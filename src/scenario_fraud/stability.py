"""E7 stability (EXPERIMENT_PLAN §5.4; DESIGN §9.5, §10.5; CONTRACTS §6).

On a seeded subsample of the explained set (default 2,000 provider-years, stratified by test year), every method is
recomputed (Phase 4 engine, same backgrounds) under
- the seeds 0..S-1 of each fold's primary model (Phase 3; seed 0 is the reference run),
- NPI-bootstrap retrains of the reference configuration: fold k's final-fit rows (negatives subsampled as in Phase 3)
  with NPIs drawn with replacement (row weight × the NPI's draw count), the tuned parameters, seed 0,
- perturbed inputs: every shrunk-rate numerator k redrawn as Binomial(round(n), k/n) (rescaled to n) given its
  denominator n, the shrunk rate and robust z recomputed with that year's persisted peer statistics; other
  features unchanged; reference model.
Reported per family and method: top-1 agreement with the reference run (seeds, bootstraps, perturbations) and the
mean pairwise Kendall τ_b over the 7 scheme groups within the seed runs and within the bootstrap runs.
Runs: data/processed/stability/{family}/{seed,boot,perturb}{i}.parquet; models: stability/models/; summary:
outputs/phase6/stability.json.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import polars as pl

from . import features, models, splits, train
from .attribute import col_groups, explain_rows
from .attribution import background as bgm
from .config import load_config
from .evaluate import METHODS, group_vectors, out_dir, write_json
from .explained_set import load_explained_set
from .groups import GROUP_ORDER
from .modelmatrix import load_design, matrix
from .paths import Paths

M = len(GROUP_ORDER)
KINDS = ("seed", "boot", "perturb")
_SUB, _BOOT, _PERT = 901, 902, 903


def stab_dir(paths: Paths) -> "Paths":
    return paths.processed / "stability"


def subsample(paths: Paths, n: int, seed: int) -> pl.DataFrame:
    """n provider-years of the explained set (largest-remainder allocation over test years, seeded hash order)."""
    es = load_explained_set(paths).select("npi", pl.col("year").cast(pl.Int32), "fold").unique(["npi", "year"])
    if es.height <= n:
        return es.sort("year", "npi")
    sizes = es.group_by("year").len().sort("year")
    quota = sizes["len"].to_numpy() * n / es.height
    base = np.floor(quota).astype(int)
    base[np.argsort(-(quota - base), kind="stable")[: n - base.sum()]] += 1
    alloc = pl.DataFrame({"year": sizes["year"], "_q": base})
    h = splits.npi_keys(es["npi"], splits._salt(seed, _SUB)) ^ es["year"].cast(pl.UInt64).to_numpy()
    return (es.with_columns(pl.Series("_h", h)).join(alloc, on="year").sort("year", "_h")
            .filter(pl.int_range(pl.len()).over("year") < pl.col("_q")).select("npi", "year", "fold").sort("year", "npi"))


def year_features(paths: Paths, years) -> dict[int, pl.DataFrame]:
    """Feature rows of each year with cohort cells and folds (candidates for backgrounds)."""
    fold_tab = splits.load_folds(paths).select("npi", "fold", "strat_specialty")
    cells = pl.scan_parquet(paths.processed / "cohort.parquet").select(
        "npi", pl.col("year").cast(pl.Int32), "cell_spec_ent", "cell_ent").collect()
    return {int(y): features.load_features(paths, [int(y)]).with_columns(pl.col("year").cast(pl.Int32))
            .join(cells, on=["npi", "year"], how="left").join(fold_tab, on="npi", how="inner") for y in years}


def vectors(rows: pl.DataFrame) -> pl.DataFrame:
    """Store rows -> npi, year, fold and one 8-vector per method (method C: diagonal of M)."""
    cols = [pl.Series(m, group_vectors(rows, m).tolist(), dtype=pl.List(pl.Float64)) for m in METHODS]
    return rows.select("npi", "year", "fold").with_columns(cols).sort("year", "npi")


def attribute_subsample(paths: Paths, model_for_fold: Callable[[int], models.Fitted], sub: pl.DataFrame,
                        feats: dict[int, pl.DataFrame], baseline: str,
                        perturb: Callable[[pl.DataFrame, int, int], pl.DataFrame] | None = None) -> pl.DataFrame:
    acfg = load_config("attribution", paths)
    bcfg = bgm.BackgroundConfig.load(paths)
    parts = []
    for (yr, k), g in sub.group_by("year", "fold"):
        yr, k = int(yr), int(k)
        fy = feats[yr]
        exf = fy.join(g.select("npi"), on="npi", how="inner").sort("npi")
        if exf.is_empty():
            continue
        cand = fy.filter(pl.col("fold") != k).sort("npi")
        assign, members = bgm.peer_backgrounds(exf, cand, bcfg, k, yr)    # backgrounds from the unperturbed rows
        if perturb is not None:
            exf = perturb(exf, yr, k)
        f = model_for_fold(k)
        parts.append(explain_rows(f, exf, cand, baseline, col_groups(f.columns, paths), acfg, bcfg, k, yr,
                                  assign=assign, members=members))
    return vectors(pl.concat(parts, how="diagonal_relaxed"))


# ----------------------------------------------------------------------------------------------- bootstrap retrains

def bootstrap_model(paths: Paths, family: str, fold: int, b: int, df=None) -> models.Fitted:
    d = stab_dir(paths) / "models" / family / f"boot{b}" / f"fold{fold}"
    if (d / "fitted.json").is_file():
        return models.load(d)
    mcfg = models.ModelConfig.load(paths)
    scfg = splits.SplitConfig.load(paths)
    seed = int(load_config("evaluation", paths)["seed"])
    df = df if df is not None else load_design(paths, "primary")
    fr, _, _, final, _ = train.fold_rows(df, fold, mcfg, scfg)
    npis = final["npi"].unique().sort()
    rng = np.random.default_rng(np.random.SeedSequence([seed, _BOOT, fold, b]))
    cnt = np.bincount(rng.integers(0, npis.len(), npis.len()), minlength=npis.len())
    draw = pl.DataFrame({"npi": npis, "_c": cnt}).filter(pl.col("_c") > 0)
    fin = final.join(draw, on="npi", how="inner").with_columns((pl.col("weight") * pl.col("_c")).alias("weight"))
    X = matrix(fr, df.columns, fin["_i"].to_numpy())
    f = models.fit(family, X, fin["y"].to_numpy(), fin["weight"].to_numpy(), train._reused_params(paths, family, fold),
                   train._seed(mcfg["tuning_seed"], 17, train.FAMILY_SALT[family], fold, b), df.columns, mcfg, paths)
    models.save(f, d)
    return f


# ----------------------------------------------------------------------------------------------- perturbation

def shrunk_z(exf: pl.DataFrame, stats: pl.DataFrame, fcfg: features.FeatureConfig,
             k_new: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Robust z of every shrunk-rate feature of `exf` from its persisted raw__{ID}__k / __n and the year's peer
    statistics (features.transform's formula); `k_new` replaces numerators. Rows without a positive n keep their z."""
    out = {}
    for s in fcfg.robust:
        if not s.shrink or s.id not in exf.columns or f"{features.RAW}{s.id}__n" not in exf.columns:
            continue
        st = stats.filter(pl.col("feature") == s.id).select("peer_cell", "median", "scale", "alpha", "prior_mean")
        j = exf.select("peer_cell").join(st, on="peer_cell", how="left", maintain_order="left")
        k = exf[f"{features.RAW}{s.id}__k"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        n = exf[f"{features.RAW}{s.id}__n"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        if k_new is not None and s.id in k_new:
            k = k_new[s.id]
        a, m = j["alpha"].to_numpy().astype(float), j["prior_mean"].to_numpy().astype(float)
        med, sc = j["median"].to_numpy().astype(float), j["scale"].to_numpy().astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(n > 0, (k + a * m) / (n + a), np.nan)
            z = np.clip((v - med) / sc, -fcfg.z_clip, fcfg.z_clip)
        z = np.where(np.isfinite(z), z, 0.0)
        if s.fit_on == "dme_referrers" and "DI0" in exf.columns:
            z = np.where(exf["DI0"].to_numpy() == 1, 0.0, z)
        keep = ~(n > 0)
        out[s.id] = np.where(keep, exf[s.id].to_numpy().astype(float), z)
    return out


def binomial_numerators(exf: pl.DataFrame, fcfg: features.FeatureConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    out = {}
    for s in fcfg.robust:
        kc, nc = f"{features.RAW}{s.id}__k", f"{features.RAW}{s.id}__n"
        if not s.shrink or kc not in exf.columns or s.id not in exf.columns:
            continue
        k = exf[kc].cast(pl.Float64).fill_null(np.nan).to_numpy()
        n = exf[nc].cast(pl.Float64).fill_null(np.nan).to_numpy()
        ok = np.isfinite(k) & np.isfinite(n) & (n >= 1)
        nn = np.where(ok, np.round(n), 1).astype(np.int64)
        p = np.where(ok, np.clip(k / np.where(ok, n, 1), 0, 1), 0)
        draw = rng.binomial(nn, p) * np.where(ok, n / nn, 0)
        out[s.id] = np.where(ok, draw, k)
    return out


def perturber(paths: Paths, p: int, seed: int) -> Callable[[pl.DataFrame, int, int], pl.DataFrame]:
    fcfg = features.FeatureConfig.load(paths)
    stats: dict[int, pl.DataFrame] = {}

    def fn(exf: pl.DataFrame, yr: int, k: int) -> pl.DataFrame:
        if yr not in stats:
            stats[yr] = pl.read_parquet(paths.processed / "peer" / str(yr) / "stats.parquet")
        rng = np.random.default_rng(np.random.SeedSequence([seed, _PERT, p, yr, k]))
        z = shrunk_z(exf, stats[yr], fcfg, binomial_numerators(exf, fcfg, rng))
        return exf.with_columns([pl.Series(c, v, dtype=pl.Float64) for c, v in z.items()])

    return fn


# ----------------------------------------------------------------------------------------------- metrics

def kendall_tau_b(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Row-wise Kendall τ_b of two n × m matrices (NaN when a row is constant)."""
    i, j = np.triu_indices(X.shape[1], 1)
    dx, dy = np.sign(X[:, i] - X[:, j]), np.sign(Y[:, i] - Y[:, j])
    n0 = len(i)
    den = np.sqrt((n0 - (dx == 0).sum(axis=1)) * (n0 - (dy == 0).sum(axis=1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (dx * dy).sum(axis=1) / den, np.nan)


def _mat(run: pl.DataFrame, method: str) -> np.ndarray:
    v = run[method].to_list()
    return np.array([x[1:] if x is not None else [np.nan] * (M - 1) for x in v], dtype=float)


def agreement(ref: np.ndarray, other: np.ndarray) -> float:
    ok = np.isfinite(ref).all(axis=1) & np.isfinite(other).all(axis=1)
    return float(np.mean(np.argmax(ref[ok], axis=1) == np.argmax(other[ok], axis=1))) if ok.any() else float("nan")


def mean_pairwise_tau(mats: list[np.ndarray]) -> float:
    vals = []
    for a, b in combinations(mats, 2):
        ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        vals.extend(kendall_tau_b(a[ok], b[ok]).tolist())
    v = np.array(vals, dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def summarise(runs: dict[tuple[str, int], pl.DataFrame]) -> list[dict]:
    ref = runs.get(("seed", 0))
    if ref is None:
        return []
    out = []
    for meth in METHODS:
        r = _mat(ref, meth)
        if not np.isfinite(r).all(axis=1).any():
            continue
        row = {"method": meth, "n": int(np.isfinite(r).all(axis=1).sum())}
        for kind in KINDS:
            mats = [_mat(v, meth) for (kd, i), v in sorted(runs.items()) if kd == kind and not (kd == "seed" and i == 0)]
            row[f"{kind}_runs"] = len(mats)
            row[f"top1_agree_{kind}"] = float(np.mean([agreement(r, m) for m in mats])) if mats else None
        seeds = [_mat(v, meth) for (kd, _), v in sorted(runs.items()) if kd == "seed"]
        boots = [_mat(v, meth) for (kd, _), v in sorted(runs.items()) if kd == "boot"]
        row["kendall_seeds"] = mean_pairwise_tau(seeds) if len(seeds) > 1 else None
        row["kendall_boot"] = mean_pairwise_tau(boots) if len(boots) > 1 else None
        out.append(row)
    return out


# ----------------------------------------------------------------------------------------------- orchestration

def run(paths: Paths, *, families=None, kinds=KINDS) -> dict:
    train.check_frozen(paths)
    sc = load_config("evaluation", paths)
    st = sc["stability"]
    seed = int(sc["seed"])
    sub = subsample(paths, int(st["n"]), seed)
    feats = year_features(paths, sorted(sub["year"].unique().to_list()))
    base = st["baseline"]
    folds = sorted(sub["fold"].unique().to_list())
    out = {"n": sub.height, "baseline": base, "families": {}}
    df = None
    for fam in families or st["families"]:
        mdirs = {k: train.model_dir(paths, "primary", fam, k) for k in folds}
        if not all((d / "seed0").is_dir() for d in mdirs.values()):
            continue
        n_seeds = min(int(st["seeds"]), *[len(list(d.glob("seed*"))) for d in mdirs.values()])
        runs: dict[tuple[str, int], pl.DataFrame] = {}
        if "seed" in kinds:
            for s in range(n_seeds):
                runs[("seed", s)] = attribute_subsample(
                    paths, lambda k, s=s: models.load(mdirs[k] / f"seed{s}"), sub, feats, base)
        else:
            runs[("seed", 0)] = attribute_subsample(paths, lambda k: models.load(mdirs[k] / "seed0"), sub, feats, base)
        if "boot" in kinds and int(st["bootstraps"]):
            df = df if df is not None else load_design(paths, "primary")
            for b in range(int(st["bootstraps"])):
                runs[("boot", b)] = attribute_subsample(
                    paths, lambda k, b=b: bootstrap_model(paths, fam, k, b, df), sub, feats, base)
        if "perturb" in kinds:
            for p in range(int(st["perturbations"])):
                runs[("perturb", p)] = attribute_subsample(
                    paths, lambda k: models.load(mdirs[k] / "seed0"), sub, feats, base, perturber(paths, p, seed))
        keys = runs[("seed", 0)].select("npi", "year")
        for (kind, i), r in runs.items():
            if not r.select("npi", "year").equals(keys):
                raise RuntimeError(f"stability run {kind}{i} of {fam} explains different rows than the reference")
            p_ = stab_dir(paths) / fam / f"{kind}{i}.parquet"
            p_.parent.mkdir(parents=True, exist_ok=True)
            r.write_parquet(p_)
        out["families"][fam] = {"seeds": n_seeds, "summary": summarise(runs)}
    write_json(out_dir(paths) / "stability.json", out)
    return out
