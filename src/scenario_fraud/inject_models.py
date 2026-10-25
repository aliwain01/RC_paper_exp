"""Phase 5: injection models, stress test, attribution of injected providers, calibration (DESIGN §7.7;
EXPERIMENT_PLAN §6; CONTRACTS §5).

- train(paths, draw): use (b). One injection model per family and fold on the injected copy: label = injected (1)
  vs clean non-injected (0); NPIs positive under any LEIE tier removed; training rows = the primary design's
  TRAIN_ROLES of fold k (train years), negatives subsampled as for the LEIE models; parameters = the LEIE model's
  tuned parameters for the same family and fold (seed 0). Test rows (fold k, test years) are scored out of fold.
- stress_test(paths, draw): use (a). The LEIE models (primary, seed 0, own fold) score the injected test providers;
  each gets its percentile among the clean test rows of its year and fold.
- explain(paths, draw): methods A-E (Phase 4 engine) for every injected test provider under the injection model of
  its fold, against backgrounds of clean training-fold providers of the injected copy.
- calibrate(paths): for each scheme and intensity on the grid, the peer-cell percentile of the scheme's primary
  feature after injection into a sample of clean providers (the human accepts intensities at G5).
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from . import features, injection, models, splits, train
from .attribute import col_groups, explain_rows
from .attribution import background as bgm
from .config import load_config
from .modelmatrix import matrix, model_column_order
from .paths import Paths


EXTRA = ("injected", "fold", "ever_positive", "strat_specialty", "y", "cell_spec_ent", "cell_ent")


def _cohort_cells(paths: Paths, draw: int, years) -> pl.DataFrame:
    parts = []
    for y in years:
        c = pl.read_parquet(injection.draw_dir(paths, draw) / "peer" / str(int(y)) / "cohort.parquet")
        parts.append(c.select("npi", pl.col("year").cast(pl.Int32), "cell_spec_ent", "cell_ent"))
    return pl.concat(parts)


def _frame(paths: Paths, draw: int, years) -> pl.DataFrame:
    """Injected copy of the given years with label `y` = injected, LEIE-positive NPIs (any tier) removed."""
    folds = splits.load_folds(paths).select("npi", "fold", "ever_positive", "strat_specialty")
    f = injection.load_injected_features(paths, draw, years).with_columns(pl.col("year").cast(pl.Int32))
    f = f.join(folds, on="npi", how="inner").filter(~pl.col("ever_positive"))
    return f.with_columns(pl.col("injected").cast(pl.Int8).alias("y"))


def model_dir(paths: Paths, draw: int, family: str, fold: int):
    return injection.draw_dir(paths, draw) / "models" / family / f"fold{fold}"


def train_models(paths: Paths, draw: int, *, families=None, folds=None) -> dict:
    train.check_frozen(paths)
    icfg = injection.InjectionConfig.load(paths)
    mcfg = models.ModelConfig.load(paths)
    scfg = splits.SplitConfig.load(paths)
    d = scfg.design("primary")
    years = sorted({*d.train_years, *d.test_years})
    df = _frame(paths, draw, years)
    cols = model_column_order([c for c in df.columns if c not in EXTRA], paths=paths)
    plan = pl.read_parquet(injection.draw_dir(paths, draw) / "plan.parquet")
    out = {}
    for k in sorted(folds if folds is not None else range(scfg.n_folds)):
        role = splits.assign_roles(df.select("npi", "year"), splits.load_folds(paths), k, "primary", scfg)["role"]
        fr = df.with_columns(role.alias("role"), pl.int_range(pl.len()).alias("_i"))
        tr = splits.subsample_negatives(
            fr.filter(pl.col("role").is_in(list(splits.TRAIN_ROLES))).select("npi", "year", "y", "_i"), scfg)
        te = fr.filter(pl.col("role") == "test")
        X, y, w = matrix(fr, cols, tr["_i"].to_numpy()), tr["y"].to_numpy(), tr["weight"].to_numpy()
        Xt = matrix(fr, cols, te["_i"].to_numpy())
        for fam in families or icfg["families"]:
            params = train._reused_params(paths, fam, k)
            f = models.fit(fam, X, y, w, params, train._seed(mcfg["tuning_seed"], 13, train.FAMILY_SALT[fam], k, draw),
                           cols, mcfg, paths)
            models.save(f, model_dir(paths, draw, fam, k) / "seed0")
            s = te.select("npi", "year", "fold", "y", "peer_cell").with_columns(
                pl.Series("margin", models.margin(f, Xt)))
            s = s.join(plan.filter(pl.col("part") == "test").select("npi", "year", "schemes", "intensity", "pair",
                                                                    "answer_groups", "distractor", "distractor_groups"),
                       on=["npi", "year"], how="left")
            p = injection.draw_dir(paths, draw) / "scores" / fam / f"fold{k}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            s.write_parquet(p)
            out.setdefault(fam, {})[k] = {"train_rows": tr.height, "train_injected": int(y.sum()),
                                          "test_rows": te.height, "test_injected": int(te["y"].sum())}
    return out


def stress_test(paths: Paths, draw: int, *, families=None) -> pl.DataFrame:
    """Use (a): percentile of each injected test provider's LEIE-model margin among clean test rows (same year and
    fold), seed-0 primary models."""
    icfg = injection.InjectionConfig.load(paths)
    scfg = splits.SplitConfig.load(paths)
    years = scfg.design("primary").test_years
    inj = _frame(paths, draw, years).filter(pl.col("y") == 1)
    plan = pl.read_parquet(injection.draw_dir(paths, draw) / "plan.parquet").filter(pl.col("part") == "test")
    rows = []
    for fam in families or icfg["families"]:
        sc = train.load_scores(paths, "primary", fam).filter(pl.col("seed") == 0)
        for k in range(scfg.n_folds):
            g = inj.filter(pl.col("fold") == k)
            if g.is_empty():
                continue
            f = models.load(train.model_dir(paths, "primary", fam, k) / "seed0")
            m = models.margin(f, matrix(g, f.columns))
            for yr in years:
                sel = (g["year"] == yr).to_numpy()
                ref = np.sort(sc.filter((pl.col("fold") == k) & (pl.col("year") == yr))["margin"].to_numpy())
                if ref.size == 0 or not sel.any():
                    continue
                pct = np.searchsorted(ref, m[sel], side="right") / ref.size
                rows.append(g.filter(pl.Series(sel)).select("npi", "year", "fold").with_columns(
                    pl.lit(fam).alias("family"), pl.Series("margin", m[sel]), pl.Series("percentile", pct)))
    out = pl.concat(rows).join(plan.select("npi", "year", "schemes", "intensity", "pair"), on=["npi", "year"],
                               how="left")
    out.write_parquet(injection.draw_dir(paths, draw) / "use_a.parquet")
    return out


def explain(paths: Paths, draw: int, *, families=None, baselines=None, limit: int | None = None) -> dict:
    """Methods A-E for injected test providers under the injection model of their fold."""
    train.check_frozen(paths)
    icfg = injection.InjectionConfig.load(paths)
    acfg = load_config("attribution", paths)
    bcfg = bgm.BackgroundConfig.load(paths)
    scfg = splits.SplitConfig.load(paths)
    years = scfg.design("primary").test_years
    df = _frame(paths, draw, years).join(_cohort_cells(paths, draw, years), on=["npi", "year"], how="left")
    plan = pl.read_parquet(injection.draw_dir(paths, draw) / "plan.parquet").filter(pl.col("part") == "test")
    out_root = injection.draw_dir(paths, draw) / "attributions"
    written = []
    for fam in families or icfg["families"]:
        for base in baselines or icfg["attribution_baselines"]:
            for k in range(scfg.n_folds):
                parts = []
                f = models.load(model_dir(paths, draw, fam, k) / "seed0")
                cg = col_groups(f.columns, paths)
                for yr in years:
                    ex = df.filter((pl.col("fold") == k) & (pl.col("year") == yr) & (pl.col("y") == 1)).sort("npi")
                    if limit is not None:
                        ex = ex.head(limit)
                    if ex.is_empty():
                        continue
                    cand = df.filter((pl.col("fold") != k) & (pl.col("year") == yr) & (pl.col("y") == 0)).sort("npi")
                    parts.append(explain_rows(f, ex, cand, base, cg, acfg, bcfg, k, yr))
                if parts:
                    a = pl.concat(parts, how="diagonal_relaxed").join(
                        plan.select("npi", "year", "schemes", "intensity", "pair", "answer_groups", "distractor",
                                    "distractor_groups"), on=["npi", "year"], how="left")
                    p = out_root / fam / base / f"fold{k}.parquet"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    a.write_parquet(p)
                    written.append(str(p))
                (out_root / fam).mkdir(parents=True, exist_ok=True)
                (out_root / fam / "columns.json").write_text(json.dumps(
                    {"columns": f.columns, "col_group": cg.tolist()}), encoding="utf-8")
    return {"draw": draw, "written": len(written)}


def load_injected_attributions(paths: Paths, family: str, baseline: str = "peer", draws=None) -> pl.DataFrame:
    root = paths.processed / "injection"
    ds = sorted(int(p.name[4:]) for p in root.glob("draw*")) if draws is None else list(draws)
    parts = []
    for d in ds:
        for f in sorted((root / f"draw{d}" / "attributions" / family / baseline).glob("fold*.parquet")):
            parts.append(pl.read_parquet(f).with_columns(pl.lit(d, dtype=pl.Int32).alias("draw")))
    if not parts:
        raise FileNotFoundError(f"no injected attributions for {family}/{baseline}")
    return pl.concat(parts, how="diagonal_relaxed")


# ----------------------------------------------------------------------------------------------- calibration

def calibrate(paths: Paths) -> dict:
    """EXPERIMENT_PLAN §6: for each scheme and grid intensity, inject into a stratified sample of applicable clean
    providers of `calibration.year` and measure the peer-cell percentile of the scheme's primary feature (z) among
    reference providers. Proposes, for each scheme, the smallest grid value whose median percentile reaches the
    target; the human accepts it (G5)."""
    icfg = injection.InjectionConfig.load(paths)
    fcfg = features.FeatureConfig.load(paths)
    cal = icfg["calibration"]
    y = int(cal["year"])
    elig = injection.eligibility(paths, [y], icfg)
    elig = elig.with_columns(pl.Series("_h", injection._h(elig["npi"], int(icfg["seed"]), 299, y)))
    out = {"year": y, "target_percentile": float(cal["target_percentile"]), "schemes": {}}
    full = injection._year_frames(paths, y)
    for s in injection.SCHEME_ORDER:
        if s not in icfg["schemes"]:
            continue
        sample = injection._take_stratified(elig.filter(pl.col(f"ok_{s}")), int(cal["sample_per_scheme"]))
        npis = sample["npi"]
        coh = injection.injected_cohort(paths, y, npis)
        coh_in = coh.filter(pl.col("in_cohort"))
        lazy = {k: v.lazy() for k, v in full.items()}
        lazy["nppes"] = None
        refs = features.fit_peer_reference(lazy, coh_in, y, fcfg)
        raw_ref = features.raw_statistics(lazy, coh_in.filter(~pl.col("injected")), refs, y, fcfg)
        ps_ = features.fit_peer_stats(raw_ref, coh_in, y, fcfg)
        feat_ref = features.transform(raw_ref, coh_in.filter(~pl.col("injected")), ps_, refs, y, fcfg)
        feat_name = icfg["schemes"][s]["primary_feature"]
        ref_vals = feat_ref.select("peer_cell", pl.col(feat_name).alias("v"))
        rows = []
        for x in cal["grid"][s]:
            who = sample.drop("peer_cell").join(coh.select("npi", "peer_cell"), on="npi").with_columns(
                pl.lit(float(x)).alias(f"x_{s}"))
            sub = {k: v.filter(pl.col("npi").is_in(npis.implode())) for k, v in full.items()}
            sub["_full"] = full
            edited, _ = injection.apply_year(sub, who, coh, refs, fcfg, icfg, -1, y)
            # lazy, so the featurisation's semi-join on the injected NPIs never copies the full year (memory)
            frames = {k: pl.concat([full[k].lazy().filter(~pl.col("npi").is_in(npis.implode())), edited[k].lazy()])
                      for k in features.DATASETS}
            frames["nppes"] = None
            inj_coh = coh_in.filter(pl.col("injected"))
            raw = features.raw_statistics(frames, inj_coh, refs, y, fcfg)
            f = features.transform(raw, inj_coh, ps_, refs, y, fcfg).select("npi", "peer_cell", pl.col(feat_name).alias("z"))
            pct = []
            for r in f.to_dicts():
                ref = ref_vals.filter(pl.col("peer_cell") == r["peer_cell"])["v"].to_numpy()
                pct.append(float(np.mean(ref <= r["z"])) if ref.size else float("nan"))
            pct = np.array(pct)
            rows.append({"intensity": float(x), "median_percentile": float(np.nanmedian(pct)),
                         "share_at_or_above_target": float(np.nanmean(pct >= float(cal["target_percentile"]))),
                         "median_z": float(f["z"].median()), "n": int(f.height)})
        hit = [r["intensity"] for r in rows if r["median_percentile"] >= float(cal["target_percentile"])]
        cur = icfg["schemes"][s]["intensities"]
        out["schemes"][s] = {"primary_feature": feat_name, "grid": rows, "current": cur,
                             "proposed_medium": min(hit) if hit else None}
    p = paths.outputs / "phase5" / "calibration.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
