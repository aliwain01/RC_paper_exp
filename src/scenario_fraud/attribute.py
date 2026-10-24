"""Phase 4 orchestration (DESIGN §9; EXPERIMENT_PLAN §4; CONTRACTS §4): every attribution method for the explained set.

For each test year, fold k, family (reference seed of fold k's primary model) and baseline (peer, global):
backgrounds (C9) → coalition table V (C10) → B, C, E, deletion scores; method A (C11) for tree families; method D
(C12) for glass boxes. Path-dependent diagnostics (A-path, feature-level TreeSHAP interactions) on a subsample.
Store: data/processed/attributions/{family}/{baseline}/fold{k}.parquet (+ columns.json per family).
Resumable (DESIGN §17 D123): each (family, baseline, year, fold) is computed in chunks of about `chunk_rows`
explained providers, each written to {family}/{baseline}/parts/{year}_fold{k}/ as it finishes; a rerun skips the
chunks of a piece whose plan.json fingerprint (settings, model, providers, chunking) still matches, and the fold
file is assembled from the parts. Providers are independent given their background, so chunking changes no value.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time

import numpy as np
import polars as pl

from . import features, models, splits, train
from .attribution import background as bgm
from .attribution import coalition as C
from .attribution import glassbox, summed_shap
from .config import load_config
from .explained_set import load_explained_set
from .groups import GROUP_ORDER, feature_groups
from .modelmatrix import matrix
from .paths import Paths

M = len(GROUP_ORDER)


def store_path(paths: Paths, family: str, baseline: str, fold: int):
    return paths.processed / "attributions" / family / baseline / f"fold{fold}.parquet"


def col_groups(columns: list[str], paths: Paths) -> np.ndarray:
    g = feature_groups(columns, paths=paths)
    return np.array([GROUP_ORDER.index(g[c]) for c in columns])


def _arr(name: str, a: np.ndarray | None, n: int) -> pl.Series:
    if a is None:
        return pl.Series(name, [None] * n, dtype=pl.List(pl.Float64))
    return pl.Series(name, np.asarray(a, dtype=np.float64).reshape(n, -1).tolist(), dtype=pl.List(pl.Float64))


def _methods(f: models.Fitted, x: np.ndarray, xb: np.ndarray, cg: np.ndarray, acfg: dict, seed: int) -> dict:
    """Every method for providers x sharing background xb."""
    pred = lambda Z: models.margin(f, Z)  # noqa: E731
    V = C.coalition_values(pred, x, xb, cg, M, int(acfg["batch_rows"]))
    phi = C.shapley(V)
    Mx = C.interaction_matrix(V, phi)
    E, Ep = C.counterfactual(V)
    out = {"v": V, "B": phi, "C": Mx, "E": E, "E_prob": Ep, "A": None, "A_feat": None, "A_gap": None, "D": None,
           "D_pair": None, "D_feat": None, "margin": V[:, -1], "base": V[:, 0]}
    if f.family in acfg["shap_families"]:
        phi_f, _ = summed_shap.feature_shap(f, x, xb)
        out["A"], out["A_feat"] = summed_shap.group_sums(phi_f, cg, M), phi_f
        # |Σφ − (f(x) − mean_B f)| per provider (D126): above 1e-4, method A is not exact for that row
        out["A_gap"] = np.abs(phi_f.sum(axis=1) - (V[:, -1] - V[:, 0]))
    if f.family == "lr":
        out["D"], _ = glassbox.lr_contrib(f, x, xb, GROUP_ORDER)
        out["D_feat"] = glassbox.lr_feature_contrib(f, x, xb)
    elif f.family in ("ebm", "ga2m"):
        out["D"], out["D_pair"], out["D_feat"] = glassbox.ebm_contrib(f, x, xb, cg, M)
    for name in ("B", "A", "D", "E"):
        if out[name] is not None:
            o = C.order_by(out[name])
            out[f"del_{name}"] = C.deletion_score(V, o)
            out[f"del_{name}_reverse"] = C.deletion_score(V, o[:, ::-1])
    out["del_random"] = C.random_deletion_score(V, int(acfg["deletion"]["random_orders"]), seed)
    return out


STORE_LISTS = ("v", "B", "C", "E", "E_prob", "A", "A_feat", "D", "D_pair", "D_feat")


def _plan(exf: pl.DataFrame, cand: pl.DataFrame, base: str, bcfg: bgm.BackgroundConfig, k: int, yr: int, *,
          assign=None, members=None, gl=None) -> tuple[pl.DataFrame, list]:
    """Store metadata per row of `exf` and the jobs (row indices into exf, row indices into cand) that share one
    background."""
    cidx = {n: i for i, n in enumerate(cand["npi"].to_list())}
    if base == "peer":
        if assign is None or members is None:
            assign, members = bgm.peer_backgrounds(exf, cand, bcfg, k, yr)
        # rows of exf by NPI (joins do not promise row order)
        meta = exf.select("npi").with_row_index("_r").join(assign, on="npi", how="left").sort("_r")
        if meta["bg_key"].null_count():
            raise RuntimeError("explained provider without a background")
        mem = members.group_by("bg_key").agg(pl.col("npi"))
        mem = dict(zip(mem["bg_key"].to_list(), mem["npi"].to_list()))
        jobs = [(np.asarray(r["_r"]), [cidx[n] for n in mem[r["bg_key"]]])
                for r in meta.group_by("bg_key", maintain_order=True).agg(pl.col("_r")).to_dicts()]
        meta = meta.drop("_r")
    else:
        if gl is None:
            gl = bgm.global_background(cand, bcfg, k, yr)
        jobs = [(np.arange(exf.height), [cidx[n] for n in gl.to_list()])]
        meta = exf.select("npi").with_columns(pl.lit(f"global={yr}").alias("bg_key"), pl.lit("global").alias("bg_level"),
                                              pl.lit(gl.len(), dtype=pl.UInt32).alias("bg_n"))
    return meta, jobs


def _chunks(jobs: list, chunk_rows: int) -> list[list]:
    """Jobs packed into chunks of about `chunk_rows` explained rows, in job order; a large job is split by rows
    (its background is unchanged)."""
    out, cur, n = [], [], 0
    for r_idx, b_idx in jobs:
        for i in range(0, len(r_idx), chunk_rows):
            piece = r_idx[i:i + chunk_rows]
            if cur and n + len(piece) > chunk_rows:
                out.append(cur)
                cur, n = [], 0
            cur.append((piece, b_idx))
            n += len(piece)
    if cur:
        out.append(cur)
    return out


def _run(f: models.Fitted, Xe: np.ndarray, Xc: np.ndarray, meta: pl.DataFrame, jobs: list, cg: np.ndarray,
         acfg: dict, bcfg: bgm.BackgroundConfig, k: int, yr: int) -> pl.DataFrame:
    parts = []
    for r_idx, b_idx in jobs:
        res = _methods(f, Xe[r_idx], Xc[b_idx], cg, acfg, bcfg.seed + k)
        n = len(r_idx)
        frame = meta[r_idx].with_columns(pl.lit(yr, dtype=pl.Int32).alias("year"), pl.lit(k, dtype=pl.Int32).alias("fold"))
        cols = [_arr(c, res[c], n) for c in STORE_LISTS]
        cols += [pl.Series(c, np.asarray(res[c], dtype=np.float64)) for c in
                 ("margin", "base", *(["A_gap"] if res.get("A_gap") is not None else []),
                  *sorted(c for c in res if c.startswith("del_")))]
        parts.append(frame.with_columns(cols))
    return pl.concat(parts, how="diagonal_relaxed")


def explain_rows(f: models.Fitted, exf: pl.DataFrame, cand: pl.DataFrame, base: str, cg: np.ndarray, acfg: dict,
                 bcfg: bgm.BackgroundConfig, k: int, yr: int, *, assign=None, members=None, gl=None) -> pl.DataFrame:
    """Store rows (CONTRACTS §4) for the providers of `exf` (one fold k and year) under model f, against peer or
    global backgrounds drawn from `cand` (training-fold candidates of that year, sorted by NPI). Backgrounds already
    drawn for these frames may be passed in (assign/members, gl)."""
    meta, jobs = _plan(exf, cand, base, bcfg, k, yr, assign=assign, members=members, gl=gl)
    return _run(f, matrix(exf, f.columns), matrix(cand, f.columns), meta, jobs, cg, acfg, bcfg, k, yr)


class _Log:
    """Progress lines on stderr and appended to outputs/phase4/explain_progress.log."""

    def __init__(self, paths: Paths):
        self.path = paths.outputs / "phase4" / "explain_progress.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, file=sys.stderr, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def parts_dir(paths: Paths, family: str, baseline: str, yr: int, fold: int):
    return paths.processed / "attributions" / family / baseline / "parts" / f"{yr}_fold{fold}"


def _fingerprint(paths: Paths, acfg: dict, bcfg: bgm.BackgroundConfig, design: str, fam: str, k: int, base: str,
                 exf: pl.DataFrame, chunk_rows: int, n_chunks: int) -> str:
    """Everything a piece's values depend on: attribution settings, background settings, the model (its tuning
    record), the explained providers and the chunking."""
    keep = {key: acfg.get(key) for key in ("background", "deletion", "shap_families", "reference_seed", "seed")}
    tj = train.model_dir(paths, design, fam, k) / "tuning.json"
    model = json.loads(tj.read_text(encoding="utf-8")) if tj.is_file() else {}
    doc = {"acfg": keep, "bcfg": {k2: str(v) for k2, v in sorted(vars(bcfg).items())}, "base": base,
           "model": {k2: model.get(k2) for k2 in ("best_params", "fingerprint", "seeds", "n_trials")},
           "npis": hashlib.sha256("\n".join(exf["npi"].cast(pl.Utf8).to_list()).encode()).hexdigest(),
           "chunk_rows": chunk_rows, "n_chunks": n_chunks}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _write(df: pl.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def explain(paths: Paths, *, families=None, baselines=None, folds=None, limit: int | None = None,
            resume: bool = True) -> dict:
    train.check_frozen(paths)
    acfg = load_config("attribution", paths)
    bcfg = bgm.BackgroundConfig.load(paths)
    scfg = splits.SplitConfig.load(paths)
    design = acfg["explained"]["design"]
    chunk_rows = int(acfg.get("chunk_rows", 1000))
    fams = list(families or acfg["coalition_families"])
    bases = list(baselines or acfg["baselines"])
    es = load_explained_set(paths)
    fold_tab = splits.load_folds(paths).select("npi", "fold", "strat_specialty")
    cells = pl.scan_parquet(paths.processed / "cohort.parquet").select(
        "npi", pl.col("year").cast(pl.Int32), "cell_spec_ent", "cell_ent").collect()
    years = sorted(es["year"].unique().to_list())
    fold_ids = sorted(int(k) for k in (folds if folds is not None else range(scfg.n_folds)))
    log = _Log(paths)
    log(f"explain: families {fams}, baselines {bases}, folds {fold_ids}, years {years}, chunk_rows {chunk_rows}, "
        f"resume={resume}")
    done: dict = {(fam, b, k): [] for fam in fams for b in bases for k in fold_ids}
    summary: dict = {"design": design, "families": fams, "baselines": bases, "seconds": {}, "chunks": {}}
    for yr in years:
        feats = (features.load_features(paths, [yr]).with_columns(pl.col("year").cast(pl.Int32))
                 .join(cells, on=["npi", "year"], how="left").join(fold_tab, on="npi", how="inner"))
        for k in fold_ids:
            ex = es.filter((pl.col("year") == yr) & (pl.col("fold") == k)).select("npi")
            if limit is not None:
                ex = ex.with_columns(pl.Series("_h", splits.npi_keys(ex["npi"], splits._salt(bcfg.seed, 104, k, yr)))
                                     ).sort("_h").head(limit).drop("_h")
            exf = feats.join(ex, on="npi", how="inner").sort("npi")
            cand = feats.filter(pl.col("fold") != k).sort("npi")
            if exf.height == 0:
                continue
            assign, members = bgm.peer_backgrounds(exf, cand, bcfg, k, yr)
            gl = bgm.global_background(cand, bcfg, k, yr)
            # provenance (DESIGN §11.2): which providers every explanation was compared with
            bp = paths.processed / "attributions" / "backgrounds" / f"{yr}_fold{k}.parquet"
            bp.parent.mkdir(parents=True, exist_ok=True)
            pl.concat([members, pl.DataFrame({"bg_key": [f"global={yr}"] * gl.len(), "npi": gl})]).with_columns(
                pl.lit(yr, dtype=pl.Int32).alias("year"), pl.lit(k, dtype=pl.Int32).alias("fold")).write_parquet(bp)
            for fam in fams:
                t0 = time.perf_counter()
                f = models.load(train.model_dir(paths, design, fam, k) / f"seed{int(acfg['reference_seed'])}")
                cg = col_groups(f.columns, paths)
                Xe = Xc = None
                for base in bases:
                    if base == "global" and fam not in acfg.get("global_families", fams):
                        continue
                    meta, jobs = _plan(exf, cand, base, bcfg, k, yr, assign=assign, members=members, gl=gl)
                    chunks = _chunks(jobs, chunk_rows)
                    fp = _fingerprint(paths, acfg, bcfg, design, fam, k, base, exf, chunk_rows, len(chunks))
                    pdir = parts_dir(paths, fam, base, yr, k)
                    plan = pdir / "plan.json"
                    if not resume or not plan.is_file() or json.loads(plan.read_text(encoding="utf-8")).get("fp") != fp:
                        if pdir.exists():
                            if resume:
                                log(f"{fam}/{base} {yr} fold {k}: settings or providers changed; recomputing")
                            shutil.rmtree(pdir)
                        pdir.mkdir(parents=True)
                        plan.write_text(json.dumps({"fp": fp, "n_chunks": len(chunks), "rows": exf.height}),
                                        encoding="utf-8")
                    files, skipped = [], 0
                    for j, chunk in enumerate(chunks):
                        cp = pdir / f"c{j:04d}.parquet"
                        files.append(cp)
                        if cp.is_file():
                            skipped += 1
                            continue
                        if Xe is None:
                            Xe, Xc = matrix(exf, f.columns), matrix(cand, f.columns)
                        t1 = time.perf_counter()
                        _write(_run(f, Xe, Xc, meta, chunk, cg, acfg, bcfg, k, yr), cp)
                        log(f"{fam}/{base} {yr} fold {k}: chunk {j + 1}/{len(chunks)} "
                            f"({sum(len(r) for r, _ in chunk)} providers) {time.perf_counter() - t1:.1f}s")
                    if skipped:
                        log(f"{fam}/{base} {yr} fold {k}: {skipped} of {len(chunks)} chunks already done, skipped")
                    done[(fam, base, k)].extend(files)
                    summary["chunks"][f"{fam}/{base}/{yr}/fold{k}"] = len(chunks)
                summary["seconds"][f"{fam}/{yr}/fold{k}"] = round(time.perf_counter() - t0, 2)
                (paths.processed / "attributions" / fam).mkdir(parents=True, exist_ok=True)
                (paths.processed / "attributions" / fam / "columns.json").write_text(json.dumps(
                    {"columns": f.columns, "col_group": cg.tolist(), "groups": list(GROUP_ORDER)}), encoding="utf-8")
    written = []
    for (fam, base, k), files in done.items():
        if files:
            p = store_path(paths, fam, base, k)
            _write(pl.concat([pl.read_parquet(x) for x in files], how="diagonal_relaxed"), p)
            written.append(paths.rel(p))
            log(f"{fam}/{base} fold {k}: store written ({len(files)} chunks)")
    summary["written"] = written
    return summary


def load_attributions(paths: Paths, family: str, baseline: str) -> pl.DataFrame:
    files = sorted((paths.processed / "attributions" / family / baseline).glob("fold*.parquet"))
    if not files:
        raise FileNotFoundError(f"no attributions for {family}/{baseline}")
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def diagnostics(paths: Paths, *, families=None) -> dict:
    """DESIGN §9.4: path-dependent TreeSHAP group sums and feature-level interaction values on a subsample of the
    explained set, compared with the interventional (peer) method A and with method C."""
    acfg = load_config("attribution", paths)
    n_total = int(acfg["diagnostics"]["path_dependent_n"])
    out = {}
    for fam in families or acfg["shap_families"]:
        try:
            att = load_attributions(paths, fam, "peer")
        except FileNotFoundError:
            continue
        att = att.with_columns(pl.Series("_h", splits.npi_keys(att["npi"], splits._salt(7, 105)))).sort("_h").head(n_total)
        rows = []
        for (k,), g in att.group_by("fold"):
            f = models.load(train.model_dir(paths, acfg["explained"]["design"], fam, int(k))
                            / f"seed{int(acfg['reference_seed'])}")
            feats = pl.concat([features.load_features(paths, [y]).with_columns(pl.col("year").cast(pl.Int32))
                               for y in g["year"].unique().to_list()], how="diagonal_relaxed")
            gx = g.select("npi", "year").join(feats, on=["npi", "year"], how="left")
            A_path, IV = summed_shap.path_dependent(f, matrix(gx, f.columns), col_groups(f.columns, paths), M)
            A = np.array(g["A"].to_list())
            Cm = np.array(g["C"].to_list()).reshape(-1, M, M)
            rows.append({"fold": int(k), "n": g.height,
                         "max_abs_A_path_minus_A": float(np.abs(A_path - A).max()),
                         "mean_abs_A_path_minus_A": float(np.abs(A_path - A).mean()),
                         "corr_offdiag_IV_vs_C": float(np.corrcoef(
                             (IV + np.transpose(IV, (0, 2, 1)))[:, ~np.eye(M, dtype=bool)].ravel(),
                             (Cm + np.transpose(Cm, (0, 2, 1)))[:, ~np.eye(M, dtype=bool)].ravel())[0, 1])})
        out[fam] = rows
    p = paths.outputs / "phase4" / "diagnostics.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
