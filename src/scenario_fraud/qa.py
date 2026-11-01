"""Phase 2 QA report (DESIGN §15 Phase 2 exit criterion; CONTRACTS §2.4; FEATURE_CATALOG leakage checklist).

phase2_qa(paths) -> outputs/phase2/QA_report.md (+ QA_report.json): cohort counts and peer-level mix per year,
feature quantiles, undefined and structural-zero rates, CX8 coverage, near-duplicate pairs, fold balance, and
the FEATURE_CATALOG leakage checklist, evaluated programmatically where possible:

  L1 labels / LEIE / later years    imports and artefact names used by the C4/C5 modules, and one feature year
                                    rebuilt in a scratch project that holds only that year's harmonised files
                                    and cohort rows (no labels, no LEIE, no other year) -> identical tables
  L2 peer stats non-injected only   every peer statistic's n, every code-reference cell's n_cell, every PM2/PM3
                                    fit's n and every cohort n_cell_* equal the number of non-injected in-cohort
                                    providers of that cell (with a defined value); re-run after Phase 5
  L3 label-dependent fits           split roles keep to training folds and years (model fits: Phase 3, pending)
  L4 provider folds disjoint        every design x fold: no NPI both test and training; tune sets disjoint
  L5 injection on raw rows          pending (Phase 5)
  L6 fixed before test results      hashes of cohort threshold, high-risk DME list and grouping; no later-phase
                                    output exists, or the hashes equal the freeze record (G2_freeze.json) that
                                    `phase2 report` wrote before the first later-phase output (never rewritten)
A section whose inputs are missing says so. The report states facts; the G2 decisions belong to the human.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from . import acquire, cohort, features, labels, provenance, splits
from . import groups as groups_mod
from .cohort import LEVELS as PEER_LEVELS, REASONS
from .config import load_config, load_yaml
from .paths import Paths

PASS, FAIL, PARTIAL, PENDING, NOT_RUN = "pass", "fail", "partial", "pending", "not_run"
CHECKLIST = (   # FEATURE_CATALOG "Leakage checklist (Phase 2 QA)", verbatim
    ("L1", "No feature uses labels, LEIE data or information from years after t."),
    ("L2", "Peer statistics per year use non-injected providers only."),
    ("L3", "Every label-dependent fit (models, calibration, thresholds, tuning) uses training folds and "
           "training years only."),
    ("L4", "Provider folds verified disjoint across train/test for every fold."),
    ("L5", "Injected providers are edited on raw rows, re-suppressed, then featurised."),
    ("L6", "Cohort threshold, high-risk DME list and grouping are fixed before any test result is seen."),
)
FEATURE_MODULES = ("cohort.py", "features.py", "groups.py", "provenance.py")   # C4, C5 and what they import
FORBIDDEN_IMPORTS = ("labels", "acquire", "splits")              # label-bearing modules
FORBIDDEN_NAMES = ("labels_", "exclusions", "leie")               # label artefacts / LEIE source (lower case)
LATER_PHASES = tuple(f"phase{i}" for i in range(3, 8))
FREEZE_FILE = "G2_freeze.json"      # outputs/phase2: protocol hashes recorded before any Phase 3-7 output exists
FREEZE_KEYS = ("cohort_threshold", "dme_high_risk", "grouping")   # leakage checklist item 6
ROOT_ENV = "SCENARIO_FRAUD_ROOT"


def _fmt(x):
    if isinstance(x, (float, np.floating)):
        return "nan" if np.isnan(x) else f"{float(x):.4g}"
    return x


def md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    """Markdown table: floats to 4 significant digits, '|' inside cells escaped (peer-cell ids contain it)."""
    if not rows:
        return "_none_\n"
    cols = cols or list(rows[0])

    def cell(v) -> str:
        v = _fmt(v)
        return "" if v is None else str(v).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(cell(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- provenance

def data_provenance(paths: Paths) -> dict:
    """Which inputs are synthetic test data (tests/synth.py marks the Parquet metadata, the `.meta.json` sidecar and
    the manifest notes), and which harmonised files have unknown provenance (no sidecar, no metadata flag)."""
    files = provenance.interim_files(paths.interim)
    kinds = {f: provenance.file_provenance(f) for f in files}
    synth = [paths.rel(f) for f, k in kinds.items() if k == provenance.SYNTHETIC]
    unknown = [paths.rel(f) for f, k in kinds.items() if k == provenance.UNKNOWN]
    manifest = acquire.load_manifest(paths)
    synth_raw = [e.path for e in manifest if "synthetic" in (e.notes or "").lower()]
    return {"synthetic": bool(synth or synth_raw), "harmonised_files": len(files),
            "synthetic_harmonised_files": synth, "unknown_provenance_files": unknown,
            "registered_files": len(manifest), "synthetic_registered_files": synth_raw}


def provenance_lines(prov: dict) -> list[str]:
    """Report banner. Mentions synthetic data only when some input is synthetic; warns when provenance is unknown."""
    if prov["synthetic"]:
        return ["> **SYNTHETIC DATA: NOT A RESULT.** "
                f"{len(prov['synthetic_harmonised_files'])} of {prov['harmonised_files']} harmonised files and "
                f"{len(prov['synthetic_registered_files'])} of {prov['registered_files']} registered raw files are "
                "synthetic test data generated by `tests/synth.py`. Every number below describes that generator, "
                "not Medicare providers, and must never be reported as a finding.", ""]
    if not prov["harmonised_files"]:
        return ["_No harmonised input files found under `data/interim`._", ""]
    lines = [f"Inputs: {prov['harmonised_files']} harmonised CMS files under `data/interim` and "
             f"{prov['registered_files']} registered raw files (`data/raw/manifest.json`).", ""]
    unknown = prov.get("unknown_provenance_files") or []
    if unknown:
        lines = [f"> **PROVENANCE UNKNOWN: NOT A RESULT YET.** {len(unknown)} of {prov['harmonised_files']} harmonised "
                 f"files (e.g. `{unknown[0]}`) have neither a `.meta.json` sidecar from `phase1 harmonise` nor a "
                 "provenance flag in their Parquet metadata, so they may be copied or generated data. Re-harmonise "
                 "them before quoting any number below.", "", *lines]
    return lines


# ----------------------------------------------------------------------------------------------- hashes

def protocol_hashes(paths: Paths) -> dict:
    """sha256 of the canonical JSON of what leakage item 6 says must be fixed before test results, plus the
    other frozen protocol sections."""
    exp, feat = load_config("experiment", paths), load_config("features", paths)
    return {
        "cohort_threshold": _h(exp["cohort"]["min_nondrug_services"]),
        "cohort_peer_cells": _h([exp["cohort"], load_yaml(paths.configs / "census_divisions.yaml")]),
        "dme_high_risk": _h(feat["dme_high_risk"]),
        "grouping": _h(load_config("groups", paths)),
        "features": _h(feat),
        "splits": _h(exp.get("splits")),
        "labels": _h(load_config("labels", paths)),
    }


# ----------------------------------------------------------------------------------------------- sections

def feature_years(paths: Paths) -> list[int]:
    d = paths.processed / "features"
    return sorted(int(p.stem) for p in d.glob("[0-9][0-9][0-9][0-9].parquet")) if d.is_dir() else []


def cohort_section(paths: Paths) -> dict | None:
    f = paths.processed / "cohort.parquet"
    if not f.is_file():
        return None
    meta_p = f.with_name("cohort.meta.json")
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
    lf = pl.scan_parquet(f)
    counts = lf.group_by("year").agg(
        pl.len().alias("rows"), pl.col("in_cohort").sum().alias("in_cohort"),
        *[(pl.col("cohort_exclusion_reason") == r).sum().alias(r) for r in REASONS],
        (pl.col("in_cohort") & (pl.col("census_division") == "other")).sum().alias("in_cohort_division_other"),
        (pl.col("in_cohort") & (pl.col("ruca_class") == "unknown")).sum().alias("in_cohort_ruca_unknown"),
    ).sort("year").collect()
    rows = []
    for r in counts.to_dicts():
        conf = (meta.get("per_year", {}).get(str(r["year"]), {}).get("attribute_conflicts") or {})
        rows.append({**r, "attribute_conflicts": sum(conf.values()) if conf else None})
    inc = lf.filter("in_cohort")
    cells = inc.group_by("year", "peer_level", "peer_cell").len().collect()
    mix = []
    for (y,), g in sorted(cells.group_by("year"), key=lambda t: t[0]):
        row = {"year": y}
        for lvl in PEER_LEVELS:
            s = g.filter(pl.col("peer_level") == lvl)
            row[f"providers_{lvl}"] = int(s["len"].sum())
            row[f"cells_{lvl}"] = s.height
        row["smallest_cell"] = int(g["len"].min())
        row["undersized_ent_cells"] = len(meta.get("per_year", {}).get(str(y), {}).get("undersized_ent_cells") or {})
        mix.append(row)
    stale = None
    if meta.get("config"):
        cur = load_config("experiment", paths)["cohort"]
        if (float(meta["config"].get("min_nondrug_services", -1)) != float(cur["min_nondrug_services"])
                or meta["config"].get("refine_min") != cur["peer"]["refine_min"]
                or meta["config"].get("min_cell") != cur["peer"]["min_cell"]):
            stale = "cohort.parquet was built with a different cohort config than configs/experiment.yaml"
    return {"counts": rows, "peer_mix": mix, "stale": stale}


def feature_section(paths: Paths, years: list[int], qs: list[float]) -> dict:
    """Per year: raw-statistic quantiles, undefined share and |z| clipping of robust_z features; quantiles of
    context_raw features (CX8 = coverage); indicator shares (structural zeros)."""
    fcfg = features.FeatureConfig.load(paths)
    out = {}
    for y in years:
        df = pl.read_parquet(paths.processed / "features" / f"{y}.parquet")
        n = df.height
        rows = []
        for s in fcfg.specs:
            if s.kind == "robust_z":
                col = f"{features.RAW}{s.id}"
            elif s.kind == "context_raw" and s.id in df.columns:
                col = s.id
            else:
                continue
            v = df[col].cast(pl.Float64).drop_nulls().drop_nans().to_numpy()
            q = np.quantile(v, qs) if v.size else [None] * len(qs)
            row = {"feature": s.id, "kind": s.kind, "undefined_share": (n - v.size) / n if n else None,
                   **{f"p{round(100 * a, 2):g}": b for a, b in zip(qs, q)}}
            if s.kind == "robust_z":
                row["clipped_share"] = (float((df[s.id].abs() >= fcfg.z_clip).mean())
                                        if s.id in df.columns and n else None)
            rows.append(row)
        ind = {s.id: float(df[s.id].cast(pl.Float64).mean()) for s in fcfg.specs
               if s.kind == "indicator" and s.id in df.columns and n}
        meta_p = paths.processed / "features" / f"{y}.meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
        out[y] = {"rows": n, "features": rows, "indicator_shares": ind,
                  "structural_zero_shares": {"UP0_zero": 1 - ind["UP0"] if "UP0" in ind else None,
                                             "DI0_one": ind.get("DI0"), "PM4_missing": ind.get("PM4_missing")},
                  "not_computable": meta.get("not_computable", {}), "inputs_present": meta.get("inputs_present", {}),
                  "model_columns": len(features.model_columns(df.columns))}
    return out


def near_duplicates(paths: Paths, years: list[int]) -> list[dict]:
    """features.prune_candidates per year on the model columns, merged over years."""
    fcfg = features.FeatureConfig.load(paths)
    found: dict[tuple, dict] = {}
    for y in years:
        f = paths.processed / "features" / f"{y}.parquet"
        cols = features.model_columns(pl.read_parquet_schema(f).keys())
        df = pl.read_parquet(f, columns=cols)
        pc = features.prune_candidates(df, groups_mod.feature_groups(cols, paths=paths), fcfg)
        for r in pc.to_dicts():
            k = (r["group"], r["feature_a"], r["feature_b"])
            e = found.setdefault(k, {"group": k[0], "feature_a": k[1], "feature_b": k[2], "years": [], "rho": []})
            e["years"].append(y)
            e["rho"].append(r["rho"])
    return [{"group": e["group"], "feature_a": e["feature_a"], "feature_b": e["feature_b"],
             "years": ",".join(map(str, e["years"])), "min_abs_rho": min(abs(r) for r in e["rho"]),
             "max_abs_rho": max(abs(r) for r in e["rho"])}
            for e in sorted(found.values(), key=lambda e: (groups_mod.GROUP_ORDER.index(e["group"]), e["feature_a"],
                                                           e["feature_b"]))]


def _scale(v: pl.Expr, fcfg: features.FeatureConfig) -> pl.Expr:
    """Robust scale as in features (1.4826 MAD, else IQR / 1.349, else SD); null when there is no spread."""
    mad = (v - v.median()).abs().median() * fcfg.mad_constant
    iqr = (v.quantile(0.75, interpolation="linear") - v.quantile(0.25, interpolation="linear")) / fcfg.iqr_constant
    sd = v.std(ddof=1).fill_nan(None)
    eps = fcfg.scale_eps
    return pl.when(mad > eps).then(mad).when(iqr > eps).then(iqr).when(sd > eps).then(sd)


def region_section(paths: Paths, years: list[int], threshold: float) -> list[dict]:
    """Evidence for DESIGN §19 Q2 ("is region needed?"): for every provider whose peer cell is a Census-division
    cell (spec x ent x division), how far its division cell's median of the raw statistic sits from the median of
    the whole spec x ent cell, in units of the spec x ent cell's robust scale (the shift of a typical member's z if
    the division refinement were dropped), and the division cell's scale relative to the spec x ent cell's. Over
    reference providers (in-cohort, not injected), all feature years pooled. Reference-dependent features (SM1, SM2,
    CH4, PM2, PM3) are left out: their raw value itself changes with the reference. Shrunk rates use the unshrunk
    raw rate."""
    fcfg = features.FeatureConfig.load(paths)
    specs = [s for s in fcfg.robust if s.id not in features.REF_DEPENDENT]
    coh = pl.scan_parquet(paths.processed / "cohort.parquet")
    coh = coh.filter(cohort.counted(coh))
    parts = []
    for y in years:
        f = pl.scan_parquet(paths.processed / "features" / f"{y}.parquet").select(
            "npi", "DI0", *[f"{features.RAW}{s.id}" for s in specs])
        R = (coh.filter(pl.col("year") == y).select("npi", "peer_level", "cell_spec_ent", "cell_spec_ent_div")
             .join(f, on="npi", how="inner").collect())
        for s in specs:
            v = pl.col(f"{features.RAW}{s.id}")
            ok = v.is_not_null() & v.is_finite() & ((pl.col("DI0") == 0) if s.fit_on == "dme_referrers" else pl.lit(True))
            D = R.filter(ok).select("npi", "peer_level", "cell_spec_ent", "cell_spec_ent_div", v.alias("v"))
            se = D.group_by("cell_spec_ent").agg(pl.col("v").median().alias("med_s"), _scale(pl.col("v"), fcfg).alias("sc_s"))
            dv = (D.filter(pl.col("peer_level") == "spec_ent_div").group_by("cell_spec_ent_div", "cell_spec_ent")
                  .agg(pl.len().alias("n_d"), pl.col("v").median().alias("med_d"),
                       _scale(pl.col("v"), fcfg).alias("sc_d")))
            parts.append(dv.join(se, on="cell_spec_ent").select(
                pl.lit(s.id).alias("feature"), "n_d",
                ((pl.col("med_d") - pl.col("med_s")) / pl.col("sc_s")).abs().alias("shift"),
                (pl.col("sc_d") / pl.col("sc_s")).alias("ratio")))
    if not parts:
        return []
    refined = coh.filter(pl.col("year").is_in(years)).select(
        (pl.col("peer_level") == "spec_ent_div").mean()).collect().item()
    P = pl.concat(parts).drop_nulls("shift")
    # one entry per refined provider: repeat each cell's values n_d times (provider-weighted quantiles)
    P = P.select(pl.col("feature", "shift", "ratio").repeat_by("n_d").explode())
    out = P.group_by("feature").agg(
        pl.len().alias("refined_providers"), pl.col("shift").median().alias("median_abs_shift"),
        pl.col("shift").quantile(0.9, interpolation="linear").alias("p90_abs_shift"),
        (pl.col("shift") > threshold).mean().alias(f"share_shift_gt_{threshold:g}"),
        pl.col("ratio").median().alias("median_scale_ratio"))
    order = {s.id: i for i, s in enumerate(specs)}
    return [{**r, "refined_share_of_providers": refined}
            for r in sorted(out.to_dicts(), key=lambda r: order[r["feature"]])]


def _label_frame(paths: Paths) -> tuple[pl.DataFrame, str | None]:
    """In-cohort (npi, year) rows with the primary-tier label y at the primary W, if labels exist."""
    frame = pl.scan_parquet(paths.processed / "cohort.parquet").filter("in_cohort").select("npi", "year")
    cfg = labels.LabelConfig.load(paths)
    lab = paths.processed / f"labels_{cfg.primary_tier}_W{cfg.W}.parquet"
    if not lab.is_file():
        return frame.collect().sort("npi", "year"), None
    y = pl.scan_parquet(lab).select("npi", pl.col("year").cast(pl.Int32), "y")
    return (frame.join(y, on=["npi", "year"], how="left").collect().sort("npi", "year"), paths.rel(lab))


def split_checks(frame: pl.DataFrame, folds: pl.DataFrame,
                 cfg: splits.SplitConfig) -> tuple[list[dict], dict[str, list[str]]]:
    """Every design x fold: role counts (and positives if `frame` has y) and violations, checked against the
    folds table and the design years. Returns (rows, {"roles": [...], "npis": [...]}): `roles` = rows whose role
    breaks the fold / year rules (leakage item 3), `npis` = NPIs on both sides of a split or test rows not
    scored exactly once (item 4)."""
    splits.check_folds(folds, cfg)
    has_y = "y" in frame.columns
    frame = frame.select("npi", "year", *(["y"] if has_y else []))
    fk = folds.select("npi", pl.col("fold").alias("_fold"))
    holds = {k: splits.holdout_npis(folds, k, cfg) for k in range(cfg.n_folds)}
    n_out = {k: int((folds["fold"] != k).sum()) for k in range(cfg.n_folds)}
    rows, bad = [], {"roles": [], "npis": []}
    for d in cfg.all_designs:
        test_rows = 0
        for k in range(cfg.n_folds):
            r = splits.assign_roles(frame, folds, k, d, cfg, holdout=holds[k], check=False).join(
                fk, on="npi", how="left")
            role, yr, fold = pl.col("role"), pl.col("year"), pl.col("_fold")
            train = role.is_in(list(splits.TRAIN_ROLES))
            viol = {
                "test outside fold k or test years": (role == "test") & ((fold != k) | ~yr.is_between(*d.test)),
                "training role inside fold k or outside train years": train & ((fold == k) | ~yr.is_between(*d.train)),
                "tune_valid outside the last train year": (role == "tune_valid") & (yr != d.last_train_year),
                "tune_train in or after the last train year": (role == "tune_train") & (yr >= d.last_train_year),
            }
            pos = (pl.col("y") == 1) if has_y else pl.lit(False)
            c = r.select(*[e.sum().alias(n) for n, e in viol.items()],
                         (role == "test").sum().alias("test_rows"), train.sum().alias("train_set_rows"),
                         (role == "tune_train").sum().alias("tune_train_rows"),
                         (role == "tune_valid").sum().alias("tune_valid_rows"),
                         (pos & (role == "test")).sum().alias("test_pos"), (pos & train).sum().alias("train_set_pos"),
                         (pos & (role == "tune_valid")).sum().alias("tune_valid_pos")).row(0, named=True)
            bad["roles"] += [f"{d.name} fold {k}: {n} rows: {c[n]}" for n in viol if c[n]]
            per_npi = r.group_by("npi").agg((role == "test").any().alias("te"), train.any().alias("tr"),
                                            (role == "tune_train").any().alias("tt"),
                                            (role == "tune_valid").any().alias("tv"))
            ov = per_npi.select((pl.col("te") & pl.col("tr")).sum().alias("o"),
                                (pl.col("tt") & pl.col("tv")).sum().alias("t")).row(0, named=True)
            if ov["o"]:
                bad["npis"].append(f"{d.name} fold {k}: {ov['o']} NPIs both test and training")
            if ov["t"]:
                bad["npis"].append(f"{d.name} fold {k}: {ov['t']} NPIs in both tune_train and tune_valid")
            row = {"design": d.name, "fold": k, **{n: int(c[n]) for n in ("test_rows", "train_set_rows",
                                                                         "tune_train_rows", "tune_valid_rows")},
                   "holdout_share": holds[k].len() / n_out[k] if n_out[k] else None,
                   "test_train_overlap_npis": int(ov["o"]), "tune_overlap_npis": int(ov["t"])}
            if has_y:
                row.update({n: int(c[n]) for n in ("test_pos", "train_set_pos", "tune_valid_pos")})
            rows.append(row)
            test_rows += row["test_rows"]
        expected = int(frame["year"].is_between(*d.test).sum())
        if test_rows != expected:
            bad["npis"].append(f"{d.name}: test rows over all folds {test_rows} != rows in test years {expected}")
    return rows, bad


# ----------------------------------------------------------------------------------------------- leakage

def _forbidden_uses(path: Path) -> list[str]:
    """Imports of label-bearing modules and non-docstring string constants naming label artefacts or LEIE."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docs.add(id(first.value))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module] + [f"{node.module}.{m}" for m in mods]
            for m in mods:
                if set(m.split(".")) & set(FORBIDDEN_IMPORTS):
                    hits.append(f"{path.name}:{node.lineno} imports {m}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            low = node.value.lower()
            hits += [f"{path.name}:{node.lineno} names {w!r}" for w in FORBIDDEN_NAMES if w in low]
    return hits


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def rebuild_check(paths: Paths, year: int) -> dict:
    """Rebuild `year`'s features in a scratch project holding only the configs, that year's harmonised files
    (+ the NPPES extract, if any) and that year's cohort rows; compare with the persisted tables."""
    persisted = [paths.processed / "features" / f"{year}.parquet",
                 *[paths.processed / "peer" / str(year) / f"{t}.parquet" for t in ("codes", "stats", "pm_models")]]
    missing = [paths.rel(p) for p in persisted if not p.is_file()]
    if missing:
        return {"status": NOT_RUN, "detail": f"persisted tables missing: {missing}"}
    with tempfile.TemporaryDirectory(prefix=".qa_rebuild_", dir=paths.data) as td:
        tp = Paths(Path(td) / "proj")
        (tp.root / "design").mkdir(parents=True)
        (tp.root / "pyproject.toml").write_text("[project]\nname = 'qa-rebuild'\n", encoding="utf-8")
        shutil.copytree(paths.configs, tp.configs)
        for ds in features.DATASETS:
            src = paths.interim / ds / f"{year}.parquet"
            if src.is_file():
                _link(src.resolve(), tp.interim / ds / src.name)
        for src in sorted((paths.interim / "nppes").glob("*.parquet")) if (paths.interim / "nppes").is_dir() else []:
            _link(src.resolve(), tp.interim / "nppes" / src.name)
        tp.processed.mkdir(parents=True)
        (pl.scan_parquet(paths.processed / "cohort.parquet").filter(pl.col("year") == year).collect()
         .write_parquet(tp.processed / "cohort.parquet"))
        # Point implicit root discovery (Paths.discover, load_config without paths) at the scratch project, so the
        # rebuild cannot read the real project's labels or LEIE through a path it did not receive.
        env = os.environ.get(ROOT_ENV)
        os.environ[ROOT_ENV] = str(tp.root)
        try:
            features.build_features(tp, [year])
        except Exception as exc:     # any failure of the label-free rebuild is an L1 failure, not a QA crash
            return {"status": FAIL, "detail": f"{year}: the rebuild without labels, LEIE and other years fails: "
                                              f"{type(exc).__name__}: {exc}"}
        finally:
            if env is None:
                os.environ.pop(ROOT_ENV, None)
            else:
                os.environ[ROOT_ENV] = env
        differ = [paths.rel(p) for p in persisted
                  if not pl.read_parquet(p).equals(pl.read_parquet(tp.root / p.relative_to(paths.root)))]
    if differ:
        return {"status": FAIL, "detail": f"{year}: rebuilt without labels, LEIE and other years, these tables "
                                          f"differ from the persisted ones: {differ}"}
    return {"status": PASS, "detail": f"{year}: features and peer tables rebuilt from that year's harmonised files "
                                      "and cohort rows only (no labels, LEIE or other years) are identical"}


def peer_stats_check(paths: Paths, years: list[int]) -> dict:
    """Every stats.parquet row's n must equal the number of reference providers (in-cohort, not injected) of its
    stat cell whose value is defined (fit_on rule of features.yaml), recomputed from the persisted features."""
    fcfg = features.FeatureConfig.load(paths)
    coh = pl.scan_parquet(paths.processed / "cohort.parquet")
    has_inj = "injected" in coh.collect_schema().names()
    n_inj = int(coh.select(pl.col("injected").sum()).collect().item() or 0) if has_inj else 0
    bad, checked = [], 0
    for y in years:
        sp = paths.processed / "peer" / str(y) / "stats.parquet"
        if not sp.is_file():
            bad.append(f"{y}: {paths.rel(sp)} missing")
            continue
        ref = coh.filter((pl.col("year") == y) & pl.col("in_cohort"))
        if has_inj:
            ref = ref.filter(~pl.col("injected"))
        ref = ref.select("npi", *[features.CELL_COL[lv] for lv in features.PEER_LEVELS]).with_columns(
            pl.lit(str(y)).alias(features.CELL_COL["national"]))
        feat = pl.scan_parquet(paths.processed / "features" / f"{y}.parquet")
        R = ref.join(feat.select("npi", "DI0", *[f"{features.RAW}{s.id}" for s in fcfg.robust]), on="npi",
                     how="inner").collect()
        n_ref = ref.select(pl.len()).collect().item()
        if R.height != n_ref:
            bad.append(f"{y}: {n_ref - R.height} reference providers have no feature row")

        def defined(s: features.FeatureSpec) -> pl.Expr:
            v = pl.col(f"{features.RAW}{s.id}")
            ok = v.is_not_null() & v.is_finite()
            return (ok & (pl.col("DI0") == 0)) if s.fit_on == "dme_referrers" else ok

        exp = pl.concat([
            R.group_by(features.CELL_COL[lv]).agg(*[defined(s).sum().alias(s.id) for s in fcfg.robust])
            .rename({features.CELL_COL[lv]: "stat_cell"}).with_columns(pl.lit(lv).alias("level"))
            .unpivot(index=["level", "stat_cell"], variable_name="feature", value_name="n_expected")
            for lv in features.LEVELS])
        st = pl.read_parquet(sp).join(exp, on=["level", "stat_cell", "feature"], how="left").with_columns(
            pl.col("n_expected").fill_null(0))
        mism = st.filter(pl.col("n") != pl.col("n_expected"))
        checked += st.height
        if mism.height:
            ex = mism.select("peer_cell", "feature", "level", "n", "n_expected").head(3).to_dicts()
            bad.append(f"{y}: {mism.height} of {st.height} peer statistics have n != reference count, e.g. {ex}")

        # Code reference (SM1, SM2, CH4): each cell's n_cell = its reference members, n_providers <= n_cell.
        members = pl.concat([R.group_by(features.CELL_COL[lv]).len().rename({features.CELL_COL[lv]: "peer_cell"})
                             .with_columns(pl.lit(lv).alias("level")) for lv in features.LEVELS])
        cp = sp.with_name("codes.parquet")
        if cp.is_file():
            codes = (pl.scan_parquet(cp).group_by("level", "peer_cell")
                     .agg(pl.col("n_cell").min().alias("lo"), pl.col("n_cell").max().alias("hi"),
                          (pl.col("n_providers") > pl.col("n_cell")).sum().alias("over")).collect()
                     .join(members, on=["level", "peer_cell"], how="left").with_columns(pl.col("len").fill_null(0)))
            cm = codes.filter((pl.col("lo") != pl.col("len")) | (pl.col("hi") != pl.col("len")) | (pl.col("over") > 0))
            checked += codes.height
            if cm.height:
                bad.append(f"{y}: {cm.height} of {codes.height} code-reference cells (codes.parquet) are not fitted on "
                           f"exactly their reference providers, e.g. {cm.head(3).to_dicts()}")
        else:
            bad.append(f"{y}: {paths.rel(cp)} missing")

        # PM2/PM3 regressions: each fit cell's n = its reference members with a defined outcome.
        pp = sp.with_name("pm_models.parquet")
        if pp.is_file():
            fits = pl.read_parquet(pp).select("target", "level", "fit_cell", "n").unique()
            exp_pm = pl.concat([
                R.group_by(features.CELL_COL[lv]).agg(*[pl.col(f"{features.RAW}{t}").is_not_null().sum().alias(t)
                                                       for t in features.PM_TARGETS])
                .rename({features.CELL_COL[lv]: "fit_cell"}).with_columns(pl.lit(lv).alias("level"))
                .unpivot(index=["level", "fit_cell"], variable_name="target", value_name="n_expected")
                for lv in features.LEVELS])
            pm = fits.join(exp_pm, on=["target", "level", "fit_cell"], how="left").with_columns(
                pl.col("n_expected").fill_null(0))
            pmm = pm.filter(pl.col("n") != pl.col("n_expected"))
            checked += pm.height
            if pmm.height:
                bad.append(f"{y}: {pmm.height} of {pm.height} PM2/PM3 fits (pm_models.parquet) are not fitted on exactly "
                           f"their reference providers, e.g. {pmm.head(3).to_dicts()}")
        else:
            bad.append(f"{y}: {paths.rel(pp)} missing")
    cell_bad, cells_checked = peer_cell_check(paths)
    return {"injected_rows": n_inj, "has_injected_column": has_inj, "checked": checked + cells_checked,
            "problems": bad + cell_bad}


def peer_cell_check(paths: Paths) -> tuple[list[str], int]:
    """cohort.parquet: every n_cell_* equals the number of counted providers (in-cohort, not injected) with that
    cell id, and peer_cell / peer_level follow the DESIGN §7.2 rule under the config the cohort was built with."""
    f = paths.processed / "cohort.parquet"
    meta_p = f.with_name("cohort.meta.json")
    conf = (json.loads(meta_p.read_text(encoding="utf-8")).get("config") or {}) if meta_p.is_file() else {}
    peer = load_config("experiment", paths)["cohort"]["peer"]
    refine, min_cell = conf.get("refine_min", peer["refine_min"]), conf.get("min_cell", peer["min_cell"])
    lf = pl.scan_parquet(f)
    c = lf.filter(cohort.counted(lf))
    df = lf.select("npi", "year", "peer_cell", "peer_level", *[f"cell_{lv}" for lv in PEER_LEVELS],
                   *[f"n_cell_{lv}" for lv in PEER_LEVELS])
    for lv in PEER_LEVELS:
        col = f"cell_{lv}"
        df = df.join(c.filter(pl.col(col).is_not_null()).group_by(col).agg(pl.len().cast(pl.Int64).alias(f"_n_{lv}")),
                     on=col, how="left").with_columns(
            pl.when(pl.col(col).is_not_null()).then(pl.col(f"_n_{lv}").fill_null(0)).alias(f"_n_{lv}"))
    level = (pl.when(pl.col("_n_spec_ent_div") >= refine).then(pl.lit("spec_ent_div"))
             .when(pl.col("_n_spec_ent") >= min_cell).then(pl.lit("spec_ent"))
             .when(pl.col("cell_ent").is_not_null()).then(pl.lit("ent")))
    out = df.with_columns(level.alias("_level")).select(
        pl.len().alias("rows"),
        *[(pl.col(f"n_cell_{lv}").ne_missing(pl.col(f"_n_{lv}"))).sum().alias(f"n_cell_{lv}") for lv in PEER_LEVELS],
        pl.col("peer_level").ne_missing(pl.col("_level")).sum().alias("peer_level"),
        pl.col("peer_cell").ne_missing(
            pl.when(pl.col("_level") == "spec_ent_div").then(pl.col("cell_spec_ent_div"))
            .when(pl.col("_level") == "spec_ent").then(pl.col("cell_spec_ent")).otherwise(pl.col("cell_ent"))
        ).sum().alias("peer_cell")).collect().row(0, named=True)
    rows = out.pop("rows")
    wrong = {k: v for k, v in out.items() if v}
    return ([f"cohort.parquet: rows whose cell sizes or peer cell do not match the counted (in-cohort, non-injected) "
             f"providers: {wrong}"] if wrong else []), rows


def _item(iid: str, checks: list[dict], pending: str | None = None, note: str | None = None) -> dict:
    """Item status: fail if a check fails; pass if all pass; partial if some pass and a part is pending (or
    not run); pending if nothing can be checked yet; not_run otherwise. `note` = re-check reminder."""
    st = [c["status"] for c in checks]
    if FAIL in st:
        status = FAIL
    elif st and all(s == PASS for s in st):
        status = PARTIAL if pending else PASS
    elif any(s == PASS for s in st):
        status = PARTIAL
    else:
        status = PENDING if pending else NOT_RUN
    return {"id": iid, "item": dict(CHECKLIST)[iid], "status": status, "checks": checks, "pending": pending,
            "note": note}


def leakage_checklist(paths: Paths, years: list[int], split_bad: dict | None, split_error: str | None,
                      rebuild: bool, rebuild_year: int | None) -> list[dict]:
    pkg = Path(__file__).resolve().parent
    items = []

    # L1
    hits = [h for m in FEATURE_MODULES for h in _forbidden_uses(pkg / m)]
    l1 = [{"check": f"modules {', '.join(FEATURE_MODULES)} import no label module and name no label artefact",
           "status": FAIL if hits else PASS, "detail": "; ".join(hits) or "none found"}]
    if not years:
        l1.append({"check": "rebuild without labels / LEIE / other years", "status": NOT_RUN,
                   "detail": "no feature tables (run phase2 features)"})
    elif not rebuild:
        l1.append({"check": "rebuild without labels / LEIE / other years", "status": NOT_RUN,
                   "detail": "disabled (qa.leakage_rebuild false or --no-rebuild)"})
    else:
        y = rebuild_year if rebuild_year is not None else years[0]
        r = rebuild_check(paths, y) if y in years else {"status": NOT_RUN, "detail": f"no feature table for {y}"}
        l1.append({"check": f"features of {y} rebuilt without labels, LEIE and other years are identical", **r})
    items.append(_item("L1", l1))

    # L2
    if years:
        ps = peer_stats_check(paths, years)
        l2 = [{"check": "peer statistics (stats n), code reference (codes n_cell), PM2/PM3 fits (pm_models n) and "
                        "cohort cell sizes count exactly the non-injected in-cohort providers of each cell",
               "status": FAIL if ps["problems"] else PASS,
               "detail": "; ".join(ps["problems"]) or f"{ps['checked']} statistics over {len(years)} years match"},
              {"check": "injected providers in the cohort", "status": PASS,
               "detail": f"{ps['injected_rows']} rows flagged injected" + (
                   "" if ps["injected_rows"] else " (none exist before Phase 5)")}]
    else:
        l2 = [{"check": "peer statistics", "status": NOT_RUN, "detail": "no feature tables"}]
    items.append(_item("L2", l2, note="re-run after Phase 5 adds injected providers"))

    # L3, L4
    if split_error:
        sc = [{"check": "split roles for every design and fold", "status": FAIL, "detail": split_error}]
        items += [_item("L3", sc, pending="model, calibration, threshold and tuning fits: Phase 3"), _item("L4", sc)]
    elif split_bad is None:
        sc = [{"check": "split roles for every design and fold", "status": NOT_RUN,
               "detail": "no folds (run phase2 folds)"}]
        items += [_item("L3", sc, pending="model, calibration, threshold and tuning fits: Phase 3"), _item("L4", sc)]
    else:
        role_bad, npi_bad = split_bad["roles"], split_bad["npis"]
        items.append(_item("L3", [{
            "check": "training / tuning roles only in training folds and years; tune_valid only in the last train "
                     "year; every train year precedes every test year (config validated)",
            "status": FAIL if role_bad else PASS, "detail": "; ".join(role_bad) or "all designs and folds"}],
            pending="model, calibration, threshold and tuning fits are checked in Phase 3"))
        items.append(_item("L4", [{
            "check": "no NPI both test and training (or tune_train and tune_valid) in any design x fold; each test "
                     "provider-year scored by exactly one fold", "status": FAIL if npi_bad else PASS,
            "detail": "; ".join(npi_bad) or "all designs and folds"}]))

    # L5
    items.append(_item("L5", [{"check": "injection engine (C6)", "status": PENDING, "detail": "Phase 5"}],
                       pending="Phase 5"))

    # L6
    hashes = protocol_hashes(paths)
    later = later_outputs(paths)
    freeze = load_freeze(paths)
    prev = (freeze or {}).get("protocol_hashes")
    changed = [k for k in FREEZE_KEYS if prev and prev.get(k) != hashes[k]]
    when = f" recorded {freeze['recorded_utc']}" if freeze else ""
    if not later:
        st, det = PASS, "no Phase 3-7 outputs exist, so no test result can have been seen"
        if changed:
            det += f"; changed since the freeze record{when} (the next `phase2 report` re-records it): {changed}"
    elif prev and not changed:
        st, det = PASS, f"unchanged since the freeze{when} (outputs/phase2/{FREEZE_FILE}), made before any Phase 3-7 output"
    else:
        st, det = FAIL, (f"{len(later)} later-phase output files exist (e.g. {later[:2]}) and "
                         + (f"{changed} changed since the freeze{when}: log the change and its reason in DESIGN §17 "
                            "(EXPERIMENT_PLAN header)" if prev else
                            f"no freeze was recorded before them (outputs/phase2/{FREEZE_FILE} missing)"))
    items.append(_item("L6", [
        {"check": "hashes recorded", "status": PASS,
         "detail": ", ".join(f"{k} {hashes[k][:12]}" for k in FREEZE_KEYS)},
        {"check": "fixed before any test result", "status": st, "detail": det}]))
    return items


def later_outputs(paths: Paths) -> list[str]:
    """Files under outputs/phase3 ... phase7 (their existence means test results may have been seen)."""
    return sorted(paths.rel(p) for d in LATER_PHASES if (paths.outputs / d).is_dir()
                  for p in (paths.outputs / d).rglob("*") if p.is_file())


def load_freeze(paths: Paths) -> dict | None:
    """The protocol freeze record (outputs/phase2/G2_freeze.json), written by `phase2 report` only while no Phase 3-7
    output exists and never overwritten afterwards; None if absent."""
    f = paths.phase2 / FREEZE_FILE
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


# ----------------------------------------------------------------------------------------------- report

def phase2_qa(paths: Paths, *, rebuild: bool | None = None, rebuild_year: int | None = None) -> Path:
    """Write outputs/phase2/QA_report.md and QA_report.json from whatever Phase 2 artefacts exist."""
    exp = load_config("experiment", paths)
    qcfg = exp["qa"]
    rebuild = bool(qcfg["leakage_rebuild"]) if rebuild is None else rebuild
    rebuild_year = rebuild_year if rebuild_year is not None else qcfg.get("leakage_rebuild_year")
    qs = [float(q) for q in qcfg["quantiles"]]
    prov = data_provenance(paths)
    data: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(), "provenance": prov}
    md = ["# Phase 2 QA report: cohort, features, folds", "", *provenance_lines(prov),
          f"Generated {data['generated_utc']} by `scenario-fraud phase2 qa`. Regenerate after any change to the "
          "cohort, features, folds or configs. This report states facts; G2 decisions belong to the human "
          "(HANDOFF §6).", ""]

    coh = cohort_section(paths)
    data["cohort"] = coh
    md += ["## 1. Cohort per year (DESIGN §7.2)", ""]
    if coh is None:
        md += ["_Not available: run `scenario-fraud phase2 cohort`._", ""]
    else:
        if coh["stale"]:
            md += [f"**Warning:** {coh['stale']}.", ""]
        md += [md_table(coh["counts"]), "## 2. Peer-level mix of in-cohort providers", "",
               "Providers and cells per peer level; `undersized_ent_cells` = entity-type cells below `peer.min_cell` "
               "(no further fallback).", "", md_table(coh["peer_mix"])]

    years = feature_years(paths)
    data["feature_years"] = years
    md += ["## 3. Features per year (FEATURE_CATALOG)", ""]
    if not years:
        md += ["_Not available: run `scenario-fraud phase2 features`._", ""]
        fs = {}
    else:
        fs = feature_section(paths, years, qs)
        data["features"] = {str(y): v for y, v in fs.items()}
        md += ["Quantiles of the raw statistic r (`raw__{ID}`) of robust-z features and of context_raw features; "
               "`undefined_share` = r undefined (zero denominator; z set to the peer median); `clipped_share` = "
               "|z| at the clip.", ""]
        for y in years:
            md += [f"### {y} ({fs[y]['rows']} providers, {fs[y]['model_columns']} model columns)", "",
                   md_table(fs[y]["features"])]
        md += ["## 4. Undefined and structural-zero rates", "",
               md_table([{"year": y, **fs[y]["structural_zero_shares"]} for y in years]),
               "Undefined share of r by feature and year:", "",
               md_table([{"feature": f, **{str(y): next((x["undefined_share"] for x in fs[y]["features"]
                                                        if x["feature"] == f), None) for y in years}}
                         for f in dict.fromkeys(x["feature"] for y in years for x in fs[y]["features"])])]
        md += ["## 5. Data coverage CX8 (share of services surviving suppression)", "",
               md_table([{"year": y, **{k: v for x in fs[y]["features"] if x["feature"] == "CX8"
                                        for k, v in x.items() if k not in ("feature", "kind")}} for y in years])]
        nd = near_duplicates(paths, years)
        data["near_duplicates"] = nd
        rho = features.FeatureConfig.load(paths).prune_rho
        md += [f"## 6. Near-duplicate pairs (|Spearman ρ| > {rho} within a group, model columns)", "",
               "Candidates only: pruning is decided at G2 and recorded in `configs/features.yaml` `pruned`.", "",
               md_table(nd) if nd else "_none_\n"]

    thr = float(qcfg["region_shift_threshold"])
    md += ["## 7. Census-division refinement (DESIGN §19 Q2: is region needed?)", ""]
    if not years or coh is None:
        md += ["_Not available: needs the cohort and feature tables._", ""]
        data["region"] = None
    else:
        reg = region_section(paths, years, thr)
        data["region"] = reg
        md += ["For providers whose peer cell is a Census-division cell (spec × ent × division, ≥ `peer.refine_min` "
               "providers): `abs_shift` = |division-cell median − spec × ent cell median| of the raw statistic in units "
               "of the spec × ent cell's robust scale, i.e. how far a typical member's z would move if region "
               f"refinement were dropped; `share_shift_gt_{thr:g}` = share of those providers whose cell moves by more "
               f"than {thr:g} (`qa.region_shift_threshold`); `median_scale_ratio` = division-cell scale / spec × ent "
               "scale (< 1: region makes cells more homogeneous). Reference providers, feature years pooled; "
               "reference-dependent features (SM1, SM2, CH4, PM2, PM3) are left out. Input to the G2 decision on "
               "peer granularity; sensitivity S10 runs the other choice.", "",
               md_table(reg) if reg else "_No provider has a division-level peer cell._\n"]

    md += ["## 8. Folds and splits (DESIGN §7.6; EXPERIMENT_PLAN §2)", ""]
    split_bad, split_err, fold_info = None, None, None
    fpath = paths.processed / "folds.parquet"
    if not fpath.is_file():
        md += ["_Not available: run `scenario-fraud phase2 folds`._", ""]
    elif coh is None:
        md += ["_Not available: cohort missing._", ""]
    else:
        try:
            scfg = splits.SplitConfig.load(paths)
            folds = pl.read_parquet(fpath)
            frame, lab = _label_frame(paths)
            rows, split_bad = split_checks(frame, folds, scfg)
            bal = splits.fold_balance(folds, scfg.n_folds)
            fold_info = {"balance": bal, "splits": rows, "violations": split_bad, "labels": lab}
            md += [f"{bal['npis']} NPIs in {scfg.n_folds} folds over {bal['strata']} strata (modal specialty × "
                   f"ever-positive, any tier); largest per-stratum spread across folds {bal['max_stratum_spread']}, "
                   f"fold-size spread {bal['fold_size_spread']}, ever-positive spread {bal['ever_positive_spread']}.",
                   "", md_table(bal["per_fold"]),
                   "Rows per role and design × fold" + (f" (positives `y = 1` from `{lab}`)" if lab else
                                                        " (no label file: positives not shown)")
                   + ". `train_set_rows` = all training-set roles (train, tune_train, tune_valid); how the final model "
                   "and its calibrator divide them is decided in Phase 3 (DESIGN §17).",
                   "", md_table(rows)]
        except splits.SplitError as exc:
            split_err = str(exc)
            md += [f"**Split check failed:** {exc}", ""]
    data["folds"] = fold_info

    items = leakage_checklist(paths, years, split_bad, split_err, rebuild, rebuild_year)
    data["leakage_checklist"] = items
    data["protocol_hashes"] = protocol_hashes(paths)
    md += ["## 9. Leakage checklist (FEATURE_CATALOG)", "",
           md_table([{"id": i["id"], "item": i["item"], "status": i["status"],
                       "pending / note": i["pending"] or i["note"] or ""} for i in items]), ""]
    for i in items:
        md += [f"**{i['id']}** {i['item']}", ""]
        md += [f"- [{c['status']}] {c['check']}: {c['detail']}" for c in i["checks"]] + [""]

    out = paths.phase2 / "QA_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    (paths.phase2 / "QA_report.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return out
