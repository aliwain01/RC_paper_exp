"""Gate G5 report (HANDOFF §6 G5; EXPERIMENT_PLAN §6; CONTRACTS §5).

build_report(paths) -> outputs/phase5/G5_report.md (+ .json): calibration (peer percentile of each scheme's primary
feature per intensity, current and proposed "medium"), the injection plan against its targets, pools, distractor
shares, re-suppression effects, the feature-shift profile of injected test providers by group (spill-over), the
leakage checks for injected data (checklist L2 and L5) and the injection-model row counts. The human accepts the
calibrated intensity grid (G5).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from . import features, injection, qa
from .groups import GROUP_ORDER
from .paths import Paths


def _draws(paths: Paths) -> list[int]:
    root = paths.processed / "injection"
    return sorted(int(p.name[4:]) for p in root.glob("draw*") if (p / "plan.parquet").is_file()) if root.is_dir() else []


def leakage_checks(paths: Paths, draw: int, icfg: injection.InjectionConfig) -> list[dict]:
    d = injection.draw_dir(paths, draw)
    plan = pl.read_parquet(d / "plan.parquet")
    out = []
    sup = icfg["suppression"]
    for ds, key in (("cms_ps", "cms_ps"), ("dme_s", "dme_s")):
        files = sorted((d / "edited" / ds).glob("*.parquet"))
        bad = sum(pl.read_parquet(f).filter(pl.col(sup[key]["field"]) <= int(sup[key]["max_suppressed"])).height
                  for f in files)
        out.append({"id": "L5", "check": f"edited {ds} rows re-suppressed ({sup[key]['field']} > "
                                         f"{sup[key]['max_suppressed']})", "status": "pass" if bad == 0 else "fail",
                    "detail": f"{bad} rows at or below the threshold"})
    foreign = 0
    for ds in features.DATASETS:
        for f in sorted((d / "edited" / ds).glob("*.parquet")):
            y = int(f.stem)
            planned = plan.filter(pl.col("year") == y)["npi"]
            foreign += pl.read_parquet(f).filter(~pl.col("npi").is_in(planned.implode())).height
    out.append({"id": "L5", "check": "only planned providers' rows edited", "status": "pass" if foreign == 0 else "fail",
                "detail": f"{foreign} edited rows of unplanned NPIs"})
    bad_n, cells = 0, 0
    for yd in sorted((d / "peer").glob("*")):
        coh = pl.read_parquet(yd / "cohort.parquet")
        ref = coh.filter(pl.col("in_cohort") & ~pl.col("injected"))
        st = pl.read_parquet(yd / "stats.parquet")
        for level in ("spec_ent_div", "spec_ent", "ent", "national"):
            sl = st.filter(pl.col("level") == level)
            if sl.is_empty():
                continue
            counts = (ref.with_columns(pl.lit(yd.name).alias("stat_cell")).select("stat_cell") if level == "national"
                      else ref.select(pl.col(f"cell_{level}").alias("stat_cell"))).group_by("stat_cell").len()
            j = sl.join(counts, on="stat_cell", how="left")
            bad_n += j.filter(pl.col("n") > pl.col("len").fill_null(0)).height
            cells += j.height
        for lvl in ("ent", "spec_ent", "spec_ent_div"):          # cohort cell sizes exclude injected providers
            n_ref = ref.group_by(f"cell_{lvl}").len()
            jj = coh.select(f"cell_{lvl}", f"n_cell_{lvl}").unique().drop_nulls().join(n_ref, on=f"cell_{lvl}", how="left")
            bad_n += jj.filter(pl.col(f"n_cell_{lvl}") != pl.col("len").fill_null(0)).height
    out.append({"id": "L2", "check": "peer statistics and cell sizes count only non-injected providers",
                "status": "pass" if bad_n == 0 else "fail", "detail": f"{bad_n} violations over {cells} statistic cells"})
    return out


def shift_profile(paths: Paths, draw: int, icfg: injection.InjectionConfig) -> list[dict]:
    """Mean change in each group's mean direction-up z (injected minus the same provider-year's clean features) for
    injected test providers, by scheme set."""
    fcfg = features.FeatureConfig.load(paths)
    up = {g: [s.id for s in fcfg.robust if s.group == g and s.direction == "up"] for g in GROUP_ORDER[1:]}
    plan = pl.read_parquet(injection.draw_dir(paths, draw) / "plan.parquet").filter(pl.col("part") == "test")
    years = sorted(plan["year"].unique().to_list())
    inj = injection.load_injected_features(paths, draw, years)
    cl = features.load_features(paths, years)
    rows = []
    for (key,), g in plan.with_columns(pl.col("schemes").list.join("+").alias("k")).group_by("k"):
        a = g.select("npi", "year").join(inj, on=["npi", "year"]).sort("npi", "year")
        b = g.select("npi", "year").join(cl, on=["npi", "year"]).sort("npi", "year")
        row = {"schemes": key, "n": a.height}
        for grp, cols in up.items():
            cols = [c for c in cols if c in a.columns and c in b.columns]
            if cols:
                row[grp] = round(float((a.select(cols).mean_horizontal() - b.select(cols).mean_horizontal()).mean()), 3)
        rows.append(row)
    return sorted(rows, key=lambda r: r["schemes"])


def build_report(paths: Paths) -> Path:
    icfg = injection.InjectionConfig.load(paths)
    prov = qa.data_provenance(paths)
    gen = datetime.now(timezone.utc).isoformat()
    md = ["# Gate G5 report: scheme injection (Phase 5)", "", *qa.provenance_lines(prov),
          f"Generated {gen} by `scenario-fraud phase5 report`. HANDOFF §6 G5: injection intensities actually needed; the "
          "human accepts the calibrated grid. Use (a) and use (b) results are evaluated in Phase 6.", ""]
    data: dict = {"generated_utc": gen, "provenance": prov}
    cp = paths.outputs / "phase5" / "calibration.json"
    if cp.is_file():
        cal = json.loads(cp.read_text(encoding="utf-8"))
        data["calibration"] = cal
        md += [f"## Calibration ({cal['year']}; target: median peer percentile ≥ {cal['target_percentile']})", ""]
        for s, v in cal["schemes"].items():
            md += [f"**{s}** (primary feature {v['primary_feature']}; current low / medium / high = "
                   f"{v['current']['low']} / {v['current']['medium']} / {v['current']['high']}; proposed medium = "
                   f"{v['proposed_medium'] if v['proposed_medium'] is not None else '**none on the grid reaches the target**'})",
                   "", qa.md_table(v["grid"])]
    else:
        md += ["_No calibration: run `phase5 calibrate`._", ""]
    draws = _draws(paths)
    data["draws"] = {}
    for d in draws:
        dd = injection.draw_dir(paths, d)
        plan = pl.read_parquet(dd / "plan.parquet")
        cnt = (plan.with_columns(pl.col("schemes").list.join("+").alias("scheme"))
               .group_by("part", "scheme", "intensity").agg(pl.len().alias("planned"), pl.col("cell_target").first(),
                                                            pl.col("distractor").mean().alias("distractor_share"))
               .sort("part", "scheme", "intensity"))
        short = cnt.filter(pl.col("planned") < pl.col("cell_target"))
        entry = {"plan": cnt.to_dicts(), "shortfalls": short.to_dicts(),
                 "leakage": leakage_checks(paths, d, icfg)}
        sp = dd / "summary.json"
        if sp.is_file():
            entry["build"] = json.loads(sp.read_text(encoding="utf-8"))
        if (dd / "features").is_dir():
            entry["shift"] = shift_profile(paths, d, icfg)
        sc = sorted((dd / "scores").glob("*/fold*.parquet"))
        if sc:
            s = pl.concat([pl.read_parquet(f).with_columns(pl.lit(f.parent.name).alias("family")) for f in sc])
            entry["models"] = s.group_by("family").agg(pl.len().alias("test_rows"), pl.col("y").sum().alias(
                "test_injected")).sort("family").to_dicts()
        data["draws"][d] = entry
        md += [f"## Draw {d}", "", "Planned injections (test: per scheme × intensity and per pair; train: ≈ "
               f"{icfg['train']['share']:.1%} of training provider-years split over the same cells):", "",
               qa.md_table(entry["plan"])]
        if entry["shortfalls"]:
            md += ["**Cells below target** (too few applicable clean provider-years):", "", qa.md_table(entry["shortfalls"])]
        if "build" in entry:
            md += ["Edits and re-suppression per year:", "",
                   qa.md_table([{"year": y, **v} for y, v in entry["build"]["years"].items()])]
        if "shift" in entry:
            md += ["Feature-shift profile of injected test providers (mean change of each group's mean direction-up z; "
                   "the answer groups should move most, spill-over groups somewhat):", "",
                   qa.md_table(entry["shift"], ["schemes", "n", *GROUP_ORDER[1:]])]
        md += ["Leakage checks on the injected data (FEATURE_CATALOG checklist L2, L5):", "",
               qa.md_table(entry["leakage"])]
        if "models" in entry:
            md += ["Injection models (use b): test rows and injected test rows per family:", "", qa.md_table(entry["models"])]
    if not draws:
        md += ["_No injection draws: run `phase5 build`._", ""]
    md += ["## Decisions for the human (G5)", "",
           "- Accept or change the intensity grid in `configs/injection.yaml` (calibration above).",
           "- The answer key and scheme transforms (DESIGN §17 D90-D97, proposed).",
           "- The DMEPOS re-suppression field (beneficiaries vs claims; DESIGN §18).", ""]
    out = paths.outputs / "phase5" / "G5_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    return out
