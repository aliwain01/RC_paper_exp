"""Gate G4 report (HANDOFF §6 G4; EXPERIMENT_PLAN §4; CONTRACTS §4).

build_report(paths) -> outputs/phase4/G4_report.md (+ .json): the explained set per test year, the timing benchmark
with projected wall-clock per family and baseline against the compute budget, the background levels actually used,
the EXPERIMENT_PLAN §4 additivity checks on the attribution store, and the path-dependent diagnostics. The human
sets |B| and the explained-set size (G4).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from . import qa
from .attribute import M, load_attributions
from .config import load_config
from .explained_set import explained_path
from .paths import Paths

TOL = {"B": 1e-6, "C": 1e-6, "A": 1e-4, "D": 1e-6}   # EXPERIMENT_PLAN §4 (A: interventional TreeSHAP, 1e-4)


def store_checks(paths: Paths, acfg: dict) -> list[dict]:
    rows = []
    for fam in acfg["coalition_families"]:
        for base in acfg["baselines"]:
            try:
                a = load_attributions(paths, fam, base)
            except FileNotFoundError:
                continue
            V = np.array(a["v"].to_list())
            tot = V[:, -1] - V[:, 0]
            row = {"family": fam, "baseline": base, "providers": a.height,
                   "bg_levels": dict(sorted(a.group_by("bg_level").len().iter_rows())),
                   "bg_n_min": int(a["bg_n"].min()), "bg_n_median": float(a["bg_n"].median())}
            checks = {"B": np.array(a["B"].to_list()).sum(1),
                      "C": np.array(a["C"].to_list()).sum(1)}
            if a["A"].null_count() == 0:
                checks["A"] = np.array(a["A"].to_list()).sum(1)
            if a["D"].null_count() == 0:
                checks["D"] = np.array(a["D"].to_list()).sum(1)
            ok = True
            for k, v in checks.items():
                err = float(np.abs(v - tot).max())
                row[f"max_err_{k}"] = err
                ok &= err < TOL[k]
            if "A" in checks:      # D126: providers whose method A is not exact (excluded from method A comparisons)
                row["A_inexact"] = int((np.abs(checks["A"] - tot) >= TOL["A"]).sum())
            row["additivity"] = "pass" if ok else "FAIL"
            rows.append(row)
    return rows


def build_report(paths: Paths) -> Path:
    acfg = load_config("attribution", paths)
    prov = qa.data_provenance(paths)
    gen = datetime.now(timezone.utc).isoformat()
    md = ["# Gate G4 report: attribution timing and store checks (Phase 4)", "", *qa.provenance_lines(prov),
          f"Generated {gen} by `scenario-fraud phase4 report`. HANDOFF §6 G4: predictions per second and the projected "
          "wall-clock of the coalition tables; the human sets |B| and the explained-set size.", ""]
    data: dict = {"generated_utc": gen, "provenance": prov}
    ep = explained_path(paths)
    if ep.is_file():
        es = pl.read_parquet(ep)
        comp = es.group_by("year").agg(pl.len().alias("explained"), pl.col("top").sum().alias("top"),
                                       pl.col("random").sum().alias("random"),
                                       pl.col("injected").sum().alias("injected")).sort("year")
        data["explained"] = comp.to_dicts()
        md += ["## Explained set", "", f"Top {float(acfg['explained']['top_frac']):.0%} by the out-of-fold margin of "
               f"`{es['score_family'][0]}` (best GBM on validation) ∪ stratified random "
               f"{acfg['explained']['random_n']} ∪ injected test providers (Phase 5).", "", qa.md_table(comp.to_dicts())]
    else:
        md += ["_No explained set: run `phase4 explained-set`._", ""]
    bp = paths.outputs / "phase4" / "bench.json"
    if bp.is_file():
        bench = json.loads(bp.read_text(encoding="utf-8"))
        data["bench"] = bench
        opts = [*map(str, acfg["bench"]["B_options"]), "global"]
        rows = []
        for fam, r in bench["families"].items():
            row = {"family": fam, "pred/s": f"{r['pred_per_s']:,.0f}"}
            row.update({f"hours |B|={o}" if o != "global" else "hours global": round(r["projections_hours"][o], 2)
                        for o in opts})
            row["SHAP hours (B=200)"] = round(r["shap_hours_peer_B200"], 2) if "shap_hours_peer_B200" in r else "–"
            rows.append(row)
        md += ["## Timing benchmark and projections", "",
               f"{bench['explained_total']} explained provider-years; coalition table = 2^{M} subsets × |B| "
               f"predictions per provider. Budget set in attribution.yaml: {bench['budget_hours']} h.", "",
               qa.md_table(rows),
               "Totals by baseline choice (all families):", "",
               qa.md_table([{"setting": o, "hours": round(sum(r["projections_hours"][o] for r in
                                                              bench["families"].values()), 2),
                             "within budget": sum(r["projections_hours"][o] for r in bench["families"].values())
                             <= bench["budget_hours"]} for o in opts])]
    else:
        md += ["_No benchmark: run `phase4 bench`._", ""]
    checks = store_checks(paths, acfg)
    data["store_checks"] = checks
    md += ["## Attribution store: EXPERIMENT_PLAN §4 additivity checks", "",
           "Σ Φᴮ, Σ M (method C), Σ Φᴬ and Σ D against v(N) − v(∅) per provider (A tolerance 1e-4, others 1e-6).",
           "", qa.md_table([{k: v for k, v in r.items() if k != "bg_levels"} for r in checks],
                           [c for c in ("family", "baseline", "providers", "bg_n_min", "bg_n_median", "max_err_B",
                                        "max_err_C", "max_err_A", "A_inexact", "max_err_D", "additivity")]) if checks
           else "_No attributions yet: run `phase4 explain`._\n",
           "Background levels used (peer baseline falls back up the hierarchy below `min_background`, D82):", "",
           qa.md_table([{"family": r["family"], "baseline": r["baseline"], **r["bg_levels"]} for r in checks],
                       ["family", "baseline", *sorted({lv for r in checks for lv in r["bg_levels"]})])
           if checks else "_none_\n"]
    dp = paths.outputs / "phase4" / "diagnostics.json"
    if dp.is_file():
        diag = json.loads(dp.read_text(encoding="utf-8"))
        data["diagnostics"] = diag
        md += ["## Path-dependent diagnostics (DESIGN §9.4)", "",
               qa.md_table([{"family": f, **r} for f, rows in diag.items() for r in rows])]
    md += ["## Decisions for the human (G4)", "",
           "- |B| (attribution.yaml `background.B`) and the explained-set size (`explained.top_frac`, `random_n`).",
           "- Which families get coalition tables (`coalition_families`; E6 asks for every model × method pair).",
           "- `background.min_background` (D82).", ""]
    out = paths.outputs / "phase4" / "G4_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    return out
