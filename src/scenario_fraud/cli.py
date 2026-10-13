"""C16 entry point: one sub-command group per phase. Every command writes a run log.

    scenario-fraud phase1 register --source cms_ps --year 2023 --kind data FILE [...]
    scenario-fraud phase1 fetch --source leie --year 2026 --kind current --route oig --url URL --as-of 2026-10-12
    scenario-fraud phase1 datagov-list SLUG
    scenario-fraud phase1 wayback-list
    scenario-fraud phase1 wayback-fetch TIMESTAMP [...]
    scenario-fraud phase1 verify
    scenario-fraud phase1 headers [--refresh]
    scenario-fraud phase1 draft-colmap --dataset cms_ps --year 2023
    scenario-fraud phase1 harmonise [--dataset cms_ps --year 2023]
    scenario-fraud phase1 labels
    scenario-fraud phase1 report

    scenario-fraud phase2 cohort [--year 2022 ...]
    scenario-fraud phase2 features [--year 2022 ...]
    scenario-fraud phase2 folds
    scenario-fraud phase2 qa [--no-rebuild] [--rebuild-year 2018]
    scenario-fraud phase2 report
    scenario-fraud phase3 train [--design primary] [--family xgb ...] [--fold 0 ...] [--seeds N] [--trials N]
    scenario-fraud phase3 score-showcase [--family xgb ...]
    scenario-fraud phase3 report
    scenario-fraud phase4 explained-set | explain [--family F ...] [--baseline peer|global] [--fold K ...] [--limit N]
    scenario-fraud phase4 diagnostics | bench [--family F ...] | report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from . import (acquire, cohort, features, g1_report, g2_report, g3_report, harmonise, labels, provenance, qa, splits,
               train, attribute, bench, explained_set, g4_report)
from .paths import Paths
from .runlog import run_log


def _p1(sub: argparse._SubParsersAction) -> None:
    p1 = sub.add_parser("phase1", help="Phase 1: data acquisition (C1-C3) and the G1 report")
    s = p1.add_subparsers(dest="cmd", required=True)

    r = s.add_parser("register", help="register manually downloaded files (route C)")
    r.add_argument("files", nargs="+", type=Path)
    r.add_argument("--source", required=True)
    r.add_argument("--year", type=int, required=True)
    r.add_argument("--kind", required=True)
    r.add_argument("--route", default="manual")
    r.add_argument("--source-page", default="")
    r.add_argument("--file-url", default="manual")
    r.add_argument("--as-of", default=None, help="LEIE: snapshot/supplement/download date (YYYY-MM[-DD])")
    r.add_argument("--move", action="store_true", help="move instead of copy")
    r.add_argument("--notes", default="")

    f = s.add_parser("fetch", help="download one URL taken from a landing page or catalogue listing")
    f.add_argument("--url", required=True)
    f.add_argument("--source", required=True)
    f.add_argument("--year", type=int, required=True)
    f.add_argument("--kind", required=True)
    f.add_argument("--route", required=True)
    f.add_argument("--source-page", default="")
    f.add_argument("--filename", default=None)
    f.add_argument("--as-of", default=None)

    d = s.add_parser("datagov-list", help="route A: list a data.gov entry's distributions")
    d.add_argument("slug")

    s.add_parser("wayback-list", help="list Internet Archive captures of the LEIE file (CDX)")
    w = s.add_parser("wayback-fetch", help="fetch LEIE captures by timestamp (id_ raw bytes)")
    w.add_argument("timestamps", nargs="+")

    s.add_parser("verify", help="recompute every manifest checksum")
    h = s.add_parser("headers", help="dump column names + row counts for every registered table")
    h.add_argument("--refresh", action="store_true")

    c = s.add_parser("draft-colmap", help="draft a column map from a real header dump")
    c.add_argument("--dataset", required=True)
    c.add_argument("--year", type=int, required=True)
    c.add_argument("--force", action="store_true")

    hm = s.add_parser("harmonise", help="raw CSV -> data/interim Parquet via reviewed column maps")
    hm.add_argument("--dataset", default=None)
    hm.add_argument("--year", type=int, default=None)

    s.add_parser("labels", help="build exclusions and provider-year labels (C3)")
    s.add_parser("report", help="write outputs/phase1/G1_report.md")


def _run_phase1(paths: Paths, a: argparse.Namespace) -> dict:
    if a.cmd == "register":
        es = [acquire.register(paths, f, a.source, a.year, a.kind, route=a.route, source_page=a.source_page,
                               file_url=a.file_url, as_of=a.as_of, move=a.move, notes=a.notes) for f in a.files]
        return {"registered": [e.path for e in es]}
    if a.cmd == "fetch":
        e = acquire.fetch(paths, a.url, a.source, a.year, a.kind, route=a.route, source_page=a.source_page,
                          filename=a.filename, as_of=a.as_of)
        return {"fetched": e.path, "sha256": e.sha256, "size_bytes": e.size_bytes}
    if a.cmd == "datagov-list":
        res = acquire.datagov_distributions(paths, a.slug)
        return {"n_resources": len(res), "resources": [{"name": r["name"], "url": r["url"]} for r in res]}
    if a.cmd == "wayback-list":
        snaps = acquire.wayback_snapshots(paths)
        return {"n_snapshots": len(snaps), "timestamps": [s["timestamp"] for s in snaps]}
    if a.cmd == "wayback-fetch":
        return {"fetched": [acquire.fetch_wayback_snapshot(paths, ts).path for ts in a.timestamps]}
    if a.cmd == "verify":
        problems = acquire.verify(paths)
        if problems:
            raise SystemExit("checksum problems:\n" + json.dumps(problems, indent=2))
        return {"verified": len(acquire.load_manifest(paths))}
    if a.cmd == "headers":
        dumps = acquire.dump_all_headers(paths, refresh=a.refresh)
        return {"dumped": [(d["path"], d["n_rows"], d["n_columns"]) for d in dumps]}
    if a.cmd == "draft-colmap":
        return {"draft": paths.rel(harmonise.draft_column_map(paths, a.dataset, a.year, force=a.force))}
    if a.cmd == "harmonise":
        if a.dataset and a.year:
            targets = [(a.dataset, a.year)]
        else:  # every map present, optionally filtered by dataset
            targets = sorted(
                (p.stem.rsplit("_", 1)[0], int(p.stem.rsplit("_", 1)[1]))
                for p in paths.column_maps.glob("*_[0-9][0-9][0-9][0-9].yaml")
            )
            targets = [t for t in targets if a.dataset in (None, t[0])]
        return {"harmonised": [paths.rel(harmonise.harmonise(paths, d, y)) for d, y in targets]}
    if a.cmd == "labels":
        return labels.run_labels(paths)
    if a.cmd == "report":
        return {"report": paths.rel(g1_report.build_report(paths))}
    raise SystemExit(f"unknown command {a.cmd}")


def _p2(sub: argparse._SubParsersAction) -> None:
    p2 = sub.add_parser("phase2", help="Phase 2: cohort (C4), features (C5), folds (C7), QA and the G2 report")
    s = p2.add_subparsers(dest="cmd", required=True)
    c = s.add_parser("cohort", help="cohort + peer cells -> data/processed/cohort.parquet")
    c.add_argument("--year", type=int, action="append", default=None, help="repeatable; default: every year")
    f = s.add_parser("features", help="features + peer statistics -> data/processed/{features,peer}/")
    f.add_argument("--year", type=int, action="append", default=None, help="repeatable; default: every cohort year")
    s.add_parser("folds", help="stratified provider folds -> data/processed/folds.parquet")
    q = s.add_parser("qa", help="write outputs/phase2/QA_report.md (incl. the leakage checklist)")
    q.add_argument("--no-rebuild", action="store_true", help="skip the label-free feature rebuild (checklist L1)")
    q.add_argument("--rebuild-year", type=int, default=None)
    s.add_parser("report", help="write outputs/phase2/G2_report.md")


def _run_phase2(paths: Paths, a: argparse.Namespace) -> dict:
    if a.cmd == "cohort":
        df = cohort.build_cohort(paths, a.year)
        per = df.group_by("year").agg(pl.len().alias("rows"), pl.col("in_cohort").sum().alias("in_cohort")).sort("year")
        return {"cohort": paths.rel(paths.processed / "cohort.parquet"),
                "per_year": {str(r["year"]): {"rows": r["rows"], "in_cohort": r["in_cohort"]} for r in per.to_dicts()}}
    if a.cmd == "features":
        out = features.build_features(paths, a.year)
        return {"features": {str(y): {"rows": m["rows"], "model_columns": len(m["model_columns"]),
                                      "seconds": m["seconds"]} for y, m in out.items()}}
    if a.cmd == "folds":
        folds = splits.assign_folds(paths)
        bal = splits.fold_balance(folds, splits.SplitConfig.load(paths).n_folds)
        return {"folds": paths.rel(paths.processed / "folds.parquet"), "npis": bal["npis"],
                "per_fold": bal["per_fold"], "max_stratum_spread": bal["max_stratum_spread"]}
    if a.cmd == "qa":
        out = qa.phase2_qa(paths, rebuild=False if a.no_rebuild else None, rebuild_year=a.rebuild_year)
        rep = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        return {"report": paths.rel(out), "synthetic_inputs": rep["provenance"]["synthetic"],
                "leakage_checklist": {i["id"]: i["status"] for i in rep["leakage_checklist"]}}
    if a.cmd == "report":
        return {"report": paths.rel(g2_report.build_report(paths))}
    raise SystemExit(f"unknown command {a.cmd}")


def _p3(sub: argparse._SubParsersAction) -> None:
    p3 = sub.add_parser("phase3", help="Phase 3: models (C8), detection metrics and the G3 report")
    s = p3.add_subparsers(dest="cmd", required=True)
    t = s.add_parser("train", help="tune, fit and score one design -> data/processed/{models,scores}/")
    t.add_argument("--design", default="primary", help="primary | O1 | O2 | O3 (rolling origins reuse primary params)")
    t.add_argument("--family", action="append", default=None, help="repeatable; default: models.yaml families")
    t.add_argument("--fold", type=int, action="append", default=None, help="repeatable; default: every fold")
    t.add_argument("--seeds", type=int, default=None, help="override models.yaml seeds for this design")
    t.add_argument("--trials", type=int, default=None, help="override models.yaml n_trials")
    c = s.add_parser("score-showcase", help="score the showcase year with the primary models (no accuracy claims)")
    c.add_argument("--family", action="append", default=None)
    s.add_parser("report", help="write outputs/phase3/G3_report.md")


def _run_phase3(paths: Paths, a: argparse.Namespace) -> dict:
    if a.cmd == "train":
        return train.train_design(paths, a.design, families=a.family, folds=a.fold, seeds=a.seeds, n_trials=a.trials)
    if a.cmd == "score-showcase":
        return train.score_showcase(paths, families=a.family)
    if a.cmd == "report":
        out = g3_report.build_report(paths)
        rep = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        prim = rep["designs"].get("primary", {})
        return {"report": paths.rel(out), "best": prim.get("best"),
                "h1_supported": (prim.get("delta") or {}).get("h1_supported")}
    raise SystemExit(f"unknown command {a.cmd}")


def _p4(sub: argparse._SubParsersAction) -> None:
    p4 = sub.add_parser("phase4", help="Phase 4: attribution (C9-C12), timing benchmark and the G4 report")
    s = p4.add_subparsers(dest="cmd", required=True)
    s.add_parser("explained-set", help="top GBM scores + stratified random sample per test year")
    e = s.add_parser("explain", help="methods A-E for the explained set -> data/processed/attributions/")
    e.add_argument("--family", action="append", default=None)
    e.add_argument("--baseline", action="append", default=None, choices=["peer", "global"])
    e.add_argument("--fold", type=int, action="append", default=None)
    e.add_argument("--limit", type=int, default=None, help="explain at most N providers per fold and year")
    s.add_parser("diagnostics", help="path-dependent TreeSHAP diagnostics on a subsample (DESIGN §9.4)")
    b = s.add_parser("bench", help="timing benchmark -> outputs/phase4/bench.json")
    b.add_argument("--family", action="append", default=None)
    s.add_parser("report", help="write outputs/phase4/G4_report.md")


def _run_phase4(paths: Paths, a: argparse.Namespace) -> dict:
    if a.cmd == "explained-set":
        es = explained_set.build_explained_set(paths)
        return {"explained": paths.rel(explained_set.explained_path(paths)),
                "per_year": {str(y): n for y, n in es.group_by("year").len().sort("year").iter_rows()}}
    if a.cmd == "explain":
        return attribute.explain(paths, families=a.family, baselines=a.baseline, folds=a.fold, limit=a.limit)
    if a.cmd == "diagnostics":
        return attribute.diagnostics(paths)
    if a.cmd == "bench":
        return bench.benchmark(paths, a.family)
    if a.cmd == "report":
        return {"report": paths.rel(g4_report.build_report(paths))}
    raise SystemExit(f"unknown command {a.cmd}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scenario-fraud", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="phase", required=True)
    _p1(sub)
    _p2(sub)
    _p3(sub)
    _p4(sub)
    a = ap.parse_args(argv)
    paths = Paths.discover()
    args = {k: v for k, v in vars(a).items() if k not in ("phase", "cmd")}
    with run_log(paths, a.phase, a.cmd, args) as result:
        result.update({"phase1": _run_phase1, "phase2": _run_phase2, "phase3": _run_phase3,
                       "phase4": _run_phase4}[a.phase](paths, a))
        if a.phase in ("phase2", "phase3", "phase4"):     # numbers in this log describe synthetic data when this is true (HANDOFF §8 r7)
            result["synthetic_inputs"] = provenance.inputs_synthetic(provenance.interim_files(paths.interim))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
