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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import acquire, g1_report, harmonise, labels
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scenario-fraud", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="phase", required=True)
    _p1(sub)
    a = ap.parse_args(argv)
    paths = Paths.discover()
    args = {k: v for k, v in vars(a).items() if k not in ("phase", "cmd")}
    with run_log(paths, a.phase, a.cmd, args) as result:
        result.update(_run_phase1(paths, a))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
