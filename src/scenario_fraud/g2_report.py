"""Gate G2 report (HANDOFF §6 G2; CONTRACTS §2.4): what the human needs to freeze the protocol before Phase 3.

build_report(paths) -> outputs/phase2/G2_report.md (+ G2_report.json):
  1. the final feature list after `pruned` (configs/features.yaml) with group, kind, direction, monotone
     constraint and the model columns each feature produced;
  2. features that could not be computed, or were structurally empty in a year, and why;
  3. near-duplicate candidates for the pruning decision (from the QA report);
  4. the protocol diff: every DESIGN §17 row added after the v1.1 baseline, plus the parameters and config
     hashes to freeze;
  5. the leakage checklist (from the QA report) and the decisions the human takes at G2, incl. the open DESIGN §19
     questions (HANDOFF §5 Q7).
While no Phase 3-7 output exists it (re)writes the freeze record outputs/phase2/G2_freeze.json (protocol hashes);
once one exists the record is never rewritten, so leakage item 6 compares against the pre-results protocol.
It reports; the human decides (HANDOFF §6). Run `phase2 qa` first.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


from . import features
from . import groups as groups_mod
from .config import file_sha256, load_config
from .paths import Paths
from .qa import (FREEZE_FILE, FREEZE_KEYS, data_provenance, feature_years, later_outputs, load_freeze, md_table,
                 protocol_hashes, provenance_lines)

CONFIG_FILES = ("experiment.yaml", "features.yaml", "groups.yaml", "labels.yaml", "census_divisions.yaml")
DME_INPUTS = ("dme_s", "dme_p")
_ROW = re.compile(r"^\|\s*D(\d+)\s*\|(.*)\|\s*$")


def decision_rows(paths: Paths) -> list[dict] | None:
    """Rows of DESIGN_DOCUMENT §17 (Decision log) as {number, id, cells}; None if the document is absent."""
    f = paths.root / "design" / "DESIGN_DOCUMENT.md"
    if not f.is_file():
        return None
    rows, inside = [], False
    for line in f.read_text(encoding="utf-8").splitlines():
        if re.match(r"^##\s+17\.", line):
            inside = True
            continue
        if inside and re.match(r"^##\s", line):
            break
        m = _ROW.match(line) if inside else None
        if m:
            rows.append({"number": int(m.group(1)), "id": f"D{m.group(1)}",
                         "cells": [c.strip() for c in m.group(2).split("|")]})
    return rows


def open_questions(paths: Paths) -> list[str] | None:
    """Numbered items of DESIGN_DOCUMENT §19 (Open questions); None if the document is absent."""
    f = paths.root / "design" / "DESIGN_DOCUMENT.md"
    if not f.is_file():
        return None
    out, inside = [], False
    for line in f.read_text(encoding="utf-8").splitlines():
        if re.match(r"^##\s+19\.", line):
            inside = True
            continue
        if inside and re.match(r"^##\s", line):
            break
        m = re.match(r"^\s*(\d+)\.\s+(.*\S)\s*$", line) if inside else None
        if m:
            out.append(f"Q{m.group(1)} {m.group(2)}")
    return out


def _metas(paths: Paths, years: list[int]) -> dict[int, dict]:
    out = {}
    for y in years:
        p = paths.processed / "features" / f"{y}.meta.json"
        out[y] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    return out


def feature_list(paths: Paths, metas: dict[int, dict]) -> tuple[list[dict], list[str]]:
    """One row per catalog feature, and the union of model columns over the feature years."""
    cfg = features.FeatureConfig.load(paths)
    cols = sorted({c for m in metas.values() for c in m.get("model_columns", [])})
    nc = {fid for m in metas.values() for fid in (m.get("not_computable") or {})}
    rows = []
    for s in cfg.specs:
        own = [c for c in cols if c == s.id or c.startswith(f"{s.id}__")]
        years = [y for y, m in metas.items() if any(c == s.id or c.startswith(f"{s.id}__")
                                                   for c in m.get("model_columns", []))]
        status = ("pruned (features.yaml)" if s.id in cfg.pruned else
                  "not computable" if s.id in nc else
                  "model feature" if years else "no feature table yet")
        rows.append({"id": s.id, "name": s.name, "group": s.group, "kind": s.kind, "direction": s.direction,
                     "monotone": s.monotone, "shrink": s.shrink, "suppression": s.suppression,
                     "model_columns": len(own), "years": len(years), "status": status})
    return rows, cols


def not_computable(metas: dict[int, dict], qa: dict | None) -> list[dict]:
    """Features absent (meta `not_computable`), structurally empty (an input file missing in a year), or
    undefined for every provider of a year (QA)."""
    rows: dict[tuple, dict] = {}

    def add(fid: str, why: str, y: int) -> None:
        rows.setdefault((fid, why), {"feature": fid, "why": why, "years": []})["years"].append(y)

    for y, m in metas.items():
        for fid, why in (m.get("not_computable") or {}).items():
            add(fid, why, y)
        missing = [ds for ds in DME_INPUTS if (m.get("inputs_present") or {}).get(ds) is False]
        if missing:
            add("DI0-DI4, DC1-DC2", f"{', '.join(missing)} not harmonised for the year: every provider has DI0 = 1 "
                                    "and raw DI1-DI4, DC1-DC2 = 0", y)
    for y, f in ((qa or {}).get("features") or {}).items():
        for r in f["features"]:
            if r["undefined_share"] == 1:
                add(r["feature"], "r undefined for every provider (z = 0 for all)", int(y))
    return [{**r, "years": ",".join(map(str, sorted(r["years"])))} for r in rows.values()]


def parameters(paths: Paths) -> list[dict]:
    """The protocol parameters frozen at G2, with their config source."""
    exp, feat, lab = (load_config(n, paths) for n in ("experiment", "features", "labels"))
    s, c, t = exp.get("splits", {}), exp["cohort"], feat["transform"]
    out = [
        ("years train / test / showcase / flagged", [exp["years"]["train"], exp["years"]["test"],
                                                     exp["years"]["showcase"], exp["years"]["flagged"]], "experiment.yaml"),
        ("cohort.min_nondrug_services", c["min_nondrug_services"], "experiment.yaml"),
        ("cohort.peer refine_min / min_cell", [c["peer"]["refine_min"], c["peer"]["min_cell"]], "experiment.yaml"),
        ("cohort.ruca_classes", c["ruca_classes"], "experiment.yaml"),
        ("labels primary_tier / W / W_sensitivity", [lab["primary_tier"], lab["W"], lab["W_sensitivity"]], "labels.yaml"),
        ("labels other_excluded_policy", lab["other_excluded_policy"], "labels.yaml"),
        ("splits n_folds / seed / valid_frac", [s.get("n_folds"), s.get("seed"), s.get("valid_frac")], "experiment.yaml"),
        ("splits designs", s.get("designs"), "experiment.yaml"),
        ("splits showcase", s.get("showcase"), "experiment.yaml"),
        ("negative subsample rate / sensitivity / seed",
         [s.get("negative_subsample", {}).get(k) for k in ("rate", "sensitivity_rates", "seed")], "experiment.yaml"),
        ("transform z_clip / mad_constant / scale_fallbacks", [t["z_clip"], t["mad_constant"], t["scale_fallbacks"]],
         "features.yaml"),
        ("peer_stats.min_defined", feat["peer_stats"]["min_defined"], "features.yaml"),
        ("shrinkage alpha_min / alpha_max / alpha_max_median_n",
         [feat["shrinkage"][k] for k in ("alpha_min", "alpha_max", "alpha_max_median_n")], "features.yaml"),
        ("SM2 rare_provider_share", feat["atypical_mix"]["rare_provider_share"], "features.yaml"),
        ("CH4 min providers cell / national", [feat["charge_index"]["min_providers_cell"],
                                               feat["charge_index"]["min_providers_national"]], "features.yaml"),
        ("PM2/PM3 min_n / min_n_per_term", [feat["patient_mix"]["min_n"], feat["patient_mix"]["min_n_per_term"]],
         "features.yaml"),
        ("high-risk DME list (status, ranges)", [feat["dme_high_risk"].get("status"),
                                                 [f"{r['from']}-{r['to']}" for r in feat["dme_high_risk"]["hcpcs_ranges"]]],
         "features.yaml"),
        ("pruned", feat.get("pruned") or [], "features.yaml"),
        ("near-duplicate threshold abs(rho)", feat["prune"]["spearman_abs"], "features.yaml"),
    ]
    return [{"parameter": p, "value": json.dumps(v, default=str), "config": f} for p, v, f in out]


def build_report(paths: Paths) -> Path:
    exp = load_config("experiment", paths)
    base = int(exp["g2"]["baseline_last_decision"])
    version = exp["g2"]["protocol_version"]
    fcfg = features.FeatureConfig.load(paths)
    prov = data_provenance(paths)
    qa_p = paths.phase2 / "QA_report.json"
    qa = json.loads(qa_p.read_text(encoding="utf-8")) if qa_p.is_file() else None
    years = feature_years(paths)
    metas = _metas(paths, years)
    data: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(), "provenance": prov, "feature_years": years}
    md = ["# Gate G2 report: freeze the protocol before Phase 3", "", *provenance_lines(prov),
          f"Generated {data['generated_utc']} by `scenario-fraud phase2 report`. HANDOFF §6 G2: final feature list "
          "after pruning, features that could not be computed, protocol diff. The human freezes the protocol; this "
          "report only states facts.", ""]
    if qa is None:
        md += ["**The QA report is missing: run `scenario-fraud phase2 qa` first.** Sections that need it are "
               "marked.", ""]

    rows, cols = feature_list(paths, metas)
    data["features"], data["model_columns"] = rows, cols
    in_model = [r for r in rows if r["status"] == "model feature"]
    md += ["## 1. Final feature list (after `pruned`)", ""]
    if not years:
        md += ["_No feature tables yet: run `scenario-fraud phase2 features`. The list below is the configured "
               "catalog._", ""]
    else:
        gm = groups_mod.feature_groups(cols, paths=paths)
        per_group = {g: sum(1 for v in gm.values() if v == g) for g in groups_mod.GROUP_ORDER}
        data["model_columns_per_group"] = per_group
        md += [f"{len(in_model)} catalog features in the model ({len(cols)} model columns over {years[0]}-{years[-1]} "
               f"after one-hot expansion; pruned: {list(fcfg.pruned) or 'none'}). Model columns per group: "
               + ", ".join(f"{g} {n}" for g, n in per_group.items()) + ".", ""]
    md += [md_table(rows)]

    nc = not_computable(metas, qa)
    data["not_computable"] = nc
    md += ["## 2. Features that could not be computed, and why", "", md_table(nc) if nc else "_none_\n"]

    nd = (qa or {}).get("near_duplicates")
    data["near_duplicates"] = nd
    md += [f"## 3. Near-duplicate candidates (|Spearman ρ| > {fcfg.prune_rho} within a group)", ""]
    md += (["_QA report missing._", ""] if qa is None else [md_table(nd) if nd else "_none_\n"])

    drows = decision_rows(paths)
    md += [f"## 4. Protocol diff against v{version}", "",
           f"### 4.1 DESIGN §17 rows added after D{base}", ""]
    if drows is None:
        new = None
        md += ["_design/DESIGN_DOCUMENT.md not found._", ""]
    else:
        new = [r for r in drows if r["number"] > base]
        md += [md_table([{"#": r["id"], "decision": r["cells"][0], "reason / status": " | ".join(r["cells"][1:])}
                          for r in new]) if new else
               f"_No rows after D{base}: every Phase 2 implementation choice still has to be logged in §17._\n"]
    data["decisions_after_baseline"] = new
    par = parameters(paths)
    hashes = protocol_hashes(paths)
    files = {f: file_sha256(paths.configs / f) for f in CONFIG_FILES if (paths.configs / f).is_file()}
    data.update(parameters=par, protocol_hashes=hashes, config_sha256=files)
    later = later_outputs(paths)
    freeze = load_freeze(paths)
    if not later:       # before any result: (re)record the freeze
        freeze = {"recorded_utc": data["generated_utc"], "protocol_hashes": hashes, "config_sha256": files}
        paths.phase2.mkdir(parents=True, exist_ok=True)
        (paths.phase2 / FREEZE_FILE).write_text(json.dumps(freeze, indent=2), encoding="utf-8")
        fz = {"status": "recorded", "recorded_utc": freeze["recorded_utc"], "changed": []}
        fz_md = (f"Recorded in `outputs/phase2/{FREEZE_FILE}` ({freeze['recorded_utc']}). No Phase 3-7 output exists, "
                 "so the record is rewritten by every report until the first one appears, and never afterwards.")
    elif freeze is None:
        fz = {"status": "missing", "recorded_utc": None, "changed": list(FREEZE_KEYS)}
        fz_md = (f"**No freeze record**: {len(later)} Phase 3-7 output files exist and `outputs/phase2/{FREEZE_FILE}` "
                 "was never written before them, so leakage item 6 cannot pass; it is not written now.")
    else:
        changed = [k for k in FREEZE_KEYS if freeze["protocol_hashes"].get(k) != hashes[k]]
        fz = {"status": "kept", "recorded_utc": freeze["recorded_utc"], "changed": changed}
        fz_md = (f"Freeze recorded {freeze['recorded_utc']} kept unchanged (Phase 3-7 outputs exist). "
                 + (f"**Changed since the freeze: {changed}.**" if changed else "Nothing frozen has changed since."))
    data["freeze"] = fz
    md += ["### 4.2 Parameters to freeze", "", md_table(par),
           "### 4.3 Hashes (leakage checklist item 6; QA compares later runs with the freeze record)", "", fz_md, "",
           md_table([{"what": k, "sha256": v} for k, v in hashes.items()]
                     + [{"what": f"configs/{k}", "sha256": v} for k, v in files.items()])]

    items = (qa or {}).get("leakage_checklist")
    data["leakage_checklist"] = items
    md += ["## 5. Leakage checklist (from the QA report)", ""]
    md += (["_QA report missing._", ""] if not items else
           [md_table([{"id": i["id"], "item": i["item"], "status": i["status"],
                        "pending / note": i.get("pending") or i.get("note") or ""} for i in items])])

    todo = []
    if (load_config("features", paths)["dme_high_risk"].get("status") or "") != "frozen":
        todo.append("Freeze the high-risk DME list (features.yaml `dme_high_risk.status` is not `frozen`; "
                    "leakage item 6) and cite its DOJ/OIG sources.")
    if nd and not fcfg.pruned:
        todo.append(f"Decide the pruning of {len(nd)} near-duplicate pair(s) (features.yaml `pruned`).")
    for r in nc:
        todo.append(f"Accept or remedy: {r['feature']} ({r['why']}; {r['years']}).")
    if new is not None and not new:
        todo.append(f"Log the Phase 2 implementation choices as DESIGN §17 rows after D{base} before freezing.")
    if new and any("proposed" in " ".join(r["cells"]).lower() for r in new):
        todo.append("Confirm or change the §17 rows marked proposed.")
    if new and any("(open" in " ".join(r["cells"]).lower() for r in new):
        todo.append("Decide the §17 rows marked open, e.g. which training rows the final fold model and its isotonic "
                    "calibrator use (DESIGN §7.8 'calibration on the validation slice').")
    todo.append("Decide peer granularity (DESIGN §19 Q2, 'is region needed?'; decided empirically in Phase 2) from "
                "QA report §7 (shift of division-cell medians against the spec × entity cell); sensitivity S10 runs "
                "the other choice.")
    qs = open_questions(paths)
    if qs:
        todo.append("Answer the open DESIGN §19 questions that change the protocol before the freeze (HANDOFF §5 "
                    "Q7): " + "; ".join(qs) + ".")
    if fz["status"] == "kept" and fz["changed"]:
        todo.append(f"{fz['changed']} changed after the freeze: log each change and its reason in DESIGN §17 "
                    "(EXPERIMENT_PLAN header); leakage item 6 fails until then.")
    elif fz["status"] == "missing":
        todo.append("Phase 3-7 outputs exist without a freeze record: report this deviation (leakage item 6 fails).")
    for i in items or []:
        if i["status"] == "fail":
            todo.append(f"Resolve the failing leakage check {i['id']}.")
    if qa is None:
        todo.append("Run `scenario-fraud phase2 qa` and regenerate this report.")
    todo.append("Freeze the protocol (EXPERIMENT_PLAN header): no test-year result may be looked at before.")
    data["decisions_needed"] = todo
    md += ["## 6. Decisions needed at G2", "", *[f"{n}. {t}" for n, t in enumerate(todo, 1)], ""]

    out = paths.phase2 / "G2_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    (paths.phase2 / "G2_report.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return out
