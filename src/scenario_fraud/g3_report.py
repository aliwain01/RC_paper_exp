"""Gate G3 report (HANDOFF §6 G3; EXPERIMENT_PLAN §3, §9 T1; CONTRACTS §3.5).

build_report(paths) -> outputs/phase3/G3_report.md (+ .json). From the out-of-fold test scores of every trained
design: validation average precision per family (how "best GBM" and "best glass box" are chosen, never on test
data), T1 for the reference models (seed 0) with NPI-bootstrap CIs, the paired bootstrap of Δ = best GBM − best
glass box with Holm correction across k, the H1 check, seed spread, per-year results, rolling origins, calibration
in the top 5 % and the label-horizon caveat. It states which DESIGN §10.8 row the detection numbers are
consistent with; the human decides the branch (G3).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from . import metrics, models, qa, splits, train
from .config import load_config
from .paths import Paths

REF_SEED = 0


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _ci(v, ci):
    return f"{_fmt(v)} [{_fmt(ci[0])}, {_fmt(ci[1])}]"


def _families_with_scores(paths: Paths, design: str) -> list[str]:
    d = paths.processed / "scores" / design
    return sorted(p.name for p in d.iterdir() if p.is_dir() and any(p.glob("fold*.parquet"))) if d.is_dir() else []


def choose_best(tuning: pl.DataFrame, candidates: list[str]) -> tuple[str | None, dict]:
    """Family with the highest mean validation average precision over folds (ties: models.yaml order)."""
    if tuning.is_empty():
        return None, {}
    t = tuning.filter(pl.col("family").is_in(candidates)).group_by("family").agg(
        pl.col("valid_ap").mean().alias("mean_valid_ap"), pl.len().alias("folds"))
    if t.is_empty():
        return None, {}
    scores = {r["family"]: r["mean_valid_ap"] for r in t.to_dicts()}
    best = max(candidates, key=lambda f: (scores.get(f, float("-inf")), -candidates.index(f)))
    return (best if best in scores else None), scores


def _ref(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("seed") == REF_SEED).select("npi", "year", "y", "margin", "prob")


def design_section(paths: Paths, design: str, mcfg: models.ModelConfig, ev: dict, *, full: bool) -> dict | None:
    fams = [f for f in mcfg["families"] if f in _families_with_scores(paths, design)]
    if not fams:
        return None
    k_fracs = [float(k) for k in ev["k_fracs"]]
    tuning = train.load_tuning(paths, design)
    out: dict = {"design": design, "families": fams, "validation": tuning.to_dicts() if not tuning.is_empty() else []}
    best = {}
    for role in ("glass_box", "gbm", "gbm_mono"):
        best[role], out[f"valid_ap_{role}"] = choose_best(tuning, [f for f in mcfg[role] if f in fams])
    out["best"] = best
    scores = {f: train.load_scores(paths, design, f) for f in fams}
    t1 = {}
    for i, f in enumerate(fams):
        ref = _ref(scores[f])
        det = metrics.detection_metrics(ref, k_fracs, reliability_top=float(ev["reliability_top"]),
                                        reliability_bins=int(ev["reliability_bins"]))
        row = {"point": det}
        if full:
            row["ci"] = metrics.bootstrap_ci(ref, k_fracs, int(ev["bootstrap_reps"]), int(ev["bootstrap_seed"]) + i,
                                             float(ev["ci"]))
        seeds = scores[f]["seed"].unique().sort().to_list()
        if len(seeds) > 1:
            per = [metrics.detection_metrics(scores[f].filter(pl.col("seed") == s).select(
                "npi", "year", "y", "margin", "prob"), k_fracs)["pooled"] for s in seeds]
            row["seed_spread"] = {m: float(pl.Series([p[m] for p in per]).std())
                                  for m in ("pr_auc", f"p@{ev['primary_k']}")}
        t1[f] = row
    out["t1"] = t1
    a, b = best["gbm"], best["glass_box"]
    if full and a and b:
        boot = metrics.paired_bootstrap(_ref(scores[a]), _ref(scores[b]), k_fracs, int(ev["bootstrap_reps"]),
                                        int(ev["bootstrap_seed"]), float(ev["ci"]))
        hk = metrics.holm({f"p@{k}": boot[f"p@{k}"]["p_value"] for k in k_fracs}, float(ev["alpha"]))
        pk = f"p@{ev['primary_k']}"
        h1 = (boot["pr_auc"]["delta_ci"][0] > 0) and (boot[pk]["delta_ci"][0] > 0) and hk[pk]["reject"]
        out["delta"] = {"a": a, "b": b, "bootstrap": boot, "holm": hk, "h1_supported": bool(h1)}
    horizon = pl.concat([scores[fams[0]].filter(pl.col("seed") == REF_SEED)]).group_by("year").agg(
        pl.len().alias("rows"), (~pl.col("label_horizon_complete")).sum().alias("horizon_incomplete"),
        pl.col("y").sum().alias("positives")).sort("year") if "label_horizon_complete" in scores[fams[0]].columns \
        else pl.DataFrame()
    out["horizon"] = horizon.to_dicts()
    return out


def _branch(sec: dict) -> str:
    d = sec.get("delta")
    if not d:
        return "Not assessed (the best GBM or best glass box is missing)."
    if d["h1_supported"]:
        return ("Δ PR-AUC and Δ precision@1 % both have CIs above 0 (Holm-adjusted): consistent with the DESIGN §10.8 "
                "rows that start \"GBM better\". Which of them applies depends on the explanation results (Phases 4-6).")
    return ("H1 is not supported (a CI includes 0 or lies below it): consistent with the DESIGN §10.8 row \"No "
            "significant difference\" if the gap is not significant, or with a glass-box advantage if Δ < 0.")


def build_report(paths: Paths) -> Path:
    mcfg = models.ModelConfig.load(paths)
    exp = load_config("experiment", paths)
    ev = exp["evaluation"]
    scfg = splits.SplitConfig.load(paths)
    prov = qa.data_provenance(paths)
    gen = datetime.now(timezone.utc).isoformat()
    data: dict = {"generated_utc": gen, "provenance": prov, "designs": {}}
    freeze = qa.load_freeze(paths)
    md = ["# Gate G3 report: detection results (end of Phase 3)", "", *qa.provenance_lines(prov),
          f"Generated {gen} by `scenario-fraud phase3 report`. HANDOFF §6 G3: detection results with CIs; the human "
          "decides which DESIGN §10.8 / §16 branch the paper follows. Rankings use raw margins; probabilities are "
          "isotonic-calibrated (models.yaml calibration_source). T1 uses the reference model (seed 0) of each family.",
          "", f"- G2 freeze record: {'recorded ' + freeze['recorded_utc'] if freeze else '**missing**'}",
          f"- Final fit / calibration: `{mcfg['final_fit']}` / `{mcfg['calibration_source']}` (DESIGN §17 D59, open)",
          ""]
    ks = [float(k) for k in ev["k_fracs"]]
    for design in [*scfg.designs, "showcase"]:
        if design == "showcase":
            fams = _families_with_scores(paths, "showcase")
            if fams:
                n = train.load_scores(paths, "showcase", fams[0]).filter(pl.col("seed") == REF_SEED).height
                md += ["## Showcase", "", f"{scfg.showcase_year}: {n} provider-years scored by the primary models of "
                       f"their own fold ({', '.join(fams)}). No accuracy claims (DESIGN §7.6).", ""]
                data["designs"]["showcase"] = {"families": fams, "rows": n}
            continue
        sec = design_section(paths, design, mcfg, ev, full=(design == "primary"))
        if sec is None:
            continue
        data["designs"][design] = sec
        d = scfg.design(design)
        md += [f"## Design {design}: train {d.train[0]}–{d.train[1]}, test {d.test[0]}–{d.test[1]}", ""]
        md += ["Validation average precision (mean over folds of the tuning model on `tune_valid`), used to choose "
               "the best models:", ""]
        val = {f: v for role in ("glass_box", "gbm", "gbm_mono") for f, v in sec[f"valid_ap_{role}"].items()}
        md += [qa.md_table([{"family": f, "mean_valid_ap": _fmt(val.get(f)),
                             "chosen_as": ", ".join(r for r, b in sec["best"].items() if b == f) or ""}
                            for f in sec["families"]])]
        rows = []
        for f in sec["families"]:
            p = sec["t1"][f]["point"]["pooled"]
            ci = sec["t1"][f].get("ci")
            row = {"model": f, "PR-AUC": _ci(p["pr_auc"], ci["pr_auc"]["ci"]) if ci else _fmt(p["pr_auc"]),
                   "ROC-AUC": _fmt(p["roc_auc"])}
            for k in ks:
                row[f"P@{k:.1%}"] = _ci(p[f"p@{k}"], ci[f"p@{k}"]["ci"]) if ci else _fmt(p[f"p@{k}"])
            row["R@1.0%"] = _fmt(p.get(f"r@{ev['primary_k']}"))
            row["Brier"] = _fmt(p.get("brier"), 5)
            sp = sec["t1"][f].get("seed_spread")
            row["seed SD (PR-AUC)"] = _fmt(sp["pr_auc"], 4) if sp else "–"
            rows.append(row)
        md += [f"**T1 ({'pooled ' + '–'.join(map(str, d.test)) if len(d.test_years) > 1 else d.test[0]}, "
               f"{'95 % NPI-bootstrap CIs, ' if design == 'primary' else ''}reference models)**", "",
               qa.md_table(rows)]
        if "delta" in sec:
            dl = sec["delta"]
            drows = [{"metric": m, "Δ": _fmt(v["delta"]), "CI": f"[{_fmt(v['delta_ci'][0])}, {_fmt(v['delta_ci'][1])}]",
                      "p": _fmt(v["p_value"]), "p (Holm over k)": _fmt(dl["holm"].get(m, {}).get("p_holm"))}
                     for m, v in dl["bootstrap"].items() if m == "pr_auc" or m.startswith("p@")]
            md += [f"**Δ = best GBM (`{dl['a']}`) − best glass box (`{dl['b']}`)**, paired NPI bootstrap "
                   f"({ev['bootstrap_reps']} replicates):", "", qa.md_table(drows),
                   f"- H1 (Δ PR-AUC and Δ precision@{float(ev['primary_k']):.0%} CIs > 0, Holm across k): "
                   f"**{'supported' if dl['h1_supported'] else 'not supported'}**", f"- {_branch(sec)}", ""]
        per = []
        for f in sec["families"]:
            for y, p in sec["t1"][f]["point"]["per_year"].items():
                per.append({"model": f, "year": y, "N": p["N"], "positives": p["positives"],
                            "base rate": _fmt(p["base_rate"], 5), "PR-AUC": _fmt(p["pr_auc"]),
                            f"P@{float(ev['primary_k']):.0%}": _fmt(p[f"p@{ev['primary_k']}"])})
        md += ["Per test year:", "", qa.md_table(per)]
        if sec["horizon"]:
            md += ["Label horizon (DESIGN §5.4: rows whose W-year window is not fully observed; 2023 has ≈ 2.8 of 3 "
                   "years with an October 2026 snapshot):", "", qa.md_table(sec["horizon"])]
        if design == "primary":
            for role in ("gbm", "glass_box"):
                f = sec["best"].get(role)
                rel = sec["t1"][f]["point"].get("reliability_top") if f else None
                if rel:
                    md += [f"Calibration within the top {float(ev['reliability_top']):.0%} (`{f}`):", "",
                           qa.md_table([{k2: _fmt(v, 4) if isinstance(v, float) else v for k2, v in r.items()}
                                        for r in rel])]
    if not data["designs"]:
        md += ["_No scores found: run `scenario-fraud phase3 train` first._", ""]
    out = paths.phase3 / "G3_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    return out
