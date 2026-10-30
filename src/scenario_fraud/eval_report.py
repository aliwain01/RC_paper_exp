"""Phase 6 evaluation report (DESIGN §10; EXPERIMENT_PLAN §5, §7-§9; CONTRACTS §6).

build_report(paths) -> outputs/phase6/evaluation_report.md (+ .json): the result table shells T1 (from the G3 report),
T2 correctness, T3 consistency, T4 faithfulness and stability, the use (a) stress test (E5), the baseline comparison
(E8), the sensitivity suite (E9) and the fairness profile (E10), each from outputs/phase6/{part}.json. It states which
DESIGN §10.8 row the numbers are consistent with, by the rule in DESIGN §17 D108; the human decides the branch.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from . import evaluate, qa, sensitivity
from .config import load_config
from .paths import Paths

T2_ROWS = (("gbm", "A"), ("gbm", "B"), ("gbm", "C"), ("gbm", "E"), ("ga2m", "D"), ("ebm", "D"), ("lr", "D"))
LABEL = {"gbm": "GBM", "ga2m": "GA²M", "ebm": "EBM", "lr": "LR"}


def fmt(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    if isinstance(x, bool):
        return str(x)
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def wilson_low(p: float | None, n: int, z: float = 1.96) -> float | None:
    """Lower end of the Wilson score interval of a proportion."""
    if p is None or not n:
        return None
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    return (c - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den


def _family(role: str, best: dict) -> str | None:
    return best.get("gbm") if role == "gbm" else role


def t2_rows(corr: dict, best: dict) -> list[dict]:
    tab = {(r["family"], r["method"]): r for r in corr.get("table", [])}
    rows, chance = [], None
    for role, meth in T2_ROWS:
        fam = _family(role, best)
        r = tab.get((fam, meth))
        if r is None:
            continue
        chance = chance or r
        pair = meth == "C" or (role == "ga2m" and meth == "D")
        rows.append({"model · method": f"{LABEL[role]} ({fam}) · {meth}" if role == "gbm" else f"{LABEL[role]} · {meth}",
                     "n": r["n"], "Top-1": fmt(r["top1"]), "Top-3": fmt(r["top3"]), "MRR": fmt(r["mrr"]),
                     "Distractor top-1": fmt(r["distractor_top1"]), "Compound both-in-top-3": fmt(r["both_top3"]),
                     "Pair hit": fmt(r["pair_hit"]) if pair else "—"})
    if chance:
        rows.append({"model · method": "chance (this answer key)", "n": "", "Top-1": fmt(chance["chance_top1"]),
                     "Top-3": fmt(chance["chance_top3"]), "MRR": fmt(chance["chance_mrr"]),
                     "Distractor top-1": fmt(chance["chance_top1"]), "Compound both-in-top-3": fmt(chance["chance_both_top3"]),
                     "Pair hit": fmt(chance["chance_pair_hit"])})
    return rows


def t3_rows(cons: dict, best: dict, stratum: str = "both") -> list[dict]:
    rows = []
    for r in cons.get("summary", []):
        if r["gbm"] != best.get("gbm") or r["stratum"] != stratum or r["glass_box"] == "lr":
            continue
        rows.append({"GBM method": f"{r['gbm']} · {r['method']}", "vs": r["glass_box"], "n": r["n"],
                     "Sign agree": fmt(r["sign_agree"]["median"]), "Spearman ρ": fmt(r["spearman"]["median"]),
                     "Top-3 Jaccard": fmt(r["jaccard3"]["median"]), "Share cos": fmt(r["cosine"]["median"]),
                     "Permutation baseline (cos)": fmt(r["cosine"]["perm_median"]),
                     "Magnitude ratio": fmt(r["mag_ratio"]["median"])})
    return rows


def t4_rows(faith: dict, stab: dict) -> list[dict]:
    st = {(fam, r["method"]): r for fam, v in (stab or {}).get("families", {}).items() for r in v.get("summary", [])}
    rows = []
    for r in faith.get("summary", []):
        if r["subset"] != "all" or r["baseline"] != "peer":
            continue
        s = st.get((r["family"], r["method"]), {})
        rows.append({"model · method": f"{r['family']} · {r['method']}", "n": r["n"],
                     "Deletion ↓ (median)": fmt(r["median"]), "Random order (median)": fmt(r.get("random_median")),
                     "Reverse order (median)": fmt(r.get("reverse_median")),
                     "Share below random": fmt(r.get("below_random")),
                     "Top-1 agree (seeds)": fmt(s.get("top1_agree_seed")), "Kendall τ (seeds)": fmt(s.get("kendall_seeds")),
                     "Top-1 agree (bootstrap)": fmt(s.get("top1_agree_boot")),
                     "Kendall τ (bootstrap)": fmt(s.get("kendall_boot")),
                     "Top-1 agree (perturbation)": fmt(s.get("top1_agree_perturb"))})
    return rows


def branch(g3: dict | None, corr: dict, cons: dict, best: dict, method: str = "B") -> dict:
    """DESIGN §10.8 row the numbers are consistent with (D108): H1 from G3; 'correct' = the best GBM's method
    (default B, §9.6) has a Wilson 95 % lower bound of top-1 above chance on the T2 rows; 'consistent' = in the
    both-flag stratum its median share cosine with each of EBM and GA²M exceeds the permutation baseline's."""
    h1 = None
    if g3:
        d = (g3.get("designs", {}).get("primary") or {}).get("delta")
        h1 = d.get("h1_supported") if d else None
    row = next((r for r in corr.get("table", []) if r["family"] == best.get("gbm") and r["method"] == method), None)
    correct = None
    if row and row["n"]:
        lo = wilson_low(row["top1"], int(row["n"]))
        correct = bool(lo is not None and lo > row["chance_top1"])
    cs = [r for r in cons.get("summary", []) if r["gbm"] == best.get("gbm") and r["method"] == method
          and r["stratum"] == "both" and r["glass_box"] in ("ebm", "ga2m")]
    consistent = None
    if cs and all(r["cosine"]["median"] is not None and r["cosine"]["perm_median"] is not None for r in cs):
        consistent = all(r["cosine"]["median"] > r["cosine"]["perm_median"] for r in cs)
    if h1 is None:
        text = "H1 not assessed (no G3 result): no DESIGN §10.8 row can be named."
    elif not h1:
        text = "\"No significant difference\": recommend the glass box; the scenario framework and method comparison stand."
    elif correct is None:
        text = "GBM better on detection; explanation correctness not assessed (no injected attributions)."
    elif correct and consistent:
        text = "\"GBM significantly better; explanations consistent and correct\"."
    elif correct:
        text = "\"GBM better; disagrees with glass box but correct on injected\"."
    else:
        text = "\"GBM better; incorrect on injected\": the extra performance is not explainable at scheme level."
    return {"h1_supported": h1, "method": method, "correct": correct, "consistent": consistent, "row": text}


def build_report(paths: Paths) -> Path:
    prov = qa.data_provenance(paths)
    gen = datetime.now(timezone.utc).isoformat()
    ecfg = load_config("evaluation", paths)
    parts = {p: evaluate.load_part(paths, p) or {} for p in evaluate.PARTS}
    stab = evaluate.load_part(paths, "stability") or {}
    sens = sensitivity.load_results(paths)
    g3p = paths.phase3 / "G3_report.json"
    g3 = json.loads(g3p.read_text(encoding="utf-8")) if g3p.is_file() else None
    try:
        best = evaluate.best_families(paths)
    except Exception:  # noqa: BLE001 - no tuning yet: report what exists
        best = {}
    data = {"generated_utc": gen, "provenance": prov, "best": best, "parts": parts, "stability": stab,
            "sensitivity": sens}
    md = ["# Evaluation report (Phase 6)", "", *qa.provenance_lines(prov),
          f"Generated {gen} by `scenario-fraud phase6 report`. DESIGN §10, EXPERIMENT_PLAN §5, §7-§9. Explanation "
          "vectors are the 7 scheme groups (context excluded); a group counts as found only with a positive "
          "contribution; method C ranks by its diagonal (pure effects) (DESIGN §17 D101-D108).", "",
          f"- Best GBM / monotone GBM / glass box (validation, D74): `{best.get('gbm')}` / `{best.get('gbm_mono')}` / "
          f"`{best.get('glass_box')}`", ""]
    md += ["## T1 Detection (E1)", ""]
    if g3:
        prim = g3.get("designs", {}).get("primary")
        if prim:
            rows = []
            for f, v in prim["t1"].items():
                p, ci = v["point"]["pooled"], v.get("ci") or {}
                row = {"model": f, "PR-AUC": fmt(p["pr_auc"])}
                if "pr_auc" in ci:
                    row["PR-AUC"] += f" [{fmt(ci['pr_auc']['ci'][0])}, {fmt(ci['pr_auc']['ci'][1])}]"
                row.update({"ROC-AUC": fmt(p["roc_auc"]), **{f"P@{float(k[2:]):.1%}": fmt(p[k]) for k in p
                                                             if k.startswith("p@")}, "Brier": fmt(p.get("brier"), 5)})
                rows.append(row)
            md += [qa.md_table(rows), "From `outputs/phase3/G3_report.json` (CIs and Δ with Holm there).", ""]
    else:
        md += ["_No G3 report: run `phase3 report`._", ""]
    corr = parts["correctness"]
    md += ["## T2 Correctness (E4; injection models, use (b))", ""]
    if corr.get("available"):
        md += [f"Rows: single schemes at intensity `{ecfg['correctness']['table_intensity']}`; compound columns: the "
               f"pairs; draws {corr.get('draws')}; baseline `{corr['baseline']}`.", "", qa.md_table(t2_rows(corr, best))]
        md += ["Top-1 by intensity (best GBM, all methods; glass boxes, method D; the rest in `correctness.json`):", "",
               qa.md_table([{"family": r["family"], "method": r["method"], "intensity": r["intensity"], "n": r["n"],
                             "top-1": fmt(r["top1"]), "chance": fmt(r["chance_top1"])} for r in corr["by_intensity"]
                            if r["family"] == best.get("gbm") or r["method"] == "D"])]
        md += ["Top-1 by scheme (medium and other intensities) and spill-over shares are in `correctness.json`.", ""]
    else:
        md += [f"_{corr.get('detail', 'not run')}_", ""]
    st = parts["stress"]
    md += ["## E5 Detection stress test (use (a))", ""]
    if st.get("available"):
        md += [qa.md_table([{k: fmt(v) if isinstance(v, float) else v for k, v in r.items()} for r in st["summary"]])]
    else:
        md += [f"_{st.get('detail', 'not run')}_", ""]
    cons = parts["consistency"]
    md += [f"## T3 Consistency (E3; best GBM vs EBM / GA²M, both flag at k = {float(cons.get('flag_k') or 0):.0%})", ""]
    if cons.get("available"):
        md += [qa.md_table(t3_rows(cons, best)), "All strata (both / GBM only / glass box only / neither) and LR are in "
               "`consistency.json`.", ""]
    else:
        md += [f"_{cons.get('detail', 'not run')}_", ""]
    md += ["## T4 Faithfulness (E6) and stability (E7)", ""]
    if parts["faithfulness"].get("available"):
        md += ["Deletion score = area under the normalised group-deletion curve (lower is more faithful), providers "
               "above their baseline, peer baseline. Medians: the normalised area is unbounded when a provider's group "
               "contributions have mixed signs and f(x) is close to its baseline, so means are dominated by those "
               "providers (means in `faithfulness.json`; DESIGN §17 D104).", "",
               qa.md_table(t4_rows(parts["faithfulness"], stab))]
    else:
        md += [f"_{parts['faithfulness'].get('detail', 'not run')}_", ""]
    if stab:
        md += [f"Stability subsample: {stab.get('n')} provider-years; seeds per family: "
               f"{ {f: v['seeds'] for f, v in stab.get('families', {}).items()} }.", ""]
    bl = parts["baselines"]
    md += ["## E8 Peer vs global baseline (RQ5)", ""]
    if bl.get("available"):
        md += [qa.md_table([{k: fmt(v) if isinstance(v, float) else v for k, v in r.items()} for r in bl["summary"]])]
    else:
        md += ["_Needs peer and global attributions of a GBM (phase4 explain)._", ""]
    md += ["## E9 Sensitivity suite", ""]
    if sens:
        rows = []
        for vid, r in sens.items():
            row = {"variant": vid, "what": r.get("what") if not r.get("not_estimable")
                   else f"{r.get('what')} (not estimable: {r['not_estimable']})"}
            fam = best.get("gbm")
            det, det0 = (r.get("detection") or {}).get(fam), (r.get("detection_main") or {}).get(fam)
            row["PR-AUC (variant / main)"] = f"{fmt(det['pr_auc']) if det else '–'} / {fmt(det0['pr_auc']) if det0 else '–'}"
            ag = next((a for a in r.get("agreement_with_main", []) if a["family"] == fam and a["method"] == "B"), None)
            row["top-1 agree with main (B)"] = fmt(ag["top1_agreement"]) if ag else "–"
            cs = next((c for c in r.get("consistency", []) if c["gbm"] == fam and c["method"] == "B"), None)
            row["cos vs glass box (B)"] = fmt(cs["cosine"]) if cs else "–"
            fa = next((f for f in r.get("faithfulness", []) if f["family"] == fam and f["method"] == "B"), None)
            row["deletion (B)"] = fmt(fa["deletion"]) if fa else "–"
            co = next((c for c in r.get("correctness", []) if c["family"] == fam and c["method"] == "B"), None)
            row["top-1 injected (B)"] = fmt(co["top1"]) if co else "–"
            rows.append(row)
        md += [qa.md_table(rows), "S5 and S9 are read from the main evaluation (`sensitivity/S5.json`, `S9.json`).", ""]
    else:
        md += ["_No sensitivity variants run (`phase6 sensitivity`)._", ""]
    fa = parts["fairness"]
    md += ["## E10 Fairness profile (descriptive)", ""]
    if fa.get("available"):
        md += ["Flag rate, precision and flag rate ÷ positive rate at each k, pooled over the test years. LEIE reflects "
               "enforcement patterns, which limits interpretation (DESIGN §10.7). Largest groups shown; all in "
               "`fairness.json`.", ""]
        for fam, rows in fa["families"].items():
            show = [r for r in rows if r["N"] >= int(ecfg["fairness"]["min_rows"])]
            by = {}
            for r in show:
                by.setdefault((r["attribute"], r["k"]), []).append(r)
            for (attr, k), rs in sorted(by.items()):
                md += [f"`{fam}`, {attr}, k = {float(k):.1%}:", "",
                       qa.md_table([{"value": r["value"], "N": r["N"], "positives": r["positives"],
                                     "flagged": r["flagged"], "flag rate": fmt(r["flag_rate"], 4),
                                     "precision": fmt(r["precision"]),
                                     "flag/positive rate": fmt(r["flag_to_positive_ratio"], 2)} for r in rs[:12]])]
    else:
        md += ["_Not run._", ""]
    br = branch(g3, corr, cons, best)
    data["branch"] = br
    md += ["## DESIGN §10.8: which row the numbers are consistent with", "",
           f"- H1 (G3): {br['h1_supported']}; explanations of the best GBM (method {br['method']}) correct on injected "
           f"(Wilson 95 % lower bound of top-1 > chance): {br['correct']}; consistent with EBM and GA²M (median share "
           f"cosine > permutation baseline, both-flag stratum): {br['consistent']}.",
           f"- Consistent with: {br['row']}", "- The human decides the branch the paper follows (HANDOFF §8 rule 7).", ""]
    out = evaluate.out_dir(paths) / "evaluation_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    evaluate.write_json(out.with_suffix(".json"), data)
    return out
