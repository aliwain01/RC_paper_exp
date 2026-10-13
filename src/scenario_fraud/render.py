"""C14 indicator & view renderer (DESIGN §9.6, §11; CONTRACTS §7).

`build_record(paths, npi, year)` assembles one decision record (configs/decision_record.schema.json): the score of the
reference model (seed 0) with its rank and calibrated probability, every method's group attributions against the peer
baseline (Phase 4 store), up to 3 indicators by the primary method (positive, ≥ 0.1 log-odds) with 1–2 supporting
facts each (largest per-feature contributions in the group: method A within group for trees, method D for glass
boxes; raw value vs the peer-cell median), compound pairs from method C, mitigating groups, the context contribution
as a peer-group adjustment (never a reason), versions (model, feature table, configs, attribution store) and an
optional human decision. Views: analyst (everything), provider and beneficiary notices (only after a decision whose
action is external; §11.3 rules enforced on the rendered text), oversight (model card and provenance chain), and a
queue summary for unit managers. Decisions here are mock decisions (DESIGN §11.2): every notice carries a MOCK-UP
banner. Output: outputs/phase7/records/{record_id}.json and outputs/phase7/views/{record_id}/.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from . import evaluate, features, metrics, provenance, qa, runlog, splits, train
from .attribute import load_attributions, store_path
from .config import file_sha256, load_config
from .groups import CONTEXT, GROUP_ORDER, SCHEME_GROUPS
from .paths import Paths

SCHEMA_VERSION = "1.0"
M = len(GROUP_ORDER)
EXTERNAL = ("provider", "beneficiary")


class RenderError(RuntimeError):
    pass


def views_dir(paths: Paths) -> Path:
    return paths.phase7 / "views"


def records_dir(paths: Paths) -> Path:
    return paths.phase7 / "records"


def schema(paths: Paths) -> dict:
    return json.loads((paths.configs / "decision_record.schema.json").read_text(encoding="utf-8"))


def validate(paths: Paths, record: dict) -> None:
    import jsonschema

    try:
        jsonschema.validate(record, schema(paths))
    except jsonschema.ValidationError as exc:
        raise RenderError(f"decision record does not match the schema: {exc.message} at {list(exc.absolute_path)}") from exc


def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def _vec(v) -> dict:
    return {g: _f(x) for g, x in zip(GROUP_ORDER, v)}


# ----------------------------------------------------------------------------------------------- record

def _score(paths: Paths, family: str, npi: str, year: int) -> dict:
    scfg = splits.SplitConfig.load(paths)
    design = "showcase" if year == scfg.showcase_year else "primary"
    s = train.load_scores(paths, design, family).filter((pl.col("seed") == 0) & (pl.col("year") == year))
    if s.is_empty() or s.filter(pl.col("npi") == npi).is_empty():
        raise RenderError(f"{npi} {year}: no {design} score of {family} (run phase3)")
    s = s.with_columns(pl.Series("_tk", metrics.tie_keys(s["npi"], s["year"]))).sort(
        ["margin", "_tk"], descending=[True, True]).with_row_index("rank", offset=1)
    r = s.filter(pl.col("npi") == npi).row(0, named=True)
    return {"family": family, "seed": 0, "margin": float(r["margin"]), "probability": _f(r.get("prob")),
            "rank": int(r["rank"]), "n_scored": s.height, "label_tier": r.get("label_tier"),
            "W": int(r["W"]) if r.get("W") is not None else None, "fold": int(r["fold"])}


def _facts(paths: Paths, row: dict, feat: dict, peer: pl.DataFrame, cols: dict, g: int, n: int,
           fcfg: features.FeatureConfig) -> list[dict]:
    contrib = row.get("A_feat") or row.get("D_feat")
    if contrib is None:
        return []
    idx = [j for j, cg in enumerate(cols["col_group"]) if cg == g]
    idx = sorted(idx, key=lambda j: -contrib[j])
    out = []
    for j in idx:
        if contrib[j] <= 0 or len(out) >= n:
            break
        col = cols["columns"][j]
        fid = col.split("__")[0]
        try:
            spec = fcfg.spec(fid)
        except KeyError:
            continue
        rawc = f"{features.RAW}{fid}"
        val = feat.get(rawc, feat.get(col))
        med = peer[rawc].median() if rawc in peer.columns else peer[col].median()
        out.append({"feature": fid, "name": spec.name, "value": _f(val), "peer_median": _f(med),
                    "z": _f(feat.get(col)) if spec.kind == "robust_z" else None, "contribution": float(contrib[j]),
                    "unit": "share" if "share" in spec.name.lower() else "value"})
    return out


def build_record(paths: Paths, npi: str, year: int, *, family: str | None = None, decision: dict | None = None) -> dict:
    train.check_frozen(paths)
    vcfg = load_config("views", paths)
    fcfg = features.FeatureConfig.load(paths)
    fam = family or vcfg.get("family") or evaluate.best_families(paths)["gbm"]
    base = vcfg["baseline"]
    att = load_attributions(paths, fam, base).filter((pl.col("npi") == npi) & (pl.col("year") == year))
    if att.is_empty():
        raise RenderError(f"{npi} {year}: no {fam}/{base} attributions (only the explained set is attributed; "
                          "run phase4 explain)")
    row = att.row(0, named=True)
    cols = json.loads((paths.processed / "attributions" / fam / "columns.json").read_text(encoding="utf-8"))
    ft = features.load_features(paths, [year]).with_columns(pl.col("year").cast(pl.Int32))
    me = ft.filter(pl.col("npi") == npi)
    if me.is_empty():
        raise RenderError(f"{npi} {year}: no feature row")
    feat = me.row(0, named=True)
    peer = ft.filter(pl.col("peer_cell") == feat["peer_cell"])
    coh = pl.read_parquet(paths.processed / "cohort.parquet").filter(
        (pl.col("npi") == npi) & (pl.col("year").cast(pl.Int32) == year)).row(0, named=True)
    score = _score(paths, fam, npi, year)
    fold = score.pop("fold")
    mdir = train.model_dir(paths, "primary", fam, fold) / "seed0"
    cal = joblib.load(mdir / "calibrator.joblib")
    meth = vcfg["primary_method"]
    vec = evaluate.group_vectors(att, meth)[0]
    V = np.asarray(row["v"], dtype=float)
    full = 2 ** M - 1
    order = sorted(range(1, M), key=lambda g: -vec[g])
    indicators = []
    for g in order:
        if len(indicators) >= int(vcfg["max_indicators"]) or vec[g] < float(vcfg["min_contribution"]):
            break
        name = GROUP_ORDER[g]
        indicators.append({"group": name, "label": vcfg["groups"][name]["label"], "contribution": float(vec[g]),
                           "probability_if_peer": _f(cal.predict([V[full & ~(1 << g)]])[0]),
                           "facts": _facts(paths, row, feat, peer, cols, g, int(vcfg["supporting_facts"]), fcfg)})
    Cm = np.asarray(row["C"], dtype=float).reshape(M, M)
    compound = sorted(({"groups": [GROUP_ORDER[g], GROUP_ORDER[h]], "value": float(Cm[g, h] + Cm[h, g])}
                       for g in range(1, M) for h in range(g + 1, M)
                       if Cm[g, h] + Cm[h, g] >= float(vcfg["compound_min"])), key=lambda c: -c["value"])
    mitigating = [{"group": GROUP_ORDER[g], "label": vcfg["groups"][GROUP_ORDER[g]]["label"],
                   "contribution": float(vec[g])} for g in range(1, M) if vec[g] <= float(vcfg["mitigating_max"])]
    methods = {m: _vec(evaluate.group_vectors(att, m)[0]) for m in evaluate.METHODS
               if evaluate.method_available(att, m)}
    sp = store_path(paths, fam, base, int(row["fold"]))
    record = {
        "schema_version": SCHEMA_VERSION, "record_id": "", "created_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic": bool(qa.data_provenance(paths)["synthetic"]),
        "provider": {"npi": npi, "year": int(year), "peer_cell": feat["peer_cell"], "peer_level": coh.get("peer_level"),
                     "specialty": coh.get("prvdr_type"), "entity": coh.get("ent_cd"),
                     "census_division": coh.get("census_division"), "ruca_class": coh.get("ruca_class"),
                     "data_coverage": _f(feat.get("CX8"))},
        "score": score,
        "baseline": {"kind": base, "bg_key": row["bg_key"], "bg_level": row.get("bg_level"), "bg_n": int(row["bg_n"]),
                     "base": float(row["base"])},
        "attributions": {"primary_method": meth, "methods": methods, "interaction_matrix": Cm.tolist(),
                         "counterfactual_probability": _vec(row["E_prob"]) if row.get("E_prob") else {}},
        "indicators": indicators, "compound": compound, "mitigating": mitigating,
        "peer_adjustment": float(vec[0]) if np.isfinite(vec[0]) else 0.0,
        "versions": {"model": {"dir": paths.rel(mdir), "fitted_sha256": file_sha256(mdir / "fitted.json")},
                     "features": {"file": f"data/processed/features/{year}.parquet",
                                  "sha256": file_sha256(paths.processed / "features" / f"{year}.parquet"),
                                  "synthetic_inputs": provenance.inputs_synthetic(
                                      provenance.interim_files(paths.interim))},
                     "configs": {p.name: file_sha256(p) for p in sorted(paths.configs.glob("*.y*ml"))},
                     "attribution_store": {"file": paths.rel(sp), "sha256": file_sha256(sp)},
                     "git_commit": runlog._git(paths, "rev-parse", "HEAD")},
        "decision": decision, "notices": []}
    key = f"{npi}|{year}|{fam}|{base}|{record['versions']['model']['fitted_sha256']}"
    record["record_id"] = hashlib.blake2b(key.encode(), digest_size=8).hexdigest()
    if any(i["group"] == CONTEXT for i in indicators):
        raise RenderError("context can never be a reason (§11.3 rule 2)")
    validate(paths, record)
    return record


def mock_decision(record: dict, action: str, vcfg: dict) -> dict:
    if action not in vcfg["actions"]:
        raise RenderError(f"unknown action {action!r}; known {sorted(vcfg['actions'])}")
    return {"action": action, "decided_by": "MOCK reviewer (no human decision was taken)",
            "decided_utc": datetime.now(timezone.utc).isoformat(),
            "policy_reasons": [i["group"] for i in record["indicators"][:2]], "mock": True,
            "note": "Mock decision for the design proposal (DESIGN §11.2, §11.5)"}


# ----------------------------------------------------------------------------------------------- views

def _num(x: float | None, unit: str) -> str:
    if x is None:
        return "n/a"
    return f"{x:.0%}" if unit == "share" else f"{x:.3g}"


def analyst_view(record: dict, vcfg: dict) -> str:
    p, s = record["provider"], record["score"]
    prob = f"{s['probability']:.1%}" if s["probability"] is not None else "n/a"
    lines = []
    if record.get("synthetic"):
        lines += ["> SYNTHETIC DATA: NOT A RESULT. Mock-up from tests/synth.py data.", ""]
    lines += [f"# Analyst view · record {record['record_id']}", "",
              f"**Rank {s['rank']} of {s['n_scored']} scored providers · P(fraud-related exclusion within "
              f"{s.get('W') or 'W'} yrs) = {prob}** · peer group: {p.get('specialty')}, {p.get('entity')}, "
              f"{p.get('census_division')}, {p['year']} ({p.get('peer_level')}) · data coverage "
              f"{_num(p.get('data_coverage'), 'share')}", "",
              f"Model `{s['family']}` (seed {s['seed']}), method {record['attributions']['primary_method']}, "
              f"{record['baseline']['kind']} baseline ({record['baseline']['bg_n']} providers, "
              f"`{record['baseline']['bg_key']}`). Contributions are log-odds; they explain the model, not fraud.", ""]
    if not record["indicators"]:
        lines += ["No scheme group contributes ≥ "
                  f"{vcfg['min_contribution']} log-odds.", ""]
    for i, ind in enumerate(record["indicators"], 1):
        facts = "; ".join(f"{f['name']} {_num(f['value'], f['unit'])} (peer median {_num(f['peer_median'], f['unit'])})"
                          for f in ind["facts"]) or "no supporting feature with a positive contribution"
        lines.append(f"{i}. **{ind['label']}** (+{ind['contribution']:.2f}): {facts}")
        for c in record["compound"]:
            if ind["group"] in c["groups"] and c["groups"][0] == ind["group"]:
                other = vcfg["groups"][c["groups"][1]]["label"]
                lines.append(f"   ↳ **Compound: {ind['label']} × {other}** (+{c['value']:.2f})")
    if record["mitigating"]:
        lines += ["", "*Mitigating:* " + ", ".join(f"{m['label']} ({m['contribution']:+.2f})" for m in record["mitigating"])]
    lines += ["", f"*Peer-group adjustment (context, not a reason):* {record['peer_adjustment']:+.2f}"]
    if record["indicators"] and record["indicators"][0]["probability_if_peer"] is not None and s["probability"] is not None:
        top = record["indicators"][0]
        lines += [f"*If {top['label'].lower()} matched peers:* risk {prob} → {top['probability_if_peer']:.1%}"]
    lines += ["", "| method | " + " | ".join(SCHEME_GROUPS) + " |", "|---" * (len(SCHEME_GROUPS) + 1) + "|"]
    for m, v in record["attributions"]["methods"].items():
        lines.append(f"| {m} | " + " | ".join("–" if v[g] is None else f"{v[g]:+.2f}" for g in SCHEME_GROUPS) + " |")
    if record.get("decision"):
        d = record["decision"]
        lines += ["", f"Decision: **{vcfg['actions'][d['action']]['text']}** by {d['decided_by']} ({d['decided_utc']})"
                  + (" · MOCK" if d["mock"] else "")]
    return "\n".join(lines) + "\n"


def provider_view(record: dict, vcfg: dict) -> str:
    d = record["decision"]
    facts = []
    for g in d["policy_reasons"]:
        ind = next((i for i in record["indicators"] if i["group"] == g), None)
        said = [f"{f['name'].lower()} was {_num(f['value'], f['unit'])} (peer median {_num(f['peer_median'], f['unit'])})"
                for f in (ind["facts"] if ind else [])]
        said = [t for t in said if not _forbidden(t, vcfg["forbidden"]["provider"])]   # e.g. "average risk score"
        if said:
            facts.append(said[0])
        elif g in vcfg["groups"]:
            facts.append(vcfg["groups"][g]["policy_reason"])
    body = vcfg["templates"]["provider"].format(year=record["provider"]["year"],
                                                facts="; ".join(facts) or "see the enclosed list")
    head = f"Action: {vcfg['actions'][d['action']]['text']}. Reference {record['record_id']}-P."
    return _banner(record) + head + "\n\n" + " ".join(body.split()) + "\n"


def beneficiary_view(record: dict, vcfg: dict) -> str:
    return _banner(record) + " ".join(vcfg["templates"]["beneficiary"].split()) + \
        f"\n\nReference {record['record_id']}-B.\n"


def _banner(record: dict) -> str:
    out = ""
    if (record.get("decision") or {}).get("mock"):
        out += "MOCK-UP: not a real notice; no human decision was taken.\n\n"
    if record.get("synthetic"):
        out += "SYNTHETIC DATA: NOT A RESULT.\n\n"
    return out


def oversight_view(record: dict) -> str:
    v = record["versions"]
    lines = [f"# Oversight view · record {record['record_id']}", "",
             f"- Model: `{v['model']['dir']}` (fitted.json sha256 {v['model']['fitted_sha256'][:12]}…), family "
             f"`{record['score']['family']}`, label tier `{record['score'].get('label_tier')}`, W = {record['score'].get('W')}",
             f"- Data: `{v['features']['file']}` (sha256 {v['features']['sha256'][:12]}…), synthetic inputs: "
             f"{v['features'].get('synthetic_inputs')}",
             f"- Attribution snapshot: `{v['attribution_store']['file']}` (sha256 {v['attribution_store']['sha256'][:12]}…), "
             f"baseline `{record['baseline']['bg_key']}`",
             f"- Code: git {v.get('git_commit') or 'unknown'}; configs: {len(v['configs'])} files hashed in the record",
             "- Fairness audit: `outputs/phase6/fairness.json`; model evaluation: `outputs/phase6/evaluation_report.md`",
             "", "Provenance chain: " + " → ".join([*([n["notice_id"] for n in record["notices"]] or ["(no notice)"]),
                                                  f"record {record['record_id']}", "attribution snapshot",
                                                  "model and data version"])]
    if record.get("decision"):
        d = record["decision"]
        lines.append(f"- Decision: {d['action']} by {d['decided_by']} at {d['decided_utc']}"
                     + (" (MOCK)" if d["mock"] else ""))
    return "\n".join(lines) + "\n"


def _forbidden(text: str, words) -> list[str]:
    low = text.lower()
    return [w for w in words if re.search(rf"(?<![\w']){re.escape(w.lower())}(?![\w'])", low)]


def check_external(text: str, audience: str, record: dict, vcfg: dict) -> None:
    """§11.3: external views never show the score, model internals, context as a reason or (beneficiaries) fraud
    language. Raises RenderError on a violation."""
    bad = _forbidden(text, vcfg["forbidden"][audience])
    s = record["score"]
    nums = [f"{s['margin']:.2f}"] + ([f"{s['probability']:.1%}"] if s["probability"] is not None else [])
    leaked = [n for n in nums if re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", text)]
    if bad or leaked:
        raise RenderError(f"{audience} view breaks DESIGN §11.3: forbidden words {bad}, score values {leaked}")


def to_html(title: str, text: str) -> str:
    body, in_list = [], False
    for line in text.splitlines():
        t = html.escape(line)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        item = re.match(r"^\s*(?:\d+\.|-)\s+(.*)$", t)
        if item and not line.startswith("|"):
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{item.group(1)}</li>")
            continue
        if in_list:
            body.append("</ul>")
            in_list = False
        if line.startswith("# "):
            body.append(f"<h1>{t[2:]}</h1>")
        elif line.startswith("&gt; "):
            body.append(f"<p class='banner'>{t[5:]}</p>")
        elif t.strip():
            body.append(f"<p>{t}</p>")
    if in_list:
        body.append("</ul>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><title>" + html.escape(title) +
            "</title><style>body{font-family:system-ui,sans-serif;max-width:46rem;margin:2rem auto;padding:0 1rem;"
            "color:#0b0b0b;background:#fcfcfb}.banner{border:1px solid #d03b3b;padding:.5rem}</style></head><body>"
            + "\n".join(body) + "</body></html>\n")


# ----------------------------------------------------------------------------------------------- orchestration

def render(paths: Paths, npi: str, year: int, *, family: str | None = None, action: str | None = None) -> dict:
    vcfg = load_config("views", paths)
    rec = build_record(paths, npi, year, family=family)
    if action:
        rec["decision"] = mock_decision(rec, action, vcfg)
    out = views_dir(paths) / rec["record_id"]
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    views = {"analyst": analyst_view(rec, vcfg)}
    if rec["decision"] and vcfg["actions"][rec["decision"]["action"]]["external"]:
        views["provider"] = provider_view(rec, vcfg)
        views["beneficiary"] = beneficiary_view(rec, vcfg)
        for aud in EXTERNAL:
            check_external(views[aud], aud, rec, vcfg)
            rec["notices"].append({"notice_id": f"{rec['record_id']}-{aud[0].upper()}", "audience": aud,
                                   "record_id": rec["record_id"], "path": paths.rel(out / f"{aud}.md")})
    views["oversight"] = oversight_view(rec)
    for name, text in views.items():
        (out / f"{name}.md").write_text(text, encoding="utf-8")
        (out / f"{name}.html").write_text(to_html(f"{name} view {rec['record_id']}", text), encoding="utf-8")
        written[name] = paths.rel(out / f"{name}.md")
    validate(paths, rec)
    rp = records_dir(paths) / f"{rec['record_id']}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return {"record": paths.rel(rp), "views": written}


def sample(paths: Paths, n: int) -> pl.DataFrame:
    """The n highest-scored providers of the latest test year's explained set (top 2 % part) that have attributions."""
    vcfg = load_config("views", paths)
    fam = vcfg.get("family") or evaluate.best_families(paths)["gbm"]
    att = load_attributions(paths, fam, vcfg["baseline"]).select("npi", "year", "margin")
    yr = att["year"].max()
    return att.filter(pl.col("year") == yr).sort(["margin", "npi"], descending=[True, False]).head(n).select("npi", "year")


def queue_summary(paths: Paths, *, year: int | None = None) -> Path:
    """Unit-manager view (§11.2): over the explained set's top-flagged providers of a year, how often each scheme
    group is the leading indicator and its mean contribution; precision at capacity from the G3 report."""
    vcfg = load_config("views", paths)
    fam = vcfg.get("family") or evaluate.best_families(paths)["gbm"]
    att = load_attributions(paths, fam, vcfg["baseline"])
    es = pl.read_parquet(paths.processed / "attributions" / "explained_set.parquet").select(
        "npi", pl.col("year").cast(pl.Int32), "top")
    att = att.join(es, on=["npi", "year"], how="left").filter(pl.col("top").fill_null(False))
    yr = int(year or att["year"].max())
    att = att.filter(pl.col("year") == yr)
    vec = evaluate.group_vectors(att, vcfg["primary_method"])[:, 1:] if att.height else np.zeros((0, M - 1))
    lead = np.argmax(vec, axis=1) if len(vec) else np.array([], dtype=int)
    rows = [{"scheme group": vcfg["groups"][g]["label"], "leading indicator (share)": float(np.mean(lead == i)) if len(lead) else None,
             "mean contribution": float(vec[:, i].mean()) if len(vec) else None} for i, g in enumerate(SCHEME_GROUPS)]
    g3p = paths.phase3 / "G3_report.json"
    cap = []
    if g3p.is_file():
        t1 = json.loads(g3p.read_text(encoding="utf-8")).get("designs", {}).get("primary", {}).get("t1", {}).get(fam)
        if t1:
            cap = [{"capacity k": k[2:], "precision": v} for k, v in t1["point"]["per_year"].get(str(yr), {}).items()
                   if k.startswith("p@")]
    banner = qa.provenance_lines(qa.data_provenance(paths))
    md = [f"# Queue summary · {yr} · `{fam}`", "", *banner,
          f"{att.height} top-flagged providers in the explained set (no individual drill-down).", "",
          qa.md_table(rows), "Precision at audit capacity (G3, out of fold):", "", qa.md_table(cap)]
    out = views_dir(paths) / f"queue_{yr}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    return out
