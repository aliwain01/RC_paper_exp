"""Gate G1 report (HANDOFF §6; DESIGN §15 Phase 1 go/no-go).

Reports, from whatever Phase 1 artefacts exist: file inventory with sizes and row counts, whether
DMEPOS 2024 exists, column-map status, LEIE sources and EXCLTYPE coverage, positive NPI counts per
year x label tier, NPI-match losses, Internet Archive gap intervals and the reinstatement-loss profile.
Every section says plainly when its inputs are missing. It reports; the human decides.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from . import acquire, labels
from .config import load_config
from .harmonise import ColumnMap, map_path
from .paths import Paths

CMS_SOURCES = ("cms_ps", "cms_prov", "dme_s", "dme_p")


def _md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_none_\n"
    cols = cols or list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out) + "\n"


def _years(rng: list[int]) -> list[int]:
    return list(range(int(rng[0]), int(rng[1]) + 1))


def _ranges(ys: list[int]) -> str:
    """[2018, 2019, 2022, 2023, 2024] -> '2018–2019, 2022–2024'."""
    out, start = [], ys[0]
    for a, b in zip(ys, ys[1:] + [None]):
        if b is None or b != a + 1:  # run ends at a
            out.append(str(start) if start == a else f"{start}–{a}")
            start = b
    return ", ".join(out)


# ------------------------------------------------------------------------------------------------ sections

def inventory(paths: Paths) -> tuple[list[dict], list[dict], dict]:
    entries = acquire.load_manifest(paths)
    dumps = {d["path"]: d for d in acquire.dump_all_headers(paths)} if entries else {}
    rows = [{
        "source": e.source, "year": e.year, "kind": e.kind, "file": Path(e.path).name,
        "MB": round(e.size_bytes / 1e6, 1), "rows": dumps.get(e.path, {}).get("n_rows"),
        "cols": dumps.get(e.path, {}).get("n_columns"), "route": e.route, "as_of": e.as_of,
        "sha256": e.sha256[:12],
    } for e in entries]
    years = _years(load_config("sources", paths)["years"]["primary"])
    have = {(e.source, e.year, e.kind) for e in entries}
    missing = []
    for s in CMS_SOURCES:
        for k in ("data", "dictionary", "methodology"):
            ys = [y for y in years if (s, y, k) not in have]
            if ys:
                missing.append({"source": s, "kind": k, "missing_years": _ranges(ys)})
    leie_kinds = {e.kind for e in entries if e.source == "leie"}
    missing += [{"source": "leie", "kind": k, "missing_years": "-"}
                for k in ("current", "record_layout", "supplement_rein", "snapshot") if k not in leie_kinds]
    dme24 = {s: next((r for r in rows if r["source"] == s and r["year"] == 2024 and r["kind"] == "data"), None)
             for s in ("dme_s", "dme_p")}
    return rows, missing, dme24


def column_map_status(paths: Paths) -> list[dict]:
    years = _years(load_config("sources", paths)["years"]["primary"])
    out = []
    for s in CMS_SOURCES:
        for y in years:
            mp = map_path(paths, s, y)
            meta_p = paths.interim / s / f"{y}.meta.json"
            meta = json.loads(meta_p.read_text()) if meta_p.is_file() else None
            row = {"dataset": s, "year": y, "map": "missing", "harmonised_rows": None, "bene_cc_cols": None}
            if mp.is_file():
                cm = ColumnMap.load(mp)
                row["map"] = "reviewed" if cm.reviewed else "draft"
                if s == "cms_prov":
                    row["bene_cc_cols"] = sum(1 for h in cm.columns.values() if h and h.startswith("bene_cc_"))
            if meta:
                row["harmonised_rows"] = meta["n_rows"]
            out.append(row)
    return out


def excltype_counts(exc: pl.DataFrame, cfg: labels.LabelConfig) -> list[dict]:
    g = exc.group_by("excltype_raw", "authority").agg(
        pl.len().alias("events"), pl.col("npi").is_not_null().sum().alias("with_npi")
    ).sort("events", descending=True)
    rows = []
    for r in g.to_dicts():
        r["tiers"] = ",".join(t for t, a in cfg.tiers.items() if r["authority"] in a) or "-"
        rows.append(r)
    return rows


def positives_by_year(paths: Paths, cfg: labels.LabelConfig, exp: dict) -> dict:
    """Counts per year x tier at the primary W, unique positive NPIs over the gate window, W sensitivity.
    Also after the DESIGN §7.2 volume rule when the service-level file is harmonised for that year."""
    vol = _volume_ok(paths, exp["cohort"]["min_nondrug_services"])
    gate_years = _years(exp["gate_g1"]["window"])
    per_year, unique, sens = [], [], []
    for tier in cfg.tiers:
        f = paths.processed / f"labels_{tier}_W{cfg.W}.parquet"
        if not f.is_file():
            continue
        df = pl.read_parquet(f)
        if vol is not None:
            df = df.join(vol, on=["npi", "year"], how="left")
        for r in df.group_by("year").agg(
            (pl.col("y") == 1).sum().alias("pos"), (pl.col("y") == 0).sum().alias("neg"),
            pl.col("y").is_null().sum().alias("dropped"),
            *([((pl.col("y") == 1) & pl.col("vol_ok").fill_null(False)).sum().alias("pos_after_volume_rule")]
              if vol is not None else []),
        ).sort("year").to_dicts():
            per_year.append({"tier": tier, **r})
        win = df.filter(pl.col("year").is_in(gate_years) & (pl.col("y") == 1))
        unique.append({
            "tier": tier, "W": cfg.W, "unique_positive_npis": win["npi"].n_unique(),
            "after_volume_rule": (win.filter(pl.col("vol_ok").fill_null(False))["npi"].n_unique()
                                  if vol is not None else None),
            "threshold": exp["gate_g1"]["min_unique_positive_npis"],
        })
    for W in sorted({cfg.W, *cfg.W_sensitivity}):
        f = paths.processed / f"labels_{cfg.primary_tier}_W{W}.parquet"
        if f.is_file():
            d = pl.read_parquet(f).filter(pl.col("year").is_in(gate_years))
            sens.append({"tier": cfg.primary_tier, "W": W,
                         "unique_positive_npis": d.filter(pl.col("y") == 1)["npi"].n_unique(),
                         "positive_rows": int((d["y"] == 1).sum())})
    return {"per_year": per_year, "unique": unique, "W_sensitivity": sens, "volume_rule_applied": vol is not None}


def _volume_ok(paths: Paths, min_services: int) -> pl.DataFrame | None:
    files = sorted((paths.interim / "cms_ps").glob("*.parquet"))
    if not files:
        return None
    parts = [
        pl.scan_parquet(f).filter(pl.col("hcpcs_drug_ind") != "Y").group_by("npi", "year")
        .agg(pl.col("tot_srvcs").sum().alias("srv")).collect()
        for f in files
    ]
    return pl.concat(parts).select("npi", "year", (pl.col("srv") >= min_services).alias("vol_ok"))


def npi_match_losses(exc: pl.DataFrame, cfg: labels.LabelConfig, paths: Paths, first_year: int, snap: date) -> list[dict]:
    """Exclusion events that could label a study year (first_year < excl_year <= snapshot year), by tier:
    how many lack an NPI, carry an invalid one, or have an NPI absent from every by-Provider year."""
    py = labels.provider_years_from_interim(paths)
    known = set(py["npi"].unique().to_list()) if py is not None else None
    rel = exc.filter((pl.col("excl_year") > first_year) & (pl.col("excl_year") <= snap.year))
    rows = []
    for tier, auths in cfg.tiers.items():
        t = rel.filter(pl.col("authority").is_in(auths))
        with_npi = t.filter(pl.col("npi").is_not_null())
        row = {
            "tier": tier, "events": t.height,
            "no_npi": int((t["npi_status"] == "none").sum()),
            "invalid_npi": int((t["npi_status"] == "invalid").sum()),
            "unique_npis": with_npi["npi"].n_unique(),
            "npis_in_part_b": None, "npis_not_in_part_b": None,
        }
        if known is not None:
            u = set(with_npi["npi"].unique().to_list())
            row["npis_in_part_b"] = len(u & known)
            row["npis_not_in_part_b"] = len(u - known)
        rows.append(row)
    return rows


def horizon_table(cfg: labels.LabelConfig, snap: date, years: list[int]) -> list[dict]:
    out = []
    for t in years:
        obs = min(cfg.W, max(0.0, (snap - date(t + 1, 1, 1)).days / 365.25))
        out.append({"year": t, "years_observed": round(obs, 2), "W": cfg.W,
                    "complete": snap >= date(t + cfg.W, 12, 31)})
    return out


# ------------------------------------------------------------------------------------------------ report

def build_report(paths: Paths) -> Path:
    cfg = labels.LabelConfig.load(paths)
    exp = load_config("experiment", paths)
    src_years = load_config("sources", paths)["years"]["primary"]
    data: dict = {"generated_utc": datetime.now(timezone.utc).isoformat()}
    md: list[str] = [
        "# Gate G1 report: end of Phase 1 (data acquisition)",
        "",
        f"Generated {data['generated_utc']} by `scenario-fraud phase1 report`. Regenerate after any change to "
        "`data/raw/manifest.json`, column maps or labels. This report states facts; the go/no-go decision "
        "belongs to the human (HANDOFF §6, G1).",
        "",
    ]

    inv, missing, dme24 = inventory(paths)
    data.update(inventory=inv, missing=missing, dmepos_2024=dme24)
    md += ["## 1. Files acquired", "", _md_table(inv) if inv else "_No files registered in `data/raw/manifest.json`._\n"]
    md += ["### Expected but not registered", "", _md_table(missing)]
    md += ["### DMEPOS 2024", ""]
    for s, r in dme24.items():
        md.append(f"- `{s}` 2024: " + (f"registered, {r['rows']} rows, {r['MB']} MB" if r else
                                       "not registered, so whether it is published is **still unknown**"))
    md.append("")

    cms = column_map_status(paths)
    data["column_maps"] = cms
    md += ["## 2. Column maps (C2) and harmonised files", "", _md_table(cms)]

    md += ["## 3. LEIE sources and EXCLTYPE coverage (C3)", ""]
    md.append(f"- LEIE layout verified against the OIG record-layout PDF: **{cfg.layout_verified}**")
    md.append(f"- EXCLTYPE map verified: **{cfg.excltype_verified}**")
    exc_p = paths.processed / "exclusions.parquet"
    exc = pl.read_parquet(exc_p) if exc_p.is_file() else None
    if exc is None:
        md += ["", "_No exclusions table: run `scenario-fraud phase1 labels` after registering the LEIE files._", ""]
    else:
        snap = labels.snapshot_date(paths, cfg)
        md += [f"- Snapshot date (label horizon): {snap.isoformat()}",
               f"- Exclusion events after union: {exc.height}; not in the current file (recovered): "
               f"{int((~exc['in_current']).sum())}", ""]
        et = excltype_counts(exc, cfg)
        data["excltype_counts"] = et
        md += [_md_table(et)]

    md += ["## 4. Positive NPI counts per year × label tier", ""]
    if exc is None or not any((paths.processed / f"labels_{t}_W{cfg.W}.parquet").is_file() for t in cfg.tiers):
        md += ["_Not available: needs the LEIE files and harmonised by-Provider (`cms_prov`) years._", ""]
    else:
        pos = positives_by_year(paths, cfg, exp)
        data["positives"] = pos
        note = ("after the ≥ {} non-drug services rule where `cms_ps` is harmonised".format(
            exp["cohort"]["min_nondrug_services"]) if pos["volume_rule_applied"]
            else "before the cohort volume rule (`cms_ps` not harmonised yet)")
        md += [f"Counts over all by-Provider NPI-years, {note}. W = {cfg.W}.", "", _md_table(pos["per_year"]),
               f"**Unique positive NPIs over {exp['gate_g1']['window'][0]}–{exp['gate_g1']['window'][1]}** "
               f"(DESIGN §14 threshold ≈ {exp['gate_g1']['min_unique_positive_npis']}):", "",
               _md_table(pos["unique"]), "W sensitivity (primary tier):", "", _md_table(pos["W_sensitivity"])]

    md += ["## 5. NPI-match losses", ""]
    if exc is None:
        md += ["_Not available._", ""]
    else:
        nm = npi_match_losses(exc, cfg, paths, int(src_years[0]), snap)
        data["npi_match"] = nm
        md += [f"Exclusion events with {src_years[0]} < exclusion year ≤ {snap.year}, by tier. "
               "`npis_in_part_b` needs harmonised `cms_prov` years.", "", _md_table(nm)]

    md += ["## 6. Reinstatement recovery: Internet Archive gaps and loss profile", ""]
    if exc is None:
        md += ["_Not available._", ""]
    else:
        gaps = labels.coverage_gaps(paths, date(int(src_years[0]) + 1, 1, 1), snap)
        longest = max((i["days"] for i in gaps["uncovered_intervals"]), default=None)
        loss = labels.reinstatement_loss_profile(exc, cfg, longest)
        data["coverage"] = gaps
        data["loss_profile"] = loss.to_dicts()
        md += [f"- Full-file snapshots used: {', '.join(gaps['snapshot_dates']) or 'none'}",
               f"- Reinstatement supplement months: {', '.join(gaps['rein_supplement_months']) or 'none'}",
               f"- Uncovered days in window {gaps['window'][0]} → {gaps['window'][1]}: {gaps['uncovered_days_total']}",
               "", "Uncovered intervals (an exclusion that starts and is reinstated inside one is invisible):", "",
               _md_table(gaps["uncovered_intervals"]),
               "Reinstated records by authority (from supplements and snapshots):", "", _md_table(loss.to_dicts())]

    md += ["## 7. Label horizon", ""]
    try:
        snap_h = labels.snapshot_date(paths, cfg)
        md += [_md_table(horizon_table(cfg, snap_h, _years(src_years)))]
    except labels.LabelError as exc_h:
        md += [f"_Not available: {exc_h}_", ""]

    out = paths.phase1 / "G1_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    (paths.phase1 / "G1_report.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return out
