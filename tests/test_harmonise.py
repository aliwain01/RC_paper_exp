"""C2: draft maps from real headers, strict validation, typed Parquet output."""

from __future__ import annotations

import json

import polars as pl
import pytest

from scenario_fraud import acquire, harmonise
from tests.conftest import write_csv

PS_HEADER = [
    "Rndrng_NPI", "Rndrng_Prvdr_Last_Org_Name", "Rndrng_Prvdr_Ent_Cd", "Rndrng_Prvdr_State_Abrvtn",
    "Rndrng_Prvdr_RUCA", "Rndrng_Prvdr_Type", "Rndrng_Prvdr_Mdcr_Prtcptg_Ind", "HCPCS_Cd", "HCPCS_Drug_Ind",
    "Place_Of_Srvc", "Tot_Benes", "Tot_Srvcs", "Tot_Bene_Day_Srvcs", "Avg_Sbmtd_Chrg", "Avg_Mdcr_Alowd_Amt",
    "Avg_Mdcr_Pymt_Amt", "Avg_Mdcr_Stdzd_Amt", "Some_New_Col",
]
ROW = ["1234567893", "DOE", "I", "TX", "1", "Internal Medicine", "Y", "99215", "N", "O",
       "25", "40.5", "38", "210.0", "130.2", "100.1", "98.7", "zzz"]


def _setup(proj, tmp_path, rows=None, header=PS_HEADER):
    f = write_csv(tmp_path / "ps.csv", header, rows or [ROW, ["1999999992", *ROW[1:]]])
    e = acquire.register(proj, f, "cms_ps", 2023, "data")
    acquire.header_dump(proj, e)
    return e


def _finalise(proj, edits: dict[str, str]):
    p = harmonise.map_path(proj, "cms_ps", 2023)
    text = p.read_text()
    for k, v in edits.items():
        text = text.replace(k, v)
    p.write_text(text)


def test_draft_prefills_known_and_marks_unknown(proj, tmp_path):
    _setup(proj, tmp_path)
    p = harmonise.draft_column_map(proj, "cms_ps", 2023)
    cmap = harmonise.ColumnMap.load(p)
    assert cmap.columns["Rndrng_NPI"] == "npi"
    assert cmap.columns["Rndrng_Prvdr_Last_Org_Name"] is None
    assert cmap.columns["Some_New_Col"] == harmonise.TODO
    assert not cmap.reviewed
    with pytest.raises(harmonise.SchemaError, match="exists"):
        harmonise.draft_column_map(proj, "cms_ps", 2023)


def test_draft_or_todo_maps_are_refused(proj, tmp_path):
    _setup(proj, tmp_path)
    harmonise.draft_column_map(proj, "cms_ps", 2023)
    with pytest.raises(harmonise.SchemaError) as ei:
        harmonise.harmonise(proj, "cms_ps", 2023)
    msg = str(ei.value)
    assert "draft" in msg and "TODO" in msg and "Some_New_Col" in msg


def test_harmonise_writes_typed_parquet(proj, tmp_path):
    e = _setup(proj, tmp_path)
    harmonise.draft_column_map(proj, "cms_ps", 2023)
    _finalise(proj, {"reviewed: false": "reviewed: true", '"Some_New_Col": TODO': '"Some_New_Col": null'})
    out = harmonise.harmonise(proj, "cms_ps", 2023)
    df = pl.read_parquet(out)
    assert df.height == 2
    assert df.schema["tot_benes"] == pl.Int64 and df.schema["tot_srvcs"] == pl.Float64
    assert df.schema["npi"] == pl.String and df["year"].to_list() == [2023, 2023]
    assert "Rndrng_Prvdr_Last_Org_Name" not in df.columns
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["source_sha256"] == e.sha256 and meta["n_rows"] == 2


def test_unmapped_stale_and_missing_required_fail(proj, tmp_path):
    _setup(proj, tmp_path)
    harmonise.draft_column_map(proj, "cms_ps", 2023)
    _finalise(proj, {
        "reviewed: false": "reviewed: true",
        '"Some_New_Col": TODO    # not in template: map or drop by hand\n': "",   # now unmapped
        '"Tot_Benes": tot_benes': '"Tot_Benes": null\n  "Ghost_Col": null',          # stale + required missing
    })
    with pytest.raises(harmonise.SchemaError) as ei:
        harmonise.harmonise(proj, "cms_ps", 2023)
    msg = str(ei.value)
    assert "unmapped raw columns" in msg and "Some_New_Col" in msg
    assert "not in the file" in msg and "Ghost_Col" in msg
    assert "required harmonised columns not produced" in msg and "tot_benes" in msg


def test_bad_values_fail_loudly(proj, tmp_path):
    bad = list(ROW)
    bad[PS_HEADER.index("Tot_Benes")] = "12.5"
    _setup(proj, tmp_path, rows=[ROW, bad])
    harmonise.draft_column_map(proj, "cms_ps", 2023)
    _finalise(proj, {"reviewed: false": "reviewed: true", '"Some_New_Col": TODO': '"Some_New_Col": null'})
    with pytest.raises(harmonise.SchemaError, match="does not cast"):
        harmonise.harmonise(proj, "cms_ps", 2023)


def test_bad_npi_fails(proj, tmp_path):
    bad = list(ROW)
    bad[0] = "12345"
    _setup(proj, tmp_path, rows=[ROW, bad])
    harmonise.draft_column_map(proj, "cms_ps", 2023)
    _finalise(proj, {"reviewed: false": "reviewed: true", '"Some_New_Col": TODO': '"Some_New_Col": null'})
    with pytest.raises(harmonise.SchemaError, match="not 10 digits"):
        harmonise.harmonise(proj, "cms_ps", 2023)


def test_map_tied_to_file_checksum(proj, tmp_path):
    _setup(proj, tmp_path)
    harmonise.draft_column_map(proj, "cms_ps", 2023)
    _finalise(proj, {"reviewed: false": "reviewed: true", '"Some_New_Col": TODO': '"Some_New_Col": null'})
    p = harmonise.map_path(proj, "cms_ps", 2023)
    cmap = harmonise.ColumnMap.load(p)
    p.write_text(p.read_text().replace(cmap.sha256, "0" * 64))
    with pytest.raises(harmonise.SchemaError, match="different file"):
        harmonise.harmonise(proj, "cms_ps", 2023)
    # An unquoted all-digit hash would load as an int and silently skip the check: refuse it.
    p.write_text(p.read_text().replace('"' + "0" * 64 + '"', "0" * 64))
    with pytest.raises(harmonise.SchemaError, match="quoted 64-hex"):
        harmonise.harmonise(proj, "cms_ps", 2023)


def test_dmepos_draft_uses_capital_hcpcs_and_drop_prefixes(proj, tmp_path):
    header = ["Rfrg_NPI", "Tot_Suplr_Benes", "DME_Tot_Suplr_Benes", "DME_Sprsn_Ind", "POS_Tot_Suplrs",
              "Bene_Avg_Age", "Rfrg_Prvdr_Spclty_Desc", "Mystery_Col"]
    f = write_csv(tmp_path / "dmep.csv", header, [["1234567893", "20", "15", "", "3", "71", "Podiatry", "x"]])
    e = acquire.register(proj, f, "dme_p", 2023, "data")
    acquire.header_dump(proj, e)
    cols = harmonise.ColumnMap.load(harmonise.draft_column_map(proj, "dme_p", 2023)).columns
    assert cols["DME_Tot_Suplr_Benes"] == "dme_tot_suplr_benes"  # exact map wins over the DME_ drop prefix
    assert cols["DME_Sprsn_Ind"] is None and cols["POS_Tot_Suplrs"] is None and cols["Bene_Avg_Age"] is None
    assert cols["Rfrg_Prvdr_Spclty_Desc"] == "prvdr_type"
    assert cols["Mystery_Col"] == harmonise.TODO


def test_prefix_columns_for_chronic_conditions(proj, tmp_path):
    header = ["Rndrng_NPI", "Tot_HCPCS_Cds", "Tot_Benes", "Tot_Srvcs", "Tot_Sbmtd_Chrg", "Tot_Mdcr_Alowd_Amt",
              "Tot_Mdcr_Pymt_Amt", "Tot_Mdcr_Stdzd_Amt", "Bene_Avg_Risk_Scre", "Bene_Dual_Cnt",
              "Bene_CC_BH_Depress_V1_Pct"]
    row = ["1234567893", "20", "150", "900", "1e5", "5e4", "4e4", "3.9e4", "1.21", "", "31"]
    f = write_csv(tmp_path / "prov.csv", header, [row])
    e = acquire.register(proj, f, "cms_prov", 2023, "data")
    acquire.header_dump(proj, e)
    p = harmonise.draft_column_map(proj, "cms_prov", 2023)
    assert harmonise.ColumnMap.load(p).columns["Bene_CC_BH_Depress_V1_Pct"] == "bene_cc_bh_depress_v1_pct"
    p.write_text(p.read_text().replace("reviewed: false", "reviewed: true"))
    df = pl.read_parquet(harmonise.harmonise(proj, "cms_prov", 2023))
    assert df["bene_dual_cnt"].to_list() == [None]  # suppressed -> null, not 0
    assert df["bene_cc_bh_depress_v1_pct"].to_list() == [31.0]
