"""C3: LEIE parsing, union across sources, label rule (DESIGN §5.4), horizon flags, coverage gaps."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from scenario_fraud import acquire, labels
from tests.conftest import write_csv

HDR = ["LASTNAME", "FIRSTNAME", "MIDNAME", "BUSNAME", "GENERAL", "SPECIALTY", "UPIN", "NPI", "DOB", "ADDRESS",
       "CITY", "STATE", "ZIP", "EXCLTYPE", "EXCLDATE", "REINDATE", "WAIVERDATE", "WVRSTATE"]


def rec(last, npi, excltype, excldate, reindate="00000000", busname=""):
    return [last, "A", "", busname, "IND", "MD", "", npi, "19600101", "1 Main", "Town", "TX", "75001",
            excltype, excldate, reindate, "00000000", ""]


A, B, C, D = "1111111112", "2222222224", "3333333336", "4444444448"
SNAP = date(2026, 10, 12)


@pytest.fixture
def cfg(proj):
    return labels.LabelConfig.load(proj)


def test_read_parses_and_classifies(proj, tmp_path, cfg):
    f = write_csv(tmp_path / "UPDATED.csv", HDR, [
        rec("X", A, "1128b7", "20220501"),
        rec("Y", "0000000000", "1128a1", "20200115"),
        rec("Z", "12AB", "1128(b)(4)", "20210301"),
        rec("W", D, "1128Aa", "20190101"),
    ])
    df = labels.read_leie_file(f, cfg)
    assert df["authority"].to_list() == ["1128(b)(7)", "1128(a)(1)", "1128(b)(4)", "1128A(a)"]
    assert df["npi_status"].to_list() == ["valid", "none", "invalid", "valid"]
    assert df["npi"].to_list() == [A, None, None, D]
    assert df["excl_date"].to_list()[0] == date(2022, 5, 1)
    assert df["rein_date"].null_count() == 4


def test_unknown_excltype_fails_with_counts(proj, tmp_path, cfg):
    f = write_csv(tmp_path / "u.csv", HDR, [rec("X", A, "1128zz9", "20220501")] * 2)
    with pytest.raises(labels.LabelError, match=r"'1128zz9'×2"):
        labels.read_leie_file(f, cfg)


def test_missing_columns_and_bad_dates_fail(proj, tmp_path, cfg):
    f = write_csv(tmp_path / "m.csv", [h for h in HDR if h != "EXCLTYPE"], [rec("X", A, "1128b7", "2022")[:13] + ["2022", "0", "0", ""]])
    with pytest.raises(labels.LabelError, match="key columns missing"):
        labels.read_leie_file(f, cfg)
    g = write_csv(tmp_path / "d.csv", HDR, [rec("X", A, "1128b7", "2022-05-01")])
    with pytest.raises(labels.LabelError, match="unparseable dates"):
        labels.read_leie_file(g, cfg)


def _register_sources(proj, tmp_path):
    cur = write_csv(tmp_path / "cur" / "UPDATED.csv", HDR, [
        rec("A", A, "1128b7", "20220501"),
        rec("D", D, "1128b4", "20190301"),
        rec("D", D, "1128a1", "20230601"),
    ])
    snap = write_csv(tmp_path / "snap" / "UPDATED_20190110.csv", HDR, [
        rec("B", B, "1128b4", "20210101"),           # not plausible in a 2019 file, but tests the union
        rec("A", A, "1128b7", "20220501"),
        rec("R", "5555555551", "1128b7", "20180601"),  # later reinstated: only in snapshot + supplement
    ])
    rein = write_csv(tmp_path / "rein" / "2501REIN.csv", HDR, [
        rec("R", "5555555551", "1128b7", "20180601", reindate="20250115"),
    ])
    acquire.register(proj, cur, "leie", 2026, "current", route="oig", as_of="2026-10-12")
    acquire.register(proj, snap, "leie", 2019, "snapshot", route="wayback", as_of="2019-01-10")
    acquire.register(proj, rein, "leie", 2025, "supplement_rein", route="oig", as_of="2025-01")


def test_union_dedupes_and_recovers_reinstated(proj, tmp_path, cfg):
    _register_sources(proj, tmp_path)
    exc = labels.build_exclusions(proj, cfg)
    assert exc.height == 5  # A counted once across current + snapshot
    a = exc.filter(pl.col("npi") == A).row(0, named=True)
    assert a["in_current"] and a["seen_in"] == "current,snapshot"
    r = exc.filter(pl.col("npi") == "5555555551").row(0, named=True)
    assert not r["in_current"] and r["rein_date"] == date(2025, 1, 15)
    assert (proj.processed / "exclusions.parquet").is_file()
    assert labels.snapshot_date(proj, cfg) == SNAP


def _labelled(proj, tmp_path, cfg, tier="primary", policy="drop", W=3):
    _register_sources(proj, tmp_path)
    exc = labels.build_exclusions(proj, cfg)
    firsts = labels.first_exclusions(exc, cfg)
    py = pl.DataFrame({"npi": [n for n in (A, B, C, D) for _ in range(2018, 2025)],
                       "year": [y for _ in range(4) for y in range(2018, 2025)]})
    out = labels.apply_label_rule(py, firsts, tier, W, SNAP, policy)
    return {(r["npi"], r["year"]): r for r in out.to_dicts()}


def test_label_rule_primary(proj, tmp_path, cfg):
    L = _labelled(proj, tmp_path, cfg)
    # A: first primary exclusion 2022 -> positive 2019-2021, onset unknown 2018, dropped 2022+
    assert L[(A, 2018)]["drop_reason"] == "onset_unknown"
    assert [L[(A, y)]["y"] for y in (2019, 2020, 2021)] == [1, 1, 1]
    assert L[(A, 2022)]["drop_reason"] == "at_or_after_exclusion" and L[(A, 2022)]["y"] is None
    assert L[(A, 2021)]["excl_type"] == "1128(b)(7)" and L[(A, 2021)]["excl_year"] == 2022
    # B: only a removed authority ((b)(4)) -> not a negative, all rows dropped
    assert {L[(B, y)]["drop_reason"] for y in range(2018, 2025)} == {"other_excluded"}
    # C: never excluded -> unlabelled negative in every year
    assert {L[(C, y)]["y"] for y in range(2018, 2025)} == {0}
    # D: (b)(4) 2019 then (a)(1) 2023 -> primary yE = 2023
    assert [L[(D, y)]["y"] for y in (2020, 2021, 2022)] == [1, 1, 1]
    assert L[(D, 2019)]["drop_reason"] == "onset_unknown"


def test_label_rule_tiers_policy_and_window(proj, tmp_path, cfg):
    L = _labelled(proj, tmp_path, cfg, tier="prior_work")
    assert [L[(B, y)]["y"] for y in (2018, 2019, 2020)] == [1, 1, 1]  # (b)(4) is in the FAU set
    L = _labelled(proj, tmp_path, cfg, policy="negative")
    assert [L[(B, y)]["y"] for y in (2018, 2019, 2020)] == [0, 0, 0]
    assert L[(B, 2021)]["drop_reason"] == "at_or_after_exclusion"
    L = _labelled(proj, tmp_path, cfg, W=1)
    assert L[(A, 2021)]["y"] == 1 and L[(A, 2020)]["drop_reason"] == "onset_unknown"


def test_label_horizon(proj, tmp_path, cfg):
    L = _labelled(proj, tmp_path, cfg)
    assert L[(C, 2022)]["label_horizon_complete"] is True
    assert L[(C, 2023)]["label_horizon_complete"] is False
    assert L[(C, 2023)]["label_horizon_years"] == pytest.approx(2.78, abs=0.01)  # DESIGN §5.4: ~2.8 of 3
    assert L[(C, 2024)]["label_horizon_years"] == pytest.approx(1.78, abs=0.01)  # ~1.8


def test_coverage_gaps_and_loss_profile(proj, tmp_path, cfg):
    _register_sources(proj, tmp_path)
    g = labels.coverage_gaps(proj, date(2019, 1, 1), SNAP)
    assert g["snapshot_dates"] == ["2019-01-10", "2026-10-12"]
    assert [i["from"] for i in g["uncovered_intervals"]] == ["2019-01-01", "2019-01-10"]
    full = len(labels._month_range(date(2019, 1, 10), SNAP))
    assert g["uncovered_intervals"][1]["uncovered_months"] == full - 1  # 2025-01 has a REIN supplement
    exc = labels.build_exclusions(proj, cfg)
    prof = labels.reinstatement_loss_profile(exc, cfg, gap_days=max(i["days"] for i in g["uncovered_intervals"]))
    row = prof.row(0, named=True)
    assert row["authority"] == "1128(b)(7)" and row["n_reinstated"] == 1 and row["in_primary"]
