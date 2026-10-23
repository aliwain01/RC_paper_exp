"""End-to-end Phase 1 through the CLI on tiny fixture files: register -> headers -> draft map -> review ->
harmonise -> labels -> G1 report. Fixture values are synthetic and exist only to exercise the code."""

from __future__ import annotations

import json

import polars as pl

from scenario_fraud import cli
from tests.conftest import write_csv
from tests.test_harmonise import PS_HEADER, ROW
from tests.test_labels import HDR, rec

PROV_HEADER = ["Rndrng_NPI", "Rndrng_Prvdr_Last_Org_Name", "Tot_HCPCS_Cds", "Tot_Benes", "Tot_Srvcs",
               "Tot_Sbmtd_Chrg", "Tot_Mdcr_Alowd_Amt", "Tot_Mdcr_Pymt_Amt", "Tot_Mdcr_Stdzd_Amt",
               "Bene_Avg_Risk_Scre", "Bene_Dual_Cnt"]
NPIS = ["1111111112", "2222222224", "3333333336"]


def _review(path):
    path.write_text(path.read_text().replace("reviewed: false", "reviewed: true")
                    .replace('"Some_New_Col": TODO', '"Some_New_Col": null'))


def test_phase1_end_to_end(proj, tmp_path, capsys):
    for y in (2020, 2021):
        prov = write_csv(tmp_path / f"prov{y}.csv", PROV_HEADER,
                         [[n, "X", "10", "100", "500", "1", "1", "1", "1", "1.1", "20"] for n in NPIS])
        ps_rows = [[n, *ROW[1:]] for n in NPIS]
        for r, srv in zip(ps_rows, ("60", "60", "10")):  # third NPI is below the 50-service volume rule
            r[PS_HEADER.index("Tot_Srvcs")] = srv
        ps = write_csv(tmp_path / f"ps{y}.csv", PS_HEADER, ps_rows)
        assert cli.main(["phase1", "register", "--source", "cms_prov", "--year", str(y), "--kind", "data", str(prov)]) == 0
        assert cli.main(["phase1", "register", "--source", "cms_ps", "--year", str(y), "--kind", "data", str(ps)]) == 0
    leie = write_csv(tmp_path / "UPDATED.csv", HDR, [
        rec("A", NPIS[0], "1128b7", "20220301"),
        rec("N", "0000000000", "1128a1", "20220301"),
        rec("Q", "9999999995", "1128b6", "20230301"),   # NPI never seen in Part B
    ])
    cli.main(["phase1", "register", "--source", "leie", "--year", "2026", "--kind", "current",
              "--as-of", "2026-10-12", str(leie)])
    assert cli.main(["phase1", "verify"]) == 0
    assert cli.main(["phase1", "headers"]) == 0
    for ds in ("cms_prov", "cms_ps"):
        for y in (2020, 2021):
            cli.main(["phase1", "draft-colmap", "--dataset", ds, "--year", str(y)])
            _review(proj.column_maps / f"{ds}_{y}.yaml")
    assert cli.main(["phase1", "harmonise"]) == 0
    assert cli.main(["phase1", "labels"]) == 0
    capsys.readouterr()
    assert cli.main(["phase1", "report"]) == 0

    lab = pl.read_parquet(proj.processed / "labels_primary_W3.parquet")
    assert lab.filter(pl.col("npi") == NPIS[0])["y"].to_list() == [1, 1]
    rep = json.loads((proj.phase1 / "G1_report.json").read_text())
    uniq = {r["tier"]: r for r in rep["positives"]["unique"]}
    assert uniq["primary"]["unique_positive_npis"] == 1 and uniq["primary"]["after_volume_rule"] == 1
    nm = {r["tier"]: r for r in rep["npi_match"]}["primary"]
    assert (nm["events"], nm["no_npi"], nm["npis_in_part_b"], nm["npis_not_in_part_b"]) == (3, 1, 1, 1)
    assert rep["dmepos_2024"] == {"dme_s": None, "dme_p": None}
    md = (proj.phase1 / "G1_report.md").read_text()
    assert "still unknown" in md and "Positive NPI counts" in md
    runs = list((proj.phase1 / "runs").glob("*_report.json"))
    assert runs and json.loads(runs[0].read_text())["status"] == "ok"


def test_report_with_no_data_says_so(proj):
    assert cli.main(["phase1", "report"]) == 0
    md = (proj.phase1 / "G1_report.md").read_text()
    assert "No files registered" in md and "_Not available" in md
