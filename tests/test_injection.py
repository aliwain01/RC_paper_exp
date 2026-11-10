"""C6 transforms on hand-built frames: exact arithmetic, summary-total updates, re-suppression (EXPERIMENT_PLAN §6)."""

from __future__ import annotations

import polars as pl
import pytest

from scenario_fraud import features, injection

NPI = "1234567893"


def _ps(rows):
    base = {"npi": NPI, "ent_cd": "I", "state": "TX", "ruca": "1", "prvdr_type": "Internal Medicine",
            "mdcr_prtcptg_ind": "Y", "hcpcs_drug_ind": "N", "place_of_srvc": "O", "avg_sbmtd_chrg": 100.0,
            "avg_mdcr_alowd_amt": 50.0, "avg_mdcr_pymt_amt": 40.0, "avg_mdcr_stdzd_amt": 45.0, "year": 2022}
    df = pl.DataFrame([{**base, **r} for r in rows])
    return df.with_columns(pl.col("tot_benes").cast(pl.Int64), pl.col("tot_srvcs").cast(pl.Float64),
                           pl.col("tot_bene_day_srvcs").cast(pl.Float64), pl.col("year").cast(pl.Int32))


@pytest.fixture
def icfg(proj):
    return injection.InjectionConfig.load(proj)


def _tabs():
    codes = ["99214", "99215"]
    return {"cell": pl.DataFrame({"peer_cell": ["c", "c"], "hcpcs_cd": codes, "place_of_srvc": ["O", "O"],
                                  **{c: [65.0, 70.0] for c in injection.AMT}, "n": [5, 5]}),
            "nat_place": pl.DataFrame({"hcpcs_cd": codes, "place_of_srvc": ["O", "O"],
                                       **{c: [75.0, 80.0] for c in injection.AMT}, "n": [9, 9]}),
            "nat": pl.DataFrame({"hcpcs_cd": codes, **{c: [85.0, 90.0] for c in injection.AMT}})}


def test_upcoding_moves_services_and_prices_new_rows(icfg):
    ps = _ps([{"hcpcs_cd": "99213", "tot_srvcs": 100, "tot_benes": 60, "tot_bene_day_srvcs": 90},
              {"hcpcs_cd": "99212", "tot_srvcs": 40, "tot_benes": 30, "tot_bene_day_srvcs": 40},
              {"hcpcs_cd": "99214", "tot_srvcs": 20, "tot_benes": 15, "tot_bene_day_srvcs": 20, "avg_mdcr_alowd_amt": 77.0},
              {"hcpcs_cd": "J1100", "hcpcs_drug_ind": "Y", "tot_srvcs": 5.5, "tot_benes": 12, "tot_bene_day_srvcs": 5}])
    who = pl.DataFrame({"npi": [NPI], "peer_cell": ["c"], "x": [0.5]})
    out = injection.upcoding(ps, who, pl.DataFrame({"npi": [NPI], "prov_benes": [70]}), _tabs(), icfg)
    by = {r["hcpcs_cd"]: r for r in out.to_dicts()}
    assert by["99213"]["tot_srvcs"] == 50 and by["99212"]["tot_srvcs"] == 20
    assert by["99214"]["tot_srvcs"] == 20 + 35 and by["99215"]["tot_srvcs"] == 35      # 70 moved, split 50/50
    assert out["tot_srvcs"].sum() == ps["tot_srvcs"].sum()                               # services conserved
    assert by["99214"]["avg_mdcr_alowd_amt"] == 77.0                                     # provider's own average kept
    assert by["99215"]["avg_mdcr_alowd_amt"] == 70.0                                     # new row: peer-cell median
    assert by["99214"]["tot_benes"] == min(15 + 22, 70) and by["99215"]["tot_benes"] == 23
    assert by["J1100"]["tot_srvcs"] == 5.5                                               # drug line untouched


def test_overutilization_and_charge_inflation(icfg):
    ps = _ps([{"hcpcs_cd": "99213", "tot_srvcs": 10, "tot_benes": 12, "tot_bene_day_srvcs": 9},
              {"hcpcs_cd": "J1100", "hcpcs_drug_ind": "Y", "tot_srvcs": 5, "tot_benes": 12, "tot_bene_day_srvcs": 5}])
    who = pl.DataFrame({"npi": [NPI], "peer_cell": ["c"], "x": [1.6]})
    o = injection.overutilization(ps, who)
    assert o["tot_srvcs"].to_list() == [16.0, 5.0] and o["tot_benes"].to_list() == [12, 12]
    assert o["tot_bene_day_srvcs"].to_list() == [14.0, 5.0]
    c = injection.charge_inflation(ps, who)
    assert c["avg_sbmtd_chrg"].to_list() == [160.0, 160.0] and c["avg_mdcr_alowd_amt"].to_list() == [50.0, 50.0]
    other = injection.overutilization(ps, who.with_columns(pl.lit("9999999995").alias("npi")))
    assert other.equals(ps)                                                               # other providers untouched


def test_apply_year_updates_totals_and_resuppresses(proj, icfg):
    fcfg = features.FeatureConfig.load(proj)
    ps = _ps([{"hcpcs_cd": "99213", "tot_srvcs": 100, "tot_benes": 14, "tot_bene_day_srvcs": 90},
              {"hcpcs_cd": "99212", "tot_srvcs": 12, "tot_benes": 12, "tot_bene_day_srvcs": 12}])
    prov = pl.DataFrame({"npi": [NPI], "tot_hcpcs_cds": [2], "tot_benes": [80], "tot_srvcs": [500.0],
                         "tot_sbmtd_chrg": [1e5], "tot_mdcr_alowd_amt": [5e4], "tot_mdcr_pymt_amt": [4e4],
                         "tot_mdcr_stdzd_amt": [4.5e4], "year": [2022]}).with_columns(pl.col("year").cast(pl.Int32))
    empty_ds = features.empty_frame(proj, "dme_s", 2022)
    empty_dp = features.empty_frame(proj, "dme_p", 2022)
    who = pl.DataFrame({"npi": [NPI], "peer_cell": ["c"], "x_overutilization": [2.0]})

    class _R:      # reference tables are only priced from for new rows; overutilization adds none
        codes = pl.DataFrame(schema={"level": pl.String, "peer_cell": pl.String, "hcpcs_cd": pl.String,
                                     "provider_share": pl.Float64})
    frames = {"cms_ps": ps, "cms_prov": prov, "dme_s": empty_ds, "dme_p": empty_dp,
              "_full": {"cms_ps": ps, "cms_prov": prov, "dme_s": empty_ds, "dme_p": empty_dp}}
    coh = pl.DataFrame({"npi": [NPI], "peer_cell": ["c"], "in_cohort": [True], "injected": [False]})
    edited, log = injection.apply_year(frames, who, coh, _R(), fcfg, icfg, 0, 2022)
    p = edited["cms_prov"].row(0, named=True)
    assert p["tot_srvcs"] == 500 + 112                                   # visible services doubled: +112
    assert p["tot_mdcr_alowd_amt"] == pytest.approx(5e4 + 112 * 50)
    assert p["tot_hcpcs_cds"] == 2 and p["tot_benes"] == 80
    assert edited["cms_ps"]["tot_srvcs"].to_list() == [200.0, 24.0]       # rows with > 10 beneficiaries kept
    assert set(edited["cms_ps"]["year"].to_list()) == {2022}


def test_resuppression_drops_small_rows(proj, icfg):
    ps = _ps([{"hcpcs_cd": "99213", "tot_srvcs": 30, "tot_benes": 20, "tot_bene_day_srvcs": 30},
              {"hcpcs_cd": "99212", "tot_srvcs": 30, "tot_benes": 12, "tot_bene_day_srvcs": 30}])
    who = pl.DataFrame({"npi": [NPI], "peer_cell": ["c"], "x": [0.5]})
    out = injection.upcoding(ps, who, pl.DataFrame({"npi": [NPI], "prov_benes": [40]}), _tabs(), icfg)
    sup = icfg["suppression"]["cms_ps"]
    kept = out.filter(pl.col(sup["field"]) > int(sup["max_suppressed"]))
    assert "99212" not in kept["hcpcs_cd"].to_list()                      # 12 -> 6 beneficiaries: suppressed


def test_config_validates_answer_groups(proj):
    import yaml

    p = proj.configs / "injection.yaml"
    d = yaml.safe_load(p.read_text())
    d["schemes"]["upcoding"]["groups"] = ["context"]
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(injection.InjectionError, match="scheme groups"):
        injection.InjectionConfig.load(proj)
