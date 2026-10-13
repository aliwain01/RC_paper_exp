"""C4 cohort and peer cells (DESIGN §7.2; CONTRACTS §2.1): division and RUCA mapping, the 50-service rule on
visible non-drug services, one-file-only NPIs, in-cohort cell counts, the peer hierarchy with fallbacks,
loud failures and determinism. Hand-built frames pin the rules; the synthetic world checks the whole path."""

from __future__ import annotations

import json

import polars as pl
import pytest
import yaml

from scenario_fraud import cohort
from scenario_fraud.cohort import CohortConfig, CohortError
from tests.synth import make_synthetic

Y = 2022


@pytest.fixture
def cfg(proj) -> CohortConfig:
    return CohortConfig.load(proj)


def P(npi, spec="Cardiology", ent="I", state="TX", ruca="1", nondrug=60.0, drug=0.0, in_ps=True, in_prov=True,
      prov_attrs=True, part="Y"):
    return dict(npi=npi, spec=spec, ent=ent, state=state, ruca=ruca, nondrug=nondrug, drug=drug, in_ps=in_ps,
                in_prov=in_prov, prov_attrs=prov_attrs, part=part)


def frames(provs: list[dict], year: int = Y) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Harmonised-shape cms_ps / cms_prov frames: one non-drug row (99213, O) and optionally one drug row."""
    ps, prov = [], []
    for p in provs:
        attrs = {"ent_cd": p["ent"], "state": p["state"], "ruca": p["ruca"], "prvdr_type": p["spec"],
                 "mdcr_prtcptg_ind": p["part"]}
        if p["in_ps"]:
            ps.append({"npi": p["npi"], **attrs, "hcpcs_cd": "99213", "hcpcs_drug_ind": "N", "place_of_srvc": "O",
                       "tot_benes": 20, "tot_srvcs": float(p["nondrug"])})
            if p["drug"]:
                ps.append({"npi": p["npi"], **attrs, "hcpcs_cd": "J1100", "hcpcs_drug_ind": "Y",
                           "place_of_srvc": "O", "tot_benes": 15, "tot_srvcs": float(p["drug"])})
        if p["in_prov"]:
            row = {"npi": p["npi"], "tot_benes": 100, "tot_srvcs": 1000.0}
            row.update(attrs if p["prov_attrs"] else {k: None for k in attrs})
            prov.append(row)
    ps_schema = {"npi": pl.String, "ent_cd": pl.String, "state": pl.String, "ruca": pl.String,
                 "prvdr_type": pl.String, "mdcr_prtcptg_ind": pl.String, "hcpcs_cd": pl.String,
                 "hcpcs_drug_ind": pl.String, "place_of_srvc": pl.String, "tot_benes": pl.Int64,
                 "tot_srvcs": pl.Float64}
    prov_schema = {"npi": pl.String, "tot_benes": pl.Int64, "tot_srvcs": pl.Float64, "ent_cd": pl.String,
                   "state": pl.String, "ruca": pl.String, "prvdr_type": pl.String, "mdcr_prtcptg_ind": pl.String}
    yr = pl.lit(year, dtype=pl.Int32).alias("year")
    return (pl.DataFrame(ps, schema=ps_schema).with_columns(yr),
            pl.DataFrame(prov, schema=prov_schema).with_columns(yr))


def run(provs, cfg, year=Y) -> dict[str, dict]:
    df, _ = cohort.cohort_year(*frames(provs, year), year, cfg)
    return {r["npi"]: r for r in df.to_dicts()}


def npis(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:06d}" for i in range(n)]


# ------------------------------------------------------------------------------------------- mappings

def test_census_divisions_incl_territories_and_military(cfg):
    assert len(cfg.divisions) == 51 and len(set(cfg.divisions.values())) == 9
    states = ["TX", "tx", " NY ", "DC", "AK", "PR", "VI", "GU", "AS", "MP", "AA", "AE", "AP", "ZZ", "XX", None]
    got = pl.DataFrame({"state": states}, schema={"state": pl.String}).select(
        cohort.census_division_expr(cfg))["state"].to_list()
    assert got[:5] == ["West South Central", "West South Central", "Middle Atlantic", "South Atlantic", "Pacific"]
    assert set(got[5:]) == {"other"}


@pytest.mark.parametrize("edit, match", [
    (lambda d: d["divisions"]["Pacific"].append("TX"), "listed under"),
    (lambda d: d["divisions"]["Mountain"].remove("WY"), "51 codes"),
    (lambda d: d["divisions"]["Mountain"].append("Wyo"), "two-letter"),
    (lambda d: d.update(other="Pacific"), "distinct `other`"),
])
def test_division_config_is_validated(proj, edit, match):
    f = proj.configs / "census_divisions.yaml"
    doc = yaml.safe_load(f.read_text())
    edit(doc)
    f.write_text(yaml.safe_dump(doc))
    with pytest.raises(CohortError, match=match):
        CohortConfig.load(proj)


def test_ruca_classes_incl_decimals_and_nulls(cfg):
    codes = ["1", "1.0", "2.1", "3", "4", "4.1", "6", "7.2", "9", "10", "10.1", "10.3", "99", "0", "", "abc", None, " 5 "]
    want = ["urban", "urban", "urban", "urban", "large_rural", "large_rural", "large_rural", "small_rural",
            "small_rural", "isolated", "isolated", "isolated", "unknown", "unknown", "unknown", "unknown",
            "unknown", "large_rural"]
    got = pl.DataFrame({"ruca": codes}, schema={"ruca": pl.String}).select(cohort.ruca_class_expr(cfg))
    assert got.to_series().to_list() == want


# ------------------------------------------------------------------------------------------- cohort rule

def test_threshold_uses_visible_nondrug_services_only(cfg):
    r = run([
        P("A", nondrug=49, drug=5000),       # drug units never count
        P("B", nondrug=50),                  # exactly the threshold: in
        P("C", nondrug=49.5),
        P("D", nondrug=10, drug=0),          # by-Provider tot_srvcs = 1000 does not count either
    ], cfg)
    assert r["A"]["nondrug_srvcs"] == 49 and r["A"]["cohort_exclusion_reason"] == cohort.BELOW_MIN_SERVICES
    assert r["B"]["in_cohort"] and r["B"]["cohort_exclusion_reason"] is None
    assert not r["C"]["in_cohort"] and not r["D"]["in_cohort"]
    assert r["B"]["prov_tot_benes"] == 100


def test_nondrug_sum_spans_rows_and_places(cfg):
    ps, prov = frames([P("A", nondrug=30)])
    extra = ps.with_columns(pl.lit("F").alias("place_of_srvc"), pl.lit(25.0).alias("tot_srvcs"))
    df, _ = cohort.cohort_year(pl.concat([ps, extra]), prov, Y, cfg)
    assert df.row(0, named=True)["nondrug_srvcs"] == 55 and df["in_cohort"].all()


def test_npis_in_only_one_part_b_file_get_a_reason(cfg):
    r = run([
        P("A"),
        P("E", in_ps=False, spec="Podiatry"),                    # summary row only (all service rows suppressed)
        P("F", in_prov=False),                                    # service rows only
        P("G", in_ps=False, prov_attrs=False),                    # summary row only, no attributes there
        P("H", spec=None),                                        # no specialty: no peer cell possible
    ], cfg)
    assert r["E"]["cohort_exclusion_reason"] == cohort.NOT_IN_SERVICE_FILE and not r["E"]["in_cohort"]
    assert r["E"]["nondrug_srvcs"] == 0 and r["E"]["prvdr_type"] == "Podiatry" and r["E"]["prov_tot_benes"] == 100
    assert r["F"]["cohort_exclusion_reason"] == cohort.NOT_IN_PROVIDER_FILE and r["F"]["prov_tot_benes"] is None
    assert r["G"]["cohort_exclusion_reason"] == cohort.NOT_IN_SERVICE_FILE
    assert r["G"]["peer_cell"] is None and r["G"]["cell_ent"] is None and r["G"]["n_cell_ent"] is None
    assert r["H"]["cohort_exclusion_reason"] == cohort.MISSING_PEER_ATTRIBUTES and r["H"]["cell_spec_ent"] is None
    assert r["H"]["peer_level"] == "ent"
    assert r["A"]["in_cohort"] and set(r) == {"A", "E", "F", "G", "H"}


# ------------------------------------------------------------------------------------------- peer cells

def _hierarchy_world():
    provs = (
        [P(n, state="TX") for n in npis("cTX", 200)]                          # 200 in-cohort: refined
        + [P(n, state="OK", nondrug=10) for n in npis("cOK", 30)]             # same cell, out of cohort
        + [P(n, state="NY") for n in npis("cNY", 199)]                        # 199: not refined
        + [P(n, spec="Podiatry", state="CA") for n in npis("pI", 99)]         # 99 < min_cell: ent
        + [P(n, spec="Podiatry", state="CA", nondrug=5) for n in npis("pX", 50)]
        + [P(n, spec="Podiatry", ent="O", state="CA") for n in npis("pO", 100)]  # exactly min_cell
    )
    return provs


def test_cell_counts_count_in_cohort_providers_only(cfg):
    r = run(_hierarchy_world(), cfg)
    a = r["cTX000000"]
    assert a["cell_spec_ent_div"] == "2022|I|Cardiology|West South Central"
    assert a["cell_spec_ent"] == "2022|I|Cardiology" and a["cell_ent"] == "2022|I"
    assert a["n_cell_spec_ent_div"] == 200
    assert a["n_cell_spec_ent"] == 200 + 199                     # the 30 out-of-cohort OK providers excluded
    assert a["n_cell_ent"] == 200 + 199 + 99
    assert r["cOK000000"]["n_cell_spec_ent_div"] == 200 and not r["cOK000000"]["in_cohort"]
    assert r["pX000000"]["n_cell_spec_ent"] == 99


def test_peer_hierarchy_and_fallbacks(cfg):
    assert (cfg.refine_min, cfg.min_cell) == (200, 100)
    r = run(_hierarchy_world(), cfg)
    assert (r["cTX000000"]["peer_level"], r["cTX000000"]["peer_cell"]) == (
        "spec_ent_div", "2022|I|Cardiology|West South Central")
    assert (r["cNY000000"]["peer_level"], r["cNY000000"]["peer_cell"]) == ("spec_ent", "2022|I|Cardiology")
    assert (r["pI000000"]["peer_level"], r["pI000000"]["peer_cell"]) == ("ent", "2022|I")
    assert (r["pO000000"]["peer_level"], r["pO000000"]["peer_cell"]) == ("spec_ent", "2022|O|Podiatry")
    # out-of-cohort rows get the cell their attributes map to (they never enlarge it)
    assert r["cOK000000"]["peer_level"] == "spec_ent_div"      # OK is West South Central too: refined cell
    assert r["pX000000"]["peer_level"] == "ent"


def test_peer_thresholds_come_from_config(proj):
    f = proj.configs / "experiment.yaml"
    doc = yaml.safe_load(f.read_text())
    doc["cohort"]["peer"].update(refine_min=150, min_cell=99)
    f.write_text(yaml.safe_dump(doc))
    r = run(_hierarchy_world(), CohortConfig.load(proj))
    assert r["cNY000000"]["peer_level"] == "spec_ent_div" and r["pI000000"]["peer_level"] == "spec_ent"
    doc["cohort"]["peer"].update(refine_min=50, min_cell=99)
    f.write_text(yaml.safe_dump(doc))
    with pytest.raises(CohortError, match="min_cell"):
        CohortConfig.load(proj)
    del doc["cohort"]["peer"]
    f.write_text(yaml.safe_dump(doc))
    with pytest.raises(CohortError, match="incomplete"):
        CohortConfig.load(proj)


def test_cells_are_per_year(cfg):
    df, _ = cohort.cohort_year(*frames([P("A")], 2019), 2019, cfg)
    row = df.row(0, named=True)
    assert row["year"] == 2019 and row["peer_cell"] == "2019|I" and df.schema["year"] == pl.Int32


# ------------------------------------------------------------------------------------------- failures

@pytest.mark.parametrize("breaker, match", [
    (lambda ps, pv: (ps.with_columns(pl.lit("X").alias("hcpcs_drug_ind")), pv), "hcpcs_drug_ind"),
    (lambda ps, pv: (ps.with_columns(pl.lit(None, pl.String).alias("hcpcs_drug_ind")), pv), "hcpcs_drug_ind"),
    (lambda ps, pv: (pl.concat([ps, ps]), pv), "duplicated cms_ps grain"),
    (lambda ps, pv: (ps, pl.concat([pv, pv])), "duplicated NPIs in cms_prov"),
    (lambda ps, pv: (ps.with_columns(pl.lit(2021, pl.Int32).alias("year")), pv), "not 2022"),
    (lambda ps, pv: (ps.with_columns(pl.lit("A|B").alias("prvdr_type")), pv), "separator"),
    (lambda ps, pv: (ps, pv.with_columns(pl.lit("A|B").alias("prvdr_type"))), "separator"),
    (lambda ps, pv: (ps.with_columns(pl.lit(None, pl.Float64).alias("tot_srvcs")), pv), "tot_srvcs"),
    (lambda ps, pv: (ps.drop("place_of_srvc"), pv), "missing"),
    (lambda ps, pv: (ps, pv.drop("tot_benes")), "missing"),
])
def test_violated_assumptions_fail_loudly(cfg, breaker, match):
    ps, pv = breaker(*frames([P("A"), P("B")]))
    with pytest.raises(CohortError, match=match):
        cohort.cohort_year(ps, pv, Y, cfg)


# ------------------------------------------------------------------------------------------- determinism

def test_attributes_first_non_null_in_grain_order_and_order_free(cfg):
    ps, prov = frames([P("A", ruca=None), P("B")])
    rows = pl.concat([
        ps.filter(pl.col("npi") == "A").with_columns(pl.lit(c).alias("hcpcs_cd"), pl.lit(r, pl.String).alias("ruca"))
        for c, r in (("99214", "4.1"), ("99212", None), ("99213", "7"), ("99211", None))
    ] + [ps.filter(pl.col("npi") == "B")])
    results = []
    for seed in range(5):
        df, summary = cohort.cohort_year(rows.sample(fraction=1.0, shuffle=True, seed=seed), prov, Y, cfg)
        results.append(df)
    assert all(r.equals(results[0]) for r in results)
    a = results[0].filter(pl.col("npi") == "A").row(0, named=True)
    assert a["ruca"] == "7" and a["ruca_class"] == "small_rural"    # 99213 is the first non-null in grain order
    assert summary["attribute_conflicts"]["ruca"] == 1 and summary["attribute_conflicts"]["state"] == 0


# ------------------------------------------------------------------------------------------- whole path

def test_build_cohort_on_synthetic_world(proj):
    truth = make_synthetic(proj, n_npi=800, years=[2021, 2022], seed=5)
    out = cohort.build_cohort(proj)
    assert out.columns == list(cohort.COLUMNS)
    assert pl.read_parquet(proj.processed / "cohort.parquet").equals(out)
    assert sorted(out["year"].unique().to_list()) == [2021, 2022]
    for y in (2021, 2022):
        ps = pl.read_parquet(proj.interim / "cms_ps" / f"{y}.parquet")
        prov = pl.read_parquet(proj.interim / "cms_prov" / f"{y}.parquet")
        c = out.filter(pl.col("year") == y)
        assert set(c["npi"].to_list()) == set(ps["npi"].to_list()) | set(prov["npi"].to_list())
        assert c["npi"].is_unique().all() and c["npi"].to_list() == sorted(c["npi"].to_list())
        # independent recomputation of the volume rule
        nd = ps.filter(pl.col("hcpcs_drug_ind") != "Y").group_by("npi").agg(pl.col("tot_srvcs").sum().alias("nd"))
        exp = set(nd.filter(pl.col("nd") >= 50).join(prov.select("npi"), on="npi")["npi"].to_list())
        assert set(c.filter("in_cohort")["npi"].to_list()) == exp
        assert set(c.filter(pl.col("cohort_exclusion_reason") == cohort.NOT_IN_SERVICE_FILE)["npi"].to_list()) == (
            set(prov["npi"].to_list()) - set(ps["npi"].to_list()))
        assert (c["cohort_exclusion_reason"] == cohort.BELOW_MIN_SERVICES).any()
        # stored counts and levels agree with the rule
        inc = c.filter("in_cohort")
        for lvl in ("ent", "spec_ent", "spec_ent_div"):
            n = inc.group_by(f"cell_{lvl}").len()
            chk = c.join(n, on=f"cell_{lvl}", how="left").select(
                (pl.col(f"n_cell_{lvl}") == pl.col("len").fill_null(0)).all()).item()
            assert chk, lvl
        rule = (pl.when(pl.col("n_cell_spec_ent_div") >= 200).then(pl.lit("spec_ent_div"))
                .when(pl.col("n_cell_spec_ent") >= 100).then(pl.lit("spec_ent")).otherwise(pl.lit("ent")))
        assert c.select((pl.col("peer_level") == rule).all()).item()
        assert c.select((pl.col("peer_cell") == pl.when(pl.col("peer_level") == "ent").then("cell_ent")
                         .when(pl.col("peer_level") == "spec_ent").then("cell_spec_ent")
                         .otherwise("cell_spec_ent_div")).all()).item()
    meta = json.loads((proj.processed / "cohort.meta.json").read_text())
    assert meta["config"]["min_cell"] == 100 and meta["per_year"]["2022"]["rows"] == out.filter(pl.col("year") == 2022).height
    assert sum(meta["per_year"]["2022"]["peer_level_mix"].values()) == meta["per_year"]["2022"]["in_cohort"]
    # determinism, and a run for a subset of years writes exactly that subset
    assert cohort.build_cohort(proj).equals(out)
    sub = cohort.build_cohort(proj, years=[2022])
    assert sub.equals(out.filter(pl.col("year") == 2022))
    assert pl.read_parquet(proj.processed / "cohort.parquet")["year"].unique().to_list() == [2022]
    assert truth["n_npi"] == 800


def test_peer_refinement_happens_at_scale(proj):
    make_synthetic(proj, n_npi=12_000, years=[2022], seed=2)
    out = cohort.build_cohort(proj)
    levels = set(out.filter("in_cohort")["peer_level"].to_list())
    assert levels == {"spec_ent_div", "spec_ent", "ent"}


def test_missing_or_lonely_years_fail(proj):
    make_synthetic(proj, n_npi=200, years=[2022], seed=1)
    with pytest.raises(CohortError, match="harmonised files missing"):
        cohort.build_cohort(proj, years=[2023])
    (proj.interim / "cms_prov" / "2022.parquet").rename(proj.interim / "cms_prov" / "2023.parquet")
    with pytest.raises(CohortError, match=r"only one of cms_ps / cms_prov.*2022, 2023"):
        cohort.build_cohort(proj)


def test_no_data_fails(proj):
    with pytest.raises(CohortError, match="no harmonised Part B years"):
        cohort.build_cohort(proj)
