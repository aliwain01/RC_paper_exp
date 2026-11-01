"""C5 features (DESIGN §7.4-7.5; FEATURE_CATALOG; CONTRACTS §2.2).

Hand-built frames pin every feature family to hand-computed values (UP, OU, SM incl. a known JSD, CH, PM, DI, DC,
CX), the beta-binomial shrinkage and its alpha caps, robust z with clipping and every scale fallback, and the
structural zeros. Synthetic worlds (tests/synth.py; never real data, never results) check the injection API,
determinism, label independence, the group map, persisted artefacts and runtime."""

from __future__ import annotations

import math
import shutil
import statistics
import time
from datetime import date

import numpy as np
import polars as pl
import pytest
import yaml
from scipy.spatial.distance import jensenshannon

from scenario_fraud import cohort, features, groups
from scenario_fraud.features import FeatureConfig, FeatureError
from tests.synth import make_synthetic

Y = 2022
PS_SCHEMA = {"npi": pl.String, "hcpcs_cd": pl.String, "hcpcs_drug_ind": pl.String, "place_of_srvc": pl.String,
             "tot_srvcs": pl.Float64, "tot_bene_day_srvcs": pl.Float64, "avg_sbmtd_chrg": pl.Float64,
             "avg_mdcr_alowd_amt": pl.Float64, "avg_mdcr_pymt_amt": pl.Float64, "avg_mdcr_stdzd_amt": pl.Float64}
PROV_SCHEMA = {"npi": pl.String, "tot_benes": pl.Int64, "tot_srvcs": pl.Float64, "tot_hcpcs_cds": pl.Int64,
               "bene_avg_risk_scre": pl.Float64, "bene_dual_cnt": pl.Int64}
DS_SCHEMA = {"npi": pl.String, "hcpcs_cd": pl.String, "suplr_rentl_ind": pl.String, "tot_suplrs": pl.Int64,
             "tot_suplr_clms": pl.Int64, "tot_suplr_srvcs": pl.Float64, "avg_suplr_mdcr_pymt_amt": pl.Float64}
DP_SCHEMA = {"npi": pl.String, "tot_suplr_benes": pl.Int64}


# ------------------------------------------------------------------------------------------- builders

def C(npi, spec="Cardiology", ent="I", div="Pacific", level="spec_ent", in_cohort=True, part="Y", ruca="urban",
      injected=None):
    """A cohort row (CONTRACTS §2.1 columns used by C5)."""
    ce = f"{Y}|{ent}"
    cse, csed = f"{ce}|{spec}", f"{ce}|{spec}|{div}"
    row = dict(npi=npi, year=Y, in_cohort=in_cohort, peer_cell={"spec_ent_div": csed, "spec_ent": cse, "ent": ce}[level],
               peer_level=level, cell_ent=ce, cell_spec_ent=cse, cell_spec_ent_div=csed, prvdr_type=spec, ent_cd=ent,
               census_division=div, ruca_class=ruca, mdcr_prtcptg_ind=part)
    if injected is not None:
        row["injected"] = injected
    return row


def S(npi, code, srv, place="O", days=None, allowed=100.0, sbmtd=200.0, pymt=80.0, stdzd=90.0, drug=None):
    """A visible P&S row."""
    return dict(npi=npi, hcpcs_cd=code, hcpcs_drug_ind=drug or ("Y" if code.startswith("J") else "N"),
                place_of_srvc=place, tot_srvcs=float(srv), tot_bene_day_srvcs=float(srv if days is None else days),
                avg_sbmtd_chrg=sbmtd, avg_mdcr_alowd_amt=allowed, avg_mdcr_pymt_amt=pymt, avg_mdcr_stdzd_amt=stdzd)


def P(npi, benes=100, srvcs=1000.0, cds=10, risk=1.5, dual=30, **cc):
    return dict(npi=npi, tot_benes=benes, tot_srvcs=srvcs, tot_hcpcs_cds=cds, bene_avg_risk_scre=risk,
                bene_dual_cnt=dual, **cc)


def DS(npi, code, rent="N", suplrs=1, clms=20, srv=20.0, pymt=100.0):
    return dict(npi=npi, hcpcs_cd=code, suplr_rentl_ind=rent, tot_suplrs=suplrs, tot_suplr_clms=clms,
                tot_suplr_srvcs=float(srv), avg_suplr_mdcr_pymt_amt=pymt)


def frames(ps, prov, ds=(), dp=(), cc=()):
    yr = pl.lit(Y, dtype=pl.Int32).alias("year")
    return {
        "cms_ps": pl.DataFrame(list(ps), schema=PS_SCHEMA).with_columns(yr),
        "cms_prov": pl.DataFrame(list(prov), schema={**PROV_SCHEMA, **{c: pl.Float64 for c in cc}}).with_columns(yr),
        "dme_s": pl.DataFrame(list(ds), schema=DS_SCHEMA).with_columns(yr),
        "dme_p": pl.DataFrame([dict(npi=n, tot_suplr_benes=b) for n, b in dp], schema=DP_SCHEMA).with_columns(yr),
    }


def cohort_df(rows) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col("year").cast(pl.Int32))


def tune(proj, **sections) -> FeatureConfig:
    """Small-world settings (no hierarchy fallback for tiny cells) plus overrides {section: {key: value}}."""
    f = proj.configs / "features.yaml"
    doc = yaml.safe_load(f.read_text(encoding="utf-8"))
    doc["peer_stats"]["min_defined"] = 1
    doc["patient_mix"].update(min_n=1, min_n_per_term=1)
    doc["charge_index"].update(min_providers_cell=1, min_providers_national=1)
    for sec, kv in sections.items():
        if isinstance(kv, dict):
            doc[sec].update(kv)
        else:
            doc[sec] = kv
    f.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return FeatureConfig.load(proj)


def run(fr, coh, cfg):
    refs = features.fit_peer_reference(fr, coh, Y, cfg)
    raw = features.raw_statistics(fr, coh, refs, Y, cfg)
    ps = features.fit_peer_stats(raw, coh, Y, cfg)
    return refs, raw, ps, features.transform(raw, coh, ps, refs, Y, cfg)


def row(df: pl.DataFrame, npi: str) -> dict:
    return df.filter(pl.col("npi") == npi).row(0, named=True)


def stat(ps, cell: str, fid: str) -> dict:
    return ps.stats.filter((pl.col("peer_cell") == cell) & (pl.col("feature") == fid)).row(0, named=True)


# ------------------------------------------------------------------------------------------- config & groups

def test_config_lists_every_catalog_feature_with_contract_fields(proj):
    cfg = FeatureConfig.load(proj)
    assert [s.id for s in cfg.specs] == list(features.CATALOG)
    for s in cfg.specs:
        assert s.monotone == (1 if s.direction == "up" and s.kind == "robust_z" else 0)
        assert s.kind in features.KINDS and s.suppression in ("V", "T", "VT")
    assert {s.id for s in cfg.specs if s.shrink} >= {"UP1", "UP2", "UP3", "UP4", "SM2", "DC2", "PM4"}
    assert [s.id for s in cfg.one_hot] == ["CX1", "CX2", "CX3", "CX4", "CX6"]
    assert cfg.pruned == () and cfg.z_clip == 10 and cfg.mad_constant == 1.4826
    assert ("L0450", "L0651") in cfg.high_risk and ("K0800", "K0899") in cfg.high_risk


@pytest.mark.parametrize("edit, match", [
    (lambda d: d["features"].pop(3), "missing"),
    (lambda d: d["features"][1].update(monotone=0), "monotone"),
    (lambda d: d["features"][1].update(group="overutilization"), "does not list it"),
    (lambda d: d["features"][0].update(shrink=True), "only robust_z"),
    (lambda d: d["dme_high_risk"]["hcpcs_ranges"].append({"from": "L0999", "to": "K0001"}), "same letter"),
    (lambda d: d.update(pruned=["XX9"]), "unknown IDs"),
    (lambda d: d["transform"].update(z_clip=0), "out of range"),
    (lambda d: d["transform"].update(scale_fallbacks=["iqr", "mad", "std", "parent", "unit"]), "scale_fallbacks"),
])
def test_feature_config_is_validated(proj, edit, match):
    f = proj.configs / "features.yaml"
    doc = yaml.safe_load(f.read_text(encoding="utf-8"))
    edit(doc)
    f.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    with pytest.raises(FeatureError, match=match):
        FeatureConfig.load(proj)


def test_feature_groups_maps_each_column_to_exactly_one_group(proj):
    m = groups.feature_groups(["UP1", "CX1__cardiology", "CX5", "PM4_missing", "DC2", "CH2", "PM1"])
    assert m == {"UP1": "upcoding", "CX1__cardiology": "context", "CX5": "context", "PM4_missing": "patient_mix",
                 "DC2": "dme_concentration", "CH2": "charge_inflation", "PM1": "patient_mix"}
    s7 = groups.feature_groups(["CH2", "PM1", "UP1"], variant="S7")
    assert s7 == {"CH2": "overutilization", "PM1": "context", "UP1": "upcoding"}
    with pytest.raises(groups.GroupsError, match="exactly one group"):
        groups.feature_groups(["UP1", "raw__UP1", "npi"])
    f = proj.configs / "groups.yaml"
    doc = yaml.safe_load(f.read_text(encoding="utf-8"))
    doc["groups"][2]["features"].append("UP1")
    f.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(groups.GroupsError, match="listed in groups"):
        groups.load_groups(proj)


# ------------------------------------------------------------------------------------------- upcoding

def test_upcoding_raw_values_and_structural_zeros(proj):
    cfg = tune(proj)
    ps = [S("A", "99212", 10), S("A", "99213", 20), S("A", "99214", 30), S("A", "99215", 40, allowed=200.0),
          S("A", "99202", 5), S("A", "99205", 15), S("A", "99201", 10), S("A", "93000", 50),
          S("B", "93000", 80),
          S("C", "99283", 10), S("C", "99285", 30), S("C", "93000", 20)]
    fr = frames(ps, [P("A"), P("B"), P("C")])
    refs, raw, pst, feat = run(fr, cohort_df([C("A"), C("B"), C("C")]), cfg)
    a, b, c = row(raw, "A"), row(raw, "B"), row(raw, "C")
    assert (a["UP0"], a["UP1"], a["UP1__k"], a["UP1__n"]) == (1, 0.4, 40.0, 100.0)
    assert a["UP2"] == pytest.approx(0.7)
    assert (a["UP3"], a["UP3__n"]) == (0.75, 20.0)          # 99201 is not in UP3's denominator
    assert a["UP4"] is None and a["UP5"] == pytest.approx(4.0)
    # UP6 counts 99201 as an E&M visit: (60*100 + 40*200 + 30*100) / 130
    assert a["UP6"] == pytest.approx(17000 / 130)
    assert b["UP0"] == 0 and all(b[f] is None for f in ("UP1", "UP2", "UP3", "UP4", "UP5", "UP6"))
    assert (c["UP0"], c["UP4"], c["UP1"], c["UP6"]) == (1, 0.75, None, 100.0)
    fb = row(feat, "B")
    assert fb["UP0"] == 0 and all(fb[f] == 0.0 for f in ("UP1", "UP2", "UP3", "UP4", "UP5", "UP6"))
    assert fb["raw__UP1"] is None and row(feat, "C")["UP1"] == 0.0 and row(feat, "A")["UP4"] == 0.0


# ------------------------------------------------------------------------------------------- shrinkage

def _rate_world(cells: dict[str, list[tuple[int, int]]], extra=()):
    """UP1 = k/n from 99215 (k) and 99211 (n - k) services, one peer cell per specialty."""
    ps, prov, coh = [], [], []
    for spec, kn in cells.items():
        for i, (k, n) in enumerate(kn):
            npi = f"{spec[:3]}{i}"
            ps += [S(npi, "99215", k)] if k else []
            ps += [S(npi, "99211", n - k)] if n - k else []
            prov.append(P(npi))
            coh.append(C(npi, spec=spec, injected=False))
    for npi, spec, k, n in extra:
        ps += [S(npi, "99215", k)] + ([S(npi, "99211", n - k)] if n - k else [])
        prov.append(P(npi))
        coh.append(C(npi, spec=spec, injected=True))
    return frames(ps, prov), cohort_df(coh)


CELLS = {"Cardiology": [(2, 10), (5, 10), (8, 10)],          # overdispersed: alpha by the method of moments
         "Dermatology": [(5, 10), (5, 10), (5, 10)],         # rho < 0: no overdispersion -> cap
         "Oncology": [(0, 10), (10, 10), (0, 10), (10, 10)],  # rho >= 1 -> alpha_min
         "Urology": [(0, 10), (0, 10), (0, 10)]}             # pbar = 0: not identified -> alpha_min


def test_beta_binomial_alpha_and_shrunk_z_exact(proj):
    cfg = tune(proj)
    fr, coh = _rate_world(CELLS, extra=[("INJ", "Cardiology", 10, 10)])
    _, raw, ps, feat = run(fr, coh, cfg)
    # Cardiology: pbar = .5, S = 1.8, I = 3, N = 30, sum n^2 / N = 10 -> rho = (1.8/.25 - 2) / (30 - 10 - 2) = 5.2/18
    alpha = 18 / 5.2 - 1
    card = stat(ps, f"{Y}|I|Cardiology", "UP1")
    assert card["alpha"] == pytest.approx(alpha) and card["prior_mean"] == 0.5 and card["n"] == 3
    shrunk = [(k + alpha * 0.5) / (10 + alpha) for k in (2, 5, 8)]
    assert card["median"] == pytest.approx(0.5)
    assert card["scale"] == pytest.approx(1.4826 * 3 / (10 + alpha)) and card["scale_method"] == "mad"
    assert shrunk[0] == pytest.approx(0.25926, abs=1e-5)
    # z = +-(3/(10+alpha)) / (1.4826 * 3/(10+alpha)) = +-1/1.4826, the injected 10/10 provider (not in the fit)
    # gets (5/(10+alpha)) / (1.4826 * 3/(10+alpha)) = 5/(3*1.4826).
    z = dict(feat.select("npi", "UP1").iter_rows())
    assert z["Car0"] == pytest.approx(-1 / 1.4826) and z["Car2"] == pytest.approx(1 / 1.4826) and z["Car1"] == 0
    assert z["INJ"] == pytest.approx(5 / (3 * 1.4826))
    assert row(feat, "Car0")["raw__UP1"] == 0.2 and row(feat, "INJ")["raw__UP1__n"] == 10.0
    assert stat(ps, f"{Y}|I|Dermatology", "UP1")["alpha"] == 10.0     # cap = 1.0 x median n (10)
    assert stat(ps, f"{Y}|I|Oncology", "UP1")["alpha"] == 1.0
    uro = stat(ps, f"{Y}|I|Urology", "UP1")
    assert uro["alpha"] == 1.0 and uro["prior_mean"] == 0.0


@pytest.mark.parametrize("over, derm, card", [
    ({"alpha_max": 5.0}, 5.0, 18 / 5.2 - 1),            # absolute cap
    ({"alpha_max_median_n": 0.2}, 2.0, 2.0),            # cap = 0.2 x median n = 2 binds both cells
    ({"alpha_min": 3.0}, 10.0, 3.0),                    # floor
])
def test_alpha_caps(proj, over, derm, card):
    cfg = tune(proj, shrinkage=over)
    _, _, ps, _ = run(*_rate_world(CELLS), cfg)
    assert stat(ps, f"{Y}|I|Dermatology", "UP1")["alpha"] == pytest.approx(derm)
    assert stat(ps, f"{Y}|I|Cardiology", "UP1")["alpha"] == pytest.approx(card)


def test_alpha_expr_matches_method_of_moments_by_hand(proj):
    cfg = tune(proj)
    k, n = np.array([1, 15, 1, 30, 2.0]), np.array([20, 30, 10, 40, 25.0])
    p, N = k / n, n.sum()
    pbar = k.sum() / N
    S_ = (n * (p - pbar) ** 2).sum()
    rho = (S_ / (pbar * (1 - pbar)) - 4) / (N - (n ** 2).sum() / N - 4)
    df = pl.DataFrame({"X__I": [5], "X__N": [N], "X__Sn2": [(n ** 2).sum()], "X__pbar": [pbar], "X__S": [S_],
                       "X__medn": [float(np.median(n))]})
    got = df.select(features.alpha_expr("X", cfg)).item()
    assert 0 < rho < 1 and got == pytest.approx(min(1 / rho - 1, 25.0))


# ------------------------------------------------------------------------------------------- robust z

def test_robust_z_scale_fallbacks_and_clipping(proj):
    """OU4 = Tot_HCPCS_Cds / Tot_Benes (100), so cds/100 sets r exactly; OU3 = 1 for every reference provider."""
    cfg = tune(proj)
    cells = {"MadSpec": [1, 2, 3, 4, 5], "IqrSpec": [1, 1, 1, 1, 1, 2, 3, 4, 5], "StdSpec": [1, 1, 1, 1, 2],
             "FlatSpec": [7, 7, 7]}
    ps, prov, coh = [], [], []
    for spec, cds in cells.items():
        for i, c in enumerate(cds):
            npi = f"{spec[:3]}{i}"
            ps.append(S(npi, "99213", 60))
            prov.append(P(npi, cds=c))
            coh.append(C(npi, spec=spec, injected=False))
    for i, c in enumerate([50, 51, 52, 53, 54]):            # an organisation cell, outside the I hierarchy
        ps.append(S(f"Hig{i}", "99213", 60))
        prov.append(P(f"Hig{i}", cds=c))
        coh.append(C(f"Hig{i}", spec="HighSpec", ent="O", injected=False))
    for npi, spec, ent, c, days in [("UP", "MadSpec", "I", 100, 60), ("DOWN", "HighSpec", "O", 1, 60),
                                    ("FLAT", "FlatSpec", "I", 8, 60), ("UNIT", "MadSpec", "I", 3, 30)]:
        ps.append(S(npi, "99213", 60, days=days))
        prov.append(P(npi, cds=c))
        coh.append(C(npi, spec=spec, ent=ent, injected=True))
    _, _, st, feat = run(frames(ps, prov), cohort_df(coh), cfg)
    z = dict(feat.select("npi", "OU4").iter_rows())
    mad = stat(st, f"{Y}|I|MadSpec", "OU4")
    assert (mad["median"], mad["scale_method"]) == (pytest.approx(0.03), "mad")
    assert mad["scale"] == pytest.approx(1.4826 * 0.01)
    assert z["Mad4"] == pytest.approx(0.02 / (1.4826 * 0.01))
    iqr = stat(st, f"{Y}|I|IqrSpec", "OU4")          # MAD 0; linear quantiles .01 and .03
    assert iqr["scale_method"] == "iqr" and iqr["scale"] == pytest.approx(0.02 / 1.349)
    assert z["Iqr8"] == pytest.approx(0.04 * 1.349 / 0.02)
    sd = stat(st, f"{Y}|I|StdSpec", "OU4")           # MAD 0, IQR 0
    assert sd["scale_method"] == "std" and sd["scale"] == pytest.approx(statistics.stdev([.01, .01, .01, .01, .02]))
    flat = stat(st, f"{Y}|I|FlatSpec", "OU4")        # no spread: scale of the ent cell (median 2, MAD 1, /100)
    assert flat["scale_method"] == "parent_mad" and flat["scale"] == pytest.approx(1.4826 * 0.01)
    assert flat["level"] == "spec_ent" and flat["median"] == pytest.approx(0.07)
    assert z["FLAT"] == pytest.approx(0.01 / (1.4826 * 0.01))
    assert z["UP"] == 10.0 and z["DOWN"] == -10.0                    # clipped to +-z_clip
    unit = stat(st, f"{Y}|I|MadSpec", "OU3")         # every reference value is 1 at every level
    assert (unit["median"], unit["scale"], unit["scale_method"]) == (1.0, 1.0, "unit")
    assert row(feat, "UNIT")["OU3"] == pytest.approx(1.0)        # (60/30 - 1) / 1


def test_hierarchy_fallback_when_cell_has_few_defined_values(proj):
    cfg = tune(proj, peer_stats={"min_defined": 4})
    fr, coh = _rate_world({"Cardiology": [(2, 10), (5, 10), (8, 10)], "Dermatology": [(1, 10)] * 5})
    _, _, ps, _ = run(fr, coh, cfg)
    card, derm = stat(ps, f"{Y}|I|Cardiology", "UP1"), stat(ps, f"{Y}|I|Dermatology", "UP1")
    assert (card["level"], card["stat_cell"], card["n"]) == ("ent", f"{Y}|I", 8)
    assert (derm["level"], derm["n"]) == ("spec_ent", 5)


# ------------------------------------------------------------------------------------------- overutilization, context

def test_overutilization_coverage_and_size(proj):
    cfg = tune(proj)
    ps = [S("X", "99213", 100, days=80), S("X", "97110", 60, days=30), S("X", "J1100", 500, days=20)]
    _, raw, _, feat = run(frames(ps, [P("X", benes=40, cds=7, srvcs=800.0)]), cohort_df([C("X")]), cfg)
    x = row(raw, "X")
    assert (x["OU1"], x["OU2"], x["OU4"]) == (4.0, 2.75, 0.175)
    assert x["OU3"] == pytest.approx(160 / 110) and x["OU5"] == pytest.approx(math.log(161))
    assert x["CX5"] == pytest.approx(math.log(40)) and x["CX8"] == pytest.approx(660 / 800)
    f = row(feat, "X")
    assert f["CX5"] == pytest.approx(math.log(40)) and f["CX8"] == pytest.approx(0.825)


def test_context_one_hot_names_and_tenure(proj):
    cfg = tune(proj)
    coh = cohort_df([C("A", spec="Hematology-Oncology", div="Middle Atlantic", ruca="large_rural"),
                     C("B", spec="Nurse Practitioner", div="other", part=None), C("D", ent="O")])
    ps = [S(n, "99213", 60) for n in "ABD"]
    fr = frames(ps, [P(n) for n in "ABD"])
    fr["nppes"] = pl.DataFrame({"npi": ["A", "B", "D", "Z"],
                                "enumeration_date": [date(2012, 12, 31), date(2023, 6, 1), date(2020, 1, 1), None]})
    _, _, _, feat = run(fr, coh, cfg)
    hot = [c for c in feat.columns if c.startswith(("CX1__", "CX2__", "CX3__", "CX4__", "CX6__"))]
    assert hot == ["CX1__cardiology", "CX1__hematology_oncology", "CX1__nurse_practitioner", "CX2__i", "CX2__o",
                   "CX3__middle_atlantic", "CX3__other", "CX3__pacific", "CX4__large_rural", "CX4__urban",
                   "CX6__na", "CX6__y"]
    a, b = row(feat, "A"), row(feat, "B")
    assert (a["CX1__hematology_oncology"], a["CX1__nurse_practitioner"], a["CX4__large_rural"], a["CX6__y"]) == (1, 0, 1, 1)
    assert (b["CX6__na"], b["CX6__y"], b["CX3__other"]) == (1, 0, 1)
    assert feat.schema["CX2__i"] == pl.Int8
    assert a["CX7"] == pytest.approx(3652 / 365.25) and b["CX7"] == 0.0
    assert row(feat, "D")["CX7"] == pytest.approx(1095 / 365.25) and feat["CX7"].null_count() == 0
    _, _, _, feat2 = run(frames(ps, [P(n) for n in "ABD"]), coh, cfg)
    assert "CX7" not in feat2.columns                                 # no NPPES extract: not computable


@pytest.mark.parametrize("dates, match", [
    ({"A": date(2010, 1, 1)}, "no enumeration date"),                         # D missing: post-t deactivation leak
    ({"A": date(2010, 1, 1), "D": None}, "no enumeration date"),              # blank date (deactivated NPI)
    ({"A": date(2010, 1, 1), "D": date(2011, 1, 1), "D ": None}, "several rows"),
])
def test_tenure_refuses_missing_or_duplicated_enumeration_dates(proj, dates, match):
    """FEATURE_CATALOG leakage item 1: NPPES blanks the dates of NPIs deactivated after t, so a missing CX7 would
    encode a post-t event; it is refused, never imputed or flagged. Duplicated NPIs would make CX7 ambiguous."""
    cfg = tune(proj)
    npis = [n.strip() for n in dates]
    fr = frames([S(n, "99213", 60) for n in "AD"], [P(n) for n in "AD"])
    fr["nppes"] = pl.DataFrame({"npi": npis, "enumeration_date": list(dates.values())},
                               schema={"npi": pl.String, "enumeration_date": pl.Date})
    with pytest.raises(FeatureError, match=match):
        run(fr, cohort_df([C("A"), C("D")]), cfg)


def test_one_hot_collision_fails(proj):
    cfg = tune(proj)
    coh = cohort_df([C("A", spec="Hematology-Oncology"), C("B", spec="Hematology Oncology")])
    with pytest.raises(FeatureError, match="collide"):
        features.fit_peer_reference(frames([S("A", "99213", 60)], [P("A")]), coh, Y, cfg)


# ------------------------------------------------------------------------------------------- atypical mix

def test_service_mix_known_jsd_hhi_and_drug_share(proj):
    cfg = tune(proj)
    ps = [S("R1", "A", 100), S("R1", "C", 50), S("R2", "A", 50), S("R2", "C", 100),   # pooled q = (A .5, C .5)
          S("X", "A", 30, pymt=20.0), S("X", "A", 20, place="F", pymt=20.0), S("X", "B", 50, pymt=60.0),
          S("X", "J1100", 10, pymt=100.0)]
    coh = cohort_df([C("R1", injected=False), C("R2", injected=False), C("X", injected=True)])
    refs, raw, _, _ = run(frames(ps, [P("R1"), P("R2"), P("X")]), coh, cfg)
    x, r1 = row(raw, "X"), row(raw, "R1")
    # p = (A .5, B .5), q = (A .5, C .5): JSD = 1/2 KL(p||m) + 1/2 KL(q||m) = 1/2 + 1/2 * ... = 0.5 (base 2)
    assert x["SM1"] == pytest.approx(0.5, abs=1e-12)
    assert r1["SM1"] == pytest.approx(jensenshannon([2 / 3, 1 / 3], [0.5, 0.5], base=2) ** 2)
    assert (x["SM2"], x["SM2__k"], x["SM2__n"]) == (0.5, 50.0, 100.0)   # B is billed by no reference provider
    # payment by code (drug included): A 1000, B 3000, J1100 1000 -> HHI = (1 + 9 + 1) / 25
    assert x["SM3"] == pytest.approx(0.44) and x["SM4"] == pytest.approx(0.2)
    cell = refs.codes.filter(pl.col("peer_cell") == f"{Y}|I|Cardiology").sort("hcpcs_cd")
    assert cell["hcpcs_cd"].to_list() == ["A", "C"] and cell["srv_share"].to_list() == [0.5, 0.5]


@pytest.mark.parametrize("share, rare", [(0.34, True), (1 / 3, False)])
def test_rare_codes_strictly_below_threshold(proj, share, rare):
    cfg = tune(proj, atypical_mix={"rare_provider_share": share})
    ps = [S("R1", "A", 100), S("R1", "D", 10), S("R2", "A", 100), S("R3", "A", 100), S("X", "A", 90), S("X", "Z", 10)]
    coh = cohort_df([C(n, injected=n == "X") for n in ("R1", "R2", "R3", "X")])
    refs, raw, _, _ = run(frames(ps, [P(n) for n in ("R1", "R2", "R3", "X")]), coh, cfg)
    d = refs.codes.filter((pl.col("level") == "spec_ent") & (pl.col("hcpcs_cd") == "D")).row(0, named=True)
    assert d["provider_share"] == pytest.approx(1 / 3) and d["rare"] is rare
    assert row(raw, "R1")["SM2__k"] == (10.0 if rare else 0.0)
    assert row(raw, "X")["SM2"] == pytest.approx(0.1)                  # Z: no reference provider bills it


# ------------------------------------------------------------------------------------------- charge inflation

def test_charge_inflation_and_code_adjusted_price_index(proj):
    cfg = tune(proj, charge_index={"min_providers_cell": 2, "min_providers_national": 3})
    ps = [S("R1", "A", 20, allowed=100.0), S("R1", "A", 20, place="F", allowed=60.0), S("R1", "B", 20, allowed=45.0),
          S("R2", "A", 20, allowed=100.0), S("R3", "A", 20, allowed=120.0),
          S("R4", "B", 20, allowed=40.0), S("R5", "B", 20, allowed=60.0),
          S("X", "A", 10, allowed=110.0, sbmtd=220.0, stdzd=100.0),
          S("X", "A", 10, place="F", allowed=66.0, sbmtd=132.0, stdzd=60.0),
          S("X", "B", 30, allowed=54.0, sbmtd=108.0, stdzd=50.0),
          S("X", "J1100", 10, allowed=5.0, sbmtd=6.0, stdzd=5.0),
          S("X", "Q", 5, allowed=999.0, sbmtd=1998.0, stdzd=900.0)]
    coh = cohort_df([C("R1", injected=False), C("R2", injected=False), C("R3", injected=False),
                     C("R4", spec="Dermatology", injected=False), C("R5", spec="Dermatology", injected=False),
                     C("X", injected=True)])
    names = ("R1", "R2", "R3", "R4", "R5")
    refs, raw, _, _ = run(frames(ps, [P(n) for n in names] + [P("X", benes=20)]), coh, cfg)
    x = row(raw, "X")
    assert x["CH1"] == pytest.approx(16810 / 8425)
    assert x["CH2"] == pytest.approx(7650 / 20)
    assert x["CH3"] == pytest.approx(7600 / 55)                       # non-drug numerator and denominator
    # A/O: cell median 100 (3 >= 2) -> 1.1; A/F: 1 in cell, 1 national (< 3) -> left out;
    # B/O: 1 in cell -> national median of (45, 40, 60) = 45 -> 1.2; Q: unknown -> left out; drug: left out
    assert x["CH4"] == pytest.approx((10 * 1.1 + 30 * 1.2) / 40)
    nat_b = refs.codes.filter((pl.col("level") == "national") & (pl.col("hcpcs_cd") == "B")).row(0, named=True)
    assert (nat_b["med_allowed_o"], nat_b["n_allowed_o"]) == (45.0, 3)


# ------------------------------------------------------------------------------------------- patient mix

E = [0.1, -0.2, 0.0, 0.2, -0.1]          # sums to 0 and is orthogonal to the risk scores 1..5


def _pm_world(cc=None):
    ps, prov, coh = [], [], []
    for i, e in enumerate(E):
        r = i + 1.0
        npi = f"Car{i}"
        ps.append(S(npi, "99213", 100 * math.exp(0.1 + 0.2 * r + e)))
        prov.append(P(npi, risk=r, **({} if cc is None else {"bene_cc_x_pct": cc[i]})))
        coh.append(C(npi))
    for i, r in enumerate([1.2, 2.5, 3.1]):
        npi = f"Der{i}"
        ps.append(S(npi, "99213", 100 * math.exp(0.5 + 0.1 * r + 0.05 * i)))
        prov.append(P(npi, risk=r, **({} if cc is None else {"bene_cc_x_pct": 25.0 + i})))
        coh.append(C(npi, spec="Dermatology"))
    return frames(ps, prov, cc=() if cc is None else ("bene_cc_x_pct",)), cohort_df(coh)


def test_patient_mix_residuals_exact(proj):
    cfg = tune(proj)
    _, raw, ps, feat = run(*_pm_world(), cfg)
    got = feat.filter(pl.col("npi").str.starts_with("Car")).sort("npi")
    assert got["raw__PM2"].to_list() == pytest.approx(E, abs=1e-9)
    assert got["raw__PM3"].to_list() == pytest.approx(E, abs=1e-9)   # log CH2 = log(0.9) + log OU1
    m = ps.pm.filter((pl.col("peer_cell") == f"{Y}|I|Cardiology") & (pl.col("target") == "PM2"))
    coef = dict(m.select("term", "coef").iter_rows())
    assert coef["intercept"] == pytest.approx(0.1) and coef["PM1"] == pytest.approx(0.2)
    assert set(m["level"]) == {"spec_ent"} and ps.covariates == ("PM1",)
    assert row(raw, "Car0")["PM1"] == 1.0 and row(raw, "Car0")["PM4"] == 0.3


def test_patient_mix_fallback_and_imputation(proj):
    cfg = tune(proj, patient_mix={"min_n": 6})
    _, _, ps, _ = run(*_pm_world(), cfg)
    m = ps.pm.filter(pl.col("peer_cell") == f"{Y}|I|Cardiology")
    assert set(m["level"]) == {"ent"} and set(m["fit_cell"]) == {f"{Y}|I"} and set(m["n"]) == {8}

    cfg = tune(proj, patient_mix={"min_n": 1})
    cc = [10.0, 20.0, None, 40.0, 50.0]
    fr, coh = _pm_world(cc)
    _, raw, ps, feat = run(fr, coh, cfg)
    assert ps.covariates == ("PM1", "bene_cc_x_pct")
    m = ps.pm.filter((pl.col("peer_cell") == f"{Y}|I|Cardiology") & (pl.col("target") == "PM2"))
    assert dict(m.select("term", "impute").iter_rows())["bene_cc_x_pct"] == 30.0   # cell median of 10, 20, 40, 50
    car = raw.filter(pl.col("npi").str.starts_with("Car")).sort("npi")
    A = np.column_stack([np.ones(5), car["PM1"].to_numpy(), [10, 20, 30, 40, 50.0]])
    y = car["PM2__y"].to_numpy()
    want = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    got = feat.filter(pl.col("npi").str.starts_with("Car")).sort("npi")["raw__PM2"].to_numpy()
    assert np.allclose(got, want, atol=1e-9)


def test_dual_share_missing_indicator(proj):
    cfg = tune(proj)
    fr = frames([S(n, "99213", 60) for n in "ABC"], [P("A", dual=30), P("B", dual=50), P("C", dual=None)])
    _, raw, _, feat = run(fr, cohort_df([C(n) for n in "ABC"]), cfg)
    assert (row(raw, "A")["PM4"], row(raw, "A")["PM4__k"], row(raw, "A")["PM4__n"]) == (0.3, 30.0, 100.0)
    c = row(feat, "C")
    assert (c["PM4_missing"], c["PM4"], c["raw__PM4"]) == (1, 0.0, None) and row(feat, "A")["PM4_missing"] == 0


# ------------------------------------------------------------------------------------------- DME

def test_dme_features_and_structural_zero(proj):
    cfg = tune(proj)
    names = ("R1", "R2", "R3", "X", "M", "N")
    ps = [S(n, "99213", 60) for n in names]
    ds = [DS("R1", "E0601", rent="Y", clms=30, srv=30, pymt=50.0), DS("R2", "A4253", clms=40, srv=80, pymt=9.0),
          DS("R3", "L1833", suplrs=2, clms=20, srv=20, pymt=480.0),
          DS("X", "L0650", suplrs=1, clms=24, srv=24, pymt=800.0),
          DS("X", "E0601", rent="Y", suplrs=3, clms=90, srv=90, pymt=70.0)]
    dp = [("R1", 15), ("R2", 25), ("R3", 18), ("X", 20), ("M", 14)]    # M: every DME-S row suppressed
    _, raw, ps_, feat = run(frames(ps, [P(n) for n in names], ds, dp), cohort_df([C(n) for n in names]), cfg)
    x = row(raw, "X")
    assert (x["DI0"], x["DI1"], x["DI2"], x["DI4"]) == (0, 1275.0, 0.2, 5.7)
    assert x["DI3"] == pytest.approx(19200 / 25500)                    # L0650 is high-risk, E0601 is not
    assert x["DC1"] == pytest.approx((24 * 24 + 90 * 30) / 114)
    assert (x["DC2"], x["DC2__k"], x["DC2__n"]) == (pytest.approx(90 / 114), 90.0, 114.0)
    n = row(raw, "N")
    # FEATURE_CATALOG DI0: no DME-P row -> raw DI1-DI4 = 0; DC1, DC2 (referral concentration) are undefined
    assert n["DI0"] == 1 and all(n[f] == 0.0 for f in ("DI1", "DI2", "DI3", "DI4"))
    assert all(n[f] is None for f in ("DC1", "DC2", "DC2__k", "DC2__n"))
    m = row(raw, "M")
    assert (m["DI0"], m["DI1"], m["DI4"], m["DI3"], m["DC1"], m["DC2"]) == (0, 0.0, 0.0, None, None, None)
    cell = f"{Y}|I|Cardiology"
    assert stat(ps_, cell, "DI1")["n"] == 5 and stat(ps_, cell, "DI2")["n"] == 6 and stat(ps_, cell, "DI3")["n"] == 4
    assert stat(ps_, cell, "DC1")["n"] == 4 and stat(ps_, cell, "DC2")["n"] == 4
    fn = row(feat, "N")
    # structural zero: z = 0 (peer median of the referrers) for DI1, DI3, DI4, DC1, DC2; DI0 carries the information,
    # so a non-referrer is not pushed to the clip and does not lift a typical referrer's G7 above its peers
    assert fn["DI0"] == 1 and all(fn[f] == 0.0 for f in ("DI1", "DI3", "DI4", "DC1", "DC2"))
    assert (fn["raw__DI1"], fn["raw__DI4"], fn["raw__DC1"], fn["raw__DC2"]) == (0.0, 0.0, None, None)
    di2 = stat(ps_, cell, "DI2")                       # DI2 = 0 is a genuine share: z-scored over all providers
    assert fn["DI2"] == pytest.approx(np.clip((0 - di2["median"]) / di2["scale"], -10, 10))
    assert row(feat, "M")["DI3"] == 0.0


def test_high_risk_dme_ranges(proj):
    cfg = FeatureConfig.load(proj)
    codes = ["L0650", "L0449", "L0651", "K0800", "K0899", "K0900", "E0601", "L3960", "l0650", "L065", None]
    got = pl.DataFrame({"hcpcs_cd": codes}, schema={"hcpcs_cd": pl.String}).select(
        features.high_risk_expr(cfg)).to_series().to_list()
    assert got == [True, False, True, True, True, False, False, True, False, False, False]


# ------------------------------------------------------------------------------------------- loud failures

@pytest.mark.parametrize("edit, match", [
    (lambda fr, coh: (fr | {"cms_ps": fr["cms_ps"].with_columns(pl.lit("X").alias("place_of_srvc"))}, coh),
     "place_of_srvc"),
    (lambda fr, coh: (fr | {"cms_ps": fr["cms_ps"].with_columns(pl.lit(None, pl.Float64).alias("avg_mdcr_pymt_amt"))}, coh),
     "nulls"),
    (lambda fr, coh: (fr | {"cms_prov": pl.concat([fr["cms_prov"], fr["cms_prov"]])}, coh), "duplicated grain"),
    (lambda fr, coh: ({k: v for k, v in fr.items() if k != "dme_p"}, coh), "frames lack"),
    (lambda fr, coh: (fr, coh.with_columns(pl.lit(2021, pl.Int32).alias("year"))), "another year"),
    (lambda fr, coh: (fr, coh.with_columns(pl.lit(None, pl.String).alias("peer_cell"))), "without a peer cell"),
    (lambda fr, coh: (fr | {"dme_s": pl.DataFrame([DS("A", "E0601", rent="?")], schema=DS_SCHEMA)
                            .with_columns(pl.lit(Y, pl.Int32).alias("year"))}, coh), "suplr_rentl_ind"),
])
def test_fails_loudly(proj, edit, match):
    cfg = tune(proj)
    fr, coh = edit(frames([S("A", "99213", 60), S("B", "99213", 60)], [P("A"), P("B")]),
                   cohort_df([C("A"), C("B")]))
    with pytest.raises(FeatureError, match=match):
        run(fr, coh, cfg)


def test_unknown_peer_cell_at_scoring_fails(proj):
    cfg = tune(proj)
    fr = frames([S("A", "99213", 60), S("B", "99213", 60)], [P("A"), P("B")])
    coh = cohort_df([C("A"), C("B", spec="Dermatology", injected=True)]).with_columns(
        pl.col("injected").fill_null(False))
    refs = features.fit_peer_reference(fr, coh, Y, cfg)
    with pytest.raises(FeatureError, match="without reference providers"):
        features.raw_statistics(fr, coh, refs, Y, cfg)


# ------------------------------------------------------------------------------------------- prune helpers

def test_prune_candidates_within_group_only(proj):
    cfg = FeatureConfig.load(proj)
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    df = pl.DataFrame({"npi": [str(i) for i in range(500)], "UP1": x, "UP2": 2 * x + rng.normal(0, 1e-3, 500),
                       "UP5": rng.normal(size=500), "OU1": x, "CX2__i": (x > 0).astype(np.int8),
                       "CX2__o": (x <= 0).astype(np.int8), "raw__UP1": x})
    got = features.prune_candidates(df, cfg=cfg)
    assert got.select("group", "feature_a", "feature_b").rows() == [("upcoding", "UP1", "UP2")]
    assert got["rho"].item() > 0.95


def test_pruned_ids_are_dropped_from_the_model_columns(proj):
    cfg = tune(proj, pruned=["SM3", "CX6"])
    _, _, _, feat = run(frames([S(n, "99213", 60) for n in "AB"], [P("A"), P("B")]), cohort_df([C("A"), C("B")]), cfg)
    assert "SM3" not in feat.columns and not any(c.startswith("CX6__") for c in feat.columns)
    assert "raw__SM3" in feat.columns and "SM4" in feat.columns


# ------------------------------------------------------------------------------------------- synthetic worlds

def _world(proj, n_npi, years):
    truth = make_synthetic(proj, n_npi=n_npi, years=years)
    cohort.build_cohort(proj)
    return truth


def test_injection_api_peer_stats_unaffected(proj):
    _world(proj, 2500, [Y])
    cfg = FeatureConfig.load(proj)
    lazy, _ = features.load_frames(proj, Y)
    fr = {k: v.collect() for k, v in lazy.items() if v is not None}
    coh = pl.read_parquet(proj.processed / "cohort.parquet").filter(pl.col("in_cohort"))
    refs, raw, ps, feat = run(fr, coh, cfg)
    donors = feat.filter((pl.col("raw__UP1__n") > 50) & (pl.col("raw__UP1") > 0)).sort("npi")["npi"].head(5).to_list()
    new = {d: "9" + d[1:] for d in donors}
    sel = pl.col("npi").is_in(donors)
    rename = pl.col("npi").replace_strict(new, default=None)
    up = {"cms_ps": [pl.when(pl.col("hcpcs_cd") == "99215").then(pl.col("tot_srvcs") * 3).otherwise(
        pl.col("tot_srvcs")).alias("tot_srvcs")]}
    aug = {k: pl.concat([v, v.filter(sel).with_columns(rename, *up.get(k, []))]) for k, v in fr.items()}
    aug_coh = pl.concat([coh.with_columns(pl.lit(False).alias("injected")),
                         coh.filter(sel).with_columns(rename, pl.lit(True).alias("injected"))])
    # fit_* ignore injected rows: identical reference and peer statistics
    refs2 = features.fit_peer_reference(aug, aug_coh, Y, cfg)
    raw2 = features.raw_statistics(aug, aug_coh, refs, Y, cfg)
    ps2 = features.fit_peer_stats(raw2, aug_coh, Y, cfg)
    assert refs2.codes.equals(refs.codes) and ps2.stats.equals(ps.stats) and ps2.pm.equals(ps.pm)
    # extra providers passed only to raw_statistics / transform leave everyone else's features unchanged
    feat2 = features.transform(raw2, aug_coh, ps, refs, Y, cfg)
    assert feat2.filter(~pl.col("npi").is_in(list(new.values()))).equals(feat)
    inj = feat2.filter(pl.col("npi").is_in(list(new.values()))).sort("npi")
    base = feat.filter(sel).sort("npi")
    assert inj.height == 5 and (inj["raw__UP1"].to_numpy() > base["raw__UP1"].to_numpy()).all()
    assert (inj["UP1"].to_numpy() >= base["UP1"].to_numpy()).all()


def test_peer_statistics_centre_every_peer_cell_on_one_reference(proj):
    """Review finding: SM1/SM2/CH4/PM2/PM3 are measured against the peer cell's own reference, so a cell's
    statistics must summarise every member's value against that same reference. Invariant: featurising all reference
    members of a peer cell (its cell id at the peer level, e.g. every individual for an entity-type cell) against
    that cell gives median z = 0 for every robust_z feature whose statistics come from the cell itself."""
    _world(proj, 12_000, [Y])
    cfg = FeatureConfig.load(proj)
    lazy, _ = features.load_frames(proj, Y)
    fr = {k: v.collect() for k, v in lazy.items() if v is not None}
    coh = pl.read_parquet(proj.processed / "cohort.parquet").filter(pl.col("in_cohort"))
    refs, raw, ps, feat = run(fr, coh, cfg)
    assert set(coh["peer_level"]) == set(features.PEER_LEVELS)
    used_se = refs.cells.filter(pl.col("peer_level") == "spec_ent")["peer_cell"].implode()
    assert coh.filter(pl.col("cell_spec_ent").is_in(used_se) & (pl.col("peer_level") == "spec_ent_div")).height
    own = ps.stats.join(refs.cells.select("peer_cell", "peer_level"), on="peer_cell").filter(
        pl.col("level") == pl.col("peer_level"))
    for f in features.REF_DEPENDENT:                     # always from the peer cell itself
        sub = ps.stats.filter(pl.col("feature") == f)
        assert sub.height == own.filter(pl.col("feature") == f).height
        assert set(sub["scale_method"]) <= {"mad", "iqr", "std", "parent_mad", "parent_iqr", "parent_std"}
    checked = 0
    for lv in features.PEER_LEVELS:
        cells = refs.cells.filter(pl.col("peer_level") == lv)["peer_cell"].implode()
        members = coh.filter(pl.col(features.CELL_COL[lv]).is_in(cells)).with_columns(
            pl.col(features.CELL_COL[lv]).alias("peer_cell"), pl.lit(lv).alias("peer_level"))
        fm = features.transform(features.raw_statistics(fr, members, refs, Y, cfg), members, ps, refs, Y, cfg)
        pairs = own.filter(pl.col("peer_level") == lv).select("peer_cell", "feature")
        med = fm.unpivot(index="peer_cell", on=[s.id for s in cfg.robust], variable_name="feature", value_name="z") \
            .join(pairs, on=["peer_cell", "feature"]).group_by("peer_cell", "feature").agg(pl.col("z").median())
        bad = med.filter(pl.col("z").abs() > 1e-9)
        assert bad.is_empty(), bad.sort("feature").head(10)
        checked += med.filter(pl.col("feature").is_in(list(features.REF_DEPENDENT))).height
    assert checked >= 5 * refs.cells.height


def test_features_do_not_depend_on_input_row_order(proj):
    """CONTRACTS §0 determinism, incl. the DMEPOS sums (a lazy sort before group_by is dropped by polars)."""
    _world(proj, 2000, [Y])
    cfg = FeatureConfig.load(proj)
    lazy, _ = features.load_frames(proj, Y)
    fr = {k: v.collect() for k, v in lazy.items() if v is not None}
    coh = pl.read_parquet(proj.processed / "cohort.parquet").filter(pl.col("in_cohort"))
    a = run(fr, coh, cfg)
    shuffled = {k: v.sample(fraction=1.0, shuffle=True, seed=11) for k, v in fr.items()}
    b = run(shuffled, coh.sample(fraction=1.0, shuffle=True, seed=12), cfg)
    assert a[0].codes.equals(b[0].codes) and a[2].stats.equals(b[2].stats) and a[2].pm.equals(b[2].pm)
    assert a[3].equals(b[3])
    assert a[3]["raw__DI1"].drop_nulls().len() > 50                # DME sums are exercised


def test_build_features_artefacts_groups_determinism_and_no_labels(proj):
    _world(proj, 1500, [2021, 2022])
    out = features.build_features(proj)
    assert sorted(out) == [2021, 2022]
    cfg = FeatureConfig.load(proj)
    feat = pl.read_parquet(proj.processed / "features" / "2022.parquet")
    coh = pl.read_parquet(proj.processed / "cohort.parquet").filter((pl.col("year") == 2022) & pl.col("in_cohort"))
    assert feat.columns[:3] == ["npi", "year", "peer_cell"] and feat["npi"].to_list() == sorted(coh["npi"].to_list())
    cols = features.model_columns(feat.columns)
    gm = groups.feature_groups(cols)
    assert set(gm.values()) == set(groups.GROUP_ORDER)
    scheme = [s.id for s in cfg.specs if s.kind != "one_hot" and s.id != "CX7"]
    assert [c for c in cols if not c.startswith(tuple(f"{s.id}__" for s in cfg.one_hot))] == scheme
    assert feat.select(cols).null_count().sum_horizontal().item() == 0
    z = [s.id for s in cfg.robust]
    assert feat.select(pl.max_horizontal(pl.col(z).abs().max())).item() <= cfg.z_clip
    assert all(f"raw__{f}" in feat.columns for f in z)
    meta = out[2022]
    assert meta["not_computable"] == {"CX7": meta["not_computable"]["CX7"]} and meta["rows"] == feat.height
    peer = proj.processed / "peer" / "2022"
    assert pl.read_parquet(peer / "codes.parquet").columns == list(features.CODES_COLS)
    assert pl.read_parquet(peer / "stats.parquet").columns == list(features.STATS_COLS)
    assert pl.read_parquet(peer / "pm_models.parquet").columns == list(features.PM_COLS)
    both = features.load_features(proj, [2021, 2022])
    assert both.height == feat.height + out[2021]["rows"]
    assert both.select(pl.col("^CX1__.*$").null_count()).sum_horizontal().item() == 0

    files = sorted((proj.processed / "features").glob("*.parquet")) + sorted((proj.processed / "peer").rglob("*.parquet"))
    first = {f: pl.read_parquet(f) for f in files}
    for f in [*proj.processed.glob("labels_*.parquet"), proj.processed / "exclusions.parquet",
              proj.processed / "first_exclusions.parquet"]:
        f.unlink()
    shutil.rmtree(proj.raw / "leie")
    features.build_features(proj)                       # deterministic, and needs no label or LEIE file
    for f, df in first.items():
        assert pl.read_parquet(f).equals(df), f


def test_planted_schemes_move_their_group(proj):
    truth = _world(proj, 8000, [Y])
    features.build_features(proj)
    feat = pl.read_parquet(proj.processed / "features" / f"{Y}.parquet")
    planted = truth["planted"].filter(pl.col("year") == Y).explode("schemes")
    clean = feat.filter(~pl.col("npi").is_in(planted["npi"].implode()))
    for scheme, fid in [("upcoding", "UP2"), ("overutilization", "OU1"), ("atypical_mix", "SM2"),
                        ("charge_inflation", "CH1"), ("dme_mill", "DI3")]:
        hit = feat.filter(pl.col("npi").is_in(planted.filter(pl.col("schemes") == scheme)["npi"].implode()))
        assert hit.height >= 5 and hit[fid].median() > clean[fid].quantile(0.75), scheme


def test_runtime_per_10k_providers(proj):
    _world(proj, 10_000, [Y])
    t0 = time.perf_counter()
    out = features.build_features(proj)
    secs = time.perf_counter() - t0
    per_10k = secs / out[Y]["rows"] * 1e4
    print(f"\nfeatures: {out[Y]['rows']} providers in {secs:.2f} s -> {per_10k:.2f} s per 10k providers per year")
    assert per_10k < 20
