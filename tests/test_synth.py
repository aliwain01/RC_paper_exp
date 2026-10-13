"""tests/synth.py: the synthetic data meet the harmonised contract (schema, dtypes, NPI format, suppression,
grain), look like CMS/LEIE files in the ways later phases depend on, carry learnable planted signal, and are
deterministic. Synthetic numbers checked here are properties of the generator, not results."""

from __future__ import annotations

import json
import shutil
import time

import numpy as np
import polars as pl
import pytest

from scenario_fraud.cohort import CohortConfig
from scenario_fraud.harmonise import DTYPES, NPI_RE, _target_spec, load_schema
from scenario_fraud.paths import Paths
from tests.conftest import REPO
from tests.synth import HIGH_RISK_DME, RARE, SCHEMES, SPECS, make_npis, make_synthetic

DATASETS = ("cms_ps", "cms_prov", "dme_s", "dme_p")
GRAIN = {"cms_ps": ["npi", "hcpcs_cd", "place_of_srvc"], "cms_prov": ["npi"],
         "dme_s": ["npi", "hcpcs_cd", "suplr_rentl_ind"], "dme_p": ["npi"]}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One default-size synthetic project shared by the read-only tests below."""
    proj = _new_proj(tmp_path_factory.mktemp("synth"))
    truth = make_synthetic(proj)
    return proj, truth


def _new_proj(base) -> Paths:
    root = base / "proj with space"
    (root / "design").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    shutil.copytree(REPO / "configs", root / "configs")
    return Paths(root)


def _read(proj: Paths, ds: str, years=None) -> pl.DataFrame:
    files = sorted((proj.interim / ds).glob("*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files if years is None or int(f.stem) in years])


def test_files_follow_harmonised_schema(world):
    proj, truth = world
    schema = load_schema(proj)
    assert truth["years"] == list(range(2018, 2025))
    for ds in DATASETS:
        for y in truth["years"]:
            f = proj.interim / ds / f"{y}.parquet"
            df = pl.read_parquet(f)
            assert df.columns[-1] == "year" and df.schema["year"] == pl.Int32 and df["year"].unique().to_list() == [y]
            for col in df.columns[:-1]:
                spec = _target_spec(schema[ds], col)
                assert spec is not None, f"{ds}.{col} not in _schema.yaml"
                assert df.schema[col] == DTYPES[spec["dtype"]], f"{ds}.{col}: {df.schema[col]}"
            required = [c for c, s in schema[ds]["columns"].items() if s.get("required")]
            assert set(required) <= set(df.columns), ds
            assert df["npi"].str.contains(NPI_RE).all() and df["npi"].null_count() == 0
            assert df.select(pl.struct(GRAIN[ds]).is_duplicated().any()).item() is False, f"{ds} {y} grain"
            meta = json.loads(f.with_suffix(".meta.json").read_text())
            assert meta["synthetic"] is True and meta["n_rows"] == df.height
    prov = _read(proj, "cms_prov")
    assert any(c.startswith("bene_cc_") for c in prov.columns)
    assert any(c.startswith("cat_") for c in _read(proj, "dme_p").columns)


def _npi_ok(npi: str) -> bool:
    """CMS NPI check digit: Luhn over the prefix 80840 + the 10 digits."""
    total = 0
    for i, d in enumerate(int(c) for c in reversed("80840" + npi)):
        d = d * 2 if i % 2 else d
        total += d - 9 if d > 9 else d
    return total % 10 == 0


def test_npis_are_unique_and_luhn_valid(world):
    assert _npi_ok("1234567893") and not _npi_ok("1234567890")
    npis = make_npis(np.random.default_rng(1), 5000)
    assert len(set(npis)) == 5000 and all(_npi_ok(n) for n in npis)
    _, truth = world
    assert all(_npi_ok(n) for n in truth["npis"] + truth["dme_only_npis"] + truth["leie_only_npis"])


def test_suppression_applied(world):
    proj, _ = world
    ps, prov, ds, dp = (_read(proj, d) for d in DATASETS)
    assert ps["tot_benes"].min() > 10 and ds["tot_suplr_benes"].min() > 10
    assert prov["tot_benes"].min() > 10 and dp["tot_suplr_benes"].min() > 10
    for col in ("bene_dual_cnt", "bene_ndual_cnt"):
        assert 0 < prov[col].null_count() < prov.height and prov[col].drop_nulls().min() > 10
    cc = [c for c in prov.columns if c.startswith("bene_cc_")]
    assert all(prov[c].null_count() > 0 for c in cc) and prov.select(cc).null_count().sum_horizontal().item() < prov.height * len(cc) / 2
    sub = dp["dme_tot_suplr_benes"].drop_nulls()
    assert dp["dme_tot_suplr_benes"].null_count() > 0 and ((sub == 0) | (sub > 10)).all()


def test_summary_totals_exceed_visible_rows(world):
    proj, _ = world
    ps, prov = _read(proj, "cms_ps"), _read(proj, "cms_prov")
    vis = ps.group_by("npi", "year").agg(
        pl.col("tot_srvcs").sum().alias("vis_srv"), pl.col("hcpcs_cd").n_unique().alias("vis_codes"),
        pl.col("tot_benes").max().alias("max_benes"),
        (pl.col("tot_srvcs") * pl.col("avg_mdcr_alowd_amt")).sum().alias("vis_allowed"))
    j = prov.join(vis, on=["npi", "year"], how="left")
    both = j.filter(pl.col("vis_srv").is_not_null())
    assert (both["tot_srvcs"] >= both["vis_srv"] - 0.05).all()
    # averages are rounded (as CMS rounds them), so dollar totals agree to a relative 1e-6
    assert (both["tot_mdcr_alowd_amt"] >= both["vis_allowed"] * (1 - 1e-6) - 0.05).all()
    assert (both["tot_hcpcs_cds"] >= both["vis_codes"]).all() and (both["tot_benes"] >= both["max_benes"]).all()
    assert (both["tot_srvcs"] > both["vis_srv"] + 0.05).mean() > 0.3        # suppression hides some services
    assert j["vis_srv"].null_count() > 0                                    # every service row suppressed
    assert ps.join(prov, on=["npi", "year"], how="anti").is_empty()         # no service rows without a summary


def test_coverage_of_real_world_situations(world):
    proj, truth = world
    ps, prov, ds, dp = (_read(proj, d) for d in DATASETS)
    cfg = CohortConfig.load(proj)
    states = set(prov["state"].to_list())
    assert {"PR", "GU"} <= states and states & {"AA", "AE", "AP"}
    assert {cfg.divisions[s] for s in states if s in cfg.divisions} == set(cfg.divisions.values())
    ruca = prov["ruca"]
    assert ruca.null_count() > 0 and ruca.str.contains(r"\.").any() and (ruca == "10").any()
    assert set(prov["ent_cd"].unique()) == {"I", "O"}
    types = prov.group_by("prvdr_type").agg(pl.col("npi").n_unique().alias("n"))
    assert types.height == len(SPECS) and 1 <= types["n"].min() < 0.02 * truth["n_npi"]
    codes = ps.group_by("year").agg(pl.col("hcpcs_cd").unique())
    by_year = {r["year"]: set(r["hcpcs_cd"]) for r in codes.to_dicts()}
    assert all("99201" in by_year[y] for y in (2018, 2019, 2020))
    assert all("99201" not in by_year[y] for y in (2021, 2022, 2023, 2024))
    assert {f"9921{i}" for i in range(1, 6)} | {f"9920{i}" for i in range(2, 6)} | {f"9928{i}" for i in range(1, 6)} <= by_year[2022]
    em = {f"992{i:02d}" for i in range(1, 16)} | {f"9928{i}" for i in range(1, 6)}
    procs = ps.filter((pl.col("year") == 2022) & (pl.col("hcpcs_drug_ind") == "N") & ~pl.col("hcpcs_cd").is_in(list(em)))
    assert procs["hcpcs_cd"].n_unique() >= 30
    drug = ps.filter(pl.col("hcpcs_drug_ind") == "Y")
    assert drug["hcpcs_cd"].str.starts_with("J").all() and (drug["tot_srvcs"] % 1 != 0).any()
    assert set(ps["place_of_srvc"].unique()) == {"F", "O"} and set(ps["hcpcs_drug_ind"].unique()) == {"N", "Y"}
    assert (ps.group_by("npi", "year", "hcpcs_cd").len()["len"] == 2).any()      # same code at F and at O
    assert set(ds["suplr_rentl_ind"].unique()) == {"Y", "N"} and ds["rbcs_id"].null_count() == 0
    hc = ds["hcpcs_cd"]
    assert hc.str.starts_with("L").any() and hc.str.starts_with("E").any() and hc.str.starts_with("A").any()
    assert hc.is_in([f"K{i:04d}" for i in range(800, 900)]).any()
    assert prov.join(dp, on=["npi", "year"], how="anti").height > prov.height / 2   # most refer no DME
    dme_only = set(truth["dme_only_npis"])                                    # referrers with no Part B row
    part_b = set(ps["npi"].to_list()) | set(prov["npi"].to_list())
    assert dme_only & set(dp["npi"].to_list()) and not dme_only & part_b
    assert set(HIGH_RISK_DME) >= {"L0650", "L1833", "L3960", "K0823"}


def test_labels_and_ground_truth(world):
    proj, truth = world
    lab = pl.read_parquet(proj.processed / "labels_primary_W3.parquet")
    firsts = pl.read_parquet(proj.processed / "first_exclusions.parquet")
    assert (proj.processed / "labels_broad_W3.parquet").is_file() and truth["labels"]["snapshot_date"] == "2026-10-12"
    n = truth["n_npi"]
    assert 0.03 * n <= len(truth["excluded_npis"]) <= 0.05 * n
    ex = truth["exclusions"].filter("in_part_b")
    assert ex["excltype"].is_in(["1128b7", "1128a1"]).any() and ex["excltype"].is_in(["1128b4", "1128b14"]).any()
    yrs = ex["excl_date"].dt.year()
    assert yrs.min() == 2017 and yrs.max() == 2026 and ex["excl_date"].max().isoformat() <= "2026-10-12"
    leie = pl.read_csv(next((proj.raw / "leie" / "2026").glob("*.csv")), infer_schema=False)
    assert (leie["NPI"] == "0000000000").sum() == truth["n_leie_no_npi"] >= 3
    pos = lab.filter(pl.col("y") == 1)
    assert pos.height > 0 and set(pos["npi"].to_list()) <= set(truth["primary_npis"])
    assert set(firsts.filter(pl.col("tier") == "any")["npi"].to_list()) >= set(truth["excluded_npis"])
    planted = truth["planted"]
    assert set(planted["source"].unique()) == {"excluded", "unlabelled"}
    assert set(s for sch in truth["schemes"].values() for s in sch) <= set(SCHEMES)
    assert all(1 <= len(s) <= 2 for s in truth["schemes"].values())
    # excluded providers' scheme years are exactly their positive window, where they still bill
    exc_planted = planted.filter(pl.col("source") == "excluded").join(
        lab.select("npi", "year", "y", "excl_year"), on=["npi", "year"], how="inner")
    assert exc_planted.height > 0 and (exc_planted["y"] == 1).all()
    assert ((exc_planted["excl_year"] - exc_planted["year"]).is_between(1, 3)).all()
    assert not set(truth["unlabelled_npis"]) & set(truth["excluded_npis"])


def _stat(ps: pl.DataFrame, prov: pl.DataFrame, ds: pl.DataFrame) -> pl.DataFrame:
    """A few FEATURE_CATALOG raw statistics, computed crudely, to check the planted signal exists."""
    est = ps.filter(pl.col("hcpcs_cd").is_in(["99211", "99212", "99213", "99214", "99215"]))
    up1 = est.group_by("npi", "year").agg(
        (pl.col("tot_srvcs").filter(pl.col("hcpcs_cd") == "99215").sum() / pl.col("tot_srvcs").sum()).alias("UP1"))
    nd = ps.filter(pl.col("hcpcs_drug_ind") == "N").group_by("npi", "year").agg(
        pl.col("tot_srvcs").sum().alias("nd_srv"),
        (pl.col("tot_srvcs") * pl.col("avg_sbmtd_chrg")).sum().alias("sb"),
        (pl.col("tot_srvcs") * pl.col("avg_mdcr_alowd_amt")).sum().alias("al"),
        (pl.col("tot_srvcs").filter(pl.col("hcpcs_cd").is_in(list(RARE))).sum()).alias("rare_srv"))
    dc1 = ds.group_by("npi", "year").agg((pl.col("tot_suplr_clms") / pl.col("tot_suplrs")).mean().alias("DC1"))
    return (prov.select("npi", "year", "prvdr_type", "tot_benes")
            .join(up1, on=["npi", "year"], how="left").join(nd, on=["npi", "year"], how="left")
            .join(dc1, on=["npi", "year"], how="left")
            .with_columns((pl.col("nd_srv") / pl.col("tot_benes")).alias("OU1"), (pl.col("sb") / pl.col("al")).alias("CH1"),
                          (pl.col("rare_srv") / pl.col("nd_srv")).alias("SM2")))


def test_planted_signal_is_learnable(tmp_path):
    proj = _new_proj(tmp_path)
    truth = make_synthetic(proj, n_npi=4000, years=range(2019, 2024), seed=3)
    st = _stat(_read(proj, "cms_ps"), _read(proj, "cms_prov"), _read(proj, "dme_s"))
    planted = truth["planted"].explode("schemes")
    clean = st.join(truth["planted"].select("npi"), on="npi", how="anti")
    for scheme, col in (("upcoding", "UP1"), ("overutilization", "OU1"), ("charge_inflation", "CH1"),
                        ("atypical_mix", "SM2"), ("dme_mill", "DC1")):
        hit = st.join(planted.filter(pl.col("schemes") == scheme).select("npi", "year"), on=["npi", "year"])
        hit = hit.filter(pl.col(col).is_not_null())
        assert hit.height >= 5, scheme
        # compare within specialty: median of planted provider-years vs 90th percentile of clean peers
        ref = clean.group_by("prvdr_type").agg(pl.col(col).fill_null(0).quantile(0.9).alias("p90"))
        above = hit.join(ref, on="prvdr_type").select((pl.col(col) > pl.col("p90")).mean()).item()
        assert above > 0.5, (scheme, above)


def test_deterministic_given_seed(tmp_path):
    a, b, c = (_new_proj(tmp_path / k) for k in "abc")
    ta = make_synthetic(a, n_npi=300, years=[2021, 2022], seed=7)
    tb = make_synthetic(b, n_npi=300, years=[2021, 2022], seed=7)
    make_synthetic(c, n_npi=300, years=[2021, 2022], seed=8)
    for ds in DATASETS:
        for y in (2021, 2022):
            fa, fb, fc = (pl.read_parquet(p.interim / ds / f"{y}.parquet") for p in (a, b, c))
            assert fa.equals(fb), (ds, y)
            assert not fa.equals(fc), (ds, y)
    assert ta["schemes"] == tb["schemes"] and ta["excluded_npis"] == tb["excluded_npis"]
    assert pl.read_parquet(a.processed / "labels_primary_W3.parquet").equals(
        pl.read_parquet(b.processed / "labels_primary_W3.parquet"))


def test_scales_to_benchmark_size(tmp_path):
    proj = _new_proj(tmp_path)
    t0 = time.perf_counter()
    truth = make_synthetic(proj, n_npi=20_000, years=[2022], seed=1)
    elapsed = time.perf_counter() - t0
    ps = pl.read_parquet(proj.interim / "cms_ps" / "2022.parquet")
    prov = pl.read_parquet(proj.interim / "cms_prov" / "2022.parquet")
    assert prov.height > 0.75 * 20_000 and ps.height > 5 * prov.height
    assert len(truth["npis"]) == 20_000
    assert elapsed < 30, f"20k NPIs x 1 year took {elapsed:.1f}s"


def _snapshot(root):
    d = root / "data"
    return sorted((str(p), p.stat().st_mtime_ns) for p in d.rglob("*")) if d.is_dir() else []


def test_refuses_the_checkout_and_projects_with_real_data(tmp_path):
    """CONTRACTS §0 (review finding): synthetic files only ever land in throw-away roots. The checkout (or a folder
    inside it) and any root holding non-synthetic harmonised or registered files are refused before any write."""
    before = _snapshot(REPO)
    for root in (REPO, REPO / "tests" / "scratch"):
        with pytest.raises(ValueError, match="inside the repository checkout"):
            make_synthetic(Paths(root), n_npi=60, years=[2022])
    assert _snapshot(REPO) == before and not (REPO / "synthetic_src").exists() and not (REPO / "tests" / "scratch").exists()

    proj = _new_proj(tmp_path / "a")
    real = proj.interim / "cms_ps" / "2022.parquet"
    real.parent.mkdir(parents=True)
    pl.DataFrame({"npi": ["1234567893"]}).write_parquet(real)
    real.with_suffix(".meta.json").write_text(json.dumps({"dataset": "cms_ps", "year": 2022}))   # phase1 harmonise
    with pytest.raises(ValueError, match="non-synthetic harmonised files"):
        make_synthetic(proj, n_npi=60, years=[2022])
    assert pl.read_parquet(real).height == 1 and not (proj.interim / "cms_prov").exists()

    proj = _new_proj(tmp_path / "b")
    from scenario_fraud import acquire
    acquire.save_manifest(proj, [acquire.Entry(
        path="data/raw/leie/2026/UPDATED.csv", source="leie", year=2026, kind="current", route="oig", source_page="",
        file_url="manual", download_date="2026-10-12", size_bytes=1, sha256="0" * 64, as_of="2026-10-12")])
    with pytest.raises(ValueError, match="registered non-synthetic raw files"):
        make_synthetic(proj, n_npi=60, years=[2022])

    proj = _new_proj(tmp_path / "c")                           # a synthetic project can be regenerated
    make_synthetic(proj, n_npi=60, years=[2022])
    f = proj.interim / "cms_ps" / "2022.parquet"
    f.with_suffix(".meta.json").unlink()                        # Parquet metadata alone still marks it synthetic
    from scenario_fraud import provenance
    assert provenance.file_provenance(f) == provenance.SYNTHETIC
