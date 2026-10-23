"""C7 folds and splits (DESIGN §7.6; EXPERIMENT_PLAN §2; CONTRACTS §2.3). Synthetic inputs only."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
import yaml

from scenario_fraud import cohort, labels, splits
from scenario_fraud.splits import ROLES, TRAIN_ROLES, SplitConfig, SplitError
from tests.synth import make_npis, make_synthetic

YEARS = list(range(2017, 2025))


def _edit_splits(proj, fn) -> None:
    p = proj.configs / "experiment.yaml"
    doc = yaml.safe_load(p.read_text())
    fn(doc)
    p.write_text(yaml.safe_dump(doc, sort_keys=False))


def _population(n: int = 6000, seed: int = 3) -> tuple[pl.DataFrame, list[str]]:
    """Cohort-like rows over 2018-2021: 13 specialties of very different sizes (some tiny), some NPIs switching
    specialty, some years out of cohort, some NPIs never in cohort; about 2 % ever-positive NPIs."""
    rng = np.random.default_rng(seed)
    npis = make_npis(rng, n)
    sizes = np.array([30, 20, 12, 10, 8, 6, 5, 4, 2, 1, 0.6, 0.3, 0.1])
    spec = rng.choice([f"Spec {chr(65 + i)}" for i in range(13)], n, p=sizes / sizes.sum())
    years = np.arange(2018, 2022)
    ii, jj = np.nonzero(rng.random((n, 4)) < 0.85)
    st = spec[ii].copy()
    switch = rng.random(ii.size) < 0.05
    st[switch] = "Spec A"
    df = pl.DataFrame({"npi": npis[ii], "year": years[jj].astype(np.int32), "prvdr_type": st,
                       "in_cohort": rng.random(ii.size) < 0.9})
    pos = sorted(rng.choice(npis, int(0.02 * n), replace=False).tolist())
    return df, pos


# ------------------------------------------------------------------------------------------------ config

def test_config_designs_and_showcase(proj):
    cfg = SplitConfig.load(proj)
    assert (cfg.n_folds, cfg.valid_frac, cfg.neg_rate) == (5, 0.2, 0.15)
    assert list(cfg.designs) == ["primary", "O1", "O2", "O3"]
    assert cfg.design("primary").train_years == (2018, 2019, 2020, 2021)
    assert cfg.design("primary").test_years == (2022, 2023)
    assert [(cfg.design(d).train, cfg.design(d).test) for d in ("O1", "O2", "O3")] == [
        ((2018, 2020), (2021, 2021)), ((2018, 2021), (2022, 2022)), ((2018, 2022), (2023, 2023))]
    sc = cfg.design("showcase")
    assert (sc.train, sc.test, sc.accuracy) == ((2018, 2021), (2024, 2024), False)
    assert [d.name for d in cfg.all_designs] == ["primary", "O1", "O2", "O3", "showcase"]
    with pytest.raises(SplitError, match="unknown design"):
        cfg.design("O4")


@pytest.mark.parametrize("edit, match", [
    (lambda d: d["splits"]["designs"]["primary"].update(train=[2018, 2020]), "disagrees with years"),
    (lambda d: d["splits"]["designs"]["O1"].update(test=[2020, 2021]), "precede every test year"),
    (lambda d: d["splits"]["designs"]["O2"].update(train=[2021, 2021]), ">= 2 train years"),
    (lambda d: d["splits"].update(valid_frac=1.0), "valid_frac"),
    (lambda d: d["splits"].update(valid_frac=0), "valid_frac"),
    (lambda d: d["splits"].update(n_folds=1), "n_folds"),
    (lambda d: d["splits"]["showcase"].update(design="O9"), "showcase design"),
    (lambda d: d["splits"]["showcase"].update(year=2021), "showcase year|disagrees"),
    (lambda d: d["splits"]["negative_subsample"].update(rate=0), "rates must be in"),
    (lambda d: d["splits"]["designs"].update(showcase={"train": [2018, 2021], "test": [2024, 2024]}), "showcase"),
    (lambda d: d["splits"].pop("seed"), "incomplete or malformed"),
    (lambda d: d["splits"]["designs"]["O3"].update(test=[2023]), "incomplete or malformed"),
])
def test_config_is_validated(proj, edit, match):
    _edit_splits(proj, edit)
    with pytest.raises(SplitError, match=match):
        SplitConfig.load(proj)


# ------------------------------------------------------------------------------------------------ folds

def _strata_counts(folds: pl.DataFrame) -> pl.DataFrame:
    return (folds.group_by("strat_specialty", "ever_positive", "fold").len()
            .pivot(on="fold", index=["strat_specialty", "ever_positive"], values="len").fill_null(0))


def test_folds_partition_stratified_and_balanced(proj):
    cfg = SplitConfig.load(proj)
    df, pos = _population()
    folds = splits.make_folds(df, pos, cfg)
    assert folds.columns == list(splits.FOLD_COLS)
    inc = df.filter("in_cohort")
    assert folds["npi"].to_list() == sorted(inc["npi"].unique().to_list())      # every in-cohort NPI, once
    assert folds["fold"].is_between(0, 4).all() and folds["fold"].n_unique() == 5
    # ever_positive = NPI in the positive list
    assert set(folds.filter("ever_positive")["npi"]) == set(pos) & set(folds["npi"])
    # balance: fold sizes, positives per fold and every stratum's count per fold differ by at most one
    per = folds.group_by("fold").agg(pl.len(), pl.col("ever_positive").sum())
    assert per["len"].max() - per["len"].min() <= 1
    assert per["ever_positive"].max() - per["ever_positive"].min() <= 1
    sc = _strata_counts(folds)
    ks = [str(k) for k in range(5) if str(k) in sc.columns]
    spread = sc.select(pl.max_horizontal(ks) - pl.min_horizontal(ks)).to_series()
    tiny = sc.filter(pl.sum_horizontal(ks) < 5)       # a stratum smaller than 5 cannot reach every fold
    assert spread.max() <= 1 and tiny.height > 0
    bal = splits.fold_balance(folds, 5)
    assert bal["max_stratum_spread"] == spread.max() and bal["fold_size_spread"] <= 1
    assert bal["ever_positive_spread"] <= 1 and bal["npis"] == folds.height and bal["strata"] == sc.height
    # modal specialty over the NPI's in-cohort years
    modal = (inc.group_by("npi", "prvdr_type").agg(pl.len(), pl.col("year").max())
             .sort(["npi", "len", "year"], descending=[False, True, True]).unique("npi", keep="first")
             .select("npi", pl.col("prvdr_type").alias("m")))
    assert folds.join(modal, on="npi").filter(pl.col("m") != pl.col("strat_specialty")).is_empty()


def test_modal_specialty_rules(proj):
    cfg = SplitConfig.load(proj)
    a, b, c = "1000000004", "1000000012", "1000000020"
    rows = [  # (npi, year, specialty, in cohort)
        (a, 2018, "Cardiology", True), (a, 2019, "Cardiology", True), (a, 2020, "Internal Medicine", True),
        (b, 2018, "Cardiology", True), (b, 2019, "Internal Medicine", True),       # tie -> latest year's
        (c, 2018, "Podiatry", False), (c, 2019, "Podiatry", False), (c, 2020, "Podiatry", False),
        (c, 2021, "Dermatology", True),                                             # out-of-cohort years ignored
        ("1000000038", 2021, "Dermatology", False),                                 # never in cohort: no fold
    ]
    df = pl.DataFrame(rows, schema=["npi", "year", "prvdr_type", "in_cohort"], orient="row")
    f = splits.make_folds(df, [b, "9999999999"], cfg)
    assert f.select("npi", "strat_specialty", "ever_positive").rows() == [
        (a, "Cardiology", False), (b, "Internal Medicine", True), (c, "Dermatology", False)]


def test_folds_deterministic_order_free_and_seeded(proj):
    cfg = SplitConfig.load(proj)
    df, pos = _population(3000, seed=5)
    f1 = splits.make_folds(df, pos, cfg)
    f2 = splits.make_folds(df.sample(fraction=1.0, shuffle=True, seed=11), list(reversed(pos)), cfg)
    assert f1.equals(f2)
    _edit_splits(proj, lambda d: d["splits"].update(seed=12345))
    f3 = splits.make_folds(df, pos, SplitConfig.load(proj))
    assert (f3["fold"] != f1["fold"]).mean() > 0.5
    assert splits.fold_balance(f3, 5)["max_stratum_spread"] <= 1


def test_npi_keys_are_splitmix64(proj):
    """The hash is plain splitmix64 arithmetic (version-independent): check against a pure-Python version."""
    mask = (1 << 64) - 1

    def ref(x: int) -> int:
        z = (x + 0x9E3779B97F4A7C15) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return z ^ (z >> 31)

    salt = splits._salt(20261013, 0)
    npis = ["1234567893", "9999999999", "1000000004"]
    got = splits.npi_keys(pl.Series(npis), salt)
    assert [int(g) for g in got] == [ref(int(n) ^ int(salt)) for n in npis]
    with pytest.raises(SplitError, match="10-digit"):
        splits.npi_keys(pl.Series(["12345"]), salt)


def test_holdout_fraction_stratified_deterministic_and_per_fold(proj):
    cfg = SplitConfig.load(proj)
    df, pos = _population()
    folds = splits.make_folds(df, pos, cfg)
    seen = []
    for k in range(5):
        h = splits.holdout_npis(folds, k, cfg)
        out = folds.filter(pl.col("fold") != k)
        assert set(h) <= set(out["npi"]) and h.n_unique() == h.len()
        assert h.len() == math.floor(0.2 * out.height)                       # exactly the 20 % share
        hf = out.with_columns(pl.col("npi").is_in(h.implode()).alias("h"))
        per = hf.group_by("strat_specialty", "ever_positive").agg(pl.len().alias("n"), pl.col("h").sum())
        assert (per["h"] - 0.2 * per["n"]).abs().max() < 1                    # each stratum within one
        npos = hf.filter("ever_positive")
        assert abs(npos["h"].sum() - 0.2 * npos.height) < 1
        assert h.equals(splits.holdout_npis(folds, k, cfg))                  # deterministic
        seen.append(set(h))
    assert all(seen[i] != seen[j] for i in range(5) for j in range(i + 1, 5))  # differs per fold
    _edit_splits(proj, lambda d: d["splits"].update(valid_frac=0.25))
    cfg25 = SplitConfig.load(proj)
    assert splits.holdout_npis(folds, 0, cfg25).len() == math.floor(0.25 * folds.filter(pl.col("fold") != 0).height)


# ------------------------------------------------------------------------------------------------ roles

A, B, C, D, E, F = (f"10000000{i:02d}" for i in range(6))
HAND_FOLDS = pl.DataFrame({"npi": [A, B, C, D, E, F], "fold": [0, 0, 1, 1, 2, 3],
                           "strat_specialty": ["X"] * 6, "ever_positive": [False] * 6},
                          schema_overrides={"fold": pl.Int32})


def _hand_frame() -> pl.DataFrame:
    rows = [(n, y, f"{n[-1]}{y}") for n in (A, B, C, D, E, F) for y in YEARS]
    return pl.DataFrame(rows, schema=["npi", "year", "x"], orient="row").with_columns(pl.col("year").cast(pl.Int32))


def _expected(spec: dict[str, dict[int, str]]) -> list[str]:
    return [spec.get(n, {}).get(y, "none") for n in (A, B, C, D, E, F) for y in YEARS]


def test_exact_roles_on_a_hand_built_frame(proj):
    cfg = SplitConfig.load(proj)
    fr = _hand_frame()
    held = [C, E]
    test, valid_pattern, tune_pattern = "test", {2018: "train", 2019: "train", 2020: "train", 2021: "tune_valid"}, \
        {2018: "tune_train", 2019: "tune_train", 2020: "tune_train", 2021: "train"}
    primary = splits.assign_roles(fr, HAND_FOLDS, 0, "primary", cfg, holdout=held)
    assert primary.columns == ["npi", "year", "x", "role"] and primary["x"].equals(fr["x"])   # order kept
    assert primary.schema["role"] == splits.ROLE_DTYPE
    assert primary["role"].cast(pl.String).to_list() == _expected({
        A: {2022: test, 2023: test}, B: {2022: test, 2023: test},
        C: valid_pattern, E: valid_pattern, D: tune_pattern, F: tune_pattern})
    o1 = splits.assign_roles(fr, HAND_FOLDS, 0, "O1", cfg, holdout=held)
    assert o1["role"].cast(pl.String).to_list() == _expected({
        A: {2021: test}, B: {2021: test},
        C: {2018: "train", 2019: "train", 2020: "tune_valid"}, E: {2018: "train", 2019: "train", 2020: "tune_valid"},
        D: {2018: "tune_train", 2019: "tune_train", 2020: "train"},
        F: {2018: "tune_train", 2019: "tune_train", 2020: "train"}})
    show = splits.assign_roles(fr, HAND_FOLDS, 0, "showcase", cfg, holdout=held)
    assert show["role"].cast(pl.String).to_list() == _expected({
        A: {2024: test}, B: {2024: test}, C: valid_pattern, E: valid_pattern, D: tune_pattern, F: tune_pattern})
    f1 = splits.assign_roles(fr, HAND_FOLDS, 1, "primary", cfg, holdout=[A, E])
    assert f1["role"].cast(pl.String).to_list() == _expected({
        C: {2022: test, 2023: test}, D: {2022: test, 2023: test},
        A: valid_pattern, E: valid_pattern, B: tune_pattern, F: tune_pattern})
    assert set(primary["role"].cast(pl.String)) == set(ROLES)
    # the training set (TRAIN_ROLES) is every non-fold-0 row of 2018-2021
    fit = primary.filter(pl.col("role").is_in(list(TRAIN_ROLES)))
    assert sorted(fit.select("npi", "year").rows()) == [(n, y) for n in (C, D, E, F) for y in range(2018, 2022)]


def test_exclude_years_moves_validation_to_the_last_remaining_train_year(proj):
    """Review finding: O1's last train year is the flagged 2020, so under sensitivity S4 (exclude 2020) it had no
    tune_valid rows. With exclude_years, excluded rows get `none` and validation moves to 2019."""
    cfg = SplitConfig.load(proj)
    fr = _hand_frame()
    held = [C, E]
    o1 = splits.assign_roles(fr, HAND_FOLDS, 0, "O1", cfg, holdout=held, exclude_years=[2020])
    assert o1["role"].cast(pl.String).to_list() == _expected({
        A: {2021: "test"}, B: {2021: "test"},
        C: {2018: "train", 2019: "tune_valid"}, E: {2018: "train", 2019: "tune_valid"},
        D: {2018: "tune_train", 2019: "train"}, F: {2018: "tune_train", 2019: "train"}})
    prim = splits.assign_roles(fr, HAND_FOLDS, 0, "primary", cfg, holdout=held, exclude_years=[2020])
    assert prim.filter(pl.col("year") == 2020)["role"].cast(pl.String).unique().to_list() == ["none"]
    assert set(prim.filter(pl.col("role") == "tune_valid")["year"]) == {2021}       # primary still validates on 2021
    assert splits.assign_roles(fr, HAND_FOLDS, 0, "O1", cfg, holdout=held, exclude_years=()).equals(
        splits.assign_roles(fr, HAND_FOLDS, 0, "O1", cfg, holdout=held))
    with pytest.raises(SplitError, match="tuning needs >= 2"):
        splits.assign_roles(fr, HAND_FOLDS, 0, "O1", cfg, holdout=held, exclude_years=[2019, 2020])


def test_roles_fail_loudly(proj):
    cfg = SplitConfig.load(proj)
    fr = _hand_frame()
    with pytest.raises(SplitError, match="have no fold"):
        splits.assign_roles(fr.vstack(pl.DataFrame({"npi": ["1999999999"], "year": [2019], "x": ["z"]},
                                                   schema=fr.schema)), HAND_FOLDS, 0, "primary", cfg, holdout=[C])
    with pytest.raises(SplitError, match="fold must be"):
        splits.assign_roles(fr, HAND_FOLDS, 5, "primary", cfg, holdout=[C])
    with pytest.raises(SplitError, match="unknown design"):
        splits.assign_roles(fr, HAND_FOLDS, 0, "rolling", cfg, holdout=[C])
    with pytest.raises(SplitError, match="columns missing"):
        splits.assign_roles(fr.drop("year"), HAND_FOLDS, 0, "primary", cfg, holdout=[C])
    with pytest.raises(SplitError, match="already has"):
        splits.assign_roles(fr.with_columns(pl.lit("x").alias("role")), HAND_FOLDS, 0, "primary", cfg, holdout=[C])
    with pytest.raises(SplitError, match="exactly one fold"):
        splits.assign_roles(fr, HAND_FOLDS.vstack(HAND_FOLDS.head(1).with_columns(pl.lit(4, dtype=pl.Int32)
                                                                                   .alias("fold"))), 0, "primary", cfg)
    with pytest.raises(SplitError, match="outside fold k"):
        splits.assign_roles(fr, HAND_FOLDS, 0, "primary", cfg, holdout=[A])


def test_split_rows_on_a_synthetic_world_disjoint_for_every_fold_and_design(proj):
    make_synthetic(proj, n_npi=900, years=range(2018, 2025))
    coh = cohort.build_cohort(proj)
    folds = splits.assign_folds(proj)
    cfg = SplitConfig.load(proj)
    assert (proj.processed / "folds.parquet").is_file() and (proj.processed / "folds.meta.json").is_file()
    inc = coh.filter("in_cohort").select("npi", "year")
    assert folds["npi"].to_list() == sorted(inc["npi"].unique().to_list())
    # ever_positive: an exclusion with an NPI under any tier's authorities
    lc = labels.LabelConfig.load(proj)
    auths = {a for t in lc.tiers.values() for a in t}
    exc = pl.read_parquet(proj.processed / "exclusions.parquet").filter(pl.col("authority").is_in(list(auths)))
    assert set(folds.filter("ever_positive")["npi"]) == set(exc["npi"].drop_nulls()) & set(folds["npi"])
    assert folds["ever_positive"].sum() >= 5
    fold_of = dict(folds.select("npi", "fold").iter_rows())
    for d in cfg.all_designs:
        tested = []
        for k in range(cfg.n_folds):
            r = splits.split_rows(inc, k, d.name, proj)
            assert r.height == inc.height
            role = pl.col("role")
            te = set(r.filter(role == "test")["npi"])
            tr = set(r.filter(role.is_in(list(TRAIN_ROLES)))["npi"])
            tt, tv = set(r.filter(role == "tune_train")["npi"]), set(r.filter(role == "tune_valid")["npi"])
            assert te and tr and tt and tv and not te & tr and not tt & tv
            assert all(fold_of[n] == k for n in te) and all(fold_of[n] != k for n in tr)
            yrs = {x: set(r.filter(role == x)["year"]) for x in ROLES}
            assert yrs["test"] <= set(d.test_years) and (yrs["train"] | yrs["tune_train"]) <= set(d.train_years)
            assert yrs["tune_valid"] == {d.last_train_year} and max(yrs["tune_train"]) < d.last_train_year
            held = set(splits.holdout_npis(folds, k, cfg))
            assert tv == held & set(inc.filter(pl.col("year") == d.last_train_year)["npi"])
            assert tt == (set(inc.filter(pl.col("year") < d.last_train_year)["npi"]) - held
                          - {n for n, f in fold_of.items() if f == k})
            tested.append(r.filter(role == "test").select("npi", "year"))
        allt = pl.concat(tested)
        assert not allt.is_duplicated().any()
        assert allt.height == inc.filter(pl.col("year").is_between(*d.test)).height   # every test row once


def test_assign_folds_needs_cohort_and_exclusions(proj):
    with pytest.raises(SplitError, match="cohort.parquet missing"):
        splits.assign_folds(proj)
    make_synthetic(proj, n_npi=200, years=[2018, 2019])
    cohort.build_cohort(proj)
    (proj.processed / "exclusions.parquet").unlink()
    with pytest.raises(SplitError, match="exclusions.parquet missing"):
        splits.assign_folds(proj)
    with pytest.raises(SplitError, match="folds.parquet missing"):
        splits.load_folds(proj)


# ------------------------------------------------------------------------------------------------ subsampling

def test_negative_subsample_rate_weights_and_identity(proj):
    cfg = SplitConfig.load(proj)
    rng = np.random.default_rng(9)
    n = 40_000
    npis = make_npis(rng, n // 4)
    df = pl.DataFrame({"npi": np.repeat(npis, 4), "year": np.tile(np.arange(2018, 2022, dtype=np.int32), n // 4),
                       "y": (rng.random(n) < 0.01).astype(np.int8)})
    s = splits.subsample_negatives(df, cfg)
    neg = int((df["y"] == 0).sum())
    kept_neg = s.filter(pl.col("y") == 0)
    assert s.filter(pl.col("y") == 1).height == int(df["y"].sum())                        # all positives
    assert abs(kept_neg.height / neg - 0.15) < 3 * math.sqrt(0.15 * 0.85 / neg)
    assert set(s.filter(pl.col("y") == 1)["weight"]) == {1.0} and set(kept_neg["weight"]) == {1 / 0.15}
    key = ["npi", "year"]
    shuffled = splits.subsample_negatives(df.sample(fraction=1.0, shuffle=True, seed=1), cfg)
    assert shuffled.sort(key).equals(s.sort(key))                                          # row order free
    sub = df.filter(pl.col("year") == 2019)
    assert splits.subsample_negatives(sub, cfg).sort(key).equals(s.filter(pl.col("year") == 2019).sort(key))
    s30 = splits.subsample_negatives(df, cfg, rate=0.30)
    assert set(kept_neg.select(key).rows()) <= set(s30.select(key).rows())                # nested (S12)
    assert splits.subsample_negatives(df, cfg, rate=1.0).height == df.height
    with pytest.raises(SplitError, match="without nulls"):
        splits.subsample_negatives(df.with_columns(pl.lit(None, dtype=pl.Int8).alias("y")), cfg)
    with pytest.raises(SplitError, match="rate"):
        splits.subsample_negatives(df, cfg, rate=0.0)
