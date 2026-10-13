"""Detection metrics (EXPERIMENT_PLAN §3): hand-computed precision/recall@k, AP against scikit-learn, weighted
forms against repeated rows, paired bootstrap, Holm."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score

from scenario_fraud import metrics


def _frame(y, m, years=None, npis=None, prob=None):
    n = len(y)
    d = {"npi": npis or [f"{1000000000 + i}" for i in range(n)], "year": years or [2022] * n,
         "y": y, "margin": [float(v) for v in m]}
    if prob is not None:
        d["prob"] = prob
    return pl.DataFrame(d).with_columns(pl.col("year").cast(pl.Int32), pl.col("y").cast(pl.Int8))


def test_precision_recall_at_k_ceil():
    y = [1, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    m = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    d = metrics.detection_metrics(_frame(y, m), [0.15, 0.3])["pooled"]
    assert d["p@0.15"] == pytest.approx(1 / 2)          # ceil(1.5) = 2 flagged: rows 9, 8
    assert d["p@0.3"] == pytest.approx(2 / 3)           # ceil(3) = 3 flagged
    assert d["r@0.3"] == pytest.approx(2 / 3)
    assert d["lift@0.3"] == pytest.approx((2 / 3) / 0.3)


def test_pooled_sums_per_year_counts():
    y = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    m = [5, 4, 3, 2, 1, 1, 5, 4, 3, 2]
    years = [2022] * 5 + [2023] * 5
    d = metrics.detection_metrics(_frame(y, m, years), [0.2])
    assert d["per_year"]["2022"]["p@0.2"] == 1.0 and d["per_year"]["2023"]["p@0.2"] == 0.0
    assert d["pooled"]["p@0.2"] == pytest.approx(0.5)   # (1 + 0) / (1 + 1)


def test_ties_broken_independently_of_row_order():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200).tolist()
    m = rng.integers(0, 5, 200).tolist()                 # many ties
    f = _frame(y, m)
    a = metrics.detection_metrics(f, [0.05, 0.1])["pooled"]
    b = metrics.detection_metrics(f.sample(fraction=1.0, shuffle=True, seed=3), [0.05, 0.1])["pooled"]
    assert a == b


def test_ap_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    m = np.round(rng.normal(size=500), 1)
    d = metrics.detection_metrics(_frame(y.tolist(), m.tolist()), [0.01])["pooled"]
    assert d["pr_auc"] == pytest.approx(average_precision_score(y, m), abs=1e-12)


def test_weighted_equals_repeated_rows():
    rng = np.random.default_rng(2)
    n = 60
    y = rng.integers(0, 2, n).tolist()
    m = rng.normal(size=n).tolist()
    years = [2022 if i % 2 else 2023 for i in range(n)]
    npis = [f"{2000000000 + i}" for i in range(n)]
    f = _frame(y, m, years, npis)
    r = metrics.rank(f)
    cnt = rng.integers(0, 4, n).astype(float)
    code = {p: i for i, p in enumerate(sorted(npis))}
    w = cnt[r.npi_code]                                   # each row weighted by its NPI's draw count
    got = metrics.weighted_metrics(r, w, [0.1, 0.25])
    rep = f.with_columns(pl.Series("c", cnt[[code[p] for p in f["npi"]]].astype(int)))
    rows = rep.select(pl.all().repeat_by("c").explode()).drop("c")
    rows = rows.with_columns((pl.col("npi") + pl.int_range(pl.len()).cast(pl.String)).alias("npi"))
    # a repeated row's copies are adjacent ties: precision@k only depends on how many copies fall in the top n
    want_ap = average_precision_score(rows["y"].to_numpy(), rows["margin"].to_numpy())
    assert got["pr_auc"] == pytest.approx(want_ap, abs=1e-12)
    for k in (0.1, 0.25):
        n_tot = tp = 0
        for yr in (2022, 2023):
            g = rows.filter(pl.col("year") == yr).sort("margin", descending=True)
            n = math.ceil(k * g.height)
            n_tot += n
            tp += int(g.head(n)["y"].sum())
        assert got[f"p@{k}"] == pytest.approx(tp / n_tot)


def test_paired_bootstrap_identical_models_and_seed():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300).tolist()
    m = rng.normal(size=300).tolist()
    f = _frame(y, m)
    b = metrics.paired_bootstrap(f, f, [0.05], reps=50, seed=7)
    assert b["pr_auc"]["delta"] == 0 and b["pr_auc"]["delta_ci"] == [0.0, 0.0] and b["pr_auc"]["p_value"] == 1.0
    g = f.with_columns(pl.col("margin") + pl.Series(rng.normal(size=300)))
    assert metrics.paired_bootstrap(f, g, [0.05], 50, 7) == metrics.paired_bootstrap(f, g, [0.05], 50, 7)
    with pytest.raises(metrics.MetricsError, match="same"):
        metrics.paired_bootstrap(f, f.head(10), [0.05], 5, 1)


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(4)
    y = (rng.random(400) < 0.2).astype(int)
    m = y + rng.normal(size=400)
    ci = metrics.bootstrap_ci(_frame(y.tolist(), m.tolist()), [0.1], reps=200, seed=1)
    for v in ci.values():
        assert v["ci"][0] <= v["value"] <= v["ci"][1]


def test_holm():
    h = metrics.holm({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05)
    assert h["a"]["p_holm"] == pytest.approx(0.03) and h["c"]["p_holm"] == pytest.approx(0.06)
    assert h["b"]["p_holm"] == pytest.approx(0.06)               # monotone step-down
    assert h["a"]["reject"] and not h["b"]["reject"]


def test_reliability_bins_in_top_share():
    n = 1000
    rng = np.random.default_rng(5)
    m = rng.normal(size=n)
    p = 1 / (1 + np.exp(-m))
    y = (rng.random(n) < p).astype(int)
    rel = metrics.reliability(_frame(y.tolist(), m.tolist(), prob=p.tolist()), 0.1, 5)
    assert sum(r["n"] for r in rel) == 100 and len(rel) == 5


def test_rank_rejects_duplicates_and_nulls():
    with pytest.raises(metrics.MetricsError, match="duplicated"):
        metrics.rank(_frame([1, 0], [1, 2], npis=["1111111112", "1111111112"]))
