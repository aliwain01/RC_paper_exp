"""C13 metric functions on hand-built inputs (EXPERIMENT_PLAN §5, §8)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from scenario_fraud import evaluate as ev
from scenario_fraud import injection
from scenario_fraud.attribution import coalition as C
from scenario_fraud.groups import SCHEME_GROUPS

G = {g: i + 1 for i, g in enumerate(SCHEME_GROUPS)}     # index in the 8-vector (context = 0)


def _vec(**vals) -> list[float]:
    v = [0.0] * 8
    for g, x in vals.items():
        v[0 if g == "context" else G[g]] = x
    return v


def _att(rows):
    base = {"year": 2022, "fold": 0, "intensity": "medium", "distractor": False, "D_pair": None}
    df = pl.DataFrame([{**base, **r} for r in rows])
    return df.with_columns(pl.col("D_pair").cast(pl.List(pl.Float64)))


@pytest.fixture
def icfg(proj):
    return injection.InjectionConfig.load(proj)


def test_ranks_break_ties_by_order():
    r = ev.ranks(np.array([[0.1, 0.3, 0.3, -1.0]]))
    assert r.tolist() == [[3, 1, 2, 4]]


def test_chance_levels_are_exact():
    one = ev.chance((frozenset({0}),))
    assert one["top1"] == pytest.approx(1 / 7) and one["top3"] == pytest.approx(3 / 7)
    assert one["mrr"] == pytest.approx(sum(1 / r for r in range(1, 8)) / 7)
    two = ev.chance((frozenset({5, 6}),))                     # DME mill: either DME group counts (D90)
    assert two["top1"] == pytest.approx(2 / 7) and two["top3"] == pytest.approx(1 - math.comb(5, 3) / math.comb(7, 3))
    pair = ev.chance((frozenset({0}), frozenset({1})))
    assert pair["pair_hit"] == pytest.approx(1 / 21)
    assert pair["both_top3"] == pytest.approx((3 * 2 * math.factorial(5)) / math.factorial(7))
    assert ev.chance((frozenset({5, 6}), frozenset({1})))["pair_hit"] == pytest.approx(2 / 21)


def test_correctness_rows(icfg):
    Cm = np.zeros((8, 8))
    Cm[G["upcoding"], G["overutilization"]] = Cm[G["overutilization"], G["upcoding"]] = 0.4
    Cm[G["charge_inflation"], G["patient_mix"]] = Cm[G["patient_mix"], G["charge_inflation"]] = 0.1
    np.fill_diagonal(Cm, _vec(upcoding=0.5, overutilization=0.2, charge_inflation=0.6))
    rows = [
        {"npi": "1", "schemes": ["upcoding"], "pair": False, "answer_groups": ["upcoding"],
         "B": _vec(context=2.0, upcoding=0.9, charge_inflation=0.3), "C": Cm.ravel().tolist()},
        {"npi": "2", "schemes": ["upcoding"], "pair": False, "answer_groups": ["upcoding"],
         "B": _vec(upcoding=0.5, charge_inflation=0.9), "C": Cm.ravel().tolist()},
        {"npi": "3", "schemes": ["upcoding"], "pair": False, "answer_groups": ["upcoding"],
         "B": _vec(upcoding=-0.5, charge_inflation=0.9), "C": Cm.ravel().tolist()},
        {"npi": "4", "schemes": ["dme_mill"], "pair": False, "answer_groups": ["dme_intensity", "dme_concentration"],
         "B": _vec(dme_concentration=0.4, dme_intensity=0.1), "C": Cm.ravel().tolist()},
        {"npi": "5", "schemes": ["upcoding", "overutilization"], "pair": True,
         "answer_groups": ["upcoding", "overutilization"],
         "B": _vec(upcoding=0.9, overutilization=0.6, charge_inflation=0.1), "C": Cm.ravel().tolist()},
    ]
    att = _att(rows)
    b = ev.correctness_rows(att, "B", icfg).sort("npi")
    assert b["top1"].to_list() == [1.0, 0.0, 0.0, 1.0, 1.0]       # context is never ranked
    assert b["top3"].to_list() == [1.0, 1.0, 0.0, 1.0, 1.0]       # a negative contribution is not found
    assert b["rr"].to_list() == [1.0, 0.5, 0.0, 1.0, 1.0]
    assert b["chance_top1"].to_list() == pytest.approx([1 / 7, 1 / 7, 1 / 7, 2 / 7, 2 / 7])
    assert b["answer_share"][0] == pytest.approx(0.9 / 1.2) and b["spill_share"][0] == pytest.approx(0.3 / 1.2)
    assert b["both_top3"].to_list()[4] == 1.0 and b["both_top3"].null_count() == 4
    assert b["pair_hit"].null_count() == 5                        # B has no pair cells
    c = ev.correctness_rows(att, "C", icfg).sort("npi")
    assert c["top1"].to_list()[0] == 0.0                          # diagonal: charge inflation 0.6 > upcoding 0.5
    assert c["pair_hit"].to_list()[4] == 1.0                      # M_up,ou + M_ou,up = 0.8 is the largest pair


def test_consistency_metrics():
    a = np.array([[0.5, 0.2, -0.1, 0.0, 0.3, 0.01, 0.02]])
    same = ev.consistency_metrics(a, a * 2, 0.05)
    assert same["sign_agree"][0] == 1 and same["spearman"][0] == pytest.approx(1)
    assert same["jaccard3"][0] == 1 and same["cosine"][0] == pytest.approx(1) and same["mag_ratio"][0] == 0.5
    opp = ev.consistency_metrics(a, -a, 0.05)
    assert opp["sign_agree"][0] == 0 and opp["spearman"][0] == pytest.approx(-1) and opp["cosine"][0] == pytest.approx(-1)
    zero = ev.consistency_metrics(np.zeros((1, 7)), np.zeros((1, 7)), 0.05)
    assert np.isnan(zero["sign_agree"][0]) and np.isnan(zero["cosine"][0])


def test_flags_take_top_k_per_year():
    n = 300
    s = pl.DataFrame({"npi": [f"{1000000000 + i}" for i in range(2 * n)], "year": [2022] * n + [2023] * n,
                      "margin": np.r_[np.arange(n), -np.arange(n)].astype(float)})
    f = ev.flags(s, 0.01).filter(pl.col("flagged")).sort("year", "npi")
    assert f.height == 6
    assert f.filter(pl.col("year") == 2022)["npi"].to_list() == ["1000000297", "1000000298", "1000000299"]
    assert f.filter(pl.col("year") == 2023)["npi"].to_list() == ["1000000300", "1000000301", "1000000302"]


def test_derangement_stays_in_stratum():
    keys = pl.DataFrame({"npi": [f"{1000000000 + i}" for i in range(9)], "year": [2022] * 9,
                         "stratum": ["a"] * 4 + ["b"] * 4 + ["c"]})
    d = ev.derangement(keys, ["year", "stratum"], 1)
    assert d[8] == -1
    for i in range(8):
        assert d[i] != i and keys["stratum"][int(d[i])] == keys["stratum"][i]


def test_deletion_table_prefers_the_true_order():
    rng = np.random.default_rng(0)
    w = np.array([0.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.0])
    x = rng.uniform(0.5, 1.5, size=(20, 8))
    masks = C.column_masks(np.arange(8), 8)
    V = np.stack([(x * masks[s] * w).sum(axis=1) for s in range(256)], axis=1)   # additive model, zero background
    att = pl.DataFrame({"npi": [str(i) for i in range(20)], "year": [2022] * 20,
                        "v": V.tolist(), "B": (x * w).tolist(), "del_random": np.full(20, 0.5)})
    t, curves = ev.deletion_table(att, ("B",))
    assert t["valid"].all() and (t["B"] < t["B_reverse"]).all()
    assert curves["B"][0] == pytest.approx(1.0) and curves["B"][-1] == pytest.approx(0.0)


def test_fairness_table():
    s = pl.DataFrame({"npi": [f"{1000000000 + i}" for i in range(200)], "year": [2022] * 200,
                      "margin": np.arange(200, dtype=float)[::-1], "y": [1, 1] + [0] * 196 + [1, 0]})
    attrs = s.select("npi", pl.col("year").cast(pl.Int32)).with_columns(
        pl.Series("ent_cd", ["I"] * 100 + ["O"] * 100))
    t = {r["value"]: r for r in ev.fairness_table(s, attrs, 0.01, "ent_cd")}
    assert t["I"]["flagged"] == 2 and t["I"]["precision"] == 1.0 and t["O"]["flagged"] == 0
    assert t["I"]["flag_rate"] == 0.02 and t["I"]["flag_to_positive_ratio"] == pytest.approx(1.0)
    assert t["O"]["positive_rate"] == 0.01 and t["O"]["precision"] is None
