"""C9 background sampler: peer hierarchy and fallback, cap at B, determinism, global stratification."""

from __future__ import annotations

import polars as pl
import pytest

from scenario_fraud.attribution import background as bgm

CFG = bgm.BackgroundConfig(B=5, min_background=3, B_global=10, seed=1)


def _cand(spec: dict) -> pl.DataFrame:
    """spec: {(peer_cell, spec_ent, ent): n}"""
    rows = []
    i = 0
    for (pc, se, en), n in spec.items():
        for _ in range(n):
            rows.append({"npi": f"{1000000000 + i}", "peer_cell": pc, "cell_spec_ent": se, "cell_ent": en,
                         "strat_specialty": se.split("|")[-1]})
            i += 1
    return pl.DataFrame(rows)


CAND = _cand({("2022|I|card|pac", "2022|I|card", "2022|I"): 8,      # big refined cell
              ("2022|I|card|nen", "2022|I|card", "2022|I"): 2,      # small refined cell -> spec_ent
              ("2022|I|derm", "2022|I|derm", "2022|I"): 1,          # tiny spec cell -> spec_ent has 1 -> ent
              ("2022|O", "2022|O|amb", "2022|O"): 2})               # ent cell too small -> whole year


def _exp(*cells):
    return pl.DataFrame([{"npi": f"9{i:09d}", "peer_cell": c[0], "cell_spec_ent": c[1], "cell_ent": c[2]}
                         for i, c in enumerate(cells)])


def test_hierarchy_and_fallback():
    ex = _exp(("2022|I|card|pac", "2022|I|card", "2022|I"), ("2022|I|card|nen", "2022|I|card", "2022|I"),
              ("2022|I|derm", "2022|I|derm", "2022|I"), ("2022|O", "2022|O|amb", "2022|O"))
    a, m = bgm.peer_backgrounds(ex, CAND, CFG, 0, 2022)
    lv = dict(zip(a["npi"], a["bg_level"]))
    assert list(lv.values()) == ["peer_cell", "cell_spec_ent", "cell_ent", "year"]
    n = dict(zip(a["npi"], a["bg_n"]))
    assert list(n.values()) == [5, 5, 5, 5]                         # capped at B
    for key, g in m.group_by("bg_key"):
        lvl, val = key[0].split("=", 1)
        col = {"peer_cell": "peer_cell", "cell_spec_ent": "cell_spec_ent", "cell_ent": "cell_ent"}.get(lvl)
        members = CAND.filter(pl.col("npi").is_in(g["npi"].implode()))
        if col:
            assert (members[col] == val).all()


def test_deterministic_and_order_free():
    ex = _exp(("2022|I|card|pac", "2022|I|card", "2022|I"))
    a1, m1 = bgm.peer_backgrounds(ex, CAND, CFG, 0, 2022)
    a2, m2 = bgm.peer_backgrounds(ex, CAND.sample(fraction=1.0, shuffle=True, seed=3), CFG, 0, 2022)
    assert a1.equals(a2) and m1.sort("npi").equals(m2.sort("npi"))
    _, m3 = bgm.peer_backgrounds(ex, CAND, CFG, 1, 2022)              # another fold draws another sample
    assert not m1.sort("npi").equals(m3.sort("npi"))


def test_whole_year_too_small_fails():
    ex = _exp(("2022|O", "2022|O|amb", "2022|O"))
    with pytest.raises(ValueError, match="whole year"):
        bgm.peer_backgrounds(ex, CAND.head(2), CFG, 0, 2022)


def test_global_is_stratified_and_capped():
    g = bgm.global_background(CAND, CFG, 0, 2022)
    assert g.len() == 10 and g.n_unique() == 10
    spec = CAND.filter(pl.col("npi").is_in(g.implode()))["strat_specialty"].value_counts()
    counts = dict(zip(spec["strat_specialty"], spec["count"]))
    assert counts["card"] == 8                                        # 10 of 13 candidates: 10*10/13 = 7.7 -> 8
    assert bgm.global_background(CAND.head(4), CFG, 0, 2022).len() == 4
