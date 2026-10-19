"""Phase 5 end to end through the CLI on synthetic data: phase2 -> phase3 (lr, xgb) -> phase5 calibrate, build,
train, stress, explain, report. Every number here comes from tests/synth.py and is not a result."""

from __future__ import annotations

import json
import shutil

import numpy as np
import polars as pl
import pytest
import yaml

from scenario_fraud import cli, features, inject_models, injection, splits
from scenario_fraud.paths import Paths
from tests.conftest import REPO, copy_configs
from tests.synth import make_synthetic

FAMS = ["lr", "xgb"]


def _small(root):
    m = yaml.safe_load((root / "configs" / "models.yaml").read_text())
    m["min_valid_positives"] = 1
    m["designs"]["primary"]["seeds"] = 1
    m["xgb"]["space"]["n_estimators"] = {"type": "int", "low": 20, "high": 40}
    (root / "configs" / "models.yaml").write_text(yaml.safe_dump(m, sort_keys=False))
    a = yaml.safe_load((root / "configs" / "attribution.yaml").read_text())
    a["background"].update(B=20, min_background=8, B_global=40)
    (root / "configs" / "attribution.yaml").write_text(yaml.safe_dump(a, sort_keys=False))
    i = yaml.safe_load((root / "configs" / "injection.yaml").read_text())
    i["test"].update(per_cell=3, per_pair=3)
    i["train"]["share"] = 0.02
    i["families"] = FAMS
    i["calibration"]["sample_per_scheme"] = 10
    i["calibration"]["grid"] = {s: v[2:4] for s, v in i["calibration"]["grid"].items()}
    (root / "configs" / "injection.yaml").write_text(yaml.safe_dump(i, sort_keys=False))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("w5") / "proj with space"
    (root / "design").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    copy_configs(root)
    _small(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SCENARIO_FRAUD_ROOT", str(root))
        make_synthetic(Paths(root), n_npi=3000, excluded_frac=0.15)
        for c in (["cohort"], ["features"], ["folds"], ["qa", "--no-rebuild"], ["report"]):
            assert cli.main(["phase2", *c]) == 0
        assert cli.main(["phase3", "train", "--trials", "1", *[x for f in FAMS for x in ("--family", f)]]) == 0
        codes = [cli.main(["phase5", *c]) for c in (["calibrate"], ["build", "--draw", "0"], ["train", "--draw", "0"],
                                                     ["stress", "--draw", "0"],
                                                     ["explain", "--draw", "0", "--limit", "3"], ["report"])]
    return root, codes


@pytest.fixture
def w(world, tmp_path, monkeypatch) -> Paths:
    root = tmp_path / "proj with space"
    shutil.copytree(world[0], root, symlinks=True)
    monkeypatch.setenv("SCENARIO_FRAUD_ROOT", str(root))
    return Paths(root)


def test_commands_and_report(world, w):
    assert world[1] == [0] * 6
    md = (w.phase5 / "G5_report.md").read_text()
    assert "SYNTHETIC DATA: NOT A RESULT" in md and "Calibration" in md
    rep = json.loads((w.phase5 / "G5_report.json").read_text())
    assert all(c["status"] == "pass" for c in rep["draws"]["0"]["leakage"])
    assert rep["draws"]["0"]["shortfalls"] == []


def test_plan_is_clean_unique_and_deterministic(w):
    p = pl.read_parquet(injection.draw_dir(w, 0) / "plan.parquet")
    folds = splits.load_folds(w)
    assert p.join(folds.filter(pl.col("ever_positive")), on="npi", how="semi").height == 0
    assert not p.select(pl.struct("npi", "year").is_duplicated().any()).item()
    assert set(p.filter(pl.col("part") == "test")["year"].unique()) <= {2022, 2023}
    assert set(p.filter(pl.col("part") == "train")["year"].unique()) <= {2018, 2019, 2020, 2021}
    again = injection.plan(w, 0)
    assert again.select(p.columns).equals(p)
    other = injection.plan(w, 1)
    assert not set(zip(other["npi"], other["year"])) == set(zip(p["npi"], p["year"]))
    up = p.filter(pl.col("schemes").list.contains("upcoding"))
    elig = injection.eligibility(w, sorted(up["year"].unique().to_list()), injection.InjectionConfig.load(w))
    assert up.join(elig.filter(pl.col("ok_upcoding")), on=["npi", "year"], how="anti").height == 0


def test_injected_features_flag_and_clean_raw_unchanged(w):
    inj = injection.load_injected_features(w, 0, [2022])
    clean = features.load_features(w, [2022])
    p = pl.read_parquet(injection.draw_dir(w, 0) / "plan.parquet").filter(pl.col("year") == 2022)
    assert set(inj.filter(pl.col("injected"))["npi"]) == set(p["npi"])
    j = inj.filter(~pl.col("injected")).select("npi", "raw__OU1", "raw__CH1").join(
        clean.select("npi", pl.col("raw__OU1").alias("o"), pl.col("raw__CH1").alias("c")), on="npi")
    assert j.height > 0
    np.testing.assert_allclose(j["raw__OU1"].fill_null(-1), j["o"].fill_null(-1))   # other providers' rows untouched
    np.testing.assert_allclose(j["raw__CH1"].fill_null(-1), j["c"].fill_null(-1))
    ov = p.filter((pl.col("schemes") == ["overutilization"]) & (pl.col("intensity") == "high"))
    if ov.height:
        a = ov.join(inj, on="npi").select("npi", "raw__OU1").join(clean.select("npi", pl.col("raw__OU1").alias("o")),
                                                                  on="npi")
        assert (a["raw__OU1"] > a["o"]).all()


def test_injection_models_score_out_of_fold(w):
    folds = splits.load_folds(w)
    for fam in FAMS:
        s = pl.concat([pl.read_parquet(f) for f in (injection.draw_dir(w, 0) / "scores" / fam).glob("*.parquet")])
        assert s.join(folds, on="npi").filter(pl.col("fold") != pl.col("fold_right")).height == 0
        assert set(s["year"].unique()) == {2022, 2023} and s["y"].sum() > 0
        assert s.filter(pl.col("y") == 1)["schemes"].null_count() == 0


def test_stress_and_attributions(w):
    a = pl.read_parquet(injection.draw_dir(w, 0) / "use_a.parquet")
    assert a["percentile"].is_between(0, 1).all() and set(a["family"]) == set(FAMS)
    x = inject_models.load_injected_attributions(w, "xgb")
    B = np.array(x["B"].to_list())
    V = np.array(x["v"].to_list())
    np.testing.assert_allclose(B.sum(1), V[:, -1] - V[:, 0], atol=1e-6)
    assert x["answer_groups"].null_count() == 0 and x["A"].null_count() == 0
    lr = inject_models.load_injected_attributions(w, "lr")
    assert lr["D"].null_count() == 0


def test_calibration_written(w):
    c = json.loads((w.phase5 / "calibration.json").read_text())
    assert set(c["schemes"]) == set(injection.SCHEME_ORDER)
    for v in c["schemes"].values():
        assert len(v["grid"]) == 2 and all(0 <= g["median_percentile"] <= 1 for g in v["grid"])
