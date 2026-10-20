"""Phase 3 end to end through the CLI on synthetic data (tests/synth.py): phase2 -> phase3 train (primary, O1) ->
score-showcase -> report. Every number here comes from the synthetic generator and is not a result."""

from __future__ import annotations

import json
import shutil

import joblib
import numpy as np
import polars as pl
import pytest
import yaml

from scenario_fraud import cli, models, splits, train
from scenario_fraud.paths import Paths
from tests.conftest import REPO, copy_configs
from tests.synth import make_synthetic

FAMS = ["lr", "ebm", "xgb", "lgbm_mono"]


def _small_configs(root):
    m = yaml.safe_load((root / "configs" / "models.yaml").read_text())
    m["min_valid_positives"] = 1
    m["ebm"]["fixed"]["outer_bags"] = 1
    m["designs"]["primary"]["seeds"] = 2
    for f in ("xgb", "xgb_mono", "lgbm", "lgbm_mono"):
        m[f]["space"]["n_estimators"] = {"type": "int", "low": 20, "high": 40}
    (root / "configs" / "models.yaml").write_text(yaml.safe_dump(m, sort_keys=False))
    e = (root / "configs" / "experiment.yaml").read_text().replace("bootstrap_reps: 1000", "bootstrap_reps: 50")
    (root / "configs" / "experiment.yaml").write_text(e)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("w3") / "proj with space"
    (root / "design").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    copy_configs(root)
    _small_configs(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SCENARIO_FRAUD_ROOT", str(root))
        make_synthetic(Paths(root), n_npi=3000, excluded_frac=0.15)
        for c in (["cohort"], ["features"], ["folds"], ["qa", "--no-rebuild"], ["report"]):
            assert cli.main(["phase2", *c]) == 0
        fam = [x for f in FAMS for x in ("--family", f)]
        codes = [cli.main(["phase3", "train", "--design", "primary", "--trials", "2", *fam]),
                 cli.main(["phase3", "train", "--design", "O1", "--family", "lr", "--family", "xgb"]),
                 cli.main(["phase3", "score-showcase", *fam]),
                 cli.main(["phase3", "report"])]
    return root, codes


@pytest.fixture
def w(world, tmp_path, monkeypatch) -> Paths:
    root = tmp_path / "proj with space"
    shutil.copytree(world[0], root, symlinks=True)
    monkeypatch.setenv("SCENARIO_FRAUD_ROOT", str(root))
    return Paths(root)


def test_commands_succeed_and_report_says_synthetic(world, w):
    assert world[1] == [0, 0, 0, 0]
    md = (w.phase3 / "G3_report.md").read_text()
    assert "SYNTHETIC DATA: NOT A RESULT" in md and "## Design primary" in md and "## Design O1" in md
    rep = json.loads((w.phase3 / "G3_report.json").read_text())
    prim = rep["designs"]["primary"]
    tun = train.load_tuning(w, "primary").group_by("family").agg(pl.col("valid_ap").mean())
    va = dict(zip(tun["family"], tun["valid_ap"]))
    assert prim["best"]["glass_box"] == max(["lr", "ebm"], key=va.get)       # chosen on validation, not test
    assert prim["best"]["gbm"] == "xgb" and prim["best"]["gbm_mono"] == "lgbm_mono"
    assert prim["delta"]["a"] == "xgb" and isinstance(prim["delta"]["h1_supported"], bool)
    runs = [json.loads(p.read_text()) for p in sorted((w.phase3 / "runs").glob("*.json"))]
    assert {r["status"] for r in runs} == {"ok"} and all(r["result"]["synthetic_inputs"] for r in runs)


def test_scores_are_out_of_fold_and_complete(w):
    folds = splits.load_folds(w)
    lab = pl.read_parquet(w.processed / "labels_primary_W3.parquet").filter(pl.col("y").is_not_null())
    feats = pl.concat([pl.read_parquet(w.processed / "features" / f"{y}.parquet", columns=["npi", "year"])
                       for y in (2022, 2023)])
    expected = feats.join(lab.select("npi", "year"), on=["npi", "year"]).height
    for fam in FAMS:
        s = train.load_scores(w, "primary", fam)
        assert set(s["year"].unique()) == {2022, 2023}
        assert s.join(folds, on="npi").filter(pl.col("fold") != pl.col("fold_right")).height == 0
        for seed in (0, 1):
            ss = s.filter(pl.col("seed") == seed)
            assert ss.height == expected and not ss.select(pl.struct("npi", "year").is_duplicated().any()).item()


def test_saved_models_reproduce_scores_and_calibration_is_monotone(w):
    from scenario_fraud.modelmatrix import load_design, matrix, roles

    df = load_design(w, "primary")
    scfg = splits.SplitConfig.load(w)
    test = df.frame.filter(roles(df, 2, scfg) == "test")
    s = train.load_scores(w, "primary", "xgb").filter((pl.col("fold") == 2) & (pl.col("seed") == 0))
    d = train.model_dir(w, "primary", "xgb", 2) / "seed0"
    f = models.load(d)
    m = models.margin(f, matrix(test, f.columns))
    got = test.select("npi", "year").with_columns(pl.Series("m", m)).join(s, on=["npi", "year"])
    np.testing.assert_allclose(got["m"], got["margin"], rtol=1e-6)
    c = joblib.load(d / "calibrator.joblib")
    grid = np.linspace(m.min() - 1, m.max() + 1, 200)
    assert (np.diff(c.predict(grid)) >= 0).all()


def test_rolling_origin_reuses_primary_params(w):
    for fam in ("lr", "xgb"):
        for k in range(5):
            o = json.loads((train.model_dir(w, "O1", fam, k) / "tuning.json").read_text())
            p = json.loads((train.model_dir(w, "primary", fam, k) / "tuning.json").read_text())
            assert o["params_source"].startswith("primary") and o["best_params"] == p["best_params"]
            assert o["trials"] == [] and o["seeds"] == 1
    assert set(train.load_scores(w, "O1", "lr")["year"].unique()) == {2021}


def test_showcase_scores_2024_out_of_fold(w):
    s = train.load_scores(w, "showcase", "xgb")
    assert set(s["year"].unique()) == {2024}
    assert s.join(splits.load_folds(w), on="npi").filter(pl.col("fold") != pl.col("fold_right")).height == 0
    assert set(s["seed"].unique()) == {0, 1}


def test_training_is_deterministic(w):
    before = train.load_scores(w, "primary", "xgb").filter(pl.col("fold") == 0).sort(["seed", "npi", "year"])
    train.train_design(w, "primary", families=["xgb"], folds=[0], n_trials=2, resume=False)
    after = train.load_scores(w, "primary", "xgb").filter(pl.col("fold") == 0).sort(["seed", "npi", "year"])
    assert before.equals(after)


def test_resume_skips_finished_pieces(w):
    d = train.model_dir(w, "primary", "xgb", 0)
    stamp = (d / "tuning.json").stat().st_mtime_ns
    train.train_design(w, "primary", families=["xgb"], folds=[0], n_trials=2)
    assert (d / "tuning.json").stat().st_mtime_ns == stamp
    assert "already done" in (w.phase3 / "train_progress.log").read_text()
    train.train_design(w, "primary", families=["xgb"], folds=[0], n_trials=3)      # other budget: retrained
    assert json.loads((d / "tuning.json").read_text())["n_trials"] == 3


def test_resume_after_interruption_matches_uninterrupted_run(w, monkeypatch):
    key = ["seed", "npi", "year"]
    train.train_design(w, "primary", families=["xgb"], folds=[0], n_trials=3, resume=False)
    d = train.model_dir(w, "primary", "xgb", 0)
    full = train.load_scores(w, "primary", "xgb").filter(pl.col("fold") == 0).sort(key)
    best = json.loads((d / "tuning.json").read_text())["best_params"]
    # interrupted during the 2nd trial (a half-written line) and before seed 1 was saved
    tp = d / "progress" / "trials.jsonl"
    rows = tp.read_text().splitlines()
    tp.write_text(rows[0] + "\n" + rows[1][:15])
    (d / "tuning.json").unlink()
    train.score_path(w, "primary", "xgb", 0).unlink()
    shutil.rmtree(d / "seed1")
    fits = []
    real_fit = train.models.fit
    monkeypatch.setattr(train.models, "fit", lambda *a, **k: fits.append(a[0]) or real_fit(*a, **k))
    train.train_design(w, "primary", families=["xgb"], folds=[0], n_trials=3)
    assert len(fits) == 4          # trials 2 and 3, the tuning model, seed 1 (seed 0 loaded)
    assert json.loads((d / "tuning.json").read_text())["best_params"] == best
    assert [json.loads(r) for r in tp.read_text().splitlines()][0] == json.loads(rows[0])
    resumed = train.load_scores(w, "primary", "xgb").filter(pl.col("fold") == 0).sort(key)
    assert resumed.select(key).equals(full.select(key))
    np.testing.assert_allclose(resumed["margin"].to_numpy(), full["margin"].to_numpy(), rtol=1e-6, atol=1e-6)
    assert "1 logged trials replayed" in (w.phase3 / "train_progress.log").read_text()


def test_training_refuses_without_or_after_changing_the_freeze(w):
    (w.phase2 / "G2_freeze.json").rename(w.phase2 / "moved.json")
    with pytest.raises(train.TrainError, match="freeze"):
        train.train_design(w, "primary", families=["lr"], folds=[0], n_trials=1)
    (w.phase2 / "moved.json").rename(w.phase2 / "G2_freeze.json")
    e = w.configs / "experiment.yaml"
    e.write_text(e.read_text().replace("min_nondrug_services: 50", "min_nondrug_services: 40"))
    with pytest.raises(train.TrainError, match="protocol changed"):
        train.train_design(w, "primary", families=["lr"], folds=[0], n_trials=1)


def test_too_few_validation_positives_fails_loudly(w):
    m = w.configs / "models.yaml"
    doc = yaml.safe_load(m.read_text())
    doc["min_valid_positives"] = 10_000
    m.write_text(yaml.safe_dump(doc))
    with pytest.raises(train.TrainError, match="min_valid_positives"):
        train.train_design(w, "primary", families=["lr"], folds=[0], n_trials=1)


def test_bench_times_one_fold_without_scoring(w):
    before = sorted((w.processed / "scores").rglob("*.parquet"))
    assert cli.main(["phase3", "bench", "--family", "lr", "--family", "xgb", "--trials", "2"]) == 0
    b = json.loads((w.phase3 / "bench.json").read_text())
    assert set(b["families"]) == {"lr", "xgb"} and b["projected_hours_total"] >= 0
    assert all(len(v["trial_seconds"]) == 2 for v in b["families"].values())
    assert sorted((w.processed / "scores").rglob("*.parquet")) == before      # no test row scored
