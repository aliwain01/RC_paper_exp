"""C8 model families: fit / margin / save-load, monotone constraints, LR design matrix (method D), GA2M pairs."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from scenario_fraud import models

COLS = ["UP1", "UP0", "OU1", "SM1", "CH3", "CX1__cardiology"]   # robust_z, indicator, splines, one-hot


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(COLS))).astype(np.float32)
    X[:, 1] = rng.integers(0, 2, n)
    X[:, 5] = rng.integers(0, 2, n)
    logit = 1.2 * X[:, 0] + 0.8 * X[:, 2] * X[:, 3] - 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    w = np.where(y == 1, 1.0, 1 / 0.15)
    return X, y, w


@pytest.fixture
def cfg(proj):
    return models.ModelConfig.load(proj)


PARAMS = {
    "lr": {"C": 1.0, "l1_ratio": 0.5, "pos_weight": 2.0},
    "ebm": {"learning_rate": 0.05, "max_bins": 64, "max_leaves": 2, "min_samples_leaf": 5, "pos_weight": 2.0},
    "ga2m": {"interactions": 3, "learning_rate": 0.05, "max_bins": 64, "min_samples_leaf": 5, "pos_weight": 2.0},
    "xgb": {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 1.0,
            "min_child_weight": 1.0, "reg_lambda": 1.0, "pos_weight": 2.0},
    "lgbm": {"n_estimators": 30, "num_leaves": 7, "learning_rate": 0.1, "feature_fraction": 1.0,
             "bagging_fraction": 1.0, "min_child_samples": 5, "lambda_l2": 1.0, "pos_weight": 2.0},
}
PARAMS["xgb_mono"], PARAMS["lgbm_mono"] = PARAMS["xgb"], PARAMS["lgbm"]


@pytest.mark.parametrize("family", models.FAMILIES)
def test_fit_margin_roundtrip(proj, cfg, tmp_path, family):
    X, y, w = _data()
    f = models.fit(family, X, y, w, PARAMS[family], 3, COLS, cfg, proj)
    m = models.margin(f, X)
    assert m.shape == (len(y),) and np.isfinite(m).all()
    assert np.corrcoef(m, X[:, 0])[0, 1] > 0.2          # learned the planted main effect
    models.save(f, tmp_path / family)
    g = models.load(tmp_path / family)
    np.testing.assert_allclose(models.margin(g, X), m, rtol=1e-6, atol=1e-6)
    assert g.params == PARAMS[family] and g.columns == COLS
    np.testing.assert_allclose(models.margin(f, X, batch=97), m)   # batching does not change results


def test_fit_is_deterministic(proj, cfg):
    X, y, w = _data()
    for fam in ("xgb", "lgbm", "ebm"):
        a = models.margin(models.fit(fam, X, y, w, PARAMS[fam], 5, COLS, cfg, proj), X)
        b = models.margin(models.fit(fam, X, y, w, PARAMS[fam], 5, COLS, cfg, proj), X)
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("family", ["xgb_mono", "lgbm_mono"])
def test_monotone_constraint_respected(proj, cfg, family):
    X, y, w = _data()
    assert models.monotone_vector(COLS, proj)[0] == 1 and models.monotone_vector(COLS, proj)[5] == 0
    f = models.fit(family, X, y, w, PARAMS[family], 1, COLS, cfg, proj)
    grid = np.repeat(X[:50], 21, axis=0)
    grid[:, 0] = np.tile(np.linspace(-3, 3, 21), 50)
    m = models.margin(f, grid).reshape(50, 21)
    assert (np.diff(m, axis=1) >= -1e-6).all()


def test_lr_design_reproduces_decision_and_groups(proj, cfg):
    X, y, w = _data()
    f = models.fit("lr", X, y, w, PARAMS["lr"], 0, COLS, cfg, proj)
    Z, names, grp = models.lr_design(f, X, proj)
    lr = f.model.named_steps["lr"]
    np.testing.assert_allclose(Z @ lr.coef_.ravel() + lr.intercept_[0], models.margin(f, X), rtol=1e-5, atol=1e-6)  # float32 input
    spl = [n for n in names if "__spl" in n]
    assert {n.split("__spl")[0] for n in spl} == {"OU1", "SM1", "CH3"}         # spline features in COLS
    assert dict(zip(names, grp))["OU1__spl0"] == "overutilization"
    assert dict(zip(names, grp))["CX1__cardiology"] == "context"
    assert len(names) == Z.shape[1] and set(COLS) <= set(names)                  # linear terms kept


def test_ga2m_fast_finds_planted_pair(proj, cfg):
    """DESIGN §18: EBM pairwise detection (FAST) in InterpretML, confirmed in Phase 3."""
    X, y, w = _data(n=3000)
    f = models.fit("ga2m", X, y, w, PARAMS["ga2m"], 0, COLS, cfg, proj)
    assert "OU1 & SM1" in f.extra["pair_terms"]
    e = models.fit("ebm", X, y, w, PARAMS["ebm"], 0, COLS, cfg, proj)
    assert e.extra["pair_terms"] == []


def test_pos_weight_multiplies_positive_weights():
    y, w = np.array([1, 0, 1]), np.array([1.0, 6.0, 1.0])
    np.testing.assert_allclose(models._weights(y, w, 3.0), [3.0, 6.0, 3.0])


def test_lr_nonconvergence_recorded_not_raised(proj, cfg, tmp_path):
    X, y, w = _data()
    doc = yaml.safe_load((proj.configs / "models.yaml").read_text())
    doc["lr"]["fixed"]["max_iter"] = 1
    (proj.configs / "models.yaml").write_text(yaml.safe_dump(doc))
    f = models.fit("lr", X, y, w, PARAMS["lr"], 0, COLS, models.ModelConfig.load(proj), proj)
    assert f.extra == {"converged": False}


def test_config_rejects_leaky_calibration(proj):
    doc = yaml.safe_load((proj.configs / "models.yaml").read_text())
    doc["calibration_source"] = "final_model"
    (proj.configs / "models.yaml").write_text(yaml.safe_dump(doc))
    with pytest.raises(models.ModelError, match="exclude_holdout_npis"):
        models.ModelConfig.load(proj)
    doc["final_fit"] = "exclude_holdout_npis"
    (proj.configs / "models.yaml").write_text(yaml.safe_dump(doc))
    models.ModelConfig.load(proj)


def test_margin_rejects_wrong_width(proj, cfg):
    X, y, w = _data()
    f = models.fit("xgb", X, y, w, PARAMS["xgb"], 0, COLS, cfg, proj)
    with pytest.raises(models.ModelError, match="columns"):
        models.margin(f, X[:, :3])
