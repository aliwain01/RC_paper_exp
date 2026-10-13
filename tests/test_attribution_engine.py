"""EXPERIMENT_PLAN §4 unit tests for C10 (coalition table), C11 (summed SHAP) and C12 (glass box), plus deletion
curves and counterfactuals. All on small synthetic arrays."""

from __future__ import annotations

from itertools import permutations
from math import factorial

import numpy as np
import pytest

from scenario_fraud import models
from scenario_fraud.attribution import coalition as C
from scenario_fraud.attribution import glassbox, summed_shap
from scenario_fraud.groups import GROUP_ORDER

M = 8
# 2 columns per group: column j belongs to group j // 2
COL_GROUP = np.repeat(np.arange(M), 2)
P = len(COL_GROUP)


def _xb(n=300, nb=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, P)).astype(np.float32), rng.normal(size=(nb, P)).astype(np.float32)


def test_additivity_and_interaction_matrix_sums():
    X, Xb = _xb()
    beta = np.linspace(-1, 1, P)
    f = lambda Z: Z @ beta + 0.7 * Z[:, 2] * Z[:, 9] - 0.3 * Z[:, 0] * Z[:, 1] + np.sin(Z[:, 5])  # noqa: E731
    V = C.coalition_values(f, X[:20], Xb, COL_GROUP, M, batch_rows=10_000)
    phi = C.shapley(V)
    full = 2 ** M - 1
    np.testing.assert_allclose(phi.sum(1), V[:, full] - V[:, 0], atol=1e-6)            # §4 test 1
    Mx = C.interaction_matrix(V, phi)
    np.testing.assert_allclose(Mx.sum(axis=(1, 2)), V[:, full] - V[:, 0], atol=1e-6)
    np.testing.assert_allclose(Mx.sum(axis=2), phi, atol=1e-9)                          # row sums = Φ
    np.testing.assert_allclose(Mx, np.transpose(Mx, (0, 2, 1)), atol=1e-12)
    np.testing.assert_allclose(V[:, full], f(X[:20]), atol=1e-5)                       # v(N) = f(x)
    np.testing.assert_allclose(V[:, 0], np.full(20, f(Xb).mean()), atol=1e-5)          # v(∅) = E_B f


def test_shapley_matches_permutation_definition():
    rng = np.random.default_rng(1)
    m = 4
    V = rng.normal(size=(3, 2 ** m))
    phi = C.shapley(V)
    brute = np.zeros((3, m))
    for perm in permutations(range(m)):
        s = 0
        for g in perm:
            brute[:, g] += V[:, s | (1 << g)] - V[:, s]
            s |= 1 << g
    np.testing.assert_allclose(phi, brute / factorial(m), atol=1e-12)


def test_planted_pairwise_term_recovered_exactly():
    """§4 test 2: f = additive + c·x_p·x_q with p in group 1, q in group 4."""
    X, Xb = _xb(seed=2)
    beta = np.random.default_rng(3).normal(size=P)
    c, p, q = 1.3, 2, 9                  # columns 2 (group 1) and 9 (group 4)
    f = lambda Z: Z @ beta + c * Z[:, p] * Z[:, q]  # noqa: E731
    x = X[:15]
    V = C.coalition_values(f, x, Xb, COL_GROUP, M)
    phi = C.shapley(V)
    Mx = C.interaction_matrix(V, phi)
    Ezp, Ezq, Ezpq = Xb[:, p].mean(), Xb[:, q].mean(), (Xb[:, p] * Xb[:, q]).mean()
    I = c * (x[:, p] * x[:, q] - x[:, p] * Ezq - Ezp * x[:, q] + Ezpq)
    np.testing.assert_allclose(Mx[:, 1, 4], I / 2, atol=1e-5)
    np.testing.assert_allclose(Mx[:, 4, 1] + Mx[:, 1, 4], I, atol=1e-5)
    off = [(g, h) for g in range(M) for h in range(M) if g != h and {g, h} != {1, 4}]
    assert max(abs(Mx[:, g, h]).max() for g, h in off) < 1e-5                         # no other pair interacts
    T = lambda a, b: c * a * b  # noqa: E731
    lin = lambda g: (x[:, COL_GROUP == g] - Xb[:, COL_GROUP == g].mean(0)) @ beta[COL_GROUP == g]  # noqa: E731
    phi1 = lin(1) + 0.5 * (T(x[:, p], Ezq) - c * Ezpq) + 0.5 * (T(x[:, p], x[:, q]) - Ezp * c * x[:, q])
    np.testing.assert_allclose(phi[:, 1], phi1, atol=1e-5)
    np.testing.assert_allclose(phi[:, 0], lin(0), atol=1e-5)


def test_linear_model_shapley_equals_lr_contributions():
    """§4 test 3 on a pure linear model: Φᴮ_G = Σ_{j∈G} β_j (x_j − mean_B z_j)."""
    X, Xb = _xb(seed=4)
    beta = np.random.default_rng(5).normal(size=P)
    V = C.coalition_values(lambda Z: Z @ beta + 0.4, X[:10], Xb, COL_GROUP, M)
    want = np.stack([((X[:10] - Xb.mean(0)) * beta)[:, COL_GROUP == g].sum(1) for g in range(M)], 1)
    np.testing.assert_allclose(C.shapley(V), want, atol=1e-5)
    assert abs(C.interaction_matrix(V)[:, ~np.eye(M, dtype=bool)]).max() < 1e-6


def test_counterfactual_and_deletion():
    X, Xb = _xb(seed=6)
    beta = np.zeros(P)
    beta[[0, 4, 6]] = [2.0, 1.0, -1.0]   # groups 0, 2, 3
    x = X[(X @ beta) > 1.0][:5]          # deletion is defined for providers riskier than the baseline
    V = C.coalition_values(lambda Z: Z @ beta, x, Xb, COL_GROUP, M)
    d, dp = C.counterfactual(V)
    np.testing.assert_allclose(d, C.shapley(V), atol=1e-5)            # additive model: Δ_G = Φ_G
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    np.testing.assert_allclose(dp[:, 0], sig(V[:, -1]) - sig(V[:, -1] - d[:, 0]), atol=1e-12)
    order = C.order_by(C.shapley(V))
    curve = C.deletion_curve(V, order)
    full = 2 ** M - 1
    np.testing.assert_allclose(curve[:, 0], V[:, full]) and np.testing.assert_allclose(curve[:, -1], V[:, 0])
    best = C.deletion_score(V, order)
    worst = C.deletion_score(V, order[:, ::-1])
    rnd = C.random_deletion_score(V, 50, 1)
    assert np.all(best <= rnd + 1e-12) and np.all(rnd <= worst + 1e-12)


def test_batching_does_not_change_values():
    X, Xb = _xb(seed=7)
    f = lambda Z: np.tanh(Z @ np.arange(P) / P) + Z[:, 3] * Z[:, 12]  # noqa: E731
    a = C.coalition_values(f, X[:7], Xb, COL_GROUP, M, batch_rows=1)
    b = C.coalition_values(f, X[:7], Xb, COL_GROUP, M, batch_rows=10 ** 7)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_context_attribution_zero_when_background_matches():
    """DESIGN §9.3: a context trait shared by the provider and every background row gets exactly 0."""
    X, Xb = _xb(seed=8)
    Xb[:, 0] = Xb[:, 1] = 1.0
    x = X[:5].copy()
    x[:, 0] = x[:, 1] = 1.0              # group 0 = context, matched
    f = lambda Z: 3 * Z[:, 0] + Z[:, 0] * Z[:, 5] + Z[:, 7]  # noqa: E731
    phi = C.shapley(C.coalition_values(f, x, Xb, COL_GROUP, M))
    assert abs(phi[:, 0]).max() < 1e-9


# --------------------------------------------------------------------------------- real model families (C11, C12)

COLS = ["UP1", "UP2", "OU1", "OU2", "SM1", "SM2", "CH1", "CH3", "PM1", "PM2", "DI1", "DI2", "DC1", "DC2",
        "CX5", "CX8"]


@pytest.fixture
def fam_data(proj):
    from scenario_fraud.groups import feature_groups

    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, len(COLS))).astype(np.float32)
    logit = X[:, 0] + 0.8 * X[:, 2] * X[:, 6] - 2
    y = (rng.random(1500) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    grp = feature_groups(COLS, paths=proj)
    col_group = np.array([GROUP_ORDER.index(grp[c]) for c in COLS])
    return X, y, np.ones(1500), col_group, models.ModelConfig.load(proj)


def _fit(fam, data, proj, **params):
    X, y, w, _, cfg = data
    base = {"lr": {"C": 1.0, "l1_ratio": 0.0},
            "ebm": {"learning_rate": 0.05, "max_bins": 64, "max_leaves": 2, "min_samples_leaf": 5},
            "ga2m": {"interactions": 4, "learning_rate": 0.05, "max_bins": 64, "min_samples_leaf": 5},
            "xgb": {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1, "subsample": 1.0,
                    "colsample_bytree": 1.0, "min_child_weight": 1.0, "reg_lambda": 1.0}}[fam]
    return models.fit(fam, X, y, w, {**base, **params}, 0, COLS, cfg, proj)


@pytest.mark.parametrize("fam", ["lr", "ebm"])
def test_additive_glass_box_D_equals_B(proj, fam_data, fam):
    """LR (splines act per feature) and EBM without pairs are additive: D equals B exactly (§4 test 3)."""
    X, _, _, cg, _ = fam_data
    f = _fit(fam, fam_data, proj)
    x, xb = X[:12], X[1000:1080]
    V = C.coalition_values(lambda Z: models.margin(f, Z), x, xb, cg, M)
    if fam == "lr":
        D, _ = glassbox.lr_contrib(f, x, xb, GROUP_ORDER)
        per = glassbox.lr_feature_contrib(f, x, xb)
    else:
        D, Mx, per = glassbox.ebm_contrib(f, x, xb, cg, M)
        assert abs(Mx[:, ~np.eye(M, dtype=bool)]).max() == 0
    np.testing.assert_allclose(D, C.shapley(V), atol=1e-5)
    full = models.margin(f, x) - models.margin(f, xb).mean()
    np.testing.assert_allclose(D.sum(1), full, atol=1e-5)
    np.testing.assert_allclose(per.sum(1), full, atol=1e-5)


def test_ga2m_D_sums_and_pair_cells(proj, fam_data):
    X, _, _, cg, _ = fam_data
    f = _fit("ga2m", fam_data, proj)
    x, xb = X[:12], X[1000:1080]
    D, Mx, _ = glassbox.ebm_contrib(f, x, xb, cg, M)
    full = models.margin(f, x) - models.margin(f, xb).mean()
    np.testing.assert_allclose(D.sum(1), full, atol=1e-5)
    np.testing.assert_allclose(Mx.sum(2), D, atol=1e-9)
    np.testing.assert_allclose(Mx, np.transpose(Mx, (0, 2, 1)), atol=1e-12)


def test_summed_shap_additivity_and_stumps_equal_B(proj, fam_data):
    """C11 unit test: |Σφ − (f(x) − mean_B f)| < 1e-4 with a 200-row background (no silent subsampling); depth-1
    trees are additive, so A equals B."""
    X, _, _, cg, _ = fam_data
    f = _fit("xgb", fam_data, proj)
    x, xb = X[:20], X[1000:1200]
    phi, ev = summed_shap.feature_shap(f, x, xb)
    assert abs(phi.sum(1) - (models.margin(f, x) - models.margin(f, xb).mean())).max() < 1e-4
    assert abs(ev - models.margin(f, xb).mean()) < 1e-4
    s = _fit("xgb", fam_data, proj, max_depth=1)
    A = summed_shap.group_sums(summed_shap.feature_shap(s, x, xb)[0], cg, M)
    V = C.coalition_values(lambda Z: models.margin(s, Z), x, xb, cg, M)
    np.testing.assert_allclose(A, C.shapley(V), atol=1e-4)


def test_path_dependent_diagnostics_shapes(proj, fam_data):
    X, _, _, cg, _ = fam_data
    f = _fit("xgb", fam_data, proj)
    g, iv = summed_shap.path_dependent(f, X[:10], cg, M)
    assert g.shape == (10, M) and iv.shape == (10, M, M)
    np.testing.assert_allclose(iv.sum(2), g, atol=1e-4)
