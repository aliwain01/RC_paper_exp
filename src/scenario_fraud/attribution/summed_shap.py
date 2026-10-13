"""C11 summed SHAP, method A (DESIGN §9.2, §9.4; EXPERIMENT_PLAN §4).

Interventional TreeSHAP against the provider's background B on the raw margin, then summed per group:
Φᴬ_G = Σ_{j∈G} φ_j, with Σ_j φ_j = f(x) − mean_B f(z). The background is passed as an `Independent` masker sized to
|B|: shap's default masker subsamples backgrounds above 100 rows, which silently breaks additivity against B
(shap 0.53; DESIGN §18). Path-dependent TreeSHAP and its interaction values are diagnostics only (DESIGN §9.4).
"""

from __future__ import annotations

import warnings

import numpy as np

from ..models import Fitted

TREE_FAMILIES = ("xgb", "xgb_mono", "lgbm", "lgbm_mono")


def _check(f: Fitted) -> None:
    if f.family not in TREE_FAMILIES:
        raise ValueError(f"method A needs a tree model; got {f.family}")


def feature_shap(f: Fitted, X: np.ndarray, Xb: np.ndarray) -> tuple[np.ndarray, float]:
    """(φ: n × p, expected value = mean_B f) by interventional TreeSHAP on the raw margin."""
    import shap

    _check(f)
    Xb = np.asarray(Xb, dtype=np.float32)
    e = shap.TreeExplainer(f.model, data=shap.maskers.Independent(Xb, max_samples=len(Xb)),
                           feature_perturbation="interventional", model_output="raw")
    phi = np.asarray(e.shap_values(np.asarray(X, dtype=np.float32)), dtype=np.float64)
    if phi.ndim != 2 or phi.shape != (len(X), Xb.shape[1]):
        raise ValueError(f"unexpected SHAP output shape {phi.shape}")
    return phi, float(np.ravel(e.expected_value)[0])


def group_sums(phi: np.ndarray, col_group: np.ndarray, m: int) -> np.ndarray:
    """n × m sums of per-column values by group."""
    G = np.zeros((phi.shape[1], m))
    G[np.arange(phi.shape[1]), col_group] = 1.0
    return phi @ G


def path_dependent(f: Fitted, X: np.ndarray, col_group: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostics: path-dependent group sums (n × m) and feature-level TreeSHAP interaction values aggregated to
    groups (n × m × m; cell GH = Σ_{j∈G, k∈H} φ_jk)."""
    import shap

    _check(f)
    e = shap.TreeExplainer(f.model, feature_perturbation="tree_path_dependent")
    X = np.asarray(X, dtype=np.float32)
    with warnings.catch_warnings():
        # shap 0.53 announces a list-of-arrays output for LightGBM classifiers but returns the raw-margin array
        # for a Booster (checked: values + expected_value = raw score); the shape check below guards it.
        warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer shap values")
        phi = np.asarray(e.shap_values(X), dtype=np.float64)
        iv = np.asarray(e.shap_interaction_values(X), dtype=np.float64)
    if phi.shape != X.shape or iv.shape != (*X.shape, X.shape[1]):
        raise ValueError(f"unexpected path-dependent SHAP shapes {phi.shape}, {iv.shape}")
    G = np.zeros((X.shape[1], m))
    G[np.arange(X.shape[1]), col_group] = 1.0
    return phi @ G, np.einsum("njk,jg,kh->ngh", iv, G, G)
