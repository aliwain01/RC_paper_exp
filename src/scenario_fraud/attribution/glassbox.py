"""C12 glass-box native contributions, method D (DESIGN §9.2; EXPERIMENT_PLAN §4).

LR: c_j = β_j (Z_j(x) − mean_B Z_j(z)) per design column (standardised features and spline basis columns), summed
per group. EBM / GA²M: c_t = f_t(x) − mean_B f_t(z) per term; a single-feature term or a pair inside one group goes
to that group; a cross-group pair term goes to the matrix cell (G, H) and is split ½/½ between G and H for the
group vector (comparable with A and B). The pair matrix follows method C's convention: off-diagonal M_GH = M_HG =
c/2 (so M_GH + M_HG is the pair's compound effect) and the diagonal makes row sums equal the group vector.
Every form sums to f(x) − mean_B f(z).
"""

from __future__ import annotations

import numpy as np

from ..models import Fitted, lr_design


def lr_contrib(f: Fitted, X: np.ndarray, Xb: np.ndarray, groups_order: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """(group vector n × m, per-design-column contributions n × q) for an LR model."""
    Z, _, zg = lr_design(f, np.asarray(X, dtype=np.float64))
    Zb, _, _ = lr_design(f, np.asarray(Xb, dtype=np.float64))
    coef = f.model.named_steps["lr"].coef_.ravel()
    c = (Z - Zb.mean(axis=0)) * coef
    gi = np.array([groups_order.index(g) for g in zg])
    G = np.zeros((len(gi), len(groups_order)))
    G[np.arange(len(gi)), gi] = 1.0
    return c @ G, c


def lr_feature_contrib(f: Fitted, X: np.ndarray, Xb: np.ndarray) -> np.ndarray:
    """LR contributions per model column (a feature's spline basis columns added to it): n × p."""
    Z, names, _ = lr_design(f, np.asarray(X, dtype=np.float64))
    Zb, _, _ = lr_design(f, np.asarray(Xb, dtype=np.float64))
    c = (Z - Zb.mean(axis=0)) * f.model.named_steps["lr"].coef_.ravel()
    out = np.zeros((len(X), len(f.columns)))
    for j, name in enumerate(names):
        out[:, f.columns.index(name.split("__spl")[0])] += c[:, j]
    return out


def ebm_contrib(f: Fitted, X: np.ndarray, Xb: np.ndarray, col_group: np.ndarray, m: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(group vector n × m with ½-splits, pair matrix n × m × m, per-column contributions n × p with pair terms
    split ½/½ between their two columns) for an EBM / GA²M model."""
    if f.family not in ("ebm", "ga2m"):
        raise ValueError("ebm_contrib needs an ebm or ga2m model")
    T = f.model.eval_terms(np.asarray(X, dtype=np.float32))
    Tb = f.model.eval_terms(np.asarray(Xb, dtype=np.float32))
    c = T - Tb.mean(axis=0)
    n = len(X)
    vec = np.zeros((n, m))
    M = np.zeros((n, m, m))
    per_col = np.zeros((n, len(f.columns)))
    for t, feats in enumerate(f.model.term_features_):
        gs = [int(col_group[j]) for j in feats]
        for j in feats:
            per_col[:, j] += c[:, t] / len(feats)
        if len(set(gs)) == 1:
            vec[:, gs[0]] += c[:, t]
            M[:, gs[0], gs[0]] += c[:, t]
        elif len(feats) == 2:
            g, h = gs
            vec[:, g] += c[:, t] / 2
            vec[:, h] += c[:, t] / 2
            M[:, g, h] += c[:, t] / 2
            M[:, h, g] += c[:, t] / 2
        else:
            raise ValueError(f"EBM term {t} spans more than two features")
    return vec, M, per_col
