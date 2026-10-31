"""C10 coalition-table engine (DESIGN §9.1-9.2, §10.4; EXPERIMENT_PLAN §4-5).

For a provider x, background B and m groups, the coalition value of S ⊆ N is
    v(S) = (1/|B|) Σ_{z∈B} f(x_S, z_{N∖S})
(groups in S keep the provider's values, the others take each background row's; interventional). Subsets are
bitmasks over the groups in groups.yaml order: bit g set = group g kept. One table V (n × 2^m) gives
    B  group Shapley values                 Φ_G = Σ_{S∌G} |S|!(m−|S|−1)!/m! · [v(S∪G) − v(S)]
    C  group interaction matrix             M_GH = I_GH / 2 (Grabisch-Roubens index), M_GG = Φ_G − Σ_{H≠G} M_GH
    E  peer counterfactual                  Δ_G = v(N) − v(N∖{G})
    deletion curves                         f_j = v(N ∖ {π_1..π_j}), j = 0..m
All cells of M sum to v(N) − v(∅); row sums equal Φ.
"""

from __future__ import annotations

from math import factorial
from typing import Callable

import numpy as np


def column_masks(col_group: np.ndarray, m: int) -> np.ndarray:
    """(2^m, p) bool: column j is taken from the provider in subset s iff its group's bit is set."""
    s = np.arange(2 ** m)[:, None]
    return ((s >> np.asarray(col_group)[None, :]) & 1).astype(bool)


def coalition_values(predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray, Xb: np.ndarray,
                     col_group: np.ndarray, m: int, batch_rows: int = 500_000) -> np.ndarray:
    """V (n × 2^m) for providers X (n × p) sharing background Xb (|B| × p). `predict` returns raw margins."""
    X = np.asarray(X, dtype=np.float32)
    Xb = np.asarray(Xb, dtype=np.float32)
    n, p = X.shape
    if Xb.ndim != 2 or Xb.shape[1] != p or len(col_group) != p:
        raise ValueError("X, Xb and col_group disagree")
    if len(Xb) == 0:
        raise ValueError("empty background")
    if np.max(col_group) >= m or np.min(col_group) < 0:
        raise ValueError("col_group outside 0..m-1")
    cm = column_masks(col_group, m)
    S, nb = cm.shape[0], len(Xb)
    per = S * nb
    chunk = max(1, batch_rows // per)
    V = np.empty((n, S), dtype=np.float64)
    for a in range(0, n, chunk):
        xc = X[a:a + chunk]
        H = np.where(cm[None, :, None, :], xc[:, None, None, :], Xb[None, None, :, :])
        out = np.asarray(predict(H.reshape(-1, p)), dtype=np.float64)
        V[a:a + len(xc)] = out.reshape(len(xc), S, nb).mean(axis=2)
    return V


def _m(V: np.ndarray) -> int:
    m = int(round(np.log2(V.shape[1])))
    if 2 ** m != V.shape[1]:
        raise ValueError("V must have 2^m columns")
    return m


def shapley(V: np.ndarray) -> np.ndarray:
    """Method B: exact group Shapley values (n × m)."""
    m = _m(V)
    S = np.arange(2 ** m)
    size = np.array([bin(s).count("1") for s in S])
    out = np.empty((V.shape[0], m))
    for g in range(m):
        without = S[((S >> g) & 1) == 0]
        w = np.array([factorial(k) * factorial(m - k - 1) / factorial(m) for k in size[without]])
        out[:, g] = (V[:, without | (1 << g)] - V[:, without]) @ w
    return out


def interaction_matrix(V: np.ndarray, phi: np.ndarray | None = None) -> np.ndarray:
    """Method C: n × m × m, off-diagonal I_GH/2 (pairwise Shapley interaction index), diagonal Φ_G − Σ off-diagonal."""
    m = _m(V)
    phi = shapley(V) if phi is None else phi
    S = np.arange(2 ** m)
    size = np.array([bin(s).count("1") for s in S])
    M = np.zeros((V.shape[0], m, m))
    for g in range(m):
        for h in range(g + 1, m):
            base = S[(((S >> g) & 1) == 0) & (((S >> h) & 1) == 0)]
            w = np.array([factorial(k) * factorial(m - k - 2) / factorial(m - 1) for k in size[base]])
            gi, hi = 1 << g, 1 << h
            I = (V[:, base | gi | hi] - V[:, base | gi] - V[:, base | hi] + V[:, base]) @ w
            M[:, g, h] = M[:, h, g] = I / 2
    idx = np.arange(m)
    M[:, idx, idx] = phi - (M.sum(axis=2) - M[:, idx, idx])
    return M


def counterfactual(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Method E: Δ_G = v(N) − v(N∖{G}) in log-odds and in probability units."""
    m = _m(V)
    full = 2 ** m - 1
    drop = np.array([full & ~(1 << g) for g in range(m)])
    vn = V[:, [full]]
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
    return vn - V[:, drop], sig(vn) - sig(V[:, drop])


def deletion_curve(V: np.ndarray, order: np.ndarray) -> np.ndarray:
    """f_j = v(N minus the first j groups of `order`), j = 0..m (n × (m+1)); `order` is n × m or m."""
    m = _m(V)
    order = np.broadcast_to(np.asarray(order), (V.shape[0], m))
    removed = np.zeros(V.shape[0], dtype=np.int64)
    full = 2 ** m - 1
    cols = [np.full(V.shape[0], full)]
    for j in range(m):
        removed |= (1 << order[:, j]).astype(np.int64)
        cols.append(full & ~removed)
    idx = np.stack(cols, axis=1)
    return np.take_along_axis(V, idx, axis=1)


def deletion_score(V: np.ndarray, order: np.ndarray) -> np.ndarray:
    """EXPERIMENT_PLAN §5.3: area under (f_j − f_m)/(f_0 − f_m) over j = 0..m, divided by m (trapezoid); lower is
    more faithful. Meaningful for providers above their baseline (f_0 > f_m, e.g. flagged providers); NaN when
    f_0 = f_m. Evaluation (C13) restricts it to providers with f_0 − f_m > 0."""
    f = deletion_curve(V, order)
    den = f[:, :1] - f[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(np.abs(den) > 1e-12, (f - f[:, -1:]) / den, np.nan)
    m = f.shape[1] - 1
    return (r[:, :-1] + r[:, 1:]).sum(axis=1) / (2 * m)


def order_by(attr: np.ndarray) -> np.ndarray:
    """Removal order: groups by decreasing attribution (ties by group index)."""
    return np.argsort(-attr, axis=1, kind="stable")


def random_deletion_score(V: np.ndarray, n_orders: int, seed: int) -> np.ndarray:
    """Mean deletion score over `n_orders` random orders (the same orders for every provider)."""
    m = _m(V)
    rng = np.random.default_rng(seed)
    scores = [deletion_score(V, rng.permutation(m)) for _ in range(n_orders)]
    return np.stack(scores, axis=1).mean(axis=1)   # f_0 − f_m is order-free: all orders NaN or none
