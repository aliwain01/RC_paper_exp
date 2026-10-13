"""Detection metrics (EXPERIMENT_PLAN §3; DESIGN §10.1; CONTRACTS §3.4), shared with C13.

Scores are ranked by the raw margin within each test year. precision@k = positives in the top ceil(k*N) / ceil(k*N)
with N the scored provider-years of that year; pooled over years = per-year top-k counts summed. Ties are broken
by a fixed hash of (npi, year), so results do not depend on row order. PR-AUC is average precision (scikit-learn's
definition), computed here on presorted arrays so the paired NPI bootstrap can reweight rows without re-sorting:
a bootstrap replicate resamples NPIs with replacement and gives each row its NPI's draw count as weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from .splits import _mix64, npi_keys

_TIE_SALT = np.uint64(0x5EED_7E57_0000_0001)


class MetricsError(RuntimeError):
    pass


def tie_keys(npi: pl.Series, year: pl.Series) -> np.ndarray:
    return _mix64(npi_keys(npi, _TIE_SALT) ^ year.cast(pl.UInt64).to_numpy())


@dataclass
class Ranked:
    """One model's test scores, sorted by (year, margin desc, tie key) with NPI codes for the bootstrap."""
    year: np.ndarray
    y: np.ndarray
    margin: np.ndarray
    prob: np.ndarray | None
    npi_code: np.ndarray       # integer code of the NPI (shared coding between paired models)
    years: tuple[int, ...]
    # pooled order (margin desc, tie key) for PR-AUC over all years
    pooled_order: np.ndarray
    pooled_group_end: np.ndarray   # True at the last row of each tied-margin group in pooled order


def rank(df: pl.DataFrame, npi_codes: dict[str, int] | None = None) -> Ranked:
    """df: npi, year, y, margin (+ prob). Rows must be unique per (npi, year)."""
    need = [c for c in ("npi", "year", "y", "margin") if c not in df.columns]
    if need:
        raise MetricsError(f"score frame missing {need}")
    if df.select(pl.struct("npi", "year").is_duplicated().any()).item():
        raise MetricsError("score frame has duplicated (npi, year) rows")
    if df["y"].null_count() or df["margin"].null_count():
        raise MetricsError("score frame has null y or margin")
    tk = tie_keys(df["npi"], df["year"])
    d = df.with_columns(pl.Series("_tk", tk)).sort(["year", "margin", "_tk"], descending=[False, True, True])
    if npi_codes is None:
        u = d["npi"].unique().sort()
        npi_codes = dict(zip(u.to_list(), range(u.len())))
    code = d["npi"].replace_strict(npi_codes, return_dtype=pl.Int64).to_numpy()
    year = d["year"].to_numpy()
    m = d["margin"].to_numpy().astype(np.float64)
    po = np.lexsort((-d["_tk"].to_numpy().astype(np.float64), -m))
    pm = m[po]
    pend = np.r_[pm[1:] != pm[:-1], True]
    return Ranked(year, d["y"].to_numpy().astype(np.int8), m,
                  d["prob"].to_numpy().astype(np.float64) if "prob" in d.columns else None,
                  code, tuple(sorted(set(year.tolist()))), po, pend)


# ------------------------------------------------------------------------------------------------ weighted metrics

def _ap(y: np.ndarray, w: np.ndarray, group_end: np.ndarray) -> float:
    """Average precision on rows already sorted by score desc; ties form groups (scikit-learn definition)."""
    tp = np.cumsum(w * y)[group_end]
    fp = np.cumsum(w * (1 - y))[group_end]
    P = tp[-1] if tp.size else 0.0
    if P <= 0:
        return float("nan")
    prec = tp / np.maximum(tp + fp, 1e-300)
    rec = tp / P
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def _topk(y: np.ndarray, w: np.ndarray, k: float) -> tuple[float, float, float]:
    """(flagged n, positives in top ceil(k*N), positives) for rows sorted by score desc, weights = multiplicity."""
    N = float(w.sum())
    if N <= 0:
        return 0.0, 0.0, 0.0
    n = math.ceil(k * N - 1e-9)
    cw = np.cumsum(w)
    idx = np.searchsorted(cw, n, side="left")         # row where the cumulative weight reaches n
    tp_before = float(np.sum((w * y)[:idx]))
    take = n - (cw[idx - 1] if idx > 0 else 0.0)       # copies of row idx inside the top n
    tp = tp_before + (take * y[idx] if idx < len(y) else 0.0)
    return float(n), tp, float(np.sum(w * y))


def weighted_metrics(r: Ranked, w: np.ndarray, k_fracs) -> dict:
    """PR-AUC (pooled), precision/recall@k pooled over years, for row weights w (aligned with r's rows)."""
    out = {"pr_auc": _ap(r.y[r.pooled_order].astype(np.float64), w[r.pooled_order], r.pooled_group_end)}
    for k in k_fracs:
        n = tp = P = 0.0
        for yr in r.years:
            sel = r.year == yr
            a, b, c = _topk(r.y[sel].astype(np.float64), w[sel], k)
            n, tp, P = n + a, tp + b, P + c
        out[f"p@{k}"] = tp / n if n else float("nan")
        out[f"r@{k}"] = tp / P if P else float("nan")
    return out


# ------------------------------------------------------------------------------------------------ point metrics

def detection_metrics(df: pl.DataFrame, k_fracs, *, reliability_top: float = 0.05, reliability_bins: int = 10) -> dict:
    """Pooled and per-year metrics of one model's test scores (npi, year, y, margin, prob)."""
    from sklearn.metrics import roc_auc_score

    r = rank(df)
    w = np.ones(len(r.y))

    def block(sel: np.ndarray | None) -> dict:
        if sel is None:
            yy, mm, pp, years = r.y, r.margin, r.prob, r.years
            ww = w
            sub = r
        else:
            yy, mm, pp = r.y[sel], r.margin[sel], (r.prob[sel] if r.prob is not None else None)
            ww = w[sel]
            years = tuple(sorted(set(r.year[sel].tolist())))
            sub = rank(df.filter(pl.col("year").is_in(list(years))))
        N, P = len(yy), int(yy.sum())
        base = P / N if N else float("nan")
        wm = weighted_metrics(sub, np.ones(len(sub.y)), k_fracs)
        res = {"years": list(years), "N": N, "positives": P, "base_rate": base, "pr_auc": wm["pr_auc"],
               "roc_auc": float(roc_auc_score(yy, mm)) if 0 < P < N else float("nan")}
        for k in k_fracs:
            res[f"p@{k}"] = wm[f"p@{k}"]
            res[f"r@{k}"] = wm[f"r@{k}"]
            res[f"lift@{k}"] = wm[f"p@{k}"] / base if base else float("nan")
        if pp is not None:
            res["brier"] = float(np.mean((pp - yy) ** 2))
        return res

    out = {"pooled": block(None), "per_year": {str(y): block(r.year == y) for y in r.years}}
    if r.prob is not None:
        out["reliability_top"] = reliability(df, reliability_top, reliability_bins)
    return out


def reliability(df: pl.DataFrame, top: float, bins: int) -> list[dict]:
    """Calibration within the top `top` share of each year by margin: `bins` quantile bins of predicted
    probability with mean predicted vs observed rate."""
    parts = []
    for (yr,), g in df.sort("year").group_by("year", maintain_order=True):
        n = math.ceil(top * g.height)
        parts.append(g.sort("margin", descending=True).head(n))
    t = pl.concat(parts)
    if t.height == 0:
        return []
    t = t.with_columns((pl.col("prob").rank("ordinal") * bins / (t.height + 1)).floor().cast(pl.Int32).alias("bin"))
    return (t.group_by("bin").agg(pl.len().alias("n"), pl.col("prob").mean().alias("mean_predicted"),
                                   pl.col("y").cast(pl.Float64).mean().alias("observed"))
            .sort("bin").to_dicts())


# ------------------------------------------------------------------------------------------------ bootstrap

def paired_bootstrap(a: pl.DataFrame, b: pl.DataFrame, k_fracs, reps: int, seed: int, ci: float = 0.95) -> dict:
    """Paired NPI bootstrap of model a and model b on the same test rows. Returns, per metric (PR-AUC and
    precision/recall@k pooled), the point values, a's and b's percentile CIs, and Δ = a − b with CI and a
    two-sided bootstrap p-value (2 × min share of Δ* ≤ 0, share ≥ 0)."""
    ka = a.select("npi", "year").sort(["npi", "year"])
    kb = b.select("npi", "year").sort(["npi", "year"])
    if not ka.equals(kb):
        raise MetricsError("paired bootstrap needs both models scored on the same (npi, year) rows")
    u = a["npi"].unique().sort()
    codes = dict(zip(u.to_list(), range(u.len())))
    ra, rb = rank(a, codes), rank(b, codes)
    rng = np.random.default_rng(seed)
    n_npi = u.len()
    pa, pb = weighted_metrics(ra, np.ones(len(ra.y)), k_fracs), weighted_metrics(rb, np.ones(len(rb.y)), k_fracs)
    names = list(pa)
    A = np.empty((reps, len(names)))
    B = np.empty((reps, len(names)))
    for i in range(reps):
        cnt = np.bincount(rng.integers(0, n_npi, n_npi), minlength=n_npi).astype(np.float64)
        ma = weighted_metrics(ra, cnt[ra.npi_code], k_fracs)
        mb = weighted_metrics(rb, cnt[rb.npi_code], k_fracs)
        A[i] = [ma[n] for n in names]
        B[i] = [mb[n] for n in names]
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    out = {}
    for j, n in enumerate(names):
        D = A[:, j] - B[:, j]
        ok = np.isfinite(D)
        Dk = D[ok]
        p = (min(1.0, 2 * min(np.mean(Dk <= 0), np.mean(Dk >= 0)))) if Dk.size else float("nan")
        out[n] = {"a": pa[n], "b": pb[n], "delta": pa[n] - pb[n],
                  "a_ci": _q(A[:, j], lo, hi), "b_ci": _q(B[:, j], lo, hi), "delta_ci": _q(D, lo, hi),
                  "p_value": float(p), "valid_reps": int(ok.sum())}
    return out


def bootstrap_ci(df: pl.DataFrame, k_fracs, reps: int, seed: int, ci: float = 0.95) -> dict:
    """Percentile CIs of one model's PR-AUC and precision/recall@k (NPI bootstrap)."""
    r = rank(df)
    rng = np.random.default_rng(seed)
    n_npi = int(r.npi_code.max()) + 1
    point = weighted_metrics(r, np.ones(len(r.y)), k_fracs)
    names = list(point)
    M = np.empty((reps, len(names)))
    for i in range(reps):
        cnt = np.bincount(rng.integers(0, n_npi, n_npi), minlength=n_npi).astype(np.float64)
        m = weighted_metrics(r, cnt[r.npi_code], k_fracs)
        M[i] = [m[n] for n in names]
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    return {n: {"value": point[n], "ci": _q(M[:, j], lo, hi)} for j, n in enumerate(names)}


def _q(x: np.ndarray, lo: float, hi: float) -> list[float]:
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, lo)), float(np.quantile(x, hi))] if x.size else [float("nan")] * 2


def holm(pvalues: dict[str, float], alpha: float) -> dict[str, dict]:
    """Holm step-down adjustment: adjusted p-values (monotone) and rejection at `alpha`."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[name] = {"p": p, "p_holm": adj, "reject": bool(adj <= alpha)}
    return out
