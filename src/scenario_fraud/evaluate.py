"""C13 evaluation suite (DESIGN §10.2-10.7; EXPERIMENT_PLAN §5, §6 use (a), §8; CONTRACTS §6).

Explanation vectors a_i are the 7 scheme-group entries of a method's group vector (context excluded). Methods A, B,
D, E use their stored group vectors; method C ranks by the diagonal of M (pure scheme effects) and its pair hit uses
M_GH + M_HG; GA²M's method D pair hit uses its pair cells (DESIGN §17 D101). A group counts as found only with a
positive contribution ("rankings use positive contributions"); ranks break ties by group order.

- correctness (E4): top-1, top-3, MRR, compound both-in-top-3, pair hit, answer / spill-over shares, distractors,
  with exact chance levels for each answer key (enumerated over the 7! group orders);
- consistency (E3): GBM methods vs glass-box D, by flag agreement at k = 1 %, against a permutation baseline;
- faithfulness (E6): deletion scores read from the coalition tables (+ method C's order), random and reverse orders;
- baselines (E8): peer vs global baseline for GBM methods A, B, E;
- stress (E5): use (a) percentiles of injected providers under the LEIE models;
- fairness (E10): flag rate, precision and flag rate / positive rate by provider attributes at k ∈ {0.5, 1} %.
Per-row results go to data/processed/evaluation/, summaries to outputs/phase6/{part}.json. Stability (E7) is in
`stability.py`, the sensitivity suite (E9) in `sensitivity.py`.
"""

from __future__ import annotations

import itertools
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

from . import injection, metrics, models, splits, train
from .attribute import load_attributions
from .attribution import coalition as C
from .config import load_config
from .g3_report import choose_best
from .groups import GROUP_ORDER, SCHEME_GROUPS
from .inject_models import load_injected_attributions
from .paths import Paths

M = len(GROUP_ORDER)
S = len(SCHEME_GROUPS)
METHODS = ("A", "B", "C", "D", "E")
PARTS = ("correctness", "consistency", "faithfulness", "baselines", "stress", "fairness")
STRATA = ("both", "gbm_only", "glass_box_only", "neither")


class EvalError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- helpers

def clean(obj):
    """JSON-safe copy: NaN / inf -> None, numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=1, default=str), encoding="utf-8")


def eval_dir(paths: Paths) -> Path:
    return paths.processed / "evaluation"


def out_dir(paths: Paths) -> Path:
    return paths.outputs / "phase6"


def stack(series: pl.Series, width: int) -> np.ndarray:
    """List column -> n × width float array; null entries -> NaN rows."""
    vals = series.to_list()
    out = np.full((len(vals), width), np.nan)
    for i, v in enumerate(vals):
        if v is not None:
            out[i] = v
    return out


def group_vectors(att: pl.DataFrame, method: str) -> np.ndarray:
    """n × 8 group vector of a method (NaN rows where the method was not computed)."""
    if method == "C":
        Mx = stack(att["C"], M * M).reshape(-1, M, M)
        return np.diagonal(Mx, axis1=1, axis2=2).copy()
    if method not in att.columns:
        return np.full((att.height, M), np.nan)
    return stack(att[method], M)


def pair_matrix(att: pl.DataFrame, method: str) -> np.ndarray | None:
    """n × 8 × 8 group interaction cells: method C's M, or method D's pair cells (GA²M); else None."""
    col = {"C": "C", "D": "D_pair"}.get(method)
    if col is None or col not in att.columns or att[col].null_count() == att.height:
        return None
    return stack(att[col], M * M).reshape(-1, M, M)


def ranks(a: np.ndarray) -> np.ndarray:
    """1-based rank of each column by decreasing value, ties by column order (n × k)."""
    order = np.argsort(-a, axis=1, kind="stable")
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.arange(1, a.shape[1] + 1)[None, :].repeat(a.shape[0], 0), axis=1)
    return r


def group_mask(names_per_row, n: int) -> np.ndarray:
    """n × 7 boolean mask of scheme groups listed per row."""
    idx = {g: i for i, g in enumerate(SCHEME_GROUPS)}
    out = np.zeros((n, S), dtype=bool)
    for i, names in enumerate(names_per_row):
        for g in names or ():
            if g not in idx:
                raise EvalError(f"answer group {g!r} is not a scheme group")
            out[i, idx[g]] = True
    return out


def best_rank(a7: np.ndarray, r: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Best rank among masked groups with a positive contribution; inf if none."""
    return np.where(mask & (a7 > 0), r, np.inf).min(axis=1)


@lru_cache(maxsize=1)
def _perm_ranks() -> np.ndarray:
    return np.array(list(itertools.permutations(range(1, S + 1))))      # 5040 × 7: rank of each group


@lru_cache(maxsize=None)
def chance(answer_sets: tuple[frozenset, ...], top_n: int = 3) -> dict:
    """Exact chance levels under a uniformly random order of the 7 scheme groups: top-1 / top-n / MRR for the union
    of the answer sets, both-in-top-n (each set has a member in the top n) and the pair hit (share of the 21
    unordered pairs that match one group of each set) when there are two sets."""
    P = _perm_ranks()
    union = sorted(set().union(*answer_sets))
    best = P[:, union].min(axis=1)
    out = {"top1": float(np.mean(best == 1)), "top3": float(np.mean(best <= top_n)), "mrr": float(np.mean(1.0 / best))}
    if len(answer_sets) == 2:
        a, b = (sorted(s) for s in answer_sets)
        out["both_top3"] = float(np.mean((P[:, a].min(axis=1) <= top_n) & (P[:, b].min(axis=1) <= top_n)))
        hits = {frozenset((i, j)) for i in a for j in b if i != j}
        out["pair_hit"] = len(hits) / (S * (S - 1) / 2)
    return out


# ----------------------------------------------------------------------------------------------- correctness (E4)

def scheme_groups(icfg: injection.InjectionConfig, scheme: str) -> list[str]:
    return list(icfg["schemes"][scheme]["groups"])


def correctness_rows(att: pl.DataFrame, method: str, icfg: injection.InjectionConfig, top_n: int = 3) -> pl.DataFrame:
    """Per injected provider: rank of the best-ranked answer group (union of the schemes' groups), top-1, top-n,
    reciprocal rank, both-in-top-n and pair hit for compound injections, shares of positive attribution on the
    answer and expected spill-over groups, and the chance levels of its answer key."""
    a8 = group_vectors(att, method)
    n = att.height
    valid = np.isfinite(a8).all(axis=1)
    a7 = np.where(valid[:, None], a8[:, 1:], 0.0)
    r = ranks(a7)
    schemes = att["schemes"].to_list()
    ans = group_mask(att["answer_groups"].to_list(), n)
    rank_best = best_rank(a7, r, ans)
    pos = np.clip(a7, 0, None)
    tot = pos.sum(axis=1)
    spill_names = [sorted({g for s in sc for g in icfg["schemes"][s]["spill"]} - set(ag or ()))
                   for sc, ag in zip(schemes, att["answer_groups"].to_list())]
    spill = group_mask(spill_names, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        ans_share = np.where(tot > 0, (pos * ans).sum(axis=1) / tot, np.nan)
        spill_share = np.where(tot > 0, (pos * spill).sum(axis=1) / tot, np.nan)
    both = np.full(n, np.nan)
    pair_hit = np.full(n, np.nan)
    PM = pair_matrix(att, method)
    iu = np.triu_indices(S, 1)
    ch = []
    for i, sc in enumerate(schemes):
        sets = tuple(frozenset(SCHEME_GROUPS.index(g) for g in scheme_groups(icfg, s)) for s in sc)
        ch.append(chance(sets, top_n))
        if len(sc) == 2 and valid[i]:
            m1 = group_mask([scheme_groups(icfg, sc[0])], 1)
            m2 = group_mask([scheme_groups(icfg, sc[1])], 1)
            both[i] = float(best_rank(a7[i:i + 1], r[i:i + 1], m1)[0] <= top_n
                            and best_rank(a7[i:i + 1], r[i:i + 1], m2)[0] <= top_n)
            if PM is not None and np.isfinite(PM[i]).all():
                P = PM[i, 1:, 1:] + PM[i, 1:, 1:].T
                vals = P[iu]
                j = int(np.argmax(vals))
                g, h = int(iu[0][j]), int(iu[1][j])
                hit = (m1[0, g] and m2[0, h]) or (m1[0, h] and m2[0, g])
                pair_hit[i] = float(bool(hit) and vals[j] > 0)
    return att.select("npi", "year", "fold", *[c for c in ("draw",) if c in att.columns]).with_columns(
        pl.Series("schemes_key", ["+".join(s) for s in schemes]), att["intensity"], att["pair"], att["distractor"],
        pl.Series("valid", valid),
        pl.Series("rank", np.where(valid & np.isfinite(rank_best), rank_best, np.nan)),
        pl.Series("top1", np.where(valid, (rank_best == 1).astype(float), np.nan)),
        pl.Series("top3", np.where(valid, (rank_best <= top_n).astype(float), np.nan)),
        pl.Series("rr", np.where(valid, np.where(np.isfinite(rank_best), 1.0 / rank_best, 0.0), np.nan)),
        pl.Series("both_top3", both), pl.Series("pair_hit", pair_hit),
        pl.Series("answer_share", np.where(valid, ans_share, np.nan)),
        pl.Series("spill_share", np.where(valid & spill.any(axis=1), spill_share, np.nan)),
        pl.Series("chance_top1", [c["top1"] for c in ch]), pl.Series("chance_top3", [c["top3"] for c in ch]),
        pl.Series("chance_mrr", [c["mrr"] for c in ch]),
        pl.Series("chance_both_top3", [c.get("both_top3", np.nan) for c in ch]),
        pl.Series("chance_pair_hit", [c.get("pair_hit", np.nan) for c in ch])).fill_nan(None)


CORR_COLS = ("top1", "top3", "rr", "both_top3", "pair_hit", "answer_share", "spill_share", "chance_top1",
             "chance_top3", "chance_mrr", "chance_both_top3", "chance_pair_hit")


def summarise(rows: pl.DataFrame, by: list[str], cols=CORR_COLS) -> list[dict]:
    aggs = [pl.len().alias("n")] + [pl.col(c).mean().alias(c) for c in cols if c in rows.columns]
    return rows.group_by(by).agg(aggs).sort(by).to_dicts()


def best_families(paths: Paths) -> dict:
    """Best GBM, best monotone GBM and best glass box of the primary design (validation, D74)."""
    mcfg = models.ModelConfig.load(paths)
    tun = train.load_tuning(paths, "primary")
    return {role: choose_best(tun, list(mcfg[role]))[0] for role in ("gbm", "gbm_mono", "glass_box")}


def method_available(att: pl.DataFrame, method: str) -> bool:
    if method == "C":
        return "C" in att.columns and att["C"].null_count() < att.height
    return method in att.columns and att[method].null_count() < att.height


def run_correctness(paths: Paths, *, families=None, baseline: str | None = None, store: Path | None = None) -> dict:
    """E4 over every injection draw with attributions; store = a directory of per-row output (default evaluation/)."""
    ecfg = load_config("evaluation", paths)
    icfg = injection.InjectionConfig.load(paths)
    cc = ecfg["correctness"]
    base = baseline or cc["baseline"]
    top_n = int(cc["top_n"])
    folds = splits.load_folds(paths).select("npi", "strat_specialty")
    parts = []
    for fam in families or icfg["families"]:
        try:
            att = load_injected_attributions(paths, fam, base)
        except FileNotFoundError:
            continue
        for meth in ecfg["methods"]:
            if method_available(att, meth):
                parts.append(correctness_rows(att, meth, icfg, top_n).with_columns(
                    pl.lit(fam).alias("family"), pl.lit(meth).alias("method")))
    if not parts:
        return {"available": False, "detail": "no injected attributions (run phase5 explain)"}
    rows = pl.concat(parts, how="diagonal_relaxed").join(folds, on="npi", how="left")
    p = (store or eval_dir(paths)) / f"correctness_{base}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(p)
    single = rows.filter(~pl.col("pair"))
    pairs = rows.filter(pl.col("pair"))
    tab_int = cc["table_intensity"]
    return {
        "available": True, "baseline": base, "rows": rows.height,
        "draws": sorted(rows["draw"].unique().to_list()) if "draw" in rows.columns else [],
        "by_scheme_intensity": summarise(single, ["family", "method", "schemes_key", "intensity"]),
        "by_intensity": summarise(single, ["family", "method", "intensity"]),
        "distractor": summarise(single, ["family", "method", "distractor"]),
        "compound": summarise(pairs, ["family", "method", "schemes_key"]),
        "spill": summarise(rows, ["family", "method", "schemes_key"], ("answer_share", "spill_share")),
        "by_specialty": summarise(single.filter(pl.col("intensity") == tab_int),
                                  ["family", "method", "strat_specialty"], ("top1", "top3", "rr", "chance_top1")),
        "by_draw": summarise(single.filter(pl.col("intensity") == tab_int), ["family", "method", "draw"],
                             ("top1", "top3", "rr")) if "draw" in rows.columns else [],
        "table": t2(single.filter(pl.col("intensity") == tab_int), pairs),
    }


def t2(single: pl.DataFrame, pairs: pl.DataFrame) -> list[dict]:
    """EXPERIMENT_PLAN §9 T2 cells per family × method (the report picks the shell's rows)."""
    out = []
    keys = sorted(set(zip(single["family"].to_list(), single["method"].to_list())))
    for fam, meth in keys:
        s = single.filter((pl.col("family") == fam) & (pl.col("method") == meth) & pl.col("valid"))
        d = s.filter(pl.col("distractor"))
        p = pairs.filter((pl.col("family") == fam) & (pl.col("method") == meth) & pl.col("valid"))
        out.append({"family": fam, "method": meth, "n": s.height, "top1": s["top1"].mean(), "top3": s["top3"].mean(),
                    "mrr": s["rr"].mean(), "distractor_n": d.height, "distractor_top1": d["top1"].mean(),
                    "pairs_n": p.height, "both_top3": p["both_top3"].mean(),
                    "pair_hit": p["pair_hit"].mean() if p["pair_hit"].null_count() < p.height else None,
                    "chance_top1": s["chance_top1"].mean(), "chance_top3": s["chance_top3"].mean(),
                    "chance_mrr": s["chance_mrr"].mean(), "chance_both_top3": p["chance_both_top3"].mean(),
                    "chance_pair_hit": p["chance_pair_hit"].mean()})
    return out


# ----------------------------------------------------------------------------------------------- consistency (E3)

def _rankdata(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(a, axis=1)


def consistency_metrics(a: np.ndarray, b: np.ndarray, sign_threshold: float, top_n: int = 3) -> dict[str, np.ndarray]:
    """Per row of two n × 7 attribution matrices: sign agreement over groups with max(|a|, |b|) > threshold, Spearman
    ρ, top-n Jaccard, cosine of the L1-normalised vectors (= cosine of the vectors) and the L1 magnitude ratio."""
    with np.errstate(invalid="ignore", divide="ignore"):
        big = np.maximum(np.abs(a), np.abs(b)) > sign_threshold
        agree = (np.sign(a) == np.sign(b)) & big
        sign = np.where(big.any(axis=1), agree.sum(axis=1) / big.sum(axis=1), np.nan)
        ra, rb = _rankdata(a), _rankdata(b)
        ra, rb = ra - ra.mean(axis=1, keepdims=True), rb - rb.mean(axis=1, keepdims=True)
        den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
        spear = np.where(den > 0, (ra * rb).sum(axis=1) / den, np.nan)
        ta = ranks(a) <= top_n
        tb = ranks(b) <= top_n
        jac = (ta & tb).sum(axis=1) / (ta | tb).sum(axis=1)
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        cos = np.where((na > 0) & (nb > 0), (a * b).sum(axis=1) / (na * nb), np.nan)
        l1a, l1b = np.abs(a).sum(axis=1), np.abs(b).sum(axis=1)
        mag = np.where(l1b > 0, l1a / l1b, np.nan)
    return {"sign_agree": sign, "spearman": spear, "jaccard3": jac, "cosine": cos, "mag_ratio": mag}


CONS_COLS = ("sign_agree", "spearman", "jaccard3", "cosine", "mag_ratio")


def flags(scores: pl.DataFrame, k: float) -> pl.DataFrame:
    """npi, year, flagged: the top ceil(k·N) of each year by margin (ties by the fixed (npi, year) hash, as metrics)."""
    d = scores.select("npi", pl.col("year").cast(pl.Int32), "margin").with_columns(
        pl.Series("_tk", metrics.tie_keys(scores["npi"], scores["year"])))
    d = d.sort(["year", "margin", "_tk"], descending=[False, True, True])
    n = d.group_by("year").len().with_columns(
        (pl.col("len") * k - 1e-9).ceil().cast(pl.Int64).alias("_n")).select("year", "_n")
    return d.join(n, on="year").with_columns(
        (pl.int_range(pl.len()).over("year") < pl.col("_n")).alias("flagged")).select("npi", "year", "flagged")


def derangement(keys: pl.DataFrame, by: list[str], seed: int) -> np.ndarray:
    """Index of a different row in the same stratum for each row (seeded hash order, shift by one); -1 when a
    stratum has a single row."""
    h = splits.npi_keys(keys["npi"], splits._salt(seed, 601)) ^ keys["year"].cast(pl.UInt64).to_numpy()
    k = keys.with_row_index("_r").with_columns(pl.Series("_h", h)).sort([*by, "_h"])
    nxt = k.with_columns(pl.col("_r").shift(-1).over(by).alias("_nx"), pl.col("_r").first().over(by).alias("_f"),
                         pl.len().over(by).alias("_n"))
    nxt = nxt.with_columns(pl.when(pl.col("_n") < 2).then(-1).otherwise(pl.col("_nx").fill_null(pl.col("_f")))
                           .alias("_to"))
    out = np.full(keys.height, -1, dtype=np.int64)
    out[nxt["_r"].to_numpy()] = nxt["_to"].to_numpy()
    return out


def _scores0(paths: Paths, fam: str) -> pl.DataFrame:
    return train.load_scores(paths, "primary", fam).filter(pl.col("seed") == 0).select("npi", "year", "margin")


def run_consistency(paths: Paths, *, gbms=None, glass_boxes=None, store: Path | None = None) -> dict:
    ecfg = load_config("evaluation", paths)
    acfg = load_config("attribution", paths)
    cc = ecfg["consistency"]
    thr, k, base = float(cc["sign_threshold"]), float(cc["flag_k"]), cc["baseline"]
    best = best_families(paths)
    gbms = [g for g in (gbms or acfg["shap_families"])]
    rows, summary = [], []
    for g in gbms:
        try:
            ag = load_attributions(paths, g, base)
        except FileNotFoundError:
            continue
        for gb in glass_boxes or cc["glass_boxes"]:
            try:
                ab = load_attributions(paths, gb, base)
            except FileNotFoundError:
                continue
            j = ag.join(ab.select("npi", "year", pl.col("D").alias("_Db")), on=["npi", "year"], how="inner")
            if j.is_empty():
                continue
            fg = flags(_scores0(paths, g), k).rename({"flagged": "flag_gbm"})
            fb = flags(_scores0(paths, gb), k).rename({"flagged": "flag_gb"})
            j = j.join(fg, on=["npi", "year"], how="left").join(fb, on=["npi", "year"], how="left").with_columns(
                pl.when(pl.col("flag_gbm") & pl.col("flag_gb")).then(pl.lit("both"))
                .when(pl.col("flag_gbm")).then(pl.lit("gbm_only"))
                .when(pl.col("flag_gb")).then(pl.lit("glass_box_only")).otherwise(pl.lit("neither")).alias("stratum"))
            b = stack(j["_Db"], M)[:, 1:]
            perm = derangement(j.select("npi", "year", "stratum"), ["year", "stratum"], int(ecfg["seed"]))
            bp = np.where((perm >= 0)[:, None], b[np.maximum(perm, 0)], np.nan)
            for meth in cc["gbm_methods"]:
                if not method_available(j, meth):
                    continue
                a = group_vectors(j, meth)[:, 1:]
                ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
                mt = consistency_metrics(a, b, thr)
                mp = consistency_metrics(a, bp, thr)
                fr = j.select("npi", "year", "stratum").with_columns(
                    pl.lit(g).alias("gbm"), pl.lit(meth).alias("method"), pl.lit(gb).alias("glass_box"),
                    *[pl.Series(c, np.where(ok, mt[c], np.nan)) for c in CONS_COLS],
                    *[pl.Series(f"perm_{c}", np.where(ok, mp[c], np.nan)) for c in CONS_COLS]).fill_nan(None)
                rows.append(fr)
                for st in ("all", *STRATA):
                    s = fr if st == "all" else fr.filter(pl.col("stratum") == st)
                    ent = {"gbm": g, "method": meth, "glass_box": gb, "stratum": st, "n": s.height,
                           "best_gbm": g == best["gbm"]}
                    for c in CONS_COLS:
                        v = s[c].drop_nulls().to_numpy()
                        vp = s[f"perm_{c}"].drop_nulls().to_numpy()
                        ent[c] = {"median": float(np.median(v)) if v.size else None,
                                  "iqr": [float(np.quantile(v, 0.25)), float(np.quantile(v, 0.75))] if v.size else None,
                                  "perm_median": float(np.median(vp)) if vp.size else None}
                    summary.append(ent)
    if not rows:
        return {"available": False, "detail": "need attributions of a GBM and a glass box (phase4 explain)"}
    p = (store or eval_dir(paths)) / "consistency.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(rows).write_parquet(p)
    return {"available": True, "baseline": base, "flag_k": k, "sign_threshold": thr, "best": best, "summary": summary}


# ----------------------------------------------------------------------------------------------- faithfulness (E6)

def deletion_table(att: pl.DataFrame, methods=METHODS) -> tuple[pl.DataFrame, dict]:
    """Per provider above its baseline (v(N) > v(∅), D86): deletion scores of each method's order, its reverse and
    the random-order mean; method C's order (diagonal of M) is computed here from v. Also the mean normalised
    deletion curve per method (9 points)."""
    V = stack(att["v"], 2 ** M)
    keep = np.isfinite(V).all(axis=1) & (V[:, -1] - V[:, 0] > 1e-12)
    out = {"npi": att["npi"], "year": att["year"]}
    curves = {}
    for meth in methods:
        if not method_available(att, meth):
            continue
        a = group_vectors(att, meth)
        ok = keep & np.isfinite(a).all(axis=1)
        o = C.order_by(np.where(ok[:, None], a, 0.0))
        out[f"{meth}"] = np.where(ok, C.deletion_score(V, o), np.nan)
        out[f"{meth}_reverse"] = np.where(ok, C.deletion_score(V, o[:, ::-1]), np.nan)
        if ok.any():
            f = C.deletion_curve(V[ok], o[ok])
            curves[meth] = ((f - f[:, -1:]) / (f[:, :1] - f[:, -1:])).mean(axis=0).tolist()
    out["random"] = np.where(keep, att["del_random"].to_numpy(), np.nan)
    return pl.DataFrame(out).with_columns(pl.Series("valid", keep)).fill_nan(None), curves


def run_faithfulness(paths: Paths, *, families=None, store: Path | None = None) -> dict:
    ecfg = load_config("evaluation", paths)
    acfg = load_config("attribution", paths)
    es = None
    ep = paths.processed / "attributions" / "explained_set.parquet"
    if ep.is_file():
        es = pl.read_parquet(ep).select("npi", pl.col("year").cast(pl.Int32), "top")
    summary, curves, rows = [], {}, []
    for fam in families or acfg["coalition_families"]:
        for base in ecfg["faithfulness"]["baselines"]:
            try:
                att = load_attributions(paths, fam, base)
            except FileNotFoundError:
                continue
            t, cv = deletion_table(att)
            if es is not None:
                t = t.join(es, on=["npi", "year"], how="left").with_columns(pl.col("top").fill_null(False))
            else:
                t = t.with_columns(pl.lit(False).alias("top"))
            curves[f"{fam}/{base}"] = cv
            rows.append(t.with_columns(pl.lit(fam).alias("family"), pl.lit(base).alias("baseline")))
            for subset in ("all", "top"):
                s = t.filter(pl.col("valid") & (pl.col("top") if subset == "top" else pl.lit(True)))
                for meth in [m for m in METHODS if m in t.columns]:
                    v = s[meth].drop_nulls()
                    rv, rd = s[f"{meth}_reverse"].drop_nulls(), s["random"].drop_nulls()
                    summary.append({"family": fam, "baseline": base, "method": meth, "subset": subset,
                                    "n": v.len(), "deletion": v.mean(), "median": v.median(),
                                    "reverse": rv.mean(), "reverse_median": rv.median(),
                                    "random": rd.mean(), "random_median": rd.median(),
                                    "below_random": (s.filter(pl.col(meth).is_not_null())
                                                     .select((pl.col(meth) < pl.col("random")).mean()).item())})
    if not rows:
        return {"available": False, "detail": "no attributions (run phase4 explain)"}
    p = (store or eval_dir(paths)) / "faithfulness.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(rows, how="diagonal_relaxed").write_parquet(p)
    return {"available": True, "summary": summary, "curves": curves}


# ----------------------------------------------------------------------------------------------- baselines (E8)

def run_baselines(paths: Paths, *, families=None) -> dict:
    """RQ5: for GBM methods A, B, E, how much attribution the context group takes under each baseline, how often the
    top group (all 8) is context, and how far the scheme-group explanations agree between baselines."""
    ecfg = load_config("evaluation", paths)
    acfg = load_config("attribution", paths)
    out = []
    for fam in families or acfg["shap_families"]:
        try:
            peer, glob = load_attributions(paths, fam, "peer"), load_attributions(paths, fam, "global")
        except FileNotFoundError:
            continue
        cols = ["npi", "year", *[m for m in ecfg["baselines"]["methods"] if m in peer.columns]]
        j = peer.select(cols).join(glob.select(cols), on=["npi", "year"], suffix="_g")
        for meth in ecfg["baselines"]["methods"]:
            if meth not in peer.columns or j.is_empty():
                continue
            a, g = stack(j[meth], M), stack(j[f"{meth}_g"], M)
            ok = np.isfinite(a).all(axis=1) & np.isfinite(g).all(axis=1)
            a, g = a[ok], g[ok]
            if not len(a):
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                cs_p = np.abs(a[:, 0]) / np.abs(a).sum(axis=1)
                cs_g = np.abs(g[:, 0]) / np.abs(g).sum(axis=1)
            cm = consistency_metrics(a[:, 1:], g[:, 1:], float(ecfg["consistency"]["sign_threshold"]))
            out.append({"family": fam, "method": meth, "n": int(ok.sum()),
                        "context_share_peer": float(np.nanmean(cs_p)), "context_share_global": float(np.nanmean(cs_g)),
                        "context_top_peer": float(np.mean(np.argmax(a, axis=1) == 0)),
                        "context_top_global": float(np.mean(np.argmax(g, axis=1) == 0)),
                        "top1_agreement": float(np.mean(np.argmax(a[:, 1:], axis=1) == np.argmax(g[:, 1:], axis=1))),
                        "spearman_median": float(np.nanmedian(cm["spearman"])),
                        "cosine_median": float(np.nanmedian(cm["cosine"]))})
    return {"available": bool(out), "summary": out}


# ----------------------------------------------------------------------------------------------- stress (E5)

def run_stress(paths: Paths) -> dict:
    """Use (a): where the LEIE models rank injected test providers among clean test rows (percentile), by family,
    scheme and intensity; share in the top k for the audit-capacity k values."""
    ks = [float(k) for k in load_config("experiment", paths)["evaluation"]["k_fracs"]]
    root = paths.processed / "injection"
    files = sorted(root.glob("draw*/use_a.parquet")) if root.is_dir() else []
    if not files:
        return {"available": False, "detail": "no use (a) results (run phase5 stress)"}
    u = pl.concat([pl.read_parquet(f).with_columns(pl.lit(int(f.parent.name[4:]), dtype=pl.Int32).alias("draw"))
                   for f in files]).with_columns(pl.col("schemes").list.join("+").alias("schemes_key"))
    aggs = [pl.len().alias("n"), pl.col("percentile").median().alias("median_percentile")]
    aggs += [(pl.col("percentile") >= 1 - k).mean().alias(f"in_top_{k}") for k in ks]
    return {"available": True, "draws": sorted(u["draw"].unique().to_list()),
            "summary": u.group_by("family", "schemes_key", "intensity").agg(aggs)
            .sort("family", "schemes_key", "intensity").to_dicts()}


# ----------------------------------------------------------------------------------------------- fairness (E10)

def fairness_table(scores: pl.DataFrame, attrs: pl.DataFrame, k: float, by: str) -> list[dict]:
    """Flag rate, precision and flag rate ÷ positive rate per value of `by`, flags = each test year's top ceil(k·N)
    (pooled over years)."""
    f = flags(scores, k)
    d = scores.select("npi", pl.col("year").cast(pl.Int32), "y").join(f, on=["npi", "year"]).join(
        attrs.select("npi", "year", by), on=["npi", "year"], how="left").with_columns(
        pl.col(by).cast(pl.String).fill_null("unknown"))
    t = d.group_by(by).agg(pl.len().alias("N"), pl.col("y").sum().alias("positives"),
                           pl.col("flagged").sum().alias("flagged"),
                           (pl.col("flagged") & (pl.col("y") == 1)).sum().alias("flagged_positives"))
    t = t.with_columns((pl.col("flagged") / pl.col("N")).alias("flag_rate"),
                       (pl.col("positives") / pl.col("N")).alias("positive_rate"),
                       pl.when(pl.col("flagged") > 0).then(pl.col("flagged_positives") / pl.col("flagged"))
                       .alias("precision"))
    t = t.with_columns(pl.when(pl.col("positive_rate") > 0).then(pl.col("flag_rate") / pl.col("positive_rate"))
                       .alias("flag_to_positive_ratio"))
    return t.sort("N", descending=True).with_columns(pl.lit(by).alias("attribute"), pl.col(by).alias("value"),
                                                      pl.lit(k).alias("k")).drop(by).to_dicts()


def run_fairness(paths: Paths, *, families=None) -> dict:
    ecfg = load_config("evaluation", paths)
    fc = ecfg["fairness"]
    best = best_families(paths)
    fams = [f for f in (families or [best["gbm"], best["glass_box"]]) if f]
    attrs = pl.read_parquet(paths.processed / "cohort.parquet").select(
        "npi", pl.col("year").cast(pl.Int32), *fc["by"])
    out = {}
    for fam in fams:
        try:
            sc = train.load_scores(paths, "primary", fam).filter(pl.col("seed") == 0)
        except train.TrainError:
            continue
        out[fam] = [r for k in fc["k_fracs"] for by in fc["by"] for r in fairness_table(sc, attrs, float(k), by)]
    return {"available": bool(out), "best": best, "families": out}


# ----------------------------------------------------------------------------------------------- orchestration

RUNNERS = {"correctness": run_correctness, "consistency": run_consistency, "faithfulness": run_faithfulness,
           "baselines": run_baselines, "stress": run_stress, "fairness": run_fairness}


def run(paths: Paths, parts=None) -> dict:
    """Run the requested parts (default all), writing outputs/phase6/{part}.json. Refuses before the G2 freeze."""
    train.check_frozen(paths)
    done = {}
    for part in parts or PARTS:
        if part not in RUNNERS:
            raise EvalError(f"unknown part {part!r}; known {PARTS}")
        res = RUNNERS[part](paths)
        write_json(out_dir(paths) / f"{part}.json", res)
        done[part] = bool(res.get("available"))
    return done


def load_part(paths: Paths, part: str) -> dict | None:
    p = out_dir(paths) / f"{part}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
