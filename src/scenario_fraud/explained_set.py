"""Explained set (DESIGN §9.5; EXPERIMENT_PLAN §4; CONTRACTS §4).

Per test year of the attribution design: the top `top_frac` of provider-years by the out-of-fold margin of the best
GBM (chosen on validation, DESIGN §17 D74; reference seed), ∪ a random `random_n` stratified by specialty
(largest-remainder allocation, seeded hash order), ∪ every injected test provider (Phase 5; `injected` frame with
npi, year). Written to data/processed/attributions/explained_set.parquet.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from . import models, splits, train
from .config import load_config
from .g3_report import choose_best
from .paths import Paths

_RANDOM = 103


def explained_path(paths: Paths) -> "Paths":
    return paths.processed / "attributions" / "explained_set.parquet"


def best_gbm(paths: Paths, design: str) -> str:
    mcfg = models.ModelConfig.load(paths)
    best, _ = choose_best(train.load_tuning(paths, design), list(mcfg["gbm"]))
    if best is None:
        raise ValueError(f"no trained GBM ({mcfg['gbm']}) for design {design}")
    return best


def build_explained_set(paths: Paths, injected: pl.DataFrame | None = None) -> pl.DataFrame:
    a = load_config("attribution", paths)
    e = a["explained"]
    design, ref = e["design"], int(a["reference_seed"])
    gbm = best_gbm(paths, design)
    s = train.load_scores(paths, design, gbm).filter(pl.col("seed") == ref).select("npi", "year", "fold", "margin")
    folds = splits.load_folds(paths).select("npi", "strat_specialty")
    s = s.join(folds, on="npi", how="left")
    parts = []
    for (yr,), g in s.sort("year").group_by("year", maintain_order=True):
        n_top = math.ceil(float(e["top_frac"]) * g.height)
        top = g.sort(["margin", "npi"], descending=[True, False]).head(n_top).select("npi")
        h = pl.Series("_h", splits.npi_keys(g["npi"], splits._salt(int(a["seed"]), _RANDOM, int(yr))))
        gg = g.with_columns(h)
        n = min(int(e["random_n"]), g.height)
        sizes = gg.group_by("strat_specialty").len().sort("strat_specialty")
        quota = sizes["len"].to_numpy() * n / g.height
        base = np.floor(quota).astype(int)
        base[np.lexsort((np.arange(len(quota)), -(quota - base)))[: n - base.sum()]] += 1
        alloc = pl.DataFrame({"strat_specialty": sizes["strat_specialty"], "_q": base})
        rnd = (gg.join(alloc, on="strat_specialty").sort(["strat_specialty", "_h"])
               .filter(pl.int_range(pl.len()).over("strat_specialty") < pl.col("_q")).select("npi"))
        parts.append(g.select("npi", "year", "fold").with_columns(
            pl.col("npi").is_in(top["npi"].implode()).alias("top"),
            pl.col("npi").is_in(rnd["npi"].implode()).alias("random")))
    out = pl.concat(parts).with_columns(pl.lit(False).alias("injected"))
    if injected is not None and injected.height:
        inj = injected.select("npi", pl.col("year").cast(pl.Int32), "fold").with_columns(
            pl.lit(False).alias("top"), pl.lit(False).alias("random"), pl.lit(True).alias("injected"))
        out = pl.concat([out, inj], how="vertical_relaxed")
    out = (out.filter(pl.col("top") | pl.col("random") | pl.col("injected"))
           .with_columns(pl.lit(gbm).alias("score_family")).sort(["year", "npi"]))
    p = explained_path(paths)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(p)
    return out


def load_explained_set(paths: Paths) -> pl.DataFrame:
    p = explained_path(paths)
    if not p.is_file():
        raise FileNotFoundError("explained set missing: run `phase4 explained-set` first")
    return pl.read_parquet(p)
