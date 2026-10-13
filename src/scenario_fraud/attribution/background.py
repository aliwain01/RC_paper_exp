"""C9 background sampler (DESIGN §9.1, §9.3; EXPERIMENT_PLAN §4; CONTRACTS §4).

Peer baseline: for an explained provider of fold k in year t, up to B training-fold providers (NPIs not in fold k)
of the same year and peer cell. If the peer cell has fewer than `min_background` such providers, the next level up
is used: specialty × entity → entity → the whole year (DESIGN §17 D82). Global baseline: B_global training-fold
providers of the year, stratified by specialty. Draws are deterministic: candidates are ordered by a seeded hash
of the NPI, so every provider sharing a background key gets the same background.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..config import load_config
from ..paths import Paths
from ..splits import _salt, npi_keys

LEVELS = ("peer_cell", "cell_spec_ent", "cell_ent", "year")
_PEER, _GLOBAL = 101, 102


@dataclass(frozen=True)
class BackgroundConfig:
    B: int
    min_background: int
    B_global: int
    seed: int

    @classmethod
    def load(cls, paths: Paths) -> "BackgroundConfig":
        a = load_config("attribution", paths)
        b = a["background"]
        cfg = cls(int(b["B"]), int(b["min_background"]), int(b["B_global"]), int(a["seed"]))
        if not 1 <= cfg.min_background <= cfg.B or cfg.B_global < 1:
            raise ValueError("attribution.yaml background: need 1 <= min_background <= B and B_global >= 1")
        return cfg


def _with_hash(df: pl.DataFrame, salt) -> pl.DataFrame:
    return df.with_columns(pl.Series("_h", npi_keys(df["npi"], salt)))


def peer_backgrounds(explained: pl.DataFrame, candidates: pl.DataFrame, cfg: BackgroundConfig, fold: int,
                     year: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """explained and candidates: npi, peer_cell, cell_spec_ent, cell_ent (one year; candidates = training-fold
    providers). Returns (assignment: npi, bg_key, bg_level, bg_n; members: bg_key, npi)."""
    cand = candidates.with_columns(pl.lit("all").alias("year_cell"))
    exp = explained.with_columns(pl.lit("all").alias("year_cell"))
    cols = {"peer_cell": "peer_cell", "cell_spec_ent": "cell_spec_ent", "cell_ent": "cell_ent", "year": "year_cell"}
    counts = {lv: cand.group_by(c).len().rename({c: "_v", "len": "_n"}) for lv, c in cols.items()}
    out = exp.select("npi").with_columns(pl.lit(None, dtype=pl.String).alias("bg_level"),
                                         pl.lit(None, dtype=pl.String).alias("_v"))
    for lv, c in cols.items():
        hit = (exp.select("npi", pl.col(c).alias("_v")).join(counts[lv], on="_v", how="left")
               .filter(pl.col("_n") >= cfg.min_background).select("npi", pl.col("_v").alias("_v2")))
        out = (out.join(hit, on="npi", how="left", maintain_order="left")
               .with_columns(pl.when(pl.col("bg_level").is_null() & pl.col("_v2").is_not_null()).then(pl.lit(lv))
                             .otherwise(pl.col("bg_level")).alias("bg_level"),
                             pl.when(pl.col("bg_level").is_null() & pl.col("_v2").is_not_null()).then(pl.col("_v2"))
                             .otherwise(pl.col("_v")).alias("_v"))
               .drop("_v2"))
    if out["bg_level"].null_count():
        raise ValueError(f"{year} fold {fold}: fewer than min_background training-fold providers in the whole year")
    out = out.with_columns((pl.col("bg_level") + "=" + pl.col("_v")).alias("bg_key")).drop("_v")
    ranked = _with_hash(cand, _salt(cfg.seed, _PEER, fold, year))
    parts = []
    for lv, c in cols.items():
        keys = out.filter(pl.col("bg_level") == lv)["bg_key"].unique()
        if keys.len() == 0:
            continue
        m = (ranked.with_columns((pl.lit(lv + "=") + pl.col(c)).alias("bg_key"))
             .filter(pl.col("bg_key").is_in(keys.implode()))
             .sort(["bg_key", "_h"]).group_by("bg_key", maintain_order=True).head(cfg.B)
             .select("bg_key", "npi"))
        parts.append(m)
    members = pl.concat(parts)
    sizes = members.group_by("bg_key").len().rename({"len": "bg_n"})
    return out.join(sizes, on="bg_key", how="left").select("npi", "bg_key", "bg_level", "bg_n"), members


def global_background(candidates: pl.DataFrame, cfg: BackgroundConfig, fold: int, year: int) -> pl.Series:
    """B_global training-fold NPIs of the year, stratified by `strat_specialty` (largest-remainder allocation)."""
    c = _with_hash(candidates.select("npi", "strat_specialty"), _salt(cfg.seed, _GLOBAL, fold, year))
    n = min(cfg.B_global, c.height)
    sizes = c.group_by("strat_specialty").len().sort("strat_specialty")
    quota = sizes["len"].to_numpy() * n / c.height
    base = np.floor(quota).astype(int)
    rem = n - base.sum()
    order = np.lexsort((np.arange(len(quota)), -(quota - base)))
    base[order[:rem]] += 1
    alloc = pl.DataFrame({"strat_specialty": sizes["strat_specialty"], "_q": base})
    return (c.join(alloc, on="strat_specialty").sort(["strat_specialty", "_h"])
            .filter(pl.int_range(pl.len()).over("strat_specialty") < pl.col("_q"))["npi"].sort())
