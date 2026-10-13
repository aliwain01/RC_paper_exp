"""Model matrix (CONTRACTS §3.1): features + labels + folds of one design, in a fixed column order.

`load_design(paths, design)` reads the feature tables of the design's years once (in-cohort rows), joins the
provider-year labels (rows whose label is dropped, `y` null, are left out) and the fold of every NPI. Per fold,
`roles(frame, fold)` assigns the C7 roles and `matrix(frame, columns, rows)` gives the float32 numpy matrix.
Columns: groups.yaml order, then features.yaml order within a group, one-hot columns last and sorted. A trained
model's column list is saved and re-applied when scoring other years (one-hot vocabularies differ by year,
DESIGN §17 D42): absent one-hot columns are 0; any other missing column fails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from . import features, groups, splits
from .labels import LabelConfig
from .paths import Paths

META = ("npi", "year", "peer_cell", "fold", "y", "label_horizon_complete")


class MatrixError(RuntimeError):
    pass


@dataclass
class DesignFrame:
    design: str
    label_tier: str
    W: int
    years: tuple[int, ...]
    frame: pl.DataFrame          # META + model columns, one row per labelled in-cohort provider-year
    columns: list[str]           # model columns in fixed order
    groups: dict[str, str]       # column -> group
    folds: pl.DataFrame


def column_order(columns, fcfg: features.FeatureConfig | None = None, paths: Paths | None = None) -> list[str]:
    """Model columns ordered by group (groups.yaml), then features.yaml order, one-hots last and sorted."""
    fcfg = fcfg or features.FeatureConfig.load(paths)
    cols = features.model_columns(columns)
    grp = groups.feature_groups(cols, paths=paths)
    rank = {s.id: i for i, s in enumerate(fcfg.specs)}
    gi = {g: i for i, g in enumerate(groups.GROUP_ORDER)}

    def key(c: str):
        base = c.split("__")[0]
        return (gi[grp[c]], "__" in c, rank.get(base, len(rank)), c)

    return sorted(cols, key=key)


def column_kinds(columns, fcfg: features.FeatureConfig | None = None, paths: Paths | None = None) -> dict[str, str]:
    """features.yaml kind of every model column (one-hot columns -> one_hot)."""
    fcfg = fcfg or features.FeatureConfig.load(paths)
    out = {}
    for c in columns:
        base = c.split("__")[0]
        try:
            out[c] = "one_hot" if "__" in c else fcfg.spec(base).kind
        except Exception as exc:  # noqa: BLE001 - FeatureError for an unknown id
            raise MatrixError(f"column {c!r} is not a features.yaml feature") from exc
    return out


def design_years(design: str, scfg: splits.SplitConfig) -> tuple[int, ...]:
    d = scfg.design(design)
    return tuple(sorted({*d.train_years, *d.test_years}))


def load_design(paths: Paths, design: str, *, label_tier: str | None = None, W: int | None = None,
                years=None) -> DesignFrame:
    lcfg = LabelConfig.load(paths)
    scfg = splits.SplitConfig.load(paths)
    tier, W = label_tier or lcfg.primary_tier, int(W or lcfg.W)
    years = tuple(sorted(int(y) for y in years)) if years is not None else design_years(design, scfg)
    lab_path = paths.processed / f"labels_{tier}_W{W}.parquet"
    if not lab_path.is_file():
        raise MatrixError(f"{paths.rel(lab_path)} missing (run phase1 labels)")
    feats = features.load_features(paths, years)
    labels = (pl.scan_parquet(lab_path).filter(pl.col("year").is_in(list(years)) & pl.col("y").is_not_null())
              .select("npi", pl.col("year").cast(pl.Int32), pl.col("y").cast(pl.Int8), "label_horizon_complete")
              .collect())
    folds = splits.load_folds(paths)
    df = (feats.with_columns(pl.col("year").cast(pl.Int32))
          .join(labels, on=["npi", "year"], how="inner")
          .join(folds.select("npi", "fold"), on="npi", how="left"))
    if df["fold"].null_count():
        raise MatrixError(f"{df['fold'].null_count()} rows have no fold (folds.parquet is older than the features?)")
    cols = column_order(feats.columns, paths=paths)
    nulls = [c for c in cols if df[c].null_count()]
    if nulls:
        raise MatrixError(f"model columns with nulls (features must be complete): {nulls[:10]}")
    df = df.select(*META, *cols).sort(["year", "npi"])
    return DesignFrame(design, tier, W, years, df, cols, groups.feature_groups(cols, paths=paths), folds)


def roles(df: DesignFrame, fold: int, scfg: splits.SplitConfig, *, design: str | None = None,
          exclude_years=()) -> pl.Series:
    """C7 role of every row of `df.frame` for fold `fold` (Enum)."""
    keys = df.frame.select("npi", "year")
    return splits.assign_roles(keys, df.folds, fold, design or df.design, scfg, check=False,
                               exclude_years=exclude_years)["role"]


def matrix(frame: pl.DataFrame, columns: list[str], rows: np.ndarray | None = None) -> np.ndarray:
    """float32 matrix of `columns` (absent one-hot columns = 0; any other absent column fails)."""
    sub = frame if rows is None else frame[rows]
    missing = [c for c in columns if c not in sub.columns]
    hard = [c for c in missing if "__" not in c]
    if hard:
        raise MatrixError(f"columns the model needs are missing: {hard}")
    if missing:
        sub = sub.with_columns(pl.lit(0, dtype=pl.Int8).alias(c) for c in missing)
    X = sub.select(pl.col(c).cast(pl.Float32) for c in columns).to_numpy()
    if not np.isfinite(X).all():
        raise MatrixError("model matrix has non-finite values")
    return np.ascontiguousarray(X)
