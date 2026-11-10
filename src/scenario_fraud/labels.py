"""C3 LEIE labeller (DESIGN §5.3-5.4, §7.3).

LEIE current file + monthly supplements + Internet Archive snapshots -> one row per exclusion event
(data/processed/exclusions.parquet) -> first exclusion per NPI and tier -> provider-year labels.

The label rule, for first tier exclusion in year yE and window W:
  positive   if t < yE <= t + W
  dropped    if t >= yE            (barred or partial year)
  dropped    if t <  yE - W        (onset unknown)
  negative   if never excluded under any authority (unlabelled negatives, PU setting)
Providers excluded only under other authorities follow `other_excluded_policy` (default: drop).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from .acquire import Entry, load_manifest
from .config import load_config
from .paths import Paths

LEIE_KINDS = ("current", "supplement_excl", "supplement_rein", "snapshot")
NAME_FIELDS = ("lastname", "firstname", "midname", "busname", "dob")
_STRIP = str.maketrans("", "", " \t()§.")


class LabelError(RuntimeError):
    pass


def normalise_excltype(raw: str | None) -> str:
    return (raw or "").translate(_STRIP)


@dataclass
class LabelConfig:
    W: int
    tiers: dict[str, list[str]]
    other_excluded_policy: str
    layout: dict
    excltype_map: dict[str, str]
    mandatory: list[str]
    snapshot_date: date | None
    primary_tier: str
    W_sensitivity: list[int]
    layout_verified: bool
    excltype_verified: bool

    @classmethod
    def load(cls, paths: Paths) -> "LabelConfig":
        c = load_config("labels", paths)
        if c["other_excluded_policy"] not in ("drop", "negative"):
            raise LabelError("other_excluded_policy must be drop or negative")
        return cls(
            W=int(c["W"]),
            tiers={k: list(v) for k, v in c["tiers"].items()},
            other_excluded_policy=c["other_excluded_policy"],
            layout=c["leie_layout"],
            excltype_map={normalise_excltype(k): v for k, v in c["excltype_map"]["map"].items()},
            mandatory=list(c.get("mandatory", [])),
            snapshot_date=date.fromisoformat(str(c["snapshot_date"])) if c.get("snapshot_date") else None,
            primary_tier=c["primary_tier"],
            W_sensitivity=[int(w) for w in c.get("W_sensitivity", [])],
            layout_verified=bool(c["leie_layout"].get("verified")),
            excltype_verified=bool(c["excltype_map"].get("verified")),
        )


# ----------------------------------------------------------------------------------------------- reading

def read_leie_file(path: Path, cfg: LabelConfig, kind: str = "current", as_of: str | None = None) -> pl.DataFrame:
    """Read one LEIE-layout CSV into the standard columns. Fails on missing key columns, unparseable dates
    and unknown EXCLTYPE values (with counts), so an unverified layout cannot pass silently."""
    cols = cfg.layout["columns"]
    raw = pl.read_csv(path, infer_schema=False, encoding="utf8-lossy", null_values=[])
    raw = raw.rename({c: c.strip() for c in raw.columns})
    missing = [cols[k] for k in ("npi", "excltype", "excldate", "reindate") if cols[k] not in raw.columns]
    if missing:
        raise LabelError(f"{path.name}: LEIE key columns missing {missing}; header is {raw.columns}")

    def col(key: str) -> pl.Expr:
        name = cols.get(key)
        if name and name in raw.columns:
            return pl.col(name).str.strip_chars()
        return pl.lit(None, dtype=pl.String)

    df = raw.select(
        col("npi").alias("npi_raw"),
        col("excltype").alias("excltype_raw"),
        col("excldate").alias("excldate_raw"),
        col("reindate").alias("reindate_raw"),
        *[col(k).alias(k) for k in NAME_FIELDS],
        col("state").alias("state"),
        col("general").alias("general"),
        col("specialty").alias("specialty"),
    )

    # EXCLTYPE -> canonical authority; unknown codes stop the run.
    df = df.with_columns(
        pl.col("excltype_raw").map_elements(normalise_excltype, return_dtype=pl.String).alias("excltype_key")
    )
    unknown = df.filter(~pl.col("excltype_key").is_in(list(cfg.excltype_map))).group_by("excltype_raw").len()
    if unknown.height:
        listing = ", ".join(f"{r['excltype_raw']!r}×{r['len']}" for r in unknown.sort("len", descending=True).to_dicts())
        raise LabelError(f"{path.name}: EXCLTYPE values not in configs/labels.yaml excltype_map: {listing}")
    df = df.with_columns(
        pl.col("excltype_key").replace_strict(cfg.excltype_map, return_dtype=pl.String).alias("authority")
    )

    # Dates.
    fmt, empties = cfg.layout["date_format"], [str(v) for v in cfg.layout["empty_dates"]]

    def parse(c: str) -> pl.Expr:
        return pl.when(pl.col(c).is_null() | pl.col(c).is_in(empties)).then(None).otherwise(
            pl.col(c).str.strptime(pl.Date, fmt, strict=False)
        )

    df = df.with_columns(parse("excldate_raw").alias("excl_date"), parse("reindate_raw").alias("rein_date"))
    bad = df.filter(
        (pl.col("excl_date").is_null())
        | (pl.col("rein_date").is_null() & pl.col("reindate_raw").is_not_null() & ~pl.col("reindate_raw").is_in(empties))
    )
    if bad.height:
        ex = bad.select("excldate_raw", "reindate_raw").head(5).to_dicts()
        raise LabelError(f"{path.name}: {bad.height} rows with missing/unparseable dates, e.g. {ex}")

    # NPI: blank / all-zero = none; anything that is not 10 digits is reported, not matched.
    empty_npi = [str(v) for v in cfg.layout["empty_npi"]]
    df = df.with_columns(
        pl.when(pl.col("npi_raw").is_null() | pl.col("npi_raw").is_in(empty_npi))
        .then(pl.lit("none"))
        .when(pl.col("npi_raw").str.contains(r"^\d{10}$"))
        .then(pl.lit("valid"))
        .otherwise(pl.lit("invalid"))
        .alias("npi_status")
    ).with_columns(
        pl.when(pl.col("npi_status") == "valid").then(pl.col("npi_raw")).otherwise(None).alias("npi")
    )
    return df.with_columns(
        pl.lit(kind).alias("file_kind"),
        pl.lit(as_of).alias("file_as_of"),
        pl.lit(path.name).alias("file_name"),
    ).drop("excltype_key")


def leie_entries(paths: Paths) -> list[Entry]:
    return [e for e in load_manifest(paths) if e.source == "leie" and e.kind in LEIE_KINDS]


def snapshot_date(paths: Paths, cfg: LabelConfig) -> date:
    if cfg.snapshot_date:
        return cfg.snapshot_date
    current = [e for e in leie_entries(paths) if e.kind == "current"]
    if not current:
        raise LabelError("no `current` LEIE file registered and labels.yaml snapshot_date is null")
    newest = max(current, key=lambda e: e.as_of or e.download_date)
    return date.fromisoformat((newest.as_of or newest.download_date)[:10])


# ----------------------------------------------------------------------------------------------- union

def union_exclusions(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """One row per exclusion event across all LEIE sources.
    Key: (npi, authority, excl_date) when an NPI exists, else (name fields + dob, authority, excl_date)."""
    if not frames:
        raise LabelError("no LEIE files to union")
    allr = pl.concat(frames, how="vertical_relaxed").with_columns(
        pl.when(pl.col("npi").is_not_null())
        .then(pl.concat_str([pl.lit("npi:"), pl.col("npi")]))
        .otherwise(
            pl.concat_str(
                [pl.lit("name:"), *[pl.col(f).fill_null("").str.to_uppercase() for f in NAME_FIELDS]], separator="|"
            )
        )
        .alias("party_key")
    )
    key = ["party_key", "authority", "excl_date"]
    return (
        allr.sort(["file_as_of", "file_kind"], nulls_last=True)
        .group_by(key, maintain_order=True)
        .agg(
            pl.col("npi").drop_nulls().first(),
            pl.col("npi_status").first(),
            pl.col("excltype_raw").first(),
            pl.col("rein_date").drop_nulls().max(),
            *[pl.col(f).drop_nulls().first() for f in NAME_FIELDS],
            pl.col("state").drop_nulls().first(),
            pl.col("general").drop_nulls().first(),
            pl.col("specialty").drop_nulls().first(),
            (pl.col("file_kind") == "current").any().alias("in_current"),
            pl.col("file_kind").unique().sort().str.join(",").alias("seen_in"),
            pl.col("file_as_of").min().alias("first_seen_as_of"),
            pl.col("file_as_of").max().alias("last_seen_as_of"),
        )
        .with_columns(pl.col("excl_date").dt.year().alias("excl_year"))
        .sort(key)
    )


def build_exclusions(paths: Paths, cfg: LabelConfig | None = None) -> pl.DataFrame:
    cfg = cfg or LabelConfig.load(paths)
    entries = leie_entries(paths)
    if not any(e.kind == "current" for e in entries):
        raise LabelError("no `current` LEIE file registered (data/raw/leie/<year>/, kind=current)")
    frames = [read_leie_file(paths.abs(e.path), cfg, e.kind, e.as_of or e.download_date) for e in entries]
    exc = union_exclusions(frames)
    out = paths.processed / "exclusions.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    exc.write_parquet(out)
    return exc


# ----------------------------------------------------------------------------------------------- label rule

def first_exclusions(exclusions: pl.DataFrame, cfg: LabelConfig) -> pl.DataFrame:
    """Per NPI: first exclusion under each tier, and first exclusion under any authority."""
    exc = exclusions.filter(pl.col("npi").is_not_null())
    parts = []
    for tier, auths in cfg.tiers.items():
        parts.append(
            exc.filter(pl.col("authority").is_in(auths))
            .sort(["npi", "excl_date", "authority"])
            .group_by("npi", maintain_order=True)
            .agg(pl.col("excl_date").first(), pl.col("authority").first())
            .with_columns(pl.lit(tier).alias("tier"))
        )
    parts.append(
        exc.sort(["npi", "excl_date", "authority"])
        .group_by("npi", maintain_order=True)
        .agg(pl.col("excl_date").first(), pl.col("authority").first())
        .with_columns(pl.lit("any").alias("tier"))
    )
    return pl.concat(parts).with_columns(pl.col("excl_date").dt.year().alias("excl_year"))


def apply_label_rule(
    provider_years: pl.DataFrame,
    firsts: pl.DataFrame,
    tier: str,
    W: int,
    snapshot: date,
    other_excluded_policy: str = "drop",
) -> pl.DataFrame:
    """provider_years has columns npi (str) and year (int). Returns one row per input row with
    y (1/0, null when dropped), drop_reason, label_tier, excl_type, excl_year, label_horizon_complete,
    label_horizon_years (DESIGN §7.3)."""
    if other_excluded_policy not in ("drop", "negative"):
        raise LabelError("other_excluded_policy must be drop or negative")
    t_first = firsts.filter(pl.col("tier") == tier).select(
        "npi", pl.col("excl_year").alias("yE"), pl.col("authority").alias("excl_type")
    )
    a_first = firsts.filter(pl.col("tier") == "any").select("npi", pl.col("excl_year").alias("yA"))
    df = (
        provider_years.select("npi", pl.col("year").cast(pl.Int32))
        .join(t_first, on="npi", how="left")
        .join(a_first, on="npi", how="left")
    )
    t, yE, yA = pl.col("year"), pl.col("yE"), pl.col("yA")
    tier_pos = yE.is_not_null()
    other = yE.is_null() & yA.is_not_null()
    reason = (
        pl.when(tier_pos & (t >= yE)).then(pl.lit("at_or_after_exclusion"))
        .when(tier_pos & (t < yE - W)).then(pl.lit("onset_unknown"))
    )
    if other_excluded_policy == "drop":
        reason = reason.when(other).then(pl.lit("other_excluded"))
    else:  # negative: other-excluded rows are negatives until the year of their first exclusion
        reason = reason.when(other & (t >= yA)).then(pl.lit("at_or_after_exclusion"))
    reason = reason.otherwise(None)
    y = pl.when(reason.is_not_null()).then(None).when(tier_pos).then(1).otherwise(0).cast(pl.Int8)

    jan1_next = pl.date(t + 1, 1, 1)
    years_obs = ((pl.lit(snapshot) - jan1_next).dt.total_days() / 365.25).clip(0, W)
    complete = pl.lit(snapshot) >= pl.date(t + W, 12, 31)
    return df.with_columns(
        y.alias("y"),
        reason.alias("drop_reason"),
        pl.lit(tier).alias("label_tier"),
        pl.col("yE").alias("excl_year"),
        complete.alias("label_horizon_complete"),
        years_obs.round(2).alias("label_horizon_years"),
        pl.lit(W).cast(pl.Int8).alias("W"),
    ).select(
        "npi", "year", "y", "drop_reason", "label_tier", "excl_type", "excl_year",
        "label_horizon_complete", "label_horizon_years", "W",
    )


def provider_years_from_interim(paths: Paths) -> pl.DataFrame | None:
    """Every (npi, year) in the harmonised by-Provider files (pre-cohort: the C4 volume rule is Phase 2)."""
    files = sorted((paths.interim / "cms_prov").glob("*.parquet"))
    if not files:
        return None
    return pl.concat([pl.read_parquet(f, columns=["npi", "year"]) for f in files]).unique().sort(["npi", "year"])


def run_labels(paths: Paths, cfg: LabelConfig | None = None) -> dict:
    """C3 end to end: exclusions, first exclusions, and (if by-Provider years are harmonised) provider-year
    labels for every tier at the primary W, plus the W sensitivity for the primary tier."""
    cfg = cfg or LabelConfig.load(paths)
    exc = build_exclusions(paths, cfg)
    firsts = first_exclusions(exc, cfg)
    firsts.write_parquet(paths.processed / "first_exclusions.parquet")
    snap = snapshot_date(paths, cfg)
    written = []
    py = provider_years_from_interim(paths)
    if py is not None:
        runs = [(t, cfg.W) for t in cfg.tiers] + [(cfg.primary_tier, w) for w in cfg.W_sensitivity]
        for tier, W in runs:
            out = paths.processed / f"labels_{tier}_W{W}.parquet"
            apply_label_rule(py, firsts, tier, W, snap, cfg.other_excluded_policy).write_parquet(out)
            written.append(paths.rel(out))
    return {
        "snapshot_date": snap.isoformat(),
        "n_exclusion_events": exc.height,
        "n_npis_excluded": firsts.filter(pl.col("tier") == "any").height,
        "provider_years": None if py is None else py.height,
        "label_files": written,
        "layout_verified": cfg.layout_verified,
        "excltype_verified": cfg.excltype_verified,
    }


# ----------------------------------------------------------------------------------------------- coverage

def _month_range(a: date, b: date) -> list[tuple[int, int]]:
    out, y, m = [], a.year, a.month
    while (y, m) <= (b.year, b.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def coverage_gaps(paths: Paths, window_start: date, window_end: date) -> dict:
    """Intervals in which an exclusion could start and be reinstated without any source seeing it.
    A reinstated exclusion is recovered if a full-file snapshot (or the current file) was taken while it
    was in effect, or if its reinstatement month has a reinstatement supplement (DESIGN §5.4)."""
    entries = leie_entries(paths)
    snaps = sorted({
        date.fromisoformat((e.as_of or e.download_date)[:10])
        for e in entries if e.kind in ("snapshot", "current")
    })
    rein_months = sorted({tuple(map(int, (e.as_of or "")[:7].split("-"))) for e in entries
                          if e.kind == "supplement_rein" and e.as_of})
    points = [window_start] + [d for d in snaps if window_start < d < window_end] + [window_end]
    intervals = []
    for a, b in zip(points[:-1], points[1:]):
        months = [ym for ym in _month_range(a, b) if ym not in set(rein_months)]
        if months and (b - a).days > 0:
            intervals.append({
                "from": a.isoformat(), "to": b.isoformat(), "days": (b - a).days,
                "uncovered_months": len(months), "bounded_by_snapshot": b in snaps,
            })
    return {
        "window": [window_start.isoformat(), window_end.isoformat()],
        "snapshot_dates": [d.isoformat() for d in snaps],
        "rein_supplement_months": [f"{y:04d}-{m:02d}" for y, m in rein_months],
        "uncovered_intervals": intervals,
        "uncovered_days_total": sum(i["days"] for i in intervals),
    }


def reinstatement_loss_profile(exclusions: pl.DataFrame, cfg: LabelConfig, gap_days: int | None) -> pl.DataFrame:
    """Per authority, from records with a reinstatement date: count, duration quantiles, and the share
    short enough to fall entirely inside the longest uncovered interval (an upper-bound loss indicator)."""
    r = exclusions.filter(pl.col("rein_date").is_not_null()).with_columns(
        (pl.col("rein_date") - pl.col("excl_date")).dt.total_days().alias("dur_days")
    )
    agg = [
        pl.len().alias("n_reinstated"),
        pl.col("dur_days").median().alias("median_days"),
        pl.col("dur_days").quantile(0.1).alias("p10_days"),
    ]
    if gap_days:
        agg.append((pl.col("dur_days") < gap_days).mean().alias("share_shorter_than_longest_gap"))
    return r.group_by("authority").agg(agg).with_columns(
        pl.col("authority").is_in(cfg.mandatory).alias("mandatory"),
        *[pl.col("authority").is_in(a).alias(f"in_{t}") for t, a in cfg.tiers.items()],
    ).sort("n_reinstated", descending=True)
