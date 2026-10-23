"""C5 Feature builder (DESIGN §7.4-7.5; FEATURE_CATALOG v1.1; CONTRACTS §2.2).

harmonised cms_ps + cms_prov + dme_s + dme_p (one year) + cohort peer cells
    -> data/processed/features/{year}.parquet (+ {year}.meta.json)
    -> data/processed/peer/{year}/{codes,stats,pm_models}.parquet

Four steps, so injected providers (C6) can be featurised against peer statistics fitted on real data only:

    refs = fit_peer_reference(frames, cohort, year)    # per peer cell x HCPCS: pooled shares, rare flags, allowed medians
    raw  = raw_statistics(frames, cohort, refs, year)  # raw statistic r per feature (+ k, n of rates), any provider
    ps   = fit_peer_stats(raw, cohort, year)           # medians, scales, beta-binomial priors, PM2/PM3 regressions
    feat = transform(raw, cohort, ps, refs, year)      # model features (FEATURE_CATALOG IDs) + raw__{ID}

The fit_* steps use reference providers only: cohort rows with in_cohort true and, if the cohort has a boolean
`injected` column, injected false. Nothing here reads labels (DESIGN §7.4, D14). raw_statistics and transform
featurise every cohort row they are given; each row needs a peer cell the fit has seen.

Transform (DESIGN §7.4): robust z = (v - median) / scale within the peer cell, clipped to +-z_clip; v = r, or for
count-based rates the beta-binomial shrunk rate (k + alpha*m) / (n + alpha). A peer cell's members are all reference
providers whose cell at the peer cell's level is that cell (an entity-type cell holds every provider of that entity
type). A feature's peer statistics come from the most specific level (peer cell -> spec x ent -> ent -> national,
same year) with >= min_defined defined reference values; scale = 1.4826 MAD, else IQR/1.349, else SD, else the next
level's, else 1. Reference-dependent features (REF_DEPENDENT: SM1, SM2, CH4, PM2, PM3, whose r is measured against
the peer cell's code reference or regression) take their statistics from the peer cell itself, over every member's
r recomputed against that same reference (no hierarchy fallback; scale fallback mad, iqr, std, unit). An undefined
r gets z = 0 (the peer median); structural zeros have indicators UP0, DI0, PM4_missing; a suppressed DME patient count has DI2_missing. No DME-P row: DI0 = 1, raw
DI1-DI4 = 0 (FEATURE_CATALOG DI0), raw DC1-DC2 undefined, and z = 0 for DI1, DI3, DI4, DC1, DC2 (DI0 carries the
information). Parameters, code sets and every choice the catalog leaves open: configs/features.yaml.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import polars as pl

from . import groups as groups_mod
from . import provenance
from .config import load_config
from .harmonise import DTYPES, load_schema
from .paths import Paths

LEVELS = ("spec_ent_div", "spec_ent", "ent", "national")
PEER_LEVELS = LEVELS[:3]
CELL_COL = {"spec_ent_div": "cell_spec_ent_div", "spec_ent": "cell_spec_ent", "ent": "cell_ent",
            "national": "cell_national"}
KEYS = ("npi", "year", "peer_cell")
RAW = "raw__"
NA = "na"   # one-hot token for a null category
KINDS = ("robust_z", "indicator", "context_raw", "one_hot")
FIT_ON = ("defined", "dme_referrers")
SCALE_FALLBACKS = ("mad", "iqr", "std", "parent", "unit")   # the order _level_stats / _robust_stats implement
PM_TARGETS = ("PM2", "PM3")
# Features whose raw statistic is measured against the peer cell's own reference (pooled code distribution, rare
# flags, allowed medians, PM regression) -> the columns that change with the reference. Their peer statistics use
# every member's value recomputed against the same reference ("{column}@{level}" in raw / fit_peer_stats).
REF_DEPENDENT = {"SM1": ("SM1",), "SM2": ("SM2__k",), "CH4": ("CH4",), "PM2": ("PM2",), "PM3": ("PM3",)}
AT = "@"
CATALOG = (
    "UP0", "UP1", "UP2", "UP3", "UP4", "UP5", "UP6", "OU1", "OU2", "OU3", "OU4", "OU5", "SM1", "SM2", "SM3", "SM4",
    "CH1", "CH2", "CH3", "CH4", "PM1", "PM2", "PM3", "PM4", "PM4_missing", "DI0", "DI1", "DI2", "DI2_missing", "DI3", "DI4",
    "DC1", "DC2", "CX1", "CX2", "CX3", "CX4", "CX5", "CX6", "CX7", "CX8",
)
DATASETS = ("cms_ps", "cms_prov", "dme_s", "dme_p")
PS_NUM = ("tot_srvcs", "tot_bene_day_srvcs", "avg_sbmtd_chrg", "avg_mdcr_alowd_amt", "avg_mdcr_pymt_amt",
          "avg_mdcr_stdzd_amt")
PS_COLS = ("npi", "hcpcs_cd", "hcpcs_drug_ind", "place_of_srvc", *PS_NUM)
PROV_NUM = ("tot_benes", "tot_srvcs", "tot_hcpcs_cds")
PROV_COLS = ("npi", *PROV_NUM, "bene_avg_risk_scre", "bene_dual_cnt")
DMES_NUM = ("tot_suplrs", "tot_suplr_clms", "tot_suplr_srvcs", "avg_suplr_mdcr_pymt_amt")
DMES_COLS = ("npi", "hcpcs_cd", "suplr_rentl_ind", *DMES_NUM)
DMEP_COLS = ("npi", "tot_suplr_benes")
NEED = {"cms_ps": PS_COLS, "cms_prov": PROV_COLS, "dme_s": DMES_COLS, "dme_p": DMEP_COLS}
PS_GRAIN = ("npi", "hcpcs_cd", "place_of_srvc")
DMES_GRAIN = ("npi", "hcpcs_cd", "suplr_rentl_ind")
COHORT_COLS = ("npi", "year", "in_cohort", "peer_cell", "peer_level", "cell_ent", "cell_spec_ent", "cell_spec_ent_div")
CODES_COLS = ("level", "peer_cell", "hcpcs_cd", "n_cell", "srv", "srv_share", "n_providers", "provider_share", "rare",
              "med_allowed_f", "n_allowed_f", "med_allowed_o", "n_allowed_o")
STATS_COLS = ("peer_cell", "feature", "level", "stat_cell", "n", "median", "scale", "scale_method", "alpha",
              "prior_mean")
PM_COLS = ("peer_cell", "target", "level", "fit_cell", "n", "term", "coef", "impute")


class FeatureError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- config

@dataclass(frozen=True)
class FeatureSpec:
    id: str
    name: str
    group: str
    direction: str
    suppression: str
    kind: str
    shrink: bool
    monotone: int
    fit_on: str = "defined"
    source: str | None = None
    optional: bool = False


@dataclass(frozen=True)
class FeatureConfig:
    specs: tuple[FeatureSpec, ...]
    z_clip: float
    mad_constant: float
    iqr_constant: float
    scale_eps: float
    min_defined: int
    alpha_min: float
    alpha_max: float
    alpha_max_median_n: float
    rare_share: float
    ch4_min_cell: int
    ch4_min_national: int
    pm_prefix: str
    pm_min_n: int
    pm_min_n_per_term: int
    prune_rho: float
    code_sets: dict
    high_risk: tuple[tuple[str, str], ...]
    pruned: tuple[str, ...]

    @property
    def robust(self) -> tuple[FeatureSpec, ...]:
        return tuple(s for s in self.specs if s.kind == "robust_z")

    @property
    def one_hot(self) -> tuple[FeatureSpec, ...]:
        return tuple(s for s in self.specs if s.kind == "one_hot")

    def spec(self, fid: str) -> FeatureSpec:
        for s in self.specs:
            if s.id == fid:
                return s
        raise KeyError(fid)

    @classmethod
    def load(cls, paths: Paths | None = None) -> "FeatureConfig":
        doc = load_config("features", paths)
        try:
            t, sh, cs = doc["transform"], doc["shrinkage"], doc["code_sets"]
            if tuple(t["scale_fallbacks"]) != SCALE_FALLBACKS:
                raise FeatureError(f"transform.scale_fallbacks must be {list(SCALE_FALLBACKS)} (the order the code "
                                   f"implements); got {t['scale_fallbacks']}")
            specs = tuple(FeatureSpec(
                id=str(f["id"]), name=str(f["name"]), group=str(f["group"]), direction=str(f["direction"]),
                suppression=str(f["suppression"]), kind=str(f["kind"]), shrink=bool(f["shrink"]),
                monotone=int(f["monotone"]), fit_on=str(f.get("fit_on", "defined")), source=f.get("source"),
                optional=bool(f.get("optional", False))) for f in doc["features"])
            cfg = cls(
                specs=specs, z_clip=float(t["z_clip"]), mad_constant=float(t["mad_constant"]),
                iqr_constant=float(t["iqr_constant"]), scale_eps=float(t["scale_eps"]),
                min_defined=int(doc["peer_stats"]["min_defined"]), alpha_min=float(sh["alpha_min"]),
                alpha_max=float(sh["alpha_max"]), alpha_max_median_n=float(sh["alpha_max_median_n"]),
                rare_share=float(doc["atypical_mix"]["rare_provider_share"]),
                ch4_min_cell=int(doc["charge_index"]["min_providers_cell"]),
                ch4_min_national=int(doc["charge_index"]["min_providers_national"]),
                pm_prefix=str(doc["patient_mix"]["covariate_prefix"]), pm_min_n=int(doc["patient_mix"]["min_n"]),
                pm_min_n_per_term=int(doc["patient_mix"]["min_n_per_term"]),
                prune_rho=float(doc["prune"]["spearman_abs"]),
                code_sets={k: (dict(v) if isinstance(v, dict) else tuple(str(c) for c in v)) for k, v in cs.items()},
                high_risk=tuple((str(r["from"]), str(r["to"])) for r in doc["dme_high_risk"]["hcpcs_ranges"]),
                pruned=tuple(str(p) for p in (doc.get("pruned") or ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FeatureError(f"configs/features.yaml incomplete or malformed: {exc!r}") from exc
        cfg._validate(groups_mod.load_groups(paths))
        return cfg

    def _validate(self, groups: tuple[groups_mod.Group, ...]) -> None:
        ids = [s.id for s in self.specs]
        problems = []
        if len(ids) != len(set(ids)) or set(ids) != set(CATALOG):
            problems.append(f"feature IDs must be exactly the catalog's once each; missing "
                            f"{sorted(set(CATALOG) - set(ids))}, extra {sorted(set(ids) - set(CATALOG))}")
        by_group = {g.name: g for g in groups}
        for s in self.specs:
            if (s.direction not in ("up", "none") or s.suppression not in ("V", "T", "VT") or s.kind not in KINDS
                    or s.fit_on not in FIT_ON):
                problems.append(f"{s.id}: bad direction/suppression/kind/fit_on")
            if s.shrink and s.kind != "robust_z":
                problems.append(f"{s.id}: only robust_z features can be shrunk")
            if s.monotone != (1 if (s.direction == "up" and s.kind == "robust_z") else 0):
                problems.append(f"{s.id}: monotone must be +1 exactly for `up` robust_z features")
            if (s.kind == "one_hot") != (s.source is not None):
                problems.append(f"{s.id}: one_hot features (and only they) need a cohort `source` column")
            g = by_group.get(s.group)
            listed = g is not None and (f"{s.id}__" in g.prefixes if s.kind == "one_hot" else s.id in g.features)
            if not listed:
                problems.append(f"{s.id}: group {s.group!r} in features.yaml does not list it in groups.yaml")
        extra = {f for g in groups for f in g.features} - set(ids)
        if extra:
            problems.append(f"groups.yaml lists features not in features.yaml: {sorted(extra)}")
        cs = self.code_sets
        try:
            if (len(cs["established"]) != 5 or not set(cs["established_high"]) <= set(cs["established"])
                    or cs["level5"]["established"] not in cs["established"] or cs["level5"]["new"] not in cs["new"]
                    or cs["level5"]["ed"] not in cs["ed"]):
                problems.append("code_sets: established must list levels 1-5; level-5 codes must be in their sets")
            _ = cs["new_deleted"]
        except KeyError as exc:
            problems.append(f"code_sets: missing {exc}")
        for lo, hi in self.high_risk:
            ok = all(len(c) == 5 and c[0].isalpha() and c[0].isupper() and c[1:].isdigit() for c in (lo, hi))
            if not (ok and lo[0] == hi[0] and lo <= hi):
                problems.append(f"dme_high_risk range {lo}-{hi}: need 5-character codes, same letter, from <= to")
        if set(self.pruned) - set(ids):
            problems.append(f"pruned lists unknown IDs {sorted(set(self.pruned) - set(ids))}")
        if not (self.z_clip > 0 and self.mad_constant > 0 and self.iqr_constant > 0 and self.scale_eps >= 0
                and self.min_defined >= 1 and 0 < self.alpha_min <= self.alpha_max and self.alpha_max_median_n > 0
                and 0 < self.rare_share < 1 and self.ch4_min_cell >= 1 and self.ch4_min_national >= 1
                and self.pm_min_n >= 1 and self.pm_min_n_per_term >= 1 and 0 < self.prune_rho <= 1):
            problems.append("a numeric parameter is out of range")
        if problems:
            raise FeatureError("configs/features.yaml: " + "; ".join(problems))


# ----------------------------------------------------------------------------------------------- inputs

def _lf(x, what: str) -> pl.LazyFrame:
    if isinstance(x, pl.LazyFrame):
        return x
    if isinstance(x, pl.DataFrame):
        return x.lazy()
    raise FeatureError(f"{what}: expected a polars DataFrame or LazyFrame, got {type(x).__name__}")


def _df(x, what: str) -> pl.DataFrame:
    return x if isinstance(x, pl.DataFrame) else _lf(x, what).collect()


def _frames(frames: Mapping, year: int) -> dict[str, pl.LazyFrame]:
    missing = [d for d in DATASETS if d not in frames]
    if missing:
        raise FeatureError(f"{year}: frames lack {missing} (DMEPOS may be empty frames with the schema columns)")
    out = {}
    for ds in DATASETS:
        lf = _lf(frames[ds], ds)
        names = lf.collect_schema().names()
        miss = [c for c in (*NEED[ds], "year") if c not in names]
        if miss:
            raise FeatureError(f"{year} {ds}: harmonised columns missing {miss}")
        out[ds] = lf
    if frames.get("nppes") is not None:
        lf = _lf(frames["nppes"], "nppes")
        miss = [c for c in ("npi", "enumeration_date") if c not in lf.collect_schema().names()]
        if miss:
            raise FeatureError(f"nppes extract: columns missing {miss}")
        out["nppes"] = lf
    return out


def _check_values(lfs: dict[str, pl.LazyFrame], year: int, datasets: Iterable[str]) -> None:
    """Fail loudly on rows of another year, unknown flags, nulls / negatives in the numbers used, and
    duplicated grains (CONTRACTS §0)."""
    def not_in(col: str, ok: list[str]) -> pl.Expr:
        return (~pl.col(col).is_in(ok).fill_null(False)).sum()

    def nulls(cols) -> pl.Expr:
        return pl.any_horizontal([pl.col(c).is_null() for c in cols]).sum()

    def negative(cols) -> pl.Expr:
        return pl.any_horizontal([(pl.col(c) < 0).fill_null(False) for c in cols]).sum()

    checks = {
        "cms_ps": (("npi", "hcpcs_cd", "place_of_srvc"), [
            ("hcpcs_drug_ind not in {Y, N}", not_in("hcpcs_drug_ind", ["Y", "N"])),
            ("place_of_srvc not in {F, O}", not_in("place_of_srvc", ["F", "O"])),
            ("nulls in used columns", nulls(PS_COLS)), ("negative numbers", negative(PS_NUM))]),
        "cms_prov": (("npi",), [("nulls in npi/totals", nulls(("npi", *PROV_NUM))),
                                ("negative numbers", negative((*PROV_NUM, "bene_dual_cnt")))]),
        "dme_s": (("npi", "hcpcs_cd", "suplr_rentl_ind"), [
            ("suplr_rentl_ind not in {Y, N}", not_in("suplr_rentl_ind", ["Y", "N"])),
            ("nulls in used columns", nulls(DMES_COLS)), ("negative numbers", negative(DMES_NUM))]),
        # Tot_Suplr_Benes is blank when 1-10 (CMS suppression, D118): only the NPI must be present
        "dme_p": (("npi",), [("nulls in npi", nulls(("npi",))),
                             ("negative numbers", negative(("tot_suplr_benes",)))]),
    }
    problems = []
    for ds in datasets:
        grain, exprs = checks[ds]
        lf = lfs[ds]
        row = lf.select((pl.col("year") != year).fill_null(True).sum().alias("wrong year"),
                        *[e.alias(name) for name, e in exprs]).collect().row(0, named=True)
        row["duplicated grain"] = lf.group_by(grain).len().filter(pl.col("len") > 1).select(pl.len()).collect().item()
        problems += [f"{ds}: {v} rows with {k}" for k, v in row.items() if v]
    if problems:
        raise FeatureError(f"{year}: " + "; ".join(problems))


def _cohort(cohort, year: int) -> pl.DataFrame:
    df = _df(cohort, "cohort")
    miss = [c for c in COHORT_COLS if c not in df.columns]
    if miss:
        raise FeatureError(f"{year}: cohort columns missing {miss}")
    if (df["year"] != year).fill_null(True).any():
        raise FeatureError(f"cohort rows of another year than {year} (pass one year at a time)")
    if df["npi"].null_count() or df["npi"].is_duplicated().any():
        raise FeatureError(f"{year}: cohort has null or duplicated NPIs")
    if df["in_cohort"].null_count():
        raise FeatureError(f"{year}: null in_cohort")
    if "injected" in df.columns and (df.schema["injected"] != pl.Boolean or df["injected"].null_count()):
        raise FeatureError("cohort column `injected` must be boolean without nulls")
    return df.with_columns(pl.lit(str(year)).alias("cell_national"))


def _need_cells(df: pl.DataFrame, what: str) -> None:
    bad = df.filter(pl.col("peer_cell").is_null() | ~pl.col("peer_level").is_in(list(PEER_LEVELS)).fill_null(False))
    if bad.height:
        raise FeatureError(f"{what}: {bad.height} rows without a peer cell, e.g. {bad['npi'].head(3).to_list()}")


def _reference(df: pl.DataFrame) -> pl.DataFrame:
    """Reference providers: in-cohort and not injected (CONTRACTS §2.2: the only rows fit_* may use)."""
    ref = df.filter(pl.col("in_cohort"))
    if "injected" in df.columns:
        ref = ref.filter(~pl.col("injected"))
    if ref.is_empty():
        raise FeatureError("no reference providers (in_cohort and not injected) to fit peer statistics on")
    _need_cells(ref, "reference providers")
    return ref


def _ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """num / den, null (undefined) unless den > 0."""
    return pl.when(den > 0).then(num / den).otherwise(None)


def _sanitise(expr: pl.Expr) -> pl.Expr:
    """One-hot value -> column suffix: lower case, non-alphanumerics -> '_', null -> 'na' (CONTRACTS §2.2)."""
    return expr.cast(pl.String).fill_null(NA).str.to_lowercase().str.replace_all(r"[^a-z0-9]", "_")


def high_risk_expr(cfg: FeatureConfig, col: str = "hcpcs_cd") -> pl.Expr:
    """True for HCPCS codes in configs/features.yaml dme_high_risk ranges (inclusive)."""
    c = pl.col(col)
    shape = c.str.contains(r"^[A-Z][0-9]{4}$").fill_null(False)
    inside = pl.any_horizontal([(c >= lo) & (c <= hi) for lo, hi in cfg.high_risk]).fill_null(False)
    return shape & inside


# ----------------------------------------------------------------------------------------------- step 1

@dataclass(frozen=True)
class PeerReference:
    """Label-free code reference of one year (fit on reference providers)."""
    year: int
    codes: pl.DataFrame                    # CODES_COLS; non-drug codes per peer cell (+ level 'national' rows)
    cells: pl.DataFrame                    # peer_cell, peer_level, n_reference
    vocab: dict[str, tuple[str, ...]]      # one-hot ID -> sanitised values


def fit_peer_reference(frames: Mapping, cohort, year: int, cfg: FeatureConfig | None = None) -> PeerReference:
    """Per peer cell x non-drug HCPCS: pooled service share (SM1), share of the cell's reference providers billing
    it and the `rare` flag (SM2), median allowed per place of service (CH4); national rows for the CH4 fallback.
    A peer cell's members are all reference providers whose cell at that level is the peer cell."""
    cfg = cfg or FeatureConfig.load()
    lfs = _frames(frames, year)
    _check_values(lfs, year, ("cms_ps",))
    ref = _reference(_cohort(cohort, year))
    nd = (lfs["cms_ps"].filter((pl.col("hcpcs_drug_ind") == "N") & (pl.col("tot_srvcs") > 0))
          .join(ref.lazy().select("npi"), on="npi", how="semi")
          .select("npi", "hcpcs_cd", "place_of_srvc", "tot_srvcs", "avg_mdcr_alowd_amt").collect().sort(PS_GRAIN))
    pc = nd.group_by("npi", "hcpcs_cd", maintain_order=True).agg(pl.col("tot_srvcs").sum().alias("srv"))
    used = ref.select("peer_cell", "peer_level").unique()
    parts = []
    for level in LEVELS:
        col = CELL_COL[level]
        members = ref.select("npi", pl.col(col).alias("peer_cell"))
        if level != "national":
            cells = used.filter(pl.col("peer_level") == level)["peer_cell"]
            if cells.is_empty():
                continue
            members = members.filter(pl.col("peer_cell").is_in(cells.implode()))
        parts.append(_code_table(pc, nd, members, cfg).with_columns(pl.lit(level).alias("level")))
    codes = pl.concat(parts).select(CODES_COLS).sort("level", "peer_cell", "hcpcs_cd")
    cells = ref.group_by("peer_cell", "peer_level").agg(pl.len().cast(pl.Int64).alias("n_reference")).sort("peer_cell")
    return PeerReference(year=year, codes=codes, cells=cells, vocab=_vocab(ref, cfg))


def _code_table(pc: pl.DataFrame, nd: pl.DataFrame, members: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    n_cell = members.group_by("peer_cell").agg(pl.len().cast(pl.Int64).alias("n_cell"))
    codes = (pc.join(members, on="npi", maintain_order="left").group_by("peer_cell", "hcpcs_cd")
             .agg(pl.col("srv").sum(), pl.len().cast(pl.Int64).alias("n_providers"))
             .join(n_cell, on="peer_cell").sort("peer_cell", "hcpcs_cd")
             .with_columns((pl.col("srv") / pl.col("srv").sum().over("peer_cell")).alias("srv_share"),
                           (pl.col("n_providers") / pl.col("n_cell")).alias("provider_share"))
             .with_columns((pl.col("provider_share") < cfg.rare_share).alias("rare")))
    allowed = (nd.join(members, on="npi", maintain_order="left").group_by("peer_cell", "hcpcs_cd", "place_of_srvc")
               .agg(pl.col("avg_mdcr_alowd_amt").median().alias("med"), pl.len().cast(pl.Int64).alias("n")))
    for place in ("F", "O"):
        p = place.lower()
        codes = codes.join(allowed.filter(pl.col("place_of_srvc") == place).select(
            "peer_cell", "hcpcs_cd", pl.col("med").alias(f"med_allowed_{p}"), pl.col("n").alias(f"n_allowed_{p}")),
            on=["peer_cell", "hcpcs_cd"], how="left").with_columns(pl.col(f"n_allowed_{p}").fill_null(0))
    return codes


def _vocab(ref: pl.DataFrame, cfg: FeatureConfig) -> dict[str, tuple[str, ...]]:
    out = {}
    for s in cfg.one_hot:
        if s.source not in ref.columns:
            raise FeatureError(f"{s.id}: cohort column {s.source!r} missing")
        vals = ref.select(pl.col(s.source).unique()).with_columns(_sanitise(pl.col(s.source)).alias("_s"))
        clash = vals.group_by("_s").agg(pl.col(s.source)).filter(pl.col(s.source).list.len() > 1)
        if clash.height or (vals["_s"] == "").any():
            raise FeatureError(f"{s.id}: values collide or vanish after sanitising: {clash.to_dicts()}")
        out[s.id] = tuple(sorted(vals["_s"].to_list()))
    return out


# ----------------------------------------------------------------------------------------------- step 2

def raw_statistics(frames: Mapping, cohort, refs: PeerReference, year: int,
                   cfg: FeatureConfig | None = None) -> pl.DataFrame:
    """Raw statistic r of every feature (FEATURE_CATALOG formulas) for every cohort row given, plus k / n of
    the shrunk rates, PM2/PM3 regression inputs (PMx__y, bene_cc_*) and the indicators. One row per NPI."""
    cfg = cfg or FeatureConfig.load()
    if refs.year != year:
        raise FeatureError(f"peer reference of {refs.year} used for {year}")
    lfs = _frames(frames, year)
    _check_values(lfs, year, DATASETS)
    coh = _cohort(cohort, year)
    _need_cells(coh, "cohort rows to featurise")
    unknown = coh.filter(~pl.col("peer_cell").is_in(refs.cells["peer_cell"].implode()))
    if unknown.height:
        raise FeatureError(f"{unknown.height} rows in peer cells without reference providers, e.g. "
                           f"{unknown['peer_cell'].head(3).to_list()}")
    ids = coh.select("npi", "peer_cell")
    keep = ids.lazy().select("npi")
    # Canonical row order (grain), so every sum runs in the same order whatever the input order.
    ps = lfs["cms_ps"].join(keep, on="npi", how="semi").select(PS_COLS).collect().sort(PS_GRAIN)
    cc = sorted(c for c in lfs["cms_prov"].collect_schema().names() if c.startswith(cfg.pm_prefix))
    prov = (lfs["cms_prov"].join(keep, on="npi", how="semi")
            .select(*PROV_COLS, *[pl.col(c).cast(pl.Float64) for c in cc])
            .rename({"tot_srvcs": "prov_tot_srvcs"}).collect())
    df = (ids.join(_ps_sums(ps, cfg), on="npi", how="left")
          .join(_mix(ps, ids, refs), on="npi", how="left")
          .join(_hhi(ps), on="npi", how="left")
          .join(_price_index(ps, ids, refs, cfg), on="npi", how="left")
          .join(prov, on="npi", how="left")
          .join(_dme_sums(lfs, keep, cfg), on="npi", how="left"))
    if "nppes" in lfs:
        df = df.join(_enumeration_dates(lfs["nppes"], ids, year), on="npi", how="left")
    raw = _raw_columns(df, cfg, cc, year, "nppes" in lfs).sort("npi")
    return raw.join(_reference_levels(ps, coh, refs, raw, cfg), on="npi", how="left", maintain_order="left")


def _enumeration_dates(nppes: pl.LazyFrame, ids: pl.DataFrame, year: int) -> pl.DataFrame:
    """CX7 input. Every featurised NPI needs exactly one enumeration date: a public NPPES file blanks the dates of
    NPIs deactivated later, so a missing date would carry information from after year t (leakage checklist
    item 1) and is refused rather than imputed or flagged."""
    e = (nppes.join(ids.lazy().select("npi"), on="npi", how="semi")
         .select("npi", pl.col("enumeration_date").cast(pl.Date)).collect())
    dup = e.filter(pl.col("npi").is_duplicated())["npi"].unique().sort()
    if dup.len():
        raise FeatureError(f"{year}: NPPES extract has {dup.len()} NPIs on several rows, e.g. {dup.head(3).to_list()}")
    miss = ids.join(e.drop_nulls("enumeration_date"), on="npi", how="anti")["npi"].sort()
    if miss.len():
        raise FeatureError(
            f"{year}: {miss.len()} featurised NPIs have no enumeration date in the NPPES extract (e.g. "
            f"{miss.head(3).to_list()}). NPPES blanks the dates of NPIs deactivated later, so CX7's missingness would "
            "carry information from after year t (leakage checklist item 1). Use an extract with an enumeration date "
            "for every NPI (injected providers included), or remove data/interim/nppes so that CX7 is not computed "
            "(HANDOFF §10.4).")
    return e


def _sm1() -> pl.Expr:
    """SM1 from _mix's sums: in-support terms + 1/2 x the reference mass on codes the provider does not bill."""
    return (pl.col("_jsd_in") + 0.5 * (1.0 - pl.col("_q_in")).clip(lower_bound=0.0)).clip(0.0, 1.0)


def _reference_levels(ps: pl.DataFrame, coh: pl.DataFrame, refs: PeerReference, raw: pl.DataFrame,
                      cfg: FeatureConfig) -> pl.DataFrame:
    """For each peer level L, SM1, SM2__k and CH4 of every row against the reference of its level-L cell, when that
    cell is a peer cell of the year ("{col}@{L}"; the row's own value when its peer cell is that cell; null when the
    cell is no peer cell). fit_peer_stats summarises a peer cell over all its members with these values, so every
    member's r is measured against the same reference as the scored provider's (FEATURE_CATALOG SM1, SM2, CH4)."""
    own = coh.select("npi", "peer_level", *[CELL_COL[lv] for lv in PEER_LEVELS]).join(
        raw.select("npi", "SM1", "SM2__k", "CH4"), on="npi", how="left")
    out = own.select("npi")
    for lv in PEER_LEVELS:
        cells = refs.cells.filter(pl.col("peer_level") == lv)["peer_cell"]
        other = own.filter((pl.col("peer_level") != lv) & pl.col(CELL_COL[lv]).is_in(cells.implode())).select(
            "npi", pl.col(CELL_COL[lv]).alias("peer_cell"))
        alt = (other.select("npi")
               .join(_mix(ps, other, refs), on="npi", how="left")
               .join(_price_index(ps, other, refs, cfg), on="npi", how="left")
               .select("npi", _sm1().alias("SM1"), pl.col("_rare_srv").fill_null(0.0).cast(pl.Float64).alias("SM2__k"),
                       _ratio(pl.col("_ch4_num"), pl.col("_ch4_den")).alias("CH4")))
        j = own.join(alt, on="npi", how="left", suffix="_alt", maintain_order="left")
        is_own = pl.col("peer_level") == lv
        out = out.hstack(j.select(*[pl.when(is_own).then(pl.col(c)).otherwise(pl.col(f"{c}_alt")).alias(f"{c}{AT}{lv}")
                                    for c in ("SM1", "SM2__k", "CH4")]))
    return out


def _ps_sums(ps: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    cs = cfg.code_sets
    est, new, ed = list(cs["established"]), list(cs["new"]), list(cs["ed"])
    em = est + new + list(cs["new_deleted"]) + ed
    code, srv, nd = pl.col("hcpcs_cd"), pl.col("tot_srvcs"), pl.col("hcpcs_drug_ind") == "N"
    level = code.replace_strict(est, list(range(1, 6)), default=None, return_dtype=pl.Float64)

    def s(x: pl.Expr, cond: pl.Expr) -> pl.Expr:
        return x.filter(cond).sum()

    pay, al, sd = srv * pl.col("avg_mdcr_pymt_amt"), srv * pl.col("avg_mdcr_alowd_amt"), srv * pl.col("avg_mdcr_stdzd_amt")
    return ps.group_by("npi").agg(
        s(srv, code.is_in(est)).alias("_est"), s(srv, code == cs["level5"]["established"]).alias("_up1k"),
        s(srv, code.is_in(list(cs["established_high"]))).alias("_up2k"),
        s(srv, code.is_in(new)).alias("_new"), s(srv, code == cs["level5"]["new"]).alias("_up3k"),
        s(srv, code.is_in(ed)).alias("_ed"), s(srv, code == cs["level5"]["ed"]).alias("_up4k"),
        s(srv * level, code.is_in(est)).alias("_up5num"),
        code.is_in(em).any().alias("_em_any"), s(srv, code.is_in(em)).alias("_em_srv"),
        s(al, code.is_in(em)).alias("_em_al"),
        s(srv, nd).alias("_nd_srv"), s(pl.col("tot_bene_day_srvcs"), nd).alias("_nd_days"),
        srv.sum().alias("_all_srv"), pay.sum().alias("_pay"), s(pay, ~nd).alias("_drug_pay"),
        (srv * pl.col("avg_sbmtd_chrg")).sum().alias("_sb"), al.sum().alias("_al"), sd.sum().alias("_sd"),
        s(sd, nd).alias("_nd_sd"),
    )


def _mix(ps: pl.DataFrame, ids: pl.DataFrame, refs: PeerReference) -> pl.DataFrame:
    """SM1 (Jensen-Shannon divergence, base 2, of the provider's non-drug service distribution p from the peer
    cell's pooled distribution q) and SM2's numerator. Codes the provider bills contribute
    1/2 p log2(2p/(p+q)) + 1/2 q log2(2q/(p+q)); codes it does not bill contribute q/2 each, i.e.
    1/2 (1 - sum of q over its codes) in total."""
    cell = refs.codes.filter(pl.col("level") != "national").select("peer_cell", "hcpcs_cd", "srv_share", "rare")
    p, q = pl.col("_p"), pl.col("srv_share").fill_null(0.0)
    m = (p + q) / 2
    term = 0.5 * p * (p / m).log(2) + pl.when(q > 0).then(0.5 * q * (q / m).log(2)).otherwise(0.0)
    return (ps.filter((pl.col("hcpcs_drug_ind") == "N") & (pl.col("tot_srvcs") > 0))
            .group_by("npi", "hcpcs_cd", maintain_order=True).agg(pl.col("tot_srvcs").sum().alias("srv"))
            .join(ids, on="npi", maintain_order="left")
            .join(cell, on=["peer_cell", "hcpcs_cd"], how="left", maintain_order="left")
            .with_columns((pl.col("srv") / pl.col("srv").sum().over("npi")).alias("_p"))
            .group_by("npi").agg(term.sum().alias("_jsd_in"), q.sum().alias("_q_in"),
                                 pl.col("srv").filter(pl.col("rare").fill_null(True)).sum().alias("_rare_srv")))


def _hhi(ps: pl.DataFrame) -> pl.DataFrame:
    """SM3: sum over codes of (payment share)^2 = sum pay_c^2 / (sum pay_c)^2, all lines."""
    pay = (ps.group_by("npi", "hcpcs_cd", maintain_order=True)
           .agg((pl.col("tot_srvcs") * pl.col("avg_mdcr_pymt_amt")).sum().alias("pay")))
    return pay.group_by("npi").agg(pl.col("pay").sum().alias("_hhi_tot"), (pl.col("pay") ** 2).sum().alias("_hhi_sq"))


def _price_index(ps: pl.DataFrame, ids: pl.DataFrame, refs: PeerReference, cfg: FeatureConfig) -> pl.DataFrame:
    """CH4: service-weighted mean over non-drug (code, place) lines of allowed / reference median allowed."""
    med = ["med_allowed_f", "n_allowed_f", "med_allowed_o", "n_allowed_o"]
    cell = refs.codes.filter(pl.col("level") != "national").select("peer_cell", "hcpcs_cd", *med)
    nat = refs.codes.filter(pl.col("level") == "national").select(
        "hcpcs_cd", *[pl.col(c).alias(f"{c}_nat") for c in med])
    f = pl.col("place_of_srvc") == "F"

    def pick(stem: str, suffix: str = "") -> pl.Expr:
        return pl.when(f).then(pl.col(f"{stem}_f{suffix}")).otherwise(pl.col(f"{stem}_o{suffix}"))

    cmed, cn = pick("med_allowed"), pick("n_allowed").fill_null(0)
    nmed, nn = pick("med_allowed", "_nat"), pick("n_allowed", "_nat").fill_null(0)
    den = (pl.when((cn >= cfg.ch4_min_cell) & (cmed > 0)).then(cmed)
           .when((nn >= cfg.ch4_min_national) & (nmed > 0)).then(nmed).otherwise(None))
    rows = (ps.filter((pl.col("hcpcs_drug_ind") == "N") & (pl.col("tot_srvcs") > 0))
            .select("npi", "hcpcs_cd", "place_of_srvc", "tot_srvcs", "avg_mdcr_alowd_amt")
            .join(ids, on="npi", maintain_order="left")
            .join(cell, on=["peer_cell", "hcpcs_cd"], how="left", maintain_order="left")
            .join(nat, on="hcpcs_cd", how="left", maintain_order="left")
            .with_columns((pl.col("avg_mdcr_alowd_amt") / den).alias("_ratio")))
    ok = pl.col("_ratio").is_not_null()
    return rows.group_by("npi").agg((pl.col("tot_srvcs") * pl.col("_ratio")).filter(ok).sum().alias("_ch4_num"),
                                    pl.col("tot_srvcs").filter(ok).sum().alias("_ch4_den"))


def _dme_sums(lfs: dict[str, pl.LazyFrame], keep: pl.LazyFrame, cfg: FeatureConfig) -> pl.DataFrame:
    srv, pay = pl.col("tot_suplr_srvcs"), pl.col("tot_suplr_srvcs") * pl.col("avg_suplr_mdcr_pymt_amt")
    has_sup = pl.col("tot_suplrs") > 0
    # Collect, then sort by grain eagerly: a lazy sort before group_by is dropped by the optimiser, and the sums
    # would then depend on the input row order.
    ds = (lfs["dme_s"].join(keep, on="npi", how="semi").select(DMES_COLS).collect().sort(DMES_GRAIN).group_by("npi").agg(
        pay.sum().alias("_dme_pay"), pay.filter(high_risk_expr(cfg)).sum().alias("_hr_pay"),
        pl.col("tot_suplr_clms").sum().alias("_clms"), srv.sum().alias("_dsrv"),
        srv.filter(pl.col("suplr_rentl_ind") == "Y").sum().alias("_rent"),
        (srv * pl.col("tot_suplr_clms") / pl.col("tot_suplrs")).filter(has_sup).sum().alias("_dc1_num"),
        srv.filter(has_sup).sum().alias("_dc1_den"), pl.len().alias("_dme_s_rows")))
    dp = lfs["dme_p"].join(keep, on="npi", how="semi").select(
        "npi", pl.col("tot_suplr_benes").cast(pl.Float64).alias("_dme_benes"), pl.lit(True).alias("_in_dmep")).collect()
    return dp.join(ds, on="npi", how="full", coalesce=True)


def _raw_columns(df: pl.DataFrame, cfg: FeatureConfig, cc: list[str], year: int, nppes: bool) -> pl.DataFrame:
    c = pl.col
    benes = c("tot_benes").cast(pl.Float64)
    ou1 = _ratio(c("_nd_srv"), benes)
    ch2 = _ratio(c("_sd"), benes)
    no_dme = ~c("_in_dmep").fill_null(False)

    def dme(x: pl.Expr) -> pl.Expr:          # FEATURE_CATALOG DI0: no DME-P row -> raw DI1-DI4 = 0
        return pl.when(no_dme).then(0.0).otherwise(x)

    def referral(x: pl.Expr) -> pl.Expr:     # DC1, DC2 describe referrals: undefined without a DME-P row
        return pl.when(no_dme).then(None).otherwise(x)

    def rate(fid: str, k: pl.Expr, n: pl.Expr, wrap=lambda e: e) -> list[pl.Expr]:
        k, n = k.cast(pl.Float64), n.cast(pl.Float64)
        return [wrap(_ratio(k, n)).alias(fid), wrap(k).alias(f"{fid}__k"), wrap(n).alias(f"{fid}__n")]

    dual = c("bene_dual_cnt").cast(pl.Float64)
    cols = [
        c("_em_any").fill_null(False).cast(pl.Int8).alias("UP0"),
        *rate("UP1", c("_up1k"), c("_est")), *rate("UP2", c("_up2k"), c("_est")),
        *rate("UP3", c("_up3k"), c("_new")), *rate("UP4", c("_up4k"), c("_ed")),
        _ratio(c("_up5num"), c("_est")).alias("UP5"), _ratio(c("_em_al"), c("_em_srv")).alias("UP6"),
        ou1.alias("OU1"), _ratio(c("_nd_days"), benes).alias("OU2"), _ratio(c("_nd_srv"), c("_nd_days")).alias("OU3"),
        _ratio(c("tot_hcpcs_cds").cast(pl.Float64), benes).alias("OU4"), (c("_nd_srv") + 1.0).log().alias("OU5"),
        _sm1().alias("SM1"),
        *rate("SM2", c("_rare_srv").fill_null(0.0), c("_nd_srv")),
        _ratio(c("_hhi_sq"), c("_hhi_tot") ** 2).alias("SM3"), _ratio(c("_drug_pay"), c("_pay")).alias("SM4"),
        _ratio(c("_sb"), c("_al")).alias("CH1"), ch2.alias("CH2"), _ratio(c("_nd_sd"), c("_nd_srv")).alias("CH3"),
        _ratio(c("_ch4_num"), c("_ch4_den")).alias("CH4"),
        c("bene_avg_risk_scre").cast(pl.Float64).alias("PM1"),
        pl.when(ou1 > 0).then(ou1.log()).alias("PM2__y"), pl.when(ch2 > 0).then(ch2.log()).alias("PM3__y"),
        *rate("PM4", dual, pl.when(dual.is_not_null()).then(benes)),
        dual.is_null().cast(pl.Int8).alias("PM4_missing"),
        no_dme.cast(pl.Int8).alias("DI0"),
        dme(_ratio(c("_dme_pay").fill_null(0.0), c("_dme_benes"))).alias("DI1"),
        dme(_ratio(c("_dme_benes"), benes)).alias("DI2"),
        (~no_dme & c("_dme_benes").is_null()).cast(pl.Int8).alias("DI2_missing"),   # D118: DI1, DI2, DI4 undefined
        dme(_ratio(c("_hr_pay"), c("_dme_pay"))).alias("DI3"),
        dme(_ratio(c("_clms").cast(pl.Float64).fill_null(0.0), c("_dme_benes"))).alias("DI4"),
        referral(_ratio(c("_dc1_num"), c("_dc1_den"))).alias("DC1"),
        *rate("DC2", c("_rent").fill_null(0.0), c("_dsrv").fill_null(0.0), referral),
        pl.when(benes > 0).then(benes.log()).alias("CX5"),
        _ratio(c("_all_srv"), c("prov_tot_srvcs")).alias("CX8"),
        c("_dme_s_rows").fill_null(0).cast(pl.Int64).alias("_dme_s_rows"),
    ]
    if nppes:   # tenure at the end of year t (every NPI has a date: _enumeration_dates); a date after t -> 0
        end = pl.date(year, 12, 31)
        cols.append(((end - c("enumeration_date")).dt.total_days() / 365.25).clip(lower_bound=0.0).alias("CX7"))
    return df.select("npi", pl.lit(year, dtype=pl.Int32).alias("year"), "peer_cell", *cols, *cc)


# ----------------------------------------------------------------------------------------------- step 3

@dataclass(frozen=True)
class PeerStats:
    """Label-free peer statistics of one year (fit on reference providers)."""
    year: int
    stats: pl.DataFrame              # STATS_COLS, one row per peer cell x robust_z feature
    pm: pl.DataFrame                 # PM_COLS, one row per peer cell x target x term
    covariates: tuple[str, ...]      # PM2/PM3 covariates (PM1 first)
    dropped_covariates: tuple[str, ...]


def fit_peer_stats(raw, cohort, year: int, cfg: FeatureConfig | None = None) -> PeerStats:
    """Peer medians, scales and beta-binomial priors per peer cell x feature, and the PM2/PM3 regressions, from
    the raw statistics of reference providers (rows of other providers in `raw` are ignored)."""
    cfg = cfg or FeatureConfig.load()
    ref = _reference(_cohort(cohort, year))
    raw = _df(raw, "raw")
    R = raw.drop("peer_cell", "year", strict=False).join(
        ref.select("npi", "peer_cell", "peer_level", *CELL_COL.values()), on="npi", how="inner").sort("npi")
    if R.is_empty():
        raise FeatureError(f"{year}: raw statistics contain no reference provider")
    cand = _candidates(ref)
    covs = ["PM1", *sorted(c for c in R.columns if c.startswith(cfg.pm_prefix))]
    dropped = tuple(c for c in covs if R[c].null_count() == R.height)
    covs = tuple(c for c in covs if c not in dropped)
    pm = pl.concat([_fit_pm(R, cand, t, covs, cfg) for t in PM_TARGETS])
    R = R.with_columns(*[pl.Series(t, _pm_residual(R, pm, t, covs)).fill_nan(None) for t in PM_TARGETS])
    missing = [f"{c}{AT}{lv}" for lv in PEER_LEVELS for f, cols in REF_DEPENDENT.items() if f not in PM_TARGETS
               for c in cols if f"{c}{AT}{lv}" not in R.columns]
    if missing:
        raise FeatureError(f"{year}: raw statistics lack {missing} (produced by raw_statistics)")
    # PM2/PM3 of every member of a peer cell against that cell's model (REF_DEPENDENT; cf. _reference_levels).
    for lv in PEER_LEVELS:
        cells = ref.filter(pl.col("peer_level") == lv)["peer_cell"].unique()
        sub = R.filter((pl.col("peer_level") != lv) & pl.col(CELL_COL[lv]).is_in(cells.implode())).with_columns(
            pl.col(CELL_COL[lv]).alias("peer_cell"))
        alt = sub.select("npi", *[pl.Series(f"{t}_alt", _pm_residual(sub, pm, t, covs) if sub.height else [],
                                            dtype=pl.Float64).fill_nan(None) for t in PM_TARGETS])
        R = R.join(alt, on="npi", how="left", maintain_order="left").with_columns(
            *[pl.when(pl.col("peer_level") == lv).then(pl.col(t)).otherwise(pl.col(f"{t}_alt")).alias(f"{t}{AT}{lv}")
              for t in PM_TARGETS]).drop([f"{t}_alt" for t in PM_TARGETS])
    return PeerStats(year=year, stats=_robust_stats(R, cand, cfg), pm=pm, covariates=covs,
                     dropped_covariates=dropped)


def _candidates(ref: pl.DataFrame) -> pl.DataFrame:
    """Per peer cell, the cells to draw statistics from, most specific first (r = 0, 1, ...)."""
    cells = ref.select("peer_cell", "peer_level", *CELL_COL.values()).unique("peer_cell")
    parts = []
    for level in PEER_LEVELS:
        chain = LEVELS[LEVELS.index(level):]
        sub = cells.filter(pl.col("peer_level") == level)
        parts += [sub.select("peer_cell", pl.lit(r, dtype=pl.Int32).alias("r"), pl.lit(lv).alias("level"),
                             pl.col(CELL_COL[lv]).alias("cell")) for r, lv in enumerate(chain)]
    return pl.concat(parts).sort("peer_cell", "r")


def _fit_mask(s: FeatureSpec) -> pl.Expr:
    return (pl.col("DI0") == 0) if s.fit_on == "dme_referrers" else pl.lit(True)


def _bb_aggs(s: FeatureSpec) -> list[pl.Expr]:
    """Per cell sums for the beta-binomial method of moments, and m = median rate (n > 0, fit mask)."""
    mask = _fit_mask(s) & (pl.col(f"{s.id}__n") > 0) & pl.col(f"{s.id}__k").is_not_null()
    k, n = pl.col(f"{s.id}__k").filter(mask), pl.col(f"{s.id}__n").filter(mask)
    pbar = k.sum() / n.sum()
    return [mask.sum().alias(f"{s.id}__I"), n.sum().alias(f"{s.id}__N"), (n * n).sum().alias(f"{s.id}__Sn2"),
            pbar.alias(f"{s.id}__pbar"), (n * (k / n - pbar) ** 2).sum().alias(f"{s.id}__S"),
            (k / n).median().alias(f"{s.id}__m"), n.median().alias(f"{s.id}__medn")]


def alpha_expr(fid: str, cfg: FeatureConfig) -> pl.Expr:
    """Beta-binomial prior strength alpha = a + b = 1/rho - 1 by the method of moments with unequal n
    (Kleinman 1973): S = sum n_i (p_i - pbar)^2 has expectation pbar(1-pbar)[I - 1 + rho (N - sum n_i^2/N - I + 1)].
    rho <= 0 (no overdispersion) -> the cap min(alpha_max, alpha_max_median_n x median n); rho >= 1 or not
    identified (fewer than 2 providers, pbar in {0, 1}, zero denominator) -> alpha_min."""
    I, N, Sn2, pbar, S, medn = (pl.col(f"{fid}__{x}") for x in ("I", "N", "Sn2", "pbar", "S", "medn"))
    denom = N - Sn2 / N - (I - 1)
    rho = (S / (pbar * (1 - pbar)) - (I - 1)) / denom
    cap = pl.min_horizontal(pl.lit(cfg.alpha_max), cfg.alpha_max_median_n * medn).clip(lower_bound=cfg.alpha_min)
    unidentified = (I < 2) | ~(pbar > 0) | ~(pbar < 1) | ~(denom > 0) | ~rho.is_finite()
    return (pl.when(unidentified).then(pl.lit(cfg.alpha_min)).when(rho <= 0).then(cap)
            .when(rho >= 1).then(pl.lit(cfg.alpha_min)).otherwise((1 / rho - 1).clip(cfg.alpha_min, cap)))


def _level_stats(R: pl.DataFrame, level: str, cfg: FeatureConfig, specs: tuple[FeatureSpec, ...] | None = None,
                 consistent: bool = True) -> pl.DataFrame:
    """Long table (level, cell, feature, n, median, scale_within, method_within, alpha, prior_mean) over the
    reference providers of each cell at `level`; shrink features are summarised as shrunk rates. With `consistent`,
    reference-dependent columns take every member's value against the level's own reference ("{col}@{level}"; none
    at the national level); otherwise each member's own r (used only to borrow a parent scale)."""
    specs = cfg.robust if specs is None else specs
    G = R.with_columns(pl.col(CELL_COL[level]).alias("cell"))
    if consistent:
        G = G.with_columns(*[(pl.col(f"{c}{AT}{level}") if level in PEER_LEVELS else pl.lit(None, dtype=pl.Float64))
                             .alias(c) for cols in REF_DEPENDENT.values() for c in cols])
    shrink = [s for s in specs if s.shrink]
    if shrink:
        pri = G.group_by("cell").agg(*[e for s in shrink for e in _bb_aggs(s)])
        pri = pri.select("cell", *[e for s in shrink for e in (
            alpha_expr(s.id, cfg).alias(f"{s.id}__alpha"), pl.col(f"{s.id}__m"))])
        G = G.join(pri, on="cell", how="left", maintain_order="left")
    vals = []
    for s in specs:
        if s.shrink:
            k, n, a, m = (pl.col(f"{s.id}__{x}") for x in ("k", "n", "alpha", "m"))
            v = pl.when(_fit_mask(s) & (n > 0) & k.is_not_null()).then((k + a * m) / (n + a))
        else:
            v = pl.when(_fit_mask(s) & pl.col(s.id).is_finite()).then(pl.col(s.id))
        vals.append(v.alias(f"_v_{s.id}"))
    G = G.select("cell", *vals, *[pl.col(f"{s.id}__{x}") for s in shrink for x in ("alpha", "m")])
    aggs = []
    for s in specs:
        v = pl.col(f"_v_{s.id}")
        aggs += [v.count().cast(pl.Int64).alias(f"{s.id}|n"), v.median().alias(f"{s.id}|median"),
                 (v - v.median()).abs().median().alias(f"{s.id}|mad"),
                 v.quantile(0.25, interpolation="linear").alias(f"{s.id}|q25"),
                 v.quantile(0.75, interpolation="linear").alias(f"{s.id}|q75"), v.std(ddof=1).alias(f"{s.id}|sd")]
        if s.shrink:
            aggs += [pl.col(f"{s.id}__alpha").first().alias(f"{s.id}|alpha"),
                     pl.col(f"{s.id}__m").first().alias(f"{s.id}|prior_mean")]
    A = G.group_by("cell").agg(aggs)
    eps = cfg.scale_eps
    mad = pl.col("mad") * cfg.mad_constant
    iqr = (pl.col("q75") - pl.col("q25")) / cfg.iqr_constant
    sd = pl.col("sd").fill_nan(None)
    out = []
    for s in specs:
        cols = ["n", "median", "mad", "q25", "q75", "sd"] + (["alpha", "prior_mean"] if s.shrink else [])
        part = A.select("cell", *[pl.col(f"{s.id}|{x}").alias(x) for x in cols])
        if not s.shrink:
            part = part.with_columns(pl.lit(None, dtype=pl.Float64).alias("alpha"),
                                     pl.lit(None, dtype=pl.Float64).alias("prior_mean"))
        out.append(part.with_columns(
            pl.lit(s.id).alias("feature"),
            pl.when(mad > eps).then(mad).when(iqr > eps).then(iqr).when(sd > eps).then(sd).alias("scale_within"),
            pl.when(mad > eps).then(pl.lit("mad")).when(iqr > eps).then(pl.lit("iqr"))
            .when(sd > eps).then(pl.lit("std")).alias("method_within"),
        ).select("cell", "feature", "n", "median", "scale_within", "method_within", "alpha", "prior_mean"))
    return pl.concat(out).with_columns(pl.lit(level).alias("level"))


def _robust_stats(R: pl.DataFrame, cand: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    """Pick, per peer cell x feature, the first candidate level with >= min_defined defined values (else the
    last, national); its median / alpha / m; the scale of that level, else of the next level up with a
    non-zero scale ('parent_<method>'), else 1 ('unit'). Reference-dependent features take median, n and prior from
    the peer cell itself (values against its reference); only when that cell has no spread at all do they borrow
    the scale of the next level up, computed over the members' own values (the statistic's typical spread)."""
    long = pl.concat([_level_stats(R, lv, cfg) for lv in LEVELS])
    refdep = pl.col("feature").is_in(list(REF_DEPENDENT))
    X = cand.join(long, on=["level", "cell"], how="left").filter(~refdep | (pl.col("r") == 0))
    dep = tuple(s for s in cfg.robust if s.id in REF_DEPENDENT)
    if dep:
        own = pl.concat([_level_stats(R, lv, cfg, dep, consistent=False) for lv in LEVELS[1:]])
        X = pl.concat([X, cand.filter(pl.col("r") > 0).join(own, on=["level", "cell"], how="inner")
                       .select(X.columns)])
    X = X.sort("peer_cell", "feature", "r")
    first_ok = pl.col("r").filter(pl.col("n") >= cfg.min_defined).min().fill_null(pl.col("r").max())
    r_star = pl.when(pl.col("feature").first().is_in(list(REF_DEPENDENT))).then(pl.lit(0, dtype=pl.Int32)) \
        .otherwise(first_ok)
    at = pl.col("r") == r_star
    has = (pl.col("r") >= r_star) & pl.col("scale_within").is_not_null()
    sel = X.group_by("peer_cell", "feature", maintain_order=True).agg(
        r_star.alias("_r"), pl.col("level").filter(at).first(), pl.col("cell").filter(at).first().alias("stat_cell"),
        *[pl.col(x).filter(at).first() for x in ("n", "median", "alpha", "prior_mean")],
        pl.col("scale_within").filter(has).first().alias("scale"),
        pl.col("method_within").filter(has).first().alias("_m"), pl.col("r").filter(has).first().alias("_rs"))
    sel = sel.with_columns(
        pl.when(pl.col("scale").is_null()).then(pl.lit("unit"))
        .when(pl.col("_rs") > pl.col("_r")).then(pl.lit("parent_") + pl.col("_m"))
        .otherwise(pl.col("_m")).alias("scale_method"),
        pl.col("scale").fill_null(1.0), pl.col("n").fill_null(0))
    return sel.select(STATS_COLS).sort("peer_cell", "feature")


def _fit_pm(R: pl.DataFrame, cand: pl.DataFrame, target: str, covs: tuple[str, ...],
            cfg: FeatureConfig) -> pl.DataFrame:
    """OLS of PMx__y on [1, covs] per fit cell (FEATURE_CATALOG PM2/PM3); the fit cell is the first candidate
    level with n >= max(min_n, min_n_per_term x (len(covs) + 1)) defined reference providers, else national.
    Missing covariates are imputed with the fit cell's median (national median, else 0)."""
    ycol = f"{target}__y"
    D = R.filter(pl.col(ycol).is_finite())
    min_n = max(cfg.pm_min_n, cfg.pm_min_n_per_term * (len(covs) + 1))
    counts = pl.concat([D.group_by(CELL_COL[lv]).agg(pl.len().cast(pl.Int64).alias("n"))
                        .rename({CELL_COL[lv]: "cell"}).with_columns(pl.lit(lv).alias("level")) for lv in LEVELS])
    X = (cand.join(counts, on=["level", "cell"], how="left").with_columns(pl.col("n").fill_null(0))
         .sort("peer_cell", "r"))
    r_star = pl.col("r").filter(pl.col("n") >= min_n).min().fill_null(pl.col("r").max())
    at = pl.col("r") == r_star
    choice = X.group_by("peer_cell", maintain_order=True).agg(
        pl.col("level").filter(at).first(), pl.col("cell").filter(at).first().alias("fit_cell"),
        pl.col("n").filter(at).first())
    nat = np.array([D[c].median() if D[c].null_count() < D.height else np.nan for c in covs], dtype=float)
    nat = np.where(np.isnan(nat), 0.0, nat)
    terms = ("intercept", *covs)
    models = []
    for lv in LEVELS:
        cells = choice.filter(pl.col("level") == lv)["fit_cell"].unique()
        col = CELL_COL[lv]
        sub = D.filter(pl.col(col).is_in(cells.implode())).sort(col, "npi")
        if sub.is_empty():
            continue
        y = sub[ycol].to_numpy().astype(float)
        Xm = np.column_stack([sub[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in covs]) \
            if covs else np.empty((sub.height, 0))
        bounds = sub.with_row_index("_i").group_by(col, maintain_order=True).agg(
            pl.col("_i").min().alias("lo"), pl.col("_i").max().alias("hi"),
            *[pl.col(c).cast(pl.Float64).median().alias(f"_med_{c}") for c in covs])
        for row in bounds.iter_rows():
            cell, lo, hi = row[:3]
            imp = np.array([np.nan if v is None else v for v in row[3:]], dtype=float)
            imp = np.where(np.isnan(imp), nat, imp)
            xc = Xm[lo:hi + 1]
            A = np.column_stack([np.ones(len(xc)), np.where(np.isnan(xc), imp, xc)])
            coef = np.linalg.lstsq(A, y[lo:hi + 1], rcond=None)[0]
            models.append(pl.DataFrame({"level": lv, "fit_cell": cell, "term": list(terms), "coef": coef,
                                        "impute": [None, *imp.tolist()]},
                                       schema={"level": pl.String, "fit_cell": pl.String, "term": pl.String,
                                               "coef": pl.Float64, "impute": pl.Float64}))
    mod = (pl.concat(models) if models else
           pl.DataFrame(schema={"level": pl.String, "fit_cell": pl.String, "term": pl.String, "coef": pl.Float64,
                                "impute": pl.Float64}))
    return (choice.join(mod, on=["level", "fit_cell"], how="left").with_columns(pl.lit(target).alias("target"))
            .select(PM_COLS).sort("peer_cell", "term"))


def _pm_residual(df: pl.DataFrame, pm: pl.DataFrame, target: str, covs: tuple[str, ...]) -> np.ndarray:
    """PMx = PMx__y - prediction of the peer cell's fitted model (NaN where y or the model is undefined)."""
    miss = [c for c in (*covs, f"{target}__y", "peer_cell") if c not in df.columns]
    if miss:
        raise FeatureError(f"{target}: raw statistics lack the regression inputs {miss}")
    m = pm.filter((pl.col("target") == target) & pl.col("term").is_not_null())
    cells = sorted(pm.filter(pl.col("target") == target)["peer_cell"].unique().to_list())
    terms = ["intercept", *covs]
    B = np.full((len(cells), len(terms)), np.nan)
    IMP = np.full((len(cells), len(terms)), np.nan)
    if m.height:
        ci = m["peer_cell"].replace_strict(cells, list(range(len(cells))), return_dtype=pl.Int64).to_numpy()
        ti = m["term"].replace_strict(terms, list(range(len(terms))), default=-1, return_dtype=pl.Int64).to_numpy()
        if (ti < 0).any():
            raise FeatureError(f"{target}: model terms {set(m['term'].to_list()) - set(terms)} not among covariates")
        B[ci, ti] = m["coef"].to_numpy()
        IMP[ci, ti] = m["impute"].fill_null(np.nan).to_numpy()
    idx = df["peer_cell"].replace_strict(cells, list(range(len(cells))), default=-1, return_dtype=pl.Int64).to_numpy()
    if (idx < 0).any():
        raise FeatureError(f"{target}: peer cells without a fitted model: {sorted(set(df['peer_cell'].to_numpy()[idx < 0]))[:3]}")
    pred = B[idx, 0].copy()
    for j, c in enumerate(covs, start=1):
        x = df[c].cast(pl.Float64).to_numpy().astype(float)
        pred += B[idx, j] * np.where(np.isnan(x), IMP[idx, j], x)
    return df[f"{target}__y"].cast(pl.Float64).to_numpy().astype(float) - pred


# ----------------------------------------------------------------------------------------------- step 4

def transform(raw, cohort, ps: PeerStats, refs: PeerReference, year: int,
              cfg: FeatureConfig | None = None) -> pl.DataFrame:
    """Model features (CONTRACTS §2.2 names) + raw__{ID} (+ raw__{ID}__k / __n for shrunk rates) per raw row,
    against peer statistics `ps` / `refs` fitted on reference providers. Pruned IDs are dropped. One-hot columns
    follow the reference vocabulary of the year; a category no reference provider has is all zeros."""
    cfg = cfg or FeatureConfig.load()
    if ps.year != year or refs.year != year:
        raise FeatureError(f"peer statistics of {ps.year} / reference of {refs.year} used for {year}")
    coh = _cohort(cohort, year)
    sources = [s.source for s in cfg.one_hot]
    D = (_df(raw, "raw").drop("peer_cell", "year", strict=False)
         .join(coh.select("npi", "peer_cell", "peer_level", *sources), on="npi", how="left").sort("npi"))
    _need_cells(D, "raw rows (each needs a cohort row with a peer cell)")
    cells = sorted(ps.stats["peer_cell"].unique().to_list())
    ci = D["peer_cell"].replace_strict(cells, list(range(len(cells))), default=-1, return_dtype=pl.Int64).to_numpy()
    if (ci < 0).any():
        raise FeatureError(f"peer cells without fitted statistics: {sorted(set(D['peer_cell'].to_numpy()[ci < 0]))[:3]}")
    st = {}
    for name in ("median", "scale", "alpha", "prior_mean"):
        w = ps.stats.pivot(on="feature", index="peer_cell", values=name).sort("peer_cell")
        st[name] = {c: w[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in w.columns if c != "peer_cell"}
    resid = {t: _pm_residual(D, ps.pm, t, ps.covariates) for t in PM_TARGETS}
    di0 = D["DI0"].to_numpy()

    def arr(col: str) -> np.ndarray:
        return D[col].cast(pl.Float64).fill_null(np.nan).to_numpy().astype(float)

    model, rawcols = [], []
    for s in cfg.specs:
        if s.kind == "robust_z":
            r = resid[s.id] if s.id in PM_TARGETS else arr(s.id)
            if s.id not in st["median"]:
                raise FeatureError(f"{s.id}: no fitted peer statistics")
            if s.shrink:
                k, n = arr(f"{s.id}__k"), arr(f"{s.id}__n")
                a, m = st["alpha"][s.id][ci], st["prior_mean"][s.id][ci]
                with np.errstate(invalid="ignore", divide="ignore"):
                    v = np.where(n > 0, (k + a * m) / (n + a), np.nan)
            else:
                v = r
            with np.errstate(invalid="ignore", divide="ignore"):
                z = np.clip((v - st["median"][s.id][ci]) / st["scale"][s.id][ci], -cfg.z_clip, cfg.z_clip)
            z = np.where(np.isfinite(z), z, 0.0)
            if s.fit_on == "dme_referrers":
                z = np.where(di0 == 1, 0.0, z)          # structural zero -> peer median; DI0 carries it (UP0 rule)
            model.append(pl.Series(s.id, z))
            rawcols.append(pl.Series(f"{RAW}{s.id}", r).fill_nan(None))
            if s.shrink:
                rawcols += [pl.Series(f"{RAW}{s.id}__k", k).fill_nan(None), pl.Series(f"{RAW}{s.id}__n", n).fill_nan(None)]
        elif s.kind == "indicator":
            model.append(D[s.id].cast(pl.Int8))
        elif s.kind == "context_raw":
            if s.id in D.columns:
                model.append(D[s.id].cast(pl.Float64))
            elif not s.optional:
                raise FeatureError(f"{s.id}: missing from the raw statistics")
        else:
            vals = D.select(_sanitise(pl.col(s.source))).to_series()
            model += [(vals == u).cast(pl.Int8).alias(f"{s.id}__{u}") for u in refs.vocab[s.id]]
    model = [c for c in model if c.name.split("__")[0] not in cfg.pruned]
    keys = D.select("npi", pl.lit(year, dtype=pl.Int32).alias("year"), "peer_cell")
    return keys.with_columns(*model, *rawcols)


# ----------------------------------------------------------------------------------------------- helpers

def model_columns(columns: Iterable[str]) -> list[str]:
    """The model feature columns of a feature table (no keys, no raw__ statistics)."""
    return [c for c in columns if c not in KEYS and not c.startswith(RAW)]


def empty_frame(paths: Paths, dataset: str, year: int) -> pl.DataFrame:
    """An empty harmonised frame with the schema columns (e.g. a year without DMEPOS files)."""
    spec = load_schema(paths)[dataset]["columns"]
    return pl.DataFrame(schema={**{c: DTYPES[s["dtype"]] for c, s in spec.items()}, "year": pl.Int32})


def load_frames(paths: Paths, year: int) -> tuple[dict[str, pl.LazyFrame], dict[str, bool]]:
    """That year's harmonised frames (lazy); DMEPOS files may be absent (empty frames). NPPES extract if any."""
    frames, present = {}, {}
    for ds in DATASETS:
        f = paths.interim / ds / f"{year}.parquet"
        present[ds] = f.is_file()
        if present[ds]:
            frames[ds] = pl.scan_parquet(f)
        elif ds in ("dme_s", "dme_p"):
            frames[ds] = empty_frame(paths, ds, year).lazy()
        else:
            raise FeatureError(f"{year}: {paths.rel(f) if f.parent.exists() else f} missing (run phase1 harmonise)")
    nppes = sorted((paths.interim / "nppes").glob("*.parquet")) if (paths.interim / "nppes").is_dir() else []
    if len(nppes) > 1:
        raise FeatureError(f"more than one NPPES extract in data/interim/nppes: {[p.name for p in nppes]}")
    frames["nppes"] = pl.scan_parquet(nppes[0]) if nppes else None
    present["nppes"] = bool(nppes)
    return frames, present


def _write(df: pl.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def build_features(paths: Paths, years: Iterable[int] | None = None) -> dict:
    """C5 for the given years (default: every cohort year), one year at a time, over in-cohort providers.
    Reads only data/processed/cohort.parquet and data/interim (never labels). Writes features/{year}.parquet
    (+ .meta.json) and peer/{year}/{codes,stats,pm_models}.parquet; returns the per-year summaries."""
    cfg = FeatureConfig.load(paths)
    cpath = paths.processed / "cohort.parquet"
    if not cpath.is_file():
        raise FeatureError("data/processed/cohort.parquet missing (run phase2 cohort first)")
    have = set(pl.scan_parquet(cpath).select(pl.col("year").unique()).collect()["year"].to_list())
    years = sorted(have) if years is None else sorted({int(y) for y in years})
    if not years or set(years) - have:
        raise FeatureError(f"years {sorted(set(years) - have) or years} not in cohort.parquet (has {sorted(have)})")
    out = {}
    for y in years:
        t0 = time.perf_counter()
        frames, present = load_frames(paths, y)
        coh = pl.scan_parquet(cpath).filter((pl.col("year") == y) & pl.col("in_cohort")).collect()
        refs = fit_peer_reference(frames, coh, y, cfg)
        raw = raw_statistics(frames, coh, refs, y, cfg)
        ps = fit_peer_stats(raw, coh, y, cfg)
        feat = transform(raw, coh, ps, refs, y, cfg)
        _write(feat, paths.processed / "features" / f"{y}.parquet")
        peer = paths.processed / "peer" / str(y)
        _write(refs.codes, peer / "codes.parquet")
        _write(ps.stats, peer / "stats.parquet")
        _write(ps.pm, peer / "pm_models.parquet")
        meta = _meta(feat, raw, ps, refs, cfg, present, time.perf_counter() - t0)
        meta["synthetic_inputs"] = provenance.inputs_synthetic(paths.interim / ds / f"{y}.parquet" for ds in DATASETS)
        (paths.processed / "features" / f"{y}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        out[y] = meta
    return out


def _meta(feat: pl.DataFrame, raw: pl.DataFrame, ps: PeerStats, refs: PeerReference, cfg: FeatureConfig,
          present: dict, seconds: float) -> dict:
    per = {}
    for s in cfg.robust:
        st = ps.stats.filter(pl.col("feature") == s.id)
        r = feat[f"{RAW}{s.id}"]
        entry = {
            "undefined": int(r.null_count()), "undefined_share": round(r.null_count() / max(feat.height, 1), 6),
            "clipped": int((feat[s.id].abs() >= cfg.z_clip).sum()) if s.id in feat.columns else None,
            "stats_level": dict(sorted(st.group_by("level").len().iter_rows())),
            "scale_method": dict(sorted(st.group_by("scale_method").len().iter_rows())),
        }
        if s.shrink:
            entry["alpha_min_median_max"] = [st["alpha"].min(), st["alpha"].median(), st["alpha"].max()]
        per[s.id] = entry
    cx7 = "CX7" in feat.columns
    return {
        "year": ps.year, "rows": feat.height, "reference_providers": int(refs.cells["n_reference"].sum()),
        "peer_cells": refs.cells.height, "inputs_present": present,
        "not_computable": {} if cx7 else {"CX7": "no NPPES extract in data/interim/nppes/*.parquet (HANDOFF §10.4)"},
        "pruned": list(cfg.pruned), "dme_high_risk_ranges": [list(r) for r in cfg.high_risk],
        "pm_covariates": list(ps.covariates), "pm_dropped_covariates": list(ps.dropped_covariates),
        "pm_fit_level": {t: dict(sorted(ps.pm.filter(pl.col("target") == t).unique("peer_cell")
                                        .group_by("level").len().iter_rows())) for t in PM_TARGETS},
        "structural": {"UP0_zero": int((raw["UP0"] == 0).sum()), "DI0_one": int((raw["DI0"] == 1).sum()),
                       "PM4_missing": int((raw["PM4_missing"] == 1).sum())},
        "dme_s_rows_without_dme_p": int(((raw["DI0"] == 1) & (raw["_dme_s_rows"] > 0)).sum()),
        "cx8_quantiles": {str(q): raw["CX8"].quantile(q, interpolation="linear") for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
        "features": per, "model_columns": model_columns(feat.columns), "seconds": round(seconds, 2),
    }


def load_features(paths: Paths, years: Iterable[int]) -> pl.DataFrame:
    """Feature tables of several years stacked; one-hot columns absent in a year (a category without reference
    providers that year) are 0."""
    cfg = FeatureConfig.load(paths)
    files = [paths.processed / "features" / f"{int(y)}.parquet" for y in years]
    missing = [paths.rel(f) for f in files if not f.is_file()]
    if missing:
        raise FeatureError(f"feature tables missing: {missing}")
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    hot = tuple(f"{s.id}__" for s in cfg.one_hot)
    return df.with_columns(pl.col(c).fill_null(0) for c in df.columns if c.startswith(hot))


def prune_candidates(features_df: pl.DataFrame, groups: Mapping[str, str] | None = None,
                     cfg: FeatureConfig | None = None) -> pl.DataFrame:
    """Within-group feature pairs with |Spearman rho| > prune.spearman_abs (FEATURE_CATALOG near-duplicates).
    One-hot context columns are left out (complementary by construction). Pruning itself is decided at G2."""
    cfg = cfg or FeatureConfig.load()
    cols = model_columns(features_df.columns)
    groups = dict(groups) if groups is not None else groups_mod.feature_groups(cols)
    hot = tuple(f"{s.id}__" for s in cfg.one_hot)
    use = [c for c in cols if c in groups and not c.startswith(hot)]
    rows = []
    for g in groups_mod.GROUP_ORDER:
        members = [c for c in use if groups[c] == g]
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                pair = features_df.select(a, b).drop_nulls()
                rho = pair.select(pl.corr(a, b, method="spearman")).item() if pair.height > 2 else None
                if rho is not None and np.isfinite(rho) and abs(rho) > cfg.prune_rho:
                    rows.append({"group": g, "feature_a": a, "feature_b": b, "rho": float(rho)})
    return pl.DataFrame(rows, schema={"group": pl.String, "feature_a": pl.String, "feature_b": pl.String,
                                      "rho": pl.Float64})
