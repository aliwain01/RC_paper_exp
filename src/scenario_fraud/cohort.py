"""C4 Cohort and peer cells (DESIGN §7.2; CONTRACTS §2.1).

harmonised cms_ps + cms_prov, one year at a time -> data/processed/cohort.parquet (+ cohort.meta.json)

One row per (npi, year) found in either Part B file. A row is in the cohort iff the NPI is in both files,
has entity type and specialty, and bills >= `cohort.min_nondrug_services` visible non-drug services;
every other row carries `cohort_exclusion_reason`. DMEPOS-only referrers are out of scope (DESIGN §7.2).

Peer cells (label-free; counts are in-cohort providers of the cell):
  spec_ent_div  "{year}|{ent}|{spec}|{division}"  if that cell has >= peer.refine_min providers
  spec_ent      "{year}|{ent}|{spec}"             else if that cell has >= peer.min_cell providers
  ent           "{year}|{ent}"                    otherwise (no further fallback; undersized ent cells are
                                                  listed in the meta file)
Rows outside the cohort get the cell their attributes map to, so they can be featurised against it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import polars as pl

from .config import load_config, load_yaml
from .paths import Paths

ATTRS = ("ent_cd", "prvdr_type", "state", "ruca", "mdcr_prtcptg_ind")
PS_COLS = ("npi", "hcpcs_cd", "place_of_srvc", "hcpcs_drug_ind", "tot_srvcs", *ATTRS)
PS_GRAIN = ("npi", "hcpcs_cd", "place_of_srvc")
LEVELS = ("spec_ent_div", "spec_ent", "ent")
SEP = "|"

NOT_IN_SERVICE_FILE = "not_in_service_file"      # by-Provider row only (e.g. every service row suppressed)
NOT_IN_PROVIDER_FILE = "not_in_provider_file"    # service rows but no by-Provider row
MISSING_PEER_ATTRIBUTES = "missing_peer_attributes"  # no entity type or specialty: no peer cell
BELOW_MIN_SERVICES = "below_min_nondrug_services"
REASONS = (NOT_IN_SERVICE_FILE, NOT_IN_PROVIDER_FILE, MISSING_PEER_ATTRIBUTES, BELOW_MIN_SERVICES)

COLUMNS = (
    "npi", "year", "ent_cd", "prvdr_type", "state", "ruca", "mdcr_prtcptg_ind", "census_division", "ruca_class",
    "nondrug_srvcs", "prov_tot_benes", "in_cohort", "cohort_exclusion_reason",
    "cell_ent", "cell_spec_ent", "cell_spec_ent_div", "n_cell_ent", "n_cell_spec_ent", "n_cell_spec_ent_div",
    "peer_cell", "peer_level",
)


class CohortError(RuntimeError):
    pass


@dataclass(frozen=True)
class CohortConfig:
    min_nondrug_services: float
    refine_min: int
    min_cell: int
    ruca_classes: dict[str, tuple[int, int]]
    divisions: dict[str, str]          # state code -> division name
    other: str

    @classmethod
    def load(cls, paths: Paths) -> "CohortConfig":
        c = load_config("experiment", paths)["cohort"]
        try:
            peer = c["peer"]
            cfg = cls(
                min_nondrug_services=float(c["min_nondrug_services"]),
                refine_min=int(peer["refine_min"]),
                min_cell=int(peer["min_cell"]),
                ruca_classes={k: (int(v[0]), int(v[1])) for k, v in c["ruca_classes"].items()},
                **_load_divisions(paths),
            )
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise CohortError(f"configs/experiment.yaml cohort section incomplete or malformed: {exc!r}") from exc
        if not 1 <= cfg.min_cell <= cfg.refine_min:
            raise CohortError(f"need 1 <= peer.min_cell ({cfg.min_cell}) <= peer.refine_min ({cfg.refine_min})")
        spans = sorted(cfg.ruca_classes.values())
        if any(lo > hi for lo, hi in spans) or any(a[1] >= b[0] for a, b in zip(spans, spans[1:])):
            raise CohortError(f"cohort.ruca_classes ranges must be ordered and disjoint: {cfg.ruca_classes}")
        if "unknown" in cfg.ruca_classes:
            raise CohortError("`unknown` is reserved for unclassifiable RUCA codes")
        return cfg


def _load_divisions(paths: Paths) -> dict:
    doc = load_yaml(paths.configs / "census_divisions.yaml")
    divs, other = doc["divisions"], doc["other"]
    mapping: dict[str, str] = {}
    for div, states in divs.items():
        for s in states:
            if not (isinstance(s, str) and len(s) == 2 and s.isalpha() and s.isupper()):
                raise CohortError(f"census_divisions.yaml: {div!r} lists {s!r}, not a two-letter code")
            if s in mapping:
                raise CohortError(f"census_divisions.yaml: {s} listed under {mapping[s]!r} and {div!r}")
            mapping[s] = div
    if len(divs) != 9 or len(mapping) != 51 or other in divs:
        raise CohortError(f"census_divisions.yaml: expected 9 divisions over 51 codes (50 states + DC) and a "
                          f"distinct `other`; got {len(divs)} divisions, {len(mapping)} codes")
    return {"divisions": mapping, "other": other}


# ----------------------------------------------------------------------------------------------- expressions

def census_division_expr(cfg: CohortConfig, col: str = "state") -> pl.Expr:
    """State code -> Census division; territories, military, foreign, unknown and null -> cfg.other."""
    return (pl.col(col).str.strip_chars().str.to_uppercase()
            .replace_strict(cfg.divisions, default=cfg.other, return_dtype=pl.String).fill_null(cfg.other))


def ruca_class_expr(cfg: CohortConfig, col: str = "ruca") -> pl.Expr:
    """Primary RUCA = integer part of the code ("10.1" -> 10), classed by cfg.ruca_classes, else unknown."""
    primary = pl.col(col).str.strip_chars().str.extract(r"^(\d+)(?:\.\d+)?$", 1).cast(pl.Int32, strict=False)
    expr = pl.lit("unknown")
    for name, (lo, hi) in reversed(cfg.ruca_classes.items()):
        expr = pl.when(primary.is_between(lo, hi)).then(pl.lit(name)).otherwise(expr)
    return expr


# ----------------------------------------------------------------------------------------------- one year

def _check_inputs(ps: pl.LazyFrame, prov: pl.LazyFrame, year: int) -> None:
    """Fail on violated assumptions: missing columns, foreign years, drug flags other than Y/N, null or
    negative service counts, duplicate grain rows, duplicate by-Provider NPIs, '|' inside cell parts."""
    ps_cols, prov_cols = ps.collect_schema().names(), prov.collect_schema().names()
    miss = [c for c in (*PS_COLS, "year") if c not in ps_cols]
    miss += [f"cms_prov.{c}" for c in ("npi", "tot_benes", "year") if c not in prov_cols]
    if miss:
        raise CohortError(f"{year}: harmonised columns missing: {miss}")
    def sep_rows(cols: list[str]) -> pl.Expr:
        hits = [pl.col(c).str.contains(SEP, literal=True).fill_null(False) for c in ("ent_cd", "prvdr_type") if c in cols]
        return pl.sum_horizontal(hits).gt(0).sum().alias("sep") if hits else pl.lit(0).alias("sep")

    s = ps.select(
        (pl.col("year") != year).sum().alias("wrong_year"),
        (~pl.col("hcpcs_drug_ind").is_in(["Y", "N"]) | pl.col("hcpcs_drug_ind").is_null()).sum().alias("bad_drug"),
        (pl.col("tot_srvcs").is_null() | (pl.col("tot_srvcs") < 0)).sum().alias("bad_srvcs"),
        pl.col("npi").is_null().sum().alias("null_npi"),
        sep_rows(ps_cols),
    ).collect().row(0, named=True)
    dup = ps.group_by(PS_GRAIN).len().filter(pl.col("len") > 1).select(pl.len()).collect().item()
    p = prov.select(
        (pl.col("year") != year).sum().alias("wrong_year"),
        pl.col("npi").is_null().sum().alias("null_npi"),
        (pl.len() - pl.col("npi").n_unique()).alias("dup_npi"),
        sep_rows(prov_cols),
    ).collect().row(0, named=True)
    problems = []
    if s["wrong_year"] or p["wrong_year"]:
        problems.append(f"rows whose `year` is not {year}: cms_ps {s['wrong_year']}, cms_prov {p['wrong_year']}")
    if s["bad_drug"]:
        problems.append(f"{s['bad_drug']} cms_ps rows with hcpcs_drug_ind not in {{Y, N}}")
    if s["bad_srvcs"]:
        problems.append(f"{s['bad_srvcs']} cms_ps rows with null or negative tot_srvcs")
    if s["null_npi"] or p["null_npi"]:
        problems.append(f"null NPIs: cms_ps {s['null_npi']}, cms_prov {p['null_npi']}")
    if s["sep"] or p["sep"]:
        problems.append(f"{SEP!r} (peer-cell id separator) inside ent_cd/prvdr_type: cms_ps {s['sep']} rows, "
                        f"cms_prov {p['sep']} rows")
    if dup:
        problems.append(f"{dup} duplicated cms_ps grain keys {PS_GRAIN}")
    if p["dup_npi"]:
        problems.append(f"{p['dup_npi']} duplicated NPIs in cms_prov")
    if problems:
        raise CohortError(f"{year}: " + "; ".join(problems))


def cohort_year(ps: pl.LazyFrame | pl.DataFrame, prov: pl.LazyFrame | pl.DataFrame, year: int,
                cfg: CohortConfig) -> tuple[pl.DataFrame, dict]:
    """Cohort rows and peer cells for one year, from that year's harmonised cms_ps and cms_prov frames.
    Returns (frame with COLUMNS sorted by npi, summary dict for the meta file)."""
    ps, prov = ps.lazy(), prov.lazy()
    _check_inputs(ps, prov, year)

    # Provider attributes: first non-null in grain order (constant within npi-year in CMS files; any
    # conflict is counted, not hidden). Non-drug services over visible rows only (FEATURE_CATALOG OU1).
    svc = (
        ps.select(PS_COLS).sort(PS_GRAIN)
        .group_by("npi")
        .agg(
            *[pl.col(a).drop_nulls().first() for a in ATTRS],
            *[(pl.col(a).drop_nulls().n_unique() > 1).alias(f"_conflict_{a}") for a in ATTRS],
            pl.col("tot_srvcs").filter(pl.col("hcpcs_drug_ind") != "Y").sum().alias("nondrug_srvcs"),
        )
        .with_columns(pl.lit(True).alias("_in_ps"))
    )
    prov_cols = prov.collect_schema().names()
    pv = prov.select(
        "npi", pl.col("tot_benes").alias("prov_tot_benes"),
        *[pl.col(a).alias(f"_prov_{a}") for a in ATTRS if a in prov_cols],
        pl.lit(True).alias("_in_prov"),
    )
    df = svc.join(pv, on="npi", how="full", coalesce=True).collect()
    conflicts = {a: int(df[f"_conflict_{a}"].sum() or 0) for a in ATTRS}
    in_ps, in_prov = pl.col("_in_ps").fill_null(False), pl.col("_in_prov").fill_null(False)
    # NPIs without service rows take their attributes from the by-Provider file (optional columns there).
    df = df.with_columns(
        *[pl.coalesce(a, f"_prov_{a}").alias(a) if f"_prov_{a}" in df.columns else pl.col(a) for a in ATTRS],
        pl.col("nondrug_srvcs").fill_null(0.0),
        pl.lit(year, dtype=pl.Int32).alias("year"),
    )

    reason = (
        pl.when(~in_ps).then(pl.lit(NOT_IN_SERVICE_FILE))
        .when(~in_prov).then(pl.lit(NOT_IN_PROVIDER_FILE))
        .when(pl.col("ent_cd").is_null() | pl.col("prvdr_type").is_null()).then(pl.lit(MISSING_PEER_ATTRIBUTES))
        .when(pl.col("nondrug_srvcs") < cfg.min_nondrug_services).then(pl.lit(BELOW_MIN_SERVICES))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    y = pl.lit(str(year))
    df = df.with_columns(
        census_division_expr(cfg).alias("census_division"),
        ruca_class_expr(cfg).alias("ruca_class"),
        reason.alias("cohort_exclusion_reason"),
    ).with_columns(
        pl.col("cohort_exclusion_reason").is_null().alias("in_cohort"),
        pl.concat_str([y, "ent_cd"], separator=SEP).alias("cell_ent"),
        pl.concat_str([y, "ent_cd", "prvdr_type"], separator=SEP).alias("cell_spec_ent"),
        pl.concat_str([y, "ent_cd", "prvdr_type", "census_division"], separator=SEP).alias("cell_spec_ent_div"),
    )
    df = assign_peer_cells(df, cfg)
    df = df.select(COLUMNS).sort("npi")
    return df, _summary(df, cfg, conflicts)


def assign_peer_cells(df: pl.DataFrame, cfg: CohortConfig) -> pl.DataFrame:
    """Add n_cell_* (in-cohort providers per cell) and the chosen peer_cell / peer_level (DESIGN §7.2)."""
    for lvl in ("ent", "spec_ent", "spec_ent_div"):
        cell = f"cell_{lvl}"
        counts = df.filter(pl.col("in_cohort") & pl.col(cell).is_not_null()).group_by(cell).agg(
            pl.len().cast(pl.Int64).alias(f"n_{cell}"))
        df = df.drop(f"n_{cell}", strict=False).join(counts, on=cell, how="left").with_columns(
            pl.when(pl.col(cell).is_not_null()).then(pl.col(f"n_{cell}").fill_null(0)).alias(f"n_{cell}"))
    pick = (
        pl.when(pl.col("n_cell_spec_ent_div") >= cfg.refine_min).then(pl.struct(
            peer_cell=pl.col("cell_spec_ent_div"), peer_level=pl.lit("spec_ent_div")))
        .when(pl.col("n_cell_spec_ent") >= cfg.min_cell).then(pl.struct(
            peer_cell=pl.col("cell_spec_ent"), peer_level=pl.lit("spec_ent")))
        .when(pl.col("cell_ent").is_not_null()).then(pl.struct(
            peer_cell=pl.col("cell_ent"), peer_level=pl.lit("ent")))
    )
    return df.with_columns(pick.alias("_pick")).unnest("_pick")


def _summary(df: pl.DataFrame, cfg: CohortConfig, conflicts: dict) -> dict:
    inc = df.filter("in_cohort")
    ent_cells = inc.group_by("cell_ent").len().sort("cell_ent")
    return {
        "rows": df.height,
        "in_cohort": inc.height,
        "exclusion_reasons": {r: int((df["cohort_exclusion_reason"] == r).sum()) for r in REASONS},
        "peer_level_mix": {lvl: int((inc["peer_level"] == lvl).sum()) for lvl in LEVELS},
        "n_peer_cells": inc["peer_cell"].n_unique(),
        "smallest_peer_cell": None if inc.is_empty() else int(inc.group_by("peer_cell").len()["len"].min()),
        "undersized_ent_cells": {r["cell_ent"]: r["len"] for r in ent_cells.iter_rows(named=True)
                                 if r["len"] < cfg.min_cell},
        "attribute_conflicts": conflicts,
        "census_division_other": int((inc["census_division"] == cfg.other).sum()),
        "ruca_unknown": int((inc["ruca_class"] == "unknown").sum()),
    }


# ----------------------------------------------------------------------------------------------- all years

def available_years(paths: Paths) -> list[int]:
    """Years with harmonised Part B files. A year present in only one of the two files is an error."""
    found = {ds: {int(p.stem) for p in (paths.interim / ds).glob("[0-9][0-9][0-9][0-9].parquet")}
             for ds in ("cms_ps", "cms_prov")}
    lonely = found["cms_ps"] ^ found["cms_prov"]
    if lonely:
        raise CohortError(f"years with only one of cms_ps / cms_prov harmonised: {sorted(lonely)}")
    return sorted(found["cms_ps"])


def build_cohort(paths: Paths, years: Iterable[int] | None = None) -> pl.DataFrame:
    """C4 for the given years (default: every harmonised year), one year at a time. Writes exactly those
    years to data/processed/cohort.parquet and a summary to cohort.meta.json; returns the frame."""
    cfg = CohortConfig.load(paths)
    years = available_years(paths) if years is None else sorted({int(y) for y in years})
    if not years:
        raise CohortError("no harmonised Part B years found under data/interim/{cms_ps,cms_prov}")
    parts, summary = [], {}
    for y in years:
        f_ps, f_prov = (paths.interim / ds / f"{y}.parquet" for ds in ("cms_ps", "cms_prov"))
        missing = [paths.rel(f) for f in (f_ps, f_prov) if not f.is_file()]
        if missing:
            raise CohortError(f"{y}: harmonised files missing: {missing}")
        df, summary[y] = cohort_year(pl.scan_parquet(f_ps), pl.scan_parquet(f_prov), y, cfg)
        parts.append(df)
    out = pl.concat(parts)
    path = paths.processed / "cohort.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    out.write_parquet(tmp)
    tmp.replace(path)
    meta = {
        "years": years,
        "config": {"min_nondrug_services": cfg.min_nondrug_services, "refine_min": cfg.refine_min,
                   "min_cell": cfg.min_cell, "ruca_classes": cfg.ruca_classes},
        "per_year": {str(y): s for y, s in summary.items()},
    }
    path.with_name("cohort.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out
