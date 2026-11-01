"""C6 scheme injection engine (DESIGN §7.7; EXPERIMENT_PLAN §6; CONTRACTS §5).

1. plan(paths, draw): clean provider-years (in cohort, never positive under any tier), stratified by specialty and
   restricted to those the scheme applies to, get one scheme (or a pair) and an intensity; test part = test years,
   per_cell per scheme × intensity and per_pair per pair; train part = train years, `share` of provider-years split
   evenly over the same cells. Distractors: already above the peer 95th percentile in another group.
2. apply_year(...): edits the harmonised service-level rows of the planned providers (schemes in a fixed order),
   updates the summary-file totals by the change in visible amounts, then re-applies CMS suppression to the edited
   rows. Returns edited frames for the year plus an edit log.
3. build(paths, draw): for every planned year, the edited year is featurised with the Phase 2 code: injected
   providers are flagged in the cohort (they never count toward peer cells and are excluded from every peer
   statistic), so peer statistics come from real providers only (DESIGN §7.7 step 5).
Outputs under data/processed/injection/draw{d}/.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from . import cohort as cohort_mod
from . import features, splits
from .config import load_config
from .groups import GROUP_ORDER
from .paths import Paths

SCHEME_ORDER = ("upcoding", "overutilization", "atypical_mix", "charge_inflation", "dme_mill")
AMT = {"avg_sbmtd_chrg": "tot_sbmtd_chrg", "avg_mdcr_alowd_amt": "tot_mdcr_alowd_amt",
       "avg_mdcr_pymt_amt": "tot_mdcr_pymt_amt", "avg_mdcr_stdzd_amt": "tot_mdcr_stdzd_amt"}
DME_AMT = ("avg_suplr_sbmtd_chrg", "avg_suplr_mdcr_alowd_amt", "avg_suplr_mdcr_pymt_amt", "avg_suplr_mdcr_stdzd_amt")
PROV_ATTRS = ("ent_cd", "state", "ruca", "prvdr_type", "mdcr_prtcptg_ind", "zip5")
_PLAN, _CODES, _DME = 201, 202, 203


class InjectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class InjectionConfig:
    doc: dict

    @classmethod
    def load(cls, paths: Paths) -> "InjectionConfig":
        d = load_config("injection", paths)
        bad = [s for s in d["schemes"] if s not in SCHEME_ORDER]
        if bad:
            raise InjectionError(f"injection.yaml: unknown schemes {bad}")
        for s, v in d["schemes"].items():
            if not set(v["groups"]) <= set(GROUP_ORDER[1:]) or not set(v["spill"]) <= set(GROUP_ORDER[1:]):
                raise InjectionError(f"injection.yaml {s}: groups / spill must be scheme groups")
            if set(v["intensities"]) != {"low", "medium", "high"}:
                raise InjectionError(f"injection.yaml {s}: intensities need low, medium, high")
        for p in d["pairs"]:
            if len(p) != 2 or not set(p) <= set(d["schemes"]):
                raise InjectionError(f"injection.yaml: bad pair {p}")
        return cls(d)

    def __getitem__(self, k):
        return self.doc[k]

    def intensity(self, scheme: str, label: str) -> float:
        return float(self.doc["schemes"][scheme]["intensities"][label])

    def answer_groups(self, schemes) -> list[str]:
        return sorted({g for s in schemes for g in self.doc["schemes"][s]["groups"]})


def draw_dir(paths: Paths, draw: int) -> Path:
    return paths.processed / "injection" / f"draw{int(draw)}"


def _years(r) -> list[int]:
    return list(range(int(r[0]), int(r[1]) + 1))


def _h(npi: pl.Series, *salt: int) -> np.ndarray:
    return splits.npi_keys(npi, splits._salt(*salt))


def _str_keys(strings, *salt: int) -> np.ndarray:
    """Stable uint64 hash of arbitrary strings (blake2b; independent of library versions and row order)."""
    pre = "|".join(map(str, salt)).encode()
    return np.array([int.from_bytes(hashlib.blake2b(pre + b"|" + s.encode(), digest_size=8).digest(), "little")
                     for s in strings], dtype=np.uint64)


# ----------------------------------------------------------------------------------------------- plan

def eligibility(paths: Paths, years: list[int], icfg: InjectionConfig) -> pl.DataFrame:
    """One row per in-cohort, never-positive provider-year with the scheme applicability flags."""
    fcfg = features.FeatureConfig.load(paths)
    est = list(fcfg.code_sets["established"])
    coh = (pl.scan_parquet(paths.processed / "cohort.parquet").filter(pl.col("in_cohort") & pl.col("year").is_in(years))
           .select("npi", pl.col("year").cast(pl.Int32), "prvdr_type", "peer_cell").collect())
    folds = splits.load_folds(paths).select("npi", "fold", "strat_specialty", "ever_positive")
    parts = []
    for y in years:
        ps = pl.scan_parquet(paths.interim / "cms_ps" / f"{y}.parquet")
        estv = (ps.filter(pl.col("hcpcs_cd").is_in(est) & (pl.col("hcpcs_drug_ind") == "N")).group_by("npi")
                .agg(pl.col("tot_srvcs").sum().alias("est_srv")).collect())
        ds = paths.interim / "dme_s" / f"{y}.parquet"
        dp = paths.interim / "dme_p" / f"{y}.parquet"
        dme_s = pl.scan_parquet(ds).select("npi").unique().collect() if ds.is_file() else pl.DataFrame(schema={"npi": pl.String})
        dme_p = pl.scan_parquet(dp).select("npi").unique().collect() if dp.is_file() else pl.DataFrame(schema={"npi": pl.String})
        c = (coh.filter(pl.col("year") == y).join(estv, on="npi", how="left")
             .with_columns(pl.col("est_srv").fill_null(0.0),
                           pl.col("npi").is_in(dme_s["npi"].implode()).alias("has_dme_s"),
                           pl.col("npi").is_in(dme_p["npi"].implode()).alias("dme_referrer")))
        c = c.with_columns(pl.col("dme_referrer").cast(pl.Float64).mean().over("prvdr_type").alias("spec_dme_share"))
        parts.append(c)
    df = pl.concat(parts).join(folds, on="npi", how="inner").filter(~pl.col("ever_positive"))
    pool = icfg["pools"]
    return df.with_columns(
        (pl.col("est_srv") >= float(pool["upcoding_min_established"])).alias("ok_upcoding"),
        pl.lit(True).alias("ok_overutilization"), pl.lit(True).alias("ok_atypical_mix"),
        pl.lit(True).alias("ok_charge_inflation"),
        (pl.col("has_dme_s") | (pl.col("spec_dme_share") >= float(pool["dme_specialty_referrer_share"]))
         ).alias("ok_dme_mill"))


def _cells(icfg: InjectionConfig) -> list[tuple[tuple[str, ...], str]]:
    single = [((s,), lab) for s in SCHEME_ORDER if s in icfg["schemes"] for lab in ("low", "medium", "high")]
    pairs = [(tuple(sorted(p, key=SCHEME_ORDER.index)), icfg["pair_intensity"]) for p in icfg["pairs"]]
    return single + pairs


def _take_stratified(df: pl.DataFrame, n: int) -> pl.DataFrame:
    """n rows, specialty quotas by largest remainder over df's specialty sizes, first by `_h` within a specialty."""
    n = min(n, df.height)
    if n == 0:
        return df.head(0)
    sizes = df.group_by("strat_specialty").len().sort("strat_specialty")
    quota = sizes["len"].to_numpy() * n / df.height
    base = np.floor(quota).astype(int)
    base[np.lexsort((np.arange(len(quota)), -(quota - base)))[: n - base.sum()]] += 1
    alloc = pl.DataFrame({"strat_specialty": sizes["strat_specialty"], "_q": base})
    return (df.join(alloc, on="strat_specialty").sort(["strat_specialty", "_h"])
            .filter(pl.int_range(pl.len()).over("strat_specialty") < pl.col("_q")).drop("_q"))


def plan(paths: Paths, draw: int) -> pl.DataFrame:
    icfg = InjectionConfig.load(paths)
    seed = int(icfg["seed"])
    test_years, train_years = _years(icfg["test"]["years"]), _years(icfg["train"]["years"])
    elig = eligibility(paths, sorted({*test_years, *train_years}), icfg)
    cells = _cells(icfg)
    n_train_rows = (pl.scan_parquet(paths.processed / "cohort.parquet")
                    .filter(pl.col("in_cohort") & pl.col("year").is_in(train_years)).select(pl.len()).collect().item())
    per_train = max(1, round(float(icfg["train"]["share"]) * n_train_rows / len(cells)))
    rows = []
    for part, years in (("test", test_years), ("train", train_years)):
        pool = elig.filter(pl.col("year").is_in(years))
        pool = pool.with_columns(pl.Series("_h", _h(pool["npi"], seed, _PLAN, draw, 0 if part == "test" else 1)
                                           ^ pool["year"].cast(pl.UInt64).to_numpy()))
        used: set = set()
        for ci, (schemes, label) in enumerate(cells):
            n = (int(icfg["test"]["per_pair"]) if len(schemes) > 1 else int(icfg["test"]["per_cell"])) \
                if part == "test" else per_train
            ok = pl.all_horizontal([pl.col(f"ok_{s}") for s in schemes])
            avail = pool.filter(ok)
            if used:
                avail = avail.filter(~pl.concat_str(["npi", pl.col("year").cast(pl.String)], separator="|")
                                     .is_in(list(used)))
            got = _take_stratified(avail, n)
            used |= set((got["npi"] + "|" + got["year"].cast(pl.String)).to_list())
            rows.append(got.select("npi", "year", "fold", "strat_specialty", "peer_cell").with_columns(
                pl.lit(part).alias("part"), pl.lit(list(schemes)).alias("schemes"), pl.lit(label).alias("intensity"),
                pl.lit(len(schemes) > 1).alias("pair"), pl.lit(ci, dtype=pl.Int32).alias("cell"),
                pl.lit(n, dtype=pl.Int32).alias("cell_target")))
    p = pl.concat(rows).with_columns(pl.lit(int(draw), dtype=pl.Int32).alias("draw"))
    p = p.with_columns(*[pl.when(pl.col("schemes").list.contains(s))
                         .then(pl.col("intensity").replace_strict(
                             {lab: icfg.intensity(s, lab) for lab in ("low", "medium", "high")}, return_dtype=pl.Float64))
                         .otherwise(None).alias(f"x_{s}") for s in SCHEME_ORDER if s in icfg["schemes"]])
    p = p.with_columns(pl.col("schemes").map_elements(lambda s: icfg.answer_groups(list(s)),
                                                      return_dtype=pl.List(pl.String)).alias("answer_groups"))
    return add_distractors(paths, p, icfg)


def add_distractors(paths: Paths, p: pl.DataFrame, icfg: InjectionConfig) -> pl.DataFrame:
    """EXPERIMENT_PLAN §5.1: before injection, the provider's group score (max z of the group's direction-up
    features) is at or above the peer cell's `percentile` in some group outside the answer groups."""
    fcfg = features.FeatureConfig.load(paths)
    up = {g: [s.id for s in fcfg.robust if s.group == g and s.direction == "up"] for g in GROUP_ORDER[1:]}
    q = float(icfg["distractor"]["percentile"])
    out = []
    for (y,), g in p.group_by("year"):
        f = features.load_features(paths, [int(y)])
        f = f.with_columns(*[pl.max_horizontal([pl.col(c) for c in cols if c in f.columns]).alias(f"_g_{grp}")
                             for grp, cols in up.items() if any(c in f.columns for c in cols)])
        gcols = [c for c in f.columns if c.startswith("_g_")]
        f = f.with_columns(*[((pl.col(c).rank("max").over("peer_cell") / pl.len().over("peer_cell")) >= q).alias(c)
                             for c in gcols]).select("npi", *gcols)
        j = g.join(f, on="npi", how="left")
        hi = pl.concat_list([pl.when(pl.col(c).fill_null(False)).then(pl.lit(c[3:])).otherwise(None) for c in gcols]
                            ).list.drop_nulls()
        j = j.with_columns(hi.alias("_hi")).with_columns(
            pl.col("_hi").list.set_difference(pl.col("answer_groups")).alias("distractor_groups"))
        out.append(j.drop(gcols, "_hi").with_columns((pl.col("distractor_groups").list.len() > 0).alias("distractor")))
    return pl.concat(out).sort(["part", "cell", "year", "npi"])


# ----------------------------------------------------------------------------------------------- references

def reference_tables(frames: dict, coh: pl.DataFrame, refs: features.PeerReference,
                     fcfg: features.FeatureConfig, icfg: InjectionConfig) -> dict:
    """Label-free medians from reference providers (in cohort, not injected) used to price added rows. The year's
    frames may be lazy: only the aggregates are collected (memory)."""
    ref = coh.filter(pl.col("in_cohort") & ~pl.col("injected")).select("npi", "peer_cell").lazy()
    ps = frames["cms_ps"].lazy().join(ref, on="npi", how="inner").filter(pl.col("hcpcs_drug_ind") == "N")
    aggs = [pl.col(c).median().alias(c) for c in AMT] + [pl.len().alias("n")]
    cell = ps.group_by("peer_cell", "hcpcs_cd", "place_of_srvc").agg(aggs)
    nat_place = ps.group_by("hcpcs_cd", "place_of_srvc").agg(aggs)
    nat = (ps.group_by("hcpcs_cd").agg(
        *[pl.col(c).median().alias(c) for c in AMT], pl.col("npi").n_unique().alias("n_providers"),
        (pl.col("tot_srvcs") / pl.col("tot_benes")).median().alias("srv_per_bene"),
        (pl.col("tot_bene_day_srvcs") / pl.col("tot_srvcs")).median().alias("bds_per_srv"),
        pl.col("place_of_srvc").mode().first().alias("place")))
    rare_cell = refs.codes.filter(pl.col("level") != "national").select(
        "peer_cell", "hcpcs_cd", pl.col("provider_share"))
    cell, nat_place, nat = pl.collect_all([cell, nat_place, nat])
    out = {"cell": cell, "nat_place": nat_place, "nat": nat, "rare_cell": rare_cell}
    ds = frames["dme_s"].lazy().join(ref, on="npi", how="inner")
    hr = ds.filter(features.high_risk_expr(fcfg))
    out["dme_codes"] = (hr.group_by("hcpcs_cd").agg(
        *[pl.col(c).median().alias(c) for c in DME_AMT], pl.col("npi").n_unique().alias("n_providers"),
        (pl.col("tot_suplr_srvcs") / pl.col("tot_suplr_benes")).median().alias("srv_per_bene"),
        (pl.col("tot_suplr_clms") / pl.col("tot_suplr_srvcs")).median().alias("clms_per_srv"),
        pl.col("suplr_rentl_ind").mode().first().alias("rental"), pl.col("rbcs_lvl").mode().first().alias("rbcs_lvl"),
        pl.col("rbcs_id").mode().first().alias("rbcs_id"))
        .filter(pl.col("n_providers") >= int(icfg["transforms"]["dme"]["min_national_providers"])).sort("hcpcs_cd")
        .collect())
    spend = ds.group_by("npi", "peer_cell").agg((pl.col("tot_suplr_srvcs") * pl.col("avg_suplr_mdcr_pymt_amt")).sum()
                                                .alias("spend")).collect()
    out["dme_spend_cell"] = spend.group_by("peer_cell").agg(pl.col("spend").median().alias("med_spend"))
    out["dme_spend_nat"] = float(spend["spend"].median()) if spend.height else float("nan")
    return out


# ----------------------------------------------------------------------------------------------- transforms

def _cast_like(df: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    for c in schema.names():
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))
    return df.select([pl.col(c).cast(schema[c]) for c in schema.names()])


def _ps_sums(ps: pl.DataFrame) -> pl.DataFrame:
    return ps.group_by("npi").agg(pl.col("tot_srvcs").sum().alias("srv"), pl.col("hcpcs_cd").n_unique().alias("codes"),
                                  *[(pl.col("tot_srvcs") * pl.col(c)).sum().alias(t) for c, t in AMT.items()])


def _price(rows: pl.DataFrame, tabs: dict) -> pl.DataFrame:
    """Averages for new rows (npi, peer_cell, hcpcs_cd, place_of_srvc): peer cell → national (code, place) →
    national (code) medians."""
    out = rows.join(tabs["cell"].drop("n"), on=["peer_cell", "hcpcs_cd", "place_of_srvc"], how="left")
    for key, tab in ((["hcpcs_cd", "place_of_srvc"], tabs["nat_place"].drop("n")),
                     (["hcpcs_cd"], tabs["nat"].select("hcpcs_cd", *AMT))):
        out = out.join(tab.rename({c: f"{c}__f" for c in AMT}), on=key, how="left").with_columns(
            *[pl.coalesce(c, f"{c}__f").alias(c) for c in AMT]).drop([f"{c}__f" for c in AMT])
    miss = out.filter(pl.any_horizontal(pl.col(c).is_null() for c in AMT))
    if miss.height:
        raise InjectionError(f"no reference price for {miss['hcpcs_cd'].unique().to_list()[:5]}")
    return out


def _attrs(ps: pl.DataFrame) -> pl.DataFrame:
    keep = [c for c in PROV_ATTRS if c in ps.columns]
    return ps.sort("hcpcs_cd", "place_of_srvc").group_by("npi").agg(*[pl.col(c).drop_nulls().first() for c in keep])


def upcoding(ps: pl.DataFrame, who: pl.DataFrame, prov_benes: pl.DataFrame, tabs: dict, icfg: InjectionConfig
             ) -> pl.DataFrame:
    """who: npi, peer_cell, x (share p). Moves p of 99212-13 services (and proportional beneficiaries and
    beneficiary-days) to 99214-15 in the same place of service; existing target rows keep the provider's own
    averages; new rows are priced at peer-cell medians (EXPERIMENT_PLAN §6)."""
    t = icfg["transforms"]["upcoding"]
    src_codes, targets = list(t["from"]), dict(t["to"])
    sel = (pl.col("npi").is_in(who["npi"].implode()) & pl.col("hcpcs_cd").is_in(src_codes)
           & (pl.col("hcpcs_drug_ind") == "N"))
    moved = (ps.filter(sel).join(who.select("npi", "x"), on="npi")
             .with_columns((pl.col("tot_srvcs") * pl.col("x")).round(0).alias("m_srv"),
                           (pl.col("tot_benes") * pl.col("x")).round(0).cast(pl.Int64).alias("m_ben"),
                           (pl.col("tot_bene_day_srvcs") * pl.col("x")).round(0).alias("m_bds")))
    ps = (ps.join(moved.select("npi", "hcpcs_cd", "place_of_srvc", "m_srv", "m_ben", "m_bds"),
                  on=["npi", "hcpcs_cd", "place_of_srvc"], how="left", maintain_order="left")
          .with_columns((pl.col("tot_srvcs") - pl.col("m_srv").fill_null(0)).alias("tot_srvcs"),
                        (pl.col("tot_benes") - pl.col("m_ben").fill_null(0)).alias("tot_benes"),
                        (pl.col("tot_bene_day_srvcs") - pl.col("m_bds").fill_null(0)).alias("tot_bene_day_srvcs"))
          .drop("m_srv", "m_ben", "m_bds").filter(pl.col("tot_srvcs") > 0))
    tot = moved.group_by("npi", "place_of_srvc").agg(pl.col("m_srv").sum(), pl.col("m_ben").sum(), pl.col("m_bds").sum())
    codes = list(targets)
    adds = []
    for i, code in enumerate(codes):
        w = float(targets[code])
        if i < len(codes) - 1:
            a = tot.with_columns((pl.col("m_srv") * w).round(0).alias("a_srv"), (pl.col("m_ben") * w).round(0)
                                 .cast(pl.Int64).alias("a_ben"), (pl.col("m_bds") * w).round(0).alias("a_bds"))
        else:   # the last target takes the remainder so services are conserved
            prev = pl.concat(adds).group_by("npi", "place_of_srvc").agg(pl.col("a_srv").sum().alias("p_srv"),
                                                                         pl.col("a_ben").sum().alias("p_ben"),
                                                                         pl.col("a_bds").sum().alias("p_bds")) \
                if adds else tot.select("npi", "place_of_srvc").with_columns(pl.lit(0.0).alias("p_srv"),
                                                                              pl.lit(0).alias("p_ben"),
                                                                              pl.lit(0.0).alias("p_bds"))
            a = tot.join(prev, on=["npi", "place_of_srvc"]).with_columns(
                (pl.col("m_srv") - pl.col("p_srv")).alias("a_srv"), (pl.col("m_ben") - pl.col("p_ben")).alias("a_ben"),
                (pl.col("m_bds") - pl.col("p_bds")).alias("a_bds"))
        adds.append(a.select("npi", "place_of_srvc", "a_srv", "a_ben", "a_bds").with_columns(pl.lit(code).alias("hcpcs_cd")))
    add = pl.concat(adds).filter(pl.col("a_srv") > 0)
    key = ["npi", "hcpcs_cd", "place_of_srvc"]
    ps = (ps.join(add, on=key, how="left", maintain_order="left").join(prov_benes, on="npi", how="left")
          .with_columns((pl.col("tot_srvcs") + pl.col("a_srv").fill_null(0)).alias("tot_srvcs"),
                        pl.min_horizontal(pl.col("tot_benes") + pl.col("a_ben").fill_null(0),
                                          pl.col("prov_benes").fill_null(pl.col("tot_benes") + pl.col("a_ben").fill_null(0))
                                          ).alias("tot_benes"),
                        (pl.col("tot_bene_day_srvcs") + pl.col("a_bds").fill_null(0)).alias("tot_bene_day_srvcs"))
          .drop("a_srv", "a_ben", "a_bds", "prov_benes"))
    new = add.join(ps.select(key), on=key, how="anti")
    if new.height:
        new = (_price(new.join(who.select("npi", "peer_cell"), on="npi"), tabs).join(_attrs(ps), on="npi", how="left")
               .join(prov_benes, on="npi", how="left")
               .with_columns(pl.col("a_srv").alias("tot_srvcs"),
                             pl.min_horizontal("a_ben", "prov_benes").alias("tot_benes"),
                             pl.col("a_bds").alias("tot_bene_day_srvcs"), pl.lit("N").alias("hcpcs_drug_ind")))
        ps = pl.concat([ps, _cast_like(new, ps.schema)])
    return ps


def overutilization(ps: pl.DataFrame, who: pl.DataFrame) -> pl.DataFrame:
    """Multiplies services and beneficiary-days of existing non-drug rows by m (beneficiaries unchanged)."""
    j = ps.join(who.select("npi", "x"), on="npi", how="left", maintain_order="left")
    hit = pl.col("x").is_not_null() & (pl.col("hcpcs_drug_ind") == "N")
    return j.with_columns(pl.when(hit).then((pl.col("tot_srvcs") * pl.col("x")).round(0)).otherwise(pl.col("tot_srvcs"))
                          .alias("tot_srvcs"),
                          pl.when(hit).then((pl.col("tot_bene_day_srvcs") * pl.col("x")).round(0))
                          .otherwise(pl.col("tot_bene_day_srvcs")).alias("tot_bene_day_srvcs")).drop("x")


def charge_inflation(ps: pl.DataFrame, who: pl.DataFrame) -> pl.DataFrame:
    """Multiplies Avg_Sbmtd_Chrg by c on every row; allowed and paid unchanged."""
    j = ps.join(who.select("npi", "x"), on="npi", how="left", maintain_order="left")
    return j.with_columns((pl.col("avg_sbmtd_chrg") * pl.col("x").fill_null(1.0)).alias("avg_sbmtd_chrg")).drop("x")


def atypical_mix(ps: pl.DataFrame, who: pl.DataFrame, prov: pl.DataFrame, tabs: dict, icfg: InjectionConfig,
                 draw: int, year: int, rare_share: float) -> pl.DataFrame:
    """Adds rows for codes billed by < rare_share of the peer cell (not billed by the provider) worth a share q of the
    provider's payment, at national median prices, with >= min_new_row_benes beneficiaries."""
    t = icfg["transforms"]["atypical"]
    nat = tabs["nat"].filter(pl.col("n_providers") >= int(t["min_national_providers"]))
    rare = tabs["rare_cell"]
    prov_pay = prov.select("npi", pl.col("tot_mdcr_pymt_amt").alias("pay"), pl.col("tot_benes").alias("prov_benes"))
    min_b = int(icfg["transforms"]["min_new_row_benes"])
    billed = ps.group_by("npi").agg(pl.col("hcpcs_cd").unique())
    billed = dict(zip(billed["npi"].to_list(), billed["hcpcs_cd"].to_list()))
    codes_all = nat["hcpcs_cd"].to_numpy()
    rows = []
    for r in who.join(prov_pay, on="npi").to_dicts():
        common = set(rare.filter((pl.col("peer_cell") == r["peer_cell"]) & (pl.col("provider_share") >= rare_share))
                     ["hcpcs_cd"].to_list())
        own = set(billed.get(r["npi"], []))
        cand = np.array([c for c in codes_all if c not in common and c not in own])
        if cand.size < int(t["n_codes"]):
            raise InjectionError(f"{r['npi']} {year}: fewer than {t['n_codes']} rare codes to add")
        h = _str_keys([r["npi"] + "|" + c for c in cand], int(icfg["seed"]), _CODES, draw, year)
        pick = cand[np.argsort(h)[: int(t["n_codes"])]]
        add_pay = r["x"] / (1 - r["x"]) * r["pay"] / len(pick)
        for c in pick:
            rows.append({"npi": r["npi"], "peer_cell": r["peer_cell"], "hcpcs_cd": str(c), "add_pay": add_pay,
                         "prov_benes": r["prov_benes"]})
    if not rows:
        return ps
    new = (pl.DataFrame(rows).join(nat.select("hcpcs_cd", pl.col("place").alias("place_of_srvc"), "srv_per_bene",
                                              "bds_per_srv"), on="hcpcs_cd"))
    new = _price(new, tabs).with_columns(
        pl.max_horizontal(pl.lit(1.0), (pl.col("add_pay") / pl.col("avg_mdcr_pymt_amt")).round(0)).alias("tot_srvcs"))
    new = new.with_columns(
        pl.min_horizontal(pl.max_horizontal(pl.lit(min_b), (pl.col("tot_srvcs") / pl.col("srv_per_bene")).round(0)
                                            .cast(pl.Int64)), pl.col("prov_benes")).alias("tot_benes"))
    new = new.with_columns(pl.min_horizontal(pl.col("tot_srvcs"), pl.max_horizontal(
        pl.col("tot_benes").cast(pl.Float64), (pl.col("tot_srvcs") * pl.col("bds_per_srv")).round(0))).alias(
        "tot_bene_day_srvcs"), pl.lit("N").alias("hcpcs_drug_ind")).join(_attrs(ps), on="npi", how="left")
    return pl.concat([ps, _cast_like(new, ps.schema)])


def dme_mill(dme_s: pl.DataFrame, dme_p: pl.DataFrame, who: pl.DataFrame, prov: pl.DataFrame, tabs: dict,
             icfg: InjectionConfig, draw: int, year: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Adds high-risk DME-S rows worth d × the peer cell's median DME spend (national median if the cell has no
    referrers) filled by 1-2 suppliers; DME-P beneficiary totals updated (row created if absent)."""
    t = icfg["transforms"]["dme"]
    codes = tabs["dme_codes"]
    if codes.height < int(t["n_codes"]):
        raise InjectionError(f"{year}: fewer than {t['n_codes']} high-risk DME codes with reference prices")
    min_b = int(icfg["transforms"]["min_new_row_benes"])
    spend = who.join(tabs["dme_spend_cell"], on="peer_cell", how="left").with_columns(
        pl.col("med_spend").fill_null(tabs["dme_spend_nat"]))
    pb = prov.select("npi", pl.col("tot_benes").alias("prov_benes"), "prvdr_type") if "prvdr_type" in prov.columns \
        else prov.select("npi", pl.col("tot_benes").alias("prov_benes"))
    code_arr = codes["hcpcs_cd"].to_numpy()
    rows = []
    for r in spend.join(pb, on="npi").to_dicts():
        if not math.isfinite(r["med_spend"]) or r["med_spend"] <= 0:
            raise InjectionError(f"{r['npi']} {year}: no reference DME spend")
        h = _str_keys([r["npi"] + "|" + c for c in code_arr], int(icfg["seed"]), _DME, draw, year)
        pick = code_arr[np.argsort(h)[: int(t["n_codes"])]]
        for i, c in enumerate(pick):
            rows.append({"npi": r["npi"], "hcpcs_cd": str(c), "spend": r["x"] * r["med_spend"] / len(pick),
                         "suppliers": int(t["suppliers"][i % len(t["suppliers"])]), "prov_benes": r["prov_benes"]})
    new = pl.DataFrame(rows).join(codes, on="hcpcs_cd")
    new = new.with_columns(
        pl.max_horizontal(pl.lit(1.0), (pl.col("spend") / pl.col("avg_suplr_mdcr_pymt_amt")).round(0)).alias("srv"))
    new = new.with_columns(pl.min_horizontal(pl.max_horizontal(pl.lit(min_b), (pl.col("srv") / pl.col("srv_per_bene"))
                                                               .round(0).cast(pl.Int64)), pl.col("prov_benes"))
                           .alias("benes"))
    new = new.with_columns(pl.max_horizontal(pl.col("benes"), (pl.col("srv") * pl.col("clms_per_srv")).round(0)
                                             .cast(pl.Int64)).alias("clms"))
    key = ["npi", "hcpcs_cd", "suplr_rentl_ind"]
    new = new.rename({"rental": "suplr_rentl_ind"})
    j = dme_s.join(new.select(*key, "srv", "benes", "clms", "prov_benes", *DME_AMT).rename(
        {c: f"{c}__n" for c in ("srv", "benes", "clms", "prov_benes", *DME_AMT)}), on=key, how="left", maintain_order="left")
    hit = pl.col("srv__n").is_not_null()
    tot = pl.col("tot_suplr_srvcs") + pl.col("srv__n")
    j = j.with_columns(
        *[pl.when(hit).then((pl.col(c) * pl.col("tot_suplr_srvcs") + pl.col(f"{c}__n") * pl.col("srv__n")) / tot)
          .otherwise(pl.col(c)).alias(c) for c in DME_AMT],
        pl.when(hit).then(tot).otherwise(pl.col("tot_suplr_srvcs")).alias("tot_suplr_srvcs"),
        # a blank count is 1-10 (D114/D118): 0 + added is its lower bound; min_horizontal would skip a null
        pl.when(hit).then(pl.min_horizontal(pl.col("tot_suplr_benes").fill_null(0) + pl.col("benes__n"),
                                            pl.col("prov_benes__n")))
        .otherwise(pl.col("tot_suplr_benes")).alias("tot_suplr_benes"),
        pl.when(hit).then(pl.col("tot_suplr_clms") + pl.col("clms__n")).otherwise(pl.col("tot_suplr_clms"))
        .alias("tot_suplr_clms")).drop([c for c in j.columns if c.endswith("__n")])
    fresh = new.join(dme_s.select(key), on=key, how="anti").with_columns(
        pl.col("srv").alias("tot_suplr_srvcs"), pl.col("benes").alias("tot_suplr_benes"),
        pl.col("clms").alias("tot_suplr_clms"), pl.col("suppliers").alias("tot_suplrs"))
    dme_s = pl.concat([j, _cast_like(fresh, dme_s.schema)]) if fresh.height else j
    added = new.group_by("npi").agg(pl.col("benes").sum().alias("add_b"), pl.col("prov_benes").first())
    dp = dme_p.join(added, on="npi", how="left", maintain_order="left").with_columns(
        *[pl.when(pl.col("add_b").is_not_null()).then(pl.min_horizontal(pl.col(c).fill_null(0) + pl.col("add_b"),
                                                                          pl.col("prov_benes")))
          .otherwise(pl.col(c)).alias(c) for c in ("tot_suplr_benes", "dme_tot_suplr_benes")]).drop("add_b", "prov_benes")
    missing = added.join(dme_p.select("npi"), on="npi", how="anti")
    if missing.height:
        newp = missing.join(pb, on="npi", how="left", suffix="_p").with_columns(
            pl.min_horizontal("add_b", "prov_benes").alias("tot_suplr_benes")).with_columns(
            pl.col("tot_suplr_benes").alias("dme_tot_suplr_benes"))
        dp = pl.concat([dp, _cast_like(newp, dme_p.schema)])
    return dme_s, dp


def apply_year(frames: dict[str, pl.DataFrame], who: pl.DataFrame, coh: pl.DataFrame, refs, fcfg, icfg,
               draw: int, year: int) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Edit the planned providers' rows of one year. `frames`: that year's harmonised frames restricted to the
    planned NPIs (eager); `who`: plan rows (npi, peer_cell, x_<scheme>). Returns edited frames + edit log."""
    tabs = reference_tables(frames["_full"], coh, refs, fcfg, icfg)
    ps, prov, ds, dp = (frames[k] for k in ("cms_ps", "cms_prov", "dme_s", "dme_p"))
    before = _ps_sums(ps)
    prov_benes = prov.select("npi", pl.col("tot_benes").alias("prov_benes"))
    rare_share = fcfg.rare_share
    for s in SCHEME_ORDER:
        col = f"x_{s}"
        if col not in who.columns:
            continue
        w = who.filter(pl.col(col).is_not_null()).select("npi", "peer_cell", pl.col(col).alias("x"))
        if w.is_empty():
            continue
        if s == "upcoding":
            ps = upcoding(ps, w, prov_benes, tabs, icfg)
        elif s == "overutilization":
            ps = overutilization(ps, w)
        elif s == "atypical_mix":
            ps = atypical_mix(ps, w, prov, tabs, icfg, draw, year, rare_share)
        elif s == "charge_inflation":
            ps = charge_inflation(ps, w)
        elif s == "dme_mill":
            ds, dp = dme_mill(ds, dp, w, prov, tabs, icfg, draw, year)
    after = _ps_sums(ps)
    d = (after.join(before, on="npi", how="full", suffix="_b", coalesce=True).fill_null(0))
    delta = d.select("npi", (pl.col("srv") - pl.col("srv_b")).alias("d_srv"),
                     (pl.col("codes") - pl.col("codes_b")).alias("d_codes"),
                     *[(pl.col(t) - pl.col(f"{t}_b")).alias(f"d_{t}") for t in AMT.values()])
    prov = (prov.join(delta, on="npi", how="left", maintain_order="left").with_columns(
        (pl.col("tot_srvcs") + pl.col("d_srv").fill_null(0)).alias("tot_srvcs"),
        (pl.col("tot_hcpcs_cds") + pl.col("d_codes").fill_null(0)).alias("tot_hcpcs_cds"),
        *[(pl.col(t) + pl.col(f"d_{t}").fill_null(0)).alias(t) for t in AMT.values()])
        .drop([c for c in delta.columns if c != "npi"]))
    sup = icfg["suppression"]
    n_ps, n_ds = ps.height, ds.height
    ps = ps.filter(pl.col(sup["cms_ps"]["field"]) > int(sup["cms_ps"]["max_suppressed"]))
    ds = ds.filter(pl.col(sup["dme_s"]["field"]) > int(sup["dme_s"]["max_suppressed"]))
    log = delta.with_columns(pl.lit(year, dtype=pl.Int32).alias("year"),
                             pl.lit(n_ps - ps.height).alias("ps_rows_suppressed_total"),
                             pl.lit(n_ds - ds.height).alias("dme_rows_suppressed_total"))
    edited = {"cms_ps": ps, "cms_prov": prov, "dme_s": ds, "dme_p": dp}
    edited = {k: v.with_columns(pl.lit(year).cast(v.schema["year"]).alias("year")) for k, v in edited.items()}
    return edited, log


# ----------------------------------------------------------------------------------------------- build

def _year_frames(paths: Paths, year: int) -> dict[str, pl.LazyFrame]:
    """The year's harmonised frames, lazy: a year of service-level rows does not fit in memory twice."""
    out = {}
    for ds in features.DATASETS:
        f = paths.interim / ds / f"{year}.parquet"
        out[ds] = pl.scan_parquet(f) if f.is_file() else features.empty_frame(paths, ds, year).lazy()
    return out


def injected_cohort(paths: Paths, year: int, npis: pl.Series) -> pl.DataFrame:
    ccfg = cohort_mod.CohortConfig.load(paths)
    coh = pl.read_parquet(paths.processed / "cohort.parquet").filter(pl.col("year") == year)
    coh = coh.with_columns(pl.col("npi").is_in(npis.implode()).alias("injected"))
    return cohort_mod.assign_peer_cells(coh, ccfg)


BUILD_VERSION = 2   # bumped when the build's output changes for the same settings (2: D131 drop of unobservable rows)


def _build_fingerprint(paths: Paths, icfg: InjectionConfig, p: pl.DataFrame) -> str:
    """Injection and feature settings plus the plan itself: a finished year is reused only under the same ones."""
    h = hashlib.sha256(json.dumps([BUILD_VERSION, icfg.doc, load_config("features", paths)], sort_keys=True,
                                  default=str).encode())
    h.update(p.sort("npi", "year").write_ndjson().encode())   # a provider-year is planned at most once
    return h.hexdigest()[:16]


def build(paths: Paths, draw: int, p: pl.DataFrame | None = None, *, years=None, resume: bool = True) -> dict:
    """Plan (if not given), edit and featurise every planned year of draw `draw` (or only `years`). Each finished
    year leaves years/{y}.json and years/{y}_log.parquet; with `resume`, a year finished under the same fingerprint
    is skipped, so a killed build restarts where it stopped. summary.json and edit_log.parquet cover the finished
    planned years."""
    icfg = InjectionConfig.load(paths)
    fcfg = features.FeatureConfig.load(paths)
    out_dir = draw_dir(paths, draw)
    p = plan(paths, draw) if p is None else p
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = _build_fingerprint(paths, icfg, p)
    yd = out_dir / "years"
    fpf = out_dir / "build.json"
    if not (resume and fpf.is_file() and json.loads(fpf.read_text())["fingerprint"] == fp):
        for f in yd.glob("*") if yd.is_dir() else []:
            f.unlink()
    yd.mkdir(exist_ok=True)
    fpf.write_text(json.dumps({"fingerprint": fp}), encoding="utf-8")
    p.write_parquet(out_dir / "plan.parquet")
    planned = sorted(p["year"].unique().to_list())
    for y in planned if years is None else [int(v) for v in years]:
        if (yd / f"{y}.json").is_file():
            print(f"draw {draw} {y}: done earlier, skipped", flush=True)
            continue
        who = p.filter(pl.col("year") == y)
        npis = who["npi"]
        full = _year_frames(paths, y)
        coh = injected_cohort(paths, y, npis)
        coh_in = coh.filter(pl.col("in_cohort"))
        refs = features.fit_peer_reference(full, coh_in, y, fcfg)
        sub = {k: v.filter(pl.col("npi").is_in(npis.implode())).collect() for k, v in full.items()}
        sub["_full"] = full
        # peer cells after re-counting without the injected providers (they never count, D22)
        who_y = who.drop("peer_cell").join(coh.select("npi", "peer_cell"), on="npi", how="left")
        edited, log = apply_year(sub, who_y, coh, refs, fcfg, icfg, draw, y)
        for k, v in edited.items():
            d = out_dir / "edited" / k
            d.mkdir(parents=True, exist_ok=True)
            v.write_parquet(d / f"{y}.parquet")
        # lazy, so the year is not copied a second time before featurisation (memory)
        lazy = {k: pl.concat([full[k].filter(~pl.col("npi").is_in(npis.implode())), edited[k].lazy()],
                             how="vertical") for k in features.DATASETS}
        lazy["nppes"] = None
        raw = features.raw_statistics(lazy, coh_in, refs, y, fcfg)
        ps_ = features.fit_peer_stats(raw, coh_in, y, fcfg)
        feat = features.transform(raw, coh_in, ps_, refs, y, fcfg)
        feat = feat.with_columns(pl.col("npi").is_in(npis.implode()).alias("injected"))
        # D131: an injected provider whose every service-level row is suppressed after the edit would not appear in
        # the public files (its P&S features are undefined): it leaves the injected copy
        lost = npis.filter(~npis.is_in(edited["cms_ps"]["npi"].unique().implode()))
        feat = feat.filter(~pl.col("npi").is_in(lost.implode()))
        fd = out_dir / "features"
        fd.mkdir(parents=True, exist_ok=True)
        feat.write_parquet(fd / f"{y}.parquet")
        pd_ = out_dir / "peer" / str(y)
        pd_.mkdir(parents=True, exist_ok=True)
        ps_.stats.write_parquet(pd_ / "stats.parquet")
        coh.write_parquet(pd_ / "cohort.parquet")
        log.write_parquet(yd / f"{y}_log.parquet")
        (yd / f"{y}.json").write_text(json.dumps(
            {"injected": int(npis.len()), "rows_suppressed_ps": int(log["ps_rows_suppressed_total"][0]),
             "rows_suppressed_dme": int(log["dme_rows_suppressed_total"][0]), "dropped_no_visible_rows": lost.len(),
             "featurised": feat.height}), encoding="utf-8")
        print(f"draw {draw} {y}: {npis.len()} injected, {feat.height} featurised", flush=True)
        del full, sub, edited, lazy, raw   # free the year before the next one is read
    done = [y for y in planned if (yd / f"{y}.json").is_file()]
    summary = {"draw": draw, "years": {y: json.loads((yd / f"{y}.json").read_text()) for y in done},
               "missing_years": [y for y in planned if y not in done]}
    if done:
        pl.concat([pl.read_parquet(yd / f"{y}_log.parquet") for y in done], how="diagonal_relaxed").write_parquet(
            out_dir / "edit_log.parquet")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def load_injected_features(paths: Paths, draw: int, years, *, model_only: bool = False) -> pl.DataFrame:
    """The draw's feature tables of `years`. `model_only`: no raw__ statistics and float columns as Float32, which
    is what the model matrix uses anyway (memory: six years of the full table do not fit beside a model fit)."""
    fcfg = features.FeatureConfig.load(paths)
    files = [draw_dir(paths, draw) / "features" / f"{int(y)}.parquet" for y in years]
    miss = [str(f) for f in files if not f.is_file()]
    if miss:
        raise InjectionError(f"injected features missing: {miss}")
    if model_only:
        def lean(f):
            lf = pl.scan_parquet(f)
            sch = lf.collect_schema()
            return lf.select([pl.col(c).cast(pl.Float32) if sch[c] == pl.Float64 else pl.col(c)
                              for c in sch.names() if not c.startswith(features.RAW)]).collect()
        df = pl.concat([lean(f) for f in files], how="diagonal_relaxed")
    else:
        df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    hot = tuple(f"{s.id}__" for s in fcfg.one_hot)
    return df.with_columns(pl.col(c).fill_null(0) for c in df.columns if c.startswith(hot))
