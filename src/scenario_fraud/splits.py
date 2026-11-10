"""C7 Fold and split assigner (DESIGN §7.6, D12-D13; EXPERIMENT_PLAN §2; CONTRACTS §2.3).

cohort + LEIE exclusions -> data/processed/folds.parquet (+ folds.meta.json); any (npi, year) frame -> role.

Folds. Every NPI with at least one in-cohort year is assigned once to one of `splits.n_folds` folds, stratified
by strat_specialty (modal `prvdr_type` over its in-cohort years; a tie goes to the specialty of the latest of
the tied years) x ever_positive (an exclusion with an NPI under any label tier's authorities, at any date).
Within a stratum NPIs are ordered by a seeded 64-bit hash of the NPI; strata are laid end to end (ever-positive
strata first) and position i goes to fold i mod n_folds. Fold sizes, ever-positive NPIs per fold and each
stratum's count per fold therefore differ by at most one, and the partition depends only on the NPI set and
the seed (not on row order).

Roles (column `role`, Enum of ROLES) for fold k under a design with train years [a, b] and test years [c, e]
(`splits.designs`):

    test        NPI in fold k, year in [c, e]
    tune_valid  NPI not in k and in the fold's validation hold-out H_k, year b (the last train year)
    tune_train  NPI not in k and not in H_k, year in [a, b - 1]
    train       NPI not in k, year in [a, b], and neither of the two above, i.e. H_k's rows in [a, b - 1]
                and the other NPIs' rows in year b
    none        everything else: fold-k NPIs in train years, other NPIs in test years, any other year

TRAIN_ROLES (train, tune_train, tune_valid) are the training set of fold k (NPIs not in k, train years). Tuning fits
on tune_train and validates on tune_valid, so tuning never sees a tune_valid NPI or the last train year. Which of
the TRAIN_ROLES rows the final model and its isotonic calibrator use (DESIGN §7.8 "calibration on the validation
slice") is decided in Phase 3 (DESIGN §17, open at G2). With `exclude_years` (sensitivity S4, e.g. the flagged
2020), rows of those years get `none` and the validation year is the last remaining train year.
H_k is a deterministic, stratified `splits.valid_frac` share of all NPIs not in fold k
(same strata, a fold-specific hash order, the NPIs at positions i with floor((i+1) f) > floor(i f)); it
differs per fold and is the same for every design. The showcase design (`splits.showcase`) trains like its
model design and tests on the showcase year (no accuracy claims).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import polars as pl

from . import labels, provenance
from .config import load_config
from .paths import Paths

ROLES = ("train", "test", "tune_train", "tune_valid", "none")
ROLE_DTYPE = pl.Enum(ROLES)
TRAIN_ROLES = ("train", "tune_train", "tune_valid")
FOLD_COLS = ("npi", "fold", "strat_specialty", "ever_positive")
SHOWCASE = "showcase"
# Sub-seed purposes: SeedSequence([seed, purpose, fold]).
_FOLDS, _HOLDOUT, _NEGATIVES = 0, 1, 2
# splitmix64 finaliser constants (Steele, Lea & Flood 2014): a stable, version-independent 64-bit hash.
_GOLDEN, _M1, _M2 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)
_TMP = ("__split_in_k", "__split_held")


class SplitError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- config

@dataclass(frozen=True)
class Design:
    """Inclusive train and test year ranges of one split design."""
    name: str
    train: tuple[int, int]
    test: tuple[int, int]
    accuracy: bool = True          # False for the showcase year

    @property
    def train_years(self) -> tuple[int, ...]:
        return tuple(range(self.train[0], self.train[1] + 1))

    @property
    def test_years(self) -> tuple[int, ...]:
        return tuple(range(self.test[0], self.test[1] + 1))

    @property
    def last_train_year(self) -> int:
        return self.train[1]


@dataclass(frozen=True)
class SplitConfig:
    n_folds: int
    seed: int
    valid_frac: float
    designs: dict[str, Design]     # splits.designs (primary, O1, O2, O3)
    showcase_year: int
    showcase_design: str
    neg_rate: float
    neg_sensitivity_rates: tuple[float, ...]
    neg_seed: int

    @classmethod
    def load(cls, paths: Paths) -> "SplitConfig":
        exp = load_config("experiment", paths)
        try:
            s = exp["splits"]

            def rng(v) -> tuple[int, int]:
                if len(v) != 2:
                    raise ValueError(f"year range {v} is not [first, last]")
                return int(v[0]), int(v[1])

            designs = {str(k): Design(str(k), rng(v["train"]), rng(v["test"])) for k, v in s["designs"].items()}
            neg = s["negative_subsample"]
            cfg = cls(
                n_folds=int(s["n_folds"]), seed=int(s["seed"]), valid_frac=float(s["valid_frac"]), designs=designs,
                showcase_year=int(s["showcase"]["year"]), showcase_design=str(s["showcase"]["design"]),
                neg_rate=float(neg["rate"]),
                neg_sensitivity_rates=tuple(float(r) for r in neg.get("sensitivity_rates") or ()),
                neg_seed=int(neg["seed"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SplitError(f"configs/experiment.yaml splits section incomplete or malformed: {exc!r}") from exc
        cfg._validate(exp.get("years"))
        return cfg

    def _validate(self, years: dict | None) -> None:
        problems = []
        if self.n_folds < 2:
            problems.append("n_folds must be >= 2")
        if not 0 < self.valid_frac < 1:
            problems.append("valid_frac must be in (0, 1)")
        if "primary" not in self.designs or SHOWCASE in self.designs:
            problems.append(f"designs must include `primary` and must not be named {SHOWCASE!r}")
        for d in self.designs.values():
            if d.train[0] > d.train[1] or d.test[0] > d.test[1]:
                problems.append(f"{d.name}: a year range runs backwards")
            if d.train[1] - d.train[0] < 1:
                problems.append(f"{d.name}: needs >= 2 train years (tune_train before, tune_valid in the last)")
            if d.train[1] >= d.test[0]:
                problems.append(f"{d.name}: every train year must precede every test year")
        if years and "primary" in self.designs:
            p = self.designs["primary"]
            if [list(p.train), list(p.test)] != [list(map(int, years["train"])), list(map(int, years["test"]))]:
                problems.append(f"splits.designs.primary {p.train}->{p.test} disagrees with years.train/test "
                                f"{years['train']}->{years['test']}")
            if years.get("showcase") is not None and int(years["showcase"]) != self.showcase_year:
                problems.append("splits.showcase.year disagrees with years.showcase")
        base = self.designs.get(self.showcase_design)
        if base is None:
            problems.append(f"showcase design {self.showcase_design!r} is not in designs")
        elif self.showcase_year <= base.train[1]:
            problems.append("the showcase year must come after the train years of its model design")
        if not all(0 < r <= 1 for r in (self.neg_rate, *self.neg_sensitivity_rates)):
            problems.append("negative_subsample rates must be in (0, 1]")
        if problems:
            raise SplitError("configs/experiment.yaml splits: " + "; ".join(problems))

    def design(self, name: str) -> Design:
        """A design of `splits.designs`, or `showcase` (train like its model design, test = showcase year)."""
        if name == SHOWCASE:
            base = self.designs[self.showcase_design]
            return Design(SHOWCASE, base.train, (self.showcase_year, self.showcase_year), accuracy=False)
        if name not in self.designs:
            raise SplitError(f"unknown design {name!r}; known: {[*self.designs, SHOWCASE]}")
        return self.designs[name]

    @property
    def all_designs(self) -> tuple[Design, ...]:
        return (*self.designs.values(), self.design(SHOWCASE))


# ----------------------------------------------------------------------------------------------- hashing

def _salt(seed: int, *path: int) -> np.uint64:
    return np.random.SeedSequence([int(seed), *path]).generate_state(1, dtype=np.uint64)[0]


def _mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser on a uint64 array (wraps modulo 2**64)."""
    z = x + _GOLDEN
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def _npi_u64(npi: pl.Series) -> np.ndarray:
    ok = npi.str.contains(r"^\d{10}$").fill_null(False)
    if not ok.all():
        raise SplitError(f"NPIs must be 10-digit strings; got e.g. {npi.filter(~ok).head(3).to_list()}")
    return npi.cast(pl.UInt64).to_numpy()


def npi_keys(npi: pl.Series, salt: np.uint64) -> np.ndarray:
    """Seeded 64-bit key per NPI: depends only on the NPI and the salt (not on row order or the NPI set)."""
    return _mix64(_npi_u64(npi) ^ salt)


# ----------------------------------------------------------------------------------------------- folds

def _stratified_order(df: pl.DataFrame, salt: np.uint64) -> pl.DataFrame:
    """Rows ordered stratum by stratum (ever-positive strata first, then by specialty), by key within a stratum;
    `_pos` = position in that order."""
    return (df.with_columns(pl.Series("_key", npi_keys(df["npi"], salt), dtype=pl.UInt64))
            .sort(["ever_positive", "strat_specialty", "_key", "npi"], descending=[True, False, False, False])
            .with_row_index("_pos").drop("_key"))


def make_folds(cohort: pl.DataFrame | pl.LazyFrame, positive_npis, cfg: SplitConfig) -> pl.DataFrame:
    """FOLD_COLS for every NPI with an in-cohort year (DESIGN §7.6), sorted by NPI. `positive_npis`: the NPIs
    with an exclusion under any label tier (any iterable / Series of NPI strings)."""
    lf = cohort.lazy()
    miss = [c for c in ("npi", "year", "prvdr_type", "in_cohort") if c not in lf.collect_schema().names()]
    if miss:
        raise SplitError(f"cohort columns missing: {miss}")
    inc = lf.filter(pl.col("in_cohort")).select("npi", "year", "prvdr_type").collect()
    if inc.is_empty():
        raise SplitError("no in-cohort rows to partition")
    if inc["npi"].null_count() or inc["prvdr_type"].null_count():
        raise SplitError("in-cohort rows with null npi or prvdr_type")
    if inc.select(pl.struct("npi", "year").is_duplicated().any()).item():
        raise SplitError("cohort has duplicated (npi, year) rows")
    spec = (inc.group_by("npi", "prvdr_type").agg(pl.len().alias("_n"), pl.col("year").max().alias("_last"))
            .sort(["npi", "_n", "_last"], descending=[False, True, True])
            .unique("npi", keep="first", maintain_order=True)
            .select("npi", pl.col("prvdr_type").alias("strat_specialty")))
    pos = pl.Series("npi", list(positive_npis), dtype=pl.String)
    df = spec.with_columns(pl.col("npi").is_in(pos.implode()).alias("ever_positive"))
    o = _stratified_order(df, _salt(cfg.seed, _FOLDS))
    return (o.with_columns((pl.col("_pos") % cfg.n_folds).cast(pl.Int32).alias("fold"))
            .select(FOLD_COLS).sort("npi"))


def ever_positive_npis(paths: Paths) -> pl.Series:
    """NPIs with an exclusion under the authorities of any tier in configs/labels.yaml (any date)."""
    f = paths.processed / "exclusions.parquet"
    if not f.is_file():
        raise SplitError("data/processed/exclusions.parquet missing (run phase1 labels): folds are stratified "
                         "by ever-positive status")
    cfg = labels.LabelConfig.load(paths)
    auths = sorted({a for t in cfg.tiers.values() for a in t})
    return (pl.scan_parquet(f).filter(pl.col("npi").is_not_null() & pl.col("authority").is_in(auths))
            .select("npi").unique().collect()["npi"].sort())


def fold_balance(folds: pl.DataFrame, n_folds: int) -> dict:
    """Per-fold counts and the largest per-stratum spread (max - min count over folds)."""
    per = folds.group_by("fold").agg(pl.len().alias("npis"), pl.col("ever_positive").sum().alias("ever_positive"))
    per = pl.DataFrame({"fold": list(range(n_folds))}, schema={"fold": pl.Int32}).join(
        per, on="fold", how="left").fill_null(0).sort("fold")
    strata = (folds.group_by("strat_specialty", "ever_positive", "fold").len()
              .pivot(on="fold", index=["strat_specialty", "ever_positive"], values="len").fill_null(0))
    for k in range(n_folds):
        if str(k) not in strata.columns:
            strata = strata.with_columns(pl.lit(0, dtype=pl.UInt32).alias(str(k)))
    ks = [str(k) for k in range(n_folds)]
    spread = strata.select((pl.max_horizontal(ks) - pl.min_horizontal(ks)).max()).item() if strata.height else 0
    return {
        "npis": folds.height, "strata": strata.height, "max_stratum_spread": int(spread or 0),
        "per_fold": per.to_dicts(),
        "fold_size_spread": int(per["npis"].max() - per["npis"].min()),
        "ever_positive_spread": int(per["ever_positive"].max() - per["ever_positive"].min()),
    }


def assign_folds(paths: Paths) -> pl.DataFrame:
    """C7 folds from data/processed/cohort.parquet and the LEIE exclusions; writes data/processed/folds.parquet
    and folds.meta.json (config + balance). Returns the frame."""
    cfg = SplitConfig.load(paths)
    cpath = paths.processed / "cohort.parquet"
    if not cpath.is_file():
        raise SplitError("data/processed/cohort.parquet missing (run phase2 cohort first)")
    pos = ever_positive_npis(paths)
    folds = make_folds(pl.scan_parquet(cpath), pos, cfg)
    out = paths.processed / "folds.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    folds.write_parquet(tmp)
    tmp.replace(out)
    years = pl.scan_parquet(cpath).filter("in_cohort").select(pl.col("year").unique()).collect()["year"].sort()
    meta = {
        "config": {"n_folds": cfg.n_folds, "seed": cfg.seed, "valid_frac": cfg.valid_frac},
        "cohort_years": years.to_list(),
        "synthetic_inputs": provenance.inputs_synthetic(
            paths.interim / ds / f"{y}.parquet" for y in years.to_list() for ds in ("cms_ps", "cms_prov")),
        "positive_npis_any_tier": pos.len(),
        "positive_npis_in_folds": int(folds["ever_positive"].sum()),
        **fold_balance(folds, cfg.n_folds),
    }
    out.with_name("folds.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folds


def load_folds(paths: Paths) -> pl.DataFrame:
    f = paths.processed / "folds.parquet"
    if not f.is_file():
        raise SplitError("data/processed/folds.parquet missing (run phase2 folds first)")
    return pl.read_parquet(f)


def check_folds(folds: pl.DataFrame, cfg: SplitConfig) -> None:
    """Fail unless every NPI appears once, with a fold in [0, n_folds) and non-null strata."""
    miss = [c for c in FOLD_COLS if c not in folds.columns]
    if miss:
        raise SplitError(f"folds columns missing: {miss}")
    problems = []
    if folds["npi"].null_count() or folds["npi"].is_duplicated().any():
        problems.append(f"{int(folds['npi'].is_duplicated().sum())} rows with a duplicated NPI (or null NPIs): "
                        "an NPI must sit in exactly one fold")
    if not folds["fold"].is_between(0, cfg.n_folds - 1).fill_null(False).all():
        problems.append(f"fold values outside 0..{cfg.n_folds - 1}")
    if folds["ever_positive"].null_count() or folds["strat_specialty"].null_count():
        problems.append("null strata")
    if problems:
        raise SplitError("folds: " + "; ".join(problems))


# ----------------------------------------------------------------------------------------------- roles

def _check_fold(fold: int, cfg: SplitConfig) -> int:
    if isinstance(fold, bool) or not (isinstance(fold, (int, np.integer)) and 0 <= int(fold) < cfg.n_folds):
        raise SplitError(f"fold must be an integer in 0..{cfg.n_folds - 1}; got {fold!r}")
    return int(fold)


def holdout_npis(folds: pl.DataFrame, fold: int, cfg: SplitConfig) -> pl.Series:
    """H_k: the validation hold-out of fold k, a stratified valid_frac share of the NPIs not in fold k
    (sorted). Exactly floor(valid_frac x their number) NPIs; each stratum within one of its share."""
    fold = _check_fold(fold, cfg)
    check_folds(folds, cfg)
    tr = folds.filter(pl.col("fold") != fold).select("npi", "strat_specialty", "ever_positive")
    frac = Fraction(str(cfg.valid_frac)).limit_denominator(1_000_000)
    p, q = frac.numerator, frac.denominator
    i = pl.col("_pos").cast(pl.Int64)
    o = _stratified_order(tr, _salt(cfg.seed, _HOLDOUT, fold))
    return o.filter((i + 1) * p // q > i * p // q)["npi"].sort()


def assign_roles(frame: pl.DataFrame | pl.LazyFrame, folds: pl.DataFrame, fold: int, design: Design | str,
                 cfg: SplitConfig, holdout=None, *, check: bool = True, exclude_years=()) -> pl.DataFrame:
    """`frame` (needs npi, year) + column `role` (Enum of ROLES) for fold k under `design` (module docstring).
    Row order is kept. `holdout` overrides H_k (NPIs not in fold k); every frame NPI needs a fold. check=False
    skips re-validating `folds` and `holdout` (for callers that already did, e.g. looping over folds).
    `exclude_years` (sensitivity S4): those years' rows get `none`; tune_valid moves to the last remaining train
    year, which needs >= 2 remaining train years."""
    df = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
    miss = [c for c in ("npi", "year") if c not in df.columns]
    if miss:
        raise SplitError(f"frame columns missing: {miss}")
    clash = [c for c in ("role", *_TMP) if c in df.columns]
    if clash:
        raise SplitError(f"frame already has column(s) {clash}")
    if df["year"].null_count():
        raise SplitError("frame has null years")
    fold = _check_fold(fold, cfg)
    if check:
        check_folds(folds, cfg)
    d = cfg.design(design) if isinstance(design, str) else design
    if holdout is None:
        held = holdout_npis(folds, fold, cfg)
    else:
        held = holdout if isinstance(holdout, pl.Series) else pl.Series("npi", list(holdout), dtype=pl.String)
        if check and (folds.filter(pl.col("npi").is_in(held.implode()) & (pl.col("fold") == fold)).height
                      or held.is_in(folds["npi"].implode()).not_().any()):
            raise SplitError("holdout must list NPIs of the folds table outside fold k")
    f = folds.select("npi", (pl.col("fold") == fold).alias(_TMP[0]), pl.col("npi").is_in(held.implode()).alias(_TMP[1]))
    j = df.join(f, on="npi", how="left", maintain_order="left")
    lost = j.filter(pl.col(_TMP[0]).is_null())["npi"]
    if lost.len():
        raise SplitError(f"{lost.len()} frame rows ({lost.n_unique()} NPIs) have no fold, e.g. "
                         f"{lost.unique().sort().head(3).to_list()} (folds.parquet is built from the cohort)")
    in_k, held_e, yr = pl.col(_TMP[0]), pl.col(_TMP[1]), pl.col("year")
    excl = sorted({int(y) for y in exclude_years})
    train_years = [y for y in d.train_years if y not in excl]
    if len(train_years) < 2:
        raise SplitError(f"{d.name} without years {excl} keeps train years {train_years}: tuning needs >= 2")
    in_train = ~in_k & yr.is_in(train_years)
    last = train_years[-1]
    role = (pl.when(in_k & yr.is_between(*d.test) & ~yr.is_in(excl)).then(pl.lit("test"))
            .when(in_train & held_e & (yr == last)).then(pl.lit("tune_valid"))
            .when(in_train & ~held_e & (yr < last)).then(pl.lit("tune_train"))
            .when(in_train).then(pl.lit("train"))
            .otherwise(pl.lit("none"))).cast(ROLE_DTYPE)
    return j.with_columns(role.alias("role")).drop(list(_TMP))


def split_rows(frame: pl.DataFrame | pl.LazyFrame, fold: int, design: str, paths: Paths, *,
               folds: pl.DataFrame | None = None, cfg: SplitConfig | None = None, exclude_years=()) -> pl.DataFrame:
    """CONTRACTS §2.3: `frame` + `role` for fold `fold` under design `design` (primary, O1, O2, O3 or showcase),
    with folds from data/processed/folds.parquet and parameters from experiment.yaml `splits`. `exclude_years`:
    sensitivity S4 (e.g. experiment.yaml years.flagged)."""
    cfg = cfg or SplitConfig.load(paths)
    return assign_roles(frame, load_folds(paths) if folds is None else folds, fold, design, cfg,
                        exclude_years=exclude_years)


# ----------------------------------------------------------------------------------------------- subsampling

def subsample_negatives(frame: pl.DataFrame, cfg: SplitConfig, *, rate: float | None = None,
                        label: str = "y") -> pl.DataFrame:
    """EXPERIMENT_PLAN §2 (Phase 3 training rows): keep every positive and each negative (npi, year) with
    probability `rate` (default splits.negative_subsample.rate), decided by a seeded hash of (npi, year), so the
    sample is the same for every model family, fold and design, and a smaller rate's sample is nested in a
    larger one's (S12). Adds `weight` = 1 for positives, 1 / rate for negatives."""
    rate = cfg.neg_rate if rate is None else float(rate)
    if not 0 < rate <= 1:
        raise SplitError(f"subsampling rate must be in (0, 1]; got {rate}")
    miss = [c for c in ("npi", "year", label) if c not in frame.columns]
    if miss or "weight" in frame.columns:
        raise SplitError(f"frame needs npi, year, {label} and no `weight` column (missing {miss})")
    y = frame[label]
    if y.null_count() or not y.is_in([0, 1]).all():
        raise SplitError(f"`{label}` must be 0/1 without nulls (drop label-dropped rows first)")
    key = _mix64(npi_keys(frame["npi"], _salt(cfg.neg_seed, _NEGATIVES))
                 ^ frame["year"].cast(pl.UInt64).to_numpy())
    keep = np.ones(frame.height, dtype=bool) if rate == 1 else key < np.uint64(int(rate * 2.0 ** 64))
    pos = (y == 1).to_numpy()
    return (frame.filter(pl.Series(keep | pos))
            .with_columns(pl.when(pl.col(label) == 1).then(1.0).otherwise(1.0 / rate).alias("weight")))
