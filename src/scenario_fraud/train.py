"""C8 model trainer (DESIGN §7.8; EXPERIMENT_PLAN §2-3; CONTRACTS §3.3).

For one design (primary, O1, O2, O3), every family and every fold k:
1. rows of fold k's training set (TRAIN_ROLES), negatives subsampled identically for every family (C7);
2. tuning (designs.<d>.tune): Optuna TPE, `n_trials`, objective = average precision on `tune_valid` (all rows,
   unweighted) of a model fit on `tune_train`; otherwise the primary design's best parameters are reused;
3. the tuning model (best parameters, fit on tune_train) gives the validation average precision and, by default,
   the isotonic calibrator (models.yaml calibration_source, DESIGN §17 D59);
4. final models, one per seed, on all training rows (models.yaml final_fit);
5. out-of-fold scores of the test rows (margin and calibrated probability).
`score_showcase` scores 2024 with the primary models of each NPI's own fold (no accuracy claims).
Training refuses to run unless the G2 freeze record exists and still matches the protocol (HANDOFF §8 rule 3).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from . import models, qa, splits
from .modelmatrix import DesignFrame, load_design, matrix, roles
from .paths import Paths

FAMILY_SALT = {f: i for i, f in enumerate(models.FAMILIES)}
DESIGN_SALT = {"primary": 0, "O1": 1, "O2": 2, "O3": 3}


class TrainError(RuntimeError):
    pass


def check_frozen(paths: Paths) -> dict:
    """HANDOFF §8 rule 3: no model is trained before the G2 freeze record exists and matches the protocol."""
    freeze = qa.load_freeze(paths)
    if freeze is None:
        raise TrainError("no G2 freeze record (outputs/phase2/G2_freeze.json): run `phase2 report` and take the G2 "
                         "decision before Phase 3")
    now = qa.protocol_hashes(paths)
    changed = [k for k in qa.FREEZE_KEYS if freeze["protocol_hashes"].get(k) != now.get(k)]
    if changed:
        raise TrainError(f"protocol changed since the G2 freeze ({changed}); a frozen protocol must not change")
    return freeze


def model_dir(paths: Paths, design: str, family: str, fold: int) -> Path:
    return paths.processed / "models" / design / family / f"fold{fold}"


def score_path(paths: Paths, design: str, family: str, fold: int) -> Path:
    return paths.processed / "scores" / design / family / f"fold{fold}.parquet"


def _seed(*parts: int) -> int:
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0] % (2 ** 31 - 1))


def _ap(y: np.ndarray, m: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, m))


def _calibrator(m: np.ndarray, y: np.ndarray):
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True).fit(m, y)


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _reused_params(paths: Paths, family: str, fold: int) -> dict:
    p = model_dir(paths, "primary", family, fold) / "tuning.json"
    if not p.is_file():
        raise TrainError(f"{paths.rel(p)} missing: train the primary design first (this design reuses its parameters)")
    return json.loads(p.read_text(encoding="utf-8"))["best_params"]


def train_design(paths: Paths, design: str, *, families=None, folds=None, seeds: int | None = None,
                 n_trials: int | None = None, resume: bool = True) -> dict:
    """Train one design. With `resume` (the default) a (family, fold) whose tuning.json records the same trial budget
    and seed count is skipped, an interrupted tuning replays its logged trials (progress/trials.jsonl; the seeded
    sampler re-proposes the same parameters, so the run is identical to an uninterrupted one) and seed models already
    saved with the best parameters are loaded instead of refitted. `resume=False` retrains from scratch."""
    check_frozen(paths)
    mcfg = models.ModelConfig.load(paths)
    scfg = splits.SplitConfig.load(paths)
    if design not in DESIGN_SALT:
        raise TrainError(f"design must be one of {sorted(DESIGN_SALT)} (showcase: use score_showcase)")
    dcfg = mcfg.design(design)
    fams = list(families or mcfg["families"])
    unknown = [f for f in fams if f not in mcfg["families"]]
    if unknown:
        raise TrainError(f"families {unknown} not in models.yaml families")
    n_seeds = int(seeds if seeds is not None else dcfg["seeds"])
    trials = int(n_trials if n_trials is not None else mcfg["n_trials"]) if dcfg["tune"] else 0
    fold_ids = sorted(int(k) for k in (folds if folds is not None else range(scfg.n_folds)))
    log = _Log(paths)
    log(f"{design}: families {fams}, folds {fold_ids}, {trials} trials, {n_seeds} seeds, resume={resume}")
    df = load_design(paths, design)
    summary = {"design": design, "label_tier": df.label_tier, "W": df.W, "rows": df.frame.height,
               "columns": len(df.columns), "folds": {}}
    for k in fold_ids:
        done = {f: _done(paths, design, f, k, trials, n_seeds, fingerprint(mcfg, f, df.columns)) for f in fams} \
            if resume else {}
        for f, info in done.items():
            if info:
                log(f"{design} fold {k} {f}: already done (valid AP {info['valid_ap']:.5f}), skipped")
        pending = [f for f in fams if not done.get(f)]
        out = _train_fold(paths, df, k, pending, mcfg, scfg, dcfg, n_seeds, trials, resume, log) if pending else {}
        summary["folds"][k] = {f: out[f] if f in out else _brief(done[f]) for f in fams}
    log(f"{design}: finished")
    return summary


class _Log:
    """Progress lines on stderr and appended to outputs/phase3/train_progress.log (validation numbers only)."""

    def __init__(self, paths: Paths):
        self.path = paths.phase3 / "train_progress.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, file=sys.stderr, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _brief(info: dict) -> dict:
    return {k: info[k] for k in ("valid_ap", "n_valid_pos", "n_test", "n_test_pos", "seconds")}


def fingerprint(mcfg: models.ModelConfig, family: str, columns: list[str]) -> str:
    """Hash of everything in models.yaml that changes a fit of `family` besides the tuned parameters (the family's
    section, the pos_weight space, seeds, final fit and calibration) and of the design columns. Resume reuses logged
    trials, saved seeds and finished pieces only under the same fingerprint."""
    doc = {"family": mcfg[family], "pos_weight": mcfg["pos_weight"], "tuning_seed": mcfg["tuning_seed"],
           "final_fit": mcfg["final_fit"], "calibration_source": mcfg["calibration_source"], "columns": columns}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _done(paths: Paths, design: str, family: str, fold: int, trials: int, n_seeds: int, fp: str) -> dict | None:
    """The tuning.json of a finished (family, fold) with this budget and fingerprint, else None. tuning.json is
    written last. A tuning.json without a fingerprint (written before fingerprints existed) is accepted."""
    p = model_dir(paths, design, family, fold) / "tuning.json"
    if not p.is_file() or not score_path(paths, design, family, fold).is_file():
        return None
    info = json.loads(p.read_text(encoding="utf-8"))
    ok = info.get("n_trials") == trials and info.get("seeds") == n_seeds and info.get("fingerprint", fp) == fp
    return info if ok else None


def _read_trials(p: Path) -> list[dict]:
    """Logged trials; a line cut short by an interruption ends the log."""
    out = []
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out


def _write_trials(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(p)


def _append_trial(p: Path, row: dict) -> None:
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def fold_rows(df: DesignFrame, k: int, mcfg: models.ModelConfig, scfg: splits.SplitConfig):
    """Fold k's rows: (frame with `role` and row index `_i`, tune_train sample, tune_valid, final-fit sample, test).
    Training rows are TRAIN_ROLES with negatives subsampled (weights 1/rate); tune_valid and test are not subsampled."""
    role = roles(df, k, scfg)
    fr = df.frame.with_columns(role.alias("role"), pl.int_range(pl.len()).alias("_i"))
    train_rows = fr.filter(pl.col("role").is_in(list(splits.TRAIN_ROLES))).select("npi", "year", "y", "role", "_i")
    sub = splits.subsample_negatives(train_rows, scfg)
    tt = sub.filter(pl.col("role") == "tune_train")
    tv = fr.filter(pl.col("role") == "tune_valid")
    if mcfg["final_fit"] == "exclude_holdout_npis":
        held = splits.holdout_npis(df.folds, k, scfg)
        final = sub.filter(~pl.col("npi").is_in(held.implode()))
    else:
        final = sub
    test = fr.filter(pl.col("role") == "test")
    return fr, tt, tv, final, test


def _tune(paths, df: DesignFrame, k: int, fam: str, mcfg, trials: int, data, resume: bool, log) -> tuple[dict, list]:
    """Optuna TPE by ask/tell with every trial logged to progress/trials.jsonl as it ends. On resume the logged
    trials are replayed: the seeded sampler proposes the same parameters given the same history, so a logged value
    is told instead of refitting; from the first mismatch on, trials are fitted again."""
    import optuna

    X_tt, y_tt, w_tt, X_tv, y_tv = data
    fs = FAMILY_SALT[fam]
    space = mcfg.space(fam)
    tp = model_dir(paths, df.design, fam, k) / "progress" / "trials.jsonl"
    fpp = tp.with_name("fingerprint.json")
    fp = fingerprint(mcfg, fam, df.columns)
    logged = _read_trials(tp)[:trials] if resume else []
    if logged and fpp.is_file() and json.loads(fpp.read_text(encoding="utf-8")) != fp:   # no file: older log, kept
        log(f"{df.design} fold {k} {fam}: settings changed since the logged trials; tuning restarts")
        logged = []
    _write_trials(tp, logged)
    fpp.write_text(json.dumps(fp), encoding="utf-8")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
        seed=_seed(mcfg["tuning_seed"], DESIGN_SALT[df.design], fs, k)))
    replayed = 0
    for i in range(trials):
        trial = study.ask()
        p = models.suggest(trial, space)
        if i < len(logged) and logged[i]["params"] == p:
            value = logged[i]["value"]
            replayed += 1
        else:
            if i < len(logged):
                log(f"{df.design} fold {k} {fam}: logged trial {i} has other parameters; refitting from here")
                logged = logged[:i]
                _write_trials(tp, logged)
            t1 = time.perf_counter()
            f = models.fit(fam, X_tt, y_tt, w_tt, p, _seed(mcfg["tuning_seed"], 7, fs, k), df.columns, mcfg, paths)
            value = _ap(y_tv, models.margin(f, X_tv))
            row = {"number": i, "value": value, "params": p, "seconds": round(time.perf_counter() - t1, 1)}
            _append_trial(tp, row)
            logged.append(row)
            best = max(r["value"] for r in logged)
            log(f"{df.design} fold {k} {fam}: trial {i + 1}/{trials} valid AP {value:.5f} (best {best:.5f}), "
                f"{row['seconds']}s")
        study.tell(trial, value)
    if replayed:
        log(f"{df.design} fold {k} {fam}: {replayed} logged trials replayed without refitting")
    return dict(study.best_params), [{"number": t.number, "value": t.value, "params": t.params} for t in study.trials]


def _train_fold(paths, df: DesignFrame, k: int, fams, mcfg, scfg, dcfg, n_seeds: int, trials: int,
                resume: bool = False, log=None) -> dict:
    import optuna

    log = log or (lambda msg: None)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fr, tt, tv, final, test = fold_rows(df, k, mcfg, scfg)
    n_valid_pos = int(tv["y"].sum())
    if n_valid_pos < int(mcfg["min_valid_positives"]):
        raise TrainError(f"{df.design} fold {k}: tune_valid has {n_valid_pos} positives "
                         f"(< models.yaml min_valid_positives = {mcfg['min_valid_positives']})")
    if test.height == 0:
        raise TrainError(f"{df.design} fold {k}: no test rows")
    X_tt, y_tt, w_tt = matrix(fr, df.columns, tt["_i"].to_numpy()), tt["y"].to_numpy(), tt["weight"].to_numpy()
    X_tv, y_tv = matrix(fr, df.columns, tv["_i"].to_numpy()), tv["y"].to_numpy()
    X_f, y_f, w_f = matrix(fr, df.columns, final["_i"].to_numpy()), final["y"].to_numpy(), final["weight"].to_numpy()
    X_te = matrix(fr, df.columns, test["_i"].to_numpy())
    log(f"{df.design} fold {k}: tune_train {tt.height:,}, tune_valid {tv.height:,} ({n_valid_pos} positives), "
        f"final {final.height:,}, test {test.height:,} rows")
    out = {}
    for fam in fams:
        t0 = time.perf_counter()
        fs = FAMILY_SALT[fam]
        trials_log = []
        log(f"{df.design} fold {k} {fam}: start")
        if dcfg["tune"]:
            best, trials_log = _tune(paths, df, k, fam, mcfg, trials, (X_tt, y_tt, w_tt, X_tv, y_tv), resume, log)
            source = "tuned"
        else:
            best = _reused_params(paths, fam, k)
            source = "primary design, same family and fold"
        tuning_model = models.fit(fam, X_tt, y_tt, w_tt, best, _seed(mcfg["tuning_seed"], 7, fs, k), df.columns,
                                  mcfg, paths)
        m_tv = models.margin(tuning_model, X_tv)
        valid_ap = _ap(y_tv, m_tv)
        log(f"{df.design} fold {k} {fam}: tuning model valid AP {valid_ap:.5f}, best params {best}")
        cal = _calibrator(m_tv, y_tv) if mcfg["calibration_source"] == "tuning_model" else None
        d = model_dir(paths, df.design, fam, k)
        fp = fingerprint(mcfg, fam, df.columns)
        scores = []
        for s in range(n_seeds):
            sd = d / f"seed{s}"
            mark = sd / "params.json"
            t1 = time.perf_counter()
            if resume and mark.is_file() and json.loads(mark.read_text(encoding="utf-8")) == {"params": best,
                                                                                         "fingerprint": fp}:
                f = models.load(sd)
                c = joblib.load(sd / "calibrator.joblib")
                how = "loaded (resume)"
            else:
                mark.unlink(missing_ok=True)
                f = models.fit(fam, X_f, y_f, w_f, best, _seed(mcfg["tuning_seed"], 11, fs, k, s), df.columns, mcfg,
                               paths)
                c = cal if cal is not None else _calibrator(models.margin(f, X_tv), y_tv)
                models.save(f, sd)
                joblib.dump(c, sd / "calibrator.joblib")
                mark.write_text(json.dumps({"params": best, "fingerprint": fp}),
                                encoding="utf-8")                      # written last: marks a complete seed
                how = "fitted"
            m = models.margin(f, X_te)
            log(f"{df.design} fold {k} {fam}: seed {s + 1}/{n_seeds} {how}, {time.perf_counter() - t1:.1f}s")
            scores.append(test.select("npi", "year", "fold", "y", "peer_cell", "label_horizon_complete").with_columns(
                pl.lit(s, dtype=pl.Int32).alias("seed"), pl.Series("margin", m),
                pl.Series("prob", c.predict(m)), pl.lit(df.label_tier).alias("label_tier"),
                pl.lit(df.W, dtype=pl.Int32).alias("W")))
        _write(pl.concat(scores), score_path(paths, df.design, fam, k))
        info = {"design": df.design, "family": fam, "fold": k, "params_source": source, "best_params": best,
                "valid_ap": valid_ap, "n_tune_train": tt.height, "n_tune_valid": tv.height,
                "n_valid_pos": n_valid_pos, "n_final_train": final.height, "n_final_pos": int(y_f.sum()),
                "n_test": test.height, "n_test_pos": int(test["y"].sum()), "seeds": n_seeds, "n_trials": trials, "fingerprint": fp,
                "final_fit": mcfg["final_fit"], "calibration_source": mcfg["calibration_source"],
                "pair_terms": tuning_model.extra.get("pair_terms"), "trials": trials_log,
                "seconds": round(time.perf_counter() - t0, 2)}
        (d / "columns.json").write_text(json.dumps(df.columns), encoding="utf-8")
        tmp = d / "tuning.json.tmp"
        tmp.write_text(json.dumps(info, indent=1), encoding="utf-8")
        tmp.replace(d / "tuning.json")                                  # written last: marks a finished (family, fold)
        log(f"{df.design} fold {k} {fam}: done in {info['seconds'] / 60:.1f} min")
        out[fam] = _brief(info)
    return out


def score_showcase(paths: Paths, *, families=None) -> dict:
    """2024 (experiment.yaml splits.showcase) scored by the primary models of each NPI's own fold."""
    check_frozen(paths)
    mcfg = models.ModelConfig.load(paths)
    scfg = splits.SplitConfig.load(paths)
    df = load_design(paths, "showcase", years=[scfg.showcase_year])
    out = {}
    for fam in families or mcfg["families"]:
        rows = 0
        for k in range(scfg.n_folds):
            d = model_dir(paths, "primary", fam, k)
            if not d.is_dir():
                raise TrainError(f"{paths.rel(d)} missing: train the primary design first")
            role = roles(df, k, scfg, design="showcase")
            test = df.frame.filter(role == "test")
            parts = []
            for sd in sorted(d.glob("seed*")):
                f = models.load(sd)
                X = matrix(test, f.columns)
                m = models.margin(f, X)
                c = joblib.load(sd / "calibrator.joblib")
                parts.append(test.select("npi", "year", "fold", "y", "peer_cell").with_columns(
                    pl.lit(int(sd.name[4:]), dtype=pl.Int32).alias("seed"), pl.Series("margin", m),
                    pl.Series("prob", c.predict(m))))
            _write(pl.concat(parts), score_path(paths, "showcase", fam, k))
            rows += test.height
        out[fam] = {"rows": rows, "year": scfg.showcase_year}
    return out


def load_scores(paths: Paths, design: str, family: str) -> pl.DataFrame:
    files = sorted((paths.processed / "scores" / design / family).glob("fold*.parquet"))
    if not files:
        raise TrainError(f"no scores for {design}/{family}")
    return pl.concat([pl.read_parquet(f) for f in files])


def load_tuning(paths: Paths, design: str) -> pl.DataFrame:
    """One row per (family, fold) with the validation average precision (for choosing the best models)."""
    rows = []
    for p in sorted((paths.processed / "models" / design).glob("*/fold*/tuning.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        rows.append({k: t[k] for k in ("family", "fold", "valid_ap", "n_valid_pos", "params_source")})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def benchmark(paths: Paths, *, families=None, fold: int = 0, trials: int = 3) -> dict:
    """Timing run before the full Phase 3 (EXPERIMENT_PLAN §10; HANDOFF §7 "time it first"): on fold k of the
    primary design, `trials` tuning trials per family (parameters drawn by the same seeded TPE sampler, so its random
    start) fitted on tune_train and scored on tune_valid, plus one final-model fit on all training rows. No test row
    is scored. Projects the hours of the full run: per family and fold, n_trials tuning fits + the tuning model +
    `seeds` final fits (primary) + one final fit per rolling origin. Writes outputs/phase3/bench.json."""
    import optuna

    check_frozen(paths)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mcfg = models.ModelConfig.load(paths)
    scfg = splits.SplitConfig.load(paths)
    fams = list(families or mcfg["families"])
    t0 = time.perf_counter()
    df = load_design(paths, "primary")
    load_s = time.perf_counter() - t0
    fr, tt, tv, final, _ = fold_rows(df, fold, mcfg, scfg)
    X_tt, y_tt, w_tt = matrix(fr, df.columns, tt["_i"].to_numpy()), tt["y"].to_numpy(), tt["weight"].to_numpy()
    X_tv, y_tv = matrix(fr, df.columns, tv["_i"].to_numpy()), tv["y"].to_numpy()
    X_f, y_f, w_f = matrix(fr, df.columns, final["_i"].to_numpy()), final["y"].to_numpy(), final["weight"].to_numpy()
    n_trials, seeds = int(mcfg["n_trials"]), int(mcfg.design("primary")["seeds"])
    rolling = [d for d in DESIGN_SALT if d != "primary" and d in scfg.designs]
    out = {"fold": fold, "rows": {"tune_train": int(len(y_tt)), "tune_valid": int(len(y_tv)), "final": int(len(y_f)),
                                  "columns": len(df.columns)},
           "load_design_seconds": round(load_s, 1), "threads": int(mcfg["threads"]), "n_trials": n_trials,
           "seeds": seeds, "rolling_origins": rolling, "families": {}}
    for fam in fams:
        fs = FAMILY_SALT[fam]
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
            seed=_seed(mcfg["tuning_seed"], DESIGN_SALT["primary"], fs, fold)))
        times, params = [], []
        for _ in range(trials):
            trial = study.ask()
            p = models.suggest(trial, mcfg.space(fam))
            t1 = time.perf_counter()
            f = models.fit(fam, X_tt, y_tt, w_tt, p, _seed(mcfg["tuning_seed"], 7, fs, fold), df.columns, mcfg, paths)
            ap = _ap(y_tv, models.margin(f, X_tv))
            times.append(time.perf_counter() - t1)
            params.append(p)
            study.tell(trial, ap)
        t1 = time.perf_counter()
        models.fit(fam, X_f, y_f, w_f, params[0], _seed(mcfg["tuning_seed"], 11, fs, fold, 0), df.columns, mcfg, paths)
        t_final = time.perf_counter() - t1
        t_trial = float(np.mean(times))
        per_fold = n_trials * t_trial + t_trial + seeds * t_final + len(rolling) * t_final
        out["families"][fam] = {"trial_seconds": [round(t, 1) for t in times], "mean_trial_seconds": round(t_trial, 1),
                                "final_fit_seconds": round(t_final, 1), "params": params,
                                "projected_hours_per_fold": round(per_fold / 3600, 2),
                                "projected_hours_all_folds": round(scfg.n_folds * per_fold / 3600, 2)}
    out["projected_hours_total"] = round(sum(v["projected_hours_all_folds"] for v in out["families"].values()), 1)
    p = paths.phase3 / "bench.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out
