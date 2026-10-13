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

import json
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
                 n_trials: int | None = None) -> dict:
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
    trials = int(n_trials if n_trials is not None else mcfg["n_trials"])
    df = load_design(paths, design)
    fold_ids = sorted(int(k) for k in (folds if folds is not None else range(scfg.n_folds)))
    summary = {"design": design, "label_tier": df.label_tier, "W": df.W, "rows": df.frame.height,
               "columns": len(df.columns), "folds": {}}
    for k in fold_ids:
        summary["folds"][k] = _train_fold(paths, df, k, fams, mcfg, scfg, dcfg, n_seeds, trials)
    return summary


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


def _train_fold(paths, df: DesignFrame, k: int, fams, mcfg, scfg, dcfg, n_seeds: int, trials: int) -> dict:
    import optuna

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
    out = {}
    for fam in fams:
        t0 = time.perf_counter()
        fs = FAMILY_SALT[fam]
        trials_log = []
        if dcfg["tune"]:
            space = mcfg.space(fam)

            def objective(trial):
                p = models.suggest(trial, space)
                f = models.fit(fam, X_tt, y_tt, w_tt, p, _seed(mcfg["tuning_seed"], 7, fs, k), df.columns, mcfg, paths)
                return _ap(y_tv, models.margin(f, X_tv))

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
                seed=_seed(mcfg["tuning_seed"], DESIGN_SALT[df.design], fs, k)))
            study.optimize(objective, n_trials=trials)
            best = dict(study.best_params)
            trials_log = [{"number": t.number, "value": t.value, "params": t.params} for t in study.trials]
            source = "tuned"
        else:
            best = _reused_params(paths, fam, k)
            source = "primary design, same family and fold"
        tuning_model = models.fit(fam, X_tt, y_tt, w_tt, best, _seed(mcfg["tuning_seed"], 7, fs, k), df.columns,
                                  mcfg, paths)
        m_tv = models.margin(tuning_model, X_tv)
        valid_ap = _ap(y_tv, m_tv)
        cal = _calibrator(m_tv, y_tv) if mcfg["calibration_source"] == "tuning_model" else None
        d = model_dir(paths, df.design, fam, k)
        scores = []
        for s in range(n_seeds):
            f = models.fit(fam, X_f, y_f, w_f, best, _seed(mcfg["tuning_seed"], 11, fs, k, s), df.columns, mcfg, paths)
            c = cal if cal is not None else _calibrator(models.margin(f, X_tv), y_tv)
            models.save(f, d / f"seed{s}")
            joblib.dump(c, d / f"seed{s}" / "calibrator.joblib")
            m = models.margin(f, X_te)
            scores.append(test.select("npi", "year", "fold", "y", "peer_cell", "label_horizon_complete").with_columns(
                pl.lit(s, dtype=pl.Int32).alias("seed"), pl.Series("margin", m),
                pl.Series("prob", c.predict(m)), pl.lit(df.label_tier).alias("label_tier"),
                pl.lit(df.W, dtype=pl.Int32).alias("W")))
        _write(pl.concat(scores), score_path(paths, df.design, fam, k))
        info = {"design": df.design, "family": fam, "fold": k, "params_source": source, "best_params": best,
                "valid_ap": valid_ap, "n_tune_train": tt.height, "n_tune_valid": tv.height,
                "n_valid_pos": n_valid_pos, "n_final_train": final.height, "n_final_pos": int(y_f.sum()),
                "n_test": test.height, "n_test_pos": int(test["y"].sum()), "seeds": n_seeds,
                "final_fit": mcfg["final_fit"], "calibration_source": mcfg["calibration_source"],
                "pair_terms": tuning_model.extra.get("pair_terms"), "trials": trials_log,
                "seconds": round(time.perf_counter() - t0, 2)}
        (d / "columns.json").write_text(json.dumps(df.columns), encoding="utf-8")
        (d / "tuning.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        out[fam] = {k2: info[k2] for k2 in ("valid_ap", "n_valid_pos", "n_test", "n_test_pos", "seconds")}
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
