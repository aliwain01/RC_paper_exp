"""Model families behind one interface (C8; DESIGN §7.8; CONTRACTS §3.2).

fit(family, X, y, w, params, seed, columns) -> Fitted; margin(fitted, X) -> log-odds (raw margin, before
calibration); save / load (native formats for XGBoost and LightGBM so `shap.TreeExplainer` can read them).
`lr_design` exposes the logistic regression's transformed matrix and the group of each coefficient (method D).

Weights: the caller passes inverse-sampling weights; `pos_weight` (a tuned parameter of every family) multiplies
the positives' weights here, the same mechanism for all families.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from . import features, groups
from .config import load_config
from .modelmatrix import column_kinds
from .paths import Paths

FAMILIES = ("lr", "ebm", "ga2m", "xgb", "xgb_mono", "lgbm", "lgbm_mono")
MONO = {"xgb_mono", "lgbm_mono"}
BATCH = 1_000_000


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    doc: dict

    @classmethod
    def load(cls, paths: Paths | None = None) -> "ModelConfig":
        doc = load_config("models", paths)
        bad = [f for f in doc["families"] if f not in FAMILIES]
        if bad:
            raise ModelError(f"models.yaml: unknown families {bad}; known {FAMILIES}")
        if doc["final_fit"] not in ("all_train", "exclude_holdout_npis"):
            raise ModelError("models.yaml final_fit must be all_train or exclude_holdout_npis")
        if doc["calibration_source"] not in ("tuning_model", "final_model"):
            raise ModelError("models.yaml calibration_source must be tuning_model or final_model")
        if doc["calibration_source"] == "final_model" and doc["final_fit"] != "exclude_holdout_npis":
            raise ModelError("calibration_source final_model needs final_fit exclude_holdout_npis: otherwise the "
                             "calibrator is fitted on providers the final model was trained on (DESIGN §17 D59)")
        for k in ("glass_box", "gbm", "gbm_mono"):
            if not set(doc[k]) <= set(doc["families"]):
                raise ModelError(f"models.yaml {k} lists families not in `families`")
        return cls(doc)

    def __getitem__(self, k):
        return self.doc[k]

    def design(self, name: str) -> dict:
        d = (self.doc.get("designs") or {}).get(name)
        if d is None:
            raise ModelError(f"models.yaml designs has no entry for {name!r}")
        return {"tune": bool(d.get("tune", False)), "seeds": int(d.get("seeds", self.doc["seeds"]))}

    def space(self, family: str) -> dict:
        return {**(self.doc[family].get("space") or {}), "pos_weight": self.doc["pos_weight"]}

    def fixed(self, family: str) -> dict:
        return dict(self.doc[family].get("fixed") or {})


def suggest(trial, space: dict) -> dict:
    """Optuna parameters from a models.yaml search space."""
    out = {}
    for name, s in space.items():
        t = s["type"]
        if t == "float":
            out[name] = trial.suggest_float(name, float(s["low"]), float(s["high"]), log=bool(s.get("log")))
        elif t == "int":
            out[name] = trial.suggest_int(name, int(s["low"]), int(s["high"]), log=bool(s.get("log")))
        elif t == "cat":
            out[name] = trial.suggest_categorical(name, list(s["choices"]))
        else:
            raise ModelError(f"unknown search-space type {t!r} for {name}")
    return out


@dataclass
class Fitted:
    family: str
    model: object
    columns: list[str]
    params: dict
    seed: int
    extra: dict = field(default_factory=dict)


def monotone_vector(columns, paths: Paths | None = None) -> list[int]:
    """features.yaml `monotone` per model column (one-hot columns 0)."""
    fcfg = features.FeatureConfig.load(paths)
    return [0 if "__" in c else int(fcfg.spec(c).monotone) for c in columns]


def _weights(y: np.ndarray, w: np.ndarray, pos_weight: float) -> np.ndarray:
    return np.where(y == 1, w * float(pos_weight), w).astype(np.float64)


# ------------------------------------------------------------------------------------------------ fit

def fit(family: str, X: np.ndarray, y: np.ndarray, w: np.ndarray, params: dict, seed: int, columns: list[str],
        cfg: ModelConfig, paths: Paths | None = None) -> Fitted:
    if family not in FAMILIES:
        raise ModelError(f"unknown family {family!r}")
    if X.shape[1] != len(columns):
        raise ModelError("X and columns disagree")
    given = dict(params)
    params = dict(params)
    sw = _weights(y, w, params.pop("pos_weight", 1.0))
    seed = int(seed) % (2 ** 31 - 1)
    threads = int(cfg["threads"])
    fixed = cfg.fixed(family)
    base = "xgb" if family.startswith("xgb") else "lgbm" if family.startswith("lgbm") else family
    if base == "lr":
        model, extra = _fit_lr(X, y, sw, params, fixed, seed, columns, cfg, paths)
    elif base in ("ebm", "ga2m"):
        from interpret.glassbox import ExplainableBoostingClassifier

        kw = {**fixed, **params}
        if base == "ebm":
            kw["interactions"] = 0
        elif int(kw.get("interactions", 0)) > int(cfg["ga2m"]["max_interactions"]):
            raise ModelError("ga2m interactions above models.yaml ga2m.max_interactions")
        model = ExplainableBoostingClassifier(feature_names=list(columns), random_state=seed, n_jobs=threads, **kw)
        model.fit(X, y, sample_weight=sw)
        extra = {"pair_terms": [n for n, t in zip(model.term_names_, model.term_features_) if len(t) == 2]}
    elif base == "xgb":
        import xgboost as xgb

        p = {**fixed, **params}
        rounds = int(p.pop("n_estimators"))
        p.update(seed=seed, nthread=threads, verbosity=0)
        if family in MONO:
            p["monotone_constraints"] = "(" + ",".join(map(str, monotone_vector(columns, paths))) + ")"
        d = xgb.DMatrix(X, label=y, weight=sw, feature_names=list(columns), nthread=threads)
        model = xgb.train(p, d, num_boost_round=rounds)
        extra = {}
    else:
        import lightgbm as lgb

        p = {**fixed, **params}
        rounds = int(p.pop("n_estimators"))
        p.update(seed=seed, deterministic=True, force_row_wise=True, num_threads=threads, verbose=-1,
                 bagging_freq=1 if p.get("bagging_fraction", 1.0) < 1.0 else 0)
        if family in MONO:
            p["monotone_constraints"] = monotone_vector(columns, paths)
        d = lgb.Dataset(X, label=y, weight=sw, feature_name=list(columns), free_raw_data=True)
        model = lgb.train(p, d, num_boost_round=rounds)
        extra = {}
    return Fitted(family, model, list(columns), given, seed, extra)


def _fit_lr(X, y, sw, params, fixed, seed, columns, cfg: ModelConfig, paths):
    from sklearn.compose import ColumnTransformer
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    kinds = column_kinds(columns, paths=paths)
    spl = [i for i, c in enumerate(columns) if c in set(cfg["lr"]["spline_features"])]
    cont = [i for i, c in enumerate(columns) if kinds[c] in ("robust_z", "context_raw")]
    binary = [i for i, c in enumerate(columns) if kinds[c] in ("indicator", "one_hot")]
    if len(cont) + len(binary) != len(columns):
        raise ModelError("lr: every column must be continuous or binary")
    ct = ColumnTransformer([
        ("spl", SplineTransformer(n_knots=int(cfg["lr"]["n_knots"]), degree=int(cfg["lr"]["degree"]),
                                  knots="quantile", extrapolation="linear", include_bias=False), spl),
        ("std", StandardScaler(), cont),
        ("bin", "passthrough", binary),
    ])
    pipe = Pipeline([("design", ct), ("lr", LogisticRegression(random_state=seed, **fixed, **params))])
    # Weights rescaled to mean 1: the optimum is unchanged up to C, which is tuned (DESIGN §17 D77); saga
    # converges far better. Non-convergence is recorded in the model's metadata, not raised or hidden.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe.fit(X, y, lr__sample_weight=sw / sw.mean())
    converged = not any(issubclass(c.category, ConvergenceWarning) for c in caught)
    for c in caught:
        if not issubclass(c.category, ConvergenceWarning):
            warnings.warn_explicit(c.message, c.category, c.filename, c.lineno)
    return pipe, {"converged": converged}


def lr_design(f: Fitted, X: np.ndarray, paths: Paths | None = None) -> tuple[np.ndarray, list[str], list[str]]:
    """The LR's transformed matrix Z (decision = Z @ coef + intercept), Z's column names, and each Z column's
    group (spline basis columns take their source feature's group)."""
    if f.family != "lr":
        raise ModelError("lr_design needs an lr model")
    ct = f.model.named_steps["design"]
    Z = ct.transform(X)
    names = list(ct.get_feature_names_out())
    grp = groups.feature_groups(f.columns, paths=paths)
    out_names, out_groups = [], []
    for n in names:
        m = re.match(r"^(spl|std|bin)__x(\d+)(?:_sp_(\d+))?$", n)
        if not m:
            raise ModelError(f"unexpected LR design column {n!r}")
        src = f.columns[int(m.group(2))]
        out_names.append(src if m.group(1) != "spl" else f"{src}__spl{m.group(3)}")
        out_groups.append(grp[src])
    return np.asarray(Z, dtype=np.float64), out_names, out_groups


# ------------------------------------------------------------------------------------------------ predict

def margin(f: Fitted, X: np.ndarray, batch: int = BATCH) -> np.ndarray:
    """Raw margin (log-odds) in batches of `batch` rows."""
    if X.shape[1] != len(f.columns):
        raise ModelError(f"X has {X.shape[1]} columns, model expects {len(f.columns)}")
    out = np.empty(X.shape[0], dtype=np.float64)
    for a in range(0, X.shape[0], batch):
        b = X[a:a + batch]
        if f.family.startswith("xgb"):
            out[a:a + batch] = f.model.inplace_predict(b, predict_type="margin")
        elif f.family.startswith("lgbm"):
            out[a:a + batch] = f.model.predict(b, raw_score=True)
        else:
            out[a:a + batch] = f.model.decision_function(b)
    return out


# ------------------------------------------------------------------------------------------------ persistence

def save(f: Fitted, d: Path) -> None:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    if f.family.startswith("xgb"):
        f.model.save_model(str(d / "model.json"))
    elif f.family.startswith("lgbm"):
        f.model.save_model(str(d / "model.txt"))
    else:
        joblib.dump(f.model, d / "model.joblib")
    (d / "fitted.json").write_text(json.dumps({"family": f.family, "columns": f.columns, "params": f.params,
                                               "seed": f.seed, "extra": f.extra}, indent=1), encoding="utf-8")


def load(d: Path) -> Fitted:
    d = Path(d)
    meta = json.loads((d / "fitted.json").read_text(encoding="utf-8"))
    fam = meta["family"]
    if fam.startswith("xgb"):
        import xgboost as xgb

        model = xgb.Booster()
        model.load_model(str(d / "model.json"))
    elif fam.startswith("lgbm"):
        import lightgbm as lgb

        model = lgb.Booster(model_file=str(d / "model.txt"))
    else:
        model = joblib.load(d / "model.joblib")
    return Fitted(fam, model, meta["columns"], meta["params"], meta["seed"], meta.get("extra", {}))
