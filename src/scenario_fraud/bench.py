"""Phase 4 timing benchmark (DESIGN §9.5; EXPERIMENT_PLAN §4 "timing benchmark"; HANDOFF §6 G4).

Measures, per family, raw-margin predictions per second on hybrid rows built like the coalition table's, and
interventional TreeSHAP seconds per provider; then projects the wall-clock of the coalition tables and of method A
for the explained set at each |B| option and for the global baseline. The human sets |B| and the explained-set
size at G4 from these numbers.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import polars as pl

from . import features, models, train
from .attribute import M, col_groups
from .attribution import coalition as C
from .attribution import summed_shap
from .config import load_config
from .explained_set import explained_path
from .modelmatrix import matrix
from .paths import Paths


def _explained_counts(paths: Paths, acfg: dict) -> dict[int, int]:
    p = explained_path(paths)
    if p.is_file():
        es = pl.read_parquet(p)
        return {int(y): int(n) for y, n in es.group_by("year").len().sort("year").iter_rows()}
    raise FileNotFoundError("explained set missing: run `phase4 explained-set` first")


def benchmark(paths: Paths, families=None) -> dict:
    acfg = load_config("attribution", paths)
    b = acfg["bench"]
    design, ref = acfg["explained"]["design"], int(acfg["reference_seed"])
    counts = _explained_counts(paths, acfg)
    year = min(counts)
    feats = features.load_features(paths, [year])
    n_exp = sum(counts.values())
    out = {"explained_per_year": counts, "explained_total": n_exp, "rows_timed": int(b["rows"]), "families": {}}
    for fam in families or acfg["coalition_families"]:
        f = models.load(train.model_dir(paths, design, fam, 0) / f"seed{ref}")
        X = matrix(feats, f.columns)
        cg = col_groups(f.columns, paths)
        nb = min(len(X), max(int(x) for x in b["B_options"]))
        cm = C.column_masks(cg, M)
        reps = math.ceil(int(b["rows"]) / (cm.shape[0] * nb))
        H = np.where(cm[None, :, None, :], X[:reps, None, None, :], X[None, None, :nb, :]).reshape(-1, X.shape[1])
        models.margin(f, H[:1000])                                        # warm-up
        best = math.inf
        for _ in range(2):
            t = time.perf_counter()
            models.margin(f, H)
            best = min(best, time.perf_counter() - t)
        rate = len(H) / best
        row = {"pred_per_s": rate, "rows": len(H), "projections_hours": {}}
        for B in [*b["B_options"], "global"]:
            nB = int(load_config("attribution", paths)["background"]["B_global"]) if B == "global" else int(B)
            row["projections_hours"][str(B)] = n_exp * 2 ** M * nB / rate / 3600
        if fam in acfg["shap_families"]:
            k = min(10, len(X) - nb) if len(X) > nb else min(10, len(X))
            t = time.perf_counter()
            summed_shap.feature_shap(f, X[:k], X[-nb:])
            s = (time.perf_counter() - t) / max(k, 1)
            row["shap_s_per_provider_at_B"] = {"B": nb, "seconds": s}
            row["shap_hours_peer_B200"] = n_exp * s * (200 / nb) / 3600
        out["families"][fam] = row
    out["budget_hours"] = float(b["budget_hours"])
    p = paths.outputs / "phase4" / "bench.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
