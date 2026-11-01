"""E9 sensitivity suite (EXPERIMENT_PLAN §7; DESIGN §10.6; CONTRACTS §6).

Each variant in `configs/evaluation.yaml: sensitivity.variants` runs in its own scratch project
data/processed/sensitivity/{id}/: the main project's configs with the variant's overrides, the main interim data and
every processed artefact the variant does not rebuild linked in place, and a freeze record derived from the main G2
freeze (the variants are pre-registered in the protocol). Stages, in order: `cohort`, `features` (S10), `train`
(primary design, tuned hyperparameters of the main project, one seed), `explain` (Phase 4 engine, peer baseline, a
seeded subsample of the main explained set), `inject_train`, `inject_explain` (Phase 5 models / attributions of the
configured draws). Grouping variants (S6 correlation clusters, S7 moved features) rewrite groups.yaml in the scratch
project. `from_main` variants (S5 baselines, S9 monotone) are read from the main project's evaluation.

Per variant, outputs/phase6/sensitivity/{id}.json: detection (seed-0 PR-AUC, precision@k of the variant's models
and of the same main models), consistency and faithfulness on the variant's explanations, agreement with the main
explanations of the same providers, and correctness on injected providers where re-explained.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from . import cohort, evaluate, features, groups, inject_models, metrics, qa, train
from .attribute import explain as explain_store
from .attribute import load_attributions
from .config import load_config
from .evaluate import write_json
from .explained_set import explained_path, load_explained_set
from .paths import Paths
from .stability import _mat, kendall_tau_b, subsample

STAGES = ("cohort", "features", "train", "explain", "inject_train", "inject_explain")
ALWAYS_FRESH = ("attributions", "evaluation", "stability", "sensitivity")


class SensitivityError(RuntimeError):
    pass


def variant_root(paths: Paths, vid: str) -> Path:
    return paths.processed / "sensitivity" / vid


def deep_update(d: dict, u: dict) -> dict:
    out = dict(d)
    for k, v in u.items():
        out[k] = deep_update(out.get(k) or {}, v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def mirror(src: Path, dst: Path, fresh: set[str], skip: set[str] = frozenset()) -> None:
    """Link every entry of src into dst except `fresh` paths (left for the variant to write) and `skip` paths;
    directories that contain a fresh or skipped path are created and mirrored recursively."""
    dst.mkdir(parents=True, exist_ok=True)
    for e in sorted(src.iterdir()):
        name = e.name
        if name in fresh or name in skip:
            continue
        inner_f = {f[len(name) + 1:] for f in fresh if f.startswith(name + "/")}
        inner_s = {s[len(name) + 1:] for s in skip if s.startswith(name + "/")}
        if e.is_dir() and (inner_f or inner_s):
            mirror(e, dst / name, inner_f, inner_s)
        else:
            qa._link(e.resolve(), dst / name)


@contextlib.contextmanager
def as_root(root: Path):
    """Point implicit root discovery at the scratch project while a variant runs."""
    old = os.environ.get(qa.ROOT_ENV)
    os.environ[qa.ROOT_ENV] = str(root)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(qa.ROOT_ENV, None)
        else:
            os.environ[qa.ROOT_ENV] = old


def variant_families(paths: Paths, spec: dict) -> list[str]:
    best = evaluate.best_families(paths)
    sel = spec.get("families", "best")
    if sel == "best":
        fams = [best["gbm"], best["glass_box"]]
    elif sel == "gbm_lr":
        fams = [best["gbm"], "lr"]
    else:
        fams = list(sel)
    if not all(fams):
        raise SensitivityError(f"best families not available ({best}): train and tune the primary design first")
    return list(dict.fromkeys(fams))


# ----------------------------------------------------------------------------------------------- grouping variants

def cluster_moves(paths: Paths, ccfg: dict, seed: int) -> tuple[dict[str, str], list[dict]]:
    """S6: hierarchical clustering (distance 1 − |Spearman ρ|) of the scheme-group features over the primary design's
    training years (label-free), cut at n_clusters; each cluster takes the name of the scheme group it overlaps most
    (one-to-one assignment), so the answer key still applies. Returns feature moves and the cluster table."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import squareform
    from scipy.stats import rankdata

    from . import splits

    scfg = splits.SplitConfig.load(paths)
    years = scfg.design("primary").train_years
    base = groups.load_groups(paths)
    feat_group = {f: g.name for g in base for f in g.features if g.name != groups.CONTEXT}
    df = features.load_features(paths, years)
    cols = [c for c in features.model_columns(df.columns) if c in feat_group]
    n = df.height
    if n > int(ccfg["max_rows"]):
        h = splits.npi_keys(df["npi"], splits._salt(seed, 951)) ^ df["year"].cast(pl.UInt64).to_numpy()
        df = df.with_columns(pl.Series("_h", h)).sort("_h").head(int(ccfg["max_rows"]))
    X = rankdata(df.select(cols).to_numpy().astype(float), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(X, rowvar=False)
    R = np.nan_to_num(np.abs(R), nan=0.0)
    D = np.clip(1.0 - R, 0.0, None)
    np.fill_diagonal(D, 0.0)
    lab = fcluster(linkage(squareform(D, checks=False), method=str(ccfg["linkage"])), int(ccfg["n_clusters"]),
                   criterion="maxclust")
    clusters = sorted(set(lab.tolist()))
    names = list(groups.SCHEME_GROUPS)
    overlap = np.array([[sum(1 for c, l in zip(cols, lab) if l == cl and feat_group[c] == g) for g in names]
                        for cl in clusters])
    ri, ci = linear_sum_assignment(-overlap)
    name_of = {clusters[r]: names[c] for r, c in zip(ri, ci)}
    moves, table = {}, []
    for cl in clusters:
        mem = [c for c, l in zip(cols, lab) if l == cl]
        tgt = name_of.get(cl)
        if tgt is None:
            raise SensitivityError("more clusters than scheme groups")
        for c in mem:
            if feat_group[c] != tgt:
                moves[c] = tgt
        table.append({"cluster": int(cl), "named": tgt, "features": mem,
                      "a_priori": sorted({feat_group[c] for c in mem})})
    return moves, table


def grouping_doc(paths: Paths, moves: dict[str, str]) -> dict:
    """groups.yaml with `moves` applied to the base grouping (variants list dropped)."""
    doc = load_config("groups", paths)
    by = {g.name: g for g in groups.load_groups(paths)}
    out = []
    for gd in doc["groups"]:
        g = by[gd["name"]]
        feats = [f for f in g.features if f not in moves] + [f for f, t in moves.items() if t == g.name]
        out.append({**gd, "features": feats})
    return {"groups": out, "variants": {}}


# ----------------------------------------------------------------------------------------------- setup

def variant_fingerprint(paths: Paths, vid: str, spec: dict, fams: list[str], draws: list[int]) -> str:
    """The variant's spec, families and draws, every main config file and the main G2 freeze: a scratch project or
    result made under the same fingerprint is reused (D132)."""
    frozen = (qa.load_freeze(paths) or {}).get("recorded_utc")
    h = hashlib.sha256(json.dumps([vid, spec, fams, draws, frozen], sort_keys=True, default=str).encode())
    for f in sorted(paths.configs.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(paths.configs).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def setup(paths: Paths, vid: str, spec: dict, fams: list[str], draws: list[int], *, resume: bool = True,
          fp: str | None = None) -> tuple[Paths, dict]:
    """Create the scratch project of one variant (configs, linked data, derived freeze record). With `resume`, a
    project whose setup.json records the same fingerprint is kept as it is, so the stages inside it (training,
    attribution, injection models) resume where they stopped."""
    sc = load_config("evaluation", paths)
    root = variant_root(paths, vid)
    fp = fp or variant_fingerprint(paths, vid, spec, fams, draws)
    sj = root / "setup.json"
    if resume and sj.is_file():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        if rec.get("fp") == fp:
            return Paths(root), rec["info"]
    if root.exists():
        shutil.rmtree(root)
    vp = Paths(root)
    (root / "design").mkdir(parents=True)
    (root / "pyproject.toml").write_text(f"[project]\nname = 'sensitivity-{vid}'\n", encoding="utf-8")
    shutil.copytree(paths.configs, vp.configs)
    info: dict = {}
    over = dict(spec.get("overrides") or {})
    over["models"] = deep_update(over.get("models") or {}, {"designs": {"primary": {"tune": False, "seeds": 1}}})
    for name, upd in over.items():
        p = vp.configs / f"{name}.yaml"
        if not p.is_file():
            raise SensitivityError(f"{vid}: override of unknown config {name}.yaml")
        p.write_text(yaml.safe_dump(deep_update(yaml.safe_load(p.read_text(encoding="utf-8")), upd), sort_keys=False),
                     encoding="utf-8")
    grouping = spec.get("grouping")
    if grouping:
        if grouping == "clusters":
            moves, table = cluster_moves(paths, sc["sensitivity"]["clusters"], int(sc["seed"]))
            info["clusters"] = table
        else:
            moves = dict((load_config("groups", paths).get("variants") or {})[grouping])
        info["moves"] = moves
        (vp.configs / "groups.yaml").write_text(yaml.safe_dump(grouping_doc(paths, moves), sort_keys=False),
                                                encoding="utf-8")
        fp = vp.configs / "features.yaml"           # features.yaml names each feature's group too (C5 validates both)
        fdoc = yaml.safe_load(fp.read_text(encoding="utf-8"))
        for f in fdoc["features"]:
            if f["id"] in moves:
                f["group"] = moves[f["id"]]
        fp.write_text(yaml.safe_dump(fdoc, sort_keys=False), encoding="utf-8")
    stages = list(spec.get("stages") or [])
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SensitivityError(f"{vid}: unknown stages {bad}; known {STAGES}")
    fresh = set(ALWAYS_FRESH)
    if "cohort" in stages:
        fresh |= {"cohort.parquet", "cohort.meta.json"}
    if "features" in stages:
        fresh |= {"features", "peer"}
    if "train" in stages:
        fresh |= {"models/primary", "scores/primary"}
    inj = paths.processed / "injection"
    main_draws = sorted(int(p.name[4:]) for p in inj.glob("draw*")) if inj.is_dir() else []
    skip = {f"injection/draw{d}" for d in main_draws if d not in draws}
    for d in draws:
        if "inject_train" in stages:
            fresh |= {f"injection/draw{d}/models", f"injection/draw{d}/scores"}
        if "inject_explain" in stages or grouping:
            fresh |= {f"injection/draw{d}/attributions"}
    if paths.interim.is_dir():
        qa._link(paths.interim.resolve(), vp.interim)
    mirror(paths.processed, vp.processed, fresh, skip)
    if "train" in stages:                       # tuned hyperparameters of the main project (D105)
        for fam in fams:
            for t in sorted((paths.processed / "models" / "primary" / fam).glob("fold*/tuning.json")):
                dst = vp.processed / "models" / "primary" / fam / t.parent.name / "tuning.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(t, dst)
    freeze = qa.load_freeze(paths)
    rec = {"recorded_utc": datetime.now(timezone.utc).isoformat(), "protocol_hashes": qa.protocol_hashes(vp),
           "variant_of": {"variant": vid, "what": spec.get("what"), "main_recorded_utc": freeze["recorded_utc"],
                          "main_protocol_hashes": freeze["protocol_hashes"]}}
    (vp.phase2).mkdir(parents=True, exist_ok=True)
    (vp.phase2 / qa.FREEZE_FILE).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    sj.write_text(json.dumps({"fp": fp, "info": info}, default=str), encoding="utf-8")   # written last
    return vp, info


def write_explained(paths: Paths, vp: Paths, n: int, seed: int, fams: list[str], trained: bool) -> int:
    """The variant explains a seeded subsample of the main explained set, restricted to the provider-years the
    variant scores (e.g. individuals only)."""
    sub = subsample(paths, n, seed)
    es = load_explained_set(paths).with_columns(pl.col("year").cast(pl.Int32)).join(
        sub.select("npi", "year"), on=["npi", "year"], how="semi")
    if trained:
        keep = pl.concat([train.load_scores(vp, "primary", f).select("npi", pl.col("year").cast(pl.Int32)).unique()
                          for f in fams]).unique()
        es = es.join(keep, on=["npi", "year"], how="semi")
    p = explained_path(vp)
    p.parent.mkdir(parents=True, exist_ok=True)
    es.write_parquet(p)
    return es.height


# ----------------------------------------------------------------------------------------------- evaluation

def detection(paths: Paths, fams: list[str]) -> dict:
    ev = load_config("experiment", paths)["evaluation"]
    ks = [float(k) for k in ev["k_fracs"]]
    out = {}
    for f in fams:
        try:
            s = train.load_scores(paths, "primary", f).filter(pl.col("seed") == 0)
        except train.TrainError:
            continue
        s = metrics.screening_score(s.select("npi", "year", "fold", "y", "margin"), ev.get("ranking", "pooled_margin"))
        p = metrics.detection_metrics(s.select("npi", "year", "y", "margin"), ks)["pooled"]
        out[f] = {"N": p["N"], "positives": p["positives"], "pr_auc": p["pr_auc"], "roc_auc": p["roc_auc"],
                  **{f"p@{k}": p[f"p@{k}"] for k in ks}}
    return out


def agreement_with_main(paths: Paths, vp: Paths, fams: list[str]) -> list[dict]:
    rows = []
    for f in fams:
        try:
            a, b = load_attributions(paths, f, "peer"), load_attributions(vp, f, "peer")
        except FileNotFoundError:
            continue
        j = b.select("npi", "year", "v", *[m for m in ("A", "B", "C", "D", "E") if m in b.columns]).join(
            a, on=["npi", "year"], how="inner", suffix="_main")
        if j.is_empty():
            continue
        for m in evaluate.METHODS:
            if not evaluate.method_available(j, m):
                continue
            va = evaluate.group_vectors(j, m)[:, 1:]
            main = j.select("npi", "year", *[pl.col(f"{c}_main").alias(c) for c in ("v", "A", "B", "C", "D", "E")
                                             if f"{c}_main" in j.columns])
            vm = evaluate.group_vectors(main, m)[:, 1:]
            ok = np.isfinite(va).all(axis=1) & np.isfinite(vm).all(axis=1)
            if not ok.any():
                continue
            tau = kendall_tau_b(va[ok], vm[ok])
            rows.append({"family": f, "method": m, "n": int(ok.sum()),
                         "top1_agreement": float(np.mean(np.argmax(va[ok], 1) == np.argmax(vm[ok], 1))),
                         "kendall_median": float(np.nanmedian(tau)) if np.isfinite(tau).any() else None})
    return rows


def _compact_consistency(res: dict) -> list[dict]:
    return [{"gbm": r["gbm"], "method": r["method"], "glass_box": r["glass_box"], "n": r["n"],
             **{c: r[c]["median"] for c in evaluate.CONS_COLS}, "perm_cosine": r["cosine"]["perm_median"]}
            for r in res.get("summary", []) if r["stratum"] == "all"]


def evaluate_variant(paths: Paths, vp: Paths, fams: list[str], stages: list[str]) -> dict:
    out = {"detection": detection(vp, fams), "detection_main": detection(paths, fams)}
    if "explain" in stages:
        best = [f for f in fams if f in load_config("attribution", vp)["shap_families"]]
        gbs = [f for f in fams if f not in best]
        with as_root(vp.root):
            cons = evaluate.run_consistency(vp, gbms=best, glass_boxes=gbs)
            faith = evaluate.run_faithfulness(vp, families=fams)
        out["consistency"] = _compact_consistency(cons)
        out["faithfulness"] = [r for r in faith.get("summary", []) if r["subset"] == "all"]
        out["agreement_with_main"] = agreement_with_main(paths, vp, fams)
    if "inject_explain" in stages:
        with as_root(vp.root):
            corr = evaluate.run_correctness(vp, families=fams)
        out["correctness"] = corr.get("table", [])
    return out


def from_main(paths: Paths, kind: str) -> dict:
    """S5 (baselines) and S9 (monotone) from the main project's evaluation (run `phase6 evaluate` first)."""
    if kind == "baselines":
        b = evaluate.load_part(paths, "baselines") or evaluate.run_baselines(paths)
        return {"baselines": b.get("summary", [])}
    if kind == "monotone":
        mcfg = load_config("models", paths)
        pairs = [(g, f"{g}_mono") for g in mcfg["gbm"] if f"{g}_mono" in mcfg["gbm_mono"]]
        fams = [f for p in pairs for f in p]
        corr = evaluate.load_part(paths, "correctness") or {}
        faith = evaluate.load_part(paths, "faithfulness") or {}
        cons = evaluate.load_part(paths, "consistency") or {}
        return {"pairs": pairs, "detection": detection(paths, fams),
                "correctness": [r for r in corr.get("table", []) if r["family"] in fams],
                "faithfulness": [r for r in faith.get("summary", []) if r["family"] in fams and r["subset"] == "all"],
                "consistency": [r for r in _compact_consistency(cons) if r["gbm"] in fams]}
    raise SensitivityError(f"unknown from_main kind {kind!r}")


# ----------------------------------------------------------------------------------------------- orchestration

def run_variant(paths: Paths, vid: str, *, resume: bool = True) -> dict:
    """One variant. With `resume`, a variant whose result records the same fingerprint is skipped, and an interrupted
    one continues in its scratch project."""
    sc = load_config("evaluation", paths)
    spec = (sc["sensitivity"]["variants"] or {}).get(vid)
    if spec is None:
        raise SensitivityError(f"unknown variant {vid!r}; known {sorted(sc['sensitivity']['variants'])}")
    if spec.get("from_main"):
        res = {"variant": vid, "what": spec.get("what"), "from_main": spec["from_main"], **from_main(paths, spec["from_main"])}
        write_json(evaluate.out_dir(paths) / "sensitivity" / f"{vid}.json", res)
        return res
    fams = variant_families(paths, spec)
    draws = [int(d) for d in sc["sensitivity"]["draws"]]
    stages = list(spec.get("stages") or [])
    fp = variant_fingerprint(paths, vid, spec, fams, draws)
    rp = evaluate.out_dir(paths) / "sensitivity" / f"{vid}.json"
    if resume and rp.is_file():
        done = json.loads(rp.read_text(encoding="utf-8"))
        if done.get("fingerprint") == fp:
            return done
    vp, info = setup(paths, vid, spec, fams, draws, resume=resume, fp=fp)
    res: dict = {"variant": vid, "what": spec.get("what"), "families": fams, "stages": stages,
                 "root": paths.rel(vp.root), "fingerprint": fp, **info}
    with as_root(vp.root):
        if "cohort" in stages:
            cohort.build_cohort(vp)
        if "features" in stages:
            features.build_features(vp)
        if "train" in stages:
            try:
                train.train_design(vp, "primary", families=fams)
            except train.TrainError as e:
                if "min_valid_positives" not in str(e):
                    raise
                # D134: too few validation positives to fit under the protocol's guard: reported, not forced
                res["not_estimable"] = str(e)
                write_json(evaluate.out_dir(paths) / "sensitivity" / f"{vid}.json", res)
                return res
        if "explain" in stages or spec.get("grouping"):
            res["explained"] = write_explained(paths, vp, int(sc["sensitivity"]["explain_n"]), int(sc["seed"]), fams,
                                              "train" in stages)
            explain_store(vp, families=fams, baselines=["peer"])
        for d in draws:
            if "inject_train" in stages:
                inject_models.train_models(vp, d, families=fams)
            if "inject_explain" in stages:
                inject_models.explain(vp, d, families=fams, baselines=["peer"],
                                      limit=sc["sensitivity"].get("inject_limit"))
    res.update(evaluate_variant(paths, vp, fams, stages))
    write_json(evaluate.out_dir(paths) / "sensitivity" / f"{vid}.json", res)
    return res


def run(paths: Paths, variants=None, *, resume: bool = True) -> dict:
    train.check_frozen(paths)
    sc = load_config("evaluation", paths)
    ids = list(variants or sc["sensitivity"]["variants"])
    return {vid: {"done": True, "families": run_variant(paths, vid, resume=resume).get("families")} for vid in ids}


def load_results(paths: Paths) -> dict[str, dict]:
    d = evaluate.out_dir(paths) / "sensitivity"
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))} if d.is_dir() else {}
