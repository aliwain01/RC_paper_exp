"""Phase 4 end to end through the CLI on synthetic data: phase2 -> phase3 (lr, ebm, xgb) -> phase4 explained-set ->
explain -> diagnostics -> bench -> report. Every number here comes from tests/synth.py and is not a result."""

from __future__ import annotations

import json
import shutil

import numpy as np
import polars as pl
import pytest
import yaml

from scenario_fraud import attribute, cli, explained_set, splits, train
from scenario_fraud.paths import Paths
from tests.conftest import REPO, copy_configs
from tests.synth import make_synthetic

FAMS = ["lr", "ebm", "xgb"]


def _small(root):
    m = yaml.safe_load((root / "configs" / "models.yaml").read_text())
    m["min_valid_positives"] = 1
    m["ebm"]["fixed"]["outer_bags"] = 1
    m["designs"]["primary"]["seeds"] = 1
    m["xgb"]["space"]["n_estimators"] = {"type": "int", "low": 20, "high": 40}
    (root / "configs" / "models.yaml").write_text(yaml.safe_dump(m, sort_keys=False))
    a = yaml.safe_load((root / "configs" / "attribution.yaml").read_text())
    a["background"].update(B=20, min_background=8, B_global=40)
    a["explained"]["random_n"] = 30
    a["coalition_families"] = FAMS
    a["global_families"] = FAMS
    a["shap_families"] = ["xgb"]
    a["bench"]["rows"] = 20000
    a["diagnostics"]["path_dependent_n"] = 20
    (root / "configs" / "attribution.yaml").write_text(yaml.safe_dump(a, sort_keys=False))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("w4") / "proj with space"
    (root / "design").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    copy_configs(root)
    _small(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SCENARIO_FRAUD_ROOT", str(root))
        make_synthetic(Paths(root), n_npi=3000, excluded_frac=0.15)
        for c in (["cohort"], ["features"], ["folds"], ["qa", "--no-rebuild"], ["report"]):
            assert cli.main(["phase2", *c]) == 0
        assert cli.main(["phase3", "train", "--trials", "1", *[x for f in FAMS for x in ("--family", f)]]) == 0
        codes = [cli.main(["phase4", *c]) for c in (["explained-set"], ["explain", "--limit", "8"], ["diagnostics"],
                                                     ["bench"], ["report"])]
    return root, codes


@pytest.fixture
def w(world, tmp_path, monkeypatch) -> Paths:
    root = tmp_path / "proj with space"
    shutil.copytree(world[0], root, symlinks=True)
    monkeypatch.setenv("SCENARIO_FRAUD_ROOT", str(root))
    return Paths(root)


def test_commands_and_report(world, w):
    assert world[1] == [0] * 5
    md = (w.phase4 / "G4_report.md").read_text()
    assert "SYNTHETIC DATA: NOT A RESULT" in md and "Timing benchmark" in md
    rep = json.loads((w.phase4 / "G4_report.json").read_text())
    assert {(r["family"], r["baseline"]) for r in rep["store_checks"]} == {(f, b) for f in FAMS for b in ("peer", "global")}
    assert all(r["additivity"] == "pass" for r in rep["store_checks"])
    assert set(rep["bench"]["families"]) == set(FAMS)


def test_explained_set_top_is_best_gbm_top(w):
    es = explained_set.load_explained_set(w)
    gbm = es["score_family"][0]
    s = train.load_scores(w, "primary", gbm).filter(pl.col("seed") == 0)
    for yr in (2022, 2023):
        sy = s.filter(pl.col("year") == yr)
        n_top = int(np.ceil(0.02 * sy.height))
        want = set(sy.sort(["margin", "npi"], descending=[True, False]).head(n_top)["npi"])
        assert set(es.filter((pl.col("year") == yr) & pl.col("top"))["npi"]) == want
    assert es.filter(pl.col("random")).group_by("year").len()["len"].to_list() == [30, 30]


def test_attribution_margin_matches_scores_and_backgrounds_are_out_of_fold(w):
    folds = splits.load_folds(w)
    for fam in FAMS:
        a = attribute.load_attributions(w, fam, "peer")
        s = train.load_scores(w, "primary", fam).filter(pl.col("seed") == 0)
        j = a.select("npi", "year", "margin").join(s.select("npi", "year", pl.col("margin").alias("m")),
                                                   on=["npi", "year"])
        assert j.height == a.height
        np.testing.assert_allclose(j["margin"], j["m"], rtol=1e-5, atol=1e-5)    # v(N) = f(x)
    bg = pl.concat([pl.read_parquet(p) for p in (w.processed / "attributions" / "backgrounds").glob("*.parquet")])
    j = bg.join(folds.select("npi", pl.col("fold").alias("member_fold")), on="npi")
    assert j.filter(pl.col("member_fold") == pl.col("fold")).height == 0         # training-fold providers only
    a = attribute.load_attributions(w, "xgb", "peer")
    used = a.select("bg_key", "year", "fold").unique()
    assert used.join(bg.select("bg_key", "year", "fold").unique(), on=["bg_key", "year", "fold"], how="anti").height == 0


def test_methods_present_per_family(w):
    x = attribute.load_attributions(w, "xgb", "peer")
    assert x["A"].null_count() == 0 and x["D"].null_count() == x.height
    for fam in ("lr", "ebm"):
        g = attribute.load_attributions(w, fam, "global")
        assert g["D"].null_count() == 0 and g["A"].null_count() == g.height
        assert set(g["bg_level"]) == {"global"}
    assert len(x["v"][0]) == 256 and len(x["C"][0]) == 64 and len(x["A_feat"][0]) == len(
        json.loads((w.processed / "attributions" / "xgb" / "columns.json").read_text())["columns"])


def test_diagnostics_written(w):
    d = json.loads((w.phase4 / "diagnostics.json").read_text())
    assert set(d) == {"xgb"} and all(r["n"] > 0 for r in d["xgb"])


def test_explain_refuses_after_protocol_change(w):
    e = w.configs / "experiment.yaml"
    e.write_text(e.read_text().replace("min_nondrug_services: 50", "min_nondrug_services: 40"))
    with pytest.raises(train.TrainError, match="protocol changed"):
        attribute.explain(w, families=["lr"], limit=2)


def _chunk_rows(w, n):
    p = w.configs / "attribution.yaml"
    a = yaml.safe_load(p.read_text())
    a["chunk_rows"] = n
    p.write_text(yaml.safe_dump(a, sort_keys=False))


def _store(w, base):
    return pl.read_parquet(attribute.store_path(w, "xgb", base, 0)).sort(["year", "npi"])


def _same(a, b):
    """Equal up to floating-point rounding: batch sizes change BLAS summation order (about 1e-15)."""
    assert a.columns == b.columns and a.height == b.height
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype in (pl.Float64, pl.List(pl.Float64)):
            xa = np.array(x.to_list(), dtype=float) if x.dtype == pl.Float64 else x.explode().to_numpy()
            ya = np.array(y.to_list(), dtype=float) if y.dtype == pl.Float64 else y.explode().to_numpy()
            np.testing.assert_allclose(xa, ya, rtol=0, atol=1e-12, equal_nan=True, err_msg=c)
        else:
            assert x.equals(y), c


def test_chunked_store_equals_unchunked(w):
    whole = {b: _store(w, b) for b in ("peer", "global")}
    _chunk_rows(w, 3)
    attribute.explain(w, families=["xgb"], folds=[0], limit=8, resume=False)
    assert len(list(attribute.parts_dir(w, "xgb", "peer", 2022, 0).glob("c*.parquet"))) >= 3
    for b in ("peer", "global"):
        _same(_store(w, b), whole[b])


def test_resume_skips_written_chunks_and_matches(w, monkeypatch):
    _chunk_rows(w, 3)
    attribute.explain(w, families=["xgb"], folds=[0], limit=8, resume=False)
    full = {b: _store(w, b) for b in ("peer", "global")}
    gone = []
    for b in ("peer", "global"):
        for d in (attribute.parts_dir(w, "xgb", b, yr, 0) for yr in (2022, 2023)):
            chunks = sorted(d.glob("c*.parquet"))
            for c in chunks[len(chunks) // 2:]:
                c.unlink()
                gone.append(c)
    calls = []
    real = attribute._run
    monkeypatch.setattr(attribute, "_run", lambda *a, **k: calls.append(1) or real(*a, **k))
    attribute.explain(w, families=["xgb"], folds=[0], limit=8)
    assert len(calls) == len(gone) > 0
    for b in ("peer", "global"):
        assert _store(w, b).equals(full[b])      # same chunking: identical bytes, not just close
    assert "already done, skipped" in (w.phase4 / "explain_progress.log").read_text()


def test_resume_recomputes_after_a_settings_change(w):
    _chunk_rows(w, 3)
    attribute.explain(w, families=["xgb"], folds=[0], limit=8, baselines=["global"], resume=False)
    p = w.configs / "attribution.yaml"
    a = yaml.safe_load(p.read_text())
    a["background"]["B_global"] = 30
    p.write_text(yaml.safe_dump(a, sort_keys=False))
    attribute.explain(w, families=["xgb"], folds=[0], limit=8, baselines=["global"])
    assert "settings or providers changed" in (w.phase4 / "explain_progress.log").read_text()
    assert (_store(w, "global")["bg_n"] == 30).all()


def test_global_baseline_only_for_global_families(w):
    p = w.configs / "attribution.yaml"
    a = yaml.safe_load(p.read_text())
    a["global_families"] = ["xgb"]
    p.write_text(yaml.safe_dump(a, sort_keys=False))
    shutil.rmtree(w.processed / "attributions" / "lr" / "global")
    out = attribute.explain(w, families=["lr", "xgb"], folds=[0], limit=4)
    assert not (w.processed / "attributions" / "lr" / "global").exists()
    assert any("xgb/global" in x for x in out["written"]) and any("lr/peer" in x for x in out["written"])
