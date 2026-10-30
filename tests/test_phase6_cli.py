"""Phase 6 end to end through the CLI on synthetic data: phase2 -> phase3 (lr, ebm, xgb; 2 seeds) -> phase4 ->
phase5 (draw 0) -> phase6 evaluate, stability, sensitivity (a subset of variants), report. Every number here comes
from tests/synth.py and is not a result."""

from __future__ import annotations

import json
import shutil

import numpy as np
import polars as pl
import pytest
import yaml

from scenario_fraud import cli, evaluate, features, sensitivity, stability
from scenario_fraud.attribute import load_attributions
from scenario_fraud.config import load_config
from scenario_fraud.paths import Paths
from tests.conftest import REPO, copy_configs
from tests.synth import make_synthetic

FAMS = ["lr", "ebm", "xgb"]
VARIANTS = ["S4", "S5", "S6", "S7", "S8", "S9", "S10"]


def _small(root):
    def edit(name, fn):
        p = root / "configs" / f"{name}.yaml"
        d = yaml.safe_load(p.read_text())
        fn(d)
        p.write_text(yaml.safe_dump(d, sort_keys=False))

    def models(m):
        m["min_valid_positives"] = 1
        m["ebm"]["fixed"]["outer_bags"] = 1
        m["designs"]["primary"]["seeds"] = 2
        m["xgb"]["space"]["n_estimators"] = {"type": "int", "low": 20, "high": 40}

    def attribution(a):
        a["background"].update(B=20, min_background=8, B_global=40)
        a["explained"]["random_n"] = 30
        a["coalition_families"] = FAMS
        a["shap_families"] = ["xgb"]

    def inj(i):
        i["test"].update(per_cell=3, per_pair=3)
        i["train"]["share"] = 0.02
        i["families"] = FAMS

    def ev(e):
        e["stability"].update(n=24, families=["lr", "xgb"], bootstraps=1, perturbations=1)
        e["sensitivity"].update(explain_n=24, inject_limit=2)
        e["sensitivity"]["clusters"]["max_rows"] = 2000

    edit("models", models)
    edit("attribution", attribution)
    edit("injection", inj)
    edit("evaluation", ev)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("w6") / "proj with space"
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
        assert cli.main(["phase3", "report"]) == 0
        for c in (["explained-set"], ["explain", "--limit", "6"]):
            assert cli.main(["phase4", *c]) == 0
        for c in (["build", "--draw", "0"], ["train", "--draw", "0"], ["stress", "--draw", "0"],
                  ["explain", "--draw", "0", "--limit", "4"]):
            assert cli.main(["phase5", *c]) == 0
        codes = [cli.main(["phase6", *c]) for c in
                 (["evaluate"], ["stability"], ["sensitivity", *[x for v in VARIANTS for x in ("--variant", v)]],
                  ["report"])]
    return root, codes


@pytest.fixture
def w(world, tmp_path, monkeypatch) -> Paths:
    root = tmp_path / "proj with space"
    shutil.copytree(world[0], root, symlinks=True)
    monkeypatch.setenv("SCENARIO_FRAUD_ROOT", str(root))
    return Paths(root)


def test_commands_and_report(world, w):
    assert world[1] == [0] * 4
    md = (w.phase6 / "evaluation_report.md").read_text()
    assert "SYNTHETIC DATA: NOT A RESULT" in md
    for h in ("T1 Detection", "T2 Correctness", "T3 Consistency", "T4 Faithfulness", "E8", "E9", "E10", "§10.8"):
        assert h in md
    for part in evaluate.PARTS:
        assert (w.phase6 / f"{part}.json").is_file()


def test_correctness_rows_and_chance(w):
    c = evaluate.load_part(w, "correctness")
    assert c["available"] and c["table"]
    rows = pl.read_parquet(w.processed / "evaluation" / "correctness_peer.parquet")
    assert set(rows["method"]) >= {"A", "B", "C", "D", "E"}
    dme = rows.filter(pl.col("schemes_key") == "dme_mill")
    if dme.height:
        assert np.allclose(dme["chance_top1"].to_numpy(), 2 / 7)
    ok = rows.filter(pl.col("valid"))
    assert ok["top1"].is_between(0, 1).all() and (ok["top3"] >= ok["top1"]).all()


def test_consistency_uses_flag_strata(w):
    c = evaluate.load_part(w, "consistency")
    assert c["available"]
    assert {r["stratum"] for r in c["summary"]} == {"all", *evaluate.STRATA}
    assert {r["glass_box"] for r in c["summary"]} == {"lr", "ebm"}


def test_faithfulness_matches_store(w):
    f = evaluate.load_part(w, "faithfulness")
    att = load_attributions(w, "xgb", "peer")
    keep = (att["margin"] - att["base"]) > 1e-12
    stored = att.filter(keep)["del_B"].mean()
    row = next(r for r in f["summary"] if r["family"] == "xgb" and r["method"] == "B" and r["subset"] == "all")
    assert row["deletion"] == pytest.approx(stored, rel=1e-9)
    assert len(f["curves"]["xgb/peer"]["B"]) == 9


def test_stability_runs(w):
    s = json.loads((w.phase6 / "stability.json").read_text())
    assert set(s["families"]) == {"lr", "xgb"} and s["families"]["xgb"]["seeds"] == 2
    r = {x["method"]: x for x in s["families"]["xgb"]["summary"]}
    assert r["B"]["seed_runs"] == 1 and r["B"]["boot_runs"] == 1 and r["B"]["perturb_runs"] == 1
    assert 0 <= r["B"]["top1_agree_seed"] <= 1 and -1 <= r["B"]["kendall_seeds"] <= 1
    assert (w.processed / "stability" / "models" / "xgb" / "boot0" / "fold0" / "fitted.json").is_file()


def test_unperturbed_shrunk_z_reproduces_features(w):
    fcfg = features.FeatureConfig.load(w)
    f = features.load_features(w, [2022])
    st = pl.read_parquet(w.processed / "peer" / "2022" / "stats.parquet")
    z = stability.shrunk_z(f, st, fcfg)
    assert z
    for c, v in z.items():
        np.testing.assert_allclose(v, f[c].to_numpy(), atol=1e-9)


def test_sensitivity_variants(w):
    res = sensitivity.load_results(w)
    assert set(res) == set(VARIANTS)
    s4 = res["S4"]
    assert s4["detection"] and s4["agreement_with_main"]
    vp = Paths(w.processed / "sensitivity" / "S4")
    assert (vp.processed / "models" / "primary").is_dir() and not (vp.processed / "models" / "primary").is_symlink()
    assert (vp.processed / "labels_primary_W3.parquet").is_symlink()
    fr = json.loads((vp.phase2 / "G2_freeze.json").read_text())
    assert fr["variant_of"]["variant"] == "S4"
    s7 = res["S7"]
    assert s7["moves"] == {"CH2": "overutilization", "PM1": "context"} and s7["correctness"]
    cols = json.loads((Paths(w.processed / "sensitivity" / "S7").processed / "attributions" / "xgb" / "columns.json")
                      .read_text())
    assert cols["col_group"][cols["columns"].index("CH2")] == 2 and cols["col_group"][cols["columns"].index("PM1")] == 0
    s8 = res["S8"]
    cols8 = json.loads((Paths(w.processed / "sensitivity" / "S8").processed / "models" / "primary" / "xgb" / "fold0"
                        / "columns.json").read_text())
    assert not any(c.startswith("CX") for c in cols8) and s8["correctness"]
    assert len(res["S6"]["clusters"]) <= 7 and res["S6"]["correctness"]
    assert res["S9"]["pairs"] and "baselines" in res["S5"]
    s10 = Paths(w.processed / "sensitivity" / "S10")
    coh = pl.read_parquet(s10.processed / "cohort.parquet")
    assert (coh["peer_level"] != "spec_ent_div").all()
    assert not (s10.processed / "cohort.parquet").is_symlink()
    # variants write their own stores, never through a link into the main project
    for v in ("S4", "S7", "S10"):
        a = Paths(w.processed / "sensitivity" / v).processed / "attributions"
        assert a.is_dir() and not a.is_symlink()


def test_stability_resumes(w, capsys):
    """Runs already written are reused (same values); a deleted run is recomputed identically; a run-count change
    keeps the runs; a subsample change recomputes the family."""
    d = w.processed / "stability" / "xgb"
    before = json.loads((w.phase6 / "stability.json").read_text())["families"]["xgb"]
    ref = pl.read_parquet(d / "perturb0.parquet")
    (d / "perturb0.parquet").unlink()
    stability.run(w, families=["xgb"])
    err = capsys.readouterr().err
    assert "xgb perturb0:" in err and "seed0:" not in err and "boot0:" not in err
    assert pl.read_parquet(d / "perturb0.parquet").equals(ref)
    assert json.loads((w.phase6 / "stability.json").read_text())["families"]["xgb"] == before
    e = yaml.safe_load((w.configs / "evaluation.yaml").read_text())
    e["stability"]["perturbations"] = 0                  # a count change keeps the runs written
    (w.configs / "evaluation.yaml").write_text(yaml.safe_dump(e, sort_keys=False))
    stability.run(w, families=["xgb"])
    assert "recomputing" not in capsys.readouterr().err and (d / "perturb0.parquet").is_file()
    e["stability"]["n"] = 20                             # a different subsample does not
    (w.configs / "evaluation.yaml").write_text(yaml.safe_dump(e, sort_keys=False))
    stability.run(w, families=["xgb"], kinds=("seed",))
    assert "settings or models changed; recomputing" in capsys.readouterr().err
    assert not (d / "perturb0.parquet").exists() and (d / "seed0.parquet").is_file()


def test_sensitivity_resumes(w, monkeypatch):
    """A finished variant is skipped; an interrupted one keeps its scratch project (setup.json)."""
    vid = VARIANTS[0]
    res = json.loads((w.phase6 / "sensitivity" / f"{vid}.json").read_text())
    assert res.get("fingerprint")
    calls = []
    monkeypatch.setattr(sensitivity, "setup", lambda *a, **k: calls.append(1))
    assert sensitivity.run_variant(w, vid) == res and calls == []
    monkeypatch.undo()
    monkeypatch.setenv("SCENARIO_FRAUD_ROOT", str(w.root))
    root = sensitivity.variant_root(w, vid)
    marker = root / "kept.txt"
    marker.write_text("x")
    spec = load_config("evaluation", w)["sensitivity"]["variants"][vid]
    fams = sensitivity.variant_families(w, spec)
    sensitivity.setup(w, vid, spec, fams, [0])
    assert marker.is_file()
    sensitivity.setup(w, vid, spec, fams, [0], resume=False)
    assert not marker.exists()


def test_lean_design_gives_the_same_matrix(w):
    """D133: the lean design (model columns as Float32, no raw__) yields the model matrix of the full design."""
    from scenario_fraud.modelmatrix import load_design, matrix
    full, lean = load_design(w, "primary"), load_design(w, "primary", lean=True)
    assert full.columns == lean.columns and not any(c.startswith(features.RAW) for c in lean.frame.columns)
    np.testing.assert_array_equal(matrix(full.frame, full.columns), matrix(lean.frame, lean.columns))
    assert full.frame.select("npi", "year", "y", "fold").equals(lean.frame.select("npi", "year", "y", "fold"))


def test_variant_without_validation_positives_is_not_estimable(w, monkeypatch):
    """D134: a variant whose training stops on the min_valid_positives guard is recorded as not estimable."""
    from scenario_fraud import train as train_mod

    def refuse(*a, **k):
        raise train_mod.TrainError("primary fold 0: tune_valid has 0 positives (< models.yaml min_valid_positives = 5)")

    monkeypatch.setattr(train_mod, "train_design", refuse)
    res = sensitivity.run_variant(w, "S4", resume=False)
    assert "min_valid_positives" in res["not_estimable"]
    assert json.loads((w.phase6 / "sensitivity" / "S4.json").read_text())["not_estimable"] == res["not_estimable"]
    monkeypatch.setattr(train_mod, "train_design", lambda *a, **k: (_ for _ in ()).throw(train_mod.TrainError("other")))
    with pytest.raises(train_mod.TrainError):
        sensitivity.run_variant(w, "S4", resume=False)
