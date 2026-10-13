"""Phase 7 end to end through the CLI on synthetic data: phase2 -> phase3 (lr, ebm, xgb) -> phase4 -> phase5 (draw 0)
-> phase6 evaluate + report -> phase7 render, render-sample, queue, tables, figures, draft, package. Every number
here comes from tests/synth.py and is not a result."""

from __future__ import annotations

import json
import shutil

import polars as pl
import pytest
import yaml

from scenario_fraud import cli, render
from scenario_fraud.attribute import load_attributions
from scenario_fraud.config import load_config
from scenario_fraud.paths import Paths
from tests.conftest import REPO
from tests.synth import make_synthetic

FAMS = ["lr", "ebm", "xgb"]


def _small(root):
    def edit(name, fn):
        p = root / "configs" / f"{name}.yaml"
        d = yaml.safe_load(p.read_text())
        fn(d)
        p.write_text(yaml.safe_dump(d, sort_keys=False))

    def models(m):
        m["min_valid_positives"] = 1
        m["ebm"]["fixed"]["outer_bags"] = 1
        m["designs"]["primary"]["seeds"] = 1
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

    edit("models", models)
    edit("attribution", attribution)
    edit("injection", inj)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("w7") / "proj with space"
    (root / "design").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    shutil.copytree(REPO / "configs", root / "configs")
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
        for c in (["evaluate"], ["report"]):
            assert cli.main(["phase6", *c]) == 0
        codes = [cli.main(["phase7", *c]) for c in (["report"],)]
    return root, codes


@pytest.fixture
def w(world, tmp_path, monkeypatch) -> Paths:
    root = tmp_path / "proj with space"
    shutil.copytree(world[0], root, symlinks=True)
    monkeypatch.setenv("SCENARIO_FRAUD_ROOT", str(root))
    return Paths(root)


def test_report_outputs(world, w):
    assert world[1] == [0]
    for t in ("T1_detection", "T2_correctness", "T3_consistency", "T4_faithfulness_stability", "T6_fairness"):
        for ext in ("md", "csv", "tex"):
            assert (w.phase7 / "tables" / f"{t}.{ext}").is_file(), (t, ext)
    assert "SYNTHETIC DATA: NOT A RESULT" in (w.phase7 / "tables" / "T2_correctness.md").read_text()
    figs = {p.stem for p in (w.phase7 / "figures").glob("*.png")}
    assert {"F2_scheme_importance", "F3_recovery", "F4_consistency", "F5_deletion_curves", "F6_interaction_matrix",
            "F7_peer_vs_global", "F8_fairness"} <= figs
    draft = (w.phase7 / "draft.md").read_text()
    assert "SYNTHETIC DATA" in draft and "T2 Correctness" in draft and "[Authors:" in draft
    pkg = json.loads((w.phase7 / "reproducibility.json").read_text())
    assert pkg["g2_freeze"] and pkg["run_logs"] and "evaluation.yaml" in pkg["configs_sha256"]
    assert (w.phase7 / "REPRODUCE.md").is_file()
    assert list((w.phase7 / "views").glob("queue_*.md"))


def test_records_match_schema_and_rules(w):
    recs = sorted((w.phase7 / "records").glob("*.json"))
    assert len(recs) == 3
    vcfg = load_config("views", w)
    for p in recs:
        rec = json.loads(p.read_text())
        render.validate(w, rec)
        assert rec["synthetic"] and rec["decision"]["mock"]
        assert all(i["group"] != "context" for i in rec["indicators"])
        assert all(i["contribution"] >= vcfg["min_contribution"] for i in rec["indicators"])
        assert {n["audience"] for n in rec["notices"]} == {"provider", "beneficiary"}
        d = w.phase7 / "views" / rec["record_id"]
        for name in ("analyst", "provider", "beneficiary", "oversight"):
            assert (d / f"{name}.md").is_file() and (d / f"{name}.html").is_file()
        prov = (d / "provider.md").read_text()
        assert prov.startswith("MOCK-UP") and "risk" not in prov.lower() and "score" not in prov.lower()
        render.check_external(prov, "provider", rec, vcfg)
        render.check_external((d / "beneficiary.md").read_text(), "beneficiary", rec, vcfg)
        assert "Rank" in (d / "analyst.md").read_text()


def test_record_values_come_from_the_store(w):
    rec = json.loads(sorted((w.phase7 / "records").glob("*.json"))[0].read_text())
    p = rec["provider"]
    att = load_attributions(w, rec["score"]["family"], "peer").filter(
        (pl.col("npi") == p["npi"]) & (pl.col("year") == p["year"])).row(0, named=True)
    assert rec["score"]["margin"] == pytest.approx(att["margin"], abs=1e-5)
    assert [rec["attributions"]["methods"]["B"][g] for g in rec["attributions"]["methods"]["B"]] == pytest.approx(att["B"])
    assert rec["score"]["rank"] >= 1 and rec["score"]["n_scored"] > rec["score"]["rank"] - 1


def test_external_rules_reject_leaks(w):
    rec = json.loads(sorted((w.phase7 / "records").glob("*.json"))[0].read_text())
    vcfg = load_config("views", w)
    with pytest.raises(render.RenderError, match="forbidden"):
        render.check_external("Your fraud risk score is high.", "provider", rec, vcfg)
    with pytest.raises(render.RenderError, match="forbidden"):
        render.check_external("This is part of an investigation.", "beneficiary", rec, vcfg)
    bad = dict(rec, indicators=[{**rec["indicators"][0], "group": "context"}]) if rec["indicators"] else None
    if bad:
        with pytest.raises(render.RenderError, match="schema"):
            render.validate(w, bad)


def test_no_decision_no_notice(w):
    n, y = render.sample(w, 1).row(0)
    out = render.render(w, n, int(y))
    assert set(out["views"]) == {"analyst", "oversight"}
    rec = json.loads((w.root / out["record"]).read_text())
    assert rec["decision"] is None and rec["notices"] == []
