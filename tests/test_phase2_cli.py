"""Phase 2 end to end through the CLI on synthetic data (tests/synth.py): cohort -> features -> folds -> qa ->
report; then the QA leakage checks on tampered copies, the synthetic-data statement, and the G2 protocol diff.
Every number here comes from the synthetic generator and is not a result."""

from __future__ import annotations

import json
import shutil

import polars as pl
import pytest
import yaml

from scenario_fraud import cli, cohort, features, g2_report, provenance, qa, splits
from scenario_fraud.paths import Paths
from tests.conftest import REPO
from tests.synth import make_synthetic

YEARS = range(2018, 2025)
COMMANDS = (["cohort"], ["features"], ["folds"], ["qa"], ["report"])


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """One synthetic world taken through every phase2 command (built once, copied per test)."""
    root = tmp_path_factory.mktemp("world") / "proj with space"
    (root / "design").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    shutil.copytree(REPO / "configs", root / "configs")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SCENARIO_FRAUD_ROOT", str(root))
        p = Paths(root)
        truth = make_synthetic(p, n_npi=1200, years=YEARS)
        codes = [cli.main(["phase2", *c]) for c in COMMANDS]
    return root, truth, codes


@pytest.fixture
def w(world, tmp_path, monkeypatch) -> Paths:
    root = tmp_path / "proj with space"
    shutil.copytree(world[0], root, symlinks=True)
    monkeypatch.setenv("SCENARIO_FRAUD_ROOT", str(root))
    return Paths(root)


def _qa(paths: Paths, *extra: str) -> dict:
    assert cli.main(["phase2", "qa", *extra]) == 0
    return json.loads((paths.phase2 / "QA_report.json").read_text())


def _status(rep: dict) -> dict[str, str]:
    return {i["id"]: i["status"] for i in rep["leakage_checklist"]}


def test_phase2_end_to_end_reports_exist_and_say_synthetic(world, w):
    assert world[2] == [0] * len(COMMANDS)
    pr = w.processed
    assert (pr / "cohort.parquet").is_file() and (pr / "folds.parquet").is_file()
    assert all((pr / "features" / f"{y}.parquet").is_file() for y in YEARS)
    assert all((pr / "peer" / str(y) / "stats.parquet").is_file() for y in YEARS)
    for name in ("QA_report.md", "QA_report.json", "G2_report.md", "G2_report.json"):
        assert (w.phase2 / name).is_file(), name
    runs = sorted((w.phase2 / "runs").glob("*.json"))
    logged = {json.loads(r.read_text())["command"]: json.loads(r.read_text())["status"] for r in runs}
    assert logged == {c[0]: "ok" for c in COMMANDS}
    assert all(json.loads(r.read_text())["result"]["synthetic_inputs"] is True for r in runs)
    assert json.loads((pr / "features" / "2018.meta.json").read_text())["synthetic_inputs"] is True
    assert json.loads((pr / "folds.meta.json").read_text())["synthetic_inputs"] is True

    qmd, gmd = (w.phase2 / "QA_report.md").read_text(), (w.phase2 / "G2_report.md").read_text()
    assert "SYNTHETIC DATA: NOT A RESULT" in qmd and "SYNTHETIC DATA: NOT A RESULT" in gmd
    rep = json.loads((w.phase2 / "QA_report.json").read_text())
    assert rep["provenance"]["synthetic"] is True
    assert _status(rep) == {"L1": "pass", "L2": "pass", "L3": "partial", "L4": "pass", "L5": "pending", "L6": "pass"}
    l1 = {c["check"].split(" ")[0]: c for c in rep["leakage_checklist"][0]["checks"]}
    assert "identical" in l1["features"]["detail"]                       # the label-free rebuild ran
    for sec in ("1. Cohort per year", "2. Peer-level mix", "3. Features per year", "4. Undefined",
                "5. Data coverage CX8", "6. Near-duplicate", "7. Census-division refinement", "8. Folds and splits",
                "9. Leakage checklist"):
        assert f"## {sec}" in qmd, sec
    assert rep["region"] == [] and "No provider has a division-level peer cell" in qmd     # 1,200 NPIs: none
    # cohort counts agree with the cohort table; fold balance; every design x fold disjoint
    coh = pl.read_parquet(pr / "cohort.parquet")
    assert [r["in_cohort"] for r in rep["cohort"]["counts"]] == \
        coh.group_by("year").agg(pl.col("in_cohort").sum()).sort("year")["in_cohort"].to_list()
    sp = rep["folds"]["splits"]
    assert len(sp) == 5 * 5 and {r["design"] for r in sp} == {"primary", "O1", "O2", "O3", "showcase"}
    assert all(r["test_train_overlap_npis"] == 0 and r["tune_overlap_npis"] == 0 for r in sp)
    assert all(abs(r["holdout_share"] - 0.2) < 0.01 for r in sp)
    assert rep["folds"]["balance"]["max_stratum_spread"] <= 1 and rep["folds"]["labels"]
    assert sum(r["test_pos"] for r in sp if r["design"] == "primary") > 0

    g2 = json.loads((w.phase2 / "G2_report.json").read_text())
    assert any(r["feature"] == "CX7" for r in g2["not_computable"])
    assert {r["id"]: r["status"] for r in g2["features"]}["CX7"] == "not computable"
    assert g2["protocol_hashes"] == rep["protocol_hashes"]
    assert any("high-risk DME" in t for t in g2["decisions_needed"])
    assert any("DESIGN §19 Q2" in t for t in g2["decisions_needed"])
    assert "## 4. Protocol diff against v1.1" in gmd and "design/DESIGN_DOCUMENT.md not found" in gmd
    assert g2["freeze"]["status"] == "recorded"
    assert qa.load_freeze(w)["protocol_hashes"] == g2["protocol_hashes"]


def _unmark(paths: Paths, sidecars: bool = True) -> None:
    """Strip every synthetic marker: Parquet metadata, sidecar flag (or the sidecar itself), manifest notes."""
    for f in provenance.interim_files(paths.interim):
        pl.read_parquet(f).write_parquet(f)                              # rewritten without key-value metadata
        m = f.with_suffix(".meta.json")
        if sidecars:
            meta = json.loads(m.read_text())
            meta.pop("synthetic")
            meta["note"] = "harmonised"
            m.write_text(json.dumps(meta))
        else:
            m.unlink()
    man = json.loads(paths.manifest.read_text())
    for e in man["files"]:
        e["notes"] = ""
    paths.manifest.write_text(json.dumps(man))


def test_reports_do_not_say_synthetic_for_unmarked_inputs(w):
    _unmark(w)
    rep = _qa(w, "--no-rebuild")
    assert cli.main(["phase2", "report"]) == 0
    assert rep["provenance"]["synthetic"] is False and rep["provenance"]["unknown_provenance_files"] == []
    for name in ("QA_report.md", "G2_report.md"):
        assert "synthetic" not in (w.phase2 / name).read_text().lower(), name
    assert "Inputs: 28 harmonised CMS files" in (w.phase2 / "QA_report.md").read_text()
    assert "PROVENANCE UNKNOWN" not in (w.phase2 / "QA_report.md").read_text()


def test_synthetic_files_copied_without_sidecars_are_still_flagged(w):
    """Review finding: provenance used to rest on the sidecars only. The Parquet metadata flag survives copying a
    synthetic file alone; files with neither marker get a PROVENANCE UNKNOWN banner instead of silence."""
    for m in w.interim.glob("*/*.meta.json"):
        m.unlink()
    man = json.loads(w.manifest.read_text())
    for e in man["files"]:
        e["notes"] = ""
    w.manifest.write_text(json.dumps(man))
    rep = _qa(w, "--no-rebuild")
    assert rep["provenance"]["synthetic"] is True and len(rep["provenance"]["synthetic_harmonised_files"]) == 28
    assert "SYNTHETIC DATA: NOT A RESULT" in (w.phase2 / "QA_report.md").read_text()
    for f in provenance.interim_files(w.interim):
        pl.read_parquet(f).write_parquet(f)
    rep = _qa(w, "--no-rebuild")
    assert rep["provenance"]["synthetic"] is False and len(rep["provenance"]["unknown_provenance_files"]) == 28
    assert "PROVENANCE UNKNOWN: NOT A RESULT YET" in (w.phase2 / "QA_report.md").read_text()
    assert cli.main(["phase2", "report"]) == 0
    assert "PROVENANCE UNKNOWN" in (w.phase2 / "G2_report.md").read_text()
    runs = sorted((w.phase2 / "runs").glob("*_qa.json"))
    assert json.loads(runs[-1].read_text())["result"]["synthetic_inputs"] is None


def test_qa_detects_features_that_differ_from_a_label_free_rebuild(w):
    f = w.processed / "features" / "2018.parquet"
    df = pl.read_parquet(f)
    df.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(pl.col("UP1") + 0.5).otherwise(pl.col("UP1"))
                    .alias("UP1")).write_parquet(f)
    st = _status(_qa(w))
    assert st["L1"] == "fail" and st["L2"] == "pass"
    st = _status(_qa(w, "--rebuild-year", "2019"))                          # another year rebuilds identically
    assert st["L1"] == "pass"


def test_qa_detects_peer_stats_fitted_on_injected_providers(w):
    p = w.processed / "cohort.parquet"
    coh = pl.read_parquet(p)
    inj = coh.filter((pl.col("year") == 2019) & pl.col("in_cohort"))["npi"].head(25).implode()
    coh.with_columns((pl.col("npi").is_in(inj) & (pl.col("year") == 2019)).alias("injected")).write_parquet(p)
    rep = _qa(w, "--no-rebuild")
    l2 = rep["leakage_checklist"][1]
    assert l2["status"] == "fail" and "2019" in l2["checks"][0]["detail"]
    assert "25 rows flagged injected" in l2["checks"][1]["detail"]


def test_qa_reports_broken_folds_instead_of_crashing(w):
    p = w.processed / "folds.parquet"
    f = pl.read_parquet(p)
    f.vstack(f.head(3).with_columns(((pl.col("fold") + 1) % 5).alias("fold"))).write_parquet(p)
    st = _status(_qa(w, "--no-rebuild"))
    assert st["L3"] == "fail" and st["L4"] == "fail"
    assert "exactly one fold" in (w.phase2 / "QA_report.md").read_text()


def _widen_dme_list(paths: Paths) -> None:
    fy = paths.configs / "features.yaml"
    doc = yaml.safe_load(fy.read_text())
    doc["dme_high_risk"]["hcpcs_ranges"].append({"from": "E0100", "to": "E0105", "label": "canes"})
    fy.write_text(yaml.safe_dump(doc, sort_keys=False))


def test_qa_item6_freeze_against_later_outputs(w):
    """Leakage item 6: the freeze is recorded by `phase2 report` only while no Phase 3-7 output exists. A report run
    after results exist must not re-baseline it (review finding: it used to, and L6 then passed after a change)."""
    later = w.outputs / "phase3" / "metrics.json"
    later.parent.mkdir(parents=True)
    later.write_text("{}")
    (w.phase2 / qa.FREEZE_FILE).unlink()
    assert _status(_qa(w, "--no-rebuild"))["L6"] == "fail"                 # results exist, nothing recorded
    assert cli.main(["phase2", "report"]) == 0                              # too late: nothing is recorded now
    assert not (w.phase2 / qa.FREEZE_FILE).exists()
    assert _status(_qa(w, "--no-rebuild"))["L6"] == "fail"
    g2 = json.loads((w.phase2 / "G2_report.json").read_text())
    assert g2["freeze"]["status"] == "missing" and any("without a freeze record" in t for t in g2["decisions_needed"])

    later.unlink()                                                          # back before any result
    assert cli.main(["phase2", "report"]) == 0                              # records the freeze
    frozen = json.loads((w.phase2 / qa.FREEZE_FILE).read_text())
    later.write_text("{}")
    assert _status(_qa(w, "--no-rebuild"))["L6"] == "pass"
    _widen_dme_list(w)                                                      # a change after results exist ...
    assert cli.main(["phase2", "report"]) == 0                              # ... is not re-baselined by a report
    assert json.loads((w.phase2 / qa.FREEZE_FILE).read_text()) == frozen
    rep = _qa(w, "--no-rebuild")
    assert _status(rep)["L6"] == "fail" and "dme_high_risk" in rep["leakage_checklist"][5]["checks"][1]["detail"]
    g2 = json.loads((w.phase2 / "G2_report.json").read_text())
    assert g2["freeze"] == {"status": "kept", "recorded_utc": frozen["recorded_utc"], "changed": ["dme_high_risk"]}
    assert any("changed after the freeze" in t for t in g2["decisions_needed"])


def test_g2_protocol_diff_lists_decision_rows_after_the_baseline(w):
    (w.root / "design" / "DESIGN_DOCUMENT.md").write_text("\n".join([
        "# Design", "## 17. Decision log", "", "| # | Decision | Reason |", "|---|---|---|",
        "| D1 | Drop Kaggle data | Rule-generated labels |", "| D18 | One record | Lifecycle |",
        "| D19 | peer.min_cell = 100 | SM2 and MAD stability; proposed, confirm at G2 |",
        "| D20 | Folds stratified by modal specialty | Ties go to the latest year |",
        "", "## 18. Claim verification status", "| D99 | not a decision row | x |"]))
    rows = g2_report.decision_rows(w)
    assert [r["id"] for r in rows] == ["D1", "D18", "D19", "D20"]
    assert cli.main(["phase2", "report"]) == 0
    g2 = json.loads((w.phase2 / "G2_report.json").read_text())
    assert [r["id"] for r in g2["decisions_after_baseline"]] == ["D19", "D20"]
    assert any("marked proposed" in t for t in g2["decisions_needed"])
    md = (w.phase2 / "G2_report.md").read_text()
    assert "| D19 | peer.min_cell = 100 |" in md and "D18 | One record" not in md


def test_reports_on_an_empty_project_say_what_is_missing(proj):
    assert cli.main(["phase2", "qa"]) == 0
    md = (proj.phase2 / "QA_report.md").read_text()
    assert "run `scenario-fraud phase2 cohort`" in md and "run `scenario-fraud phase2 features`" in md
    assert "No harmonised input files" in md
    st = {i["id"]: i["status"] for i in json.loads((proj.phase2 / "QA_report.json").read_text())["leakage_checklist"]}
    assert st["L2"] == "not_run" and st["L4"] == "not_run" and st["L5"] == "pending"
    assert cli.main(["phase2", "report"]) == 0
    assert "No feature tables yet" in (proj.phase2 / "G2_report.md").read_text()
    (proj.phase2 / "QA_report.json").unlink()
    assert cli.main(["phase2", "report"]) == 0
    assert "QA report is missing" in (proj.phase2 / "G2_report.md").read_text()


def test_static_scan_flags_label_imports_and_artefact_names(tmp_path):
    m = tmp_path / "mod.py"
    m.write_text('"""Docstring may mention labels_ and LEIE."""\nfrom . import labels\n'
                 'from .config import load_config\nX = "labels_primary_W3.parquet"\n')
    hits = qa._forbidden_uses(m)
    assert len(hits) == 2 and "imports labels" in hits[0] and "'labels_'" in hits[1]


def test_qa_detects_code_reference_and_pm_fits_that_include_injected_providers(w):
    """Review finding: L2 used to check stats.parquet only. A code reference (SM1/SM2/CH4) or PM2/PM3 regression
    fitted with injected providers must fail L2 even when stats.parquet is right."""
    y = 2019
    p = w.processed / "cohort.parquet"
    coh = pl.read_parquet(p)
    inj = coh.filter((pl.col("year") == y) & pl.col("in_cohort"))["npi"].head(25).implode()
    flagged = coh.with_columns((pl.col("npi").is_in(inj) & (pl.col("year") == y)).alias("injected"))
    flagged = cohort.assign_peer_cells(flagged, cohort.CohortConfig.load(w))      # injected never count
    flagged.write_parquet(p)
    features.build_features(w, [y])                                                 # honest fits (fit_* skip them)
    rep = _qa(w, "--no-rebuild")
    assert _status(rep)["L2"] == "pass" and "25 rows flagged injected" in rep["leakage_checklist"][1]["checks"][1]["detail"]
    fcfg = features.FeatureConfig.load(w)
    lazy, _ = features.load_frames(w, y)
    fr = {k: v.collect() for k, v in lazy.items() if v is not None}
    leaky = flagged.filter((pl.col("year") == y) & pl.col("in_cohort")).drop("injected")
    refs = features.fit_peer_reference(fr, leaky, y, fcfg)
    ps = features.fit_peer_stats(features.raw_statistics(fr, leaky, refs, y, fcfg), leaky, y, fcfg)
    peer = w.processed / "peer" / str(y)
    refs.codes.write_parquet(peer / "codes.parquet")
    rep = _qa(w, "--no-rebuild")
    detail = rep["leakage_checklist"][1]["checks"][0]["detail"]
    assert _status(rep)["L2"] == "fail" and "codes.parquet" in detail and "pm_models" not in detail
    features.build_features(w, [y])
    ps.pm.write_parquet(peer / "pm_models.parquet")
    rep = _qa(w, "--no-rebuild")
    detail = rep["leakage_checklist"][1]["checks"][0]["detail"]
    assert _status(rep)["L2"] == "fail" and "pm_models.parquet" in detail and "codes.parquet" not in detail


def test_qa_detects_cohort_cells_counting_injected_providers(w):
    p = w.processed / "cohort.parquet"
    coh = pl.read_parquet(p)
    inj = coh.filter((pl.col("year") == 2020) & pl.col("in_cohort"))["npi"].head(3).implode()
    coh.with_columns((pl.col("npi").is_in(inj) & (pl.col("year") == 2020)).alias("injected")).write_parquet(p)
    bad, _ = qa.peer_cell_check(w)                       # n_cell_* still count the 3 injected rows
    assert bad and "n_cell_ent" in bad[0]


def test_split_checks_report_leaky_role_assignments(w, monkeypatch):
    """Review finding: the L3/L4 detectors were untested. A fold-k NPI in training, an H_k NPI in both tune sets and
    a test row scored by a second fold must each be reported, and L3/L4 must fail."""
    scfg, folds = splits.SplitConfig.load(w), splits.load_folds(w)
    frame, _ = qa._label_frame(w)
    assert qa.split_checks(frame, folds, scfg)[1] == {"roles": [], "npis": []}
    orig = splits.assign_roles

    def leaky(frame, folds, fold, design, cfg, holdout=None, *, check=True, exclude_years=()):
        r = orig(frame, folds, fold, design, cfg, holdout, check=check)
        d = design if isinstance(design, splits.Design) else cfg.design(design)
        npi, yr, k = pl.col("npi"), pl.col("year"), folds.filter(pl.col("fold") == fold)["npi"].implode()
        first, test = d.train[0], d.test[0]
        a = r.filter(npi.is_in(k) & (yr == test) & npi.is_in(r.filter(yr == first)["npi"].implode()))["npi"][0]
        tv = r.filter(pl.col("role") == "tune_valid")["npi"].implode()
        b = r.filter(npi.is_in(tv) & (yr == first))["npi"][0]
        c = r.filter(~npi.is_in(k) & (yr == test))["npi"][0]
        role = (pl.when((npi == a) & (yr == first)).then(pl.lit("train"))
                .when((npi == b) & (yr == first)).then(pl.lit("tune_train"))
                .when((npi == c) & (yr == test)).then(pl.lit("test"))
                .otherwise(pl.col("role").cast(pl.String))).cast(splits.ROLE_DTYPE)
        return r.with_columns(role.alias("role"))

    monkeypatch.setattr(splits, "assign_roles", leaky)
    _, bad = qa.split_checks(frame, folds, scfg)
    assert any("training role inside fold k" in b for b in bad["roles"])
    assert any("test outside fold k" in b for b in bad["roles"])
    assert any("both test and training" in b for b in bad["npis"])
    assert any("tune_train and tune_valid" in b for b in bad["npis"])
    assert any("test rows over all folds" in b for b in bad["npis"])
    st = {i["id"]: i["status"] for i in qa.leakage_checklist(w, [], bad, None, False, None)}
    assert st["L3"] == "fail" and st["L4"] == "fail"


def test_label_free_rebuild_is_isolated_from_the_real_project(w, monkeypatch):
    """Review finding: implicit root discovery inside the L1 rebuild must see the scratch project, not the real one,
    and any exception of the rebuild is an L1 failure rather than a QA crash."""
    seen = {}
    orig = features.build_features

    def spy(paths, years=None):
        seen["discovered"], seen["given"] = Paths.discover().root, paths.root
        return orig(paths, years)

    monkeypatch.setattr(features, "build_features", spy)
    assert qa.rebuild_check(w, 2018)["status"] == "pass"
    assert seen["discovered"] == seen["given"] != w.root and ".qa_rebuild_" in str(seen["given"])
    assert Paths.discover().root == w.root                                     # environment restored

    def boom(paths, years=None):
        raise KeyError("would have read labels_primary_W3.parquet")

    monkeypatch.setattr(features, "build_features", boom)
    r = qa.rebuild_check(w, 2018)
    assert r["status"] == "fail" and "KeyError" in r["detail"] and Paths.discover().root == w.root


def test_region_section_measures_division_shifts(proj):
    """DESIGN §19 Q2 evidence: division-cell median vs spec x ent median in robust-z units, over counted providers."""
    fcfg = features.FeatureConfig.load(proj)
    ids = [s.id for s in fcfg.robust if s.id not in features.REF_DEPENDENT]
    se = "2022|I|Cardiology"
    coh, feat = [], []

    def add(npi, level, div, v, in_cohort=True, injected=False):
        coh.append(dict(npi=npi, year=2022, in_cohort=in_cohort, injected=injected, peer_level=level,
                        cell_spec_ent=se, cell_spec_ent_div=f"{se}|{div}"))
        feat.append({"npi": npi, "DI0": 0, **{f"raw__{f}": 1.0 for f in ids}, "raw__OU1": float(v)})

    for i, v in enumerate([10, 11, 12, 13, 14]):
        add(f"P{i}", "spec_ent_div", "Pacific", v)
    for i, v in enumerate([0, 1, 2, 3, 4]):
        add(f"N{i}", "spec_ent", "New England", v)
    add("X0", "spec_ent_div", "Pacific", 1000.0, injected=True)              # never counted
    add("X1", "spec_ent", "New England", -1000.0, in_cohort=False)
    proj.processed.joinpath("features").mkdir(parents=True)
    pl.DataFrame(coh).with_columns(pl.col("year").cast(pl.Int32)).write_parquet(proj.processed / "cohort.parquet")
    pl.DataFrame(feat).with_columns(pl.col("DI0").cast(pl.Int8)).write_parquet(proj.processed / "features" / "2022.parquet")
    rep = qa.region_section(proj, [2022], 0.5)
    assert [r["feature"] for r in rep] == ["OU1"]                             # constant features have no scale
    r = rep[0]
    # spec x ent: median 7, MAD 5 -> scale 7.413; Pacific: median 12, MAD 1 -> shift 5 / 7.413, ratio 1 / 5
    assert r["refined_providers"] == 5 and r["median_abs_shift"] == pytest.approx(5 / (1.4826 * 5))
    assert r["median_scale_ratio"] == pytest.approx(0.2) and r["share_shift_gt_0.5"] == 1.0
    assert r["refined_share_of_providers"] == 0.5
