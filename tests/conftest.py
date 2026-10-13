"""Shared fixtures: a throw-away project root with the real configs, so tests never touch data/."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scenario_fraud.paths import Paths

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def proj(tmp_path, monkeypatch) -> Paths:
    root = tmp_path / "proj with space"
    (root / "design").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='t'\n", encoding="utf-8")
    shutil.copytree(REPO / "configs", root / "configs")
    monkeypatch.setenv("SCENARIO_FRAUD_ROOT", str(root))
    return Paths(root)


def write_csv(path: Path, header: list[str], rows: list[list], encoding: str = "utf-8") -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding, newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path
