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
    copy_configs(root)
    monkeypatch.setenv("SCENARIO_FRAUD_ROOT", str(root))
    return Paths(root)


def copy_configs(root: Path) -> None:
    """The repo's configs without the column maps written for the real CMS files (tests draft their own)."""
    shutil.copytree(REPO / "configs", root / "configs",
                    ignore=lambda d, names: [n for n in names if Path(d).name == "column_maps" and not n.startswith("_")])


def write_csv(path: Path, header: list[str], rows: list[list], encoding: str = "utf-8") -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding, newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path
