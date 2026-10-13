"""Project paths (C16). The project path may contain spaces: always use pathlib, never shell strings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def find_root(start: Path | None = None) -> Path:
    """Project root = $SCENARIO_FRAUD_ROOT, else the nearest ancestor holding pyproject.toml and design/."""
    env = os.environ.get("SCENARIO_FRAUD_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").is_file() and (p / "design").is_dir():
            return p
    # Fall back to the checkout this package was installed from (editable install).
    pkg_root = Path(__file__).resolve().parents[2]
    if (pkg_root / "pyproject.toml").is_file():
        return pkg_root
    raise FileNotFoundError("Cannot locate project root (pyproject.toml + design/); set SCENARIO_FRAUD_ROOT")


@dataclass(frozen=True)
class Paths:
    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "Paths":
        return cls(find_root(start))

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def column_maps(self) -> Path:
        return self.configs / "column_maps"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def manifest(self) -> Path:
        return self.raw / "manifest.json"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def phase1(self) -> Path:
        return self.outputs / "phase1"

    @property
    def phase2(self) -> Path:
        return self.outputs / "phase2"

    @property
    def phase3(self) -> Path:
        return self.outputs / "phase3"

    @property
    def phase4(self) -> Path:
        return self.outputs / "phase4"

    @property
    def phase5(self) -> Path:
        return self.outputs / "phase5"

    @property
    def headers(self) -> Path:
        return self.phase1 / "headers"

    def rel(self, p: Path) -> str:
        """Path relative to the root, POSIX-style, for manifests and reports."""
        return Path(p).resolve().relative_to(self.root).as_posix()

    def abs(self, rel: str) -> Path:
        return self.root / Path(rel)
