"""YAML config loading (C16). Configs are the single source for parameters; code holds no magic numbers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from .paths import Paths


def load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_config(name: str, paths: Paths | None = None) -> dict:
    paths = paths or Paths.discover()
    return load_yaml(paths.configs / f"{name}.yaml")
