"""Input provenance (CONTRACTS §0; HANDOFF §8 rule 7): are the harmonised inputs synthetic test data?

tests/synth.py marks every file it writes twice: a `{year}.meta.json` sidecar with `synthetic: true`, and the Parquet
key-value metadata SYNTHETIC_KEY = "true", which survives copying the Parquet file without its sidecar. Phase 1
harmonise writes a sidecar without the key (harmonised CMS data). A Parquet file with neither has unknown provenance:
reports then warn instead of staying silent, because its numbers could come from a copied synthetic file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

SYNTHETIC_KEY = "scenario_fraud.synthetic"
SYNTHETIC, HARMONISED, UNKNOWN = "synthetic", "harmonised", "unknown"


def file_provenance(path: Path) -> str:
    """SYNTHETIC if the Parquet metadata or the sidecar says so, HARMONISED if a sidecar exists without the flag,
    else UNKNOWN."""
    path = Path(path)
    try:
        kv = pq.read_metadata(path).metadata or {}
    except (OSError, ValueError):
        kv = {}
    if kv.get(SYNTHETIC_KEY.encode()) == b"true":
        return SYNTHETIC
    meta = path.with_suffix(".meta.json")
    if not meta.is_file():
        return UNKNOWN
    try:
        doc = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return UNKNOWN
    return SYNTHETIC if doc.get("synthetic") is True else HARMONISED


def interim_files(interim: Path) -> list[Path]:
    """Every harmonised `{dataset}/{year}.parquet` under data/interim."""
    return sorted(interim.glob("*/[0-9][0-9][0-9][0-9].parquet")) if interim.is_dir() else []


def inputs_synthetic(files: Iterable[Path]) -> bool | None:
    """True if any existing file is synthetic; None if none is but some has unknown provenance; else False.
    Recorded as `synthetic_inputs` in the Phase 2 meta files and run logs."""
    kinds = {file_provenance(f) for f in files if Path(f).is_file()}
    if SYNTHETIC in kinds:
        return True
    return None if UNKNOWN in kinds else False
