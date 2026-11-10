"""Run log (C16): every CLI invocation records what ran, on which code and configs, and when."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .config import file_sha256
from .paths import Paths

_PACKAGES = ["numpy", "pandas", "pyarrow", "polars", "duckdb", "pyyaml", "requests"]


def _git(paths: Paths, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=paths.root, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in _PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


@contextmanager
def run_log(paths: Paths, phase: str, command: str, args: dict):
    """Write outputs/{phase}/runs/<utc-timestamp>_<command>.json on exit (also on failure)."""
    started = datetime.now(timezone.utc)
    record: dict = {
        "phase": phase,
        "command": command,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in args.items()},
        "started_utc": started.isoformat(),
        "git_commit": _git(paths, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(paths, "status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _versions(),
        "config_sha256": {
            p.relative_to(paths.root).as_posix(): file_sha256(p) for p in sorted(paths.configs.rglob("*.yaml"))
        },
        "result": {},
    }
    status = "ok"
    try:
        yield record["result"]
    except BaseException as exc:  # record the failure, then re-raise
        status = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        record["status"] = status
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        out_dir = paths.outputs / phase / "runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{started.strftime('%Y%m%dT%H%M%S%fZ')}_{command}.json"
        (out_dir / name).write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
