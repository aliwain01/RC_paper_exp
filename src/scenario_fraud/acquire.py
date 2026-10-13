"""C1 Acquisition & manifest (DESIGN §8, HANDOFF §10).

Every raw file lives under data/raw/{source}/{year}/ and has one manifest entry recording where it came
from (source page, file URL or "manual"), the route, the download date, size and SHA-256. Files can be
fetched (data.gov distribution links, OIG, Internet Archive) or downloaded by hand and registered.
Nothing here guesses a file URL: URLs come from the landing page, the catalogue listing or the CDX index.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import requests

from .config import file_sha256, load_config
from .paths import Paths

ROUTES = {"manual", "datagov", "cms_api", "oig", "wayback", "other"}
DATA_KINDS = {"data", "current", "supplement_excl", "supplement_rein", "snapshot"}  # tabular files
USER_AGENT = "scenario-fraud-research/0.1 (academic research; public-data download)"


class ManifestError(RuntimeError):
    pass


@dataclass
class Entry:
    path: str                 # relative to project root, POSIX
    source: str               # key in configs/sources.yaml
    year: int                 # data year (CMS) or file year (LEIE)
    kind: str                 # one of the source's kinds
    route: str                # manual | datagov | cms_api | oig | wayback | other
    source_page: str          # landing / catalogue page the file was taken from
    file_url: str             # direct URL, or "manual"
    download_date: str        # ISO date
    size_bytes: int
    sha256: str
    as_of: str | None = None  # LEIE: snapshot / supplement / download date
    notes: str = ""
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------- manifest

def load_manifest(paths: Paths) -> list[Entry]:
    if not paths.manifest.is_file():
        return []
    rows = json.loads(paths.manifest.read_text(encoding="utf-8"))
    return [Entry(**r) for r in rows["files"]]


def save_manifest(paths: Paths, entries: Iterable[Entry]) -> None:
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(entries, key=lambda e: (e.source, e.year, e.kind, e.path))
    doc = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "files": [asdict(e) for e in entries],
    }
    tmp = paths.manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(paths.manifest)


def _upsert(paths: Paths, new: Entry) -> Entry:
    entries = [e for e in load_manifest(paths) if e.path != new.path]
    entries.append(new)
    save_manifest(paths, entries)
    return new


def _validate(paths: Paths, source: str, year: int, kind: str, route: str) -> None:
    sources = load_config("sources", paths)["sources"]
    if source not in sources:
        raise ManifestError(f"unknown source {source!r}; known: {sorted(sources)}")
    if kind not in sources[source]["kinds"]:
        raise ManifestError(f"kind {kind!r} not valid for {source}; valid: {sources[source]['kinds']}")
    if route not in ROUTES:
        raise ManifestError(f"route {route!r} not in {sorted(ROUTES)}")
    if not (1990 <= int(year) <= 2100):
        raise ManifestError(f"implausible year {year}")


def dest_dir(paths: Paths, source: str, year: int) -> Path:
    return paths.raw / source / str(int(year))


# ----------------------------------------------------------------------------------------------- register

def register(
    paths: Paths,
    file: Path,
    source: str,
    year: int,
    kind: str,
    *,
    route: str = "manual",
    source_page: str = "",
    file_url: str = "manual",
    as_of: str | None = None,
    move: bool = False,
    notes: str = "",
    download_date: str | None = None,
) -> Entry:
    """Register a file already on disk (e.g. downloaded by hand in a browser). Copies (or moves) it into
    data/raw/{source}/{year}/ unless it is already there, then checksums it and writes the manifest."""
    _validate(paths, source, year, kind, route)
    file = Path(file).resolve()
    if not file.is_file():
        raise ManifestError(f"not a file: {file}")
    target_dir = dest_dir(paths, source, year)
    target = target_dir / file.name
    if file != target.resolve():
        target_dir.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_sha256(target) != file_sha256(file):
                raise ManifestError(f"{paths.rel(target)} already exists with different content")
        elif move:
            shutil.move(str(file), str(target))
        else:
            shutil.copy2(file, target)
    if not source_page:
        source_page = load_config("sources", paths)["sources"][source].get("landing_page", "")
    entry = Entry(
        path=paths.rel(target),
        source=source,
        year=int(year),
        kind=kind,
        route=route,
        source_page=source_page,
        file_url=file_url,
        download_date=download_date or date.today().isoformat(),
        size_bytes=target.stat().st_size,
        sha256=file_sha256(target),
        as_of=as_of,
        notes=notes,
    )
    return _upsert(paths, entry)


# ----------------------------------------------------------------------------------------------- fetch

def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s  # honours HTTPS_PROXY / REQUESTS_CA_BUNDLE from the environment


def fetch(
    paths: Paths,
    url: str,
    source: str,
    year: int,
    kind: str,
    *,
    route: str,
    source_page: str = "",
    filename: str | None = None,
    as_of: str | None = None,
    notes: str = "",
    timeout: float = 60.0,
    session: requests.Session | None = None,
) -> Entry:
    """Stream a URL to data/raw/{source}/{year}/<filename>, checksum it, and record it in the manifest."""
    _validate(paths, source, year, kind, route)
    session = session or _session()
    name = filename or Path(requests.utils.urlparse(url).path).name or "download.bin"
    target_dir = dest_dir(paths, source, year)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name
    part = target.with_name(target.name + ".part")
    try:
        with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
            if r.status_code != 200:
                raise ManifestError(f"HTTP {r.status_code} for {url}")
            with open(part, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            final_url = r.url
    except BaseException:
        part.unlink(missing_ok=True)  # never leave a truncated file that looks like data
        raise
    if target.exists() and file_sha256(target) != file_sha256(part):
        part.unlink()
        raise ManifestError(f"{paths.rel(target)} already exists with different content; not overwritten")
    part.replace(target)
    entry = register(
        paths, target, source, year, kind,
        route=route, source_page=source_page, file_url=url, as_of=as_of, notes=notes,
    )
    if final_url != url:
        entry.extra["final_url"] = final_url
        _upsert(paths, entry)
    return entry


# ----------------------------------------------------------------------------------------------- route discovery

def datagov_distributions(paths: Paths, slug: str, session: requests.Session | None = None) -> list[dict]:
    """Route A: list a data.gov catalogue entry's distributions (per-year download links). Saved for review;
    the human or the runbook picks URLs from this listing, nothing is downloaded here."""
    cfg = load_config("sources", paths)["datagov"]
    url = cfg["package_show"].format(slug=slug)
    r = (session or _session()).get(url, timeout=60)
    r.raise_for_status()
    body = r.json()
    if not body.get("success"):
        raise ManifestError(f"data.gov package_show failed for {slug}: {body.get('error')}")
    keep = ("name", "description", "url", "format", "created", "last_modified", "size")
    resources = [{k: res.get(k) for k in keep} for res in body["result"].get("resources", [])]
    out = paths.phase1 / "routes" / f"datagov_{slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"slug": slug, "queried": url, "resources": resources}, indent=2), encoding="utf-8")
    return resources


def wayback_snapshots(paths: Paths, session: requests.Session | None = None) -> list[dict]:
    """List Internet Archive captures of the LEIE full file (CDX API), de-duplicated by content digest."""
    cfg = load_config("sources", paths)["wayback"]
    params = {
        "url": cfg["target_url"],
        "output": "json",
        "fl": "timestamp,original,statuscode,digest,length",
        "filter": "statuscode:200",
        "collapse": "digest",
    }
    r = (session or _session()).get(cfg["cdx_api"], params=params, timeout=120)
    r.raise_for_status()
    rows = r.json()
    snaps = [dict(zip(rows[0], row)) for row in rows[1:]] if rows else []
    out = paths.phase1 / "routes" / "wayback_cdx.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"query": params, "snapshots": snaps}, indent=2), encoding="utf-8")
    return snaps


def wayback_closest(paths: Paths, timestamp: str, session: requests.Session | None = None) -> dict | None:
    """Availability API: the capture closest to YYYYMMDD (checked during design)."""
    cfg = load_config("sources", paths)["wayback"]
    r = (session or _session()).get(
        cfg["available_api"], params={"url": cfg["target_url"], "timestamp": timestamp}, timeout=60
    )
    r.raise_for_status()
    return r.json().get("archived_snapshots", {}).get("closest")


def fetch_wayback_snapshot(paths: Paths, timestamp: str, original: str | None = None, **kw) -> Entry:
    """Fetch the raw bytes (`id_`) of one LEIE capture and register it as leie/<year>/UPDATED_<ts>.csv."""
    cfg = load_config("sources", paths)["wayback"]
    original = original or f"https://{cfg['target_url']}"
    url = cfg["raw_prefix"].format(timestamp=timestamp, url=original)
    as_of = f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
    return fetch(
        paths, url, "leie", int(timestamp[:4]), "snapshot",
        route="wayback", source_page="https://web.archive.org/", filename=f"UPDATED_{timestamp}.csv",
        as_of=as_of, **kw,
    )


# ----------------------------------------------------------------------------------------------- verification

def verify(paths: Paths) -> list[dict]:
    """Recompute every checksum. Returns one row per problem (missing file, size or hash mismatch)."""
    problems = []
    for e in load_manifest(paths):
        p = paths.abs(e.path)
        if not p.is_file():
            problems.append({"path": e.path, "problem": "missing"})
            continue
        size = p.stat().st_size
        if size != e.size_bytes:
            problems.append({"path": e.path, "problem": f"size {size} != manifest {e.size_bytes}"})
            continue
        if file_sha256(p) != e.sha256:
            problems.append({"path": e.path, "problem": "sha256 mismatch"})
    return problems


# ----------------------------------------------------------------------------------------------- header dumps

def _sniff_encoding(path: Path) -> str:
    with open(path, "rb") as fh:
        head = fh.read(1 << 20)
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            head.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _count_rows_python(path: Path, encoding: str) -> int:
    with open(path, "r", encoding=encoding, newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return sum(1 for _ in reader)


def _count_rows(path: Path, encoding: str) -> tuple[int, str]:
    """Data rows (excluding header). DuckDB is fast on multi-GB files; fall back to the csv module."""
    if encoding in ("utf-8", "utf-8-sig"):
        try:
            import duckdb

            con = duckdb.connect()
            n = con.execute(
                "SELECT count(*) FROM read_csv(?, header=true, all_varchar=true, strict_mode=true)", [str(path)]
            ).fetchone()[0]
            return int(n), "duckdb"
        except Exception:  # noqa: BLE001 - any parser complaint -> slower but tolerant path
            pass
    return _count_rows_python(path, encoding), "python-csv"


def header_dump(paths: Paths, entry: Entry, count_rows: bool = True) -> dict:
    """Column names + row count for one tabular file -> outputs/phase1/headers/<source>_<year>_<stem>.json.
    These dumps are the inputs to the C2 column maps and to the G1 report."""
    p = paths.abs(entry.path)
    if p.suffix.lower() != ".csv":
        raise ManifestError(f"header dump supports .csv only (got {entry.path}); unpack archives first")
    enc = _sniff_encoding(p)
    with open(p, "r", encoding=enc, newline="") as fh:
        columns = next(csv.reader(fh))
    columns = [c.strip() for c in columns]
    rows, counter = _count_rows(p, enc) if count_rows else (None, None)
    dump = {
        "path": entry.path,
        "source": entry.source,
        "year": entry.year,
        "kind": entry.kind,
        "as_of": entry.as_of,
        "sha256": entry.sha256,
        "size_bytes": entry.size_bytes,
        "encoding": enc,
        "n_columns": len(columns),
        "columns": columns,
        "n_rows": rows,
        "row_counter": counter,
    }
    out = header_path(paths, entry)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dump, indent=2), encoding="utf-8")
    return dump


def header_path(paths: Paths, entry: Entry) -> Path:
    return paths.headers / f"{entry.source}_{entry.year}_{Path(entry.path).stem}.json"


def dump_all_headers(paths: Paths, refresh: bool = False) -> list[dict]:
    dumps = []
    for e in load_manifest(paths):
        if e.kind not in DATA_KINDS or not e.path.lower().endswith(".csv"):
            continue
        hp = header_path(paths, e)
        if hp.is_file() and not refresh:
            d = json.loads(hp.read_text(encoding="utf-8"))
            if d.get("sha256") == e.sha256:
                dumps.append(d)
                continue
        dumps.append(header_dump(paths, e))
    return dumps
