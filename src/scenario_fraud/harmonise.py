"""C2 Schema harmoniser (DESIGN §7.1, HANDOFF §7 "CMS column names").

raw CSV + configs/column_maps/{dataset}_{year}.yaml -> data/interim/{dataset}/{year}.parquet

Column maps are written from the *actual* CSV header of each downloaded year (header dumps from C1),
with the CMS data dictionary as reference. `draft_column_map` pre-fills a draft from known names; a
human-reviewed map sets `reviewed: true`. Everything that does not line up fails loudly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .acquire import Entry, header_path, load_manifest
from .config import file_sha256, load_yaml
from .paths import Paths

TODO = "TODO"
DTYPES = {"str": pl.String, "int": pl.Int64, "float": pl.Float64}
NPI_RE = r"^\d{10}$"


class SchemaError(RuntimeError):
    pass


@dataclass
class ColumnMap:
    dataset: str
    year: int
    columns: dict[str, str | None]   # raw -> harmonised (None = drop)
    null_values: list[str]
    reviewed: bool
    sha256: str | None               # sha of the file the map was written from
    path: Path

    @classmethod
    def load(cls, path: Path) -> "ColumnMap":
        doc = load_yaml(path)
        sha = (doc.get("written_from") or {}).get("sha256")
        if sha is not None and not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
            raise SchemaError(f"{path.name}: written_from.sha256 must be a quoted 64-hex string, got {sha!r}")
        return cls(
            dataset=doc["dataset"],
            year=int(doc["year"]),
            columns=dict(doc["columns"]),
            null_values=[str(v) for v in doc.get("null_values", [""])],
            reviewed=doc.get("reviewed") is True,
            sha256=sha,
            path=path,
        )


def map_path(paths: Paths, dataset: str, year: int) -> Path:
    return paths.column_maps / f"{dataset}_{year}.yaml"


def load_schema(paths: Paths) -> dict:
    return load_yaml(paths.column_maps / "_schema.yaml")


def _target_spec(schema_ds: dict, name: str) -> dict | None:
    if name in schema_ds["columns"]:
        return schema_ds["columns"][name]
    for prefix, spec in (schema_ds.get("prefix") or {}).items():
        if name.startswith(prefix) and len(name) > len(prefix):
            return spec
    return None


def validate_map(cmap: ColumnMap, raw_columns: list[str], schema: dict, allow_draft: bool = False) -> dict[str, str]:
    """Check a column map against a file header and the harmonised schema. Returns {raw: harmonised} kept."""
    if cmap.dataset not in schema:
        raise SchemaError(f"{cmap.path.name}: dataset {cmap.dataset!r} not in _schema.yaml")
    ds = schema[cmap.dataset]
    errors: list[str] = []
    if not cmap.reviewed and not allow_draft:
        errors.append("map is a draft (reviewed: false); check every line against the header and data dictionary")
    raw_set = set(raw_columns)
    if len(raw_set) != len(raw_columns):
        dups = sorted({c for c in raw_columns if raw_columns.count(c) > 1})
        errors.append(f"duplicate raw columns in file: {dups}")
    unmapped = [c for c in raw_columns if c not in cmap.columns]
    if unmapped:
        errors.append(f"unmapped raw columns (add them, or map to null to drop): {unmapped}")
    stale = [c for c in cmap.columns if c not in raw_set]
    if stale:
        errors.append(f"map entries for columns not in the file: {stale}")
    todo = [c for c, h in cmap.columns.items() if h == TODO]
    if todo:
        errors.append(f"unresolved TODO entries: {todo}")
    kept = {r: h for r, h in cmap.columns.items() if h not in (None, TODO) and r in raw_set}
    unknown = [h for h in kept.values() if _target_spec(ds, h) is None]
    if unknown:
        errors.append(f"harmonised names not in _schema.yaml[{cmap.dataset}]: {unknown}")
    targets = list(kept.values())
    dup_t = sorted({t for t in targets if targets.count(t) > 1})
    if dup_t:
        errors.append(f"several raw columns map to the same harmonised name: {dup_t}")
    missing = [n for n, spec in ds["columns"].items() if spec.get("required") and n not in targets]
    if missing:
        errors.append(f"required harmonised columns not produced: {missing}")
    if errors:
        raise SchemaError(f"{cmap.path.name}:\n  - " + "\n  - ".join(errors))
    return kept


def _data_entry(paths: Paths, dataset: str, year: int, sha256: str | None) -> Entry:
    cands = [e for e in load_manifest(paths) if e.source == dataset and e.year == year and e.kind == "data"]
    if sha256:
        match = [e for e in cands if e.sha256 == sha256]
        if not match:
            raise SchemaError(
                f"no registered {dataset} {year} data file with sha256 {sha256[:12]}... (the map was written for "
                f"a different file; registered: {[e.path for e in cands]})"
            )
        return match[0]
    if len(cands) != 1:
        raise SchemaError(f"expected exactly one registered {dataset} {year} data file, found {len(cands)}")
    return cands[0]


def harmonise(paths: Paths, dataset: str, year: int, allow_draft: bool = False) -> Path:
    schema = load_schema(paths)
    cmap = ColumnMap.load(map_path(paths, dataset, year))
    if (cmap.dataset, cmap.year) != (dataset, year):
        raise SchemaError(f"{cmap.path.name} declares {cmap.dataset} {cmap.year}")
    entry = _data_entry(paths, dataset, year, cmap.sha256)
    src = paths.abs(entry.path)
    hp = header_path(paths, entry)
    encoding = json.loads(hp.read_text(encoding="utf-8"))["encoding"] if hp.is_file() else "utf-8"
    pl_encoding = "utf8" if encoding.startswith("utf-8") else "utf8-lossy"

    lf = pl.scan_csv(src, infer_schema=False, encoding=pl_encoding, null_values=cmap.null_values)
    raw_columns = [c.strip() for c in lf.collect_schema().names()]
    lf = lf.rename({old: new for old, new in zip(lf.collect_schema().names(), raw_columns) if old != new})
    kept = validate_map(cmap, raw_columns, schema, allow_draft=allow_draft)

    ds = schema[dataset]
    exprs = []
    for raw, name in kept.items():
        dtype = DTYPES[_target_spec(ds, name)["dtype"]]
        e = pl.col(raw).str.strip_chars()
        e = e if dtype == pl.String else e.cast(dtype, strict=True)
        exprs.append(e.alias(name))
    lf = lf.select(exprs).with_columns(pl.lit(year, dtype=pl.Int32).alias("year"))

    out = paths.interim / dataset / f"{year}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    try:
        lf.sink_parquet(tmp)
    except pl.exceptions.InvalidOperationError as exc:
        tmp.unlink(missing_ok=True)
        raise SchemaError(f"{dataset} {year}: value does not cast to the schema dtype: {exc}") from exc

    df = pl.scan_parquet(tmp)
    stats = df.select(
        pl.len().alias("n_rows"),
        (~pl.col("npi").str.contains(NPI_RE)).sum().alias("bad_npi"),
        *[pl.col(c).null_count().alias(f"null__{c}") for c in kept.values()],
    ).collect().row(0, named=True)
    if stats["bad_npi"]:
        tmp.unlink(missing_ok=True)
        raise SchemaError(f"{dataset} {year}: {stats['bad_npi']} NPI values are not 10 digits")
    tmp.replace(out)
    meta = {
        "dataset": dataset,
        "year": year,
        "source_path": entry.path,
        "source_sha256": entry.sha256,
        "column_map": paths.rel(cmap.path),
        "column_map_sha256": file_sha256(cmap.path),
        "map_reviewed": cmap.reviewed,
        "encoding": encoding,
        "n_rows": stats["n_rows"],
        "null_counts": {k[6:]: v for k, v in stats.items() if k.startswith("null__")},
        "dropped_raw_columns": [r for r, h in cmap.columns.items() if h is None],
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------------------------- draft maps

def draft_column_map(paths: Paths, dataset: str, year: int, force: bool = False) -> Path:
    """Write configs/column_maps/{dataset}_{year}.yaml from the real header dump, pre-filling known names
    from _templates.yaml. Anything not recognised is TODO; the map stays `reviewed: false` until checked."""
    out = map_path(paths, dataset, year)
    if out.exists() and not force:
        raise SchemaError(f"{paths.rel(out)} exists; pass force to overwrite")
    entries = [e for e in load_manifest(paths) if e.source == dataset and e.year == year and e.kind == "data"]
    if len(entries) != 1:
        raise SchemaError(f"expected exactly one registered {dataset} {year} data file, found {len(entries)}")
    entry = entries[0]
    hp = header_path(paths, entry)
    if not hp.is_file():
        raise SchemaError(f"no header dump for {entry.path}; run `phase1 headers` first")
    dump = json.loads(hp.read_text(encoding="utf-8"))
    tmpl = load_yaml(paths.column_maps / "_templates.yaml").get(dataset, {})
    exact = tmpl.get("map", {})
    lower = {k.lower(): v for k, v in exact.items()}
    prefixes = tmpl.get("prefix", {}) or {}
    drops = {d.lower() for d in tmpl.get("drop", [])}

    lines = []
    for col in dump["columns"]:
        if col in exact:
            target, how = exact[col], "template"
        elif col.lower() in lower:
            target, how = lower[col.lower()], "template (case differs: check)"
        elif any(col.startswith(p) for p in prefixes):
            p = next(p for p in prefixes if col.startswith(p))
            target = prefixes[p] + re.sub(r"[^0-9a-z]+", "_", col[len(p):].lower()).strip("_")
            how = "template prefix (check name and meaning)"
        elif col.lower() in drops:
            target, how = None, "template drop (identifying / unused)"
        else:
            target, how = TODO, "not in template: map or drop by hand"
        value = "null" if target is None else target
        lines.append(f"  {json.dumps(col)}: {value}    # {how}")

    text = "\n".join([
        f"# Column map for {dataset} {year}. Draft generated from the real header dump; review every line",
        "# against the file header and that year's CMS data dictionary, then set reviewed: true.",
        f"dataset: {dataset}",
        f"year: {year}",
        "reviewed: false",
        "written_from:",
        f"  path: {json.dumps(entry.path)}",
        f"  sha256: {json.dumps(entry.sha256)}",
        f"  header_dump: {json.dumps(paths.rel(hp))}",
        f"  n_columns: {dump['n_columns']}",
        'null_values: [""]',
        "columns:",
        *lines,
        "",
    ])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out
