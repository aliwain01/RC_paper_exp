"""Scheme groups (DESIGN §7.5, Fig. 4; CONTRACTS §2.2): configs/groups.yaml -> model column -> group.

Eight groups in a fixed order: context first (an input, never a reason), then the seven scheme groups.
Feature IDs are matched exactly; one-hot context columns by prefix (`CX1__cardiology` -> context).
`feature_groups` maps every column it is given to exactly one group, or fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import load_config
from .paths import Paths

GROUP_ORDER = ("context", "upcoding", "overutilization", "atypical_mix", "charge_inflation", "patient_mix",
               "dme_intensity", "dme_concentration")
CONTEXT = GROUP_ORDER[0]
SCHEME_GROUPS = GROUP_ORDER[1:]


class GroupsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Group:
    id: str
    name: str
    features: tuple[str, ...]
    prefixes: tuple[str, ...] = ()


def load_groups(paths: Paths | None = None, variant: str | None = None) -> tuple[Group, ...]:
    """Groups in contract order G0..G7, optionally with a variant from groups.yaml `variants` applied
    (feature ID -> new group, e.g. S7 moves CH2 to overutilization and PM1 to context)."""
    doc = load_config("groups", paths)
    try:
        groups = [Group(id=str(g["id"]), name=str(g["name"]), features=tuple(g.get("features") or ()),
                        prefixes=tuple(g.get("prefixes") or ())) for g in doc["groups"]]
    except (KeyError, TypeError) as exc:
        raise GroupsError(f"configs/groups.yaml malformed: {exc!r}") from exc
    got = tuple(g.name for g in groups)
    if got != GROUP_ORDER or tuple(g.id for g in groups) != tuple(f"G{i}" for i in range(len(GROUP_ORDER))):
        raise GroupsError(f"groups.yaml must list G0..G7 = {GROUP_ORDER} in order; got {got}")
    seen: dict[str, str] = {}
    for g in groups:
        for f in g.features:
            if f in seen:
                raise GroupsError(f"feature {f} listed in groups {seen[f]!r} and {g.name!r}")
            seen[f] = g.name
    prefixes = [p for g in groups for p in g.prefixes]
    if len(prefixes) != len(set(prefixes)) or any(f.startswith(p) for f in seen for p in prefixes):
        raise GroupsError("groups.yaml prefixes must be unique and must not match a listed feature ID")
    if variant is None:
        return tuple(groups)
    moves = (doc.get("variants") or {}).get(variant)
    if not moves:
        raise GroupsError(f"unknown grouping variant {variant!r}; known: {sorted(doc.get('variants') or {})}")
    for f, target in moves.items():
        if f not in seen or target not in GROUP_ORDER:
            raise GroupsError(f"variant {variant}: cannot move {f!r} to {target!r}")
    return tuple(Group(g.id, g.name,
                       tuple(f for f in g.features if f not in moves)
                       + tuple(f for f, t in moves.items() if t == g.name),
                       g.prefixes) for g in groups)


def feature_groups(columns: Iterable[str], variant: str | None = None,
                   paths: Paths | None = None) -> dict[str, str]:
    """{model column: group name}. Fails on a column that matches no group or more than one."""
    groups = load_groups(paths, variant)
    out: dict[str, str] = {}
    bad: list[str] = []
    for col in columns:
        hits = [g.name for g in groups if col in g.features or any(col.startswith(p) for p in g.prefixes)]
        if len(hits) != 1:
            bad.append(f"{col} -> {hits or 'no group'}")
        else:
            out[col] = hits[0]
    if bad:
        raise GroupsError("every model column must map to exactly one group: " + "; ".join(bad))
    return out
