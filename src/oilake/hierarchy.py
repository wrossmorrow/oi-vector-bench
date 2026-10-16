"""Flatten the boxable-class hierarchy JSON into a transitive closure.

The JSON is a tree of {"LabelName": mid, "Subcategory": [...], "Part": [...]}.
Only "Subcategory" is an is-a relation; "Part" (e.g. Wheel is part of Car) is
not, so it is ignored. A class can appear under several parents, so this is a
DAG; we keep the shortest depth per (ancestor, descendant) pair.
"""

from __future__ import annotations

from typing import Iterator

ROOT_ENTITY = "/m/0bl9f"  # synthetic "Entity" root; not a real label


def _walk(node: dict, path: tuple[str, ...]) -> Iterator[tuple[str, str, int]]:
    mid = node["LabelName"]
    path = path + (mid,)
    for depth, ancestor in enumerate(reversed(path)):
        yield ancestor, mid, depth
    for child in node.get("Subcategory", []) or []:
        yield from _walk(child, path)


def closure(tree: dict, drop_root: bool = True) -> list[tuple[str, str, int]]:
    """Return sorted (ancestor, descendant, depth) rows, depth 0 = self."""
    best: dict[tuple[str, str], int] = {}
    for anc, desc, depth in _walk(tree, ()):
        if drop_root and ROOT_ENTITY in (anc, desc):
            continue
        key = (anc, desc)
        if key not in best or depth < best[key]:
            best[key] = depth
    return sorted((a, d, dep) for (a, d), dep in best.items())
