"""Small pure helpers: sampling keys, rotation parsing, work-unit planning.

No third-party imports, so these are testable anywhere.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

_SAMPLE_SALT = b"oilake-sample-v1"


def sample_key(image_id: str) -> int:
    """Stable pseudo-random non-negative 63-bit int per image.

    63 bits, not 64: Delta has no unsigned types, so this must fit a signed
    int64 (Delta `long`).

    Sorting by this key gives one fixed random order of the dataset, so
    "first N by sample_key" is a reproducible sample, and samples are nested:
    the 1M sample is a subset of the 1.9M set, which is a subset of anything
    larger built the same way. Image IDs are already hex, but they are not
    documented as uniform, so hash them instead of trusting that.
    """
    digest = hashlib.blake2b(image_id.encode("ascii"), digest_size=8, key=_SAMPLE_SALT).digest()
    return int.from_bytes(digest, "big") >> 1


def parse_rotation(value: str | None) -> int | None:
    """Open Images `Rotation`: degrees CCW to the uploader's intended orientation.

    Values seen: "0", "90.0", "180", "270", "nan", "". Missing -> None.
    """
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() == "nan":
        return None
    deg = int(round(float(value))) % 360
    if deg not in (0, 90, 180, 270):
        raise ValueError(f"unexpected rotation {value!r}")
    return deg


def chunk_units(n_items: int, unit_size: int) -> list[tuple[int, int]]:
    """Split [0, n_items) into contiguous [start, stop) units."""
    if unit_size <= 0:
        raise ValueError("unit_size must be positive")
    return [(s, min(s + unit_size, n_items)) for s in range(0, n_items, unit_size)]


def unit_name(index: int) -> str:
    return f"unit-{index:05d}"


def parse_worker(spec: str) -> tuple[int, int]:
    """'k/N' -> (k, N): this worker takes units whose index % N == k."""
    k, n = (int(x) for x in spec.split("/"))
    if not (n > 0 and 0 <= k < n):
        raise ValueError(f"bad worker spec {spec!r}")
    return k, n


def units_for_worker(unit_indices: Iterable[int], worker: tuple[int, int],
                     done: Sequence[str] = ()) -> list[int]:
    """Units this worker should still run (skips ones already written)."""
    k, n = worker
    done_set = set(done)
    return [i for i in unit_indices if i % n == k and unit_name(i) not in done_set]
