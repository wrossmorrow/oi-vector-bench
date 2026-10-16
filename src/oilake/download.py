"""Fetch upstream metadata files into a local cache (idempotent)."""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from urllib.parse import urlparse


def cached(url: str, cache_dir: str) -> str:
    """Download `url` into `cache_dir` once; return the local path.

    Writes to a .part file and renames, so an interrupted download is never
    mistaken for a complete one.
    """
    os.makedirs(cache_dir, exist_ok=True)
    name = os.path.basename(urlparse(url).path)
    dest = os.path.join(cache_dir, name)
    if os.path.exists(dest):
        return dest
    tmp = dest + ".part"
    print(f"downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=8 << 20)
    os.replace(tmp, dest)
    return dest
