"""oilake command line.

oilake ingest   --lake s3://bucket/oi [--cache ~/.cache/oilake] [--only classes images labels]
oilake plan     --lake ... --model pe-core-l14-336 [--unit-size 20000] [--limit N]
oilake embed    --lake ... --model pe-core-l14-336 [--worker 0/1] [--batch-size 256] ...
oilake finalize --lake ... --model pe-core-l14-336
oilake sample   --lake ... --model pe-core-l14-336 --name mirror_1m --n 1000000
"""

from __future__ import annotations

import argparse
import os

from .keys import parse_worker
from .lake import Lake


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="oilake")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, model: bool = True):
        p = sub.add_parser(name)
        p.add_argument("--lake", required=True, help="local dir or s3://bucket/prefix")
        if model:
            p.add_argument("--model", default="pe-core-l14-336")
        return p

    p = add("ingest", model=False)
    p.add_argument("--cache", default=os.path.expanduser("~/.cache/oilake"))
    p.add_argument("--only", nargs="*", choices=["classes", "images", "labels"])

    p = add("plan")
    p.add_argument("--unit-size", type=int, default=20_000)
    p.add_argument("--limit", type=int, help="plan only the first N images by sample_key")

    p = add("embed")
    p.add_argument("--worker", default="0/1", help="k/N: take units with index %% N == k")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--num-workers",
        type=int,
        default=24,
        help="DataLoader processes; each does its own S3 GETs + JPEG decode",
    )
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument(
        "--no-rotation",
        action="store_true",
        help="don't apply the Open Images Rotation field before embedding",
    )
    p.add_argument("--local-images", help="read images from this directory instead of S3")
    p.add_argument("--max-units", type=int, help="stop after this many units (smoke tests)")

    p = add("finalize")
    p.add_argument("--allow-missing", action="store_true")

    p = add("sample")
    p.add_argument("--name", required=True)
    p.add_argument("--n", type=int, help="omit for every eligible image")
    p.add_argument(
        "--tail",
        action="store_true",
        help="take the last N in sample_key order (use for the query holdout)",
    )
    p.add_argument("--exclude", help="name of a sample whose images to leave out")

    a = ap.parse_args(argv)
    lake = Lake.from_root(a.lake)

    if a.cmd == "ingest":
        from .ingest import ingest_all

        ingest_all(lake, a.cache, a.only)
    elif a.cmd == "plan":
        from .embed import plan

        plan(lake, a.model, a.unit_size, a.limit)
    elif a.cmd == "embed":
        from .embed import run

        run(
            lake,
            a.model,
            worker=parse_worker(a.worker),
            batch_size=a.batch_size,
            num_workers=a.num_workers,
            device=a.device,
            apply_rotation=not a.no_rotation,
            local_images=a.local_images,
            max_units=a.max_units,
        )
    elif a.cmd == "finalize":
        from .embed import finalize

        finalize(lake, a.model, a.allow_missing)
    elif a.cmd == "sample":
        from .sample import build_sample

        build_sample(lake, a.model, a.name, a.n, tail=a.tail, exclude=a.exclude)


if __name__ == "__main__":
    main()
