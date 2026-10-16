"""Measure the three embedding bottlenecks separately, before a real run.

    uv run python scripts/preflight.py [--fetch-threads 64] [--n 2000]

1. S3 fetch: concurrent unsigned GETs from the mirror (network-bound ceiling).
2. Decode + preprocess: one process, so multiply by vCPUs for the CPU ceiling.
3. GPU encode: random input at batch sizes 256 / 512 (model-bound ceiling).

The full job runs at roughly min(fetch, decode x workers, gpu).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oilake import sources  # noqa: E402
from oilake.embed import S3MirrorFetcher, decode, load_encoder  # noqa: E402

IMAGE_SIZE = 336


def list_keys(n: int) -> list[str]:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", region_name=sources.MIRROR_REGION,
                      config=Config(signature_version=UNSIGNED))
    keys, token = [], None
    while len(keys) < n:
        kw = {"Bucket": sources.MIRROR_BUCKET, "Prefix": "validation/", "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".jpg")]
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return keys[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--fetch-threads", type=int, default=64)
    ap.add_argument("--model", default="pe-core-l14-336")
    a = ap.parse_args()

    keys = list_keys(a.n)
    print(f"listed {len(keys)} mirror keys")

    # 1. fetch
    fetch = S3MirrorFetcher()
    t0 = time.time()
    with ThreadPoolExecutor(a.fetch_threads) as pool:
        blobs = list(pool.map(fetch, keys))
    dt = time.time() - t0
    mb = sum(map(len, blobs)) / 1e6
    print(f"fetch:  {len(blobs) / dt:7.0f} img/s  {mb / dt:6.0f} MB/s  "
          f"({a.fetch_threads} threads, avg {mb / len(blobs) * 1000:.0f} KB/img)")

    # 2. decode + preprocess, single process
    import torch

    enc = load_encoder(a.model)
    sample = blobs[:300]
    t0 = time.time()
    for b in sample:
        enc.preprocess(decode(b, None, False))
    per_proc = len(sample) / (time.time() - t0)
    cpus = os.cpu_count() or 1
    print(f"decode: {per_proc:7.0f} img/s per process -> ~{per_proc * (cpus - 2):.0f} img/s "
          f"with {cpus - 2} workers ({cpus} vCPUs)")

    # 3. GPU encode
    print(f"device: {enc.device}"
          + (f" ({torch.cuda.get_device_name()})" if enc.device == "cuda" else ""))
    for bs in (256, 512):
        x = torch.randn(bs, 3, IMAGE_SIZE, IMAGE_SIZE)
        enc.encode(x)  # warmup
        if enc.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        reps = 5
        for _ in range(reps):
            z = enc.encode(x)
        print(f"gpu:    {bs * reps / (time.time() - t0):7.0f} img/s  (batch {bs}, dim {z.shape[1]})")

    # sanity: real images embed to unit vectors, and two different images differ
    real = torch.stack([enc.preprocess(decode(b, None, False)) for b in blobs[:2]])
    z = enc.encode(real)
    print(f"check:  norms {z[0] @ z[0]:.3f} {z[1] @ z[1]:.3f}, cos(img0, img1) = {z[0] @ z[1]:.3f}")


if __name__ == "__main__":
    main()
