"""Resumable image-embedding job over the S3 mirror.

Three steps, each safe to re-run:

  plan      Read the images table, keep mirror images, sort by sample_key and
            cut into fixed-size units. Saved as <staging>/plan.parquet.
            Because units follow sample_key order, embedding the first ~50
            units (1M images) yields exactly the 1M benchmark sample, so you
            can stop early and still have a clean tier.
  run       Embed units that don't have output yet. Each unit is one Parquet
            file in staging, written only when complete, so a spot
            interruption loses at most one unit. `--worker k/N` splits units
            across machines.
  finalize  Check every planned unit is present, then load staging into the
            Delta `embeddings` table (partition model_id), replacing that
            model's rows in one commit. Single writer, so no Delta locking.

Embeddings are stored as float32 arrays: Delta has no float16 type, and
37 GB for 9M x 1024 is cheap in S3. They are L2-normalized.
"""

from __future__ import annotations

import io
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import sources
from .keys import chunk_units, unit_name, units_for_worker
from .lake import Lake

# Concurrent S3 GETs per DataLoader worker process. Inside AWS a few suffice;
# from Lambda or home, per-request latency is higher, so overlap more of them.
FETCH_THREADS = int(os.environ.get("OILAKE_FETCH_THREADS", "16"))

# model_id -> (open_clip model name, open_clip pretrained tag or None, embedding dim)
MODELS = {
    "pe-core-l14-336": ("hf-hub:timm/PE-Core-L-14-336", None, 1024),
    # 768-d: under pg_vectorsearch's ~945-d limit for fastscan posting pages.
    "clip-l14-336": ("ViT-L-14-336", "openai", 768),
}

PLAN_SCHEMA = pa.schema([
    ("unit", pa.int32()),
    ("image_id", pa.string()),
    ("s3_key", pa.string()),
    ("rotation", pa.int16()),
    ("sample_key", pa.int64()),
])


# --------------------------------------------------------------------------- plan

def make_plan(images: pa.Table, unit_size: int, limit: int | None = None) -> pa.Table:
    """images: columns image_id, s3_key, rotation, sample_key, in_s3_mirror."""
    t = images.filter(pc.field("in_s3_mirror"))
    t = t.sort_by([("sample_key", "ascending"), ("image_id", "ascending")])
    if limit is not None:
        t = t.slice(0, limit)
    unit = pa.array([i // unit_size for i in range(t.num_rows)], pa.int32())
    return pa.table([unit, t["image_id"], t["s3_key"], t["rotation"], t["sample_key"]],
                    schema=PLAN_SCHEMA)


def plan(lake: Lake, model_id: str, unit_size: int = 20_000, limit: int | None = None) -> str:
    from deltalake import DeltaTable

    _check_model(model_id)
    dt = DeltaTable(lake.table("images"), storage_options=lake.storage_options)
    images = dt.to_pyarrow_dataset().to_table(
        columns=["image_id", "s3_key", "rotation", "sample_key", "in_s3_mirror"],
        filter=pc.field("in_s3_mirror"))
    p = make_plan(images, unit_size, limit)
    fs, _ = lake.filesystem()
    path = lake.fs_path(lake.staging(model_id)) + "/plan.parquet"
    fs.create_dir(os.path.dirname(path), recursive=True)
    pq.write_table(p, path, filesystem=fs)
    n_units = len(chunk_units(p.num_rows, unit_size))
    print(f"planned {p.num_rows:,} images in {n_units} units -> {path}", file=sys.stderr)
    return path


def _check_model(model_id: str) -> None:
    if model_id not in MODELS:
        raise SystemExit(f"unknown model {model_id!r}; known: {', '.join(MODELS)}")


def _load_plan(lake: Lake, model_id: str):
    fs, _ = lake.filesystem()
    stage = lake.fs_path(lake.staging(model_id))
    return fs, stage, pq.read_table(stage + "/plan.parquet", filesystem=fs)


def _done_units(fs, stage: str) -> list[str]:
    import pyarrow.fs as pafs

    sel = pafs.FileSelector(stage + "/units", allow_not_found=True)
    return [os.path.basename(i.path)[: -len(".parquet")]
            for i in fs.get_file_info(sel) if i.path.endswith(".parquet")]


# --------------------------------------------------------------------------- encoders

@dataclass
class Encoder:
    preprocess: Callable          # PIL.Image -> CHW tensor
    encode: Callable              # NCHW tensor -> (N, dim) L2-normalized float32 numpy
    dim: int
    device: str


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_encoder(model_id: str, device: str = "auto") -> Encoder:
    import open_clip
    import torch

    _check_model(model_id)
    name, pretrained, dim = MODELS[model_id]
    device = pick_device(device)
    model, preprocess = (open_clip.create_model_from_pretrained(name, pretrained=pretrained)
                         if pretrained else open_clip.create_model_from_pretrained(name))
    model = model.eval().to(device)
    use_fp16 = device == "cuda"

    @torch.inference_mode()
    def encode(x):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            z = model.encode_image(x)
        z = torch.nn.functional.normalize(z.float(), dim=-1)
        return z.cpu().numpy()

    return Encoder(preprocess, encode, dim, device)


# --------------------------------------------------------------------------- fetching

class S3MirrorFetcher:
    """Unsigned reads from the public mirror; one boto3 client per process."""

    def __init__(self, bucket: str = sources.MIRROR_BUCKET):
        self.bucket = bucket
        self._client = None
        self._pid = None

    def __getstate__(self):  # boto3 clients don't pickle; rebuild in each worker
        return {"bucket": self.bucket, "_client": None, "_pid": None}

    def __call__(self, key: str) -> bytes:
        if self._client is None or self._pid != os.getpid():
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config

            # Pin the mirror's region: hosts outside AWS (Lambda, the Mac) may have
            # no AWS config at all, and the lake bucket's region is irrelevant here.
            self._client = boto3.client("s3", region_name=sources.MIRROR_REGION, config=Config(
                signature_version=UNSIGNED, retries={"max_attempts": 8, "mode": "adaptive"},
                max_pool_connections=FETCH_THREADS))
            self._pid = os.getpid()
        return self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read()


class LocalFetcher:
    """Read keys under a local directory (an `aws s3 sync` copy, or test fixtures)."""

    def __init__(self, root: str):
        self.root = root

    def __call__(self, key: str) -> bytes:
        with open(os.path.join(self.root, key), "rb") as f:
            return f.read()


def decode(data: bytes, rotation: int | None, apply_rotation: bool):
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    if apply_rotation and rotation:
        # Open Images Rotation is degrees counter-clockwise; PIL rotates CCW too.
        img = img.rotate(rotation, expand=True)
    return img


# --------------------------------------------------------------------------- run

def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"[:500]


class UnitDataset:
    """Map-style dataset over one unit's rows.

    Module-level and picklable (no live pool or client in its state), because
    on macOS DataLoader workers are spawned and receive the dataset by pickle.
    Duck-typed rather than subclassing torch Dataset so this module imports
    without torch.
    """

    def __init__(self, rows: list[dict], fetch, preprocess, apply_rotation: bool):
        self.rows = rows
        self.fetch = fetch
        self.preprocess = preprocess
        self.apply_rotation = apply_rotation
        self._pool = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pool"] = None
        return state

    def __len__(self):
        return len(self.rows)

    def _process(self, i, data):
        try:
            img = decode(data, self.rows[i]["rotation"], self.apply_rotation)
            return self.preprocess(img), i, ""
        except Exception as e:  # one bad image must not kill the unit
            return None, i, _err(e)

    def _get(self, i):
        try:
            return self.fetch(self.rows[i]["s3_key"]), None
        except Exception as e:
            return None, _err(e)

    def __getitem__(self, i):
        data, err = self._get(i)
        return (None, i, err) if err else self._process(i, data)

    def __getitems__(self, idx):
        # DataLoader hands each worker a whole batch of indices here. Fetch them
        # concurrently (I/O bound, GIL released), then decode serially (CPU
        # bound; parallelism comes from the worker processes).
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(FETCH_THREADS)
        fetched = list(self._pool.map(self._get, idx))
        return [(None, i, err) if err else self._process(i, data)
                for i, (data, err) in zip(idx, fetched)]


def _collate(batch):
    import torch

    ok = [(x, i) for x, i, err in batch if x is not None]
    errs = [(i, err) for x, i, err in batch if x is None]
    xs = torch.stack([x for x, _ in ok]) if ok else None
    return xs, [i for _, i in ok], errs


def embed_unit(rows: list[dict], encoder: Encoder, fetch, *, model_id: str,
               batch_size: int, num_workers: int, apply_rotation: bool) -> pa.Table:
    import numpy as np
    import torch

    ds = UnitDataset(rows, fetch, encoder.preprocess, apply_rotation)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate,
        pin_memory=encoder.device == "cuda", prefetch_factor=4 if num_workers else None,
        persistent_workers=False)

    n = len(rows)
    vecs = np.zeros((n, encoder.dim), dtype=np.float32)
    errors: list[str | None] = [None] * n
    for xs, idx, errs in loader:
        if xs is not None:
            vecs[idx] = encoder.encode(xs)
        for i, err in errs:
            errors[i] = err

    mask = pa.array([e is not None for e in errors])
    emb = pa.FixedSizeListArray.from_arrays(pa.array(vecs.reshape(-1)), encoder.dim, mask=mask)
    return pa.table({
        "image_id": pa.array([r["image_id"] for r in rows], pa.string()),
        "embedding": emb,
        "error": pa.array(errors, pa.string()),
        "rotation_applied": pa.array([bool(apply_rotation and r["rotation"]) for r in rows]),
        "model_id": pa.array([model_id] * n, pa.string()),
    })


def run(lake: Lake, model_id: str, *, worker: tuple[int, int] = (0, 1), batch_size: int = 256,
        num_workers: int = 24, device: str = "auto", apply_rotation: bool = True,
        local_images: str | None = None, max_units: int | None = None) -> None:
    fs, stage, p = _load_plan(lake, model_id)
    fs.create_dir(stage + "/units", recursive=True)
    unit_ids = sorted(set(p["unit"].to_pylist()))
    todo = units_for_worker(unit_ids, worker, _done_units(fs, stage))
    if max_units is not None:
        todo = todo[:max_units]
    print(f"worker {worker[0]}/{worker[1]}: {len(todo)} units to do", file=sys.stderr)
    if not todo:
        return

    encoder = load_encoder(model_id, device)
    fetch = LocalFetcher(local_images) if local_images else S3MirrorFetcher()
    for u in todo:
        t0 = time.time()
        rows = p.filter(pc.equal(p["unit"], u)).select(
            ["image_id", "s3_key", "rotation"]).to_pylist()
        out = embed_unit(rows, encoder, fetch, model_id=model_id, batch_size=batch_size,
                         num_workers=num_workers, apply_rotation=apply_rotation)
        # S3 objects appear atomically on completion; locally, write-then-rename.
        final = f"{stage}/units/{unit_name(u)}.parquet"
        tmp = final if lake.is_s3 else final + ".tmp"
        pq.write_table(out, tmp, filesystem=fs)
        if tmp != final:
            fs.move(tmp, final)
        n_err = len(rows) - out["error"].null_count
        dt = time.time() - t0
        print(f"{unit_name(u)}: {len(rows):,} images, {n_err:,} errors, "
              f"{len(rows) / dt:,.0f} img/s", file=sys.stderr)


# --------------------------------------------------------------------------- finalize

def finalize(lake: Lake, model_id: str, allow_missing: bool = False) -> None:
    from deltalake import DeltaTable, write_deltalake

    fs, stage, p = _load_plan(lake, model_id)
    planned = {unit_name(u) for u in set(p["unit"].to_pylist())}
    done = set(_done_units(fs, stage))
    missing = sorted(planned - done)
    if missing and not allow_missing:
        raise SystemExit(f"{len(missing)} units not embedded yet (first: {missing[0]}); "
                         "run more workers or pass --allow-missing")

    dim = MODELS[model_id][2]
    schema = pa.schema([
        ("image_id", pa.string()),
        ("embedding", pa.list_(pa.float32())),  # Delta has no fixed-size list
        ("error", pa.string()),
        ("rotation_applied", pa.bool_()),
        ("model_id", pa.string()),
    ])

    def batches():
        for name in sorted(done & planned):
            t = pq.read_table(f"{stage}/units/{name}.parquet", filesystem=fs)
            t = t.set_column(1, "embedding", t["embedding"].cast(pa.list_(pa.float32())))
            yield from t.select(schema.names).cast(schema).to_batches()

    reader = pa.RecordBatchReader.from_batches(schema, batches())
    uri = lake.table("embeddings")
    exists = DeltaTable.is_deltatable(uri, storage_options=lake.storage_options)
    write_deltalake(uri, reader, mode="overwrite", partition_by=["model_id"],
                    predicate=f"model_id = '{model_id}'" if exists else None,
                    storage_options=lake.storage_options)
    print(f"finalized {len(done & planned)} units of {model_id} (dim {dim}) -> {uri}",
          file=sys.stderr)
