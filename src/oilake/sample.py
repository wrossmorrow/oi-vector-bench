"""Named, nested benchmark samples.

A sample is the first N images, in sample_key order, that have a good
embedding for a model. Because the order is fixed, samples are nested
(mirror_1m is a subset of mirror_all) and reproducible from the lake alone.
"""

from __future__ import annotations

import sys

import pyarrow as pa
import pyarrow.compute as pc

from .lake import Lake

SAMPLES_SCHEMA = pa.schema(
    [
        ("sample_name", pa.string()),
        ("model_id", pa.string()),
        ("image_id", pa.string()),
        ("rank", pa.int64()),  # position in sample_key order
    ]
)


def choose(
    images: pa.Table,
    embedded_ok: pa.Array,
    n: int | None,
    *,
    tail: bool = False,
    exclude: pa.Array | None = None,
) -> pa.Table:
    """images: image_id, sample_key. Returns image_id, rank for the first n eligible.

    tail=True takes the LAST n in sample_key order instead. That is how the
    query set is held out: `--name queries --n 5000 --tail`, then every other
    sample built with `--exclude queries`. mirror_1m (the first 1M) can never
    overlap a tail of 5000; mirror_all needs the exclude.
    """
    t = images.filter(pc.is_in(images["image_id"], value_set=embedded_ok))
    if exclude is not None and len(exclude):
        t = t.filter(pc.invert(pc.is_in(t["image_id"], value_set=exclude)))
    order = "descending" if tail else "ascending"
    t = t.sort_by([("sample_key", order), ("image_id", order)])
    if n is not None:
        if t.num_rows < n:
            raise SystemExit(f"only {t.num_rows:,} eligible images, asked for {n:,}")
        t = t.slice(0, n)
    return pa.table({"image_id": t["image_id"], "rank": pa.array(range(t.num_rows), pa.int64())})


def read_sample_ids(lake: Lake, name: str) -> pa.Table:
    """image_id, rank for a named sample, in rank order."""
    from deltalake import DeltaTable

    ds = DeltaTable(lake.table("samples"), storage_options=lake.storage_options).to_pyarrow_dataset()
    t = ds.to_table(columns=["image_id", "rank"], filter=pc.field("sample_name") == name)
    if t.num_rows == 0:
        raise SystemExit(f"no sample named {name!r}")
    return t.sort_by("rank")


def build_sample(
    lake: Lake,
    model_id: str,
    name: str,
    n: int | None,
    *,
    tail: bool = False,
    exclude: str | None = None,
) -> None:
    from deltalake import DeltaTable, write_deltalake

    opts = lake.storage_options
    emb = DeltaTable(lake.table("embeddings"), storage_options=opts).to_pyarrow_dataset()
    ok = emb.to_table(
        columns=["image_id"],
        filter=(pc.field("model_id") == model_id) & pc.field("error").is_null(),
    )
    images = (
        DeltaTable(lake.table("images"), storage_options=opts)
        .to_pyarrow_dataset()
        .to_table(columns=["image_id", "sample_key"], filter=pc.field("in_s3_mirror"))
    )
    excl = read_sample_ids(lake, exclude)["image_id"].combine_chunks() if exclude else None
    chosen = choose(images, ok["image_id"].combine_chunks(), n, tail=tail, exclude=excl)
    out = pa.table(
        {
            "sample_name": pa.array([name] * chosen.num_rows, pa.string()),
            "model_id": pa.array([model_id] * chosen.num_rows, pa.string()),
            "image_id": chosen["image_id"],
            "rank": chosen["rank"],
        },
        schema=SAMPLES_SCHEMA,
    )

    uri = lake.table("samples")
    exists = DeltaTable.is_deltatable(uri, storage_options=opts)
    write_deltalake(
        uri,
        out,
        mode="overwrite",
        partition_by=["sample_name"],
        predicate=f"sample_name = '{name}'" if exists else None,
        storage_options=opts,
    )
    print(f"sample {name}: {out.num_rows:,} images -> {uri}", file=sys.stderr)
