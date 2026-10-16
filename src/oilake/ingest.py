"""Open Images metadata -> Delta tables.

Tables written (all under <lake>/delta/):

  images         one row per image (9.18M). in_s3_mirror marks the ~1.9M whose
                 pixels are in s3://open-images-dataset; s3_key is set only for those.
  image_labels   image-level labels, human + machine, partitioned by `kind`.
                 Human negatives (confidence 0) are kept on purpose: they are
                 the cleanest source of anti-correlated filter queries.
  classes        MID -> display name, a dense int label_id (for int[] columns
                 in Postgres), and boxable/trainable flags.
  class_closure  (ancestor, descendant, depth) over the 600 boxable classes,
                 so a filter on "Animal" can include every descendant.

Everything streams through pyarrow record batches, so memory stays bounded
even for the multi-GB label files.
"""

from __future__ import annotations

import json
import sys
from typing import Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from deltalake import write_deltalake

from . import sources
from .download import cached
from .hierarchy import closure
from .keys import parse_rotation, sample_key
from .lake import Lake

_BLOCK = 64 << 20


def _open_csv(path: str, columns: list[str], header: bool = True) -> pacsv.CSVStreamingReader:
    """Stream a CSV with every column read as string (no type inference surprises)."""
    read = pacsv.ReadOptions(block_size=_BLOCK,
                             column_names=None if header else columns,
                             autogenerate_column_names=False)
    parse = pacsv.ParseOptions(newlines_in_values=True)  # titles contain newlines
    convert = pacsv.ConvertOptions(column_types={c: pa.string() for c in columns},
                                   include_columns=columns,
                                   strings_can_be_null=False)
    return pacsv.open_csv(path, read_options=read, parse_options=parse, convert_options=convert)


def _write(lake: Lake, table: str, reader: pa.RecordBatchReader, partition_by=None) -> None:
    write_deltalake(lake.table(table), reader, mode="overwrite", schema_mode="overwrite",
                    partition_by=partition_by, storage_options=lake.storage_options)
    print(f"wrote {lake.table(table)}", file=sys.stderr)


# --------------------------------------------------------------------------- images

IMAGES_SCHEMA = pa.schema([
    ("image_id", pa.string()),
    ("subset", pa.string()),
    ("in_s3_mirror", pa.bool_()),
    ("s3_key", pa.string()),                  # e.g. "train/000002b66c9c498e.jpg"
    ("sample_key", pa.int64()),              # see keys.sample_key
    ("rotation", pa.int16()),                 # degrees CCW; null if unknown
    ("author", pa.string()),
    ("author_profile_url", pa.string()),      # stable grouping key for "per photographer"
    ("license", pa.string()),
    ("title", pa.string()),
    ("original_url", pa.string()),
    ("original_landing_url", pa.string()),
    ("thumbnail_300k_url", pa.string()),      # for the later Flickr crawl tier
    ("original_size", pa.int64()),
    ("original_md5", pa.string()),
])

_IMAGE_COLS = ["ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "License",
               "AuthorProfileURL", "Author", "Title", "OriginalSize", "OriginalMD5",
               "Thumbnail300KURL", "Rotation"]


def _mirror_ids(cache_dir: str) -> pa.Array:
    ids = []
    for url in sources.MIRROR_IMAGE_LISTS.values():
        path = cached(url, cache_dir)
        for batch in _open_csv(path, ["ImageID"]):
            ids.append(batch.column(0))
    return pa.concat_arrays([a.combine_chunks() if isinstance(a, pa.ChunkedArray) else a for a in ids])


def _empty_to_null(arr: pa.Array) -> pa.Array:
    return pc.if_else(pc.equal(arr, ""), pa.scalar(None, pa.string()), arr)


def _image_batches(path: str, mirror: pa.Array) -> Iterator[pa.RecordBatch]:
    for b in _open_csv(path, _IMAGE_COLS):
        ids = b.column("ImageID")
        subset = b.column("Subset")
        in_mirror = pc.is_in(ids, value_set=mirror)
        key = pc.binary_join_element_wise(subset, ids, "/")
        key = pc.binary_join_element_wise(key, pa.scalar(".jpg"), "")
        key = pc.if_else(in_mirror, key, pa.scalar(None, pa.string()))
        rot = pa.array([parse_rotation(v) for v in b.column("Rotation").to_pylist()], pa.int16())
        skey = pa.array([sample_key(v) for v in ids.to_pylist()], pa.int64())
        size = pc.cast(_empty_to_null(b.column("OriginalSize")), pa.int64())
        yield pa.RecordBatch.from_arrays([
            ids, subset, in_mirror, key, skey, rot,
            _empty_to_null(b.column("Author")),
            _empty_to_null(b.column("AuthorProfileURL")),
            _empty_to_null(b.column("License")),
            _empty_to_null(b.column("Title")),
            _empty_to_null(b.column("OriginalURL")),
            _empty_to_null(b.column("OriginalLandingURL")),
            _empty_to_null(b.column("Thumbnail300KURL")),
            size,
            _empty_to_null(b.column("OriginalMD5")),
        ], schema=IMAGES_SCHEMA)


def ingest_images(lake: Lake, cache_dir: str) -> None:
    mirror = _mirror_ids(cache_dir)
    print(f"{len(mirror):,} image ids in the S3 mirror lists", file=sys.stderr)
    path = cached(sources.ALL_IMAGE_IDS, cache_dir)
    reader = pa.RecordBatchReader.from_batches(IMAGES_SCHEMA, _image_batches(path, mirror))
    _write(lake, "images", reader)


# --------------------------------------------------------------------------- labels

LABELS_SCHEMA = pa.schema([
    ("image_id", pa.string()),
    ("label", pa.string()),          # MID, e.g. /m/0bt9lr (Dog)
    ("confidence", pa.float32()),    # human: 1 present / 0 verified absent; machine: score
    ("source", pa.string()),         # verification | crowdsource-verification | machine
    ("subset", pa.string()),
    ("kind", pa.string()),           # human | machine (partition column)
])


def _label_batches(files: list[tuple[str, str, str]]) -> Iterator[pa.RecordBatch]:
    for path, subset, kind in files:
        print(f"reading {path}", file=sys.stderr)
        for b in _open_csv(path, ["ImageID", "Source", "LabelName", "Confidence"]):
            n = b.num_rows
            yield pa.RecordBatch.from_arrays([
                b.column("ImageID"),
                b.column("LabelName"),
                pc.cast(b.column("Confidence"), pa.float32()),
                b.column("Source"),
                pa.array([subset] * n, pa.string()),
                pa.array([kind] * n, pa.string()),
            ], schema=LABELS_SCHEMA)


def ingest_labels(lake: Lake, cache_dir: str) -> None:
    files = []
    for kind, urls in (("human", sources.HUMAN_LABELS), ("machine", sources.MACHINE_LABELS)):
        for subset, url in urls.items():
            files.append((cached(url, cache_dir), subset, kind))
    reader = pa.RecordBatchReader.from_batches(LABELS_SCHEMA, _label_batches(files))
    _write(lake, "image_labels", reader, partition_by=["kind"])


# --------------------------------------------------------------------------- classes

def _read_two_col(path: str) -> list[tuple[str, str]]:
    """Class description CSVs; some versions ship a header row, some don't."""
    rows = []
    for b in _open_csv(path, ["mid", "name"], header=False):
        rows.extend(zip(b.column(0).to_pylist(), b.column(1).to_pylist()))
    if rows and rows[0][0] == "LabelName":
        rows = rows[1:]
    return rows


def build_classes(descriptions: list[tuple[str, str]], boxable: set[str],
                  trainable: set[str]) -> pa.Table:
    """Dense label_id assigned in sorted-MID order, so it is stable across runs."""
    names = dict(descriptions)
    mids = sorted(names)
    return pa.table({
        "label_id": pa.array(range(len(mids)), pa.int32()),
        "label": pa.array(mids, pa.string()),
        "display_name": pa.array([names[m] for m in mids], pa.string()),
        "is_boxable": pa.array([m in boxable for m in mids], pa.bool_()),
        "is_trainable": pa.array([m in trainable for m in mids], pa.bool_()),
    })


def ingest_classes(lake: Lake, cache_dir: str) -> None:
    descriptions = _read_two_col(cached(sources.CLASS_DESCRIPTIONS, cache_dir))
    boxable = {m for m, _ in _read_two_col(cached(sources.BOXABLE_CLASS_DESCRIPTIONS, cache_dir))}
    with open(cached(sources.TRAINABLE_CLASSES, cache_dir)) as f:
        trainable = {line.strip() for line in f if line.strip()}
    table = build_classes(descriptions, boxable, trainable)
    _write(lake, "classes", table.to_reader())

    with open(cached(sources.BOXABLE_HIERARCHY, cache_dir)) as f:
        rows = closure(json.load(f))
    clo = pa.table({
        "ancestor": pa.array([r[0] for r in rows], pa.string()),
        "descendant": pa.array([r[1] for r in rows], pa.string()),
        "depth": pa.array([r[2] for r in rows], pa.int16()),
    })
    _write(lake, "class_closure", clo.to_reader())


def ingest_all(lake: Lake, cache_dir: str, only: list[str] | None = None) -> None:
    steps = {"classes": ingest_classes, "images": ingest_images, "labels": ingest_labels}
    for name, fn in steps.items():
        if only is None or name in only:
            fn(lake, cache_dir)
