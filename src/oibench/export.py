"""Lake -> local numpy dataset (the only oibench step that reads Delta/S3).

Labels for items:
  human-verified positives (confidence 1) by default, plus machine labels at
  or above --machine-threshold if given. With --expand (default), each label
  also implies its ancestors in the boxable hierarchy, so a filter on
  "Animal" matches images only labeled "Dog". Negatives are NOT expanded:
  "not Animal" implies "not Dog", but not the other way round.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import numpy as np

from oilake.lake import Lake
from oilake.sample import read_sample_ids

from .data import CSR, Dataset


def _log(*a) -> None:
    print(*a, file=sys.stderr)


def _delta(lake: Lake, name: str):
    from deltalake import DeltaTable

    return DeltaTable(lake.table(name), storage_options=lake.storage_options).to_pyarrow_dataset()


def _embeddings(lake: Lake, model_id: str, image_ids: list[str], dim_hint: int | None = None):
    """(len(image_ids), D) float16 in the given order; raises if any are missing."""
    import pyarrow as pa
    import pyarrow.compute as pc

    pos = {iid: i for i, iid in enumerate(image_ids)}
    out = None
    found = np.zeros(len(image_ids), dtype=bool)
    ds = _delta(lake, "embeddings")
    flt = (
        (pc.field("model_id") == model_id)
        & pc.field("error").is_null()
        & pc.field("image_id").isin(pa.array(image_ids))
    )
    for batch in ds.to_batches(columns=["image_id", "embedding"], filter=flt):
        if batch.num_rows == 0:
            continue
        col = batch.column("embedding")
        flat = pc.list_flatten(col).to_numpy(zero_copy_only=False)
        dim = len(flat) // batch.num_rows
        if out is None:
            out = np.zeros((len(image_ids), dim), dtype=np.float16)
        rows = np.fromiter((pos[x] for x in batch.column("image_id").to_pylist()), dtype=np.int64)
        out[rows] = flat.reshape(-1, dim).astype(np.float16)
        found[rows] = True
    if out is None or not found.all():
        missing = int((~found).sum()) if out is not None else len(image_ids)
        raise SystemExit(f"{missing:,} images have no good {model_id} embedding")
    return out


def _labels(
    lake: Lake,
    image_ids: list[str],
    label_id: dict,
    closure: dict | None,
    positive: bool,
    machine_threshold: float | None,
) -> CSR:
    import pyarrow as pa
    import pyarrow.compute as pc

    pos = {iid: i for i, iid in enumerate(image_ids)}
    ds = _delta(lake, "image_labels")
    ids = pa.array(image_ids)
    conf = 1.0 if positive else 0.0
    filters = [(pc.field("kind") == "human") & (pc.field("confidence") == conf)]
    if positive and machine_threshold is not None:
        filters.append((pc.field("kind") == "machine") & (pc.field("confidence") >= machine_threshold))
    rows, cols = [], []
    for f in filters:
        t = ds.to_table(columns=["image_id", "label"], filter=f & pc.field("image_id").isin(ids))
        r = np.fromiter((pos[x] for x in t.column("image_id").to_pylist()), dtype=np.int64)
        lab = t.column("label").to_pylist()
        c = np.fromiter((label_id.get(m, -1) for m in lab), dtype=np.int64)
        keep = c >= 0
        rows.append(r[keep])
        cols.append(c[keep])
        _log(f"  {t.num_rows:,} label rows ({int((~keep).sum()):,} with unknown MIDs dropped)")
    rows_a, cols_a = np.concatenate(rows), np.concatenate(cols)
    if closure:
        # add ancestors: closure maps descendant label_id -> ancestor label_ids (incl. self)
        er, ec = [rows_a], [cols_a]
        for d, ancs in closure.items():
            hit = cols_a == d
            if hit.any():
                for a in ancs:
                    if a != d:
                        er.append(rows_a[hit])
                        ec.append(np.full(int(hit.sum()), a, dtype=np.int64))
        rows_a, cols_a = np.concatenate(er), np.concatenate(ec)
    return CSR.from_pairs(rows_a, cols_a, len(image_ids))


def export(
    lake: Lake,
    out_dir: str,
    *,
    sample: str,
    queries: str,
    model_id: str,
    expand: bool = True,
    machine_threshold: float | None = None,
) -> None:
    import pyarrow.compute as pc

    item_ids = read_sample_ids(lake, sample)["image_id"].to_pylist()
    q_ids = read_sample_ids(lake, queries)["image_id"].to_pylist()
    overlap = set(item_ids) & set(q_ids)
    if overlap:
        raise SystemExit(
            f"{len(overlap)} query images are also in {sample!r}; rebuild {sample!r} with --exclude {queries}"
        )
    _log(f"{len(item_ids):,} items from {sample!r}, {len(q_ids):,} queries from {queries!r}")

    classes_t = _delta(lake, "classes").to_table().sort_by("label_id")
    classes = list(
        zip(
            classes_t["label_id"].to_pylist(),
            classes_t["label"].to_pylist(),
            classes_t["display_name"].to_pylist(),
            strict=True,
        )
    )
    label_id = {mid: i for i, mid, _ in classes}

    closure = None
    if expand:
        clo = _delta(lake, "class_closure").to_table()
        closure = {}
        for a, d in zip(clo["ancestor"].to_pylist(), clo["descendant"].to_pylist(), strict=True):
            if a in label_id and d in label_id:
                closure.setdefault(label_id[d], []).append(label_id[a])

    _log("embeddings: items")
    emb = _embeddings(lake, model_id, item_ids)
    _log("embeddings: queries")
    q_emb = _embeddings(lake, model_id, q_ids)
    _log("labels: items")
    labels = _labels(lake, item_ids, label_id, closure, True, machine_threshold)
    _log("labels: query positives")
    q_pos = _labels(lake, q_ids, label_id, closure, True, machine_threshold)
    _log("labels: query verified negatives")
    q_neg = _labels(lake, q_ids, label_id, None, False, None)

    _log("authors")
    img = _delta(lake, "images").to_table(
        columns=["image_id", "author_profile_url"],
        filter=pc.field("image_id").isin(item_ids),
    )
    url_of = dict(zip(img["image_id"].to_pylist(), img["author_profile_url"].to_pylist(), strict=True))
    codes: dict = {}
    author = np.empty(len(item_ids), dtype=np.int64)
    for i, iid in enumerate(item_ids):
        u = url_of.get(iid)
        author[i] = codes.setdefault(u, len(codes)) if u else -(i + 1)

    ds = Dataset(
        emb=emb,
        labels=labels,
        author=author,
        image_ids=item_ids,
        q_emb=q_emb,
        q_pos=q_pos,
        q_neg=q_neg,
        q_image_ids=q_ids,
        classes=classes,
        meta={
            "lake": lake.root,
            "sample": sample,
            "queries": queries,
            "model_id": model_id,
            "expand_hierarchy": expand,
            "machine_threshold": machine_threshold,
            "label_rows": int(len(labels.indices)),
            "known_authors": len(codes),
            "created": datetime.now(UTC).isoformat(),
        },
    )
    ds.save(out_dir)
    _log(f"wrote {out_dir}: {ds.n:,} x {ds.dim}, {len(labels.indices):,} item labels, {len(codes):,} authors")
