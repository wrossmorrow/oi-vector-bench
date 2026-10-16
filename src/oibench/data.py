"""The local benchmark dataset: plain numpy files.

Item i (row i of emb.npy) is Postgres id i. Labels are CSR lists of dense
label_ids (oilake `classes.label_id`). Embeddings are float16, because the
Postgres column is halfvec: storing and computing ground truth on the same
fp16-rounded values means "exact" is exactly what the database could return.

  <dir>/meta.json
  <dir>/classes.tsv                    label_id, mid, display_name
  <dir>/items/emb.npy                  (N, D) float16
  <dir>/items/labels_{indptr,indices}.npy
  <dir>/items/author.npy               (N,) int64, distinct negative ids for unknown authors
  <dir>/items/image_ids.txt
  <dir>/queries/emb.npy                (Q, D) float16
  <dir>/queries/pos_{indptr,indices}.npy   labels present (human-verified, maybe expanded)
  <dir>/queries/neg_{indptr,indices}.npy   labels human-verified ABSENT
  <dir>/queries/image_ids.txt
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CSR:
    indptr: np.ndarray   # (n_rows + 1,) int64
    indices: np.ndarray  # (nnz,) int32

    @property
    def n_rows(self) -> int:
        return len(self.indptr) - 1

    def row(self, i: int) -> np.ndarray:
        return self.indices[self.indptr[i]:self.indptr[i + 1]]

    @classmethod
    def from_lists(cls, rows) -> "CSR":
        lens = np.fromiter((len(r) for r in rows), dtype=np.int64)
        indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        flat = [x for r in rows for x in r]
        return cls(indptr, np.asarray(flat, dtype=np.int32))

    @classmethod
    def from_pairs(cls, rows: np.ndarray, cols: np.ndarray, n_rows: int) -> "CSR":
        """Build from (row, col) pairs; duplicates removed, cols sorted per row."""
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if len(rows):
            key = np.unique(rows * (int(cols.max()) + 1) + cols)
            width = int(cols.max()) + 1
            rows, cols = key // width, key % width
        counts = np.bincount(rows, minlength=n_rows)
        indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return cls(indptr, cols.astype(np.int32))

    def inverted(self, n_cols: int) -> "CSR":
        """Column -> sorted row ids (label -> items)."""
        row_of = np.repeat(np.arange(self.n_rows, dtype=np.int64), np.diff(self.indptr))
        order = np.argsort(self.indices, kind="stable")
        counts = np.bincount(self.indices, minlength=n_cols)
        indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return CSR(indptr, row_of[order].astype(np.int32))

    def save(self, prefix: str) -> None:
        np.save(prefix + "_indptr.npy", self.indptr)
        np.save(prefix + "_indices.npy", self.indices)

    @classmethod
    def load(cls, prefix: str) -> "CSR":
        return cls(np.load(prefix + "_indptr.npy"), np.load(prefix + "_indices.npy"))


@dataclass
class Dataset:
    emb: np.ndarray
    labels: CSR
    author: np.ndarray
    image_ids: list[str]
    q_emb: np.ndarray
    q_pos: CSR
    q_neg: CSR
    q_image_ids: list[str]
    classes: list[tuple[int, str, str]]
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.emb.shape[0]

    @property
    def dim(self) -> int:
        return self.emb.shape[1]

    @property
    def n_labels(self) -> int:
        return max(c[0] for c in self.classes) + 1 if self.classes else 0

    def label_name(self, label_id: int) -> str:
        return self._names().get(label_id, str(label_id))

    def _names(self) -> dict:
        if not hasattr(self, "_name_cache"):
            self._name_cache = {i: name for i, _, name in self.classes}
        return self._name_cache

    def save(self, d: str) -> None:
        os.makedirs(os.path.join(d, "items"), exist_ok=True)
        os.makedirs(os.path.join(d, "queries"), exist_ok=True)
        np.save(os.path.join(d, "items/emb.npy"), self.emb.astype(np.float16))
        self.labels.save(os.path.join(d, "items/labels"))
        np.save(os.path.join(d, "items/author.npy"), self.author.astype(np.int64))
        _write_lines(os.path.join(d, "items/image_ids.txt"), self.image_ids)
        np.save(os.path.join(d, "queries/emb.npy"), self.q_emb.astype(np.float16))
        self.q_pos.save(os.path.join(d, "queries/pos"))
        self.q_neg.save(os.path.join(d, "queries/neg"))
        _write_lines(os.path.join(d, "queries/image_ids.txt"), self.q_image_ids)
        with open(os.path.join(d, "classes.tsv"), "w") as f:
            for i, mid, name in self.classes:
                f.write(f"{i}\t{mid}\t{name.replace(chr(9), ' ')}\n")
        meta = dict(self.meta, n_items=self.n, dim=self.dim, n_queries=len(self.q_image_ids))
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, d: str, mmap: bool = True) -> "Dataset":
        mode = "r" if mmap else None
        classes = []
        with open(os.path.join(d, "classes.tsv")) as f:
            for line in f:
                i, mid, name = line.rstrip("\n").split("\t")
                classes.append((int(i), mid, name))
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        return cls(
            emb=np.load(os.path.join(d, "items/emb.npy"), mmap_mode=mode),
            labels=CSR.load(os.path.join(d, "items/labels")),
            author=np.load(os.path.join(d, "items/author.npy")),
            image_ids=_read_lines(os.path.join(d, "items/image_ids.txt")),
            q_emb=np.load(os.path.join(d, "queries/emb.npy")),
            q_pos=CSR.load(os.path.join(d, "queries/pos")),
            q_neg=CSR.load(os.path.join(d, "queries/neg")),
            q_image_ids=_read_lines(os.path.join(d, "queries/image_ids.txt")),
            classes=classes,
            meta=meta,
        )


def _write_lines(path: str, lines: list[str]) -> None:
    with open(path, "w") as f:
        for x in lines:
            f.write(x + "\n")


def _read_lines(path: str) -> list[str]:
    with open(path) as f:
        return [x.rstrip("\n") for x in f]
