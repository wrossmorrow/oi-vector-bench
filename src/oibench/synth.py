"""Synthetic dataset with the same shape as an export, for testing the
Postgres path before real embeddings exist.

Items sit in clusters, and label l is attached mostly to cluster l % n_clusters
with frequency decaying in l, so labels span several selectivity decades and
correlate with position in embedding space (pos/neg filters behave like the
real thing, roughly).
"""

from __future__ import annotations

import numpy as np

from .data import CSR, Dataset


def synthetic(n: int = 3000, q: int = 60, d: int = 32, n_labels: int = 40,
              n_clusters: int = 8, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, d))

    def points(m):
        c = rng.integers(0, n_clusters, size=m)
        x = centers[c] + 0.6 * rng.normal(size=(m, d))
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16), c

    emb, ci = points(n)
    q_emb, qc = points(q)

    def label_rows(clusters, neg=False):
        rows = []
        for c in clusters:
            ls = [l for l in range(n_labels)
                  if (l % n_clusters == c) == (not neg)
                  and rng.random() < 0.6 * 0.75 ** (l // n_clusters)]
            rows.append(sorted(ls))
        return rows

    labels = CSR.from_lists(label_rows(ci))
    q_pos = CSR.from_lists(label_rows(qc))
    q_neg = CSR.from_lists([sorted(set(r) - set(q_pos.row(i).tolist()))
                            for i, r in enumerate(label_rows(qc, neg=True))])
    author = rng.integers(0, max(1, n // 4), size=n)
    return Dataset(emb=emb, labels=labels, author=author,
                   image_ids=[f"i{i:08d}" for i in range(n)],
                   q_emb=q_emb, q_pos=q_pos, q_neg=q_neg,
                   q_image_ids=[f"q{i:06d}" for i in range(q)],
                   classes=[(l, f"/m/l{l}", f"label {l}") for l in range(n_labels)],
                   meta={"sample": "synthetic", "seed": seed})
