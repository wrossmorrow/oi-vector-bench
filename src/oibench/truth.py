"""Exact (brute-force) answers, using cosine distance as pgvector defines it.

Embeddings arrive as float16 (what Postgres stores in halfvec) and are
widened to float32 once; queries are rounded through float16 too, since the
SQL casts them `::halfvec`. Distances are 1 - cos(a, b) with both sides
renormalized, matching pgvector's <=> on non-unit (fp16-rounded) vectors.
"""

from __future__ import annotations

import numpy as np

_CHUNK = 1 << 16


def as_query(q: np.ndarray) -> np.ndarray:
    """Round a query the way `::halfvec` does, then unit-normalize."""
    q = np.asarray(q, dtype=np.float16).astype(np.float32)
    return q / np.linalg.norm(q)


class Exact:
    def __init__(self, emb16: np.ndarray):
        e = np.asarray(emb16, dtype=np.float32)
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.emb = e / norms  # unit rows of the fp16-rounded vectors
        self.n = e.shape[0]

    # ------------------------------------------------------------- primitives

    def sims_subset(self, q: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """cos(q, item) for items `idx`, gathered in chunks to bound memory."""
        out = np.empty(len(idx), dtype=np.float32)
        for s in range(0, len(idx), _CHUNK):
            out[s : s + _CHUNK] = self.emb[idx[s : s + _CHUNK]] @ q
        return out

    @staticmethod
    def _top(ids: np.ndarray, sims: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(ids))
        if k == 0:
            return ids[:0], sims[:0].astype(np.float32)
        part = np.argpartition(-sims, k - 1)[:k] if k < len(ids) else np.arange(len(ids))
        order = part[np.lexsort((ids[part], -sims[part]))]  # by distance, then id
        return ids[order], (1.0 - sims[order]).astype(np.float32)

    def topk(self, q: np.ndarray, k: int, idx: np.ndarray | None = None):
        """Exact top-k by cosine distance, over all items or the subset `idx`.

        Returns (ids, distances), nearest first.
        """
        q = as_query(q)
        if idx is None:
            sims = self.sims_all(q[None, :])[0]
            ids = np.arange(self.n)
        else:
            idx = np.asarray(idx, dtype=np.int64)
            sims = self.sims_subset(q, idx)
            ids = idx
        return self._top(ids, sims, k)

    def sims_all(self, Q: np.ndarray) -> np.ndarray:
        """(B, N) similarities for already-normalized queries. Use for small B."""
        return (self.emb @ Q.T).T

    def topk_batch(self, Q: np.ndarray, k: int, batch: int = 256) -> np.ndarray:
        """Unfiltered top-k ids for many queries: (Qn, k) int64, nearest first."""
        Qn = np.stack([as_query(q) for q in Q])
        out = np.empty((len(Qn), k), dtype=np.int64)
        for b in range(0, len(Qn), batch):
            qb = Qn[b : b + batch]
            best_ids = np.zeros((len(qb), 0), dtype=np.int64)
            best_sims = np.zeros((len(qb), 0), dtype=np.float32)
            for s in range(0, self.n, _CHUNK * 2):
                block = self.emb[s : s + _CHUNK * 2] @ qb.T  # (chunk, B)
                ids = np.arange(s, s + block.shape[0])
                kk = min(k, block.shape[0])
                p = np.argpartition(-block, kk - 1, axis=0)[:kk].T  # (B, kk)
                cand_ids = np.concatenate([best_ids, ids[p]], axis=1)
                cand_sims = np.concatenate([best_sims, np.take_along_axis(block.T, p, axis=1)], axis=1)
                kk2 = min(k, cand_ids.shape[1])
                p2 = np.argpartition(-cand_sims, kk2 - 1, axis=1)[:, :kk2]
                best_ids = np.take_along_axis(cand_ids, p2, axis=1)
                best_sims = np.take_along_axis(cand_sims, p2, axis=1)
            order = np.argsort(-best_sims, axis=1, kind="stable")
            out[b : b + len(qb)] = np.take_along_axis(best_ids, order, axis=1)
        return out

    # ------------------------------------------------------------- query types

    def filtered(self, q, k: int, label_items: np.ndarray):
        """Top-k among items carrying a label."""
        return self.topk(q, k, label_items)

    def grouped(self, q, n: int, groups: dict[int, np.ndarray]) -> dict[int, list[int]]:
        """Top-n per group (label -> its items)."""
        return {g: self.topk(q, n, items)[0].tolist() for g, items in groups.items()}

    def diverse(
        self,
        q,
        k: int,
        cap: int,
        author: np.ndarray,
        candidates: np.ndarray | None = None,
    ) -> tuple[list[int], bool]:
        """Top-k with at most `cap` items per author, greedy in distance order.

        `candidates` (nearest first) avoids a full scan; if it runs out before
        k picks, falls back to the full ordering. Returns (ids, used_fallback).
        """

        def greedy(order):
            seen: dict[int, int] = {}
            picked = []
            for i in order:
                a = int(author[i])
                if seen.get(a, 0) < cap:
                    seen[a] = seen.get(a, 0) + 1
                    picked.append(int(i))
                    if len(picked) == k:
                        break
            return picked

        if candidates is not None:
            picked = greedy(candidates)
            if len(picked) == k:
                return picked, False
        ids, _ = self.topk(q, self.n)
        return greedy(ids), True
