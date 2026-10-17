"""Generate the query workload and its exact answers.

Three kinds of query, each from a held-out query image q:

  filter       top-k among items with label L.
               ftype says how L relates to q:
                 pos   q has L                       (filter correlated with the query)
                 rand  q neither has nor lacks L     (roughly independent)
                 neg   q is human-verified NOT L     (anti-correlated)
               Each also records `lift`: among q's unfiltered top-T neighbors,
               the fraction carrying L, divided by L's overall selectivity.
               lift >> 1 means L's items sit near q; lift << 1 means they are
               far away, which is where post-filtering index scans run short.
  group_label  top-n per label for G labels drawn from one selectivity bin.
  diverse      top-k with at most `cap` items per author.

Selectivity bins are half-decades: (10^-0.5, 1], (10^-1, 10^-0.5], ...
Every cell (kind, ftype, bin, k, G/cap) gets up to n_per_cell queries.

Output: workload.jsonl (one query per line) and truth.jsonl (same qids).
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field

import numpy as np

from .data import Dataset
from .truth import Exact


@dataclass
class Spec:
    ks: list[int] = field(default_factory=lambda: [10, 100])
    n_per_cell: int = 50
    log10_top: float = 0.0  # bins from 10^0 ...
    log10_bottom: float = -5.0  # ... down to 10^-5
    bin_step: float = 0.5
    ftypes: list[str] = field(default_factory=lambda: ["pos", "rand", "neg"])
    group_sizes: list[int] = field(default_factory=lambda: [10, 100])
    group_n: int = 3
    diverse_caps: list[int] = field(default_factory=lambda: [1, 2])
    lift_top: int = 1000  # T for lift; also the diverse candidate pool
    seed: int = 0


def bins(spec: Spec) -> list[tuple[float, float]]:
    """[(lo, hi)] selectivity intervals, lo exclusive, hi inclusive."""
    n = round((spec.log10_top - spec.log10_bottom) / spec.bin_step)
    edges = [spec.log10_top - i * spec.bin_step for i in range(n + 1)]
    return [(10 ** edges[i + 1], 10 ** edges[i]) for i in range(n)]


def bin_name(b: tuple[float, float]) -> str:
    return f"{b[0]:.0e}-{b[1]:.0e}"


def generate(ds: Dataset, spec: Spec, log=lambda *a: print(*a, file=sys.stderr)):
    rng = np.random.default_rng(spec.seed)
    n_labels = ds.n_labels
    inv = ds.labels.inverted(n_labels)
    counts = np.diff(inv.indptr)
    sel = counts / ds.n
    exact = Exact(ds.emb)
    Q = len(ds.q_image_ids)

    log(f"unfiltered top-{spec.lift_top} for {Q} queries")
    top_t = exact.topk_batch(ds.q_emb, spec.lift_top)

    def items_of(label: int) -> np.ndarray:
        return inv.indices[inv.indptr[label] : inv.indptr[label + 1]]

    def lift(qi: int, label: int) -> float:
        frac = np.isin(top_t[qi], items_of(label), assume_unique=True).mean()
        return float(frac / sel[label]) if sel[label] > 0 else math.nan

    workload: list[dict] = []
    truth: list[dict] = []

    def emit(item: dict, answer: dict) -> None:
        item["qid"] = len(workload)
        answer["qid"] = item["qid"]
        workload.append(item)
        truth.append(answer)

    for b in bins(spec):
        in_bin = (sel > b[0]) & (sel <= b[1]) & (counts > 0)
        bname = bin_name(b)

        # ---- filter queries: pick (q, L) per ftype, answer once at kmax, slice per k
        for ftype in spec.ftypes:
            for k in spec.ks:
                eligible = np.flatnonzero(in_bin & (counts >= k))
                if len(eligible) == 0:
                    continue
                elig_set = set(eligible.tolist())
                picked = 0
                for qi in rng.permutation(Q):
                    pos, neg = ds.q_pos.row(qi), ds.q_neg.row(qi)
                    if ftype == "pos":
                        cand = [x for x in pos.tolist() if x in elig_set]
                    elif ftype == "neg":
                        pos_set = set(pos.tolist())
                        cand = [x for x in neg.tolist() if x in elig_set and x not in pos_set]
                    else:
                        excluded = set(pos.tolist()) | set(neg.tolist())
                        cand = []
                        for _ in range(20):
                            x = int(rng.choice(eligible))
                            if x not in excluded:
                                cand = [x]
                                break
                    if not cand:
                        continue
                    label = int(rng.choice(cand))
                    ids, dists = exact.filtered(ds.q_emb[qi], k, items_of(label))
                    emit(
                        {
                            "kind": "filter",
                            "ftype": ftype,
                            "bin": bname,
                            "k": k,
                            "q": int(qi),
                            "labels": [label],
                            "selectivity": float(sel[label]),
                            "lift": lift(qi, label),
                        },
                        {"ids": ids.tolist(), "dists": dists.tolist()},
                    )
                    picked += 1
                    if picked >= spec.n_per_cell:
                        break
                log(f"filter {ftype:4s} {bname} k={k}: {picked}")

        # ---- grouped queries: G labels from this bin, top-n each
        eligible = np.flatnonzero(in_bin & (counts >= spec.group_n))
        for G in spec.group_sizes:
            if len(eligible) < G:
                continue
            for qi in rng.choice(Q, size=min(spec.n_per_cell, Q), replace=False):
                labels = sorted(int(x) for x in rng.choice(eligible, size=G, replace=False))
                groups = {L: items_of(L) for L in labels}
                emit(
                    {
                        "kind": "group_label",
                        "bin": bname,
                        "k": spec.group_n,
                        "G": G,
                        "q": int(qi),
                        "labels": labels,
                        "selectivity": float(np.mean(sel[labels])),
                    },
                    {"groups": {str(g): v for g, v in exact.grouped(ds.q_emb[qi], spec.group_n, groups).items()}},
                )
            log(f"group_label {bname} G={G}: {min(spec.n_per_cell, Q)}")

    # ---- diversity queries
    fallbacks = 0
    for cap in spec.diverse_caps:
        for k in spec.ks:
            for qi in rng.choice(Q, size=min(spec.n_per_cell, Q), replace=False):
                ids, fb = exact.diverse(ds.q_emb[qi], k, cap, ds.author, top_t[qi])
                fallbacks += fb
                emit(
                    {"kind": "diverse", "k": k, "cap": cap, "q": int(qi), "labels": []},
                    {"ids": ids},
                )
    log(f"diverse: {len(spec.diverse_caps) * len(spec.ks)} cells, {fallbacks} full-scan fallbacks")
    return workload, truth


def write(out_dir: str, spec: Spec, workload: list[dict], truth: list[dict]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "workload.jsonl"), "w") as f:
        for w in workload:
            f.write(json.dumps(w) + "\n")
    with open(os.path.join(out_dir, "truth.jsonl"), "w") as f:
        for t in truth:
            f.write(json.dumps(t) + "\n")
    with open(os.path.join(out_dir, "spec.json"), "w") as f:
        json.dump(asdict(spec), f, indent=2)


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]
