"""Score raw results against ground truth and summarize.

Per query:
  recall   filter/diverse: |returned ∩ truth| / |truth|
           group_label:    mean over groups of the same, per group
  short    returned fewer rows than the truth has (per group for group_label:
           the fraction of groups that came back short)

Summary rows group by config x strategy x cell and report n, recall mean and
10th percentile, short fraction, error fraction, latency p50 / p95.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .workload import read_jsonl

CELL = ["kind", "ftype", "bin", "k", "G", "cap"]
CONFIG = ["method", "mode", "gucs_s", "strategy"]


def score(item: dict, truth: dict, res: dict) -> tuple[float, float]:
    if "error" in res:
        return 0.0, 1.0
    if item["kind"] == "group_label":
        rec, short = [], []
        got = res.get("groups", {})
        for g, ids in truth["groups"].items():
            r = got.get(g, [])
            if not ids:
                continue
            rec.append(len(set(r) & set(ids)) / len(ids))
            short.append(len(r) < len(ids))
        return float(np.mean(rec)), float(np.mean(short))
    ids = truth["ids"]
    r = res.get("ids", [])
    return len(set(r) & set(ids)) / len(ids), float(len(r) < len(ids))


def summarize(workload_dir: str, results_dir: str):
    import pandas as pd

    items = {w["qid"]: w for w in read_jsonl(os.path.join(workload_dir, "workload.jsonl"))}
    truth = {t["qid"]: t for t in read_jsonl(os.path.join(workload_dir, "truth.jsonl"))}
    rows = []
    with open(os.path.join(results_dir, "results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            w = items[r["qid"]]
            rec, short = score(w, truth[r["qid"]], r)
            rows.append(
                {
                    "method": r["method"],
                    "mode": r["mode"],
                    "gucs_s": ",".join(f"{k}={v}" for k, v in sorted(r["gucs"].items())) or "-",
                    "strategy": r["strategy"] or "-",
                    **{c: w.get(c) for c in CELL},
                    "selectivity": w.get("selectivity"),
                    "lift": w.get("lift"),
                    "qid": r["qid"],
                    "recall": rec,
                    "short": short,
                    "error": "error" in r,
                    "latency_ms": r["latency_ms"],
                    "route": r.get("route"),  # routed methods only: "index" | "exact"
                }
            )
    df = pd.DataFrame(rows)
    df[CELL] = df[CELL].astype(object).where(df[CELL].notna(), "-")
    agg = (
        df.groupby(CONFIG + CELL, dropna=False)
        .agg(
            n=("recall", "size"),
            recall=("recall", "mean"),
            recall_p10=("recall", lambda s: s.quantile(0.10)),
            short=("short", "mean"),
            errors=("error", "mean"),
            lat_p50=("latency_ms", "median"),
            lat_p95=("latency_ms", lambda s: s.quantile(0.95)),
            routed_exact=(
                "route",
                lambda s: (s == "exact").mean() if s.notna().any() else float("nan"),
            ),
        )
        .reset_index()
    )
    return df, agg


_PLAN_KEEP = ("Scan", "Sort", "Limit")


def compact_plan(nodes: list[str]) -> str:
    """Keep the nodes that say how a query was answered: scans, sorts, limits."""
    return " > ".join(n for n in nodes if any(k in n.split(":")[0] for k in _PLAN_KEEP))


def plan_summary(workload_dir: str, results_dir: str) -> list[tuple[str, str, str, int]]:
    """(method|mode, kind/strategy, compact plan, count) from explain.jsonl."""
    from collections import Counter

    kinds = {w["qid"]: w["kind"] for w in read_jsonl(os.path.join(workload_dir, "workload.jsonl"))}
    c: Counter = Counter()
    with open(os.path.join(results_dir, "explain.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            method, mode = r["config"].split("|")[:2]
            kind = kinds[r["qid"]] + (f"/{r['strategy']}" if r.get("strategy") else "")
            c[(f"{method}|{mode}", kind, compact_plan(r["nodes"]))] += 1
    return sorted((a, b, p, n) for (a, b, p), n in c.items())


def filter_pivot(agg, k: int = 10, value: str = "recall"):
    f = agg[(agg.kind == "filter") & (agg.k == k)]
    return f.pivot_table(index=["method", "mode", "gucs_s"], columns=["ftype", "bin"], values=value).round(3)


def write(workload_dir: str, results_dir: str) -> None:
    df, agg = summarize(workload_dir, results_dir)
    df.to_csv(os.path.join(results_dir, "scored.csv"), index=False)
    agg.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
    with open(os.path.join(results_dir, "report.md"), "w") as f:
        for k in sorted(agg[agg.kind == "filter"].k.unique()):
            for value in ("recall", "short", "lat_p50"):
                f.write(f"## filter {value}, k={k}\n\n")
                f.write(filter_pivot(agg, int(k), value).to_markdown() + "\n\n")
        for kind in ("group_label", "diverse"):
            sub = agg[agg.kind == kind]
            if len(sub):
                f.write(f"## {kind}\n\n")
                cols = (
                    CONFIG
                    + [c for c in CELL if c != "kind"]
                    + ["n", "recall", "recall_p10", "short", "lat_p50", "lat_p95"]
                )
                f.write(sub[cols].round(3).to_markdown(index=False) + "\n\n")
    print(f"wrote scored.csv, summary.csv, report.md in {results_dir}")
