"""oibench command line.

oibench export   --lake s3://.../oi --sample mirror_1m --queries queries --out data/m1
oibench workload --data data/m1 --out work/m1 [--n-per-cell 50]
oibench load     --data data/m1 --dsn postgresql://... --prefix m1
oibench run      --data data/m1 --workload work/m1 --sweep sweep.toml --out results/m1-a --dsn ...
oibench report   --workload work/m1 --results results/m1-a
"""

from __future__ import annotations

import argparse
import json
import os


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="oibench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    dsn_default = os.environ.get("OIBENCH_DSN", "postgresql://postgres:bench@localhost:5432/postgres")

    p = sub.add_parser("export")
    p.add_argument("--lake", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--queries", default="queries")
    p.add_argument("--model", default="pe-core-l14-336")
    p.add_argument("--out", required=True)
    p.add_argument("--no-expand", action="store_true", help="don't add hierarchy ancestors")
    p.add_argument(
        "--machine-threshold",
        type=float,
        help="also use machine labels with confidence >= this",
    )

    p = sub.add_parser("synth", help="synthetic dataset for testing the Postgres path")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=50_000)
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--labels", type=int, default=200)

    p = sub.add_parser("workload")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-per-cell", type=int, default=50)
    p.add_argument("--ks", type=int, nargs="+", default=[10, 100])
    p.add_argument("--group-sizes", type=int, nargs="+", default=[10, 100])
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("load")
    p.add_argument("--data", required=True)
    p.add_argument("--dsn", default=dsn_default)
    p.add_argument("--prefix", required=True)
    p.add_argument("--hnsw-m", type=int, default=16)
    p.add_argument("--hnsw-ef-construction", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)

    p = sub.add_parser("run")
    p.add_argument("--data", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--sweep", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dsn", default=dsn_default)
    p.add_argument("--methods", nargs="*", help="only these methods from the sweep")
    p.add_argument("--prefix", help="override the sweep's table prefix")
    p.add_argument("--limit", type=int, help="first N queries only (smoke test)")

    p = sub.add_parser("report")
    p.add_argument("--workload", required=True)
    p.add_argument("--results", required=True)

    p = sub.add_parser("plans", help="which plans each config actually used (from EXPLAIN)")
    p.add_argument("--workload", required=True)
    p.add_argument("--results", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "export":
        from oilake.lake import Lake

        from .export import export

        export(
            Lake.from_root(a.lake),
            a.out,
            sample=a.sample,
            queries=a.queries,
            model_id=a.model,
            expand=not a.no_expand,
            machine_threshold=a.machine_threshold,
        )
    elif a.cmd == "synth":
        from .synth import synthetic

        ds = synthetic(n=a.n, q=a.queries, d=a.dim, n_labels=a.labels, n_clusters=20)
        ds.save(a.out)
        print(f"synthetic {ds.n:,} x {ds.dim} -> {a.out}")
    elif a.cmd == "workload":
        from .data import Dataset
        from .workload import Spec, generate, write

        spec = Spec(ks=a.ks, n_per_cell=a.n_per_cell, group_sizes=a.group_sizes, seed=a.seed)
        wl, tr = generate(Dataset.load(a.data, mmap=False), spec)
        write(a.out, spec, wl, tr)
        print(f"{len(wl)} queries -> {a.out}")
    elif a.cmd == "load":
        from .data import Dataset
        from .pg import load

        stats = load(
            a.dsn,
            a.prefix,
            Dataset.load(a.data),
            hnsw_m=a.hnsw_m,
            hnsw_ef_construction=a.hnsw_ef_construction,
            parallel_workers=a.workers,
        )
        with open(os.path.join(a.data, f"load-{a.prefix}.json"), "w") as f:
            json.dump(stats, f, indent=2, default=str)
    elif a.cmd == "run":
        from .run import run

        run(a.dsn, a.data, a.workload, a.sweep, a.out, a.methods, a.limit, a.prefix)
    elif a.cmd == "report":
        from .report import write

        write(a.workload, a.results)
    elif a.cmd == "plans":
        from .report import plan_summary

        for cfg, kind, plan, n in plan_summary(a.workload, a.results):
            print(f"{n:5d}  {cfg:16s} {kind:28s} {plan}")


if __name__ == "__main__":
    main()
