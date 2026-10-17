"""Run the workload against Postgres over a parameter sweep.

A sweep is a TOML file (see sweep.example.toml). Each *config* is one
method + one point in its parameter grid + one planner mode:

  mode "forced"   seqscan/bitmapscan/sort off: the ordered vector index scan is
                  the only plan left, so this measures the index itself.
                  Check explain.jsonl `nodes` to confirm (see `oibench plans`).
  mode "planner"  defaults: Postgres may pick an exact GIN bitmap scan + sort
                  for selective filters. This is what an application gets.

Query shapes per kind:

  filter        WHERE labels @> ARRAY[L] ORDER BY embedding <=> q LIMIT k
  group_label   "lateral":      one filtered top-n subquery per label
                "overfetch:M":  top-M unfiltered, then top-n per label in SQL
  diverse       "overfetch:M":  top-M unfiltered, then <= cap per author, top-k

Tables and indexes are pg_prewarm'ed once, then each query runs once per
config, timed client-side. Results stream to results.jsonl; EXPLAIN ANALYZE
plans for the first few queries of each cell go to explain.jsonl.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import sys
import time
import tomllib
from dataclasses import dataclass, field

import numpy as np

from .data import Dataset
from .pg import connect, table_names
from .workload import read_jsonl

MODE_GUCS = {
    # Sort off as well: otherwise the planner can walk the primary-key btree,
    # filter, and sort (an exact plan) instead of using the vector index. The
    # ordered vector index scan is the only plan that needs no sort. Window
    # queries still sort internally; PG 18 picks the plan with the fewest
    # disabled nodes, which keeps the vector scan underneath.
    "forced": {
        "enable_seqscan": "off",
        "enable_bitmapscan": "off",
        "enable_sort": "off",
    },
    "planner": {},
}


@dataclass
class Config:
    method: str
    table: str
    gucs: dict
    mode: str
    strategies: dict = field(default_factory=dict)
    kinds: frozenset = frozenset({"filter", "group_label", "diverse"})
    router: dict | None = None  # probe-then-route for filter queries; see run_routed

    @property
    def key(self) -> str:
        g = ",".join(f"{k}={v}" for k, v in sorted(self.gucs.items()))
        key = f"{self.method}|{self.mode}|{g}"
        if self.router:
            key += "|router=" + ",".join(f"{k}={v}" for k, v in sorted(self.router.items()))
        return key


def configs(sweep: dict) -> list[Config]:
    tables = table_names(sweep["prefix"])
    out = []
    default_modes = sweep.get("modes", ["forced", "planner"])
    default_kinds = sweep.get("kinds", ["filter", "group_label", "diverse"])
    for method, m in sweep["methods"].items():
        grid = m.get("grid", {})
        fixed = m.get("gucs", {})
        names = list(grid)
        for values in itertools.product(*(grid[n] for n in names)) if names else [()]:
            gucs = dict(fixed, **dict(zip(names, values, strict=True)))
            for mode in m.get("modes", default_modes):
                out.append(
                    Config(
                        method,
                        tables[m["table"]],
                        gucs,
                        mode,
                        sweep.get("strategies", {}),
                        frozenset(m.get("kinds", default_kinds)),
                        m.get("router"),
                    )
                )
    return out


def expected_rows(cfg: Config, items: list[dict]) -> int:
    """Result rows a complete run of `cfg` writes for these items."""
    return sum(len(strategies_for(w, cfg)) for w in items if w["kind"] in cfg.kinds)


def resume_state(out_dir: str, cfgs: list[Config], items: list[dict], known: list[Config] | None = None) -> set[str]:
    """Config keys of `cfgs` already complete in out_dir.

    Rows of incomplete configs (an interrupted run) are dropped from results.jsonl /
    explain.jsonl, so a rerun of the same command picks up where it stopped without
    duplicates. Completeness is a row count against each config's own expected rows.
    `known` is every config in the sweep file (not just the --methods being run), so a
    method with narrower `kinds` is judged by its own kinds. Rows of configs the sweep
    doesn't know are never dropped, only reported. The originals are backed up first."""
    res_path = os.path.join(out_dir, "results.jsonl")
    if not os.path.exists(res_path):
        return set()
    counts: dict = {}
    with open(res_path) as f:
        for line in f:
            k = json.loads(line)["config"]
            counts[k] = counts.get(k, 0) + 1
    want = {c.key: expected_rows(c, items) for c in (known or []) + cfgs}
    unknown = sorted(k for k in counts if k not in want)
    if unknown:
        print(f"keeping rows of {len(unknown)} config(s) not in this sweep: {unknown}", file=sys.stderr)
    done = {k for k, n in counts.items() if k in want and n >= want[k]}
    partial = {k for k in counts if k in want and k not in done}
    if partial:
        print(
            f"dropping partial rows of {len(partial)} interrupted config(s): {sorted(partial)}",
            file=sys.stderr,
        )
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for name in ("results.jsonl", "explain.jsonl"):
            path = os.path.join(out_dir, name)
            if not os.path.exists(path):
                continue
            shutil.copyfile(path, f"{path}.bak-{stamp}")
            with open(path) as f:
                keep = [line for line in f if json.loads(line)["config"] not in partial]
            with open(path + ".tmp", "w") as f:
                f.writelines(keep)
            os.replace(path + ".tmp", path)
    return done & {c.key for c in cfgs}


# --------------------------------------------------------------------------- SQL


def sql_for(item: dict, table: str, strategy: str | None) -> tuple[str, dict]:
    kind = item["kind"]
    if kind == "filter":
        return (
            f"SELECT id FROM {table} WHERE labels @> ARRAY[%(l)s]::int4[] "
            f"ORDER BY embedding <=> %(q)s LIMIT {int(item['k'])}",
            {"l": item["labels"][0]},
        )
    if kind == "group_label":
        n = int(item["k"])
        if strategy == "lateral":
            return (
                f"SELECT g.l, x.id FROM unnest(%(ls)s::int4[]) AS g(l) CROSS JOIN LATERAL "
                f"(SELECT id FROM {table} WHERE labels @> ARRAY[g.l] "
                f"ORDER BY embedding <=> %(q)s LIMIT {n}) x",
                {"ls": item["labels"]},
            )
        m = _overfetch(strategy)
        return (
            f"SELECT l, id FROM (SELECT u.l, c.id, row_number() OVER "
            f"(PARTITION BY u.l ORDER BY c.d) AS rn FROM "
            f"(SELECT id, labels, embedding <=> %(q)s AS d FROM {table} "
            f"ORDER BY embedding <=> %(q)s LIMIT {m}) c "
            f"CROSS JOIN LATERAL unnest(c.labels) AS u(l) "
            f"WHERE u.l = ANY(%(ls)s::int4[])) s WHERE rn <= {n}",
            {"ls": item["labels"]},
        )
    if kind == "diverse":
        m = _overfetch(strategy)
        return (
            f"SELECT id FROM (SELECT id, d, row_number() OVER "
            f"(PARTITION BY author_id ORDER BY d) AS rn FROM "
            f"(SELECT id, author_id, embedding <=> %(q)s AS d FROM {table} "
            f"ORDER BY embedding <=> %(q)s LIMIT {m}) c) s "
            f"WHERE rn <= {int(item['cap'])} ORDER BY d LIMIT {int(item['k'])}",
            {},
        )
    raise ValueError(kind)


def _overfetch(strategy: str | None) -> int:
    if not strategy or not strategy.startswith("overfetch:"):
        raise ValueError(f"bad strategy {strategy!r}")
    return int(strategy.split(":", 1)[1])


def strategies_for(item: dict, cfg: Config) -> list[str | None]:
    if item["kind"] == "filter":
        return [None]
    return list(cfg.strategies.get(item["kind"], []))


def cell_of(item: dict) -> tuple:
    return (
        item["kind"],
        item.get("ftype"),
        item.get("bin"),
        item["k"],
        item.get("G"),
        item.get("cap"),
    )


# --------------------------------------------------------------------------- explain helpers


def plan_nodes(plan: dict) -> list[str]:
    """Flatten a JSON plan to 'Node Type[:Index Name]' strings."""
    out = []

    def walk(p) -> None:
        s = p.get("Node Type", "?")
        if "Index Name" in p:
            s += ":" + p["Index Name"]
        out.append(s)
        for c in p.get("Plans", []):
            walk(c)

    walk(plan["Plan"] if "Plan" in plan else plan)
    return out


# --------------------------------------------------------------------------- routing


def probe_sql(table: str, n: int) -> str:
    """How many of the query's n nearest neighbours (unfiltered, via the vector
    index) pass the filter, and how many rows the index returned at all (HNSW
    returns at most ef_search)."""
    return (
        "SELECT count(*) FILTER (WHERE labels @> ARRAY[%(l)s]::int4[]), count(*) "
        f"FROM (SELECT labels FROM {table} ORDER BY embedding <=> %(q)s LIMIT {int(n)}) p"
    )


def route_decision(matches: int, k: int, factor: float = 1.0) -> str:
    """'index' if the probe already saw enough matching rows near the query for a
    post-filtered index scan to fill k; otherwise 'exact' (label index + sort).

    This measures the filter's lift around this particular query, which the
    planner can't see: it only knows the label's overall selectivity."""
    return "index" if matches >= factor * k else "exact"


def run_routed(conn, cfg: Config, item: dict, params: dict) -> tuple[list, dict]:
    """Probe, decide, then run the filter query on the chosen plan. The caller
    times the whole thing, so latency includes the probe."""
    n = int(cfg.router.get("probe", 200))
    factor = float(cfg.router.get("factor", 1.0))
    matches, returned = conn.execute(probe_sql(cfg.table, n), params).fetchone()
    route = route_decision(matches, int(item["k"]), factor)
    sql, _ = sql_for(item, cfg.table, None)
    if route == "exact":
        with conn.transaction():  # SET LOCAL reverts at commit
            conn.execute("SET LOCAL enable_indexscan = off")
            rows = conn.execute(sql, params).fetchall()
    else:
        rows = conn.execute(sql, params).fetchall()
    return rows, {"route": route, "probe_matches": matches, "probe_rows": returned}


# --------------------------------------------------------------------------- run


def prewarm(conn, prefix: str) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
    for t in table_names(prefix).values():
        rels = [
            r[0]
            for r in conn.execute(
                "SELECT indexrelid::regclass::text FROM pg_index WHERE indrelid = %s::regclass",
                (t,),
            )
        ]
        for rel in [t] + rels:
            t0 = time.time()
            n = conn.execute("SELECT pg_prewarm(%s::regclass)", (rel,)).fetchone()[0]
            print(f"prewarm {rel}: {n:,} blocks, {time.time() - t0:.1f}s", file=sys.stderr)


def run(
    dsn: str,
    data_dir: str,
    workload_dir: str,
    sweep_path: str,
    out_dir: str,
    only_methods: list[str] | None = None,
    limit: int | None = None,
    prefix: str | None = None,
) -> None:
    from pgvector import HalfVector

    with open(sweep_path, "rb") as f:
        sweep = tomllib.load(f)
    if prefix:
        sweep["prefix"] = prefix
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(sweep_path, os.path.join(out_dir, "sweep.toml"))

    ds = Dataset.load(data_dir)
    items = read_jsonl(os.path.join(workload_dir, "workload.jsonl"))
    kinds = set(sweep.get("kinds", ["filter", "group_label", "diverse"]))
    items = [w for w in items if w["kind"] in kinds]
    if limit:
        items = items[:limit]
    qvec = {}

    conn = connect(dsn)
    # No server-side prepared statements. psycopg prepares a query after 5
    # executions, and Postgres keeps a prepared statement's cached plan even
    # after enable_* / index GUCs change, so the first config's plan would
    # silently be reused by every later config with the same SQL text. Each
    # query is planned fresh under the current config instead (planning time
    # is included in latency, equally for every method).
    conn.prepare_threshold = None
    prewarm(conn, sweep["prefix"])
    explain_n = int(sweep.get("explain_per_cell", 3))
    timeout = sweep.get("statement_timeout", "60s")

    all_cfgs = configs(sweep)
    cfgs = [c for c in all_cfgs if not only_methods or c.method in only_methods]
    done = resume_state(out_dir, cfgs, items, known=all_cfgs)
    if done:
        print(
            f"resuming: {len(done)} config(s) already complete, skipping them",
            file=sys.stderr,
        )
    print(
        f"{len(cfgs) - len(done)} configs to run x {len(items)} queries",
        file=sys.stderr,
    )
    res_f = open(os.path.join(out_dir, "results.jsonl"), "a")
    exp_f = open(os.path.join(out_dir, "explain.jsonl"), "a")

    for ci, cfg in enumerate(cfgs):
        if cfg.key in done:
            continue
        conn.execute("RESET ALL")
        conn.execute("DEALLOCATE ALL")  # belt and braces: no plan survives a config change
        conn.execute(f"SET statement_timeout = '{timeout}'")
        for k, v in {**MODE_GUCS[cfg.mode], **cfg.gucs}.items():
            conn.execute(f"SET {k} = '{v}'")
        t_cfg = time.time()
        explained: dict = {}
        n_err = 0
        for item in items:
            if item["kind"] not in cfg.kinds:
                continue
            q = item["q"]
            if q not in qvec:
                qvec[q] = HalfVector(np.asarray(ds.q_emb[q]))
            for strategy in strategies_for(item, cfg):
                sql, params = sql_for(item, cfg.table, strategy)
                params["q"] = qvec[q]
                row = {
                    "config": cfg.key,
                    "method": cfg.method,
                    "mode": cfg.mode,
                    "gucs": cfg.gucs,
                    "strategy": strategy,
                    "qid": item["qid"],
                }
                t0 = time.perf_counter()
                try:
                    if cfg.router and item["kind"] == "filter":
                        rows, info = run_routed(conn, cfg, item, params)
                        row.update(info)
                    else:
                        rows = conn.execute(sql, params).fetchall()
                    row["latency_ms"] = (time.perf_counter() - t0) * 1000
                    if item["kind"] == "group_label":
                        groups: dict = {}
                        for label, rid in rows:
                            groups.setdefault(str(label), []).append(rid)
                        row["groups"] = groups
                    else:
                        row["ids"] = [r[0] for r in rows]
                except Exception as e:  # timeouts etc.: record and move on
                    row["latency_ms"] = (time.perf_counter() - t0) * 1000
                    row["error"] = f"{type(e).__name__}: {e}"[:300]
                    n_err += 1  # autocommit: a failed statement leaves no open transaction
                res_f.write(json.dumps(row) + "\n")

                ckey = (cell_of(item), strategy)
                # routed queries are several statements; EXPLAIN of one would mislead
                if explained.get(ckey, 0) < explain_n and "error" not in row and not cfg.router:
                    explained[ckey] = explained.get(ckey, 0) + 1
                    try:
                        plan = conn.execute(
                            "EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON) " + sql,
                            params,
                        ).fetchone()[0][0]
                        exp_f.write(
                            json.dumps(
                                {
                                    "config": cfg.key,
                                    "strategy": strategy,
                                    "qid": item["qid"],
                                    "nodes": plan_nodes(plan),
                                    "plan": plan,
                                }
                            )
                            + "\n"
                        )
                    except Exception as e:
                        print(f"  explain failed: {e}", file=sys.stderr)
        res_f.flush()
        exp_f.flush()
        print(
            f"[{ci + 1}/{len(cfgs)}] {cfg.key}: {time.time() - t_cfg:.0f}s, {n_err} errors",
            file=sys.stderr,
        )
    res_f.close()
    exp_f.close()
