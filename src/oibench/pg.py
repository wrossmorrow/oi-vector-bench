"""Postgres schema, bulk load, and index builds.

Two identical tables, one index type each, so the planner can never pick the
"wrong" vector index and no hinting extension is needed:

  <p>_hnsw   pgvector HNSW   (halfvec_cosine_ops)
  <p>_prism  pg_vectorsearch (vec16_cosine_ops on the same halfvec column)

Both get a GIN index on labels, which is what lets the planner choose an
exact bitmap-scan + sort plan for selective filters (measured as "planner"
mode in run.py).

halfvec(1024) is 2,052 bytes: just over the ~2 kB TOAST threshold, so the
column is STORAGE PLAIN to keep vectors inline (3 rows per 8 kB page) and
make reranks one heap fetch.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from .data import Dataset


def connect(dsn: str):
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def table_names(prefix: str) -> dict[str, str]:
    return {"hnsw": f"{prefix}_hnsw", "prism": f"{prefix}_prism"}


def create_tables(conn, prefix: str, dim: int) -> None:
    t = table_names(prefix)
    for name in t.values():
        conn.execute(f"DROP TABLE IF EXISTS {name}")
    conn.execute(f"""
        CREATE TABLE {t['hnsw']} (
            id        bigint PRIMARY KEY,
            image_id  text NOT NULL,
            labels    int4[] NOT NULL,
            author_id bigint NOT NULL,
            embedding halfvec({dim}) STORAGE PLAIN NOT NULL
        )""")
    conn.execute(f"CREATE TABLE {t['prism']} (LIKE {t['hnsw']} INCLUDING ALL)")


def copy_items(conn, prefix: str, ds: Dataset, batch_log: int = 100_000) -> float:
    """Binary COPY into the hnsw table, then server-side copy to the prism table."""
    from pgvector import HalfVector

    t = table_names(prefix)
    t0 = time.time()
    with conn.cursor() as cur:
        with cur.copy(f"COPY {t['hnsw']} (id, image_id, labels, author_id, embedding) "
                      "FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int8", "text", "int4[]", "int8", "halfvec"])
            for i in range(ds.n):
                copy.write_row((i, ds.image_ids[i], ds.labels.row(i).tolist(),
                                int(ds.author[i]), HalfVector(np.asarray(ds.emb[i]))))
                if (i + 1) % batch_log == 0:
                    print(f"  copied {i + 1:,}/{ds.n:,}", file=sys.stderr)
    conn.execute(f"INSERT INTO {t['prism']} SELECT * FROM {t['hnsw']}")
    return time.time() - t0


def _timed(conn, sql: str) -> float:
    print(f"  {sql}", file=sys.stderr)
    t0 = time.time()
    conn.execute(sql)
    return time.time() - t0


def build_indexes(conn, prefix: str, *, hnsw_m: int = 16, hnsw_ef_construction: int = 64,
                  prism_options: dict | None = None, maintenance_work_mem: str = "16GB",
                  parallel_workers: int = 8) -> dict:
    t = table_names(prefix)
    conn.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}'")
    conn.execute(f"SET max_parallel_maintenance_workers = {parallel_workers}")
    stats: dict = {}
    for name in t.values():
        stats[f"{name}.gin_s"] = _timed(conn, f"CREATE INDEX ON {name} USING gin (labels)")
        conn.execute(f"ANALYZE {name}")

    stats["hnsw.build_s"] = _timed(
        conn, f"CREATE INDEX {t['hnsw']}_vec ON {t['hnsw']} USING hnsw "
              f"(embedding halfvec_cosine_ops) WITH (m = {hnsw_m}, "
              f"ef_construction = {hnsw_ef_construction})")

    conn.execute("CREATE EXTENSION IF NOT EXISTS pg_vectorsearch")
    with_ = ""
    if prism_options:
        with_ = " WITH (" + ", ".join(f"{k} = {v}" for k, v in prism_options.items()) + ")"
    stats["prism.build_s"] = _timed(
        conn, f"CREATE INDEX {t['prism']}_vec ON {t['prism']} USING prism "
              f"(embedding vec16_cosine_ops){with_}")

    for idx in (f"{t['hnsw']}_vec", f"{t['prism']}_vec"):
        stats[f"{idx}.bytes"] = conn.execute(
            "SELECT pg_relation_size(%s::regclass)", (idx,)).fetchone()[0]
    for name in t.values():
        stats[f"{name}.heap_bytes"] = conn.execute(
            "SELECT pg_relation_size(%s::regclass)", (name,)).fetchone()[0]
    stats["prism.settings"] = {r[0]: [r[1], r[2]] for r in conn.execute(
        "SELECT * FROM prism_index_settings(%s::regclass)", (f"{t['prism']}_vec",))}
    stats["versions"] = {r[0]: r[1] for r in conn.execute(
        "SELECT extname, extversion FROM pg_extension "
        "WHERE extname IN ('vector', 'pg_vectorsearch')")}
    stats["versions"]["postgres"] = conn.execute("SHOW server_version").fetchone()[0]
    return stats


def load(dsn: str, prefix: str, ds: Dataset, **index_kw) -> dict:
    conn = connect(dsn)
    create_tables(conn, prefix, ds.dim)
    stats = {"prefix": prefix, "n_items": ds.n, "dim": ds.dim, "meta": ds.meta}
    stats["copy_s"] = copy_items(conn, prefix, ds)
    stats.update(build_indexes(conn, prefix, **index_kw))
    for name in table_names(prefix).values():
        conn.execute(f"VACUUM (ANALYZE) {name}")
    print(json.dumps(stats, indent=2, default=str), file=sys.stderr)
    return stats
