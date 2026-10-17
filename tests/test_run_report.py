"""Sweep expansion, SQL shapes, plan flattening, scoring (no Postgres needed)."""

import json
import os
import sys
import tomllib
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oibench.report import score  # noqa: E402
from oibench.run import cell_of, configs, plan_nodes, sql_for, strategies_for  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


class TestSweep(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, "sweep.example.toml"), "rb") as f:
            self.sweep = tomllib.load(f)

    def test_example_expands(self):
        cfgs = configs(self.sweep)
        by = {}
        for c in cfgs:
            by.setdefault(c.method, []).append(c)
        self.assertEqual(len(by["exact"]), 1)  # planner mode only
        self.assertEqual(len(by["hnsw"]), 3 * 2)  # ef_search x modes
        self.assertEqual(len(by["hnsw_iter"]), 1 * 2 * 2 * 2)  # ef x iter x max_tuples x modes
        self.assertTrue(all(c.kinds == frozenset({"filter", "diverse"}) for c in by["hnsw_iter"]))
        self.assertEqual(len(by["prism"]), 4 * 2)
        self.assertTrue(all(c.table == "m1_prism" for c in by["prism"]))
        self.assertEqual(by["exact"][0].gucs, {"enable_indexscan": "off"})
        self.assertEqual(len({c.key for c in cfgs}), len(cfgs))  # keys unique

    def test_strategies(self):
        c = configs(self.sweep)[0]
        self.assertEqual(strategies_for({"kind": "filter"}, c), [None])
        self.assertEqual(
            strategies_for({"kind": "group_label"}, c),
            ["lateral", "overfetch:1000", "overfetch:10000"],
        )


class TestResume(unittest.TestCase):
    def _setup(self, d):
        import json

        from oibench.run import Config

        items = [
            {"qid": 0, "kind": "filter"},
            {"qid": 1, "kind": "group_label"},
            {"qid": 2, "kind": "diverse"},
        ]
        strat = {"group_label": ["lateral", "overfetch:10"], "diverse": ["overfetch:5"]}
        full = Config("a", "t", {"x": 1}, "forced", strat)  # 1 + 2 + 1 = 4 rows
        partial = Config("b", "t", {"x": 2}, "forced", strat)
        narrow = Config("c", "t", {}, "forced", strat, frozenset({"filter"}))  # 1 row
        rows = (
            [{"config": full.key}] * 4
            + [{"config": partial.key}] * 2
            + [{"config": "old|forced|y=1"}] * 4
            + [{"config": "old|forced|y=2"}] * 2
            + [{"config": narrow.key}]
        )
        with open(os.path.join(d, "results.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        with open(os.path.join(d, "explain.jsonl"), "w") as f:
            f.writelines(json.dumps({"config": k}) + "\n" for k in (full.key, partial.key))
        return items, [full, partial, narrow]

    def test_resume(self):
        import tempfile

        from oibench.run import expected_rows, resume_state

        with tempfile.TemporaryDirectory() as d:
            items, (full, partial, narrow) = self._setup(d)
            self.assertEqual(expected_rows(full, items), 4)
            self.assertEqual(expected_rows(narrow, items), 1)
            done = resume_state(d, [full, partial, narrow], items)
            self.assertEqual(done, {full.key, narrow.key})
            with open(os.path.join(d, "results.jsonl")) as f:
                keys = [json.loads(line)["config"] for line in f]
            self.assertNotIn(partial.key, keys)  # partial rows dropped
            self.assertEqual(keys.count("old|forced|y=1"), 4)  # complete foreign config kept
            self.assertNotIn("old|forced|y=2", keys)  # interrupted foreign config dropped
            with open(os.path.join(d, "explain.jsonl")) as f:
                self.assertEqual(len(f.readlines()), 1)

    def test_method_kinds(self):
        sweep = {
            "prefix": "p",
            "kinds": ["filter", "diverse"],
            "methods": {
                "a": {"table": "hnsw"},
                "b": {"table": "hnsw", "kinds": ["filter"], "modes": ["forced"]},
            },
        }
        cfgs = configs(sweep)
        self.assertEqual(cfgs[0].kinds, frozenset({"filter", "diverse"}))
        self.assertEqual([c.kinds for c in cfgs if c.method == "b"], [frozenset({"filter"})])


class TestSQL(unittest.TestCase):
    def test_filter(self):
        sql, p = sql_for({"kind": "filter", "k": 10, "labels": [7]}, "m1_hnsw", None)
        self.assertIn("labels @> ARRAY[%(l)s]::int4[]", sql)
        self.assertTrue(sql.endswith("LIMIT 10"))
        self.assertEqual(p, {"l": 7})

    def test_group_shapes(self):
        item = {"kind": "group_label", "k": 3, "labels": [1, 2]}
        lat, p = sql_for(item, "t", "lateral")
        self.assertIn("CROSS JOIN LATERAL", lat)
        self.assertIn("LIMIT 3", lat)
        of, _ = sql_for(item, "t", "overfetch:500")
        self.assertIn("LIMIT 500", of)
        self.assertIn("rn <= 3", of)
        self.assertEqual(p, {"ls": [1, 2]})
        with self.assertRaises(ValueError):
            sql_for(item, "t", "bogus")

    def test_diverse(self):
        sql, _ = sql_for({"kind": "diverse", "k": 10, "cap": 2}, "t", "overfetch:200")
        self.assertIn("PARTITION BY author_id", sql)
        self.assertIn("rn <= 2", sql)
        self.assertTrue(sql.endswith("LIMIT 10"))

    def test_cell(self):
        self.assertEqual(
            cell_of({"kind": "diverse", "k": 10, "cap": 1}),
            ("diverse", None, None, 10, None, 1),
        )


class TestPlanAndScore(unittest.TestCase):
    def test_plan_nodes(self):
        plan = {
            "Plan": {
                "Node Type": "Limit",
                "Plans": [{"Node Type": "Index Scan", "Index Name": "m1_prism_vec"}],
            }
        }
        self.assertEqual(plan_nodes(plan), ["Limit", "Index Scan:m1_prism_vec"])

    def test_compact_plan(self):
        from oibench.report import compact_plan

        nodes = ["Limit", "Result", "Sort", "Index Scan:syn_hnsw_pkey", "Gather"]
        self.assertEqual(compact_plan(nodes), "Limit > Sort > Index Scan:syn_hnsw_pkey")

    def test_forced_disables_sort(self):
        from oibench.run import MODE_GUCS

        self.assertEqual(MODE_GUCS["forced"]["enable_sort"], "off")

    def test_score(self):
        f = {"kind": "filter"}
        self.assertEqual(score(f, {"ids": [1, 2, 3, 4]}, {"ids": [2, 4, 9]}), (0.5, 1.0))
        self.assertEqual(score(f, {"ids": [1, 2]}, {"ids": [2, 1]}), (1.0, 0.0))
        self.assertEqual(score(f, {"ids": [1]}, {"error": "timeout"}), (0.0, 1.0))
        g = {"kind": "group_label"}
        truth = {"groups": {"5": [1, 2], "6": [3, 4]}}
        rec, short = score(g, truth, {"groups": {"5": [1, 2], "6": [3]}})
        self.assertAlmostEqual(rec, 0.75)
        self.assertAlmostEqual(short, 0.5)


if __name__ == "__main__":
    unittest.main()
