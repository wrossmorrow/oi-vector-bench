"""Tests for logic with no third-party dependencies (runs with plain unittest)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oilake.hierarchy import ROOT_ENTITY, closure  # noqa: E402
from oilake.keys import (chunk_units, parse_rotation, parse_worker, sample_key,  # noqa: E402
                         unit_name, units_for_worker)

TREE = {
    "LabelName": ROOT_ENTITY,
    "Subcategory": [
        {"LabelName": "animal", "Subcategory": [
            {"LabelName": "dog"},
            {"LabelName": "cat"},
            {"LabelName": "pet", "Subcategory": [{"LabelName": "dog"}]},
        ]},
        {"LabelName": "vehicle", "Subcategory": [{"LabelName": "car"}],
         "Part": [{"LabelName": "wheel"}]},
    ],
}


class TestClosure(unittest.TestCase):
    def setUp(self):
        self.rows = closure(TREE)
        self.pairs = {(a, d): dep for a, d, dep in self.rows}

    def test_self_rows(self):
        for mid in ["animal", "dog", "cat", "pet", "vehicle", "car"]:
            self.assertEqual(self.pairs[(mid, mid)], 0)

    def test_transitive(self):
        self.assertEqual(self.pairs[("animal", "dog")], 1)  # shortest path wins over animal>pet>dog
        self.assertEqual(self.pairs[("pet", "dog")], 1)
        self.assertEqual(self.pairs[("vehicle", "car")], 1)

    def test_root_dropped_and_parts_ignored(self):
        self.assertFalse(any(ROOT_ENTITY in (a, d) for a, d, _ in self.rows))
        self.assertNotIn(("vehicle", "wheel"), self.pairs)
        self.assertNotIn(("wheel", "wheel"), self.pairs)

    def test_no_duplicates(self):
        self.assertEqual(len(self.rows), len(self.pairs))


class TestKeys(unittest.TestCase):
    def test_sample_key_stable_and_spread(self):
        self.assertEqual(sample_key("000002b66c9c498e"), sample_key("000002b66c9c498e"))
        keys = [sample_key(f"{i:016x}") for i in range(20_000)]
        self.assertEqual(len(set(keys)), len(keys))
        # must fit a signed int64: Delta has no unsigned types
        self.assertTrue(all(0 <= k < 2**63 for k in keys))
        # sequential ids should not give sequential keys: roughly half above the midpoint
        frac_high = sum(k >= 2**62 for k in keys) / len(keys)
        self.assertAlmostEqual(frac_high, 0.5, delta=0.02)

    def test_parse_rotation(self):
        self.assertEqual(parse_rotation("0"), 0)
        self.assertEqual(parse_rotation("90.0"), 90)
        self.assertEqual(parse_rotation("270"), 270)
        self.assertIsNone(parse_rotation("nan"))
        self.assertIsNone(parse_rotation(""))
        self.assertIsNone(parse_rotation(None))
        with self.assertRaises(ValueError):
            parse_rotation("45")

    def test_chunk_units(self):
        self.assertEqual(chunk_units(5, 2), [(0, 2), (2, 4), (4, 5)])
        self.assertEqual(chunk_units(0, 2), [])
        with self.assertRaises(ValueError):
            chunk_units(5, 0)

    def test_worker_split_covers_everything_once(self):
        units = list(range(10))
        got = sorted(u for k in range(3) for u in units_for_worker(units, (k, 3)))
        self.assertEqual(got, units)

    def test_worker_skips_done(self):
        todo = units_for_worker(range(4), (0, 1), done=[unit_name(1), unit_name(3)])
        self.assertEqual(todo, [0, 2])

    def test_parse_worker(self):
        self.assertEqual(parse_worker("2/4"), (2, 4))
        for bad in ["4/4", "-1/2", "0/0"]:
            with self.assertRaises(ValueError):
                parse_worker(bad)


if __name__ == "__main__":
    unittest.main()
