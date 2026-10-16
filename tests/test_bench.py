"""oibench data / truth / workload on a small synthetic dataset (numpy only)."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oibench.data import CSR, Dataset  # noqa: E402
from oibench.synth import synthetic  # noqa: E402
from oibench.truth import Exact, as_query  # noqa: E402
from oibench.workload import Spec, bins, generate, read_jsonl, write  # noqa: E402


class TestCSR(unittest.TestCase):
    def test_from_pairs_dedups_and_sorts(self):
        c = CSR.from_pairs(np.array([2, 0, 2, 2, 0]), np.array([5, 1, 3, 5, 0]), n_rows=4)
        self.assertEqual([c.row(i).tolist() for i in range(4)], [[0, 1], [], [3, 5], []])

    def test_inverted(self):
        c = CSR.from_lists([[1, 2], [], [2], [0, 2]])
        inv = c.inverted(3)
        self.assertEqual([inv.row(i).tolist() for i in range(3)], [[3], [0], [0, 2, 3]])


class TestExact(unittest.TestCase):
    def setUp(self):
        self.ds = synthetic()
        self.ex = Exact(self.ds.emb)

    def brute(self, q, idx):
        e = self.ds.emb.astype(np.float32)[idx]
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        d = 1 - e @ as_query(q)
        o = np.lexsort((idx, d))
        return idx[o], d[o]

    def test_topk_matches_brute_force(self):
        q = self.ds.q_emb[0]
        ids, dists = self.ex.topk(q, 25)
        bids, bd = self.brute(q, np.arange(self.ds.n))
        self.assertEqual(ids.tolist(), bids[:25].tolist())
        np.testing.assert_allclose(dists, bd[:25], atol=1e-6)

    def test_subset_and_short_subset(self):
        q = self.ds.q_emb[1]
        idx = np.arange(0, self.ds.n, 7)
        ids, _ = self.ex.topk(q, 10, idx)
        self.assertEqual(ids.tolist(), self.brute(q, idx)[0][:10].tolist())
        ids, _ = self.ex.topk(q, 10, idx[:4])
        self.assertEqual(len(ids), 4)

    def test_topk_batch_matches_single(self):
        Q = self.ds.q_emb[:5]
        got = self.ex.topk_batch(Q, 50, batch=2)
        for i in range(5):
            self.assertEqual(got[i].tolist(), self.ex.topk(Q[i], 50)[0].tolist())

    def test_diverse_cap_and_fallback(self):
        q = self.ds.q_emb[2]
        ids, fb = self.ex.diverse(q, 20, 1, self.ds.author, self.ex.topk(q, 500)[0])
        self.assertFalse(fb)
        self.assertEqual(len(set(self.ds.author[ids])), 20)
        # tiny candidate pool forces the fallback, same answer
        ids2, fb2 = self.ex.diverse(q, 20, 1, self.ds.author, self.ex.topk(q, 5)[0])
        self.assertTrue(fb2)
        self.assertEqual(ids, ids2)


class TestWorkload(unittest.TestCase):
    def test_bins(self):
        b = bins(Spec(log10_top=0, log10_bottom=-2, bin_step=0.5))
        self.assertEqual(len(b), 4)
        self.assertAlmostEqual(b[0][1], 1.0)
        self.assertAlmostEqual(b[-1][0], 0.01)

    def test_generate_and_roundtrip(self):
        ds = synthetic()
        spec = Spec(ks=[5, 20], n_per_cell=6, log10_bottom=-3, group_sizes=[3],
                    group_n=2, diverse_caps=[1], lift_top=200)
        wl, tr = generate(ds, spec, log=lambda *a: None)
        self.assertEqual([w["qid"] for w in wl], list(range(len(wl))))
        ex = Exact(ds.emb)
        inv = ds.labels.inverted(ds.n_labels)
        kinds = {w["kind"] for w in wl}
        self.assertEqual(kinds, {"filter", "group_label", "diverse"})
        lifts = {"pos": [], "neg": []}
        for w, t in zip(wl, tr):
            if w["kind"] == "filter":
                L = w["labels"][0]
                items = inv.row(L)
                self.assertGreaterEqual(len(items), w["k"])
                self.assertEqual(len(t["ids"]), w["k"])
                self.assertTrue(set(t["ids"]) <= set(items.tolist()))
                # ftype semantics
                if w["ftype"] == "pos":
                    self.assertIn(L, ds.q_pos.row(w["q"]).tolist())
                if w["ftype"] == "neg":
                    self.assertIn(L, ds.q_neg.row(w["q"]).tolist())
                if w["ftype"] in lifts:
                    lifts[w["ftype"]].append(w["lift"])
                self.assertEqual(t["ids"], ex.topk(ds.q_emb[w["q"]], w["k"], items)[0].tolist())
            elif w["kind"] == "group_label":
                self.assertEqual(len(w["labels"]), 3)
                self.assertEqual(sorted(int(g) for g in t["groups"]), w["labels"])
            else:
                self.assertEqual(len(set(ds.author[t["ids"]])), w["k"])
        # the synthetic labels follow clusters, so pos filters sit near the query
        self.assertGreater(np.mean(lifts["pos"]), 1.5)
        self.assertLess(np.mean(lifts["neg"]), 0.7)

        with tempfile.TemporaryDirectory() as d:
            write(d, spec, wl, tr)
            self.assertEqual(read_jsonl(os.path.join(d, "workload.jsonl")), wl)

    def test_dataset_roundtrip(self):
        ds = synthetic(n=200, q=10)
        with tempfile.TemporaryDirectory() as d:
            ds.save(d)
            back = Dataset.load(d)
            np.testing.assert_array_equal(back.emb, ds.emb)
            self.assertEqual(back.labels.row(5).tolist(), ds.labels.row(5).tolist())
            self.assertEqual(back.q_image_ids, ds.q_image_ids)
            self.assertEqual(back.meta["n_items"], 200)


if __name__ == "__main__":
    unittest.main()
