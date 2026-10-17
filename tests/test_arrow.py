"""Tests for the pyarrow transforms. Skipped where pyarrow isn't installed."""

import importlib.util
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HAVE_ARROW = importlib.util.find_spec("pyarrow") is not None
HAVE_DELTA = importlib.util.find_spec("deltalake") is not None

IMAGES_CSV = (
    "ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,"
    "OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation\n"
    "aaaa000000000001,train,https://x/1_o.jpg,https://l/1,cc-by,https://p/alice,Alice,"
    '"Two\nlines, quoted",1234,md5a,https://x/1_z.jpg,90.0\n'
    "aaaa000000000002,train,https://x/2_o.jpg,https://l/2,cc-by,https://p/bob,Bob,Plain,"
    "99,md5b,,nan\n"
    "aaaa000000000003,validation,https://x/3_o.jpg,https://l/3,cc-by,,,,,md5c,,0\n"
)


@unittest.skipUnless(HAVE_ARROW and HAVE_DELTA, "needs pyarrow + deltalake")
class TestImages(unittest.TestCase):
    def test_image_batches(self):
        import pyarrow as pa

        from oilake.ingest import IMAGES_SCHEMA, _image_batches

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ids.csv")
            with open(path, "w") as f:
                f.write(IMAGES_CSV)
            mirror = pa.array(["aaaa000000000001", "aaaa000000000003"])
            t = pa.Table.from_batches(list(_image_batches(path, mirror)), schema=IMAGES_SCHEMA)
        rows = {r["image_id"]: r for r in t.to_pylist()}
        self.assertEqual(rows["aaaa000000000001"]["s3_key"], "train/aaaa000000000001.jpg")
        self.assertIsNone(rows["aaaa000000000002"]["s3_key"])
        self.assertFalse(rows["aaaa000000000002"]["in_s3_mirror"])
        self.assertEqual(rows["aaaa000000000003"]["s3_key"], "validation/aaaa000000000003.jpg")
        self.assertEqual(rows["aaaa000000000001"]["rotation"], 90)
        self.assertIsNone(rows["aaaa000000000002"]["rotation"])
        self.assertEqual(rows["aaaa000000000001"]["title"], "Two\nlines, quoted")
        self.assertIsNone(rows["aaaa000000000002"]["thumbnail_300k_url"])
        self.assertIsNone(rows["aaaa000000000003"]["original_size"])


@unittest.skipUnless(HAVE_ARROW and HAVE_DELTA, "needs pyarrow + deltalake")
class TestModels(unittest.TestCase):
    def test_registry_shape(self):
        from oilake.embed import MODELS

        for entry in MODELS.values():
            name, pretrained, dim = entry
            self.assertIsInstance(name, str)
            self.assertTrue(pretrained is None or isinstance(pretrained, str))
            self.assertGreater(dim, 0)
        # the point of adding CLIP: fits pg_vectorsearch fastscan posting pages
        self.assertLess(MODELS["clip-l14-336"][2], 946)


@unittest.skipUnless(HAVE_ARROW and HAVE_DELTA, "needs pyarrow + deltalake")
class TestPlanAndSample(unittest.TestCase):
    def _images(self):
        import pyarrow as pa

        from oilake.keys import sample_key

        ids = [f"{i:016x}" for i in range(100)]
        return pa.table(
            {
                "image_id": ids,
                "s3_key": [f"train/{i}.jpg" for i in ids],
                "rotation": pa.array([0] * 100, pa.int16()),
                "sample_key": pa.array([sample_key(i) for i in ids], pa.int64()),
                "in_s3_mirror": [i % 4 != 0 for i in range(100)],
            }
        )

    def test_plan_order_units_and_prefix_property(self):
        from oilake.embed import make_plan

        images = self._images()
        p = make_plan(images, unit_size=10)
        self.assertEqual(p.num_rows, 75)
        keys = p["sample_key"].to_pylist()
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(p["unit"].to_pylist()[:10], [0] * 10)
        self.assertEqual(max(p["unit"].to_pylist()), 7)
        # a limited plan is a prefix of the full plan
        small = make_plan(images, unit_size=10, limit=30)
        self.assertEqual(small["image_id"].to_pylist(), p["image_id"].to_pylist()[:30])

    def test_sample_nested(self):
        import pyarrow as pa

        from oilake.sample import choose

        images = self._images().select(["image_id", "sample_key"])
        ok = pa.array([f"{i:016x}" for i in range(100) if i % 3])
        a = choose(images, ok, 10)["image_id"].to_pylist()
        b = choose(images, ok, 40)["image_id"].to_pylist()
        self.assertEqual(b[:10], a)

    def test_tail_holdout_and_exclude(self):

        from oilake.sample import choose

        images = self._images().select(["image_id", "sample_key"])
        ok = images["image_id"].combine_chunks()
        q = choose(images, ok, 5, tail=True)["image_id"]
        rest = choose(images, ok, None, exclude=q.combine_chunks())
        self.assertEqual(rest.num_rows, 95)
        self.assertFalse(set(q.to_pylist()) & set(rest["image_id"].to_pylist()))
        head = choose(images, ok, 95)["image_id"].to_pylist()
        self.assertEqual(head, rest["image_id"].to_pylist())  # tail holdout = first 95 exactly


if __name__ == "__main__":
    unittest.main()
