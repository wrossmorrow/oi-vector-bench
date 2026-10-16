"""UnitDataset: concurrent fetch, error isolation, rotation, picklability."""

import importlib.util
import os
import pickle
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HAVE = all(importlib.util.find_spec(m) for m in ("pyarrow", "PIL"))


@unittest.skipUnless(HAVE, "needs pyarrow + pillow")
class TestUnitDataset(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        self.dir = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self.dir.name, "train"))
        Image.new("RGB", (40, 20), "red").save(os.path.join(self.dir.name, "train/a.jpg"))
        with open(os.path.join(self.dir.name, "train/bad.jpg"), "wb") as f:
            f.write(b"not a jpeg")
        self.rows = [
            {"image_id": "a", "s3_key": "train/a.jpg", "rotation": 90},
            {"image_id": "bad", "s3_key": "train/bad.jpg", "rotation": 0},
            {"image_id": "gone", "s3_key": "train/missing.jpg", "rotation": None},
            {"image_id": "a2", "s3_key": "train/a.jpg", "rotation": None},
        ]

    def tearDown(self):
        self.dir.cleanup()

    def _ds(self, rotate=True):
        from oilake.embed import LocalFetcher, UnitDataset

        return UnitDataset(self.rows, LocalFetcher(self.dir.name), _size, rotate)

    def test_batch_fetch_isolates_errors_and_rotates(self):
        out = self._ds().__getitems__([0, 1, 2, 3])
        self.assertEqual([i for _, i, _ in out], [0, 1, 2, 3])
        self.assertEqual(out[0][0], (20, 40))       # rotated 90 -> portrait
        self.assertEqual(out[3][0], (40, 20))       # rotation unknown -> untouched
        self.assertIsNone(out[1][0])
        self.assertIn("UnidentifiedImageError", out[1][2])
        self.assertIsNone(out[2][0])
        self.assertIn("FileNotFoundError", out[2][2])

    def test_no_rotation_flag(self):
        self.assertEqual(self._ds(rotate=False)[0][0], (40, 20))

    def test_picklable_after_use(self):
        ds = self._ds()
        ds.__getitems__([0])                         # creates the thread pool
        clone = pickle.loads(pickle.dumps(ds))
        self.assertEqual(clone[0][0], (20, 40))

    def test_s3_fetcher_pickles_without_client(self):
        from oilake.embed import S3MirrorFetcher

        f = S3MirrorFetcher()
        f._client = object()                         # stand-in for a live boto3 client
        self.assertIsNone(pickle.loads(pickle.dumps(f))._client)


def _size(img):  # module-level so the dataset stays picklable
    return img.size


if __name__ == "__main__":
    unittest.main()
