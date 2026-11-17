"""2026-11 review: silhouette gate + small-part repair for the single-source edit (worker side)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import test_two_workflows as T

PNG = T.load("png_io")
RW = T.load("result_worker")


class SourceDetailStepTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        h, w = 120, 200
        flat = np.zeros((h, w, 4), np.uint8)
        flat[40:100, 40:160] = (40, 140, 60, 255)          # product body
        flat[80:100, 55:75] = (20, 20, 20, 255)            # wheel
        flat[80:100, 125:145] = (20, 20, 20, 255)
        self.flat = self.dir / "detail.png"
        PNG.write_png(self.flat, flat)
        self.helpers = {"detail": str(self.flat), "keep": "", "matid": ""}
        self.h, self.w = h, w
        self.flat_rgba = flat

    def scene(self, dx=0):
        img = np.zeros((self.h, self.w, 3), np.uint8)
        img[:] = (200, 170, 120)
        img[100:] = (180, 130, 70)
        prod = np.roll(self.flat_rgba, dx, axis=1)
        a = prod[..., 3:4] / 255.0
        img = (img * (1 - a) + prod[..., :3] * a).astype(np.uint8)
        path = self.dir / f"raw_{dx}.png"
        PNG.write_png(path, img)
        return path

    def test_aligned_product_is_checked_and_reports_iou(self):
        raw = self.scene(0)
        out = RW.source_detail_step(str(raw), 1, self.helpers, self.w, self.h, self.dir)
        self.assertIsNotNone(out["iou"])
        self.assertGreaterEqual(out["iou"]["iou"], RW.SOURCE_DETAIL_MIN_IOU)
        self.assertIsNotNone(out["report"])
        self.assertIn("轮廓 IoU", out["note"])
        self.assertEqual(out["raw"]["sha256"], T.load("source_reference").fingerprint(raw)["sha256"])

    def test_moved_product_skips_repair_without_calling_the_model(self):
        raw = self.scene(60)
        calls = []
        out = RW.source_detail_step(str(raw), 1, self.helpers, self.w, self.h, self.dir,
                                    edit_fn=lambda *a, **k: calls.append(a))
        # either gate may catch a moved product: the silhouette IoU or detail_refine's own
        # alignment fit; in both cases nothing is sent to the model and nothing is pasted
        self.assertEqual(out["refined"], "")
        self.assertEqual(calls, [])
        self.assertTrue(out["report"] is None or out["report"].get("skipped"))

    def test_missing_helper_never_raises(self):
        raw = self.scene(0)
        out = RW.source_detail_step(str(raw), 1, {"detail": ""}, self.w, self.h, self.dir)
        self.assertIn("缺少产品细节底图", out["note"])


if __name__ == "__main__":
    unittest.main()
