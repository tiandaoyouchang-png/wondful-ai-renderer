"""1.2.3 固定坐标验收: deterministic contour IoU (no fitting), holes = background, thin rods,
bounds -> 通过 / 未通过 / 待复核."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

C = load_package_module("contour_iou")
PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"

H, W = 360, 640
yy, xx = np.mgrid[0:H, 0:W]


def product_mask(cx=320, cy=190, rx=200, ry=110):
    return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0)


def scene(mask, fg=(0.75, 0.15, 0.12), seed=3):
    rng = np.random.default_rng(seed)
    bg = np.dstack([0.25 + 0.15 * (yy / H), 0.45 + 0.05 * np.sin(xx / 23.0), 0.30 + 0.0 * xx]).astype(np.float32)
    bg += rng.normal(0, 0.015, bg.shape).astype(np.float32)
    img = np.where(mask[..., None], np.array(fg, np.float32) + 0.08 * (yy / H)[..., None], bg)
    return np.clip(img, 0, 1).astype(np.float32)


class ContourIoUTests(unittest.TestCase):
    def test_exact_placement_passes(self):
        M = product_mask()
        rep = C.score(scene(M), M.astype(np.float32))
        self.assertGreater(rep["iou"], 0.98)
        self.assertEqual(rep["verdict"], "通过")
        self.assertTrue(rep["passed"])

    def test_fixed_coordinates_shift_is_measured_not_fitted(self):
        M = product_mask()
        shifted = product_mask(cx=320 + 12)
        rep = C.score(scene(shifted), M.astype(np.float32))
        expected = (shifted & M).sum() / (shifted | M).sum()
        self.assertAlmostEqual(rep["iou"], expected, delta=0.02)
        self.assertLess(rep["iou"], 0.95)
        self.assertEqual(rep["verdict"], "未通过")

    def test_appearance_term_does_not_hide_a_shift(self):
        # the Blender detail pass (appearance at the Blender position) must not pull the outline back
        M = product_mask()
        shifted = product_mask(cx=320 + 10, cy=190 + 4)
        rep = C.score(scene(shifted), M.astype(np.float32), detail_rgb=scene(M))
        expected = (shifted & M).sum() / (shifted | M).sum()
        self.assertAlmostEqual(rep["iou"], expected, delta=0.02)
        self.assertNotEqual(rep["verdict"], "通过")
        same = C.score(scene(M), M.astype(np.float32), detail_rgb=scene(M))
        self.assertEqual(same["verdict"], "通过")

    def test_scale_up_is_measured(self):
        M = product_mask()
        big = product_mask(rx=214, ry=118)
        rep = C.score(scene(big), M.astype(np.float32))
        expected = (big & M).sum() / (big | M).sum()
        # +7 % reaches the band limit (3 % of the diagonal): the reading is capped there, but it
        # still fails clearly — it can never be flattered into a pass
        self.assertAlmostEqual(rep["iou"], expected, delta=0.035)
        self.assertEqual(rep["verdict"], "未通过")
        small = product_mask(rx=196, ry=108)
        rep2 = C.score(scene(small), M.astype(np.float32))
        self.assertAlmostEqual(rep2["iou"], (small & M).sum() / (small | M).sum(), delta=0.015)

    def test_hole_is_background(self):
        M = product_mask()
        hole = (((xx - 320) / 60.0) ** 2 + ((yy - 190) / 35.0) ** 2 <= 1.0)
        Mh = M & ~hole
        good = C.score(scene(Mh), Mh.astype(np.float32))
        self.assertGreater(good["iou"], 0.97)
        self.assertGreater(good["hole_px"], 0)
        # the AI filled the handle gap with product: the IoU must drop
        filled = C.score(scene(M), Mh.astype(np.float32))
        self.assertGreater(filled["holes_filled_px"], 0)
        self.assertLess(filled["iou_lo"], 0.95)
        self.assertNotEqual(filled["verdict"], "通过")

    def test_thin_rod_kept_and_missing_rod_lowers_the_bound(self):
        M = product_mask()
        rod = (np.abs(xx - 500) <= 2) & (yy > 20) & (yy < 120)
        Mr = M | rod
        present = C.score(scene(Mr), Mr.astype(np.float32))
        self.assertGreater(present["thin_px"], 0)
        self.assertGreater(present["iou"], 0.97)
        missing = C.score(scene(M), Mr.astype(np.float32))
        self.assertLess(missing["iou_lo"], present["iou_lo"])

    def test_no_contrast_is_uncertain_not_passed(self):
        M = product_mask()
        flat = np.full((H, W, 3), 0.4, np.float32)
        rep = C.score(flat, M.astype(np.float32))
        self.assertNotEqual(rep["verdict"], "通过")

    def test_trace_and_fill_reproduce_the_mask(self):
        M = product_mask()
        pts = C.trace_boundary(M)
        F = C.fill_polygon(pts.astype(np.float64), M.shape)
        self.assertGreater((F & M).sum() / (F | M).sum(), 0.985)

    def test_verdict_rules(self):
        self.assertEqual(C.verdict(0.951, 0.97, 0.0), "通过")
        self.assertEqual(C.verdict(0.90, 0.949, 0.0), "未通过")
        self.assertEqual(C.verdict(0.94, 0.96, 0.0), "待复核")
        self.assertEqual(C.verdict(0.96, 0.97, 0.5), "待复核")   # ran past the band: never 通过
        self.assertEqual(C.verdict(0.96, 0.97, 0.0, 0.8), "待复核")   # no edges to judge by
        self.assertEqual(C.PASS_IOU, 0.95)                        # the bar is fixed

    def test_score_file_resamples_but_never_fits(self):
        import tempfile
        png = load_package_module("png_io")
        M = product_mask()
        tmp = Path(tempfile.mkdtemp())
        mask = np.zeros((H, W, 4), np.uint8)
        mask[..., 3] = M * 255
        png.write_png(tmp / "mask.png", mask)
        # AI image at 2x the canvas size (same aspect): resampled only
        img = scene(M)
        big = np.repeat(np.repeat(img, 2, 0), 2, 1)
        png.write_png(tmp / "ai.png", (big * 255).astype(np.uint8))
        rep = C.score_file(tmp / "ai.png", tmp / "mask.png")
        self.assertEqual(rep["verdict"], "通过", rep)
        self.assertEqual(rep["size"], [W, H])

    def test_pure_numpy(self):
        for name in ("contour_iou.py", "feature_lines.py", "result_worker.py"):
            src = (PKG / name).read_text(encoding="utf-8")
            self.assertNotIn("import bpy", src, name)
            self.assertNotIn("from bpy", src, name)


if __name__ == "__main__":
    unittest.main()
