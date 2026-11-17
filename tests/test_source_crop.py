"""2026-11 review: local canvas around the product, pasted back into the photograph."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import test_two_workflows as T

PNG = T.load("png_io")
SC = T.load("source_crop")


class SourceCropTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_box_has_endpoint_aspect_and_contains_product(self):
        a = np.zeros((816, 1456), np.float32)
        a[330:600, 420:880] = 1
        x0, y0, x1, y1 = SC.crop_box(a)
        self.assertAlmostEqual((x1 - x0) / (y1 - y0), SC.ASPECT, delta=0.02)
        self.assertTrue(x0 <= 420 and x1 >= 880 and y0 <= 330 and y1 >= 600)
        self.assertAlmostEqual(460 / (x1 - x0), SC.FILL, delta=0.03)
        self.assertTrue(0 <= x0 and 0 <= y0 and x1 <= 1456 and y1 <= 816)

    def test_box_is_skipped_when_product_fills_the_frame(self):
        a = np.zeros((400, 600), np.float32)
        a[20:380, 30:570] = 1
        self.assertIsNone(SC.crop_box(a))

    def test_box_near_the_edge_is_shifted_inside(self):
        a = np.zeros((816, 1456), np.float32)
        a[600:800, 1200:1450] = 1
        x0, y0, x1, y1 = SC.crop_box(a)
        self.assertTrue(x1 <= 1456 and y1 <= 816 and x0 >= 0 and y0 >= 0)

    def test_paste_back_keeps_outside_pixels_and_matches_exposure(self):
        base = np.full((200, 300, 3), 100, np.uint8)
        base[:, :, 1] = 120
        PNG.write_png(self.dir / "base.png", base)
        box = (60, 40, 210, 140)                       # 150×100
        ai = np.full((256, 384, 3), 118, np.uint8)       # provider size, brighter overall
        PNG.write_png(self.dir / "ai.png", ai)
        rep = SC.paste_back(self.dir / "ai.png", self.dir / "base.png", box, self.dir / "out.png")
        out = PNG.read_png(self.dir / "out.png")[..., :3]
        self.assertEqual(out.shape, base.shape)
        np.testing.assert_array_equal(out[10, 10], base[10, 10])          # outside: untouched
        np.testing.assert_array_equal(out[199, 299], base[199, 299])
        centre = out[90, 135].astype(int)
        self.assertLess(abs(centre[0] - 100), 4)                          # gain matched to the ring
        self.assertEqual(rep["box"], list(box))

    def test_registration_snaps_a_shifted_car_back_onto_the_silhouette(self):
        h, w = 240, 360
        alpha = np.zeros((h, w), np.float32)
        alpha[90:170, 80:280] = 1.0
        img = np.full((h, w, 3), 180, np.float32)
        img[96:176, 86:286] = (30, 110, 40)              # the model drew the car 6 px right/down
        out, info = SC.register_product(img, alpha)
        self.assertTrue(info["applied"], info)
        inside = out[95:165, 90:270].mean(axis=(0, 1))
        self.assertLess(abs(inside[1] - 110), 12)
        self.assertLess(abs(out[92, 82, 1] - 110), 15)   # silhouette's top-left corner: car now
        np.testing.assert_allclose(out[5, 5], img[5, 5])  # far background untouched

    def test_registration_leaves_an_aligned_car_alone(self):
        h, w = 240, 360
        alpha = np.zeros((h, w), np.float32)
        alpha[90:170, 80:280] = 1.0
        img = np.full((h, w, 3), 180, np.float32)
        img[90:170, 80:280] = (30, 110, 40)
        out, info = SC.register_product(img, alpha)
        self.assertFalse(info["applied"])                 # already aligned: identity
        np.testing.assert_array_equal(out, img)

    def test_registration_refuses_large_drift(self):
        h, w = 240, 360
        alpha = np.zeros((h, w), np.float32)
        alpha[90:170, 80:280] = 1.0
        img = np.full((h, w, 3), 180, np.float32)
        img[60:140, 60:260] = (30, 110, 40)              # 30 px up: the model moved the car
        out, info = SC.register_product(img, alpha)
        self.assertFalse(info["applied"])


if __name__ == "__main__":
    unittest.main()
