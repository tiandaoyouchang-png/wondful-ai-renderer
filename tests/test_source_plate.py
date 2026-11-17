"""2026-11 review: clean plate before compositing the product (old-vehicle remnants)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import test_two_workflows as T

PNG = T.load("png_io")
SP = T.load("source_plate")


class SourcePlateTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        import os
        os.environ["WONDFUL_PLATE_CACHE"] = str(self.dir / "cache")
        h, w = 90, 160
        self.src = np.zeros((h, w, 3), np.uint8)
        self.src[:] = (200, 180, 140)
        self.src[30:70, 30:130] = (60, 60, 60)          # the old, larger vehicle
        self.src_path = self.dir / "src.png"
        PNG.write_png(self.src_path, self.src)
        self.alpha = np.zeros((h, w), np.float32)
        self.alpha[40:68, 45:115] = 1.0                 # the new, smaller product

    def test_region_covers_old_vehicle_margin_and_ground(self):
        region = SP.edit_region(self.alpha)
        self.assertTrue(region[40:68, 45:115].all())
        self.assertTrue(region[35, 50])                  # above the new hood line
        self.assertTrue(region[75, 80])                  # ground band below
        self.assertFalse(region[2, 2])

    def test_plate_restores_outside_region_and_caches(self):
        calls = []

        def fake_edit(base, mask, out):
            calls.append((base, mask))
            m = PNG.read_png(mask)
            self.assertEqual(m.shape[:2], self.src.shape[:2])
            img = PNG.read_png(base)[..., :3].copy()
            img[:] = (200, 180, 140)                     # "vehicle removed", but also the corners changed
            img[0:5, 0:5] = (0, 0, 255)
            PNG.write_png(out, img)
            return out

        info = SP.make_plate(self.src_path, self.alpha, self.dir / "w", fake_edit)
        plate = PNG.read_png(info["path"])[..., :3]
        self.assertFalse(info["cached"])
        np.testing.assert_array_equal(plate[2, 2], self.src[2, 2])      # outside region: photograph
        self.assertLess(abs(int(plate[50, 80, 0]) - 200), 3)             # inside: vehicle gone
        again = SP.make_plate(self.src_path, self.alpha, self.dir / "w2", fake_edit)
        self.assertTrue(again["cached"])
        self.assertEqual(len(calls), 1)

    def test_resized_provider_output_is_brought_back_to_photo_size(self):
        def fake_edit(base, mask, out):
            PNG.write_png(out, np.full((100, 178, 3), (200, 180, 140), np.uint8))
            return out
        info = SP.make_plate(self.src_path, self.alpha, self.dir / "w3", fake_edit)
        self.assertEqual(PNG.read_png(info["path"]).shape[:2], self.src.shape[:2])


if __name__ == "__main__":
    unittest.main()
