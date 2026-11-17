"""Mode 2 (2026-11): reference photo projected to the user's camera + view plate."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import test_two_workflows as T

PNG = T.load("png_io")
SP = T.load("scene_projection")


def uv_grid(h, w, sx=1.0, sy=1.0):
    uv = np.zeros((h, w, 4), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    uv[..., 0] = xx / w * sx
    uv[..., 1] = yy / h * sy
    uv[..., 3] = 1
    return uv


class ConfidenceTests(unittest.TestCase):
    def test_identity_mapping_fully_trusted(self):
        self.assertGreater(SP.confidence(uv_grid(64, 96), (96, 64)).mean(), 0.99)

    def test_strong_magnification_untrusted(self):
        self.assertLess(SP.confidence(uv_grid(64, 96, sx=0.1, sy=0.1), (96, 64)).mean(), 0.01)

    def test_holes_zero(self):
        uv = uv_grid(64, 96)
        uv[:10, :, 3] = 0
        c = SP.confidence(uv, (96, 64))
        self.assertEqual(float(c[:10].max()), 0.0)
        self.assertGreater(float(c[20:].mean()), 0.99)


class ViewPlateTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        h, w = 60, 90
        self.proj = np.zeros((h, w, 4), np.uint8)
        self.proj[..., :3] = (120, 140, 90)
        self.proj[..., 3] = 255
        self.proj[:15, :, 3] = 0
        self.proj[:15, :, :3] = 0
        self.proj_path = self.dir / "proj.png"
        PNG.write_png(self.proj_path, self.proj)
        self.conf = np.ones((h, w), np.float32)
        self.conf[:15] = 0
        self.ref = self.dir / "ref.png"
        PNG.write_png(self.ref, self.proj[..., :3].copy())

    def test_push_pull_keeps_trusted_and_fills_holes(self):
        rgb = self.proj[..., :3].astype(np.float32)
        f = SP.push_pull_fill(rgb, self.conf)
        self.assertLess(np.abs(f[30:] - rgb[30:]).max(), 1e-3)
        self.assertLess(np.abs(f[2] - np.array([120, 140, 90])).max(), 2.0)

    def test_view_plate_restores_trusted_pixels(self):
        calls = []

        def edit(base, mask, ref, out):
            calls.append((base, mask, ref))
            PNG.write_png(out, np.full((60, 90, 3), 255, np.uint8))
            return out

        info = SP.make_view_plate(self.proj_path, self.conf, self.ref, self.dir / "w", edit)
        self.assertTrue(info["edited"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], str(self.ref))
        out = PNG.read_png(info["path"])
        self.assertTrue((out[40:, :, :3] == (120, 140, 90)).all())
        self.assertTrue((out[:10, :, :3] == 255).all())

    def test_no_holes_no_edit(self):
        PNG.write_png(self.proj_path, np.dstack([self.proj[..., :3], np.full((60, 90), 255, np.uint8)]))
        info = SP.make_view_plate(self.proj_path, np.ones((60, 90), np.float32), self.ref, self.dir / "w2",
                                  lambda *a: self.fail("no edit expected"))
        self.assertFalse(info["edited"])


if __name__ == "__main__":
    unittest.main()
