"""Mode 2 (2026-11): 按参考风格渲染 — canvas, whole-frame registration, ranking."""
import unittest

import numpy as np

import test_two_workflows as T

SR = T.load("style_render")


class StyleRenderTests(unittest.TestCase):
    def test_gradient_follows_horizon(self):
        env = {"sky": [0.5, 0.6, 0.9], "horizon": [0.8, 0.6, 0.3], "ground": [0.1, 0.08, 0.05]}
        g = SR.gradient_background(env, 40, 100, horizon=0.3)
        self.assertEqual(g.shape, (100, 40, 3))
        self.assertGreater(int(g[0, 0, 2]), int(g[0, 0, 0]))      # sky blue on top
        self.assertGreater(int(g[30, 0, 0]), int(g[30, 0, 2]))    # warm horizon at 30 %
        self.assertLess(int(g[-1].mean()), int(g[30].mean()))     # dark ground below

    def test_prompt_locks_geometry_and_uses_style_only(self):
        text = SR.style_prompt()
        for phrase in ("完全一致", "不需要复制图 2 的构图", "不得用身份图或其裁切重设计", "三视图或身份图只确定各部件的材质属性"):
            self.assertIn(phrase, text)

    def test_mirrored_sampling_identity_and_no_smear(self):
        img = np.random.rand(30, 50, 3).astype(np.float32)
        same = SR.sample_mirrored(img, 1.0, 0.0, 0.0, (25, 15))
        self.assertLess(np.abs(same - img).max(), 1e-5)
        shifted = SR.sample_mirrored(img, 1.0, 0.0, -5.0, (25, 15))
        # rows exposed at the top mirror the picture instead of repeating row 0
        self.assertLess(np.abs(shifted[0] - img[5]).max(), 1e-5)
        self.assertGreater(np.abs(shifted[0] - img[0]).max(), 1e-3)

    def test_register_whole_snaps_enlarged_car(self):
        h, w = 120, 200
        alpha = np.zeros((h, w), np.float32)
        alpha[50:90, 60:140] = 1
        # the "AI" drew the car 10 % larger and 6 px lower
        ai = np.full((h, w, 3), 200.0, np.float32)
        ys, xs = np.nonzero(alpha)
        cy, cx = ys.mean(), xs.mean()
        y0, y1 = int(round(cy + (50 - cy) * 1.1 + 6)), int(round(cy + (90 - cy) * 1.1 + 6))
        x0, x1 = int(round(cx + (60 - cx) * 1.1)), int(round(cx + (140 - cx) * 1.1))
        ai[y0:y1, x0:x1] = 20.0
        out, info = SR.register_whole(ai, alpha)
        self.assertTrue(info["applied"])
        dark = out[..., 0] < 110
        iou = (dark & (alpha > 0.5)).sum() / max(1, (dark | (alpha > 0.5)).sum())
        self.assertGreater(iou, 0.9)

    def test_rank_prefers_higher_iou(self):
        self.assertGreater(SR.rank({"iou": {"iou": 0.97}}), SR.rank({"iou": {"iou": 0.9}}))
        self.assertEqual(SR.rank({}), -1.0)

    def test_exposed_mask_marks_shifted_strip(self):
        m = SR.exposed_mask((100, 60), 1.0, 0.0, 20.0, (30, 50))
        self.assertTrue(m[-20:].all())
        self.assertFalse(m[:79].any())

    def test_cover_fit_crops_instead_of_bars(self):
        import tempfile
        from pathlib import Path
        PNG = T.load("png_io")
        d = Path(tempfile.mkdtemp())
        src = np.full((1717, 916, 3), 90, np.uint8)
        PNG.write_png(d / "raw.png", src)
        self.assertTrue(SR.cover_fit(d / "raw.png", d / "fit.png", 108, 192))
        out = PNG.read_png(d / "fit.png")
        self.assertEqual(out.shape[:2], (192, 108))
        self.assertTrue((out[..., :3] > 60).all())    # no black bars

    def test_fill_border_only_touches_strip(self):
        import tempfile
        from pathlib import Path
        PNG = T.load("png_io")
        d = Path(tempfile.mkdtemp())
        img = np.full((80, 60, 3), 100, np.uint8)
        PNG.write_png(d / "reg.png", img)
        exposed = np.zeros((80, 60), np.uint8); exposed[-10:] = 255
        PNG.write_png(d / "exp.png", np.repeat(exposed[..., None], 3, axis=2))

        def edit(base, mask, out):
            PNG.write_png(out, np.full((80, 60, 3), 250, np.uint8))
            return out

        info = SR.fill_border(d / "reg.png", d / "exp.png", d / "out.png", edit)
        out = PNG.read_png(info["path"])
        self.assertTrue((out[:40, :, :3] == 100).all())
        self.assertGreater(int(out[-2, 30, 0]), 200)


if __name__ == "__main__":
    unittest.main()
