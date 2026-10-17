"""1.2.3: bounded regeneration wiring, worker-thread result path, feature-line validation and
the cheap fixes for the confirmed GPT-6 counterexamples."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

DP = load_package_module("detail_paste")
MC = load_package_module("material_color")
IS = load_package_module("identity_score")
FL = load_package_module("feature_lines")
RW = load_package_module("result_worker")
PNG = load_package_module("png_io")
PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"

H, W = 360, 640
yy, xx = np.mgrid[0:H, 0:W]
PROD = (((xx - 320) / 220.0) ** 2 + ((yy - 180) / 120.0) ** 2 <= 1).astype(np.float32)


class KeepPerRegion(unittest.TestCase):
    def test_big_misdetected_region_does_not_kill_a_real_logo(self):
        keep = np.zeros((H, W, 4), np.float32)
        big = (PROD > 0) & (xx < 260)
        logo = (PROD > 0) & (abs(xx - 400) < 10) & (abs(yy - 180) < 6)
        for m in (big, logo):
            keep[m, 1] = 1
            keep[m, 3] = 1
        k, _g, _m, info = DP.merge_keep_masks(keep, PROD)
        self.assertEqual(int((k[logo] > 0.5).sum()), int(logo.sum()))   # was 0 in 1.2.2
        self.assertEqual(int((k[big] > 0.5).sum()), 0)
        self.assertEqual(info["auto_dropped_regions"], 1)
        self.assertTrue(info["auto_used"])


class ExactKeepIsExact(unittest.TestCase):
    def test_red_logo_keeps_its_red(self):
        ai = np.zeros((H, W, 3), np.float32)
        ai[...] = (0.45, 0.30, 0.25)
        det = np.zeros((H, W, 4), np.float32)
        det[..., :3] = (0.45, 0.30, 0.25)
        det[..., 3] = PROD
        logo = (PROD > 0) & (abs(xx - 330) < 30) & (abs(yy - 180) < 14)
        det[logo, :3] = (0.9, 0.05, 0.05)
        ai[logo] = (0.62, 0.22, 0.18)              # the AI drew the logo, in a duller red
        keep = np.zeros((H, W, 4), np.float32)
        keep[logo, 0] = 1
        keep[logo, 3] = 1
        fit = {"scale": 1.0, "scale_x": 1.0, "scale_y": 1.0, "dx": 0.0, "dy": 0.0}
        out = DP.paste_details(ai, det, PROD, keep_rgba=keep, fit=fit)
        core = logo & (abs(xx - 330) < 20) & (abs(yy - 180) < 8)
        mean = out["image"][core].mean(0)
        self.assertGreater(mean[0], 0.85)
        self.assertLess(mean[1], 0.1)


class CastNeedsReference(unittest.TestCase):
    def setUp(self):
        self.labels = np.where(PROD > 0, 0, -1).astype(np.int32)
        self.model = np.zeros((H, W, 3), np.float32)
        self.model[...] = (0.1, 0.6, 0.2)
        self.ai = np.zeros((H, W, 3), np.float32)
        self.ai[...] = (0.6, 0.1, 0.1)

    def test_single_paint_wrong_hue_is_corrected(self):
        mats = [{"index": 0, "name": "CarPaint", "kind": "reflective"}]
        res = MC.correct_colours(self.ai, self.model, self.labels, mats, PROD, strength=1.0)
        self.assertEqual(res["cast"]["ref"], "none")
        self.assertGreater(res["changed_px"], 1000)            # 1.2.2 changed 0 px

    def test_neutral_trim_is_the_reference(self):
        labels = self.labels.copy()
        labels[(PROD > 0) & (yy > 230)] = 1
        model = self.model.copy()
        model[labels == 1] = (0.05, 0.05, 0.05)
        ai = self.ai.copy()
        ai[labels == 1] = (0.05, 0.05, 0.05)
        mats = [{"index": 0, "name": "CarPaint", "kind": "reflective"}, {"index": 1, "name": "Trim", "kind": "default"}]
        res = MC.correct_colours(ai, model, labels, mats, PROD, strength=1.0)
        self.assertEqual(res["cast"]["ref"], "neutral")
        self.assertGreater(res["changed_px"], 1000)

    def test_two_ordinary_materials_still_measure_a_cast(self):
        stats = [{"kind": "default", "px": 100, "ai": np.array([50.0, 20, 10]), "model": np.array([50.0, 10, 5])},
                 {"kind": "matte", "px": 100, "ai": np.array([60.0, 18, 30]), "model": np.array([60.0, 8, 25])}]
        cast = MC.estimate_cast(stats)
        self.assertEqual(cast["ref"], "materials")
        self.assertGreater(cast["da"], 5)


class FeatureLinesAndFakeSeams(unittest.TestCase):
    def test_feature_lines_from_normals_and_depth(self):
        n = np.zeros((H, W, 3), np.float32)
        n[..., 2] = 1.0
        n[:, 320:] = (0.7071, 0.0, 0.7071)          # a 45° crease at x = 320
        depth = np.full((H, W), 0.5, np.float32)
        depth[100:140, 100:200] = 0.2              # a part in front: depth step
        prod = np.ones((H, W), bool)
        lines = FL.feature_lines(n, depth, prod)
        self.assertTrue(lines[:, 318:321].any())
        self.assertTrue(lines[99:101, 120:180].any())
        self.assertFalse(lines[:, 10:90].any())

    def test_decode_roundtrip(self):
        rgba = np.zeros((4, 4, 4), np.float32)
        rgba[..., 0] = 0.5
        rgba[..., 1] = 0.5
        rgba[..., 2] = 0.3
        rgba[..., 3] = 1
        n, d, prod = FL.decode(rgba, encoded_srgb=False)
        self.assertAlmostEqual(float(n[0, 0, 2]), 1.0, places=4)
        self.assertTrue(prod.all())

    def _paint_case(self, seam):
        base = np.zeros((H, W, 3), np.float32)
        base[...] = (0.1, 0.6, 0.2)
        ai = base.copy()
        if seam:
            ai[(PROD > 0) & (abs(xx - 330) <= 1)] = (0.02, 0.05, 0.02)
        ai = ai * PROD[..., None] + 0.8 * (1 - PROD[..., None])
        det = np.dstack([base, PROD])
        labels = np.where(PROD > 0, 0, -1).astype(np.int32)
        mats = [{"index": 0, "name": "CarPaint", "kind": "reflective"}]
        return IS.score(ai, det, PROD, (labels, mats), features=np.zeros((H, W), bool))

    def test_ai_only_seam_on_flat_paint_is_named(self):
        rep = self._paint_case(True)
        self.assertEqual(rep["fake_lines"], ["CarPaint"])
        self.assertIn("CarPaint 可疑线条", IS.weakest_parts(rep))
        self.assertEqual(self._paint_case(False)["fake_lines"], [])

    def test_validation_is_decoupled_from_the_paste_source(self):
        # checkerboard only in the detail pass: pasting it must not improve 细节边缘 (GPT-6 #6)
        chk = (((xx // 6) + (yy // 6)) % 2).astype(np.float32)
        det = np.zeros((H, W, 4), np.float32)
        det[..., :3] = (0.3 + 0.4 * chk)[..., None] * PROD[..., None] + 0.8 * (1 - PROD[..., None])
        det[..., 3] = PROD
        ai = np.full((H, W, 3), 0.5, np.float32) * PROD[..., None] + 0.8 * (1 - PROD[..., None])
        pasted = ai.copy()
        pasted[PROD > 0] = det[..., :3][PROD > 0]
        geo = np.zeros((H, W), bool)               # the real geometry is a smooth plate
        geo[178:181, 150:490] = True               # … with one real crease
        geo &= PROD > 0
        raw = IS.score(ai, det, PROD, features=geo)
        out = IS.score(pasted, det, PROD, features=geo)
        self.assertEqual(out["edge_ref"], "geometry")
        self.assertGreaterEqual(out["edge_px"], raw["edge_px"] - 0.5)
        # the old detail-pass reading is still reported, as a diagnostic
        self.assertIn("edge_px_detail", out)


class WorkerResultPath(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="wondful123_"))
        mask = np.zeros((H, W, 4), np.uint8)
        mask[..., 3] = (PROD > 0) * 255
        PNG.write_png(self.tmp / "01_product_mask.png", mask)
        img = np.zeros((H * 2 + 2, W * 2, 3), np.float32)   # AI size differs (aspect almost equal)
        img[...] = (0.3, 0.5, 0.35)
        big = np.repeat(np.repeat(PROD > 0, 2, 0), 2, 1)
        img[1:-1][big] = (0.8, 0.2, 0.1)
        PNG.write_png(self.tmp / "cand.png", (img * 255).astype(np.uint8))

    def test_fit_to_canvas_and_gate(self):
        self.assertTrue(RW.fit_to_canvas(self.tmp / "cand.png", self.tmp / "fit.png", W, H))
        self.assertEqual(PNG.read_png(self.tmp / "fit.png").shape[:2], (H, W))
        rep = RW.gate_score(self.tmp / "cand.png", str(self.tmp / "01_product_mask.png"), W, H, self.tmp, "t")
        self.assertEqual(rep["verdict"], "通过", rep)
        self.assertFalse((self.tmp / "gate_t.png").exists())

    def test_unreadable_candidate_does_not_burn_quota(self):
        (self.tmp / "x.png").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
        rep = RW.gate_score(self.tmp / "x.png", str(self.tmp / "01_product_mask.png"), W, H, self.tmp, "x")
        self.assertIsNone(rep["iou"])
        self.assertFalse(rep["readable"])

    def test_cancel(self):
        with self.assertRaises(RW.Cancelled):
            RW.gate_score(self.tmp / "cand.png", str(self.tmp / "01_product_mask.png"), W, H, self.tmp, "c",
                          cancel_check=lambda: True)

    def test_finish_variant_without_detail_spec(self):
        res = RW.finish_variant(str(self.tmp / "cand.png"), 1, None, W, H, self.tmp)
        self.assertTrue(Path(res["final"]).is_file())
        self.assertEqual(PNG.read_png(res["final"]).shape[:2], (H, W))

    def test_rank_and_notes(self):
        p = {"iou": 0.951, "verdict": "通过", "passed": True}
        f = {"iou": 0.97, "verdict": "未通过"}
        r = {"iou": 0.96, "verdict": "待复核", "iou_lo": 0.94, "iou_hi": 0.97}
        self.assertEqual(max([f, r, p], key=RW.gate_rank), p)
        self.assertGreater(RW.gate_rank(r), RW.gate_rank(f))
        note = RW.gate_note({"tries": [f, f, f], "best": {"iou": 0.931, "verdict": "未通过"}, "exhausted": True})
        self.assertIn("未通过", note)
        self.assertIn("3 次", note)
        self.assertIn("通过", RW.gate_note({"tries": [p], "best": p}))
        self.assertEqual(RW.GATE_MAX, 3)


class OperatorWiring(unittest.TestCase):
    def setUp(self):
        self.ops = (PKG / "operators.py").read_text(encoding="utf-8")

    def test_gate_loop(self):
        self.assertIn("planned_attempts = gate_max", self.ops)
        self.assertIn('if gate_rep.get("passed"):', self.ops)
        self.assertIn("continue        # fresh regeneration from the original base", self.ops)
        self.assertIn("result_worker.gate_score(", self.ops)
        self.assertIn("result_worker.finish_variant(", self.ops)
        self.assertIn("_IoU未通过", self.ops)
        self.assertIn("_候选", self.ops)
        # regeneration never edits a previous output: the gate branch continues before any
        # correction / repair region is set
        loop = self.ops[self.ops.index("if gate_mask:\n                        if gate_rep is None"):]
        self.assertLess(loop.index("continue"), loop.index("correction = "))

    def test_props_and_ui(self):
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        self.assertIn("iou_gate: BoolProperty(", props)
        self.assertIn("default=3, min=1, max=3", props)
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        adv = ui[ui.index("# ---- 高级"):]
        self.assertIn("_draw_iou_gate(adv", adv)
        self.assertIn('"iou_gate"', ui)

    def test_geometry_pass_rendered_with_detail(self):
        vc = (PKG / "viewport_capture.py").read_text(encoding="utf-8")
        self.assertIn("_render_geometry(scene, product, geo_path", vc)
        self.assertIn('"View Distance"', vc)


if __name__ == "__main__":
    unittest.main()
