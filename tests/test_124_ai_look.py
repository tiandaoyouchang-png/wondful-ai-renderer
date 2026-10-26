"""1.2.4 降低 AI 感: 照片质感 (deterministic finish, product protected), 多角度 统一调色 (co-visible
environment only), low quality-tier retry inside the existing budget, Lanczos upscale, photographic prompt flag."""
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402
from test_123_multiview import cam, view, texture  # noqa: E402

PF = load_package_module("photo_finish")
LO = load_package_module("layered_output")
RW = load_package_module("result_worker")
MV = load_package_module("multiview")
MC = load_package_module("material_color")
PNG = load_package_module("png_io")
CD = load_package_module("codex_direct")
PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"

H, W = 180, 320
yy, xx = np.mgrid[0:H, 0:W]
PROD = (((xx - 160) / 90.0) ** 2 + ((yy - 100) / 50.0) ** 2 <= 1).astype(np.float32)


def scene_image(seed=3):
    rng = np.random.default_rng(seed)
    base = np.stack([0.35 + 0.3 * np.sin(xx / 23.0), 0.45 + 0.2 * np.cos(yy / 17.0), 0.6 - 0.25 * (yy / H)], -1)
    base = base + 0.02 * rng.standard_normal((H, W, 3))
    prod = np.array([0.80, 0.12, 0.10])            # saturated red paint, below the highlight shoulder
    img = np.where(PROD[..., None] > 0.5, prod, base)
    return np.clip(img, 0, 1).astype(np.float32)


def de(a, b):
    return np.linalg.norm(MC.srgb_to_lab(a) - MC.srgb_to_lab(b), axis=-1)


class PhotoFinish(unittest.TestCase):
    def test_deterministic_for_same_input_and_seed(self):
        img = scene_image()
        a = PF.apply(img, PROD, "SUBTLE", seed=1234)
        b = PF.apply(img, PROD, "SUBTLE", seed=1234)
        self.assertTrue(np.array_equal(a, b))
        c = PF.apply(img, PROD, "SUBTLE", seed=99)
        self.assertGreater(float(np.abs(a - c).mean()), 1e-4)
        # default seed = content seed: same content → same output
        self.assertTrue(np.array_equal(PF.apply(img, PROD, "SUBTLE"), PF.apply(img.copy(), PROD, "SUBTLE")))

    def test_off_is_identity(self):
        img = scene_image()
        self.assertTrue(np.array_equal(PF.apply(img, PROD, "OFF"), img))

    def test_saturation_compression_never_touches_the_product(self):
        img = scene_image()
        img[PROD < 0.5] = [0.15, 0.75, 0.20]          # very saturated environment
        out = PF.tone(img, PF.protect_mask(PROD, img.shape))
        inner = LO.erode(PROD > 0.5, 3)
        self.assertTrue(np.allclose(out[inner], img[inner], atol=1e-6))   # product pixels unchanged
        env = ~LO.dilate(PROD > 0.5, 6)
        s_in = (img.max(-1) - img.min(-1)) / img.max(-1)
        s_out = (out.max(-1) - out.min(-1)) / np.maximum(out.max(-1), 1e-6)
        self.assertLess(float(s_out[env].mean()), float(s_in[env].mean()) - 0.05)

    def test_product_colour_shift_below_one_delta_e(self):
        img = scene_image()
        out = PF.apply(img, PROD, "SUBTLE")
        inner = LO.erode(PROD > 0.5, 3)
        k = 2   # colour, not grain: compare blurred images
        d = de(LO.blur(out, k), LO.blur(img, k))
        self.assertLess(float(d[inner].mean()), 1.0)

    def test_grain_level_and_highlight_rolloff(self):
        flat = np.full((H, W, 3), 0.5, np.float32)
        out = PF.apply(flat, None, "SUBTLE", seed=5)
        sd = float((out * 255).std())
        self.assertGreater(sd, 0.4)
        self.assertLess(sd, 3.0)
        self.assertLess(abs(float(out.mean()) - float(PF.apply(flat, None, "OFF").mean())), 0.01)
        hot = np.full((8, 8, 3), [0.99, 0.95, 0.90], np.float32)
        t = PF.tone(hot)
        self.assertLess(float(t.max()), 0.985)
        self.assertGreater(float(t.max()), 0.9)
        mid = np.full((4, 4, 3), [0.85, 0.5, 0.4], np.float32)
        self.assertLessEqual(float(PF.tone(mid)[..., 0].max()), 0.85 + 1e-6)   # never brightens
        white = np.ones((4, 4, 3), np.float32)
        self.assertTrue(np.allclose(PF.tone(white), PF.tone(white)[0, 0]))       # neutral stays neutral

    def test_apply_file_keeps_alpha_padding_and_reports(self):
        tmp = Path(tempfile.mkdtemp())
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[..., :3] = np.rint(scene_image() * 255).astype(np.uint8)
        rgba[..., 3] = 255
        rgba[:10] = 0                                  # transparent padding rows
        PNG.write_png(tmp / "in.png", rgba)
        m = np.zeros((H, W, 4), np.uint8)
        m[..., 3] = (PROD * 255).astype(np.uint8)
        PNG.write_png(tmp / "mask.png", m)
        rep = PF.apply_file(tmp / "in.png", tmp / "out.png", str(tmp / "mask.png"), "SUBTLE")
        self.assertTrue(rep["applied"] and rep["protected"], rep)
        out = PNG.read_png(tmp / "out.png")
        self.assertTrue(np.array_equal(out[:10], rgba[:10]))
        self.assertTrue(np.array_equal(out[..., 3], rgba[..., 3]))
        rep2 = PF.apply_file(tmp / "in.png", tmp / "out2.png", str(tmp / "mask.png"), "SUBTLE")
        self.assertEqual((tmp / "out.png").read_bytes(), (tmp / "out2.png").read_bytes())
        self.assertEqual(rep["seed"], rep2["seed"])

    def test_worker_finish_step_and_clean_sidecar(self):
        tmp = Path(tempfile.mkdtemp())
        rgba = np.full((H, W, 4), 255, np.uint8)
        rgba[..., :3] = np.rint(scene_image() * 255).astype(np.uint8)
        PNG.write_png(tmp / "cand.png", rgba)
        res = RW.finish_variant(str(tmp / "cand.png"), 1, None, W, H, tmp, finish={"level": "SUBTLE", "mask": ""})
        self.assertTrue(res["final"].endswith("_finish.png"))
        self.assertEqual(res["clean"], res["canvas"])
        self.assertTrue(Path(res["final"]).is_file())
        off = RW.finish_variant(str(tmp / "cand.png"), 2, None, W, H, tmp, finish={"level": "OFF"})
        self.assertNotIn("clean", off)
        self.assertEqual(off["final"], off["canvas"])
        # 多角度 uses the pre-finish sidecar of an exported 主视角
        exported = tmp / "Wondful_x.png"
        PNG.write_png(exported, rgba)
        self.assertEqual(PF.clean_source(exported), str(exported))
        PF.clean_path(exported).parent.mkdir(parents=True)
        PNG.write_png(PF.clean_path(exported), rgba)
        self.assertEqual(PF.clean_source(exported), str(PF.clean_path(exported)))
        self.assertIn(".wondful", PF.clean_source(exported))


class ColourMatch(unittest.TestCase):
    def setUp(self):
        self.m = view(cam("m", (4, -7, 2)), master=True, with_image=True)
        self.k = view(cam("k", (6.5, -4.5, 2)))
        self.truth = np.where(self.k["pos"][..., 3:4] > 0.5, texture(self.k["pos"][..., :3]),
                              np.array([110, 160, 230.0])).astype(np.float32)

    def test_fit_recovers_an_env_cast_and_leaves_the_product_untouched(self):
        cast = np.clip(self.truth * np.array([1.12, 0.97, 0.86]) + np.array([6, 0, -4]), 0, 255)
        mask = self.k["mask"]
        prod_col = np.array([20, 200, 40.0])           # the product looks nothing like the env
        img_k = np.where(mask[..., None] > 0.5, prod_col, cast).astype(np.float32)
        g = MV.fit_grade(self.m, self.k, img_k)
        self.assertIsNotNone(g)
        self.assertLess(g["dE_after"], g["dE_before"])
        out = MV.apply_grade(img_k, g, mask)
        prod = mask > 0.5
        self.assertTrue(np.array_equal(out[prod], img_k[prod]))        # product untouched
        env = (~LO.dilate(mask > 0.5, 4)) & (self.k["pos"][..., 3] > 0.5)
        before = MV.lab(cast)[env] - MV.lab(self.truth)[env]
        after = MV.lab(out)[env] - MV.lab(self.truth)[env]
        self.assertLess(float(np.linalg.norm(after.mean(0))), 0.5 * float(np.linalg.norm(before.mean(0))))

    def test_product_colour_does_not_steer_the_fit(self):
        mask = self.k["mask"]
        img_a = np.where(mask[..., None] > 0.5, np.array([250, 10, 10.0]), self.truth).astype(np.float32)
        img_b = np.where(mask[..., None] > 0.5, np.array([10, 10, 250.0]), self.truth).astype(np.float32)
        ga, gb = MV.fit_grade(self.m, self.k, img_a), MV.fit_grade(self.m, self.k, img_b)
        if ga is None or gb is None:      # nothing to correct on a perfect match is fine too
            self.assertEqual(ga is None, gb is None)
            return
        self.assertTrue(np.allclose(np.asarray(ga["A"]), np.asarray(gb["A"]), atol=0.05))

    def test_no_overlap_means_no_grade(self):
        k = dict(self.k)
        k["mask"] = np.ones_like(self.k["mask"])        # nothing but product → nothing to fit on
        self.assertIsNone(MV.fit_grade(self.m, k, self.truth))
        self.assertTrue(np.array_equal(MV.apply_grade(self.truth, None, self.k["mask"]), self.truth))

    def test_wired_into_multiview_env_only(self):
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        self.assertIn("MV.fit_grade(master_view, vk,", ops)
        self.assertIn("MV.apply_grade(img_full, g, mfull)", ops)
        self.assertIn('getattr(props, "mv_colour_match", True)', ops)
        self.assertIn("clean_source(master_img)", ops)
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        self.assertIn("mv_colour_match: BoolProperty", props)


class LowTierRetry(unittest.TestCase):
    def test_rule(self):
        self.assertTrue(RW.low_tier_retry("low", 1, 3, False))
        self.assertTrue(RW.low_tier_retry("LOW", 1, 2, False))
        self.assertFalse(RW.low_tier_retry("low", 2, 3, False))     # only after the first try
        self.assertFalse(RW.low_tier_retry("low", 1, 3, True))      # at most once per image
        self.assertFalse(RW.low_tier_retry("low", 1, 1, False))     # never past the budget
        self.assertFalse(RW.low_tier_retry("medium", 1, 3, False))
        self.assertFalse(RW.low_tier_retry("", 1, 3, False))

    def test_simulated_budget(self):
        """Replay of the gate loop: a pass at low tier costs exactly one extra generation, never more."""
        def run(results, planned):
            used, calls = False, 0
            for i in range(1, planned + 1):
                calls += 1
                passed, tier = results[i - 1]
                if passed:
                    if RW.low_tier_retry(tier, i, planned, used):
                        used = True
                        continue
                    break
            return calls
        self.assertEqual(run([(True, "low"), (True, "low"), (True, "low")], 3), 2)
        self.assertEqual(run([(True, "medium")], 3), 1)
        self.assertEqual(run([(False, "low"), (True, "low"), (True, "medium")], 3), 2)
        self.assertEqual(run([(True, "low")], 1), 1)

    def test_rank_prefers_non_low_among_passes_only(self):
        p_low = {"verdict": "通过", "iou": 0.98, "image_quality": "low"}
        p_med = {"verdict": "通过", "iou": 0.96, "image_quality": "medium"}
        f_low = {"verdict": "未通过", "iou": 0.94, "image_quality": "low"}
        f_med = {"verdict": "未通过", "iou": 0.90, "image_quality": "medium"}
        self.assertEqual(max([p_low, p_med], key=RW.gate_rank), p_med)
        self.assertEqual(max([f_low, f_med], key=RW.gate_rank), f_low)     # failures: IoU decides
        self.assertEqual(max([f_med, p_low], key=RW.gate_rank), p_low)

    def test_multiview_choose_and_wiring(self):
        a = {"identity": {"edge_px_detail": 2.0}, "product_violations": [], "env_violations": [], "image_quality": "low"}
        b = {"identity": {"edge_px_detail": 3.0}, "product_violations": [], "env_violations": [], "image_quality": "medium"}
        self.assertIs(MV.choose([a, b]), b)
        self.assertEqual(MV.MAX_CALLS, 2)
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        self.assertIn("RW.low_tier_retry(t[\"image_quality\"], n, mv_calls, False)", ops)
        self.assertIn("result_worker.low_tier_retry(tier, attempt_index, planned_attempts, low_retry_used)", ops)
        self.assertIn('"image_quality": tier', ops)

    def test_gate_note_mentions_the_retry(self):
        gate = {"tries": [{"iou": 0.97}, {"iou": 0.96}], "best": {"iou": 0.96, "verdict": "通过"},
                "passed": True, "low_tier_retry": True}
        self.assertIn("low", RW.gate_note(gate))
        gate["low_tier_retry"] = False
        self.assertNotIn("low", RW.gate_note(gate))


class Lanczos(unittest.TestCase):
    def test_enlarging_is_sharper_than_bilinear_and_exact_on_constants(self):
        rng = np.random.default_rng(0)
        small = LO.blur(rng.uniform(0, 255, (60, 100, 3)).astype(np.float32), 1)
        big_l = LO.upscale(small, 160, 96)
        big_b = LO.resize(small, 160, 96)
        self.assertEqual(big_l.shape, (96, 160, 3))
        hf = lambda a: float(np.abs(np.diff(a, axis=1)).mean())
        self.assertGreater(hf(big_l), hf(big_b))
        c = np.full((20, 30, 4), 77.0, np.float32)
        self.assertTrue(np.allclose(LO.upscale(c, 61, 41), 77.0, atol=1e-3))
        self.assertGreaterEqual(float(big_l.min()), float(small.min()) - 1e-3)
        self.assertLessEqual(float(big_l.max()), float(small.max()) + 1e-3)

    def test_same_size_and_shrink_paths(self):
        a = np.random.default_rng(1).uniform(0, 1, (40, 50)).astype(np.float32)
        self.assertTrue(np.array_equal(LO.upscale(a, 50, 40), a))
        self.assertTrue(np.array_equal(LO.upscale(a, 20, 16), LO.resize(a, 20, 16)))

    def test_canvas_fit_uses_lanczos(self):
        tmp = Path(tempfile.mkdtemp())
        rgba = np.full((45, 80, 4), 255, np.uint8)
        rgba[..., :3] = np.random.default_rng(2).integers(0, 255, (45, 80, 3), dtype=np.uint8)
        PNG.write_png(tmp / "s.png", rgba)
        self.assertTrue(RW.fit_to_canvas(tmp / "s.png", tmp / "f.png", 160, 90))
        got = PNG.read_png(tmp / "f.png").astype(np.float32)
        want = np.clip(np.rint(LO.upscale(rgba.astype(np.float32), 160, 90)), 0, 255)
        self.assertTrue(np.array_equal(got, want))
        src = (PKG / "image_manager.py").read_text(encoding="utf-8")
        self.assertNotIn("source_image.scale(", src)
        self.assertIn("_resample_pixels(", src)


class PhotoPrompt(unittest.TestCase):
    def test_default_off_keeps_the_old_text(self):
        self.assertFalse(CD.PHOTO_PROMPT)
        t = CD.build_simple_prompt("沙漠公路", vehicle=True, env_text="沙丘")
        self.assertIn(CD.CLEARCOAT_CG, t)
        self.assertIn(CD.ENDING_CG, t)

    def test_flag_on_rewrites_vehicle_and_product_text(self):
        try:
            CD.PHOTO_PROMPT = True
            car = CD.build_simple_prompt("沙漠公路", vehicle=True, env_text="沙丘")
            chair = CD.build_simple_prompt("客厅", vehicle=False, env_text="地毯")
        finally:
            CD.PHOTO_PROMPT = False
        for t in (car, chair):
            self.assertNotIn("锐利的高光带", t)
            self.assertNotIn("地平线反射线", t)
            self.assertIn(CD.FOREGROUND_PHOTO, t)
            self.assertNotIn("50mm", t)
            self.assertIn("保持 Blender 相机的机位、投影、焦段观感和景深", t)
            self.assertTrue(t.endswith("只输出一张图。"))
        self.assertIn(CD.CLEARCOAT_PHOTO, car)
        self.assertIn(CD.MATERIAL_PHOTO_TAIL, chair)
        self.assertEqual(CD.photo_rewrite(CD.build_simple_prompt("沙漠公路", vehicle=True, env_text="沙丘"), True), car)

    def test_env_override(self):
        import os
        old = os.environ.get("WONDFUL_PHOTO_PROMPT")
        os.environ["WONDFUL_PHOTO_PROMPT"] = "1"
        try:
            mod = importlib.reload(CD)
            self.assertTrue(mod.PHOTO_PROMPT)
        finally:
            if old is None:
                os.environ.pop("WONDFUL_PHOTO_PROMPT", None)
            else:
                os.environ["WONDFUL_PHOTO_PROMPT"] = old
            importlib.reload(CD)
            CD.PHOTO_PROMPT = False


class Wiring(unittest.TestCase):
    def test_finish_is_under_advanced_and_last(self):
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        self.assertIn('elif tab == "FINISH":', ui)
        self.assertIn('row.prop(props, "photo_finish", expand=True)', ui)
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        self.assertIn('("FINISH", "成片"', props)
        self.assertIn('default="SUBTLE"', props[props.index("photo_finish: EnumProperty"):][:1200])
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        self.assertIn("finish=self._finish", ops)
        self.assertIn("sidecar_path(exported, CLEAN_SUFFIX)", ops)
        rw = (PKG / "result_worker.py").read_text(encoding="utf-8")
        body = rw[rw.index("def finish_variant"):rw.index("def photo_finish_step")]
        self.assertLess(body.index("paste_and_score("), body.index("photo_finish_step("))


if __name__ == "__main__":
    unittest.main()
