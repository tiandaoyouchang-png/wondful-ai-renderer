"""1.2.0 细节回贴 › 按材质校色: material-ID pass decoding, per-material colour pull, wiring."""
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "wondful_ai_renderer"


def _load(name):
    if "wondful_ai_renderer" not in sys.modules:
        pkg = types.ModuleType("wondful_ai_renderer")
        pkg.__path__ = [str(PKG)]
        sys.modules["wondful_ai_renderer"] = pkg
    return importlib.import_module(f"wondful_ai_renderer.{name}")


MC = _load("material_color")
DP = _load("detail_paste")
PNG = _load("png_io")

H, W = 200, 300
MATS = [{"index": 0, "name": "fabric Velvet", "kind": "matte"},
        {"index": 1, "name": "wood Brown", "kind": "matte"},
        {"index": 2, "name": "Windows", "kind": "glass"}]


def _labels():
    """Fabric block (top), wood legs (bottom strip), glass patch; -1 background."""
    lab = np.full((H, W), -1, np.int32)
    lab[30:120, 60:240] = 0
    lab[120:185, 70:230] = 1
    lab[40:70, 180:230] = 2
    return lab


def _lab_img(lab_px):
    return MC.lab_to_srgb(lab_px.astype(np.float32))


def _scene(wood_ai_ab=(16.0, 15.0), cast_ab=(-2.0, -3.0)):
    """Model: flat-lit albedo (fabric orange, wood dull brown, glass grey). AI: same parts lit by a
    left→right gradient (L 30..60), fabric = model + cast, wood drawn too red (``wood_ai_ab``)."""
    lab = _labels()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    model = np.zeros((H, W, 3), np.float32); model[..., 0] = 50
    model[lab == 0] = (57, 51, 47); model[lab == 1] = (31, 3, 9); model[lab == 2] = (47, 0, 0)
    ai = np.zeros((H, W, 3), np.float32)
    ai[..., 0] = 40; ai[..., 1] = -5; ai[..., 2] = 20          # background (pavement)
    shade = 30 + 30 * xx / W
    tex = 1.5 * np.sin(xx * 0.9) * np.cos(yy * 0.7)              # fine texture / shading variation
    f = lab == 0
    ai[f, 0] = (shade + tex)[f] * 57 / 45
    ai[f, 1] = 51 * (ai[f, 0] / 57) + cast_ab[0]
    ai[f, 2] = 47 * (ai[f, 0] / 57) + cast_ab[1]
    wd = lab == 1
    ai[wd, 0] = (shade + tex)[wd] * 0.5
    ai[wd, 1] = wood_ai_ab[0] * ai[wd, 0] / 22
    ai[wd, 2] = wood_ai_ab[1] * ai[wd, 0] / 22
    g = lab == 2
    ai[g, 0] = 35; ai[g, 1] = 4; ai[g, 2] = -14                  # glass shows a blue reflection
    weight = ((lab >= 0)).astype(np.float32)
    return _lab_img(ai), _lab_img(model), lab, weight


def _err(img, model, lab, idx, cast):
    valid = MC.erode(lab >= 0, 0) if hasattr(MC, "erode") else (lab >= 0)
    m = MC.measure(img, model, lab, MATS, lab >= 0, cast=cast)
    return m["materials"][MATS[idx]["name"]]


class PaletteTest(unittest.TestCase):
    def test_palette_roundtrip_and_background(self):
        lab = np.array([[-1, 0, 1, 2], [7, 50, 214, -1]], np.int32)
        rgba = np.zeros(lab.shape + (4,), np.uint8)
        for (y, x), v in np.ndenumerate(lab):
            if v >= 0:
                rgba[y, x, :3] = MC.palette_colour(v); rgba[y, x, 3] = 255
        out = MC.decode_ids(rgba, 215)
        np.testing.assert_array_equal(out, lab)
        # unknown index (beyond the material count) and off-palette colours -> -1
        self.assertEqual(MC.decode_ids(rgba, 3)[1, 1], -1)
        rgba[0, 1, :3] = (25, 128, 77)
        self.assertEqual(MC.decode_ids(rgba, 215)[0, 1], -1)
        self.assertEqual(len({MC.palette_colour(i) for i in range(MC.MAX_MATERIALS)}), MC.MAX_MATERIALS)
        self.assertNotIn((0, 0, 0), {MC.palette_colour(i) for i in range(MC.MAX_MATERIALS)})

    def test_srgb_linear_inverse(self):
        for v in (0.0, 0.2, 0.6, 1.0):
            lin = MC.srgb_to_linear_scalar(v)
            back = 1.055 * lin ** (1 / 2.4) - 0.055 if lin > 0.0031308 else lin * 12.92
            self.assertAlmostEqual(back, v, places=5)

    def test_resize_and_warp_labels_are_nearest(self):
        lab = _labels()
        small = MC.resize_labels(lab, W // 2, H // 2)
        self.assertEqual(set(np.unique(small)) - set(np.unique(lab)), set())
        moved = MC.warp_labels(lab, 1.0, 5.0, 0.0, (W / 2, H / 2))
        np.testing.assert_array_equal(moved[:, 70], lab[:, 65])
        self.assertTrue((moved[:, :5] == -1).all())


class KindTest(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(MC.material_kind("Windows", glass=True), "glass")
        self.assertEqual(MC.material_kind("CAR_L", emissive=True), "emissive")
        self.assertEqual(MC.material_kind("Mirrors", metallic=1.0, roughness=0.0), "mirror")
        self.assertEqual(MC.material_kind("LIGHT_CHROME_SH"), "mirror")
        self.assertEqual(MC.material_kind("CarPaint"), "reflective")
        self.assertEqual(MC.material_kind("车漆"), "reflective")
        self.assertEqual(MC.material_kind("RIM", metallic=0.75, roughness=0.16), "reflective")
        self.assertEqual(MC.material_kind("fabric Peacock Velvet", matte=True, metallic=1.0), "matte")
        self.assertEqual(MC.material_kind("tyre_3"), "default")

    def test_lab_roundtrip(self):
        rng = np.random.default_rng(1)
        rgb = rng.random((40, 40, 3)).astype(np.float32)
        self.assertLess(np.abs(MC.lab_to_srgb(MC.srgb_to_lab(rgb)) - rgb).max(), 2e-3)


class CorrectTest(unittest.TestCase):
    def setUp(self):
        self.ai, self.model, self.lab, self.weight = _scene()

    def test_wood_pulled_to_model_fabric_and_cast_kept(self):
        cast = MC.measure(self.ai, self.model, self.lab, MATS, self.lab >= 0)["cast"]
        self.assertAlmostEqual(cast["da"], -2.0, delta=1.0)    # fabric defines the scene cast
        res = MC.correct_colours(self.ai, self.model, self.lab, MATS, self.weight, strength=1.0)
        before = _err(self.ai, self.model, self.lab, 1, cast)
        after = _err(res["image"], self.model, self.lab, 1, cast)
        self.assertGreater(before["dEab"], 15)
        self.assertLess(after["dEab"], before["dEab"] * 0.35)
        fab_b = _err(self.ai, self.model, self.lab, 0, cast)["dEab"]
        fab_a = _err(res["image"], self.model, self.lab, 0, cast)["dEab"]
        self.assertLess(abs(fab_a - fab_b), 0.5)
        names = {r["name"]: r for r in res["regions"]}
        self.assertEqual(names["Windows"].get("skipped"), "glass")

    def test_lighting_gradient_and_texture_preserved(self):
        res = MC.correct_colours(self.ai, self.model, self.lab, MATS, self.weight, strength=0.7)
        wd = MC.erode(self.lab == 1, 2)
        L0 = MC.srgb_to_lab(self.ai)[..., 0][wd]; L1 = MC.srgb_to_lab(res["image"])[..., 0][wd]
        self.assertGreater(np.corrcoef(L0, L1)[0, 1], 0.999)
        # factors, not offsets: no wood pixel is pushed through grey into the complementary hue
        lab1 = MC.srgb_to_lab(res["image"])[wd]
        hue = np.degrees(np.arctan2(lab1[:, 2], lab1[:, 1]))
        self.assertTrue(((hue > -10) & (hue < 120)).all())

    def test_background_glass_and_zero_weight_untouched(self):
        w = self.weight.copy(); w[:, 200:] = 0           # e.g. the outer AI edge
        res = MC.correct_colours(self.ai, self.model, self.lab, MATS, w, strength=1.0)
        out = res["image"]
        np.testing.assert_array_equal(out[self.lab < 0], self.ai[self.lab < 0])
        np.testing.assert_array_equal(out[:, 200:], self.ai[:, 200:])
        g = MC.erode(self.lab == 2, 3)
        np.testing.assert_array_equal(out[g], self.ai[g])

    def test_strength_zero_identity_and_monotone(self):
        r0 = MC.correct_colours(self.ai, self.model, self.lab, MATS, self.weight, strength=0.0)
        np.testing.assert_array_equal(r0["image"], self.ai)
        cast = MC.measure(self.ai, self.model, self.lab, MATS, self.lab >= 0)["cast"]
        errs = [_err(MC.correct_colours(self.ai, self.model, self.lab, MATS, self.weight, strength=s)["image"],
                     self.model, self.lab, 1, cast)["dEab"] for s in (0.3, 0.7, 1.0)]
        self.assertTrue(errs[0] > errs[1] > errs[2])

    def test_reflective_gentle_and_protect(self):
        mats = [dict(m) for m in MATS]; mats[1]["kind"] = "reflective"
        full = MC.correct_colours(self.ai, self.model, self.lab, MATS, self.weight, strength=1.0)
        gentle = MC.correct_colours(self.ai, self.model, self.lab, mats, self.weight, strength=1.0)
        wd = MC.erode(self.lab == 1, 3)
        d_full = np.abs(full["image"] - self.ai)[wd].mean()
        d_gentle = np.abs(gentle["image"] - self.ai)[wd].mean()
        self.assertLess(d_gentle, 0.5 * d_full)
        prot = np.zeros((H, W), np.float32); prot[130:180, 100:150] = 1
        res = MC.correct_colours(self.ai, self.model, self.lab, MATS, self.weight, strength=1.0, protect=prot)
        np.testing.assert_array_equal(res["image"][135:175, 105:145], self.ai[135:175, 105:145])

    def test_shift_is_capped(self):
        ai, model, lab, weight = _scene(wood_ai_ab=(-30.0, -35.0))     # wildly wrong (teal) wood
        res = MC.correct_colours(ai, model, lab, MATS, weight, strength=1.0)
        r = {x["name"]: x for x in res["regions"]}["wood Brown"]
        self.assertLessEqual(np.hypot(r["shift"][1], r["shift"][2]), MC.MAX_AB_DESAT + 1e-6)
        self.assertLessEqual(abs(r["hue_deg"]), MC.MAX_HUE_DEG + 1e-6)
        plan = MC.plan_shift(np.array([30.0, 2.0, 3.0]), np.array([30.0, 40.0, 45.0]))
        self.assertLessEqual(np.exp(plan["log_c"]), MC.CHROMA_RANGE[1] + 1e-6)   # never made garish
        self.assertLessEqual(np.hypot(plan["shift"][1], plan["shift"][2]), MC.MAX_AB_SHIFT + 1e-6)

    def test_small_regions_skipped(self):
        lab = np.full((H, W), -1, np.int32); lab[50:58, 50:58] = 1; lab[100:190, 20:280] = 0
        res = MC.correct_colours(self.ai, self.model, lab, MATS, (lab >= 0).astype(np.float32), 1.0)
        self.assertEqual({r["name"]: r for r in res["regions"]}["wood Brown"].get("skipped"), "too_small")


class PasteIntegrationTest(unittest.TestCase):
    def _inputs(self):
        ai, model, lab, _w = _scene()
        mask = (lab >= 0).astype(np.float32)
        detail = np.dstack([model, mask])
        return ai, detail, mask, lab

    def test_paste_with_colour_fix_keeps_edge_and_background(self):
        ai, detail, mask, lab = self._inputs()
        fit = {"scale_x": 1.0, "scale_y": 1.0, "dx": 0.0, "dy": 0.0}
        base = DP.paste_details_full(ai, detail, mask, fit=fit)
        off = DP.paste_details_full(ai, detail, mask, fit=fit, matid=(lab, MATS), colour_strength=0.0)
        np.testing.assert_array_equal(base["image"], off["image"])
        on = DP.paste_details_full(ai, detail, mask, fit=fit, matid=(lab, MATS), colour_strength=0.7)
        hard = mask > 0.5
        r_e = DP.paste_weight(hard, max(H, W))[1]          # ≈4 px at 1280 px, 2 px on this canvas
        band = hard & ~DP.erode(hard, r_e)
        np.testing.assert_array_equal(on["image"][band], base["image"][band])
        np.testing.assert_array_equal(on["image"][~hard], ai[~hard])
        self.assertGreater(on["colour"]["changed_px"], 1000)
        self.assertIn("按材质校色", DP.summarize(dict(on, seconds=1.0)))

    def test_paste_file_reads_matid_pass_and_sidecar(self):
        ai, detail, mask, lab = self._inputs()
        u8 = lambda x: (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
        ids = np.zeros((H, W, 4), np.uint8)
        for i in range(3):
            ids[lab == i, :3] = MC.palette_colour(i); ids[lab == i, 3] = 255
        with tempfile.TemporaryDirectory() as d:
            PNG.write_png(f"{d}/ai.png", u8(ai)); PNG.write_png(f"{d}/detail.png", u8(detail))
            m = np.zeros((H, W, 4), np.uint8); m[..., 3] = u8(mask); PNG.write_png(f"{d}/mask.png", m)
            PNG.write_png(f"{d}/id.png", ids)
            Path(MC.sidecar_path(f"{d}/id.png")).write_text(json.dumps({"materials": MATS}), "utf-8")
            # 1.2.3: the colour fix only runs when the region paste kept something; here there is
            # no small detail part, so the paste (and with it the colour fix) is discarded
            r_off = DP.paste_file(f"{d}/ai.png", f"{d}/detail.png", f"{d}/mask.png", f"{d}/o1.png")
            r_on = DP.paste_file(f"{d}/ai.png", f"{d}/detail.png", f"{d}/mask.png", f"{d}/o2.png",
                                 matid_path=f"{d}/id.png", colour_strength=0.7)
            self.assertEqual(r_off.get("colour"), {})
            self.assertEqual(r_on.get("colour"), {})
            self.assertTrue(r_on["discarded"])
            self.assertFalse(Path(f"{d}/o2.png").exists())
            # missing sidecar -> no colour fix, no error
            r_none = DP.paste_file(f"{d}/ai.png", f"{d}/detail.png", f"{d}/mask.png", f"{d}/o3.png",
                                   matid_path=f"{d}/missing.png", colour_strength=0.7)
            self.assertEqual(r_none.get("colour"), {})
            json.dumps(r_on, ensure_ascii=False)    # report stays JSON-able


class WiringTest(unittest.TestCase):
    def test_props_ui_operator_capture(self):
        props = (PKG / "properties.py").read_text("utf-8")
        ui = (PKG / "ui.py").read_text("utf-8")
        ops = (PKG / "operators.py").read_text("utf-8")
        vc = (PKG / "viewport_capture.py").read_text("utf-8")
        self.assertRegex(props, r"detail_colour_fix: BoolProperty\([^)]*default=True")
        self.assertRegex(props, r"detail_colour_strength: FloatProperty\([^)]*default=0\.7")
        # under 高级 › 细节回贴 only
        start = ui.index("def _draw_detail_paste(")
        body = ui[start:ui.index("\ndef ", start + 10)]
        self.assertIn('"detail_colour_fix", text="按材质校色"', body)
        self.assertIn('"detail_colour_strength", text="强度"', body)
        self.assertEqual(ui.count("detail_colour_fix"), body.count("detail_colour_fix"))
        self.assertIn('"01_material_id.png"', ops)
        self.assertIn('matid_path=(self._detail or {}).get("matid", "")', ops)
        self.assertIn('matid_path=self._detail.get("matid", "")', ops)
        self.assertIn("colour_strength=float(spec.get(\"colour_strength\", 0.0) or 0.0)", ops)
        self.assertIn("def _render_matid(", vc)
        self.assertIn("filter_width = 0.01", vc)
        self.assertIn("dither_intensity = 0.0", vc)

    def test_spec_colour_off(self):
        src = (PKG / "operators.py").read_text("utf-8")
        start = src.index("    def _detail_spec(")
        end = src.index("    def _apply_detail_paste(")
        ns = {"Path": Path, "print": lambda *a, **k: None}
        exec("class Host:\n" + src[start:end], ns)
        host = ns["Host"]()
        host._session = types.SimpleNamespace(viewport_reference="/tmp/s/00.png", directory=Path("/tmp/s"))
        fake_vc = types.ModuleType("wondful_ai_renderer.viewport_capture")
        fake_vc.keep_exact_slots = lambda coll: {}
        fake_vc.keep_exact_texts = lambda coll: []
        fake_vc.mirrored_keep_parts = lambda coll, cam: []
        saved = sys.modules.get("wondful_ai_renderer.viewport_capture")
        sys.modules["wondful_ai_renderer.viewport_capture"] = fake_vc
        try:
            ns["__package__"] = "wondful_ai_renderer"
            exec("class Host:\n" + src[start:end], ns)
            host = ns["Host"]()
            host._session = types.SimpleNamespace(viewport_reference="/tmp/s/00.png", directory=Path("/tmp/s"))
            on = types.SimpleNamespace(detail_paste=True, last_detail_note="", product_collection=object(),
                                       detail_paste_strength=0.8, detail_colour_fix=True, detail_colour_strength=0.7)
            spec = host._detail_spec(None, on)
            self.assertTrue(spec["matid"].endswith("01_material_id.png"))
            self.assertAlmostEqual(spec["colour_strength"], 0.7)
            off = types.SimpleNamespace(**dict(vars(on), detail_colour_fix=False))
            spec = host._detail_spec(None, off)
            # 1.2.3: the ID pass is still rendered (part classes for the region paste); no colour fix
            self.assertTrue(spec["matid"].endswith("01_material_id.png"))
            self.assertEqual(spec["colour_strength"], 0.0)
        finally:
            if saved is not None:
                sys.modules["wondful_ai_renderer.viewport_capture"] = saved
            else:
                sys.modules.pop("wondful_ai_renderer.viewport_capture", None)


if __name__ == "__main__":
    unittest.main()
