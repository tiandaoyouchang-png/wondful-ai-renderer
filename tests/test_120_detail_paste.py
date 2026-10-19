"""1.2.0 细节回贴: frequency blend, alignment, keep-exact masks, toggle-off wiring."""
import importlib
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


DP = _load("detail_paste")
PNG = _load("png_io")


def _ellipse(h, w, cx, cy, rx, ry):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0).astype(np.float32)


def _scene(shift=(0, 0), h=240, w=360):
    """AI image: product (ellipse, smooth colour gradient = 'lighting', wrong fine stripes)
    on a dark background. Detail pass: same product at the mask position with the right
    fine checker texture under flat light."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = _ellipse(h, w, 180, 120, 120, 70)
    ai_mask = _ellipse(h, w, 180 + shift[0], 120 + shift[1], 120, 70)
    light = np.dstack([0.25 + 0.5 * xx / w, 0.3 + 0.2 * yy / h, 0.6 - 0.3 * xx / w])
    wrong = 0.08 * np.sin(xx * 1.3)[..., None]                    # AI's redrawn fine detail (vertical stripes)
    bg = np.dstack([np.full((h, w), 0.1), np.full((h, w), 0.12), np.full((h, w), 0.15)])
    ai = np.where(ai_mask[..., None] > 0.5, np.clip(light + wrong, 0, 1), bg).astype(np.float32)
    checker = (((xx // 3) + (yy // 3)) % 2).astype(np.float32)
    det = np.dstack([0.5 + 0.2 * (checker - 0.5)] * 3)
    detail = np.dstack([det, mask]).astype(np.float32)
    return ai, detail, mask, checker, light


def _hp(img, r=4):
    return img - DP.blur(img, r, 2)


class NamesTest(unittest.TestCase):
    def test_heuristic_names(self):
        for name in ("carPlate_front", "License_Plate", "BonnetLogoFont", "Badge.001", "Emblem", "前车牌", "车标", "铭牌"):
            self.assertTrue(DP.name_looks_keep_exact(name), name)
        for name in ("CarPaint", "Body", "Template_mesh", "plateau_rock", "Seat", ""):
            self.assertFalse(DP.name_looks_keep_exact(name), name)

    def test_classify_explicit_wins_and_false_switches_off(self):
        self.assertEqual(DP.classify_slot("Body", 1, "Paint", None), "explicit")
        self.assertEqual(DP.classify_slot("Body", None, "Paint", True), "explicit")
        self.assertEqual(DP.classify_slot("carPlate", None, "CAR_L", None), "auto")
        self.assertEqual(DP.classify_slot("Body", None, "logo_decal", None), "auto")
        self.assertEqual(DP.classify_slot("carPlate", 0, "CAR_L", None), "")
        self.assertEqual(DP.classify_slot("Body", None, "Paint", None), "")

    def test_matte_names(self):
        for name in ("fabric Mystere Mango Velvet", "wood Brown", "Leather_Seat", "绒布"):
            self.assertTrue(DP.name_looks_matte(name), name)
        self.assertFalse(DP.name_looks_matte("CarPaint"))
        self.assertFalse(DP.name_looks_matte("tyre_3"))  # wet tyres show sharp highlights

    def test_glass_names(self):
        self.assertTrue(DP.name_looks_glass("Windows"))
        self.assertTrue(DP.name_looks_glass("Light_Glass"))
        self.assertFalse(DP.name_looks_glass("CarPaint"))

    def test_keep_prompt(self):
        self.assertIn("车牌", DP.keep_prompt(True))
        self.assertIn("一字不改", DP.keep_prompt(False))
        self.assertNotIn("车牌", DP.keep_prompt(False))
        self.assertIn("「XIAOMI YU7」", DP.keep_prompt(True, ["XIAOMI YU7"]))

    def test_prompt_carries_keep_text(self):
        CD = _load("codex_direct")
        text = CD.build_simple_prompt("城市街道", vehicle=True, keep_text=DP.keep_prompt(True, ["XIAOMI YU7"]))
        self.assertIn("XIAOMI YU7", text)
        self.assertNotIn("一字不改", CD.build_simple_prompt("城市街道", vehicle=True))


class KeepMaskTest(unittest.TestCase):
    def _rgba(self, h=100, w=100):
        return np.zeros((h, w, 4), np.uint8)

    def test_merge_channels(self):
        prod = np.zeros((100, 100), np.float32); prod[10:90, 10:90] = 1
        k = self._rgba()
        k[20:30, 20:40] = (255, 0, 0, 255)      # explicit
        k[50:60, 20:40] = (0, 255, 0, 255)      # automatic
        k[70:80, 20:40] = (0, 0, 255, 255)      # glass
        k[80:85, 20:40] = (0, 0, 0, 255)        # matte (fabric / wood)
        keep, glass, matte, info = DP.merge_keep_masks(k, prod)
        self.assertEqual(int((keep > 0.5).sum()), 400)
        self.assertEqual(int((glass > 0.5).sum()), 200)
        self.assertEqual(int((matte > 0.5).sum()), 100)
        self.assertTrue(info["auto_used"])
        self.assertEqual(info["explicit_px"], 200)

    def test_auto_dropped_when_it_is_most_of_the_product(self):
        prod = np.zeros((100, 100), np.float32); prod[10:90, 10:90] = 1
        k = self._rgba()
        k[10:90, 10:90] = (0, 255, 0, 255)      # a product whose name says "plate"
        k[20:25, 20:25] = (255, 0, 0, 255)      # but explicit marks still count
        keep, _glass, _matte, info = DP.merge_keep_masks(k, prod)
        self.assertFalse(info["auto_used"])
        self.assertEqual(int((keep > 0.5).sum()), 25)

    def test_keep_clipped_to_product(self):
        prod = np.zeros((50, 50), np.float32); prod[10:40, 10:40] = 1
        k = self._rgba(50, 50)
        k[0:20, 0:20] = (255, 0, 0, 255)
        keep, _g, _m, _i = DP.merge_keep_masks(k, prod)
        self.assertEqual(float(keep[5, 5]), 0.0)
        self.assertEqual(float(keep[15, 15]), 1.0)


class BlendTest(unittest.TestCase):
    """Legacy whole-product blend (``paste_details_full``, unused by the plugin since 1.2.3)."""
    def test_frequency_split_low_from_ai_high_from_detail(self):
        ai, detail, mask, checker, light = _scene()
        res = DP.paste_details_full(ai, detail, mask, strength=1.0, fit={"scale_x": 1.0, "scale_y": 1.0, "dx": 0.0, "dy": 0.0})
        out = res["image"]
        inner = DP.erode(mask > 0.5, 25)
        # low frequencies (the 'lighting' gradient) stay the AI's
        lo_err = np.abs(DP.blur(out, 8, 2) - DP.blur(ai, 8, 2))[inner].mean()
        self.assertLess(lo_err, 0.02)
        # high frequencies follow the detail pass's checker, not the AI's stripes
        hp_out = DP.luminance(_hp(out))[inner]
        corr_detail = np.corrcoef(hp_out, _hp(checker)[inner])[0, 1]
        corr_ai = np.corrcoef(hp_out, DP.luminance(_hp(ai))[inner])[0, 1]
        self.assertGreater(corr_detail, 0.8)
        self.assertLess(corr_ai, 0.3)

    def test_strength_zero_is_identity(self):
        ai, detail, mask, _c, _l = _scene()
        res = DP.paste_details_full(ai, detail, mask, strength=0.0, fit={"scale_x": 1.0, "scale_y": 1.0, "dx": 0.0, "dy": 0.0})
        self.assertLess(np.abs(res["image"] - ai).max(), 1e-4)

    def test_flat_detail_keeps_ai_reflections(self):
        h, w = 200, 300
        mask = _ellipse(h, w, 150, 100, 110, 70)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        streak = (np.abs(xx - yy - 40) < 2).astype(np.float32)    # a sharp neon reflection line
        ai = np.dstack([0.2 + 0.7 * streak, 0.1 + 0.2 * streak, 0.3 + 0.6 * streak]).astype(np.float32)
        detail = np.dstack([np.full((h, w, 3), 0.5, np.float32), mask])   # flat paint in the detail pass
        res = DP.paste_details_full(ai, detail, mask, strength=0.8, fit={"scale_x": 1.0, "scale_y": 1.0, "dx": 0.0, "dy": 0.0})
        inner = DP.erode(mask > 0.5, 20)
        self.assertLess(np.abs(res["image"] - ai)[inner].max(), 0.02)

    def test_matte_parts_take_detail_texture_even_where_flat(self):
        h, w = 200, 300
        mask = _ellipse(h, w, 150, 100, 110, 70)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        crumple = 0.08 * np.sin(xx * 0.9) * np.sin(yy * 1.1)                  # AI-invented fabric texture
        ai = np.dstack([0.7 + crumple, 0.35 + crumple, 0.25 + crumple]).astype(np.float32)
        detail = np.dstack([np.full((h, w, 3), 0.5, np.float32), mask])        # smooth velvet in Blender
        fit = {"scale_x": 1.0, "scale_y": 1.0, "dx": 0.0, "dy": 0.0}
        keep = np.zeros((h, w, 4), np.uint8); keep[..., 3] = (mask * 255).astype(np.uint8)   # all matte
        inner = DP.erode(mask > 0.5, 20)
        plain = DP.paste_details_full(ai, detail, mask, strength=1.0, fit=fit)
        matte = DP.paste_details_full(ai, detail, mask, keep_rgba=keep, strength=1.0, fit=fit)
        self.assertGreater(np.abs(_hp(DP.luminance(plain["image"])))[inner].mean(), 0.03)  # glossy rule keeps it
        self.assertLess(np.abs(_hp(DP.luminance(matte["image"])))[inner].mean(), 0.01)     # matte rule removes it
        self.assertEqual(matte["keep"]["matte_px"], int((mask > 0.5).sum()))

    def test_no_halo_silhouette_stays_ai(self):
        ai, detail, mask, _c, _l = _scene()
        detail[..., :3] = 1.0 - detail[..., :3]       # very different detail pass
        res = DP.paste_details_full(ai, detail, mask, strength=1.0, fit={"scale_x": 1.0, "scale_y": 1.0, "dx": 0.0, "dy": 0.0})
        edge = (mask > 0.5) & ~DP.erode(mask > 0.5, 2)
        outside = mask < 0.5
        self.assertLess(np.abs(res["image"] - ai)[edge].max(), 1e-4)
        self.assertLess(np.abs(res["image"] - ai)[outside].max(), 1e-6)

    def test_keep_exact_region_replaced_with_colour_match(self):
        ai, detail, mask, _c, _l = _scene()
        h, w = mask.shape
        letters = np.zeros((h, w), np.float32); letters[110:130, 150:210:4] = 1   # "text" on the plate
        detail[..., :3] = np.where(letters[..., None] > 0, 0.05, 0.9)
        keep = np.zeros((h, w, 4), np.uint8); keep[105:135, 145:215] = (255, 0, 0, 255)
        res = DP.paste_details_full(ai, detail, mask, keep_rgba=keep, strength=0.8,
                               fit={"scale_x": 1.0, "scale_y": 1.0, "dx": 0.0, "dy": 0.0})
        out = DP.luminance(res["image"])
        self.assertGreater(res["keep"]["pasted_px"], 1500)
        # letters dark, plate background light: the detail pass's text pattern is there
        self.assertLess(out[112:128, 150:210:4].mean(), out[112:128, 152:210:4].mean() - 0.3)

    def test_tiny_badge_not_replaced_outright(self):
        ai, detail, mask, _c, _l = _scene()
        keep = np.zeros(mask.shape + (4,), np.uint8); keep[120:123, 180:183] = (0, 255, 0, 255)   # 3×3 px logo
        res = DP.paste_details_full(ai, detail, mask, keep_rgba=keep, strength=0.8,
                               fit={"scale_x": 1.0, "scale_y": 1.0, "dx": 0.0, "dy": 0.0})
        self.assertEqual(res["keep"]["pasted_px"], 0)

    def test_colour_match_follows_cast(self):
        src = np.full((10, 10, 3), 0.5, np.float32)
        ref = np.dstack([np.full((10, 10), 0.7), np.full((10, 10), 0.3), np.full((10, 10), 0.6)]).astype(np.float32)
        out = DP.colour_match(src, ref, np.ones((10, 10)), amount=1.0)
        self.assertGreater(out[..., 0].mean(), out[..., 1].mean() + 0.3)


class AlignTest(unittest.TestCase):
    def test_warp_image_shift(self):
        img = np.zeros((40, 40, 2), np.float32); img[10:20, 10:20] = 1
        out = DP.warp_image(img, 1.0, 5, 3, (20, 20))
        self.assertEqual(float(out[13:23, 15:25].min()), 1.0)
        self.assertEqual(float(out[11, 11, 0]), 0.0)

    def test_fit_aligns_detail_to_moved_product(self):
        ai, detail, mask, checker, _l = _scene(shift=(6, 4))
        res = DP.paste_details_full(ai, detail, mask, strength=1.0)
        fit = res["fit"]
        self.assertAlmostEqual(fit["dx"], 6, delta=1.5)
        self.assertAlmostEqual(fit["dy"], 4, delta=1.5)
        # the background next to the moved product stays the AI's: nothing pasted outside it
        ai_mask = _ellipse(*mask.shape, 186, 124, 120, 70)
        self.assertLess(np.abs(res["image"] - ai)[ai_mask < 0.5].max(), 1e-6)


    def test_wrong_outline_fit_is_rejected_by_inner_edges(self):
        ai, detail, mask, checker, _l = _scene()
        ai = ai.copy(); ai[..., :3] += 0.25 * (checker[..., None] - 0.5) * (mask[..., None] > 0.5)  # same inner detail
        bad = {"scale_x": 0.89, "scale_y": 0.88, "dx": 14.0, "dy": -8.0}
        al = DP.verify_alignment(ai, detail[..., :3], detail[..., 3], mask, bad)
        self.assertEqual(al["source"], "identity")
        self.assertTrue(al["ok"])
        self.assertLess(al["ncc_outline"], al["ncc_identity"])

    def test_unrelated_detail_pass_skips_paste(self):
        ai, detail, mask, _c, _l = _scene()
        rng = np.random.default_rng(3)
        noise = rng.random(mask.shape).astype(np.float32)
        detail = np.dstack([DP.blur(noise, 6, 2)[..., None].repeat(3, -1) * 0 + 0.5, mask])   # featureless
        res = DP.paste_details(ai, detail, mask, strength=1.0)
        self.assertIn("skipped", res)
        self.assertLess(np.abs(res["image"] - ai).max(), 1e-6)


class FileTest(unittest.TestCase):
    def test_paste_file_roundtrip_and_raw_path(self):
        """1.2.3: nothing small to paste here (no keep-exact part, no material-ID pass) -> the
        paste is discarded, no file is written and the report says the AI image was kept."""
        ai, detail, mask, checker, _l = _scene()
        ai = ai.copy(); ai[..., :3] += 0.1 * (checker[..., None] - 0.5) * (mask[..., None] > 0.5)
        with tempfile.TemporaryDirectory() as d:
            u8 = lambda x: (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
            PNG.write_png(f"{d}/ai.png", u8(ai))
            PNG.write_png(f"{d}/detail.png", u8(detail))
            m = np.zeros(mask.shape + (4,), np.uint8); m[..., 3] = u8(mask)
            PNG.write_png(f"{d}/mask.png", m)
            rep = DP.paste_file(f"{d}/ai.png", f"{d}/detail.png", f"{d}/mask.png", f"{d}/out.png")
            self.assertFalse(Path(f"{d}/out.png").exists())
            self.assertTrue(rep["discarded"])
            self.assertIn("细节回贴：已放弃", DP.summarize(rep))
        self.assertEqual(DP.raw_path("/x/Wondful_S_C_1.png"), "/x/Wondful_S_C_1_raw.png")


class WiringTest(unittest.TestCase):
    def test_properties_ui_operator_hooks(self):
        props = (PKG / "properties.py").read_text("utf-8")
        ui = (PKG / "ui.py").read_text("utf-8")
        ops = (PKG / "operators.py").read_text("utf-8")
        init = (PKG / "__init__.py").read_text("utf-8")
        vc = (PKG / "viewport_capture.py").read_text("utf-8")
        self.assertIn("detail_paste: BoolProperty", props)
        self.assertRegex(props, r"detail_paste: BoolProperty\([^)]*default=True")   # 1.2.3 细节修复: on again
        self.assertIn("detail_paste_strength: FloatProperty", props)
        self.assertIn("default=0.8", props)
        adv = ui.index("# ---- 高级")
        self.assertGreater(ui.rindex("_draw_detail_paste(adv, context, props, busy)"), adv)
        self.assertIn('"wondful.mark_keep_exact"', ui)
        self.assertIn("WONDFUL_OT_mark_keep_exact,", init)
        self.assertIn("detail_path=(self._detail or {}).get(\"detail\", \"\")", ops)
        self.assertIn("keep_text=direct_keep_text", ops)
        self.assertIn("self._apply_detail_paste(props, variant_final, variant_index)", ops)
        self.assertIn("raw_path(exported)", ops)
        # paste happens before export, so layers (_submit_layers) see the pasted result
        self.assertLess(ops.index("self._apply_detail_paste(props, variant_final, variant_index)"),
                        ops.index("self._submit_layers(props, Path(exported))"))
        self.assertIn("def render_detail_pass(", vc)
        self.assertIn("ShaderNodeHoldout", vc)

    def test_toggle_off_is_old_behaviour(self):
        """detail_paste off -> no spec -> variants pass through untouched (no _raw copy)."""
        src = (PKG / "operators.py").read_text("utf-8")
        start = src.index("    def _detail_spec(")
        end = src.index("    def _submit_layers(")
        ns = {"Path": Path, "print": lambda *a, **k: None}
        exec("class Host:\n" + src[start:end], ns)
        host = ns["Host"]()
        props = types.SimpleNamespace(detail_paste=False, last_detail_note="x", product_collection=object())
        host._session = types.SimpleNamespace(viewport_reference="/tmp/s/00.png", directory=Path("/tmp/s"))
        self.assertIsNone(host._detail_spec(None, props))
        host._detail = None
        self.assertEqual(host._apply_detail_paste(props, Path("/tmp/v.png"), 1), (Path("/tmp/v.png"), None))
        # missing passes also fall through unchanged
        host._detail = {"detail": "/nonexistent/d.png", "mask": "/nonexistent/m.png", "keep": "", "strength": 0.8,
                        "reports": []}
        self.assertEqual(host._apply_detail_paste(props, Path("/tmp/v.png"), 1), (Path("/tmp/v.png"), None))


if __name__ == "__main__":
    unittest.main()
