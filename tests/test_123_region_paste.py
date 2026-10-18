"""1.2.3 细节回贴 rework: default OFF, only small detail regions, lighting matched per region,
per-region A/B gate against the AI-only image, whole paste discarded when nothing survives."""
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

DP = load_package_module("detail_paste")
PNG = load_package_module("png_io")
PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"
FIT = {"scale": 1.0, "scale_x": 1.0, "scale_y": 1.0, "dx": 0.0, "dy": 0.0}

H, W = 360, 640
yy, xx = np.mgrid[0:H, 0:W]
PROD = (((xx - 320) / 260.0) ** 2 + ((yy - 180) / 140.0) ** 2 <= 1).astype(np.float32)   # ≈ 114 000 px
PLATE = (np.abs(xx - 320) < 26) & (np.abs(yy - 200) < 10)                                # ≈ 1 000 px (< 1.5 %)
LETTERS = PLATE & (np.abs(yy - 200) < 5) & ((xx // 3) % 3 == 0) & (np.abs(xx - 320) < 22)


def _scene(ai_plate, ai_ring, det_plate, det_ring, blur_ai=True):
    """AI image: product of brightness ``ai_ring`` with a plate (``ai_plate``) whose letters are
    smeared; detail pass: same geometry, flat light, sharp letters."""
    rng = np.random.default_rng(5)
    ai = np.full((H, W, 3), 0.5, np.float32)
    ai[PROD > 0] = ai_ring
    ai += (0.01 * rng.standard_normal((H, W)))[..., None].astype(np.float32)     # photo grain
    ai[PLATE] = ai_plate
    letters_ai = DP.blur(LETTERS.astype(np.float32), 2, 2) if blur_ai else LETTERS.astype(np.float32)
    ai = np.clip(ai - 0.5 * ai_plate * letters_ai[..., None] * PLATE[..., None], 0, 1)
    det = np.zeros((H, W, 4), np.float32)
    det[..., :3] = det_ring
    det[PLATE, :3] = det_plate
    det[LETTERS, :3] = 0.05
    det[..., 3] = PROD
    keep = np.zeros((H, W, 4), np.float32)
    keep[PLATE, 0] = 1
    keep[PLATE, 3] = 1
    return ai.astype(np.float32), det, keep


class DefaultOff(unittest.TestCase):
    def test_property_default_off_and_tooltip_one_line(self):
        props = (PKG / "properties.py").read_text("utf-8")
        m = re.search(r"detail_paste: BoolProperty\((.*?)\n    \)", props, re.S)
        self.assertIsNotNone(m)
        self.assertIn("default=False", m.group(1))
        desc = re.search(r'description="([^"]*)"', m.group(1))
        self.assertIsNotNone(desc)
        self.assertNotIn('"\n', m.group(1).split("description=")[1].split("default")[0].rstrip().rstrip(","))
        self.assertLess(len(desc.group(1)), 70)
        # 按材质校色 keeps its own default (on), it only acts inside 细节回贴
        self.assertRegex(props, r"detail_colour_fix: BoolProperty\([^)]*default=True")

    def test_operator_fallbacks_are_off(self):
        ops = (PKG / "operators.py").read_text("utf-8")
        self.assertNotIn('getattr(props, "detail_paste", True)', ops)
        self.assertIn('getattr(props, "detail_paste", False)', ops)

    def test_colour_fix_does_nothing_when_paste_off(self):
        src = (PKG / "operators.py").read_text("utf-8")
        start = src.index("    def _detail_spec(")
        end = src.index("    def _apply_detail_paste(")
        ns = {"Path": Path, "print": lambda *a, **k: None}
        exec("class Host:\n" + src[start:end], ns)
        host = ns["Host"]()
        host._session = types.SimpleNamespace(viewport_reference="/tmp/s/00.png", directory=Path("/tmp/s"))
        props = types.SimpleNamespace(detail_paste=False, last_detail_note="x", product_collection=object(),
                                      detail_colour_fix=True, detail_colour_strength=0.7)
        self.assertIsNone(host._detail_spec(None, props))     # no spec -> no paste, no colour fix, no _raw
        self.assertEqual(props.last_detail_note, "")
        # an unset property (fresh / old .blend) reads as off
        bare = types.SimpleNamespace(last_detail_note="", product_collection=object())
        self.assertIsNone(host._detail_spec(None, bare))


class RegionFilter(unittest.TestCase):
    def test_region_classes(self):
        c = DP.region_class
        for name, kind in (("tyre_3", "default"), ("Tire_Rubber", "default"), ("轮胎", "default"),
                           ("Windows", "glass"), ("D_Glass", "glass"), ("CarPaint", "reflective"),
                           ("Body_Panel", "default"), ("Mirrors", "mirror"), ("INT_Seat_Back", "reflective")):
            self.assertEqual(c(name, kind), "exclude", name)
        self.assertEqual(c("Yu7RIM_C1", "reflective"), "wheel")
        for name, kind in (("Light_Glass", "emissive"), ("LIGHT_CHROME_SH", "emissive"), ("CAR_L", "emissive"),
                           ("Caliper_C1", "default"), ("DISK", "reflective"), ("Grille_Text", "default"),
                           ("Badge", "default")):
            self.assertEqual(c(name, kind), "detail", name)
        self.assertEqual(c("Black_Mat", "reflective"), "")
        self.assertEqual(c("Plastic_SH", "default"), "")

    def test_large_region_rejected(self):
        big = (np.abs(xx - 320) < 70) & (np.abs(yy - 180) < 30)          # ≈ 8 400 px ≈ 7 % of the product
        ai, det, keep = _scene(0.8, 0.3, 0.85, 0.35)
        keep[...] = 0
        keep[big, 0] = 1
        keep[big, 3] = 1
        res = DP.paste_details(ai, det, PROD, keep_rgba=keep, fit=FIT)
        self.assertTrue(res["discarded"])
        self.assertEqual(res["regions"][0]["reason"], "区域过大")
        np.testing.assert_array_equal(res["image"], ai)

    def test_tyre_and_glass_never_pasted(self):
        ai, det, _keep = _scene(0.8, 0.3, 0.85, 0.35)
        tyre = (np.abs(xx - 200) < 20) & (np.abs(yy - 250) < 20)
        glass = (np.abs(xx - 420) < 20) & (np.abs(yy - 120) < 12)
        det[tyre, :3] = np.where(((xx[tyre] // 2) % 2 == 0)[:, None], 0.9, 0.1)   # strong structure
        det[glass, :3] = np.where(((yy[glass] // 2) % 2 == 0)[:, None], 0.9, 0.1)
        labels = np.where(PROD > 0, 0, -1).astype(np.int32)
        labels[tyre] = 1
        labels[glass] = 2
        mats = [{"index": 0, "name": "Plastic_SH", "kind": "default"},
                {"index": 1, "name": "tyre_3", "kind": "default"},
                {"index": 2, "name": "Windows", "kind": "glass"}]
        keep = np.zeros((H, W, 4), np.float32)
        keep[tyre | glass, 1] = 1                       # even if a name heuristic marked them keep-exact
        keep[tyre | glass, 3] = 1
        res = DP.paste_details(ai, det, PROD, keep_rgba=keep, fit=FIT, matid=(labels, mats))
        self.assertEqual(res["regions"], [])
        self.assertTrue(res["discarded"])
        np.testing.assert_array_equal(res["image"], ai)
        self.assertIn("已放弃", DP.summarize(dict(res, seconds=1.0)))


class Gate(unittest.TestCase):
    def test_good_paste_kept_only_inside_region(self):
        ai, det, keep = _scene(ai_plate=0.8, ai_ring=0.3, det_plate=0.85, det_ring=0.35)
        res = DP.paste_details(ai, det, PROD, keep_rgba=keep, fit=FIT)
        self.assertFalse(res["discarded"], res["regions"])
        self.assertEqual(res["kept"], 1)
        out = res["image"]
        # sharp letters now, background of the plate stays at the AI's brightness
        lum = DP.luminance
        self.assertLess(float(lum(out)[LETTERS].mean()), float(lum(ai)[LETTERS].mean()) - 0.1)
        bg = PLATE & ~DP.dilate(LETTERS, 2)
        self.assertLess(abs(float(np.median(lum(out)[bg])) - float(np.median(lum(ai)[bg]))), 0.05)
        # no halo: nothing outside the plate changes at all
        np.testing.assert_array_equal(out[~PLATE], ai[~PLATE])
        note = DP.summarize(dict(res, seconds=1.0))
        self.assertIn("细节回贴：保留 1 处 · 放弃 0 处", note)

    def test_brightness_mismatch_discarded(self):
        # night scene: the AI plate is dark; the flat-lit Blender plate is far brighter than the
        # clamped match can bring down -> the gate rejects it and the AI image is kept
        ai, det, keep = _scene(ai_plate=0.12, ai_ring=0.08, det_plate=0.95, det_ring=0.6)
        res = DP.paste_details(ai, det, PROD, keep_rgba=keep, fit=FIT)
        self.assertTrue(res["discarded"])
        self.assertEqual(res["kept"], 0)
        self.assertEqual(res["dropped"], 1)
        self.assertIn(res["regions"][0]["reason"], ("明暗不匹配", "边缘发灰 / 偏色", "接缝明显"))
        np.testing.assert_array_equal(res["image"], ai)
        self.assertEqual(DP.summarize(dict(res, seconds=1.0)), "细节回贴：已放弃，原图更自然")

    def test_gate_function_directly(self):
        ai, det, _keep = _scene(0.8, 0.3, 0.85, 0.35)
        ring = DP.dilate(PLATE, 4) & ~PLATE
        bad = ai.copy()
        bad[PLATE] = np.clip(bad[PLATE] + 0.35, 0, 1)                    # a bright sticker
        self.assertFalse(DP.gate_region(ai, bad, det[..., :3], PLATE, ring)["ok"])
        flat = ai.copy()
        flat[PLATE] = ai[PLATE].mean(0)                                  # detail wiped out -> flatter
        g = DP.gate_region(ai, flat, det[..., :3], PLATE, ring)
        self.assertFalse(g["ok"])
        self.assertEqual(g["reason"], "比原图更平")

    def test_ai_already_right_is_left_alone(self):
        ai, det, keep = _scene(0.85, 0.35, 0.85, 0.35, blur_ai=False)   # the AI drew the same sharp letters
        res = DP.paste_details(ai, det, PROD, keep_rgba=keep, fit=FIT)
        self.assertTrue(res["discarded"])
        self.assertEqual(res["regions"][0]["reason"], "AI 已画对")

    def test_region_near_silhouette_is_trimmed(self):
        ai, det, keep = _scene(0.8, 0.3, 0.85, 0.35)
        edge = (PROD > 0.5) & ~DP.erode(PROD > 0.5, 1)
        keep[edge, 0] = 1
        keep[edge, 3] = 1
        res = DP.paste_details(ai, det, PROD, keep_rgba=keep, fit=FIT)
        margin = max(2, int(round(DP.EDGE_MARGIN * W)))
        band = (PROD > 0.5) & ~DP.erode(PROD > 0.5, margin)
        np.testing.assert_array_equal(res["image"][band], ai[band])

    def test_paste_file_discarded_writes_nothing(self):
        ai, det, keep = _scene(ai_plate=0.12, ai_ring=0.08, det_plate=0.95, det_ring=0.6)
        u8 = lambda x: (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
        with tempfile.TemporaryDirectory() as d:
            PNG.write_png(f"{d}/ai.png", u8(ai))
            PNG.write_png(f"{d}/detail.png", u8(det))
            PNG.write_png(f"{d}/keep.png", u8(keep))
            m = np.zeros((H, W, 4), np.uint8); m[..., 3] = u8(PROD)
            PNG.write_png(f"{d}/mask.png", m)
            rep = DP.paste_file(f"{d}/ai.png", f"{d}/detail.png", f"{d}/mask.png", f"{d}/out.png",
                                keep_path=f"{d}/keep.png")
            self.assertTrue(rep.get("skipped"))
            self.assertFalse(Path(f"{d}/out.png").exists())
            self.assertEqual(rep["output"], "")
            ai2, det2, keep2 = _scene(0.8, 0.3, 0.85, 0.35)
            PNG.write_png(f"{d}/ai.png", u8(ai2)); PNG.write_png(f"{d}/detail.png", u8(det2))
            rep = DP.paste_file(f"{d}/ai.png", f"{d}/detail.png", f"{d}/mask.png", f"{d}/out.png",
                                keep_path=f"{d}/keep.png")
            self.assertFalse(rep.get("skipped"), rep.get("regions"))
            self.assertTrue(Path(f"{d}/out.png").is_file())
            import json
            json.dumps(rep, ensure_ascii=False, default=str)


class Wiring(unittest.TestCase):
    def test_last_render_report_shows_decision(self):
        ui = (PKG / "ui.py").read_text("utf-8")
        diag = ui[ui.index('_section(adv, "上次渲染"'):]
        diag = diag[:diag.index('_section(adv, "工具"')]
        self.assertIn("last_detail_note", diag)
        self.assertIn('detail_note.startswith("细节回贴：")', ui)

    def test_matid_always_requested_when_paste_on(self):
        ops = (PKG / "operators.py").read_text("utf-8")
        self.assertIn('"matid": str(work / "01_material_id.png"),', ops)
        self.assertIn('"matid": v["files"]["matid"], "strength"', ops)


if __name__ == "__main__":
    unittest.main()
