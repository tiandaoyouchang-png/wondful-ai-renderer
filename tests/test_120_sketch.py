"""1.2.0 草图模式: silhouettes, sheet split, visual-hull carving, surface nets, prompt + payload wiring."""
import base64
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402
from test_321_direct_edit import DirectEditTestBase, rgb_png  # noqa: E402

SM = load_package_module("sketch_model")
png_io = load_package_module("png_io")
CD = load_package_module("codex_direct")
PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"


def disc(n=300):
    y, x = np.mgrid[:n, :n]
    return ((x - n / 2 + 0.5) ** 2 + (y - n / 2 + 0.5) ** 2) < (n / 2) ** 2


def on_white(mask, colour=(0.2, 0.3, 0.6), pad=30, noise=0.0, seed=0):
    h, w = mask.shape
    a = np.ones((h + 2 * pad, w + 2 * pad, 4), np.float32)
    if noise:
        a[:, :, :3] += np.random.default_rng(seed).normal(0, noise, (h + 2 * pad, w + 2 * pad, 3))
    a[pad:pad + h, pad:pad + w, :3][mask] = colour
    return np.clip(a, 0, 1)


class ComponentTests(unittest.TestCase):
    def test_labels_and_sizes(self):
        m = np.zeros((20, 30), bool)
        m[2:5, 2:6] = True          # 12
        m[10:18, 10:20] = True      # 80
        m[10:12, 25:28] = True      # 6
        m[0:20, 8] = False
        labels, sizes = SM.label_components(m)
        self.assertEqual(sorted(sizes[1:]), [6, 12, 80])
        self.assertEqual(int(SM.largest_component(m).sum()), 80)
        # U shape is one component (runs merge through the bottom)
        u = np.zeros((10, 10), bool); u[1:9, 1] = u[1:9, 8] = True; u[8, 1:9] = True
        self.assertEqual(len(SM.label_components(u)[1]) - 1, 1)

    def test_fill_holes_size_limit(self):
        ring = np.ones((100, 100), bool)
        ring[40:43, 40:43] = False      # small hole → filled
        ring[60:90, 10:40] = False      # big opening → kept with max_hole
        self.assertTrue(SM.fill_holes(ring).all())
        part = SM.fill_holes(ring, max_hole=0.02)
        self.assertTrue(part[41, 41])
        self.assertFalse(part[75, 25])


class SilhouetteTests(unittest.TestCase):
    def test_alpha_path(self):
        m = disc(120)
        a = np.zeros((160, 160, 4), np.float32)
        a[20:140, 20:140, 3] = m
        a[:, :, :3] = 1.0  # white RGB everywhere: only alpha tells the shape
        mask, how = SM.silhouette_mask(a)
        self.assertEqual(how, "alpha")
        self.assertEqual(int(mask.sum()), int(m.sum()))

    def test_background_with_noise_and_speck(self):
        m = disc(200)
        a = on_white(m, noise=0.02)
        a[5:8, 5:8, :3] = 0.0  # dust speck in a corner: not the largest blob
        mask, how = SM.silhouette_mask(a)
        self.assertEqual(how, "background")
        crop, box = SM.crop_to_mask(mask)
        self.assertEqual(crop.shape, (200, 200))
        self.assertGreater((crop == m).mean(), 0.995)

    def test_line_drawing_is_filled(self):
        m = np.pad(disc(200), 5)
        outline = m & ~SM.erode(m, 3)
        mask, _how = SM.silhouette_mask(on_white(outline, colour=(0, 0, 0)))
        crop, _ = SM.crop_to_mask(mask)
        self.assertGreater(crop.mean(), 0.75)  # disc fill ratio π/4
        self.assertGreater((crop == m[5:-5, 5:-5]).mean(), 0.99)

    def test_light_product_parts_kept_but_real_opening_stays(self):
        m = np.ones((200, 300), bool)
        m[60:140, 80:220] = False  # a window through the product (16% of area)
        mask, _ = SM.silhouette_mask(on_white(m, colour=(0.1, 0.1, 0.1)))
        crop, _ = SM.crop_to_mask(mask)
        self.assertAlmostEqual(float(crop.mean()), float(m.mean()), places=2)

    def test_png_roundtrip_loader(self):
        with tempfile.TemporaryDirectory() as d:
            arr = (on_white(disc(64)) * 255).astype(np.uint8)
            png_io.write_png(Path(d) / "v.png", arr)
            rgba = SM.load_rgba(Path(d) / "v.png")
            self.assertEqual(rgba.shape, (124, 124, 4))
            crop, info = SM.view_silhouette(rgba)
            self.assertEqual(info["size"], [64, 64])


class CarveTests(unittest.TestCase):
    def test_box_volume_and_auto_size(self):
        v, q, info = SM.build_hull(np.ones((100, 300), bool), np.ones((100, 200), bool), None,
                                   known_axis="LENGTH", known_value=2.0, resolution=128)
        self.assertEqual(info["dims"], [3.0, 2.0, 1.0])
        self.assertAlmostEqual(info["volume_m3"], 6.0, delta=0.06)
        self.assertAlmostEqual(v[:, 2].min(), 0.0, places=6)          # sits on the ground
        self.assertAlmostEqual(v[:, 0].mean(), 0.0, delta=0.02)        # centred
        self.assertAlmostEqual(np.ptp(v[:, 1]), 2.0, delta=0.04)

    def test_known_height_and_manual(self):
        _v, _q, info = SM.build_hull(np.ones((50, 100), bool), np.ones((50, 150), bool),
                                     known_axis="HEIGHT", known_value=0.5, resolution=64)
        self.assertEqual(info["dims"], [1.0, 1.5, 0.5])
        _v, _q, info = SM.build_hull(np.ones((50, 100), bool), np.ones((50, 150), bool), size_mode="MANUAL",
                                     length=4.0, width=2.0, height=1.5, resolution=64)
        self.assertEqual(info["dims"], [2.0, 4.0, 1.5])

    def test_sphere_silhouettes_give_tricylinder(self):
        d = disc(400)
        v, q, info = SM.build_hull(d, d, d, dims=(1, 1, 1), resolution=128)
        ideal = (16 - 8 * 2 ** 0.5) * 0.5 ** 3   # Steinmetz tricylinder, r = 0.5
        self.assertAlmostEqual(info["volume_m3"], ideal, delta=0.02 * ideal)
        self.assertEqual(len(v) - len(q), 2)       # closed genus-0 quad mesh: V - F = 2
        _v2, _q2, info2 = SM.build_hull(d, d, None, dims=(1, 1, 1), resolution=128)
        self.assertAlmostEqual(info2["volume_m3"], 16 / 3 * 0.125, delta=0.02 * 16 / 3 * 0.125)  # bicylinder

    def test_side_orientation_front_is_minus_y(self):
        # wedge: tall at the front.  Side view drawn with the front on the LEFT.
        side = np.zeros((100, 200), bool)
        for c in range(200):
            side[int(c / 2):, c] = True
        front = np.ones((100, 100), bool)
        for side_front, img in (("LEFT", side), ("RIGHT", side[:, ::-1])):
            v, _q, _i = SM.build_hull(front, img, None, dims=(1, 2, 1), side_front=side_front, resolution=96)
            tall = v[v[:, 2] > 0.8]
            self.assertLess(tall[:, 1].mean(), -0.5, side_front)  # tall end at -Y (product front)

    def test_top_auto_rotation(self):
        front = np.ones((40, 40), bool)    # W:H 1:1
        side = np.ones((40, 120), bool)    # L:H 3:1
        top_sideways = np.ones((40, 120), bool)  # drawn lying sideways (rows = width)
        v, _q, info = SM.build_hull(front, side, top_sideways, known_axis="LENGTH", known_value=3.0, resolution=96)
        self.assertTrue(info["top_rotated"])
        self.assertAlmostEqual(info["volume_m3"], 3.0, delta=0.06)
        self.assertEqual(SM.auto_top_front(np.ones((120, 40), bool), 1 / 3.0)[1], False)

    def test_empty_intersection_raises(self):
        front = np.zeros((40, 40), bool); front[:, :5] = True; front[:, 35:] = True
        top = np.zeros((40, 40), bool); top[:, 15:25] = True
        with self.assertRaises(ValueError):
            SM.build_hull(front, np.ones((40, 40), bool), top, dims=(1, 1, 1), resolution=40)

    def test_resample_area_average(self):
        m = np.zeros((10, 10), bool); m[:, :4] = True
        r = SM.resample_mask(m, 4, 4)   # each output column covers 2.5 input columns
        self.assertTrue(np.allclose(r, [[1.0, 0.6, 0.0, 0.0]] * 4), r)


class SheetTests(unittest.TestCase):
    def sheet(self, rotate_top=False, extra=False):
        a = np.ones((520, 900, 4), np.float32)
        # front 160w × 120h, side 300w × 120h (same row), top 160w × 300h below the front (third angle)
        a[40:160, 40:200, :3] = 0.2
        a[40:160, 300:600, :3] = 0.2
        a[195:200, 300:600, :3] = 0.2        # a thin ground line under the side view: small blob, ignored
        if rotate_top:
            a[220:380, 300:600, :3] = 0.2   # top drawn sideways 300w × 160h
        else:
            a[200:500, 40:200, :3] = 0.2
        if extra:
            a[300:420, 650:850, :3] = 0.2   # a fourth view of similar size
        return a

    def test_split_l_layout(self):
        views, info = SM.split_sheet(self.sheet())
        self.assertEqual(info["views"], 3)
        self.assertEqual(views["front"][0].shape, (120, 160))
        self.assertEqual(views["side"][0].shape, (120, 300))
        self.assertEqual(views["top"][0].shape, (300, 160))
        self.assertFalse(info["top_rotated"])

    def test_split_rotated_top_then_hull(self):
        views, info = SM.split_sheet(self.sheet(rotate_top=True))
        self.assertTrue(info["top_rotated"])
        self.assertEqual(views["top"][0].shape, (160, 300))
        v, q, hinfo = SM.build_hull(views["front"][0], views["side"][0], views["top"][0], known_axis="LENGTH",
                                    known_value=3.0, resolution=96)
        self.assertTrue(hinfo["top_rotated"])
        W, L, H = hinfo["dims"]
        self.assertAlmostEqual(hinfo["volume_m3"], W * L * H, delta=0.03 * W * L * H)

    def test_rejects_unreliable_sheets(self):
        one = np.ones((300, 300, 4), np.float32); one[50:250, 50:250, :3] = 0.2
        with self.assertRaises(ValueError):
            SM.split_sheet(one)
        with self.assertRaises(ValueError):
            SM.split_sheet(self.sheet(extra=True))


class PromptTests(unittest.TestCase):
    def test_sketch_prompt(self):
        text = CD.build_simple_prompt("雪山清晨", sketch_views=["正视图", "侧视图", "俯视图"])
        self.assertIn("白模是根据三视图生成的粗略外形，保持位置、朝向、大小和透视", text)
        self.assertIn("外形细节、材质、颜色、Logo 以三视图为准，可修正粗模的块面和缺口", text)
        self.assertIn("图 2：正视图、图 3：侧视图、图 4：俯视图", text)
        self.assertNotIn("不增减、不改形任何部件", text)
        self.assertNotIn("固有色", text)
        plain = CD.build_simple_prompt("雪山清晨")
        self.assertIn("不增减、不改形任何部件", plain)
        self.assertNotIn("三视图", plain)
        car = CD.build_simple_prompt("", env_text="雪山。车身质感：清漆", vehicle=True, sketch_views=["正视图", "侧视图"])
        self.assertIn("车漆颜色和材质以三视图为准", car)
        self.assertIn("以白模（位置、朝向、大小）和三视图", car)


class SketchPayloadTests(DirectEditTestBase):
    def test_simple_path_sends_clay_plus_views_not_env(self):
        views = [(rgb_png(self.tmp / f"v{i}.png", 400, 300, (i * 40, 90, 90)), lab)
                 for i, lab in enumerate(["正视图", "侧视图", "俯视图"])]
        result = self.edit(simple_brief="海边黄昏", sketch_refs=views + [(self.product, "多余")])
        self.assertEqual(result["sketch_views"], ["正视图", "侧视图", "俯视图"])
        body = self.state.raw_bodies[-1]
        self.assertIn("rough proxy", body["instructions"])
        content = body["input"][0]["content"]
        images = [c for c in content if c["type"] == "input_image"]
        self.assertEqual(len(images), 4)                       # clay + 3 views, the env image is NOT attached
        self.assertEqual(base64.b64decode(images[0]["image_url"].split(",", 1)[1]), Path(self.camera).read_bytes())
        self.assertEqual(base64.b64decode(images[2]["image_url"].split(",", 1)[1]), Path(views[1][0]).read_bytes())
        self.assertIn("粗略外形", content[0]["text"])

    def test_simple_path_without_sketch_unchanged(self):
        self.edit(simple_brief="海边黄昏")
        body = self.state.raw_bodies[-1]
        self.assertEqual(body["instructions"], CD.INSTRUCTIONS)
        self.assertEqual(len([c for c in body["input"][0]["content"] if c["type"] == "input_image"]), 1)

    def test_non_simple_ignores_sketch_refs(self):
        result = self.edit(sketch_refs=[(self.product, "正视图")])
        self.assertEqual(result["sketch_views"], [])


class WiringTests(unittest.TestCase):
    def test_plugin_wiring(self):
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        self.assertIn('bl_idname = "wondful.sketch_build"', ops)
        self.assertIn("sketch_refs=sketch_refs or None", ops)
        init = (PKG / "__init__.py").read_text(encoding="utf-8")
        self.assertEqual(init.count("WONDFUL_OT_sketch_build"), 2)
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        adv = ui[ui.index("# ---- 高级"):]
        self.assertIn("_draw_sketch(adv, props, busy)", adv)
        self.assertIn("草图模式（概念图）", ui)
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        for name in ("sketch_mode", "sketch_front_path", "sketch_side_path", "sketch_top_path", "sketch_sheet_path",
                     "sketch_known_value", "sketch_resolution"):
            self.assertIn(f"{name}:", props)

    def test_only_numpy_stdlib(self):
        for name in ("sketch_model.py", "sketch_proxy.py"):
            src = (PKG / name).read_text(encoding="utf-8")
            for mod in ("PIL", "cv2", "scipy", "skimage", "trimesh"):
                self.assertNotIn(f"import {mod}", src)
                self.assertNotIn(f"from {mod}", src)
        self.assertNotIn("import bpy", (PKG / "sketch_model.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
