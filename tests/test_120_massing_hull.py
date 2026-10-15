"""1.2.0 环境体块 › 从三视图生成建筑体块: silhouettes, sheet split, visual-hull carving, surface nets, wiring.

The product is always the real 3D model; the visual hull only builds environment buildings."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

SM = load_package_module("sketch_model")
png_io = load_package_module("png_io")
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


class BuildingScaleTests(unittest.TestCase):
    def test_known_height_20m_building(self):
        front = np.ones((200, 80), bool)          # 80 wide × 200 tall elevation
        side = np.ones((200, 60), bool)
        v, q, info = SM.build_hull(front, side, None, known_axis="HEIGHT", known_value=20.0, resolution=64)
        W, L, H = info["dims"]
        self.assertAlmostEqual(H, 20.0, places=3)
        self.assertAlmostEqual(W, 8.0, places=2)
        self.assertAlmostEqual(L, 6.0, places=2)
        self.assertAlmostEqual(float(v[:, 2].min()), 0.0, places=5)   # sits on the ground
        self.assertLess(abs(float(v[:, 0].mean())), 0.3)             # centred on X


class WiringTests(unittest.TestCase):
    def test_product_sketch_mode_removed(self):
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        cd = (PKG / "codex_direct.py").read_text(encoding="utf-8")
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        for src in (ops, cd, ui, props):
            self.assertNotIn("sketch_refs", src)
            self.assertNotIn("sketch_mode", src)
            self.assertNotIn("草图模式", src)
        self.assertNotIn("rough proxy", cd)
        self.assertNotIn("product_collection =", (PKG / "massing_hull.py").read_text(encoding="utf-8"))
        self.assertFalse((PKG / "sketch_proxy.py").exists())

    def test_hull_wiring(self):
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        self.assertIn('bl_idname = "wondful.massing_hull_build"', ops)
        self.assertIn("massing_hull.build_block(context, props)", ops)
        init = (PKG / "__init__.py").read_text(encoding="utf-8")
        for cls in ("WONDFUL_OT_massing_hull_build", "WONDFUL_OT_block_ref_add", "WONDFUL_OT_block_ref_clear",
                    "WONDFUL_OT_block_select"):
            self.assertEqual(init.count(cls), 2, cls)
        hull = (PKG / "massing_hull.py").read_text(encoding="utf-8")
        self.assertIn("MS.ensure_collection(scene, props)", hull)       # into the massing collection
        self.assertIn("MS.set_refs(obj, paths)", hull)                  # three views become its 建筑参考
        self.assertIn("obj.location = (cur.x, cur.y, 0.0)", hull)       # 3D cursor, on the ground
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        self.assertIn("_draw_massing_hull(box, props, busy)", ui)
        self.assertIn("从三视图生成建筑体块", ui)
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        for name in ("hull_front_path", "hull_side_path", "hull_top_path", "hull_sheet_path", "hull_known_value",
                     "hull_resolution", "last_hull_note"):
            self.assertIn(f"{name}:", props)
        self.assertIn('hull_known_value: FloatProperty(name="已知尺寸", default=20.0', props)
        self.assertIn('default="HEIGHT"', props)

    def test_only_numpy_stdlib(self):
        for name in ("sketch_model.py", "massing_hull.py", "massing.py"):
            src = (PKG / name).read_text(encoding="utf-8")
            for mod in ("PIL", "cv2", "scipy", "skimage", "trimesh"):
                self.assertNotIn(f"import {mod}", src)
                self.assertNotIn(f"from {mod}", src)
        self.assertNotIn("import bpy", (PKG / "sketch_model.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
