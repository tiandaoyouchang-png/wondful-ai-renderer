"""1.2.3 环境体块: 「从三视图生成建筑体块」 at a placeholder cube + 建筑参考 label for 场景粗模 blocks.

Pure maths (placeholder box in its own yaw frame, uniform fit, bottom / centre placement) and the label run
anywhere; the real build (scene blockout fixture + a user collection + a rotated, scaled cube) runs in a
headless Blender subprocess (tests/hull_placeholder_runner.py, skipped when ``blender`` is not on PATH)."""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

MS = load_package_module("massing")
HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "wondful_ai_renderer"


def cube_corners(loc, dims, yaw_deg, bottom_origin=False):
    """World corners of a box object: centre ``loc`` (or bottom centre), ``dims``, rotated about Z."""
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    out = []
    for x in (-0.5, 0.5):
        for y in (-0.5, 0.5):
            for z in ((0.0, 1.0) if bottom_origin else (-0.5, 0.5)):
                px, py, pz = x * dims[0], y * dims[1], z * dims[2]
                out.append((loc[0] + c * px - s * py, loc[1] + s * px + c * py, loc[2] + pz))
    return out


def rot_matrix(yaw_deg, scale=(1, 1, 1)):
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    return [[c * scale[0], -s * scale[1], 0, 0], [s * scale[0], c * scale[1], 0, 0], [0, 0, scale[2], 0], [0, 0, 0, 1]]


def place(block_min, block_max, tf, local):
    """World position of a block-local point under ``fit_to_placeholder``'s transform."""
    s, a = tf["scale"], tf["yaw"]
    x, y, z = (v * s for v in local)
    return (tf["location"][0] + math.cos(a) * x - math.sin(a) * y,
            tf["location"][1] + math.sin(a) * x + math.cos(a) * y, tf["location"][2] + z)


class PlacementMathTests(unittest.TestCase):
    def test_yaw_ignores_scale(self):
        for deg in (0, 30, -75, 120, 179):
            self.assertAlmostEqual(math.degrees(MS.yaw_of(rot_matrix(deg, (3, 0.5, 2)))), deg, places=6)

    def test_box_in_own_frame(self):
        box = MS.placeholder_box(cube_corners((5, -3, 2), (8, 4, 6), 35), math.radians(35))
        self.assertAlmostEqual(box["centre"][0], 5, places=6)
        self.assertAlmostEqual(box["centre"][1], -3, places=6)
        self.assertAlmostEqual(box["bottom"], -1, places=6)
        for got, want in zip(box["dims"], (8, 4, 6)):
            self.assertAlmostEqual(got, want, places=6)

    def test_fit_uniform_inside_bottom_and_centre(self):
        box = MS.placeholder_box(cube_corners((5, -3, 2), (8, 4, 6), 35), math.radians(35))
        bmin, bmax = (-10, -5, 0), (10, 5, 30)        # 20 x 10 x 30 hull, centred, on z = 0
        tf = MS.fit_to_placeholder(bmin, bmax, box, fit=True)
        self.assertAlmostEqual(tf["scale"], 0.2, places=6)        # height limits: 6 / 30
        for got, lim in zip(tf["dims"], (8, 4, 6)):
            self.assertLessEqual(got, lim + 1e-9)
        self.assertAlmostEqual(tf["dims"][0] / tf["dims"][2], 20 / 30, places=6)   # proportions kept
        bottom = place(bmin, bmax, tf, (0, 0, 0))
        self.assertAlmostEqual(bottom[0], 5, places=6)
        self.assertAlmostEqual(bottom[1], -3, places=6)
        self.assertAlmostEqual(bottom[2], -1, places=6)
        self.assertAlmostEqual(math.degrees(tf["yaw"]), 35, places=6)
        self.assertTrue(tf["fitted"])

    def test_off_centre_mesh_is_recentred(self):
        box = MS.placeholder_box(cube_corners((0, 0, 0), (10, 10, 10), 90, bottom_origin=True), math.radians(90))
        bmin, bmax = (2, 4, 1), (6, 8, 5)            # 4 x 4 x 4, not at the origin, not on the ground
        tf = MS.fit_to_placeholder(bmin, bmax, box, fit=True)
        self.assertAlmostEqual(tf["scale"], 2.5, places=6)
        mid = place(bmin, bmax, tf, (4, 6, 1))
        self.assertAlmostEqual(mid[0], 0, places=6)
        self.assertAlmostEqual(mid[1], 0, places=6)
        self.assertAlmostEqual(mid[2], 0, places=6)

    def test_known_size_keeps_size(self):
        box = MS.placeholder_box(cube_corners((1, 2, 0), (2, 2, 2), -20, bottom_origin=True), math.radians(-20))
        tf = MS.fit_to_placeholder((-10, -5, 0), (10, 5, 30), box, fit=False)
        self.assertEqual(tf["scale"], 1.0)
        self.assertFalse(tf["fitted"])
        self.assertEqual([round(d, 6) for d in tf["dims"]], [20, 10, 30])
        p = place((-10, -5, 0), (10, 5, 30), tf, (0, 0, 0))
        self.assertAlmostEqual(p[0], 1, places=6)
        self.assertAlmostEqual(p[1], 2, places=6)
        self.assertAlmostEqual(p[2], 0, places=6)

    def test_flat_placeholder_rejected(self):
        box = MS.placeholder_box(cube_corners((0, 0, 0), (4, 4, 0), 0), 0.0)
        with self.assertRaises(ValueError):
            MS.fit_to_placeholder((-1, -1, 0), (1, 1, 2), box, fit=True)
        MS.fit_to_placeholder((-1, -1, 0), (1, 1, 2), box, fit=False)   # known size: only position


class LabelTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(MS.block_ref_label("建筑_03", 0, True), "建筑_03 · 来自场景参考图")
        self.assertEqual(MS.block_ref_label("建筑_03", 2, True), "建筑_03 · 2 张参考")
        self.assertEqual(MS.block_ref_label("Cube", 0, False), "Cube · 无参考")
        self.assertEqual(MS.block_ref_label("Cube", 1, False), "Cube · 1 张参考")

    def test_scene_blockout_detection(self):
        class Obj(dict):
            def __init__(self, colls=(), **kw):
                super().__init__(**kw)
                self.users_collection = list(colls)
        sb = {"wondful_scene_blockout": 1}
        self.assertTrue(MS.is_scene_blockout_block(Obj(wondful_sb_group=4)))
        self.assertTrue(MS.is_scene_blockout_block(Obj(wondful_sb_wrap=1)))
        self.assertTrue(MS.is_scene_blockout_block(Obj(colls=[sb])))
        self.assertFalse(MS.is_scene_blockout_block(Obj(colls=[{}], wondful_hull_block=1)))
        self.assertFalse(MS.is_scene_blockout_block(Obj()))
        self.assertEqual(MS.SB_GROUP_PROP, load_package_module("scene_blockout").GROUP_PROP)

    def test_ui_wiring(self):
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        self.assertIn("MS.block_ref_label(obj.name, n, MS.is_scene_blockout_block(obj))", ui)
        self.assertIn("选中一个占位立方体，新体块会放到它的位置并按它大小缩放", ui)
        hull = (PKG / "massing_hull.py").read_text(encoding="utf-8")
        self.assertIn("已放到占位体", hull)
        self.assertIn("ph.hide_render = True", hull)
        self.assertIn("ph.hide_viewport = True", hull)
        self.assertIn('fit = size_mode != "MANUAL"', hull)

    def test_massing_objects_without_bpy(self):
        class Obj:
            name, hide_render, data = "A", False, SimpleNamespace(materials=[])
        o = Obj()
        coll = SimpleNamespace(all_objects=[o, o])
        self.assertEqual(MS.massing_objects(coll), [o])


BLENDER = shutil.which("blender")


@unittest.skipUnless(BLENDER and os.environ.get("WONDFUL_SKIP_BLENDER_TESTS") != "1", "blender not on PATH")
class BlenderPlaceholderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hull_ph_")
        out = os.path.join(cls.tmp, "rep.json")
        args = [BLENDER, "-b", "--factory-startup", "--python", str(HERE / "hull_placeholder_runner.py"), "--",
                str(HERE / "fixtures" / "scene_blockout" / "desert_v3.json"), out, cls.tmp]
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        with open(out, encoding="utf-8") as fh:
            cls.rep = json.load(fh)

    def test_ran(self):
        self.assertTrue(self.rep.get("ok"), self.rep.get("error"))

    def test_placeholder_transform(self):
        a = self.rep["a"]
        self.assertAlmostEqual(a["yaw_deg"], 30.0, places=3)
        self.assertAlmostEqual(a["bottom_z"], a["ph_bottom"], places=3)
        self.assertLess(math.dist(a["centre_xy"], a["ph_centre"]), 1e-3)
        sx, sy, sz = a["scale"]
        self.assertAlmostEqual(sx, sy, places=6)
        self.assertAlmostEqual(sx, sz, places=6)
        for got, lim in zip(a["dims_own"], a["ph_dims"]):
            self.assertLessEqual(got, lim + 1e-3)
        self.assertTrue(any(abs(g - l) < 1e-3 for g, l in zip(a["dims_own"], a["ph_dims"])))   # touches one side
        self.assertIn("已放到占位体 占位Cube 的位置", a["summary"])
        self.assertIn("等比缩放", a["summary"])

    def test_collection_hidden_and_tag(self):
        a = self.rep["a"]
        self.assertEqual(a["collections"], ["新建筑"])
        self.assertTrue(a["ph_hide_render"] and a["ph_hide_viewport"] and a["ph_exists"])
        self.assertEqual(a["placeholder_prop"], "占位Cube")
        self.assertTrue(a["extra_tag"])
        self.assertTrue(a["in_massing"])
        self.assertFalse(a["ph_in_massing"])
        self.assertTrue(a["in_prompt_items"], a["prompt_names"])
        self.assertIn(a["hint"], a["prompt_text"])
        self.assertIn(a["name"], a["multiview_words"])
        self.assertEqual(a["label_sb"], "来自场景参考图")
        self.assertEqual(a["label_new"], "3 张参考")

    def test_replace_keeps_one_block(self):
        b = self.rep["b"]
        self.assertEqual(b["hull_blocks"], 1)
        self.assertEqual(b["name"], self.rep["a"]["name"])
        self.assertIn("已替换", b["summary"])
        self.assertLess(math.dist(b["centre_xy"], self.rep["a"]["ph_centre"]), 1e-3)
        self.assertAlmostEqual(b["bottom_z"], self.rep["a"]["ph_bottom"], places=3)
        self.assertEqual(b["collections"], ["新建筑"])

    def test_manual_size_and_cursor(self):
        c = self.rep["c"]
        self.assertEqual(c["scale"], [1.0, 1.0, 1.0])
        self.assertIn("保持填写的长宽高", c["summary"])
        self.assertAlmostEqual(c["yaw_deg"], -45.0, places=3)
        self.assertLess(math.dist(c["centre_xy"], c["ph_centre"]), 1e-3)
        self.assertEqual(c["collections"], [c["massing_name"]])
        self.assertFalse(c["extra_tag"])
        d = self.rep["d"]
        self.assertEqual(d["location"], [7.0, 8.0, 0.0])      # nothing selected: 3D cursor, on the ground
        self.assertEqual(d["mode"], "cursor")
        e = self.rep["e"]
        self.assertEqual(e["mode"], "cursor")                 # a 场景粗模 block selected is not a placeholder
        self.assertFalse(e["sb_hidden"])
        self.assertTrue(self.rep["product_unmoved"])


if __name__ == "__main__":
    unittest.main()
