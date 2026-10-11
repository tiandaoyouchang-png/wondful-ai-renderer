"""1.1.0: custom terrain (高级 › 地面 › 自定义) — presets, height field, pose and prompt text."""
import math, sys, unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402
T = load_package_module("terrain")
CD = load_package_module("codex_direct")
ROOT = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"


class TerrainMathTests(unittest.TestCase):
    def test_presets_fill_four_directions(self):
        self.assertEqual(set(T.PRESET_ORDER), {"CLIMB", "DESCENT", "SIDE", "RIDGE", "VALLEY"})
        for key in T.PRESET_ORDER:
            s = T.preset_settings(key)
            for d in T.DIRECTIONS:
                self.assertIn(s[d][0], T.MODES)
                self.assertTrue(0.0 <= s[d][1] <= 35.0)

    def test_pose_follows_front_back_and_left_right(self):
        p, r = T.pose_angles(*T.pad_gradients(T.preset_settings("CLIMB")))
        self.assertAlmostEqual(p, 15.0, places=4); self.assertAlmostEqual(r, 0.0)
        p, r = T.pose_angles(*T.pad_gradients(T.preset_settings("DESCENT")))
        self.assertAlmostEqual(p, -15.0, places=4)
        p, r = T.pose_angles(*T.pad_gradients(T.preset_settings("SIDE")))
        self.assertAlmostEqual(r, -12.0, places=4)  # right side down
        for key in ("RIDGE", "VALLEY"):  # symmetric: level pad on top / at the bottom
            self.assertEqual(T.pad_gradients(T.preset_settings(key)), (0.0, 0.0))

    def test_follow_off_keeps_pad_level(self):
        s = T.preset_settings("CLIMB"); s["follow"] = False
        self.assertEqual(T.pad_gradients(s), (0.0, 0.0))

    def test_angles_are_clamped(self):
        s = T.normalize({"front": ("UP", 80), "back": ("SIDEWAYS", 10), "left": ("DOWN", 0)})
        self.assertEqual(s["front"], ("UP", 35.0))
        self.assertEqual(s["back"], ("FLAT", 0.0))
        self.assertEqual(s["left"], ("FLAT", 0.0))

    def test_height_field_shape(self):
        hf = T.HeightField(T.preset_settings("RIDGE"), 2.8, 1.2, 1.4)
        self.assertAlmostEqual(hf.height(0.0, 0.0), 0.0)
        self.assertAlmostEqual(hf.height(2.0, 1.0), 0.0)  # pad is flat on a ridge
        far = hf.height(0.0, 30.0)
        self.assertLess(far, -(30 - 1.2 - 1.4) * math.tan(math.radians(20)) + 1e-6)
        hf = T.HeightField(T.preset_settings("CLIMB"), 2.8, 1.2, 1.4)
        self.assertAlmostEqual(hf.height(1.0, 0.0), math.tan(math.radians(15)), places=6)
        # numpy and scalar paths agree, and the profile is continuous across the pad edge
        xs = np.linspace(-20, 20, 801)
        hs = hf.height(xs, np.zeros_like(xs))
        self.assertTrue(np.allclose(hs, [hf.height(float(x), 0.0) for x in xs]))
        self.assertLess(np.abs(np.diff(hs)).max(), 0.05 * 1.0 + 1e-9)

    def test_grid_offsets_cover_pad_and_far(self):
        g = T.grid_offsets(2.8, 1.4, 5.0)
        self.assertEqual(g, sorted(g))
        self.assertIn(0.0, g); self.assertIn(2.8, g); self.assertGreaterEqual(g[-1], 300.0)


class TerrainPromptTests(unittest.TestCase):
    def test_car_wording(self):
        text = T.describe(T.preset_settings("CLIMB"), vehicle=True)
        self.assertIn("车头朝上爬坡约15°", text)
        self.assertIn("车轮", text)
        text = T.describe(T.preset_settings("SIDE"), vehicle=True, hints={"right": "画面左侧", "left": "画面右侧"})
        self.assertIn("车身右侧（画面左侧）低", text)
        self.assertIn("约12°", text)

    def test_generic_wording_for_other_products(self):
        for key in T.PRESET_ORDER:
            text = T.describe(T.preset_settings(key), vehicle=False)
            self.assertNotIn("车", text, key)
            self.assertIn("产品", text)
        self.assertIn("山脊", T.describe(T.preset_settings("RIDGE")))
        self.assertIn("山谷", T.describe(T.preset_settings("VALLEY")))

    def test_flat_terrain_has_no_text(self):
        self.assertEqual(T.describe(T.normalize({})), "")

    def test_terrain_text_reaches_simple_prompt(self):
        tt = T.describe(T.preset_settings("CLIMB"), vehicle=True)
        text = CD.build_simple_prompt("雪山", terrain_text=tt)
        self.assertIn(tt, text)
        self.assertNotIn("本次地形", CD.build_simple_prompt("雪山"))


class TerrainWiringTests(unittest.TestCase):
    def test_props_ui_operator_wired(self):
        props = (ROOT / "properties.py").read_text(encoding="utf-8")
        self.assertIn('("CUSTOM", "自定义"', props)
        for d in T.DIRECTIONS:
            self.assertIn(f"clay_terrain_{d}: EnumProperty", props)
            self.assertIn(f"clay_terrain_{d}_angle: FloatProperty", props)
        self.assertIn("clay_terrain_follow: BoolProperty", props)
        self.assertIn('name="姿态跟随地形"', props)
        ui = (ROOT / "ui.py").read_text(encoding="utf-8")
        self.assertIn('"wondful.terrain_preset"', ui)
        self.assertIn('== "CUSTOM"', ui)
        ops = (ROOT / "operators.py").read_text(encoding="utf-8")
        self.assertIn('bl_idname = "wondful.terrain_preset"', ops)
        self.assertIn("terrain=settings_from_props(props)", ops)
        self.assertIn("terrain_text=direct_terrain_text", ops)
        init = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(init.count("WONDFUL_OT_terrain_preset"), 2)
        vc = (ROOT / "viewport_capture.py").read_text(encoding="utf-8")
        self.assertIn("for undo in reversed(restore)", vc)  # pose is always undone in finally

    def test_preset_to_props(self):
        class P:  # minimal stand-in for the PropertyGroup
            pass
        p = P()
        T.apply_preset_to_props(p, "SIDE")
        self.assertEqual((p.clay_terrain_left, p.clay_terrain_right), ("UP", "DOWN"))
        self.assertEqual(p.clay_terrain_front, "FLAT")
        s = T.settings_from_props(p)
        self.assertEqual(s["right"], ("DOWN", 12.0))


if __name__ == "__main__":
    unittest.main()
