"""1.1.0: custom terrain (高级 › 地面 › 自定义) — four corners, 坡顶, 凹凸, rigid rest pose, prompt text."""
import math, sys, unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402
T = load_package_module("terrain")
CD = load_package_module("codex_direct")
ROOT = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"

F = np.array([0.0, -1.0, 0.0])
S = np.cross(F, [0.0, 0.0, 1.0])


def four_legs(half=0.25, height=0.8, com_offset=(0.02, -0.03)):
    """A chair-like test product: four vertical legs, centre of mass slightly off-centre."""
    pts = []
    for su in (-1, 1):
        for sv in (-1, 1):
            for k in range(20):
                pts.append(F * su * half + S * sv * half + np.array([0.0, 0.0, k * height / 20]))
    com = F * com_offset[0] + S * com_offset[1] + np.array([0.0, 0.0, 0.4 * height])
    return np.array(pts), com


def settle(name, follow=True, **over):
    s = T.preset_settings(name)
    s.update(over)
    s["follow"] = follow
    P, com = four_legs()
    hf = T.HeightField(s, 0.28, 0.28, 0.28, 0.8)
    R, t, p, r = T.rest_pose(T.support_points(P, F, S), com, hf, F, S, follow=follow)
    Q = P @ R.T + t
    rep = T.contact_report(Q, R @ com + t, hf, F, S, 0.004 * 0.8, 0.5)
    return p, r, t[2], rep


class TerrainMathTests(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(set(T.PRESET_ORDER), {"CREST", "TWIST", "TWIST_MIRROR", "BUMPY", "CLIMB", "DESCENT", "SIDE"})
        for key in T.PRESET_ORDER:
            s = T.preset_settings(key)
            for c in T.CORNERS:
                self.assertTrue(-35.0 <= s[c] <= 35.0)
            self.assertFalse(T.is_flat(s), key)
        # the user's 左上右下高、右上左下低: 左前 + 右后 high, 右前 + 左后 low
        s = T.preset_settings("TWIST")
        self.assertGreater(s["fl"], 0); self.assertGreater(s["br"], 0)
        self.assertLess(s["fr"], 0); self.assertLess(s["bl"], 0)
        self.assertGreater(T.components(s)["twist"], 0)
        self.assertLess(T.components(T.preset_settings("TWIST_MIRROR"))["twist"], 0)
        self.assertGreater(T.components(T.preset_settings("CREST"))["dome"], 10)
        self.assertGreater(T.preset_settings("BUMPY")["bump"], 0.3)
        self.assertEqual(T.preset_settings("RIDGE"), T.preset_settings("CREST"))  # old name remapped

    def test_components(self):
        c = T.components(T.preset_settings("CLIMB"))
        self.assertAlmostEqual(c["pitch"], 15.0); self.assertAlmostEqual(c["roll"], 0.0)
        c = T.components(T.preset_settings("SIDE"))
        self.assertAlmostEqual(c["roll"], -12.0)  # right side low
        c = T.components({"fl": -20, "fr": -20, "bl": -20, "br": -20})
        self.assertAlmostEqual(c["dome"], 20.0)  # all four corners down = crest

    def test_clamp_and_flat(self):
        s = T.normalize({"fl": 80, "fr": -99, "crest": 3, "bump": -1})
        self.assertEqual((s["fl"], s["fr"], s["crest"], s["bump"]), (35.0, -35.0, 1.0, 0.0))
        self.assertTrue(T.is_flat(T.normalize({})))

    def test_legacy_directions_migrate(self):
        old = {"front": ("UP", 15), "back": ("DOWN", 15), "left": ("FLAT", 0), "right": ("FLAT", 0)}
        c = T.components(T.normalize(old))
        self.assertAlmostEqual(c["pitch"], 15.0)
        old = {"front": ("FLAT", 0), "back": ("FLAT", 0), "left": ("DOWN", 20), "right": ("DOWN", 20)}
        self.assertGreater(T.components(T.normalize(old))["dome"], 15)  # 山脊 → crest

        class P(dict):  # a PropertyGroup holding the old raw ID properties
            pass
        p = P({"clay_terrain_front": 1, "clay_terrain_front_angle": 10.0, "clay_terrain_back": 2,
               "clay_terrain_back_angle": 10.0})
        self.assertTrue(T.migrate_legacy_props(p))
        self.assertEqual((p.clay_terrain_fl, p.clay_terrain_fr, p.clay_terrain_bl, p.clay_terrain_br),
                         (10.0, 10.0, -10.0, -10.0))
        self.assertNotIn("clay_terrain_front", p)
        self.assertFalse(T.migrate_legacy_props(p))

    def test_height_field_shapes(self):
        hf = T.HeightField(T.preset_settings("CLIMB"), 2.8, 1.2, 1.4, 1.6)
        self.assertAlmostEqual(hf.height(1.0, 0.0), math.tan(math.radians(15)), places=6)
        hf = T.HeightField(T.preset_settings("CREST"), 2.8, 1.2, 1.4, 1.6)
        self.assertAlmostEqual(hf.height(2.0, 1.0), 0.0)  # flat plateau under the product
        for u, v in ((20, 0), (-20, 0), (0, 20), (0, -20), (15, 15)):
            self.assertLess(hf.height(u, v), -3.0)  # falls away all round
        hf = T.HeightField(T.preset_settings("TWIST"), 2.8, 1.2, 1.4, 1.6)
        self.assertGreater(hf.height(8, -8), 1.0)   # 左前 (front, left) high
        self.assertGreater(hf.height(-8, 8), 1.0)   # 右后 high
        self.assertLess(hf.height(8, 8), -1.0)      # 右前 low
        self.assertLess(hf.height(-8, -8), -1.0)    # 左后 low
        pad = abs(hf.height(2.8, -1.2))
        self.assertLess(pad, 0.3)  # gentler under the product (vs ~0.65 at full slope)
        self.assertGreater(pad, 0.15)  # ... but enough that a rigid product visibly lifts one corner
        self.assertAlmostEqual(pad, T.TWIST_PAD_GAP * 1.6 * math.tan(math.radians(12)), places=6)
        # numpy and scalar paths agree; continuous
        xs = np.linspace(-20, 20, 801)
        for name in T.PRESET_ORDER:
            hf = T.HeightField(T.preset_settings(name), 2.8, 1.2, 1.4, 1.6)
            hs = hf.height(xs, 0.7 * xs)
            self.assertTrue(np.allclose(hs, [hf.height(float(x), 0.7 * float(x)) for x in xs]), name)
            self.assertLess(np.abs(np.diff(hs)).max(), 0.05, name)

    def test_bumps_deterministic(self):
        a = T.HeightField(T.preset_settings("BUMPY"), 1, 1, 0.5, 1).height(np.arange(10.0), np.zeros(10))
        b = T.HeightField(T.preset_settings("BUMPY"), 1, 1, 0.5, 1).height(np.arange(10.0), np.zeros(10))
        self.assertTrue(np.allclose(a, b))
        self.assertGreater(np.ptp(a), 0.01)

    def test_grid_offsets(self):
        g = T.grid_offsets(2.8, 1.4, 5.0)
        self.assertEqual(g, sorted(g))
        self.assertIn(0.0, g); self.assertIn(2.8, g); self.assertGreaterEqual(g[-1], 300.0)
        self.assertLess(max(np.diff([x for x in g if abs(x) < 4.0])), 5.0 / 30)


class RestPoseTests(unittest.TestCase):
    def test_planes_rest_on_four_legs(self):
        for name, pitch, roll in (("CLIMB", 15, 0), ("DESCENT", -15, 0), ("SIDE", 0, -12), ("CREST", 0, 0)):
            p, r, lift, rep = settle(name)
            self.assertAlmostEqual(p, pitch, places=1, msg=name)
            self.assertAlmostEqual(r, roll, places=1, msg=name)
            self.assertEqual(rep["contact_groups"], 4, name)
            self.assertGreaterEqual(rep["min_clearance"], -1e-9)

    def test_twist_three_point_contact_one_corner_lifted(self):
        for name in ("TWIST", "TWIST_MIRROR"):
            p, r, lift, rep = settle(name)
            self.assertLess(max(abs(p), abs(r)), 16.0, name)  # tilts, but does not tip over absurdly
            self.assertEqual(rep["contact_groups"], 3, name)
            self.assertTrue(rep["com_inside"], name)
            self.assertGreaterEqual(rep["min_clearance"], -1e-9)
            corner, gap = T.lifted_corner(rep, 0.5)
            self.assertIn(corner, ("fr", "bl") if name == "TWIST" else ("fl", "br"))  # a low corner hangs
            self.assertGreater(gap, 0.12)  # a 0.8 m product: the gap must read in the picture
            self.assertLess(gap, 0.35)

    def test_bumpy_stable_no_penetration(self):
        p, r, lift, rep = settle("BUMPY")
        self.assertGreaterEqual(rep["contact_groups"], 3)
        self.assertTrue(rep["com_inside"])
        self.assertGreaterEqual(rep["min_clearance"], -1e-9)

    def test_follow_off_keeps_level(self):
        p, r, lift, rep = settle("TWIST", follow=False)
        self.assertEqual((p, r), (0.0, 0.0))
        self.assertGreaterEqual(rep["min_clearance"], -1e-9)


class TerrainPromptTests(unittest.TestCase):
    def test_car_wording(self):
        text = T.describe(T.preset_settings("CLIMB"), vehicle=True)
        self.assertIn("车头朝上爬坡约15°", text)
        self.assertIn("车轮", text)
        text = T.describe(T.preset_settings("SIDE"), vehicle=True, hints={"right": "画面左侧", "left": "画面右侧"})
        self.assertIn("车身右侧（画面左侧）低", text)
        text = T.describe(T.preset_settings("TWIST"), vehicle=True, pose={"pitch": -2, "roll": 2, "lifted": "fr", "gap": 0.26})
        self.assertIn("左前方与右后方地面较高", text)
        self.assertIn("三个车轮着地", text)
        self.assertIn("右前轮明显悬空、离地约26厘米", text)

    def test_generic_wording(self):
        for key in T.PRESET_ORDER:
            text = T.describe(T.preset_settings(key), vehicle=False, pose={"lifted": "fr"})
            self.assertNotIn("车", text, key)
            self.assertIn("产品", text)
        text = T.describe(T.preset_settings("TWIST"), pose={"pitch": -3, "roll": 3, "lifted": "fr"},
                          hints={"fr": "画面左侧近处"})
        self.assertIn("产品三点着地", text)
        self.assertIn("右前角（画面左侧近处）明显悬空", text)
        self.assertIn("空隙", text)
        self.assertIn("保持悬空", text)
        self.assertIn("坡顶", T.describe(T.preset_settings("CREST")))
        self.assertIn("凹凸不平", T.describe(T.preset_settings("BUMPY")))

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
        for c in T.CORNERS:
            self.assertIn(f"clay_terrain_{c}: FloatProperty", props)
        for d in T.LEGACY_DIRECTIONS:
            self.assertNotIn(f"clay_terrain_{d}: EnumProperty", props)
        self.assertIn("clay_terrain_crest: FloatProperty", props)
        self.assertIn("clay_terrain_bump: FloatProperty", props)
        self.assertIn("clay_terrain_follow: BoolProperty", props)
        self.assertIn('name="姿态跟随地形"', props)
        ui = (ROOT / "ui.py").read_text(encoding="utf-8")
        self.assertIn('"wondful.terrain_preset"', ui)
        self.assertIn('== "CUSTOM"', ui)
        for key in ("CREST", "TWIST", "BUMPY"):
            self.assertIn(f'("{key}", ', ui)
        ops = (ROOT / "operators.py").read_text(encoding="utf-8")
        self.assertIn('bl_idname = "wondful.terrain_preset"', ops)
        self.assertIn("terrain=settings_from_props(props)", ops)
        self.assertIn("terrain_text=direct_terrain_text", ops)
        init = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(init.count("WONDFUL_OT_terrain_preset"), 2)
        self.assertIn("migrate_legacy_props", init)
        vc = (ROOT / "viewport_capture.py").read_text(encoding="utf-8")
        self.assertIn("for undo in reversed(restore)", vc)  # pose is always undone in finally
        self.assertIn("TR.rest_pose(", vc)

    def test_preset_to_props(self):
        class P:  # minimal stand-in for the PropertyGroup
            pass
        p = P()
        T.apply_preset_to_props(p, "TWIST")
        self.assertEqual((p.clay_terrain_fl, p.clay_terrain_fr, p.clay_terrain_bl, p.clay_terrain_br),
                         (12.0, -12.0, -12.0, 12.0))
        self.assertEqual((p.clay_terrain_crest, p.clay_terrain_bump), (0.0, 0.0))
        p.clay_terrain_follow = True
        s = T.settings_from_props(p)
        self.assertAlmostEqual(T.components(s)["twist"], 12.0)


if __name__ == "__main__":
    unittest.main()
