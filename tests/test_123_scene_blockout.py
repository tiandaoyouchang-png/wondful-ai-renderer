"""1.2.3 高级 › 环境体块 › 从参考图生成场景粗模.

Pure-Python parts (schema, rules, labels, scene note, fit score, yaw maths) run anywhere; the JSON → Blender
scene builder runs in a headless Blender subprocess (skipped when ``blender`` is not on PATH) on fixture
JSONs from the accepted prototype runs (tests/fixtures/scene_blockout, no network)."""
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

SB = load_package_module("scene_blockout")
MS = load_package_module("massing")
SM = load_package_module("sketch_model")
HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "wondful_ai_renderer"
FIX = HERE / "fixtures" / "scene_blockout"
FIXTURES = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(FIX.glob("*.json"))}


def _walk(node, path="$"):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


class SchemaTests(unittest.TestCase):
    def test_strict_schema_shape(self):
        # OpenAI strict json_schema: every object closed and every property required
        for path, obj in _walk(SB.SCHEMA):
            self.assertIs(obj.get("additionalProperties"), False, path)
            self.assertEqual(sorted(obj["required"]), sorted(obj["properties"]), path)

    def test_label_fields_everywhere(self):
        P = SB.SCHEMA["properties"]
        for key in ("roads", "mountain_bands", "foliage_masses", "primitives", "cliffs"):
            self.assertIn("label_zh", P[key]["items"]["properties"], key)
        self.assertIn("label_zh", P["terrain"]["properties"])
        self.assertIn("lock_zh", P["palette"]["properties"])

    def test_rules_frame_and_product(self):
        content = SB.analysis_content("data:image/png;base64,AAAA", (1200, 1600), (0.7, 0.7, 0.9), vehicle=False)
        text = content[0]["text"]
        self.assertIn("1200x1600", text)
        self.assertIn("3:4", text)
        self.assertIn("tan(hfov/2)*1.3333", text)
        self.assertIn("0.70 m wide x 0.70 m deep x\n 0.90 m high", text)
        self.assertIn("small object", text)
        self.assertNotIn("{", text.split("COORDINATES")[0])
        images = [c for c in content if c["type"] == "input_image"]
        self.assertEqual(len(images), 1)          # only the reference photo, only to the analysis model
        car = SB.analysis_content("x", (1600, 900), (1.9, 4.6, 1.5), vehicle=True)[0]["text"]
        self.assertIn("16:9", car)
        self.assertIn("a car", car)

    def test_payload_strict_and_toolless(self):
        p = SB.payload([{"type": "input_text", "text": "x"}], "m", "high")
        self.assertEqual(p["text"]["format"]["type"], "json_schema")
        self.assertTrue(p["text"]["format"]["strict"])
        self.assertEqual(p["tools"], [])
        self.assertEqual(p["reasoning"], {"effort": "high"})
        self.assertFalse(p["store"])

    def test_correction_content(self):
        base = SB.analysis_content("ref", (1600, 900))
        prev = dict(FIXTURES["desert_v3"])
        c = SB.correction_content(base, prev, {"score": 0.5}, ["x 0.1-0.2: photo skyline at y 0.30"], "r", "o")
        self.assertEqual([x["image_url"] for x in c if x["type"] == "input_image"], ["ref", "r", "o"])
        joined = "".join(x.get("text", "") for x in c)
        self.assertIn("photo skyline", joined)
        self.assertNotIn('"_frame"', joined)

    def test_parse_reply(self):
        self.assertEqual(SB.parse_reply('noise {"a": 1} tail'), {"a": 1})
        with self.assertRaises(ValueError):
            SB.parse_reply("no json")


class NormalizeAndLabelTests(unittest.TestCase):
    def test_fixtures_normalize(self):
        for name, data in FIXTURES.items():
            d = SB.normalize(data)
            for key in ("roads", "cliffs", "surface_zones", "foliage_masses", "primitives", "mountain_bands"):
                self.assertIsInstance(d[key], list, name)
            self.assertIn("has_sea", d["terrain"], name)
            self.assertTrue(20 <= d["camera"]["hfov_deg"] <= 120)
        self.assertNotIn("cliffs", FIXTURES["desert_v3"])   # v3 fixture really is the old schema

    def test_normalize_clamps_and_rejects(self):
        d = SB.normalize({"camera": {"hfov_deg": 400, "pitch_deg": -200, "height_m": "x"},
                          "primitives": [{"type": "spaceship", "shape": "torus", "w": None}]})
        self.assertEqual(d["camera"]["hfov_deg"], 120.0)
        self.assertEqual(d["camera"]["pitch_deg"], -80.0)
        self.assertEqual(d["camera"]["height_m"], 1.6)
        self.assertEqual((d["primitives"][0]["type"], d["primitives"][0]["shape"]), ("building", "box"))
        with self.assertRaises(ValueError):
            SB.normalize([1, 2])

    def test_labels(self):
        self.assertEqual(SB.clean_label("Red brick 红砖厂房 01"), "红砖厂房")
        self.assertEqual(SB.clean_label("building", "建筑"), "建筑")
        self.assertEqual(SB.element_label("primitive", {"type": "building", "label_zh": "玻璃幕墙写字楼"}), "玻璃幕墙写字楼")
        self.assertEqual(SB.element_label("primitive", {"type": "furniture", "id": "coffee_table"}, True), "茶几")
        self.assertEqual(SB.element_label("primitive", {"type": "wall", "id": "w1"}, True), "墙面")
        self.assertEqual(SB.element_label("primitive", {"type": "pole", "id": "p3"}), "杆子")
        self.assertEqual(SB.element_label("road", {"kind": "sidewalk", "id": "s"}), "人行道")
        self.assertEqual(SB.element_label("band", {"kind": "snowy", "id": "b"}), "远处雪山")
        self.assertEqual(SB.element_label("foliage", {"kind": "palm", "id": "f"}), "棕榈树")
        self.assertEqual(SB.terrain_label(FIXTURES["interior_v3"]), "地板")
        self.assertEqual(SB.terrain_label(FIXTURES["snow_v4"]), "雪地")
        self.assertIn(SB.terrain_label(FIXTURES["desert_v3"]), ("沙地", "沙丘"))
        self.assertEqual(SB.terrain_label({"terrain": {"label_zh": "金色沙丘"}}), "金色沙丘")

    def test_labels_feed_massing_names(self):
        # the builder names objects "<label>_NN"; the prompt naming reads the label back from the name
        self.assertEqual(MS.hint_from_name("棕榈树_36"), "棕榈树")
        self.assertEqual(MS.hint_from_name("沙发坐垫底座_14_02"), "沙发坐垫底座")
        self.assertEqual(MS.derive_hint("远处雪山_07", "远处雪山", True)[0], "远处雪山")

    def test_scene_note(self):
        note = SB.scene_note(FIXTURES["snow_v4"])
        self.assertTrue(note.startswith("场景（按环境参考图分析的季节与色彩"))
        self.assertIn("冬季", note)
        self.assertIn("积雪", note)
        locked = SB.scene_note({"palette": {"lock_zh": "盛夏晴天：草地鲜绿"}, "environment_zh": "海边公路"})
        self.assertIn("盛夏晴天：草地鲜绿", locked)
        self.assertIn("海边公路", locked)
        self.assertEqual(SB.scene_note({}), "")
        objs = [{"x": 1}, {"wondful_sb_scene_note": "场景：A"}]
        self.assertEqual(MS.scene_note_of(objs), "场景：A")
        self.assertEqual(MS.scene_note_of([{}]), "")

    def test_summary(self):
        s = SB.summary({"blocks": 12, "kind": "interior", "camera": True, "scores": [0.6, 0.7], "round": 2,
                        "notes": ["移除了与产品重叠的茶几（t）"]})
        self.assertIn("12 个体块", s)
        self.assertIn("室内", s)
        self.assertIn(SB.CAMERA_NAME, s)
        self.assertIn("0.600→0.700（采用校正轮）", s)
        self.assertIn("注意：移除了与产品重叠的茶几", s)
        self.assertIn("保留第 1 轮", SB.summary({"blocks": 1, "scores": [0.6, 0.5], "round": 1}))
        self.assertGreater(SB.MIN_GAIN, 0)
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        self.assertIn('sc["score"] <= sb["scores"][0] + SB.MIN_GAIN', ops)   # kept only when the score really rises


class GeometryMathTests(unittest.TestCase):
    def test_lens_and_horizon(self):
        self.assertAlmostEqual(SB.lens_from_hfov(90), 18.0, places=6)
        self.assertAlmostEqual(SB.horizon_from_pitch(0, 80, 9 / 16), 0.5)
        self.assertLess(SB.horizon_from_pitch(-10, 80, 9 / 16), 0.5)    # looking down: horizon above centre
        for name, d in FIXTURES.items():
            c = SB.normalize(d)["camera"]
            hz = SB.horizon_from_pitch(c["pitch_deg"], c["hfov_deg"], 9 / 16)
            self.assertLess(abs(hz - c["horizon_y_frac"]), 0.08, name)

    def test_yaw(self):
        self.assertAlmostEqual(SB.yaw_to(0, 1), 0.0)
        self.assertAlmostEqual(SB.yaw_to(-1, 0), math.pi / 2)    # +Y turned 90° CCW points to -X
        # spot heading 20° must land on a product long axis at 50° (mod 180)
        a = SB.align_yaw(math.radians(50), 20)
        self.assertAlmostEqual(math.degrees(a), 30.0, places=6)
        a = SB.align_yaw(math.radians(-170), 20)
        self.assertTrue(-90 <= math.degrees(a) <= 90)
        self.assertAlmostEqual((math.degrees(a) + 20 - (-170)) % 180, 0.0, places=6)


class ProductSpotTests(unittest.TestCase):
    PRIMS = [{"type": "furniture", "id": "coffee_table", "x": 0.08, "y": 2.17, "w": 0.85, "d": 1.65, "h": 0.4,
              "z_offset": 0.0, "rot_z_deg": 0},
             {"type": "furniture", "id": "media_cabinet", "x": 1.49, "y": 2.05, "w": 0.58, "d": 3.7, "h": 0.5,
              "z_offset": 0.0, "rot_z_deg": 0},
             {"type": "rug", "id": "rug", "x": 0.8, "y": 2.4, "w": 3, "d": 3, "h": 0.02, "z_offset": 0, "rot_z_deg": 0},
             {"type": "furniture", "id": "wall_shelf", "x": 0.84, "y": 2.4, "w": 1, "d": 0.3, "h": 0.03,
              "z_offset": 1.4, "rot_z_deg": 0}]

    def test_rect_dist(self):
        p = {"x": 0, "y": 0, "w": 2, "d": 1, "rot_z_deg": 0}
        self.assertEqual(SB.rect_dist(0.5, 0.2, p), 0.0)
        self.assertAlmostEqual(SB.rect_dist(2.0, 0.0, p), 1.0)
        self.assertAlmostEqual(SB.rect_dist(0.0, 2.0, dict(p, rot_z_deg=90)), 1.0)

    def test_free_spot_keeps_a_free_spot(self):
        self.assertEqual(SB.free_spot(self.PRIMS, -2.0, 4.0, 0.45, 0.9), (-2.0, 4.0, 0.0))

    def test_free_spot_nudges_out_of_a_tight_gap(self):
        # real interior analysis: a chair spot squeezed between the coffee table and the TV cabinet (rug and the
        # raised shelf do not block)
        x, y, shift = SB.free_spot(self.PRIMS, 0.84, 2.4, 0.45, 0.9, hfov_deg=74)
        self.assertGreater(shift, 0)
        self.assertLessEqual(shift, 1.2 + 1e-9)
        obs = SB.blocking_footprints(self.PRIMS, 0.9)
        self.assertEqual([o["id"] for o in obs], ["coffee_table", "media_cabinet"])
        self.assertGreaterEqual(SB.clearance(obs, x, y), 0.45)
        self.assertTrue(0.1 <= 0.5 + x / (2 * y * math.tan(math.radians(37))) <= 0.9)

    def test_free_spot_gives_up(self):
        wall = [{"type": "building", "id": "b", "x": 0, "y": 10, "w": 50, "d": 50, "h": 10, "z_offset": 0, "rot_z_deg": 0}]
        self.assertIsNone(SB.free_spot(wall, 0, 10, 2.0, 1.5))


class FitScoreTests(unittest.TestCase):
    def photo(self, sky_rows=80, h=180, w=320):
        img = np.zeros((h, w, 3), np.float32)
        img[:sky_rows] = (0.55, 0.70, 0.95)       # blue sky
        img[sky_rows:] = (0.45, 0.40, 0.30)       # ground
        img[sky_rows - 30:sky_rows, 100:160] = (0.3, 0.3, 0.32)  # a building against the sky
        return img

    def render(self, sky_rows=80, h=180, w=320, dx=0):
        r = np.zeros((h, w, 4), np.float32)
        r[..., :3] = 0.6
        r[..., 3] = 1.0
        r[:sky_rows, :, 3] = 0.0
        r[sky_rows - 30:sky_rows, 100 + dx:160 + dx, 3] = 1.0
        r[sky_rows - 30:sky_rows, 100 + dx:160 + dx, :3] = 0.3
        return r

    def test_sky_mask(self):
        sky = SB.photo_sky(self.photo())
        self.assertGreater(sky[:50].mean(), 0.95)
        self.assertLess(sky[100:].mean(), 0.01)

    def test_score_orders_fits(self):
        good = SB.fit_score(self.photo(), self.render())
        bad = SB.fit_score(self.photo(), self.render(sky_rows=40, dx=90))
        self.assertGreater(good["sky_iou"], 0.95)
        self.assertGreater(good["score"], bad["score"])
        self.assertTrue(any("too high" in h for h in bad["hints"]))
        self.assertEqual(good["hints"], [])
        u8 = SB.fit_score((self.photo() * 255).astype(np.uint8), (self.render() * 255).astype(np.uint8))
        self.assertAlmostEqual(u8["score"], good["score"], delta=0.05)

    def test_score_interior_without_sky(self):
        photo = np.full((90, 160, 3), 0.5, np.float32)
        photo[:, 70:90] = 0.2
        render = np.ones((90, 160, 4), np.float32) * 0.6
        render[:, 70:90, :3] = 0.25
        sc = SB.fit_score(photo, render)
        self.assertNotIn("sky_iou", sc)
        self.assertEqual(sc["score"], sc["edge_f"])
        self.assertGreater(sc["score"], 0.5)

    def test_overlay(self):
        o = SB.overlay(np.zeros((10, 20, 3), np.uint8), np.full((5, 10, 3), 200, np.uint8))
        self.assertEqual(o.shape, (10, 20, 3))
        self.assertEqual(int(o[0, 0, 0]), 100)


class WiringTests(unittest.TestCase):
    def test_registered_and_in_main_flow_step4(self):
        init = (PKG / "__init__.py").read_text(encoding="utf-8")
        self.assertEqual(init.count("WONDFUL_OT_scene_blockout_build"), 2)
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        # 1.2.3: moved out of 高级 › 场景 › 环境体块 into the main flow as ④ 场景粗模（可选）
        massing = ui[ui.index("def _draw_massing("):ui.index("def _draw_block_refs")]
        self.assertNotIn("scene_blockout", massing)
        main = ui.split("class WONDFUL_PT_main")[1]
        self.assertEqual(main.count("_draw_blockout_step("), 1)
        self.assertEqual(ui.count('"wondful.scene_blockout_build"'), 1)
        self.assertIn('"wondful.scene_blockout_build", text="生成场景粗模"', ui)
        for label in ("同时设置相机", "自动校正一轮"):
            self.assertIn(label, ui)
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        self.assertIn("sb_set_camera: BoolProperty(name=\"同时设置相机\", default=True", props)
        self.assertIn("sb_correct: BoolProperty(name=\"自动校正一轮\", default=False", props)
        self.assertIn("记录原图用于连续结构、材质与光照参照", props)

    def test_blockout_build_does_not_trigger_image_generation(self):
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        body = ops.split("class WONDFUL_OT_scene_blockout_build")[1].split("\nclass ")[0]
        for call in ("edit_image", "generate_image", "style_images.add", "load_reference"):
            self.assertNotIn(call, body)
        self.assertIn('bl_options = {"REGISTER", "UNDO"}', body)
        src = (PKG / "scene_blockout_build.py").read_text(encoding="utf-8")
        self.assertNotIn("edit_image", src)

    def test_terrain_is_ground_and_note_reaches_prompt(self):
        vc = (PKG / "viewport_capture.py").read_text(encoding="utf-8")
        self.assertIn('if _excluded(obj) and not obj.get("wondful_terrain")', vc)
        self.assertIn("MS.scene_note_of(blocks)", vc)

    def test_split_sheet_error_tells_what_works(self):
        one = np.ones((300, 300, 4), np.float32)
        one[50:250, 50:250, :3] = 0.2
        with self.assertRaises(ValueError) as cm:
            SM.split_sheet(one)
        msg = str(cm.exception)
        for words in ("白底或透明底", "分开排列", "留出明显空白", "边框、文字、尺寸线", "分别导入"):
            self.assertIn(words, msg)


BLENDER = shutil.which("blender")


@unittest.skipUnless(BLENDER and os.environ.get("WONDFUL_SKIP_BLENDER_TESTS") != "1", "blender not on PATH")
class BuilderBlenderTests(unittest.TestCase):
    """JSON → Blender blocks through the real builder (headless Blender, fixture JSONs, no network)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="wondful_sb_test_")
        cls.results = {}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_case(self, fixture, product="box", set_camera=True, preview=False):
        key = (fixture, product, set_camera, preview)
        if key not in self.results:
            out = os.path.join(self.tmp, f"{fixture}_{product}_{int(set_camera)}.json")
            args = [BLENDER, "-b", "--factory-startup", "--python", str(HERE / "sb_build_runner.py"), "--",
                    str(FIX / f"{fixture}.json"), out, product, "1" if set_camera else "0"]
            if preview:
                args.append(os.path.join(self.tmp, f"{fixture}.png"))
            subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
            self.results[key] = json.loads(Path(out).read_text(encoding="utf-8"))
        rep = self.results[key]
        self.assertTrue(rep.get("ok"), rep.get("error"))
        return rep

    def common(self, rep, outdoor=True):
        self.assertEqual(rep["collections_named"], 1)          # re-run replaced, not duplicated
        self.assertTrue(rep["rebuild_same_count"])
        self.assertTrue(rep["parent_ok"])                       # inside 环境体块 → used by clay / 多角度场景一致
        self.assertTrue(rep["found_by_massing"])
        self.assertTrue(rep["all_labelled"])
        self.assertTrue(rep["text_block"])
        self.assertTrue(rep["product_unmoved"])
        self.assertTrue(rep["user_cam_unchanged"])
        self.assertEqual(len(rep["terrain"]), 1)
        self.assertGreaterEqual(rep["roots"], 5)
        self.assertEqual(rep["roots"], rep["info"]["blocks"])
        for hint in rep["root_hints"]:
            self.assertRegex(hint, r"[\u3400-\u9fff]")
        for name in rep["root_names"]:
            self.assertRegex(name, r"^[\u3400-\u9fff（）、·]+_\d\d$")
        self.assertTrue(rep["scene_note"].startswith("场景（"))
        if outdoor:
            self.assertGreater(rep["wrap_copies"], 0)          # 360° bands / outer ground for other cameras
        else:
            self.assertEqual(rep["wrap_copies"], 0)

    def test_desert_with_car_and_camera(self):
        rep = self.run_case("desert_v3", preview=True)
        self.common(rep)
        self.assertIn("foliage_palm", rep["env_types"])
        self.assertIn("road", rep["env_types"])
        self.assertIn("band_dune", rep["env_types"])
        self.assertEqual(rep["scene_camera"], "Wondful 参考相机")
        self.assertTrue(rep["camera"]["is_scene_camera"])
        self.assertFalse(rep["camera"]["in_blockout"])
        self.assertAlmostEqual(rep["camera"]["lens"], rep["info"]["lens"], places=3)
        self.assertLess(abs(rep["ground_gap"]), 0.05)            # car base on the road surface, not sunk / floating
        self.assertTrue(rep["ground_hit"].startswith("道路"))
        self.assertTrue(rep["preview"])
        self.assertEqual(rep["scenes_after_preview"], 1)         # temporary preview scene removed
        spot = rep["info"]["spot_world"]
        self.assertAlmostEqual(spot[0], 3.0, places=3)
        self.assertAlmostEqual(spot[1], -2.0, places=3)

    def test_interior_with_chair(self):
        rep = self.run_case("interior_v3", product="chair")
        self.common(rep, outdoor=False)
        self.assertEqual(rep["info"]["kind"], "interior")
        for word in ("茶几", "沙发靠背", "电视", "天花板"):
            self.assertIn(word, rep["root_hints"])
        self.assertEqual(rep["info"]["notes"], [])                # nothing removed around the analysed spot
        self.assertLess(abs(rep["ground_gap"]), 0.02)
        self.assertTrue(rep["ground_hit"].startswith("地板"))

    def test_coast_cliffs_and_sea(self):
        rep = self.run_case("coast_v4")
        self.common(rep)
        for t in ("cliff", "water_plane", "road"):
            self.assertIn(t, rep["env_types"])
        self.assertLess(abs(rep["ground_gap"]), 0.05)

    def test_snow_crowns_without_product(self):
        rep = self.run_case("snow_v4", product="none")
        self.common(rep)
        self.assertIn("trunk", rep["env_types"])
        self.assertIn("雪地", rep["root_hints"])
        self.assertAlmostEqual(rep["info"]["spot_world"][0], 0.0, places=3)   # no product: spot at the origin

    def test_street_without_camera_follows_user_camera(self):
        rep = self.run_case("street_v4", set_camera=False)
        self.common(rep)
        self.assertIsNone(rep["camera"])
        self.assertEqual(rep["scene_camera"], "UserCam")
        self.assertAlmostEqual(rep["info"]["yaw_deg"], 45.0, places=1)   # +Y turned to the user camera's heading
        self.assertIn("building", rep["env_types"])


if __name__ == "__main__":
    unittest.main()
