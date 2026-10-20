"""1.2.6 多角度场景一致: 室内误判 / 被遮挡就调视角 / 变成非霓虹场景 (real cyber orbit test 2026-10-20).

User: 「视角问题可以通过相机设置解决吧，误判才是大问题，还有变成非霓虹场景」「被遮挡就调视角」.
1. 室内误判: 场景补全's 「推测建筑」「推测窗」 tripped the name rule (wall_like >= 2) → enclosure walls + 「室内物体」
   prompt on outdoor streets. Now the ④ analysis kind decides; completion / aux objects never count; names only
   with partial enclosure.
2. 被遮挡就调视角: every 多角度 camera is ray-checked before any image call; occluded → moved closer / nudged
   (temporarily), or skipped with 「产品被遮挡，已跳过/请调整相机」.
3. 风格与光线锁定: the 主视角 look (night, light sources, colours) goes into every new-view prompt; night colour match
   keeps local contrast / emissive pixels; no low-tier retry on night scenes.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PKG = ROOT / "wondful_ai_renderer"
FIX = HERE / "fixtures" / "scene_blockout"
sys.path.insert(0, str(HERE))
from test_316_regressions import load_package_module  # noqa: E402
from test_124_ai_look import cam, view, texture  # noqa: E402

MV = load_package_module("multiview")
LO = load_package_module("layered_output")
CYBER = Path("/workspace/work/scene_blockout_impl/orbit5_cyber/run")
BLENDER = shutil.which("blender")


class InteriorDecision(unittest.TestCase):
    def test_analysis_kind_is_the_authority(self):
        # the cyber street: 0 / 72 rays hit, 场景补全 names would have given wall_like 2 → was interior
        self.assertFalse(MV.decide_interior("outdoor", 0, False, 2))
        self.assertFalse(MV.decide_interior("outdoor", 70, True, 9))     # even a canyon-like street stays outdoor
        self.assertTrue(MV.decide_interior("interior", 0, False, 0))

    def test_unknown_kind_prefers_shape_over_names(self):
        self.assertFalse(MV.decide_interior("", 0, False, 2))            # names alone: not a room
        self.assertFalse(MV.decide_interior("", 10, False, 5))
        self.assertTrue(MV.decide_interior("", 20, False, 2))            # names + partial enclosure
        self.assertTrue(MV.decide_interior("", 40, False, 0))            # rays mostly hit blocks
        self.assertTrue(MV.decide_interior("", 0, True, 0))              # slab overhead / walls on 2 sides

    def test_wiring(self):
        mv = (PKG / "multiview.py").read_text(encoding="utf-8")
        self.assertNotIn("interior = hits >= 36 or wall_like >= 2", mv)
        self.assertIn("own = [o for o in blocks if not o.get(MS.INFERRED_PROP) and not o.get(AUX_TAG)]", mv)
        self.assertIn("interior = decide_interior(kind, hits, MS.is_interior(boxes, pbox), wall_like)", mv)
        ms = (PKG / "massing.py").read_text(encoding="utf-8")
        self.assertIn("kind = blockout_kind(objs) if kind is None else str(kind or \"\")", ms)
        sbb = (PKG / "scene_blockout_build.py").read_text(encoding="utf-8")
        self.assertIn('coll["wondful_sb_kind"]', sbb)


class LightWording(unittest.TestCase):
    NIGHT = {"light_type": "interior_mixed", "key_rel_azimuth_deg": 75, "key_elevation_deg": 30,
             "sky_words": "夜间室外街巷，无明显日光；主光来自右侧店铺橱窗及上方灯笼、街灯，暖黄低色温，多光源混合"}

    def test_outdoor_night_never_says_room(self):
        m = cam("m", (0, -7, 1.6))
        light = MV.world_light(self.NIGHT, m)
        k = cam("k", (7, 0, 1.6))
        out = MV.light_sentence(light, k, outdoor=True)
        self.assertNotIn("房间", out)
        self.assertNotIn("窗户（主光）", out)
        self.assertIn("夜景", out)
        self.assertIn("不要改成白天", out)
        # unknown / interior keeps the room wording (real interiors unchanged)
        self.assertIn("房间", MV.light_sentence(light, k))
        self.assertIn("房间", MV.light_sentence(light, k, outdoor=False))

    def test_outdoor_day_interior_estimate_becomes_soft_daylight(self):
        m = cam("m", (0, -7, 1.6))
        light = MV.world_light({"light_type": "interior_window", "key_rel_azimuth_deg": 0, "key_elevation_deg": 20,
                                "sky_words": "上午柔和天光"}, m)
        out = MV.light_sentence(light, m, outdoor=True)
        self.assertIn("阴天", out)
        self.assertNotIn("房间", out)

    def test_schema_offers_night_artificial(self):
        self.assertIn("night_artificial", MV.LIGHT_SCHEMA["properties"]["light_type"]["enum"])
        self.assertIn("night_artificial", MV.LIGHT_PROMPT)

    @unittest.skipUnless((CYBER / "out").is_dir(), "cyber orbit run not present")
    def test_recorded_cyber_light(self):
        lp = next((CYBER / "out").rglob("light.json"))
        light = json.loads(lp.read_text(encoding="utf-8"))["light"]
        k = cam("k", (7, 0, 1.6))
        self.assertEqual(MV.light_kind(light, True), "night_artificial")
        self.assertNotIn("房间", MV.light_sentence(light, k, outdoor=True))


def night_street(h=90, w=160, neon=True):
    img = np.full((h, w, 3), 18.0, np.float32)
    img[:30] = 8.0                                     # dark sky
    img[30:, :] = 30.0                                 # dark facades / road
    if neon:
        img[35:45, 10:30] = (250, 30, 40)              # red sign
        img[50:58, 120:150] = (40, 90, 255)            # blue sign
        img[40:46, 70:90] = (255, 240, 200)            # warm-white lightbox
    return img


class StyleLock(unittest.TestCase):
    def test_night_with_neon_is_measured(self):
        img = night_street()
        sky = np.zeros(img.shape[:2], bool)
        sky[:30] = True
        st = MV.master_style(img, sky)
        self.assertTrue(st["night"])
        self.assertGreater(st["emissive"], 0.02)
        self.assertIn("红", st["colours"])
        self.assertIn("蓝", st["colours"])

    def test_day_is_not_night(self):
        img = np.full((90, 160, 3), 150.0, np.float32)
        img[:30] = (120, 170, 235)
        sky = np.zeros(img.shape[:2], bool)
        sky[:30] = True
        st = MV.master_style(img, sky)
        self.assertFalse(st["night"])
        t = MV.style_lock_text(st, {"palette": {"lock_zh": "初夏晴天"}})
        self.assertIn("风格与光线锁定", t)
        self.assertIn("初夏晴天", t)
        self.assertNotIn("招牌", t)
        self.assertNotIn("夜景", t)

    def test_night_text_locks_signs_and_lighting(self):
        an = {"environment_description": "Large yellow-white lightboxes and red, blue, and orange signs, lanterns "
                                         "and street lamps at night", "palette": {"lock_zh": "夏夜街景"}}
        st = MV.master_style(night_street())
        t = MV.style_lock_text(st, an, LightWording.NIGHT)
        for w in ("必须保持", "夜景", "灯箱", "灯笼", "招牌", "真的在发光", "不要画成白天", "棕灰", "夏夜街景",
                  "不要画成湿路"):
            self.assertIn(w, t)
        wet = MV.style_lock_text(st, {"environment_description": "wet asphalt with neon reflections"})
        self.assertIn("倒影", wet)

    def test_view_text_carries_the_lock(self):
        t = MV.view_text("体块", "光线锁定：夜景", ["主视角"], style_line="风格与光线锁定（必须保持）：夜景")
        self.assertIn("风格与光线锁定", t)
        self.assertLess(t.index("光线锁定：夜景"), t.index("风格与光线锁定"))
        self.assertNotIn("风格与光线锁定", MV.view_text("体块", "", ["主视角"]))

    @unittest.skipUnless((CYBER / "master.png").is_file(), "cyber master not present")
    def test_recorded_cyber_master_is_night_with_lights(self):
        PNG = load_package_module("png_io")
        img = PNG.read_png(CYBER / "master.png").astype(np.float32)[..., :3]
        st = MV.master_style(img)
        self.assertTrue(st["night"], st)
        self.assertGreater(st["emissive"], 0.005, st)


class NightGrade(unittest.TestCase):
    def setUp(self):
        self.m = view(cam("m", (4, -7, 2)), master=True, with_image=True)
        self.m = dict(self.m, img=np.asarray(self.m["img"], np.float32) * 0.25)      # a dark master
        self.k = view(cam("k", (6.5, -4.5, 2)))
        hit = self.k["pos"][..., 3:4] > 0.5
        self.truth = np.where(hit, texture(self.k["pos"][..., :3]) * 0.25, np.array([10, 12, 20.0])).astype(np.float32)

    def test_emissive_and_local_contrast_kept(self):
        img = np.clip(self.truth * 1.6 + 6, 0, 255)
        H, W = img.shape[:2]
        img[H // 3:H // 3 + 6, W // 8:W // 8 + 12] = (250, 30, 40)                # a lit sign
        g = MV.fit_grade(self.m, self.k, img, night=True)
        self.assertIsNotNone(g)
        self.assertTrue(g["night"])
        A = np.asarray(g["A"])
        for c in range(3):
            self.assertGreaterEqual(A[c, c], MV.GRADE_MIN_GAIN - 1e-6)
        out = MV.apply_grade(img, g, self.k["mask"])
        sign = (slice(H // 3 + 1, H // 3 + 5), slice(W // 8 + 1, W // 8 + 11))
        self.assertLess(float(np.abs(out[sign] - img[sign]).max()), 3.0)            # the sign still glows
        env = (~LO.dilate(self.k["mask"] > 0.5, 4)) & (self.k["pos"][..., 3] > 0.5)
        hp = lambda x: MV.lab(x)[..., 0] - LO.blur(MV.lab(x)[..., 0], 3)          # noqa: E731
        self.assertGreater(float(hp(out)[env].std()), 0.85 * float(hp(img)[env].std()))
        prod = self.k["mask"] > 0.5
        self.assertTrue(np.array_equal(out[prod], img[prod]))

    def test_day_path_unchanged(self):
        img = np.clip(self.truth * 1.6 + 6, 0, 255)
        g = MV.fit_grade(self.m, self.k, img)
        if g is not None:
            self.assertNotIn("night", g)


class OcclusionPlan(unittest.TestCase):
    def test_candidates_go_straight_in_first(self):
        c = MV.pose_candidates((12.0, 0.0, 1.8), (0.0, 0.0, 0.8), 3.0)
        first = [x for x in c if x["d_az"] == 0 and x["dh"] == 0]
        self.assertEqual(c[:len(first)], first)
        self.assertEqual([x["r"] for x in first], sorted([x["r"] for x in first], reverse=True))
        for x in c:
            self.assertGreaterEqual(x["dist"], 3.0 - 1e-6)
            self.assertAlmostEqual(math.hypot(x["pos"][0], x["pos"][1]), x["dist"], places=3)
        az = [abs(x["d_az"]) for x in c[len(first):]]
        self.assertLessEqual(az[0], 5.0)
        self.assertEqual(c[0]["pos"][2], 1.8)

    def street_eval(self, half=4.5):
        """Narrow street along Y, buildings at |x| > half: a camera behind a facade sees nothing."""
        def ev(c):
            x, y, _z = c["pos"]
            return {"occ": 1.0 if abs(x) > half else 0.0, "free": True, "lens_scale": 1.0}
        return ev

    def test_narrow_street_orbit_moves_inside_the_street(self):
        a = math.radians(72)
        pos = (12.6 * math.sin(a), 12.6 * math.cos(a), 1.86)
        best, tried = MV.plan_clear_pose(MV.pose_candidates(pos, (0, 0, 1.0), 3.7), self.street_eval())
        self.assertIsNotNone(best)
        self.assertEqual(best["d_az"], 0.0)                  # kept its azimuth: moved straight in
        self.assertLessEqual(abs(best["pos"][0]), 4.5)
        self.assertGreater(best["dist"], 3.7)
        self.assertGreater(tried, 1)

    def test_no_clear_pose_means_skip(self):
        best, tried = MV.plan_clear_pose(MV.pose_candidates((12, 0, 1.8), (0, 0, 1), 3.7),
                                         lambda c: {"occ": 1.0, "free": True, "lens_scale": 1.0})
        self.assertIsNone(best)
        self.assertGreater(tried, 50)
        unframed = MV.plan_clear_pose(MV.pose_candidates((12, 0, 1.8), (0, 0, 1), 3.7),
                                      lambda c: {"occ": 0.0, "free": True, "lens_scale": None})[0]
        self.assertIsNone(unframed)

    def test_notes(self):
        res = [{"name": "A", "ok": True, "calls": 1, "occlusion": {"status": "adjusted", "occ_before": 0.61,
                                                                    "occ_after": 0.02, "dist": 12.6, "dist_after": 4.4}},
               {"name": "B", "ok": False, "calls": 0, "occlusion": {"status": "blocked", "occ_before": 1.0}}]
        n = MV.occlusion_note(res)
        self.assertIn("A（遮挡 61%→2%", n)
        self.assertIn("产品被遮挡，已跳过/请调整相机：B", n)
        self.assertIn(n, MV.summary_line(res))

    def test_operator_wiring(self):
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        i_adj = ops.index("occ = MV.adjust_camera_for_occlusion(context, cam, coll)")
        i_render = ops.index("ok = render_clay_base(context, files[\"clay\"], coll", i_adj)
        self.assertLess(i_adj, i_render)                      # checked before the passes / any image call
        i_skip = ops.index('res["error"] = "产品被遮挡，已跳过/请调整相机"')
        i_edit = ops.index("r = CD.edit_image(prompt=\"\", edit_base_path=base", i_skip)
        self.assertLess(i_skip, i_edit)
        self.assertIn("MV.restore_camera(snap)", ops)
        self.assertIn("not night and RW.low_tier_retry(t[\"image_quality\"], n, mv_calls, False)", ops)
        self.assertIn("MV.fit_grade(master_view, vk, resize(img_full, ww, wh), night=night)", ops)
        self.assertIn("MV.light_sentence(light, rec, outdoor=outdoor)", ops)
        self.assertIn("style_line=style_line", ops)
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        self.assertIn("mv_avoid_occlusion: BoolProperty(", props)
        self.assertIn('"mv_avoid_occlusion", text="被遮挡就调视角"', (PKG / "ui.py").read_text(encoding="utf-8"))


@unittest.skipUnless(BLENDER and os.environ.get("WONDFUL_SKIP_BLENDER_TESTS") != "1", "blender not on PATH")
class BlenderCases(unittest.TestCase):
    """Real builder + widen_blockout / describe_blocks / occlusion adjust in headless Blender (no network)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="wondful_129_")
        cls.cache = {}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_case(self, case, fixture=""):
        if case not in self.cache:
            out = os.path.join(self.tmp, case + ".json")
            args = [BLENDER, "-b", "--factory-startup", "--python", str(HERE / "interior_occlusion_runner.py"), "--",
                    case, out] + ([str(FIX / fixture)] if fixture else [])
            subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
            self.cache[case] = json.loads(Path(out).read_text(encoding="utf-8"))
        rep = self.cache[case]
        self.assertTrue(rep.get("ok"), rep.get("error"))
        return rep

    def test_street_with_completion_names_stays_outdoor(self):
        rep = self.run_case("street", "street_v4.json")
        self.assertEqual(rep["kind"], "outdoor")
        self.assertFalse(rep["widen"]["interior"])
        self.assertEqual(rep["widen"]["closure"], [])
        self.assertGreaterEqual(rep["widen"]["wall_like"], 0)
        self.assertEqual(rep["items_interior"], 0)
        self.assertFalse(rep["text_interior"])
        self.assertEqual(rep["aux_left"], 0)

    def test_untagged_street_names_alone_not_a_room(self):
        rep = self.run_case("street_untagged", "street_v4.json")
        self.assertEqual(rep["kind"], "")
        self.assertGreaterEqual(rep["widen"]["wall_like"], 2)          # the old rule would have closed it in
        self.assertFalse(rep["widen"]["interior"])
        self.assertFalse(rep["text_interior"])

    def test_real_interior_with_chair_still_a_room(self):
        rep = self.run_case("interior", "interior_v3.json")
        self.assertEqual(rep["kind"], "interior")
        self.assertTrue(rep["widen"]["interior"])
        self.assertTrue(rep["text_interior"])
        self.assertEqual(rep["items_interior"], rep["items"])

    def test_untagged_interior_found_by_shape(self):
        rep = self.run_case("interior_untagged", "interior_v3.json")
        self.assertEqual(rep["kind"], "")
        self.assertTrue(rep["widen"]["interior"])
        self.assertTrue(rep["text_interior"])

    def test_occlusion_adjust_restore_and_skip(self):
        rep = self.run_case("occlusion")
        mv = rep["moved"]
        self.assertEqual(mv["status"], "adjusted")
        self.assertGreater(mv["occ_before"], 0.9)
        self.assertLessEqual(mv["occ_after"], MV.OCCLUDE_MAX)
        self.assertEqual(mv["d_az_deg"], 0.0)
        self.assertLess(mv["dist_after"], 5.0)                          # in front of the facade at x = 5
        self.assertIn("原相机未改", mv["message"])
        self.assertTrue(rep["moved_matrix_changed"])
        self.assertTrue(rep["restored"])
        self.assertEqual(rep["clear"]["status"], "clear")
        self.assertNotIn("pos_after", rep["clear"])
        self.assertEqual(rep["blocked"]["status"], "blocked")
        self.assertIn("已跳过", rep["blocked"]["message"])
        self.assertTrue(rep["blocked_untouched"])


if __name__ == "__main__":
    unittest.main()


# ----------------------------------------------------------------- orbit-minecraft findings (2026-10-21)
SS = load_package_module("scene_style")
CD = load_package_module("codex_direct")
DBk = load_package_module("detail_blockout")
MC = Path("/workspace/work/scene_blockout_impl/orbit5_minecraft/blockout_cache/scene_analysis_r1.json")
VOXEL = {"scene_kind": "outdoor",
         "scene_summary": "A broad cobblestone street in a block-built village",
         "environment_description": "This is a stylized, voxel-built village in summer … Preserve the square-edged "
                                    "construction, pixelated material variation, and cubic vegetation.",
         "environment_zh": "夏日方块村庄，蓝天方形白云", "palette": {"lock_zh": "夏日晴天", "sky_words": "vivid blue sky"}}
NIGHT_AN = {"scene_kind": "outdoor", "environment_description": "A warm summer night street with red and blue neon signs, "
                                                                "lightboxes and lanterns", "palette": {"lock_zh": "夏夜街景"}}
REAL = {"scene_kind": "outdoor", "environment_description": "A sunny suburban street with brick houses, hedges, a dark "
                                                            "asphalt road and blocky concrete garages",
        "palette": {"lock_zh": "初夏晴天"}}


class MinecraftInterior(unittest.TestCase):
    def test_llm_named_walls_and_windows_do_not_make_a_room(self):
        # 21 wall / window names (山墙房、墙柱、大窗、石矮墙 — the LLM's own), 0 / 72 rays hit
        self.assertFalse(MV.decide_interior("outdoor", 0, False, 21))
        self.assertFalse(MV.decide_interior("", 0, False, 21))
        self.assertFalse(MV.decide_interior("", 17, False, 21))


class SceneStyle(unittest.TestCase):
    def test_detects_stylized_only_when_described(self):
        self.assertTrue(SS.art_style(SS.analysis_text(VOXEL)))
        self.assertEqual(SS.art_style(SS.analysis_text(REAL)), [])
        self.assertEqual(SS.art_style(SS.analysis_text(NIGHT_AN)), [])
        self.assertTrue(SS.art_style("", "我的世界方块风格"))        # the user's 一句话需求
        self.assertEqual(SS.art_style("不要画成纯色方块，不要保留灰蓝色"), [])   # massing wording is not a style
        for fx in FIX.glob("*.json"):
            self.assertEqual(SS.art_style(SS.analysis_text(json.loads(fx.read_text(encoding="utf-8")))), [], fx.name)

    @unittest.skipUnless(MC.is_file(), "minecraft analysis not present")
    def test_recorded_minecraft_analysis(self):
        an = json.loads(MC.read_text(encoding="utf-8"))
        self.assertIn("体素方块", SS.art_style(SS.analysis_text(an)))
        self.assertIn("画风锁定", SS.master_lock(an))

    def test_master_lock(self):
        self.assertEqual(SS.master_lock(REAL), "")                     # realistic day: prompt unchanged
        v = SS.master_lock(VOXEL)
        self.assertIn("画风锁定", v)
        self.assertIn("不要画成写实照片", v)
        n = SS.master_lock(NIGHT_AN)
        self.assertIn("夜景", n)
        self.assertIn("霓虹灯", n)
        self.assertNotIn("画风锁定", n)

    def test_completion_hints_carry_the_look(self):
        self.assertIn("画风", DBk.inferred_hint(8, VOXEL))
        self.assertIn("体素方块", DBk.inferred_hint(10, VOXEL)[:20])    # 电杆; in front (names are cut at 60)
        self.assertTrue(DBk.inferred_hint(8, NIGHT_AN).startswith("【夜景、立面亮着招牌灯光】"))
        self.assertNotIn("招牌", DBk.inferred_hint(11, NIGHT_AN))        # a curb gets no signs
        self.assertEqual(DBk.inferred_hint(10, REAL), DBk.INFER_HINT[10])
        self.assertEqual(DBk.inferred_hint(10, None), DBk.INFER_HINT[10])
        src = (PKG / "detail_blockout.py").read_text(encoding="utf-8")
        self.assertEqual(src.count("hint=inferred_hint(int(c), analysis))"), 2)

    def test_new_view_lock_keeps_the_style(self):
        day = MV.master_style(np.full((90, 160, 3), 150.0, np.float32))
        t = MV.style_lock_text(day, VOXEL)
        self.assertIn("画风", t)
        self.assertIn("不要画成写实照片", t)
        self.assertNotIn("画风", MV.style_lock_text(day, REAL))


class SimplePromptStyle(unittest.TestCase):
    def test_realistic_wording_unchanged(self):
        p = CD.build_simple_prompt("", env_text="晴天郊区街道", vehicle=True, massing_text="体块说明")
        self.assertIn("写实的商业广告摄影照片", p)
        self.assertTrue(p.endswith(CD.ENDING_CG))
        self.assertEqual(p, CD.build_simple_prompt("", env_text="晴天郊区街道", vehicle=True, massing_text="体块说明",
                                                   style_text=""))

    def test_stylized_reference_drops_photoreal_wording(self):
        lock = SS.master_lock(VOXEL)
        for kw in ({"style_text": lock}, {"env_text": "voxel-built village with pixelated textures"}):
            p = CD.build_simple_prompt("", vehicle=True, massing_text="体块说明", **kw)
            self.assertNotIn("写实的商业广告摄影照片", p)
            self.assertNotIn("照片质感真实自然", p)
            self.assertIn("不是写实照片", p)
            self.assertIn("不要画成写实照片", p)
        p = CD.build_simple_prompt("我的世界方块风格", vehicle=False)
        self.assertNotIn("照片质感真实自然", p)
        p = CD.build_simple_prompt("", vehicle=True, massing_text="体块说明", style_text=lock)
        self.assertIn("画风锁定", p)
        once = CD.build_simple_prompt("", vehicle=True, massing_text="体块说明\n" + lock, style_text=lock)
        self.assertEqual(once.count("画风锁定"), 1)                   # 多角度: already in the view text

    def test_night_lock_reaches_a_realistic_prompt(self):
        lock = SS.master_lock(NIGHT_AN)
        p = CD.build_simple_prompt("", vehicle=True, massing_text="体块说明", style_text=lock)
        self.assertIn("写实的商业广告摄影照片", p)                     # still photoreal
        self.assertIn("夜景", p)

    def test_plumbing(self):
        cd = (PKG / "codex_direct.py").read_text(encoding="utf-8")
        self.assertIn("style_text=style_text if use_simple else \"\"", cd)
        self.assertIn("keep_text=keep_text,\n                                   style_text=style_text)", cd)
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        self.assertIn("direct_style_text = _master_lock(json.loads(_txt.as_string()), direct_simple_brief or \"\")", ops)
        self.assertIn("style_text=direct_style_text,", ops)
        self.assertIn("massing_text=text, keep_text=keep_text, style_text=style_line)", ops)


class BlockedHint(unittest.TestCase):
    def test_hint_search_is_wider_and_never_applied(self):
        self.assertGreater(min(abs(a) for a in MV.POSE_AZ_SUGGEST), max(MV.POSE_AZ))
        src = (PKG / "multiview.py").read_text(encoding="utf-8")
        i = src.index('out["suggest"] =')
        self.assertLess(i, src.index("    if apply:\n        cam_obj.matrix_world"))
