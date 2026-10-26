"""1.2.6 只做场景相关物体 — no pedestrians / other vehicles / animals in the ④ blockout or any image request.

User 2026-10-21: 「不要加行人」「只做场景相关物体」. ④ 场景粗模 (LLM blocks, 细节粗模 measured props, 场景补全 stats)
keeps only static scene objects; every image prompt (主视角, 多角度, 全景, 细节修复, 分层输出; Codex 直连, codex exec,
Antigravity) carries one Chinese + English negative line; the 一句话需求 re-enables a category only when it asks
for it (「街上有行人」), never when it mentions it negated (「不要行人」).
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402
from test_321_direct_edit import DirectEditTestBase  # noqa: E402

SS = load_package_module("static_scene")
SB = load_package_module("scene_blockout")
DB = load_package_module("detail_blockout")
MV = load_package_module("multiview")
direct = load_package_module("codex_direct")
PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"
NEON = Path("/workspace/work/scene_blockout_impl/pano_v2/neon/run/session_neon/scene_analysis_r1.json")


def prim(id_, type_, label="", material="", **kw):
    p = {"id": id_, "type": type_, "label_zh": label, "material": material, "shape": "box", "x": 0, "y": 10,
         "w": 1, "d": 1, "h": 1}
    p.update(kw)
    return p


SHIBUYA = {"primitives": [
    prim("central_portrait_billboard", "sign", "人物广告牌"), prim("left_zebra_stripes", "sign", "左侧斑马线"),
    prim("right_glass_block", "building", "玻璃商业楼"), prim("right_streetlamp", "pole", "右侧路灯"),
    prim("central_gray_coat_woman", "person", "灰衣行人"), prim("opposite_crossing_crowd", "person", "对面行人群"),
    prim("left_white_taxi", "vehicle", "白色出租车"), prim("parked_bicycle", "furniture", "停着的自行车"),
    prim("stray_cat", "furniture", "流浪猫"), prim("bench", "furniture", "长椅"),
    prim("bicycle_rack", "furniture", "自行车停放架"), prim("vending_machine", "furniture", "自动售货机")],
    "foliage_masses": [{"id": "tree", "kind": "deciduous"}], "roads": [{"id": "x", "kind": "sidewalk", "label_zh": "人行道"}]}


class Categories(unittest.TestCase):
    def test_category_first(self):
        cat = {p["id"]: SS.category_of(p) for p in SHIBUYA["primitives"]}
        self.assertEqual(cat["central_gray_coat_woman"], "person")
        self.assertEqual(cat["opposite_crossing_crowd"], "person")
        self.assertEqual(cat["left_white_taxi"], "vehicle")
        # static categories never become dynamic by their words (a billboard with a person on it)
        self.assertEqual(cat["central_portrait_billboard"], "")
        self.assertEqual(cat["left_zebra_stripes"], "")
        self.assertEqual(cat["right_glass_block"], "")

    def test_generic_category_uses_words(self):
        cat = {p["id"]: SS.category_of(p) for p in SHIBUYA["primitives"]}
        self.assertEqual(cat["parked_bicycle"], "vehicle")
        self.assertEqual(cat["stray_cat"], "animal")
        self.assertEqual(cat["bench"], "")
        self.assertEqual(cat["bicycle_rack"], "")          # a static word in the same name wins
        self.assertEqual(cat["vending_machine"], "")
        self.assertEqual(SS.category_of(prim("cabinet", "furniture", "木柜")), "")   # "cab" is a whole word only

    def test_strip_dynamic(self):
        out, dropped = SS.strip_dynamic(SHIBUYA, allow=frozenset())
        ids = [p["id"] for p in out["primitives"]]
        self.assertEqual(len(dropped), 5)
        for gone in ("central_gray_coat_woman", "opposite_crossing_crowd", "left_white_taxi", "parked_bicycle", "stray_cat"):
            self.assertNotIn(gone, ids)
        for kept in ("central_portrait_billboard", "left_zebra_stripes", "bench", "bicycle_rack", "right_streetlamp"):
            self.assertIn(kept, ids)
        self.assertEqual(out["foliage_masses"], SHIBUYA["foliage_masses"])
        self.assertEqual(len(SHIBUYA["primitives"]), 12)     # input untouched
        note = SS.dropped_note(dropped)
        self.assertIn("行人 2", note)
        self.assertIn("车辆 2", note)
        self.assertIn("动物 1", note)

    def test_strip_respects_allow(self):
        out, dropped = SS.strip_dynamic(SHIBUYA, allow={"person"})
        ids = [p["id"] for p in out["primitives"]]
        self.assertIn("central_gray_coat_woman", ids)
        self.assertNotIn("left_white_taxi", ids)
        same, none = SS.strip_dynamic({"primitives": [prim("b", "building")]}, allow=frozenset())
        self.assertEqual(none, [])

    @unittest.skipUnless(NEON.is_file(), "recorded Shibuya analysis not present")
    def test_recorded_shibuya(self):
        an = json.loads(NEON.read_text(encoding="utf-8"))
        out, dropped = SS.strip_dynamic(an, allow=frozenset())
        self.assertEqual(sum(1 for c, _ in dropped if c == "person"), 11)
        self.assertEqual(sum(1 for c, _ in dropped if c == "vehicle"), 2)
        self.assertFalse(any(p["type"] in ("person", "vehicle") for p in out["primitives"]))
        self.assertTrue(any(p["label_zh"] == "人物广告牌" for p in out["primitives"]))

    def test_measured_props(self):
        self.assertEqual(SS.is_dynamic_prop(0.5, 1.7, allow=()), "person")
        self.assertEqual(SS.is_dynamic_prop(1.8, 1.5, allow=()), "vehicle")      # car end-on
        self.assertEqual(SS.is_dynamic_prop(4.5, 1.5, allow=()), "vehicle")      # car side-on
        self.assertEqual(SS.is_dynamic_prop(1.8, 0.45, allow=()), "")            # bench
        self.assertEqual(SS.is_dynamic_prop(1.0, 1.8, allow=()), "")             # vending machine / kiosk
        self.assertEqual(SS.is_dynamic_prop(0.5, 1.7, allow={"person"}), "")


class Brief(unittest.TestCase):
    def test_asks(self):
        for b, cat in (("街上有行人走过", "person"), ("人来人往的街头", "person"), ("几个人在等红灯", "person"),
                       ("with pedestrians crossing", "person"), ("busy street with taxis", "vehicle"),
                       ("周围停着出租车", "vehicle"), ("一只狗趴在车旁", "animal")):
            self.assertIn(cat, SS.wanted(b), b)

    def test_does_not_ask(self):
        for b in ("", "雪地清晨，冷色调", "涩谷十字路口夜景，不要行人", "没有车流和行人", "no people, empty street",
                  "人行道很宽", "小米YU7停在路口", "红色的车停在雪地", "a cabin in the woods", "无人的街道"):
            self.assertEqual(SS.wanted(b), frozenset(), b)

    def test_set_brief(self):
        try:
            self.assertEqual(SS.set_brief("街上有行人"), frozenset({"person"}))
            self.assertEqual(SS.allowed(), frozenset({"person"}))
            self.assertNotIn("行人", SS.negative_text())
        finally:
            SS.set_brief("")
        self.assertEqual(SS.allowed(), frozenset())


class Prompts(unittest.TestCase):
    def test_negative_line_zh_en(self):
        t = SS.negative_text(allow=())
        for w in ("行人", "人群", "其他车辆", "自行车", "动物", "静态物体", "pedestrians", "people", "other vehicles",
                  "bicycles", "animals", "static scene", "the product itself excepted"):
            self.assertIn(w, t)
        self.assertEqual(SS.negative_text(allow={"person", "vehicle", "animal"}), "")
        t2 = SS.negative_text(allow={"person"})
        self.assertNotIn("行人", t2)
        self.assertNotIn("pedestrians", t2)
        self.assertIn("其他车辆", t2)

    def test_apply_idempotent(self):
        a = SS.apply("主视角提示词", allow=())
        self.assertTrue(a.startswith("主视角提示词\n"))
        self.assertEqual(SS.apply(a, allow=()), a)
        self.assertEqual(a.count(SS.MARKER), 1)
        self.assertIn(SS.MARKER_EN, SS.apply("", allow=()))
        self.assertEqual(SS.apply("x", allow={"person", "vehicle", "animal"}), "x")

    def test_scrub(self):
        zh = SS.scrub("夜晚涩谷十字路口，行人穿过斑马线，巨型人物广告牌发光，出租车在路口等待。人行道湿润反光。", allow=())
        self.assertNotIn("行人穿过", zh)
        self.assertNotIn("出租车", zh)
        self.assertIn("人物广告牌", zh)
        self.assertIn("人行道湿润反光", zh)
        en = SS.scrub("Crowds of pedestrians fill the crossing. Wet asphalt reflects neon at 3.5 m; taxis wait at the "
                      "kerb, pedestrian crossing stripes are white.", allow=())
        self.assertNotIn("pedestrians fill", en)
        self.assertNotIn("taxis", en)
        self.assertIn("3.5 m", en)
        self.assertIn("pedestrian crossing stripes", en)
        self.assertIn("行人", SS.scrub("街上有行人", allow={"person"}))

    def test_scene_note_scrubbed(self):
        n = SB.scene_note({"environment_zh": "夜晚十字路口，行人穿行，霓虹招牌林立", "palette": {"lock_zh": "夏夜，路面湿润"}})
        self.assertNotIn("行人", n)
        self.assertIn("霓虹招牌", n)

    def test_style_lock_scrubbed(self):
        st = {"night": False, "median_L": 60}
        t = MV.style_lock_text(st, {"palette": {"lock_zh": "初夏晴天，行人很多，草地鲜绿"}})
        self.assertNotIn("行人", t)
        self.assertIn("草地鲜绿", t)

    def test_analysis_rules(self):
        self.assertIn("MOVING THINGS", SB.RULES)
        self.assertIn("never turns these into blocks", SB.RULES)
        self.assertIn("Describe only the static scene", SB.RULES)
        SB.RULES.format(fw=1600, fh=900, aspect_txt="16:9", hw=0.5625, product_words="a car", pw=1.9, pd=4.8, ph=1.5)


class DirectEditNegative(DirectEditTestBase):
    def test_every_direct_request_carries_the_line(self):
        SS.set_brief("")
        self.edit()
        log = json.loads((self.tmp / "codex_direct_t1.json").read_text(encoding="utf-8"))
        self.assertIn(SS.MARKER, log["prompt"])
        self.assertIn(SS.MARKER_EN, log["prompt"])
        # exact prompts (分层输出 clean plate, 全景) too
        self.edit(prompt="Paint the 360 panorama.", exact_prompt=True, tag="t2")
        log2 = json.loads((self.tmp / "codex_direct_t2.json").read_text(encoding="utf-8"))
        self.assertTrue(log2["prompt"].startswith("Paint the 360 panorama."))
        self.assertIn(SS.MARKER, log2["prompt"])

    def test_brief_asking_for_people_lifts_it(self):
        try:
            SS.set_brief("街上有行人")
            self.edit(tag="t3")
            log = json.loads((self.tmp / "codex_direct_t3.json").read_text(encoding="utf-8"))
            self.assertNotIn("行人、人物或人群", log["prompt"])
            self.assertIn("其他车辆", log["prompt"])
        finally:
            SS.set_brief("")


class Wiring(unittest.TestCase):
    def test_all_image_paths(self):
        cd = (PKG / "codex_direct.py").read_text(encoding="utf-8")
        self.assertIn("text = _SS.apply(text)", cd)
        self.assertIn("env_text = _SS0.scrub(env_text)", cd)
        for f in ("codex_client.py", "antigravity_client.py"):
            src = (PKG / f).read_text(encoding="utf-8")
            i = src.index("def generate_image(")
            self.assertIn("prompt = _SS.apply(prompt)", src[i:i + 1500], f)

    def test_blockout_and_task_start(self):
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        i = ops.index("def _start_thread(")
        self.assertIn("_SS.set_brief(", ops[i:i + 900])
        self.assertIn("_SS.strip_dynamic(data, _SS.wanted(", ops)
        sbb = (PKG / "scene_blockout_build.py").read_text(encoding="utf-8")
        i = sbb.index("\ndef build(context, data")
        body = sbb[i:i + 1200]
        self.assertLess(body.index("SS.strip_dynamic(data, allow)"), body.index("S = SB.normalize(data)"))
        self.assertIn("SS.dropped_note(dropped)", sbb)
        db = (PKG / "detail_blockout.py").read_text(encoding="utf-8")
        self.assertIn("dyn = SSm.is_dynamic_prop(W, hh, D=Dm, static=is_static)", db)


if __name__ == "__main__":
    unittest.main()
