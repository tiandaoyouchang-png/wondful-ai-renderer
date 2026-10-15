"""1.2.0 环境体块 › 建筑参考: per-block reference storage, image cap / priority, prompt mapping,
direct-edit payload (clay + building refs, env still text only) and the codex exec fallback."""
import base64
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402
from test_321_direct_edit import DirectEditTestBase, rgb_png  # noqa: E402

MS = load_package_module("massing")
CD = load_package_module("codex_direct")
PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"


class FakeBlock(dict):
    """ID-property semantics of a Blender object (keys / get / [] / del) plus a parent."""

    def __init__(self, name, parent=None):
        super().__init__()
        self.name, self.parent = name, parent

    __hash__ = object.__hash__

    def __eq__(self, other):
        return self is other


class StorageTests(unittest.TestCase):
    def test_parse_dump_roundtrip(self):
        self.assertEqual(MS.parse_refs(None), [])
        self.assertEqual(MS.parse_refs(""), [])
        self.assertEqual(MS.parse_refs("/a/b.jpg"), ["/a/b.jpg"])           # a plain string = one path
        self.assertEqual(MS.parse_refs('["/a.png", "/a.png", " ", "//rel/c.jpg"]'), ["/a.png", "//rel/c.jpg"])
        self.assertEqual(MS.parse_refs(["x.png", "y.png"]), ["x.png", "y.png"])
        self.assertEqual(MS.parse_refs("{bad json"), ["{bad json"])
        self.assertEqual(MS.parse_refs("42"), [])                           # JSON, but not a list of paths
        text = MS.dump_refs(["/图/石库门.jpg", "/b.png"])
        self.assertEqual(json.loads(text), ["/图/石库门.jpg", "/b.png"])
        self.assertIn("石库门", text)                                       # readable, not \\u-escaped

    def test_merge_limit_and_duplicates(self):
        kept, dropped = MS.merge_refs('["a", "b"]', ["b", "c", "d"])
        self.assertEqual(kept, ["a", "b", "c"])
        self.assertEqual(dropped, ["d"])
        kept, dropped = MS.merge_refs([], ["a"], limit=3)
        self.assertEqual((kept, dropped), (["a"], []))

    def test_each_block_keeps_its_own_refs(self):
        tower, house = FakeBlock("高楼"), FakeBlock("低层建筑")
        MS.set_refs(tower, ["/glass.jpg"])
        MS.set_refs(house, ["/brick1.jpg", "/brick2.jpg"])
        self.assertEqual(MS.get_refs(tower), ["/glass.jpg"])
        self.assertEqual(MS.get_refs(house), ["/brick1.jpg", "/brick2.jpg"])
        self.assertIsInstance(tower[MS.REF_PROP], str)                     # stored as a JSON string ID property
        MS.set_refs(tower, [])
        self.assertNotIn(MS.REF_PROP, tower)
        self.assertEqual(MS.get_refs(tower), [])
        self.assertEqual(MS.get_refs(None), [])

    def test_block_root_walks_to_the_root_block(self):
        root = FakeBlock("楼")
        child = FakeBlock("楼_檐口", parent=root)
        other = FakeBlock("别的")
        objs = [root, child]
        self.assertIs(MS.block_root(child, objs), root)
        self.assertIs(MS.block_root(root, objs), root)
        self.assertIsNone(MS.block_root(other, objs))
        self.assertIsNone(MS.block_root(None, objs))
        self.assertEqual(MS.block_roots(objs), [root])


class SelectionTests(unittest.TestCase):
    def items(self):
        return [
            {"hint": "高楼", "place": "画面左侧，在产品左边，背景（产品后方）", "box": [0.0, 0.35, 0.54, 3.7],
             "refs": ["t1", "t2", "t3"]},                                      # area 0.35 × 0.46
            {"hint": "围墙", "place": "画面中间，在产品正后方", "box": [0.19, 0.5, 0.48, 0.58], "refs": []},
            {"hint": "", "place": "画面右侧，在产品右边", "box": [0.62, 1.08, 0.5, 0.92], "refs": ["h1", "h2"]},  # 0.38 × 0.42
            {"hint": "小亭子", "place": "画面右侧", "box": [0.9, 0.95, 0.4, 0.45], "refs": ["k1"]},  # tiny
        ]

    def test_cap_prefers_larger_blocks_one_each_first(self):
        sel = MS.select_building_refs(self.items(), cap=4)
        self.assertEqual(len(sel), 4)
        blocks = [s["block"] for s in sel]
        self.assertEqual(blocks, sorted(blocks, key=lambda b: [1, 3, 4].index(b)))   # grouped by block, priority order
        self.assertEqual(sorted(set(blocks)), [1, 3, 4])                             # every block gets one first
        self.assertEqual([s["path"] for s in sel], ["t1", "t2", "h1", "k1"])

    def test_cap_two_drops_the_smallest_block(self):
        sel = MS.select_building_refs(self.items(), cap=2)
        self.assertEqual([(s["block"], s["path"]) for s in sel], [(1, "t1"), (3, "h1")])

    def test_under_cap_takes_all_and_missing_files_skipped(self):
        it = self.items()[:3]
        sel = MS.select_building_refs(it, cap=10, exists=lambda p: p != "t2")
        self.assertEqual([s["path"] for s in sel], ["t1", "t3", "h1", "h2"])
        self.assertEqual(MS.select_building_refs([{"place": "x", "refs": []}]), [])
        self.assertEqual(MS.select_building_refs(it, cap=0), [])

    def test_default_cap_is_four_and_per_block_three(self):
        many = [{"hint": f"b{i}", "place": "画面中间", "box": [0, 1, 0, 1], "refs": [f"{i}a", f"{i}b", f"{i}c", f"{i}d"]}
                for i in range(1)]
        self.assertEqual(len(MS.select_building_refs(many)), 3)       # at most 3 per block
        self.assertEqual(MS.MAX_BUILDING_IMAGES, 4)
        self.assertAlmostEqual(MS.visible_area([-0.5, 0.5, 0.5, 2.0]), 0.25)
        self.assertEqual(MS.visible_area(None), 0.0)


class MappingTextTests(unittest.TestCase):
    SEL = [{"block": 1, "path": "/a.jpg", "hint": "高楼", "place": "画面左侧，在产品左边，背景（产品后方）"},
           {"block": 1, "path": "/b.jpg", "hint": "高楼", "place": "画面左侧，在产品左边，背景（产品后方）"},
           {"block": 3, "path": "/c.jpg", "hint": "", "place": "画面右侧，在产品右边，背景"}]

    def test_numbered_mapping(self):
        t = MS.building_ref_text(self.SEL, vehicle=True)
        self.assertIn("- 图2、图3 是画面左侧，在产品左边的体块 1「高楼」（灰蓝体块）的参考：按它的建筑风格、立面、材质、窗户和细节来画这栋建筑，"
                      "但保持体块的位置、高度、轮廓和透视；参考图里的机位、构图、天空、其他物体和车辆一律不要照搬。", t)
        self.assertIn("- 图4 是画面右侧，在产品右边的体块 3（灰蓝体块）的参考", t)
        self.assertIn("图 1 里不是灰蓝色的主体就是我们的车，它完全锁定：位置、大小、角度、透视、轮廓、颜色都与图 1 一致", t)
        self.assertIn("没有参考图的体块仍按场景风格自由设计", t)
        g = MS.building_ref_text(self.SEL[:1], vehicle=False)
        self.assertIn("我们的产品", g)
        self.assertNotIn("我们的车", g)
        self.assertEqual(MS.building_ref_text([]), "")

    def test_named_mapping_for_fallback(self):
        t = MS.building_ref_text(self.SEL, names=["a.jpg", "b.jpg", "c.jpg"])
        self.assertIn("参考图「a.jpg」、参考图「b.jpg」 是", t)
        self.assertNotIn("图2", t)
        self.assertNotIn("灰蓝", t)                      # the exec canvas is the camera capture, no grey-blue
        self.assertIn("相机底图里的产品完全锁定", t)

    def test_simple_prompt_with_building_text(self):
        mt = MS.build_massing_text([{"hint": "高楼", "place": "画面左侧"}], vehicle=True)
        bt = MS.building_ref_text(self.SEL[:1], vehicle=True)
        with_b = CD.build_simple_prompt("城市街道", env_text="晴天的街道", vehicle=True, massing_text=mt, building_text=bt)
        self.assertTrue(with_b.startswith("把图 1（Blender 白模渲染图）直接编辑成"))
        self.assertLess(with_b.index(mt), with_b.index(bt))
        self.assertIn("有建筑参考图的体块按各自参考图", with_b)
        self.assertIn("车身的光照", with_b)                                   # product lock / paint rules unchanged
        plain = CD.build_simple_prompt("城市街道", env_text="晴天的街道", vehicle=True, massing_text=mt)
        self.assertTrue(plain.startswith("把这张 Blender 白模渲染图直接编辑成"))
        self.assertNotIn("建筑参考图", plain)
        self.assertIn("它们的风格和材质也按描述来", plain)
        # building text without blocks is ignored (nothing grey-blue to map to)
        self.assertNotIn("建筑参考图", CD.build_simple_prompt("街道", building_text=bt))


class BuildingPayloadTests(DirectEditTestBase):
    def sel(self, n):
        return [{"block": i + 1, "path": rgb_png(self.tmp / f"b{i}.png", 500, 400, (30 * i, 80, 140)),
                 "hint": f"楼{i + 1}", "place": "画面左侧，在产品左边"} for i in range(n)]

    def test_simple_path_sends_clay_plus_building_refs_not_env(self):
        refs = self.sel(5)
        result = self.edit(simple_brief="老上海街道", massing_text="灰蓝色体块：……", building_refs=refs, vehicle=True)
        self.assertEqual(result["building_refs"], [1, 2, 3, 4])               # capped at 4 extra images
        self.assertEqual(result["image_roles"], ["camera", "building", "building", "building", "building"])
        body = self.state.raw_bodies[-1]
        self.assertEqual(body["instructions"], CD.BUILDING_INSTRUCTIONS)
        content = body["input"][0]["content"]
        images = [c for c in content if c["type"] == "input_image"]
        self.assertEqual(len(images), 5)                                      # the env image is NOT attached
        self.assertEqual(base64.b64decode(images[0]["image_url"].split(",", 1)[1]), Path(self.camera).read_bytes())
        self.assertEqual(base64.b64decode(images[2]["image_url"].split(",", 1)[1]), Path(refs[1]["path"]).read_bytes())
        text = content[0]["text"]
        self.assertIn("图2 是画面左侧，在产品左边的体块 1「楼1」", text)
        self.assertIn("图5 是", text)
        self.assertNotIn("图6", text)
        self.assertIn("图 1 里不是灰蓝色的主体就是我们的车", text)

    def test_no_massing_text_means_no_building_images(self):
        result = self.edit(simple_brief="海边", building_refs=self.sel(2))
        self.assertEqual(result["building_refs"], [])
        body = self.state.raw_bodies[-1]
        self.assertEqual(body["instructions"], CD.INSTRUCTIONS)
        self.assertEqual(len([c for c in body["input"][0]["content"] if c["type"] == "input_image"]), 1)

    def test_simple_path_without_refs_unchanged(self):
        self.edit(simple_brief="海边黄昏", massing_text="灰蓝色体块：……")
        body = self.state.raw_bodies[-1]
        self.assertEqual(body["instructions"], CD.INSTRUCTIONS)
        self.assertEqual(len([c for c in body["input"][0]["content"] if c["type"] == "input_image"]), 1)
        self.assertNotIn("建筑参考图", body["input"][0]["content"][0]["text"])

    def test_non_simple_ignores_building_refs(self):
        result = self.edit(building_refs=self.sel(1), massing_text="x")
        self.assertEqual(result["building_refs"], [])


class FallbackTests(unittest.TestCase):
    def test_exec_fallback_gets_refs_and_named_text(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "glass.jpg"; a.write_bytes(b"x")
            sel = [{"block": 1, "path": str(a), "hint": "高楼", "place": "画面左侧"},
                   {"block": 2, "path": str(Path(d) / "missing.jpg"), "hint": "", "place": "画面右侧"}]
            seen = {}

            def fail_edit(**kw):
                seen["direct"] = kw
                raise CD.DirectEditError("boom", kind="request")

            def fallback(**kw):
                seen["fallback"] = kw
                return {"ok": True}
            old = CD.edit_image
            CD.edit_image = fail_edit
            try:
                base = Path(d) / "clay.png"; base.write_bytes(b"x")
                CD.generate_with_fallback({"output_path": str(Path(d) / "o.png"), "reference_paths": [str(base)],
                                           "edit_base_path": str(base), "prompt": "原提示词"},
                                          fallback=fallback, state={}, size="1536x1024", simple_base=str(base),
                                          simple_brief="街道", massing_text="M", building_refs=sel, vehicle=True)
            finally:
                CD.edit_image = old
        self.assertEqual(seen["direct"]["building_refs"], sel)
        fb = seen["fallback"]
        self.assertEqual(fb["reference_paths"][-1], str(a))
        self.assertEqual(len(fb["reference_paths"]), 2)                     # the missing file is not sent
        self.assertTrue(fb["prompt"].startswith("原提示词"))
        self.assertIn("参考图「glass.jpg」 是画面左侧的体块 1「高楼」", fb["prompt"])
        self.assertIn("相机底图里的车完全锁定", fb["prompt"])
        unchanged = CD.fallback_with_building_refs({"prompt": "p"}, None)
        self.assertEqual(unchanged, {"prompt": "p"})


class WiringTests(unittest.TestCase):
    def test_operator_and_capture_wiring(self):
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        self.assertIn("direct_building_refs = self._building_refs(LAST_CLAY_INFO.get(\"massing\") or [])", ops)
        self.assertIn("building_refs=direct_building_refs or None", ops)
        self.assertIn("MS.select_building_refs(items, exists=", ops)
        self.assertIn('bl_idname = "wondful.block_ref_add"', ops)
        self.assertIn('bl_idname = "wondful.block_ref_clear"', ops)
        ms = (PKG / "massing.py").read_text(encoding="utf-8")
        self.assertIn('"refs": refs', ms)                                    # describe_blocks reads each block's refs
        self.assertIn('REF_PROP = "wondful_ref_paths"', ms)
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        self.assertIn("_draw_block_refs(box, context, props, busy, blocks)", ui)
        vc = (PKG / "viewport_capture.py").read_text(encoding="utf-8")
        self.assertIn("is_holdout", vc)                                      # blocks never enter the product mask


if __name__ == "__main__":
    unittest.main()
