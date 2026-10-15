"""1.2.0 环境体块: name hints, placement words, prompt paragraph, temporary material swap."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

MS = load_package_module("massing")
CD = load_package_module("codex_direct")
ROOT = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"


class FakeSlot:
    """Blender MaterialSlot semantics: ``material`` reads/writes the data or object level by ``link``."""

    def __init__(self, data_mat, obj_mat=None, link="DATA"):
        self.data_mat, self.obj_mat, self.link = data_mat, obj_mat, link

    @property
    def material(self):
        return self.obj_mat if self.link == "OBJECT" else self.data_mat

    @material.setter
    def material(self, m):
        if self.link == "OBJECT":
            self.obj_mat = m
        else:
            self.data_mat = m


class FakeMats(list):
    pass


class FakeData:
    def __init__(self, owner):
        self.owner = owner
        self.materials = FakeMats()

    def sync(self):
        pass


class FakeObj:
    def __init__(self, name, slots):
        self.name = name
        self.data = FakeData(self)
        self._slots = slots

    @property
    def material_slots(self):
        # an appended data material creates a DATA-linked slot
        while len(self._slots) < len(self.data.materials):
            self._slots.append(FakeSlot(self.data.materials[len(self._slots)]))
        while len(self._slots) > len(self.data.materials) and self.data.materials is not None and self._added:
            self._slots.pop()
        return self._slots

    _added = False


def state(o):
    return [(s.link, s.data_mat, s.obj_mat) for s in o._slots], list(o.data.materials)


class MaterialSwapTests(unittest.TestCase):
    def test_swap_and_restore(self):
        a = FakeObj("Tower", [FakeSlot("brick"), FakeSlot("glass", obj_mat="objglass", link="OBJECT")])
        a.data.materials.extend(["brick", "glass"])
        b = FakeObj("Empty_mesh", [])
        b._added = True
        before_a, before_b = state(a), state(b)
        restore = []
        n = MS.swap_materials([a, b], "MASS", restore)
        self.assertEqual(n, 2)
        self.assertTrue(all(s.material == "MASS" for s in a.material_slots))
        self.assertTrue(all(s.link == "OBJECT" for s in a.material_slots))
        self.assertEqual([s.data_mat for s in a._slots], ["brick", "glass"], "mesh data materials untouched")
        self.assertEqual([s.material for s in b.material_slots], ["MASS"])
        for undo in reversed(restore):
            undo()
        self.assertEqual(state(a), before_a)
        self.assertEqual(state(b)[1], before_b[1])
        self.assertEqual(len(b.material_slots), 0)

    def test_non_geometry_skipped(self):
        class E:
            name = "Empty"
            data = None
        restore = []
        self.assertEqual(MS.swap_materials([E()], "MASS", restore), 0)
        self.assertEqual(restore, [])

    def test_colour_is_grey_blue_matte(self):
        r, g, b, a = MS.massing_rgba()
        self.assertTrue(b > g > r)
        self.assertAlmostEqual(MS.srgb_to_linear(159 / 255), r, places=6)
        self.assertEqual(a, 1.0)

        class Mat:
            def __init__(self, name):
                self.name = name
                self.diffuse_color = None
        m = MS.make_material(Mat)
        self.assertEqual(m.name, MS.MAT_NAME)
        self.assertEqual(m.diffuse_color, MS.massing_rgba())


class NameAndPlacementTests(unittest.TestCase):
    def test_auto_pick(self):
        self.assertEqual(MS.auto_pick(["Car", "环境", "体块_old"]), "环境")
        self.assertEqual(MS.auto_pick(["Product", "Context Massing"]), "Context Massing")
        self.assertEqual(MS.auto_pick(["massing", "car"], exclude=["massing"]), "")
        self.assertEqual(MS.auto_pick(["Car", "Lights"]), "")

    def test_hints(self):
        self.assertEqual(MS.hint_from_name("楼_玻璃幕墙"), "楼，玻璃幕墙")
        self.assertEqual(MS.hint_from_name("Tower_glass.001"), "高楼，玻璃幕墙")
        self.assertEqual(MS.hint_from_name("Cube.003"), "")
        self.assertEqual(MS.hint_from_name("立方体.002"), "")
        self.assertEqual(MS.hint_from_name("砖墙"), "砖墙")
        self.assertEqual(MS.hint_from_name("Cube", custom="石库门弄堂口"), "石库门弄堂口")
        self.assertEqual(MS.hint_from_name("block_tree_02"), "树")

    def test_placement(self):
        t = MS.placement_words(0.05, 0.3, 0.4, 1.2, "BACK", 6.2)
        self.assertIn("画面左侧", t)
        self.assertIn("背景", t)
        self.assertIn("顶部超出画面", t)
        self.assertIn("6.2 倍", t)
        t = MS.placement_words(0.75, 1.1, 0.2, 0.4, "SAME", 0.8)
        self.assertIn("画面右侧", t)
        self.assertIn("同一深度", t)
        self.assertIn("一侧延伸出画面", t)
        self.assertIn("比产品矮", t)
        self.assertIn("画面中间", MS.placement_words(0.4, 0.6, 0.1, 0.5, "FRONT"))
        self.assertIn("前景", MS.placement_words(0.4, 0.6, 0.1, 0.5, "FRONT"))
        self.assertIn("横跨画面", MS.placement_words(-0.2, 1.2, 0.2, 0.3))
        self.assertIn("在产品左边", MS.placement_words(0.1, 0.3, 0.2, 0.9, "BACK", product_x=(0.4, 0.6)))
        self.assertIn("在产品右边", MS.placement_words(0.7, 0.9, 0.2, 0.9, "BACK", product_x=(0.4, 0.6)))
        self.assertIn("在产品正后方", MS.placement_words(0.4, 0.6, 0.2, 0.9, "BACK", product_x=(0.35, 0.65)))

    def test_depth(self):
        self.assertEqual(MS.depth_word(15, 20, 10), "BACK")
        self.assertEqual(MS.depth_word(3, 6, 10), "FRONT")
        self.assertEqual(MS.depth_word(8, 14, 10), "SAME")
        self.assertEqual(MS.depth_word(8, 14, 0), "SAME")


class PromptTests(unittest.TestCase):
    ITEMS = [{"hint": "玻璃幕墙", "place": "画面左侧，背景（产品后方）"}, {"hint": "", "place": "画面右侧，前景"}]

    def test_paragraph(self):
        t = MS.build_massing_text(self.ITEMS, vehicle=True)
        for w in ("灰蓝色", "建筑", "位置、高度、轮廓和透视", "不要画成纯色方块", "不要把体块当成产品", "车继续严格锁定",
                  "体块 1「玻璃幕墙」：画面左侧", "体块 2：画面右侧"):
            self.assertIn(w, t)
        g = MS.build_massing_text(self.ITEMS, vehicle=False)
        self.assertIn("产品继续严格锁定", g)
        self.assertNotIn("车", g)
        self.assertEqual(MS.build_massing_text([]), "")

    def test_simple_prompt_with_and_without(self):
        mt = MS.build_massing_text(self.ITEMS)
        plain = CD.build_simple_prompt("城市街道", env_text="晴天的街道，柏油路")
        with_m = CD.build_simple_prompt("城市街道", env_text="晴天的街道，柏油路", massing_text=mt)
        self.assertNotIn("灰蓝色", plain)
        self.assertIn(mt, with_m)
        self.assertIn("灰蓝色体块对应的建筑", with_m)  # env-strict sentence allows the blocks
        self.assertIn("不额外添加其他建筑", with_m)
        self.assertIn("不额外添加建筑", plain)
        self.assertIn("不删除", with_m)
        self.assertIn("目标环境", with_m)  # env text still used alongside
        car = CD.build_simple_prompt("赛博朋克", vehicle=True, massing_text=MS.build_massing_text(self.ITEMS, True))
        self.assertIn("车身", car)
        self.assertIn("灰蓝色体块按上面的要求变成建筑", car)

    def test_massing_text_reaches_edit_image(self):
        seen = {}

        def fake_edit(**kw):
            seen.update(kw)
            return {"ok": True}
        old = CD.edit_image
        CD.edit_image = fake_edit
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                base = Path(d) / "clay.png"
                base.write_bytes(b"x")
                CD.generate_with_fallback({"output_path": str(Path(d) / "o.png"), "reference_paths": [str(base)],
                                           "edit_base_path": str(base)},
                                          fallback=lambda **k: None, state={}, size="1536x1024",
                                          simple_base=str(base), simple_brief="街道", massing_text="MASSING")
        finally:
            CD.edit_image = old
        self.assertEqual(seen.get("massing_text"), "MASSING")


class WiringTests(unittest.TestCase):
    def test_sources(self):
        vc = (ROOT / "viewport_capture.py").read_text(encoding="utf-8")
        self.assertIn("massing=None", vc)
        self.assertIn("is_holdout", vc)  # blocks cut the product mask, never enter the subject layer
        self.assertIn("_MASSING_EXCLUDE.clear()", vc)
        ops = (ROOT / "operators.py").read_text(encoding="utf-8")
        self.assertIn("massing_text=direct_massing_text", ops)
        ui = (ROOT / "ui.py").read_text(encoding="utf-8")
        self.assertIn("_draw_massing(adv, context, props, busy)", ui)
        props = (ROOT / "properties.py").read_text(encoding="utf-8")
        for k in ("massing_enabled", "massing_collection", "last_massing_note"):
            self.assertIn(k, props)


if __name__ == "__main__":
    unittest.main()
