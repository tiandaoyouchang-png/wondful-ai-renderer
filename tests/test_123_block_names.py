"""1.2.3 多角度场景一致 › 体块命名: every context block gets a meaningful name in the prompt.

Sources, first match wins: the user's own hint › object name › material name › a creator's
auto-stored label › analysis label / element type › category words › a size / height guess.
Rooms are described as 室内物体, not buildings."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

MS = load_package_module("massing")
CD = load_package_module("codex_direct")
PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"


class FakeBlock(dict):
    """ID-property semantics of a Blender object plus material slots."""

    def __init__(self, name, materials=(), **props):
        super().__init__(props)
        self.name = name
        self.material_slots = [SimpleNamespace(material=SimpleNamespace(name=m)) for m in materials]

    __hash__ = object.__hash__

    def __eq__(self, other):
        return self is other


class NameWordTests(unittest.TestCase):
    def test_analysis_builder_names(self):
        cases = {
            "furniture_sofa_back": "沙发靠背",
            "furniture_sofa_seat_and_base": "沙发坐垫底座",
            "furniture_coffee_table": "茶几",
            "furniture_media_console": "电视柜",
            "furniture_book_on_table": "桌子上的书",
            "furniture_lowest_wall_shelf": "墙面搁板",
            "wall_left_wall": "墙面",
            "wall_television_wood_panel": "电视木饰面墙板",
            "ceiling_rear_ceiling_soffit": "吊顶",
            "window_balcony_glazing": "阳台玻璃门窗",
            "curtain_white_sheer_curtains_1": "白色纱帘",
            "plant_floor_houseplant": "落地盆栽绿植",
            "furniture_decorative_clock": "钟",
        }
        for name, want in cases.items():
            self.assertEqual(MS.hint_from_name(name), want, name)

    def test_old_behaviour_kept(self):
        self.assertEqual(MS.hint_from_name("楼_玻璃幕墙"), "楼，玻璃幕墙")
        self.assertEqual(MS.hint_from_name("Tower_glass.001"), "高楼，玻璃幕墙")
        self.assertEqual(MS.hint_from_name("Cube.003"), "")
        self.assertEqual(MS.hint_from_name("block_tree_02"), "树")
        self.assertEqual(MS.hint_from_name("Brick_wall"), "砖墙")
        self.assertEqual(MS.hint_from_name("建筑体块"), "建筑")
        self.assertEqual(MS.hint_from_name("地形_heightfield"), "地面")
        self.assertEqual(MS.hint_from_name("Cube", custom="石库门弄堂口"), "石库门弄堂口")

    def test_category_only_name(self):
        self.assertEqual(MS.hint_from_name("Cube_furniture"), "")
        self.assertEqual(MS.derive_hint("Cube_furniture"), ("家具", "category"))


class DeriveTests(unittest.TestCase):
    SOFA_BOX = (-1.9, 0.4, 0.0, -0.8, 3.4, 0.9)

    def test_priority(self):
        self.assertEqual(MS.derive_hint("furniture_sofa_back", custom="芥末黄丝绒沙发"), ("芥末黄丝绒沙发", "custom"))
        # an auto-stored creator label ranks below the user's own object name
        self.assertEqual(MS.derive_hint("书架", custom="建筑", auto=True), ("书架", "name"))
        self.assertEqual(MS.derive_hint("Cube.004", custom="建筑", auto=True), ("建筑", "auto"))
        self.assertEqual(MS.derive_hint("Cube.001", materials=["Brick_Red"]), ("砖墙", "material"))
        self.assertEqual(MS.derive_hint("Cube.001", materials=["_wondful_massing_mat"], env_type="wall"), ("墙面", "label"))
        self.assertEqual(MS.derive_hint("Cube.002", label="media_console"), ("电视柜", "label"))
        self.assertEqual(MS.derive_hint("Cube.002", label="木质书柜"), ("木质书柜", "label"))
        h, src = MS.derive_hint("Cube.005", box=self.SOFA_BOX, interior=True, room_top=2.8)
        self.assertEqual(src, "shape")
        self.assertEqual(h, "沙发" + MS.SHAPE_NOTE)
        self.assertEqual(MS.derive_hint("Cube.005"), ("", ""))

    def test_group_hint_reads_objects(self):
        a = FakeBlock("Cube.001", materials=["_wondful_massing_mat"])
        b = FakeBlock("Cube.002", materials=["Oak_cabinet"])
        self.assertEqual(MS.group_hint([a, b]), ("柜子", "material"))
        c = FakeBlock("Cube.003", wondful_hint="电视柜", wondful_hint_auto=1)
        self.assertEqual(MS.group_hint([c]), ("电视柜", "auto"))
        c["wondful_hint_auto"] = 0
        self.assertEqual(MS.group_hint([c]), ("电视柜", "custom"))
        d = FakeBlock("wall_x", wondful_env_type="wall")
        self.assertEqual(MS.group_hint([d], box=(0, 0, 0, 0.2, 4, 2.8), interior=True), ("墙面", "label"))
        e = FakeBlock("Cube.009")
        h, src = MS.group_hint([e], box=(0, 0, 0, 0.2, 4, 2.8), interior=True, room_top=2.8)
        self.assertEqual((h, src), ("墙面" + MS.SHAPE_NOTE, "shape"))
        self.assertEqual(MS.group_hint([e], box=(0, 0, 0, 0.2, 4, 2.8), shape=False), ("", ""))

    def test_store_label(self):
        o = FakeBlock("建筑体块")
        MS.store_label(o, "建筑")
        self.assertEqual(o["wondful_hint"], "建筑")
        self.assertEqual(o[MS.AUTO_HINT_FLAG], 1)
        MS.store_label(o, "")
        self.assertEqual(o["wondful_hint"], "建筑")


class ShapeTests(unittest.TestCase):
    def test_interior(self):
        g = lambda b: MS.shape_hint(b, 0.0, True, 2.8)
        self.assertEqual(g((-2, -2, 0, 2, 3, 0.02)), "地面（地板 / 地毯）")
        self.assertEqual(g((-2, -2, 2.7, 2, 3, 2.85)), "天花板")
        self.assertEqual(g((-2.2, -1, 0, -2.0, 4, 2.8)), "墙面")
        self.assertEqual(g((0, 0, 0, 0.8, 1.5, 0.4)), "茶几 / 矮柜")
        self.assertEqual(g((0, 0, 0, 1.0, 3.0, 0.85)), "沙发")
        self.assertEqual(g((0, 0, 0, 0.45, 1.2, 1.9)), "柜子 / 书架")
        self.assertEqual(g((0, 0, 1.2, 0.3, 0.8, 1.24)), "墙面搁板")
        self.assertEqual(g((0, 0, 0.42, 0.1, 0.1, 0.54)), "桌面 / 柜面上的小摆件")
        self.assertEqual(g((1.6, 2, 1.2, 1.68, 3.3, 1.9)), "墙上的装饰（挂画 / 电视 / 墙板）")
        self.assertEqual(g((1.3, 0.4, 0.0, 1.8, 3.5, 0.6)), "矮柜 / 电视柜")
        self.assertEqual(g((-2, 4.2, 2.6, 2, 4.9, 2.8)), "吊顶 / 梁")

    def test_outdoor(self):
        g = lambda b: MS.shape_hint(b, 0.0, False)
        self.assertEqual(g((0, 0, 0, 12, 10, 9)), "建筑")
        self.assertEqual(g((0, 0, 0, 20, 20, 60)), "高楼")
        self.assertEqual(g((0, 0, 0, 40, 6, 0.05)), "道路 / 人行道")
        self.assertEqual(g((0, 0, 0, 0.8, 0.8, 6)), "树 / 路灯 / 柱子")
        self.assertEqual(g((0, 0, 0, 10, 0.4, 1.0)), "矮墙 / 围栏 / 绿篱")

    def test_is_interior(self):
        chair = (-0.4, -0.3, 0.0, 0.4, 0.3, 0.9)
        room = [(-2.2, -1, 0, -2.0, 5, 2.8), (1.8, -1, 0, 2.0, 5, 2.8)]
        self.assertTrue(MS.is_interior(room, chair))
        self.assertTrue(MS.is_interior([(-2, -1, 2.8, 2, 5, 2.95)], chair))
        street = [(5, 5, 0, 15, 15, 12), (-15, 5, 0, -5, 15, 9)]
        self.assertFalse(MS.is_interior(street, chair))
        self.assertFalse(MS.is_interior(room, None))


class PromptTests(unittest.TestCase):
    ROOM = [{"hint": "沙发靠背", "place": "画面左侧，背景（产品后方）", "interior": True},
            {"hint": "墙面" + MS.SHAPE_NOTE, "place": "画面右侧，背景", "interior": True}]
    STREET = [{"hint": "玻璃幕墙", "place": "画面左侧，背景（产品后方）"}, {"hint": "", "place": "画面右侧，前景"}]

    def test_room_paragraph(self):
        t = MS.build_massing_text(self.ROOM)
        self.assertIn(MS.INTERIOR_MARK, t)
        self.assertIn("写的是墙就画成墙面", t)
        self.assertIn("不要把体块画成建筑", t)
        self.assertNotIn("建筑 / 结构的体量占位", t)
        self.assertIn("体块 1「沙发靠背」：画面左侧", t)
        self.assertIn("按尺寸推测", t)
        self.assertIn("产品继续严格锁定", t)

    def test_street_paragraph_unchanged(self):
        t = MS.build_massing_text(self.STREET)
        self.assertNotIn(MS.INTERIOR_MARK, t)
        self.assertIn("建筑 / 结构的体量占位", t)
        self.assertIn("体块 1「玻璃幕墙」", t)
        self.assertIn("体块 2：画面右侧", t)
        self.assertIn(MS.INTERIOR_MARK, MS.build_massing_text(self.STREET, interior=True))

    def test_simple_prompt_room_wording(self):
        room = CD.build_simple_prompt("北欧客厅", env_text="白天的客厅", massing_text=MS.build_massing_text(self.ROOM))
        self.assertIn("变成各自名称对应的室内物体", room)
        self.assertIn("灰蓝色体块对应的各自名称对应的室内物体", room)
        self.assertIn("不额外添加其他家具、建筑", room)
        street = CD.build_simple_prompt("街道", env_text="晴天", massing_text=MS.build_massing_text(self.STREET))
        self.assertIn("灰蓝色体块按上面的要求变成建筑 / 结构", street)
        self.assertNotIn("室内物体", street)


class WiringTests(unittest.TestCase):
    def test_wiring(self):
        init = (PKG / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("bpy.types.Object.wondful_hint = bpy.props.StringProperty(", init)
        self.assertIn("del bpy.types.Object.wondful_hint", init)
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        self.assertIn("_draw_mv_block_name(box, context, props, busy)", ui)
        self.assertIn('row.prop(root, "wondful_hint", text="体块名称")', ui)
        ms = (PKG / "massing.py").read_text(encoding="utf-8")
        self.assertIn("group_hint([root] + [m for m in members if m is not root], gbox, floor_z, interior, room_top)", ms)
        self.assertIn('"interior": interior', ms)
        self.assertIn('MS.store_label(obj, "建筑")', (PKG / "massing_hull.py").read_text(encoding="utf-8"))
        self.assertIn("group_hint([obj], shape=False)", (PKG / "multiview.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
