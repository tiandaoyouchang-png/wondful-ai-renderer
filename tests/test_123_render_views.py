"""1.2.3 ⑥ 渲染视角: the shared 「相机 N」 list as a multi-select of views to render (drag-in, order, checked filter)."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

MV = load_package_module("multiview")
PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"


def cam(name, kind="CAMERA"):
    return SimpleNamespace(name=name, type=kind)


def coll(objects=(), children=()):
    return SimpleNamespace(objects=list(objects), children=list(children))


class CheckedViews(unittest.TestCase):
    def test_checked_in_list_order(self):
        a, b, c = cam("A"), cam("B"), cam("C")
        self.assertEqual([o.name for o in MV.checked_views([(c, True), (a, False), (b, True)])], ["C", "B"])

    def test_drops_missing_non_camera_duplicates(self):
        a, gone, cube = cam("A"), cam("Gone"), cam("Cube", "MESH")
        slots = [(None, True), (cube, True), (gone, True), (a, True), (a, True)]
        out = MV.checked_views(slots, is_present=lambda o: o.name != "Gone")
        self.assertEqual([o.name for o in out], ["A"])

    def test_unchecked_duplicate_does_not_block(self):
        a = cam("A")
        self.assertEqual([o.name for o in MV.checked_views([(a, False), (a, True)])], ["A"])

    def test_nothing_checked(self):
        self.assertEqual(MV.checked_views([(cam("A"), False)]), [])
        self.assertEqual(MV.checked_views([]), [])


class Append(unittest.TestCase):
    def test_dedupe_against_list_and_within_drop(self):
        a, b, c = cam("A"), cam("B"), cam("C")
        out = MV.cameras_to_append(["A"], [a, b, b, c, cam("Cube", "MESH"), None])
        self.assertEqual([o.name for o in out], ["B", "C"])

    def test_empty_names_ignored(self):
        self.assertEqual([o.name for o in MV.cameras_to_append(["", None], [cam("A")])], ["A"])

    def test_drop_note(self):
        self.assertEqual(MV.drop_note(["A"], []), "已加入 A")
        self.assertEqual(MV.drop_note(["A", "B"], ["C"]), "已加入 2 个相机 · C 已在列表")
        self.assertEqual(MV.drop_note([], ["A", "B"]), "跳过已在列表的 2 个")
        self.assertEqual(MV.drop_note([], [], "空集合"), "集合「空集合」里没有相机")
        self.assertEqual(MV.drop_note([], []), "")


class CollectionRecursion(unittest.TestCase):
    def test_recursive_depth_first_once(self):
        a, b, c, d = cam("A"), cam("B"), cam("C"), cam("D")
        inner = coll([c, cam("Light", "LIGHT")])
        deep = coll([d, a])
        mid = coll([b], [inner, coll([], [deep])])
        root = coll([a, cam("Cube", "MESH")], [mid, inner])        # inner reached twice
        self.assertEqual([o.name for o in MV.cameras_in_collection(root)], ["A", "B", "C", "D"])

    def test_cycle_safe_and_none(self):
        root = coll([cam("A")])
        root.children.append(root)
        self.assertEqual([o.name for o in MV.cameras_in_collection(root)], ["A"])
        self.assertEqual(MV.cameras_in_collection(None), [])


class Order(unittest.TestCase):
    def test_move_index(self):
        self.assertEqual(MV.move_index(3, 0, -1), None)
        self.assertEqual(MV.move_index(3, 0, 1), 1)
        self.assertEqual(MV.move_index(3, 2, 1), None)
        self.assertEqual(MV.move_index(3, 2, -1), 1)
        self.assertEqual(MV.move_index(0, 0, 1), None)
        self.assertEqual(MV.move_index(3, 5, -1), None)

    def test_view_phase(self):
        self.assertEqual(MV.view_phase(1, 3, "前", True), "视角 1/3 · 前（主视角）")
        self.assertEqual(MV.view_phase(2, 3, "侧"), "视角 2/3 · 侧")


class Wiring(unittest.TestCase):
    def test_one_list_shared(self):
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        self.assertEqual(props.count("CollectionProperty(type=WONDFUL_MVCameraItem"), 1)   # one source of truth
        for name in ("mv_drop_camera", "mv_drop_collection", "update=_mv_drop_camera_update",
                     "update=_mv_drop_collection_update", "views_plan", "view_results"):
            self.assertIn(name, props)
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        self.assertIn('"WONDFUL_UL_render_views", "", props, "mv_cameras"', ui)
        # The list is shown only after the user explicitly enables multiview in
        # reference workflow; environment renders never inherit it.
        self.assertIn('if props.workflow_multiview:\n                _draw_render_views(output,context,props,busy,cells)', ui)
        init = (PKG / "__init__.py").read_text(encoding="utf-8")
        for name in ("WONDFUL_UL_render_views", "WONDFUL_OT_mv_camera_move", "WONDFUL_OT_view_result_step"):
            self.assertEqual(init.count(name), 2, name)

    def test_generate_plans_and_chains(self):
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        gen = ops[ops.index("class WONDFUL_OT_generate"):ops.index("def generate_views")]
        self.assertIn("context.scene.camera = views[0]", gen)
        self.assertIn("MV.scene_blockout_exists(scene)", ops)
        render = ops[ops.index("def _after_main_view"):ops.index("def _modal_render")]
        self.assertIn('_schedule_operator("wondful.multiview_render"', render)
        self.assertIn('"from_plan": True', render)
        mv = ops[ops.index("class WONDFUL_OT_multiview_render"):]
        self.assertIn("if plan:      # ⑥ 渲染视角: the user's list order", mv)
        self.assertIn("视角 {i + 1}/{plan_total}", mv)


if __name__ == "__main__":
    unittest.main()
