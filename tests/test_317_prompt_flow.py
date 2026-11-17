"""Regression tests for Wondful 3.1.7 prompt flow (pure Python, no Blender)."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / "wondful_ai_renderer"


def load_package_module(name):
    pkg_name = "wondful_ai_renderer_317test"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(PKG_DIR)]
        sys.modules[pkg_name] = pkg
    full = f"{pkg_name}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, PKG_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


lock = load_package_module("appearance_lock")
pe = load_package_module("prompt_engine")


class SanitizerTests(unittest.TestCase):
    def test_keeps_appearance_sentences_with_soft_terms(self):
        for text in (
            "高光位置与环境主光一致。",
            "地平线处的天空保持暖橙色。",
            "车标位置的镀铬质感必须清晰。",
            "反射中能看到窗户，透视自然，阴影方向一致。",
        ):
            clean, removed = pe.sanitize_appearance_prompt_report(text)
            self.assertEqual(clean, text)
            self.assertEqual(removed, [])

    def test_removes_structure_clauses_only(self):
        clean, removed = pe.sanitize_appearance_prompt_report("保持 Reference 1 位置不变，车漆呈亮翡翠绿。")
        self.assertEqual(clean, "车漆呈亮翡翠绿。")
        self.assertEqual(removed, ["保持 Reference 1 位置不变"])
        clean, removed = pe.sanitize_appearance_prompt_report("透视必须与底图一致，木地板纹理细腻。")
        self.assertEqual(clean, "木地板纹理细腻。")

    def test_removes_hard_structure_and_coordinates(self):
        self.assertEqual(pe.sanitize_appearance_prompt("高端广告大片。严格保持构图，不要重新构图。"), "高端广告大片。")
        self.assertEqual(pe.sanitize_appearance_prompt("主体 x=0.5 y=0.6。暖色斜阳。"), "暖色斜阳。")
        self.assertEqual(pe.sanitize_appearance_prompt("产品位置保持不变。"), "")


class AppearanceLockTests(unittest.TestCase):
    def test_color_name(self):
        self.assertTrue(lock.color_name((0.8, 0.12, 0.02)).startswith(("橙", "红", "亮")))
        self.assertTrue(lock.color_name((0.0, 0.0, 0.0)).startswith("黑色"))
        self.assertIn("#", lock.color_name((0.5, 0.5, 0.5)))

    def test_material_summary_orders_by_weight(self):
        text = lock.material_summary([
            {"name": "Legs", "base_color": (0.02, 0.01, 0.005), "roughness": 0.6, "weight": 10},
            {"name": "Fabric", "base_color": (0.6, 0.1, 0.02), "roughness": 0.9, "sheen": 1.0, "weight": 100},
        ])
        self.assertTrue(text.startswith("Fabric："))
        self.assertIn("织物绒面", text)
        self.assertIn("哑光", text)

    def test_identity_checklist_merges_and_dedupes(self):
        items = lock.identity_checklist(["前车牌、车标"], "车标、日行灯", "车标，轮毂")
        self.assertEqual(items, ["车标", "轮毂", "前车牌", "日行灯"])

    def test_lock_block(self):
        block = lock.appearance_lock_block("BLENDER", "Paint：亮绿色", "厚清漆、长柔高光")
        self.assertIn("产品外观锁定", block)
        self.assertIn("厚清漆", block)
        self.assertIn("Paint", block)
        self.assertEqual(lock.appearance_lock_block("TEXT", "Paint", ""), "")


class PromptBuildTests(unittest.TestCase):
    def setUp(self):
        sys.modules["wondful_ai_renderer_317test.composition"] = types.SimpleNamespace(
            canvas_contract=lambda *a, **k: "CANVAS")

    def test_render_prompt_keeps_lock_and_identity_unsanitized(self):
        block = lock.appearance_lock_block("BLENDER", "Paint：亮绿色", "所有场景保持一致的厚清漆")
        prompt = pe.build_render_prompt(
            "雪地清晨，冷色天光。保持构图不变。", 1280, 720, 1, 0, 1,
            color_source="BLENDER", appearance_lock=block, identity_items=["前车牌", "车标"],
        )
        self.assertIn("所有场景保持一致的厚清漆", prompt)
        self.assertIn("前车牌、车标", prompt)
        self.assertNotIn("保持构图不变", prompt)
        self.assertIn("Blender 当前材质提供车身车漆色相", prompt)

    def test_product_ref_color_source_changes_rule(self):
        text = pe.build_polish_user_text("北欧客厅", 1, 0, 1, 1280, 720, color_source="PRODUCT_REF")
        self.assertIn("兼容旧 PRODUCT_REF", text)
        self.assertIn("绝不提取或覆盖车身车漆颜色", text)
        text = pe.build_polish_user_text("北欧客厅", 1, 0, 1, 1280, 720)
        self.assertNotIn("兼容旧 PRODUCT_REF", text)

    def test_polish_text_mentions_lock(self):
        text = pe.build_polish_user_text("茶室", 0, 0, 1, 1280, 720,
                                         appearance_lock="【产品外观锁定】\nX", identity_items=["Logo"])
        self.assertIn("不要改写或重复它", text)
        self.assertIn("Logo", text)

    def test_policy_version_bumped(self):
        self.assertGreaterEqual(pe.REFERENCE_POLICY_VERSION, 2)


if __name__ == "__main__":
    unittest.main()
