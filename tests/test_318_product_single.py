"""3.1.8: single product-shape reference, three-view hint (pure Python)."""
import ast
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"


class ProductSingleTests(unittest.TestCase):
    def test_limit_constant(self):
        src = (PKG / "properties.py").read_text()
        self.assertIn("MAX_PRODUCT_REFERENCES = 1", src)

    def test_prompt_rule_mentions_three_view(self):
        import importlib.util, sys, types
        pkg = types.ModuleType("w318"); pkg.__path__ = [str(PKG)]; sys.modules["w318"] = pkg
        for name in ("composition", "prompt_engine"):
            spec = importlib.util.spec_from_file_location(f"w318.{name}", PKG / f"{name}.py")
            mod = importlib.util.module_from_spec(spec); sys.modules[f"w318.{name}"] = mod
            spec.loader.exec_module(mod)
        rule = mod.product_reference_rule("BLENDER")
        self.assertIn("三视图", rule)
        self.assertIn("不要把拼版", mod.product_reference_rule("PRODUCT_REF"))

    def test_parse_product_analysis(self):
        import sys
        self.test_prompt_rule_mentions_three_view()
        pe = sys.modules["w318.prompt_engine"]
        look, det = pe.parse_product_analysis('好的：```json\n{"appearance": "厚清漆", "details": ["车标", "车标", "前车牌"]}\n```')
        self.assertEqual(look, "厚清漆")
        self.assertEqual(det, ["车标", "前车牌"])
        look, det = pe.parse_product_analysis("产品外观：拉丝铝\n保留细节：表冠，刻度，表耳")
        self.assertEqual((look, det), ("拉丝铝", ["表冠", "刻度", "表耳"]))
        self.assertEqual(pe.parse_product_analysis("无法识别"), ("", []))
        self.assertIn("不写任何颜色", pe.product_analysis_user_text("BLENDER"))
        self.assertIn("固有色", pe.product_analysis_user_text("PRODUCT_REF"))

    def test_ui_hint_and_label(self):
        src = (PKG / "ui.py").read_text()
        components = (PKG / "ui_components.py").read_text()
        # The redesigned card keeps the guidance in its help affordance rather
        # than permanently occupying the sidebar.
        self.assertIn('UIC.help_icon(tools, key="PRODUCT")', src)
        tree = ast.parse(components)
        help_value = ast.literal_eval(next(node.value for node in tree.body
                                           if isinstance(node, ast.Assign)
                                           and any(isinstance(target, ast.Name) and target.id == "HELP"
                                                   for target in node.targets)))
        self.assertIn("正、侧、后三视图拼图", help_value["PRODUCT"])
        self.assertIn("fit='contain' if kind == 'PRODUCT'", components)
        ast.parse(src)


if __name__ == "__main__":
    unittest.main()
