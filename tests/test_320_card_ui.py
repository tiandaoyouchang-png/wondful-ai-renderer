"""3.2.0: five step cards, large reference thumbnails, staged generate progress (source-level checks)."""
import ast
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"


class CardUITests(unittest.TestCase):
    def setUp(self):
        self.src = (PKG / "ui.py").read_text()
        ast.parse(self.src)

    def test_five_steps_in_order(self):
        order = ['"连接"', '"产品三视图（可选）"', '"环境参考图"', '"提示词"', '"生成"']
        draw = self.src[self.src.index("class WONDFUL_PT_main"):]
        positions = [draw.index(f"_step_card(layout, {i + 1}, {name}") for i, name in enumerate(order)]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('STEP_NUMBERS = ("①", "②", "③", "④", "⑤")', self.src)
        self.assertIn('icon="CHECKMARK"', self.src)

    def test_cards_grey_but_never_disable(self):
        helper = self.src[self.src.index("def _step_card"):self.src.index("def _thumb_scale")]
        self.assertIn("card.active", helper)
        self.assertNotIn("card.enabled", helper)

    def test_large_thumbnails_and_compact_actions(self):
        self.assertIn("template_icon(icon_value=bpy.types.UILayout.icon(image), scale=scale)", self.src)
        for icon in ("PASTEDOWN", "FILEBROWSER", "ADD", "TRASH", "FILE_REFRESH", "X"):
            self.assertIn(f'icon="{icon}"', self.src)

    def test_generate_button_and_progress(self):
        self.assertIn("go.scale_y = 2.0", self.src)
        self.assertIn("GENERATE_STAGES", self.src)
        self.assertIn("wondful.cancel_task", self.src)

    def test_result_actions_only_three(self):
        draw = self.src[self.src.index("# ---- 结果"):self.src.index("# ---- 高级")]
        ops = [line.split('"')[1] for line in draw.splitlines() if ".operator(" in line]
        self.assertEqual(sorted(ops), sorted(["wondful.open_compare_workspace", "wondful.ai_render", "wondful.open_output_directory"]))

    def test_version(self):
        init = (PKG / "__init__.py").read_text()
        # Bumped with each release (1.1.0 candidate: custom terrain).
        self.assertIn('"version": (1, 1, 0)', init)
        self.assertIn('bl_label = "Wondful AI 渲染器 · 1.1"', self.src)
        self.assertNotIn("3.2.2", self.src)
        cd = (PKG / "codex_direct.py").read_text()
        self.assertIn('ADDON_UA = "wondful-ai-renderer/1.1.0"', cd)


if __name__ == "__main__":
    unittest.main()
