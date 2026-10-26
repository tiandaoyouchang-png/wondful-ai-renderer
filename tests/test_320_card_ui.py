"""3.2.0 step cards (1.2.3: six, ④ 场景粗模 optional), large reference thumbnails, staged generate progress (source-level checks)."""
import ast
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"


class CardUITests(unittest.TestCase):
    def setUp(self):
        self.src = (PKG / "ui.py").read_text()
        ast.parse(self.src)

    def test_current_cards_follow_workflow_order(self):
        draw = self.src[self.src.index("class WONDFUL_PT_main"):]
        positions = [draw.index('_step_card(layout, 1, "车模与相机"'),
                     draw.index('_step_card(layout, 2, "目标图与粗模" if replica else "环境参考"'),
                     draw.index('_step_card(layout, 3, "预览与调整"', draw.index("if replica:")),
                     draw.index('_step_card(layout,4,"渲染复刻图" if replica else "生成效果图"')]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('action = modes.operator("wondful.set_workflow_mode"', draw)
        self.assertIn('depress=(props.workflow_mode == value)', draw)

    def test_cards_remain_drawn_while_actions_are_gated(self):
        helper = self.src[self.src.index("def _step_card"):self.src.index("def _thumb_scale")]
        self.assertNotIn("card.enabled", helper)
        draw = self.src[self.src.index("def _draw_main"):]
        self.assertIn("controls.enabled = not busy", draw)
        self.assertIn("go.enabled=can_generate", draw)

    def test_large_thumbnails_and_compact_actions(self):
        components = (PKG / "ui_components.py").read_text()
        self.assertIn("row.template_icon(icon_value=icon, scale=scale)", components)
        self.assertIn("preview_tile_count(image, fit)", components)
        for icon in ("PASTEDOWN", "FILEBROWSER", "X"):
            self.assertIn(f"icon='{icon}'", components)

    def test_generate_button_and_progress(self):
        self.assertIn("go.scale_y=UIC.ACTION_HEIGHT", self.src)
        self.assertIn("_draw_task_progress(step5,props,cells)", self.src)
        self.assertIn('go.operator("wondful.generate"', self.src)
        self.assertIn("wondful.cancel_task", self.src)

    def test_result_actions_only_three(self):
        draw = self.src[self.src.index("# ---- 结果"):self.src.index("# ---- 高级")]
        ops = [line.split('"')[1] for line in draw.splitlines() if ".operator(" in line]
        self.assertEqual(sorted(ops), sorted(["wondful.open_compare_workspace", "wondful.generate", "wondful.open_output_directory"]))

    def test_version(self):
        init = (PKG / "__init__.py").read_text()
        # Bumped with each release (1.2.4: 一键更新 + 渲染视角).
        self.assertIn('"version": (1, 2, 4)', init)
        self.assertIn('bl_label = "Wondful AI 渲染器"', self.src)
        self.assertNotIn("3.2.2", self.src)
        cd = (PKG / "codex_direct.py").read_text()
        self.assertIn('ADDON_UA = "wondful-ai-renderer/1.2.4"', cd)


if __name__ == "__main__":
    unittest.main()
