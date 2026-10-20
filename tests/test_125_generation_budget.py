"""1.2.5: 「最多生成次数」 is the hard cap on image calls per image.

Bug (2026-10-20 细节粗模 V3 real-gen run): with 轮廓验收 on, the IoU gate's own setting (default 3) replaced the
preference entirely, and 张数 defaulted to 4, so a run configured with 最多生成次数 = 1 still spent ~19 calls."""
import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

RW = load_package_module("result_worker")
MV = load_package_module("multiview")
PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"


class GenerationBudget(unittest.TestCase):
    def test_pref_caps_gate(self):
        self.assertEqual(RW.generation_budget(1, gate_on=True, gate_max=3), 1)
        self.assertEqual(RW.generation_budget(2, gate_on=True, gate_max=3), 2)
        self.assertEqual(RW.generation_budget(3, gate_on=True, gate_max=3), 3)
        self.assertEqual(RW.generation_budget(3, gate_on=True, gate_max=1), 1)

    def test_default_behaviour_kept(self):
        # pref default is now 3 → the gate's default 3 is unchanged
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        i = props.index("alignment_max_attempts: IntProperty(")
        self.assertIn("default=3", props[i:i + 600])
        self.assertEqual(RW.generation_budget(3, gate_on=True, gate_max=3), 3)

    def test_no_gate(self):
        self.assertEqual(RW.generation_budget(3, gate_on=False, strict_retry=False), 1)
        self.assertEqual(RW.generation_budget(2, gate_on=False, strict_retry=True), 2)
        self.assertEqual(RW.generation_budget(1, gate_on=False, strict_retry=True), 1)

    def test_bad_values(self):
        self.assertEqual(RW.generation_budget(None, gate_on=True, gate_max=None), RW.GATE_MAX)
        self.assertEqual(RW.generation_budget(9, gate_on=True, gate_max=9), RW.GATE_MAX)
        self.assertEqual(RW.generation_budget(0, gate_on=True, gate_max=3), 1)

    def test_low_tier_retry_stays_inside(self):
        n = RW.generation_budget(1, gate_on=True, gate_max=3)
        self.assertFalse(RW.low_tier_retry("low", 1, n, False))

    def test_crop_regen(self):
        self.assertFalse(RW.crop_regen_allowed(1))
        self.assertTrue(RW.crop_regen_allowed(2))
        self.assertTrue(RW.crop_regen_allowed("x"))

    def test_camera_calls(self):
        self.assertEqual(MV.camera_calls(1), 1)
        self.assertEqual(MV.camera_calls(3), MV.MAX_CALLS)
        self.assertEqual(MV.camera_calls(None), MV.MAX_CALLS)

    def test_budget_note(self):
        self.assertIn("1 张 × 每张最多 1 次 = 最多 1 次", RW.call_budget_note(1, 1, crop_regen=False))
        self.assertIn("4 张 × 每张最多 3 次 = 最多 12 次", RW.call_budget_note(4, 3))


class OperatorWiring(unittest.TestCase):
    src = (PKG / "operators.py").read_text(encoding="utf-8")

    def test_gate_no_longer_replaces_pref(self):
        self.assertNotIn("planned_attempts = gate_max", self.src)
        self.assertIn("result_worker.generation_budget(", self.src)
        ast.parse(self.src)

    def test_multiview_loop_uses_cap(self):
        self.assertIn("for n in range(1, mv_calls + 1):", self.src)
        self.assertNotIn("for n in range(1, MV.MAX_CALLS + 1):", self.src)
        self.assertIn("_camera_calls(getattr(prefs, \"alignment_max_attempts\"", self.src)

    def test_crop_regen_gated(self):
        self.assertIn("crop_regen_allowed(getattr(prefs, \"alignment_max_attempts\", 3))", self.src)
        self.assertIn("if paste and paste_regen:", self.src)

    def test_pref_always_visible(self):
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        self.assertIn('            col.prop(self, "alignment_threshold")\n        col.prop(self, "alignment_max_attempts")', props)

    def test_harness_sets_one_variant(self):
        h = (PKG.parent / "tools" / "e2e_detail_mv_bg.py").read_text(encoding="utf-8")
        self.assertIn('props.render_count = "1"', h)
        self.assertIn("props.iou_gate_max = 1", h)


if __name__ == "__main__":
    unittest.main()
