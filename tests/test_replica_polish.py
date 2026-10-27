"""Focused offline contracts for mandatory replica prompt polish."""
import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "wondful_ai_renderer"


def load_prompt_engine():
    package = types.ModuleType("_replica_polish")
    package.__path__ = [str(PKG)]
    sys.modules[package.__name__] = package
    sys.modules[package.__name__ + ".composition"] = types.SimpleNamespace(
        canvas_contract=lambda *args, **kwargs: "CANVAS"
    )
    spec = importlib.util.spec_from_file_location(package.__name__ + ".prompt_engine", PKG / "prompt_engine.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_reference_bundle():
    package = sys.modules["_replica_polish"]
    spec = importlib.util.spec_from_file_location(package.__name__ + ".reference_bundle", PKG / "reference_bundle.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PE = load_prompt_engine()
RB = load_reference_bundle()
OPERATORS = (PKG / "operators.py").read_text(encoding="utf-8")


class ReplicaPromptTests(unittest.TestCase):
    def test_reference_polish_numbers_target_then_fresh_white_preview(self):
        text = PE.build_polish_user_text(
            "保留我的产品身份", 1, 0, 0, 1600, 900,
            workflow_mode="REFERENCE", replica_target_count=1, replica_preview_count=1,
        )
        self.assertIn("Reference 1 = 复刻原始目标图", text)
        self.assertIn("Reference 2 = 当前白模粗模预览", text)
        self.assertNotIn("Reference 1 = Blender Camera Base", text)
        self.assertIn("不得臆造未出现的特征", text)

    def test_reference_render_keeps_observed_visual_intent(self):
        prompt = PE.build_render_prompt(
            "低机位倾斜跟拍，运动模糊和尘土接地。", 1600, 900, 1, 0, 1,
            workflow_mode="REFERENCE",
        )
        self.assertIn("低机位倾斜跟拍", prompt)
        self.assertIn("原始复刻目标图", prompt)

    def test_overflow_bundle_keeps_scene_reference_member(self):
        plan = RB.bundle_plan([
            ("camera", ["camera.png"]),
            ("structure", ["mask.png"]),
            ("product", ["product.png"]),
            ("scene_reference", ["target.png"]),
        ], limit=3)
        members = [member for group in plan for member in group["members"]]
        self.assertIn("scene_reference", [member["role"] for member in members])
        self.assertIn("target.png", [member["path"] for member in members])

    def test_environment_render_still_strips_generic_structure_instruction(self):
        prompt = PE.build_render_prompt("严格保持构图，暖色日落。", 1600, 900, 0, 0, 0)
        self.assertNotIn("严格保持构图", prompt)
        self.assertIn("暖色日落", prompt)

    def test_replica_preserves_high_angle_static_long_lens_without_invented_features(self):
        observed = "保持原图的高机位俯拍观感，静止主体，固定长焦焦段的压缩感，冷白色漫射光。"
        self.assertEqual(PE.sanitize_appearance_prompt(observed, preserve_visual_intent=True), observed)
        rendered = PE.build_render_prompt(observed, 1600, 900, 1, 0, 1, workflow_mode="REFERENCE")
        self.assertIn(observed, rendered)
        for invented in ("运动模糊", "跟拍", "峡谷", "日落"):
            self.assertNotIn(invented, rendered)
        environment = PE.sanitize_appearance_prompt(observed)
        self.assertNotIn("高机位", environment)
        self.assertNotIn("长焦焦段", environment)
        self.assertIn("冷白色漫射光", environment)

    def test_replica_photography_exception_cannot_keep_geometry_overrides(self):
        for instruction in ("低机位跟拍 x=0.2 y=0.4", "高机位 bbox=[0,0,1,1]",
                            "将相机移动到高机位", "调整机位以获得长焦效果", "重新构图以保持跟拍感",
                            "reframe the high-angle static shot", "固定产品坐标以保持长焦观感"):
            with self.subTest(instruction=instruction):
                clean, removed = PE.sanitize_appearance_prompt_report(
                    instruction + "。冷白色漫射光。", preserve_visual_intent=True)
                self.assertEqual(clean, "冷白色漫射光。")
                self.assertTrue(removed)


class OperatorWiringTests(unittest.TestCase):
    def test_generate_is_one_polish_then_render_chain(self):
        body = OPERATORS[OPERATORS.index("def generate_needs_polish"):OPERATORS.index("def identity_preflight")]
        self.assertIn("return True", body)
        self.assertNotIn("auto_polish_on_generate", body)
        polish = OPERATORS[OPERATORS.index("class WONDFUL_OT_polish_prompt"):OPERATORS.index("class WONDFUL_OT_ai_render")]
        self.assertIn('if chain_render:\n                    props.task_phase', polish)
        self.assertNotIn('chain_render and not getattr(props, "two_step", False)', polish)

    def test_reference_pair_is_fresh_and_has_distinct_final_role(self):
        polish = OPERATORS[OPERATORS.index("class WONDFUL_OT_polish_prompt"):OPERATORS.index("class WONDFUL_OT_ai_render")]
        self.assertIn('_render_replica_preview(context, props, self._session.directory / "polish_preview")', polish)
        self.assertIn("replica_target, replica_preview", polish)
        signatures = [node for node in ast.walk(ast.parse(polish)) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == "_camera_signature"]
        self.assertTrue(signatures)
        self.assertTrue(all([ast.unparse(arg) for arg in node.args] == ["context.scene.camera", "context.scene"]
                            for node in signatures))
        render = OPERATORS[OPERATORS.index("class WONDFUL_OT_ai_render"):]
        self.assertIn('copy_references([source_reference], self._session, "scene_reference")', render)
        self.assertIn('("scene_reference" if workflow.is_reference(props) else "environment", style_upload)', render)

    def test_reference_polish_bypasses_old_style_summary_cache(self):
        polish = OPERATORS[OPERATORS.index("class WONDFUL_OT_polish_prompt"):OPERATORS.index("class WONDFUL_OT_ai_render")]
        self.assertIn("A white preview is a new geometry/camera observation every time", polish)
        self.assertIn("self._cached_style_summary = \"\"", polish)
        self.assertIn("self._style_cache_hit = False", polish)

    def test_reference_blockout_forces_and_syncs_camera_only_at_finish(self):
        blockout = OPERATORS[OPERATORS.index("class WONDFUL_OT_scene_blockout_build"):OPERATORS.index("class WONDFUL_OT_polish_prompt")]
        self.assertIn("set_camera = True", blockout)
        self.assertIn("from .camera_sync import sync_generated_camera", blockout)
        self.assertIn("sync_generated_camera(context, camera, scene=self._origin_scene)", blockout)


if __name__ == "__main__":
    unittest.main()
