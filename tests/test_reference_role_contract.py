"""Offline regression checks for the three-role source/reference contract."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module

AL = load_package_module("appearance_lock")
PE = load_package_module("prompt_engine")
PP = load_package_module("prompt_profiles")
SR = load_package_module("source_reference")
CD = load_package_module("codex_direct")
CC = load_package_module("codex_client")
AGY = load_package_module("antigravity_client")
PNG = load_package_module("png_io")


class ReferenceRoleContractTests(unittest.TestCase):
    def test_model_wins_geometry_lamps_wheels_and_paint_over_identity(self):
        text = SR.PROJECTED_IDENTITY_POLICY + "\n" + CD.HYBRID_INSTRUCTIONS
        for required in ("当前 Blender 模型", "灯组", "轮辐", "车漆色相", "身份图只"):
            self.assertIn(required, text)

    def test_legacy_product_ref_cannot_repaint_conflicting_identity(self):
        lock = AL.appearance_lock_block("PRODUCT_REF", "", "")
        rule = PE.product_reference_rule("PRODUCT_REF")
        self.assertIn("不取参考图颜色", lock)
        self.assertIn("绝不提取或覆盖车身车漆颜色", rule)
        self.assertIn("旧 PRODUCT_REF", PE.reference_lighting_policy("PRODUCT_REF"))

    def test_only_explicit_concrete_user_paint_request_can_override_model_paint(self):
        text = SR.PROJECTED_IDENTITY_POLICY + "\n" + PE.reference_lighting_policy("TEXT")
        self.assertIn("用户明确指定具体车漆颜色", text)
        self.assertNotIn("身份图决定车漆", text)

    def test_style_analysis_is_whole_scene_only_and_final_merge_is_explicit(self):
        text = PE.SOURCE_STYLE_ANALYSIS_SYSTEM_PROMPT
        for required in ("整张图", "不得复制来源图中旧产品", "Blender 模型决定目标产品几何和车漆色相", "三视图／身份图只决定各部件材质"):
            self.assertIn(required, text)

    def test_policy_and_profile_versions_invalidate_obsolete_ai_summaries(self):
        self.assertGreaterEqual(PE.REFERENCE_POLICY_VERSION, 4)
        self.assertIn("reference-role-contract", PP.PACK_VERSION)
        self.assertEqual(PP.REFERENCE_ROLE_POLICY_VERSION, 8)
        with patch.object(PP, "REFERENCE_ROLE_POLICY_VERSION", 4):
            obsolete_key = PP.style_cache_key("same-bytes", ["same-note"], "codex", "model")
        self.assertNotEqual(obsolete_key, PP.style_cache_key("same-bytes", ["same-note"], "codex", "model"))
        self.assertFalse(PE.needs_reference_policy_refresh(type("P", (), {
            "prompt_reference_policy_version": PE.REFERENCE_POLICY_VERSION,
            "last_ai_prompt": "", "style_sync_initialized": False,
            "product_images": (), "style_images": (),
        })()))

    def test_assembled_camera_direct_cli_and_agy_prompts_keep_identity_material_only(self):
        images = [("camera.png", "camera", "Blender current model"),
                  ("identity.png", "product", "three-view material reference")]
        direct = CD.build_edit_prompt("brief", images, camera_capture=True)
        simple = CD.build_simple_prompt("brief", camera_capture=True)
        hybrid = CD.build_hybrid_prompt("brief", [("source.png", "source_canvas", "source"),
                                                     ("projection.png", "product_projection", "model"),
                                                     ("identity.png", "product", "identity")])
        for text in (direct, simple, hybrid):
            self.assertIn("Blender", text)
            self.assertNotIn("身份参考控制可见细部拓扑", text)
        self.assertIn("身份图只核对各部件材质", hybrid)

        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            root = Path(directory)
            source, projection, identity = (root / "source.png", root / "projection.png", root / "identity.png")
            for path, value in ((source, 20), (projection, 90), (identity, 210)):
                PNG.write_png(path, np.full((16, 24, 3), value, np.uint8))
            output = root / "out.png"
            prompt = SR.edit_prompt("scene atmosphere", projection=True)
            captured = []

            def cli_turn(**kwargs):
                captured.append(kwargs["prompt"])
                output.write_bytes(source.read_bytes())
                return "WONDFUL_IMAGEGEN_OK"

            with patch.object(CC, "_exec_turn", side_effect=cli_turn):
                CC.generate_image(prompt=prompt, reference_paths=[str(source), str(projection), str(identity)],
                                  product_projection_path=str(projection), edit_base_path=str(source),
                                  output_path=str(output), cwd=str(root), source_only=True)

            def agy_run(_cli, args, **_kwargs):
                captured.append(args[args.index("--prompt") + 1])
                output.write_bytes(source.read_bytes())
                event = {"event": "step_update", "step_update": {"step_type": "tool", "tool_name": "generate_image",
                         "tool_info": {"output": {"path": str(output)}}}}
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps(event) + "\n" + json.dumps(
                    {"status": "SUCCESS", "response": "WONDFUL_ANTIGRAVITY_IMAGEGEN_OK"}), stderr="")

            with patch.object(AGY, "discover_antigravity_cli", return_value="/mock/agy"), \
                 patch.object(AGY, "_run", side_effect=agy_run):
                AGY.generate_image(prompt=prompt, reference_paths=[str(source), str(projection), str(identity)],
                                   product_projection_path=str(projection), edit_base_path=str(source),
                                   output_path=str(output), cwd=str(root), source_only=True)
        self.assertEqual(len(captured), 2)
        for text in captured:
            self.assertIn("只确定各部件的材质属性", text)
            self.assertIn("车漆色相保持当前 Blender 模型", text)
            self.assertNotIn("身份参考控制可见细部拓扑", text)


if __name__ == "__main__":
    unittest.main()
