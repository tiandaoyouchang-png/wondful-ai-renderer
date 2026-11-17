"""Offline source-canvas contract: capture actual payloads without auth or network."""
import ast
import base64
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module

CD = load_package_module("codex_direct")
CC = load_package_module("codex_client")
PNG = load_package_module("png_io")
COMP = load_package_module("composition")
RB = load_package_module("reference_bundle")


class ReferenceHybridTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.clay = self.picture("clay.png", 80)
        self.source = self.picture("source.png", 140)
        self.product = self.picture("identity.png", 210)
        self.candidate = self.picture("candidate.png", 110)
        self.mask = COMP.write_repair_mask(self.root / "repair_mask.png", (80, 48), (0.2, 0.3, 0.8, 0.9))
        self.refs = [self.clay, self.product, self.source]
        self.manifest = self.roles([(self.clay, "camera"), (self.product, "product"), (self.source, "scene_reference")])
        self.brief = "保持原有暗部、镜头拖影与冷光。" + "完整润色细节。" * 1000 + "全文结束标记。"

    def picture(self, name, color):
        path = self.root / name
        PNG.write_png(str(path), np.full((48, 80, 3), color, np.uint8))
        return str(path)

    @staticmethod
    def roles(items):
        return [{"path": p, "members": [{"path": p, "role": role, "reference": i}]}
                for i, (p, role) in enumerate(items, 1)]

    def kwargs(self, **changes):
        args = dict(prompt="OLD_RENDER_PROMPT with environment reconstruction", reference_paths=self.refs,
                    edit_base_path=self.clay, edit_mask_path=self.mask,
                    output_path=str(self.root / "out.png"), cwd=str(self.root), timeout=60)
        args.update(changes)
        return args

    def edit(self, **changes):
        args = dict(prompt="OLD_RENDER_PROMPT", edit_base_path=self.clay, reference_paths=self.refs,
                    output_path=str(self.root / "out.png"), size="80x48", mask_path=self.mask,
                    manifest=self.manifest, log_dir=str(self.root), simple_brief=self.brief,
                    identity_text="TARGET_IDENTITY_LOCK", scene_reference_path=self.source,
                    original_first=True, terrain_text="TERRAIN_REBUILD", massing_text="MASSING_REBUILD",
                    building_refs=[{"path": self.product, "block": 39}], style_text="STYLE_REBUILD")
        args.update(changes)
        with ExitStack() as stack:
            for name, value in (("find_auth_file", self.root / "fake_auth"), ("load_auth", {}),
                                ("token_expiring", False), ("model_chain", ["test-model"]),
                                ("available_models", ["test-model"]), ("auth_headers", {})):
                stack.enter_context(patch.object(CD, name, return_value=value))
            no_summary = stack.enter_context(patch.object(CD, "describe_environment", side_effect=AssertionError("unexpected summary")))
            response = (base64.b64encode(Path(self.candidate).read_bytes()).decode(), {}, "mock-response")
            post = stack.enter_context(patch.object(CD, "post_responses", return_value=response))
            result = CD.edit_image(**args)
            no_summary.assert_not_called()
        return post.call_args.args[2], result

    @staticmethod
    def payload_images(payload):
        return [base64.b64decode(c["image_url"].split(",", 1)[1])
                for c in payload["input"][0]["content"] if c["type"] == "input_image"]

    def test_actual_payload_source_then_projection_then_identity_no_clay_mask(self):
        payload, result = self.edit()
        self.assertEqual(self.payload_images(payload), [Path(p).read_bytes() for p in [self.source, self.clay, self.product]])
        self.assertEqual(result["image_roles"], ["source_canvas", "product_projection", "product"])
        self.assertFalse(result["mask"])
        self.assertNotIn("input_image_mask", payload["tools"][0])
        self.assertEqual(payload["instructions"], CD.HYBRID_INSTRUCTIONS)
        self.assertNotIn("keep its framing, camera, product geometry", payload["instructions"])
        text = payload["input"][0]["content"][0]["text"]
        self.assertEqual(text.count(self.brief), 1)
        self.assertIn("TARGET_IDENTITY_LOCK", text)
        self.assertIn("来源图中的旧产品必须被目标产品替换", text)
        for excluded in ("OLD_RENDER_PROMPT", "TERRAIN_REBUILD", "MASSING_REBUILD", "STYLE_REBUILD",
                         "山的最高处", "灰蓝色体块对应", CD.SCENE_REFERENCE_POLICY):
            self.assertNotIn(excluded, text)

    def test_hybrid_repair_keeps_candidate_mask_and_distinct_source_guide(self):
        payload, result = self.edit(edit_base_path=self.candidate, product_projection_path=self.clay,
                                    repair=True, correction="产品向右恢复")
        self.assertEqual(self.payload_images(payload), [Path(p).read_bytes() for p in
                                                       [self.candidate, self.clay, self.source, self.product]])
        self.assertTrue(result["mask"])
        tool_mask = payload["tools"][0]["input_image_mask"]["image_url"]
        self.assertEqual(base64.b64decode(tool_mask.split(",", 1)[1]), Path(self.mask).read_bytes())
        text = payload["input"][0]["content"][0]["text"]
        self.assertIn("产品向右恢复", text)
        self.assertIn("上一轮候选图", text)
        self.assertIn("TARGET_IDENTITY_LOCK", text)
        self.assertEqual(text.count(self.brief), 1)

    def test_empty_polish_uses_complete_fallback_once(self):
        fallback = self.brief + "旧版环境体块重建指令；Reference 1 = old camera。"
        payload, _result = self.edit(simple_brief="  ", prompt=fallback)
        text = payload["input"][0]["content"][0]["text"]
        self.assertEqual(text.count(fallback), 1)
        self.assertIn("旧环境体块、地形、建筑或构图重建指令不生效", text)
        self.assertIn("Legacy environment, terrain or building reconstruction directives do not apply", payload["instructions"])

    def test_nonempty_polish_ignores_fallback_and_empty_helper_retains_it(self):
        images, _ = CD.hybrid_images(self.clay, self.refs, self.manifest, self.source, self.clay)
        for brief in (None, "", "  "):
            text = CD.build_hybrid_prompt(brief, images, fallback_prompt=self.brief)
            self.assertEqual(text.count(self.brief), 1)
        text = CD.build_hybrid_prompt(self.brief, images, fallback_prompt="UNUSED_OLD_REBUILD")
        self.assertNotIn("UNUSED_OLD_REBUILD", text)

    def test_default_remains_clay_canvas_and_ordinary_mask(self):
        payload, result = self.edit(original_first=False, scene_reference_path="")
        self.assertEqual(self.payload_images(payload)[0], Path(self.clay).read_bytes())
        self.assertEqual(payload["instructions"], CD.BUILDING_INSTRUCTIONS)
        self.assertTrue(result["mask"])

    def test_wrapper_does_not_mistake_swapped_source_for_repair(self):
        fallback = Mock()
        for base, expected_repair in ((self.clay, False), (self.source, False), (self.candidate, True)):
            with self.subTest(base=base), patch.object(CD, "edit_image", return_value={}) as edit:
                CD.generate_with_fallback(self.kwargs(edit_base_path=base), fallback=fallback, state={}, size="80x48",
                                          manifest=self.manifest, simple_base=self.clay, simple_brief=self.brief,
                                          scene_reference_path=self.source, original_first=True, identity_text="LOCK")
                self.assertEqual(edit.call_args.kwargs["repair"], expected_repair)
                self.assertEqual(edit.call_args.kwargs["edit_base_path"], self.candidate if expected_repair else self.source)
                self.assertEqual(edit.call_args.kwargs["product_projection_path"], self.clay)
        fallback.assert_not_called()

    def fallback(self, *, repair=False, products=1):
        identities = [self.product] + [self.picture(f"identity{i}.png", 20 + i) for i in range(products - 1)]
        refs = [self.clay, self.source] + identities
        manifest = self.roles([(self.clay, "camera"), (self.source, "scene_reference")]
                              + [(p, "product") for p in identities])
        seen = {}
        self.exec_calls = []

        def execute(**kwargs):
            self.exec_calls.append(kwargs)
            seen.update(kwargs)
            Path(self.root / "out.png").write_bytes(Path(self.candidate).read_bytes())
            return "WONDFUL_IMAGEGEN_OK"

        packet = {"enabled": True, "bbox_normalized": [0.2, 0.3, 0.8, 0.9],
                  "scene_bbox_normalized": "OLD_SCENE_BOUNDARY", "generation_reference_roles": ["scene_depth"]}
        with patch.object(CC, "_exec_turn", execute):
            result = CD.generate_with_fallback(
                self.kwargs(reference_paths=refs, edit_base_path=self.candidate if repair else self.source,
                            structure_control=packet), fallback=CC.generate_image, state={"disabled": "test"},
                size="80x48", manifest=manifest, simple_base=self.clay, simple_brief=self.brief,
                identity_text="TARGET_LOCK", scene_reference_path=self.source, original_first=True,
                terrain_text="TERRAIN_REBUILD", massing_text="MASSING_REBUILD")
        return seen, result, identities

    def test_real_cli_fallback_contract_and_packing(self):
        seen, result, identities = self.fallback(products=7)
        self.assertEqual(seen["image_paths"][:2], [self.source, self.clay])
        cells = [c for s in result["collage"].get("sheets", []) for c in s["cells"]]
        retained = {Path(p).name for p in seen["image_paths"]} | {c["file"] for c in cells}
        self.assertTrue({Path(p).name for p in identities}.issubset(retained))
        self.assertFalse({Path(self.source).name, Path(self.clay).name} & {c["file"] for c in cells})
        self.assertNotIn(self.mask, seen["image_paths"])
        text = seen["prompt"]
        self.assertEqual(text.count(self.brief), 1)
        self.assertIn("TARGET_LOCK", text)
        self.assertIn("product_bbox_normalized", text)
        for excluded in ("Camera Base 是唯一整张画面", "最高结构权威", "OLD_SCENE_BOUNDARY", "OLD_RENDER_PROMPT",
                         "MASSING_REBUILD", "TERRAIN_REBUILD", "Scene Depth"):
            self.assertNotIn(excluded, text)

    def test_real_cli_repair_keeps_candidate_mask_source_and_guide(self):
        seen, _result, _identities = self.fallback(repair=True)
        self.assertEqual(seen["image_paths"][:3], [self.candidate, self.clay, self.source])
        self.assertIn(self.mask, seen["image_paths"])
        self.assertIn("透明区域可修复", seen["prompt"])

    def test_identity_capacity_failure_does_not_call_cli(self):
        with self.assertRaisesRegex(RB.ReferenceLimitError, "身份参考"):
            self.fallback(repair=True, products=10)
        self.assertFalse(self.exec_calls)

    def test_direct_fit_keeps_canvas_and_projection_independent(self):
        identities = [self.picture(f"product{i}.png", 20 + i) for i in range(7)]
        images, _ = CD.hybrid_images(self.clay, identities, self.roles([(p, "product") for p in identities]),
                                     self.source, self.clay)
        fitted, dropped, _note, info = CD.fit_images(images, 5, self.root, "128x80")
        self.assertEqual([p for p, _r, _l in fitted[:2]], [self.source, self.clay])
        self.assertFalse(dropped)
        self.assertFalse(info["text_only"])

    def test_mixed_sheet_cannot_silently_lose_identity_when_members_disappear(self):
        sheet = self.picture("mixed_sheet.png", 90)
        missing = str(self.root / "missing_identity.png")
        manifest = [{"path": sheet, "members": [{"path": self.source, "role": "scene_reference"},
                                                  {"path": missing, "role": "product"}]}]
        with self.assertRaisesRegex(CD.DirectEditError, "身份参考"):
            CD.hybrid_images(self.clay, [self.clay, sheet], manifest, self.source, self.clay)

    def test_unlabelled_identity_is_rejected_instead_of_silently_dropped(self):
        with self.assertRaisesRegex(CD.DirectEditError, "明确角色"):
            CD.hybrid_images(self.clay, self.refs, None, self.source, self.clay)

    def test_invalid_source_fails_before_auth_or_fallback(self):
        for source in ("", str(self.root / "missing.png"), self.picture("bad.png", 20)):
            if Path(source).name == "bad.png":
                Path(source).write_text("not an image")
            with self.subTest(source=source), patch.object(CD, "find_auth_file") as auth, patch.object(CD, "edit_image") as edit:
                fallback = Mock()
                with self.assertRaises(CD.DirectEditError):
                    CD.generate_with_fallback(self.kwargs(), fallback=fallback, state={}, size="80x48",
                                              manifest=self.manifest, simple_base=self.clay,
                                              scene_reference_path=source, original_first=True)
                auth.assert_not_called()
                edit.assert_not_called()
                fallback.assert_not_called()

    def test_quota_still_stops_without_cli_fallback(self):
        fallback = Mock()
        with patch.object(CD, "edit_image", side_effect=CD.DirectEditError("quota", kind="quota", fatal=True)):
            with self.assertRaises(CD.DirectEditStop):
                CD.generate_with_fallback(self.kwargs(), fallback=fallback, state={}, size="80x48",
                                          manifest=self.manifest, simple_base=self.clay,
                                          scene_reference_path=self.source, original_first=True)
        fallback.assert_not_called()

    def test_operator_flag_requires_reference_and_fresh_clay(self):
        path = Path(__file__).resolve().parents[1] / "wondful_ai_renderer" / "operators.py"
        tree = ast.parse(path.read_text())
        flags = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "direct_original_first" for t in n.targets)]
        self.assertEqual(len(flags), 2)
        self.assertIs(flags[0].value.value, False)
        self.assertIn("workflow.is_reference(props) and direct_scene_reference", ast.unparse(flags[1].value))
        fresh = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Name) and n.test.func.id == "render_clay_base"]
        self.assertTrue(any(flags[1] in list(ast.walk(n)) for n in fresh))


if __name__ == "__main__":
    unittest.main()
