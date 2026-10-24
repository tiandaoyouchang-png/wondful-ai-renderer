"""1.3.6: capture the actual edit payload without contacting a generation service."""
import base64
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module
from mock_codex_backend import solid_png

CD = load_package_module("codex_direct")
SS = load_package_module("static_scene")


class ReferenceLightingPayloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="wondful136_")
        self.root = Path(self.tmp.name)
        self.calls = []
        self.stack = ExitStack()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.stack.close)
        self.addCleanup(SS.set_brief, "")
        SS.set_brief("")
        for name, value in (("find_auth_file", self.root / "fake_auth.json"),
                            ("load_auth", {"tokens": {"access_token": "test-only"}}),
                            ("token_expiring", False), ("available_models", [CD.DEFAULT_MODEL]),
                            ("auth_headers", {})):
            self.stack.enter_context(mock.patch.object(CD, name, return_value=value))
        self.stack.enter_context(mock.patch.object(CD, "post_responses", side_effect=self.capture))
        self.camera = self.png("camera.png", (180, 180, 180))
        self.product = self.png("product.png", (0, 140, 20))
        self.env = self.png("environment.png", (200, 160, 120))
        self.structure = self.png("silhouette.png", (255, 0, 0))
        self.manifest = [self.group(self.camera, "camera"), self.group(self.structure, "structure", "mask"),
                         self.group(self.product, "product"), self.group(self.env, "environment")]
        self.refs = [self.camera, self.structure, self.product, self.env]

    def png(self, name, color=(100, 110, 120)):
        p = self.root / name
        p.write_bytes(solid_png(64, 36, color))
        return str(p)

    def group(self, path, role, sub=""):
        return {"path": path, "members": [{"reference": 1, "path": path, "role": role, "sub_role": sub}]}

    def capture(self, url, headers, payload, **kwargs):
        self.calls.append((payload, kwargs.get("want")))
        if kwargs.get("want") == "text":
            return "晴天混凝土建筑，主光来自右前上方，建筑遮挡天空光，地面暖色补光。"
        return base64.b64encode(solid_png(64, 36)).decode(), {}, "test-response"

    def edit(self, **kwargs):
        args = dict(prompt="完整上层提示词", edit_base_path=self.camera, reference_paths=self.refs,
                    manifest=self.manifest, output_path=str(self.root / "result.png"), size="1536x864",
                    simple_brief="混凝土建筑", log_dir=str(self.root), vehicle=True)
        args.update(kwargs)
        return CD.edit_image(**args)

    def image_payload(self):
        return next(body for body, want in reversed(self.calls) if want != "text")

    def payload_bytes(self, body):
        return [base64.b64decode(c["image_url"].split(",", 1)[1])
                for c in body["input"][0]["content"] if c["type"] == "input_image"]

    def test_product_identity_and_explicit_lock_reach_actual_payload(self):
        lock = "【产品外观锁定】绿色金属漆。\n【必须清晰保留的产品细节】灯组与轮毂。"
        result = self.edit(identity_text=lock)
        body = self.image_payload()
        images = self.payload_bytes(body)
        self.assertEqual(result["image_roles"], ["camera", "product"])
        self.assertEqual(images, [Path(self.camera).read_bytes(), Path(self.product).read_bytes()])
        text = body["input"][0]["content"][0]["text"]
        self.assertIn(lock, text)
        self.assertIn("图 2：产品三视图", text)
        self.assertIn("唯一相机与产品投影依据", text)
        self.assertNotIn("锐利的高光带", text)
        self.assertNotIn("约 50mm", text)
        self.assertNotIn("地平线反射线", text)

    def test_scene_source_is_standalone_and_described_once(self):
        result = self.edit(scene_reference_path=self.env)
        body = self.image_payload()
        self.assertEqual(result["image_roles"], ["camera", "scene_reference", "product"])
        self.assertEqual(self.payload_bytes(body)[:2], [Path(self.camera).read_bytes(), Path(self.env).read_bytes()])
        self.assertEqual(sum(want == "text" for _body, want in self.calls), 1)
        text = body["input"][0]["content"][0]["text"]
        self.assertIn("连续曲面的类型、连接、开口与遮挡关系", text)
        self.assertIn("不复制来源图中的车型、车漆颜色或固定高光形状", text)
        self.assertIn("保持 Blender 底图已匹配的相机与产品投影", text)

    def test_packed_reference_is_unwrapped_by_member_role(self):
        sheet = self.png("mixed_sheet.png", (10, 20, 200))
        manifest = [self.group(self.camera, "camera"),
                    {"path": sheet, "members": [
                        {"reference": 2, "path": self.product, "role": "product"},
                        {"reference": 3, "path": self.env, "role": "environment"},
                        {"reference": 4, "path": self.structure, "role": "structure", "sub_role": "mask"}]}]
        self.edit(reference_paths=[self.camera, sheet], manifest=manifest)
        self.assertEqual(self.payload_bytes(self.image_payload()),
                         [Path(self.camera).read_bytes(), Path(self.product).read_bytes()])
        env_call = next(body for body, want in self.calls if want == "text")
        self.assertEqual(self.payload_bytes(env_call), [Path(self.env).read_bytes()])

    def test_product_views_survive_budget_as_standalone_or_collage_cells(self):
        products = [self.png(f"product_{i}.png", (i * 20, 120, 30)) for i in range(7)]
        manifest = [self.group(self.camera, "camera")] + [self.group(p, "product") for p in products]
        result = self.edit(reference_paths=[self.camera] + products, manifest=manifest,
                           scene_reference_path=self.env)
        body = self.image_payload()
        self.assertLessEqual(len(self.payload_bytes(body)), CD.MAX_EDIT_IMAGES)
        self.assertEqual(self.payload_bytes(body)[:2], [Path(self.camera).read_bytes(), Path(self.env).read_bytes()])
        info = result["collage"]
        surviving = set(info["standalone"])
        surviving.update(c["file"] for sheet in info["sheets"] for c in sheet["cells"])
        self.assertTrue({Path(p).name for p in products}.issubset(surviving))
        self.assertFalse(any(it["role"] == "product" for it in info["text_only"]))

    def test_explicit_identity_geometry_kept_but_stale_camera_masks_omitted(self):
        detail = self.png("identity_detail.png", (40, 80, 200))
        result = self.edit(reference_paths=self.refs + [detail],
                           manifest=self.manifest + [self.group(detail, "structure", "identity_detail")])
        images = self.payload_bytes(self.image_payload())
        self.assertIn(Path(detail).read_bytes(), images)
        self.assertNotIn(Path(self.structure).read_bytes(), images)
        self.assertEqual(result["image_roles"], ["camera", "product", "structure"])

    def over_capacity_inputs(self):
        products = [self.png(f"view_{i:02d}.png", (i * 10, 120, 30)) for i in range(14)]
        detail = self.png("required_identity_detail.png", (40, 80, 200))
        buildings = [{"block": i + 1, "path": self.png(f"block_{i}.png"), "hint": "建筑"}
                     for i in range(4)]
        refs = [self.camera] + products + [detail]
        manifest = ([self.group(self.camera, "camera")] + [self.group(p, "product") for p in products]
                    + [self.group(detail, "structure", "identity_detail")])
        return refs, manifest, buildings

    def test_identity_detail_capacity_loss_stops_before_any_generation(self):
        # 2 fixed + 14 products + 4 buildings + 1 required detail = 21 inputs;
        # only 3 sheets / 18 reference cells remain, so the detail cannot be lost.
        refs, manifest, buildings = self.over_capacity_inputs()
        with self.assertRaises(CD.DirectEditError) as raised:
            self.edit(reference_paths=refs, manifest=manifest, scene_reference_path=self.env,
                      massing_text="灰蓝色体块", building_refs=buildings)
        self.assertEqual(raised.exception.kind, "identity_capacity")
        self.assertIn("required_identity_detail.png", str(raised.exception))
        self.assertFalse(self.calls)

    def test_capacity_error_cannot_fall_back_to_agent_or_retry_without_identity(self):
        refs, manifest, buildings = self.over_capacity_inputs()
        fallback = mock.Mock(return_value={"engine": "should-not-run"})
        state = {}
        args = dict(fallback=fallback, state=state, size="1536x864", manifest=manifest,
                    simple_base=self.camera, simple_brief="场景", scene_reference_path=self.env,
                    massing_text="灰蓝色体块", building_refs=buildings)
        kwargs = {"prompt": "完整请求", "reference_paths": refs, "edit_base_path": self.camera,
                  "output_path": str(self.root / "capacity.png"), "cwd": str(self.root)}
        for _attempt in range(2):
            with self.assertRaises(CD.DirectEditStop) as raised:
                CD.generate_with_fallback(kwargs, **args)
            self.assertEqual(raised.exception.kind, "identity_capacity")
            self.assertIn("required_identity_detail.png", str(raised.exception))
        fallback.assert_not_called()
        self.assertEqual(state["fallbacks"], 0)
        self.assertFalse(self.calls)
        note = CD.engine_note(state)
        self.assertIn("已停止：身份参考超过图像输入容量", note)
        self.assertIn("未改用代理生成", note)
        self.assertNotIn("限流", note)

    def test_early_direct_failure_cannot_bypass_identity_capacity_check(self):
        refs, manifest, buildings = self.over_capacity_inputs()
        fallback = mock.Mock(return_value={"engine": "should-not-run"})
        state = {}
        with mock.patch.object(CD, "edit_image", side_effect=CD.DirectEditError("auth unavailable", kind="auth_local", fatal=True)):
            with self.assertRaises(CD.DirectEditStop) as raised:
                CD.generate_with_fallback(
                    {"prompt": "完整请求", "reference_paths": refs, "edit_base_path": self.camera,
                     "output_path": str(self.root / "early_capacity.png")},
                    fallback=fallback, state=state, size="1536x864", manifest=manifest,
                    simple_base=self.camera, simple_brief="场景", scene_reference_path=self.env,
                    massing_text="灰蓝色体块", building_refs=buildings)
        self.assertEqual(raised.exception.kind, "identity_capacity")
        fallback.assert_not_called()
        self.assertEqual(state["fallbacks"], 0)
        self.assertFalse(self.calls)

    def test_person_reference_obeys_existing_static_scene_permission(self):
        person = self.png("person.png", (220, 80, 100))
        args = dict(reference_paths=self.refs + [person],
                    manifest=self.manifest + [self.group(person, "person")])
        self.edit(**args)
        self.assertNotIn(Path(person).read_bytes(), self.payload_bytes(self.image_payload()))
        SS.set_brief("有一位人物在车旁")
        self.edit(**args)
        self.assertIn(Path(person).read_bytes(), self.payload_bytes(self.image_payload()))

    def test_building_mapping_uses_actual_input_index_with_scene_source(self):
        building = self.png("building.png", (20, 70, 190))
        self.edit(scene_reference_path=self.env, massing_text="灰蓝色体块：建筑",
                  building_refs=[{"block": 1, "path": building, "hint": "连续曲面", "place": "画面左侧"}])
        body = self.image_payload()
        self.assertEqual(self.payload_bytes(body)[2], Path(building).read_bytes())
        self.assertIn("图3 是画面左侧的体块 1", body["input"][0]["content"][0]["text"])
        self.assertIn("product references supply identity details", body["instructions"])

    def test_old_lighting_description_cache_is_invalidated(self):
        old_key = CD._file_md5(self.env) + ":v2"
        cache = self.root / "env_descriptions.json"
        cache.write_text(json.dumps({old_key: "复制参考车的高光位置"}), encoding="utf-8")
        self.edit()
        self.assertEqual(sum(want == "text" for _body, want in self.calls), 1)
        text = self.image_payload()["input"][0]["content"][0]["text"]
        self.assertNotIn("复制参考车的高光位置", text)
        self.edit()
        self.assertEqual(sum(want == "text" for _body, want in self.calls), 1)

    def test_missing_identity_reference_fails_before_image_request(self):
        Path(self.product).unlink()
        with self.assertRaises(CD.DirectEditError):
            self.edit()
        self.assertFalse(any(want != "text" for _body, want in self.calls))

    def test_photo_mode_preserves_existing_camera_instead_of_forcing_50mm(self):
        with mock.patch.object(CD, "PHOTO_PROMPT", True):
            self.edit()
        text = self.image_payload()["input"][0]["content"][0]["text"]
        self.assertNotIn("50mm", text)
        self.assertIn("保持 Blender 相机的机位、投影、焦段观感和景深", text)

    def test_fallback_wrapper_forwards_the_narrow_identity_and_scene_contract(self):
        seen = {}
        def edit(**kwargs):
            seen.update(kwargs)
            return {"engine": "captured"}
        with mock.patch.object(CD, "edit_image", side_effect=edit):
            CD.generate_with_fallback(
                {"prompt": "p", "reference_paths": [self.camera], "edit_base_path": self.camera,
                 "output_path": str(self.root / "wrapper.png")},
                fallback=lambda **_kwargs: self.fail("unexpected fallback"), state={}, size="1536x864",
                simple_base=self.camera, simple_brief="场景", identity_text="身份锁定",
                scene_reference_path=self.env)
        self.assertEqual(seen["identity_text"], "身份锁定")
        self.assertEqual(seen["scene_reference_path"], self.env)

    def test_agent_fallback_retains_source_role_and_existing_camera(self):
        seen = {}
        def fallback(**kwargs):
            seen.update(kwargs)
            return {"engine": "captured_agent"}
        with mock.patch.object(CD, "edit_image", side_effect=CD.DirectEditError("test failure", kind="request")):
            CD.generate_with_fallback(
                {"prompt": "完整上层提示词", "reference_paths": [self.camera, self.product],
                 "edit_base_path": self.camera, "output_path": str(self.root / "fallback.png")},
                fallback=fallback, state={}, size="1536x864", manifest=self.manifest,
                simple_base=self.camera, simple_brief="场景", identity_text="身份锁定",
                scene_reference_path=self.env)
        self.assertEqual(seen["edit_base_path"], self.camera)
        self.assertEqual(seen["reference_paths"], [self.camera, self.product, self.env])
        self.assertEqual(seen["reference_roles"][self.env][0], "scene_reference")
        self.assertIn("身份锁定", seen["prompt"])
        self.assertIn(CD.SCENE_REFERENCE_POLICY, seen["prompt"])


if __name__ == "__main__":
    unittest.main()
