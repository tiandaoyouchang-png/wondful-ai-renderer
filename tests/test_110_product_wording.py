"""1.1.0: product-generic prompts — car wording only when the product is a vehicle."""
import sys, unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402
PK = load_package_module("product_kind")
CD = load_package_module("codex_direct")
T = load_package_module("terrain")
JEV = load_package_module("jev_semantics")
ROOT = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"

CAR_WORDS = ("车", "轮毂", "轮心")


def _mesh(name, data, mats):
    return NS(name=name, type="MESH", data=NS(name=data, materials=[NS(name=m) for m in mats]))


class DetectionTests(unittest.TestCase):
    def test_names(self):
        self.assertTrue(PK.names_look_like_vehicle(["Body", "Wheel_FL"]))
        self.assertTrue(PK.names_look_like_vehicle(["car_paint"]))
        self.assertTrue(PK.names_look_like_vehicle(["前轮毂"]))
        self.assertFalse(PK.names_look_like_vehicle(["SheenChair_fabric", "Scarf", "carpet", "Cardboard"]))

    def test_collection_and_props(self):
        chair = NS(name="SheenChair", all_objects=[_mesh("chair_body", "mesh0", ["fabric", "wood"])])
        car = NS(name="YU7", all_objects=[_mesh("Mesh_12", "mesh12", ["tire_rubber"])])
        self.assertFalse(PK.is_vehicle(NS(product_collection=chair, identity_details="", product_look_prompt="", scene_brief="")))
        self.assertTrue(PK.is_vehicle(NS(product_collection=car, identity_details="", product_look_prompt="", scene_brief="")))
        # the user's own text wins
        self.assertTrue(PK.is_vehicle(NS(product_collection=chair, identity_details="车标，轮毂", product_look_prompt="", scene_brief="")))
        self.assertFalse(PK.is_vehicle(None, None))


class SimplePromptWordingTests(unittest.TestCase):
    ENV = "开阔的草地山坡，主光来自画面左侧，色温偏暖。\n车身质感：车漆有明显的天空倒影。\n不要出现：画面内的太阳。"

    def test_generic_prompt_has_no_car_words(self):
        tt = T.describe(T.preset_settings("CLIMB"), vehicle=False)
        text = CD.build_simple_prompt("", env_text=self.ENV, terrain_text=tt, vehicle=False)
        for w in CAR_WORDS:
            self.assertNotIn(w, text)
        self.assertIn("产品表面材质", text)
        self.assertIn("底部接地点", text)
        self.assertIn("草地山坡", text)
        self.assertIn("不要出现：画面内的太阳", text)

    def test_generic_prompt_keeps_product_lock(self):
        text = CD.build_simple_prompt("雪山", vehicle=False)
        for phrase in ("产品与构图严格保持原图", "位置、大小、角度、透视、轮廓", "不重新设计造型",
                       "不平移、不缩放、不裁切、不旋转", "不要把地面改平", "固有色"):
            self.assertIn(phrase, text)

    def test_vehicle_prompt_keeps_car_wording(self):
        text = CD.build_simple_prompt("雪山", env_text=self.ENV, vehicle=True)
        for w in ("车漆", "车身", "车轮", "车牌", "车身质感"):
            self.assertIn(w, text)
        self.assertIn("产品与构图严格保持原图", text)

    def test_default_is_generic(self):
        self.assertNotIn("车", CD.build_simple_prompt("雪山"))

    def test_generic_texture_paragraph(self):
        text = CD.build_simple_prompt("", env_text="草地。\n表面质感：柔和漫反射。", vehicle=False)
        self.assertIn("「表面质感」只用来决定我们这件产品", text)
        self.assertNotIn("车", text)


class EnvDescriberTests(unittest.TestCase):
    def test_product_describer_never_asks_about_cars(self):
        p = CD.ENV_DESCRIBE_PROMPT_PRODUCT
        self.assertNotIn("车身质感", p)
        self.assertIn("不要提到参考图里的车辆", p)
        self.assertIn("不要描述构图", p)
        self.assertIn("不要出现", p)

    def test_describer_choice_and_cache_key(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        img = d / "env.png"; img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        sent = []

        def fake_post(url, headers, payload, **kw):
            sent.append(payload["input"][0]["content"][0]["text"])
            return "草地"
        with mock.patch.object(CD, "post_responses", side_effect=fake_post), \
             mock.patch.object(CD, "auth_headers", return_value={}):
            CD.describe_environment([img], auth={}, auth_path=d / "a.json", cache_dir=str(d), vehicle=False)
            CD.describe_environment([img], auth={}, auth_path=d / "a.json", cache_dir=str(d), vehicle=True)
            CD.describe_environment([img], auth={}, auth_path=d / "a.json", cache_dir=str(d), vehicle=False)
        self.assertEqual(sent, [CD.ENV_DESCRIBE_PROMPT_PRODUCT, CD.ENV_DESCRIBE_PROMPT])  # cached per kind


class PlumbingTests(unittest.TestCase):
    def test_vehicle_flows_from_operator_to_prompt(self):
        ops = (ROOT / "operators.py").read_text(encoding="utf-8")
        self.assertIn("direct_vehicle = _is_vehicle(props)", ops)
        self.assertIn("vehicle=direct_vehicle", ops)
        captured = {}

        def fake_edit(**kw):
            captured.update(kw)
            return {"engine": "codex_direct_edit"}
        with mock.patch.object(CD, "edit_image", side_effect=fake_edit):
            import tempfile
            base = Path(tempfile.mkdtemp()) / "clay.png"; base.write_bytes(b"x")
            CD.generate_with_fallback({"output_path": str(base.parent / "o.png"), "reference_paths": [str(base)]},
                                      fallback=lambda **k: None, state={}, size="1536x1024",
                                      simple_base=str(base), simple_brief="", vehicle=False)
        self.assertIs(captured["vehicle"], False)

    def test_shared_rules_are_generic(self):
        text = JEV._appearance_text({"change_scope": {"choice": "MATERIAL"}, "preserve_identity_probability": 0.99})
        self.assertNotIn("车标", text)
        self.assertIn("铭牌/车牌", text)
        for name in ("codex_client.py", "antigravity_client.py"):
            src = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("车辆/产品", src)
            self.assertNotIn("重新安排车辆", src)

    def test_terrain_uses_shared_detection(self):
        vc = (ROOT / "viewport_capture.py").read_text(encoding="utf-8")
        self.assertIn("names_look_like_vehicle", vc)
        s = T.settings_from_props(NS(identity_details="车标", product_look_prompt="", scene_brief=""))
        self.assertTrue(s["vehicle"])


if __name__ == "__main__":
    unittest.main()
