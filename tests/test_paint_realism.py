import copy
import unittest
from unittest.mock import patch

import test_two_workflows as T

al = T.load("appearance_lock")
sr = T.load("source_reference")
cd = T.load("codex_direct")


class TestPaintRealism(unittest.TestCase):
    def test_source_paint_lock_fallback_and_immutability(self):
        rows = [
            dict(
                name="CarPaint",
                base_color=[0, 0.27592355, 0.05033694],
                roughness=0.4,
                metallic=0,
                weight=10,
                roughness_linked=True,
            ),
            dict(
                name="Blaack_Paint",
                base_color=[0.006598276] * 3,
                roughness=0.4,
                weight=1,
            ),
        ]
        rows_orig = copy.deepcopy(rows)
        props = T.NS(
            product_collection=T.NS(
                all_objects=[T.NS(type="MESH", hide_render=False)]
            )
        )
        ns = T.function_namespace("_source_paint_lock")
        with patch.object(al, "collect_product_materials", return_value=rows):
            baseline = ns["_source_paint_lock"](props)

        self.assertIn("#008F3F", baseline)
        self.assertIn("#131313", baseline)
        second_line = baseline.splitlines()[1]
        for forbidden in ["半哑光", "哑光", "高光泽", "带清漆层"]:
            self.assertNotIn(forbidden, second_line)
        self.assertEqual(rows, rows_orig)

    def test_paint_finish_policy_and_payload_prompt(self):
        rows = [
            dict(
                name="CarPaint",
                base_color=[0, 0.27592355, 0.05033694],
                roughness=0.4,
                metallic=0,
                weight=10,
                roughness_linked=True,
            )
        ]
        props = T.NS(
            product_collection=T.NS(
                all_objects=[T.NS(type="MESH", hide_render=False)]
            )
        )
        ns = T.function_namespace("_source_paint_lock")
        with patch.object(al, "collect_product_materials", return_value=rows):
            baseline = ns["_source_paint_lock"](props)

        prompt = sr.edit_prompt(
            "完整用户要求",
            identity_text=baseline + "用户明确要求哑光漆",
            projection=True,
        )

        with patch.object(
            cd,
            "data_url",
            return_value=("data:image/png;base64,ZmFrZQ==", "image/png", 4),
        ), patch.object(cd, "_image_size", return_value=[64, 64]):
            payload, _ = cd.build_payload(
                prompt,
                [
                    ("source", "source_canvas", ""),
                    ("guide", "product_projection", ""),
                    ("identity", "product", ""),
                ],
                size="1024x512",
                instructions=sr.PROJECTED_EDIT_INSTRUCTIONS,
                image_api=True,
            )

        prompt_text = payload.get("prompt", "")
        self.assertIn("用户明确要求哑光漆", prompt_text)
        self.assertIn("#008F3F", prompt_text)
        self.assertTrue(
            ("preview lighting" in prompt_text) or ("matte finish" in prompt_text)
        )
        self.assertIn("evidenced paint gloss/clearcoat", prompt_text)
        self.assertIn("Curvature reflections", prompt_text)
        for term in ["清漆层", "基础色相", "局部 RGB", "反射形状"]:
            self.assertIn(term, prompt_text)
        self.assertIn("reflection blur", prompt_text)
        self.assertIn("hard-edged reflection bands", prompt_text)
        self.assertIn("must not redesign any visible modeled parts", prompt_text)
        self.assertIn(sr.PAINT_FINISH_POLICY, sr.PROJECTED_POLISH_SYSTEM)


if __name__ == "__main__":
    unittest.main()
