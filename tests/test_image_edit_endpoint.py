import base64
import io
import json
import os
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

# Set up paths for test harness import
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from test_316_regressions import load_package_module
import mock_codex_backend

direct = load_package_module("codex_direct")


class TestImageEditEndpoint(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.png1_bytes = mock_codex_backend.solid_png(64, 64, (255, 0, 0))
        self.png2_bytes = mock_codex_backend.solid_png(64, 64, (0, 255, 0))
        self.img1_path = str(self.temp_path / "img1.png")
        self.img2_path = str(self.temp_path / "img2.png")
        Path(self.img1_path).write_bytes(self.png1_bytes)
        Path(self.img2_path).write_bytes(self.png2_bytes)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_payload_image_api_true(self):
        images = [
            (self.img1_path, "image", "first"),
            (self.img2_path, "image", "second"),
        ]
        payload, summary = direct.build_payload(
            "Draw a cat",
            images,
            size="1024x512",
            quality="high",
            instructions="Be creative",
            image_api=True,
        )

        self.assertEqual(payload.get("model"), "gpt-image-2")
        self.assertEqual(payload.get("prompt"), "Be creative\nDraw a cat")
        self.assertEqual(payload.get("n"), 1)
        self.assertEqual(payload.get("quality"), "high")
        self.assertEqual(payload.get("background"), "opaque")
        self.assertIn("images", payload)
        self.assertEqual(len(payload["images"]), 2)

        prefix = "data:image/png;base64,"
        url1 = payload["images"][0]["image_url"]
        url2 = payload["images"][1]["image_url"]
        self.assertTrue(url1.startswith(prefix))
        self.assertTrue(url2.startswith(prefix))
        self.assertEqual(base64.b64decode(url1[len(prefix):]), self.png1_bytes)
        self.assertEqual(base64.b64decode(url2[len(prefix):]), self.png2_bytes)

        for forbidden in ("tools", "input", "reasoning", "complexity", "stream"):
            self.assertNotIn(forbidden, payload)

        self.assertEqual([row["role"] for row in summary], ["image", "image"])

    def test_build_payload_image_api_masks_reject(self):
        images = [(self.img1_path, "image", "input")]
        with self.assertRaises(direct.DirectEditError):
            direct.build_payload(
                "Fix image",
                images,
                mask_path=self.img2_path,
                size="1024x512",
                image_api=True,
            )

    def test_build_payload_legacy_schema(self):
        images = [(self.img1_path, "image", "input")]
        payload, summary = direct.build_payload(
            "Draw a cat",
            images,
            size="1024x512",
            image_api=False,
        )
        self.assertIn("tools", payload)

    def _call_post_responses(self, mock_response_data, initial_meta=None):
        captured_req = []
        meta = dict(initial_meta) if initial_meta is not None else {}

        class MockResponse:
            def __enter__(self):
                return io.BytesIO(json.dumps(mock_response_data).encode("utf-8"))

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def fake_urlopen(req, timeout=None):
            captured_req.append(req)
            return MockResponse()

        with patch.object(direct.urlrequest, "urlopen", side_effect=fake_urlopen):
            res = direct.post_responses(
                "https://chatgpt.com/backend-api/codex/images/edits",
                {},
                {"test": 1},
                timeout=5,
                want="image_edit",
                meta=meta,
            )
        return res, captured_req, meta

    def test_post_responses_success(self):
        valid_response = {
            "data": [{"b64_json": "ZmFrZV9pbWFnZQ==", "generation_id": "gen_123"}],
            "quality": "high",
            "size": "1024x512",
            "model": "gpt-image-2",
        }
        res, reqs, meta = self._call_post_responses(valid_response)
        self.assertIsInstance(res, tuple)
        b64, item, generation_id = res[:3]
        self.assertEqual(b64, "ZmFrZV9pbWFnZQ==")
        self.assertEqual(generation_id, "gen_123")
        self.assertEqual(item.get("generation_id"), "gen_123")
        self.assertEqual(len(reqs), 1)

        self.assertEqual(meta.get("item_quality"), "high")
        self.assertEqual(meta.get("item_size"), "1024x512")
        self.assertEqual(meta.get("image_model"), "gpt-image-2")
        self.assertEqual(meta.get("item_action"), "edit")

    def test_post_responses_success_without_model(self):
        response_without_model = {
            "data": [{"b64_json": "ZmFrZV9pbWFnZQ==", "generation_id": "gen_456"}],
            "quality": "standard",
            "size": "1024x512",
        }
        res, reqs, meta = self._call_post_responses(response_without_model)
        b64, item, generation_id = res[:3]
        self.assertEqual(generation_id, "gen_456")
        self.assertEqual(meta.get("item_quality"), "standard")
        self.assertEqual(meta.get("item_size"), "1024x512")
        self.assertNotIn("image_model", meta)

    def test_post_responses_reject_malformed(self):
        cases = [
            {"error": {"message": "Invalid request"}},
            {"data": []},
            {"data": [{}]},
            {"data": [{"b64_json": "abc"}, {"b64_json": "def"}]},
            {},
        ]
        for bad_resp in cases:
            with self.assertRaises(direct.DirectEditError):
                self._call_post_responses(bad_resp)

        class MalformedJsonContext:
            def __enter__(self):
                return io.BytesIO(b"not json{")

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        with patch.object(direct.urlrequest, "urlopen", return_value=MalformedJsonContext()):
            with self.assertRaises(direct.DirectEditError):
                direct.post_responses(
                    "https://chatgpt.com/backend-api/codex/images/edits",
                    {},
                    {},
                    timeout=5,
                    want="image_edit",
                    meta={},
                )


if __name__ == "__main__":
    unittest.main()
