"""Provider polish prompts preserve text defaults and honor explicit JSON contracts."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_316_regressions import load_package_module


CC = load_package_module("codex_client")
AGY = load_package_module("antigravity_client")


class PolishResponseFormatTests(unittest.TestCase):
    def test_codex_text_default_and_json_contract_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(CC, "_exec_turn", return_value="ok") as call:
            CC.polish_prompt(system_prompt="system", user_text="user", image_paths=[], cwd=directory)
            text_prompt = call.call_args.kwargs["prompt"]
            CC.polish_prompt(system_prompt="system", user_text="user", image_paths=[], cwd=directory,
                             response_format="json")
            json_prompt = call.call_args.kwargs["prompt"]
        self.assertIn("只输出最终可直接用于图像生成", text_prompt)
        self.assertNotIn("只输出最终可直接用于图像生成", json_prompt)
        self.assertIn("只输出 JSON 对象", json_prompt)

    def test_antigravity_text_default_and_json_contract_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(AGY, "_exec_text", return_value="ok") as call:
            AGY.polish_prompt(system_prompt="system", user_text="user", image_paths=[], cwd=directory)
            text_prompt = call.call_args.kwargs["prompt"]
            AGY.polish_prompt(system_prompt="system", user_text="user", image_paths=[], cwd=directory,
                              response_format="json")
            json_prompt = call.call_args.kwargs["prompt"]
        self.assertIn("只输出最终可直接用于图像生成", text_prompt)
        self.assertNotIn("只输出最终可直接用于图像生成", json_prompt)
        self.assertIn("只输出 JSON 对象", json_prompt)
