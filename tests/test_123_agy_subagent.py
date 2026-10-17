"""1.2.3 Gemini / Antigravity CLI ≥ 1.2.16: image generation runs in the built-in ``image-generator`` subagent.

A fake ``agy`` (POSIX shebang) prints stream-json for four cases: the old direct generate_image tool, an
image-generator subagent whose child conversation stores the bitmap in its brain directory, a subagent that
started but left no image (must not be re-requested), and an unrelated subagent (does not count)."""
import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

AGY = load_package_module("antigravity_client")
identity = load_package_module("identity_preserve")

FAKE = textwrap.dedent(
    r'''
    #!/usr/bin/env python3
    import json, os, struct, sys, zlib
    mode = os.environ["FAKE_AGY_MODE"]
    log = os.environ["FAKE_AGY_LOG"]
    prompt = sys.argv[sys.argv.index("--prompt") + 1] if "--prompt" in sys.argv else ""
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"prompt": prompt}) + "\n")
    def png(path):
        w, h = 32, 18
        raw = b"".join(b"\x00" + b"\x80\x80\x80\xff" * w for _ in range(h))
        def ch(k, d):
            return struct.pack(">I", len(d)) + k + d + struct.pack(">I", zlib.crc32(k + d) & 0xffffffff)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                    + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b""))
    parent = "parent-conv-0001"
    child = "child-conv-image-0001"
    brain = os.path.join(os.environ["HOME"], ".gemini", "antigravity-cli", "brain")
    if mode == "direct":
        out = os.path.join(os.getcwd(), "agy_generated.png")
        png(out)
        print(json.dumps({"event": "step_update", "conversation_id": parent,
                          "step_update": {"step_type": "tool", "tool_name": "generate_image",
                                          "tool_info": {"output": {"path": out}}}}))
    elif mode in ("subagent", "subagent_noimage", "other_subagent"):
        kind = "browser-agent" if mode == "other_subagent" else "image-generator"
        print(json.dumps({"event": "step_update", "conversation_id": parent,
                          "step_update": {"step_type": "subagent", "tool_name": "invoke_subagent",
                                          "conversation_id": parent, "step_index": 1,
                                          "subagent_info": {"subagents": [{"type_name": kind,
                                                                           "conversation_id": child}]}}}))
        if mode == "subagent":
            png(os.path.join(brain, child, "generated_image_1.png"))
        # a stale image in an unrelated conversation must never be picked up
        png(os.path.join(brain, "unrelated-conv-9999", "old.png"))
    print(json.dumps({"status": "SUCCESS", "response": "WONDFUL_ANTIGRAVITY_IMAGEGEN_OK",
                      "conversation_id": parent}))
    '''
).lstrip()


@unittest.skipIf(os.name == "nt", "fake CLI uses a POSIX shebang")
class SubagentImageTests(unittest.TestCase):
    def run_mode(self, mode):
        tmp = Path(tempfile.mkdtemp(prefix="agy_sub_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        cli = tmp / "agy"
        cli.write_text(FAKE)
        cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
        home = tmp / "home"
        home.mkdir()
        work = tmp / "work"
        work.mkdir()
        ref = tmp / "camera_base.png"
        identity.write_gray_png(ref, np.zeros((18, 32), dtype=np.uint8))
        log = tmp / "calls.jsonl"
        target = tmp / "out" / "output_candidate_v01_a01.png"
        env = {"HOME": str(home), "FAKE_AGY_MODE": mode, "FAKE_AGY_LOG": str(log)}
        with mock.patch.dict(os.environ, env):
            try:
                result = AGY.generate_image(prompt="studio car", reference_paths=[str(ref)], output_path=str(target),
                                            cwd=str(work), explicit_path=str(cli), model="", timeout=60)
                error = None
            except AGY.AntigravityError as exc:   # AntigravityImageGenerationUnavailable is a subclass
                result, error = None, exc
        calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        return result, error, calls, target, home

    def test_prompt_prefers_subagent_and_drops_only_tool_wording(self):
        _result, _error, calls, _t, _h = self.run_mode("direct")
        prompt = calls[0]["prompt"]
        self.assertIn("invoke_subagent", prompt)
        self.assertIn("image-generator", prompt)
        self.assertIn("恰好一张", prompt)
        self.assertIn("generate_image", prompt)                  # older CLI fallback still allowed
        self.assertIn("绝对路径", prompt)
        self.assertIn("WONDFUL_ANTIGRAVITY_IMAGEGEN_OK", prompt)
        self.assertIn("studio car", prompt)                      # requirements passed through
        self.assertIn("ImagePaths", prompt)
        for bad in ("第一条且唯一允许的工具调用", "不要调用任何其他工具", "唯一的工具调用"):
            self.assertNotIn(bad, prompt)

    def test_old_direct_generate_image_still_works(self):
        result, error, calls, target, _h = self.run_mode("direct")
        self.assertIsNone(error)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["image_tool"], "generate_image")
        self.assertTrue(target.is_file())
        self.assertEqual(len(calls), 1)

    def test_subagent_image_from_child_conversation(self):
        result, error, calls, target, home = self.run_mode("subagent")
        self.assertIsNone(error)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["image_tool"], "image-generator")
        self.assertIn("child-conv-image-0001", result["source"])
        self.assertNotIn("unrelated-conv", result["source"])
        self.assertTrue(target.is_file())
        self.assertEqual(len(calls), 1)

    def test_subagent_started_without_image_is_not_requested_again(self):
        result, error, calls, target, _h = self.run_mode("subagent_noimage")
        self.assertIsNone(result)
        self.assertIsInstance(error, AGY.AntigravityImageGenerationUnavailable)
        self.assertEqual(len(calls), 1)                          # no automatic second image request
        self.assertIn("未自动再次请求", str(error))
        self.assertFalse(target.exists())

    def test_unrelated_subagent_does_not_count(self):
        result, error, calls, _t, _h = self.run_mode("other_subagent")
        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertEqual(len(calls), 2)                          # no image request was made → one retry
        self.assertIn("立即执行生图", calls[1]["prompt"])
        self.assertNotIn("唯一的工具调用", calls[1]["prompt"])


class ToolInfoTests(unittest.TestCase):
    def step(self, type_name, tool_name="invoke_subagent", step_type="subagent"):
        return {"event": "step_update", "step_update": {
            "step_type": step_type, "tool_name": tool_name, "conversation_id": "parent-conv-0001",
            "subagent_info": {"subagents": [{"type_name": type_name, "conversation_id": "child-conv-0001"}]}}}

    def test_image_generator_counts(self):
        infos, called = AGY._generate_image_tool_info([self.step("image-generator")])
        self.assertTrue(called)
        self.assertEqual(len(infos), 1)
        self.assertIn("child-conv-0001", AGY._conversation_ids(infos))

    def test_other_subagents_do_not_count(self):
        for kind in ("browser-agent", "research", ""):
            infos, called = AGY._generate_image_tool_info([self.step(kind)])
            self.assertFalse(called, kind)
            self.assertEqual(infos, [])

    def test_wrong_tool_name_does_not_count(self):
        infos, called = AGY._generate_image_tool_info([self.step("image-generator", tool_name="run_command")])
        self.assertFalse(called)
        self.assertEqual(infos, [])

    def test_direct_tool_still_counts(self):
        ev = {"event": "step_update", "step_update": {"step_type": "tool", "tool_name": "generate_image",
                                                       "tool_info": {"output": {"path": "/tmp/x.png"}}}}
        infos, called = AGY._generate_image_tool_info([ev])
        self.assertTrue(called)
        self.assertEqual(infos, [{"output": {"path": "/tmp/x.png"}}])

    def test_candidate_objects_use_whole_tool_infos(self):
        src = (Path(AGY.__file__)).read_text(encoding="utf-8")
        self.assertIn("candidate_objects = tool_infos + events + [response]", src)


if __name__ == "__main__":
    unittest.main()
