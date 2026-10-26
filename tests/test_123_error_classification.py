"""1.2.3: one classifier for HTTP and in-stream Codex failures.

Quota exhausted and login rejected stop the render at once with a clear message and never fall
through to the codex exec proxy; short rate limits get a bounded wait; a cut stream is retried
once; a plain 400 still tries the lean payload. Recorded / sanitized bodies live in
mock_codex_backend.FIXTURES."""
import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_321_direct_edit import DirectEditTestBase, direct  # noqa: E402
import mock_codex_backend as mock  # noqa: E402


class ClassifierTests(unittest.TestCase):
    def k(self, status, body, headers=None):
        return direct.classify_failure(status, body if isinstance(body, str) else body.decode(), headers)

    def test_http_bodies(self):
        F = mock.FIXTURES
        self.assertEqual(self.k(401, F["http_401_real"])[:2], ("auth", False))   # refresh once first
        self.assertEqual(self.k(403, b'{"detail":"forbidden"}')[:2], ("auth", True))
        kind, fatal, wait = self.k(429, F["http_429_usage_limit"])
        self.assertEqual((kind, fatal), ("quota", True))
        self.assertEqual(wait, 7200)
        self.assertEqual(self.k(429, F["http_429_rate"], {"Retry-After": "3"}), ("rate_limit", False, 3.0))
        self.assertEqual(self.k(429, F["http_429_rate"], {"Retry-After": "900"})[0], "rate_limit_long")
        self.assertEqual(self.k(400, F["http_400_usage_not_included"])[:2], ("quota", True))
        self.assertEqual(self.k(400, F["http_400_real"])[:2], ("request", False))
        self.assertEqual(self.k(404, b'{"detail":"Not Found"}')[:2], ("endpoint", True))
        self.assertEqual(self.k(502, b"bad gateway")[0], "server")

    def test_stream_events_use_same_rules(self):
        F = mock.FIXTURES
        for key, kind in (("sse_failed_usage_limit", "quota"), ("sse_failed_rate_limit", "rate_limit"),
                          ("sse_error_token_invalidated", "auth")):
            ev = json.loads(F[key])
            lines = [f"data: {json.dumps(ev)}", ""]
            with self.assertRaises(direct.DirectEditError) as cm:
                direct.extract_image(direct.parse_sse_events(lines))
            self.assertEqual(cm.exception.kind, kind, key)
        ev = json.loads(F["sse_failed_rate_limit"])
        with self.assertRaises(direct.DirectEditError) as cm:
            direct.extract_image(direct.parse_sse_events([f"data: {json.dumps(ev)}", ""]))
        self.assertAlmostEqual(cm.exception.retry_after, 0.3)
        # old bug: an in-stream rate limit used to be called "quota"
        self.assertNotEqual(cm.exception.kind, "quota")

    def test_cut_stream_is_not_a_parse_error(self):
        lines = ['data: {"type":"response.created","response":{"id":"r"}}', ""]
        with self.assertRaises(direct.DirectEditError) as cm:
            direct.extract_image(direct.parse_sse_events(lines))
        self.assertEqual(cm.exception.kind, "stream_cut")

    def test_reported_model_is_read_from_the_echo(self):
        meta = {}
        created = {"type": "response.created", "response": {"id": "r", "model": "gpt-6.1-sol", "tools": [
            {"type": "image_generation", "model": "gpt-image-2-codex", "quality": "auto", "size": "auto"}]}}
        item = {"type": "response.output_item.done", "item": {"type": "image_generation_call", "status": "completed",
                                                             "result": "AAAA", "quality": "low", "size": "1672x941",
                                                             "action": "edit"}}
        lines = [f"data: {json.dumps(created)}", "", f"data: {json.dumps(item)}", ""]
        direct.extract_image(direct.parse_sse_events(lines), meta)
        self.assertEqual(meta["image_model"], "gpt-image-2-codex")
        self.assertEqual(meta["item_size"], "1672x941")
        self.assertEqual(meta["item_quality"], "low")

    def test_retry_seconds(self):
        self.assertAlmostEqual(direct._retry_seconds("Please try again in 11.054s."), 11.054)
        self.assertAlmostEqual(direct._retry_seconds("try again in 250ms"), 0.25)
        self.assertAlmostEqual(direct._retry_seconds("try again in 2 minutes"), 120.0)
        self.assertEqual(direct._retry_seconds("", {"Retry-After": "7"}), 7.0)


class _StopBase(DirectEditTestBase):
    def setUp(self):
        super().setUp()
        self._backoff = direct.NET_BACKOFF
        direct.NET_BACKOFF = (0.0, 0.0)      # 1.2.7 back-off retries, no waiting in tests

    def tearDown(self):
        direct.NET_BACKOFF = self._backoff
        super().tearDown()

    def run_gen(self):
        self.fb_calls = []

        def fb(**kw):
            self.fb_calls.append(kw)
            return {"engine": "codex_oauth_imagegen"}
        self.gstate = {}
        kwargs = dict(prompt="p", reference_paths=self.refs, output_path=str(self.tmp / "o.png"),
                      cwd=str(self.tmp), edit_base_path=self.camera, timeout=60)
        return direct.generate_with_fallback(kwargs, fallback=fb, state=self.gstate, size="2048x1152")


class QuotaHttpStops(_StopBase):
    mode = "quota"

    def test_quota_stops_without_fallback(self):
        with self.assertRaises(direct.DirectEditStop) as cm:
            self.run_gen()
        self.assertEqual(self.fb_calls, [])
        self.assertIn("额度已用完", str(cm.exception))
        self.assertIn("没有改用代理生成", str(cm.exception))
        self.assertEqual(self.state.response_calls, 1)          # no blind retries burning quota
        # later variants / regenerations of the same render stop before any request
        with self.assertRaises(direct.DirectEditStop):
            self.run_gen_same_state()
        self.assertEqual(self.state.response_calls, 1)

    def run_gen_same_state(self):
        kwargs = dict(prompt="p", reference_paths=self.refs, output_path=str(self.tmp / "o2.png"),
                      cwd=str(self.tmp), edit_base_path=self.camera, timeout=60)
        return direct.generate_with_fallback(kwargs, fallback=lambda **k: {}, state=self.gstate, size="2048x1152")


class QuotaStreamStops(_StopBase):
    mode = "stream_quota"

    def test_in_stream_quota_stops(self):
        with self.assertRaises(direct.DirectEditStop):
            self.run_gen()
        self.assertEqual(self.fb_calls, [])


class Quota400Stops(_StopBase):
    mode = "400_quota_text"

    def test_400_with_quota_text_stops_without_lean_retries(self):
        with self.assertRaises(direct.DirectEditStop):
            self.run_gen()
        self.assertEqual(self.fb_calls, [])
        self.assertEqual(self.state.response_calls, 1)


class AuthStops(_StopBase):
    mode = "401_always"

    def test_401_refreshes_once_then_stops(self):
        with self.assertRaises(direct.DirectEditStop) as cm:
            self.run_gen()
        self.assertEqual(self.fb_calls, [])
        self.assertEqual(self.state.token_calls, 1)
        self.assertEqual(self.state.response_calls, 2)
        self.assertIn("codex login", str(cm.exception))


class StreamAuthStops(_StopBase):
    mode = "stream_auth"

    def test_in_stream_token_invalidated_stops(self):
        with self.assertRaises(direct.DirectEditStop):
            self.run_gen()
        self.assertEqual(self.fb_calls, [])


class RateShortWaits(_StopBase):
    mode = "rate_then_ok"

    def test_short_rate_limit_waits_then_succeeds(self):
        t0 = time.time()
        r = self.run_gen()
        self.assertEqual(r["engine"], "codex_direct_edit")
        self.assertEqual(self.state.response_calls, 2)
        self.assertGreaterEqual(time.time() - t0, 0.9)          # honoured Retry-After: 1
        self.assertEqual(self.fb_calls, [])


class RateStreamWaits(_StopBase):
    mode = "stream_rate_then_ok"

    def test_in_stream_rate_limit_waits(self):
        r = self.run_gen()
        self.assertEqual(r["engine"], "codex_direct_edit")
        self.assertEqual(self.state.response_calls, 2)


class RateLongStops(_StopBase):
    mode = "rate_long"

    def test_long_rate_limit_stops_without_waiting(self):
        t0 = time.time()
        with self.assertRaises(direct.DirectEditStop) as cm:
            self.run_gen()
        self.assertLess(time.time() - t0, 10)
        self.assertEqual(self.fb_calls, [])
        self.assertIn("限流", str(cm.exception))


class StreamCutRetried(_StopBase):
    mode = "stream_cut_then_ok"

    def test_cut_stream_retried(self):
        r = self.run_gen()
        self.assertEqual(r["engine"], "codex_direct_edit")
        self.assertEqual(self.state.response_calls, 2)


class StreamCutFallsBack(_StopBase):
    mode = "stream_cut_always"

    def test_persistent_cut_falls_back_not_disabled(self):
        r = self.run_gen()
        self.assertEqual(r["engine"], "codex_oauth_imagegen")    # transient: proxy still allowed
        self.assertEqual(self.state.response_calls, 3)            # 1.2.7: 2 back-off retries, then the proxy
        self.assertFalse(self.gstate.get("stopped"))
        self.assertFalse(self.gstate.get("disabled"))


class MissingAuthStillFallsBack(_StopBase):
    def test_missing_auth_file_is_local_and_falls_back(self):
        (self.tmp / "auth.json").unlink()
        import os
        os.environ["CODEX_HOME"] = str(self.tmp / "nowhere")
        if (Path.home() / ".codex" / "auth.json").exists():
            self.skipTest("a real ~/.codex/auth.json exists on this machine")
        r = self.run_gen()
        self.assertEqual(r["engine"], "codex_oauth_imagegen")


class OperatorWiring(unittest.TestCase):
    def test_operator_stops_on_direct_stop(self):
        src = (Path(__file__).resolve().parents[1] / "wondful_ai_renderer" / "operators.py").read_text(encoding="utf-8")
        self.assertIn("DirectEditStop", src)


if __name__ == "__main__":
    unittest.main()
