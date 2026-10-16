"""1.2.2: 推理 model list merges the account's live list; CLI model rejection falls back;
「生成」 auto-chains into the AI render robustly; 1/2/3/4 images per generation."""
import ast
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

catalog = load_package_module("model_catalog")
client = load_package_module("codex_client")
chain = load_package_module("op_chain")
PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"

CLI_OLD = [
    {"id": "gpt-5.6-luna", "label": "GPT-5.6 Luna", "description": "cli", "default": False},
    {"id": "gpt-5.6-sol", "label": "GPT-5.6 Sol", "description": "cli", "default": True},
    {"id": "gpt-5.6-terra", "label": "GPT-5.6 Terra", "description": "cli", "default": False},
    {"id": "chatgpt-web/gpt-5.6", "label": "web", "description": "cli", "default": False},
]
LIVE = ["gpt-6.1-sol", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]


class MergeTests(unittest.TestCase):
    def test_live_first_dedup_cli_only_after(self):
        merged = catalog.merge_codex_models(CLI_OLD, LIVE)
        ids = [m["id"] for m in merged]
        self.assertEqual(ids[:7], LIVE)
        self.assertEqual(ids[7:], ["chatgpt-web/gpt-5.6"])
        self.assertEqual(len(ids), len(set(ids)))
        by = {m["id"]: m for m in merged}
        self.assertTrue(by["gpt-6.1-sol"]["live"])
        self.assertFalse(by["chatgpt-web/gpt-5.6"]["live"])
        self.assertEqual(by["gpt-5.6-sol"]["label"], "GPT-5.6 Sol")  # CLI metadata kept
        self.assertTrue(by["gpt-5.6-sol"]["default"])
        self.assertIn("CLI", by["gpt-6.1-sol"]["description"])

    def test_either_source_alone(self):
        self.assertEqual([m["id"] for m in catalog.merge_codex_models([], LIVE[:2])], LIVE[:2])
        self.assertEqual([m["id"] for m in catalog.merge_codex_models(CLI_OLD, [])],
                         [m["id"] for m in CLI_OLD])

    def test_discover_merged_survives_cli_failure(self):
        orig = catalog.codex_models
        try:
            def boom(cli, *, cwd, **kw):
                raise RuntimeError("old cli")
            catalog.codex_models = boom
            got = catalog.discover_codex_models_merged("codex", cwd="/tmp", live=LIVE[:3])
            self.assertEqual([m["id"] for m in got], LIVE[:3])
            with self.assertRaises(RuntimeError):
                catalog.discover_codex_models_merged("codex", cwd="/tmp", live=[])
        finally:
            catalog.codex_models = orig

    def test_codex_refresh_uses_merged(self):
        src = (PKG / "model_catalog.py").read_text()
        self.assertIn("return discover_codex_models_merged(cli, cwd=str(cwd))", src)
        self.assertIn('"version": "1.2.2"', src)


FAKE_CODEX = textwrap.dedent(r'''
    #!/usr/bin/env python3
    import os, sys
    args = sys.argv[1:]
    if "exec" not in args:
        print("Logged in using ChatGPT"); sys.exit(0)
    model = args[args.index("--model") + 1] if "--model" in args else ""
    with open(os.environ["FAKE_LOG"], "a") as f:
        f.write((model or "<default>") + "\n")
    if model in ("gpt-6.1-sol", "gpt-6-astra"):
        sys.stderr.write('ERROR: {"status":400,"error":{"message":"The \'%s\' model is not supported when using Codex with a ChatGPT account."}}\n' % model)
        sys.exit(1)
    out = args[args.index("--output-last-message") + 1]
    open(out, "w").write("POLISHED with " + (model or "default"))
''').lstrip()


@unittest.skipIf(os.name == "nt", "fake CLI uses a POSIX shebang")
class ModelFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cli = Path(self.tmp.name) / "codex"
        self.cli.write_text(FAKE_CODEX)
        self.cli.chmod(self.cli.stat().st_mode | stat.S_IEXEC)
        self.log = Path(self.tmp.name) / "log.txt"
        os.environ["FAKE_LOG"] = str(self.log)
        client.pop_model_notice()

    def tearDown(self):
        client.set_model_fallbacks([])
        self.tmp.cleanup()

    def test_detects_rejection(self):
        self.assertTrue(client.is_model_rejection("The 'x' model is not supported when using Codex with a ChatGPT account."))
        self.assertFalse(client.is_model_rejection("usage_limit_reached"))

    def test_candidates_after_rejected_then_default(self):
        cands = client.model_fallback_candidates("gpt-6.1-sol", LIVE + ["chatgpt-web/gpt-5.6"])
        self.assertEqual(cands, ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna", ""])

    def test_rejected_model_falls_back_and_notices(self):
        client.set_model_fallbacks(LIVE)
        text = client.polish_prompt(system_prompt="s", user_text="u", image_paths=[], cwd=self.tmp.name,
                                    explicit_path=str(self.cli), model="gpt-6.1-sol", timeout=30)
        self.assertEqual(text, "POLISHED with gpt-6-sol")
        self.assertEqual(self.log.read_text().split(), ["gpt-6.1-sol", "gpt-6-astra", "gpt-6-sol"])
        notice = client.pop_model_notice()
        self.assertIn("gpt-6.1-sol", notice)
        self.assertIn("更新 Codex CLI", notice)
        self.assertEqual(client.pop_model_notice(), "")

    def test_accepted_model_no_notice(self):
        client.set_model_fallbacks(LIVE)
        text = client.polish_prompt(system_prompt="s", user_text="u", image_paths=[], cwd=self.tmp.name,
                                    explicit_path=str(self.cli), model="gpt-5.6-sol", timeout=30)
        self.assertEqual(text, "POLISHED with gpt-5.6-sol")
        self.assertEqual(client.pop_model_notice(), "")


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _window(scene, *area_types):
    areas = [_Obj(type=t, regions=[_Obj(type="HEADER"), _Obj(type="WINDOW"), _Obj(type="UI")]) for t in area_types]
    return _Obj(scene=scene, screen=_Obj(areas=areas))


class ChainTests(unittest.TestCase):
    def _runner(self, invoke, windows, locked=lambda: False, report="", **kw):
        fails = []
        runner = chain.ChainRunner(invoke=invoke, contexts=lambda: chain.candidate_contexts(windows, "S"),
                                   locked=locked, on_fail=fails.append, last_report=lambda: report, **kw)
        return runner, fails

    def test_contexts_prefer_scene_window_view3d_ui_then_window(self):
        other, mine = _window("T", "VIEW_3D"), _window("S", "PROPERTIES", "VIEW_3D")
        ctxs = chain.candidate_contexts([other, mine], "S")
        self.assertIs(ctxs[0]["window"], mine)
        self.assertEqual(ctxs[0]["region"].type, "UI")
        self.assertEqual(ctxs[1]["region"].type, "WINDOW")
        self.assertEqual(ctxs[-2:], [{"window": mine}, {"window": other}])

    def test_waits_for_lock_then_starts(self):
        state = {"locked": True}
        calls = []
        runner, fails = self._runner(lambda ctx: calls.append(ctx) or {"RUNNING_MODAL"},
                                     [_window("S", "VIEW_3D")], locked=lambda: state["locked"])
        self.assertEqual(runner.step(), runner.poll_interval)
        state["locked"] = False
        self.assertIsNone(runner.step())
        self.assertEqual(runner.outcome, "started")
        self.assertEqual(len(calls), 1)
        self.assertEqual(fails, [])

    def test_context_error_tries_next_context(self):
        calls = []

        def invoke(ctx):
            calls.append(ctx)
            if ctx.get("region") is not None and ctx["region"].type == "UI":
                raise RuntimeError("Operator bpy.ops.wondful.ai_render.poll() failed, context is incorrect")
            return {"RUNNING_MODAL"}
        runner, fails = self._runner(invoke, [_window("S", "VIEW_3D")])
        self.assertIsNone(runner.step())
        self.assertEqual(runner.outcome, "started")
        self.assertEqual(len(calls), 2)

    def test_cancelled_precheck_is_shown_not_silent(self):
        runner, fails = self._runner(lambda ctx: {"CANCELLED"}, [_window("S", "VIEW_3D")],
                                     report="输出目录不可用：x")
        self.assertIsNone(runner.step())
        self.assertEqual(runner.outcome, "failed")
        self.assertEqual(fails, ["输出目录不可用：x"])

    def test_error_report_exception_fails_once_without_retry(self):
        calls = []

        def invoke(ctx):
            calls.append(ctx)
            raise chain.OperatorRefused("输出目录不可用：x")
        runner, fails = self._runner(invoke, [_window("S", "VIEW_3D")])
        self.assertIsNone(runner.step())
        self.assertEqual(len(calls), 1)
        self.assertEqual(fails, ["输出目录不可用：x"])

    def test_no_window_retries_then_fails_clearly(self):
        runner, fails = self._runner(lambda ctx: {"RUNNING_MODAL"}, [], retries=3)
        self.assertEqual(runner.step(), runner.retry_interval)
        self.assertEqual(runner.step(), runner.retry_interval)
        self.assertIsNone(runner.step())
        self.assertEqual(len(fails), 1)
        self.assertIn("窗口", fails[0])

    def test_transient_error_then_success(self):
        state = {"n": 0}

        def invoke(ctx):
            state["n"] += 1
            if state["n"] <= 2:
                raise RuntimeError("context is incorrect")
            return {"RUNNING_MODAL"}
        runner, fails = self._runner(invoke, [_window("S", "PROPERTIES")])  # window-only ctx
        self.assertEqual(runner.step(), runner.retry_interval)
        self.assertEqual(runner.step(), runner.retry_interval)
        self.assertIsNone(runner.step())
        self.assertEqual(runner.outcome, "started")
        self.assertEqual(fails, [])

    def test_lock_never_released_times_out(self):
        clock = {"t": 0.0}
        runner, fails = self._runner(lambda ctx: {"RUNNING_MODAL"}, [_window("S", "VIEW_3D")],
                                     locked=lambda: True, lock_wait=5, clock=lambda: clock["t"])
        self.assertEqual(runner.step(), runner.poll_interval)
        clock["t"] = 6
        self.assertIsNone(runner.step())
        self.assertEqual(len(fails), 1)

    def test_operator_wiring(self):
        ops = (PKG / "operators.py").read_text()
        self.assertIn('_schedule_operator("wondful.ai_render", self._origin_scene)', ops)
        self.assertIn("ChainRunner(", ops)
        # the old fire-and-forget fallback (EXEC without a window) is gone
        self.assertNotIn('getattr(getattr(bpy.ops, module), name)()\n', ops)
        ast.parse(ops)


class CountTests(unittest.TestCase):
    def test_count_property_and_ui(self):
        props = (PKG / "properties.py").read_text()
        self.assertIn("render_count: EnumProperty(", props)
        self.assertIn('default="4"', props[props.index("render_count: EnumProperty("):])
        ui = (PKG / "ui.py").read_text()
        i_count = ui.index('count_row.prop(props, "render_count", expand=True)')
        i_go = ui.index('go.operator("wondful.generate"')
        self.assertLess(i_count, i_go)
        self.assertIn('render_variant_count("antigravity" if active_is_antigravity else "codex", props)', ui)

    def test_render_variant_count_logic(self):
        ops = (PKG / "operators.py").read_text()
        start = ops.index("RENDER_VARIANT_COUNT = 4")
        end = ops.index("_ACTIVE_LOCK = threading.Lock()")
        ns = {}
        exec(ops[start:end], ns)
        rvc = ns["render_variant_count"]
        for n in (1, 2, 3, 4):
            self.assertEqual(rvc("codex", _Obj(render_count=str(n))), n)
        self.assertEqual(rvc("antigravity", _Obj(render_count="4")), 2)
        self.assertEqual(rvc("antigravity", _Obj(render_count="1")), 1)
        self.assertEqual(rvc("codex"), 4)
        self.assertEqual(rvc("antigravity"), 2)
        self.assertIn("render_variant_count(provider_id, props)", ops)
        self.assertIn('render_variant_count(provider["id"], props)', ops)


if __name__ == "__main__":
    unittest.main()
