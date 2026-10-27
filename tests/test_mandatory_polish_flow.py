"""Behavioral guards for the mandatory Generate -> polish -> render chain.

The operators are AST-extracted into fake Blender RNA contexts.  No provider,
Blender process, or network request is started here.
"""
import ast
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_two_workflows as T


NS = types.SimpleNamespace
ROOT = T.ROOT
PKG = T.PKG


class Base(T.Operator):
    def _cleanup(self, context):
        self.cleaned = True


def extract():
    tree = ast.parse((PKG / "operators.py").read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"generate_needs_polish", "_camera_signature", "workflow_preflight",
                                  "identity_preflight", "_sb_reference", "_sb_generation_reference"}
                or isinstance(node, ast.ClassDef)
                and node.name in {"WONDFUL_OT_generate", "WONDFUL_OT_polish_prompt"}]
    env = {
        "__package__": T.PACKAGE,
        "Operator": T.Operator,
        "_BaseAsyncOperator": Base,
        "workflow": T.WF,
        "Path": Path,
        "bpy": T.bpy,
        "time": __import__("time"),
        "hashlib": __import__("hashlib"),
        "_load_reference_image": lambda path: NS(size=(160, 90)),
        "image_filepath": lambda image: image.filepath,
        "AntigravityNotLoggedInError": type("AntigravityNotLoggedInError", (Exception,), {}),
        "AntigravityNotInstalledError": type("AntigravityNotInstalledError", (Exception,), {}),
        "CodexNotLoggedInError": type("CodexNotLoggedInError", (Exception,), {}),
        "CodexNotInstalledError": type("CodexNotInstalledError", (Exception,), {}),
        "REFERENCE_POLICY_VERSION": 1,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(PKG / "operators.py"), "exec"), env)
    return env


OPS = extract()


def make_props(mode, quality, prompt):
    props = T.props(workflow_mode=mode, render_mode=quality, prompt=prompt,
                    scene_brief="brief", scene_brief_used="brief",
                    auto_polish_on_generate=False, two_step=True,
                    pending_generate=False, views_plan="plan", status="IDLE")
    props.last_ai_prompt = prompt if prompt == "cached prompt" else ""
    return props


def setup_generate_env(calls):
    T.bpy.ops = NS(wondful=NS(
        polish_prompt=lambda *args: calls.append("polish") or {"RUNNING_MODAL"},
        ai_render=lambda *args: calls.append("render") or {"RUNNING_MODAL"},
    ))
    OPS.update(
        workflow_preflight=lambda context, props: "",
        _addon_prefs=lambda context: NS(),
        _status=lambda props, state, message="": (setattr(props, "status", state), setattr(props, "last_error", message)),
        _LAST_REPORT={},
    )


class MandatoryGenerateTests(unittest.TestCase):
    def setUp(self):
        env_patch = patch.dict(OPS)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_all_modes_and_quality_levels_start_polish_once_for_every_prompt_state(self):
        for mode in (T.WF.ENVIRONMENT, T.WF.REFERENCE):
            for quality in ("FAST", "NORMAL", "STRICT"):
                for prompt in ("manual prompt", "cached prompt", ""):
                    with self.subTest(mode=mode, quality=quality, prompt=prompt or "empty"):
                        calls = []
                        setup_generate_env(calls)
                        context = T.context(make_props(mode, quality, prompt))
                        op = OPS["WONDFUL_OT_generate"]()
                        op._plan_views = lambda *_args: True
                        self.assertEqual(op.execute(context), {"FINISHED"})
                        self.assertEqual(calls, ["polish"])
                        self.assertTrue(context.scene.wondful_ai.pending_generate)

    def test_polish_refusal_stops_before_low_level_render(self):
        calls = []
        setup_generate_env(calls)
        T.bpy.ops.wondful.polish_prompt = lambda *_args: calls.append("polish") or {"CANCELLED"}
        context = T.context(make_props(T.WF.ENVIRONMENT, "NORMAL", "manual"))
        op = OPS["WONDFUL_OT_generate"]()
        op._plan_views = lambda *_args: True
        self.assertEqual(op.execute(context), {"CANCELLED"})
        self.assertEqual(calls, ["polish"])
        self.assertFalse(context.scene.wondful_ai.pending_generate)


class PolishModalTests(unittest.TestCase):
    def setUp(self):
        env_patch = patch.dict(OPS)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.scheduled = []
        self.sanitize_flags = []
        self.texts = []
        T.bpy.data.texts = NS(new=lambda name: self.texts.append(NS(name=name, write=lambda value: None)) or self.texts[-1])
        self._file_hash = T.WF.file_hash
        self.addCleanup(setattr, T.WF, "file_hash", self._file_hash)
        OPS.update(
            _status=lambda props, state, message="": (setattr(props, "status", state), setattr(props, "last_error", message)),
            sanitize_appearance_prompt_report=lambda text, preserve_visual_intent=False: (self.sanitize_flags.append(preserve_visual_intent) or text, []),
            _get_prompt_text=lambda props: props.prompt,
            _set_prompt_text=lambda props, text: setattr(props, "prompt", text),
            _schedule_operator=lambda name, scene: self.scheduled.append((name, scene)),
            record_timing=lambda *args: None,
            save_metadata=lambda *args: None,
            _reference_fingerprint=lambda paths: "stable",
            _reference_paths=lambda props: {"style": []},
        )

    def modal_op(self, mode=T.WF.ENVIRONMENT, prompt="manual"):
        props = make_props(mode, "NORMAL", prompt)
        context = T.context(props)
        scene = context.scene
        op = OPS["WONDFUL_OT_polish_prompt"]()
        op._origin_scene = scene
        op._thread = NS(is_alive=lambda: False)
        op._error = None
        op._result = "polished with low canted tracking motion, dust, and canyon depth"
        op._started_at = 0
        op._provider_id = "codex"
        op._provider_label = "Codex"
        op._workflow_mode_at_start = mode
        op._reference_mode_at_start = mode == T.WF.REFERENCE
        op._prompt_at_start = prompt
        op._prompt_profile = {"target_key": "test"}
        op._session = NS(directory=Path(self.tmp.name), prompt_file=Path(self.tmp.name) / "prompt.txt")
        op._camera_output_size = (1600, 900)
        op._camera_name_at_start = "camera"
        op._reference_counts = {}
        op._reference_sources_at_start = {}
        op._reference_instructions_at_start = {}
        op._style_cache_key = ""
        op._style_summary = ""
        op._style_cache_hit = False
        op._jev_appearance_meta = {}
        op._appearance_meta = {}
        op._timing_key = "codex_polish"
        op._width, op._height = 1600, 900
        return op, context, props

    def test_success_schedules_one_render_even_with_legacy_two_step(self):
        op, context, props = self.modal_op()
        props.pending_generate = True
        self.assertEqual(op._modal(context, NS(type="TIMER")), {"FINISHED"})
        self.assertEqual(self.scheduled, [("wondful.ai_render", context.scene)])
        self.assertFalse(props.pending_generate)
        self.assertEqual(self.sanitize_flags, [False])

    def test_provider_error_stops_chain(self):
        op, context, props = self.modal_op()
        props.pending_generate = True
        op._error = (RuntimeError("provider refused"), "trace")
        self.assertEqual(op._modal(context, NS(type="TIMER")), {"CANCELLED"})
        self.assertEqual(self.scheduled, [])
        self.assertFalse(props.pending_generate)

    def test_manual_edit_stops_chain_without_overwriting_user_text(self):
        op, context, props = self.modal_op(prompt="before")
        props.pending_generate = True
        props.prompt = "user edit during polish"
        self.assertEqual(op._modal(context, NS(type="TIMER")), {"FINISHED"})
        self.assertEqual(props.prompt, "user edit during polish")
        self.assertEqual(self.scheduled, [])

    def test_reference_target_or_camera_change_stops_chain_and_keeps_visual_intent(self):
        op, context, props = self.modal_op(mode=T.WF.REFERENCE)
        props.pending_generate = True
        target = Path(self.tmp.name) / "target.png"
        preview = Path(self.tmp.name) / "white.png"
        target.write_bytes(b"target")
        preview.write_bytes(b"preview")
        OPS.update(
            _sb_reference=lambda props: (str(target), ""),
            _sb_generation_reference=lambda context, props: (str(preview), ""),
        )
        op._replica_target_hash_at_start = "different hash"
        op._replica_preview_source_at_start = str(preview.resolve())
        op._replica_snapshot_hash_at_start = "current hash"
        op._replica_camera_at_start = OPS["_camera_signature"](context.scene.camera, context.scene)
        T.WF.file_hash = lambda path: "current hash"
        self.assertEqual(op._modal(context, NS(type="TIMER")), {"FINISHED"})
        self.assertEqual(self.scheduled, [])
        self.assertEqual(self.sanitize_flags, [True])

    def test_reference_camera_change_stops_chain(self):
        op, context, props = self.modal_op(mode=T.WF.REFERENCE)
        props.pending_generate = True
        target = Path(self.tmp.name) / "target.png"
        preview = Path(self.tmp.name) / "white.png"
        target.write_bytes(b"target")
        preview.write_bytes(b"preview")
        OPS.update(
            _sb_reference=lambda props: (str(target), ""),
            _sb_generation_reference=lambda context, props: (str(preview), ""),
        )
        T.WF.file_hash = lambda path: "same hash"
        op._replica_target_hash_at_start = "same hash"
        op._replica_snapshot_hash_at_start = "same hash"
        op._replica_preview_source_at_start = str(preview.resolve())
        op._replica_camera_at_start = OPS["_camera_signature"](context.scene.camera, context.scene)
        context.scene.camera.data.lens += 5
        self.assertEqual(op._modal(context, NS(type="TIMER")), {"FINISHED"})
        self.assertEqual(self.scheduled, [])

    def test_matching_reference_pair_completes_and_schedules_one_render(self):
        op, context, props = self.modal_op(mode=T.WF.REFERENCE)
        props.pending_generate = True
        target = Path(self.tmp.name) / "target.png"
        snapshot = Path(self.tmp.name) / "snapshot.png"
        target.write_bytes(b"original target")
        snapshot.write_bytes(b"resized snapshot")
        OPS.update(_sb_reference=lambda props: (str(target), ""),
                   _sb_generation_reference=lambda context, props: (str(snapshot), ""))
        op._replica_target_hash_at_start = T.WF.file_hash(target)
        op._replica_snapshot_hash_at_start = T.WF.file_hash(snapshot)
        op._replica_preview_source_at_start = str(snapshot.resolve())
        op._replica_camera_at_start = OPS["_camera_signature"](context.scene.camera, context.scene)
        self.assertEqual(op._modal(context, NS(type="TIMER")), {"FINISHED"})
        self.assertEqual(self.scheduled, [("wondful.ai_render", context.scene)])
        self.assertEqual(self.sanitize_flags, [True])
        self.assertFalse(props.pending_generate)


class ReplicaExecuteProvenanceTests(unittest.TestCase):
    """The uploaded target is the original, not a resized blockout snapshot."""
    def setUp(self):
        env_patch = patch.dict(OPS)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        bpy_patch = patch.dict(sys.modules, {"bpy": T.bpy})
        bpy_patch.start()
        self.addCleanup(bpy_patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls = []
        T.stub("jev_semantics", analyze_appearance=lambda *args: {})

    def _configure(self, original, snapshot, preview, *, changed=False):
        def render_preview(context, props, output):
            if changed:
                original.write_bytes(b"changed")
            return str(preview)

        provider = {
            "discover": lambda: True, "id": "codex", "label": "Codex", "model": "test",
            "cli_path": "", "polish_timeout": 1,
            "polish": lambda **kwargs: self.calls.append(kwargs) or "polished",
        }
        OPS.update(
            _addon_prefs=lambda context: NS(),
            _analysis_provider=lambda props, prefs: provider,
            cleanup_old_sessions=lambda: None,
            create_session=lambda: NS(directory=Path(self.tmp.name), session_id="session"),
            _reference_paths=lambda props: {"product": [str(Path(__file__))], "person": [], "style": []},
            _reference_instructions=lambda props: {"product": ["keep YU7"], "person": [], "style": []},
            copy_references=lambda paths, session, role: list(paths),
            _render_replica_preview=render_preview,
            prepare_reference_for_upload=lambda path: "upload:" + Path(path).name,
            camera_output_dimensions=lambda context: (1600, 900),
            _reference_manifest=lambda paths: [],
            _reference_fingerprint=lambda paths: "",
            needs_reference_policy_refresh=lambda props: False,
            prompt_profiles=NS(needs_target_refresh=lambda props, prefs: False,
                               metadata=lambda *args: {"target_key": "test"},
                               style_cache_key=lambda *args: "", system_prompt=lambda text, provider: text),
            _get_prompt_text=lambda props: props.prompt,
            select_polish_source=lambda props, prompt, refresh: prompt,
            _appearance_context=lambda props: ("PRODUCT_REF", "", []),
            _product_autofill_key=lambda *args: "",
            _product_autofill_needed=lambda *args: False,
            appearance_lock_material_text=lambda props: "",
            build_polish_user_text=lambda *args, **kwargs: "request",
            _status=lambda *args: None,
            average_timing=lambda *args: 1.0,
            BASE_SYSTEM_PROMPT="base", REPLICA_POLISH_SYSTEM_PROMPT="replica",
        )
        self.original, self.snapshot = original, snapshot

    def _context(self):
        props = T.props(workflow_mode=T.WF.REFERENCE, prompt="low canted tracking shot",
                        sb_image_path=str(self.original), sb_built_target_hash=T.WF.file_hash(self.original))
        context = T.context(props)
        block = T.Collection("Owned", [T.Item("shell")], tags={"wondful_scene_blockout": 1,
            "wondful_sb_reference_image": str(self.snapshot),
            "wondful_sb_reference_sha256": T.WF.file_hash(self.snapshot),
            "wondful_sb_original_sha256": T.WF.file_hash(self.original)})
        context.scene.collection.children_recursive.append(block)
        props.massing_collection = block
        return context, props

    def test_original_target_and_fresh_white_preview_are_accepted_despite_different_bytes(self):
        original = Path(self.tmp.name) / "original-target.png"
        snapshot = Path(self.tmp.name) / "prepared-snapshot.png"
        preview = Path(self.tmp.name) / "fresh-white-preview.png"
        for path, data in ((original, b"original"), (snapshot, b"resized"), (preview, b"white")):
            path.write_bytes(data)
        self._configure(original, snapshot, preview)
        context, _props = self._context()
        op = OPS["WONDFUL_OT_polish_prompt"]()
        op._start_thread = lambda context, job: (job(), True)[1]
        self.assertEqual(op.execute(context), {"RUNNING_MODAL"})
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["image_paths"], [
            "upload:original-target.png", "upload:fresh-white-preview.png", "upload:test_mandatory_polish_flow.py",
        ])

    def test_original_upload_stays_in_session_and_provider_reads_stable_copy(self):
        user_dir = Path(self.tmp.name) / "user"
        session_dir = Path(self.tmp.name) / "session"
        user_dir.mkdir()
        session_dir.mkdir()
        original = user_dir / "original.png"
        snapshot = session_dir / "snapshot.png"
        preview = session_dir / "white.png"
        for path, data in ((original, b"original bytes"), (snapshot, b"resized"), (preview, b"white")):
            path.write_bytes(data)
        self._configure(original, snapshot, preview)
        original_hash = T.WF.file_hash(original)

        def copy_refs(paths, session, role):
            copied = []
            for index, source in enumerate(paths):
                dest = session.directory / f"{role}_{index}{Path(source).suffix}"
                dest.write_bytes(Path(source).read_bytes())
                copied.append(str(dest))
            return copied

        def prepare_upload(path):
            source = Path(path)
            upload = source.with_name(source.stem + "_upload.png")
            upload.write_bytes(source.read_bytes())
            return str(upload)

        OPS.update(create_session=lambda: NS(directory=session_dir, session_id="session"),
                   copy_references=copy_refs, prepare_reference_for_upload=prepare_upload,
                   _reference_paths=lambda props: {"product": [__file__], "person": [], "style": [str(original)]},
                   _reference_manifest=lambda paths: list(paths),
                   _reference_fingerprint=lambda paths: "|".join(T.WF.file_hash(path) for path in paths))
        context, props = self._context()
        op = OPS["WONDFUL_OT_polish_prompt"]()
        pending = []
        op._start_thread = lambda context, job: pending.append(job) or True
        self.assertEqual(op.execute(context), {"RUNNING_MODAL"})
        self.assertEqual(op._reference_sources_at_start["style"], [str(original)])
        self.assertEqual(op._style_fingerprint_at_start, original_hash)
        self.assertEqual(op._replica_target_hash_at_start, original_hash)
        self.assertEqual(props.sb_image_path, str(original))
        self.assertFalse(original.with_name("original_upload.png").exists())
        original.write_bytes(b"replaced after capture")
        pending[0]()
        uploaded = Path(self.calls[0]["image_paths"][0])
        self.assertEqual(uploaded, session_dir / "replica_original_0_upload.png")
        self.assertEqual(uploaded.read_bytes(), b"original bytes")
        self.assertEqual((session_dir / "replica_original_0.png").read_bytes(), b"original bytes")

    def test_changed_original_refuses_before_provider_submission(self):
        original = Path(self.tmp.name) / "original-target.png"
        snapshot = Path(self.tmp.name) / "prepared-snapshot.png"
        preview = Path(self.tmp.name) / "fresh-white-preview.png"
        for path, data in ((original, b"original"), (snapshot, b"resized"), (preview, b"white")):
            path.write_bytes(data)
        self._configure(original, snapshot, preview, changed=True)
        context, _props = self._context()
        op = OPS["WONDFUL_OT_polish_prompt"]()
        op._start_thread = lambda context, job: self.fail("changed original must not start provider polish")
        with self.assertRaises(RuntimeError):
            op.execute(context)
        self.assertEqual(self.calls, [])

    def test_mismatched_blockout_provenance_refuses_before_capture_or_provider(self):
        original = Path(self.tmp.name) / "original-target.png"
        snapshot = Path(self.tmp.name) / "prepared-snapshot.png"
        preview = Path(self.tmp.name) / "fresh-white-preview.png"
        original.write_bytes(b"original")
        snapshot.write_bytes(b"prepared")
        self._configure(original, snapshot, preview)
        context, props = self._context()
        props.massing_collection.tags["wondful_sb_original_sha256"] = "another original"
        OPS["_render_replica_preview"] = lambda *args: self.fail("invalid provenance must not capture a preview")
        op = OPS["WONDFUL_OT_polish_prompt"]()
        op._start_thread = lambda *args: self.fail("invalid provenance must not start polish")
        self.assertEqual(op.execute(context), {"CANCELLED"})
        self.assertEqual(self.calls, [])


class BlockoutCameraRestoreTests(unittest.TestCase):
    def test_failure_and_cancellation_restore_only_origin_including_no_previous_camera(self):
        env = T.function_namespace(classes=("WONDFUL_OT_scene_blockout_build",))
        env.update(_status=lambda *args: None, _redraw_all=lambda: None)
        T.load("sb_progress")
        for action in ("failure", "cancel"):
            for previous in (None, NS(name="Original camera")):
                with self.subTest(action=action, previous=previous):
                    props = T.props()
                    origin = NS(camera=NS(name="Unfinished camera"))
                    foreign_camera = NS(name="Other scene camera")
                    context = NS(scene=NS(camera=foreign_camera))
                    op = env["WONDFUL_OT_scene_blockout_build"]()
                    op._origin_scene = origin
                    op._sb = {"original_camera": previous, "t0": 0, "stage_id": "BUILD"}
                    op._cleanup = lambda context: None
                    if action == "failure":
                        self.assertEqual(op._fail(context, props, RuntimeError("build failed")), {"CANCELLED"})
                    else:
                        op._on_cancelled(props)
                    self.assertIs(origin.camera, previous)
                    self.assertIs(context.scene.camera, foreign_camera)


if __name__ == "__main__":
    unittest.main()
