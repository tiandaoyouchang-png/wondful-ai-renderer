"""Operator for mode 2 「按参考风格渲染」: the user's car and camera, the reference's style.

Main thread: inputs check, product capture (flat pass + lit pass) and the style canvas.
Worker thread: the image edits (in parallel per variant) and fit / registration / detail check.
A modal timer collects the results, exports them and shows the best one.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

import bpy
from bpy.types import Operator

VARIANTS = 2


def style_target(props) -> bool:
    """True when 参考图复刻 should render the user's camera in the reference's style."""
    from . import workflow
    return (workflow.is_reference(props) and not bool(getattr(props, "workflow_multiview", False))
            and getattr(props, "reference_target", "SOURCE") == "CAMERA")


def ready(context, props) -> str:
    """'' when mode 2 can start, else the reason (shown under the button)."""
    from .operators import _analysis_provider
    prefs = context.preferences.addons[__package__].preferences
    provider = _analysis_provider(props, prefs)
    if provider["id"] != "codex":
        return "按参考风格渲染需要 Codex"
    if props.codex_account_state != "LOGGED_IN":
        return "未连接账号"
    if not provider["imagegen_enabled"]:
        return "生图未启用"
    if not props.sb_image_path or not Path(bpy.path.abspath(props.sb_image_path)).is_file():
        return "未选择参考图"
    coll = props.product_collection
    if coll is None or not any(o.type in {"MESH", "CURVE", "SURFACE", "FONT", "META"} and not o.hide_render
                               for o in coll.all_objects):
        return "未选择产品集合"
    if context.scene.camera is None:
        return "未选择相机"
    return ""


class WONDFUL_OT_style_render(Operator):
    bl_idname = "wondful.style_render"
    bl_label = "按参考风格渲染"
    bl_description = "用当前相机和车模渲染，环境、光线和调色按参考图的风格生成（不复刻参考图构图）"

    _timer = None
    _thread = None

    def execute(self, context):
        from .operators import _status, _analysis_provider
        from .render_session import create_session
        from . import product_projection as PP
        from . import style_render as SR
        from . import workflow
        from .image_manager import image_filepath
        from .viewport_capture import camera_output_dimensions
        props = context.scene.wondful_ai
        reason = ready(context, props)
        if reason:
            self.report({"ERROR"}, reason)
            return {"CANCELLED"}
        prefs = context.preferences.addons[__package__].preferences
        provider = _analysis_provider(props, prefs)
        reference = bpy.path.abspath(props.sb_image_path)
        identity = workflow.identity_path(props, resolve=bpy.path.abspath, packed_export=image_filepath,
                                          strict_source=True) or ""
        session = create_session()
        self._session = session
        self._started = time.perf_counter()
        _status(props, "RENDERING")
        props.task_phase = "渲染车模光照通道"
        props.progress = 0.05
        try:
            width, height = camera_output_dimensions(context)
            record = PP.capture(context, props, str(session.directory), (width, height), source_path=reference)
            helpers = {k: v["path"] for k, v in record["helpers"].items()}
            horizon = SR.horizon_fraction(context.scene, context.scene.camera)
            canvas = SR.build_style_canvas(reference, record["path"], helpers["detail"],
                                           session.directory / "style_canvas.png", horizon)
        except Exception as exc:
            _status(props, "ERROR", f"车模光照通道失败：{exc}")
            self.report({"ERROR"}, props.last_error)
            return {"CANCELLED"}
        from .output_manager import resolve_output_directory
        configured = (props.output_directory or "").strip() or (prefs.output_directory or "").strip()
        self._out_dir = resolve_output_directory(context, configured)
        self._scene_name = context.scene.name
        self._camera_name = context.scene.camera.name
        self._meta = {"mode": "STYLE_RENDER", "version": SR.STYLE_RENDER_VERSION, "reference": reference,
                      "identity": identity, "camera": self._camera_name, "size": [width, height],
                      "horizon": canvas["horizon"], "environment": canvas["environment"]}
        self._results = {}
        self._cancel = threading.Event()
        fn = SR.codex_style_fn(model=provider["model"],
                               client_version=str(getattr(props, "codex_cli_version", "") or ""),
                               timeout=int(provider["render_timeout"]), log_dir=str(session.directory),
                               cancel_check=self._cancel.is_set)
        border = SR.codex_border_fn(model=provider["model"],
                                    client_version=str(getattr(props, "codex_cli_version", "") or ""),
                                    timeout=int(provider["render_timeout"]), log_dir=str(session.directory),
                                    cancel_check=self._cancel.is_set)

        def run(index):
            try:
                raw = fn(canvas["path"], reference, identity, str(session.directory / f"style_raw_{index:02d}.png"))
                self._results[index] = SR.finish_variant(raw, index, helpers, width, height, session.directory,
                                                         cancel_check=self._cancel.is_set, border_fn=border)
            except Exception as exc:
                self._results[index] = {"error": str(exc)[:300], "trace": traceback.format_exc()[-1500:]}

        self._threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(1, VARIANTS + 1)]
        for t in self._threads:
            t.start()
        props.task_phase = f"按参考风格生成 {VARIANTS} 张"
        props.progress = 0.15
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        props = context.scene.wondful_ai
        if props.status != "RENDERING":       # 取消任务 sets another status
            self._cancel.set()
            return self._end(context)
        done = sum(1 for t in self._threads if not t.is_alive())
        elapsed = time.perf_counter() - self._started
        props.progress = min(0.95, 0.15 + 0.8 * (done / len(self._threads)) * 0.5 + min(0.4, elapsed / 600.0))
        if done < len(self._threads):
            return {"PASS_THROUGH"}
        self._collect(context, props)
        return self._end(context)

    def _collect(self, context, props):
        from .operators import _status
        from .output_manager import export_result
        from .image_manager import load_image
        from . import style_render as SR
        ok = {i: r for i, r in self._results.items() if r.get("final")}
        errors = [r["error"] for r in self._results.values() if r.get("error")]
        self._meta["variants"] = self._results
        try:
            (self._session.directory / "style_render.json").write_text(
                json.dumps(self._meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        except Exception:
            pass
        if not ok:
            _status(props, "ERROR", "按参考风格渲染失败：" + ("；".join(errors)[:300] or "没有结果"))
            return
        exported = {}
        for i in sorted(ok):
            try:
                exported[i] = export_result(ok[i]["final"], self._out_dir, self._scene_name,
                                            self._camera_name, f"参考风格_{self._session.session_id}")
            except Exception as exc:
                errors.append(str(exc))
        if not exported:
            _status(props, "ERROR", "结果导出失败：" + "；".join(errors)[:300])
            return
        best = max(exported, key=lambda i: SR.rank(ok[i]))
        paths = [str(exported[i]) for i in sorted(exported)]
        props.variant_count = len(paths)
        props.last_export_path = props.last_result_path = str(exported[best])
        props.last_export_paths = json.dumps(paths, ensure_ascii=False)
        notes = [f"第 {i} 张：{ok[i].get('note') or ''}" for i in sorted(ok)]
        props.alignment_message = "；".join(notes)[:500]
        try:
            props.result_image = load_image(str(exported[best]), "Wondful_Result")
            props.result_workflow = "REFERENCE"
        except Exception:
            pass
        _status(props, "SUCCESS")
        props.progress = 1.0

    def _end(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        props = context.scene.wondful_ai
        props.task_phase = ""
        for area in context.screen.areas if context.screen else ():
            area.tag_redraw()
        return {"FINISHED"}


CLASSES = (WONDFUL_OT_style_render,)
