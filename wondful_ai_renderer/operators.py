# RNA Property declarations must evaluate at class creation (Blender 4.3+).

import hashlib
import shutil
import threading
import uuid
import time
import traceback
from pathlib import Path

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from .codex_client import (
    pop_model_notice as codex_pop_model_notice,
    set_model_fallbacks as codex_set_model_fallbacks,
    CodexImageGenerationUnavailable,
    CodexNotInstalledError,
    CodexNotLoggedInError,
    discover_codex_cli,
    generate_image as codex_generate_image,
    audit_alignment as codex_audit_alignment,
    get_account_status,
    login_chatgpt,
    logout as codex_logout,
    polish_prompt as codex_polish_prompt,
)
from .antigravity_client import (
    AntigravityError,
    AntigravityImageGenerationUnavailable,
    AntigravityNotInstalledError,
    AntigravityNotLoggedInError,
    audit_alignment as antigravity_audit_alignment,
    discover_antigravity_cli,
    generate_image as antigravity_generate_image,
    get_account_status as get_antigravity_account_status,
    open_google_login as open_antigravity_google_login,
    polish_prompt as antigravity_polish_prompt,
)
from .clipboard_utils import ClipboardImageError, save_clipboard_image
from .config_manager import CONFIG_DIR, average_timing, record_timing
from .composition import image_dimensions, assess_canvas, assess_audit, repair_region, write_repair_mask
from .image_manager import compute_target_size, enforce_aspect_ratio, image_filepath, load_image, prepare_reference_for_upload
from .prompt_engine import BASE_SYSTEM_PROMPT, STYLE_ANALYSIS_SYSTEM_PROMPT, PRODUCT_ANALYSIS_SYSTEM_PROMPT, product_analysis_user_text, parse_product_analysis, build_polish_user_text, build_render_prompt, sanitize_appearance_prompt, sanitize_appearance_prompt_report, needs_reference_policy_refresh, REFERENCE_POLICY_VERSION, select_polish_source
from .render_session import cleanup_old_sessions, copy_references, create_session, save_metadata
from .output_manager import export_result, resolve_output_directory, preflight_output_directory
from .viewport_capture import capture_camera_reference, camera_output_dimensions
from .structure_packet import build_structure_packet
from .properties import MAX_REFERENCES_PER_KIND, MAX_PRODUCT_REFERENCES, reference_limit
from . import prompt_profiles
from .reference_bundle import MAX_ANTIGRAVITY_IMAGE_PATHS, MAX_IMAGE_PATHS, prepare_bundle
from .model_catalog import discover_models
from .antigravity_auth import AuthSession
from .cli_transport import safe_diagnostic
import tempfile
import json
import webbrowser
from queue import Empty, Queue
from .prompt_editor import open_prompt_in_area, close_prompt_area, sync_prompt_editors


# 1.2.2: images per generation follow the scene's 「张数」 (props.render_count, 1-4);
# these are the Codex default/maximum and the Antigravity cap.
RENDER_VARIANT_COUNT = 4
ANTIGRAVITY_RENDER_VARIANT_COUNT = 2


def provider_variant_cap(provider_id: str) -> int:
    return (ANTIGRAVITY_RENDER_VARIANT_COUNT
            if str(provider_id or "").strip().lower() == "antigravity" else RENDER_VARIANT_COUNT)


def render_variant_count(provider_id: str, props=None) -> int:
    """Images per generation: the user's 1/2/3/4 choice, capped by the provider."""
    cap = provider_variant_cap(provider_id)
    try:
        wanted = int(getattr(props, "render_count", "") or 0) if props is not None else 0
    except (TypeError, ValueError):
        wanted = 0
    if wanted <= 0:
        return cap
    return max(1, min(cap, wanted))


_ACTIVE_LOCK = threading.Lock()
_ACTIVE_OPERATOR = None
_AGY_AUTH = None
_AGY_AUTH_OWNER = None
_ASYNC_STATUSES = frozenset({"AUTHENTICATING", "REFRESHING_MODELS", "CLASSIFYING", "POLISHING", "RENDERING"})


def cancel_auth_session(*, detach=False, context=None):
    if _AGY_AUTH is not None:
        _AGY_AUTH.cancel()
    if detach and _AGY_AUTH_OWNER is not None and context is not None:
        _AGY_AUTH_OWNER._cleanup(context)



def _addon_prefs(context):
    return context.preferences.addons[__package__].preferences


def _status(props, status, error=""):
    props.status = status
    props.last_error = error
    if status not in {"POLISHING", "RENDERING"}:
        props.task_phase = ""


def _apply_model_notice(props):
    """1.2.2: surface a 推理-model fallback (old Codex CLI) in step ⑤."""
    try:
        notice = codex_pop_model_notice()
        if notice:
            props.model_notice = notice
    except Exception:
        pass


def reset_stale_task_status(props) -> bool:
    """Return a persisted in-progress state to idle after a file reload.

    Async workers and modal handlers cannot survive opening a .blend file.  The
    status property can, however, be serialized while a task is in progress;
    only clear it when the process-wide task lock is free so a genuinely active
    worker is never made available to a second operation.
    """
    try:
        stale = props.status in _ASYNC_STATUSES
        if not stale or _ACTIVE_LOCK.locked():
            return False
        _status(props, "IDLE")
        props.progress = 0.0
        props.eta_seconds = 0
        return True
    except (AttributeError, ReferenceError, TypeError):
        return False


def _apply_account_status(props, status) -> None:
    props.codex_cli_resolved = status.cli_path or ""
    props.codex_cli_version = status.version or ""
    props.codex_account_message = status.message or ""
    if not status.installed:
        props.codex_account_state = "NOT_INSTALLED"
    elif status.logged_in:
        props.codex_account_state = "LOGGED_IN"
    else:
        props.codex_account_state = "LOGGED_OUT"


def _apply_antigravity_account_status(props, status) -> None:
    props.antigravity_cli_resolved = status.cli_path or ""
    props.antigravity_cli_version = status.version or ""
    props.antigravity_account_message = status.message or ""
    if not status.installed:
        props.antigravity_account_state = "NOT_INSTALLED"
    elif status.logged_in:
        props.antigravity_account_state = "LOGGED_IN"
    else:
        props.antigravity_account_state = status.state


def _analysis_provider(props, prefs):
    # The persisted property name remains `analysis_provider` for backward compatibility,
    # but since 2.11 it is the single provider for polish + audit + final image generation.
    if props.analysis_provider == "ANTIGRAVITY":
        return {
            "id": "antigravity",
            "label": "Antigravity",
            "discover": lambda: discover_antigravity_cli(prefs.antigravity_cli_path or getattr(props, "antigravity_cli_resolved", "")),
            "cli_path": str(prefs.antigravity_cli_path or getattr(props, "antigravity_cli_resolved", "") or ""),
            "model": str(prefs.antigravity_model or ""),
            "polish": antigravity_polish_prompt,
            "audit": antigravity_audit_alignment,
            "generate": antigravity_generate_image,
            "imagegen_enabled": bool(prefs.antigravity_imagegen_enabled),
            "polish_timeout": int(prefs.antigravity_polish_timeout),
            "audit_timeout": int(prefs.antigravity_audit_timeout),
            "render_timeout": int(prefs.antigravity_render_timeout),
            "reference_limit": MAX_ANTIGRAVITY_IMAGE_PATHS,
        }
    try:  # 1.2.2: run-time fallback chain if the local CLI rejects the chosen 推理 model
        codex_set_model_fallbacks([item.name for item in getattr(props, "codex_models", [])])
    except Exception:
        pass
    return {
        "id": "codex",
        "label": "Codex",
        "discover": lambda: discover_codex_cli(prefs.codex_cli_path or getattr(props, "codex_cli_resolved", "")),
        "cli_path": str(prefs.codex_cli_path or getattr(props, "codex_cli_resolved", "") or ""),
        "model": str(prefs.codex_model or ""),
        "polish": codex_polish_prompt,
        "audit": codex_audit_alignment,
        "generate": codex_generate_image,
        "imagegen_enabled": bool(prefs.codex_imagegen_enabled),
        "polish_timeout": int(prefs.codex_polish_timeout),
        "audit_timeout": int(prefs.codex_audit_timeout),
        "render_timeout": int(prefs.codex_render_timeout),
        "reference_limit": MAX_IMAGE_PATHS,
    }


def _get_prompt_text(props):
    sync_prompt_editors(props)  # flush typing even before the next 250ms timer tick
    # 3.0.1: props.prompt is the only source of truth. Older Blender fallback
    # rows synchronize into this value without inserting soft-wrap newlines.
    return (props.prompt or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _set_prompt_text(props, text):
    # RNA callback also refreshes an open full-text editor.
    props.prompt = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _product_autofill_key(props, provider):
    paths = _reference_paths(props)["product"]
    if not paths:
        return ""
    notes = _reference_instructions(props)["product"]
    return "|".join([
        _reference_fingerprint(paths),
        (notes[0] if notes else ""),
        str(getattr(props, "color_source", "BLENDER")),
        str(provider.get("id", "")),
    ])


def _product_fields_untouched(props):
    """True when both fields are empty or still hold the last automatic values."""
    look = (getattr(props, "product_look_prompt", "") or "").strip()
    details = (getattr(props, "identity_details", "") or "").strip()
    return (look in {"", (props.product_look_auto or "").strip()}
            and details in {"", (props.identity_details_auto or "").strip()})


def _product_autofill_needed(props, key):
    if not key or not _product_fields_untouched(props):
        return False
    return key != getattr(props, "product_autofill_key", "") or not (props.product_look_prompt or props.identity_details)


def _apply_product_autofill(props, key, look, details):
    """Write analyser output unless the user typed into the fields meanwhile."""
    if not _product_fields_untouched(props):
        props.product_autofill_message = "你已手动修改外观锁定，自动结果未覆盖。"
        return False
    details_text = "，".join(details)
    props.product_look_prompt = look
    props.identity_details = details_text
    props.product_look_auto = look
    props.identity_details_auto = details_text
    props.product_autofill_key = key
    props.product_autofill_message = "已根据产品参考图自动填写，可直接修改。"
    return True


def appearance_lock_material_text(props):
    """Blender material summary for the appearance lock (main thread only)."""
    from . import appearance_lock
    if str(getattr(props, "color_source", "BLENDER") or "BLENDER") != "BLENDER":
        return ""
    collection = getattr(props, "product_collection", None)
    if collection is None:
        return ""
    objects = []
    for obj in getattr(collection, "all_objects", ()):
        if getattr(obj, "type", "") != "MESH":
            continue
        try:
            if not obj.visible_get():
                continue
        except Exception:
            pass
        objects.append(obj)
    try:
        return appearance_lock.material_summary(appearance_lock.collect_product_materials(objects))
    except Exception:
        return ""


def _appearance_context(props):
    """3.1.7: colour source, locked product look and identity checklist (main thread)."""
    from . import appearance_lock
    source = str(getattr(props, "color_source", "BLENDER") or "BLENDER")
    material_text = ""
    collection = getattr(props, "product_collection", None)
    if source == "BLENDER" and collection is not None:
        objects = []
        for obj in getattr(collection, "all_objects", ()):
            if getattr(obj, "type", "") != "MESH":
                continue
            try:
                if not obj.visible_get():
                    continue
            except Exception:
                pass
            objects.append(obj)
        try:
            material_text = appearance_lock.material_summary(appearance_lock.collect_product_materials(objects))
        except Exception:
            material_text = ""
    lock = appearance_lock.appearance_lock_block(source, material_text, getattr(props, "product_look_prompt", ""))
    items = appearance_lock.identity_checklist(
        _reference_instructions(props)["product"],
        getattr(props, "jev_identity_assets", ""),
        getattr(props, "identity_details", ""),
    )
    return source, lock, items


def _reference_paths(props):
    def paths(collection):
        result = []
        for item in collection:
            path = item.source_path.strip() if getattr(item, "source_path", "") else ""
            if not path:
                path = image_filepath(item.image) or ""
            if path:
                result.append(path)
        return result

    return {
        "product": paths(props.product_images)[:MAX_PRODUCT_REFERENCES],
        "person": paths(props.person_images),
        "style": paths(props.style_images),
    }


def _reference_instructions(props):
    """Return per-image user instructions aligned with _reference_paths ordering."""
    def notes(collection):
        result = []
        for item in collection:
            path = item.source_path.strip() if getattr(item, "source_path", "") else ""
            if not path:
                path = image_filepath(item.image) or ""
            if path:
                result.append((getattr(item, "instruction", "") or "").strip())
        return result

    return {
        "product": notes(props.product_images)[:MAX_PRODUCT_REFERENCES],
        "person": notes(props.person_images),
        "style": notes(props.style_images),
    }


def _instruction_lines(label, notes):
    rows = []
    for i, note in enumerate(notes or [], start=1):
        text = str(note or "").strip()
        if text:
            rows.append(f"{label} {i} 用户指定参考重点：{text}")
    return "\n".join(rows)


def _reference_fingerprint(paths):
    """Fingerprint the actual reference bytes, not only Blender datablock identity.

    This catches same-path file replacements and prevents a prompt generated from an
    older style bitmap from being treated as synced with the current one.
    """
    digest = hashlib.sha256()
    valid = 0
    for raw in paths or []:
        try:
            p = Path(raw).expanduser().resolve()
            digest.update(str(p).encode("utf-8", errors="replace"))
            digest.update(b"\0")
            if not p.is_file():
                digest.update(b"MISSING\0")
                continue
            valid += 1
            with p.open("rb") as fh:
                while True:
                    chunk = fh.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            digest.update(b"\0")
        except Exception:
            digest.update(str(raw).encode("utf-8", errors="replace"))
            digest.update(b"\0ERR\0")
    return f"{valid}:" + digest.hexdigest()


def _reference_manifest(paths):
    """Return lightweight diagnostics proving which files were actually submitted."""
    rows = []
    for raw in paths or []:
        item = {"path": str(raw)}
        try:
            p = Path(raw).expanduser().resolve()
            item["resolved_path"] = str(p)
            if p.is_file():
                item["size"] = int(p.stat().st_size)
                h = hashlib.sha256()
                with p.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(chunk)
                item["sha256"] = h.hexdigest()
            else:
                item["missing"] = True
        except Exception as exc:
            item["error"] = str(exc)
        rows.append(item)
    return rows


def _collection_and_index(props, ref_kind):
    mapping = {
        "PRODUCT": (props.product_images, "product_image_index"),
        "PERSON": (props.person_images, "person_image_index"),
        "STYLE": (props.style_images, "style_image_index"),
    }
    return mapping[ref_kind]


def _mark_reference_changed(props, ref_kind):
    """Invalidate style-derived prompt state when the style reference set changes."""
    if ref_kind != "STYLE":
        return
    props.style_reference_revision = int(getattr(props, "style_reference_revision", 0)) + 1
    props.style_prompt_dirty = True


def _resolve_reference_index(props, ref_kind, requested_index=-1):
    collection, index_attr = _collection_and_index(props, ref_kind)
    if not collection:
        return collection, index_attr, -1
    if requested_index is None or int(requested_index) < 0:
        idx = min(getattr(props, index_attr), len(collection) - 1)
    else:
        idx = max(0, min(int(requested_index), len(collection) - 1))
        setattr(props, index_attr, idx)
    return collection, index_attr, idx


def _load_reference_image(filepath):
    image = bpy.data.images.load(str(filepath), check_existing=True)
    # If the same path was overwritten externally, Blender may return an existing
    # datablock with stale pixels. Reload it before using it as the current ref.
    try:
        image.reload()
    except Exception:
        pass
    return image


def _append_reference_from_path(props, ref_kind, filepath):
    collection, index_attr = _collection_and_index(props, ref_kind)
    if len(collection) >= reference_limit(ref_kind):
        raise RuntimeError(f"每类参考图最多 {reference_limit(ref_kind)} 张。")
    image = _load_reference_image(filepath)
    item = collection.add()
    item.image = image
    item.source_path = str(filepath)
    setattr(props, index_attr, len(collection) - 1)
    return item


def _replace_style_set(props, filepaths):
    """Atomically replace the current style set after new images load successfully."""
    loaded = []
    failures = []
    for filepath in list(filepaths or [])[:MAX_REFERENCES_PER_KIND]:
        try:
            loaded.append((str(filepath), _load_reference_image(filepath)))
        except Exception as exc:
            failures.append(f"{Path(filepath).name}: {exc}")
    if not loaded:
        return 0, failures

    collection, index_attr = _collection_and_index(props, "STYLE")
    collection.clear()
    for filepath, image in loaded:
        item = collection.add()
        item.image = image
        item.source_path = filepath
    setattr(props, index_attr, 0)
    _mark_reference_changed(props, "STYLE")
    return len(loaded), failures


def _replace_product_reference(props, filepath):
    """3.1.8: the product-shape slot holds exactly one image; a new one replaces it.

    The image is loaded first so a failed read never empties the slot."""
    image = _load_reference_image(filepath)
    collection, index_attr = _collection_and_index(props, "PRODUCT")
    old_note = (getattr(collection[0], "instruction", "") or "") if len(collection) else ""
    collection.clear()
    item = collection.add()
    item.image = image
    item.source_path = str(filepath)
    item.instruction = old_note
    setattr(props, index_attr, 0)
    return item


def _gate_summary(attempts, exhausted) -> dict:
    """1.2.3: one variant's IoU acceptance: every try, the best (pass > review > fail, then IoU)."""
    from .result_worker import gate_rank
    tries = [a.get("iou_gate") for a in attempts if a.get("iou_gate")]
    best = max(tries, key=gate_rank) if tries else None
    passed = bool(best and best.get("passed"))
    return {"tries": [{k: t.get(k) for k in ("iou", "iou_lo", "iou_hi", "verdict", "weak_frac", "out_of_band",
                                             "candidate", "seconds", "error")} for t in tries],
            "best": best, "passed": passed, "exhausted": bool(exhausted and not passed),
            "candidates": [str(a.get("path")) for a in attempts]}


_LAYER_SCENES = set()
_LAYER_BATCH = []


def _poll_layers():
    """Main-thread timer: copy finished 分层输出 jobs into the panel note."""
    from . import layered_output
    try:
        results = layered_output.RUNNER.drain()
        if results:
            _LAYER_BATCH.extend(results)
            note = layered_output.summarize(_LAYER_BATCH)
            if layered_output.RUNNER.pending > 0:
                note += f"；还有 {layered_output.RUNNER.pending} 张处理中"
            for name in list(_LAYER_SCENES):
                scene = bpy.data.scenes.get(name)
                if scene is not None:
                    scene.wondful_ai.last_layers_note = note
            for window in getattr(bpy.context.window_manager, "windows", []):
                for area in window.screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()
    except Exception as exc:
        print(f"[Wondful] 分层输出状态更新失败：{exc}")
    if layered_output.RUNNER.pending > 0:
        return 0.5
    _LAYER_SCENES.clear()
    return None


def _watch_layers(scene_name):
    _LAYER_SCENES.add(scene_name)
    if not bpy.app.timers.is_registered(_poll_layers):
        bpy.app.timers.register(_poll_layers, first_interval=0.5)


class _BaseAsyncOperator(Operator):
    _timer = None
    _thread = None
    _result = None
    _error = None
    _started_at = 0.0
    _estimated_total = 60.0
    _origin_scene = None

    @classmethod
    def poll(cls, context):
        return not _ACTIVE_LOCK.locked()

    def report(self, type, message):
        # 1.2.2: remember why an operator refused (e.g. a chained, timer-started render), for the panel.
        if "ERROR" in type or "WARNING" in type:
            _LAST_REPORT[self.bl_idname] = str(message)
        return super().report(type, message)

    def _start_thread(self, context, target):
        if not _ACTIVE_LOCK.acquire(blocking=False):
            self.report({"WARNING"}, "已有 AI / 登录任务正在运行。")
            return False
        global _ACTIVE_OPERATOR
        _ACTIVE_OPERATOR = self
        from .cli_transport import reset_cancellation
        reset_cancellation()
        self._owns_lock = True
        self._ownership_guard = threading.Lock()
        self._detached = threading.Event()
        self._origin_scene = context.scene
        self._result = None
        self._error = None
        self._started_at = time.time()

        def runner():
            try:
                self._result = target()
            except Exception as exc:
                self._error = (exc, traceback.format_exc())
            finally:
                if self._detached.is_set():
                    self._release_lock()

        self._thread = threading.Thread(target=runner, daemon=True, name="WondfulAIWorker")
        try:
            wm = context.window_manager
            self._timer = wm.event_timer_add(0.25, window=context.window)
            wm.modal_handler_add(self)
            self._thread.start()
        except Exception as exc:
            self._cleanup(context)
            _status(context.scene.wondful_ai, "ERROR", str(exc))
            self.report({"ERROR"}, str(exc))
            return False
        return True

    def modal(self, context, event):
        from .cli_transport import is_cancellation_requested
        if is_cancellation_requested():
            # Keep the modal and global lock alive until the worker actually exits.
            # Jev HTTP cannot be force-killed like a CLI subprocess, so releasing
            # the lock early would allow a second render to start concurrently.
            if self._thread and self._thread.is_alive():
                try:
                    props = self._origin_scene.wondful_ai
                    props.task_phase = "正在取消，等待当前网络/CLI请求结束"
                    props.eta_seconds = 0
                except (AttributeError, ReferenceError):
                    pass
                return {"RUNNING_MODAL"}
            self._cleanup(context)
            if self._origin_scene and self._origin_scene in list(bpy.data.scenes):
                _status(self._origin_scene.wondful_ai, "IDLE", "任务已被取消")
                self._origin_scene.wondful_ai.progress = 0.0
                self._origin_scene.wondful_ai.eta_seconds = 0
            self.report({"INFO"}, "任务已被取消")
            return {"CANCELLED"}
        try:
            scene_exists = self._origin_scene in list(bpy.data.scenes)
        except ReferenceError:
            scene_exists = False
        if not scene_exists:
            self._cleanup(context)
            return {"CANCELLED"}
        # Postprocessing, metadata and export must use the submitting scene.
        try:
            with context.temp_override(scene=self._origin_scene):
                return self._modal(context, event)
        except Exception as exc:
            self._cleanup(context)
            _status(self._origin_scene.wondful_ai, "ERROR", f"任务收尾失败：{exc}")
            self.report({"ERROR"}, f"任务收尾失败：{exc}")
            return {"CANCELLED"}

    def _progress_update(self, props, floor=0.10, ceiling=0.88):
        elapsed = max(0.0, time.time() - self._started_at)
        frac = min(1.0, elapsed / max(1.0, self._estimated_total))
        props.progress = floor + (ceiling - floor) * min(0.98, frac)
        props.eta_seconds = max(0, int(self._estimated_total - elapsed))
        props.task_phase = getattr(self, "_phase", "")

    def _release_lock(self):
        with self._ownership_guard:
            if self._owns_lock:
                self._owns_lock = False
                _ACTIVE_LOCK.release()

    def _request_cancel(self, context):
        from .cli_transport import request_cancellation
        request_cancellation()
        # Do not cleanup or release ownership here. The modal waits for the
        # worker to exit, then performs one final cleanup/release.
        try:
            if self._origin_scene and self._origin_scene in list(bpy.data.scenes):
                props = self._origin_scene.wondful_ai
                props.task_phase = "正在取消，等待当前请求结束"
                props.eta_seconds = 0
        except Exception:
            pass

    def _cleanup(self, context):
        global _ACTIVE_OPERATOR
        if _ACTIVE_OPERATOR is self:
            _ACTIVE_OPERATOR = None
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except (ReferenceError, RuntimeError):
                pass
            self._timer = None
        # Blender may cancel a modal on window close while the CLI is still busy.
        # Retain exclusivity until that worker exits; never release another task's lock.
        self._detached.set()
        if not self._thread or not self._thread.is_alive():
            self._release_lock()
        try:
            if self._origin_scene is not None:
                _apply_model_notice(self._origin_scene.wondful_ai)
        except (ReferenceError, AttributeError):
            pass

    def cancel(self, context):
        self._cleanup(context)
        try:
            _status(self._origin_scene.wondful_ai, "ERROR", "任务窗口已关闭；后台请求结束后可重新操作。")
        except (ReferenceError, AttributeError):
            pass


class WONDFUL_OT_autofill_product_look(_BaseAsyncOperator):
    """3.1.8: read the product reference and fill 产品外观 / 保留细节 now."""
    bl_idname = "wondful.autofill_product_look"
    bl_label = "根据参考自动填写"
    bl_description = "让 AI 看产品参考图，自动填写产品外观与保留细节；会覆盖这两栏当前内容"

    @classmethod
    def poll(cls, context):
        props = context.scene.wondful_ai
        return not _ACTIVE_LOCK.locked() and len(props.product_images) > 0

    def execute(self, context):
        props = context.scene.wondful_ai
        prefs = _addon_prefs(context)
        provider = _analysis_provider(props, prefs)
        paths = _reference_paths(props)["product"]
        if not paths:
            self.report({"WARNING"}, "请先放一张产品参考图。")
            return {"CANCELLED"}
        session = create_session()
        upload = [prepare_reference_for_upload(p) for p in copy_references(paths[:1], session, "product_reference")]
        note = (_reference_instructions(props)["product"] or [""])[0]
        color_source = str(getattr(props, "color_source", "BLENDER") or "BLENDER")
        self._key = _product_autofill_key(props, provider)
        self._phase = "读取产品参考图外观"
        self._estimated_total = average_timing(f"{provider['id']}_polish", 30.0)
        _status(props, "POLISHING")
        props.progress = 0.05
        polish_fn, cli_path, model = provider["polish"], provider["cli_path"], provider["model"]
        timeout = int(provider["polish_timeout"])

        def job():
            reply = polish_fn(
                system_prompt=PRODUCT_ANALYSIS_SYSTEM_PROMPT,
                user_text=product_analysis_user_text(color_source, note),
                image_paths=upload,
                cwd=str(session.directory),
                explicit_path=cli_path,
                model=model,
                timeout=timeout,
            )
            return parse_product_analysis(reply)

        if not self._start_thread(context, job):
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _modal(self, context, event):
        props = self._origin_scene.wondful_ai
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._thread and self._thread.is_alive():
            self._progress_update(props, 0.10, 0.90)
            return {"RUNNING_MODAL"}
        self._cleanup(context)
        props.progress = 0.0
        props.eta_seconds = 0
        if self._error:
            _status(props, "ERROR", f"自动填写失败：{self._error[0]}")
            self.report({"ERROR"}, props.last_error)
            return {"CANCELLED"}
        look, details = self._result or ("", [])
        if not look and not details:
            _status(props, "ERROR", "AI 没有返回可用的外观描述，两栏保持原样。")
            self.report({"WARNING"}, props.last_error)
            return {"CANCELLED"}
        # Explicit button: the user asked for a refill, so overwrite.
        props.product_look_auto = props.product_look_prompt
        props.identity_details_auto = props.identity_details
        _apply_product_autofill(props, self._key, look, details)
        _status(props, "IDLE")
        self.report({"INFO"}, props.product_autofill_message)
        return {"FINISHED"}


_LAST_REPORT = {}


def _redraw_all():
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    except Exception:
        pass


def _schedule_operator(idname, scene=None, *, label="AI 渲染"):
    """Run a modal operator once the current one has released the task lock.

    1.2.2: robust chain (see op_chain): tries every 3D View / window context, retries
    transient context errors, and when the operator cannot start shows the reason in
    the panel instead of stopping silently.  Returns the ChainRunner (for tests)."""
    from . import op_chain
    from .op_chain import ChainRunner, OperatorRefused
    module, name = idname.split(".")
    scene_name = getattr(scene, "name", "") if scene is not None else ""

    def target_scene():
        sc = bpy.data.scenes.get(scene_name) if scene_name else None
        return sc or getattr(bpy.context, "scene", None)

    def invoke(ctx):
        _LAST_REPORT.pop(idname, None)
        try:
            with bpy.context.temp_override(**ctx):
                return getattr(getattr(bpy.ops, module), name)("INVOKE_DEFAULT")
        except RuntimeError as exc:
            # bpy.ops raises RuntimeError when the operator reports an ERROR.  Before
            # 1.2.2 that exception escaped the timer callback and was only printed to the
            # (hidden on Windows) system console: the chain stopped silently.
            refused = _LAST_REPORT.get(idname, "")
            if refused:
                raise OperatorRefused(refused) from exc
            raise

    def contexts():
        return op_chain.candidate_contexts(list(getattr(bpy.context.window_manager, "windows", []) or []), target_scene())

    def on_fail(message):
        sc = target_scene()
        props = getattr(sc, "wondful_ai", None) if sc is not None else None
        text = f"提示词已写好，但没能自动开始{label}：{message}。处理后点「生成」即可继续。"
        print(f"[Wondful] {text}")
        if props is not None:
            props.pending_generate = False
            _status(props, "ERROR", text)
        _redraw_all()

    runner = ChainRunner(invoke=invoke, contexts=contexts, locked=_ACTIVE_LOCK.locked, on_fail=on_fail,
                         on_started=lambda ctx, result: _redraw_all(),
                         last_report=lambda: _LAST_REPORT.get(idname, ""))
    bpy.app.timers.register(runner.step, first_interval=0.3)
    return runner


def generate_needs_polish(props, prefs):
    """3.1.9: decide whether 「生成」 must (re)write the prompt before rendering."""
    brief = (getattr(props, "scene_brief", "") or "").strip()
    if not (props.prompt or "").strip():
        return True
    if brief != (getattr(props, "scene_brief_used", "") or "").strip():
        return True
    if needs_reference_policy_refresh(props) or prompt_profiles.needs_target_refresh(props, prefs):
        return True
    if (props.style_prompt_dirty
            or int(props.prompt_style_revision) != int(props.style_reference_revision)
            or (len(props.style_images) > 0 and not bool(props.style_sync_initialized))):
        return True
    key = _product_autofill_key(props, _analysis_provider(props, prefs))
    return bool(key) and _product_autofill_needed(props, key)


class WONDFUL_OT_generate(Operator):
    """3.1.9 single entry point: polish when needed, then render."""
    bl_idname = "wondful.generate"
    bl_label = "生成"
    bl_description = "自动完成：读产品图 → 分析环境 → 写提示词 → 渲染与验收"

    @classmethod
    def poll(cls, context):
        return not _ACTIVE_LOCK.locked()

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):
        props = context.scene.wondful_ai
        prefs = _addon_prefs(context)
        props.model_notice = ""
        brief = (props.scene_brief or "").strip()
        if generate_needs_polish(props, prefs):
            if brief != (props.scene_brief_used or "").strip() or not (props.prompt or "").strip():
                # A new brief replaces the old prompt as the polish source.
                props.prompt = brief
                props.last_ai_prompt = ""
            props.pending_generate = not props.two_step
            _LAST_REPORT.pop("wondful.polish_prompt", None)
            try:
                result = bpy.ops.wondful.polish_prompt("INVOKE_DEFAULT")
            except RuntimeError as exc:
                result = {"CANCELLED"}
                message = str(exc).replace("Error: ", "").strip()
                _status(props, "ERROR", message)
            if "RUNNING_MODAL" not in result and "FINISHED" not in result:
                props.pending_generate = False
                if props.status != "ERROR":
                    _status(props, "ERROR", _LAST_REPORT.get("wondful.polish_prompt", "") or "没能开始写提示词，请重试。")
                return {"CANCELLED"}
            return {"FINISHED"}
        _LAST_REPORT.pop("wondful.ai_render", None)
        try:
            result = bpy.ops.wondful.ai_render("INVOKE_DEFAULT")
        except RuntimeError as exc:
            _status(props, "ERROR", str(exc).replace("Error: ", "").strip())
            return {"CANCELLED"}
        if "RUNNING_MODAL" in result or "FINISHED" in result:
            return {"FINISHED"}
        if props.status != "ERROR":
            _status(props, "ERROR", _LAST_REPORT.get("wondful.ai_render", "") or "生图前检查未通过")
        return {"CANCELLED"}


class WONDFUL_OT_refresh_models(_BaseAsyncOperator):
    bl_idname = "wondful.refresh_models"
    bl_label = "刷新模型列表"
    bl_description = "从当前 CLI 读取具体推理模型 ID；不发起润色或生图请求"

    def execute(self, context):
        props, prefs = context.scene.wondful_ai, _addon_prefs(context)
        provider = _analysis_provider(props, prefs)
        self._catalog_provider = provider["id"]
        cli = provider["discover"]()
        if not cli:
            self.report({"ERROR"}, "未找到当前服务的 CLI，请先检测安装或选择路径。")
            return {"CANCELLED"}
        setattr(props, self._catalog_provider + "_cli_resolved", cli)
        _status(props, "REFRESHING_MODELS")
        setattr(props, self._catalog_provider + "_models_message", "正在读取模型列表…")
        if not self._start_thread(context, lambda: discover_models(self._catalog_provider, cli, cwd=CONFIG_DIR)):
            _status(props, "IDLE")
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._thread and self._thread.is_alive():
            return {"RUNNING_MODAL"}
        self._cleanup(context)
        props = self._origin_scene.wondful_ai
        attr = self._catalog_provider + "_models_message"
        if self._error:
            message = safe_diagnostic(str(self._error[0]))
            setattr(props, attr, "刷新失败；已有列表未更新。" + message)
            _status(props, "ERROR", message)
            return {"CANCELLED"}
        collection = getattr(props, self._catalog_provider + "_models")
        collection.clear()
        for model in self._result:
            item = collection.add()
            item.name, item.label = model["id"], model["label"]
            item.description, item.is_default = model["description"], model["default"]
        live = sum(1 for model in self._result if model.get("live"))
        setattr(props, attr, f"已读取 {len(collection)} 个模型" + (f"（账号最新 {live} 个在前）" if live else "")
                + " · 点击模型框选择")
        _status(props, "IDLE")
        return {"FINISHED"}


class WONDFUL_OT_model_default(Operator):
    bl_idname = "wondful.model_default"
    bl_label = "使用 CLI 默认模型"

    @classmethod
    def poll(cls, context):
        return not _ACTIVE_LOCK.locked()

    def execute(self, context):
        props, prefs = context.scene.wondful_ai, _addon_prefs(context)
        setattr(prefs, "antigravity_model" if props.analysis_provider == "ANTIGRAVITY" else "codex_model", "")
        return {"FINISHED"}


class WONDFUL_OT_terrain_preset(Operator):
    bl_idname = "wondful.terrain_preset"
    bl_label = "地形预设"
    bl_description = "把四个角的地面坡度、坡顶和凹凸填成预设地形（坡顶、对角扭曲、颠簸路面、爬坡、下坡、侧坡）"
    bl_options = {"REGISTER", "UNDO"}

    preset: bpy.props.EnumProperty(
        items=[("CREST", "坡顶", "产品停在坡顶最高处，四周地面向下掉落"),
               ("TWIST", "对角扭曲", "左前、右后地面高，右前、左后地面低；产品三点着地、一角明显悬空"),
               ("TWIST_MIRROR", "镜像扭曲", "右前、左后地面高，左前、右后地面低"),
               ("BUMPY", "颠簸路面", "地面高低起伏、凹凸不平"),
               ("CLIMB", "爬坡", "前高后低"), ("DESCENT", "下坡", "前低后高"), ("SIDE", "侧坡", "左高右低")],
        default="CREST",
    )

    def execute(self, context):
        from .terrain import apply_preset_to_props
        props = context.scene.wondful_ai
        apply_preset_to_props(props, self.preset)
        props.clay_ground = "CUSTOM"
        return {"FINISHED"}


def _active_block(context):
    """(root block object, massing collection) for the active object, or (None, collection)."""
    from . import massing as MS
    props = context.scene.wondful_ai
    coll = MS.find_collection(context.scene, props)
    obj = getattr(context, "active_object", None) or getattr(context.view_layer.objects, "active", None)
    if coll is None or obj is None:
        return None, coll
    objs = MS.massing_objects(coll, getattr(props, "product_collection", None))
    return MS.block_root(obj, objs), coll


class WONDFUL_OT_block_ref_add(Operator, ImportHelper):
    bl_idname = "wondful.block_ref_add"
    bl_label = "添加建筑参考"
    bl_description = ("给当前选中的环境体块添加 1–3 张建筑参考图（真实建筑照片、效果图或概念草图）。"
                      "生图时 AI 按参考图的风格、立面、材质画这栋楼，体块的位置、高度和轮廓不变")
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ""
    filter_glob: StringProperty(default="*.png;*.jpg;*.jpeg;*.webp", options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _active_block(context)[0] is not None

    def execute(self, context):
        from . import massing as MS
        block, _coll = _active_block(context)
        if block is None:
            self.report({"ERROR"}, "请先在环境体块集合里选中一个体块")
            return {"CANCELLED"}
        selected = [str(Path(self.directory) / f.name) for f in self.files if f.name] if self.files else []
        if not selected and self.filepath:
            selected = [self.filepath]
        selected = [p for p in selected if p and Path(bpy.path.abspath(p)).is_file()]
        if not selected:
            self.report({"WARNING"}, "没有选择图片")
            return {"CANCELLED"}
        kept, dropped = MS.merge_refs(MS.get_refs(block), selected, MS.MAX_BLOCK_REFS)
        MS.set_refs(block, kept)
        msg = f"{block.name}：建筑参考 {len(kept)} 张"
        if dropped:
            msg += f"（每个体块最多 {MS.MAX_BLOCK_REFS} 张，{len(dropped)} 张未添加）"
        self.report({"WARNING" if dropped else "INFO"}, msg)
        return {"FINISHED"}


class WONDFUL_OT_block_ref_clear(Operator):
    bl_idname = "wondful.block_ref_clear"
    bl_label = "清空建筑参考"
    bl_description = "清空当前选中体块的建筑参考图（只删除引用，不删除图片文件）"
    bl_options = {"REGISTER", "UNDO"}
    index: IntProperty(default=-1, options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _active_block(context)[0] is not None

    def execute(self, context):
        from . import massing as MS
        block, _coll = _active_block(context)
        if block is None:
            return {"CANCELLED"}
        refs = MS.get_refs(block)
        if 0 <= self.index < len(refs):
            refs.pop(self.index)
        else:
            refs = []
        MS.set_refs(block, refs)
        return {"FINISHED"}


class WONDFUL_OT_block_select(Operator):
    bl_idname = "wondful.block_select"
    bl_label = "选中体块"
    bl_description = "在场景中选中这个环境体块（之后可添加 / 清空它的建筑参考）"
    bl_options = {"REGISTER", "UNDO"}
    name: StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})

    def execute(self, context):
        obj = bpy.data.objects.get(self.name)
        if obj is None:
            return {"CANCELLED"}
        try:
            for other in context.selected_objects:
                other.select_set(False)
            obj.select_set(True)
        except Exception:
            pass
        context.view_layer.objects.active = obj
        return {"FINISHED"}


def keep_exact_state(context, props):
    """(selected product parts, how many are already keep-exact) for 「标为原样保留」."""
    from .viewport_capture import keep_exact_slots
    coll = getattr(props, "product_collection", None)
    product = set(coll.all_objects) if coll is not None else set()
    selected = [o for o in (getattr(context, "selected_objects", None) or ())
                if o.type in {"MESH", "CURVE", "SURFACE", "FONT", "META"} and (not product or o in product)]
    marked = {obj for (obj, _i) in keep_exact_slots(coll)} if coll is not None else set()
    if coll is None:
        from .detail_paste import KEEP_PROP, prop_state, name_looks_keep_exact
        marked = {o for o in selected if prop_state(o.get(KEEP_PROP)) is True
                  or (prop_state(o.get(KEEP_PROP)) is None and name_looks_keep_exact(o.name))}
    return selected, sum(1 for o in selected if o in marked)


class WONDFUL_OT_mark_keep_exact(Operator):
    bl_idname = "wondful.mark_keep_exact"
    bl_label = "标为原样保留"
    bl_description = ("细节回贴：把选中的产品部件（车牌、车标、铭牌、Logo 等）标为原样保留，生成后这些区域整块贴回 "
                      "Blender 里的真实像素（只轻微匹配颜色和亮度）。已标记时再点一次取消（也会关掉按名称的自动识别）")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from .detail_paste import KEEP_PROP
        props = context.scene.wondful_ai
        selected, marked = keep_exact_state(context, props)
        if not selected:
            self.report({"WARNING"}, "先在视图里选中产品里的车牌 / 标识等部件")
            return {"CANCELLED"}
        value = 0 if marked == len(selected) else 1
        for obj in selected:
            obj[KEEP_PROP] = value
        msg = (f"已标为原样保留：{len(selected)} 个部件" if value else f"已取消原样保留：{len(selected)} 个部件")
        props.last_detail_note = msg
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class WONDFUL_OT_massing_hull_build(Operator):
    bl_idname = "wondful.massing_hull_build"
    bl_label = "从三视图生成建筑体块"
    bl_description = ("从建筑的正视 / 侧视 / 俯视图轮廓雕出一个建筑体块，放进环境体块集合（没有时新建「环境体块」），"
                      "放在 3D 游标处、落在地面上，正立面朝 -Y；三视图自动作为它的建筑参考。产品不受影响")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.wondful_ai
        try:
            from . import massing_hull
            obj, info = massing_hull.build_block(context, props)
        except Exception as exc:
            props.last_hull_note = f"建筑体块生成失败：{exc}"
            self.report({"ERROR"}, props.last_hull_note[:200])
            return {"CANCELLED"}
        props.last_hull_note = massing_hull.summary(info)
        try:
            for other in context.selected_objects:
                other.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
        except Exception:
            pass
        self.report({"INFO"}, props.last_hull_note[:200])
        return {"FINISHED"}


class WONDFUL_OT_codex_check(_BaseAsyncOperator):
    bl_idname = "wondful.codex_check"
    bl_label = "检测 Codex"
    bl_description = "检测 Codex CLI 与 ChatGPT OAuth 登录状态"

    def execute(self, context):
        props = context.scene.wondful_ai
        prefs = _addon_prefs(context)
        props.codex_account_message = "正在检测…"
        self._estimated_total = 8.0

        if not self._start_thread(context, lambda: get_account_status(prefs.codex_cli_path)):
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._thread and self._thread.is_alive():
            return {"RUNNING_MODAL"}
        props = self._origin_scene.wondful_ai
        self._cleanup(context)
        if self._error:
            exc, _tb = self._error
            props.codex_account_state = "ERROR"
            props.codex_account_message = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _apply_account_status(props, self._result)
        if self._result.logged_in:
            _status(props, "IDLE")
            props.progress = 0.0
            props.eta_seconds = 0
        self.report({"INFO"}, "Codex 已登录 ChatGPT。" if self._result.logged_in else self._result.message)
        return {"FINISHED"}


class WONDFUL_OT_codex_login(_BaseAsyncOperator):
    bl_idname = "wondful.codex_login"
    bl_label = "使用 ChatGPT 登录"
    bl_description = "调用官方 Codex OAuth 登录；通常会自动打开浏览器"

    def execute(self, context):
        props = context.scene.wondful_ai
        prefs = _addon_prefs(context)
        _status(props, "AUTHENTICATING")
        props.progress = 0.05
        props.eta_seconds = 0
        props.codex_account_message = "正在启动 ChatGPT OAuth，请在浏览器完成登录…"
        self._estimated_total = 90.0

        if not self._start_thread(context, lambda: login_chatgpt(prefs.codex_cli_path, timeout=600)):
            _status(props, "IDLE")
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        props = self._origin_scene.wondful_ai
        if self._thread and self._thread.is_alive():
            # OAuth duration is user-controlled, so only show activity rather than a fake ETA.
            elapsed = max(0.0, time.time() - self._started_at)
            props.progress = min(0.85, 0.05 + elapsed / 180.0)
            return {"RUNNING_MODAL"}

        self._cleanup(context)
        props.progress = 0.0
        props.eta_seconds = 0
        if self._error:
            exc, _tb = self._error
            props.codex_account_state = "ERROR"
            props.codex_account_message = str(exc)
            _status(props, "ERROR", str(exc))
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        _apply_account_status(props, self._result)
        _status(props, "IDLE")
        self.report({"INFO"}, "ChatGPT OAuth 登录成功，Blender 已可调用 Codex。")
        return {"FINISHED"}


class WONDFUL_OT_codex_logout(Operator):
    bl_idname = "wondful.codex_logout"
    bl_label = "退出 ChatGPT"
    bl_description = "调用官方 codex logout；插件自身不保存 OAuth Token"

    def execute(self, context):
        props = context.scene.wondful_ai
        prefs = _addon_prefs(context)
        try:
            status = codex_logout(prefs.codex_cli_path)
        except Exception as exc:
            props.codex_account_state = "ERROR"
            props.codex_account_message = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _apply_account_status(props, status)
        self.report({"INFO"}, "已退出 Codex ChatGPT 登录。")
        return {"FINISHED"}


class WONDFUL_OT_antigravity_check(_BaseAsyncOperator):
    bl_idname = "wondful.antigravity_check"
    bl_label = "检测 AGY 安装"
    bl_description = "仅检测 CLI 路径与版本，不发起模型请求或浏览器登录"

    def execute(self, context):
        props = context.scene.wondful_ai
        prefs = _addon_prefs(context)
        props.antigravity_account_message = "正在检测 AGY 路径与版本…"
        self._estimated_total = 12.0
        if not self._start_thread(
            context,
            lambda: get_antigravity_account_status(prefs.antigravity_cli_path, live_check=False),
        ):
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._thread and self._thread.is_alive():
            return {"RUNNING_MODAL"}
        props = self._origin_scene.wondful_ai
        self._cleanup(context)
        if self._error:
            exc, _tb = self._error
            props.antigravity_account_state = "ERROR"
            props.antigravity_account_message = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        previous_state, previous_cli = props.antigravity_account_state, props.antigravity_cli_resolved
        _apply_antigravity_account_status(props, self._result)
        if self._result.installed and previous_state == "LOGGED_IN" and previous_cli == self._result.cli_path:
            props.antigravity_account_state = "LOGGED_IN"
        if self._result.logged_in:
            _status(props, "IDLE")
            props.progress = 0.0
            props.eta_seconds = 0
        self.report({"INFO"}, "Antigravity 已通过 Google 账号登录。" if self._result.logged_in else self._result.message)
        return {"FINISHED"}


class WONDFUL_OT_antigravity_login(_BaseAsyncOperator):
    bl_idname = "wondful.antigravity_login"
    bl_label = "验证登录"
    bl_description = "使用选定模型检查官方 CLI 登录；发起一次最小文本请求，不生成图片"
    allow_interactive: BoolProperty(default=False, options={"SKIP_SAVE"})

    def execute(self, context):
        global _AGY_AUTH, _AGY_AUTH_OWNER
        props, prefs = context.scene.wondful_ai, _addon_prefs(context)
        cli = discover_antigravity_cli(prefs.antigravity_cli_path or getattr(props, "antigravity_cli_resolved", ""))
        if not cli:
            props.antigravity_account_state = "NOT_INSTALLED"
            self.report({"ERROR"}, "未找到 agy，请在设置中选择 CLI 路径。")
            return {"CANCELLED"}
        self._auth = AuthSession(cli, cwd=CONFIG_DIR, version=props.antigravity_cli_version,
                                 model=prefs.antigravity_model, allow_interactive=self.allow_interactive,
                                 timeout=600 if self.allow_interactive else 120)
        _AGY_AUTH = self._auth
        _AGY_AUTH_OWNER = self
        context.window_manager.wondful_auth_code = ""
        props.antigravity_cli_resolved = cli
        props.auth_stage, props.auth_detail = "STARTING", ""
        _status(props, "AUTHENTICATING")
        self._estimated_total = 90.0

        def job():
            installation = get_antigravity_account_status(cli, live_check=False)
            self._auth.version = installation.version
            return self._auth.run()

        if not self._start_thread(context, job):
            _AGY_AUTH = None
            _AGY_AUTH_OWNER = None
            _status(props, "IDLE")
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _modal(self, context, event):
        global _AGY_AUTH, _AGY_AUTH_OWNER
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        props = self._origin_scene.wondful_ai
        snapshot = self._auth.snapshot()
        props.auth_stage = snapshot["stage"]
        props.auth_detail = snapshot["detail"] or snapshot["message"]
        props.auth_has_url = snapshot["has_url"]
        if self._thread and self._thread.is_alive():
            return {"RUNNING_MODAL"}
        self._cleanup(context)
        context.window_manager.wondful_auth_code = ""
        if _AGY_AUTH is self._auth:
            _AGY_AUTH = None
            _AGY_AUTH_OWNER = None
        props.auth_has_url = False
        if self._error:
            _status(props, "ERROR", safe_diagnostic(str(self._error[0])))
            return {"CANCELLED"}
        _apply_antigravity_account_status(props, self._result)
        availability = snapshot["image_tool_available"]
        props.agy_image_capability = "UNKNOWN" if availability is None else ("AVAILABLE" if availability else "NOT_LISTED")
        if self._result.logged_in:
            _status(props, "IDLE")
            self.report({"INFO"}, self._result.message)
        elif snapshot["stage"] == "CANCELED":
            _status(props, "IDLE")
        else:
            _status(props, "ERROR", self._result.message)
            self.report({"ERROR"}, self._result.message)
        return {"FINISHED"}

    def _cleanup(self, context):
        if self._thread and self._thread.is_alive():
            self._auth.cancel()
        super()._cleanup(context)

    def cancel(self, context):
        self._auth.cancel()
        context.window_manager.wondful_auth_code = ""
        super().cancel(context)


class WONDFUL_OT_antigravity_submit_code(Operator):
    bl_idname = "wondful.antigravity_submit_code"
    bl_label = "提交授权码"

    def execute(self, context):
        code = context.window_manager.wondful_auth_code
        context.window_manager.wondful_auth_code = ""
        try:
            if _AGY_AUTH is None:
                raise RuntimeError("本次登录已结束，请重新连接。")
            _AGY_AUTH.submit_code(code)
        except Exception as exc:
            self.report({"ERROR"}, safe_diagnostic(str(exc), [code]))
            return {"CANCELLED"}
        return {"FINISHED"}


class WONDFUL_OT_antigravity_auth_cancel(Operator):
    bl_idname = "wondful.antigravity_auth_cancel"
    bl_label = "取消连接"

    def execute(self, context):
        cancel_auth_session()
        context.window_manager.wondful_auth_code = ""
        return {"FINISHED"}


class WONDFUL_OT_cancel_task(Operator):
    bl_idname = "wondful.cancel_task"
    bl_label = "取消当前任务"
    bl_description = "主动中断正在执行的 AI 润色或生图任务，释放后台锁并恢复就绪状态"

    def execute(self, context):
        from .cli_transport import request_cancellation
        request_cancellation()
        cancel_auth_session(detach=True, context=context)
        global _ACTIVE_OPERATOR
        if _ACTIVE_OPERATOR is not None:
            _ACTIVE_OPERATOR._request_cancel(context)
        else:
            props = getattr(getattr(context, "scene", None), "wondful_ai", None)
            if _ACTIVE_LOCK.locked():
                # The lock is owned by a detached worker. Never release a lock
                # without its owner; wait for that worker's finally block.
                if props:
                    props.task_phase = "后台任务正在结束"
                    props.eta_seconds = 0
                self.report({"INFO"}, "后台任务正在结束，完成后会自动释放。")
                return {"FINISHED"}
            if props:
                _status(props, "IDLE", "当前没有可取消的任务")
                props.progress = 0.0
                props.eta_seconds = 0
        self.report({"INFO"}, "已发送取消请求。")
        return {"FINISHED"}


class WONDFUL_OT_auto_classify_references(_BaseAsyncOperator):
    bl_idname = "wondful.auto_classify_references"
    bl_label = "Jev 语义整理参考图"
    bl_description = "根据文件名与用户备注进行 Jev 语义整理；Jev 不读取图片像素，低置信度保持原分类"

    def execute(self, context):
        props = context.scene.wondful_ai
        from .image_manager import image_filepath
        from . import jev_semantics

        all_refs = []
        for kind in ("PRODUCT", "STYLE", "PERSON"):
            coll, _ = _collection_and_index(props, kind)
            for item in list(coll):
                img = item.image
                src = item.source_path or (image_filepath(img) if img else "")
                note = (getattr(item, "instruction", "") or "").strip()
                if src:
                    all_refs.append((kind, src, note, img))

        if not all_refs:
            self.report({"INFO"}, "当前没有参考图可整理。")
            return {"CANCELLED"}

        self._all_refs = all_refs
        self._estimated_total = 12.0
        batch_input = [
            {"filepath": src, "instruction": note, "current_kind": current_kind}
            for current_kind, src, note, _img in all_refs
        ]
        _status(props, "CLASSIFYING")
        props.progress = 0.05
        props.eta_seconds = 12

        if not self._start_thread(context, lambda: jev_semantics.classify_references_batch(batch_input)):
            _status(props, "IDLE")
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        props = self._origin_scene.wondful_ai
        if self._thread and self._thread.is_alive():
            self._progress_update(props, 0.10, 0.85)
            return {"RUNNING_MODAL"}

        self._cleanup(context)
        if self._error:
            _status(props, "ERROR", safe_diagnostic(str(self._error[0])))
            props.progress = 0.0
            self.report({"ERROR"}, props.last_error)
            return {"CANCELLED"}

        from . import jev_semantics
        judgments = list(self._result or [])
        all_refs = list(getattr(self, "_all_refs", []))
        if len(judgments) != len(all_refs):
            _status(props, "ERROR", "Jev 返回数量与参考图数量不一致，已保留原分类。")
            props.progress = 0.0
            return {"CANCELLED"}

        uncertain = 0
        jev_count = 0
        fallback_count = 0
        staged = []
        counts = {"PRODUCT": 0, "STYLE": 0, "PERSON": 0}
        for current_kind, _src, _note, _img in all_refs:
            counts[current_kind] += 1

        migration_candidates = []
        for idx, ((current_kind, src, note, img), judgment) in enumerate(zip(all_refs, judgments)):
            target_kind = str(judgment.get("choice", "")).upper()
            confidence = float(judgment.get("confidence", 0.0) or 0.0)
            backend = str(judgment.get("backend", "LOCAL_FALLBACK"))
            jev_count += int(backend == "JEV")
            fallback_count += int(backend != "JEV")
            if target_kind not in {"PRODUCT", "STYLE", "PERSON"} or confidence < jev_semantics.REFERENCE_CONFIDENCE_THRESHOLD:
                target_kind = current_kind
                uncertain += 1
            staged.append([current_kind, current_kind, src, note, img, confidence])
            if target_kind != current_kind:
                migration_candidates.append((confidence, idx, target_kind))

        # Original membership is reserved first. Migrations are then accepted
        # highest-confidence first only when the destination has free capacity.
        migrated = 0
        capacity_preserved = 0
        for _confidence, idx, target_kind in sorted(migration_candidates, reverse=True):
            current_kind = staged[idx][0]
            if counts[target_kind] >= reference_limit(target_kind):
                capacity_preserved += 1
                continue
            counts[current_kind] -= 1
            counts[target_kind] += 1
            staged[idx][1] = target_kind
            migrated += 1

        if sum(counts.values()) != len(all_refs) or any(v > reference_limit(k) and k != "PRODUCT" for k, v in counts.items()):
            _status(props, "ERROR", "参考图容量校验失败，已保留原分类。")
            props.progress = 0.0
            return {"CANCELLED"}

        style_changed = any(current != target and (current == "STYLE" or target == "STYLE")
                            for current, target, *_rest in staged)

        props.product_images.clear()
        props.style_images.clear()
        props.person_images.clear()
        restored = 0
        for _current, target_kind, src, note, img, _confidence in staged:
            coll, idx_attr = _collection_and_index(props, target_kind)
            new_item = coll.add()
            new_item.image = img
            new_item.source_path = src
            new_item.instruction = note
            setattr(props, idx_attr, len(coll) - 1)
            restored += 1

        if restored != len(all_refs):
            _status(props, "ERROR", "参考图恢复数量异常。")
            props.progress = 0.0
            return {"CANCELLED"}
        if style_changed:
            _mark_reference_changed(props, "STYLE")

        props.jev_status = "JEV" if jev_count and not fallback_count else ("MIXED" if jev_count else "LOCAL_FALLBACK")
        props.jev_status_message = (
            f"语义整理：Jev {jev_count} · 本地回退 {fallback_count} · 迁移 {migrated} · "
            f"低置信保留 {uncertain} · 容量保留 {capacity_preserved}"
        )
        props.progress = 1.0
        props.eta_seconds = 0
        _status(props, "IDLE")
        self.report({"INFO"}, props.jev_status_message)
        return {"FINISHED"}


class WONDFUL_OT_copy_conversation_id(Operator):
    bl_idname = "wondful.copy_conversation_id"
    bl_label = "复制会话 ID"
    bl_description = "将当前 AI 会话 ID 及终端恢复命令复制到剪贴板"

    def execute(self, context):
        props = context.scene.wondful_ai
        cid = props.active_conversation_id.strip()
        if not cid:
            self.report({"WARNING"}, "当前尚无活跃的会话 ID。")
            return {"CANCELLED"}
        cmd = f"agy --conversation {cid}"
        from .clipboard_utils import set_clipboard_text
        set_clipboard_text(cmd)
        self.report({"INFO"}, f"已复制终端恢复命令：{cmd}")
        return {"FINISHED"}


class WONDFUL_OT_antigravity_auth_browser(Operator):
    bl_idname = "wondful.antigravity_auth_browser"
    bl_label = "打开本次登录页"

    def execute(self, context):
        url = _AGY_AUTH.browser_url() if _AGY_AUTH else ""
        if not url:
            self.report({"WARNING"}, "当前没有有效登录链接。")
            return {"CANCELLED"}
        webbrowser.open(url)
        return {"FINISHED"}


class WONDFUL_OT_antigravity_terminal(Operator):
    bl_idname = "wondful.antigravity_terminal"
    bl_label = "Google 登录"
    bl_description = "打开官方 AGY 交互终端，完成首次设置和 Google 登录；完成后回插件点验证登录"

    @classmethod
    def poll(cls, context):
        return not _ACTIVE_LOCK.locked()

    def execute(self, context):
        prefs, props = _addon_prefs(context), context.scene.wondful_ai
        try:
            cli = discover_antigravity_cli(prefs.antigravity_cli_path or props.antigravity_cli_resolved)
            message = open_antigravity_google_login(cli or "")
            props.antigravity_cli_resolved = cli or ""
        except Exception as exc:
            self.report({"ERROR"}, safe_diagnostic(str(exc)))
            return {"CANCELLED"}
        props.antigravity_account_message = message
        _status(props, "IDLE")
        self.report({"INFO"}, message)
        return {"FINISHED"}


class WONDFUL_OT_copy_diagnostics(Operator):
    bl_idname = "wondful.copy_diagnostics"
    bl_label = "复制诊断信息"

    def execute(self, context):
        import platform
        props, prefs = context.scene.wondful_ai, _addon_prefs(context)
        provider = _analysis_provider(props, prefs)
        data = {
            "addon": "3.1.3", "platform": platform.platform(),
            "blender": getattr(bpy.app, "version_string", "unknown"),
            "provider": provider["id"], "cli_path": provider["discover"]() or provider["cli_path"],
            "addon_path": str(Path(__file__).resolve().parent),
            "codex_version": props.codex_cli_version, "codex_state": props.codex_account_state,
            "agy_version": props.antigravity_cli_version, "agy_state": props.antigravity_account_state,
            "auth_stage": props.auth_stage, "auth_detail": safe_diagnostic(props.auth_detail),
            "image_tool": props.agy_image_capability if provider["id"] == "antigravity" else "RUNTIME_MANAGED",
            "agent_model": provider["model"] or "default", "profile": prompt_profiles.metadata(provider["id"], provider["model"]),
            "model_catalog_count": len(getattr(props, provider["id"] + "_models")),
            "model_catalog_status": safe_diagnostic(getattr(props, provider["id"] + "_models_message")),
            "reference_bundle": props.last_reference_bundle,
            "error": safe_diagnostic(props.last_error),
        }
        context.window_manager.clipboard = json.dumps(data, ensure_ascii=False, indent=2)
        self.report({"INFO"}, "已复制诊断信息（不含授权码或 OAuth 链接）。")
        return {"FINISHED"}


class WONDFUL_OT_polish_prompt(_BaseAsyncOperator):
    bl_idname = "wondful.polish_prompt"
    bl_label = "AI润色"
    bl_description = "根据你输入的提示词和产品／人物／环境参考图润色，写回同一个输入框；无需先设置相机"

    _session = None
    _width = 0
    _height = 0

    def execute(self, context):
        props = context.scene.wondful_ai
        prefs = _addon_prefs(context)

        provider = _analysis_provider(props, prefs)
        if not provider["discover"]():
            if provider["id"] == "antigravity":
                props.antigravity_account_state = "NOT_INSTALLED"
                props.antigravity_account_message = "未检测到 Antigravity CLI"
            else:
                props.codex_account_state = "NOT_INSTALLED"
                props.codex_account_message = "未检测到 Codex CLI"
            self.report({"ERROR"}, f"未检测到 {provider['label']} CLI。")
            return {"CANCELLED"}

        cleanup_old_sessions()
        self._session = create_session()
        # Appearance polish needs the user's words and references, not camera geometry.
        self._width, self._height, method = 0, 0, "TEXT_AND_REFERENCES"
        refs = _reference_paths(props)
        ref_instructions = _reference_instructions(props)
        product = copy_references(refs["product"], self._session, "product_reference")
        person = copy_references(refs["person"], self._session, "person_reference")
        style = copy_references(refs["style"], self._session, "style_reference")
        self._reference_counts = {"product": len(product), "person": len(person), "style": len(style)}
        product_upload = [prepare_reference_for_upload(p) for p in product]
        person_upload = [prepare_reference_for_upload(p) for p in person]
        style_upload = [prepare_reference_for_upload(p) for p in style]
        # Keep attachment order exactly aligned with prompt_engine reference numbering:
        # Product -> Person -> Environment; no Camera Base in the polish turn.
        image_paths = [*product_upload, *person_upload, *style_upload]
        camera_w, camera_h = camera_output_dimensions(context)
        self._camera_output_size = (camera_w, camera_h)
        self._camera_name_at_start = context.scene.camera.name if context.scene.camera else ""
        self._reference_sources_at_start = {kind: _reference_manifest(paths) for kind, paths in refs.items()}
        self._reference_instructions_at_start = ref_instructions
        self._style_revision_at_start = int(getattr(props, "style_reference_revision", 0))
        self._style_fingerprint_at_start = _reference_fingerprint(refs["style"])
        fingerprint_relevant = bool(len(props.style_images) > 0 or getattr(props, "style_sync_initialized", False))
        self._style_refresh_requested = bool(
            needs_reference_policy_refresh(props)
            or prompt_profiles.needs_target_refresh(props, prefs)
            or getattr(props, "style_prompt_dirty", False)
            or int(getattr(props, "prompt_style_revision", 0)) != self._style_revision_at_start
            or (fingerprint_relevant and str(getattr(props, "prompt_style_fingerprint", "")) != self._style_fingerprint_at_start)
            or (len(props.style_images) > 0 and not bool(getattr(props, "style_sync_initialized", False)))
        )

        visible_prompt = _get_prompt_text(props)
        self._prompt_at_start = visible_prompt
        props.prompt_notice = ""
        source_prompt = select_polish_source(props, visible_prompt, self._style_refresh_requested)
        self._prompt_profile = prompt_profiles.metadata(provider["id"], provider["model"])
        self._style_cache_key = prompt_profiles.style_cache_key(
            self._style_fingerprint_at_start, ref_instructions["style"], provider["id"], provider["model"])
        self._cached_style_summary = (getattr(props, "style_summary_cache", "")
            if getattr(props, "style_summary_cache_key", "") == self._style_cache_key else "")
        self._style_cache_hit = bool(self._cached_style_summary and style_upload)
        self._phase = "准备参考图"


        color_source, appearance_lock_text, identity_items = _appearance_context(props)
        self._appearance_meta = {"color_source": color_source, "appearance_lock": appearance_lock_text,
                                 "identity_items": identity_items}
        # 3.1.8: fill 产品外观 / 保留细节 from the product reference when the user
        # has not typed their own. The material summary is captured on the main thread.
        self._autofill_key = _product_autofill_key(props, _analysis_provider(props, prefs))
        self._autofill_needed = bool(product_upload) and _product_autofill_needed(props, self._autofill_key)
        self._autofill_result = None
        _material_text = appearance_lock_material_text(props) if self._autofill_needed else ""
        _jev_assets = getattr(props, "jev_identity_assets", "")

        def _build_user_text(lock_text, items):
            return build_polish_user_text(
                source_prompt, len(product), len(person), len(style), camera_w, camera_h,
                style_refresh=self._style_refresh_requested,
                include_camera=False,
                product_instructions=ref_instructions["product"],
                person_instructions=ref_instructions["person"],
                style_instructions=ref_instructions["style"],
                color_source=color_source,
                appearance_lock=lock_text,
                identity_items=items,
            )

        user_text = _build_user_text(appearance_lock_text, identity_items)

        _status(props, "POLISHING")
        props.progress = 0.05
        props.eta_seconds = 30
        props.last_session_id = self._session.session_id
        props.last_capture_method = method
        provider = _analysis_provider(props, prefs)
        timing_key = f"{provider['id']}_polish"
        self._provider_id = provider["id"]
        self._provider_label = provider["label"]
        self._timing_key = timing_key
        self._estimated_total = average_timing(timing_key, 30.0)
        if style_upload and not self._style_cache_hit:
            # Style refresh deliberately performs a dedicated first pass that sees
            # only the CURRENT style references. This mirrors the product spec:
            # "first analyze the style image, then merge those visual features into
            # the final prompt" and prevents legacy prompt text from winning.
            self._estimated_total *= 1.7
        cli_path = provider["cli_path"]
        model = provider["model"]
        polish_fn = provider["polish"]

        def job():
            final_user_text = user_text
            if self._autofill_needed:
                self._phase = "读取产品参考图外观"
                try:
                    reply = polish_fn(
                        system_prompt=PRODUCT_ANALYSIS_SYSTEM_PROMPT,
                        user_text=product_analysis_user_text(color_source, (ref_instructions["product"] or [""])[0]),
                        image_paths=product_upload[:1],
                        cwd=str(self._session.directory),
                        explicit_path=cli_path,
                        model=model,
                        timeout=int(provider["polish_timeout"]),
                    )
                    look, details = parse_product_analysis(reply)
                except Exception:
                    look, details = "", []
                if look or details:
                    self._autofill_result = (look, details)
                    from . import appearance_lock as _al
                    lock_text = _al.appearance_lock_block(color_source, _material_text, look)
                    items = _al.identity_checklist(ref_instructions["product"], _jev_assets, "，".join(details))
                    self._appearance_meta = {"color_source": color_source, "appearance_lock": lock_text,
                                             "identity_items": items, "autofilled": True}
                    final_user_text = _build_user_text(lock_text, items)
            # 3.1.3: Jev performs fast structured judgments about change scope,
            # identity preservation and appearance parameters before the heavier
            # provider polish. This does not replace Codex/AGY generation.
            from . import jev_semantics
            self._jev_appearance_meta = jev_semantics.analyze_appearance(
                source_prompt,
                list(ref_instructions.get("style", [])),
            )
            jev_synthesized = str(self._jev_appearance_meta.get("synthesized", "") or "").strip()
            if jev_synthesized:
                final_user_text += (
                    "\n\n【Jev 结构化外观判断｜作为约束，不替代参考图】\n"
                    + jev_synthesized
                )
            self._style_summary = self._cached_style_summary
            if style_upload and not self._style_cache_hit:
                self._phase = "分析当前风格 (1/2)"
                style_summary = polish_fn(
                    system_prompt=STYLE_ANALYSIS_SYSTEM_PROMPT,
                    user_text=(
                        "请只读取本轮附带的当前环境／风格参考图，从零提取环境照明与视觉风格。首先明确产品在该环境下的受光、反射和阴影。"
                        "不要参考 Blender 构图、产品身份、人物身份，也不要复用任何旧 Prompt 风格。"
                        + (("\n" + _instruction_lines("风格参考图", ref_instructions["style"])) if _instruction_lines("风格参考图", ref_instructions["style"]) else "")
                        + "\n环境主参考确定统一照明；逐图说明限定其他风格特征和辅助环境信息，不采用产品图的原有打光。"
                    ),
                    image_paths=style_upload,
                    cwd=str(self._session.directory),
                    explicit_path=cli_path,
                    model=model,
                    timeout=int(provider["polish_timeout"]),
                )
                self._style_summary = (style_summary or "").strip()
            if self._style_summary:
                final_user_text += (
                    "\n\n【当前风格参考图独立分析结果｜本轮唯一风格依据】\n"
                    + self._style_summary
                    + "\n请用这段当前风格摘要彻底替换旧提示词中的色调、材质、布光、摄影、氛围和后期描述；"
                      "旧风格与本摘要冲突时必须丢弃旧风格。"
                )

            self._phase = "适配生图模型提示词"
            kwargs = dict(
                system_prompt=prompt_profiles.system_prompt(BASE_SYSTEM_PROMPT, provider["id"]),
                user_text=final_user_text,
                image_paths=image_paths,
                cwd=str(self._session.directory),
                explicit_path=cli_path,
                model=model,
            )
            kwargs["timeout"] = int(provider["polish_timeout"])
            return polish_fn(**kwargs)

        if not self._start_thread(context, job):
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _modal(self, context, event):
        props = self._origin_scene.wondful_ai
        if event.type == "TIMER":
            if self._thread and self._thread.is_alive():
                self._progress_update(props, 0.10, 0.90)
                return {"RUNNING_MODAL"}
            self._cleanup(context)
            if self._error:
                exc, _tb = self._error
                if isinstance(exc, AntigravityNotLoggedInError):
                    props.antigravity_account_state = "LOGGED_OUT"
                    props.antigravity_account_message = str(exc)
                elif isinstance(exc, AntigravityNotInstalledError):
                    props.antigravity_account_state = "NOT_INSTALLED"
                    props.antigravity_account_message = str(exc)
                elif isinstance(exc, CodexNotLoggedInError):
                    props.codex_account_state = "LOGGED_OUT"
                    props.codex_account_message = str(exc)
                elif isinstance(exc, CodexNotInstalledError):
                    props.codex_account_state = "NOT_INSTALLED"
                    props.codex_account_message = str(exc)
                props.pending_generate = False
                _status(props, "ERROR", str(exc))
                props.progress = 0.0
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}

            if getattr(self, "_provider_id", "codex") == "antigravity":
                props.antigravity_account_state = "LOGGED_IN"
            else:
                props.codex_account_state = "LOGGED_IN"
            if getattr(self, "_autofill_result", None):
                _apply_product_autofill(props, self._autofill_key, *self._autofill_result)
            chain_render = bool(getattr(props, "pending_generate", False))
            props.pending_generate = False
            polished, removed_fragments = sanitize_appearance_prompt_report(self._result or "")
            props.prompt_removed_notice = (
                "已移除构图类描述（由 Blender 控制）：" + "；".join(removed_fragments)[:240]
                if removed_fragments else ""
            )
            jev_meta = getattr(self, "_jev_appearance_meta", {}) or {}
            props.jev_status = str(jev_meta.get("backend", "") or props.jev_status or "LOCAL_FALLBACK")
            if jev_meta:
                scope = (jev_meta.get("change_scope") or {}).get("choice", "")
                theme = (jev_meta.get("theme") or {}).get("choice", "")
                err = str(jev_meta.get("error", "") or "")
                props.jev_status_message = (
                    f"外观判断：{props.jev_status}"
                    + (f" · {scope}" if scope else "")
                    + (f" · {theme}" if theme else "")
                    + (f" · 回退原因 {err[:80]}" if err else "")
                )
            if not polished.strip():
                _status(props, "ERROR", "AI 未返回有效外观提示词，已保留原内容。")
                self.report({"ERROR"}, props.last_error)
                return {"CANCELLED"}
            if _get_prompt_text(props) != self._prompt_at_start:
                suggestion = self._session.directory / "prompt_ai_suggestion.txt"
                suggestion.write_text(polished, encoding="utf-8")
                draft = bpy.data.texts.new("Wondful Prompt · AI 润色建议")
                draft.write(polished)
                draft.use_fake_user = True
                props.prompt_notice = "已保留你的手动修改；AI 结果在 Text Editor 的“AI 润色建议”中。"
                props.progress = 1.0
                props.eta_seconds = 0
                _status(props, "IDLE")
                if chain_render:  # 1.2.2: the chain stops here on purpose; say so in ⑤
                    _status(props, "ERROR", "写提示词期间你修改了提示词：已保留你的修改，AI 版本在“AI 润色建议”里；"
                                            "没有自动生图，确认后点「生成」继续。")
                self.report({"WARNING"}, props.prompt_notice)
                return {"FINISHED"}
            _set_prompt_text(props, polished)
            final_prompt_now = _get_prompt_text(props)
            props.last_ai_prompt = final_prompt_now
            props.prompt_target_key = self._prompt_profile["target_key"]
            current_style_revision = int(getattr(props, "style_reference_revision", 0))
            current_style_fingerprint = _reference_fingerprint(_reference_paths(props)["style"])
            same_style_set = (
                current_style_revision == int(getattr(self, "_style_revision_at_start", current_style_revision))
                and current_style_fingerprint == str(getattr(self, "_style_fingerprint_at_start", current_style_fingerprint))
            )
            if same_style_set:
                props.prompt_style_revision = current_style_revision
                props.prompt_style_fingerprint = current_style_fingerprint
                props.style_prompt_dirty = False
                props.style_sync_initialized = True
                props.prompt_reference_policy_version = REFERENCE_POLICY_VERSION
                props.style_summary_cache_key = self._style_cache_key
                props.style_summary_cache = self._style_summary
            else:
                # The user changed style references (or replaced bytes at the same
                # path) while AI was polishing. Never validate a stale prompt.
                props.style_prompt_dirty = True
            props.progress = 1.0
            props.eta_seconds = 0
            _status(props, "IDLE")
            final_prompt = _get_prompt_text(props)
            self._session.prompt_file.write_text(final_prompt, encoding="utf-8")
            elapsed = time.time() - self._started_at
            record_timing(getattr(self, "_timing_key", "codex_polish"), elapsed)
            save_metadata(
                self._session,
                {
                    "type": "prompt_polish",
                    "prompt_profile": self._prompt_profile,
                    "appearance": getattr(self, "_appearance_meta", {}),
                    "style_cache_hit": self._style_cache_hit,
                    "engine": f"{getattr(self, '_provider_id', 'codex')}_oauth",
                    "analysis_provider": getattr(self, "_provider_id", "codex"),
                    "camera_reference_size": [self._width, self._height],
                    "camera_output_size": list(self._camera_output_size),
                    "camera_name": self._camera_name_at_start,
                    "reference_bundle": getattr(self, "_reference_bundle", {}),
                    "reference_counts": dict(self._reference_counts or {}),
                    "reference_sources": self._reference_sources_at_start,
                    "reference_instructions": self._reference_instructions_at_start,
                    "style_fingerprint": str(getattr(self, "_style_fingerprint_at_start", "")),
                    "style_summary": str(getattr(self, "_style_summary", "")),
                    "capture_method": props.last_capture_method,
                    "generation_time": elapsed,
                    "remote_generation_seconds": props.last_generation_seconds,
                    "remote_audit_seconds": props.last_audit_seconds,
                },
            )
            props.scene_brief_used = (getattr(props, "scene_brief", "") or "").strip()
            if props.style_prompt_dirty:
                self.report({"WARNING"}, "提示词已生成，但风格图在生成期间又发生变化；请再次重新生成提示词。")
                if chain_render:  # 1.2.2: never stop the 生成 chain without a visible reason
                    _status(props, "ERROR", "环境参考图在写提示词期间变了，没有自动生图。再点一次「生成」即可。")
            else:
                self.report({"INFO"}, f"{getattr(self, '_provider_label', 'Codex')} 提示词生成完成，可以开始 AI 渲染。")
                if chain_render and not getattr(props, "two_step", False):
                    props.task_phase = "提示词已完成，正在开始生图…"
                    _schedule_operator("wondful.ai_render", self._origin_scene)
            return {"FINISHED"}
        return {"PASS_THROUGH"}



class WONDFUL_OT_ai_render(_BaseAsyncOperator):
    bl_idname = "wondful.ai_render"
    bl_label = "AI渲染"
    bl_description = "重新捕获当前最新 Blender Scene Camera Frame，并由当前 AI Provider 完成最终生图"

    _session = None
    _width = 0
    _height = 0
    _target_w = 0
    _target_h = 0
    _reference_counts = None
    _structure = None

    @staticmethod
    def _variant_rank(variant: dict) -> tuple[bool, int]:
        canvas = variant.get("best_canvas") or {}
        try:
            score = int(variant.get("best_score", -1))
        except (TypeError, ValueError):
            score = -1
        return bool(canvas.get("matches") or canvas.get("fit_compatible")), score

    def _building_refs(self, items):
        """1.2.0 建筑参考: the blocks' reference images for this run (capped, larger blocks first),
        copied into the session and downscaled.  Never raises."""
        try:
            from . import massing as MS
            out = []
            for i, ref in enumerate(MS.select_building_refs(items, exists=lambda p: Path(p).is_file()), start=1):
                copied = copy_references([ref["path"]], self._session, f"building_ref_{ref['block']}_{i}")
                if copied:
                    out.append(dict(ref, path=prepare_reference_for_upload(copied[0], 1600)))
            return out
        except Exception as exc:
            print(f"[Wondful] 建筑参考整理失败：{exc}")
            return []

    def _layers_spec(self, context, props, prefs, provider):
        """1.2.0 分层输出 settings for this run, or None. Never raises."""
        try:
            if not bool(getattr(props, "layered_output", False)):
                return None
            if getattr(props, "product_collection", None) is None:
                props.last_layers_note = "分层输出需要先在 ② 选择产品集合，本次已跳过"
                return None
            from .product_kind import is_vehicle
            try:
                vehicle = bool(is_vehicle(props))
            except Exception:
                vehicle = False
            props.last_layers_note = ""
            _LAYER_BATCH.clear()
            return {
                "mask": str(Path(self._session.viewport_reference).parent / "01_product_mask.png"),
                "mode": "FULL" if getattr(props, "layered_mode", "SHADOW") == "FULL" else "SHADOW",
                "model": str(getattr(prefs, "codex_model", "") or ""),
                "client_version": str(getattr(props, "codex_cli_version", "") or ""),
                "work_dir": str(self._session.directory),
                "vehicle": vehicle,
                "long_edge": int(getattr(prefs, "long_edge", 2048) or 2048),
                "scene": context.scene.name,
            }
        except Exception as exc:
            print(f"[Wondful] 分层输出设置失败：{exc}")
            return None

    def _detail_spec(self, context, props):
        """1.2.0 细节回贴 settings for this run, or None. Never raises."""
        try:
            props.last_detail_note = ""
            props.last_score_note = ""
            if not bool(getattr(props, "detail_paste", True)):
                return None
            coll = getattr(props, "product_collection", None)
            if coll is None:
                props.last_detail_note = "细节回贴需要先在 ② 选择产品集合，本次已跳过"
                return None
            work = Path(self._session.viewport_reference).parent
            from .viewport_capture import keep_exact_slots, keep_exact_texts, mirrored_keep_parts
            slots = keep_exact_slots(coll)
            try:
                mirrored = mirrored_keep_parts(coll, getattr(getattr(context, "scene", None), "camera", None))
            except Exception as exc:
                print(f"[Wondful] 镜像贴图检查失败：{exc}")
                mirrored = []
            if mirrored:
                print(f"[Wondful] 细节回贴：这些标识在模型里就是镜像的：{mirrored}")
            return {
                "detail": str(work / "01_detail_pass.png"),
                "keep": str(work / "01_keep_mask.png"),
                "mask": str(work / "01_product_mask.png"),
                "strength": float(getattr(props, "detail_paste_strength", 0.8)),
                "matid": str(work / "01_material_id.png") if bool(getattr(props, "detail_colour_fix", True)) else "",
                "colour_strength": (float(getattr(props, "detail_colour_strength", 0.7))
                                    if bool(getattr(props, "detail_colour_fix", True)) else 0.0),
                "keep_slots": len(slots),
                "keep_texts": keep_exact_texts(coll),
                "mirrored": mirrored,
                "work_dir": str(self._session.directory),
                "reports": [],
            }
        except Exception as exc:
            print(f"[Wondful] 细节回贴设置失败：{exc}")
            return None

    def _apply_detail_paste(self, props, variant_final, variant_index):
        """Paste the detail pass onto one finished variant; returns (path to use, raw path or None)."""
        spec = getattr(self, "_detail", None)
        if not spec or not Path(spec["detail"]).is_file() or not Path(spec["mask"]).is_file():
            return variant_final, None
        try:
            from .detail_paste import mirrored_note, paste_file, summarize
            out = self._session.directory / f"output_variant_{variant_index:02d}_detail.png"
            matid = spec.get("matid", "") if spec.get("matid") and Path(spec["matid"]).is_file() else ""
            raw_score = self._identity_score(props, variant_final, spec, matid)
            rep = paste_file(str(variant_final), spec["detail"], spec["mask"], str(out),
                             keep_path=spec["keep"] if Path(spec["keep"]).is_file() else "",
                             strength=spec["strength"],
                             matid_path=spec.get("matid", "") if spec.get("matid") and Path(spec["matid"]).is_file() else "",
                             colour_strength=float(spec.get("colour_strength", 0.0) or 0.0))
            spec["reports"].append(rep)
            props.last_detail_note = summarize(rep) + mirrored_note(spec.get("mirrored"))
            if rep.get("skipped"):
                return variant_final, None
            self._identity_score(props, out, spec, matid, raw=raw_score, variant_index=variant_index)
            return out, Path(variant_final)
        except Exception as exc:
            print(f"[Wondful] 细节回贴失败：{exc}")
            props.last_detail_note = f"细节回贴失败（已保留原图）：{str(exc)[:80]}"
            return variant_final, None

    def _identity_score(self, props, image, spec, matid, raw=None, variant_index=None):
        """1.2.1 识别分: score the raw AI image (raw=None) or the pasted one (raw given: writes the
        panel line under 高级 › 上次渲染 and score_XX.json in the session). Never raises."""
        try:
            from .identity_score import score_file, summarize
            rep = score_file(str(image), spec["detail"], spec["mask"], matid, placement_of=raw)
            if raw is None:
                if not rep.get("error"):
                    props.last_score_note = summarize(rep)
                return rep
            props.last_score_note = summarize(raw, rep) if raw and not raw.get("error") else ""
            if variant_index is not None:
                import json
                path = Path(spec["work_dir"]) / f"score_{variant_index:02d}.json"
                path.write_text(json.dumps({"raw": raw, "final": rep}, ensure_ascii=False, indent=1, default=str),
                                encoding="utf-8")
            return rep
        except Exception as exc:
            print(f"[Wondful] 识别分计算失败：{exc}")
            return None

    def _export_gate(self, props, exported: Path, gate: dict, variant_index: int) -> Path:
        """1.2.3: panel line + ``<name>_iou.json`` for the acceptance; a result whose every try
        failed is renamed ``…_IoU未通过.png`` and the other candidates are kept next to it."""
        from .result_worker import gate_note
        try:
            note = gate_note(gate)
            count = int(getattr(self, "_variant_count", 1) or 1)
            props.last_iou_note = (f"第 {variant_index} 张：" + note) if count > 1 else note
            best = gate.get("best") or {}
            failed = best.get("verdict") == "未通过" or (gate.get("exhausted") and not gate.get("passed")
                                                       and best.get("verdict") != "待复核")
            if failed:
                target = exported.with_name(exported.stem + "_IoU未通过" + exported.suffix)
                shutil.move(str(exported), str(target))
                exported = target
            if failed or (gate.get("exhausted") and not gate.get("passed")):
                # nothing passed: every candidate is kept next to the shown (best-IoU) one
                best_src = str(best.get("candidate") or "")
                n = 1
                for cand in gate.get("candidates") or []:
                    if cand and Path(cand).is_file() and str(Path(cand)) != str(Path(best_src)):
                        n += 1
                        shutil.copy2(cand, exported.with_name(f"{exported.stem}_候选{n}{Path(cand).suffix or '.png'}"))
            exported.with_name(exported.stem + "_iou.json").write_text(
                json.dumps(gate, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        except Exception as exc:
            print(f"[Wondful] 轮廓验收结果保存失败：{exc}")
        return exported

    def _submit_layers(self, props, exported) -> None:
        """Queue the layer split of one exported image; failures only leave a panel note."""
        spec = getattr(self, "_layers", None)
        if not spec:
            return
        try:
            from . import layered_output
            if not Path(spec["mask"]).is_file():
                raise RuntimeError("产品遮罩未生成")
            plate_fn = None
            if spec["mode"] == "FULL":
                plate_fn = layered_output.make_plate_fn(
                    model=spec["model"], client_version=spec["client_version"], log_dir=spec["work_dir"],
                    vehicle=spec["vehicle"], long_edge=min(2048, max(1024, spec["long_edge"])))
            detail = (getattr(self, "_detail", None) or {}).get("detail", "")
            layered_output.RUNNER.submit(exported, spec["mask"], spec["mode"], plate_fn, work_dir=spec["work_dir"],
                                         detail_path=detail if detail and Path(detail).is_file() else "")
            _watch_layers(spec["scene"])
            props.last_layers_note = "分层输出处理中…" + ("（含干净背景，约 30–50 秒）" if plate_fn else "")
        except Exception as exc:
            print(f"[Wondful] 分层输出未启动：{exc}")
            props.last_layers_note = f"分层输出失败：{exc}"

    def _drain_completed_variants(self, props) -> None:
        """Aspect-fit and export every finished variant as soon as it is available."""
        queue = getattr(self, "_completed_variants", None)
        if queue is None:
            return
        camera_w, camera_h = self._camera_output_size
        while True:
            try:
                variant = queue.get_nowait()
            except Empty:
                break
            try:
                variant_index = int(variant.get("variant", 0))
                expected = int(getattr(self, "_variant_count", RENDER_VARIANT_COUNT))
                if variant_index < 1 or variant_index > expected:
                    raise ValueError(f"第 {variant_index} 张候选图编号无效。")
                if variant_index in self._exported_by_variant:
                    continue
                source = Path(str(variant.get("best_path", ""))).expanduser()
                if not source.is_file():
                    raise FileNotFoundError(f"第 {variant_index} 张候选图不存在: {source}")
                processed = variant.get("processed") or None
                if processed and Path(str(processed.get("final", ""))).is_file():
                    # 1.2.3: canvas fit, 细节回贴 and 产品还原度 already ran on the worker thread
                    variant_final = Path(processed["final"])
                    variant_raw = Path(processed["raw"]) if processed.get("raw") else None
                    if image_dimensions(variant_final) != (camera_w, camera_h):
                        raise RuntimeError(f"第 {variant_index} 张结果无法适配为 {camera_w}×{camera_h}。")
                    if getattr(self, "_detail", None):
                        props.last_detail_note = processed.get("detail_note", "")
                    props.last_score_note = processed.get("score_note", "")
                else:
                    variant_final = self._session.directory / f"output_variant_{variant_index:02d}.png"
                    variant_final, _variant_postprocessed = enforce_aspect_ratio(
                        str(source), str(variant_final), camera_w, camera_h
                    )
                    if image_dimensions(variant_final) != (camera_w, camera_h):
                        raise RuntimeError(f"第 {variant_index} 张结果无法适配为 {camera_w}×{camera_h}。")
                    identity_meta = getattr(self, "_identity_meta", {}) or {}
                    if getattr(self, "_identity_hard_restore", False) and identity_meta.get("enabled"):
                        from .identity_preserve import hard_restore_identity_pixels
                        restored_path = self._session.directory / f"output_variant_{variant_index:02d}_identity_restored.png"
                        variant_final = Path(hard_restore_identity_pixels(
                            variant_final,
                            getattr(self, "_identity_source_path", ""),
                            identity_meta.get("path", ""),
                            restored_path,
                        ))
                    variant_final, variant_raw = self._apply_detail_paste(props, variant_final, variant_index)
                exported = export_result(
                    variant_final,
                    self._output_dir_at_start,
                    self._scene_name_at_start,
                    self._camera_name_at_start,
                    self._session.session_id,
                )
                gate = variant.get("iou_gate")
                if gate:
                    exported = self._export_gate(props, Path(exported), gate, variant_index)
                if variant_raw is not None:
                    try:
                        from .detail_paste import raw_path
                        shutil.copy2(variant_raw, raw_path(exported))
                    except Exception as exc:
                        print(f"[Wondful] 原始结果另存失败：{exc}")
                self._exported_by_variant[variant_index] = Path(exported)
                self._exported_variant_records[variant_index] = variant
                self._submit_layers(props, Path(exported))
            except Exception as exc:
                self._export_errors.append(str(exc))

        if not self._exported_by_variant:
            return
        best_index = max(
            self._exported_variant_records,
            key=lambda index: self._variant_rank(self._exported_variant_records[index]),
        )
        exported_paths = [
            str(self._exported_by_variant[index])
            for index in sorted(self._exported_by_variant)
        ]
        best_exported = self._exported_by_variant[best_index]
        props.variant_count = len(exported_paths)
        props.last_export_path = str(best_exported)
        props.last_result_path = str(best_exported)
        props.last_export_paths = json.dumps(exported_paths, ensure_ascii=False)
        try:
            props.result_image = load_image(str(best_exported), "Wondful_Result")
        except Exception:
            pass

    def execute(self, context):
        self._request_started_at = time.perf_counter()
        props = context.scene.wondful_ai
        prefs = _addon_prefs(context)
        provider = _analysis_provider(props, prefs)
        planned_variant_count = render_variant_count(provider["id"], props)
        if not provider["imagegen_enabled"]:
            self.report({"ERROR"}, f"{provider['label']} 生图已在插件偏好设置中关闭。")
            return {"CANCELLED"}

        if not provider["discover"]():
            if provider["id"] == "antigravity":
                props.antigravity_account_state = "NOT_INSTALLED"
                props.antigravity_account_message = "未检测到 Antigravity CLI"
            else:
                props.codex_account_state = "NOT_INSTALLED"
                props.codex_account_message = "未检测到 Codex CLI"
            self.report({"ERROR"}, f"未检测到 {provider['label']} CLI。")
            return {"CANCELLED"}

        if prompt_profiles.needs_target_refresh(props, prefs):
            self.report({"WARNING"}, "生图引擎或提示词规则已变化，请先点‘适配当前引擎’重新生成提示词。")
            return {"CANCELLED"}
        self._prompt_profile = prompt_profiles.metadata(provider["id"], provider["model"])
        current_prompt = _get_prompt_text(props)
        if not current_prompt:
            self.report({"ERROR"}, "提示词为空。可以先输入简短需求，或点击重新生成提示词。")
            return {"CANCELLED"}

        current_style_revision = int(getattr(props, "style_reference_revision", 0))
        prompt_style_revision = int(getattr(props, "prompt_style_revision", 0))
        current_style_fingerprint = _reference_fingerprint(_reference_paths(props)["style"])
        self._style_fingerprint_at_render = current_style_fingerprint
        fingerprint_relevant = bool(len(props.style_images) > 0 or getattr(props, "style_sync_initialized", False))
        policy_needs_refresh = needs_reference_policy_refresh(props)
        style_needs_refresh = bool(
            policy_needs_refresh
            or getattr(props, "style_prompt_dirty", False)
            or prompt_style_revision != current_style_revision
            or (fingerprint_relevant and str(getattr(props, "prompt_style_fingerprint", "")) != current_style_fingerprint)
            or (len(props.style_images) > 0 and not bool(getattr(props, "style_sync_initialized", False)))
        )
        if style_needs_refresh:
            message = ("参考职责已更新为产品造型、环境打光，请先重新生成提示词。" if policy_needs_refresh
                       else "环境／风格参考图尚未与当前提示词同步，请先重新生成提示词。")
            self.report({"ERROR"}, message)
            return {"CANCELLED"}

        # 2.19: Structure Lock no longer infers product meshes from the whole scene
        # or from red materials. When structure guides are enabled, one explicit
        # product Collection is required and every visible Mesh inside it is product.
        if prefs.structure_guides_enabled and prefs.strict_composition_lock and not getattr(props, "product_collection", None):
            self.report({"ERROR"}, "Blender Structure Lock 已开启，请先选择“产品集合”。集合里的 Mesh 将全部视为产品。")
            return {"CANCELLED"}

        configured_output = (props.output_directory or "").strip() or (prefs.output_directory or "").strip()
        try:
            self._output_dir_at_start = resolve_output_directory(context, configured_output)
            preflight_output_directory(self._output_dir_at_start)
        except Exception as exc:
            self.report({"ERROR"}, f"输出目录不可用：{exc}")
            return {"CANCELLED"}
        self._scene_name_at_start = context.scene.name
        self._camera_name_at_start = context.scene.camera.name if context.scene.camera else "Camera"
        cleanup_old_sessions()
        self._session = create_session()
        try:
            self._width, self._height, method = capture_camera_reference(context, str(self._session.viewport_reference))
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        refs = _reference_paths(props)
        ref_instructions = _reference_instructions(props)
        product = copy_references(refs["product"], self._session, "product_reference")
        person = copy_references(refs["person"], self._session, "person_reference")
        style = copy_references(refs["style"], self._session, "style_reference")
        self._reference_counts = {"product": len(product), "person": len(person), "style": len(style)}
        product_upload = [prepare_reference_for_upload(p) for p in product]
        person_upload = [prepare_reference_for_upload(p) for p in person]
        style_upload = [prepare_reference_for_upload(p) for p in style]

        camera_w, camera_h = camera_output_dimensions(context)
        self._camera_output_size = (camera_w, camera_h)
        self._reference_sources_at_start = {kind: _reference_manifest(paths) for kind, paths in refs.items()}
        self._reference_instructions_at_start = ref_instructions
        render_mode = str(getattr(props, "render_mode", "STANDARD") or "STANDARD")
        structure = {"enabled": False}
        if prefs.structure_guides_enabled and prefs.strict_composition_lock:
            try:
                structure = build_structure_packet(
                    context,
                    props,
                    self._session.directory,
                    camera_w,
                    camera_h,
                    camera_base_path=str(self._session.viewport_reference),
                    render_mode=render_mode,
                )
            except Exception as exc:
                structure = {"enabled": False, "error": str(exc)}
            if not structure.get("enabled"):
                reason = structure.get("reason") or structure.get("source_mode") or structure.get("error") or "UNKNOWN"
                self.report({"ERROR"}, f"产品集合无法生成 Structure Packet：{reason}。请确认集合中有当前相机可见的 Mesh。")
                return {"CANCELLED"}
        self._structure = structure
        # Collect Blender state on the main thread. The worker may call Jev using
        # this JSON-safe snapshot without touching bpy from a background thread.
        from . import jev_semantics
        self._semantic_states = jev_semantics.collect_product_object_states(
            context, props, structure.get("part_id_manifest", {}) if structure.get("enabled") else {}
        )
        props.last_structure_note = (
            "结构图：原生相机数据，已统一白模画布。" if structure.get("native_passes") else
            "结构图已降级为投影近似；遮挡和深度需人工检查。" if structure.get("enabled") else
            "未启用结构图，仅使用白模参考。"
        )
        effective_long_edge = min(int(prefs.long_edge), 1280) if render_mode == "FAST" else int(prefs.long_edge)
        self._render_mode = render_mode
        self._effective_long_edge = effective_long_edge
        self._target_w, self._target_h, target_size = compute_target_size(camera_w, camera_h, effective_long_edge)
        color_source, appearance_lock_text, identity_items = _appearance_context(props)
        if structure.get("enabled") and identity_items:
            structure["identity_checklist"] = list(identity_items)
        render_prompt = build_render_prompt(
            current_prompt,
            camera_w,
            camera_h,
            len(product),
            len(person),
            len(style),
            target_size=target_size,
            strict_lock=bool(prefs.strict_composition_lock),
            structure_guide=bool(structure.get("enabled")),
            structure_packet=structure if structure.get("enabled") else None,
            product_instructions=ref_instructions["product"],
            person_instructions=ref_instructions["person"],
            style_instructions=ref_instructions["style"],
            color_source=color_source,
            appearance_lock=appearance_lock_text,
            identity_items=identity_items,
        )
        render_prompt = prompt_profiles.render_brief(render_prompt, provider["id"])
        structure_refs = list(structure.get("generation_reference_paths", [])) if structure.get("enabled") else []
        camera_base = str(structure.get("camera_base_path") or self._session.viewport_reference)
        # Two free slots are needed if a later strict repair adds a candidate and
        # union Mask. Compose sheets on Blender's main thread, before the worker.
        reference_limit = int(provider.get("reference_limit", MAX_IMAGE_PATHS))
        # Codex currently has room for a two-image masked-repair reserve. AGY's
        # generate_image endpoint accepts only three paths total, so its repair
        # pass is reduced to the camera + edit base + mask below instead of
        # reserving slots that would make the initial bundle impossible.
        identity_preserve_enabled = bool(getattr(prefs, "identity_preserve_enabled", True))
        identity_preserve_padding = int(getattr(prefs, "identity_preserve_padding", 2))
        identity_hard_restore = bool(getattr(prefs, "identity_hard_restore", False))
        identity_possible = bool(identity_preserve_enabled and structure.get("part_index_path"))
        repair_reserve = 2 if (reference_limit >= 5 and render_mode == "STRICT" and prefs.auto_alignment_retry
                               and prefs.masked_alignment_repair and structure.get("edit_mask_path")
                               and int(prefs.alignment_max_attempts) > 1) else 0
        identity_reserve = 1 if identity_possible else 0
        reserve = max(repair_reserve, identity_reserve)
        try:
            references, bundle_note, self._reference_bundle = prepare_bundle(
                [
                    ("camera", [camera_base]),
                    ("structure", structure_refs, list(structure.get("generation_reference_roles", []))),
                    ("product", product_upload),
                    ("person", person_upload),
                    ("environment", style_upload),
                ],
                self._session.directory, reserve=reserve, limit=reference_limit,
            )
            render_prompt += bundle_note
            self._session.prompt_file.write_text(render_prompt, encoding="utf-8")
            (self._session.directory / "reference_bundle.json").write_text(
                json.dumps(self._reference_bundle, ensure_ascii=False, indent=2), encoding="utf-8")
            props.last_reference_bundle = json.dumps({k: v for k, v in self._reference_bundle.items()
                                                     if k != "groups"}, ensure_ascii=False)
        except Exception as exc:
            _status(props, "ERROR", f"参考图整理失败：{exc}")
            self.report({"ERROR"}, props.last_error)
            return {"CANCELLED"}

        self._preparation_seconds = time.perf_counter() - self._request_started_at
        props.last_preparation_seconds = self._preparation_seconds
        props.last_total_seconds = 0.0
        _status(props, "RENDERING")
        props.last_export_path = ""
        props.last_export_paths = ""
        props.variant_count = 1
        props.alignment_score = -1.0
        props.alignment_message = ""
        props.last_canvas_warning = ""
        props.last_canvas_fit = False
        props.alignment_attempts = 0
        props.progress = 0.05
        props.eta_seconds = 90 * planned_variant_count
        props.last_session_id = self._session.session_id
        props.last_capture_method = method
        # Speed policy (2.12): FAST/STANDARD skip remote vision audit; each mode
        # still produces the provider-specific independent image variants.
        auto_alignment_retry = bool(render_mode == "STRICT" and prefs.auto_alignment_retry)
        alignment_threshold = int(prefs.alignment_threshold)
        provider = _analysis_provider(props, prefs)
        generate_fn = provider["generate"]
        audit_fn = provider["audit"]
        # 3.2.1: Codex 图像编辑（直连）— edit the camera base with the hosted
        # image_generation tool; any failure falls back to the codex exec agent.
        self._direct_state = None
        props.last_engine_note = ""
        self._layers = self._layers_spec(context, props, prefs, provider)
        self._detail = self._detail_spec(context, props)
        product_mask_path = (self._layers or {}).get("mask", "") or (self._detail or {}).get("mask", "")
        if provider["id"] == "codex" and bool(getattr(prefs, "codex_direct_edit", True)):
            from . import codex_direct
            from .cli_transport import is_cancellation_requested as _cancel_requested
            direct_state = {}
            self._direct_state = direct_state
            direct_size = codex_direct.choose_edit_size(camera_w, camera_h, effective_long_edge)
            direct_manifest = list((self._reference_bundle or {}).get("groups", []))
            direct_note = bundle_note
            direct_version = str(getattr(props, "codex_cli_version", "") or "")
            exec_generate = generate_fn
            direct_simple_base = ""
            direct_simple_brief = None
            direct_terrain_text = ""
            direct_massing_text = ""
            direct_building_refs = []
            from .product_kind import is_vehicle as _is_vehicle
            try:
                direct_vehicle = _is_vehicle(props)
            except Exception:
                direct_vehicle = False
            direct_keep_text = ""
            if self._detail and (self._detail.get("keep_slots") or self._detail.get("keep_texts")):
                from .detail_paste import keep_prompt
                direct_keep_text = keep_prompt(direct_vehicle, self._detail.get("keep_texts") or ())
            direct_state["vehicle"] = direct_vehicle
            if bool(getattr(prefs, "codex_direct_simple", True)):
                clay_path = str(Path(self._session.viewport_reference).parent / "01_clay_base.png")
                try:
                    from .viewport_capture import render_clay_base, LAST_CLAY_INFO
                    from .terrain import settings_from_props
                    from .massing import find_collection as _find_massing
                    massing_coll = _find_massing(context.scene, props)
                    if render_clay_base(context, clay_path, getattr(props, "product_collection", None),
                                        ground=getattr(props, "clay_ground", "AUTO"),
                                        crest_angle=float(getattr(props, "clay_crest_angle", 18.0)),
                                        terrain=settings_from_props(props),
                                        mask_path=product_mask_path,
                                        massing=massing_coll,
                                        detail_path=(self._detail or {}).get("detail", ""),
                                        keep_path=(self._detail or {}).get("keep", ""),
                                        matid_path=(self._detail or {}).get("matid", "")):
                        direct_simple_base = clay_path
                        direct_terrain_text = str(LAST_CLAY_INFO.get("terrain_text", "") or "")
                        direct_massing_text = str(LAST_CLAY_INFO.get("massing_text", "") or "")
                        blocks = len(LAST_CLAY_INFO.get("massing") or [])
                        if direct_massing_text:
                            direct_building_refs = self._building_refs(LAST_CLAY_INFO.get("massing") or [])
                        props.last_massing_note = (f"环境体块：{massing_coll.name} · {blocks} 个体块进入画面"
                                                   + (f" · 建筑参考 {len(direct_building_refs)} 张" if direct_building_refs else "")
                                                   if massing_coll else "")
                        direct_simple_brief = (getattr(props, "scene_brief", "") or "").strip()
                except Exception as exc:
                    print(f"[Wondful] 白模底图渲染失败，改用相机截图：{exc}")
                if direct_simple_brief is None and Path(self._session.viewport_reference).is_file():
                    direct_simple_base = str(self._session.viewport_reference)
                    direct_simple_brief = (getattr(props, "scene_brief", "") or "").strip()
            direct_state["simple_base"] = Path(direct_simple_base).name if direct_simple_base else ""
            direct_state["building_refs"] = [{"block": r["block"], "file": Path(r["path"]).name}
                                             for r in direct_building_refs]

            def _direct_status(text):
                self._phase = text

            def generate_fn(**kwargs):
                return codex_direct.generate_with_fallback(
                    kwargs, fallback=exec_generate, state=direct_state, size=direct_size,
                    manifest=direct_manifest, bundle_note=direct_note, client_version=direct_version,
                    on_status=_direct_status, cancel_check=_cancel_requested,
                    simple_base=direct_simple_base, simple_brief=direct_simple_brief,
                    terrain_text=direct_terrain_text, vehicle=direct_vehicle,
                    massing_text=direct_massing_text,
                    building_refs=direct_building_refs or None,
                    keep_text=direct_keep_text,
                )
        if self._detail and not Path(self._detail["detail"]).is_file():
            # no clay base this run (other canvas / provider): render the detail pass on its own
            try:
                from .viewport_capture import render_detail_pass, render_product_mask
                from .massing import find_collection as _find_massing
                coll = getattr(props, "product_collection", None)
                mcoll = _find_massing(context.scene, props)
                info = render_detail_pass(context, self._detail["detail"], coll, keep_path=self._detail["keep"],
                                          massing=mcoll, matid_path=self._detail.get("matid", ""))
                if not info.get("detail"):
                    raise RuntimeError("细节底图未生成")
                if not Path(self._detail["mask"]).is_file() and not render_product_mask(
                        context, self._detail["mask"], coll, massing=mcoll):
                    raise RuntimeError("产品遮罩未生成")
            except Exception as exc:
                print(f"[Wondful] 细节回贴：细节底图渲染失败：{exc}")
                props.last_detail_note = f"细节回贴已跳过：细节底图渲染失败（{exc}）"
                self._detail = None
        if self._layers and not Path(self._layers["mask"]).is_file():
            try:
                from .viewport_capture import render_product_mask
                from .massing import find_collection as _find_massing
                if not render_product_mask(context, self._layers["mask"], getattr(props, "product_collection", None),
                                           massing=_find_massing(context.scene, props)):
                    raise RuntimeError("产品遮罩未生成")
            except Exception as exc:
                print(f"[Wondful] 分层输出：产品遮罩渲染失败：{exc}")
                props.last_layers_note = f"分层输出失败：产品遮罩渲染失败（{exc}）"
                self._layers = None
        # 1.2.3 轮廓验收 + 有界重生成: fixed-coordinate IoU vs the Blender product mask after every
        # generation; the first one that passes stops; at most 3 generations per image.
        from . import result_worker
        gate_mask = ""
        props.last_iou_note = ""
        if bool(getattr(props, "iou_gate", True)):
            gate_max = max(1, min(result_worker.GATE_MAX, int(getattr(props, "iou_gate_max", 3) or 3)))
            cand_mask = (self._detail or {}).get("mask", "") or (self._layers or {}).get("mask", "")
            if cand_mask and Path(cand_mask).is_file():
                gate_mask = cand_mask
            elif getattr(props, "product_collection", None) is not None:
                try:
                    from .viewport_capture import render_product_mask
                    from .massing import find_collection as _find_massing
                    path = str(Path(self._session.viewport_reference).parent / "01_product_mask.png")
                    if Path(path).is_file() or render_product_mask(
                            context, path, getattr(props, "product_collection", None),
                            massing=_find_massing(context.scene, props)):
                        gate_mask = path
                except Exception as exc:
                    print(f"[Wondful] 轮廓验收：产品遮罩渲染失败：{exc}")
            if not gate_mask:
                props.last_iou_note = "轮廓验收需要先在 ② 选择产品集合，本次未验收、未自动重生成"
        else:
            gate_max = 1
        self._gate_mask = gate_mask
        # numpy result processing runs on the worker; the old main-thread path stays for the
        # opt-in identity hard restore (it needs Blender's image loader)
        worker_post = not (identity_hard_restore and identity_possible)
        session_dir = str(self._session.directory)
        provider_cli_path = provider["cli_path"]
        provider_model = provider["model"]
        provider_id = provider["id"]
        provider_label = provider["label"]
        render_timeout = int(provider["render_timeout"])
        audit_timeout = int(provider["audit_timeout"])
        planned_attempts = max(1, int(prefs.alignment_max_attempts if auto_alignment_retry else 1))
        if gate_mask:
            planned_attempts = gate_max     # regenerations are decided by the IoU gate only
        self._provider_id = provider_id
        self._provider_label = provider_label
        timing_suffix = "strict" if auto_alignment_retry else ("fast" if render_mode == "FAST" else "standard")
        variant_count = render_variant_count(provider_id, props)
        self._variant_count = variant_count
        self._completed_variants = Queue()
        self._exported_by_variant = {}
        self._exported_variant_records = {}
        self._export_errors = []
        # Store a separate total-duration bucket for each output count; otherwise
        # a previous single-image average would be multiplied again after a
        # multi-image run.
        self._timing_key = f"{provider_id}_render_{timing_suffix}_x{self._variant_count}"
        self._estimated_total = average_timing(
            self._timing_key,
            (180.0 if render_mode != "FAST" else 120.0) * self._variant_count,
        )
        edit_mask_path = str(structure.get("edit_mask_path", "")) if structure.get("enabled") else ""
        masked_repair = bool(prefs.masked_alignment_repair and edit_mask_path)
        self._identity_hard_restore = identity_hard_restore
        self._identity_source_path = camera_base
        self._identity_meta = {}

        def job():
            from . import jev_semantics
            from .cli_transport import is_cancellation_requested
            decision_meta = jev_semantics.analyze_appearance(
                current_prompt,
                list(ref_instructions.get("style", [])),
            )
            structural_probability = decision_meta.get("structural_change_probability")
            if isinstance(structural_probability, (int, float)) and structural_probability >= 0.80:
                scope = (decision_meta.get("change_scope") or {}).get("choice", "STRUCTURAL_CHANGE")
                raise RuntimeError(
                    f"检测到 {scope} 请求。请先在 Blender 中修改几何/相机/构图，再重新执行 AI 渲染；"
                    "生图模型不会替代 Blender 结构修改。"
                )
            preserve_probability = decision_meta.get("preserve_identity_probability")
            preserve_identity = not (
                isinstance(preserve_probability, (int, float)) and preserve_probability <= 0.20
            )
            if is_cancellation_requested():
                raise RuntimeError("任务已取消")

            semantic_results = jev_semantics.resolve_object_semantics_batch(
                list(getattr(self, "_semantic_states", []))
            )
            semantic_block = jev_semantics.semantic_prompt_block(
                semantic_results,
                structure.get("part_id_manifest", {}) if structure.get("enabled") else {},
                preserve_identity=preserve_identity,
            )
            effective_render_prompt = render_prompt + semantic_block
            semantic_meta = jev_semantics.semantic_summary(semantic_results)
            semantic_meta["decision_backend"] = decision_meta.get("backend", "")
            semantic_meta["preserve_identity"] = preserve_identity
            semantic_meta["change_scope"] = (decision_meta.get("change_scope") or {}).get("choice", "")
            semantic_meta["decision_usage"] = decision_meta.get("usage", {})

            from .identity_preserve import build_identity_preserve_mask, write_identity_edit_mask
            identity_meta = {"enabled": False, "reason": "DISABLED", "path": ""}
            identity_full_edit_mask = ""
            if identity_preserve_enabled and preserve_identity and structure.get("part_index_path"):
                identity_meta = build_identity_preserve_mask(
                    structure.get("part_index_path", ""),
                    semantic_results,
                    self._session.directory / "identity_preserve_mask.png",
                    padding_px=identity_preserve_padding,
                )
                if identity_meta.get("enabled"):
                    identity_full_edit_mask = write_identity_edit_mask(
                        self._session.directory / "identity_edit_mask.png",
                        identity_meta["mask_npy_path"],
                    )
                    effective_render_prompt += (
                        "\n\n【Identity Preserve Mask｜最高优先级身份保护】\n"
                        "本轮编辑 Mask 中的不透明区域是 Logo、品牌/型号字标、铭牌/车牌文字或可读 UI 等身份资产。"
                        "这些区域不得重新绘制、改字、改形、改比例或风格化；仅透明区域允许外观编辑。"
                    )
            semantic_meta["identity_mask"] = identity_meta
            self._identity_meta = identity_meta
            if is_cancellation_requested():
                raise RuntimeError("任务已取消")

            attempts = []
            variants = []
            total_generation_seconds = 0.0
            total_audit_seconds = 0.0
            best_score = -1
            best_path = None
            best_audit = None
            best_canvas = None
            best_variant = None
            best_rank = (False, -1)
            for variant_index in range(1, variant_count + 1):
                variant_attempts = []
                variant_best_score = -1
                variant_best_path = None
                variant_best_audit = None
                variant_best_canvas = None
                variant_best_rank = (False, -1)
                correction = ""
                for attempt_index in range(1, planned_attempts + 1):
                    candidate = self._session.directory / f"output_candidate_v{variant_index:02d}_a{attempt_index:02d}.png"
                    attempt_prompt = effective_render_prompt + (
                        f"\n\n这是本轮{variant_count}张独立候选图中的第 {variant_index} 张。保持 Blender 规定的构图、几何、位置、尺度和视角不变；"
                        "允许材质细节、反射、光影和环境氛围产生自然变化，避免机械复制其他候选图。"
                    )
                    if correction:
                        attempt_prompt += (
                            "\n\n上一轮构图未通过白模对齐校验。保持用户指定的外观和环境打光，"
                            "按本轮提供的底图和修复范围恢复白模构图。具体纠偏：" + correction
                        )

                    edit_base, edit_mask = camera_base, identity_full_edit_mask
                    repair_strategy = ("IDENTITY_PRESERVE_MASK" if identity_full_edit_mask else
                                       ("WHITE_MODEL_REBASE" if correction else "WHITE_MODEL"))
                    region = None
                    if correction and variant_attempts and masked_repair:
                        previous = variant_attempts[-1]
                        region = repair_region(structure, previous["audit"],
                                               previous["canvas"].get("actual_size"),
                                               (camera_w, camera_h), alignment_threshold)
                        if region is not None:
                            edit_base = previous["path"]
                            if identity_meta.get("enabled"):
                                edit_mask = write_identity_edit_mask(
                                    self._session.directory / f"repair_identity_mask_v{variant_index:02d}_a{attempt_index:02d}.png",
                                    identity_meta["mask_npy_path"],
                                    editable_region=region,
                                    output_size=tuple(previous["canvas"].get("actual_size") or (camera_w, camera_h)),
                                )
                                repair_strategy = "UNION_MASK_PLUS_IDENTITY_PRESERVE"
                            else:
                                edit_mask = write_repair_mask(
                                    self._session.directory / f"repair_mask_v{variant_index:02d}_a{attempt_index:02d}.png",
                                    previous["canvas"]["actual_size"], region,
                                )
                                repair_strategy = "UNION_MASK"
                    if correction:
                        attempt_prompt += (
                            "\n本轮 Mask 同时覆盖偏移主体与目标位置；先清除旧位置残影，再在白模位置恢复主体。"
                            if region is not None else
                            "\n本轮重新使用原始白模整幅画布，不继承上一轮错误的比例、裁切或视角。"
                        )
                    generation_started = time.time()
                    self._phase = f"生成第 {variant_index}/{variant_count} 张 · 尝试 {attempt_index}/{planned_attempts}"
                    generation_references = references
                    if provider_id == "antigravity" and edit_mask and region is not None:
                        # AGY accepts three ImagePaths total. A masked repair already
                        # has edit_base + mask, so keep the authoritative camera only
                        # and let the previous candidate carry the appearance state.
                        generation_references = [camera_base]
                    try:
                        generation = generate_fn(
                            prompt=attempt_prompt,
                            reference_paths=generation_references,
                            output_path=str(candidate),
                            cwd=str(self._session.directory),
                            explicit_path=provider_cli_path,
                            model=provider_model,
                            edit_base_path=edit_base,
                            edit_mask_path=edit_mask,
                            structure_control=structure if structure.get("enabled") else None,
                            timeout=render_timeout,
                        )
                    except Exception:
                        # If a later strict repair is rate-limited (or the quota ran out), the
                        # previous candidate for this variant is still a completed output.
                        if variant_best_path is not None and variant_best_path.is_file():
                            partial = {
                                "variant": variant_index,
                                "best_path": str(variant_best_path),
                                "best_score": variant_best_score,
                                "best_audit": variant_best_audit,
                                "best_canvas": variant_best_canvas,
                                "attempts": list(variant_attempts),
                                "partial": True,
                            }
                            if gate_mask:
                                partial["iou_gate"] = _gate_summary(variant_attempts, False)
                            if worker_post:
                                try:
                                    partial["processed"] = result_worker.finish_variant(
                                        str(variant_best_path), variant_index, self._detail, camera_w, camera_h,
                                        session_dir)
                                except Exception as exc:
                                    print(f"[Wondful] 结果处理失败（改在主线程处理）：{exc}")
                            self._completed_variants.put(partial)
                        raise
                    generation_seconds = time.time() - generation_started
                    total_generation_seconds += generation_seconds
                    canvas = assess_canvas(image_dimensions(candidate), (camera_w, camera_h))

                    audit = None
                    score = -1
                    audit_error = ""
                    audit_seconds = 0.0
                    if auto_alignment_retry:
                        try:
                            if canvas["matches"] or canvas.get("fit_compatible"):
                                audit_started = time.time()
                                self._phase = f"检查第 {variant_index}/{variant_count} 张构图"
                                audit = audit_fn(
                                    camera_reference=camera_base,
                                    generated_image=str(candidate),
                                    cwd=str(self._session.directory),
                                    explicit_path=provider_cli_path,
                                    model=provider_model,
                                    tag=f"{variant_index}-{attempt_index}",
                                    structure_control=structure if structure.get("enabled") else None,
                                    structure_guide_path=str(structure.get("guide_path", "")) if structure.get("enabled") else "",
                                    timeout=audit_timeout,
                                )
                                audit_seconds = time.time() - audit_started
                                total_audit_seconds += audit_seconds
                            else:
                                audit = {"overall_score": 0, "passed": False, "source": "LOCAL_CANVAS_CHECK"}
                            audit = assess_audit(audit, structure, canvas, alignment_threshold)
                            score = int(audit.get("overall_score", -1))
                        except Exception as exc:
                            audit_error = str(exc)

                    gate_rep = None
                    if gate_mask:
                        self._phase = f"验收第 {variant_index}/{variant_count} 张轮廓 · 第 {attempt_index}/{planned_attempts} 次"
                        gate_rep = result_worker.gate_score(
                            str(candidate), gate_mask, camera_w, camera_h, session_dir,
                            f"v{variant_index:02d}_a{attempt_index:02d}", cancel_check=is_cancellation_requested,
                            detail_path=(self._detail or {}).get("detail", ""))
                        print(f"[Wondful] 轮廓验收 v{variant_index} a{attempt_index}: IoU {gate_rep.get('iou')} "
                              f"{gate_rep.get('verdict')}")

                    record = {
                        "iou_gate": gate_rep,
                        "variant": variant_index,
                        "attempt": attempt_index,
                        "path": str(candidate),
                        "score": score,
                        "audit": audit,
                        "audit_error": audit_error,
                        "canvas": canvas,
                        "repair_strategy": repair_strategy,
                        "repair_region": region,
                        "generation": generation,
                        "generation_seconds": generation_seconds,
                        "audit_seconds": (audit_seconds if auto_alignment_retry and audit is not None else 0.0),
                    }
                    attempts.append(record)
                    variant_attempts.append(record)

                    # A verified or aspect-fit-compatible canvas outranks an incompatible
                    # canvas even if remote audit failed.
                    rank = (bool(canvas["matches"] or canvas.get("fit_compatible")), score)
                    if gate_mask:
                        rank = (bool(canvas["matches"] or canvas.get("fit_compatible")),) + result_worker.gate_rank(gate_rep)
                    if variant_best_path is None or rank > variant_best_rank:
                        variant_best_path = candidate
                        variant_best_score = score
                        variant_best_audit = audit
                        variant_best_canvas = canvas
                        variant_best_rank = rank

                    if gate_mask:
                        if gate_rep is None or gate_rep.get("iou") is None:
                            break       # cannot judge this file: never burn quota blindly
                        if gate_rep.get("passed"):
                            break       # the first generation that passes stops
                        continue        # fresh regeneration from the original base, never a repair
                    if not auto_alignment_retry:
                        break
                    if score < 0:
                        # Audit unavailable: keep the good render rather than burning more image quota blindly.
                        break
                    if score >= alignment_threshold:
                        break
                    correction = (audit or {}).get("correction", "") or (
                        "严格恢复 Reference 1 的产品二维位置、尺度、接地点、相机高度、视角和地平线；不要重新构图。"
                    )

                if variant_best_path is None or not variant_best_path.exists():
                    continue
                variant = {
                    "variant": variant_index,
                    "best_path": str(variant_best_path),
                    "best_score": variant_best_score,
                    "best_audit": variant_best_audit,
                    "best_canvas": variant_best_canvas,
                    "attempts": variant_attempts,
                }
                if gate_mask:
                    variant["iou_gate"] = _gate_summary(variant_attempts, len(variant_attempts) >= planned_attempts)
                if worker_post:
                    self._phase = f"处理第 {variant_index}/{variant_count} 张（细节回贴 / 还原度）"
                    try:
                        variant["processed"] = result_worker.finish_variant(
                            str(variant_best_path), variant_index, self._detail, camera_w, camera_h, session_dir,
                            cancel_check=is_cancellation_requested)
                    except result_worker.Cancelled:
                        raise RuntimeError("任务已取消")
                    except Exception as exc:
                        print(f"[Wondful] 结果处理失败（改在主线程处理）：{exc}")
                variants.append(variant)
                self._completed_variants.put(variant)
                if best_path is None or variant_best_rank > best_rank:
                    best_path = variant_best_path
                    best_score = variant_best_score
                    best_audit = variant_best_audit
                    best_canvas = variant_best_canvas
                    best_variant = variant_index
                    best_rank = variant_best_rank

            if len(variants) != variant_count or best_path is None or not best_path.exists():
                raise RuntimeError(f"本轮只得到 {len(variants)} 张候选图，无法完成 {variant_count} 张生成。")
            shutil.copy2(best_path, self._session.output_raw)
            best_generation = None
            if attempts:
                best_generation = next((item.get("generation") for item in attempts if str(item.get("path")) == str(best_path)), None)
            return {
                "engine": (best_generation or {}).get("engine", f"{provider_id}_oauth_imagegen") if isinstance(best_generation, dict) else f"{provider_id}_oauth_imagegen",
                "status": "ok",
                "attempts": attempts,
                "best_score": best_score,
                "best_audit": best_audit,
                "best_canvas": best_canvas,
                "best_path": str(best_path),
                "best_variant": best_variant,
                "variants": variants,
                "variant_count": variant_count,
                "analysis_provider": provider_id,
                "structure_control": structure,
                "masked_repair": masked_repair,
                "render_mode": render_mode,
                "remote_audit_enabled": auto_alignment_retry,
                "generation_seconds": total_generation_seconds,
                "audit_seconds": total_audit_seconds,
                "semantic_summary": semantic_meta,
                "identity_mask": identity_meta,
                "identity_mask_path": str(identity_meta.get("path", "") or ""),
                "identity_hard_restore": bool(identity_hard_restore and identity_meta.get("enabled")),
                "semantic_backend": (
                    "MIXED" if semantic_meta.get("backends", {}).get("JEV", 0) and semantic_meta.get("backends", {}).get("LOCAL_FALLBACK", 0)
                    else ("JEV" if semantic_meta.get("backends", {}).get("JEV", 0) else "LOCAL_FALLBACK")
                ),
            }

        if not self._start_thread(context, job):
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _modal(self, context, event):
        props = self._origin_scene.wondful_ai
        prefs = _addon_prefs(context)
        if event.type == "TIMER":
            # Finished variants are exported on the main thread while the next
            # remote request is still running; a later 429 must not discard them.
            self._drain_completed_variants(props)
            if self._thread and self._thread.is_alive():
                self._progress_update(props, 0.10, 0.88)
                return {"RUNNING_MODAL"}

            elapsed = time.time() - self._started_at
            self._cleanup(context)
            if getattr(self, "_direct_state", None) is not None:
                from .codex_direct import engine_note
                props.last_engine_note = engine_note(self._direct_state)
            if self._error:
                exc, _tb = self._error
                if isinstance(exc, AntigravityNotLoggedInError):
                    props.antigravity_account_state = "LOGGED_OUT"
                    props.antigravity_account_message = str(exc)
                elif isinstance(exc, AntigravityNotInstalledError):
                    props.antigravity_account_state = "NOT_INSTALLED"
                    props.antigravity_account_message = str(exc)
                elif isinstance(exc, CodexNotLoggedInError):
                    props.codex_account_state = "LOGGED_OUT"
                    props.codex_account_message = str(exc)
                elif isinstance(exc, CodexNotInstalledError):
                    props.codex_account_state = "NOT_INSTALLED"
                    props.codex_account_message = str(exc)
                else:
                    from .codex_direct import DirectEditStop
                    if isinstance(exc, DirectEditStop) and exc.kind == "auth":
                        # 1.2.3: the backend rejected the login (after one token refresh)
                        props.codex_account_state = "LOGGED_OUT"
                        props.codex_account_message = str(exc)
                # Parameter/capacity/output failures do not establish an OAuth failure.
                msg = safe_diagnostic(str(exc))
                completed_count = len(getattr(self, "_exported_by_variant", {}))
                expected_count = int(getattr(self, "_variant_count", RENDER_VARIANT_COUNT))
                if completed_count:
                    msg += f" 已完成并导出 {completed_count}/{expected_count} 张；其余结果未生成。"
                if getattr(self, "_export_errors", None):
                    msg += " 部分结果导出失败：" + "；".join(self._export_errors[:3])
                _status(props, "ERROR", msg)
                props.progress = 0.0
                self.report({"ERROR"}, msg)
                return {"CANCELLED"}

            if getattr(self, "_provider_id", "codex") == "antigravity":
                props.antigravity_account_state = "LOGGED_IN"
            else:
                props.codex_account_state = "LOGGED_IN"
            result_data = self._result if isinstance(self._result, dict) else {}
            attempts = result_data.get("attempts", [])
            variants = result_data.get("variants", []) or []
            expected_variant_count = max(
                1,
                min(RENDER_VARIANT_COUNT, int(getattr(self, "_variant_count", RENDER_VARIANT_COUNT))),
            )
            props.variant_count = max(
                1,
                min(expected_variant_count, int(result_data.get("variant_count", len(variants) or 1))),
            )
            props.alignment_attempts = len(attempts)
            props.last_generation_seconds = float(result_data.get("generation_seconds", 0.0))
            props.last_audit_seconds = float(result_data.get("audit_seconds", 0.0))
            semantic_summary = result_data.get("semantic_summary") or {}
            props.jev_status = str(result_data.get("semantic_backend", "") or "LOCAL_FALLBACK")
            props.jev_semantic_summary = json.dumps(semantic_summary, ensure_ascii=False)
            identity_assets = semantic_summary.get("identity_critical", []) if isinstance(semantic_summary, dict) else []
            props.jev_identity_assets = "、".join(str(x) for x in identity_assets[:12])
            props.jev_status_message = (
                f"语义识别：{props.jev_status} · 关键身份资产 {len(identity_assets)} 个"
            )
            best_score = result_data.get("best_score", -1)
            try:
                props.alignment_score = float(best_score)
            except Exception:
                props.alignment_score = -1.0
            best_audit = result_data.get("best_audit")
            canvas = result_data.get("best_canvas") or {}
            props.last_canvas_warning = str(canvas.get("message", ""))
            props.last_canvas_fit = bool(canvas.get("fit_compatible") and not canvas.get("exact_size"))
            if isinstance(best_audit, dict):
                props.alignment_message = str(best_audit.get("correction", "")).strip()
            elif bool(result_data.get("remote_audit_enabled", False)):
                props.alignment_message = "构图校验未返回可解析评分；已保留生成结果。"
            else:
                props.alignment_message = "标准/快速模式未执行远程构图验收；可在 Wondful Compare 中人工检查，或切换严格对齐模式。"
            props.progress = 0.94
            camera_w, camera_h = self._camera_output_size
            final_path, aspect_postprocessed = enforce_aspect_ratio(
                str(self._session.output_raw),
                str(self._session.output_final),
                camera_w,
                camera_h,
            )
            if image_dimensions(final_path) != (camera_w, camera_h):
                props.last_canvas_fit = False
            try:
                output_dir = self._output_dir_at_start
                if len(variants) != expected_variant_count:
                    raise RuntimeError(f"本轮只得到 {len(variants)} 张候选图，无法完成 {expected_variant_count} 张导出。")
                if self._export_errors:
                    raise RuntimeError("；".join(self._export_errors[:3]))
                exported_by_variant = dict(self._exported_by_variant)
                if len(exported_by_variant) != expected_variant_count:
                    raise RuntimeError(f"实际导出 {len(exported_by_variant)} 张，未达到要求的 {expected_variant_count} 张。")
                exported_paths = [
                    exported_by_variant[index]
                    for index in sorted(exported_by_variant)
                ]
                best_variant = int(result_data.get("best_variant", 0) or 0)
                exported = exported_by_variant.get(best_variant) or exported_paths[0]
                # Candidate selection remains based on real canvas/alignment audit.
                # 3.1.2's file-size "TypeSafe score" was not a visual-quality metric.
                props.best_candidate_score = 0.0
                props.last_export_path = str(exported)
                props.last_export_paths = json.dumps([str(path) for path in exported_paths], ensure_ascii=False)
                cid = result_data.get("conversation_id", "")
                if not cid and attempts:
                    for att in attempts:
                        gen = att.get("generation") or {}
                        if isinstance(gen, dict) and gen.get("conversation_id"):
                            cid = gen["conversation_id"]
                            break
                if cid:
                    props.active_conversation_id = str(cid)
            except Exception as exc:
                _status(props, "ERROR", f"图片生成成功，但导出到输出目录失败: {exc}")
                self.report({"ERROR"}, props.last_error)
                return {"CANCELLED"}
            try:
                props.viewport_image = load_image(str(self._session.viewport_reference), "Wondful_Camera")
                props.result_image = load_image(str(exported), "Wondful_Result")
                mask_path = str(self._structure.get("mask_path", ""))
                props.structure_mask_image = load_image(mask_path, "Wondful_StructureMask") if mask_path and Path(mask_path).is_file() else None
            except Exception as exc:
                _status(props, "ERROR", f"图片生成成功，但 Blender 加载结果图失败: {exc}")
                self.report({"ERROR"}, props.last_error)
                return {"CANCELLED"}

            props.last_viewport_path = str(self._session.viewport_reference)
            props.last_result_path = str(exported)
            props.progress = 1.0
            props.eta_seconds = 0
            props.last_total_seconds = time.perf_counter() - self._request_started_at
            _status(props, "SUCCESS")
            save_metadata(
                self._session,
                {
                    "type": "ai_render",
                    "prompt_profile": self._prompt_profile,
                    "engine": result_data.get("engine", f"{getattr(self, '_provider_id', 'codex')}_oauth_imagegen"),
                    "analysis_provider": result_data.get("analysis_provider", props.analysis_provider.lower()),
                    "camera_reference_size": [self._width, self._height],
                    "camera_output_size": list(self._camera_output_size),
                    "camera_name": self._camera_name_at_start,
                    "reference_counts": dict(self._reference_counts or {}),
                    "reference_sources": self._reference_sources_at_start,
                    "reference_instructions": self._reference_instructions_at_start,
                    "style_fingerprint": str(getattr(self, "_style_fingerprint_at_render", "")),
                    "target_generation_size": [self._target_w, self._target_h],
                    "render_mode": getattr(self, "_render_mode", "STANDARD"),
                    "effective_long_edge": int(getattr(self, "_effective_long_edge", prefs.long_edge)),
                    "remote_audit_enabled": bool(result_data.get("remote_audit_enabled", False)),
                    "generation_time": elapsed,
                    "preparation_seconds": self._preparation_seconds,
                    "total_seconds": props.last_total_seconds,
                    "capture_method": props.last_capture_method,
                    "aspect_postprocess_used": bool(aspect_postprocessed),
                    "aspect_fit_applied": bool(aspect_postprocessed),
                    "canvas_check": canvas,
                    "structure_note": props.last_structure_note,
                    "exported_result_path": str(exported),
                    "exported_result_paths": [str(path) for path in exported_paths],
                    "variant_count": props.variant_count,
                    "output_directory": str(output_dir),
                    "alignment_score": props.alignment_score,
                    "alignment_attempts": props.alignment_attempts,
                    "alignment_message": props.alignment_message,
                    "alignment_threshold": int(prefs.alignment_threshold),
                    "strict_composition_lock": bool(prefs.strict_composition_lock),
                    "structure_control": result_data.get("structure_control", self._structure or {}),
                    "masked_alignment_repair": bool(result_data.get("masked_repair", False)),
                    "identity_preserve_mask": result_data.get("identity_mask", {}),
                    "identity_preserve_mask_path": result_data.get("identity_mask_path", ""),
                    "identity_hard_restore": bool(result_data.get("identity_hard_restore", False)),
                    "provider_response": self._result,
                    "codex_direct_edit": getattr(self, "_direct_state", None),
                },
            )
            record_timing(getattr(self, "_timing_key", f"{getattr(self, '_provider_id', 'codex')}_render_standard"), elapsed)
            score_text = f"，构图对齐 {props.alignment_score:.0f}/100" if props.alignment_score >= 0 else ""
            if props.last_canvas_warning:
                report_level = {"INFO"} if props.last_canvas_fit else {"WARNING"}
                self.report(report_level, f"{props.last_canvas_warning} 已导出 {len(exported_paths)} 张；最佳结果：{exported}")
            else:
                self.report({"INFO"}, f"{getattr(self, '_provider_label', 'Codex')} AI渲染完成，总用时 {props.last_total_seconds:.1f}s{score_text}。已导出 {len(exported_paths)} 张；最佳结果：{exported}")
            return {"FINISHED"}
        return {"PASS_THROUGH"}


class WONDFUL_OT_choose_output_directory(Operator):
    bl_idname = "wondful.choose_output_directory"
    bl_label = "选择渲染输出目录"
    bl_description = "为当前 Blender 工程指定 Wondful AI 最终渲染图输出位置"

    directory: StringProperty(name="输出目录", subtype="DIR_PATH", options={"SKIP_SAVE"})

    def invoke(self, context, event):
        props = context.scene.wondful_ai
        prefs = _addon_prefs(context)
        current = (props.output_directory or "").strip() or (prefs.output_directory or "").strip()
        if current:
            try:
                self.directory = bpy.path.abspath(current)
            except Exception:
                self.directory = current
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        path = (self.directory or "").strip()
        if not path:
            self.report({"WARNING"}, "没有选择输出目录。")
            return {"CANCELLED"}
        context.scene.wondful_ai.output_directory = path
        self.report({"INFO"}, f"当前工程输出目录：{path}")
        return {"FINISHED"}


class WONDFUL_OT_open_output_directory(Operator):
    bl_idname = "wondful.open_output_directory"
    bl_label = "打开输出目录"
    bl_description = "打开当前 Wondful AI 渲染图输出目录"

    def execute(self, context):
        prefs = _addon_prefs(context)
        props = context.scene.wondful_ai
        try:
            configured = (props.output_directory or "").strip() or (prefs.output_directory or "").strip()
            directory = resolve_output_directory(context, configured)
            bpy.ops.wm.path_open(filepath=str(directory))
        except Exception as exc:
            self.report({"ERROR"}, f"无法打开输出目录: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class WONDFUL_OT_load_reference(Operator, ImportHelper):
    bl_idname = "wondful.load_reference"
    bl_label = "添加参考图"
    bl_description = "一次选择一张或多张图片添加到当前参考图集合；第一张为主参考"

    filename_ext = ""
    filter_glob: StringProperty(default="*.png;*.jpg;*.jpeg;*.webp", options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})
    ref_kind: EnumProperty(
        items=[
            ("PRODUCT", "产品参考图", ""),
            ("PERSON", "人物参考图", ""),
            ("STYLE", "风格参考图", ""),
        ]
    )

    def execute(self, context):
        props = context.scene.wondful_ai
        collection, index_attr = _collection_and_index(props, self.ref_kind)
        selected = []
        if self.files:
            selected = [str(Path(self.directory) / f.name) for f in self.files]
        elif self.filepath:
            selected = [self.filepath]
        selected = [p for p in selected if p]
        if not selected:
            self.report({"WARNING"}, "没有选择参考图。")
            return {"CANCELLED"}

        # STYLE upload means "replace the current style set". In 2.17 this
        # button appended new images behind the first upload, while the first
        # image remained the hard-coded primary style forever.
        if self.ref_kind == "STYLE":
            added, failures = _replace_style_set(props, selected)
            if len(selected) > MAX_REFERENCES_PER_KIND:
                self.report({"WARNING"}, f"风格参考最多 {MAX_REFERENCES_PER_KIND} 张，多余图片未添加。")
            elif failures:
                self.report({"WARNING"}, f"风格已更换 {added} 张，{len(failures)} 张读取失败。")
            elif added:
                self.report({"INFO"}, f"已更换当前风格参考，共 {added} 张。")
            return {"FINISHED" if added else "CANCELLED"}

        if self.ref_kind == "PRODUCT":
            try:
                _replace_product_reference(props, selected[0])
            except Exception as exc:
                self.report({"ERROR"}, f"产品参考图读取失败：{exc}")
                return {"CANCELLED"}
            if len(selected) > 1:
                self.report({"WARNING"}, "产品造型参考只保留 1 张，已使用第一张。多个角度请拼成一张三视图。")
            else:
                self.report({"INFO"}, "已设置产品造型参考。")
            return {"FINISHED"}

        capacity = MAX_REFERENCES_PER_KIND - len(collection)
        if capacity <= 0:
            self.report({"WARNING"}, f"每类参考图最多 {MAX_REFERENCES_PER_KIND} 张。")
            return {"CANCELLED"}

        added = 0
        failures = []
        for filepath in selected[:capacity]:
            try:
                _append_reference_from_path(props, self.ref_kind, filepath)
                added += 1
            except Exception as exc:
                failures.append(f"{Path(filepath).name}: {exc}")
        if added:
            setattr(props, index_attr, len(collection) - 1)
            _mark_reference_changed(props, self.ref_kind)
        if len(selected) > capacity:
            self.report({"WARNING"}, f"已达到每类 {MAX_REFERENCES_PER_KIND} 张上限，多余图片未添加。")
        elif failures:
            self.report({"WARNING"}, f"成功添加 {added} 张，{len(failures)} 张读取失败。")
        else:
            self.report({"INFO"}, f"已添加 {added} 张参考图。")
        return {"FINISHED" if added else "CANCELLED"}


class WONDFUL_OT_paste_reference(Operator):
    bl_idname = "wondful.paste_reference"
    bl_label = "粘贴参考图"
    bl_description = "从系统图片剪贴板直接粘贴到当前参考图集合（macOS 支持截图/复制图片）"

    ref_kind: EnumProperty(items=[("PRODUCT", "产品", ""), ("PERSON", "人物", ""), ("STYLE", "风格", "")])

    def execute(self, context):
        props = context.scene.wondful_ai
        collection, index_attr = _collection_and_index(props, self.ref_kind)
        if self.ref_kind not in {"STYLE", "PRODUCT"} and len(collection) >= MAX_REFERENCES_PER_KIND:
            self.report({"WARNING"}, f"每类参考图最多 {MAX_REFERENCES_PER_KIND} 张。")
            return {"CANCELLED"}
        clipboard_dir = CONFIG_DIR / "clipboard"
        clipboard_dir.mkdir(parents=True, exist_ok=True)
        destination = clipboard_dir / f"clipboard_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        try:
            path = save_clipboard_image(destination)
            if self.ref_kind == "STYLE":
                added, failures = _replace_style_set(props, [path])
                if not added:
                    raise RuntimeError(failures[0] if failures else "无法读取剪贴板风格图")
            elif self.ref_kind == "PRODUCT":
                _replace_product_reference(props, path)
            else:
                _append_reference_from_path(props, self.ref_kind, path)
                _mark_reference_changed(props, self.ref_kind)
        except ClipboardImageError as exc:
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:
            self.report({"ERROR"}, f"粘贴参考图失败：{exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, "已从剪贴板添加参考图。")
        return {"FINISHED"}


class WONDFUL_OT_replace_reference(Operator, ImportHelper):
    bl_idname = "wondful.replace_reference"
    bl_label = "替换参考图"

    filename_ext = ""
    filter_glob: StringProperty(default="*.png;*.jpg;*.jpeg;*.webp", options={"HIDDEN"})
    ref_kind: EnumProperty(items=[("PRODUCT", "产品", ""), ("PERSON", "人物", ""), ("STYLE", "风格", "")])
    item_index: IntProperty(default=-1, options={"HIDDEN", "SKIP_SAVE"})

    def execute(self, context):
        props = context.scene.wondful_ai
        collection, index_attr, idx = _resolve_reference_index(props, self.ref_kind, self.item_index)
        if idx < 0:
            return {"CANCELLED"}
        try:
            image = _load_reference_image(self.filepath)
        except Exception as exc:
            self.report({"ERROR"}, f"无法读取参考图: {exc}")
            return {"CANCELLED"}
        collection[idx].image = image
        collection[idx].source_path = str(Path(self.filepath).expanduser())
        _mark_reference_changed(props, self.ref_kind)
        return {"FINISHED"}


class WONDFUL_OT_clear_reference(Operator):
    bl_idname = "wondful.clear_reference"
    bl_label = "删除参考图"

    ref_kind: EnumProperty(items=[("PRODUCT", "产品", ""), ("PERSON", "人物", ""), ("STYLE", "风格", "")])
    item_index: IntProperty(default=-1, options={"HIDDEN", "SKIP_SAVE"})

    def execute(self, context):
        props = context.scene.wondful_ai
        collection, index_attr, idx = _resolve_reference_index(props, self.ref_kind, self.item_index)
        if idx >= 0:
            collection.remove(idx)
            setattr(props, index_attr, max(0, min(idx, len(collection) - 1)))
            _mark_reference_changed(props, self.ref_kind)
        return {"FINISHED"}


class WONDFUL_OT_clear_all_references(Operator):
    bl_idname = "wondful.clear_all_references"
    bl_label = "清空本类参考图"

    ref_kind: EnumProperty(items=[("PRODUCT", "产品", ""), ("PERSON", "人物", ""), ("STYLE", "风格", "")])

    def execute(self, context):
        props = context.scene.wondful_ai
        collection, index_attr = _collection_and_index(props, self.ref_kind)
        had_items = bool(collection)
        collection.clear()
        setattr(props, index_attr, 0)
        if had_items:
            _mark_reference_changed(props, self.ref_kind)
        return {"FINISHED"}


class WONDFUL_OT_move_reference(Operator):
    bl_idname = "wondful.move_reference"
    bl_label = "调整参考图顺序"

    ref_kind: EnumProperty(items=[("PRODUCT", "产品", ""), ("PERSON", "人物", ""), ("STYLE", "风格", "")])
    direction: EnumProperty(items=[("UP", "前移", ""), ("DOWN", "后移", "")])
    item_index: IntProperty(default=-1, options={"HIDDEN", "SKIP_SAVE"})

    def execute(self, context):
        props = context.scene.wondful_ai
        collection, index_attr, idx = _resolve_reference_index(props, self.ref_kind, self.item_index)
        if len(collection) < 2 or idx < 0:
            return {"CANCELLED"}
        target = idx - 1 if self.direction == "UP" else idx + 1
        if target < 0 or target >= len(collection):
            return {"CANCELLED"}
        collection.move(idx, target)
        setattr(props, index_attr, target)
        _mark_reference_changed(props, self.ref_kind)
        return {"FINISHED"}


class WONDFUL_OT_open_prompt_text_editor(Operator):
    bl_idname = "wondful.open_prompt_text_editor"
    bl_label = "编辑完整提示词"
    bl_description = "在当前区域展开完整文本；完成后点击顶部“返回渲染”，保留原视图"

    def execute(self, context):
        try:
            open_prompt_in_area(context)
        except Exception as exc:
            self.report({"ERROR"}, f"无法展开提示词编辑器：{exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class WONDFUL_OT_close_prompt_text_editor(Operator):
    bl_idname = "wondful.close_prompt_text_editor"
    bl_label = "返回渲染"
    bl_description = "同步完整提示词并恢复原来的编辑区域"

    def execute(self, context):
        if context.area and close_prompt_area(context.area):
            return {"FINISHED"}
        return {"CANCELLED"}


class WONDFUL_OT_prompt_expand(Operator):
    bl_idname = "wondful.prompt_expand"
    bl_label = "展开多行编辑"

    def execute(self, context):
        props = context.scene.wondful_ai
        from .properties import sync_prompt_editor
        sync_prompt_editor(props, _get_prompt_text(props) or props.prompt or "")
        return {"FINISHED"}


class WONDFUL_OT_prompt_line_add(Operator):
    bl_idname = "wondful.prompt_line_add"
    bl_label = "添加提示词行"

    def execute(self, context):
        props = context.scene.wondful_ai
        item = props.prompt_lines.add()
        item.text = ""
        props.prompt_line_index = len(props.prompt_lines) - 1
        return {"FINISHED"}


class WONDFUL_OT_prompt_line_remove(Operator):
    bl_idname = "wondful.prompt_line_remove"
    bl_label = "删除提示词行"

    def execute(self, context):
        props = context.scene.wondful_ai
        if props.prompt_lines:
            idx = min(props.prompt_line_index, len(props.prompt_lines) - 1)
            props.prompt_lines.remove(idx)
            props.prompt_line_index = max(0, min(idx, len(props.prompt_lines) - 1))
            _set_prompt_text(props, "\n".join(item.text.rstrip() for item in props.prompt_lines).strip())
        return {"FINISHED"}


# --------------------------------------------------------------------------- 1.2.3 多角度一致

def _mv_selected_cameras(context, master):
    return [o for o in (getattr(context, "selected_objects", None) or [])
            if getattr(o, "type", "") == "CAMERA" and o != master]


class WONDFUL_OT_mv_set_master(Operator):
    """多角度一致: the current scene camera becomes 主视角, its latest result the master image."""
    bl_idname = "wondful.mv_set_master"
    bl_label = "设为主视角"
    bl_description = "把当前场景相机标为主视角，并用它最近一次的渲染结果作为主视角图（也可以在下面另选一张已满意的图）"

    def execute(self, context):
        props = context.scene.wondful_ai
        cam = context.scene.camera
        if cam is None or cam.type != "CAMERA":
            self.report({"ERROR"}, "场景里没有活动相机。")
            return {"CANCELLED"}
        props.mv_master_camera = cam
        last = (getattr(props, "last_export_path", "") or getattr(props, "last_result_path", "") or "").strip()
        if last and Path(last).is_file():
            props.mv_master_image = last
            self.report({"INFO"}, f"主视角：{cam.name}（主视角图：{Path(last).name}）")
        else:
            self.report({"WARNING"}, f"主视角：{cam.name}；还没有它的渲染结果，请先渲染或在下面选择主视角图。")
        return {"FINISHED"}


class WONDFUL_OT_multiview_render(_BaseAsyncOperator):
    """1.2.3 多角度一致: render the other selected cameras so they agree with the master view."""
    bl_idname = "wondful.multiview_render"
    bl_label = "多角度一致渲染"
    bl_description = ("按主视角渲染选中的其他相机：离主视角最近的先渲染，已完成的视角投影到环境体块上作为底图，"
                      "光照方向在世界里固定；每个相机 1 次生图，产品超标时最多重试 1 次")

    def execute(self, context):
        import numpy as np
        from . import multiview as MV
        from . import codex_direct as CD
        from .viewport_capture import render_clay_base, LAST_CLAY_INFO, keep_exact_slots, keep_exact_texts
        from .terrain import settings_from_props
        from .massing import find_collection as _find_massing
        from .png_io import read_png
        from .layered_output import resize
        scene = context.scene
        props = scene.wondful_ai
        prefs = _addon_prefs(context)
        provider = _analysis_provider(props, prefs)
        if provider["id"] != "codex" or not bool(getattr(prefs, "codex_direct_edit", True)):
            self.report({"ERROR"}, "多角度一致需要 Codex 图像编辑直连（高级 › AI 与模型）。")
            return {"CANCELLED"}
        master = getattr(props, "mv_master_camera", None)
        if master is None or master.type != "CAMERA" or master.name not in scene.objects:
            self.report({"ERROR"}, "请先把已满意的相机设为主视角。")
            return {"CANCELLED"}
        master_img = bpy.path.abspath(getattr(props, "mv_master_image", "") or "")
        if not master_img or not Path(master_img).is_file():
            self.report({"ERROR"}, "主视角图不存在：请先渲染主视角，或选择一张已满意的主视角图。")
            return {"CANCELLED"}
        coll = getattr(props, "product_collection", None)
        if coll is None:
            self.report({"ERROR"}, "多角度一致需要先在 ② 选择产品集合。")
            return {"CANCELLED"}
        others = _mv_selected_cameras(context, master)
        skipped = [o.name for o in others if getattr(o.data, "type", "PERSP") != "PERSP"]
        others = [o for o in others if o.name not in skipped]
        if not others:
            self.report({"ERROR"}, "请在视图里选中要渲染的其他相机（主视角以外，透视相机）。")
            return {"CANCELLED"}
        configured = (props.output_directory or "").strip() or (prefs.output_directory or "").strip()
        try:
            out_dir = resolve_output_directory(context, configured)
            preflight_output_directory(out_dir)
        except Exception as exc:
            self.report({"ERROR"}, f"输出目录不可用：{exc}")
            return {"CANCELLED"}
        W, H = camera_output_dimensions(context)
        ww, wh = MV.work_size(W, H)
        pool = out_dir / "多角度一致" / f"{MV.safe_name(scene.name)}_{MV.safe_name(master.name)}"
        pool.mkdir(parents=True, exist_ok=True)
        massing_coll = _find_massing(scene, props)
        made, wlog = [], {}
        old_cam = scene.camera
        views = {}
        t_prep = time.time()
        try:
            if bool(getattr(props, "mv_widen", True)) and massing_coll is not None:
                try:
                    made, wlog = MV.widen_blockout(context, massing_coll, coll, master)
                except Exception as exc:
                    print(f"[Wondful] 多角度一致：加宽体块失败：{exc}")
            for cam in [master] + others:
                scene.camera = cam
                context.view_layer.update()
                pre = pool / MV.safe_name(cam.name)
                files = {k: f"{pre}_{k}.png" for k in ("clay", "mask", "detail", "keep", "matid")}
                ok = render_clay_base(context, files["clay"], coll, ground=getattr(props, "clay_ground", "AUTO"),
                                      crest_angle=float(getattr(props, "clay_crest_angle", 18.0)),
                                      terrain=settings_from_props(props), mask_path=files["mask"], massing=massing_coll,
                                      detail_path=files["detail"], keep_path=files["keep"], matid_path=files["matid"],
                                      geometry_prefix=str(pre), geometry_size=(ww, wh))
                if not ok or not LAST_CLAY_INFO.get("geometry") or not Path(files["mask"]).is_file():
                    raise RuntimeError(f"相机 {cam.name} 的白模 / 几何通道渲染失败")
                m = read_png(files["mask"]).astype(np.float32)
                m = (m[..., 3] if m.shape[-1] == 4 else m[..., 0]) / 255.0
                np.save(f"{pre}_maskw.npy", np.clip(resize(m, ww, wh), 0, 1).astype(np.float32))
                rec = MV.scene_camera_record(scene, cam)
                blocked = hidden = 0.0
                if cam != master:
                    try:
                        blocked, hidden = MV.obstacle(context, cam, coll)
                    except Exception as exc:
                        print(f"[Wondful] 多角度一致：遮挡检查失败：{exc}")
                views[cam.name] = {"name": cam.name, "cam": rec, "files": files, "prefix": str(pre),
                                   "massing_text": str(LAST_CLAY_INFO.get("massing_text", "") or ""),
                                   "obstacle": round(blocked, 3), "hidden": round(hidden, 3)}
        except Exception as exc:
            self.report({"ERROR"}, f"多角度一致准备失败：{exc}")
            props.last_mv_note = f"多角度一致准备失败：{exc}"
            return {"CANCELLED"}
        finally:
            scene.camera = old_cam
            MV.remove_aux(made)
            try:
                context.view_layer.update()
            except Exception:
                pass
        centre = None
        b = MV.product_bounds(coll)
        if b is not None:
            centre = [(b[0].x + b[1].x) / 2, (b[0].y + b[1].y) / 2, (b[0].z + b[1].z) / 2]
        order = MV.order_views(views[master.name]["cam"], [views[o.name]["cam"] for o in others], centre or [0, 0, 0])
        from .product_kind import is_vehicle as _is_vehicle
        try:
            vehicle = bool(_is_vehicle(props))
        except Exception:
            vehicle = False
        keep_text = ""
        try:
            slots, texts = keep_exact_slots(coll), keep_exact_texts(coll)
            if slots or texts:
                from .detail_paste import keep_prompt
                keep_text = keep_prompt(vehicle, texts or ())
        except Exception:
            pass
        env_refs = [str(Path(p).resolve()) for p in _reference_paths(props)["style"] if Path(p).is_file()]
        manifest = [{"path": p, "members": [{"role": "environment"}]} for p in env_refs]
        brief = (getattr(props, "scene_brief", "") or "").strip()
        model = str(getattr(prefs, "codex_model", "") or "")
        client_version = str(getattr(props, "codex_cli_version", "") or "")
        size = CD.choose_edit_size(W, H, int(getattr(prefs, "long_edge", 2048) or 2048))
        paste = bool(getattr(props, "detail_paste", True))
        colour_fix = bool(getattr(props, "detail_colour_fix", True))
        strength = float(getattr(props, "detail_paste_strength", 0.8))
        colour_strength = float(getattr(props, "detail_colour_strength", 0.7)) if colour_fix else 0.0
        env_check = bool(getattr(props, "mv_env_check", True))
        self._mv = {"out_dir": out_dir, "pool": pool, "scene": scene.name, "views": views, "skipped": skipped,
                    "widen": wlog, "prep_seconds": round(time.time() - t_prep, 1), "master": master.name}
        self._phase = "多角度一致：准备完成"
        self._estimated_total = 15.0 + 80.0 * len(others)
        _status(props, "RENDERING")
        props.progress = 0.05
        props.last_mv_note = "多角度一致：生成中…"
        master_name = master.name

        def job():
            import numpy as np
            from .cli_transport import is_cancellation_requested as cancelled
            from .png_io import write_png
            from . import result_worker as RW
            t0 = time.time()
            mv = views[master_name]
            master_png = pool / "master.png"
            if not RW.fit_to_canvas(master_img, master_png, W, H):
                raise RuntimeError("主视角图无法读取（请用 PNG）")
            entry_m = {"name": master_name, "cam": mv["cam"], "pos": Path(mv["prefix"] + "_pos.npy").name,
                       "nrm": Path(mv["prefix"] + "_nrm.npy").name, "mask": Path(mv["prefix"] + "_maskw.npy").name,
                       "image": master_png.name, "master": True}
            data = MV.read_pool(pool)
            reused = [e for e in data.get("views", []) if not e.get("master") and e["name"] not in views
                      and (pool / e.get("image", "")).is_file() and (pool / e.get("pos", "")).is_file()]
            light, light_note = None, ""
            lp = pool / MV.LIGHT_FILE
            key = MV.file_md5(master_png)
            try:
                cached = json.loads(lp.read_text(encoding="utf-8")) if lp.is_file() else {}
            except (OSError, ValueError):
                cached = {}
            if cached.get("key") == key and cached.get("light"):
                light = cached["light"]
            else:
                self._phase = "多角度一致：读取主视角光照"
                try:
                    light = MV.world_light(MV.estimate_light(str(master_png), client_version=client_version, model=model,
                                                             cancel_check=cancelled), mv["cam"])
                    lp.write_text(json.dumps({"key": key, "light": light}, ensure_ascii=False, indent=1), encoding="utf-8")
                except CD.DirectEditError as exc:
                    if exc.kind == "cancelled" or exc.kind in CD.STOP_KINDS:
                        raise
                    light_note = f"光照估计失败（未锁定光照）：{str(exc)[:60]}"
                except Exception as exc:
                    light_note = f"光照估计失败（未锁定光照）：{str(exc)[:60]}"
            master_view = MV.load_view(pool, entry_m)
            sources = [master_view] + [MV.load_view(pool, e) for e in reused]
            results = []
            accepted = [entry_m] + reused
            stop = None
            for i, rec in enumerate(order, start=1):
                v = views[rec["name"]]
                res = {"name": rec["name"], "ok": False, "calls": 0, "obstacle": v["obstacle"], "hidden": v["hidden"]}
                results.append(res)
                if stop:
                    res["error"] = stop
                    continue
                if cancelled():
                    raise RuntimeError("任务已取消")
                self._phase = f"多角度一致：{rec['name']}（{i}/{len(order)}）"
                pre = v["prefix"]
                vk = {"name": rec["name"], "cam": rec, "pos": np.load(pre + "_pos.npy"), "nrm": np.load(pre + "_nrm.npy"),
                      "mask": np.load(pre + "_maskw.npy")}
                clay_full = read_png(v["files"]["clay"]).astype(np.float32)[..., :3]
                mfull = read_png(v["files"]["mask"]).astype(np.float32)
                mfull = (mfull[..., 3] if mfull.shape[-1] == 4 else mfull[..., 0]) / 255.0
                env, cover, _srcmap, st = MV.build_base(vk, sources, resize(clay_full, ww, wh))
                base = pre + "_base.png"
                write_png(base, np.rint(MV.compose_canvas(env, cover, clay_full, mfull)).astype(np.uint8))
                res["projection"] = st
                text = MV.view_text(v["massing_text"], MV.light_sentence(light, rec), [s["name"] for s in sources])
                Path(pre + "_prompt.txt").write_text(text, encoding="utf-8")
                work = Path(pre + "_work")
                work.mkdir(exist_ok=True)
                spec = {"detail": v["files"]["detail"], "mask": v["files"]["mask"], "keep": v["files"]["keep"],
                        "matid": v["files"]["matid"] if colour_fix else "", "strength": strength,
                        "colour_strength": colour_strength, "work_dir": str(work), "mirrored": []}
                matid = spec["matid"] if spec["matid"] and Path(spec["matid"]).is_file() else ""
                tries = []
                for n in range(1, MV.MAX_CALLS + 1):
                    raw = pre + f"_try{n}_raw.png"
                    try:
                        r = CD.edit_image(prompt="", edit_base_path=base, reference_paths=env_refs, manifest=manifest,
                                          output_path=raw, size=size, model=model, client_version=client_version,
                                          log_dir=str(pool), tag=f"mv_{MV.safe_name(rec['name'])}_try{n}",
                                          cancel_check=cancelled, simple_brief=brief, vehicle=vehicle,
                                          massing_text=text, keep_text=keep_text)
                    except CD.DirectEditError as exc:
                        res["calls"] += 1
                        if exc.kind == "cancelled":
                            raise
                        if exc.kind in CD.STOP_KINDS or exc.fatal:
                            stop = CD.stop_message(exc.kind, str(exc))
                        res["error"] = str(exc)[:160]
                        break
                    res["calls"] += 1
                    canvas = pre + f"_try{n}.png"
                    if not RW.fit_to_canvas(raw, canvas, W, H):
                        res["error"] = "结果图无法读取"
                        break
                    rep, _note = RW.identity_scores(canvas, spec, matid)
                    gate = {}
                    try:      # fixed-coordinate silhouette (轮廓验收, ~1 s): did the product really move?
                        gate = RW.gate_score(raw, v["files"]["mask"], W, H, str(work), f"try{n}", cancelled,
                                             detail_path=v["files"]["detail"])
                    except RW.Cancelled:
                        raise
                    except Exception as exc:
                        print(f"[Wondful] 多角度一致：轮廓验收失败：{exc}")
                    img_w = resize(read_png(canvas).astype(np.float32)[..., :3], ww, wh)
                    rp = MV.reproj_de(master_view, vk, img_w)
                    t = {"file": canvas, "identity": {k: rep.get(k) for k in ("shift_px", "scale_pct", "outline_px", "edge_px",
                                                                          "edge_px_detail", "colour_dE", "error")},
                         "iou": gate.get("iou"), "iou_verdict": gate.get("verdict"),
                         "reproj": rp, "product_violations": MV.product_violations(rep, gate.get("verdict")),
                         "env_violations": MV.env_violations(rp, env_check), "seconds": r.get("seconds"),
                         "image_model": r.get("image_model", "")}
                    tries.append(t)
                    if not t["product_violations"] and not t["env_violations"]:
                        break
                if not tries:
                    continue
                best = MV.choose(tries)
                res.update({"tries": tries, "chosen": Path(best["file"]).name, "identity": best["identity"],
                            "reproj": best["reproj"], "product_violations": best["product_violations"],
                            "env_violations": best["env_violations"],
                            "retry_reason": (tries[0]["product_violations"] + tries[0]["env_violations"]) if len(tries) > 1 else []})
                final = best["file"]
                if paste:
                    out = RW.paste_and_score(best["file"], 1, spec, cancelled)
                    final = out.get("final") or final
                    res["detail_note"] = out.get("detail_note", "")
                    res["score_note"] = out.get("score_note", "")
                img_name = MV.safe_name(rec["name"]) + "_accepted.png"
                shutil.copy2(final, pool / img_name)
                res["final"] = str(pool / img_name)
                res["raw"] = best["file"]
                res["ok"] = True
                entry = {"name": rec["name"], "cam": rec, "pos": Path(pre + "_pos.npy").name, "nrm": Path(pre + "_nrm.npy").name,
                         "mask": Path(pre + "_maskw.npy").name, "image": img_name, "master": False}
                accepted.append(entry)
                sources.append(MV.load_view(pool, entry))
                MV.write_pool(pool, {"master": master_name, "views": accepted})
                (pool / f"{MV.safe_name(rec['name'])}_meta.json").write_text(
                    json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            summary = {"results": results, "light": light, "light_note": light_note, "seconds": round(time.time() - t0, 1),
                       "order": [r["name"] for r in order], "reused": [e["name"] for e in reused]}
            (pool / "last_run.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            return summary

        if not self._start_thread(context, job):
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _modal(self, context, event):
        from . import multiview as MV
        props = self._origin_scene.wondful_ai
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._thread and self._thread.is_alive():
            self._progress_update(props, 0.08, 0.95)
            return {"RUNNING_MODAL"}
        self._cleanup(context)
        props.progress = 0.0
        props.eta_seconds = 0
        if self._error:
            _status(props, "ERROR", f"多角度一致失败：{self._error[0]}")
            props.last_mv_note = props.last_error
            self.report({"ERROR"}, props.last_error)
            return {"CANCELLED"}
        info = self._mv
        summary = self._result or {}
        exported = []
        for r in summary.get("results", []):
            if not r.get("ok"):
                continue
            try:
                p = export_result(r["final"], info["out_dir"], info["scene"], r["name"], "多角度一致")
                r["exported"] = str(p)
                exported.append(str(p))
                if r.get("raw"):
                    from .detail_paste import raw_path
                    shutil.copy2(r["raw"], raw_path(p))
            except Exception as exc:
                r["export_error"] = str(exc)
        notes = [MV.summary_line(summary.get("results", []))]
        warn = [f"{r['name']} 遮挡 {r['obstacle'] * 100:.0f}%" for r in summary.get("results", [])
                if (r.get("obstacle") or 0) > 0.25 or (r.get("hidden") or 0) > 0.25]
        if warn:
            notes.append("注意：产品前方有遮挡物（" + "、".join(warn) + "），可移动相机")
        if info.get("skipped"):
            notes.append("已跳过正交相机：" + "、".join(info["skipped"]))
        if summary.get("light_note"):
            notes.append(summary["light_note"])
        failed = [f"{r['name']}：{r.get('error')}" for r in summary.get("results", []) if not r.get("ok") and r.get("error")]
        if failed:
            notes.append("未完成：" + "；".join(failed)[:160])
        props.last_mv_note = " · ".join(n for n in notes if n)
        if exported:
            props.last_export_path = exported[-1]
            props.last_result_path = exported[-1]
            props.last_export_paths = json.dumps(exported, ensure_ascii=False)
            try:
                props.result_image = load_image(exported[-1], "Wondful_Result")
            except Exception:
                pass
        try:
            (Path(info["pool"]) / "last_run.json").write_text(
                json.dumps(dict(summary, exported=exported, widen=info.get("widen"), prep_seconds=info.get("prep_seconds")),
                           ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        except Exception:
            pass
        _status(props, "IDLE" if exported else "ERROR", "" if exported else props.last_mv_note)
        self.report({"INFO"} if exported else {"WARNING"}, props.last_mv_note[:200])
        return {"FINISHED"} if exported else {"CANCELLED"}
