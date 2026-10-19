"""1.2.3 更新检查 + 一键更新 — Blender side (operators, startup check, banner state).

All network and file work runs in one background thread; the UI only reads ``STATE`` and a timer
(``poll``) redraws the sidebar and finishes on the main thread (duplicate copies are disabled and removed
only after the new version is in place). Nothing here blocks Blender's UI.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from . import updater as U

PACKAGE = __package__
ADDON_DIR = Path(__file__).resolve().parent

# phase: IDLE | CHECKING | AVAILABLE | UP_TO_DATE | CHECK_FAILED | DOWNLOADING | VERIFYING | INSTALLING | DONE | FAILED
STATE = {
    "phase": "IDLE",
    "manual": False,          # last check was the manual 「检查更新」 (errors shown)
    "manifest": None,         # newest known manifest when newer than the loaded version
    "message": "",
    "progress": 0.0,
    "installed": "",          # version written to disk (DONE)
    "notes": [],              # extra lines (duplicates removed, leftovers)
}
_lock = threading.Lock()
_worker = None
_result = None                # set by the worker, consumed by poll() on the main thread


def local_version() -> str:
    mod = sys.modules.get(PACKAGE)
    info = getattr(mod, "bl_info", None) or U.read_bl_info(ADDON_DIR / "__init__.py")
    return U.version_str(info.get("version", (0, 0, 0)))


def state_dir() -> str:
    env = os.environ.get("WONDFUL_UPDATE_STATE_DIR", "").strip()
    if env:
        return env
    try:
        return bpy.utils.user_resource("CONFIG", path=PACKAGE, create=True)
    except Exception:
        return os.path.join(tempfile.gettempdir(), PACKAGE)


def _prefs():
    try:
        return bpy.context.preferences.addons[PACKAGE].preferences
    except Exception:
        return None


def auto_check_enabled() -> bool:
    prefs = _prefs()
    return bool(getattr(prefs, "auto_check_updates", True)) if prefs is not None else True


def busy() -> bool:
    return STATE["phase"] in ("CHECKING", "DOWNLOADING", "VERIFYING", "INSTALLING")


def banner_manifest():
    """What the main-panel banner shows: a pending update (not ignored), or an install in progress / done / failed."""
    if STATE["phase"] in ("DOWNLOADING", "VERIFYING", "INSTALLING", "DONE", "FAILED"):
        return STATE["manifest"]
    manifest = STATE["manifest"]
    if manifest and U.is_newer(manifest["version"], local_version()) \
            and not U.is_ignored(U.load_state(state_dir()), manifest["version"]):
        return manifest
    return None


def _set(**kw):
    with _lock:
        STATE.update(kw)


def _redraw():
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type in ("VIEW_3D", "PREFERENCES"):
                    area.tag_redraw()
    except Exception:
        pass


def addon_roots() -> list:
    """Every add-on / extension directory Blender scans (for duplicate copies)."""
    roots = []
    try:
        roots += list(bpy.utils.script_paths(subdir="addons"))
    except Exception:
        pass
    try:
        for repo in bpy.context.preferences.extensions.repos:
            path = getattr(repo, "directory", "") or getattr(repo, "custom_directory", "")
            if path:
                roots.append(path)
    except Exception:
        pass
    roots.append(str(ADDON_DIR.parent))
    out = []
    for r in roots:
        if r and os.path.isdir(r) and os.path.abspath(r) not in out:
            out.append(os.path.abspath(r))
    return out


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------

def _check_worker(manual: bool):
    global _result
    try:
        manifest = U.fetch_manifest()
        _result = ("check", manifest, None, manual)
    except U.UpdateError as exc:
        _result = ("check", None, str(exc), manual)
    except Exception as exc:  # never let a thread die silently with the phase stuck
        _result = ("check", None, f"检查失败：{exc}", manual)


def _install_worker(manifest: dict, target: str):
    global _result
    try:
        _set(phase="DOWNLOADING", progress=0.0, message="下载中")

        def _progress(got, total):
            total = total or manifest["size"]
            pct = max(0.0, min(1.0, got / float(total or 1)))
            _set(progress=pct * 0.8, message=f"下载中 {int(pct * 100)}%")

        with tempfile.TemporaryDirectory(prefix="wondful_update_dl_") as tmp:
            # verify happens per mirror inside download_zip (a bad mirror falls through to the next)
            zip_path = U.download_zip(manifest, tmp, progress=_progress)
            _set(phase="VERIFYING", progress=0.85, message="校验")
            U.verify_bytes(Path(zip_path).read_bytes(), manifest)
            _set(phase="INSTALLING", progress=0.9, message="安装")
            info = U.install_zip(zip_path, target, manifest["version"],
                                 phase=lambda text: _set(message=text))
        _result = ("install", manifest, None, info)
    except U.UpdateError as exc:
        _result = ("install", manifest, str(exc), None)
    except Exception as exc:
        _result = ("install", manifest, f"更新失败：{exc}", None)


def _start(target, args):
    global _worker, _result
    _result = None
    _worker = threading.Thread(target=target, args=args, name="WondfulUpdate", daemon=True)
    _worker.start()
    if not bpy.app.timers.is_registered(poll):
        bpy.app.timers.register(poll, first_interval=0.25)


def start_check(manual: bool) -> bool:
    if busy():
        return False
    _set(phase="CHECKING", manual=manual, message="正在检查更新…")
    _start(_check_worker, (manual,))
    return True


def start_install(manifest: dict) -> bool:
    if busy() or not manifest:
        return False
    _set(phase="DOWNLOADING", manifest=manifest, message="下载中", progress=0.0, notes=[])
    _start(_install_worker, (manifest, str(ADDON_DIR)))
    return True


def wait(timeout=120.0) -> None:
    """Tests / background mode: join the worker and finish on this (main) thread."""
    if _worker is not None:
        _worker.join(timeout)
    poll()


def _finish_check(manifest, error, manual):
    state = U.load_state(state_dir())
    if error:
        if manual:
            _set(phase="CHECK_FAILED", message="检查失败：" + error.replace("检查失败：", "") +
                 "。可稍后再试，或到 GitHub 手动下载")
        else:
            _set(phase="IDLE", message="")   # auto check: network errors are silent
        state = U.record_check(state, None)  # count the attempt; keep the cached ``latest``
        U.save_state(state_dir(), state)
        return
    U.save_state(state_dir(), U.record_check(state, manifest))
    local = local_version()
    if U.is_newer(manifest["version"], local):
        _set(phase="AVAILABLE", manifest=manifest, message=f"发现新版本 {manifest['version']}")
    else:
        _set(phase="UP_TO_DATE", manifest=None, message=f"已是最新版本 {local}")


def _finish_install(manifest, error, info):
    if error:
        _set(phase="FAILED", progress=0.0,
             message=error + "。当前版本未改动；可到 GitHub 手动下载安装包")
        return
    notes = []
    dups = U.find_duplicates(ADDON_DIR, addon_roots())
    if dups:
        try:
            import addon_utils
            for path in dups:
                name = Path(path).name
                if name != PACKAGE and name in bpy.context.preferences.addons:
                    addon_utils.disable(name, default_set=True)
        except Exception as exc:
            notes.append(f"停用旧副本失败：{exc}")
        removed, failed = U.remove_dirs(dups)
        if removed:
            notes.append(f"已删除旧版本副本 {len(removed)} 个")
        for path, reason in failed:
            notes.append(f"旧副本删除失败：{path}（{reason}）")
    if info and not info.get("backup_removed"):
        notes.append("旧文件将在下次启动时清理")
    state = U.load_state(state_dir())
    state.pop("ignored", None)
    U.save_state(state_dir(), state)
    _set(phase="DONE", progress=1.0, installed=manifest["version"], notes=notes,
         message=f"已更新到 {manifest['version']}，请重启 Blender")


def poll():
    """Timer: redraw while working; finish on the main thread. Returns None to stop."""
    global _result
    result, _result = _result, None
    if result is not None:
        kind, manifest, error, extra = result
        if kind == "check":
            _finish_check(manifest, error, extra)
        else:
            _finish_install(manifest, error, extra)
    _redraw()
    if result is None and _worker is not None and _worker.is_alive():
        return 0.25
    return None


def _startup():
    """Deferred after register: cached result at once, a network check at most once per 24 h."""
    try:
        U.cleanup_leftovers(ADDON_DIR.parent)
        state = U.load_state(state_dir())
        pending = U.pending_update(state, local_version())
        if pending:
            _set(phase="AVAILABLE", manifest=pending, message=f"发现新版本 {pending['version']}")
            _redraw()
        if auto_check_enabled() and U.should_check(state):
            start_check(manual=False)
    except Exception as exc:
        print(f"[Wondful] 更新检查跳过：{exc}")
    return None


def schedule_startup_check():
    if bpy.app.background and not os.environ.get("WONDFUL_UPDATE_IN_BACKGROUND"):
        return   # command-line renders never touch the network
    try:
        if not bpy.app.timers.is_registered(_startup):
            bpy.app.timers.register(_startup, first_interval=4.0)
    except Exception:
        pass


def unregister_timers():
    for fn in (_startup, poll):
        try:
            if bpy.app.timers.is_registered(fn):
                bpy.app.timers.unregister(fn)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

class WONDFUL_OT_update_check(Operator):
    bl_idname = "wondful.update_check"
    bl_label = "检查更新"
    bl_description = "从 GitHub（含国内镜像）检查是否有新版本"

    def execute(self, context):
        if not start_check(manual=True):
            self.report({"INFO"}, "正在处理更新，请稍候")
            return {"CANCELLED"}
        if bpy.app.background:
            wait()
        return {"FINISHED"}


class WONDFUL_OT_update_install(Operator):
    bl_idname = "wondful.update_install"
    bl_label = "更新"
    bl_description = "下载新版本，校验 sha256 后覆盖当前插件；完成后重启 Blender 生效"

    def execute(self, context):
        manifest = STATE.get("manifest")
        if not manifest or busy():
            self.report({"WARNING"}, "没有可安装的新版本" if not manifest else "正在处理更新，请稍候")
            return {"CANCELLED"}
        start_install(manifest)
        if bpy.app.background:
            wait()
        return {"FINISHED"}


class WONDFUL_OT_update_ignore(Operator):
    bl_idname = "wondful.update_ignore"
    bl_label = "忽略此版本"
    bl_description = "不再提示这个版本；有更新的版本时会再次提示"

    version: StringProperty(default="", options={"HIDDEN", "SKIP_SAVE"})

    def execute(self, context):
        manifest = STATE.get("manifest") or {}
        version = self.version or manifest.get("version", "")
        if version:
            state = U.load_state(state_dir())
            state["ignored"] = version
            U.save_state(state_dir(), state)
        if STATE["phase"] in ("AVAILABLE", "FAILED"):
            _set(phase="IDLE", message="")
        _redraw()
        return {"FINISHED"}


class WONDFUL_OT_update_open_page(Operator):
    bl_idname = "wondful.update_open_page"
    bl_label = "GitHub 手动下载"
    bl_description = "打开 GitHub 页面手动下载安装包"

    def execute(self, context):
        bpy.ops.wm.url_open(url=U.REPO_PAGE + "/tree/main/releases")
        return {"FINISHED"}


CLASSES = (WONDFUL_OT_update_check, WONDFUL_OT_update_install, WONDFUL_OT_update_ignore,
           WONDFUL_OT_update_open_page)
