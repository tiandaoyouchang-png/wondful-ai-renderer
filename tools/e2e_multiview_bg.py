"""Headless real run of 高级 › 多角度一致 (1.2.3): 1 master + N cameras through the operator.

Background Blender has no window, so the modal loop is driven by this script: the worker job
runs synchronously and ``_modal(context, TIMER)`` is called until it finishes. Everything else
is the real add-on (operators, clay / detail / geometry passes, projection, Codex direct edit,
identity score, detail paste, export) against the user's real Codex login (CODEX_HOME).

    CODEX_HOME=… blender -b scene.blend --python tools/e2e_multiview_bg.py -- <addon.zip> <out_dir> \
        <master_cam> <master_image> <cam2,cam3,…> [brief_file] [hints_json]
"""
import json
import os
import sys
import time
import traceback

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
ZIP, OUT, MASTER, MASTER_IMG, OTHERS = argv[:5]
BRIEF = open(argv[5], encoding="utf-8").read().strip() if len(argv) > 5 and argv[5] else ""
HINTS = json.load(open(argv[6], encoding="utf-8")) if len(argv) > 6 and argv[6] else {}
os.makedirs(OUT, exist_ok=True)


class FakeEvent:
    type = "TIMER"


def main():
    report = {"ok": False}
    t0 = time.time()
    bpy.ops.preferences.addon_install(filepath=ZIP, overwrite=True)
    bpy.ops.preferences.addon_enable(module="wondful_ai_renderer")
    import wondful_ai_renderer.operators as ops
    prefs = bpy.context.preferences.addons["wondful_ai_renderer"].preferences
    prefs.codex_direct_edit = True
    prefs.codex_direct_simple = True

    def sync_start_thread(self, context, target):
        import threading
        if not ops._ACTIVE_LOCK.acquire(blocking=False):
            return False
        ops._ACTIVE_OPERATOR = self
        self._owns_lock = True
        self._ownership_guard = threading.Lock()
        self._detached = threading.Event()
        self._origin_scene = context.scene
        self._result, self._error, self._started_at, self._timer, self._thread = None, None, time.time(), None, None
        try:
            self._result = target()
        except Exception as exc:
            self._error = (exc, traceback.format_exc())
            print(self._error[1])
        for _ in range(100):
            res = self._modal(bpy.context, FakeEvent())
            if "RUNNING_MODAL" not in res:
                report["modal_result"] = sorted(res)
                break
        return True
    ops._BaseAsyncOperator._start_thread = sync_start_thread

    scene = bpy.context.scene
    props = scene.wondful_ai
    props.product_collection = bpy.data.collections["产品"]
    props.massing_collection = bpy.data.collections["环境体块"]
    props.output_directory = OUT
    props.scene_brief = BRIEF
    for o in props.massing_collection.all_objects:
        for key, hint in HINTS.items():
            if key in o.name:
                o["wondful_hint"] = hint
                break
    master = bpy.data.objects[MASTER]
    scene.camera = master
    props.mv_master_camera = master
    props.mv_master_image = MASTER_IMG
    others = [bpy.data.objects[n] for n in OTHERS.split(",") if n]
    for o in bpy.context.view_layer.objects:
        o.select_set(o in others)
    bpy.context.view_layer.objects.active = others[0]
    res = bpy.ops.wondful.multiview_render()
    report["operator"] = sorted(res)
    report["note"] = props.last_mv_note
    report["status"] = props.status
    report["error"] = props.last_error
    report["exported"] = json.loads(props.last_export_paths or "[]")
    report["seconds_total"] = round(time.time() - t0, 1)
    report["camera_restored"] = scene.camera == master
    report["aux_left"] = [o.name for o in bpy.data.objects if o.get("wondful_mv_aux")]
    report["ok"] = bool(report["exported"]) and not report["aux_left"] and report["camera_restored"]
    with open(os.path.join(OUT, "e2e_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("E2E", json.dumps(report, ensure_ascii=False))


try:
    main()
except Exception:
    traceback.print_exc()
    sys.exit(1)
