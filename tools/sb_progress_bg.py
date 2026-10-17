"""Headless check of 从参考图生成场景粗模 stage progress (1.2.3), no network.

Installs the add-on ZIP, builds a small product scene, then runs WONDFUL_OT_scene_blockout_build with a fake
analysis (fixture JSON, ``SB.call_analysis`` patched to sleep then return it) and pumps the modal loop like
tools/e2e_scene_blockout_bg.py. Records every distinct (stage, progress) the panel would show.

    blender -b --factory-startup --python tools/sb_progress_bg.py -- <addon.zip | repo> <fixture.json> <out.json>

Runs: plain (no correction), with 自动校正一轮, cached analysis_json, analysis failure, cancel during analysis.
"""
import json
import math
import os
import sys
import tempfile
import threading
import time
import traceback

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
ZIP, FIX, OUT = argv[0], argv[1], argv[2]


class FakeEvent:
    type = "TIMER"
    value = "NOTHING"


def patch(OPS):
    from wondful_ai_renderer.cli_transport import reset_cancellation

    def _start_thread(self, context, target):
        if not OPS._ACTIVE_LOCK.acquire(blocking=False):
            return False
        OPS._ACTIVE_OPERATOR = self
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
        self._thread = threading.Thread(target=runner, daemon=True)
        self._timer = None
        self._thread.start()
        return True
    OPS._BaseAsyncOperator._start_thread = _start_thread


def run(OPS, props, kw, limit=120, cancel_at=None):
    seen, t = [], time.time()
    r = bpy.ops.wondful.scene_blockout_build(**kw)

    def sample():
        rec = [props.status, props.sb_stage, round(props.progress, 3), props.task_phase, props.sb_elapsed]
        if not seen or seen[-1][:3] != rec[:3]:
            seen.append(rec)
    sample()
    while time.time() - t < limit:
        op = OPS._ACTIVE_OPERATOR
        if op is None and not OPS._ACTIVE_LOCK.locked():
            break
        if cancel_at and props.sb_stage == cancel_at and time.time() - t > 0.6:
            bpy.ops.wondful.cancel_task()
            cancel_at = None
        time.sleep(0.25)
        if op is not None:
            res = op.modal(bpy.context, FakeEvent())
            if "RUNNING_MODAL" not in res and "PASS_THROUGH" not in res and OPS._ACTIVE_OPERATOR is op:
                OPS._ACTIVE_OPERATOR = None
        sample()
    return {"op": sorted(r), "trace": seen, "note": props.last_sb_note, "status": props.status,
            "failed_stage": props.sb_failed_stage, "final_stage": props.sb_stage, "elapsed": props.sb_elapsed,
            "camera": bpy.context.scene.camera.name if bpy.context.scene.camera else "",
            "seconds": round(time.time() - t, 1)}


def main():
    rep = {}
    try:
        zp = ZIP
        if ZIP == "repo":   # unit tests: zip the package straight from this checkout
            import shutil
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            zp = shutil.make_archive(os.path.join(tempfile.mkdtemp(), "wondful_repo"), "zip", root, "wondful_ai_renderer")
        bpy.ops.preferences.addon_install(filepath=zp, overwrite=True)
        bpy.ops.preferences.addon_enable(module="wondful_ai_renderer")
        from wondful_ai_renderer import operators as OPS, scene_blockout as SB, png_io
        import numpy as np
        patch(OPS)
        sc = bpy.context.scene
        for o in list(sc.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        coll = bpy.data.collections.new("产品")
        sc.collection.children.link(coll)
        me = bpy.data.meshes.new("Car")
        w, d, h = 1.9, 4.6, 1.5
        me.from_pydata([(x * w / 2, y * d / 2, z * h) for x in (-1, 1) for y in (-1, 1) for z in (0, 1)], [],
                       [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
        coll.objects.link(bpy.data.objects.new("Car", me))
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        sc.collection.objects.link(cam)
        cam.location = (6, -8, 2)
        cam.rotation_euler = (math.radians(82), 0, math.radians(37))
        sc.camera = cam
        sc.render.resolution_x, sc.render.resolution_y = 1600, 900
        props = sc.wondful_ai
        props.product_collection = coll
        tmp = tempfile.mkdtemp()
        ref = os.path.join(tmp, "env_ref.png")
        img = np.zeros((90, 160, 4), np.uint8)
        img[:45] = (120, 170, 230, 255)
        img[45:] = (190, 160, 110, 255)
        png_io.write_png(ref, img)
        props.sb_image_path = ref
        props.sb_set_camera = True
        fixture = json.load(open(FIX, encoding="utf-8"))
        calls = []
        mode = {"fail": False, "delay": 2.0}

        def fake_call(content, **kw):
            calls.append(kw.get("effort"))
            t0 = time.time()
            while time.time() - t0 < mode["delay"]:
                if kw.get("cancel_check") and kw["cancel_check"]():
                    raise RuntimeError("cancelled")
                time.sleep(0.1)
            if mode["fail"]:
                raise RuntimeError("模拟网络超时")
            return json.loads(json.dumps(fixture))
        SB.call_analysis = fake_call
        # the operator's provider gate: pretend Codex is the analysis provider
        OPS._analysis_provider = lambda props, prefs: {"id": "codex"}
        props.sb_correct = False
        rep["plain"] = run(OPS, props, {})
        props.sb_correct = True
        rep["correct"] = run(OPS, props, {})
        rep["correct"]["calls"] = list(calls)
        props.sb_correct = False
        rep["cached"] = run(OPS, props, {"analysis_json": FIX})
        mode["fail"] = True
        rep["fail"] = run(OPS, props, {})
        mode.update(fail=False, delay=30.0)
        rep["cancel"] = run(OPS, props, {}, cancel_at="ANALYZE")
        rep["ok"] = True
    except Exception:
        rep["error"] = traceback.format_exc()
        print(rep["error"])
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=str)
    os._exit(0)


main()
