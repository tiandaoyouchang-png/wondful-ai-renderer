"""Headless check of ④ 场景粗模 ·「精确模式」(1.2.5), no network for the AI step.

Same harness as tools/sb_progress_bg.py (fake analysis = fixture JSON; modal loop pumped by this script). Runs:

* baseline — 精确模式 off (the property never touched)
* off      — 精确模式 explicitly off after an on-run: must equal baseline (same stages, camera, objects, prompt)
* on       — 精确模式 on with a recorded depth measurement (synthetic: flat ground + facades, hFOV 60°, h 1.6 m)
* fallback — 精确模式 on, empty cache, every download mirror unreachable → normal mode, one analysis call
* correct  — 精确模式 on + 自动校正一轮 → 2 calls, the correction prompt carries the PRECISE MODE line
* real     — only with WONDFUL_PRECISE_REAL=<reference.jpg>: real first-use install into a temp cache (PyPI + HF)
             and real MoGe inference in this Blender's Python

    blender -b --factory-startup --python tools/e2e_precise_bg.py -- <addon.zip | repo> <fixture.json> <out.json>
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))


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


def camera_values():
    from wondful_ai_renderer import scene_blockout as SB
    txt = bpy.data.texts.get(SB.TEXT_NAME)
    cam = json.loads(txt.as_string()).get("camera", {}) if txt else {}
    c = bpy.context.scene.camera
    out = {k: round(float(cam.get(k, 0)), 3) for k in ("height_m", "pitch_deg", "roll_deg", "hfov_deg")}
    if c is not None and c.type == "CAMERA":
        out["lens"] = round(c.data.lens, 3)
        out["z"] = round(c.matrix_world.translation.z - 0.0, 3)
    return out


def run(OPS, props, kw, calls, limit=600):
    seen, t = [], time.time()
    n0 = len(calls)
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
        time.sleep(0.2)
        if op is not None:
            res = op.modal(bpy.context, FakeEvent())
            if "RUNNING_MODAL" not in res and "PASS_THROUGH" not in res and OPS._ACTIVE_OPERATOR is op:
                OPS._ACTIVE_OPERATOR = None
        sample()
    mine = calls[n0:]
    from wondful_ai_renderer import scene_blockout as SB
    coll = bpy.data.collections.get(SB.COLLECTION_NAME)
    objs = sorted(o.name for o in coll.all_objects) if coll else []
    return {"op": sorted(r), "trace": seen, "note": props.last_sb_note, "status": props.status,
            "failed_stage": props.sb_failed_stage, "final_stage": props.sb_stage,
            "camera_values": camera_values(), "objects": objs, "calls": len(mine),
            "measured_in_prompt": any("MEASURED GEOMETRY" in c["text"] for c in mine[:1]),
            "precise_text_in_correction": any("PRECISE MODE" in c["text"] for c in mine[1:]),
            "seconds": round(time.time() - t, 1)}


def main():
    rep = {}
    try:
        zp = ZIP
        if ZIP == "repo":
            import shutil
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            zp = shutil.make_archive(os.path.join(tempfile.mkdtemp(), "wondful_repo"), "zip", root, "wondful_ai_renderer")
        bpy.ops.preferences.addon_install(filepath=zp, overwrite=True)
        bpy.ops.preferences.addon_enable(module="wondful_ai_renderer")
        from wondful_ai_renderer import operators as OPS, scene_blockout as SB, png_io, precise_setup as PS
        from wondful_ai_renderer import precise_depth as PD
        import numpy as np
        import precise_synth
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
        props.sb_correct = False
        fixture = json.load(open(FIX, encoding="utf-8"))
        calls = []

        def fake_call(content, **kw):
            calls.append({"effort": kw.get("effort"),
                          "text": "\n".join(c.get("text", "") for c in content if c.get("type") == "input_text")})
            time.sleep(0.5)
            return json.loads(json.dumps(fixture))
        SB.call_analysis = fake_call
        OPS._analysis_provider = lambda props, prefs: {"id": "codex"}
        geo = precise_synth.geo({"height_m": 1.6, "pitch_deg": 0.0, "roll_deg": 0.0, "hfov_deg": 60.0},
                                boxes=[(-30, -8, 25, 40, 0, 9, PD.STRUCT), (8, 30, 20, 35, 0, 12, PD.STRUCT)])
        geo_path = os.path.join(tmp, "geo.npz")
        PD.save_geo(geo_path, geo)

        rep["baseline"] = run(OPS, props, {}, calls)
        props.sb_precise = True
        rep["on"] = run(OPS, props, {"precise_geo": geo_path}, calls)
        props.sb_precise = False
        rep["off"] = run(OPS, props, {}, calls)
        # install failure → normal mode
        props.sb_precise = True
        os.environ["WONDFUL_PRECISE_DIR"] = os.path.join(tmp, "empty_cache")
        saved = (PS.MODEL_URLS, PS.WHEEL_BASES, PS._ort_already_importable, PS.platform_key)
        PS.MODEL_URLS = ("http://127.0.0.1:9/model.onnx",)
        PS.WHEEL_BASES = ("http://127.0.0.1:9",)
        PS._ort_already_importable = lambda: False
        PS.platform_key = lambda *a, **k: "cp311-linux_x86_64"
        rep["fallback"] = run(OPS, props, {}, calls)
        PS.MODEL_URLS, PS.WHEEL_BASES, PS._ort_already_importable, PS.platform_key = saved
        os.environ.pop("WONDFUL_PRECISE_DIR", None)
        props.sb_correct = True
        rep["correct"] = run(OPS, props, {"precise_geo": geo_path}, calls)
        props.sb_correct = False
        real = os.environ.get("WONDFUL_PRECISE_REAL", "").strip()
        if real:
            os.environ["WONDFUL_PRECISE_DIR"] = os.environ.get("WONDFUL_PRECISE_REAL_DIR") or os.path.join(tmp, "real_cache")
            props.sb_image_path = real
            rep["real_status_before"] = PS.status()
            rep["real"] = run(OPS, props, {}, calls, limit=1200)
            rep["real_status_after"] = PS.status()
        rep["ok"] = True
    except Exception:
        rep["error"] = traceback.format_exc()
        print(rep["error"])
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=str)
    os._exit(0)


main()
