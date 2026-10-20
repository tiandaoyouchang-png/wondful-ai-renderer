"""Headless check of ④ 场景粗模 · 细节粗模 (1.2.5, V3 phase 1), no network for the AI step.

Fake analysis = fixture JSON, synthetic depth measurement (precise_synth). Runs: baseline (精确模式 off),
on (精确模式 + recorded geo → hidden 「Wondful 细节粗模（投影）」), the geometry pass of one orbit camera with
detail "" / "carrier" (+ again after removing the carrier: must be identical), option off, interior fallback.

    blender -b --factory-startup --python tools/e2e_detail_blockout_bg.py -- repo <fixture.json> <out.json>
"""
import json
import math
import os
import shutil
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



def run(OPS, props, kw, limit=300):
    t = time.time()
    r = bpy.ops.wondful.scene_blockout_build(**kw)
    while time.time() - t < limit:
        op = OPS._ACTIVE_OPERATOR
        if op is None and not OPS._ACTIVE_LOCK.locked():
            break
        time.sleep(0.2)
        if op is not None:
            res = op.modal(bpy.context, FakeEvent())
            if "RUNNING_MODAL" not in res and "PASS_THROUGH" not in res and OPS._ACTIVE_OPERATOR is op:
                OPS._ACTIVE_OPERATOR = None
    from wondful_ai_renderer import scene_blockout as SB, detail_blockout as DB
    coll = bpy.data.collections.get(SB.COLLECTION_NAME)
    objs = list(coll.all_objects) if coll else []
    car = [o for o in objs if o.get(DB.CARRIER_TAG)]
    tris = 0
    for o in car:
        o.data.calc_loop_triangles()
        tris += len(o.data.loop_triangles)
    return {"op": sorted(r), "note": props.last_sb_note, "objects": sorted(o.name for o in objs),
            "llm_objects": sorted(o.name for o in objs if not o.get(DB.CARRIER_TAG)),
            "carrier_objects": len(car), "carrier_tris": tris,
            "carrier_hidden": all(o.hide_render and o.hide_viewport for o in car),
            "far_kept": sum(int(o.get(DB.FAR_TAG, 0) or 0) for o in objs), "seconds": round(time.time() - t, 1)}


def geometry_pass(props, prefix, detail):
    import numpy as np
    from wondful_ai_renderer.viewport_capture import render_clay_base, LAST_CLAY_INFO
    from wondful_ai_renderer import scene_blockout as SB
    massing = bpy.data.collections.get(SB.COLLECTION_NAME)
    ok = render_clay_base(bpy.context, prefix + "_clay.png", props.product_collection, massing=massing,
                          geometry_prefix=prefix, geometry_size=(240, 135), detail=detail)
    return ok, np.load(prefix + "_pos.npy"), dict(LAST_CLAY_INFO)


def main():
    rep = {}
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        zp = shutil.make_archive(os.path.join(tempfile.mkdtemp(), "wondful_repo"), "zip", root, "wondful_ai_renderer")
        bpy.ops.preferences.addon_install(filepath=zp, overwrite=True)
        bpy.ops.preferences.addon_enable(module="wondful_ai_renderer")
        from wondful_ai_renderer import operators as OPS, scene_blockout as SB, png_io
        from wondful_ai_renderer import precise_depth as PD, detail_blockout as DB, multiview as MV
        import numpy as np
        import precise_synth
        patch(OPS)
        os.environ["WONDFUL_CLAY_ENGINE"] = "CYCLES"
        os.environ["WONDFUL_CLAY_SAMPLES"] = "2"
        os.environ["WONDFUL_CLAY_DENOISE"] = "0"
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
        state = {"fixture": fixture}
        SB.call_analysis = lambda content, **kw: json.loads(json.dumps(state["fixture"]))
        OPS._analysis_provider = lambda props, prefs: {"id": "codex"}
        geo = precise_synth.geo({"height_m": 1.6, "pitch_deg": 0.0, "roll_deg": 0.0, "hfov_deg": 60.0}, W=480, H=270,
                                boxes=[(-30, -8, 25, 40, 0, 9, PD.STRUCT), (8, 30, 20, 35, 0, 12, PD.STRUCT),
                                       (-5, -2, 14, 17, 2, 7, PD.VEG)])
        geo_path = os.path.join(tmp, "geo.npz")
        PD.save_geo(geo_path, geo)
        rep["baseline"] = run(OPS, props, {})
        props.sb_precise = True
        rep["on"] = run(OPS, props, {"precise_geo": geo_path})
        # geometry passes from an orbit camera 40° around the product
        oc = bpy.data.objects.new("Orbit", bpy.data.cameras.new("Orbit"))
        sc.collection.objects.link(oc)
        ref_cam = sc.camera
        rc = ref_cam.matrix_world.translation
        from mathutils import Matrix, Vector
        piv = Vector((0, 0, 0))
        loc = piv + Matrix.Rotation(math.radians(40), 3, "Z") @ (rc - piv) + Vector((0, 0, 1.5))
        oc.location = loc
        oc.rotation_euler = ((piv + Vector((0, 0, 1))) - loc).to_track_quat("-Z", "Y").to_euler()
        oc.data.lens = ref_cam.data.lens
        oc.data.sensor_fit, oc.data.sensor_width = "HORIZONTAL", 36
        sc.camera = oc
        vis0 = {o.name: (o.hide_render, o.hide_viewport) for o in bpy.data.objects}
        okA, A, _ia = geometry_pass(props, os.path.join(tmp, "A"), "")
        okB, B, ib = geometry_pass(props, os.path.join(tmp, "B"), "carrier")
        vis1 = {o.name: (o.hide_render, o.hide_viewport) for o in bpy.data.objects}
        DB.remove_carrier()
        okC, C, _ic = geometry_pass(props, os.path.join(tmp, "C"), "carrier")
        rec = MV.scene_camera_record(sc, oc)
        D = MV.rays(rec, 240, 135)
        below = D[..., 2] < -0.02
        hole = float(((B[..., 3] < 0.5) & below).sum()) / max(1, int(below.sum()))
        rep["passes"] = {"ok": [okA, okB, okC], "detail_info": ib.get("detail_blockout"),
                         "llm_identical_with_and_without_carrier": bool(np.array_equal(A, C)),
                         "carrier_equals_llm": bool(np.array_equal(A, B)),
                         "carrier_hole_below_horizon": round(hole, 5),
                         "llm_hole_below_horizon": round(float(((A[..., 3] < 0.5) & below).sum()) / max(1, int(below.sum())), 5),
                         "restored": vis0 == {k: v for k, v in vis1.items() if k in vis0}}
        sc.camera = ref_cam
        props.sb_detail_carrier = False
        rep["option_off"] = run(OPS, props, {"precise_geo": geo_path})
        props.sb_detail_carrier = True
        props.sb_precise = False
        rep["off"] = run(OPS, props, {})
        props.sb_precise = True
        inter = os.path.join(os.path.dirname(FIX), "interior_v3.json")
        state["fixture"] = json.load(open(inter, encoding="utf-8"))
        rep["interior"] = run(OPS, props, {"precise_geo": geo_path})
        rep["ok"] = True
    except Exception:
        rep["error"] = traceback.format_exc()
        print(rep["error"])
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=str)
    os._exit(0)


main()
