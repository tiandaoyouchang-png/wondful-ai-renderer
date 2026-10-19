"""Real headless run of ④ 场景粗模, current flow vs 精确模式 (1.2.5), with the user's Codex login (CODEX_HOME).

Builds a box-car product scene, puts <reference> in ④, runs WONDFUL_OT_scene_blockout_build with 自动校正一轮 on
(real analysis + real correction round), saves the .blend + the panel note + the session files, then renders clay
(material override, transparent film) from the reference camera, an orbit +35° around the product spot and a raised
3/4 view (same cameras as blockout_research/render_variants.py).

    CODEX_HOME=… blender -b --factory-startup --python tools/e2e_precise_compare_bg.py -- \
        repo <reference.jpg> <out_dir> <precise 0|1> [recorded_geo.npz]
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
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
REF, OUTD, PRECISE = argv[1], argv[2], argv[3] == "1"
GEO = argv[4] if len(argv) > 4 else ""
os.makedirs(OUTD, exist_ok=True)


class FakeEvent:
    type = "TIMER"
    value = "NOTHING"


def log(*a):
    print("[compare]", *a, flush=True)


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

    # background Blender has no window: the preview render used for the fit score goes through Cycles
    from wondful_ai_renderer import scene_blockout_build as SBB
    return SBB


def clay_renders(prefix):
    sc = bpy.context.scene
    m = bpy.data.materials.new("clay")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.62, 0.62, 0.64, 1)
    b.inputs["Roughness"].default_value = 0.9
    vl = bpy.context.view_layer
    old_override = vl.material_override
    vl.material_override = m
    sun = bpy.data.objects.new("clay_sun", bpy.data.lights.new("clay_sun", "SUN"))
    sc.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    sun.data.energy = 3.5
    hidden = []
    for o in sc.objects:
        if o.name.startswith("Car"):
            hidden.append((o, o.hide_render))
            o.hide_render = True
    if sc.world is None:
        sc.world = bpy.data.worlds.new("w")
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.8, 0.85, 0.9, 1)
    bg.inputs[1].default_value = 0.6
    r = sc.render
    saved = (r.engine, r.resolution_x, r.resolution_y, r.film_transparent, r.filepath)
    r.engine = "CYCLES"
    sc.cycles.samples = 12
    sc.cycles.device = "CPU"
    sc.cycles.use_denoising = False
    r.resolution_x, r.resolution_y, r.resolution_percentage = 800, 450, 100
    r.film_transparent = True
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"
    co = sc.camera
    L0, E0 = co.location.copy(), co.rotation_euler.copy()
    pivot = Vector((0, 0, 0))            # plugin build: product spot = world origin

    def shot(tag):
        r.filepath = f"{prefix}_{tag}.png"
        bpy.ops.render.render(write_still=True)
    shot("ref")
    fw = (co.matrix_world.to_3x3() @ Vector((0, 0, -1)))
    fw.z = 0
    fw.normalize()
    for tag, yaw_deg, lift in [("orbit35", 35, 0.0), ("high", 0, 1.0)]:
        if lift:
            loc = L0 - fw * 10 + Vector((0, 0, 12))
            tgt = Vector((L0.x, L0.y, 0)) + fw * 30
            loc = loc + Vector((fw.y, -fw.x, 0)) * 6
        else:
            loc = pivot + Matrix.Rotation(math.radians(yaw_deg), 3, "Z") @ (L0 - pivot)
            tgt = pivot + Vector((0, 0, 1.0))
        co.location = loc
        co.rotation_euler = (tgt - loc).to_track_quat("-Z", "Y").to_euler()
        shot(tag)
    co.location, co.rotation_euler = L0, E0
    vl.material_override = old_override
    for o, hr in hidden:
        o.hide_render = hr
    bpy.data.objects.remove(sun, do_unlink=True)
    r.engine, r.resolution_x, r.resolution_y, r.film_transparent, r.filepath = saved


def main():
    rep = {"precise": PRECISE, "ref": REF}
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        zp = shutil.make_archive(os.path.join(tempfile.mkdtemp(), "wondful_repo"), "zip", root, "wondful_ai_renderer")
        bpy.ops.preferences.addon_install(filepath=zp, overwrite=True)
        bpy.ops.preferences.addon_enable(module="wondful_ai_renderer")
        from wondful_ai_renderer import operators as OPS
        patch(OPS)
        OPS._analysis_provider = lambda props, prefs: {"id": "codex"}
        sc = bpy.context.scene
        for o in list(sc.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        coll = bpy.data.collections.new("产品")
        sc.collection.children.link(coll)
        me = bpy.data.meshes.new("Car")
        w, d, h = 1.9, 4.8, 1.5
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
        props.sb_image_path = REF
        props.sb_set_camera = True
        props.sb_correct = True
        props.sb_precise = PRECISE
        kw = {"precise_geo": GEO} if (PRECISE and GEO) else {}
        t = time.time()
        r = bpy.ops.wondful.scene_blockout_build(**kw)
        stages = []
        first_op = OPS._ACTIVE_OPERATOR
        while time.time() - t < 1500:
            op = OPS._ACTIVE_OPERATOR
            if op is None and not OPS._ACTIVE_LOCK.locked():
                break
            time.sleep(0.3)
            if op is not None:
                res = op.modal(bpy.context, FakeEvent())
                if "RUNNING_MODAL" not in res and "PASS_THROUGH" not in res and OPS._ACTIVE_OPERATOR is op:
                    OPS._ACTIVE_OPERATOR = None
            if not stages or stages[-1][0] != props.sb_stage:
                stages.append((props.sb_stage, round(time.time() - t, 1)))
                log(props.sb_stage, round(time.time() - t, 1))
        rep.update({"op": sorted(r), "stages": stages, "note": props.last_sb_note, "seconds": round(time.time() - t, 1),
                    "final_stage": props.sb_stage})
        log("NOTE", props.last_sb_note)
        try:
            sess = str(first_op._sb["session"])
            rep["session"] = sess
            shutil.copytree(sess, os.path.join(OUTD, "session"), dirs_exist_ok=True)
        except Exception as exc:
            rep["session_error"] = str(exc)
        from wondful_ai_renderer import scene_blockout as SB
        txt = bpy.data.texts.get(SB.TEXT_NAME)
        if txt:
            open(os.path.join(OUTD, "built.json"), "w", encoding="utf-8").write(txt.as_string())
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUTD, "scene.blend"))
        clay_renders(os.path.join(OUTD, "clay"))
        rep["ok"] = True
    except Exception:
        rep["error"] = traceback.format_exc()
        log(rep["error"])
    with open(os.path.join(OUTD, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=str)
    os._exit(0)


main()
