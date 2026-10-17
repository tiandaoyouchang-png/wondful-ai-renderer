"""Headless real run of 高级 › 环境体块 › 从参考图生成场景粗模 (1.2.3), then the normal 「生成」.

Installs the add-on ZIP, builds a product scene (yu7 car or SheenChair), puts the reference photo in ③,
clicks 从参考图生成场景粗模 (real analysis with the user's Codex login, CODEX_HOME), then 「生成」
(polish + simple direct edit). Background Blender has no window, so the modal loop is pumped by this script
(same harness as terrain_e2e/drive.py); the camera-frame OpenGL capture is replaced by a Cycles render.

    CODEX_HOME=… PATH=…codex…:$PATH blender -b --factory-startup --python tools/e2e_scene_blockout_bg.py -- \
        <addon.zip> <yu7|chair> <reference.jpg> <run_dir> [correct 0/1] [analysis_json]
"""
import json
import math
import os
import shutil
import sys
import threading
import time
import traceback

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
ZIP, PRODUCT, REF, RUN = argv[0], argv[1], argv[2], argv[3]
CORRECT = (argv[4] if len(argv) > 4 else "0") == "1"
CACHED = argv[5] if len(argv) > 5 else ""
os.makedirs(RUN, exist_ok=True)
LOG = open(os.path.join(RUN, "drive.log"), "a", buffering=1)
T0 = time.time()
PENDING = []
MODELS = {"yu7": "/workspace/work/bltest/yu7/yu7.glb", "chair": "/workspace/work/terrain_test/models/SheenChair.glb"}


def log(*a):
    msg = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n")


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
        log("  start_thread", type(self).__name__)
        return True
    OPS._BaseAsyncOperator._start_thread = _start_thread
    from wondful_ai_renderer import viewport_capture as VC

    def _capture(context, filepath):
        scene = context.scene
        w, h = VC.camera_output_dimensions(context)
        r = scene.render
        old = (r.filepath, r.resolution_x, r.resolution_y, r.resolution_percentage, r.engine, scene.cycles.samples,
               r.image_settings.file_format, scene.cycles.use_denoising)
        try:
            scene.cycles.use_denoising = False
            r.filepath, r.resolution_x, r.resolution_y, r.resolution_percentage = filepath, w, h, 100
            r.engine = "CYCLES"
            scene.cycles.samples = 8
            r.image_settings.file_format = "PNG"
            return "FINISHED" in bpy.ops.render.render(write_still=True) and os.path.exists(filepath)
        finally:
            (r.filepath, r.resolution_x, r.resolution_y, r.resolution_percentage, r.engine, scene.cycles.samples,
             r.image_settings.file_format, scene.cycles.use_denoising) = old
    VC._capture_camera_opengl = _capture

    real_ops = bpy.ops

    class _Op:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args, **kw):
            args = tuple("EXEC_DEFAULT" if a == "INVOKE_DEFAULT" else a for a in args)
            log("  bpy.ops.wondful." + self.name)
            return getattr(real_ops.wondful, self.name)(*args, **kw)

        def poll(self):
            return getattr(real_ops.wondful, self.name).poll()

    class _W:
        def __getattr__(self, name):
            return _Op(name)

    class _Ops:
        wondful = _W()

        def __getattr__(self, name):
            return getattr(real_ops, name)

    class _Timers:
        @staticmethod
        def register(fn, first_interval=0.0, **kw):
            PENDING.append([fn, time.time() + float(first_interval or 0.0)])

        def __getattr__(self, name):
            return getattr(bpy.app.timers, name)

    class _App:
        timers = _Timers()

        def __getattr__(self, name):
            return getattr(bpy.app, name)

    class _Bpy:
        ops = _Ops()
        app = _App()

        def __getattr__(self, name):
            return getattr(bpy, name)
    OPS.bpy = _Bpy()


def pump(OPS):
    now = time.time()
    for item in list(PENDING):
        if item[1] <= now:
            PENDING.remove(item)
            again = item[0]()
            if again is not None:
                PENDING.append([item[0], time.time() + float(again)])
    op = OPS._ACTIVE_OPERATOR
    if op is None:
        return None
    r = op.modal(bpy.context, FakeEvent())
    if "RUNNING_MODAL" not in r and "PASS_THROUGH" not in r:
        log("  modal finished", type(op).__name__, r)
        if OPS._ACTIVE_OPERATOR is op:
            OPS._ACTIVE_OPERATOR = None
    return r


def wait(OPS, limit):
    t = time.time()
    while time.time() - t < limit:
        if OPS._ACTIVE_OPERATOR is None and not OPS._ACTIVE_LOCK.locked() and not PENDING:
            return True
        time.sleep(0.5)
        pump(OPS)
    return False


def bbox(objs):
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ Vector(c) for o in objs if o.type == "MESH" for c in o.bound_box]
    return Vector([min(p[i] for p in pts) for i in range(3)]), Vector([max(p[i] for p in pts) for i in range(3)])


def build_scene():
    sc = bpy.context.scene
    for o in list(sc.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    coll = bpy.data.collections.new("产品")
    sc.collection.children.link(coll)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=MODELS[PRODUCT])
    new = [o for o in bpy.data.objects if o not in before]
    for o in new:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        coll.objects.link(o)
    root = bpy.data.objects.new("ProductRoot", None)
    coll.objects.link(root)
    for o in new:
        if o.parent is None:
            o.parent = root
    meshes = [o for o in coll.all_objects if o.type == "MESH"]
    if PRODUCT == "yu7":
        mn, mx = bbox(meshes)
        s = 5.0 / max((mx - mn).x, (mx - mn).y)
        root.scale = (s, s, s)
    mn, mx = bbox(meshes)
    ctr = (mn + mx) / 2
    root.location += Vector((-ctr.x, -ctr.y, -mn.z))
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    sc.collection.objects.link(cam)
    cam.location = (6, -8, 2)
    cam.rotation_euler = (math.radians(82), 0, math.radians(37))
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = 1600, 900
    log("scene", PRODUCT, "meshes", len(meshes), "size", tuple(round(v, 2) for v in (bbox(meshes)[1] - bbox(meshes)[0])))
    return coll


def main():
    rep = {"product": PRODUCT, "ref": REF}
    try:
        bpy.ops.preferences.addon_install(filepath=ZIP, overwrite=True)
        bpy.ops.preferences.addon_enable(module="wondful_ai_renderer")
        import addon_utils
        mod = next(m for m in addon_utils.modules() if m.__name__ == "wondful_ai_renderer")
        log("addon", mod.bl_info["version"], mod.__file__)
        from wondful_ai_renderer import operators as OPS
        patch(OPS)
        coll = build_scene()
        sc = bpy.context.scene
        props = sc.wondful_ai
        props.product_collection = coll
        r = bpy.ops.wondful.load_reference(ref_kind="STYLE", filepath=REF)
        log("load_reference", r, [i.source_path for i in props.style_images])
        props.output_directory = os.path.join(RUN, "out")
        props.ui_show_advanced = True
        props.ui_show_scene_blockout = True
        props.sb_set_camera = True
        props.sb_correct = CORRECT
        bpy.ops.wondful.codex_check()
        wait(OPS, 90)
        log("codex account", props.codex_account_state, props.codex_account_message[:80])
        # ---- 从参考图生成场景粗模 ------------------------------------------------------------
        t = time.time()
        kw = {"analysis_json": CACHED} if CACHED else {}
        r = bpy.ops.wondful.scene_blockout_build(**kw)
        log("scene_blockout_build ->", r)
        done = wait(OPS, 1500)
        rep["blockout_seconds"] = round(time.time() - t, 1)
        rep["blockout_note"] = props.last_sb_note
        rep["blockout_done"] = done
        log("blockout", rep["blockout_seconds"], "s", props.last_sb_note, props.last_error[:200])
        from wondful_ai_renderer import scene_blockout as SB
        from wondful_ai_renderer.render_session import CACHE_ROOT
        sess = sorted([p for p in CACHE_ROOT.iterdir() if (p / "scene_ref.png").is_file()], key=lambda p: p.stat().st_mtime)
        if sess:
            s = sess[-1]
            rep["blockout_session"] = str(s)
            for p in s.iterdir():
                if p.name.startswith("scene_"):
                    shutil.copy2(p, os.path.join(RUN, p.name))
        coll_sb = bpy.data.collections.get(SB.COLLECTION_NAME)
        rep["blocks"] = [o.name for o in coll_sb.objects if o.parent is None] if coll_sb else []
        rep["scene_camera"] = sc.camera.name if sc.camera else ""
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(RUN, "scene_blockout.blend"), copy=True)
        # viewport-style render of the blockout from the scene camera (own tints, product included)
        from wondful_ai_renderer import scene_blockout_build as SBB
        info = json.load(open(os.path.join(RUN, "scene_blockout_info.json"))) if os.path.isfile(
            os.path.join(RUN, "scene_blockout_info.json")) else None
        if info:
            os.environ["WONDFUL_SB_PREVIEW_ENGINE"] = "CYCLES"
            SBB.render_preview(bpy.context, info, os.path.join(RUN, "blockout_view.png"), width=800)
        # ---- 生成 (normal flow) --------------------------------------------------------------
        t = time.time()
        r = bpy.ops.wondful.generate()
        log("generate ->", r, props.status, props.last_error[:200])
        wait(OPS, 1800)
        rep["generate_seconds"] = round(time.time() - t, 1)
        rep.update({"status": props.status, "last_error": props.last_error, "last_export_path": props.last_export_path,
                    "engine_note": props.last_engine_note, "prompt": props.prompt,
                    "score_note": getattr(props, "last_score_note", "")})
        from wondful_ai_renderer import viewport_capture as VC
        rep["massing_text"] = (VC.LAST_CLAY_INFO or {}).get("massing_text", "")
        rep["clay_user_ground"] = (VC.LAST_CLAY_INFO or {}).get("user_ground", "")
        gs = CACHE_ROOT / props.last_session_id if props.last_session_id else None
        if gs and gs.is_dir():
            rep["generate_session"] = str(gs)
            for p in gs.iterdir():
                if p.name.startswith("codex_direct") or p.name in ("01_clay_base.png", "prompt.txt", "env_descriptions.json"):
                    shutil.copy2(p, os.path.join(RUN, p.name))
        if props.last_export_path and os.path.isfile(props.last_export_path):
            shutil.copy2(props.last_export_path, os.path.join(RUN, "result.png"))
        rep["ok"] = bool(props.last_export_path)
    except Exception:
        rep["error"] = traceback.format_exc()
        log("driver error", rep["error"])
    with open(os.path.join(RUN, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=str)
    log("DONE", rep.get("status"), rep.get("last_export_path"))
    LOG.close()
    os._exit(0 if rep.get("ok") else 2)


main()
