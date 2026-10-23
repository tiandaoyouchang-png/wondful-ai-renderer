"""Headless E2E of ⑥ 渲染视角 (1.2.3): 场景粗模 in the scene, 3 cameras dragged into the list (one by one + a
collection), 2 checked → 「生成 2 个视角」 → main view (polish + AI render, mock Codex backend) → 多角度场景一致
for the second view in list order → result card flips between both.  Also checks the visible output folder
holds only images (internal files under the hidden .wondful folder).

Faked (as in e2e_generate_chain_bg.py): the modal loop, bpy.app.timers, window contexts, the OpenGL camera
capture, prompt polish; the image model: Codex direct edit against tests/mock_codex_backend.py for the main
view, and for the 多角度 views a local edit that returns its projection base (no network); light estimate.
The 多角度 clay / mask / geometry passes are real Blender renders (Cycles in background).

    blender -b --factory-startup --python tools/e2e_render_views_bg.py -- <addon.zip> <out_dir>
"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ZIP, OUT = argv[0], argv[1]
PURE_SCENE = len(argv) > 2 and argv[2] == "pure-scene"
os.makedirs(OUT, exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
REPORT = {"ok": False}
TIMERS, CLOCK = [], [0.0]
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".psd", ".exr", ".tif", ".tiff"}


class FakeEvent:
    type = "TIMER"


def run_timers(props, phases, limit=900):
    t0 = time.time()
    while TIMERS and time.time() - t0 < limit:
        TIMERS.sort(key=lambda t: t[0])
        due, fn = TIMERS.pop(0)
        CLOCK[0] = max(CLOCK[0], due)
        again = fn()
        if isinstance(again, (int, float)):
            TIMERS.append([CLOCK[0] + float(again), fn])
            time.sleep(0.001)


def visible_files(folder):
    """Every file under ``folder`` outside dot-folders (what the user sees in 打开文件夹)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        out += [os.path.relpath(os.path.join(dirpath, f), folder) for f in filenames if not f.startswith(".")]
    return sorted(out)


def main():
    import mock_codex_backend as mock
    server, _state = mock.start_server(mode="ok", log_path=os.path.join(OUT, "mock.jsonl"))
    home = tempfile.mkdtemp(prefix="wondful_e2e_codex_")
    mock.write_fake_auth(os.path.join(home, "auth.json"))
    os.environ["CODEX_HOME"] = home
    os.environ["no_proxy"] = os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    port = server.server_address[1]
    os.environ["WONDFUL_CODEX_IMAGE_BASE_URL"] = f"http://127.0.0.1:{port}/backend-api/codex"
    os.environ["WONDFUL_CODEX_REFRESH_URL"] = f"http://127.0.0.1:{port}/oauth/token"

    # no GPU / EGL in background: Eevee would abort instead of raising
    for key in ("WONDFUL_CLAY_ENGINE", "WONDFUL_STRUCTURE_ENGINE", "WONDFUL_DETAIL_ENGINE", "WONDFUL_MASK_ENGINE"):
        os.environ.setdefault(key, "CYCLES")
    os.environ.setdefault("WONDFUL_CLAY_SAMPLES", "4")
    bpy.ops.preferences.addon_install(filepath=ZIP, overwrite=True)
    bpy.ops.preferences.addon_enable(module="wondful_ai_renderer")
    import wondful_ai_renderer.operators as ops
    import wondful_ai_renderer.op_chain as chain
    import wondful_ai_renderer.codex_direct as CD
    import wondful_ai_renderer.multiview as MV
    from wondful_ai_renderer import png_io
    import numpy as np
    prefs = bpy.context.preferences.addons["wondful_ai_renderer"].preferences
    prefs.codex_direct_simple = False
    prefs.structure_guides_enabled = False
    prefs.codex_direct_edit = True

    ops.discover_codex_cli = lambda *a, **k: "/usr/bin/true"
    ops.codex_polish_prompt = lambda **k: "沙漠清晨，暖色侧光的产品场景。"
    ops.codex_generate_image = lambda **k: (_ for _ in ()).throw(RuntimeError("codex exec fallback must not be used"))

    def fake_capture(context, filepath):
        w, h = ops.camera_output_dimensions(context)
        png_io.write_png(filepath, np.full((h, w, 4), 150, np.uint8))
        return w, h, "E2E_FAKE_CAPTURE"
    ops.capture_camera_reference = fake_capture

    real_edit = CD.edit_image
    mv_calls = []

    def fake_edit(**kw):
        if str(kw.get("tag", "")).startswith("mv_"):        # 多角度 view: the projection base is the "AI" image
            mv_calls.append(kw["tag"])
            shutil.copy2(kw["edit_base_path"], kw["output_path"])
            return {"seconds": 0.1, "image_model": "e2e-fake", "output_path": kw["output_path"]}
        return real_edit(**kw)
    CD.edit_image = fake_edit
    MV.estimate_light = lambda *a, **k: {"key_rel_azimuth_deg": 40.0, "key_elevation_deg": 35.0,
                                         "light_type": "sun", "sky_words": "晴朗清晨"}

    def fake_register(fn, first_interval=0.0, persistent=False):
        TIMERS.append([CLOCK[0] + float(first_interval or 0.0), fn])
    bpy.app.timers.register = fake_register
    chain.candidate_contexts = lambda windows, scene=None: [{}]
    started, phases = [], []

    def sync_start_thread(self, context, target):
        import threading
        if not ops._ACTIVE_LOCK.acquire(blocking=False):
            self.report({"WARNING"}, "已有 AI / 登录任务正在运行。")
            return False
        ops._ACTIVE_OPERATOR = self
        self._owns_lock = True
        self._ownership_guard = threading.Lock()
        self._detached = threading.Event()
        self._origin_scene = context.scene
        self._result, self._error, self._started_at, self._timer, self._thread = None, None, time.time(), None, None
        if self.bl_idname == "WONDFUL_OT_multiview_render":
            from wondful_ai_renderer import layered_output
            REPORT["layers_pending_at_multiview_start"] = layered_output.RUNNER.pending
        started.append(self.bl_idname)
        op = self

        def drive():
            try:
                op._result = target()
            except Exception as exc:
                op._error = (exc, traceback.format_exc())
                print(op._error[1])
            phases.append(getattr(op, "_phase", ""))
            for _ in range(10000):
                op._progress_update(context.scene.wondful_ai) if op._thread else None
                res = op._modal(bpy.context, FakeEvent())
                if "RUNNING_MODAL" not in res:
                    return None
            return None
        TIMERS.append([CLOCK[0] + 0.25, drive])
        return True
    ops._BaseAsyncOperator._start_thread = sync_start_thread
    # the worker sets _phase from its thread; record every phase it reaches
    orig_setattr = ops._BaseAsyncOperator.__setattr__

    def rec_setattr(self, key, value):
        if key == "_phase":
            phases.append(str(value))
        orig_setattr(self, key, value)
    ops._BaseAsyncOperator.__setattr__ = rec_setattr

    # ---- scene: product box, 场景粗模 (fixture), 3 cameras around the product --------------------------------
    scene = bpy.context.scene
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    props = scene.wondful_ai
    product = None
    if not PURE_SCENE:
        product = bpy.data.collections.new("产品")
        scene.collection.children.link(product)
        mesh = bpy.data.meshes.new("Box")
        mesh.from_pydata([(x, y, z) for x in (-.5, .5) for y in (-.5, .5) for z in (0, 1)], [],
                         [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
        product.objects.link(bpy.data.objects.new("Box", mesh))
        props.product_collection = product
    from wondful_ai_renderer import massing as MS, scene_blockout_build as SBB
    parent, _c = MS.ensure_collection(scene, props)
    fixture = os.path.join(ROOT, "tests", "fixtures", "scene_blockout", "desert_v3.json")
    SBB.build(bpy.context, json.load(open(fixture, encoding="utf-8")), product_collection=product,
              parent_collection=parent, set_camera=False)
    REPORT["blockout"] = MV.scene_blockout_exists(scene)
    import math
    from mathutils import Vector
    rig = bpy.data.collections.new("机位")
    scene.collection.children.link(rig)
    sub = bpy.data.collections.new("机位_备用")
    rig.children.link(sub)
    cams = []
    for i, (name, ang) in enumerate((("正面", -60.0), ("侧面", 20.0), ("背面", 120.0))):
        o = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        (sub if i == 2 else rig).objects.link(o)
        a = math.radians(ang)
        o.location = (6.5 * math.cos(a), 6.5 * math.sin(a), 2.0)
        o.rotation_euler = (Vector((0, 0, 0.5)) - o.location).to_track_quat("-Z", "Y").to_euler()
        cams.append(o)
    first_scene_cam = bpy.data.objects.new("原相机", bpy.data.cameras.new("原相机"))
    scene.collection.objects.link(first_scene_cam)
    first_scene_cam.location = (0, -7, 2)
    scene.camera = first_scene_cam
    scene.render.resolution_x, scene.render.resolution_y = 320, 240
    props.codex_account_state = "LOGGED_IN"
    props.output_directory = os.path.join(OUT, "renders")
    props.render_mode = "FAST"
    props.two_step = False
    props.render_count = "1"
    props.scene_brief = "沙漠清晨"
    props.layered_output = not PURE_SCENE             # 分层输出 deliverables (PNG layers + PSD) stay visible
    props.layered_mode = "SHADOW"

    # drag in: one camera, then the collection (adds the other two, recursive, skips the repeat)
    props.mv_drop_camera = cams[0]
    props.mv_drop_collection = rig
    REPORT["list"] = [i.camera.name for i in props.mv_cameras]
    REPORT["drop_note"] = props.mv_drop_note
    props.mv_cameras[1].enabled = False                  # 3 cameras, 2 checked: 正面 (主视角) + 背面
    REPORT["checked"] = [o.name for o in ops.generate_views(bpy.context, props)]

    t0 = time.time()
    REPORT["generate"] = sorted(bpy.ops.wondful.generate())
    REPORT["scene_camera_after_click"] = scene.camera.name
    run_timers(props, phases)
    from wondful_ai_renderer import layered_output as LO
    for _ in range(600):                    # the layer split runs on its own thread
        if LO.RUNNER.pending == 0:
            break
        time.sleep(0.5)
    run_timers(props, phases, 30)
    REPORT["layers_note"] = props.last_layers_note
    REPORT["seconds"] = round(time.time() - t0, 1)
    REPORT["operators"] = started
    REPORT["status"], REPORT["error"], REPORT["mv_note"] = props.status, props.last_error, props.last_mv_note
    REPORT["view_phases"] = [p for p in phases if "视角" in p]
    views = json.loads(props.view_results or "[]")
    REPORT["view_results"] = [(v["camera"], os.path.basename(v["path"]), os.path.isfile(v["path"])) for v in views]
    REPORT["mv_calls"] = mv_calls
    REPORT["plan_cleared"] = props.views_plan == ""
    flips = []
    for _ in range(len(views)):
        flips.append((int(props.view_result_index), os.path.basename(props.last_result_path)))
        bpy.ops.wondful.view_result_step(step=1)
    REPORT["flips"] = flips
    out_dir = props.output_directory
    REPORT["visible_files"] = visible_files(out_dir)
    REPORT["non_images"] = [f for f in REPORT["visible_files"] if os.path.splitext(f)[1].lower() not in IMAGE_EXT]
    REPORT["internal_files"] = sum(len(f) for _d, _s, f in os.walk(os.path.join(out_dir, ".wondful")))
    masks = list(Path(out_dir).rglob("*_maskw.npy")) if PURE_SCENE else []
    REPORT["pure_mask_empty"] = bool(masks) and all(not np.load(p).any() for p in masks) if PURE_SCENE else None
    REPORT["ok"] = bool(
        REPORT["blockout"] and REPORT["list"] == ["正面", "侧面", "背面"] and REPORT["checked"] == ["正面", "背面"]
        and REPORT["scene_camera_after_click"] == "正面"
        and started == ["WONDFUL_OT_polish_prompt", "WONDFUL_OT_ai_render", "WONDFUL_OT_multiview_render"]
        and REPORT.get("layers_pending_at_multiview_start") == 0
        and props.status == "SUCCESS" and [v[0] for v in REPORT["view_results"]] == ["正面", "背面"]
        and all(v[2] for v in REPORT["view_results"]) and any("视角 2/2" in p for p in REPORT["view_phases"])
        and REPORT["plan_cleared"] and len({f[1] for f in flips}) == 2 and not REPORT["non_images"]
        and not any(f.startswith("多角度") for f in REPORT["visible_files"])
        and (REPORT["pure_mask_empty"] if PURE_SCENE else any(f.endswith("_layers.psd") for f in REPORT["visible_files"])) and REPORT["internal_files"] > 0)


try:
    main()
except Exception:
    REPORT["exception"] = traceback.format_exc()
    print(REPORT["exception"], flush=True)
with open(os.path.join(OUT, "e2e_report.json"), "w", encoding="utf-8") as fh:
    json.dump(REPORT, fh, indent=1, ensure_ascii=False)
print("[e2e]", json.dumps(REPORT, ensure_ascii=False), flush=True)
print("[e2e] OK" if REPORT["ok"] else "[e2e] FAILED", flush=True)
os._exit(0 if REPORT["ok"] else 1)
