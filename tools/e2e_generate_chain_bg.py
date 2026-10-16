"""Headless E2E of 「生成」 → prompt analysis → automatic AI render, N = 1..4 (blender -b).

Background Blender has no windows, so modal operators cannot run for real. This harness
keeps everything else real (operators, props, the 1.2.2 chain runner, the Codex direct
image path against tests/mock_codex_backend.py, the export loop) and fakes only:
  * the modal loop: _BaseAsyncOperator._start_thread runs the worker and then drives
    ``_modal(context, TIMER)`` until it returns;
  * bpy.app.timers: callbacks are queued and fired by this script (fake clock);
  * window contexts: op_chain.candidate_contexts returns one empty override;
  * the camera capture (OpenGL is unavailable headless): a solid PNG of the camera size.

    blender -b --factory-startup --python tools/e2e_generate_chain_bg.py -- <addon.zip> <out_dir> [1,2,3,4]
"""
import json
import os
import struct
import sys
import tempfile
import time
import traceback
import zlib

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ZIP, OUT = argv[0], argv[1]
COUNTS = [int(c) for c in (argv[2] if len(argv) > 2 else "1,2,3,4").split(",")]
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
REPORT = {"runs": [], "ok": False}


def png(path, w, h, rgb=(150, 150, 150)):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def ch(k, d):
        return struct.pack(">I", len(d)) + k + d + struct.pack(">I", zlib.crc32(k + d) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + ch(b"IDAT", zlib.compress(raw)) + ch(b"IEND", b""))


TIMERS = []          # [due, fn]
CLOCK = [0.0]


class FakeEvent:
    type = "TIMER"


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

    bpy.ops.preferences.addon_install(filepath=ZIP, overwrite=True)
    bpy.ops.preferences.addon_enable(module="wondful_ai_renderer")
    import wondful_ai_renderer.operators as ops
    import wondful_ai_renderer.op_chain as chain
    prefs = bpy.context.preferences.addons["wondful_ai_renderer"].preferences
    prefs.codex_direct_simple = False      # no clay render (needs a GPU / long Cycles)
    prefs.structure_guides_enabled = False
    prefs.codex_direct_edit = True

    # ---- fakes -------------------------------------------------------------------------
    ops.discover_codex_cli = lambda *a, **k: "/usr/bin/true"
    polish_calls = [0]

    def fake_polish(**kwargs):
        polish_calls[0] += 1
        return "雪地清晨，冷色调的产品棚拍，柔和主光从左侧照来。"

    def no_exec(**kwargs):
        raise RuntimeError("codex exec fallback must not be used in this e2e")
    ops.codex_polish_prompt = fake_polish
    ops.codex_generate_image = no_exec

    def fake_capture(context, filepath):
        w, h = ops.camera_output_dimensions(context)
        png(filepath, w, h)
        return w, h, "E2E_FAKE_CAPTURE"
    ops.capture_camera_reference = fake_capture

    def fake_register(fn, first_interval=0.0, persistent=False):
        TIMERS.append([CLOCK[0] + float(first_interval or 0.0), fn])
    bpy.app.timers.register = fake_register
    chain.candidate_contexts = lambda windows, scene=None: [{}]

    started_ops = []

    def sync_start_thread(self, context, target):
        if not ops._ACTIVE_LOCK.acquire(blocking=False):
            self.report({"WARNING"}, "已有 AI / 登录任务正在运行。")
            return False
        import threading
        ops._ACTIVE_OPERATOR = self
        self._owns_lock = True
        self._ownership_guard = threading.Lock()
        self._detached = threading.Event()
        self._origin_scene = context.scene
        self._result, self._error, self._started_at, self._timer = None, None, time.time(), None
        self._thread = None
        started_ops.append(self.bl_idname)
        op = self

        def drive():
            try:
                op._result = target()
            except Exception as exc:
                op._error = (exc, traceback.format_exc())
            for _ in range(10000):
                res = op._modal(bpy.context, FakeEvent())
                if "RUNNING_MODAL" not in res:
                    return None
            return None
        # Run "modal" asynchronously in the fake timer loop, like Blender would.
        TIMERS.append([CLOCK[0] + 0.25, drive])
        return True
    ops._BaseAsyncOperator._start_thread = sync_start_thread

    # ---- scene -------------------------------------------------------------------------
    scene = bpy.context.scene
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    coll = bpy.data.collections.new("Product")
    scene.collection.children.link(coll)
    mesh = bpy.data.meshes.new("Box")
    mesh.from_pydata([(x, y, z) for x in (-.5, .5) for y in (-.5, .5) for z in (0, 1)], [],
                     [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    coll.objects.link(bpy.data.objects.new("Box", mesh))
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(cam)
    cam.location, cam.rotation_euler = (3.2, -3.2, 2.2), (1.15, 0.0, 0.785)
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = 640, 480
    props = scene.wondful_ai
    props.product_collection = coll
    props.codex_account_state = "LOGGED_IN"
    props.output_directory = os.path.join(OUT, "renders")
    props.render_mode = "FAST"
    props.two_step = False

    for i, n in enumerate(COUNTS):
        props.render_count = str(n)
        props.scene_brief = f"雪地清晨，冷色调 #{i}"     # new brief → analysis runs again
        props.last_export_paths = "[]"
        before = list(started_ops)
        calls_before = polish_calls[0]
        result = bpy.ops.wondful.generate()           # same operator the 生成 button calls
        t0 = time.time()
        statuses = []
        while TIMERS and time.time() - t0 < 180:
            TIMERS.sort(key=lambda t: t[0])
            due, fn = TIMERS.pop(0)
            CLOCK[0] = max(CLOCK[0], due)
            again = fn()
            statuses.append(props.status)
            if isinstance(again, (int, float)):
                TIMERS.append([CLOCK[0] + float(again), fn])
        chain_ops = started_ops[len(before):]
        exported = json.loads(props.last_export_paths or "[]")
        run = {"n": n, "generate": sorted(result), "operators": chain_ops,
               "polish_calls": polish_calls[0] - calls_before,
               "status": props.status, "error": props.last_error,
               "variant_count": int(props.variant_count), "exported": len(exported),
               "engine_note": props.last_engine_note,
               "files_exist": all(os.path.isfile(p) for p in exported)}
        run["ok"] = (chain_ops == ["WONDFUL_OT_polish_prompt", "WONDFUL_OT_ai_render"] and props.status == "SUCCESS"
                     and run["exported"] == n and run["variant_count"] == n and run["files_exist"]
                     and "直连" in run["engine_note"])
        REPORT["runs"].append(run)
        print("[e2e]", json.dumps(run, ensure_ascii=False), flush=True)

    # Failure path: the chained render refuses (output dir unusable) → visible error, not silence.
    props.scene_brief = "失败路径"
    props.output_directory = "/proc/definitely/not/writable"
    props.render_count = "1"
    bpy.ops.wondful.generate()
    t0 = time.time()
    while TIMERS and time.time() - t0 < 60:
        TIMERS.sort(key=lambda t: t[0])
        due, fn = TIMERS.pop(0)
        CLOCK[0] = max(CLOCK[0], due)
        again = fn()
        if isinstance(again, (int, float)):
            TIMERS.append([CLOCK[0] + float(again), fn])
    fail = {"status": props.status, "error": props.last_error,
            "ok": props.status == "ERROR" and "没能自动开始" in props.last_error and "输出目录" in props.last_error}
    REPORT["failure_path"] = fail
    print("[e2e] failure path", json.dumps(fail, ensure_ascii=False), flush=True)
    REPORT["ok"] = all(r["ok"] for r in REPORT["runs"]) and fail["ok"]


try:
    main()
except Exception:
    REPORT["exception"] = traceback.format_exc()
    print(REPORT["exception"], flush=True)
with open(os.path.join(OUT, "e2e_report.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2, ensure_ascii=False)
print("[e2e] OK" if REPORT["ok"] else "[e2e] FAILED", flush=True)
os._exit(0 if REPORT["ok"] else 1)
