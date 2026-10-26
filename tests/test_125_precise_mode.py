"""1.2.5 ④ 场景粗模 ·「精确模式」— local depth measurement (precise_depth) + first-use install (precise_setup).

Pure parts: camera override, plane → pitch/roll in the builder's Euler convention, scipy-free focal/shift recovery,
distance + width/height snap on a synthetic depth map, 3D distance error, depth prompt block, install with mocked
mirrors (fallback, sha256 mismatch, cancel, unsupported platform, bad wheel), stage lists (off = unchanged).
Blender part (skipped without ``blender``): tools/e2e_precise_bg.py runs the real operator off / on (recorded
measurement) / on with a failing install (fallback to normal mode) / on + 自动校正一轮."""
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402
import precise_synth  # noqa: E402

PD = load_package_module("precise_depth")
PS = load_package_module("precise_setup")
SP = load_package_module("sb_progress")
SB = load_package_module("scene_blockout")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BLENDER = shutil.which("blender")
CAM = {"height_m": 1.6, "pitch_deg": 0.0, "roll_deg": 0.0, "hfov_deg": 60.0}


def blender_cam_axes(pitch, roll):
    """Camera right / forward in world for the builder's Euler((90°+pitch, roll, 0), 'XYZ')."""
    ax, ay = math.radians(90 + pitch), math.radians(roll)
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    M = Ry @ Rx
    return M @ [1, 0, 0], M @ [0, 0, -1], M @ [0, 1, 0]


class CameraTest(unittest.TestCase):
    def test_apply_camera_overrides_llm(self):
        data = {"camera": {"height_m": 1.7, "pitch_deg": -2.3, "roll_deg": 0, "hfov_deg": 84, "horizon_y_frac": 0.52}}
        cam = {"height_m": 1.68, "pitch_deg": -2.03, "roll_deg": -0.81, "hfov_deg": 98.3}
        old = PD.apply_camera(data, cam, (1600, 900))
        self.assertEqual(old["hfov_deg"], 84)
        for k in cam:
            self.assertAlmostEqual(data["camera"][k], cam[k])
        self.assertAlmostEqual(data["camera"]["horizon_y_frac"], SB.horizon_from_pitch(-2.03, 98.3, 900 / 1600), 3)

    def test_plane_to_pitch_roll_matches_builder_euler(self):
        for pitch, roll in ((0, 0), (-5, 0), (8, 0), (-3, 2), (-10, -4)):
            right, fwd, up = blender_cam_axes(pitch, roll)
            # world up expressed in the Z-up camera frame (x = right, y = forward, z = camera up)
            n = np.array([right[2], fwd[2], up[2]])
            _R, h, p, r = PD.camera_from_plane(n, 1.6)
            self.assertAlmostEqual(h, 1.6)
            self.assertAlmostEqual(p, pitch, delta=0.3, msg=(pitch, roll))
            self.assertAlmostEqual(r, roll, delta=0.3, msg=(pitch, roll))

    def test_focal_shift_recovers_pinhole(self):
        W, H = 112, 70
        uv = PD.uv_grid(W, H)
        f_true, shift = 1.3, 0.7
        z = 2.0 + np.random.default_rng(0).random((H, W)) * 8
        pts = np.stack([uv[..., 0] * z / f_true, uv[..., 1] * z / f_true, z - shift], -1).astype(np.float32)
        f, s = PD.focal_shift(pts, np.ones((H, W), bool))
        self.assertAlmostEqual(f, f_true, delta=0.01)
        self.assertAlmostEqual(s, shift, delta=0.02)

    def test_infer_size_multiple_of_14(self):
        for fw, fh in ((1600, 900), (900, 1600), (1000, 1000), (2400, 1000)):
            W, H = PD.infer_size(fw, fh)
            self.assertEqual((W % 14, H % 14), (0, 0))
            self.assertEqual(max(W, H), PD.INFER_LONG)

    def test_resize_bilinear(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4, 1).repeat(3, -1)
        b = PD.resize_bilinear(a, 8, 6)
        self.assertEqual(b.shape, (6, 8, 3))
        self.assertEqual(b.dtype, np.float32)
        self.assertAlmostEqual(float(b.mean()), float(a.mean()), delta=0.3)


def wall_geo():
    # a 10 m wide, 8 m tall facade 20 m ahead (+ a 6 m tall box to the right at 35 m)
    return precise_synth.geo(CAM, boxes=[(-5, 5, 20, 24, 0, 8, PD.STRUCT), (8, 14, 35, 40, 0, 6, PD.STRUCT)])


class SnapTest(unittest.TestCase):
    def data(self, **kw):
        prim = {"id": "b", "type": "building", "shape": "box", "label_zh": "白楼", "x": 0.0, "y": 44.0, "z_offset": 0.0,
                "w": 22.0, "d": 8.8, "h": 17.6, "roof_h": 0.0, "rot_z_deg": 0.0, "repeat": 1, "repeat_dx": 0.0,
                "repeat_dy": 0.0, "material": ""}
        prim.update(kw)
        return {"camera": dict(CAM, horizon_y_frac=0.5), "primitives": [prim], "foliage_masses": []}

    def test_distance_snap_scales_about_camera(self):
        g = wall_geo()
        d = self.data()
        rep = PD.snap(d, g, g["frame"], size=False)
        p = d["primitives"][0]
        self.assertEqual(rep["snapped"], 1)
        self.assertAlmostEqual(p["y"], 22.0, delta=2.0)        # front at 20, LLM centre 44 → ~half
        self.assertAlmostEqual(p["w"] / p["y"], 22.0 / 44.0, 3)  # screen size kept

    def test_size_snap_height_and_width(self):
        g = wall_geo()
        d = self.data(w=16.0, h=10.0, y=40.0)       # too far, too narrow/short once moved
        rep = PD.snap(d, g, g["frame"], size=True)
        p = d["primitives"][0]
        self.assertAlmostEqual(p["h"], 8.0, delta=1.0)
        self.assertAlmostEqual(p["w"], 10.0, delta=1.5)
        self.assertIn("h_ratio", rep["items"][0])

    def test_size_snap_clamped(self):
        g = wall_geo()
        d = self.data(w=2.0, h=1.0, y=40.0, x=0.0)
        PD.snap(d, g, g["frame"], size=True)
        p = d["primitives"][0]
        self.assertLessEqual(p["h"], 1.0 * 0.55 * 1.5 + 1e-6)   # ratio 0.4..2.5 then ±50 % at most

    def test_thin_and_far_not_snapped(self):
        g = wall_geo()
        d = self.data(type="pole", shape="cylinder", w=0.3, d=0.3, h=8)
        self.assertEqual(PD.snap(d, g, g["frame"])["snapped"], 0)
        d = self.data(y=300.0, w=150, h=120)
        before = dict(d["primitives"][0])
        PD.snap(d, g, g["frame"])
        self.assertEqual(d["primitives"][0]["y"], before["y"])

    def test_range_error_and_precise_pass(self):
        g = wall_geo()
        d = self.data()
        err, n, worst = PD.range_error(d, g, g["frame"])
        self.assertEqual(n, 1)
        self.assertGreater(err, 50)
        nd, info = PD.precise_pass(d, g, g["frame"])
        self.assertLess(info["err_after_pct"], 15)
        self.assertGreater(info["err_before_pct"], 50)
        self.assertEqual(d["primitives"][0]["y"], 44.0)          # input untouched (deep copy)
        self.assertEqual(nd["camera"]["hfov_deg"], CAM["hfov_deg"])
        self.assertIn("_precise", nd)

    def test_depth_block(self):
        g = wall_geo()
        txt = PD.depth_block(g, g["frame"])
        self.assertIn("USE THESE CAMERA VALUES EXACTLY", txt)
        self.assertIn("hfov_deg 60.0", txt)
        self.assertIn("structure x", txt)
        self.assertEqual(len([l for l in txt.splitlines() if l.count("|") == 7]), 6)

    def test_save_load_roundtrip(self):
        g = wall_geo()
        g["cam"]["infer_s"] = 1.0
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "geo.npz")
            PD.save_geo(p, g)
            h = PD.load_geo(p)
        self.assertEqual(h["cam"]["hfov_deg"], 60.0)
        self.assertEqual(h["frame"], g["frame"])
        self.assertTrue((h["cls"] == g["cls"]).all())


def fake_opener(table):
    """url → bytes | Exception."""
    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    calls = []

    def op(url, timeout):
        calls.append(url)
        v = table.get(url, urllib.error.URLError("unreachable"))
        if isinstance(v, Exception):
            raise v
        return Resp(v)
    op.calls = calls
    return op


def wheel_bytes(ok=True):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if ok:
            zf.writestr("onnxruntime/__init__.py", "__version__ = 'fake'\n")
        zf.writestr("onnxruntime-1.20.1.dist-info/METADATA", "Name: onnxruntime\n")
    return buf.getvalue()


class InstallTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.saved = (PS.WHEELS, PS.MODEL_SHA256, PS.MODEL_SIZE, PS.MODEL_URLS, PS._ort_already_importable)
        self.model = b"onnx-model-bytes" * 1000
        self.whl = wheel_bytes()
        PS.WHEELS = {"cp311-win_amd64": ("/packages/aa/onnxruntime-1.20.1-cp311-cp311-win_amd64.whl",
                                         hashlib.sha256(self.whl).hexdigest(), len(self.whl))}
        PS.MODEL_SHA256 = hashlib.sha256(self.model).hexdigest()
        PS.MODEL_SIZE = len(self.model)
        PS.MODEL_URLS = ("https://hf.example/model.onnx", "https://mirror.example/model.onnx")
        PS._ort_already_importable = lambda: False

    def tearDown(self):
        PS.WHEELS, PS.MODEL_SHA256, PS.MODEL_SIZE, PS.MODEL_URLS, PS._ort_already_importable = self.saved
        shutil.rmtree(self.td, ignore_errors=True)

    def urls(self):
        return PS.wheel_urls("cp311-win_amd64")

    def test_platform_key_and_dirs(self):
        PS.WHEELS = self.saved[0]
        self.assertEqual(PS.platform_key("Windows", "AMD64", (3, 11)), "cp311-win_amd64")
        self.assertEqual(PS.platform_key("Darwin", "arm64", (3, 11)), "cp311-macos")
        self.assertEqual(PS.platform_key("Linux", "x86_64", (3, 13)), "cp313-linux_x86_64")
        self.assertIsNone(PS.platform_key("Windows", "ARM64", (3, 11)))
        self.assertIsNone(PS.platform_key("Windows", "AMD64", (3, 9)))
        d = PS.data_dir(env={}, home="C:/Users/张三")
        self.assertEqual(d.parts[-2:], (".wondful_ai_renderer", "precise"))
        self.assertEqual(PS.data_dir(env={"WONDFUL_PRECISE_DIR": self.td}), Path(self.td))
        for k, (path, sha, size) in PS.WHEELS.items():
            self.assertTrue(path.startswith("/packages/") and path.endswith(".whl") and PS.ORT_VERSION in path, k)
            self.assertEqual(len(sha), 64)
            self.assertGreater(size, 5_000_000)

    def test_fresh_install_with_mirror_fallback(self):
        u = self.urls()
        table = {u[0]: urllib.error.URLError("timed out"), u[1]: self.whl,
                 PS.MODEL_URLS[0]: b"x" * len(self.model),               # wrong bytes → next mirror
                 PS.MODEL_URLS[1]: self.model}
        op = fake_opener(table)
        seen = []
        st = PS.ensure_installed(root=self.td, key="cp311-win_amd64", opener=op, check_import=False,
                                 progress=lambda f, t: seen.append(f))
        self.assertTrue(st["model"])
        self.assertEqual(st["runtime"], "local")
        self.assertTrue(st["ready"])
        self.assertEqual(op.calls, [u[0], u[1], PS.MODEL_URLS[0], PS.MODEL_URLS[1]])
        self.assertTrue((PS.site_dir("cp311-win_amd64", self.td) / "onnxruntime" / "__init__.py").is_file())
        self.assertFalse(any(Path(self.td).rglob("*.part")))
        self.assertFalse(any((Path(self.td) / "downloads").glob("*.whl")))
        self.assertEqual(seen, sorted(seen))
        self.assertAlmostEqual(seen[-1], 1.0)
        # second run: nothing to download
        op2 = fake_opener({})
        PS.ensure_installed(root=self.td, key="cp311-win_amd64", opener=op2, check_import=False)
        self.assertEqual(op2.calls, [])

    def test_all_mirrors_fail(self):
        with self.assertRaises(PS.PreciseInstallError) as cm:
            PS.ensure_installed(root=self.td, key="cp311-win_amd64", opener=fake_opener({}), check_import=False)
        self.assertIn("无法下载", str(cm.exception))
        self.assertEqual(len(cm.exception.details), 3)
        self.assertFalse(PS.status(self.td, "cp311-win_amd64")["ready"])

    def test_checksum_failure_everywhere(self):
        bad = {u: b"0" * len(self.whl) for u in self.urls()}
        with self.assertRaises(PS.PreciseInstallError) as cm:
            PS.ensure_installed(root=self.td, key="cp311-win_amd64", opener=fake_opener(bad), check_import=False)
        self.assertIn("校验失败", str(cm.exception))
        self.assertFalse(any(Path(self.td).rglob("*.part")))

    def test_cancel(self):
        table = {self.urls()[0]: self.whl}
        with self.assertRaises(PS.PreciseInstallError) as cm:
            PS.ensure_installed(root=self.td, key="cp311-win_amd64", opener=fake_opener(table), check_import=False,
                                cancel=lambda: True)
        self.assertTrue(cm.exception.cancelled)
        self.assertFalse(PS.ort_installed("cp311-win_amd64", self.td))

    def test_unsupported_platform(self):
        with self.assertRaises(PS.PreciseInstallError) as cm:
            PS.ensure_installed(root=self.td, key=None, opener=fake_opener({}), check_import=False)
        self.assertIn("不支持", str(cm.exception))

    def test_bad_wheel(self):
        whl = wheel_bytes(ok=False)
        PS.WHEELS = {"cp311-win_amd64": (PS.WHEELS["cp311-win_amd64"][0], hashlib.sha256(whl).hexdigest(), len(whl))}
        with self.assertRaises(PS.PreciseInstallError):
            PS.ensure_installed(root=self.td, key="cp311-win_amd64", opener=fake_opener({self.urls()[0]: whl}),
                                check_import=False)
        self.assertFalse(PS.ort_installed("cp311-win_amd64", self.td))

    def test_import_local_site_dir(self):
        u = self.urls()
        PS.ensure_installed(root=self.td, key="cp311-win_amd64", opener=fake_opener({u[0]: self.whl,
                                                                                      PS.MODEL_URLS[0]: self.model}),
                            check_import=False)
        real = sys.modules.pop("onnxruntime", None)
        saved_path = list(sys.path)
        try:
            import builtins
            orig = builtins.__import__
            blocked = {"n": 0}

            def imp(name, *a, **kw):        # pretend no system onnxruntime the first time
                if name == "onnxruntime" and blocked["n"] == 0:
                    blocked["n"] = 1
                    raise ImportError("no system onnxruntime")
                return orig(name, *a, **kw)
            builtins.__import__ = imp
            try:
                ort = PS.import_onnxruntime(self.td, "cp311-win_amd64")
            finally:
                builtins.__import__ = orig
            self.assertEqual(ort.__version__, "fake")
        finally:
            sys.modules.pop("onnxruntime", None)
            if real is not None:
                sys.modules["onnxruntime"] = real
            sys.path[:] = saved_path


class StagesTest(unittest.TestCase):
    def test_off_unchanged(self):
        self.assertEqual(SP.stages(False, True), ["READ", "ANALYZE", "BUILD", "CAMERA", "DONE"])
        self.assertEqual(SP.stages(True, True, precise=False, install=True),
                         ["READ", "ANALYZE", "BUILD", "CAMERA", "PREVIEW", "SCORE", "CORRECT", "REBUILD", "DONE"])

    def test_precise(self):
        self.assertEqual(SP.stages(False, True, precise=True),
                         ["READ", "DEPTH", "ANALYZE", "BUILD", "CAMERA", "DONE"])
        ids = SP.stages(True, True, precise=True, install=True)
        self.assertEqual(ids[:3], ["READ", "INSTALL", "DEPTH"])
        self.assertIn("CORRECT", ids)
        a = SP.fraction(ids, "INSTALL", 0, inner=0.1)
        b = SP.fraction(ids, "INSTALL", 0, inner=0.8)
        c = SP.fraction(ids, "DEPTH", 0)
        self.assertLess(a, b)
        self.assertLessEqual(b, c)
        self.assertEqual(SP.label("DEPTH"), "测量深度")

    def test_summary(self):
        info = {"blocks": 12, "camera": True, "scores": [0.8], "precise": {"camera": {"hfov_deg": 97.8},
                                                                           "err_after_pct": 3.1, "err_before_pct": 18.0}}
        s = SB.summary(info)
        self.assertIn("精确模式（实测视角 98°） · 3D 距离误差 3.1%（对齐前 18.0%）", s)
        s = SB.summary({"blocks": 3, "precise_fallback": "无法下载（已尝试 3 个地址）"})
        self.assertIn("精确模式不可用，已用普通模式", s)
        self.assertNotIn("精确模式", SB.summary({"blocks": 3, "scores": [0.8]}))


@unittest.skipUnless(BLENDER, "blender not installed")
class BlenderPreciseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        out = Path(cls.tmp) / "precise.json"
        subprocess.run([BLENDER, "-b", "--factory-startup", "--python", str(ROOT / "tools" / "e2e_precise_bg.py"), "--",
                        "repo", str(HERE / "fixtures" / "scene_blockout" / "desert_v3.json"), str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        cls.rep = json.loads(out.read_text(encoding="utf-8"))

    def _stages(self, run):
        seq = []
        for status, stage, *_ in self.rep[run]["trace"]:
            if status == "BLOCKOUT" and (not seq or seq[-1] != stage):
                seq.append(stage)
        return seq

    def test_ran(self):
        self.assertTrue(self.rep.get("ok"), self.rep.get("error"))

    def test_off_is_unchanged(self):
        off, base = self.rep["off"], self.rep["baseline"]
        self.assertEqual(self._stages("off"), ["ANALYZE", "BUILD", "CAMERA"])
        self.assertEqual(off["camera_values"], base["camera_values"])
        self.assertEqual(off["objects"], base["objects"])
        self.assertFalse(off["measured_in_prompt"])
        self.assertNotIn("精确模式", off["note"])

    def test_on_recorded(self):
        r = self.rep["on"]
        self.assertEqual(self._stages("on"), ["DEPTH", "ANALYZE", "BUILD", "CAMERA"])
        self.assertTrue(r["measured_in_prompt"])
        self.assertAlmostEqual(r["camera_values"]["hfov_deg"], 60.0, delta=0.2)
        self.assertAlmostEqual(r["camera_values"]["height_m"], 1.6, delta=0.01)
        self.assertIn("精确模式", r["note"])
        self.assertEqual(r["final_stage"], "DONE")

    def test_install_failure_falls_back(self):
        r = self.rep["fallback"]
        self.assertEqual(self._stages("fallback")[:2], ["INSTALL", "ANALYZE"])
        self.assertEqual((r["status"], r["final_stage"]), ("IDLE", "DONE"))
        self.assertIn("精确模式不可用，已用普通模式", r["note"])
        self.assertFalse(r["measured_in_prompt"])
        self.assertEqual(r["calls"], 1)                # one normal analysis, nothing wasted on the failed install
        self.assertEqual(r["camera_values"], self.rep["baseline"]["camera_values"])

    def test_correction_round_runs(self):
        r = self.rep["correct"]
        self.assertEqual(r["calls"], 2)
        self.assertTrue(r["precise_text_in_correction"])
        self.assertIn("CORRECT", self._stages("correct"))
        self.assertEqual(r["final_stage"], "DONE")


if __name__ == "__main__":
    unittest.main()
