"""1.2.7 fixes from the Mac test (2026-10-21, 测试包 df947aa).

1. 参考图超限 → 拼图: over the 5-input limit the camera base stays standalone, the rest is merged into labelled
   sheets (letters A/B/C, each cell keeps its aspect), detail → structure become text only last; environment
   images never go to the image model. Covered for 5 / 6 / 7 / 9 inputs, the codex exec path, the direct path,
   Antigravity (3) and the panel note 「已拼合 N 张参考图」.
2. is_dynamic_prop: size-only car detection only for real car proportions; static analysis elements never removed.
3. 216° 连续性: building rows continue along the measured row line; 地面锁定 in every view prompt.
4. SSL EOF / connection reset: 2 back-off retries before the Codex proxy.
5. codex exec: one call over 10 min is stopped and retried once inside the configurable total (default 1800 s).
6. ⑤ heartbeat: 「已等 X 秒」, the bar never parks at 88 %.
7. 纯场景多角度: no product collection is allowed (no product gate).
"""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402
from test_321_direct_edit import DirectEditTestBase, direct, rgb_png  # noqa: E402

RC = load_package_module("reference_collage")
png_io = load_package_module("png_io")
SS = load_package_module("static_scene")
SC = load_package_module("scene_complete")
DB = load_package_module("detail_blockout")
MV = load_package_module("multiview")
CC = load_package_module("codex_client")
PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"
GOBI = Path("/workspace/work/scene_blockout_impl/pcfix/gobi_analysis.json")


def png(path, w, h, colour=(120, 120, 120)):
    a = np.zeros((h, w, 3), np.uint8)
    a[:] = colour
    a[: h // 3, : w // 3] = (255, 255, 255)     # an orientation mark (top-left)
    png_io.write_png(str(path), a)
    return str(path)


class CollageFit(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="wondful132_"))
        self.cam = png(self.tmp / "camera_base.png", 1600, 900)
        self.pool = [
            (png(self.tmp / "product_threeview.png", 1200, 400, (10, 120, 60)), "product"),
            (png(self.tmp / "structure_depth.png", 1600, 900, (90, 0, 0)), "structure"),
            (png(self.tmp / "structure_normal.png", 1600, 900, (0, 90, 0)), "structure"),
            (png(self.tmp / "building_ref_1_1.png", 800, 1000, (150, 120, 80)), "building"),
            (png(self.tmp / "building_ref_2_2.png", 900, 600, (160, 110, 70)), "building"),
            (png(self.tmp / "detail_crop_lamp.png", 300, 200, (200, 200, 0)), "detail"),
            (png(self.tmp / "structure_silhouette.png", 1600, 900, (0, 0, 90)), "structure"),
            (png(self.tmp / "structure_part_id.png", 1600, 900, (60, 60, 0)), "structure"),
        ]

    def fit(self, n_total, **kw):
        items = self.pool[: n_total - 1]
        roles = {p: r for p, r in items}
        paths = [self.cam] + [p for p, _r in items]
        return RC.fit_inputs(paths, 5, out_dir=str(self.tmp), roles=roles, **kw)

    def test_5_inputs_unchanged(self):
        paths, note, info = self.fit(5)
        self.assertEqual(len(paths), 5)
        self.assertEqual(note, "")
        self.assertEqual(info["merged"], 0)
        self.assertEqual(RC.panel_text(info), "")

    def test_6_inputs_one_sheet_camera_standalone(self):
        paths, note, info = self.fit(6)
        self.assertEqual(len(paths), 5)
        self.assertEqual(paths[0], str(Path(self.cam).resolve()))        # camera base first, untouched
        self.assertEqual(info["merged"], 2)
        self.assertEqual(len(info["sheets"]), 1)
        self.assertIn("已拼合 2 张参考图", RC.panel_text(info))
        self.assertIn("A = ", note)
        self.assertIn("B = ", note)
        self.assertIn("不要在结果里画出拼版", note)
        self.assertNotIn(str(Path(self.cam).resolve()), [c["file"] for s in info["sheets"] for c in s["cells"]])

    def test_7_inputs_like_the_mac(self):
        paths, note, info = self.fit(7)
        self.assertLessEqual(len(paths), 5)
        self.assertEqual(info["before"], 7)
        self.assertEqual(info["merged"], 3)
        self.assertEqual(info["text_only"], [])
        # the product three-view stays a standalone image (highest priority)
        self.assertIn("product_threeview.png", info["standalone"])

    def test_9_inputs_sheets_and_priorities(self):
        paths, note, info = self.fit(9)
        self.assertLessEqual(len(paths), 5)
        self.assertEqual(info["before"], 9)
        self.assertEqual(info["merged"] + len(info["standalone"]), 8)
        merged_roles = [c["role"] for s in info["sheets"] for c in s["cells"]]
        self.assertIn("detail", merged_roles)                # merged, not dropped
        for sh in info["sheets"]:
            self.assertTrue(Path(sh["path"]).is_file())
            self.assertLessEqual(max(sh["size"]), 2048)

    def test_sheet_long_edge_follows_request_size(self):
        paths, note, info = self.fit(9, long_edge=RC.size_long_edge("1536x1024"))
        for sh in info["sheets"]:
            self.assertLessEqual(max(sh["size"]), 1536)

    def test_cells_keep_aspect_no_crop(self):
        _p, _n, info = self.fit(9)
        sizes = [(1200, 400), (900, 600), (300, 200), (800, 1000)]
        for sh in info["sheets"]:
            img = png_io.read_png(sh["path"])[..., :3]
            w_, h_, cells = RC.layout([tuple(c["size"]) for c in sh["cells"]], 2048)
            self.assertEqual((w_, h_), tuple(sh["size"]))
            for c, (x, y, w, h) in zip(sh["cells"], cells):
                iw, ih = c["size"]
                self.assertAlmostEqual(w / h, iw / ih, delta=0.03 * iw / ih + 2.0 / h)
                self.assertGreater(int(img[y + h // 2, x + w // 2].sum()), 3 * 28)   # image, not background
        self.assertTrue(sizes)

    def test_letters_drawn_in_label_strip_above_cell(self):
        _p, note, info = self.fit(7)
        sh = info["sheets"][0]
        img = png_io.read_png(sh["path"])[..., :3]
        _w, _h, cells, tabs = RC._layout([tuple(c["size"]) for c in sh["cells"]], 2048)
        sc = RC.label_scale(2048)
        tab_h = 7 * sc + 2 * max(2, sc)
        for (x, y, w, h), (tx, ty) in zip(cells, tabs):
            self.assertLessEqual(ty + tab_h, y)                      # the tab ends above the image
            tab = img[ty:ty + tab_h, tx:tx + 5 * sc + 2 * max(2, sc)]
            self.assertTrue((tab.sum(-1) == 0).any() and (tab.sum(-1) == 765).any())   # black tab + white strokes
        self.assertIn("上方标签条", note)
        self.assertNotIn("左上角", note)

    def test_nested_sheet_numbers_not_covered(self):
        """A structure sheet that already carries its own R1/R2 tabs in its top-left keeps them visible."""
        nested = np.full((450, 800, 3), 200, np.uint8)
        RC.draw_letter(nested, 0, 0, "R", 6)                      # the sheet's own top-left number
        other = np.full((600, 900, 3), 90, np.uint8)
        sheet = RC.compose([nested, other], ["A", "B"], 2048)
        _w, _h, cells, _tabs = RC._layout([(800, 450), (900, 600)], 2048)
        for im, (x, y, w, h) in zip([nested, other], cells):
            self.assertTrue(np.array_equal(sheet[y:y + h, x:x + w], RC.resize(im, w, h)))  # image pixels untouched
        x, y, w, h = cells[0]
        corner = sheet[y:y + h // 6, x:x + w // 8]
        self.assertTrue((corner.sum(-1) == 765).any())               # the nested R strokes are still there

    def test_label_strip_keeps_long_edge(self):
        for n in (2, 3, 5, 6):
            sizes = [(1600, 900)] * n
            w_, h_, cells = RC.layout(sizes, 1536)
            self.assertLessEqual(max(w_, h_), 1536)
            sheet = RC.compose([np.full((900, 1600, 3), 120, np.uint8)] * n, list("ABCDEF")[:n], 1536)
            self.assertEqual(sheet.shape[:2], (h_, w_))

    def test_environment_never_merged(self):
        env = png(self.tmp / "env_photo.png", 1200, 634, (200, 150, 90))
        paths = [self.cam] + [p for p, _r in self.pool[:5]] + [env]
        roles = {p: r for p, r in self.pool}
        roles[env] = "environment"
        out, note, info = RC.fit_inputs(paths, 5, out_dir=str(self.tmp), roles=roles)
        self.assertNotIn(str(Path(env).resolve()), out)
        self.assertNotIn("env_photo.png", [c["file"] for s in info["sheets"] for c in s["cells"]])
        self.assertIn("环境参考图不作为图像发送", note)

    def test_text_fallback_detail_then_structure(self):
        many = [(png(self.tmp / f"structure_x{i}.png", 64, 36), "structure") for i in range(10)]
        many += [(png(self.tmp / f"detail_crop_{i}.png", 64, 36), "detail") for i in range(10)]
        paths = [self.cam, self.pool[0][0]] + [p for p, _r in many]
        roles = dict(many + [self.pool[0]])
        out, note, info = RC.fit_inputs(paths, 3, out_dir=str(self.tmp), roles=roles)
        self.assertLessEqual(len(out), 3)
        text_roles = [t["role"] for t in info["text_only"]]
        self.assertEqual(set(text_roles[:10]), {"detail"})                 # detail first …
        self.assertIn("细节图未作为图像发送", note)
        self.assertNotIn("product", text_roles)

    def test_fixed_edit_base_and_mask_stay(self):
        cand = png(self.tmp / "output_candidate_v01_a01.png", 1600, 900)
        mask = png(self.tmp / "repair_mask.png", 1600, 900)
        paths = [self.cam] + [p for p, _r in self.pool[:4]] + [cand, mask]
        out, _n, info = RC.fit_inputs(paths, 5, out_dir=str(self.tmp), roles=dict(self.pool),
                                      fixed=[self.cam, cand, mask])
        for p in (self.cam, cand, mask):
            self.assertIn(str(Path(p).resolve()), out)
        self.assertLessEqual(len(out), 5)

    def test_jpeg_inputs(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        jp = self.tmp / "product.jpg"
        Image.new("RGB", (640, 320), (30, 60, 200)).save(jp, quality=90)
        paths = [self.cam, str(jp)] + [p for p, _r in self.pool[1:6]]
        out, _n, info = RC.fit_inputs(paths, 5, out_dir=str(self.tmp), roles=dict(self.pool))
        self.assertLessEqual(len(out), 5)
        self.assertFalse(info.get("unreadable"))


class CollageReadWithoutPillow(unittest.TestCase):
    """Mac Blender has no Pillow. Blender 4.3 ``imbuf.write`` keeps a loaded JPEG a JPEG (no ``file_type``), so
    the collage reader must not take that file for a PNG: it falls back to sips / a background Blender."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="wondful132p_"))
        self.jpg = self.tmp / "ref.jpg"
        self.jpg.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF fake jpeg")
        self.saved = list(RC.CONVERTERS)
        self.pil = sys.modules.get("PIL", "absent")
        sys.modules["PIL"] = None                     # `from PIL import Image` → ImportError

    def tearDown(self):
        RC.CONVERTERS[:] = self.saved
        if self.pil == "absent":
            sys.modules.pop("PIL", None)
        else:
            sys.modules["PIL"] = self.pil

    def test_jpeg_bytes_under_png_name_rejected_then_next_converter(self):
        calls = []

        def imbuf_like(src, dst):           # what 4.3 imbuf.write does: JPEG bytes in x.png
            calls.append("imbuf")
            Path(dst).write_bytes(Path(src).read_bytes())
            return RC._is_png(dst)

        def good(src, dst):
            calls.append("good")
            a = np.zeros((10, 20, 3), np.uint8)
            a[:5] = 255
            png_io.write_png(dst, a)
            return RC._is_png(dst)
        RC.CONVERTERS[:] = [imbuf_like, good]
        a = RC.read_rgb(self.jpg)
        self.assertEqual(calls, ["imbuf", "good"])
        self.assertEqual(a.shape, (10, 20, 3))
        self.assertEqual(int(a[0, 0, 0]), 255)            # top row stays on top
        self.assertEqual(int(a[9, 0, 0]), 0)

    def test_all_converters_fail_reports_each(self):
        RC.CONVERTERS[:] = [lambda s, d: False, lambda s, d: (_ for _ in ()).throw(OSError("boom"))]
        with self.assertRaises(RC.CollageError) as cm:
            RC.read_rgb(self.jpg)
        self.assertIn("ref.jpg", str(cm.exception))
        self.assertIn("boom", str(cm.exception))

    def test_imbuf_converter_checks_real_png(self):
        class FakeIB:
            def free(self):
                pass

        class FakeImbuf:
            @staticmethod
            def load(p):
                return FakeIB()

            @staticmethod
            def write(ib, filepath):
                Path(filepath).write_bytes(b"\xff\xd8\xff\xe0 jpeg")
        old = sys.modules.get("imbuf")
        sys.modules["imbuf"] = FakeImbuf
        try:
            self.assertFalse(RC._imbuf_to_png(str(self.jpg), str(self.tmp / "o.png")))
        finally:
            if old is None:
                sys.modules.pop("imbuf", None)
            else:
                sys.modules["imbuf"] = old

    def test_unreadable_jpeg_goes_to_text_not_crash(self):
        RC.CONVERTERS[:] = [lambda s, d: False]
        cam = png(self.tmp / "camera_base.png", 160, 90)
        others = [png(self.tmp / f"structure_{i}.png", 160, 90) for i in range(4)]
        out, _n, info = RC.fit_inputs([cam] + others + [str(self.jpg)], 5, out_dir=str(self.tmp),
                                      roles={str(self.jpg): "detail"})
        self.assertLessEqual(len(out), 5)
        self.assertTrue(info.get("unreadable"))

    def test_blender_convert_script_uses_new_png_image(self):
        self.assertIn("images.new", RC._BLENDER_CONVERT)
        self.assertIn("'PNG'", RC._BLENDER_CONVERT)
        self.assertEqual(RC.CONVERTERS[0].__name__, "_imbuf_to_png")
        self.assertIn(RC._blender_to_png, RC.CONVERTERS)


class CodexExecCollage(unittest.TestCase):
    """The PC's 7-input failure was raised by the codex exec path: it now merges instead of stopping."""

    def test_exec_merges_instead_of_raising(self):
        tmp = Path(tempfile.mkdtemp(prefix="wondful132x_"))
        cam = png(tmp / "camera_base.png", 1600, 900)
        refs = [cam] + [png(tmp / f"structure_{k}.png", 800, 450) for k in ("depth", "normal", "silhouette")]
        refs += [png(tmp / "product.png", 900, 300)]
        b1, b2 = png(tmp / "building_ref_1_1.png", 600, 700), png(tmp / "building_ref_2_2.png", 600, 700)
        seen = {}

        def fake_exec(**kw):
            seen.update(kw)
            Path(kw["cwd"], "out.png").write_bytes(Path(cam).read_bytes())
            return "WONDFUL_IMAGEGEN_OK"
        orig_exec, orig_track = CC._exec_turn, CC.ArtifactTracker
        CC._exec_turn = fake_exec
        try:
            res = CC.generate_image(prompt="p", reference_paths=refs + [b1, b2], output_path=str(tmp / "out.png"),
                                    cwd=str(tmp), edit_base_path=cam,
                                    reference_roles={str(Path(b1).resolve()): ("building", "体块 1 建筑参考")})
        finally:
            CC._exec_turn = orig_exec
            CC.ArtifactTracker = orig_track
        self.assertEqual(res["collage"]["before"], 7)
        self.assertLessEqual(len(seen["image_paths"]), 5)
        self.assertEqual(seen["image_paths"][0], str(Path(cam).resolve()))
        self.assertIn("【参考图拼图】", seen["prompt"])
        self.assertIn("A = Blender 结构控制图", seen["prompt"])
        self.assertIn(str(Path(b1).resolve()), seen["image_paths"])         # building refs outrank structure maps
        self.assertTrue((tmp / "reference_fit_out.json").is_file())

    def test_fallback_passes_bundle_roles(self):
        src = (PKG / "codex_direct.py").read_text(encoding="utf-8")
        i = src.index("def generate_with_fallback(")
        body = src[i:i + 9000]
        self.assertIn("reference_roles=roles", body)
        self.assertIn('state["collage"]', body)


class DirectCollage(DirectEditTestBase):
    def test_repair_path_merges_instead_of_dropping(self):
        extra = [rgb_png(self.tmp / f"structure_{k}.png", 800, 450, (k * 30, 0, 0)) for k in range(1, 4)]
        refs = self.refs + extra
        manifest = self.manifest + [{"path": p, "members": [{"reference": 5 + i, "role": "structure", "path": p}]}
                                    for i, p in enumerate(extra)]
        cand = rgb_png(self.tmp / "output_candidate_v01_a01.png", 1600, 900)
        r = self.edit(edit_base_path=cand, reference_paths=refs, manifest=manifest, repair=True)
        self.assertEqual(r["engine"], "codex_direct_edit")
        self.assertLessEqual(r["attached_images"], 5)
        self.assertGreater(r["collage"]["merged"], 0)
        self.assertIn("candidate", r["image_roles"])
        self.assertIn("camera", r["image_roles"])
        log = json.loads((self.tmp / "codex_direct_t1.json").read_text(encoding="utf-8"))
        self.assertIn("【参考图拼图】", log["prompt"])
        self.assertEqual(direct.engine_note({"direct_ok": 1, "collage": 3}), "生图：Codex 图像编辑（直连） · 已拼合 3 张参考图")


class AntigravityCollage(unittest.TestCase):
    def test_agy_source_fits_three(self):
        src = (PKG / "antigravity_client.py").read_text(encoding="utf-8")
        self.assertIn("_RC.fit_inputs(_uniq, _AGYMAX", src)
        self.assertLess(src.index("_RC.fit_inputs(_uniq, _AGYMAX"), src.index("check_image_capacity(staged_refs"))


class DynamicPropSize(unittest.TestCase):
    def test_real_car_proportions_only(self):
        self.assertEqual(SS.is_dynamic_prop(1.8, 1.5, allow=()), "vehicle")      # car end-on
        self.assertEqual(SS.is_dynamic_prop(4.5, 1.5, allow=()), "vehicle")      # car side-on
        self.assertEqual(SS.is_dynamic_prop(2.0, 3.0, allow=()), "")             # kiosk / pillar 2.0 x 3.0 m (PC)
        self.assertEqual(SS.is_dynamic_prop(1.5, 1.5, allow=()), "")             # narrower than any car
        self.assertEqual(SS.is_dynamic_prop(6.0, 1.6, allow=()), "")             # longer than a car
        self.assertEqual(SS.is_dynamic_prop(2.6, 2.4, allow=()), "")             # van / bus shelter height
        self.assertEqual(SS.is_dynamic_prop(4.0, 2.5, allow=()), "")

    def test_end_on_swaps_length_and_width(self):
        self.assertEqual(SS.vehicle_pose(1.9, 1.5), "end")
        self.assertEqual(SS.vehicle_pose(1.9, 1.5, D=4.6), "end")              # depth = car length is fine
        self.assertEqual(SS.vehicle_pose(1.9, 1.5, D=9.0), "")                 # 9 m deep → not a car
        self.assertEqual(SS.vehicle_pose(4.6, 1.5, D=1.8), "side")
        self.assertEqual(SS.vehicle_pose(4.6, 1.5, D=4.0), "")                 # side-on but 4 m deep → a block

    def test_static_analysis_element_never_removed(self):
        self.assertEqual(SS.is_dynamic_prop(1.8, 1.5, allow=(), static=True), "")
        an = {"primitives": [{"type": "sign", "label_zh": "售货亭", "x": 3.0, "y": 12.0, "w": 2.0, "d": 1.2, "h": 3.0},
                             {"type": "wall", "label_zh": "立柱", "x": -4.0, "y": 9.0, "w": 0.5, "d": 0.5, "h": 3.0,
                              "repeat": 3, "repeat_dx": 3.0, "repeat_dy": 0.0},
                             {"type": "vehicle", "label_zh": "出租车", "x": 8.0, "y": 14.0, "w": 1.8, "d": 4.5, "h": 1.5},
                             {"type": "building", "label_zh": "商场", "x": 0.0, "y": 40.0, "w": 60, "d": 20, "h": 15}]}
        fps = SS.static_footprints(an)
        self.assertEqual(len(fps), 4)                                          # kiosk + 3 pillars; no car, no building
        self.assertTrue(SS.in_static_footprint(3.4, 12.3, fps))
        self.assertTrue(SS.in_static_footprint(2.0, 9.0, fps))                 # 3rd pillar (repeat)
        self.assertFalse(SS.in_static_footprint(8.0, 14.0, fps))

    def test_measured_kiosk_kept_in_blockout(self):
        """A measured 2.0 x 3.0 m free-standing cluster survives build_volumes (and is logged as a prop)."""
        src = (PKG / "detail_blockout.py").read_text(encoding="utf-8")
        self.assertIn("dyn = SSm.is_dynamic_prop(W, hh, D=Dm, static=is_static)", src)
        self.assertIn("static_fp = _static_scene().static_footprints(analysis)", src)
        self.assertIn('stats["static_kept"]', src)


class GroundLock(unittest.TestCase):
    AN = {"palette": {"ground_words": "Warm sandy ochre gravel with scattered gray-brown stones.",
                      "ground_hex": "#AF895E"},
          "terrain": {"surface": "Nearly level compacted sandy earth with coarse ochre gravel."},
          "roads": [{"kind": "road", "label_zh": "尘土公路", "surface": "dark gray-brown dusty surfacing, no lane markings"}]}

    def test_dry_gravel_scene(self):
        t = MV.ground_lock_text(self.AN, {"rgb": [175, 137, 94], "hex": "#AF895E"})
        self.assertIn("【地面锁定】", t)
        self.assertIn("ochre gravel", t)
        self.assertIn("#AF895E", t)
        self.assertIn("不要变湿", t)
        self.assertIn("不要把地面换成沥青", t)
        self.assertIn("GROUND LOCK", t)

    def test_wet_night_street_is_not_dried(self):
        an = {"palette": {"ground_words": "wet dark asphalt with neon reflections"},
              "roads": [{"kind": "road", "surface": "asphalt"}]}
        t = MV.ground_lock_text(an, {})
        self.assertNotIn("不要变湿", t)
        self.assertNotIn("不要把地面换成沥青", t)
        self.assertIn("保持主视角那种路面材质", t)

    def test_empty(self):
        self.assertEqual(MV.ground_lock_text({}, {}), "")

    def test_in_view_text_and_wired(self):
        t = MV.view_text("体块", "光", ["主视角"], ground_line="【地面锁定】x")
        self.assertIn("【地面锁定】x", t)
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        # 1.2.8: the measured 主视角 ground (mground) feeds the lock; each view merges its 地面分带 into it
        self.assertIn("mground = MV.master_ground(master_view)", ops)
        self.assertIn("ground_line = MV.ground_lock_text(sb_analysis, mground)", ops)
        self.assertIn("ground_line=view_ground)", ops)

    def test_master_ground_colour(self):
        H, W = 40, 60
        pos = np.zeros((H, W, 4), np.float32)
        pos[..., 3] = 1
        pos[H // 2:, :, 2] = 0.0
        pos[: H // 2, :, 2] = 5.0                                            # a wall above
        nrm = np.zeros((H, W, 3), np.float32)
        nrm[H // 2:, :, 2] = 1.0
        nrm[: H // 2, :, 1] = 1.0
        img = np.zeros((H, W, 3), np.float32)
        img[H // 2:] = (175, 137, 94)
        img[: H // 2] = (40, 40, 40)
        g = MV.master_ground({"img": img, "pos": pos, "nrm": nrm, "mask": np.zeros((H, W), np.float32)})
        self.assertEqual(g["hex"], "#AF895E")


class RowContinuation(unittest.TestCase):
    """Measured building row (master view) → inferred lots continue on the row's own line, behind the camera too."""

    def setUp(self):
        self.M = DB.Mesh()
        # a row of 4 houses along a line rotated 20° from the street (analysis road along x)
        ang = math.radians(20)
        self.u = np.array([math.cos(ang), math.sin(ang)])
        self.n = np.array([-self.u[1], self.u[0]])
        self.parts = []
        for k in range(4):
            c = self.u * (-9 + 6 * k) + self.n * 22.0
            self.parts.append({"kind": "building", "xy": [float(c[0]), float(c[1])], "u": list(map(float, self.u)),
                               "n": list(map(float, -self.n)), "W": 5.5, "D": 6.0, "top": 4.5, "g0": 0.0,
                               "dist": round(float(np.linalg.norm(c))), "inferred": False})

    def test_rows_follow_measured_line(self):
        an = {"roads": [{"kind": "road", "width_m": 7.8,
                         "points": [{"x": -60, "y": 6}, {"x": 60, "y": 6}]}]}
        st, layout = SC.complete(DB._Prims(), self.M, self.parts, lambda x, y: 0.0,
                                 lambda pts: np.zeros(len(pts), bool), pad=(0.0, 11.0, 2.5, 0.0), analysis=an)
        self.assertTrue(st["ok"], st)
        rows = layout.get("rows") or []
        self.assertTrue(rows, "measured row line recorded")
        line_dir = np.array(rows[0]["u"])
        self.assertGreater(abs(float(line_dir @ self.u)), 0.99)
        far = [b for b in layout["boxes"] if b["kind"] == "building"
               and abs(float((np.array(b["c"]) - np.array(rows[0]["c"])) @ self.n)) < 4.0
               and abs(float((np.array(b["c"]) - np.array(rows[0]["c"])) @ self.u)) > 20]
        self.assertGreaterEqual(len(far), 4, "lots continue on the measured line beyond the photo")


    def test_back_closure_without_street(self):
        """No road, too few prisms for a street: nothing to mirror → the row closes around the back."""
        st, layout = SC.complete(DB._Prims(), self.M, self.parts, lambda x, y: 0.0,
                                 lambda pts: np.zeros(len(pts), bool), pad=(0.0, 11.0, 2.5, 0.0), analysis={})
        self.assertTrue(st["ok"], st)
        self.assertEqual(st["mirrored"], [])
        self.assertGreater(st.get("back_lots", 0), 3)
        prod = np.array([0.0, 11.0])
        backs = [b for b in layout["boxes"] if b["conf"] == SC.CONF_BACK]
        # some lots stand on the camera side (y < product y): what a 216° orbit camera looks at
        self.assertTrue(any(b["c"][1] < 11.0 - 5 for b in backs))
        for b in backs:
            self.assertGreater(float(np.hypot(*(np.array(b["c"]) - prod))), layout["clear_r"])
        # never on water
        M2 = DB.Mesh()
        st2, _l = SC.complete(DB._Prims(), M2, self.parts, lambda x, y: 0.0, lambda pts: np.zeros(len(pts), bool),
                              pad=(0.0, 11.0, 2.5, 0.0), analysis={"terrain": {"has_sea": True}})
        self.assertEqual(st2.get("back_lots", 0), 0)

    def test_back_closure_skips_what_the_master_sees(self):
        seen = lambda pts: np.asarray(pts)[:, 1] < 11.0        # noqa: E731 — pretend the master sees y < 11
        st, layout = SC.complete(DB._Prims(), self.M, self.parts, lambda x, y: 0.0, seen,
                                 pad=(0.0, 11.0, 2.5, 0.0), analysis={})
        for b in layout["boxes"]:
            self.assertGreaterEqual(b["c"][1] + abs(b["a"][0]) + abs(b["b"][0]), 11.0 - 1e-6)


class NetworkRetry(unittest.TestCase):
    def test_transient_classification(self):
        E = direct.DirectEditError
        self.assertTrue(direct.transient_network(E("网络请求失败：SSLEOFError [SSL: UNEXPECTED_EOF_WHILE_READING]", kind="network")))
        self.assertTrue(direct.transient_network(E("网络请求失败：ConnectionResetError [Errno 54] Connection reset by peer",
                                                   kind="network")))
        self.assertTrue(direct.transient_network(E("图像结果传输中断", kind="stream_cut")))
        self.assertFalse(direct.transient_network(E("网络请求失败：gaierror nodename nor servname provided", kind="network")))
        self.assertEqual(direct.NET_RETRIES, 2)


class _NetBase(DirectEditTestBase):
    def setUp(self):
        super().setUp()
        self._backoff = direct.NET_BACKOFF
        direct.NET_BACKOFF = (0.0, 0.0)

    def tearDown(self):
        direct.NET_BACKOFF = self._backoff
        super().tearDown()

    def run_gen(self):
        self.fb_calls = []

        def fb(**kw):
            self.fb_calls.append(kw)
            return {"engine": "codex_oauth_imagegen"}
        self.gstate = {}
        kwargs = dict(prompt="p", reference_paths=self.refs, output_path=str(self.tmp / "o.png"),
                      cwd=str(self.tmp), edit_base_path=self.camera, timeout=60)
        return direct.generate_with_fallback(kwargs, fallback=fb, state=self.gstate, size="2048x1152")


class ResetThenOk(_NetBase):
    mode = "stream_cut_then_ok"

    def test_retry_then_direct(self):
        r = self.run_gen()
        self.assertEqual(r["engine"], "codex_direct_edit")
        self.assertEqual(self.fb_calls, [])


class ResetAlwaysFallsBack(_NetBase):
    mode = "stream_cut_always"

    def test_two_retries_then_proxy(self):
        r = self.run_gen()
        self.assertEqual(r["engine"], "codex_oauth_imagegen")
        self.assertEqual(self.state.response_calls, 1 + direct.NET_RETRIES)
        self.assertFalse(self.gstate.get("disabled"))


class ExecTimeoutRetry(unittest.TestCase):
    def test_attempt_limits(self):
        self.assertEqual(CC.render_attempt_timeouts(1800), [600, 1200])
        self.assertEqual(CC.render_attempt_timeouts(3600), [600, 3000])
        self.assertEqual(CC.render_attempt_timeouts(600), [600])
        self.assertEqual(CC.render_attempt_timeouts(0), [600, None])

    def test_slow_call_retried_once(self):
        tmp = Path(tempfile.mkdtemp(prefix="wondful132t_"))
        cam = png(tmp / "camera_base.png", 320, 180)
        calls = []

        def fake_exec(**kw):
            calls.append(kw["timeout"])
            if len(calls) == 1:
                raise CC.CodexTimeoutError("Codex 操作超过插件允许的总时长（600s）")
            Path(kw["cwd"], "out.png").write_bytes(Path(cam).read_bytes())
            return "WONDFUL_IMAGEGEN_OK"
        orig = CC._exec_turn
        CC._exec_turn = fake_exec
        try:
            res = CC.generate_image(prompt="p", reference_paths=[cam], output_path=str(tmp / "out.png"),
                                    cwd=str(tmp), edit_base_path=cam, timeout=1800)
        finally:
            CC._exec_turn = orig
        self.assertEqual(calls, [600, 1200])
        self.assertEqual(res["status"], "ok")

    def test_two_timeouts_raise_clear_error(self):
        tmp = Path(tempfile.mkdtemp(prefix="wondful132u_"))
        cam = png(tmp / "camera_base.png", 320, 180)

        def fake_exec(**kw):
            raise CC.CodexTimeoutError("超时")
        orig = CC._exec_turn
        CC._exec_turn = fake_exec
        try:
            with self.assertRaises(CC.CodexTimeoutError) as cm:
                CC.generate_image(prompt="p", reference_paths=[cam], output_path=str(tmp / "out.png"),
                                  cwd=str(tmp), edit_base_path=cam, timeout=1800)
        finally:
            CC._exec_turn = orig
        self.assertIn("自动重试 1 次", str(cm.exception))

    def test_default_total_still_1800_and_named(self):
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        i = props.index("codex_render_timeout: IntProperty(")
        self.assertIn("default=1800", props[i:i + 400])
        # Preferences now render through the shared field helper; retain the
        # timeout in the generation-settings field list.
        self.assertIn('"codex_polish_timeout", "codex_audit_timeout", "codex_render_timeout", "antigravity_polish_timeout"', props)
        self.assertIn('UIC.field(engine, self, name, None)', props)
        cc = (PKG / "codex_client.py").read_text(encoding="utf-8")
        self.assertIn("Codex ImageGen 超时（秒）", cc)


class Heartbeat(unittest.TestCase):
    def setUp(self):
        src = (PKG / "operators.py").read_text(encoding="utf-8")
        i = src.index("HEARTBEAT_AFTER = ")
        j = src.index("class _BaseAsyncOperator(")
        ns = {"math": math}
        exec(src[i:j], ns)          # noqa: S102 — the pure helper, without bpy
        self.hb = ns["heartbeat"]

    def test_never_parks_at_88(self):
        p1, _e, _t = self.hb(120, 120, 0.10, 0.88, "适配提示词")
        p2, _e, t2 = self.hb(800, 120, 0.10, 0.88, "适配提示词")
        p3, _e, _t = self.hb(1600, 120, 0.10, 0.88, "适配提示词")
        self.assertGreater(p2, p1)
        self.assertGreater(p3, p2)
        self.assertLess(p3, 0.97)
        self.assertIn("已等 13 分 20 秒", t2)
        self.assertIn("比平时慢", t2)

    def test_quiet_early_and_phase_clock(self):
        p, eta, t = self.hb(5, 60, 0.1, 0.9, "分析当前风格 (1/2)")
        self.assertEqual(t, "分析当前风格 (1/2)")
        self.assertEqual(eta, 55)
        _p, _e, t = self.hb(50, 60, 0.1, 0.9, "生成提示词 (2/2)", phase_elapsed=30)
        self.assertIn("已等 30 秒", t)

    def test_wired(self):
        src = (PKG / "operators.py").read_text(encoding="utf-8")
        i = src.index("    def _progress_update(")
        self.assertIn("heartbeat(", src[i:i + 600])


class PureSceneMultiview(unittest.TestCase):
    def test_no_product_allowed(self):
        src = (PKG / "operators.py").read_text(encoding="utf-8")
        i = src.index("class WONDFUL_OT_multiview_render(")
        body = src[i:i + 60000]
        self.assertNotIn("多角度场景一致需要先在 ② 选择产品集合", body)
        self.assertIn("no_product = coll is None", body)
        self.assertIn("if not no_product:      # 1.2.7 纯场景", body)
        self.assertIn("纯场景（未选产品集合）：已跳过产品验收", src)
        j = src.index("    def _plan_views(")
        self.assertNotIn("渲染多个视角需要先在 ② 选择产品集合", src[j:j + 1800])


if __name__ == "__main__":
    unittest.main()
