"""1.2.0 分层输出: PNG IO, mask fit, alpha, inpaint, shadow, PSD writer, driver, plugin wiring."""
import ast
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

png_io = load_package_module("png_io")
LO = load_package_module("layered_output")
PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"

try:
    from psd_tools import PSDImage  # optional, sandbox only
except Exception:  # pragma: no cover
    PSDImage = None


def _encode_png(arr, filters):
    """Reference PNG encoder applying the given per-row filter types (to test the decoder)."""
    h, w, c = arr.shape
    bpp = c
    a = arr.reshape(h, w * c).astype(np.int32)
    rows = []
    prev = np.zeros(w * c, np.int32)
    for y in range(h):
        f = filters[y % len(filters)]
        cur = a[y]
        left = np.concatenate([np.zeros(bpp, np.int32), cur[:-bpp]])
        upleft = np.concatenate([np.zeros(bpp, np.int32), prev[:-bpp]])
        if f == 0:
            out = cur
        elif f == 1:
            out = cur - left
        elif f == 2:
            out = cur - prev
        elif f == 3:
            out = cur - (left + prev) // 2
        else:
            p = left + prev - upleft
            pa, pb, pc = abs(p - left), abs(p - prev), abs(p - upleft)
            pred = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, prev, upleft))
            out = cur - pred
        rows.append(bytes([f]) + (out & 0xFF).astype(np.uint8).tobytes())
        prev = cur
    ctype = {3: 2, 4: 6, 1: 0}[c]

    def chunk(k, d):
        return struct.pack(">I", len(d)) + k + d + struct.pack(">I", zlib.crc32(k + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


def synthetic_scene(w=320, h=200, scale=1.0, dx=0, dy=0):
    """Gradient sky + textured ground, a two-tone box 'product' and a soft ground shadow."""
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    horizon = int(h * 0.45)
    img[:horizon] = np.stack([0.45 + 0.2 * yy[:horizon] / h, 0.6 + 0.1 * yy[:horizon] / h,
                              np.full_like(yy[:horizon], 0.9)], -1)
    ground = np.stack([np.full_like(yy, 0.78), np.full_like(yy, 0.66), np.full_like(yy, 0.48)], -1)
    ground += rng.normal(0, 0.012, ground.shape).astype(np.float32)
    img[horizon:] = ground[horizon:]
    cx, cy = w * 0.5, h * 0.55

    def product_mask(s, ox, oy):
        bw, bh = w * 0.36 * s, h * 0.30 * s
        return ((np.abs(xx - cx - ox) < bw / 2) & (np.abs(yy - cy - oy) < bh / 2)).astype(np.float32), bh
    true_mask, bh = product_mask(scale, dx, dy)
    bottom = cy + dy + bh / 2
    shadow = np.exp(-((yy - bottom - 4) / 6.0) ** 2) * (np.abs(xx - cx - dx) < w * 0.24 * scale)
    shadow = (shadow * 0.55 * (yy > bottom - 2)).astype(np.float32)
    plate = img.copy()
    img = img * (1 - shadow[..., None])
    body = np.where((yy < cy + dy)[..., None], [0.10, 0.55, 0.20], [0.05, 0.30, 0.10]).astype(np.float32)
    img = np.where(true_mask[..., None] > 0, body, img)
    blender_mask, _ = product_mask(1.0, 0, 0)
    return img.astype(np.float32), blender_mask, true_mask, shadow, plate


class PngIoTest(unittest.TestCase):
    def test_every_filter_type_round_trips(self):
        rng = np.random.default_rng(0)
        for channels in (1, 3, 4):
            arr = rng.integers(0, 256, (23, 31, channels), dtype=np.uint8)
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "f.png"
                path.write_bytes(_encode_png(arr, [0, 1, 2, 3, 4, 4, 3]))
                out = png_io.read_png(path)
            expect = arr if channels == 4 else np.dstack([arr if channels == 3 else np.repeat(arr, 3, 2),
                                                          np.full(arr.shape[:2] + (1,), 255, np.uint8)])
            np.testing.assert_array_equal(out, expect)

    def test_writer_reader_round_trip(self):
        rng = np.random.default_rng(1)
        arr = rng.integers(0, 256, (40, 57, 4), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as d:
            path = png_io.write_png(Path(d) / "w.png", arr)
            np.testing.assert_array_equal(png_io.read_png(path), arr)


class MaskFitTest(unittest.TestCase):
    def test_recovers_scale_and_shift(self):
        img, blender_mask, true_mask, _, _ = synthetic_scene(scale=1.08, dx=6, dy=-4)
        fit = LO.fit_mask(img, blender_mask)
        self.assertAlmostEqual((fit["scale_x"] + fit["scale_y"]) / 2, 1.08, delta=0.025)
        iou_before = (np.minimum(blender_mask, true_mask).sum() / np.maximum(blender_mask, true_mask).sum())
        m = fit["mask"] > 0.5
        iou_after = (m & (true_mask > 0)).sum() / (m | (true_mask > 0)).sum()
        self.assertGreater(iou_after, 0.95)
        self.assertGreater(iou_after, iou_before + 0.05)

    def test_aligned_mask_is_kept(self):
        img, blender_mask, _, _, _ = synthetic_scene()
        fit = LO.fit_mask(img, blender_mask)
        self.assertLessEqual(abs(fit["dx"]) + abs(fit["dy"]), 2.0)
        self.assertAlmostEqual(fit["scale"], 1.0, delta=0.012)


class AlphaInpaintShadowTest(unittest.TestCase):
    def test_alpha_follows_true_edge(self):
        img, _, true_mask, _, _ = synthetic_scene()
        # 1.2.2: the (fitted) Blender mask is trusted; colour may only move alpha within the
        # 2–3 px band and only where it agrees with the mask, so a 2-px-too-large prior gets
        # partly (not fully) pulled back, and never grows.
        prior = LO.dilate(true_mask > 0, 2).astype(np.float32)
        a = LO.refine_alpha(img, prior)["alpha"]
        err = np.abs(a - true_mask).mean()
        self.assertLess(err, 0.015)
        self.assertFalse(((a > 0.02) & (prior < 0.01)).any())
        exact = LO.refine_alpha(img, true_mask.astype(np.float32))["alpha"]
        self.assertLess(np.abs(exact - true_mask).mean(), 0.004)
        self.assertTrue((a[true_mask > 0] > 0.9).mean() > 0.98)

    def test_pushpull_fills_smoothly(self):
        h, w = 64, 96
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        base = np.dstack([xx / w, yy / h, np.full_like(xx, 0.5)]).astype(np.float32)
        hole = (np.abs(xx - 48) < 14) & (np.abs(yy - 32) < 10)
        damaged = base.copy()
        damaged[hole] = [1.0, 0.0, 1.0]
        out = LO.inpaint_pushpull(damaged, hole)
        np.testing.assert_array_equal(out[~hole], base[~hole])
        self.assertLess(np.abs(out[hole] - base[hole]).mean(), 0.05)

    def test_shadow_from_true_plate(self):
        img, _, true_mask, shadow, plate = synthetic_scene()
        reg = LO.product_region(true_mask)
        k = LO.estimate_shadow(img, plate, reg["region"], reg["mid_y"], true_mask)
        outside = true_mask == 0
        strong = (shadow > 0.3) & outside
        self.assertGreater(k[strong].mean(), 0.3)
        far = ~LO.dilate(shadow > 0.01, 4) & outside
        self.assertLess(k[far].max(), 0.1)
        self.assertEqual(float(k[true_mask > 0].max()), 0.0)

    def test_shadow_local_plate_mode(self):
        img, _, true_mask, shadow, _ = synthetic_scene()
        reg = LO.product_region(true_mask)
        approx = LO.inpaint_pushpull(img, reg["region"])
        k = LO.estimate_shadow(img, approx, reg["region"], reg["mid_y"], true_mask)
        strong = (shadow > 0.3) & (true_mask == 0)
        self.assertGreater(k[strong].mean(), 0.2)
        ys = np.nonzero(k > 0.05)[0]
        self.assertGreaterEqual(ys.min(), reg["mid_y"])


class PsdWriterTest(unittest.TestCase):
    def test_packbits_decodes(self):
        rng = np.random.default_rng(2)
        rows = [np.zeros(300, np.uint8), np.full(300, 7, np.uint8), rng.integers(0, 256, 300).astype(np.uint8),
                np.repeat(rng.integers(0, 4, 60), 5).astype(np.uint8)]
        for row in rows:
            data = LO._packbits_row(row)
            out, i = bytearray(), 0
            while i < len(data):
                n = data[i]
                if n < 128:
                    out += data[i + 1:i + 2 + n]; i += 2 + n
                elif n > 128:
                    out += bytes([data[i + 1]]) * (257 - n); i += 2
                else:
                    i += 1
            self.assertEqual(bytes(out), row.tobytes())

    def test_psd_structure_and_psd_tools(self):
        rng = np.random.default_rng(4)
        h, w = 37, 53
        bg = np.dstack([rng.integers(0, 256, (h, w, 3)), np.full((h, w), 255)]).astype(np.uint8)
        sh = np.zeros((h, w, 4), np.uint8); sh[20:, :, 3] = 90
        sub = np.zeros((h, w, 4), np.uint8); sub[5:20, 10:40] = [10, 200, 30, 255]
        with tempfile.TemporaryDirectory() as d:
            path = LO.write_psd(Path(d) / "t.psd", [{"name": "Background", "rgba": bg},
                                                    {"name": "Shadow", "rgba": sh},
                                                    {"name": "Subject", "rgba": sub, "visible": False}],
                                bg[..., :3])
            data = Path(path).read_bytes()
            self.assertEqual(data[:4], b"8BPS")
            ch, hh, ww, depth, mode = struct.unpack(">HIIHH", data[12:26])
            self.assertEqual((ch, hh, ww, depth, mode), (3, h, w, 8, 3))
            for name in (b"Background", b"Shadow", b"Subject"):
                self.assertIn(name, data)
            if PSDImage is None:
                self.skipTest("psd-tools not installed")
            psd = PSDImage.open(path)
            self.assertEqual([l.name for l in psd], ["Background", "Shadow", "Subject"])
            self.assertEqual([l.visible for l in psd], [True, True, False])
            np.testing.assert_array_equal(np.asarray(psd[0].topil()), bg)
            np.testing.assert_array_equal(np.asarray(psd[2].topil()), sub)
            self.assertEqual(int(np.asarray(psd[1].topil())[30, 5, 3]), 90)


class DriverTest(unittest.TestCase):
    def _files(self, d, **kw):
        img, blender_mask, _, _, plate = synthetic_scene(**kw)
        ip = png_io.write_png(Path(d) / "Wondful_Scene_Cam_x.png",
                              np.dstack([np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8),
                                         np.full(img.shape[:2], 255, np.uint8)]))
        mp = png_io.write_png(Path(d) / "mask.png", np.dstack([np.zeros(img.shape, np.uint8),
                                                                (blender_mask * 255).astype(np.uint8)]))
        pp = png_io.write_png(Path(d) / "plate.png", np.clip(plate * 255 + 0.5, 0, 255).astype(np.uint8))
        return ip, mp, pp

    def test_shadow_mode_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            ip, mp, _ = self._files(d, scale=1.05)
            rep = LO.build_layers(ip, mp, mode=LO.MODE_SHADOW)
            paths = LO.layer_paths(ip)
            self.assertTrue(Path(paths["subject"]).is_file() and Path(paths["shadow"]).is_file())
            self.assertFalse(Path(paths["background"]).exists())
            self.assertTrue(Path(paths["psd"]).is_file())
            self.assertEqual(rep["plate"], "local_inpaint")
            self.assertLess(rep["recomposite_err"], 3.0)
            subj = png_io.read_png(paths["subject"])
            self.assertEqual(subj.shape[2], 4)
            if PSDImage is not None:
                self.assertEqual([l.name for l in PSDImage.open(paths["psd"])], ["Original", "Shadow", "Subject"])

    def test_full_mode_uses_plate_and_never_raises_on_plate_failure(self):
        with tempfile.TemporaryDirectory() as d:
            ip, mp, pp = self._files(d)
            calls = []

            def plate_fn(image, mask, region):
                calls.append(mask)
                m = png_io.read_png(mask)
                self.assertEqual(int(m[region][..., 3].max()), 0)    # transparent = editable
                self.assertEqual(int(m[~region][..., 3].min()), 255)
                return pp
            rep = LO.build_layers(ip, mp, mode=LO.MODE_FULL, plate_fn=plate_fn, work_dir=d)
            self.assertEqual(rep["plate"], "codex_edit")
            self.assertTrue(Path(LO.layer_paths(ip)["background"]).is_file())
            self.assertLess(rep["recomposite_err"], 1.5)
            self.assertFalse(Path(calls[0]).exists())  # temporary edit mask removed

            def broken(*_a):
                raise RuntimeError("HTTP 429")
            rep = LO.build_layers(ip, mp, mode=LO.MODE_FULL, plate_fn=broken)
            self.assertEqual(rep["plate"], "local_inpaint")
            self.assertIn("429", rep["plate_error"])

    def test_runner_reports_failures(self):
        with tempfile.TemporaryDirectory() as d:
            ip, mp, _ = self._files(d)
            runner = LO.LayerRunner()
            runner.submit(ip, mp, LO.MODE_SHADOW, work_dir=d)
            runner.submit(str(Path(d) / "missing.png"), mp, LO.MODE_SHADOW)
            import time
            for _ in range(200):
                if runner.pending == 0:
                    break
                time.sleep(0.05)
            res = runner.drain()
            self.assertEqual([r["ok"] for r in res], [True, False])
            self.assertIn("分层输出完成 1 张", LO.summarize(res))
            self.assertIn("失败", LO.summarize(res))


class WiringTest(unittest.TestCase):
    def test_no_forbidden_imports(self):
        for name in ("layered_output.py", "png_io.py"):
            tree = ast.parse((PKG / name).read_text(encoding="utf-8"))
            mods = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
            mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            self.assertFalse(mods & {"cv2", "scipy", "psd_tools", "PIL", "bpy"}, name)

    def test_properties_ui_operator_hooks(self):
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        for key in ("layered_output: BoolProperty", "layered_mode: EnumProperty", "last_layers_note"):
            self.assertIn(key, props)
        self.assertIn('"主体 + 投影"', props)
        self.assertIn('"完整（含干净背景）"', props)
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        adv = ui.index('"ui_show_advanced", "高级"')
        self.assertGreater(ui.index('_section(adv, "分层输出", "RENDERLAYERS", props, "layered_output")'), adv)
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        self.assertIn('product_mask_path = (self._layers or {}).get("mask", "")', ops)
        self.assertIn("mask_path=product_mask_path", ops)
        self.assertIn("self._submit_layers(props, Path(exported))", ops)
        vc = (PKG / "viewport_capture.py").read_text(encoding="utf-8")
        self.assertIn('mask_path: str = ""', vc)

    def test_exact_prompt_sends_plate_instruction_verbatim(self):
        cd = (PKG / "codex_direct.py").read_text(encoding="utf-8")
        self.assertIn("exact_prompt=False", cd)
        self.assertIn("if exact_prompt:\n        text = prompt", cd)
        self.assertNotIn("汽车", LO.plate_prompt(False))
        self.assertIn("汽车", LO.plate_prompt(True))


if __name__ == "__main__":
    unittest.main()
