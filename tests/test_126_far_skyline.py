"""1.2.5 远山剪影 (user 2026-10-20: 「远山可以相对粗糙一些 只保留形即可」): the far range keeps only the measured
SHAPE of the sky boundary — strongly smoothed, low poly (≤ 2k tris), closed ridge with crest thickness and back
slope, one constant distance (no depth noise); unreliable measurement → the LLM 远景带 stay. Optional, off."""
import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "wondful_ai_renderer"
_spec = importlib.util.spec_from_file_location("detail_blockout_sky", PKG / "detail_blockout.py")
DB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DB)


def synth(shape=lambda a: 5 + 3 * np.sin(np.radians(a) * 6), bump=1.0, near=None, H=160, W=320, hfov=60.0,
          far_r=800.0, cam_h=1.7, noise=0.0, seed=0):
    """Pinhole geo: sky above ``shape(az)`` (+ a ±bump° saw-tooth of 1.5° period), terrain at ``far_r`` below.
    ``near`` = (az0, az1, elev) a near occluder 30 m away rising to ``elev`` degrees."""
    rng = np.random.default_rng(seed)
    f = (W / 2) / math.tan(math.radians(hfov / 2))
    xs = (np.arange(W) + 0.5 - W / 2) / f
    ys = (H / 2 - (np.arange(H) + 0.5)) / f
    X, Y = np.meshgrid(xs, ys)
    d = np.stack([X, np.ones_like(X), Y], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    az = np.degrees(np.arctan2(d[..., 0], d[..., 1]))
    el = np.degrees(np.arcsin(d[..., 2]))
    sky_el = shape(az) + bump * np.sign(np.sin(az / 1.5 * math.pi))
    mask = el < sky_el
    r = np.full(az.shape, far_r) * (1 + noise * rng.standard_normal(az.shape))
    if near is not None:
        a0, a1, e = near
        nm = (az > a0) & (az < a1) & (el < e)
        mask |= nm
        r = np.where(nm, 30.0, r)
    t = r / np.maximum(np.hypot(d[..., 0], d[..., 1]), 1e-6)
    P = d * t[..., None]
    P[..., 2] += cam_h
    P[~mask] = 0
    cls = np.where(mask, DB.STRUCT if hasattr(DB, "STRUCT") else 2, 0).astype(np.uint8)
    return {"P": P.astype(np.float32), "N": np.zeros_like(P, np.float32), "mask": mask, "cls": cls,
            "cam": {"height_m": cam_h, "pitch_deg": 0.0, "roll_deg": 0.0, "hfov_deg": hfov}}


class Skyline(unittest.TestCase):
    def test_shape_kept_bumps_removed(self):
        geo = synth()
        sk = DB.measured_skyline(geo)
        self.assertTrue(sk["ok"], sk.get("reason"))
        az = np.asarray(sk["az"])
        el = np.asarray(sk["el"])
        want = 5 + 3 * np.sin(np.radians(az) * 6)
        inner = slice(4, -4)
        # the large-scale form survives (within the removed saw-tooth's own ±1° amplitude)
        self.assertLess(float(np.abs(el - want)[inner].max()), 1.25)
        sk0 = DB.measured_skyline(synth(bump=0.0))
        self.assertLess(float(np.abs(np.asarray(sk0["el"]) - want)[inner].max()), 0.3)
        raw = np.asarray([np.nan if v is None else v for v in sk["raw"]], float)
        self.assertGreater(float(np.nanstd(np.diff(raw))), 0.4)                  # raw had the saw-tooth …
        self.assertLess(float(np.abs(np.diff(el)).max()), 0.5)                   # … the silhouette does not
        self.assertEqual(sk["dist"], 800.0)

    def test_depth_noise_never_reaches_the_ridge(self):
        a = DB.measured_skyline(synth(noise=0.0))
        b = DB.measured_skyline(synth(noise=0.15, seed=4))
        self.assertTrue(a["ok"] and b["ok"])
        self.assertLess(float(np.abs(np.asarray(a["el"]) - np.asarray(b["el"])).max()), 1e-6)   # direction only
        Vb, _ = DB.skyline_ridge(b)
        rr = np.hypot(Vb[:, 0], Vb[:, 1]).reshape(-1, len(DB.SKY_SECTION))
        self.assertLess(float(np.ptp(rr[:, 2])), 1e-6)                           # one constant crest distance

    def test_ridge_closed_low_poly_with_back_slope(self):
        for hfov in (40.0, 60.0, 100.0):
            sk = DB.measured_skyline(synth(hfov=hfov))
            self.assertTrue(sk["ok"])
            V, F = DB.skyline_ridge(sk)
            self.assertLessEqual(len(F), DB.SKY_TRI_BUDGET)
            Vw, Fw = DB.weld(V, F)
            self.assertEqual(len(DB.open_edges(Fw)), 0)
            sec = V.reshape(-1, len(DB.SKY_SECTION), 3)
            r = np.hypot(sec[..., 0], sec[..., 1])
            self.assertTrue((r[:, 3] > r[:, 2]).all())        # crest thickness
            self.assertTrue((r[:, 4] > r[:, 3]).all())        # back slope behind the crest
            self.assertTrue((sec[:, 4, 2] < sec[:, 3, 2]).all())

    def test_ridge_reproduces_the_silhouette_from_the_camera(self):
        geo = synth()
        sk = DB.measured_skyline(geo)
        V, _ = DB.skyline_ridge(sk)
        sec = V.reshape(-1, len(DB.SKY_SECTION), 3)
        az = np.degrees(np.arctan2(sec[:, 2, 0], sec[:, 2, 1]))
        el = np.degrees(np.arctan2(sec[:, 2, 2] - sk["cam_h"], np.hypot(sec[:, 2, 0], sec[:, 2, 1])))
        a0, a1 = sk["span_deg"]
        ins = (az > a0 + 3) & (az < a1 - 3)
        self.assertLess(float(np.abs(el[ins] - DB.skyline_profile(sk, az[ins])).max()), 1e-6)

    def test_profile_continuous_all_round(self):
        sk = DB.measured_skyline(synth())
        a = np.linspace(-180, 180, 2000)
        p = DB.skyline_profile(sk, a)
        self.assertLess(float(np.abs(np.diff(p)).max()), 0.3)
        self.assertLess(abs(float(DB.skyline_profile(sk, [-180.0])[0] - DB.skyline_profile(sk, [180.0])[0])), 1e-9)

    def test_fallback_when_near_hides_the_boundary(self):
        sk = DB.measured_skyline(synth(near=(-25, 15, 25)))
        self.assertFalse(sk["ok"])
        self.assertIn("挡住", sk["reason"])

    def test_fallback_on_jumps(self):
        jag = lambda a: 4 + 9 * (np.floor(a / 4) % 2)          # noqa: E731 — 9° steps every 4°
        sk = DB.measured_skyline(synth(shape=jag, bump=0.0))
        self.assertFalse(sk["ok"])
        self.assertIn("跳变", sk["reason"])

    def test_no_sky(self):
        geo = synth()
        geo["mask"][:] = True
        self.assertFalse(DB.measured_skyline(geo)["ok"])

    def test_far_band_roles(self):
        self.assertTrue(DB.is_far_band("band_rocky"))
        self.assertTrue(DB.is_far_band("band_dune"))
        self.assertFalse(DB.is_far_band("band_sea"))
        self.assertFalse(DB.is_far_band("terrain"))

    def test_recorded_measurements(self):
        rec = Path("/workspace/work/precise_blockout")
        if not (rec / "desert_geo_plugin.npz").is_file():
            self.skipTest("recorded MoGe measurements not available")
        sys.path.insert(0, str(ROOT))
        spec = importlib.util.spec_from_file_location("pd_sky", PKG / "precise_depth.py")
        PD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(PD)
        d = DB.measured_skyline(PD.load_geo(str(rec / "desert_geo_plugin.npz")))
        s = DB.measured_skyline(PD.load_geo(str(rec / "street_geo_plugin.npz")))
        self.assertTrue(d["ok"])                    # open desert: dunes against the sky
        self.assertFalse(s["ok"])                   # street: rooftops (near) hide the far range


class Wiring(unittest.TestCase):
    def test_option_default_off_under_multiview(self):
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        i = props.index("sb_detail_skyline: BoolProperty(")
        self.assertIn("default=False", props[i:i + 200])
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        self.assertIn('sub.prop(props, "sb_detail_skyline", text="远山剪影")', ui)

    def test_render_path(self):
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        self.assertIn('detail_mode += "+sky"', ops)
        vc = (PKG / "viewport_capture.py").read_text(encoding="utf-8")
        self.assertIn('skyline = detail.endswith("+sky")', vc)
        self.assertIn("skyline=skyline) as dp:", vc)
        db = (PKG / "detail_blockout.py").read_text(encoding="utf-8")
        self.assertIn("if o.get(SKY_TAG) and not sky:", db)
        self.assertIn("sky and is_far_band(", db)


if __name__ == "__main__":
    unittest.main()
