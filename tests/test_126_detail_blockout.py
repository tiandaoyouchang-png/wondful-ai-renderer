"""1.2.5 细节粗模 (V3 phase 1): pure-numpy measured near / mid blockout + its Blender wiring.

Pure-python part: numpy stand-ins for scipy, closure of every primitive, the 360° heightfield (no holes, product
pad, far blend into the LLM terrain), the near / far split of the LLM blocks, the triangle budget, the indoor /
implausible fallback. Blender part (tools/e2e_detail_blockout_bg.py): the carrier is built hidden by ④ 精确模式,
the option-off and precise-off paths are unchanged, the geometry pass uses the carrier with no holes.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("detail_blockout", ROOT / "wondful_ai_renderer" / "detail_blockout.py")
DB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DB)
import precise_synth  # noqa: E402

BLENDER = shutil.which("blender")
REC = Path("/workspace/work/scene_blockout/out")


def street_geo():
    cam = {"height_m": 1.6, "pitch_deg": 0.0, "roll_deg": 0.0, "hfov_deg": 70.0}
    boxes = [(-30, -8, 20, 40, 0, 12, DB.STRUCT), (8, 26, 18, 34, 0, 9, DB.STRUCT),
             (2.0, 3.8, 9.0, 13.5, 0, 1.5, DB.STRUCT),          # parked car end-on
             (-6, -3, 30, 33, 2.0, 7.0, DB.VEG)]                # a crown
    g = precise_synth.geo(cam, W=480, H=270, boxes=boxes)
    return g


def closed(V, F):
    Vw, Fw = DB.weld(V, F)
    return len(DB.open_edges(Fw)) == 0


class NumpyStandIns(unittest.TestCase):
    def test_label_matches_flood_fill(self):
        rng = np.random.default_rng(3)
        for eight in (False, True):
            for _ in range(5):
                m = rng.random((37, 53)) > 0.55
                lab, n = DB.label(m, eight=eight)
                # same partition as a brute-force flood fill
                ref = np.zeros(m.shape, int)
                k = 0
                nb = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if eight else [])
                for y in range(m.shape[0]):
                    for x in range(m.shape[1]):
                        if m[y, x] and not ref[y, x]:
                            k += 1
                            st = [(y, x)]
                            ref[y, x] = k
                            while st:
                                a, b = st.pop()
                                for dy, dx in nb:
                                    yy, xx = a + dy, b + dx
                                    if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1] and m[yy, xx] and not ref[yy, xx]:
                                        ref[yy, xx] = k
                                        st.append((yy, xx))
                self.assertEqual(n, k)
                self.assertTrue(((lab > 0) == m).all())
                pairs = set(zip(lab[m].tolist(), ref[m].tolist()))
                self.assertEqual(len(pairs), k)          # one-to-one

    def test_find_objects(self):
        lab = np.zeros((10, 10), int)
        lab[2:4, 3:7] = 1
        lab[8, 9] = 2
        sl = DB.find_objects(lab, 2)
        self.assertEqual(sl[0], (slice(2, 4), slice(3, 7)))
        self.assertEqual(sl[1], (slice(8, 9), slice(9, 10)))

    def test_filters(self):
        a = np.zeros((21, 21))
        a[10, 10] = 1.0
        g = DB.gaussian_filter(a, (1.0, 1.0))
        self.assertAlmostEqual(g.sum(), 1.0, places=6)
        self.assertEqual(np.unravel_index(np.argmax(g), g.shape), (10, 10))
        x = np.array([5.0, 1, 9, 2, 7, 3, 8])
        self.assertEqual(DB.median_filter1d(x, 3).tolist(), [5, 5, 2, 7, 3, 7, 8])
        self.assertEqual(DB.maximum_filter1d(x, 3).tolist(), [5, 9, 9, 9, 7, 8, 8])


class Closure(unittest.TestCase):
    def test_primitives_closed(self):
        u, n = np.array([1.0, 0]), np.array([0, -1.0])
        for V, F in (DB.box((0, 10), u, n, 6, 8, 0, 9), DB.rounded_box((3, 8), u, n, 1.8, 4.5, 0, 1.5),
                     DB.ellipsoid(np.array([0, 0, 4.0]), 2, 2, 1.5), DB.cylinder(1, 2, 0.2, 0, 6)):
            F = np.asarray(F)
            if F.shape[1] == 4:
                F = np.r_[F[:, [0, 1, 2]], F[:, [0, 2, 3]]]
            self.assertTrue(closed(V, F))

    def test_prism_windows_closed_and_hidden_faces_only(self):
        M = DB.Mesh()
        made = DB.prism_with_windows(M, (0, 20), np.array([1.0, 0]), np.array([0, -1.0]), 12, 14, -0.5, 15,
                                     np.zeros(2), {"kind": "building"})
        self.assertGreater(made, 0)
        V, F, C = M.arrays()
        b = C == DB.C_BUILDING
        self.assertTrue(closed(V, F[b]))
        w = F[C == DB.C_WINDOW]
        self.assertTrue(closed(V, w))                 # every window frame is a closed box
        # no frame on the face the camera (at the origin) sees: the front at y = 20
        wy = V[np.unique(w)][:, 1]
        self.assertFalse(np.any(np.abs(wy - 20.0) < 0.15))

    def test_skin_closed(self):
        g = street_geo()
        Q, pieces = DB.skin_quads(g["P"], g["mask"], g["cls"], 1.6, 2)
        self.assertTrue(pieces)
        for qm in pieces:
            M = DB.Mesh()
            DB.shell_piece(Q, qm, 1.6, M, DB.C_SKIN, DB.SKIN_T)
            V, F, _C = M.arrays()
            self.assertTrue(closed(V, F))
            self.assertLessEqual(M.parts[0]["t"], DB.SKIN_T)


class Ground(unittest.TestCase):
    def test_heightfield_no_holes(self):
        g = street_geo()
        V, F, T = DB.terrain(g["P"], g["mask"], g["cls"], 70.0)
        r = np.hypot(V[:, 0], V[:, 1])
        oe = DB.open_edges(DB.weld(V, F)[1])
        Vw = DB.weld(V, F)[0]
        self.assertTrue(len(oe) > 0)
        self.assertTrue(np.all(np.hypot(Vw[oe][..., 0], Vw[oe][..., 1]) > DB.R_OUT * 0.99))   # only the outer rim
        self.assertAlmostEqual(float(r.max()), DB.R_OUT, delta=1.0)
        ab = T[0]
        self.assertLess(ab[0], -170)
        self.assertGreater(ab[-1], 170)
        # under the camera flat, measured ground ≈ 0 in the photo
        self.assertLess(abs(DB.terrain_z(T, 0, 1.0)), 1e-6)
        self.assertLess(abs(DB.terrain_z(T, 0, 15.0)), 0.05)

    def test_far_blend_and_pad(self):
        g = street_geo()
        far = lambda X, Y: np.full(np.shape(X), 7.0)       # noqa: E731
        V, F, T = DB.terrain(g["P"], g["mask"], g["cls"], 70.0, far_z=far, pad=(0.0, 12.0, 3.0, 0.4))
        self.assertAlmostEqual(DB.terrain_z(T, 0, 800.0), 7.0, delta=0.05)     # LLM terrain beyond the split
        self.assertLess(abs(DB.terrain_z(T, 0, 60.0)), 0.2)                     # measured near
        self.assertAlmostEqual(DB.terrain_z(T, 0.0, 12.0), 0.4, delta=0.05)     # product pad


class SplitBudgetFallback(unittest.TestCase):
    def test_llm_role(self):
        self.assertEqual(DB.llm_role("band_dune", 10), "keep")
        self.assertEqual(DB.llm_role("band_rocky", 900), "keep")
        self.assertEqual(DB.llm_role("cliff", 50), "keep")
        self.assertEqual(DB.llm_role("terrain", 900), "replace")
        self.assertEqual(DB.llm_role("road", 40), "replace")
        self.assertEqual(DB.llm_role("building", 40), "replace")
        self.assertEqual(DB.llm_role("building", 400), "keep")
        self.assertEqual(DB.llm_role("tree", DB.R_SPLIT - 1), "replace")
        self.assertEqual(DB.llm_role("tree", DB.R_SPLIT + 1), "keep")

    def test_build_synthetic(self):
        g = street_geo()
        t = time.time()
        r = DB.build(g, pad=(2.9, 11.0, 0.5, 0.0))
        self.assertLess(time.time() - t, 2.0)
        self.assertTrue(r["ok"], r.get("reason"))
        s = r["stats"]
        self.assertGreaterEqual(s["buildings"], 2)
        self.assertGreaterEqual(s["trees"], 1)
        self.assertLessEqual(s["tris"], DB.TRI_BUDGET)
        # nothing measured beyond the near / mid split (except the ground)
        V, F, C = r["V"], r["F"], r["C"]
        nong = np.unique(F[C != DB.C_GROUND])
        self.assertLess(float(np.hypot(V[nong, 0], V[nong, 1]).max()), DB.R_SPLIT + 25)

    def test_budget_tight(self):
        g = street_geo()
        r = DB.build(g, budget=30000)
        self.assertTrue(r["ok"])
        self.assertLessEqual(r["stats"]["tris"], 32000)
        self.assertGreater(r["stats"]["skin_stride"], 2)

    @unittest.skipUnless((REC / "street_geo.npz").is_file(), "recorded measurements not present")
    def test_recorded_scenes_budget_and_time(self):
        for name in ("street", "desert", "plaza"):
            z = np.load(REC / f"{name}_geo.npz")
            g = {"P": z["P"], "N": z["N"], "mask": z["mask"].astype(bool), "cls": z["cls"], "cam": json.loads(str(z["cam"]))}
            t = time.time()
            r = DB.build(g)
            self.assertLess(time.time() - t, 2.0, name)
            self.assertTrue(r["ok"], name)
            self.assertLessEqual(r["stats"]["tris"], DB.TRI_BUDGET, name)

    def test_fallbacks(self):
        g = street_geo()
        self.assertEqual(DB.build(g, scene_kind="interior")["ok"], False)
        for key, val, word in (("height_m", 0.61, "机位高"), ("roll_deg", -9.6, "滚转"), ("pitch_deg", 50.0, "俯仰")):
            g2 = dict(g)
            g2["cam"] = dict(g["cam"], **{key: val})
            r = DB.build(g2)
            self.assertFalse(r["ok"])
            self.assertIn(word, r["reason"])
        g3 = dict(g)
        g3["cls"] = np.where(g["cls"] == DB.GROUND, DB.STRUCT, g["cls"])
        self.assertIn("地面像素", DB.build(g3)["reason"])
        self.assertIn("沿用普通粗模", DB.summary_text({"ok": False, "reason": "室内场景"}))

    @unittest.skipUnless((REC / "interior_geo.npz").is_file(), "recorded interior not present")
    def test_recorded_interior_falls_back(self):
        z = np.load(REC / "interior_geo.npz")
        g = {"P": z["P"], "N": z["N"], "mask": z["mask"].astype(bool), "cls": z["cls"], "cam": json.loads(str(z["cam"]))}
        r = DB.build(g)
        self.assertFalse(r["ok"])

    def test_option_off_is_noop(self):
        p = DB.detail_pass(None)
        self.assertIs(p.__enter__(), p)
        self.assertFalse(p.active)
        p.__exit__(None, None, None)


@unittest.skipUnless(BLENDER, "blender not installed")
class BlenderDetailTest(unittest.TestCase):
    rep = None

    @classmethod
    def setUpClass(cls):
        out = Path(tempfile.mkdtemp()) / "detail.json"
        subprocess.run([BLENDER, "-b", "--factory-startup", "--python", str(ROOT / "tools" / "e2e_detail_blockout_bg.py"),
                        "--", "repo", str(ROOT / "tests" / "fixtures" / "scene_blockout" / "street_v4.json"), str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
        cls.rep = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {"error": "no report"}

    def test_ran(self):
        self.assertTrue(self.rep.get("ok"), self.rep.get("error"))

    def test_precise_off_has_no_carrier(self):
        self.assertEqual(self.rep["off"]["carrier_objects"], 0)
        self.assertEqual(self.rep["off"]["objects"], self.rep["baseline"]["objects"])

    def test_carrier_built_hidden(self):
        on = self.rep["on"]
        self.assertGreater(on["carrier_objects"], 0)
        self.assertTrue(on["carrier_hidden"])
        self.assertLessEqual(on["carrier_tris"], 80000)
        self.assertIn("细节粗模", on["note"])
        self.assertEqual(on["llm_objects"], self.rep["baseline"]["objects"])

    def test_option_off_unchanged(self):
        self.assertEqual(self.rep["option_off"]["carrier_objects"], 0)

    def test_geometry_pass(self):
        g = self.rep["passes"]
        self.assertTrue(g["llm_identical_with_and_without_carrier"])
        self.assertFalse(g["carrier_equals_llm"])
        self.assertLess(g["carrier_hole_below_horizon"], 0.002)
        self.assertTrue(g["restored"])

    def test_scene_completion_in_blockout(self):
        cp, cl, off = self.rep["complete_precise"], self.rep["complete_llm"], self.rep["complete_off"]
        self.assertGreater(cp["inferred_objects"], 0, cp["note"])
        self.assertTrue(cp["inferred_visible"])          # part of the visible 场景粗模, not the hidden carrier
        self.assertEqual(cp["inferred_seen"], 0)         # never where the reference camera looks
        self.assertTrue(cp["layout"])
        self.assertIn("场景补全", cp["note"])
        self.assertIn("场景补全", cl["note"])
        self.assertEqual(cl["inferred_seen"], 0)
        self.assertEqual(off["inferred_objects"], 0)
        self.assertNotIn("场景补全", off["note"])

    def test_interior_fallback(self):
        self.assertEqual(self.rep["interior"]["carrier_objects"], 0)
        self.assertIn("室内", self.rep["interior"]["note"])


if __name__ == "__main__":
    unittest.main()
