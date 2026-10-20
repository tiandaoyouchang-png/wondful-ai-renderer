"""1.2.5 场景补全 · three ideas from Building Rome (arXiv 2610.08790, CC BY-NC — idea only, no code / weights).

User OK 2026-10-21 「可以的」: 近密远疏 distance bands, 空 / 可见 / 被挡住 three-state space (nothing grows in free
space, rule depths cut back), 重叠衔接 overlap band between the style fill and the projected photo (+ inferred lots
taking the neighbouring measured eaves / setback). 近密远疏 / 重叠衔接 default on; 空间三态 default off (216° 错位 ΔE got worse in the orbit test).
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_127_scene_complete as T127  # noqa: E402

DB, SC, MV, PKG = T127.DB, T127.SC, T127.MV, T127.PKG
REC = Path("/workspace/work/precise_blockout")


class ThreeStates(unittest.TestCase):
    def setUp(self):
        self.g = T127.street_geo()
        self.occ = SC.Occupancy(self.g)

    def test_states(self):
        # on the street axis in front of the facades: free (the camera sees the ground beyond)
        Q = np.array([[0.0, 30.0, 1.0],      # middle of the street, in view → free
                      [9.0, 30.0, 3.0],      # on the measured facade (x = 9) → visible
                      [14.0, 30.0, 3.0],     # inside the right block, behind its facade → hidden
                      [0.0, -20.0, 1.0]])    # behind the camera → unknown
        s = self.occ.state(Q)
        self.assertEqual(list(s), [SC.S_FREE, SC.S_VISIBLE, SC.S_HIDDEN, SC.S_UNKNOWN])

    def test_clip_depth_cuts_rule_depth_out_of_free_space(self):
        # box(c, u, n, …) extends from its front c along −n. Right block: facade x = 9, block toward +x.
        c, u = np.array([9.0, 30.0]), np.array([0.0, 1.0])
        d_bad = self.occ.clip_depth(c, u, np.array([1.0, 0.0]), 6.0, 12.0, 0.0, 6.0)   # grows into the street → cut
        self.assertLess(d_bad, 12.0)
        d_ok = self.occ.clip_depth(c, u, np.array([-1.0, 0.0]), 6.0, 8.0, 0.0, 6.0)   # into the block: hidden
        self.assertAlmostEqual(d_ok, 8.0, places=3)

    def test_build_inferred_never_in_free_space_and_switch(self):
        r = DB.build(self.g, pad=(0.0, 8.0, 2.5, 0.0), analysis=T127.analysis(), free_space=True)
        V, F, C = r["V"], r["F"], r["C"]
        inf = np.unique(F[np.isin(C, DB.C_INFERRED)])
        self.assertTrue(len(inf))
        self.assertEqual(int((self.occ.state(V[inf]) == SC.S_FREE).sum()), 0)
        self.assertTrue(r["stats"]["free_space"])
        off = DB.build(self.g, pad=(0.0, 8.0, 2.5, 0.0), analysis=T127.analysis(), free_space=False)
        self.assertFalse(off["stats"]["free_space"])
        self.assertEqual(off["stats"]["carved"], 0)

    def test_recorded_street_less_free_area(self):
        d = np.load(REC / "street_geo_plugin.npz", allow_pickle=True)
        geo = {k: d[k] for k in d.files}
        geo["cam"] = json.loads(str(d["cam"]))
        an_p = REC / "runs/street_p1b/session/scene_analysis_r1.json"
        an = json.loads(an_p.read_text(encoding="utf-8")) if an_p.is_file() else None
        occ = SC.Occupancy(geo, -0.014)

        def free_area(r):
            V, F, C = r["V"], r["F"], r["C"]
            sel = np.isin(C, (DB.C_BUILDING, DB.C_PROP, DB.C_WALL, DB.C_TREE))
            T = V[F[sel]]
            cen = T.mean(1)
            a = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
            return float(a[occ.free(cen)].sum())
        kw = dict(pad=(0.6, 7.8, 3.4, 0.086), z_offset=-0.014, analysis=an)
        new = DB.build(geo, free_space=True, **kw)
        old = DB.build(geo, free_space=False, bands=False, seam=False, **kw)
        self.assertLess(free_area(new), 0.7 * free_area(old))
        self.assertLessEqual(new["stats"]["tris"], DB.TRI_BUDGET)
        self.assertGreater(new["stats"]["carved"], 0)


class Bands(unittest.TestCase):
    def test_far_lots_merged_and_windows_near_first(self):
        g = T127.street_geo()
        kw = dict(pad=(0.0, 8.0, 2.5, 0.0), analysis=T127.analysis())
        on = DB.build(g, **kw)
        off = DB.build(g, bands=False, **kw)
        def lots(r, far):
            return [b for b in r["layout"]["boxes"] if b["kind"] == "building"
                    and (abs(b["c"][1] - 8.0) > SC.BAND_MID) == far]
        w_on = np.mean([b["a"][1] * 2 for b in lots(on, True)])
        w_off = np.mean([b["a"][1] * 2 for b in lots(off, True)])
        self.assertGreater(w_on, 1.3 * w_off)
        # no inferred window beyond BAND_MID from the product
        V, F, C = on["V"], on["F"], on["C"]
        win = F[C == SC.C_INF_WINDOW]
        if len(win):
            cen = V[win].mean(1)
            self.assertLessEqual(float(np.hypot(cen[:, 0], cen[:, 1] - 8.0).max()), SC.BAND_MID + 20.0)
        self.assertTrue(on["stats"]["bands"])
        self.assertFalse(off["stats"]["bands"])


class Seam(unittest.TestCase):
    def test_overlap_band_softens_the_step(self):
        H, W = 40, 80
        env = np.full((H, W, 3), 160.0, np.float32)
        cover = np.zeros((H, W), np.float32)
        cover[:, :40] = 1.0
        fill = np.full((H, W, 3), 100.0, np.float32)
        have = np.ones((H, W), bool)
        cls = np.full((H, W), MV.CLS_FACADE, np.int8)
        a, _c, fa = MV.apply_fill(env, cover, fill, have)
        b, _c2, fb = MV.apply_fill(env, cover, fill, have, cls=cls, seam=True)
        step_a = abs(float(a[20, 38, 0] - a[20, 41, 0]))
        step_b = abs(float(b[20, 38, 0] - b[20, 41, 0]))
        self.assertLess(step_b, 0.5 * step_a)
        # fill pulled toward the photo's colour (gain), far side still the fill's texture range
        self.assertGreater(float(b[20, 75, 0]), float(a[20, 75, 0]))
        self.assertTrue((fa == fb).all())
        # off = identical to the old behaviour
        c, _x, _y = MV.apply_fill(env, cover, fill, have, cls=cls, seam=False)
        self.assertTrue(np.allclose(a, c))

    def test_seam_blend_takes_measured_eaves(self):
        g = T127.street_geo()
        r = DB.build(g, pad=(0.0, 8.0, 2.5, 0.0), analysis=T127.analysis())
        self.assertGreater(r["stats"]["complete"]["seam_blended"], 0)
        off = DB.build(g, pad=(0.0, 8.0, 2.5, 0.0), analysis=T127.analysis(), seam=False)
        self.assertEqual(off["stats"]["complete"]["seam_blended"], 0)


class Wiring(unittest.TestCase):
    def test_props_ui_operator(self):
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        for k, on in (("sb_free_space", False), ("sb_lod_bands", True), ("sb_seam_blend", True)):
            i = props.index(f"{k}: BoolProperty(")
            self.assertIn(f"default={on}", props[i:i + 80])
        # 空间三态 stays off by default (orbit test 2026-10-21: 216° 错位 ΔE worse); the build default agrees
        import inspect
        self.assertFalse(inspect.signature(DB.build).parameters["free_space"].default)
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        for k, t in (("sb_free_space", "空间三态"), ("sb_lod_bands", "近密远疏"), ("sb_seam_blend", "接缝过渡")):
            self.assertIn(f'row.prop(props, "{k}", text="{t}")', ui)
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        self.assertIn('free_space=bool(getattr(props, "sb_free_space", False))', ops)
        self.assertIn('seam_on = bool(getattr(props, "sb_seam_blend", True))', ops)
        self.assertIn("seam=seam_on", ops)


if __name__ == "__main__":
    unittest.main()
