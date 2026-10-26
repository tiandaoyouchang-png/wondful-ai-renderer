"""1.2.5 场景补全 (④ 场景粗模, default on) + 多角度 propagation fixes.

User 2026-10-20: 「要按照当前能看到的场景推测看不到的场景里有什么 把场景模型补全」 — the blockout itself continues the
street the reference photo shows (rows of houses, curbs, hedges, poles) where the reference camera cannot see;
多角度场景一致 fills uncovered pixels with style patches rectified from the 主视角, limits AI-from-AI re-projection
to one hop, refuses stretched source pixels, and judges the environment from covis 0.4.
"""
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "wondful_ai_renderer"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PKG / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DB = _load("detail_blockout")
SC = _load("scene_complete")
import precise_synth  # noqa: E402
from test_316_regressions import load_package_module  # noqa: E402

MV = load_package_module("multiview")
REC = Path("/workspace/work/precise_blockout")


def street_geo(both=True):
    """Camera looking down a street (+Y): facades at x = ±9 from y = 14 on (one side only with both=False)."""
    cam = {"height_m": 1.6, "pitch_deg": 0.0, "roll_deg": 0.0, "hfov_deg": 70.0}
    boxes = [(9, 20, 14, 60, 0, 8, DB.STRUCT)]
    if both:
        boxes.append((-20, -9, 14, 60, 0, 7, DB.STRUCT))
    g = precise_synth.geo(cam, W=480, H=270, boxes=boxes)
    # precise_synth gives every box pixel the −Y normal; the street facades here are the x = ±9 side faces
    P, N, C = g["P"], g["N"], g["cls"]
    for (x0, x1, y0, y1, z0, z1, k) in boxes:
        on = (C == k) & (P[..., 0] >= x0 - 0.01) & (P[..., 0] <= x1 + 0.01) & (P[..., 1] >= y0 - 0.01)
        N[on & (np.abs(P[..., 0] - x0) < 0.02)] = (-1, 0, 0)
        N[on & (np.abs(P[..., 0] - x1) < 0.02)] = (1, 0, 0)
    return g


def analysis():
    return {"roads": [{"kind": "road", "width_m": 10,
                       "points": [{"x": 0, "y": 0}, {"x": 0, "y": 40}, {"x": 0, "y": 120}]}],
            "primitives": [{"type": "building", "x": 12, "y": 30, "w": 8, "d": 10}]}


def closed(V, F):
    Vw, Fw = DB.weld(V, F)
    return len(DB.open_edges(Fw)) == 0


class Completion(unittest.TestCase):
    PAD = (0.0, 8.0, 2.5, 0.0)

    def build(self, geo=None, **kw):
        return DB.build(geo or street_geo(), pad=self.PAD, analysis=kw.pop("an", analysis()), **kw)

    def test_rows_continue_behind_camera_both_sides(self):
        r = self.build()
        c = r["stats"]["complete"]
        self.assertTrue(c["ok"], c)
        boxes = [b for b in r["layout"]["boxes"] if b["kind"] == "building"]
        self.assertGreaterEqual(len(boxes), 8)
        behind = [b for b in boxes if b["c"][1] < 0]
        self.assertTrue(any(b["c"][0] > 0 for b in behind) and any(b["c"][0] < 0 for b in behind))
        # setback follows the measured facades (x = ±9), not a default
        self.assertTrue(all(8.0 <= abs(b["c"][0]) <= 10.5 for b in behind), [b["c"] for b in behind])
        self.assertGreater(c["curbs"], 0)
        self.assertEqual(c["mirrored"], [])
        self.assertEqual(r["layout"]["street"]["source"], "analysis")

    def test_inferred_parts_closed_and_tagged(self):
        r = self.build()
        V, F, C = r["V"], r["F"], r["C"]
        inf = np.isin(C, DB.C_INFERRED)
        self.assertTrue(inf.any())
        for c in np.unique(C[inf]):
            self.assertTrue(closed(V, F[C == c]), DB.C_NAME[int(c)])
        self.assertTrue(all(p.get("inferred") for p in r["parts"] if str(p.get("kind", "")).startswith("inferred")))
        confs = [p["conf"] for p in r["parts"] if p.get("inferred")]
        self.assertTrue(confs and max(confs) < 1.0)

    def test_nothing_inferred_where_the_reference_camera_sees(self):
        g = street_geo()
        r = self.build(g)
        mv = SC.MasterView(g)
        V, F, C = r["V"], r["F"], r["C"]
        inf = np.unique(F[np.isin(C, DB.C_INFERRED)])
        self.assertEqual(int(mv.visible(V[inf], margin=0.0).sum()), 0)

    def test_clear_circle_around_product(self):
        r = self.build()
        cr = r["layout"]["clear_r"]
        self.assertGreaterEqual(cr, math.hypot(0, 8) + SC.CLEAR_EXTRA - 1e-6)
        for b in r["layout"]["boxes"]:
            fp = SC._footprint_pts(np.array(b["c"]), np.array(b["u"]), np.array(b["n"]), b["a"][1] * 2, -b["b"][0])
            self.assertGreaterEqual(float(np.min(np.hypot(fp[:, 0] - 0.0, fp[:, 1] - 8.0))), cr - 1e-6)

    def test_budget_and_determinism(self):
        a = self.build()
        b = self.build()
        self.assertLessEqual(a["stats"]["tris"], DB.TRI_BUDGET)
        self.assertEqual(json.dumps(a["layout"], sort_keys=True), json.dumps(b["layout"], sort_keys=True))
        # the inferred part honours its own budget (a part is only started with ≥ 60 tris of room)
        parts = [p for p in a["parts"] if not p.get("inferred")]
        g = street_geo()
        _V, _F, T = DB.terrain(g["P"], g["mask"], g["cls"], 70.0)
        for left in (300, 1500):
            M = DB.Mesh()
            st, _lay = SC.complete(DB._Prims(), M, parts, lambda x, y: DB.terrain_z(T, x, y), SC.MasterView(g).visible,
                                   pad=self.PAD, analysis=analysis(), budget_left=left)
            self.assertLessEqual(M.tris(), left + 150)

    def test_one_side_measured_is_mirrored(self):
        r = self.build(street_geo(both=False))
        c = r["stats"]["complete"]
        self.assertTrue(c["ok"])
        self.assertEqual(c["mirrored"], ["左"])
        left = [b for b in r["layout"]["boxes"] if b["kind"] == "building" and b["c"][0] < 0]
        self.assertTrue(left and all(b["conf"] == SC.CONF_MIRROR for b in left))

    def test_no_buildings_no_rows(self):
        cam = {"height_m": 1.6, "pitch_deg": 0.0, "roll_deg": 0.0, "hfov_deg": 70.0}
        r = DB.build(precise_synth.geo(cam, W=320, H=180), pad=self.PAD, analysis=None)
        self.assertFalse(r["stats"]["complete"]["ok"])
        self.assertFalse(np.isin(r["C"], DB.C_INFERRED).any())

    def test_off_builds_nothing_inferred(self):
        r = self.build(complete=False)
        self.assertNotIn("complete", r["stats"])
        self.assertIsNone(r["layout"])
        self.assertFalse(np.isin(r["C"], DB.C_INFERRED).any())

    def test_obstacles_respected(self):
        base = self.build()
        b0 = [b for b in base["layout"]["boxes"] if b["kind"] == "building" and b["c"][1] < 0][0]
        obst = [(np.array(b0["c"]), np.array(b0["u"]), np.array(b0["n"]), b0["a"][1] * 2, -b0["b"][0])]
        r = self.build(obstacles=obst)
        for b in r["layout"]["boxes"]:
            fp = SC._footprint_pts(np.array(b["c"]), np.array(b["u"]), np.array(b["n"]), b["a"][1] * 2, -b["b"][0])
            self.assertFalse(SC._in_rects(fp, obst, pad=-0.05), b)

    def test_layout_to_world(self):
        r = self.build()
        a = math.radians(30)
        Wm = [[math.cos(a), -math.sin(a), 0, 5], [math.sin(a), math.cos(a), 0, -2], [0, 0, 1, 0.5], [0, 0, 0, 1]]
        lw = SC.layout_to_world(r["layout"], Wm)
        b, bw = r["layout"]["boxes"][0], lw["boxes"][0]
        x, y = b["c"]
        self.assertAlmostEqual(bw["c"][0], math.cos(a) * x - math.sin(a) * y + 5, places=2)
        self.assertAlmostEqual(bw["z"][0], b["z"][0] + 0.5, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(bw["u"])), 1.0, places=4)
        json.dumps(lw)

    def test_describe_and_summary(self):
        c = self.build()["stats"]["complete"]
        self.assertIn("房屋", SC.describe(c))
        self.assertIn("场景补全", DB.completion_text(c))
        self.assertIn("场景补全", DB.summary_text(self.build()))

    @unittest.skipUnless((REC / "street_geo_plugin.npz").is_file(), "recorded street measurement not available")
    def test_recorded_street(self):
        d = np.load(REC / "street_geo_plugin.npz", allow_pickle=True)
        geo = {k: d[k] for k in d.files}
        geo["cam"] = json.loads(str(d["cam"]))
        an_p = REC / "runs/street_p1b/session/scene_analysis_r1.json"
        an = json.loads(an_p.read_text(encoding="utf-8")) if an_p.is_file() else None
        r = DB.build(geo, pad=(0.6, 7.8, 3.4, 0.086), z_offset=-0.014, analysis=an)
        c = r["stats"]["complete"]
        self.assertTrue(c["ok"])
        self.assertGreaterEqual(c["buildings"], 10)
        self.assertLessEqual(r["stats"]["tris"], DB.TRI_BUDGET)
        behind = [b for b in r["layout"]["boxes"] if b["kind"] == "building" and b["c"][1] < 0]
        self.assertTrue(any(b["c"][0] > 2 for b in behind) and any(b["c"][0] < 2 for b in behind))


# ------------------------------------------------------------------ multiview side
W, H = 160, 90


def look_at(pos, target):
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = target - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = r, u, -f, pos
    return M


def cam(name, pos, target=(0, 0, 0.5), fov=60):
    return MV.camera_record(name, look_at(pos, target), math.radians(fov), "AUTO", W, H)


def ground_view(rec, master=False, img=None):
    D = MV.rays(rec, W, H)
    O = np.asarray(rec["matrix"])[:3, 3]
    t = np.where(D[..., 2] < -1e-6, -O[2] / np.minimum(D[..., 2], -1e-6), np.inf)
    t = np.where(t < 200, t, np.inf)
    hit = np.isfinite(t)
    P = O + D * np.where(hit, t, 0)[..., None]
    pos = np.dstack([P, hit.astype(np.float32)]).astype(np.float32)
    nrm = np.zeros((H, W, 3), np.float32)
    nrm[..., 2] = 1
    v = {"name": rec["name"], "cam": rec, "pos": pos, "nrm": nrm, "mask": np.zeros((H, W), np.float32), "master": master}
    if img is not None:
        v["img"] = img
        v["sky"] = ~hit
    return v


class Propagation(unittest.TestCase):
    def test_constants(self):
        self.assertAlmostEqual(MV.ENV_MIN_COVIS, 0.4)
        self.assertEqual(MV.MAX_SOURCE_HOP, 1)
        self.assertGreater(MV.MASTER_BIAS, 0.1)
        rp = {"reproj_dE": 26.1, "reproj_dE_geom": 26.1, "covis": 0.443}
        self.assertTrue(MV.env_violations(rp, True))          # the 2026-10-20 216° view now retries

    def test_hop(self):
        srcs = [{"name": "m", "master": True, "hop": 0}, {"name": "a", "hop": 1}, {"name": "b", "hop": 2}]
        self.assertEqual(MV.view_hop({"share": {"m": 0.6, "a": 0.05}}, srcs), 1)
        self.assertEqual(MV.view_hop({"share": {"m": 0.04, "a": 0.64}}, srcs), 2)
        self.assertTrue(MV.usable_source({"master": True}))
        self.assertTrue(MV.usable_source({"hop": 1}))
        self.assertFalse(MV.usable_source({"hop": 2}))
        self.assertTrue(MV.usable_source({}))               # old pool entries: hop 1

    def test_stretch_clamp(self):
        # source camera low and far away sees the near ground at a grazing angle → smeared → refused
        far = ground_view(cam("far", (0, -60, 0.4), target=(0, 0, 0)), img=np.full((H, W, 3), 200, np.float32))
        far["hop"] = 1
        tgt = ground_view(cam("t", (0, -3, 6), target=(0, 2, 0)))
        clay = np.zeros((H, W, 3), np.float32)
        _e, _c, src, st = MV.build_base(tgt, [far], clay)
        self.assertLess(st["coverage_geometry"], 0.05)
        near = ground_view(cam("near", (1, -3, 5), target=(0, 2, 0)), img=np.full((H, W, 3), 200, np.float32))
        near["hop"] = 1
        _e, _c, src, st2 = MV.build_base(tgt, [near], clay)
        self.assertGreater(st2["coverage_geometry"], 0.5)

    def test_style_fill(self):
        rec = cam("m", (0, -8, 1.7), target=(0, 10, 1.0), fov=70)
        tex = np.zeros((H, W, 3), np.float32)
        tex[..., 0] = 90
        tex[..., 1] = 60
        mv = ground_view(rec, master=True, img=tex)
        layout = {"street": {"a0": [0, 0], "t": [0, 1], "half_width": 4.0}, "boxes": [], "facades": [],
                  "veg_refs": [], "ground_z": 0.0}
        patches = MV.master_patches(mv, layout)
        self.assertIsNotNone(patches["road"])
        self.assertLess(float(np.abs(patches["road"][..., 0] - 90).max()), 2.0)
        tgt = ground_view(cam("b", (0, 16, 1.7), target=(0, 0, 1.0), fov=70))
        layout["boxes"].append({"kind": "building", "c": [6, -4], "u": [0, 1], "n": [-1, 0], "a": [-5, 5],
                                "b": [-8, 0], "z": [-0.5, 7], "style": 3})
        cls, sty = MV.class_map(tgt, layout)
        self.assertTrue((cls == MV.CLS_ROAD).any())
        img, have = MV.style_fill(tgt, cls, sty, patches, layout)
        self.assertTrue(have.any())
        env, cover, filled = MV.apply_fill(np.zeros((H, W, 3), np.float32), np.zeros((H, W), np.float32), img, have)
        self.assertGreater(float(cover.max()), 0.8)
        note = MV.fill_note(cls, filled, "两侧继续是与主视角同类型的房屋")
        self.assertIn("不要画成空白墙面", note)
        self.assertEqual(MV.fill_note(cls, np.zeros_like(filled)), "")
        self.assertIn(note, MV.view_text("", "", ["主视角"], fill_line=note))

    def test_facade_tangent_ignores_smoothed_normals(self):
        # a wall x = 3 seen along a row; normals smoothed / tilted like a Blender smooth-shaded box corner
        Hh, Ww = 4, 20
        P = np.zeros((Hh, Ww, 4))
        P[..., 0] = 3.0
        P[..., 1] = np.linspace(0, 10, Ww)[None]
        P[..., 2] = 1.0
        P[..., 3] = 1
        N = np.zeros((Hh, Ww, 3))
        N[..., 0] = -np.cos(np.linspace(-0.8, 0.8, Ww))[None]
        N[..., 1] = np.sin(np.linspace(-0.8, 0.8, Ww))[None]
        t = MV.facade_tangent(P, N)
        self.assertLess(float(np.abs(np.abs(t[:, 1:-1, 1]) - 1).max()), 1e-6)

    def test_crop_patch_fills_holes_without_streaks(self):
        col = np.zeros((40, 40, 3), np.float32)
        col[:, :20] = 100
        col[:, 20:] = 200
        ok = np.ones((40, 40), bool)
        ok[10:14, 5:9] = False
        col[~ok] = 0
        C = MV._crop_patch(col, ok, min_px=10)
        self.assertGreater(float(C[10:14, 5:9].min()), 90)
        self.assertIsNone(MV._crop_patch(col, np.zeros_like(ok), min_px=10))

    def test_in_box_and_classes(self):
        b = {"kind": "hedge", "c": [0, 0], "u": [1, 0], "n": [0, -1], "a": [-1, 1], "b": [-1, 0], "z": [0, 1]}
        P = np.array([[0.0, 0.5, 0.5], [3.0, 0.5, 0.5]])
        self.assertEqual(MV.in_box(P, b).tolist(), [True, False])
        self.assertEqual(MV.KIND_CLASS["hedge"], MV.CLS_VEG)


class Wiring(unittest.TestCase):
    def test_property_ui_operator(self):
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        i = props.index("sb_complete: BoolProperty(")
        # Completing unseen geometry is deliberately opt-in: it can add inferred
        # scene geometry, so an existing scene must not change on upgrade.
        self.assertIn("default=False", props[i:i + 120])
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        self.assertIn('extra.prop(props, "sb_complete", text="场景补全", toggle=True)', ui)
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        self.assertIn('complete = bool(getattr(props, "sb_complete", False))', ops)
        self.assertIn("complete_llm_in_blender", ops)
        self.assertIn("fillers=not has_inferred", ops)
        self.assertIn("MV.usable_source", ops)
        self.assertIn('"hop": int(st.get("hop", 1))', ops)
        src = (PKG / "detail_blockout.py").read_text(encoding="utf-8")
        self.assertIn("hidden=False", src)            # the completed street is part of the visible 场景粗模


if __name__ == "__main__":
    unittest.main()
