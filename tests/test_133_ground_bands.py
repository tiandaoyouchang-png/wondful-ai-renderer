"""1.2.8 地面分带 (PC feedback 2026-10-21: 多角度路面不一致 — the 216° view lost the 主视角's curb and bands).

Synthetic 主视角 views (a flat ground seen by a pinhole camera, bands painted across the road) check:
measurement (bands, curb, direction, product shadow, no-band fallback), the band texture fill of another view,
the curb strip geometry, the per-view prompt line merged into the 地面锁定, the ④-analysis band line of the 主视角
request (every image of 张数 2 gets it) and the operator / UI wiring.
"""
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

MV = load_package_module("multiview")
GB = load_package_module("ground_bands")
PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"
GOBI = Path("/workspace/work/scene_blockout_impl/pcfix/gobi_analysis.json")

ANALYSIS = {
    "roads": [{"id": "r", "kind": "road", "label_zh": "土路", "width_m": 9, "points": [{"x": -50, "y": 0}, {"x": 50, "y": 0}],
               "surface": "compacted dirt road with dust and fine gravel"}],
    "terrain": {"surface": "coarse ochre gravel and sandy earth"},
    "palette": {"ground_words": "dry ochre gravel"},
}


def cam_record(pos, yaw_deg, pitch_deg=-12.0, W=640, H=360, hfov=60.0):
    """Camera record (multiview format): yaw 0 = looking +y, pitch < 0 = looking down."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    up = np.cross(right, fwd)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = right, up, -fwd, pos
    tx = math.tan(math.radians(hfov) / 2)
    return {"matrix": M.tolist(), "tx": tx, "ty": tx * H / W, "res": [W, H], "ortho": False}


def paint(x, y, rot=0.0, rng=None, bands=True, gradient=False):
    """Ground colour: across-road offset a (road along x rotated by ``rot``): shoulder (coarse) | curb (dark) |
    dirt (fine). Product at the origin."""
    c, s = math.cos(rot), math.sin(rot)
    a = -s * x + c * y            # across-road offset
    n = rng.normal(0, 1, x.shape)
    if not bands:
        col = np.stack([np.full(x.shape, 180.0), np.full(x.shape, 140.0), np.full(x.shape, 100.0)], -1)
        col *= (1 + 0.04 * n)[..., None]
        if gradient:
            col *= (0.75 + 0.5 * (x - x.min()) / max(1e-6, float(x.max() - x.min())))[..., None]
        return col
    col = np.zeros(x.shape + (3,))
    sh = a < -5.3
    cb = (a >= -5.3) & (a < -4.8)
    mn = a >= -4.8
    col[sh] = np.array([172.0, 128.0, 96.0]) * (1 + 0.22 * n[sh])[..., None]
    col[cb] = np.array([62.0, 47.0, 36.0]) * (1 + 0.05 * n[cb])[..., None]
    col[mn] = np.array([196.0, 150.0, 110.0]) * (1 + 0.05 * n[mn])[..., None]
    return col


def make_view(cam_pos, yaw, rot=0.0, seed=1, W=640, H=360, product=True, shadow=False, **kw):
    rec = cam_record(np.asarray(cam_pos, float), yaw, W=W, H=H)
    d = MV.rays(rec, W, H)
    o = np.asarray(rec["matrix"])[:3, 3]
    dz = d[..., 2]
    t = np.where(dz < -1e-4, -o[2] / np.where(dz < -1e-4, dz, -1), np.inf)
    hit = np.isfinite(t) & (t < 300)
    P = np.zeros((H, W, 4))
    P[..., :3] = np.where(hit[..., None], o + d * np.where(hit, t, 0)[..., None], 0)
    P[..., 3] = hit
    N = np.zeros((H, W, 3))
    N[..., 2] = 1.0
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 150.0)
    col = paint(P[..., 0], P[..., 1], rot, rng, **kw)
    img[hit] = col[hit]
    img[~hit] = (150, 170, 200)
    mask = np.zeros((H, W), np.float32)
    if product:
        inside = hit & (np.abs(P[..., 0]) <= 2.2) & (np.abs(P[..., 1]) <= 1.0)
        mask[inside] = 1.0
        P[inside, 2] = 0.7
        img[inside] = (40, 160, 60)
        if shadow:   # a long dark cast shadow beside the product
            sh = hit & ~inside & (P[..., 0] < -2.2) & (P[..., 0] > -6.5) & (np.abs(P[..., 1]) <= 1.6)
            img[sh] *= 0.45
    v = {"cam": rec, "pos": P.astype(np.float32), "nrm": N.astype(np.float32), "mask": mask,
         "img": np.clip(img, 0, 255).astype(np.float32)}
    v["sky"] = ~hit
    if product:
        v["product_box"] = ([-2.2, -1.0], [2.2, 1.0])
    return v


MASTER = make_view((0, -10, 1.2), 0)


class MeasureTest(unittest.TestCase):
    def test_shoulder_curb_dirt(self):
        info = GB.measure(MASTER, analysis=ANALYSIS)
        self.assertTrue(info)
        kinds = [b["kind"] for b in GB.order_from_camera(info)]
        self.assertEqual(kinds, ["shoulder", "curb", "main"])
        curb = [b for b in info["bands"] if b["kind"] == "curb"][0]
        # the curb sits at across -5.3 … -4.8 (± one bin) in a frame centred on the product
        s_mid = 0.5 * (curb["s0"] + curb["s1"])
        n = np.asarray(info["n"])
        self.assertAlmostEqual(abs(s_mid), 5.05, delta=0.2)
        self.assertLessEqual(curb["width"], 0.8)
        self.assertGreater(curb["height"], 0.0)
        self.assertAlmostEqual(abs(float(np.asarray(info["t"]) @ np.array([1.0, 0.0]))), 1.0, delta=0.01)
        self.assertAlmostEqual(abs(float(n[1])), 1.0, delta=0.01)
        words = [b["zh"] for b in GB.order_from_camera(info)]
        self.assertEqual(words, ["粗砂石路肩", "深色石路沿", "压实土路面"])
        self.assertEqual(info["near_side"], "lo" if float(n[1]) > 0 else "hi")

    def test_rotated_road_direction(self):
        rot = math.radians(25)
        v = make_view((0, -10, 1.2), 0, rot=rot)
        info = GB.measure(v, analysis=ANALYSIS)
        self.assertTrue(info)
        t = np.asarray(info["t"])
        ang = math.degrees(math.acos(min(1.0, abs(float(t @ np.array([math.cos(rot), math.sin(rot)]))))))
        self.assertLess(ang, 3.0)
        self.assertIn("curb", [b["kind"] for b in info["bands"]])

    def test_street_prior_kept_when_close(self):
        info = GB.measure(MASTER, prior_t=[math.cos(math.radians(2)), math.sin(math.radians(2))], analysis=ANALYSIS)
        self.assertEqual(info["direction"], "street")
        info2 = GB.measure(MASTER, prior_t=[0.0, 1.0], analysis=ANALYSIS)     # across the bands: rejected
        self.assertEqual(info2["direction"], "measured")

    def test_no_bands_falls_back(self):
        self.assertEqual(GB.measure(make_view((0, -10, 1.2), 0, bands=False), analysis=ANALYSIS), {})
        # a light gradient across the frame is not banding
        self.assertEqual(GB.measure(make_view((0, -10, 1.2), 0, bands=False, gradient=True), analysis=ANALYSIS), {})
        self.assertEqual(GB.measure({"img": None, "pos": MASTER["pos"]}), {})

    def test_product_shadow_is_not_a_band(self):
        v = make_view((0, -10, 1.2), 0, shadow=True)
        info = GB.measure(v, analysis=ANALYSIS)
        self.assertEqual([b["kind"] for b in GB.order_from_camera(info)], ["shoulder", "curb", "main"])

    def test_grazing_far_band_not_measured(self):
        # a dark strip far behind the product (seen only at grazing angles) is not one of the measured bands
        v = make_view((0, -10, 1.2), 0)
        P = v["pos"]
        far = (P[..., 3] > 0.5) & (P[..., 1] > 9.0) & (P[..., 1] < 10.0)
        v["img"][far] = (40, 30, 25)
        info = GB.measure(v, analysis=ANALYSIS)
        self.assertEqual([b["kind"] for b in GB.order_from_camera(info)], ["shoulder", "curb", "main"])

    def test_band_index_open_edges(self):
        info = GB.measure(MASTER, analysis=ANALYSIS)
        lo, hi = info["bands"][0]["s0"], info["bands"][-1]["s1"]
        bi = GB.band_index(info, np.array([lo - 3.0, hi + 3.0, lo - GB.OPEN_EXTEND_M - 1.0]))
        self.assertEqual(list(bi[:2]), [0, len(info["bands"]) - 1])
        self.assertEqual(int(bi[2]), -1)


class FillAndTextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info = GB.measure(MASTER, analysis=ANALYSIS)
        cls.sy = GB.band_synths(GB.band_patches(MASTER, cls.info))
        cls.other = make_view((0, 12, 1.2), 180, seed=3)      # the opposite side (216°-like): curb behind the product

    def test_patches_per_band(self):
        ps = GB.band_patches(MASTER, self.info)
        self.assertEqual(len(ps), len(self.info["bands"]))
        curb_i = [i for i, b in enumerate(self.info["bands"]) if b["kind"] == "curb"][0]
        self.assertTrue(ps[curb_i]["narrow"])
        for p, b in zip(ps, self.info["bands"]):
            L = float(np.mean(MV.lab(np.asarray(p["patches"][0], np.float32).reshape(-1, 3))[:, 0]))
            self.assertAlmostEqual(L, b["lab"][0], delta=12.0)

    def test_fill_of_other_view_keeps_bands(self):
        fill, have, cls, bi = GB.band_fill(self.other, self.info, self.sy)
        self.assertTrue(have.any())
        L = MV.lab(fill)[..., 0]
        bands = self.info["bands"]
        means = {}
        for i, b in enumerate(bands):
            sel = bi == i
            self.assertTrue(sel.any(), b["kind"])
            means[b["kind"]] = float(L[sel].mean())
            self.assertTrue((cls[sel] == GB.BAND_CLS0 + i).all())
        self.assertLess(means["curb"], means["main"] - 15)
        self.assertLess(means["curb"], means["shoulder"] - 10)
        # the product stays unfilled
        self.assertFalse(have[np.asarray(self.other["mask"]) > 0.5].any())

    def test_fill_keeps_other_classes(self):
        H, W = self.other["mask"].shape
        prior = np.zeros((H, W, 3), np.float32) + 7
        prev = np.zeros((H, W), bool)
        prev[: H // 3] = True                       # e.g. facade fill above the ground
        cls0 = np.full((H, W), 3, np.int64)
        fill, have, cls, bi = GB.band_fill(self.other, self.info, self.sy, prior, prev, cls0)
        keep = (bi < 0) & prev
        self.assertTrue(keep.any())
        self.assertTrue((fill[keep] == 7).all() and (cls[keep] == 3).all())

    def test_seam_apply_fill_with_band_classes(self):
        fill, have, cls, bi = GB.band_fill(self.other, self.info, self.sy)
        H, W = have.shape
        env = np.full((H, W, 3), 120.0, np.float32)
        cover = np.zeros((H, W), np.float32)
        cover[:, : W // 2] = 1.0
        env2, cover2, filled = MV.apply_fill(env, cover, fill, have, cls=cls, seam=True)
        self.assertTrue(filled[:, W // 2 + 20:].any())
        self.assertTrue(np.isfinite(env2).all())

    def test_view_stats_and_sentence(self):
        stats = GB.view_stats(self.other, self.info)
        by = {self.info["bands"][e["i"]]["kind"]: e for e in stats}
        self.assertEqual(by["curb"]["rel"], "behind")
        self.assertEqual(by["shoulder"]["rel"], "behind")
        s = GB.view_sentence(self.info, stats)
        self.assertIn("深色石路沿在产品后方", s)
        zh, en = GB.lock_addition(self.info, stats)
        self.assertIn("粗砂石路肩 + 深色石路沿 + 压实土路面，干燥", zh)
        self.assertIn("这个视角里", zh)
        self.assertIn("略高于两侧", zh)
        self.assertIn("[GROUND BANDS]", en)
        self.assertIn("dark stone curb", en)

    def test_merged_into_ground_lock(self):
        bz = GB.lock_addition(self.info, GB.view_stats(self.other, self.info))
        t = MV.ground_lock_text(ANALYSIS, {"rgb": [180, 140, 100], "hex": "#B48C64"}, bands=bz)
        self.assertEqual(t.count("【地面锁定】"), 1)
        self.assertIn("地面分带（主视角实测", t)
        self.assertIn("[GROUND LOCK]", t)
        self.assertIn("[GROUND BANDS]", t)
        self.assertIn("地面保持干燥", t)
        # no bands → exactly the 1.2.7 line
        self.assertEqual(MV.ground_lock_text(ANALYSIS, {"rgb": [1, 2, 3], "hex": "#010203"}),
                         MV.ground_lock_text(ANALYSIS, {"rgb": [1, 2, 3], "hex": "#010203"}, bands=("", "")))
        # bands alone still give a lock line
        self.assertIn("深色石路沿", MV.ground_lock_text({}, {}, bands=bz))

    def test_fill_note_and_summary(self):
        self.assertEqual(GB.fill_note(0.01), "")
        self.assertIn("地面分带", GB.fill_note(0.2))
        self.assertIn("粗砂石路肩 + 深色石路沿 + 压实土路面", GB.summary(self.info))


class GeometryTest(unittest.TestCase):
    def test_curb_strip_raised_and_long(self):
        info = GB.measure(MASTER, analysis=ANALYSIS)
        cam_xy = [(0, -10), (0, 12), (30, 5)]
        half = GB.geometry_extent(info, cam_xy)
        self.assertGreaterEqual(half, 30 + GB.GEOM_EXTENT_M[0] - 1)
        curb = [b for b in info["bands"] if b["kind"] == "curb"][0]
        v, f = GB.strip_mesh(info, curb, half)
        V = np.asarray(v)
        tt, s = GB.frame(info, V[:, :2])
        self.assertAlmostEqual(float(tt.min()), -half, delta=GB.GEOM_STEP_M)
        self.assertAlmostEqual(float(tt.max()), half, delta=GB.GEOM_STEP_M)
        top = V[:, 2].max() - GB.plane_z(info, 0.0, 0.5 * (curb["s0"] + curb["s1"]))
        self.assertAlmostEqual(float(top), curb["height"], delta=0.03)
        self.assertLess(float(V[:, 2].min()), 0.0)            # side faces reach below the ground
        self.assertGreater(len(f), 3 * (len(V) // 4 - 1))     # top + 2 sides per segment (+ caps)
        main = [b for b in info["bands"] if b["kind"] == "main"][0]
        v2, f2 = GB.strip_mesh(info, main, half)
        z2 = np.asarray(v2)[:, 2] - GB.plane_z(info, *GB.frame(info, np.asarray(v2)[:, :2]))
        self.assertTrue(np.allclose(z2, GB.GEOM_LIFT, atol=1e-6))
        self.assertEqual(len(f2), len(v2) // 2 - 1)

    def test_only_raised_bands_become_geometry(self):
        info = GB.measure(MASTER, analysis=ANALYSIS)
        self.assertEqual([GB.raised(b) for b in info["bands"]], [b["kind"] == "curb" for b in info["bands"]])

    def test_build_geometry_names_hidden_from_block_list(self):
        src = (PKG / "ground_bands.py").read_text(encoding="utf-8")
        self.assertIn('name = f"_wondful_ground_band_', src)     # massing_objects skips "_wondful_" names
        self.assertIn('ob["wondful_mv_aux"] = 1', src)


class AnalysisBandsTest(unittest.TestCase):
    def test_gobi_analysis_curb_between_dirt(self):
        an = json.loads(GOBI.read_text(encoding="utf-8")) if GOBI.is_file() else {
            "roads": [{"kind": "road", "width_m": 7.8, "points": [{"x": -65, "y": 2.1}, {"x": 65, "y": 2.1}],
                       "surface": "dark road surfacing coated with ochre dust and fine gravel"}],
            "surface_zones": [{"id": "f", "kind": "sand", "points": [{"x": -45, "y": 6.27}, {"x": 45, "y": 6.27},
                                                                      {"x": 45, "y": 22}, {"x": -45, "y": 22}],
                               "surface": "warm ochre compacted sand"}],
            "primitives": [{"id": "roadside_curb", "label_zh": "低矮路缘", "x": 0, "y": 6.15, "w": 70, "d": 0.22,
                            "h": 0.13, "material": "weathered stone curb, dark vertical face"}],
            "product_spot": {"x": 0, "y": 11}}
        info = GB.analysis_bands(an)
        kinds = [b["kind"] for b in info["bands"]]
        self.assertIn("curb", kinds)
        i = kinds.index("curb")
        self.assertTrue(0 < i < len(kinds) - 1)
        self.assertEqual(info["bands"][i]["zh"], "深色石路沿")
        self.assertEqual(kinds[i + 1], "main")                     # the product stands behind the curb
        line = GB.analysis_bands_text(an)
        self.assertTrue(line.startswith("【地面分带】"))
        self.assertIn("来自环境参考图分析", line)
        self.assertEqual(line, GB.analysis_bands_text(an))           # deterministic: v01 and v02 get the same line

    def test_no_ground_info(self):
        self.assertEqual(GB.analysis_bands({}), {})
        self.assertEqual(GB.analysis_bands_text({"roads": []}), "")
        one = {"roads": [{"kind": "road", "width_m": 6, "points": [{"x": -9, "y": 3}, {"x": 9, "y": 3}],
                          "surface": "asphalt"}]}
        self.assertEqual(GB.analysis_bands_text(one), "")            # one band is no banding

    def test_material_grain_preference(self):
        self.assertEqual(GB._material("dust and fine gravel", grain="fine")[0], "土")
        self.assertEqual(GB._material("dust and fine gravel", grain="coarse")[0], "砂石")
        self.assertEqual(GB._material("old asphalt with dust", grain="fine")[0], "沥青")


class WiringTest(unittest.TestCase):
    def test_operator_wiring(self):
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        # 多角度: measured after the master pass, built into the blockout, removed with the other aux geometry
        self.assertIn("GB.measure_master(master_img, pool, pre, rec", ops)
        self.assertIn("got = GB.build_geometry(context, massing_coll, bands_info, cam_xy)", ops)
        self.assertIn("made.extend(got)", ops)
        self.assertIn("GB.band_synths(GB.band_patches(master_view, bands_info))", ops)
        self.assertIn("GB.band_fill(vk, bands_info, band_sy, fimg, have, cmap)", ops)
        self.assertIn("bands=GB.lock_addition(bands_info, bstats, dry=band_dry)", ops)
        # the 主视角 request: the analysis band line in the shared prompt and the simple-path style text
        self.assertIn("render_prompt = render_prompt.rstrip() + \"\\n\" + sb_band_line", ops)
        self.assertIn("direct_style_text = (direct_style_text + \"\\n\" + sb_band_line).strip()", ops)
        self.assertLess(ops.index("sb_band_line = _sb_ground_band_line(context, props)"),
                        ops.index("render_prompt += bundle_note"))

    def test_property_and_ui(self):
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        self.assertIn("sb_ground_bands: BoolProperty(name=\"地面分带\", default=True", props)
        self.assertIn('row.prop(props, "sb_ground_bands", text="地面分带")', ui)


if __name__ == "__main__":
    unittest.main()
