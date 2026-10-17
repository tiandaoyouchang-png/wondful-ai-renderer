"""1.2.3 多角度一致: projection, retry policy, light sentence, ordering, wiring."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

MV = load_package_module("multiview")
PKG = Path(__file__).resolve().parents[1] / "wondful_ai_renderer"
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


def texture(P):
    """World colour: smooth pattern on the ground and a wall at y = 6."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    return np.stack([128 + 100 * np.sin(x * 1.3), 128 + 100 * np.cos(y * 0.9), 128 + 80 * np.sin(z * 2 + x)], -1)


def trace(rec):
    """Analytic position / normal passes: ground z = 0, wall y = 6 (x in -8..8), product box at origin."""
    D = MV.rays(rec, W, H)
    O = np.asarray(rec["matrix"])[:3, 3]
    t_g = np.where(D[..., 2] < -1e-6, -O[2] / np.minimum(D[..., 2], -1e-6), np.inf)
    t_w = np.where(D[..., 1] > 1e-6, (6.0 - O[1]) / np.maximum(D[..., 1], 1e-6), np.inf)
    Pw = O + D * t_w[..., None]
    t_w = np.where((np.abs(Pw[..., 0]) < 8) & (Pw[..., 2] < 4) & (Pw[..., 2] > 0), t_w, np.inf)
    t = np.minimum(t_g, t_w)
    t = np.where(t < 60, t, np.inf)
    hit = np.isfinite(t)
    P = O + D * np.where(hit, t, 0)[..., None]
    nrm = np.where((t_w <= t_g)[..., None], np.array([0, -1.0, 0]), np.array([0, 0, 1.0]))
    pos = np.dstack([P, hit.astype(np.float32)]).astype(np.float32)
    # product: everything within 0.6 of the origin on the ground is "product"
    mask = (hit & (np.hypot(P[..., 0], P[..., 1]) < 0.6) & (P[..., 2] < 1.0)).astype(np.float32)
    return pos, nrm.astype(np.float32), mask


def view(rec, master=False, with_image=False):
    pos, nrm, mask = trace(rec)
    v = {"name": rec["name"], "cam": rec, "pos": pos, "nrm": nrm, "mask": mask, "master": master}
    if with_image:
        img = np.where(pos[..., 3:4] > 0.5, texture(pos[..., :3]), np.array([110, 160, 230.0]))
        v["img"] = img.astype(np.float32)
        v["sky"] = pos[..., 3] < 0.5
    return v


class Projection(unittest.TestCase):
    def test_project_inverts_rays(self):
        rec = cam("a", (3, -6, 2))
        D = MV.rays(rec, W, H)
        O = np.asarray(rec["matrix"])[:3, 3]
        u, v, z = MV.project(rec, O + D * 5.0, W, H)
        uu, vv = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
        self.assertLess(float(np.abs(u - uu).max()), 1e-6)
        self.assertLess(float(np.abs(v - vv).max()), 1e-6)
        self.assertTrue((z > 0).all())

    def test_portrait_fit_uses_vertical_angle(self):
        rec = MV.camera_record("p", np.eye(4), math.radians(50), "AUTO", 90, 160)
        self.assertAlmostEqual(rec["ty"], math.tan(math.radians(25)))
        self.assertAlmostEqual(rec["tx"], rec["ty"] * 90 / 160)

    def test_master_texels_land_on_the_same_world_points(self):
        master = view(cam("master", (0, -7, 1.6)), master=True, with_image=True)
        vk = view(cam("orbit", (4.5, -5.5, 1.6)))
        clay = np.full((H, W, 3), 128.0, np.float32)
        env, cover, srcmap, st = MV.build_base(vk, [master], clay)
        truth = texture(vk["pos"][..., :3])
        geo = (srcmap == 1)
        self.assertGreater(geo.mean(), 0.3)
        err = np.abs(env[geo] - truth[geo]).mean()
        self.assertLess(err, 6.0)                      # bilinear sampling only
        self.assertGreater(st["coverage"], 0.5)
        # product pixels are never painted from a source
        self.assertEqual(int((cover[vk["mask"] > 0.5] > 0.01).sum()), 0)

    def test_second_source_fills_what_the_master_never_saw(self):
        master = view(cam("master", (0, -7, 1.6)), master=True, with_image=True)
        side = view(cam("side", (7, 0.5, 1.6)), with_image=True)
        vk = view(cam("far", (6, 4, 1.8)))
        clay = np.full((H, W, 3), 128.0, np.float32)
        _e1, _c1, _s1, only_master = MV.build_base(vk, [master], clay)
        _e2, _c2, srcmap, both = MV.build_base(vk, [master, side], clay)
        self.assertGreater(both["coverage_geometry"], only_master["coverage_geometry"])
        self.assertGreater(both["share"]["side"], 0.0)

    def test_reprojection_of_a_consistent_image_is_near_zero(self):
        master = view(cam("master", (0, -7, 1.6)), master=True, with_image=True)
        vk = view(cam("orbit", (4.5, -5.5, 1.6)), with_image=True)
        rp = MV.reproj_de(master, vk, vk["img"])
        self.assertIsNotNone(rp["reproj_dE_geom"])
        self.assertLess(rp["reproj_dE_geom"], 3.0)
        self.assertGreater(rp["covis"], 0.3)
        shifted = vk["img"] * 0.6
        self.assertGreater(MV.reproj_de(master, vk, shifted)["reproj_dE_geom"], 10.0)

    def test_compose_keeps_product_and_uncovered_clay(self):
        clay_full = np.full((H * 2, W * 2, 3), 50.0, np.float32)
        env = np.full((H, W, 3), 200.0, np.float32)
        cover = np.ones((H, W), np.float32)
        cover[:, : W // 4] = 0
        mask = np.zeros((H * 2, W * 2), np.float32)
        mask[60:100, 100:200] = 1
        out = MV.compose_canvas(env, cover, clay_full, mask)
        self.assertTrue(np.allclose(out[70:90, 120:180], 50.0))
        self.assertTrue(np.allclose(out[:, :20], 50.0))
        self.assertTrue(np.allclose(out[20:40, 250:300], 200.0))

    def test_sky_mask_fallback_drops_textured_pixels(self):
        pos = np.zeros((H, W, 4), np.float32)
        pos[60:, :, 3] = 1
        img = np.full((H, W, 3), 180.0, np.float32)
        rng = np.random.default_rng(1)
        img[10:30, 40:60] = rng.uniform(0, 255, (20, 20, 3))   # a tree the AI drew into the sky
        sky = MV.sky_mask(pos, img)
        self.assertFalse(sky[15:25, 45:55].any())
        self.assertTrue(sky[5:50, 100:150].all())
        self.assertFalse(sky[62:, :].any())


class Policy(unittest.TestCase):
    def test_product_gate_thresholds(self):
        ok = {"edge_px": 9.0, "edge_px_detail": 3.0, "colour_dE": 14.9, "shift_px": 4.0, "scale_pct": -1.5}
        self.assertEqual(MV.product_violations(ok), [])      # gate reads the detail-pass edge
        bad = {"edge_px_detail": 6.2, "colour_dE": 15.5, "shift_px": 4.4, "scale_pct": 1.6}
        self.assertEqual(len(MV.product_violations(bad)), 4)
        self.assertEqual(MV.product_violations({"error": "x"}), [])
        moved = {"edge_px_detail": 5.1, "colour_dE": 3.2, "shift_px": 10.0, "scale_pct": -7.5}
        self.assertEqual(len(MV.product_violations(moved)), 2)
        self.assertEqual(MV.product_violations(moved, "通过"), [])        # silhouette sits: fit artefact
        self.assertEqual(len(MV.product_violations(moved, "待复核")), 2)
        self.assertEqual(MV.product_violations(dict(moved, colour_dE=20), "通过"), ["颜色 ΔE 20"])

    def test_env_gate_needs_shared_view_and_ignores_sky(self):
        self.assertEqual(MV.env_violations({"reproj_dE": 30, "reproj_dE_geom": 14.9, "covis": 0.9}), [])
        self.assertEqual(MV.env_violations({"reproj_dE_geom": 15.6, "covis": 0.32}), [])   # interior hero case
        self.assertEqual(len(MV.env_violations({"reproj_dE_geom": 15.6, "covis": 0.6})), 1)
        self.assertEqual(MV.env_violations({"reproj_dE_geom": 30, "covis": 0.9}, enabled=False), [])

    def test_v2_first_tries_would_not_retry(self):
        # (geom ΔE, covis) of the 12 first tries in RESULTS_multiview_v2 — none fires
        firsts = [(11.2, .57), (13.2, .43), (9.5, .81), (7.0, .83), (9.8, .67), (14.0, .32), (6.4, .74), (9.8, .52),
                  (7.5, .68), (8.4, .84), (7.8, .61), (5.5, .98)]
        self.assertFalse(any(MV.env_violations({"reproj_dE_geom": g, "covis": c}) for g, c in firsts))

    def test_choose_prefers_product_then_edge(self):
        a = {"product_violations": ["颜色 ΔE 20"], "env_violations": [], "identity": {"edge_px_detail": 1.0}}
        b = {"product_violations": [], "env_violations": ["环境"], "identity": {"edge_px_detail": 5.0}}
        c = {"product_violations": [], "env_violations": [], "identity": {"edge_px_detail": 5.5}}
        self.assertIs(MV.choose([a, b]), b)
        self.assertIs(MV.choose([b, c]), c)
        self.assertEqual(MV.MAX_CALLS, 2)

    def test_summary_line_counts_calls(self):
        line = MV.summary_line([{"name": "A", "ok": True, "calls": 1},
                                {"name": "B", "ok": True, "calls": 2, "retry_reason": ["细节边缘 7.0px"]}])
        self.assertIn("2/2", line)
        self.assertIn("图像调用 3 次", line)
        self.assertIn("B（细节边缘 7.0px）", line)


class LightAndOrder(unittest.TestCase):
    def test_world_locked_sun_changes_side_with_the_camera(self):
        master = cam("m", (0, -7, 1.6))              # looks along +Y
        light = MV.world_light({"light_type": "direct_sun", "key_rel_azimuth_deg": 90, "key_elevation_deg": 25,
                                "sky_words": "傍晚暖光"}, master)
        self.assertAlmostEqual(light["world_azimuth_deg"], 90.0, places=0)
        self.assertIn("画面右侧", MV.light_sentence(light, master))
        back = cam("b", (0, 7, 1.6))                 # looks along -Y: the same sun is now on the left
        self.assertIn("画面左侧", MV.light_sentence(light, back))
        side = cam("s", (7, 0, 1.6))                 # looks along -X toward the product: sun behind the camera
        self.assertIn("顺光", MV.light_sentence(light, side))
        self.assertIn("傍晚暖光", MV.light_sentence(light, side))
        self.assertEqual(MV.light_sentence({}, side), "")

    def test_interior_and_overcast_wording(self):
        m = cam("m", (0, -7, 1.6))
        self.assertIn("窗户", MV.light_sentence(MV.world_light({"light_type": "interior_window", "key_rel_azimuth_deg": 0,
                                                                "key_elevation_deg": 10, "sky_words": ""}, m), m))
        self.assertIn("阴天", MV.light_sentence(MV.world_light({"light_type": "overcast", "key_rel_azimuth_deg": 30,
                                                                "key_elevation_deg": 40, "sky_words": ""}, m), m))

    def test_nearest_angle_first(self):
        m = cam("m", (0, -7, 1.6))
        far = cam("far", (0, 7, 1.6))
        near = cam("near", (3, -6, 1.6))
        mid = cam("mid", (7, 0, 1.6))
        self.assertEqual([r["name"] for r in MV.order_views(m, [far, mid, near], (0, 0, 0))], ["near", "mid", "far"])
        self.assertTrue(MV.same_camera(m, cam("m", (0, -7, 1.6))))
        self.assertFalse(MV.same_camera(m, near))

    def test_view_text_mentions_sources_and_light(self):
        t = MV.view_text("画面里的灰蓝色哑光体块（共 1 个）", "光线锁定：逆光", ["主", "相机2"])
        self.assertIn("光线锁定", t)
        self.assertIn("主视角和相机2", t)
        self.assertIn("产品部分仍以白模为准", t)


class Wiring(unittest.TestCase):
    def test_detail_pass_holdouts_are_camera_only(self):
        src = (PKG / "viewport_capture.py").read_text(encoding="utf-8")
        self.assertIn("_camera_only(obj, rays)", src)
        self.assertIn("hide_lights=True, rays=rays", src)
        self.assertIn("_restore_rays(rays)", src)
        self.assertIn('"visible_shadow"', src)

    def test_clay_base_can_render_geometry_passes(self):
        src = (PKG / "viewport_capture.py").read_text(encoding="utf-8")
        self.assertIn("geometry_prefix", src)
        self.assertIn("render_geometry_now", src)

    def test_operator_ui_and_registration(self):
        init = (PKG / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("WONDFUL_OT_multiview_render", init)
        self.assertIn("WONDFUL_OT_mv_set_master", init)
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        self.assertIn("_draw_multiview(adv, context, props, busy)", ui)
        adv = ui.index("# ---- 高级")
        self.assertGreater(ui.index("        _draw_multiview(adv, context, props, busy)"), adv)   # under 高级 only
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        i = ops.index("class WONDFUL_OT_multiview_render")
        body = ops[i:]
        self.assertIn("reference_paths=env_refs", body)          # refs only as text (simple path)
        self.assertIn("MV.MAX_CALLS", body)
        self.assertFalse(any(p.suffix == ".onnx" for p in PKG.rglob("*")))


if __name__ == "__main__":
    unittest.main()
