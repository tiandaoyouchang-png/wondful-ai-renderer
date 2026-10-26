"""1.2.6 场景补全 de-tiling: the style fill no longer mirror-tiles one small master patch.

User 2026-10-20: 「需要解决重复纹理的问题」 — the 1.2.5 fill showed a repeated checkerboard on the road (orbit 144° /
216°). Now: several master patches per class (quality-filtered), histogram-preserving stochastic blending on a
world-space triangle grid (random patch / position / quarter turn / flip per vertex), prefiltered levels by pixel
footprint, slow colour drift, facade strips with per-building offsets and tint; plus a periodicity metric.
"""
import math
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402
from test_127_scene_complete import cam, ground_view  # noqa: E402

MV = load_package_module("multiview")


def noise_patch(n=66, seed=0, mean=(120, 120, 124), amp=18.0, grad=0.0):
    """Asphalt-like sample: fine noise (+ an optional light gradient, like a patch half in shade)."""
    rs = np.random.default_rng(seed)
    g = MV.blur(rs.normal(0, 1, (n, n)).astype(np.float32), 1, passes=1)
    g = g / g.std()
    ramp = np.linspace(-1, 1, n)[None, :] * grad
    return np.clip(np.array(mean, np.float32)[None, None] + (amp * g + ramp)[..., None], 0, 255)


def mirror_tile(patch, n):
    h, w = patch.shape[:2]
    yy, xx = np.mgrid[0:n, 0:n]
    return patch[MV._mirror(yy, h), MV._mirror(xx, w)]


def plane_coords(n=400):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    return xx.ravel(), yy.ravel()


class Metric(unittest.TestCase):
    def test_mirror_tiling_scores_high_noise_low(self):
        p = noise_patch()
        self.assertGreater(MV.tile_periodicity(mirror_tile(p, 400).mean(-1), min_lag=33, hp=67), 0.9)
        rs = np.random.default_rng(3)
        free = MV.blur(rs.normal(0, 1, (400, 400)).astype(np.float32), 1, passes=1)
        self.assertLess(MV.tile_periodicity(free, min_lag=33, hp=67), 0.2)

    def test_mask_and_small_region(self):
        p = mirror_tile(noise_patch(n=30), 200).mean(-1)
        m = np.zeros((200, 200), bool)
        m[20:180, 30:190] = True
        self.assertGreater(MV.tile_periodicity(p, m, min_lag=20), 0.85)
        self.assertTrue(math.isnan(MV.tile_periodicity(p, np.zeros_like(m), min_lag=20)))


class Synthesis(unittest.TestCase):
    def setUp(self):
        self.patches = [noise_patch(seed=s, grad=25.0 * (s % 2)) for s in range(4)]
        self.S = MV.make_synth(self.patches, MV.FILL_MPP_GROUND)

    def test_plane_not_periodic(self):
        u, v = plane_coords()
        img = MV.synth_ground(self.S, u, v, salt=1).reshape(400, 400, 3)
        self.assertLess(MV.tile_periodicity(img.mean(-1), min_lag=33, hp=67), 0.3)
        # one patch only (a sidewalk with a single good sample) is still not periodic
        S1 = MV.make_synth(self.patches[:1], MV.FILL_MPP_GROUND)
        img1 = MV.synth_ground(S1, u, v, salt=2).reshape(400, 400, 3)
        self.assertLess(MV.tile_periodicity(img1.mean(-1), min_lag=33, hp=67), 0.3)

    def test_histogram_preserved(self):
        u, v = plane_coords(300)
        img = MV.synth_ground(self.S, u, v, salt=1)
        ref = np.concatenate([MV._flatten(p).reshape(-1, 3) for p in self.patches])
        for c in range(3):
            self.assertLess(abs(float(img[:, c].mean() - ref[:, c].mean())), 2.0)
            self.assertGreater(float(img[:, c].std()), 0.8 * float(ref[:, c].std()))   # no blur / contrast loss
            self.assertLess(float(img[:, c].std()), 1.25 * float(ref[:, c].std()))

    def test_no_seams(self):
        """Neighbour differences of the synthesis look like the sample's own (no lines at the grid edges)."""
        u, v = plane_coords(300)
        img = MV.synth_ground(self.S, u, v, salt=1).reshape(300, 300, 3).mean(-1)
        own = np.mean([np.abs(np.diff(MV._flatten(p).mean(-1), axis=1)).mean() for p in self.patches])
        dx = np.abs(np.diff(img, axis=1))
        self.assertLess(float(dx.mean()), 1.3 * own)
        self.assertLess(float(np.percentile(dx, 99.9)), 6.0 * own)

    def test_world_space_deterministic(self):
        """Colour is a function of the world point only: the same point seen from two views gets one colour."""
        rs = np.random.default_rng(1)
        u, v = rs.uniform(-500, 500, 2000), rs.uniform(-500, 500, 2000)
        a = MV.synth_ground(self.S, u, v, salt=1)
        b = MV.synth_ground(self.S, u[::-1], v[::-1], salt=1)[::-1]
        self.assertLess(float(np.abs(a - b).max()), 1e-3)
        self.assertGreater(float(np.abs(a - MV.synth_ground(self.S, u, v, salt=2)).mean()), 1.0)

    def test_footprint_levels_average_far_pixels(self):
        u, v = plane_coords(200)
        near = MV.synth_ground(self.S, u * 8, v * 8, salt=1, lod=np.zeros(u.size))
        far = MV.synth_ground(self.S, u * 8, v * 8, salt=1, lod=np.full(u.size, 4.0))
        self.assertLess(float(far.std(0).mean()), 0.5 * float(near.std(0).mean()))
        self.assertLess(float(np.abs(far.mean(0) - near.mean(0)).max()), 4.0)

    def test_patch_quality(self):
        good = noise_patch()
        self.assertTrue(MV.patch_quality(good)["ok"])
        smear = MV.blur(np.repeat(noise_patch(seed=4)[:1], 66, 0), 1, passes=1)      # streaks down the patch
        self.assertFalse(MV.patch_quality(smear)["ok"])
        shade = good.copy()
        shade[:, :30] *= 0.45
        self.assertFalse(MV.patch_quality(shade)["ok"])
        brown = noise_patch(mean=(150, 110, 70))
        self.assertTrue(MV.patch_quality(brown)["ok"])
        self.assertFalse(MV.patch_quality(brown, good)["ok"])
        blurry = MV.blur(good, 3)
        self.assertFalse(MV.patch_quality(blurry, good)["ok"])

    def test_cost(self):
        S = MV.make_synth(self.patches, MV.FILL_MPP_GROUND)
        u, v = plane_coords(500)                       # 250 k pixels ≈ the ground of a 960×540 work view
        t0 = time.time()
        MV.synth_ground(S, u, v, salt=1, lod=np.ones(u.size))
        self.assertLess(time.time() - t0, 2.5)


class StyleFill(unittest.TestCase):
    def test_road_fill_in_a_view_not_periodic(self):
        rs = np.random.default_rng(0)
        W, H = 160, 90
        tex = np.clip(120 + 18 * rs.normal(0, 1, (H, W, 3)), 0, 255).astype(np.float32)
        mv = ground_view(cam("m", (0, -8, 1.7), target=(0, 10, 1.0), fov=70), master=True, img=tex)
        layout = {"street": {"a0": [0, 0], "t": [0, 1], "half_width": 4.0}, "boxes": [], "facades": [],
                  "veg_refs": [], "ground_z": 0.0}
        pt = MV.master_patches(mv, layout)
        self.assertGreaterEqual(len(pt["road_all"]), 1)
        self.assertIs(pt["road"], pt["road_all"][0])
        # top-down plane of road, 12 m × 12 m at the fill texel
        n = 400
        yy, xx = np.mgrid[0:n, 0:n] * MV.FILL_MPP_GROUND
        pos = np.zeros((n, n, 4))
        pos[..., 0], pos[..., 1], pos[..., 3] = xx - 2.0, yy, 1
        nrm = np.zeros((n, n, 3))
        nrm[..., 2] = 1
        plane = {"pos": pos, "nrm": nrm, "mask": np.zeros((n, n))}
        img, have = MV.style_fill(plane, np.full((n, n), MV.CLS_ROAD, np.int8), np.zeros((n, n), np.int64), pt,
                                  layout)
        self.assertTrue(have.all())
        self.assertLess(MV.tile_periodicity(img.mean(-1), min_lag=33, hp=67), 0.35)
        self.assertGreater(MV.tile_periodicity(mirror_tile(pt["road"], n).mean(-1), min_lag=33, hp=67), 0.8)

    def test_facades_differ_per_building(self):
        fac = np.zeros((60, 120, 3), np.float32)
        fac[:] = (200, 150, 150)
        for x0 in (4, 19, 31, 52, 60, 83, 97, 109):          # windows, irregular
            fac[10:25, x0:x0 + 6] = (60, 60, 70)
        fac[40:, :] = (120, 90, 80)                         # shop front at the foot
        patches = {"facade": [fac], "road": noise_patch(), "side": None, "veg": None}
        n = 80
        P = np.zeros((n, 2 * n, 4))
        P[..., 0] = 5.0
        P[..., 1] = np.linspace(0, 20, 2 * n)[None]
        P[..., 2] = np.linspace(3.5, 0.0, n)[:, None]
        P[..., 3] = 1
        N = np.zeros((n, 2 * n, 3))
        N[..., 0] = -1
        vk = {"pos": P, "nrm": N, "mask": np.zeros((n, 2 * n))}
        cls = np.full((n, 2 * n), MV.CLS_FACADE, np.int8)
        layout = {"boxes": [], "ground_z": 0.0}
        a, _h = MV.style_fill(vk, cls, np.full((n, 2 * n), 4, np.int64), patches, layout)
        b, _h = MV.style_fill(vk, cls, np.full((n, 2 * n), 8, np.int64), patches, layout)
        self.assertGreater(float(np.abs(a - b).mean()), 3.0)
        # rows stay anchored: the shop front (foot) is darker than the storey above in both
        for img in (a, b):
            self.assertLess(float(img[-5:].mean()), float(img[n // 3:n // 2].mean()))

    def test_narrow_facade_patch_does_not_repeat(self):
        """A 75-texel facade sample (4.5 m) on a 30 m wall: strips with random offsets / flips, no period."""
        rs = np.random.default_rng(2)
        fac = MV.blur(rs.normal(0, 1, (60, 75)).astype(np.float32), 1, passes=1)
        fac = np.clip(140 + 30 * fac[..., None] / fac.std(), 0, 255).repeat(3, -1)
        S = MV.make_synth([fac], MV.FILL_MPP_FACADE, rotate=False, flatten=False)
        s = np.tile(np.arange(500.0), 60)
        v = np.repeat(np.arange(60.0), 500)
        img = MV.synth_strips(S, s, v, salt=3).reshape(60, 500, 3).mean(-1)
        mir = fac.mean(-1)[:, MV._mirror(np.arange(500), 75)]
        self.assertGreater(MV.tile_periodicity(mir, min_lag=20, min_overlap=0.5), 0.9)
        self.assertLess(MV.tile_periodicity(img, min_lag=20, min_overlap=0.5), 0.5)

    def test_fill_periodicity_on_a_view(self):
        W, H = 320, 180
        rec = cam("v", (0, -4, 1.7), target=(0, 6, 0.0), fov=70)
        rec = MV.camera_record("v", np.asarray(rec["matrix"]), math.radians(70), "AUTO", W, H)
        D = MV.rays(rec, W, H)
        O = np.asarray(rec["matrix"])[:3, 3]
        t = np.where(D[..., 2] < -1e-6, -O[2] / np.minimum(D[..., 2], -1e-6), np.inf)
        hit = np.isfinite(t) & (t < 60)
        Pw = O + D * np.where(hit, t, 0)[..., None]
        pos = np.dstack([Pw, hit]).astype(np.float64)
        nrm = np.zeros((H, W, 3))
        nrm[..., 2] = 1
        vk = {"cam": rec, "pos": pos, "nrm": nrm, "mask": np.zeros((H, W))}
        p = noise_patch(grad=30)
        mir = p[MV._mirror(Pw[..., 1] / MV.FILL_MPP_GROUND, 66), MV._mirror(Pw[..., 0] / MV.FILL_MPP_GROUND, 66)]
        layout = {"street": {"a0": [0, 0], "t": [0, 1], "half_width": 4.0}, "ground_z": 0.0}
        S = MV.make_synth([noise_patch(seed=s, grad=30) for s in range(4)], MV.FILL_MPP_GROUND)
        new = MV.synth_ground(S, Pw[..., 1].ravel() / MV.FILL_MPP_GROUND, Pw[..., 0].ravel() / MV.FILL_MPP_GROUND,
                              1).reshape(H, W, 3)
        self.assertGreater(MV.fill_periodicity(vk, mir, hit, layout), 0.6)
        self.assertLess(MV.fill_periodicity(vk, new, hit, layout), 0.35)


if __name__ == "__main__":
    unittest.main()
