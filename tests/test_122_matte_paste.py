"""1.2.2: 分层输出 subject matte trusts the Blender mask (narrow band, stray pieces dropped,
edge decontamination); 细节回贴 / 按材质校色 leave mud, spray and shadow alone; mirrored
plate / logo textures are detected."""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

LO = load_package_module("layered_output")
DP = load_package_module("detail_paste")
MC = load_package_module("material_color")

GREEN = np.array([0.05, 0.55, 0.22], np.float32)


def grass_scene(h=240, w=360, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    # product: a rounded box, anti-aliased mask (Blender-like edge)
    d = np.maximum(np.abs(xx - 180) - 110, np.abs(yy - 120) - 60)
    mask = np.clip(0.5 - d, 0, 1).astype(np.float32)
    bg = np.dstack([0.25 + 0.1 * rng.random((h, w)), 0.55 + 0.25 * rng.random((h, w)), 0.12 + 0.05 * rng.random((h, w))])
    # grass blades: thin bright/dark vertical streaks everywhere outside, also crossing the bottom edge
    for x in rng.integers(0, w, 160):
        y0 = rng.integers(150, h)
        bg[max(0, y0 - 14):y0, x] = [0.45, 0.85, 0.2] if rng.random() < 0.5 else [0.1, 0.3, 0.05]
    car = np.broadcast_to(GREEN, (h, w, 3)).copy()
    car[:, ::23] *= 0.6                      # panel lines
    img = mask[..., None] * car + (1 - mask[..., None]) * bg
    return np.clip(img, 0, 1).astype(np.float32), mask


class MatteTests(unittest.TestCase):
    def test_band_is_narrow_and_mask_is_trusted(self):
        img, mask = grass_scene()
        res = LO.refine_alpha(img, mask)
        a = res["alpha"]
        self.assertLessEqual(res["band_px"], LO.EDGE_BAND_MAX_PX)
        outside = ~LO.dilate(mask > 0.01, 0)
        self.assertEqual(float(a[outside].max()), 0.0)          # never grows into the grass
        self.assertTrue((a[LO.erode(mask > 0.5, 3)] == 1.0).all())
        self.assertLess(np.abs(a - mask).mean(), 0.003)

    def test_edge_colour_is_decontaminated(self):
        img, mask = grass_scene()
        res = LO.refine_alpha(img, mask)
        fg = LO.decontaminate(img, res["alpha"], res["bg_local"], res["fg_local"])
        edge = (res["alpha"] > 0.05) & (res["alpha"] < 0.95)
        self.assertTrue(edge.any())
        # the subject's edge colour is the product's, not grass-tinted
        dist_prod = np.abs(fg[edge] - GREEN).mean()
        dist_raw = np.abs(img[edge] - GREEN).mean()
        self.assertLess(dist_prod, dist_raw * 0.6)

    def test_stray_far_piece_dropped_big_or_near_kept(self):
        h, w = 200, 300
        m = np.zeros((h, w), np.float32)
        m[60:150, 40:200] = 1                 # main body (14400 px)
        m[10:22, 260:275] = 1                 # far speck (180 px) → dropped
        m[60:100, 203:209] = 1                # detached but right next to it → kept
        m[160:195, 230:290] = 1               # detached and big (2100 px ≥ 10 %) → kept
        keep = LO.product_pieces(m)
        self.assertFalse(keep[15, 265])
        self.assertTrue(keep[80, 205])
        self.assertTrue(keep[175, 260])
        img = np.dstack([m * 0.2 + 0.5] * 3).astype(np.float32)
        res = LO.refine_alpha(img, m)
        self.assertEqual(float(res["alpha"][10:22, 260:275].max()), 0.0)
        self.assertGreater(res["dropped_px"], 150)

    def test_label_components_matches_flood_fill(self):
        rng = np.random.default_rng(5)
        for _ in range(5):
            m = rng.random((40, 60)) > 0.62
            labels, areas = LO.label_components(m)
            # reference: BFS with 8-connectivity
            ref = np.zeros(m.shape, int); n = 0
            for y, x in zip(*np.nonzero(m)):
                if ref[y, x]:
                    continue
                n += 1; stack = [(y, x)]; ref[y, x] = n
                while stack:
                    cy, cx = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = cy + dy, cx + dx
                            if 0 <= yy < 40 and 0 <= xx < 60 and m[yy, xx] and not ref[yy, xx]:
                                ref[yy, xx] = n; stack.append((yy, xx))
            self.assertEqual(len(areas), n)
            self.assertEqual(sorted(areas.values()), sorted(np.bincount(ref.ravel())[1:].tolist()))


def mud_scene(h=240, w=360, seed=11):
    """Detail pass: clean green panel with dark panel lines; AI: same panel with mud speckles
    and a dark splash shadow over the lower body."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), np.float32)
    mask[30:210, 30:330] = 1
    detail = np.dstack([np.broadcast_to(GREEN, (h, w, 3)) * 1.1, mask]).astype(np.float32)
    detail[:, ::18, :3] *= 0.35                                   # panel lines (high frequency)
    detail[60:70, 100:260, :3] = 0.9                               # a bright badge strip
    ai = np.broadcast_to(GREEN, (h, w, 3)).copy().astype(np.float32)
    ai *= (0.9 + 0.1 * np.linspace(1, 0, h))[:, None, None]       # soft lighting
    mud = np.zeros((h, w), bool)
    lower = np.zeros((h, w), bool); lower[140:210, 30:330] = True
    mud |= lower & (rng.random((h, w)) < 0.35)
    mud = LO.dilate(mud, 1) & lower
    ai[mud] = [0.18, 0.13, 0.08]                                   # brown mud
    shadow = np.zeros((h, w), bool); shadow[190:210, 30:330] = True
    ai[shadow & ~mud] *= 0.25                                      # splash / contact shadow
    ai = np.clip(ai * mask[..., None] + 0.5 * (1 - mask[..., None]), 0, 1)
    return ai, detail, mask, mud, shadow


class MudPasteTests(unittest.TestCase):
    def test_occluder_finds_mud_and_shadow_not_clean_paint(self):
        ai, detail, mask, mud, shadow = mud_scene()
        occ = DP.occluder_mask(ai, detail[..., :3], mask, max(ai.shape))
        self.assertGreater(occ[mud].mean(), 0.8)
        self.assertGreater(occ[shadow & ~mud].mean(), 0.7)
        clean = LO.erode(mask > 0.5, 3).copy(); clean[120:, :] = False
        self.assertLess(occ[clean].mean(), 0.1)

    def test_paste_never_brightens_mud_or_shadow(self):
        ai, detail, mask, mud, shadow = mud_scene()
        fit = {"scale_x": 1.0, "scale_y": 1.0, "dx": 0.0, "dy": 0.0}
        res = DP.paste_details(ai, detail, mask, fit=fit)
        out = res["image"]
        lum = LO.luminance
        dirty = (mud | shadow) & LO.erode(mask > 0.5, 6)
        self.assertLessEqual(float((lum(out) - lum(ai))[dirty].max()), 2.0 / 255)
        self.assertLess(float(np.abs(out - ai)[dirty].mean()), 1.0 / 255)
        # clean upper paint still receives the panel lines
        clean = LO.erode(mask > 0.5, 8).copy(); clean[110:, :] = False
        self.assertGreater(float(np.abs(out - ai)[clean].max()), 0.05)

    def test_colour_fix_does_not_lift_dirt(self):
        ai, detail, mask, mud, shadow = mud_scene()
        labels = np.where(mask > 0.5, 0, -1).astype(np.int32)
        mats = [{"index": 0, "name": "CarPaint", "kind": "default"}]
        model = detail.copy(); model[..., :3] = np.broadcast_to(GREEN * 1.5, model[..., :3].shape)   # brighter model paint
        weight = LO.erode(mask > 0.5, 4).astype(np.float32)
        occ = DP.occluder_mask(ai, model[..., :3], mask, max(ai.shape))
        # one material only: pin the scene cast at 0 so the brighter model paint is a real
        # per-material error (otherwise the cast estimate absorbs it and nothing moves)
        cast = {"dL": 0.0, "da": 0.0, "db": 0.0, "n": 0}
        res = MC.correct_colours(ai, model[..., :3], labels, mats, weight, strength=1.0, occluder=occ, cast=cast)
        out = res["image"]
        dirty = (mud | shadow) & (weight > 0.5)
        self.assertLess(float(np.abs(out - ai)[dirty].mean()), 1.0 / 255)
        self.assertGreater(res["changed_px"], 1000)                  # clean paint is still corrected
        # without the occluder map, very dark pixels are still never lifted
        res2 = MC.correct_colours(ai, model[..., :3], labels, mats, weight, strength=1.0, cast=cast)
        self.assertGreater(res2["changed_px"], 1000)
        L0 = MC.srgb_to_lab(ai)[..., 0]; L2 = MC.srgb_to_lab(res2["image"])[..., 0]
        med = float(res2["regions"][0]["ai"][0])
        dark = (weight > 0.5) & (L0 < 0.6 * med)                   # below ~60 % of the median: never lifted
        self.assertTrue(dark.any())
        self.assertLessEqual(float((L2 - L0)[dark].max()), 0.5)
        clean = (weight > 0.5) & ~(mud | shadow); clean[110:, :] = False
        self.assertGreater(float((L2 - L0)[clean].mean()), 0.5)    # while clean paint is lifted


REAL = Path("/workspace/work/release_122/mud_pass")


@unittest.skipUnless((REAL / "detail.png").is_file() and Path("/workspace/work/film_gen2/out_mud.png").is_file(),
                     "real mud generation + Blender passes not present (sandbox-only data)")
class RealMudTest(unittest.TestCase):
    def test_real_mud_lower_door_kept(self):
        png = load_package_module("png_io")
        I = png.read_png("/workspace/work/film_gen2/out_mud.png")[..., :3].astype(np.float32) / 255
        h, w = I.shape[:2]

        def load(p):
            a = png.read_png(str(p)).astype(np.float32) / 255
            if a.shape[-1] == 3:
                a = np.dstack([a, np.ones(a.shape[:2], np.float32)])
            return np.clip(LO.resize(a, w, h), 0, 1)
        D, M, K = load(REAL / "detail.png"), load(REAL / "mask.png")[..., 3], load(REAL / "keep.png")
        matid = DP.load_matid(str(REAL / "matid.png"), w, h)
        res = DP.paste_details(I, D, M, K, matid=matid, colour_strength=0.7)
        reg = (slice(470, 560), slice(860, 1040))                 # mud-covered lower door
        self.assertLess(float(np.abs(res["image"][reg] - I[reg]).mean() * 255), 3.0)
        # the region also holds the dark sill trim (dark by design, not an occluder), so only
        # the mud on the green paint is expected to be flagged
        self.assertGreater(float(res["occluder"][reg].mean()), 0.25)


class MirrorTests(unittest.TestCase):
    QUAD = np.array([[[0, 0, 0], [1, 0, 0], [1, 0, 1]], [[0, 0, 0], [1, 0, 1], [0, 0, 1]]], float)
    UV = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]], float)

    def test_plate_reads_correctly_or_mirrored(self):
        # vertices CCW seen from -Y (normal points to -Y): camera at -Y reads the texture correctly
        cam = (0.5, -5.0, 0.5)
        self.assertEqual(DP.mirrored_text_fraction(self.QUAD, self.UV, cam), 0.0)
        flipped_uv = self.UV.copy(); flipped_uv[..., 0] = 1 - flipped_uv[..., 0]
        self.assertEqual(DP.mirrored_text_fraction(self.QUAD, flipped_uv, cam), 1.0)
        # back side (outward normal away from the camera) does not count
        self.assertEqual(DP.mirrored_text_fraction(self.QUAD, self.UV, (0.5, 5.0, 0.5)), 0.0)

    def test_negative_scale_mirrors(self):
        q = self.QUAD.copy(); q[..., 0] *= -1                      # object scaled x = -1
        self.assertEqual(DP.mirrored_text_fraction(q, self.UV, (-0.5, -5.0, 0.5), negative_scale=True), 1.0)

    def test_note(self):
        self.assertIn("镜像", DP.mirrored_note([{"object": "carPlate", "fraction": 1.0}]))
        self.assertEqual(DP.mirrored_note([]), "")


if __name__ == "__main__":
    unittest.main()
