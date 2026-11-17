"""2026-11 review: lit product reference + placed-product canvas (pure numpy parts)."""
import unittest

import numpy as np

import test_two_workflows as T

PL = T.load("product_lookdev")


def desert(h=90, w=160):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (240, 220, 185)          # warm sky
    img[int(h * 0.55):] = (200, 140, 75)   # sand
    img[: h // 2, int(w * 0.70): int(w * 0.80)] = (255, 250, 235)   # sun glare right of centre
    return img


class LookdevTests(unittest.TestCase):
    def test_environment_colours_sample_sky_ground_and_sun_side(self):
        env = PL.environment_colours(desert())
        self.assertGreater(env["sky"][0], env["ground"][0])
        self.assertGreater(env["ground"][0], env["ground"][2])       # warm sand, linear
        self.assertTrue(all(0.0 <= c <= 1.0 for c in env["sky"] + env["ground"] + env["horizon"]))
        self.assertGreater(env["sun_x"], 0.6)
        self.assertTrue(env["sun_in_frame"])

    def test_composite_places_product_and_caps_shadow(self):
        src = desert()
        h, w = src.shape[:2]
        look = np.zeros((h, w, 4), np.uint8)
        look[30:60, 40:120] = (20, 120, 40, 255)        # product
        look[60:66, 40:120] = (0, 0, 0, 255)            # shadow catcher writes opaque black
        flat_alpha = np.zeros((h, w), np.float32)
        flat_alpha[30:60, 40:120] = 1.0
        out = PL.composite(src, look, flat_alpha)
        np.testing.assert_array_equal(out[45, 80], (20, 120, 40))
        np.testing.assert_array_equal(out[5, 5], src[5, 5])
        shadow = out[63, 80].astype(int)
        self.assertTrue((shadow < src[63, 80]).all())                 # darkened …
        self.assertTrue((shadow > 0).all())                           # … never an opaque floor

    def test_composite_resizes_lookdev_to_the_photograph(self):
        src = desert(90, 160)
        look = np.zeros((45, 80, 4), np.uint8)
        look[10:30, 20:60] = (10, 200, 30, 255)
        out = PL.composite(src, look)
        self.assertEqual(out.shape, (90, 160, 3))
        self.assertGreater(int(out[40, 80, 1]), 150)

    def test_paint_materials_by_name_then_largest_coloured_surface(self):
        rows = [dict(name="CarPaint", base_color=[0, .3, .05], weight=10),
                dict(name="Interior_Body_Leather", base_color=[.3, .1, 0], weight=50),
                dict(name="Tyre", base_color=[.02, .02, .02], weight=40)]
        self.assertEqual(PL.paint_materials(rows), ["CarPaint"])
        rows = [dict(name="Material.001", base_color=[0, .3, .05], weight=30),
                dict(name="Material.002", base_color=[.01, .01, .01], weight=90),
                dict(name="Glass", base_color=[.9, .9, .9], weight=80, transmission=1.0),
                dict(name="Material.003", base_color=[.5, .5, .5], weight=5)]
        self.assertEqual(PL.paint_materials(rows), ["Material.001"])
        self.assertEqual(PL.paint_materials([]), [])

    def test_world_exposure_lifts_dark_scenes_and_keeps_bright_ones(self):
        dusk = {"sky": [0.05, 0.037, 0.019], "horizon": [0.127, 0.07, 0.03]}
        self.assertAlmostEqual(PL.world_strength(dusk), PL.WORLD_TARGET / 0.127, places=3)
        self.assertEqual(PL.world_strength({"sky": [0.9, 0.8, 0.7], "horizon": [0.95, 0.9, 0.8]}), 1.0)
        self.assertEqual(PL.world_strength({"sky": [0, 0, 0], "horizon": [0, 0, 0]}), 20.0)

    def test_v2_defaults_have_no_mirror_glint(self):
        self.assertGreaterEqual(PL.COAT_ROUGHNESS, 0.08)
        self.assertGreater(PL.HIGHLIGHT_CLAMP, 0)
        self.assertEqual(PL.LOOKDEV_VERSION, 2)

    def test_sun_rotation_is_finite(self):
        m = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        for backlit in (False, True):
            r = PL.sun_rotation(m, 0.7, backlit)
            self.assertEqual(len(r), 3)
            self.assertTrue(all(np.isfinite(r)))


if __name__ == "__main__":
    unittest.main()
