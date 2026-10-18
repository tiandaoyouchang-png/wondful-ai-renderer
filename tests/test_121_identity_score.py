"""1.2.1 产品还原度 (识别分): pixel readings (偏移/缩放/轮廓/细节边缘/颜色 ΔE), sub-scores, wiring."""
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "wondful_ai_renderer"


def _load(name):
    if "wondful_ai_renderer" not in sys.modules:
        pkg = types.ModuleType("wondful_ai_renderer")
        pkg.__path__ = [str(PKG)]
        sys.modules["wondful_ai_renderer"] = pkg
    return importlib.import_module(f"wondful_ai_renderer.{name}")


S = _load("identity_score")
DP = _load("detail_paste")
LO = _load("layered_output")
PNG = _load("png_io")


def _product(h=270, w=480):
    """Detail pass: a rounded body with a 'wheel' of radial spokes; labels 0 = body, 1 = wheel."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    body = (((xx - 240) / 170) ** 2 + ((yy - 140) / 70) ** 2) <= 1
    wheel = ((xx - 170) ** 2 + (yy - 175) ** 2) <= 32 ** 2
    mask = (body | wheel).astype(np.float32)
    ang = np.arctan2(yy - 175, xx - 170)
    spokes = 0.5 + 0.4 * np.sign(np.sin(ang * 5))
    det = np.dstack([np.full((h, w), 0.6, np.float32)] * 3)
    det[wheel] = np.repeat(spokes[wheel][:, None], 3, 1)
    det[body & ~wheel] = [0.2, 0.6, 0.25]
    labels = np.full((h, w), -1, np.int32)
    labels[body] = 0
    labels[wheel] = 1
    mats = [{"index": 0, "name": "CarPaint", "kind": "reflective"}, {"index": 1, "name": "Rim", "kind": "default"}]
    rgba = np.dstack([det, mask])
    return rgba, mask, (labels, mats)


def _ai(rgba, mask, bg=(0.15, 0.12, 0.3)):
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    back = np.dstack([np.full((h, w), c, np.float32) for c in bg]) + 0.05 * np.sin(xx / 9)[..., None]
    light = (0.85 + 0.25 * xx / w)[..., None]
    return np.clip(np.where(mask[..., None] > 0.5, rgba[..., :3] * light, back), 0, 1)


class IdentityScoreTests(unittest.TestCase):
    def setUp(self):
        self.det, self.mask, self.matid = _product()
        self.ai = _ai(self.det, self.mask)

    def test_faithful_result_scores_high(self):
        r = S.score(self.ai, self.det, self.mask, self.matid)
        self.assertGreaterEqual(r["composition"], 95)
        self.assertGreaterEqual(r["detail"], 85)
        self.assertGreaterEqual(r["overall"], 85)
        self.assertEqual(S.weakest_parts(r), [])
        self.assertLessEqual(r["shift_px"], 2)
        self.assertLess(abs(r["scale_pct"]), 1.0)
        self.assertLess(r["edge_px"], 1.5)

    def test_enlarged_product_loses_composition(self):
        h, w = self.mask.shape
        big = DP.warp_image(self.ai, (1.1, 1.1), 0, 0, (240.0, 140.0))
        big = np.where(DP.warp_image(self.mask, (1.1, 1.1), 0, 0, (240.0, 140.0))[..., None] > 0.5, big,
                       _ai(self.det, np.zeros_like(self.mask)))
        r = S.score(big, self.det, self.mask, self.matid)
        self.assertLess(r["composition"], 70)
        self.assertAlmostEqual(r["scale_pct"], 10.0, delta=1.5)
        self.assertLess(r["overall"], S.score(self.ai, self.det, self.mask, self.matid)["overall"])

    def test_redrawn_wheel_is_named(self):
        ai = self.ai.copy()
        labels = self.matid[0]
        ai[labels == 1] = 0.5     # spokes gone
        r = S.score(ai, self.det, self.mask, self.matid)
        rim = [p for p in r["parts"] if p["name"] == "Rim"][0]
        self.assertLess(rim["detail"], 40)
        self.assertGreater(rim["edge_px"], S.WEAK_EDGE_PX)
        self.assertTrue(any(n.startswith("Rim") and n.endswith("px") for n in S.weakest_parts(r)))

    def test_wrong_colour_lowers_colour_score(self):
        ai = self.ai.copy()
        ai[self.matid[0] == 0] = [0.7, 0.2, 0.2]     # paint turned red
        r = S.score(ai, self.det, self.mask, self.matid)
        self.assertLess(r["colour"], S.score(self.ai, self.det, self.mask, self.matid)["colour"] - 10)

    def test_glass_parts_are_not_judged(self):
        labels, mats = self.matid
        mats = [dict(mats[0]), dict(mats[1], kind="glass")]
        ai = self.ai.copy()
        ai[labels == 1] = 0.5
        r = S.score(ai, self.det, self.mask, (labels, mats))
        rim = [p for p in r["parts"] if p["name"] == "Rim"]
        self.assertTrue(all("detail" not in p for p in rim))

    def test_without_matid_scores_whole_product(self):
        r = S.score(self.ai, self.det, self.mask, None)
        self.assertIsNotNone(r["detail"])
        self.assertIsNone(r["colour"])
        self.assertIsNotNone(r["overall"])

    def test_score_file_never_raises_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            u8 = lambda x: (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
            PNG.write_png(f"{d}/ai.png", u8(self.ai))
            PNG.write_png(f"{d}/det.png", u8(self.det))
            PNG.write_png(f"{d}/mask.png", u8(np.dstack([np.ones_like(self.mask)] * 3 + [self.mask])))
            r = S.score_file(f"{d}/ai.png", f"{d}/det.png", f"{d}/mask.png")
            self.assertIn("overall", r)
            line = S.summarize(r, r)
            self.assertTrue(line.startswith("480×270 · 偏移 "), line)
            self.assertIn("细节边缘", line)
            self.assertIn("回贴后", line)
            fin = S.score_file(f"{d}/ai.png", f"{d}/det.png", f"{d}/mask.png", placement_of=r)
            self.assertEqual(fin["shift_px"], r["shift_px"])
            bad = S.score_file(f"{d}/missing.png", f"{d}/det.png", f"{d}/mask.png")
            self.assertIn("error", bad)
            self.assertEqual(S.summarize(bad), "")

    def test_wiring(self):
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        ui = (PKG / "ui.py").read_text(encoding="utf-8")
        props = (PKG / "properties.py").read_text(encoding="utf-8")
        self.assertIn("_identity_score(props, variant_final", ops)
        self.assertIn("score_{variant_index:02d}.json", ops)
        self.assertIn("last_score_note: StringProperty", props)
        self.assertIn('getattr(props, "last_score_note", "")', ui)
        # not in the six main cards: only inside 上次渲染 (高级)
        self.assertGreater(ui.index('getattr(props, "last_score_note", "")'), ui.index('_section(adv, "上次渲染"'))


if __name__ == "__main__":
    unittest.main()
