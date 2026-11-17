"""2026-11: locate the photograph's own vehicle; placement guard; plate covers the old vehicle."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import test_two_workflows as T

PNG = T.load("png_io")
VL = T.load("vehicle_locate")
SP = T.load("source_plate")

REPLY = """Here you go:
{"found": true, "bbox": [0.05, 0.40, 0.95, 0.80],
 "wheels": [{"label": "front_left", "contact": [0.80, 0.79], "center": [0.80, 0.72], "visible": 1},
            {"label": "rear_left", "contact": [0.20, 0.78], "center": [0.20, 0.71], "visible": 0.9}],
 "facing": "left", "heading_deg": 90, "horizon_y": 0.38, "cut_off": [], "confidence": 0.8}"""


def alpha_box(h, w, box):
    a = np.zeros((h, w), np.float32)
    a[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)] = 1
    return a


class LocateTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        os.environ["WONDFUL_LOCATE_CACHE"] = str(self.dir / "locate.json")
        self.photo = self.dir / "photo.png"
        PNG.write_png(self.photo, np.full((60, 100, 3), 120, np.uint8))

    def test_parse_and_cache(self):
        calls = []

        def ask(prompt, path):
            calls.append(path)
            return REPLY

        a = VL.locate(self.photo, ask)
        b = VL.locate(self.photo, ask)
        self.assertEqual(len(calls), 1)
        self.assertTrue(a["found"] and b["cached"])
        self.assertEqual(a["bbox"], [0.05, 0.40, 0.95, 0.80])
        self.assertEqual(len(a["wheels"]), 2)
        self.assertAlmostEqual(a["horizon_y"], 0.38)

    def test_parse_rejects_garbage(self):
        with self.assertRaises(ValueError):
            VL.parse_reply("no json here")
        self.assertEqual(VL.parse_reply('{"found": false}'), {"found": False})

    def test_aligned_camera_passes(self):
        loc = VL.parse_reply(REPLY)
        check = VL.placement_check(loc, alpha_box(300, 500, [0.10, 0.45, 0.90, 0.80]))
        self.assertEqual(check["verdict"], "ok")

    def test_small_centred_car_fails(self):
        # the reported case: a big Jeep across the frame, the new car small in the middle
        loc = VL.parse_reply(REPLY)
        check = VL.placement_check(loc, alpha_box(300, 500, [0.30, 0.42, 0.62, 0.62]))
        self.assertEqual(check["verdict"], "fail")
        self.assertIn("对齐到参考图", check["reason"])

    def test_no_vehicle_is_unknown(self):
        self.assertEqual(VL.placement_check({"found": False}, np.ones((10, 10)))["verdict"], "unknown")

    def test_plate_region_covers_old_vehicle(self):
        loc = VL.parse_reply(REPLY)
        region = VL.old_vehicle_region(loc, (60, 100))
        self.assertTrue(region[30, 6] and region[30, 94] and region[47, 50])
        self.assertFalse(region[5, 50])


class PlateExtraRegionTests(unittest.TestCase):
    def test_extra_region_is_edited(self):
        d = Path(tempfile.mkdtemp())
        os.environ["WONDFUL_PLATE_CACHE"] = str(d / "cache")
        src = np.full((60, 100, 3), 200, np.uint8)
        PNG.write_png(d / "src.png", src)
        alpha = np.zeros((60, 100), np.float32); alpha[30:45, 40:60] = 1
        extra = np.zeros((60, 100), bool); extra[25:50, 2:98] = True

        def edit(base, mask, out):
            PNG.write_png(out, np.zeros((60, 100, 3), np.uint8))
            return out

        info = SP.make_plate(d / "src.png", alpha, d / "w", edit, extra_region=extra)
        plate = PNG.read_png(info["path"])
        self.assertLess(int(plate[37, 5, 0]), 100)       # old vehicle's far end was edited
        self.assertEqual(int(plate[2, 50, 0]), 200)      # sky untouched


if __name__ == "__main__":
    unittest.main()
