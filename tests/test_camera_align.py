"""2026-11 「对齐到参考图」: the numpy camera solve (Fable 5.2 design) on synthetic views."""
import unittest

import numpy as np

import test_two_workflows as T

AC = T.load("camera_align")
VL = T.load("vehicle_locate")

DIMS = dict(wheelbase=3.0, track=1.62, wheel_r=0.38, length=4.9, width=1.9, height=1.6)


def synth(az_deg, el_deg=6.0, dist=8.0, lens=40.0, aspect=1.5, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    tpl = AC.car_template(**DIMS)
    p = np.array([az_deg * AC.D, el_deg * AC.D, np.log(dist), 0, 0, 0, np.log(lens / 36)])
    h = 1 / aspect
    wheels = []
    names = {"FL": "front_left", "FR": "front_right", "RL": "rear_left", "RR": "rear_right"}
    C, R, f = AC.camera(p, tpl["anchor_z"])
    for k, label in names.items():
        X = tpl["kp"][k + "_contact"]
        side = (1 if k[1] == "L" else -1) * C[0] > -0.3 or abs(C[0]) < 1.0
        if not side:
            continue
        uv = AC.project(p, np.array([X, tpl["kp"][k + "_center"]]), aspect, tpl["anchor_z"])[0]
        uv[:, 1] /= h
        uv += rng.normal(0, noise, uv.shape)
        wheels.append({"label": label, "contact": uv[0].tolist(), "center": uv[1].tolist(), "visible": True})
    proxy = AC.project(p, tpl["proxy"], aspect, tpl["anchor_z"])[0]
    proxy[:, 1] /= h
    box = [*proxy.min(0), *proxy.max(0)]
    hz = AC.project(p, C + 1e5 * np.array([-R[0, 2], -R[1, 2], 0]), aspect, tpl["anchor_z"])[0][0, 1] / h
    facing = {0: "front", 90: "left", -90: "right", 180: "rear"}.get(int(round(az_deg / 90.0)) * 90 % 360 - 0, "")
    return p, {"found": True, "bbox": box, "wheels": wheels, "facing": "", "heading_deg": az_deg,
               "horizon_y": hz, "cut_off": []}


class CameraAlignTests(unittest.TestCase):
    def check(self, az):
        p_true, obs = synth(az, noise=0.004, seed=int(az) % 7)
        fit, report, tpl = AC.align(obs, DIMS, 1500, 1000)
        self.assertIn(report["status"], ("aligned", "touch_up"), report)
        self.assertLess(report["kp_rms"], 0.06)
        self.assertGreater(report["bbox_iou"], 0.75)

    def test_three_quarter_front(self):
        self.check(35)

    def test_three_quarter_rear(self):
        self.check(150)

    def test_side(self):
        self.check(90)

    def test_old_dims_sanity(self):
        dims, trusted = VL.old_dims({"dims": dict(DIMS)}, {"length": 5.0})
        self.assertTrue(trusted)
        bad = dict(DIMS, wheelbase=1.0)
        dims, trusted = VL.old_dims({"dims": bad}, {"length": 5.0})
        self.assertFalse(trusted)
        self.assertEqual(dims, {"length": 5.0})


if __name__ == "__main__":
    unittest.main()
