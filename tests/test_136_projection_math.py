"""Reference-camera projection agrees with a full XYZ view/projection matrix.

The oracle constructs and inverts a homogeneous camera transform; it does not
reuse Projector's expanded camera-axis formula. Blender integration can compare
the same points with bpy_extras.object_utils.world_to_camera_view.
"""
import copy
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module

PD = load_package_module("precise_depth")


def camera_matrix(cam):
    """Independent homogeneous XYZ Euler transform, camera local -Z forward."""
    x = math.radians(90.0 + cam["pitch_deg"])
    y = math.radians(cam.get("roll_deg", 0.0))
    rx = np.eye(4)
    rx[1:3, 1:3] = [[math.cos(x), -math.sin(x)],
                    [math.sin(x), math.cos(x)]]
    ry = np.eye(4)
    ry[0, 0], ry[0, 2] = math.cos(y), math.sin(y)
    ry[2, 0], ry[2, 2] = -math.sin(y), math.cos(y)
    translation = np.eye(4)
    translation[2, 3] = cam["height_m"]
    return translation @ ry @ rx


def matrix_projection(cam, frame, point):
    near, far = 0.1, 10000.0
    tx = math.tan(math.radians(cam["hfov_deg"]) / 2.0)
    ty = tx * frame[1] / frame[0]
    perspective = np.array([
        [1.0 / tx, 0.0, 0.0, 0.0],
        [0.0, 1.0 / ty, 0.0, 0.0],
        [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
        [0.0, 0.0, -1.0, 0.0],
    ])
    local = np.linalg.inv(camera_matrix(cam)) @ np.r_[point, 1.0]
    if -local[2] <= near:
        return None
    clip = perspective @ local
    ndc = clip[:2] / clip[3]
    return (ndc[0] + 1.0) / 2.0, (1.0 - ndc[1]) / 2.0


POSES = ((0, 0), (0, -18), (0, 18), (-14, 0), (14, 0),
         (-19, 27), (19, -27), (26, 31), (-26, -31))
FRAMES = ((1600, 900), (900, 1600), (1000, 1000))


class CameraProjection(unittest.TestCase):
    def test_matches_full_view_projection_with_signed_pitch_roll_and_aspect(self):
        points = ((0, 10, 0), (0, 10, 3), (-3, 12, 4), (4, 18, 2),
                  (6, 8, 7), (-8, 10, 1), (0, 0.8, 1.7))
        for pitch, roll in POSES:
            cam = dict(height_m=1.7, pitch_deg=pitch, roll_deg=roll, hfov_deg=78.0)
            for frame in FRAMES:
                project = PD.Projector(cam, frame)
                for point in points:
                    with self.subTest(pitch=pitch, roll=roll, frame=frame, point=point):
                        np.testing.assert_allclose(project(point), matrix_projection(cam, frame, point),
                                                   rtol=0, atol=1e-12)

    def test_optical_axis_projects_to_center_for_combined_rotations(self):
        for pitch, roll in POSES:
            cam = dict(height_m=2.3, pitch_deg=pitch, roll_deg=roll, hfov_deg=60.0)
            point = (camera_matrix(cam) @ [0, 0, -17, 1])[:3]
            for frame in FRAMES:
                with self.subTest(pitch=pitch, roll=roll, frame=frame):
                    np.testing.assert_allclose(PD.Projector(cam, frame)(point), (0.5, 0.5), atol=1e-12)

    def test_horizontal_sensor_fit_keeps_frame_corners_for_portrait_and_landscape(self):
        cam = dict(height_m=1.2, pitch_deg=-12, roll_deg=23, hfov_deg=92.0)
        world = camera_matrix(cam)
        distance = 20.0
        tx = math.tan(math.radians(cam["hfov_deg"]) / 2)
        for frame in FRAMES:
            ty = tx * frame[1] / frame[0]
            for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                point = (world @ [sx * distance * tx, sy * distance * ty, -distance, 1])[:3]
                with self.subTest(frame=frame, sx=sx, sy=sy):
                    np.testing.assert_allclose(PD.Projector(cam, frame)(point),
                                               ((sx + 1) / 2, (1 - sy) / 2), atol=1e-12)

    def test_near_and_behind_camera_are_rejected_after_rotation(self):
        cam = dict(height_m=1.7, pitch_deg=22, roll_deg=-29, hfov_deg=65.0)
        world = camera_matrix(cam)
        project = PD.Projector(cam, FRAMES[0])
        for local_z in (-0.05, 0.0, 2.0):
            with self.subTest(local_z=local_z):
                self.assertIsNone(project((world @ [0, 0, local_z, 1])[:3]))
        self.assertIsNotNone(project((world @ [0, 0, -0.2, 1])[:3]))

    def test_missing_roll_retains_legacy_zero_roll_projection(self):
        cam = dict(height_m=1.6, pitch_deg=-7.0, hfov_deg=72.0)
        for point in ((0, 10, 0), (-4, 20, 7), (2, 8, 1)):
            np.testing.assert_allclose(PD.Projector(cam, FRAMES[0])(point),
                                       matrix_projection(cam, FRAMES[0], point), atol=1e-12)

    def test_ground_plane_recovers_euler_pitch_in_presence_of_roll(self):
        for pitch, roll in POSES:
            cam = dict(height_m=1.9, pitch_deg=pitch, roll_deg=roll, hfov_deg=65.0)
            world = camera_matrix(cam)
            # Columns expressed in the depth module's right/forward/up order.
            expected = np.column_stack((world[:3, 0], -world[:3, 2], world[:3, 1]))
            normal = expected[2]
            rotation, height, actual_pitch, actual_roll = PD.camera_from_plane(normal, cam["height_m"])
            with self.subTest(pitch=pitch, roll=roll):
                self.assertAlmostEqual(height, cam["height_m"], places=12)
                self.assertAlmostEqual(actual_pitch, pitch, places=12)
                self.assertAlmostEqual(actual_roll, roll, places=12)
                np.testing.assert_allclose(rotation, expected, rtol=0, atol=1e-12)


class GroundContactDuringDepthSnap(unittest.TestCase):
    def test_depth_correction_does_not_lift_or_bury_grounded_building(self):
        cam = dict(height_m=1.8, pitch_deg=-8, roll_deg=13, hfov_deg=70.0)
        # A uniform measured structure field puts all sampled depths at 20 m.
        points = np.zeros((60, 100, 3), dtype=np.float32)
        points[..., 1] = 20.0
        points[..., 2] = 3.0
        geo = dict(P=points, mask=np.ones((60, 100), bool),
                   cls=np.full((60, 100), PD.STRUCT, dtype=np.uint8))
        for distance in (10.0, 40.0):
            primitive = dict(id="grounded", type="building", x=0.0, y=distance,
                             z_offset=0.0, w=5.0, d=3.0, h=6.0, roof_h=0.0)
            data = dict(camera=cam, primitives=[copy.deepcopy(primitive)])
            with self.subTest(distance=distance):
                result = PD.snap(data, geo, (1600, 900), size=False)
                self.assertEqual(result["snapped"], 1)
                moved = data["primitives"][0]
                self.assertAlmostEqual(moved["y"], 20.0)
                self.assertEqual(min(p[2] for p in PD.prim_corners(moved)), 0.0)
                # A camera-centred scale would move the base by h*(1-ratio).
                # This must not be used as a purported screen-preservation fix.
                self.assertNotEqual(cam["height_m"] * (1 - 20.0 / distance), 0.0)

    def test_explicit_terrain_clearance_is_not_treated_as_absolute_world_z(self):
        primitive = dict(x=1.0, y=12.0, z_offset=0.3, w=4.0, d=3.0, h=7.0)
        PD._scale_about_camera(primitive, 0.5, ("w", "d", "h"))
        self.assertEqual(primitive["z_offset"], 0.3)
        self.assertEqual(min(p[2] for p in PD.prim_corners(primitive)), 0.3)


if __name__ == "__main__":
    unittest.main()
