"""Reference-camera framing and connected architecture geometry, without network calls."""
import copy
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module

SB = load_package_module("scene_blockout")
FRAME = (1600, 900)


def box_points(width=2.0, length=5.0, height=1.5, heading=-55):
    corners = np.array([(x, y, z) for x in (-width / 2, width / 2)
                        for y in (-length / 2, length / 2) for z in (0.0, height)])
    a = math.radians(heading)
    rotation = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return corners @ rotation.T


def independently_project(camera, points):
    """Explicit camera basis, independent of scene_blockout's projection helper."""
    pitch = math.radians(camera["pitch_deg"])
    roll = math.radians(camera["roll_deg"])
    right = np.array([math.cos(roll), 0, -math.sin(roll)])
    up = np.array([math.sin(roll) * math.cos(pitch), -math.sin(pitch), math.cos(roll) * math.cos(pitch)])
    forward = np.array([math.sin(roll) * math.sin(pitch), math.cos(pitch), math.cos(roll) * math.sin(pitch)])
    delta = np.asarray(points) - [0, 0, camera["height_m"]]
    depth = delta @ forward
    tx = math.tan(math.radians(camera["hfov_deg"]) / 2)
    ty = tx * FRAME[1] / FRAME[0]
    return np.column_stack((0.5 + (delta @ right) / (2 * depth * tx),
                            0.5 - (delta @ up) / (2 * depth * ty)))


def anchors(camera, points, spot):
    projected = independently_project(camera, np.vstack((points + spot, spot)))
    lo, hi = projected[:-1].min(0), projected[:-1].max(0)
    return {"present": True, "bbox": {"x_min": lo[0], "y_min": lo[1], "x_max": hi[0], "y_max": hi[1]},
            "ground_contact": {"x_frac": projected[-1, 0], "y_frac": projected[-1, 1]},
            "heading_deg": -55.0, "heading_confidence": 1.0}


def section(left, right):
    return {"left": dict(zip(("x", "y", "z"), left)), "right": dict(zip(("x", "y", "z"), right))}


class ReferenceFramingTests(unittest.TestCase):
    def setUp(self):
        self.camera = {"height_m": 0.8, "pitch_deg": -2.0, "roll_deg": 4.0, "hfov_deg": 70.0}

    def test_legacy_json_and_invalid_anchors_do_not_activate(self):
        source = {"camera": {"hfov_deg": 65}}
        before = copy.deepcopy(source)
        result = SB.normalize(source)
        self.assertFalse(result["reference_subject"]["present"])
        self.assertEqual(result["ribbons"], [])
        self.assertEqual(source, before)
        subject = anchors(self.camera, box_points(), np.array([0.3, 10.0, 0]))
        for bad in (dict(subject, present="true"), dict(subject, bbox={}),
                    dict(subject, ground_contact={"x_frac": float("nan"), "y_frac": 0.8})):
            self.assertFalse(SB.reference_subject({"reference_subject": bad})["present"])

    def test_projection_matches_independent_camera_basis_with_roll(self):
        points = box_points() + [0.3, 10.0, 0]
        for pitch, roll in ((0, 0), (-8, 15), (9, -20)):
            camera = dict(self.camera, pitch_deg=pitch, roll_deg=roll)
            np.testing.assert_allclose(SB.project_points(camera, FRAME, points),
                                       independently_project(camera, points), atol=1e-12)

    def test_fit_recovers_known_position_from_actual_rotated_bounds(self):
        points = box_points()
        spot = np.array([0.6, 11.0, 0.0])
        subject = anchors(self.camera, points, spot)
        before = points.copy()
        result = SB.fit_subject_framing(self.camera, FRAME, subject, points, (8, 48))
        self.assertTrue(result["enabled"])
        self.assertLess(result["max_error_frac"], 1e-5)
        np.testing.assert_allclose([result["x"], result["y"]], spot[:2], atol=1e-3)
        np.testing.assert_array_equal(points, before)

    def test_taller_target_keeps_its_height(self):
        camera = dict(self.camera, pitch_deg=0, roll_deg=0)
        spot = np.array([-0.4, 10.0, 0.0])
        coupe, suv = box_points(height=1.2), box_points(height=1.85)
        subject = anchors(camera, coupe, spot)
        result = SB.fit_subject_framing(camera, FRAME, subject, suv, (2, 20))
        self.assertLess(result["max_error_frac"], 1e-5)
        self.assertFalse(result["height_matched"])
        self.assertLess(result["metrics"]["bbox"]["y_min"], subject["bbox"]["y_min"] - 0.04)
        self.assertEqual(float(suv[:, 2].max()), 1.85)

    def test_incompatible_camera_and_anchors_report_residual(self):
        camera = dict(self.camera, pitch_deg=0, roll_deg=0)
        points = box_points()
        subject = anchors(camera, points, np.array([0.0, 10.0, 0.0]))
        subject["ground_contact"]["y_frac"] = 0.95
        result = SB.fit_subject_framing(camera, FRAME, subject, points)
        self.assertEqual(result["status"], "RESIDUAL")
        self.assertGreater(result["max_error_frac"], 0.05)


class RibbonGeometryTests(unittest.TestCase):
    def curve(self):
        return [section((-3, 4, 1), (3, 4, 1)), section((-3, 7, 3), (3, 7, 4)),
                section((-2, 11, 6), (4, 11, 8)), section((1, 15, 7), (7, 15, 7))]

    def test_curve_is_connected_closed_and_has_nondegenerate_faces(self):
        sections = self.curve()
        vertices, faces = SB.ribbon_mesh(sections, 0.3, subdivisions=6)
        self.assertTrue(np.isfinite(vertices).all())
        self.assertEqual(len(vertices), ((len(sections) - 1) * 6 + 1) * 4)
        edges = Counter()
        adjacency = {i: set() for i in range(len(vertices))}
        for face in faces:
            pts = vertices[list(face)]
            area = np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
            area += np.linalg.norm(np.cross(pts[2] - pts[0], pts[3] - pts[0]))
            self.assertGreater(area, 1e-6)
            for a, b in zip(face, face[1:] + face[:1]):
                edges[tuple(sorted((a, b)))] += 1
                adjacency[a].add(b)
                adjacency[b].add(a)
        self.assertTrue(all(count == 2 for count in edges.values()))
        self.assertEqual(len(vertices) - len(edges) + len(faces), 2)
        visited, todo = set(), [0]
        while todo:
            v = todo.pop()
            if v not in visited:
                visited.add(v)
                todo.extend(adjacency[v] - visited)
        self.assertEqual(len(visited), len(vertices))
        np.testing.assert_allclose((vertices[0] + vertices[2]) / 2, [-3, 4, 1])
        np.testing.assert_allclose((vertices[-3] + vertices[-1]) / 2, [7, 15, 7])

    def test_bad_sections_are_rejected_not_coerced_into_origin_points(self):
        for rows in ([section((0, 0, 0), (0, 0, 0)), section((0, 1, 0), (1, 1, 0))],
                     [section((0, 0, 0), (2, 0, 0)), section((2, 2, 0), (0, 2, 0))],
                     [section((0, 0, 0), (2, 0, 0)), section((float("nan"), 2, 0), (2, 2, 0))]):
            clean, warning = SB.normalize_ribbons([{"sections": rows}])
            self.assertEqual(clean, [])
            self.assertTrue(warning)
            with self.assertRaises(ValueError):
                SB.ribbon_mesh(rows)

    def test_duplicate_section_is_removed_without_breaking_valid_ribbon(self):
        rows = self.curve()
        clean, warning = SB.normalize_ribbons([{"sections": [rows[0], rows[0]] + rows[1:]}])
        self.assertEqual(len(clean[0]["sections"]), 4)
        self.assertEqual(warning, [])

    def test_topology_note_survives_long_environment_description(self):
        data = SB.normalize({"ribbons": [{"label_zh": "扭转混凝土宽带", "sections": self.curve(), "thickness_m": 0.3}],
                             "environment_zh": "混凝土建筑场景" * 100,
                             "palette": {"lock_zh": "晴日暖光"}})
        note = SB.scene_note(data)
        self.assertIn("扭转混凝土宽带", note[:140])
        self.assertIn("晴日暖光", note)


class EnvironmentScoreTests(unittest.TestCase):
    def test_excluded_subject_does_not_hide_environment_errors(self):
        photo = np.full((180, 320, 3), [0.38, 0.32, 0.25], np.float32)
        photo[:70] = [0.55, 0.70, 0.95]
        photo[30:70, 40:100] = [0.3, 0.3, 0.32]
        render = np.full((180, 320, 4), [0.5, 0.5, 0.5, 1.0], np.float32)
        render[:70, :, 3] = 0
        render[30:70, 40:100, 3] = 1
        render[30:70, 40:100, :3] = 0.3
        changed = photo.copy()
        checker = np.indices((45, 120)).sum(0) % 2
        changed[115:160, 100:220] = checker[..., None]
        box = {"x_min": 100 / 320, "x_max": 220 / 320, "y_min": 115 / 180, "y_max": 160 / 180}
        baseline = SB.fit_score(photo, render, exclude_bbox=box)
        excluded = SB.fit_score(changed, render, exclude_bbox=box)
        self.assertAlmostEqual(baseline["score"], excluded["score"], places=3)
        bad_render = np.roll(render, 60, axis=1)
        bad = SB.fit_score(changed, bad_render, exclude_bbox=box)
        self.assertLess(bad["score"], excluded["score"] - 0.03)

    def test_full_exclusion_is_not_a_false_perfect_match(self):
        result = SB.fit_score(np.ones((18, 32, 3)), np.ones((18, 32, 4)), exclude_bbox=[0, 0, 1, 1])
        self.assertEqual(result["reason"], "NO_ENVIRONMENT_PIXELS")
        self.assertEqual(result["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
