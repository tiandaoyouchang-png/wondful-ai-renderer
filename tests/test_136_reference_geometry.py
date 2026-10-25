"""Reference-camera framing and connected architecture geometry, without network calls."""
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
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
        vertices, faces, layout = SB.ribbon_mesh(sections, 0.3, subdivisions=6,
                                                 width_subdivisions=3, return_layout=True)
        self.assertTrue(np.isfinite(vertices).all())
        columns = layout["columns"]
        self.assertEqual(len(vertices), ((len(sections) - 1) * 6 + 1) * 2 * columns)
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
        np.testing.assert_allclose((vertices[0] + vertices[columns]) / 2, [-3, 4, 1])
        np.testing.assert_allclose((vertices[-columns - 1] + vertices[-1]) / 2, [7, 15, 7])
        rows = vertices.reshape(layout["rows"], columns * 2, 3)
        np.testing.assert_allclose(np.linalg.norm(rows[:, :columns] - rows[:, columns:], axis=2), 0.3)

    def test_width_tessellation_resolves_twist_without_moving_source_edges(self):
        sections = [section((-20, 10, 0), (20, 10, 0)), section((-20, 11, 0), (20, 11, 4)),
                    section((-20, 12, 0), (20, 12, 8))]
        coarse, _, old = SB.ribbon_mesh(sections, 0.3, width_subdivisions=1, return_layout=True)
        dense, _, new = SB.ribbon_mesh(sections, 0.3, return_layout=True)
        self.assertGreater(old["max_quad_twist_deg"], 50)
        self.assertLess(new["max_quad_twist_deg"], 6.0)
        self.assertGreater(new["width_subdivisions"], 1)
        columns = new["columns"]
        grid = dense.reshape(new["rows"], columns * 2, 3)
        edges = grid[:, [0, columns - 1, columns, columns * 2 - 1]]
        np.testing.assert_allclose(edges, coarse.reshape(old["rows"], 4, 3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(grid[:, :columns] - grid[:, columns:], axis=2), 0.3)

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


def native_projection_regression(output):
    """A curved, rotated mesh exposes the phantom corners hidden by box tests."""
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    SBB = load_package_module("scene_blockout_build")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = FRAME
    product = bpy.data.collections.new("Projection regression")
    scene.collection.children.link(product)
    root = bpy.data.objects.new("Scaled rotated parent", None)
    product.objects.link(root)
    root.location = (0.7, 12.0, 0)
    root.rotation_euler.z = math.radians(48)
    root.scale = (1.3, 0.9, 1.7)
    n = 96
    ring = [(0.38 * math.cos(i * 2 * math.pi / n), 3 * math.sin(i * 2 * math.pi / n), 0) for i in range(n)]
    mesh = bpy.data.meshes.new("Long elliptical surface")
    mesh.from_pydata(ring + [(0, 0, 0)], [], [(n, i, (i + 1) % n) for i in range(n)])
    obj = bpy.data.objects.new("Curved non-box mesh", mesh)
    product.objects.link(obj)
    obj.parent = root
    modifier = obj.modifiers.new("Render-only thickness", "SOLIDIFY")
    modifier.thickness, modifier.offset = 0.6, 1.0
    modifier.show_viewport, modifier.show_render = False, True
    cam = bpy.data.objects.new("Known camera", bpy.data.cameras.new("Known camera"))
    scene.collection.objects.link(cam)
    camera = {"height_m": 1.6, "pitch_deg": -2.0, "roll_deg": 0.0, "hfov_deg": 70.0}
    cam.location = (0, 0, camera["height_m"])
    cam.rotation_euler = (math.radians(90 + camera["pitch_deg"]), 0, 0)
    cam.data.sensor_fit, cam.data.sensor_width = "HORIZONTAL", 36.0
    cam.data.lens = 18 / math.tan(math.radians(camera["hfov_deg"]) / 2)
    scene.camera = cam
    bpy.context.view_layer.update()
    before_matrix = np.asarray(obj.matrix_world).copy()
    geometry = SBB.product_projection_geometry(bpy.context, product)
    # Independent expected render surface: the elliptical ring extruded by the
    # render-only Solidify modifier, transformed through the nonuniform parent.
    expected = [obj.matrix_world @ Vector((x, y, z)) for x, y, _ in ring for z in (0.0, 0.6)]

    def projected_bbox(points):
        uv = np.asarray([(p.x, 1 - p.y) for point in points
                         for p in [world_to_camera_view(scene, scene.camera, Vector(point))]])
        return np.concatenate((uv.min(0), uv.max(0)))
    precise_bbox = projected_bbox(geometry["points"])
    expected_bbox = projected_bbox(expected)
    old_bbox = projected_bbox(SBB.product_info(product)["bounds"])
    anchor = world_to_camera_view(scene, cam, root.location)
    subject = {"present": True, "bbox": dict(zip(("x_min", "y_min", "x_max", "y_max"), expected_bbox)),
               "ground_contact": {"x_frac": anchor.x, "y_frac": 1 - anchor.y},
               "heading_deg": 0, "heading_confidence": 0}
    prod = SBB.product_info(product)
    source = {"camera": camera, "reference_subject": subject,
              "product_spot": {"x": -3, "y": 30, "rot_z_deg": math.degrees(prod["axis"] or 0)},
              "terrain": {"amplitude_m": 0}, "primitives": [], "ribbons": []}
    built = SBB.build(bpy.context, source, product_collection=product, frame=FRAME)
    actual_bbox = projected_bbox(expected)
    # A planar wide skin must keep its own normal at the thin rim as well as
    # in its interior; the former all-smooth closed shell tilted it by 45°.
    ribbons = bpy.data.collections.new("Rim normal regression")
    scene.collection.children.link(ribbons)
    builder = SBB._Builder({"ribbons": [{"sections": [section((-4, 8, 2), (4, 8, 2)),
                                                                 section((-4, 12, 2), (4, 12, 2))]}]}, ribbons, FRAME)
    builder.foot = 0.0
    builder.ribbons()
    ribbon = next(o for o in ribbons.objects if o.type == "MESH")
    skin_faces = ribbon["wondful_sb_skin_faces"]
    corner_deviation = []
    for face in ribbon.data.polygons[:skin_faces]:
        for loop in face.loop_indices:
            dot = np.clip(np.dot(face.normal, ribbon.data.corner_normals[loop].vector), -1, 1)
            corner_deviation.append(math.degrees(math.acos(dot)))
    result = {"geometry_basis": geometry["basis"], "fallbacks": geometry["fallbacks"],
              "hull_bbox_max_error": float(np.max(np.abs(precise_bbox - expected_bbox))),
              "old_bbox_width_error": float((old_bbox[2] - old_bbox[0]) - (expected_bbox[2] - expected_bbox[0])),
              "built_bbox_max_error": float(np.max(np.abs(actual_bbox - expected_bbox))),
              "built_width_error": float(abs((actual_bbox[2] - actual_bbox[0]) - (expected_bbox[2] - expected_bbox[0]))),
              "reported_bbox_max_error": float(np.max(np.abs(actual_bbox - np.asarray(list(built["framing_fit"]["metrics"]["bbox"].values()))))),
              "built_framing": built["framing_fit"],
              "modifier_restored": modifier.show_viewport is False and modifier.show_render is True,
              "matrix_unchanged": np.array_equal(before_matrix, np.asarray(obj.matrix_world)),
              "source_vertices_unchanged": len(mesh.vertices) == n + 1}
    result["ribbon_skin_normal_max_deviation_deg"] = max(corner_deviation)
    result["ribbon_sharp_edge_count"] = sum(e.use_edge_sharp for e in ribbon.data.edges)
    result["ribbon_rim_flat"] = all(not p.use_smooth for p in ribbon.data.polygons[skin_faces:])
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")


@unittest.skipUnless(shutil.which("blender") and os.environ.get("WONDFUL_SKIP_BLENDER_TESTS") != "1",
                     "blender not on PATH")
class EvaluatedProjectionBlenderTests(unittest.TestCase):
    def test_rotated_curved_mesh_render_modifier_and_parent_scale(self):
        with tempfile.TemporaryDirectory(prefix="wondful_projection_") as tmp:
            output = Path(tmp) / "projection.json"
            run = subprocess.run([shutil.which("blender"), "--background", "--factory-startup",
                                  "--python-exit-code", "1", "--python", str(Path(__file__).resolve()),
                                  "--", "--native-projection-regression", str(output)],
                                 capture_output=True, text=True, timeout=180)
            self.assertEqual(run.returncode, 0, run.stdout[-5000:] + run.stderr[-3000:])
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["geometry_basis"], "EVALUATED_MESH_CONVEX_HULL")
        self.assertEqual(result["fallbacks"], [])
        self.assertGreater(result["old_bbox_width_error"], 0.025)
        self.assertLess(result["hull_bbox_max_error"], 1e-6)
        # Terrain BVH height uses float32 and can shift vertical framing by a
        # fraction of a pixel. The actual reported residual must match it.
        self.assertLess(result["built_bbox_max_error"], 0.002)
        self.assertLess(result["built_width_error"], 1e-5)
        self.assertLess(result["reported_bbox_max_error"], 1e-6)
        self.assertEqual(result["built_framing"]["geometry_basis"], "EVALUATED_MESH_CONVEX_HULL")
        self.assertTrue(result["modifier_restored"])
        self.assertTrue(result["matrix_unchanged"])
        self.assertTrue(result["source_vertices_unchanged"])
        self.assertLess(result["ribbon_skin_normal_max_deviation_deg"], 0.01)
        self.assertGreater(result["ribbon_sharp_edge_count"], 0)
        self.assertTrue(result["ribbon_rim_flat"])


if __name__ == "__main__":
    if "--native-projection-regression" in sys.argv:
        native_projection_regression(sys.argv[-1])
    else:
        unittest.main()
