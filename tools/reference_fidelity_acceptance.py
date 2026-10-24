"""Native Blender CPU acceptance for reference framing, ribbons and input lighting.

Run from a checkout (no installation, network or AI generation required):
  blender -b --factory-startup --threads 2 --python-exit-code 1 \
    --python tools/reference_fidelity_acceptance.py -- --out /tmp/reference_acceptance \
    --scene-ref /path/to/user_reference.png

The rendered subjects are standard material samples and a calibration cuboid.
They are NOT a vehicle proxy or evidence of visual fidelity on the user's YU7.
The reference image is read only for provenance tests, never edited or analysed.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wondful_ai_renderer import operators as OPS
from wondful_ai_renderer import precise_depth as PD
from wondful_ai_renderer import scene_blockout as SB
from wondful_ai_renderer import scene_blockout_build as BUILD
from wondful_ai_renderer import viewport_capture as VC


def clear_scene():
    for obj in list(bpy.context.scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in list(bpy.context.scene.collection.children):
        bpy.context.scene.collection.children.unlink(collection)


def set_frame(frame, percentage=100):
    render = bpy.context.scene.render
    render.resolution_x, render.resolution_y = frame
    render.resolution_percentage = percentage
    render.pixel_aspect_x = render.pixel_aspect_y = 1.0
    render.threads_mode, render.threads = "FIXED", 2


def make_camera(name, spec):
    data = bpy.data.cameras.new(name)
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    data.type, data.sensor_fit, data.sensor_width = "PERSP", "HORIZONTAL", 36.0
    data.lens = 18.0 / math.tan(math.radians(spec["hfov_deg"]) / 2)
    data.shift_x = data.shift_y = 0.0
    obj.location = (0, 0, spec["height_m"])
    obj.rotation_euler = Euler((math.radians(90 + spec["pitch_deg"]),
                               math.radians(spec.get("roll_deg", 0)), 0), "XYZ")
    bpy.context.scene.camera = obj
    bpy.context.view_layer.update()
    return obj


def project_real(camera, points):
    result = [world_to_camera_view(bpy.context.scene, camera, Vector(p)) for p in points]
    return np.array([(p.x, 1.0 - p.y) for p in result], dtype=float)


def projection_check():
    clear_scene()
    scene = bpy.context.scene
    poses = ((0, 0), (0, -18), (0, 18), (-14, 0), (14, 0),
             (-19, 27), (19, -27), (26, 29), (-26, -29))
    frames = ((1600, 900), (900, 1600), (1000, 1000))
    points = ((0, 10, 0), (0, 10, 3), (-3, 12, 4), (4, 18, 2), (6, 8, 7), (-8, 10, 1))
    camera = make_camera("ProjectionOracle", dict(height_m=1.7, pitch_deg=0, roll_deg=0, hfov_deg=78))
    rows = []
    for pitch, roll in poses:
        spec = dict(height_m=1.7, pitch_deg=pitch, roll_deg=roll, hfov_deg=78)
        camera.rotation_euler = Euler((math.radians(90 + pitch), math.radians(roll), 0), "XYZ")
        bpy.context.view_layer.update()
        for frame in frames:
            set_frame(frame)
            actual = project_real(camera, points)
            predicted = np.array([PD.Projector(spec, frame)(p) for p in points])
            pixels = np.abs(actual - predicted) * np.array(frame)
            rows.append(dict(pitch_deg=pitch, roll_deg=roll, frame=list(frame),
                             max_abs_pixel_error=float(pixels.max())))
    maximum = max(r["max_abs_pixel_error"] for r in rows)
    return dict(ok=maximum < 0.01, measured_points=len(rows) * len(points),
                oracle="bpy_extras.object_utils.world_to_camera_view", tolerance_px=0.01,
                max_abs_pixel_error=maximum, cases=rows)


def socket_values(tree):
    if tree is None:
        return {}
    result = {}
    for node in tree.nodes:
        for socket in node.inputs:
            if not hasattr(socket, "default_value"):
                continue
            value = socket.default_value
            if isinstance(value, (bool, int, float, str)):
                result[node.name + "/" + socket.identifier] = value
            elif hasattr(value, "__iter__"):
                result[node.name + "/" + socket.identifier] = list(value)
    return result


def scene_state(scene):
    render = scene.render
    camera = scene.camera
    world = scene.world
    camera_data = camera.data if camera is not None else None
    return {
        "render": {k: getattr(render, k) for k in (
            "engine", "filepath", "resolution_x", "resolution_y", "resolution_percentage",
            "pixel_aspect_x", "pixel_aspect_y", "film_transparent", "use_border", "use_crop_to_border",
            "use_compositing", "use_sequencer", "border_min_x", "border_max_x", "border_min_y", "border_max_y")},
        "image_format": {k: getattr(render.image_settings, k) for k in
                         ("file_format", "color_mode", "color_depth", "compression")},
        "cycles": {k: getattr(scene.cycles, k) for k in
                   ("device", "samples", "use_denoising", "max_bounces", "seed", "use_animated_seed")},
        "view": {k: getattr(scene.view_settings, k) for k in ("view_transform", "look", "exposure", "gamma")},
        "world": None if world is None else dict(name=world.name, use_nodes=world.use_nodes,
                                                  sockets=socket_values(world.node_tree)),
        "camera": None if camera is None else dict(name=camera.name, matrix=[list(r) for r in camera.matrix_world],
            data={k: getattr(camera_data, k) for k in ("type", "lens", "sensor_fit", "sensor_width",
                                                       "sensor_height", "shift_x", "shift_y", "clip_start", "clip_end")}),
        "objects": {obj.name: dict(hide_render=obj.hide_render, hide_viewport=obj.hide_viewport,
            hidden=obj.hide_get(), matrix=[list(r) for r in obj.matrix_world],
            materials=[slot.material.name if slot.material else None for slot in obj.material_slots])
                    for obj in scene.objects},
        "lights": {obj.name: dict(type=obj.data.type, energy=obj.data.energy,
                                  color=list(obj.data.color), hide_render=obj.hide_render)
                   for obj in scene.objects if obj.type == "LIGHT"},
        "material_nodes": {mat.name: socket_values(mat.node_tree) for mat in bpy.data.materials},
        "datablocks": {kind: sorted(block.name for block in getattr(bpy.data, kind))
                       for kind in ("objects", "meshes", "materials", "lights", "cameras", "worlds")},
    }


def changed_state(before, after):
    return {key: dict(before=before[key], after=after[key]) for key in before if before[key] != after[key]}


def make_material(name, color, metallic=0.0, roughness=0.5, transmission=0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Transmission Weight"].default_value = transmission
    bsdf.inputs["Coat Weight"].default_value = 0.35 if metallic else 0.0
    return material


def put_in_collection(obj, collection):
    for old in list(obj.users_collection):
        old.objects.unlink(obj)
    collection.objects.link(obj)


def image_pixels(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = map(int, image.size)
        pixels = np.empty(width * height * 4, np.float32)
        image.pixels.foreach_get(pixels)
        return pixels.reshape(height, width, 4)
    finally:
        bpy.data.images.remove(image)


def lighting_check(out, temp):
    clear_scene()
    scene = bpy.context.scene
    samples = bpy.data.collections.new("DiagnosticMaterialSamples_NOT_A_VEHICLE")
    scene.collection.children.link(samples)
    materials = [make_material("Sample_GreenMetal", (0.03, 0.36, 0.065), 0.8, 0.2),
                 make_material("Sample_MatteGrey", (0.35, 0.35, 0.35), roughness=0.7),
                 make_material("Sample_Glass", (0.75, 0.85, 0.9), roughness=0.08, transmission=1.0)]
    for i, material in enumerate(materials):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.8,
                                           location=((i - 1) * 2.0, 0, 0.8))
        obj = bpy.context.object
        obj.name = "DiagnosticSphere_" + str(i)
        put_in_collection(obj, samples)
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
    camera = make_camera("LightingDiagnosticCamera", dict(height_m=3.1, pitch_deg=0, roll_deg=0, hfov_deg=55))
    camera.location = (4.5, -9.5, 4.0)
    camera.rotation_euler = (Vector((0, 0, 0.65)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    area = bpy.data.objects.new("UserStrongArea", bpy.data.lights.new("UserStrongArea", "AREA"))
    point = bpy.data.objects.new("UserStrongPoint", bpy.data.lights.new("UserStrongPoint", "POINT"))
    for obj in (area, point):
        scene.collection.objects.link(obj)
        obj.data.energy = 80000.0
    area.location, area.rotation_euler = (0, -2, 5), (0, 0, 0)
    area.data.shape, area.data.size = "RECTANGLE", 5.0
    area.data.size_y = 0.3
    point.location = (-3, -3, 3)
    point.data.color = (1.0, 0.05, 0.03)
    world = bpy.data.worlds.new("UserWorldMustSurvive")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.5, 0.02, 0.8, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 3.5
    scene.world = world
    set_frame((768, 432), 50)
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.filepath = str(temp / "user_original_output.png")
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "16"
    scene.render.film_transparent = True
    scene.render.use_border = scene.render.use_crop_to_border = True
    scene.render.border_min_x, scene.render.border_max_x = 0.2, 0.8
    scene.render.use_compositing = scene.render.use_sequencer = True
    scene.cycles.device = "CPU"
    scene.cycles.samples, scene.cycles.use_denoising = 13, True
    scene.cycles.seed, scene.cycles.use_animated_seed = 137, False
    scene.cycles.max_bounces = 6
    bpy.context.view_layer.update()
    os.environ["WONDFUL_CLAY_ENGINE"] = "CYCLES"
    os.environ["WONDFUL_CLAY_SAMPLES"] = "8"
    os.environ["WONDFUL_CLAY_DENOISE"] = "0"
    captures, restoration = [], {}
    original_render = VC._render_still
    mode = ["on"]

    def observed_render(active_scene):
        captures.append(dict(run=mode[0], lights=[dict(name=o.name, type=o.data.type, energy=o.data.energy)
            for o in active_scene.objects if o.type == "LIGHT" and not o.hide_render],
            world=active_scene.world.name))
        if mode[0] == "injected_failure":
            raise RuntimeError("reference acceptance injected render failure")
        original_render(active_scene)

    VC._render_still = observed_render
    final_png, control_png = out / "diagnostic_neutral_lighting.png", temp / "lights_off.png"
    rendered, failure_observed = {}, False
    try:
        for label, hidden, path in (("on", False, final_png), ("off", True, control_png)):
            mode[0] = label
            area.hide_render = point.hide_render = hidden
            before = scene_state(scene)
            rendered[label] = bool(VC.render_clay_base(bpy.context, str(path), samples, ground="AUTO"))
            restoration[label] = changed_state(before, scene_state(scene))
        mode[0] = "injected_failure"
        area.hide_render, point.hide_render = False, True
        before = scene_state(scene)
        try:
            VC.render_clay_base(bpy.context, str(temp / "must_not_render.png"), samples, ground="AUTO")
        except RuntimeError as exc:
            failure_observed = "injected render failure" in str(exc)
        restoration["injected_failure"] = changed_state(before, scene_state(scene))
    finally:
        VC._render_still = original_render
    a, b = image_pixels(final_png), image_pixels(control_png)
    difference = np.abs(a.astype(float) - b.astype(float))
    light_policy = all(len(row["lights"]) == 1 and row["lights"][0]["type"] == "SUN"
                       and row["lights"][0]["name"].startswith("_wondful_clay_sun") for row in captures)
    maximum = float(difference.max())
    return dict(ok=all(rendered.values()) and maximum < 1e-7 and light_policy and failure_observed
                and not any(restoration.values()), rendered=rendered,
                png=str(final_png), pixel_max_abs_difference=maximum,
                pixel_mean_abs_difference=float(difference.mean()), identical_pixel_count=int(np.all(a == b, axis=-1).sum()),
                total_pixels=int(a.shape[0] * a.shape[1]), active_lights=captures,
                no_temporary_area_or_strip_lights=light_policy, injected_failure_observed=failure_observed,
                state_changes_after_calls=restoration,
                scope="Standard material spheres only. No user vehicle or AI output was tested.")


def mesh_topology(obj):
    mesh = obj.data
    adjacency = [set() for _ in mesh.vertices]
    edge_uses = Counter()
    for face in mesh.polygons:
        indices = list(face.vertices)
        for a, b in zip(indices, indices[1:] + indices[:1]):
            adjacency[a].add(b)
            adjacency[b].add(a)
            edge_uses[tuple(sorted((a, b)))] += 1
    unseen, components = set(range(len(adjacency))), 0
    while unseen:
        components += 1
        pending = [unseen.pop()]
        while pending:
            for other in adjacency[pending.pop()] & unseen:
                unseen.remove(other)
                pending.append(other)
    return dict(vertices=len(mesh.vertices), faces=len(mesh.polygons), connected_components=components,
                nonmanifold_edges=sum(count != 2 for count in edge_uses.values()),
                euler_characteristic=len(mesh.vertices) - len(edge_uses) + len(mesh.polygons))


def ribbon_fixture():
    rows = []
    for x, y, z in ((-5, 8, 0.3), (-6, 11, 3), (-4, 16, 6), (-1, 20, 8)):
        rows.append(dict(left=dict(x=x - 1.6, y=y, z=z), right=dict(x=x + 1.6, y=y, z=z)))
    crossing = []
    for x, y, z in ((-7, 19, 4), (0, 18, 5), (7, 20, 7)):
        crossing.append(dict(left=dict(x=x, y=y - 1.1, z=z), right=dict(x=x, y=y + 1.1, z=z)))
    return [dict(id="curved", label_zh="诊断连续曲带", sections=rows, thickness_m=0.24, material="concrete"),
            dict(id="crossing", label_zh="诊断斜向曲带", sections=crossing, thickness_m=0.32, material="concrete")]


def geometry_check(out, temp, scene_ref):
    clear_scene()
    scene = bpy.context.scene
    frame = (640, 360)
    set_frame(frame)
    product = bpy.data.collections.new("CalibrationCuboid_NOT_A_VEHICLE")
    parent = bpy.data.collections.new("AcceptanceEnvironment")
    scene.collection.children.link(product)
    scene.collection.children.link(parent)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.85))
    box = bpy.context.object
    box.name = "FramingCalibrationCuboid"
    box.scale = (2.0, 5.0, 1.7)
    put_in_collection(box, product)
    bpy.context.view_layer.update()
    vertices_before = [list(v.co) for v in box.data.vertices]
    matrix_before = [list(r) for r in box.matrix_world]
    spec = dict(height_m=1.35, pitch_deg=5.0, roll_deg=7.0, hfov_deg=70.0)
    oracle_camera = make_camera(SB.CAMERA_NAME, spec)
    oracle_matrix_before = [list(r) for r in oracle_camera.matrix_world]
    expected_spot = Vector((-0.8, 13.0, 0))
    heading = 30.0
    rotation = Matrix.Rotation(math.radians(heading), 4, "Z")
    bounds_world = [box.matrix_world @ Vector(c) for c in box.bound_box]
    analysis_points = [rotation @ p + expected_spot for p in bounds_world]
    expected_uv = project_real(oracle_camera, analysis_points)
    lo, hi = expected_uv.min(0), expected_uv.max(0)
    expected_contact = project_real(oracle_camera, [expected_spot])[0]
    subject = dict(present=True, bbox=dict(x_min=float(lo[0]), y_min=float(lo[1]), x_max=float(hi[0]), y_max=float(hi[1])),
                   ground_contact=dict(x_frac=float(expected_contact[0]), y_frac=float(expected_contact[1])),
                   heading_deg=heading, heading_confidence=1.0)
    data = dict(camera=spec, terrain=dict(far_m=70, smoothness=0, surface="flat concrete"),
                product_spot=dict(x=2.0, y=24.0, rot_z_deg=-10.0), reference_subject=subject,
                ribbons=ribbon_fixture(), primitives=[])
    original_reference_bytes = None
    snapshot_path = temp / "source_reference_snapshot.png"
    if scene_ref is not None:
        original_reference_bytes = Path(scene_ref).read_bytes()
        snapshot_path.write_bytes(original_reference_bytes)
        data["_reference_image_path"] = str(snapshot_path.resolve())
        data["_reference_image_sha256"] = hashlib.sha256(original_reference_bytes).hexdigest()
    info = BUILD.build(bpy.context, data, product_collection=product, parent_collection=parent,
                       set_camera=True, frame=frame)
    collection = bpy.data.collections[info["collection"]]
    bpy.context.view_layer.update()
    actual = project_real(scene.camera, [box.matrix_world @ Vector(c) for c in box.bound_box])
    actual_contact = project_real(scene.camera, [(0, 0, 0)])[0]
    actual_lo, actual_hi = actual.min(0), actual.max(0)
    errors = dict(width=float((actual_hi[0] - actual_lo[0]) - (hi[0] - lo[0])),
                  center_x=float((actual_hi[0] + actual_lo[0] - hi[0] - lo[0]) / 2),
                  contact_x=float(actual_contact[0] - expected_contact[0]),
                  contact_y=float(actual_contact[1] - expected_contact[1]))
    topology = []
    inverse_world = Matrix(info["world"]).inverted()
    for obj in collection.all_objects:
        if not obj.get("wondful_sb_connected_surface"):
            continue
        source = next(r for r in data["ribbons"] if r["label_zh"] in obj.name)
        analysis_vertices = np.array([list(inverse_world @ obj.matrix_world @ v.co) for v in obj.data.vertices])
        thickness_errors = []
        for section in source["sections"]:
            for edge in ("left", "right"):
                midpoint = np.array([section[edge][key] for key in ("x", "y", "z")])
                nearest = np.argsort(np.linalg.norm(analysis_vertices - midpoint, axis=1))[:2]
                measured = np.linalg.norm(analysis_vertices[nearest[1]] - analysis_vertices[nearest[0]])
                thickness_errors.append(abs(float(measured) - source["thickness_m"]))
        row = mesh_topology(obj)
        row.update(name=obj.name, target_thickness_m=source["thickness_m"],
                   max_thickness_error_m=max(thickness_errors))
        topology.append(row)
    render_path = out / "diagnostic_connected_ribbons.png"
    os.environ["WONDFUL_SB_PREVIEW_ENGINE"] = "CYCLES"
    preview_ok = BUILD.render_preview(bpy.context, info, str(render_path), width=480)
    provenance = dict(skipped=scene_ref is None, reason="Pass --scene-ref for the source binding checks.")
    if scene_ref is not None:
        props = SimpleNamespace(massing_enabled=True, massing_collection=parent, product_collection=product,
                                sb_image_path=str(scene_ref))
        resolved, warning = OPS._sb_generation_reference(bpy.context, props)
        props.sb_image_path = str(render_path)
        after_ui_change, _ = OPS._sb_generation_reference(bpy.context, props)
        # A stale, invisible blockout must not supply the reference for separate
        # hand-built geometry that remains visible in the same environment root.
        manual_mesh = box.data.copy()
        manual = bpy.data.objects.new("Manual unrelated diagnostic block", manual_mesh)
        parent.objects.link(manual)
        original_visibility = [(obj, obj.hide_render) for obj in collection.all_objects]
        try:
            for obj, _hidden in original_visibility:
                obj.hide_render = True
            hidden_source, _ = OPS._sb_generation_reference(bpy.context, props)
        finally:
            for obj, hidden in original_visibility:
                obj.hide_render = hidden
            bpy.data.objects.remove(manual, do_unlink=True)
            bpy.data.meshes.remove(manual_mesh)
        snapshot_path.write_bytes(original_reference_bytes + b"acceptance tamper marker")
        rejected, rejection = OPS._sb_generation_reference(bpy.context, props)
        snapshot_path.write_bytes(original_reference_bytes)
        provenance = dict(skipped=False, original_source=str(Path(scene_ref).resolve()),
                          snapshot_hash=data["_reference_image_sha256"],
                          metadata_path_recorded=collection.get("wondful_sb_reference_image") == str(snapshot_path.resolve()),
                          metadata_hash_recorded=collection.get("wondful_sb_reference_sha256") == data["_reference_image_sha256"],
                          resolves_built_source=resolved == str(snapshot_path.resolve()),
                          ui_image_change_does_not_switch_source=after_ui_change == resolved,
                          hidden_blockout_ignored=not hidden_source,
                          tampered_snapshot_rejected=not rejected and bool(rejection), warning=warning,
                          rejection_message=rejection,
                          original_upload_untouched=Path(scene_ref).read_bytes() == original_reference_bytes)
    product_unchanged = vertices_before == [list(v.co) for v in box.data.vertices] and matrix_before == [list(r) for r in box.matrix_world]
    camera_preserved = oracle_matrix_before == [list(r) for r in oracle_camera.matrix_world] and scene.camera != oracle_camera
    topology_ok = len(topology) == 2 and all(row["connected_components"] == 1 and row["nonmanifold_edges"] == 0
        and row["euler_characteristic"] == 2 and row["faces"] > 10 and row["max_thickness_error_m"] < 1e-4 for row in topology)
    provenance_ok = provenance["skipped"] or all(provenance[key] for key in
        ("metadata_path_recorded", "metadata_hash_recorded", "resolves_built_source",
         "ui_image_change_does_not_switch_source", "hidden_blockout_ignored",
         "tampered_snapshot_rejected", "original_upload_untouched"))
    maximum = max(abs(value) for value in errors.values())
    return dict(ok=topology_ok and maximum < 0.002 and product_unchanged and camera_preserved and preview_ok and provenance_ok,
                png=str(render_path), ribbons=topology, framing_errors_fraction=errors,
                max_framing_error_fraction=maximum, framing_fit=info["framing_fit"],
                calibration_geometry_unchanged=product_unchanged, existing_same_name_camera_preserved=camera_preserved,
                reference_provenance=provenance, notes=info["notes"],
                scope="Hand-authored curved-surface fixture and cuboid projection only; not reconstruction of the uploaded scene or a YU7 render.")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene-ref")
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = dict(blender=bpy.app.version_string, renderer="Cycles CPU", render_threads=2,
                  real_user_vehicle_verified=False, ai_image_generation_performed=False,
                  limitation="Diagnostic fixtures only. Actual YU7 identity, matching the uploaded architecture and final reference-lit AI output remain unverified.",
                  checks={})
    with tempfile.TemporaryDirectory(prefix="reference_acceptance_", dir=out) as temp_dir:
        temp = Path(temp_dir)
        cases = (("projection", projection_check),
                 ("lighting_isolation_and_restore", lambda: lighting_check(out, temp)),
                 ("ribbon_geometry_framing_and_provenance", lambda: geometry_check(out, temp, args.scene_ref)))
        for name, fn in cases:
            start = time.monotonic()
            try:
                info = fn()
            except Exception as exc:
                info = dict(ok=False, error=repr(exc), traceback=traceback.format_exc())
            info["seconds"] = round(time.monotonic() - start, 3)
            report["checks"][name] = info
            print(("PASS " if info["ok"] else "FAIL ") + name, flush=True)
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["ok"] = all(info["ok"] for info in report["checks"].values())
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(report=str(out / "report.json"), ok=report["ok"])), flush=True)
    if not report["ok"]:
        raise RuntimeError("Reference fidelity acceptance failed; inspect report.json")


if __name__ == "__main__":
    main()
