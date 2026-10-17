"""Blender-side runner for tests/test_123_scene_blockout.py (no network).

    blender -b --factory-startup --python tests/sb_build_runner.py -- <fixture.json> <out.json> [product] [set_camera 0/1] [preview.png]

product: none | box (a 4.6 x 1.9 x 1.5 m car-sized box named Car at (3, -2, 0.4), rotated 30°) | chair (0.7 x 0.7 x 0.9 m)
Builds the blockout with wondful_ai_renderer.scene_blockout_build (package imported from the repo), checks
undo-free rebuild (a second build replaces the first), and dumps a summary JSON for the unittest to assert on."""
import json
import math
import os
import sys
import traceback

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
FIX, OUT = argv[0], argv[1]
PROD = argv[2] if len(argv) > 2 else "box"
SETCAM = (argv[3] if len(argv) > 3 else "1") == "1"
PREVIEW = argv[4] if len(argv) > 4 else ""
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def box(name, coll, w, d, h, loc, rz):
    me = bpy.data.meshes.new(name)
    me.from_pydata([(x * w / 2, y * d / 2, z * h) for x in (-1, 1) for y in (-1, 1) for z in (0, 1)], [],
                   [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.location = loc
    o.rotation_euler = (0, 0, math.radians(rz))
    return o


def main():
    rep = {"ok": False}
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        from wondful_ai_renderer import scene_blockout_build as SBB
        from wondful_ai_renderer import scene_blockout as SB
        from wondful_ai_renderer import massing as MS
        scene = bpy.context.scene
        scene.render.resolution_x, scene.render.resolution_y = 1600, 900
        data = json.load(open(FIX, encoding="utf-8"))
        prod = None
        if PROD != "none":
            prod = bpy.data.collections.new("产品")
            scene.collection.children.link(prod)
            if PROD == "chair":
                box("Chair", prod, 0.7, 0.7, 0.9, (1.0, 2.0, 0.0), 10)
            else:
                box("Car", prod, 1.9, 4.6, 1.5, (3.0, -2.0, 0.4), 30)
        user_cam = bpy.data.objects.new("UserCam", bpy.data.cameras.new("UserCam"))
        scene.collection.objects.link(user_cam)
        user_cam.location = (10, -10, 3)
        user_cam.rotation_euler = (math.radians(85), 0, math.radians(45))
        bpy.context.view_layer.update()
        user_cam_before = [list(r) for r in user_cam.matrix_world]
        scene.camera = user_cam
        parent = bpy.data.collections.new(MS.DEFAULT_COLLECTION)
        scene.collection.children.link(parent)
        bpy.context.view_layer.update()
        prod_before = [list(o.matrix_world.translation) for o in (prod.all_objects if prod else [])]
        info1 = SBB.build(bpy.context, data, product_collection=prod, parent_collection=parent, set_camera=SETCAM)
        n1 = len(bpy.data.collections[SB.COLLECTION_NAME].all_objects)
        info = SBB.build(bpy.context, data, product_collection=prod, parent_collection=parent, set_camera=SETCAM)
        coll = bpy.data.collections[SB.COLLECTION_NAME]
        rep["collections_named"] = sum(1 for c in bpy.data.collections if c.name.startswith(SB.COLLECTION_NAME))
        rep["rebuild_same_count"] = n1 == len(coll.all_objects)
        rep["info"] = info
        rep["first_blocks"] = info1["blocks"]
        rep["parent_ok"] = coll.name in [c.name for c in parent.children]
        rep["found_by_massing"] = MS.find_collection(scene, None) == parent
        objs = MS.massing_objects(parent, prod)
        roots = MS.block_roots(objs)
        rep["roots"] = len(roots)
        rep["root_hints"] = [MS.group_hint([r] + [o for o in objs if o.parent == r])[0] for r in roots]
        rep["root_names"] = [r.name for r in roots]
        rep["all_labelled"] = all(o.get("wondful_hint") and o.get(MS.AUTO_HINT_FLAG) for o in coll.all_objects)
        rep["terrain"] = [o.name for o in coll.all_objects if o.get("wondful_terrain")]
        rep["scene_note"] = MS.scene_note_of(objs)
        rep["env_types"] = sorted({o.get("wondful_env_type", "") for o in coll.all_objects})
        rep["wrap_copies"] = sum(1 for o in coll.all_objects if o.get("wondful_sb_wrap"))
        rep["product_unmoved"] = prod_before == [list(o.matrix_world.translation) for o in (prod.all_objects if prod else [])]
        rep["user_cam_unchanged"] = user_cam_before == [list(r) for r in user_cam.matrix_world]
        cam = bpy.data.objects.get(SB.CAMERA_NAME)
        rep["camera"] = None if cam is None else {"lens": cam.data.lens, "is_scene_camera": scene.camera == cam,
                                                   "loc": list(cam.matrix_world.translation),
                                                   "in_blockout": cam.name in coll.all_objects}
        rep["scene_camera"] = scene.camera.name if scene.camera else ""
        rep["text_block"] = SB.TEXT_NAME in bpy.data.texts
        # ground under the product: distance from the product base to the top surface below it
        if prod is not None:
            pts = [o.matrix_world @ Vector(c) for o in prod.all_objects for c in o.bound_box]
            cx = sum(p.x for p in pts) / len(pts)
            cy = sum(p.y for p in pts) / len(pts)
            zmin = min(p.z for p in pts)
            dg = bpy.context.evaluated_depsgraph_get()
            hidden = []
            for o in prod.all_objects:
                o.hide_viewport = True
                hidden.append(o)
            bpy.context.view_layer.update()
            hit = scene.ray_cast(bpy.context.evaluated_depsgraph_get(), Vector((cx, cy, zmin + 0.3)), Vector((0, 0, -1)))
            for o in hidden:
                o.hide_viewport = False
            rep["ground_gap"] = round(zmin - hit[1].z, 3) if hit[0] else None
            rep["ground_hit"] = hit[4].name if hit[0] else ""
        if PREVIEW:
            os.environ.setdefault("WONDFUL_SB_PREVIEW_ENGINE", "CYCLES")
            rep["preview"] = SBB.render_preview(bpy.context, info, PREVIEW, width=320)
            rep["scenes_after_preview"] = len(bpy.data.scenes)
        rep["ok"] = True
    except Exception:
        rep["error"] = traceback.format_exc()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=str)


main()
sys.stdout.flush()
os._exit(0)
