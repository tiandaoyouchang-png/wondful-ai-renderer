"""Blender-side runner for tests/test_123_hull_placeholder.py (no network).

    blender -b --factory-startup --python tests/hull_placeholder_runner.py -- <scene_blockout_fixture.json> <out.json> <tmpdir>

Registers the add-on from the repo, builds a 场景粗模 from the fixture around a car-sized product, adds the
user's own collection 「新建筑」 (scene root, outside 环境体块) with a rotated, non-uniformly scaled cube, writes
synthetic three-view PNGs, then clicks 「从三视图生成建筑体块」 (the real operator) in these cases:
a) cube selected (按比例) → block at the cube, fitted;  b) the new block selected → replaced in place;
c) 长宽高 mode with a cube inside 环境体块 → size kept;  d) nothing selected → 3D cursor;
e) a 场景粗模 block selected → not a placeholder (cursor)."""
import json
import math
import os
import sys
import traceback

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
FIX, OUT, TMP = argv[0], argv[1], argv[2]
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


def cube(name, coll, loc, rz_deg, scale):
    """A default 2 m cube (origin at its centre), like Shift+A › Cube."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], [],
                   [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.location = loc
    o.rotation_euler = (0, 0, math.radians(rz_deg))
    o.scale = scale
    return o


def views(png_io):
    """front: a stepped tower (wide base, narrow top); side: a rectangle; top: a rectangle (white ground)."""
    paths = {}
    f = np.full((420, 360, 3), 255, np.uint8)
    f[200:400, 30:330] = (60, 80, 140)      # base 300 wide, 200 high
    f[40:200, 110:250] = (60, 80, 140)      # tower 140 wide
    s = np.full((420, 260, 3), 255, np.uint8)
    s[40:400, 30:230] = (60, 80, 140)
    t = np.full((260, 360, 3), 255, np.uint8)
    t[30:230, 30:330] = (60, 80, 140)
    for k, a in (("front", f), ("side", s), ("top", t)):
        paths[k] = os.path.join(TMP, f"hull_{k}.png")
        png_io.write_png(paths[k], a)
    return paths


def select_only(obj):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    if obj is not None:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def own_box(MS, obj, yaw):
    corners = [tuple(obj.matrix_world @ Vector(c)) for c in obj.bound_box]
    return MS.placeholder_box(corners, yaw)


def report(MS, obj, ph_box, props):
    bpy.context.view_layer.update()
    yaw = MS.yaw_of(obj.matrix_world)
    ob = own_box(MS, obj, ph_box["yaw"] if ph_box else yaw)
    return {"name": obj.name, "yaw_deg": round(math.degrees(yaw), 4), "scale": [round(v, 6) for v in obj.scale],
            "bottom_z": ob["bottom"], "centre_xy": ob["centre"], "dims_own": ob["dims"],
            "location": [round(v, 6) for v in obj.location],
            "collections": sorted(c.name for c in obj.users_collection), "extra_tag": bool(obj.get(MS.ENV_EXTRA_TAG)),
            "placeholder_prop": str(obj.get(MS.PLACEHOLDER_PROP, "")), "summary": props.last_hull_note}


def main():
    rep = {"ok": False}
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        import wondful_ai_renderer as ADDON
        ADDON.register()
        from wondful_ai_renderer import scene_blockout_build as SBB
        from wondful_ai_renderer import scene_blockout as SB
        from wondful_ai_renderer import massing as MS
        from wondful_ai_renderer import massing_hull as MH
        from wondful_ai_renderer import multiview as MV
        from wondful_ai_renderer import png_io
        scene = bpy.context.scene
        props = scene.wondful_ai
        scene.render.resolution_x, scene.render.resolution_y = 1600, 900
        prod = bpy.data.collections.new("产品")
        scene.collection.children.link(prod)
        car = box("Car", prod, 1.9, 4.6, 1.5, (3.0, -2.0, 0.4), 30)
        props.product_collection = prod
        parent = bpy.data.collections.new(MS.DEFAULT_COLLECTION)
        scene.collection.children.link(parent)
        bpy.context.view_layer.update()
        car_before = [list(r) for r in car.matrix_world]
        SBB.build(bpy.context, json.load(open(FIX, encoding="utf-8")), product_collection=prod,
                  parent_collection=parent, set_camera=True)
        bpy.context.view_layer.update()
        p = views(png_io)
        props.hull_input = "SEPARATE"
        props.hull_front_path, props.hull_side_path, props.hull_top_path = p["front"], p["side"], p["top"]
        props.hull_resolution = 64
        cam = scene.camera
        fwd = cam.matrix_world.to_3x3() @ Vector((0, 0, -1))
        fwd.z = 0
        fwd.normalize()
        right = Vector((fwd.y, -fwd.x, 0))
        base = car.matrix_world.translation
        spot = Vector((base.x, base.y, 0.0)) + fwd * 30 + right * 6
        user = bpy.data.collections.new("新建筑")
        scene.collection.children.link(user)                     # outside 环境体块 on purpose
        ph = cube("占位Cube", user, (spot.x, spot.y, 4.5), 30, (6, 4, 4.5))   # 12 x 8 x 9 m box, bottom on z = 0
        bpy.context.view_layer.update()
        ph_box = own_box(MS, ph, MS.yaw_of(ph.matrix_world))

        # a) placeholder, 按比例
        props.hull_size_mode = "AUTO"
        select_only(ph)
        rep["a_mode"] = MH.find_target(bpy.context, props)[0]
        res = bpy.ops.wondful.massing_hull_build()
        rep["a_result"] = list(res)
        new = bpy.context.view_layer.objects.active
        a = report(MS, new, ph_box, props)
        a.update({"ph_bottom": ph_box["bottom"], "ph_centre": ph_box["centre"], "ph_dims": ph_box["dims"],
                  "ph_hide_render": ph.hide_render, "ph_hide_viewport": ph.hide_viewport,
                  "ph_exists": ph.name in bpy.data.objects})
        mcoll = MS.find_collection(scene, props)
        objs = MS.massing_objects(mcoll, prod)
        a["in_massing"] = new in objs
        a["ph_in_massing"] = ph in objs
        roots = MS.block_roots(objs)
        a["label_new"] = MS.block_ref_label(new.name, len(MS.get_refs(new)), MS.is_scene_blockout_block(new)).split(" · ")[1]
        sb_root = next(r for r in roots if MS.is_scene_blockout_block(r) and not MS.get_refs(r))
        a["label_sb"] = MS.block_ref_label(sb_root.name, 0, MS.is_scene_blockout_block(sb_root)).split(" · ")[1]
        pts = [car.matrix_world @ Vector(c) for c in car.bound_box]
        items = MS.describe_blocks(bpy.context, objs, pts)
        a["prompt_names"] = [i["name"] for i in items]
        a["in_prompt_items"] = new.name in a["prompt_names"]
        a["hint"] = next((i["hint"] for i in items if i["name"] == new.name), "")
        a["prompt_text"] = MS.build_massing_text(items, vehicle=True)
        a["multiview_words"] = MV._block_words(new)
        a["massing_collection"] = mcoll.name if mcoll else ""
        rep["a"] = a

        # b) re-run with the new block selected → replaced at the same placeholder transform
        props.hull_resolution = 48
        select_only(new)
        rep["b_mode"] = MH.find_target(bpy.context, props)[0]
        bpy.ops.wondful.massing_hull_build()
        newb = bpy.context.view_layer.objects.active
        b = report(MS, newb, ph_box, props)
        b["hull_blocks"] = sum(1 for o in bpy.data.objects if o.get(MS.HULL_TAG))
        rep["b"] = b

        # c) 长宽高 mode, placeholder inside 环境体块 → size kept, only position / rotation
        ph2 = cube("占位Cube.001", parent, (spot.x - 40 * right.x, spot.y - 40 * right.y, 1.0), -45, (1, 1, 1))
        bpy.context.view_layer.update()
        ph2_box = own_box(MS, ph2, MS.yaw_of(ph2.matrix_world))
        props.hull_size_mode = "MANUAL"
        props.hull_width, props.hull_length, props.hull_height = 10.0, 8.0, 15.0
        select_only(ph2)
        bpy.ops.wondful.massing_hull_build()
        c = report(MS, bpy.context.view_layer.objects.active, ph2_box, props)
        c.update({"ph_centre": ph2_box["centre"], "massing_name": parent.name})
        rep["c"] = c

        # d) nothing selected → 3D cursor, on the ground
        props.hull_size_mode = "AUTO"
        select_only(None)
        scene.cursor.location = (7.0, 8.0, 3.0)
        rep["d_info_mode"] = MH.find_target(bpy.context, props)[0]
        bpy.ops.wondful.massing_hull_build()
        d = report(MS, bpy.context.view_layer.objects.active, None, props)
        d["mode"] = rep["d_info_mode"]
        rep["d"] = d

        # e) a 场景粗模 block selected → not a placeholder
        sb = next(o for o in bpy.data.collections[SB.COLLECTION_NAME].all_objects if o.type == "MESH")
        select_only(sb)
        rep["e"] = {"mode": MH.find_target(bpy.context, props)[0]}
        bpy.ops.wondful.massing_hull_build()
        rep["e"]["sb_hidden"] = bool(sb.hide_render or sb.hide_viewport)
        rep["product_unmoved"] = car_before == [list(r) for r in car.matrix_world]
        rep["undo_poll"] = bool(bpy.ops.ed.undo.poll())
        rep["ok"] = True
    except Exception:
        rep["error"] = traceback.format_exc()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=str)


main()
sys.stdout.flush()
os._exit(0)
