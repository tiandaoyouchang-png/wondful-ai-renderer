"""Blender-side runner for tests/test_129_interior_neon.py (no network).

    blender -b --factory-startup --python tests/interior_occlusion_runner.py -- <case> <out.json> [fixture.json]

case:
  street    street_v4 + car box + 场景补全-like objects 「补全_推测建筑」「补全_推测窗」 (tagged wondful_detail_inferred,
            window / house hints): 室内判定 must stay outdoor (no enclosure walls, no 室内物体 wording);
  street_untagged  the same street with the ④ kind tag, the analysis text and the inferred objects removed and two
            own blocks named 「墙」「窗」 added far away: names alone must not make it a room;
  interior  interior_v3 + chair: still a room (closure, 室内物体), via the analysis tag;
  interior_untagged  interior_v3 with the kind tag and the analysis text removed: shape / rays still say room;
  occlusion empty scene, car box, a wall between the camera and the car → camera moved closer, product clear,
            the original camera restored from its snapshot; a closed ring of walls → blocked (skip).
"""
import json
import math
import os
import sys
import traceback

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
CASE, OUT = argv[0], argv[1]
FIX = argv[2] if len(argv) > 2 else ""
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def box(name, coll, w, d, h, loc, rz=0.0):
    me = bpy.data.meshes.new(name)
    me.from_pydata([(x * w / 2, y * d / 2, z * h) for x in (-1, 1) for y in (-1, 1) for z in (0, 1)], [],
                   [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.location = loc
    o.rotation_euler = (0, 0, math.radians(rz))
    return o


def camera_at(name, pos, target, lens=30.0):
    cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    bpy.context.scene.collection.objects.link(cam)
    cam.location = pos
    cam.rotation_euler = (Vector(target) - Vector(pos)).to_track_quat("-Z", "Y").to_euler()
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = 36.0
    cam.data.lens = lens
    bpy.context.view_layer.update()
    return cam


def build(fixture, product):
    from wondful_ai_renderer import scene_blockout_build as SBB
    from wondful_ai_renderer import massing as MS
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = 1600, 900
    data = json.load(open(fixture, encoding="utf-8"))
    prod = bpy.data.collections.new("产品")
    scene.collection.children.link(prod)
    if product == "chair":
        box("Chair", prod, 0.7, 0.7, 0.9, (1.0, 2.0, 0.0), 10)
    else:
        box("Car", prod, 1.9, 4.6, 1.5, (3.0, -2.0, 0.4), 30)
    parent = bpy.data.collections.new(MS.DEFAULT_COLLECTION)
    scene.collection.children.link(parent)
    bpy.context.view_layer.update()
    info = SBB.build(bpy.context, data, product_collection=prod, parent_collection=parent, set_camera=True)
    return prod, parent, info


def interior_report(prod, parent, rep):
    from wondful_ai_renderer import massing as MS, multiview as MV
    scene = bpy.context.scene
    blocks = MS.massing_objects(parent, prod)
    rep["kind"] = MS.blockout_kind(blocks)
    made, wlog = MV.widen_blockout(bpy.context, parent, prod, scene.camera, fillers=False)
    rep["widen"] = {k: v for k, v in wlog.items()}
    # what the other cameras' prompt says while the temporary blocks are in place
    pb = MV.product_bounds(prod)
    mn, mx = pb
    c = (mn + mx) / 2
    other = camera_at("Other", c + Vector((0.0, -6.0, 1.4)) if True else None, c)
    old = scene.camera
    scene.camera = other
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ Vector(cc) for o in prod.all_objects for cc in o.bound_box]
    items = MS.describe_blocks(bpy.context, MS.massing_objects(parent, prod), pts)
    text = MS.build_massing_text(items)
    rep["items"] = len(items)
    rep["items_interior"] = sum(1 for i in items if i.get("interior"))
    rep["text_interior"] = MS.INTERIOR_MARK in text
    scene.camera = old
    MV.remove_aux(made)
    rep["aux_left"] = sum(1 for o in bpy.data.objects if o.get(MV.AUX_TAG))


def add_inferred(parent):
    sub = bpy.data.collections.new("Wondful 场景补全（推测）")
    sub["wondful_detail_inferred"] = 1
    parent.children.link(sub)
    for nm, hint, loc in (("补全_推测建筑", "与主视角同一条街的房屋（推测补全）：同样的类型、颜色范围、层数、窗户和屋顶，不要画成空白墙",
                           (-12.0, 30.0, 0.0)),
                          ("补全_推测窗", "房屋窗户（推测补全）", (12.0, 30.0, 2.0))):
        o = box(nm, sub, 6.0, 8.0, 7.0, loc)
        o["wondful_detail_inferred"] = 1
        o["wondful_hint"] = hint


def untag(parent):
    from wondful_ai_renderer import massing as MS
    for c in bpy.data.collections:
        if MS.SB_KIND_PROP in c.keys():
            del c[MS.SB_KIND_PROP]
    t = bpy.data.texts.get(MS.SB_TEXT)
    if t is not None:
        bpy.data.texts.remove(t)


def same(rows, M, tol=1e-5):
    return all(abs(a - b) < tol for r0, r1 in zip(rows, M) for a, b in zip(r0, r1))


def occlusion_case(rep):
    from wondful_ai_renderer import multiview as MV
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = 1600, 900
    prod = bpy.data.collections.new("产品")
    scene.collection.children.link(prod)
    box("Car", prod, 1.9, 4.6, 1.5, (0.0, 0.0, 0.0))
    env = bpy.data.collections.new("环境")
    scene.collection.children.link(env)
    box("地面", env, 200.0, 200.0, 0.02, (0.0, 0.0, -0.02))
    wall = box("右侧商楼", env, 0.6, 30.0, 9.0, (5.0, 0.0, 0.0))        # a facade between the camera and the car
    cam = camera_at("Orbit_090", (12.0, 0.0, 1.8), (0.0, 0.0, 0.8), lens=30.0)
    before = [list(r) for r in cam.matrix_world]
    snap = MV.snapshot_camera(cam)
    occ = MV.adjust_camera_for_occlusion(bpy.context, cam, prod)
    rep["moved"] = occ
    rep["moved_matrix_changed"] = before != [list(r) for r in cam.matrix_world]
    MV.restore_camera(snap)
    bpy.context.view_layer.update()
    rep["restored"] = same(before, cam.matrix_world) and abs(cam.data.lens - 30.0) < 1e-6
    clear = camera_at("Orbit_000", (0.0, -12.0, 1.8), (0.0, 0.0, 0.8), lens=30.0)
    rep["clear"] = MV.adjust_camera_for_occlusion(bpy.context, clear, prod)
    # tight closed ring (closer than any allowed camera distance): no clear pose → blocked, camera untouched
    wall.location.x = 1.6
    for nm, w, d, loc in (("左墙", 0.6, 30.0, (-1.6, 0.0, 0.0)), ("前墙", 30.0, 0.6, (0.0, 2.9, 0.0)),
                          ("后墙", 30.0, 0.6, (0.0, -2.9, 0.0))):
        box(nm, env, w, d, 9.0, loc)
    bpy.context.view_layer.update()
    cam2 = camera_at("Orbit_045", (9.0, 9.0, 1.8), (0.0, 0.0, 0.8), lens=30.0)
    before2 = [list(r) for r in cam2.matrix_world]
    rep["blocked"] = MV.adjust_camera_for_occlusion(bpy.context, cam2, prod)
    rep["blocked_untouched"] = same(before2, cam2.matrix_world)


def main():
    rep = {"ok": False, "case": CASE}
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if CASE == "occlusion":
            occlusion_case(rep)
        else:
            product = "chair" if CASE.startswith("interior") else "box"
            prod, parent, info = build(FIX, product)
            rep["analysis_kind"] = info["kind"]
            if CASE == "street":
                add_inferred(parent)
            elif CASE == "street_untagged":
                untag(parent)
                for nm, loc in (("墙", (-40.0, 60.0, 0.0)), ("窗", (40.0, 60.0, 0.0))):
                    box(nm, parent, 4.0, 4.0, 5.0, loc)
            elif CASE == "interior_untagged":
                untag(parent)
            bpy.context.view_layer.update()
            interior_report(prod, parent, rep)
        rep["ok"] = True
    except Exception:
        rep["error"] = traceback.format_exc()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=str)


main()
sys.stdout.flush()
os._exit(0)
