"""环境体块 › 从三视图生成建筑体块 (Blender side).

The geometry work is in sketch_model (numpy only: silhouettes, sheet split, visual-hull carving,
surface nets); this module loads images through bpy (so JPG works without Pillow), builds a
grey-blue block in the massing collection (「环境体块」 when none exists) at the 3D cursor on the
ground, and stores the three views as that block's 建筑参考.  The product is never touched.

Placeholder (1.2.3): when the active, selected object is a plain mesh of the user's (not a Wondful hull /
场景粗模 block, not the product), it is a placeholder: the block takes its position (bottom on its bottom,
centred in XY) and Z rotation, is scaled uniformly to fit inside it in 按比例 mode (长宽高 mode keeps the
given size), goes into the placeholder's collection (tagged ``ENV_EXTRA_TAG`` so it counts as massing when
that collection is outside the massing one), and the placeholder is hidden (not deleted).  With a hull
block selected (or its hidden placeholder), the block is replaced at the same place.
"""
from __future__ import annotations

import time
import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from . import massing as MS
from . import sketch_model as SM

OBJECT_NAME = "建筑体块"
MATERIAL_NAME = "Wondful 环境体块"


def image_to_rgba(path, max_side: int = 1024) -> np.ndarray:
    """Float RGBA top-row-first.  PNG through png_io (fast, exact alpha), else bpy."""
    p = Path(bpy.path.abspath(str(path)))
    if not p.is_file():
        raise ValueError(f"找不到图片：{p}")
    if p.read_bytes()[:8].startswith(b"\x89PNG"):
        try:
            return SM.load_rgba(str(p), max_side)
        except Exception:
            pass  # interlaced or unusual PNG: let Blender decode it
    img = bpy.data.images.load(str(p), check_existing=False)
    try:
        w, h = int(img.size[0]), int(img.size[1])
        if w == 0 or h == 0:
            raise ValueError(f"无法读取图片：{p.name}")
        buf = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(buf)
        arr = buf.reshape(h, w, 4)[::-1]
        return SM.to_rgba_float(arr, max_side)
    finally:
        bpy.data.images.remove(img)


def view_paths(props):
    """[(view, path)] the user supplied (absolute paths, existing files only)."""
    out = []
    if getattr(props, "hull_input", "SEPARATE") == "SHEET":
        p = bpy.path.abspath(getattr(props, "hull_sheet_path", "") or "")
        return [("sheet", p)] if p and Path(p).is_file() else []
    for view in SM.VIEWS:
        p = bpy.path.abspath(getattr(props, f"hull_{view}_path", "") or "")
        if p and Path(p).is_file():
            out.append((view, p))
    return out


def silhouettes(props):
    """{view: cropped bool mask} + info from the props' images."""
    info = {}
    if getattr(props, "hull_input", "SEPARATE") == "SHEET":
        paths = view_paths(props)
        if not paths:
            raise ValueError("请先选择三视图拼图")
        views, sinfo = SM.split_sheet(image_to_rgba(paths[0][1], 1600))
        info["sheet"] = sinfo
        masks = {}
        for name, (m, _rgba) in views.items():
            masks[name], _box = SM.crop_to_mask(m)
        return masks, info
    masks = {}
    for view, path in view_paths(props):
        masks[view], info[view] = SM.view_silhouette(image_to_rgba(path))
    missing = [SM.VIEW_LABELS[v] for v in ("front", "side") if v not in masks]
    if missing:
        raise ValueError("缺少" + "、".join(missing) + "（俯视图可选）")
    return masks, info


def block_material():
    """Persistent viewport material for new blocks (the clay base swaps in its own grey-blue)."""
    mat = bpy.data.materials.get(MATERIAL_NAME)
    if mat is None:
        mat = MS.make_material(bpy.data.materials.new)
        mat.name = MATERIAL_NAME
    return mat


def _in_product(obj, props):
    prod = getattr(props, "product_collection", None)
    try:
        return prod is not None and obj.name in prod.all_objects
    except Exception:
        return False


def find_target(context, props):
    """('placeholder' | 'replace' | 'cursor', object or None) from the active, selected object."""
    obj = getattr(context, "active_object", None)
    if obj is None or obj.type != "MESH":
        return "cursor", None
    try:
        selected = obj.select_get()
    except Exception:
        selected = False
    if not selected or _in_product(obj, props):
        return "cursor", None
    if obj.get(MS.HULL_TAG):
        return "replace", obj
    linked = bpy.data.objects.get(str(obj.get(MS.PLACEHOLDER_FOR_PROP, "") or ""))
    if linked is not None and linked.get(MS.HULL_TAG):
        return "replace", linked          # the hidden placeholder picked again (outliner): redo its block
    if MS.is_scene_blockout_block(obj) or obj.name.startswith("_wondful_") or obj.get("wondful_mv_aux"):
        return "cursor", None
    return "placeholder", obj


def _box_of_placeholder(ph):
    corners = [tuple(ph.matrix_world @ Vector(c)) for c in ph.bound_box]
    return MS.placeholder_box(corners, MS.yaw_of(ph.matrix_world))


def _apply(obj, tf):
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (0.0, 0.0, tf["yaw"])
    obj.scale = (tf["scale"],) * 3
    obj.location = tf["location"]


def _in_massing(coll, massing_coll):
    return coll == massing_coll or coll in massing_coll.children_recursive


def build_block(context, props):
    """Carve a building block from the three views.  Returns (object, info dict)."""
    t0 = time.perf_counter()
    mode, target = find_target(context, props)
    masks, info = silhouettes(props)
    paths = [p for _v, p in view_paths(props)]
    size_mode = getattr(props, "hull_size_mode", "AUTO")
    verts, quads, hinfo = SM.build_hull(
        masks["front"], masks["side"], masks.get("top"),
        size_mode=size_mode,
        known_axis=getattr(props, "hull_known_axis", "HEIGHT"),
        known_value=float(getattr(props, "hull_known_value", 20.0)),
        length=float(getattr(props, "hull_length", 20.0)),
        width=float(getattr(props, "hull_width", 20.0)),
        height=float(getattr(props, "hull_height", 20.0)),
        side_front=getattr(props, "hull_side_front", "LEFT"),
        top_front=getattr(props, "hull_top_front", "BOTTOM"),
        resolution=int(getattr(props, "hull_resolution", 128)),
    )
    info.update(hinfo)
    info["mode"] = mode
    scene = context.scene
    if hasattr(props, "massing_enabled"):
        props.massing_enabled = True
    coll, created = MS.ensure_collection(scene, props)
    info["collection_created"] = bool(created)
    # where it goes: placeholder box / the replaced block's place / 3D cursor
    ph, box, old = None, None, None
    if mode == "placeholder":
        ph = target
        box = _box_of_placeholder(ph)
    elif mode == "replace":
        old = target
        ph = bpy.data.objects.get(str(old.get(MS.PLACEHOLDER_PROP, "") or ""))
        if ph is not None and ph.get(MS.HULL_TAG):
            ph = None
        if ph is not None:
            box = _box_of_placeholder(ph)
        elif old.get(MS.PLACEHOLDER_BOX_PROP):
            try:
                box = json.loads(old[MS.PLACEHOLDER_BOX_PROP])
            except Exception:
                box = None
    mesh = bpy.data.meshes.new(OBJECT_NAME)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    n = len(quads)
    mesh.loops.add(n * 4)
    mesh.loops.foreach_set("vertex_index", quads.astype(np.int32).ravel())
    mesh.polygons.add(n)
    mesh.polygons.foreach_set("loop_start", (np.arange(n, dtype=np.int32) * 4))
    mesh.polygons.foreach_set("loop_total", np.full(n, 4, np.int32))
    mesh.update(calc_edges=True)
    mesh.validate(clean_customdata=False)
    mesh.materials.append(block_material())
    obj = bpy.data.objects.new(OBJECT_NAME, mesh)
    obj[MS.HULL_TAG] = 1
    MS.store_label(obj, "建筑")
    MS.set_refs(obj, paths)  # the three views become this block's 建筑参考
    if box is not None:
        fit = size_mode != "MANUAL"     # 按比例: the placeholder gives the size; 长宽高: the user's size wins
        tf = MS.fit_to_placeholder(verts.min(axis=0), verts.max(axis=0), box, fit=fit)
        _apply(obj, tf)
        info["dims"] = [round(d, 4) for d in tf["dims"]]
        info["hull_dims"] = hinfo.get("dims")
        info["fitted"] = tf["fitted"]
        info["scale"] = round(tf["scale"], 4)
        info["yaw_deg"] = round(math.degrees(tf["yaw"]), 2)
        obj[MS.PLACEHOLDER_BOX_PROP] = json.dumps(box)
        if ph is not None:
            obj[MS.PLACEHOLDER_PROP] = ph.name
            info["placeholder"] = ph.name
        elif old is not None and old.get(MS.PLACEHOLDER_PROP):
            obj[MS.PLACEHOLDER_PROP] = str(old[MS.PLACEHOLDER_PROP])
            info["placeholder"] = str(old[MS.PLACEHOLDER_PROP])
    elif old is not None:
        obj.matrix_world = old.matrix_world.copy()    # a cursor block: same place, rotation and scale
    else:
        cur = scene.cursor.location
        obj.location = (cur.x, cur.y, 0.0)
    # collections: the placeholder's / the old block's own, else the massing collection
    if ph is not None and mode == "placeholder":
        homes = [c for c in ph.users_collection]
    elif old is not None:
        homes = [c for c in old.users_collection]
    else:
        homes = [coll]
    homes = [c for c in homes if c is not None] or [coll]
    for c in homes:
        c.objects.link(obj)
    if not any(_in_massing(c, coll) for c in homes) or (old is not None and old.get(MS.ENV_EXTRA_TAG)):
        obj[MS.ENV_EXTRA_TAG] = 1     # outside the massing collection: still a massing block
    info["collection"] = homes[0].name if homes[0] != scene.collection else "场景根集合"
    info["extra"] = bool(obj.get(MS.ENV_EXTRA_TAG))
    sb_colls = [c for c in bpy.data.collections if c.get("wondful_scene_blockout")]
    info["in_scene_blockout"] = any(c in sb_colls or any(c in p.children_recursive for p in sb_colls) for c in homes)
    if old is not None:
        name = old.name
        hint, custom = old.get("wondful_hint"), not old.get(MS.AUTO_HINT_FLAG)
        old_mesh = old.data
        if ph is None:
            for p in bpy.data.objects:
                if p.get(MS.PLACEHOLDER_FOR_PROP) == name:
                    p[MS.PLACEHOLDER_FOR_PROP] = ""
        bpy.data.objects.remove(old, do_unlink=True)
        if old_mesh is not None and old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
        obj.name = name
        if custom and hint:
            obj["wondful_hint"] = hint            # the user's own name for the block stays
            if MS.AUTO_HINT_FLAG in obj.keys():
                del obj[MS.AUTO_HINT_FLAG]
        info["replaced"] = name
    if ph is not None:
        ph.hide_render = True
        ph.hide_viewport = True
        try:
            ph.select_set(False)
        except Exception:
            pass
        ph[MS.PLACEHOLDER_FOR_PROP] = obj.name
    info["object"] = obj.name
    info["refs"] = len(MS.get_refs(obj))
    info["seconds"] = round(time.perf_counter() - t0, 2)
    return obj, info


def summary(info) -> str:
    d = info.get("dims") or [0, 0, 0]
    verb = f"已替换 {info['replaced']}" if info.get("replaced") else f"已生成 {info.get('object', OBJECT_NAME)}"
    text = (f"{verb}（{info.get('collection', '')}）：宽 {d[0]:.1f} × 深 {d[1]:.1f} × 高 {d[2]:.1f} m，"
            f"{info.get('faces', 0)} 面，{info.get('seconds', 0):.1f}s；三视图已设为它的建筑参考")
    if info.get("placeholder"):
        text += f"；已放到占位体 {info['placeholder']} 的位置"
        text += (f"，按占位体大小等比缩放 ×{info.get('scale', 1):.2f}" if info.get("fitted")
                 else "，保持填写的长宽高，只用占位体的位置和朝向")
        if info.get("mode") == "placeholder":
            text += "（占位体已隐藏）"
    elif info.get("replaced"):
        text += "；位置、朝向不变"
    if info.get("in_scene_blockout"):
        text += "；它在「场景粗模」集合里，重新生成场景粗模会删掉它"
    chk = info.get("top_aspect_check")
    if chk and abs(chk - 1.0) > 0.15:
        text += f"；俯视图比例与正 / 侧视不一致（{chk:.2f}），请检查俯视图朝向"
    if info.get("top_rotated"):
        text += "；俯视图是横放的，已按比例自动转正"
    return text
