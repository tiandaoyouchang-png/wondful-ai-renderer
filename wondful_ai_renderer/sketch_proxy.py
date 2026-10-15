"""草图模式 (Blender side): build the clay proxy object from the three-view images.

The geometry work is in sketch_model (numpy only); this module loads images through
bpy (so JPG works without Pillow), creates / updates the proxy object and its clay
material, and makes its collection the product collection.
"""
from __future__ import annotations

import time
from pathlib import Path

import bpy
import numpy as np

from . import sketch_model as SM

COLLECTION_NAME = "Wondful 草模"
OBJECT_NAME = "Wondful 草模"
MATERIAL_NAME = "Wondful 草模 Clay"
PROXY_TAG = "wondful_sketch_proxy"


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
    if getattr(props, "sketch_input", "SEPARATE") == "SHEET":
        p = bpy.path.abspath(getattr(props, "sketch_sheet_path", "") or "")
        return [("sheet", p)] if p and Path(p).is_file() else []
    for view in SM.VIEWS:
        p = bpy.path.abspath(getattr(props, f"sketch_{view}_path", "") or "")
        if p and Path(p).is_file():
            out.append((view, p))
    return out


def reference_images(props):
    """[(path, label)] sent to the image model as product appearance references."""
    labels = {"sheet": "三视图拼图（正视 / 侧视 / 俯视）", **SM.VIEW_LABELS}
    return [(p, labels[v]) for v, p in view_paths(props)]


def silhouettes(props):
    """{view: cropped bool mask} + info from the props' images."""
    info = {}
    if getattr(props, "sketch_input", "SEPARATE") == "SHEET":
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


def clay_material():
    mat = bpy.data.materials.get(MATERIAL_NAME)
    if mat is None:
        mat = bpy.data.materials.new(MATERIAL_NAME)
        mat.diffuse_color = (0.72, 0.72, 0.70, 1.0)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (0.72, 0.72, 0.70, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.65
    return mat


def proxy_collection(scene):
    coll = bpy.data.collections.get(COLLECTION_NAME)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION_NAME)
    if coll.name not in scene.collection.children and coll not in _all_children(scene.collection):
        scene.collection.children.link(coll)
    return coll


def _all_children(coll):
    out = []
    for c in coll.children:
        out.append(c)
        out.extend(_all_children(c))
    return out


def existing_proxy(coll):
    for obj in coll.objects:
        if obj.type == "MESH" and obj.get(PROXY_TAG):
            return obj
    return None


def build_proxy(context, props):
    """Build or rebuild the proxy.  Returns (object, info dict)."""
    t0 = time.perf_counter()
    masks, info = silhouettes(props)
    verts, quads, hinfo = SM.build_hull(
        masks["front"], masks["side"], masks.get("top"),
        size_mode=getattr(props, "sketch_size_mode", "AUTO"),
        known_axis=getattr(props, "sketch_known_axis", "LENGTH"),
        known_value=float(getattr(props, "sketch_known_value", 1.0)),
        length=float(getattr(props, "sketch_length", 1.0)),
        width=float(getattr(props, "sketch_width", 1.0)),
        height=float(getattr(props, "sketch_height", 1.0)),
        side_front=getattr(props, "sketch_side_front", "LEFT"),
        top_front=getattr(props, "sketch_top_front", "BOTTOM"),
        resolution=int(getattr(props, "sketch_resolution", 160)),
    )
    info.update(hinfo)
    scene = context.scene
    coll = proxy_collection(scene)
    mesh = bpy.data.meshes.new(OBJECT_NAME)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    n = len(quads)
    mesh.loops.add(n * 4)
    mesh.loops.foreach_set("vertex_index", quads.astype(np.int32).ravel())
    mesh.polygons.add(n)
    mesh.polygons.foreach_set("loop_start", (np.arange(n, dtype=np.int32) * 4))
    mesh.polygons.foreach_set("loop_total", np.full(n, 4, np.int32))
    mesh.polygons.foreach_set("use_smooth", np.ones(n, bool))
    mesh.update(calc_edges=True)
    mesh.validate(clean_customdata=False)
    mesh.materials.append(clay_material())
    obj = existing_proxy(coll)
    if obj is not None:  # rebuild in place: keep the user's placement
        old = obj.data
        obj.data = mesh
        if old.users == 0:
            bpy.data.meshes.remove(old)
        info["kept_transform"] = True
    else:
        obj = bpy.data.objects.new(OBJECT_NAME, mesh)
        obj[PROXY_TAG] = 1
        cur = scene.cursor.location
        obj.location = (cur.x, cur.y, 0.0)
        coll.objects.link(obj)
        info["kept_transform"] = False
    props.product_collection = coll
    info["seconds"] = round(time.perf_counter() - t0, 2)
    return obj, info


def summary(info) -> str:
    d = info.get("dims") or [0, 0, 0]
    text = f"草模已生成：长 {d[1]:.2f} × 宽 {d[0]:.2f} × 高 {d[2]:.2f} m，{info.get('faces', 0)} 面，{info.get('seconds', 0):.1f}s"
    chk = info.get("top_aspect_check")
    if chk and abs(chk - 1.0) > 0.15:
        text += f"；俯视图比例与正 / 侧视不一致（{chk:.2f}），请检查俯视图朝向"
    if info.get("top_rotated"):
        text += "；俯视图是横放的，已按比例自动转正"
    return text
