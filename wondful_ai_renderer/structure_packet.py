from __future__ import annotations

"""Wondful 3.0 deterministic Structure Packet.

The packet separates Blender-owned structure from AI-owned appearance.  It never
uses color/material heuristics to decide which object is the product: product
identity comes exclusively from ``props.product_collection``.

Packet assets (when native data passes are available):
    01_camera_base.png
    02_product_mask.png
    03_scene_depth.png
    04_scene_normal.png
    05_product_silhouette.png
    06_part_id.png
    07_structure.json

A compatibility/fallback path uses the existing projected-mesh structure builder
if Blender cannot render native data passes on the current scene.
"""

import hashlib
import os
import json
import math
import shutil
import time
import uuid
from pathlib import Path

from .workflow import isolated_capture

import bpy
import numpy as np

from bpy_extras.object_utils import world_to_camera_view
from .composition import image_dimensions

from .structure_control import (
    WHEEL_KEYWORDS,
    build_structure_guides,
    resolve_structure_objects,
    geometry_center_world,
    _dilate_binary,
    _save_rgba,
)


PACKET_VERSION = "3.0.11"
_NATIVE_MAX_EDGE = 1536


def _packet_resolution(width: int, height: int, max_edge: int = _NATIVE_MAX_EDGE) -> tuple[int, int]:
    width = max(4, int(width))
    height = max(4, int(height))
    longest = max(width, height)
    if longest <= max_edge:
        return width, height
    scale = float(max_edge) / float(longest)
    return max(4, int(round(width * scale))), max(4, int(round(height * scale)))


def _raster_pixel_aspect(width, height, raster_w, raster_h):
    # Integer-sized preview passes still need the exact camera display aspect.
    ratio = (width / height) / (raster_w / raster_h)
    return (ratio, 1.0) if ratio >= 1.0 else (1.0, 1.0 / ratio)


def _load_rgba(path: str | Path) -> np.ndarray:
    p = Path(path)
    image = bpy.data.images.load(str(p), check_existing=False)
    try:
        w, h = int(image.size[0]), int(image.size[1])
        values = np.empty(w * h * 4, dtype=np.float32)
        image.pixels.foreach_get(values)
        # Blender Image pixels are bottom-up; packet arrays use conventional top-down.
        return values.reshape(h, w, 4)[::-1].copy()
    finally:
        try:
            bpy.data.images.remove(image)
        except Exception:
            pass


def _save_scalar_png(path: Path, scalar: np.ndarray, name: str) -> None:
    scalar = np.asarray(scalar, dtype=np.float32)
    rgba = np.zeros((scalar.shape[0], scalar.shape[1], 4), dtype=np.float32)
    rgba[:, :, 0] = scalar
    rgba[:, :, 1] = scalar
    rgba[:, :, 2] = scalar
    rgba[:, :, 3] = 1.0
    _save_rgba(path, rgba, name)


def _erode_binary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    src = mask.astype(bool, copy=False)
    if radius <= 0:
        return src.copy()
    out = src.copy()
    h, w = src.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(src)
            y0 = max(0, dy)
            y1 = min(h, h + dy)
            x0 = max(0, dx)
            x1 = min(w, w + dx)
            sy0 = max(0, -dy)
            sy1 = sy0 + (y1 - y0)
            sx0 = max(0, -dx)
            sx1 = sx0 + (x1 - x0)
            if y1 > y0 and x1 > x0:
                shifted[y0:y1, x0:x1] = src[sy0:sy1, sx0:sx1]
            out &= shifted
    return out


def _id_color(name: str) -> tuple[float, float, float]:
    raw = hashlib.sha1(name.encode("utf-8", errors="replace")).digest()
    # Keep colors away from black so every visible product part is legible.
    return tuple(0.18 + (raw[i] / 255.0) * 0.82 for i in range(3))


# Render Layers socket names changed between Blender 4.x and 5.x
# (e.g. "IndexOB" -> "Object Index").  Try every known alias.
_PASS_SOCKET_ALIASES = {
    "Depth": ("Depth", "Z"),
    "Normal": ("Normal",),
    "IndexOB": ("IndexOB", "Object Index"),
}


def _render_layer_socket(render_layers, socket_name: str):
    for name in _PASS_SOCKET_ALIASES.get(socket_name, (socket_name,)):
        socket = render_layers.outputs.get(name)
        if socket is not None:
            return socket
    return None


def _compositor_tree(temp_scene):
    """Return (node_tree, owned_group) for the temp Scene compositor.

    Blender <= 4.x: ``scene.use_nodes`` + ``scene.node_tree``.
    Blender >= 5.0: the compositor is a node group assigned to
    ``scene.compositing_node_group``; ``scene.node_tree`` no longer exists.
    ``owned_group`` is the group we created and must remove afterwards.
    """
    if hasattr(temp_scene, "compositing_node_group"):
        group = bpy.data.node_groups.new("Wondful_StructurePacket_Comp", "CompositorNodeTree")
        temp_scene.compositing_node_group = group
        return group, group
    temp_scene.use_nodes = True
    return temp_scene.node_tree, None


def _set_exr_format(fmt, color_mode: str) -> None:
    if hasattr(fmt, "media_type"):
        try:
            fmt.media_type = "IMAGE"
        except Exception:
            pass
    fmt.file_format = "OPEN_EXR"
    fmt.color_depth = "32"
    for mode in (color_mode, "RGB", "RGBA", "BW"):
        try:
            fmt.color_mode = mode
            return
        except Exception:
            continue


def _file_output_node(nodes, links, render_layers, socket_name: str, base_path: Path, prefix: str, color_mode: str = "RGBA"):
    socket = _render_layer_socket(render_layers, socket_name)
    if socket is None:
        return None
    node = nodes.new("CompositorNodeOutputFile")
    _set_exr_format(node.format, color_mode)
    if hasattr(node, "file_output_items"):
        # Blender 5.x: directory + file_name + per-item name -> <dir>/<file_name><item>.exr
        node.directory = str(base_path)
        node.file_name = prefix
        try:
            node.file_output_items.clear()
        except Exception:
            pass
        socket_type = "FLOAT" if color_mode == "BW" else ("VECTOR" if socket_name == "Normal" else "RGBA")
        try:
            node.file_output_items.new(socket_type, "pass")
        except Exception:
            node.file_output_items.new("RGBA", "pass")
        target = node.inputs.get("pass") or node.inputs[0]
    else:
        node.base_path = str(base_path)
        node.file_slots[0].path = prefix
        target = node.inputs[0]
    links.new(socket, target)
    return node


def _find_pass_file(output_dir: Path, prefix: str, started: float) -> Path | None:
    candidates: list[Path] = []
    for p in output_dir.glob(prefix + "*.exr"):
        try:
            if p.stat().st_mtime + 2.0 >= started:
                candidates.append(p)
        except OSError:
            continue
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _visible_scene_objects(context) -> list[bpy.types.Object]:
    result = []
    for obj in context.scene.objects:
        if getattr(obj, "hide_render", False):
            continue
        try:
            if obj.type not in {"CAMERA", "LIGHT"} and not obj.visible_get(view_layer=context.view_layer):
                continue
        except Exception:
            pass
        result.append(obj)
    return result


def _depth_background_cutoff(camera) -> float:
    clip_end = 0.0
    try:
        clip_end = float(camera.data.clip_end)
    except Exception:
        clip_end = 0.0
    if not math.isfinite(clip_end) or clip_end <= 0.0:
        return 1e9
    return min(1e9, clip_end * 0.999)


def _structure_pass_engine(scene) -> str:
    """EEVEE identifier differs by version: 4.2-4.x BLENDER_EEVEE_NEXT, 5.x BLENDER_EEVEE.

    ``WONDFUL_STRUCTURE_ENGINE`` overrides it (used by headless CI without a GPU).
    """
    override = os.environ.get("WONDFUL_STRUCTURE_ENGINE", "").strip()
    try:
        available = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    except Exception:
        available = set()
    for ident in ((override,) if override else ()) + ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        if ident and (not available or ident in available or ident == "CYCLES"):
            return ident
    return scene.render.engine


def _configure_cycles_for_passes(temp_scene) -> None:
    if temp_scene.render.engine != "CYCLES":
        return
    try:
        temp_scene.cycles.samples = 1
        temp_scene.cycles.use_denoising = False
    except Exception:
        pass


def _native_data_passes(context, props, output_dir: Path, width: int, height: int, render_mode="STANDARD") -> dict:
    """Render depth/normal/object-index in an isolated temporary Scene.

    No compositor nodes, render engine, pass settings, resolution or output paths in
    the user's Scene are modified.  Original objects are linked read-only into a
    temporary Scene and object ``pass_index`` values are restored in ``finally``.
    """
    scene = context.scene
    camera = scene.camera
    product_collection = getattr(props, "product_collection", None)
    if camera is None or product_collection is None:
        raise RuntimeError("NO_CAMERA_OR_PRODUCT_COLLECTION")

    product_objects = [
        obj for obj in product_collection.all_objects
        if obj.type == "MESH" and scene.objects.get(obj.name) is not None and not getattr(obj, "hide_render", False)
    ]
    if not product_objects:
        raise RuntimeError("EMPTY_PRODUCT_COLLECTION")

    # 3.1.5: Standard also renders native Object Index at the full camera grid.
    # Small logos/wordmarks must not disappear in a lower-resolution Part-ID pass
    # before the Identity Preserve Mask is derived. FAST keeps the bounded raster.
    packet_w, packet_h = (
        (int(width), int(height))
        if render_mode in {"STRICT", "STANDARD"}
        else _packet_resolution(width, height)
    )
    raster_size = (packet_w, packet_h)
    temp_scene = bpy.data.scenes.new("Wondful_StructurePacket_" + uuid.uuid4().hex[:8])
    pass_index_snapshot: dict[str, tuple[bpy.types.Object, int]] = {}
    temp_files: list[Path] = []
    owned_comp_group = None
    try:
        # Link the exact datablocks used by the working Scene so modifiers/materials,
        # camera lens and animation evaluate consistently, while keeping Scene settings isolated.
        for obj in _visible_scene_objects(context):
            try:
                temp_scene.collection.objects.link(obj)
            except RuntimeError:
                pass
        if temp_scene.objects.get(camera.name) is None:
            try:
                temp_scene.collection.objects.link(camera)
            except RuntimeError:
                pass
        temp_scene.camera = camera
        temp_scene.world = scene.world
        temp_scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)

        temp_scene.render.resolution_x = packet_w
        temp_scene.render.resolution_y = packet_h
        temp_scene.render.resolution_percentage = 100
        temp_scene.render.use_border = False
        temp_scene.render.use_crop_to_border = False
        temp_scene.render.use_multiview = False
        temp_scene.render.use_sequencer = False
        pixel_x, pixel_y = _raster_pixel_aspect(width, height, packet_w, packet_h)
        temp_scene.render.pixel_aspect_x = pixel_x
        temp_scene.render.pixel_aspect_y = pixel_y
        temp_scene.render.film_transparent = True
        try:
            temp_scene.render.engine = _structure_pass_engine(scene)
        except Exception:
            temp_scene.render.engine = scene.render.engine
        _configure_cycles_for_passes(temp_scene)

        view_layer = temp_scene.view_layers[0]
        view_layer.use_pass_z = True
        view_layer.use_pass_normal = True
        view_layer.use_pass_object_index = True
        try:
            view_layer.update_render_passes()
        except Exception:
            pass

        # Object Index gives an occlusion-aware visible product mask and a stable
        # per-object Part-ID map.  Non-product objects keep index 0.
        ordered_products = sorted(product_objects, key=lambda o: o.name.lower())
        for obj in _visible_scene_objects(context):
            if not hasattr(obj, "pass_index"):
                continue
            pass_index_snapshot[obj.name] = (obj, int(obj.pass_index))
            obj.pass_index = 0
        part_index_to_name: dict[int, str] = {}
        for idx, obj in enumerate(ordered_products, start=1):
            pass_index_snapshot.setdefault(obj.name, (obj, int(obj.pass_index)))
            obj.pass_index = idx
            part_index_to_name[idx] = obj.name

        comp_tree, owned_comp_group = _compositor_tree(temp_scene)
        nodes = comp_tree.nodes
        links = comp_tree.links
        nodes.clear()
        render_layers = nodes.new("CompositorNodeRLayers")
        render_layers.scene = temp_scene
        try:
            render_layers.layer = view_layer.name
        except Exception:
            pass
        _file_output_node(nodes, links, render_layers, "Depth", output_dir, "__wondful_depth_", "BW")
        _file_output_node(nodes, links, render_layers, "Normal", output_dir, "__wondful_normal_", "RGBA")
        _file_output_node(nodes, links, render_layers, "IndexOB", output_dir, "__wondful_index_", "BW")

        started = time.time()
        bpy.ops.render.render(scene=temp_scene.name, write_still=False)
        depth_file = _find_pass_file(output_dir, "__wondful_depth_", started)
        normal_file = _find_pass_file(output_dir, "__wondful_normal_", started)
        index_file = _find_pass_file(output_dir, "__wondful_index_", started)
        temp_files = [p for p in (depth_file, normal_file, index_file) if p]
        if depth_file is None or normal_file is None or index_file is None:
            raise RuntimeError("NATIVE_PASS_OUTPUT_MISSING")

        depth_rgba = _load_rgba(depth_file)
        normal_rgba = _load_rgba(normal_file)
        index_rgba = _load_rgba(index_file)
        for rgba in (depth_rgba, normal_rgba, index_rgba):
            if rgba.shape[:2] != (packet_h, packet_w):
                raise RuntimeError("NATIVE_PASS_CANVAS_MISMATCH")
        # All published controls use the exact Camera Base pixel grid. Standard
        # may render lower-resolution passes, but never sends mixed-size inputs.
        if (packet_w, packet_h) != (width, height):
            depth_rgba = _resample_rgba(depth_rgba, width, height)
            normal_rgba = _resample_rgba(normal_rgba, width, height)
            index_rgba = _resample_rgba(index_rgba, width, height)
            packet_w, packet_h = int(width), int(height)
        depth = depth_rgba[:, :, 0]
        normal = normal_rgba[:, :, :3]
        index_values = np.rint(index_rgba[:, :, 0]).astype(np.int32)

        # EXR depth is camera distance.  Keep actual near/far in the manifest and
        # provide a model-friendly 0..1 PNG (near=white, far=black).
        # Background pixels carry a sentinel distance (Cycles/EEVEE write ~1e10,
        # some builds write the camera clip_end).  Including them collapses the
        # percentile window and turns the whole depth map white, so only real
        # geometry inside the camera clip range is normalized.
        finite = np.isfinite(depth) & (depth > 1e-6) & (depth < _depth_background_cutoff(camera))
        # The window is fitted to the product (Object Index > 0) so the car's own
        # surface relief spans the 0..1 range instead of a sliver of it.
        from .depth_utils import normalize_depth
        depth_norm, near, far = normalize_depth(depth, finite, index_values > 0)
        scene_depth_range = ([float(depth[finite].min()), float(depth[finite].max())]
                             if np.any(finite) else [0.0, 0.0])

        # Blender Normal pass is signed floating-point data.  Encode it as the
        # conventional RGB normal map without color-management transforms.
        normal_valid = np.linalg.norm(normal, axis=2) > 1e-6
        normal_png = np.zeros((packet_h, packet_w, 4), dtype=np.float32)
        normal_png[:, :, 3] = 1.0
        normal_png[:, :, :3] = np.clip(normal * 0.5 + 0.5, 0.0, 1.0)
        normal_png[~normal_valid, :3] = 0.0

        product_mask = index_values > 0
        if not np.any(product_mask):
            raise RuntimeError("PRODUCT_INDEX_PASS_EMPTY")
        mask_rgba = np.zeros((packet_h, packet_w, 4), dtype=np.float32)
        mask_rgba[:, :, 3] = 1.0
        mask_rgba[product_mask, :3] = 1.0

        eroded = _erode_binary(product_mask, radius=1)
        edge = product_mask & ~eroded
        edge = _dilate_binary(edge, 1)
        silhouette_rgba = np.zeros((packet_h, packet_w, 4), dtype=np.float32)
        silhouette_rgba[:, :, 3] = 1.0
        silhouette_rgba[edge, :3] = 1.0

        part_rgba = np.zeros((packet_h, packet_w, 4), dtype=np.float32)
        part_rgba[:, :, 3] = 1.0
        part_manifest: dict[str, dict] = {}
        for idx, name in part_index_to_name.items():
            visible = index_values == idx
            if not np.any(visible):
                continue
            color = _id_color(name)
            part_rgba[visible, 0] = color[0]
            part_rgba[visible, 1] = color[1]
            part_rgba[visible, 2] = color[2]
            part_manifest[name] = {
                "index": int(idx),
                "rgb": [round(float(c), 6) for c in color],
            }

        depth_path = output_dir / "03_scene_depth.png"
        normal_path = output_dir / "04_scene_normal.png"
        mask_path = output_dir / "02_product_mask.png"
        silhouette_path = output_dir / "05_product_silhouette.png"
        part_path = output_dir / "06_part_id.png"
        part_index_path = output_dir / "06_part_index.npy"
        np.save(str(part_index_path), index_values.astype(np.int32), allow_pickle=False)
        _save_scalar_png(depth_path, depth_norm, "Wondful_SceneDepth")
        _save_rgba(normal_path, normal_png, "Wondful_SceneNormal")
        _save_rgba(mask_path, mask_rgba, "Wondful_ProductMask")
        _save_rgba(silhouette_path, silhouette_rgba, "Wondful_ProductSilhouette")
        _save_rgba(part_path, part_rgba, "Wondful_PartID")

        # Published mask and edit mask share the exact full-frame pixel grid.
        target_w, target_h = int(width), int(height)
        edit_source = product_mask
        dilation_radius = max(6, min(target_w, target_h) // 80)
        editable = _dilate_binary(edit_source, dilation_radius)
        edit_rgba = np.ones((target_h, target_w, 4), dtype=np.float32)
        edit_rgba[editable, 3] = 0.0
        edit_mask_path = output_dir / "product_edit_mask.png"
        _save_rgba(edit_mask_path, edit_rgba, "Wondful_ProductEditMask3")

        yy, xx = np.nonzero(product_mask)
        bbox = (float(xx.min()), float(yy.min()), float(xx.max() + 1), float(yy.max() + 1))
        center = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)

        return {
            "native_passes": True,
            "native_raster_size": list(raster_size),
            "native_pixel_aspect": [pixel_x, pixel_y],
            "controls_resampled": raster_size != (packet_w, packet_h),
            "packet_width": packet_w,
            "packet_height": packet_h,
            "depth_near": near,
            "scene_depth_range": scene_depth_range,
            "depth_window": "PRODUCT_FOCUSED",
            "depth_far": far,
            "mask_path": str(mask_path),
            "depth_path": str(depth_path),
            "normal_path": str(normal_path),
            "silhouette_path": str(silhouette_path),
            "part_id_path": str(part_path),
            "part_index_path": str(part_index_path),
            "edit_mask_path": str(edit_mask_path),
            "part_id_manifest": part_manifest,
            "bbox_pixels_packet": [round(v, 2) for v in bbox],
            "bbox_normalized": [
                round(bbox[0] / packet_w, 6), round(bbox[1] / packet_h, 6),
                round(bbox[2] / packet_w, 6), round(bbox[3] / packet_h, 6),
            ],
            "center_normalized": [round(center[0] / packet_w, 6), round(center[1] / packet_h, 6)],
            "product_mask_available": True,
            "mask_source": "NATIVE_OBJECT_INDEX_PASS",
            "edit_mask_dilation_px": dilation_radius,
        }
    finally:
        for _name, (obj, value) in pass_index_snapshot.items():
            try:
                obj.pass_index = value
            except Exception:
                pass
        try:
            bpy.data.scenes.remove(temp_scene)
        except Exception:
            pass
        if owned_comp_group is not None:
            try:
                bpy.data.node_groups.remove(owned_comp_group)
            except Exception:
                pass
        for p in temp_files:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def _fallback_silhouette(mask_path: str | Path, output_dir: Path) -> str:
    if not mask_path or not Path(mask_path).exists():
        return ""
    try:
        rgba = _load_rgba(mask_path)
        mask = rgba[:, :, 0] > 0.5
        edge = mask & ~_erode_binary(mask, radius=1)
        edge = _dilate_binary(edge, 1)
        out = np.zeros_like(rgba, dtype=np.float32)
        out[:, :, 3] = 1.0
        out[edge, :3] = 1.0
        path = output_dir / "05_product_silhouette.png"
        _save_rgba(path, out, "Wondful_ProductSilhouetteFallback")
        return str(path)
    except Exception:
        return ""


def _select_generation_refs(packet: dict, mode: str) -> tuple[list[str], list[str]]:
    mode = (mode or "STANDARD").upper()
    candidates: list[tuple[str, str, str]] = [
        ("product_mask", packet.get("mask_path", ""), "产品可见区域/占位 Mask"),
        ("scene_depth", packet.get("depth_path", ""), "Scene Camera 深度层级"),
        ("scene_normal", packet.get("normal_path", ""), "可见表面法线/曲面朝向"),
        ("product_silhouette", packet.get("silhouette_path", ""), "产品外轮廓/关键边界"),
        ("part_id", packet.get("part_id_path", ""), "产品 Mesh 对象语义分区"),
    ]
    if mode == "FAST":
        wanted = {"product_mask", "product_silhouette"}
    elif mode == "STRICT":
        wanted = {"product_mask", "scene_depth", "scene_normal", "product_silhouette", "part_id"}
    else:
        # Standard keeps Part-ID because semantic part/material constraints need
        # a spatial carrier. Normal is less valuable than Part-ID for automotive
        # appearance transfer under limited reference slots.
        wanted = {"product_mask", "scene_depth", "product_silhouette", "part_id"}
    refs: list[str] = []
    roles: list[str] = []
    descriptions: list[str] = []
    for role, raw, desc in candidates:
        if role not in wanted or not raw:
            continue
        p = Path(raw)
        if not p.exists():
            continue
        refs.append(str(p))
        roles.append(role)
        descriptions.append(desc)
    packet["generation_reference_paths"] = refs
    packet["generation_reference_roles"] = roles
    packet["generation_reference_descriptions"] = descriptions
    return refs, roles


@isolated_capture
def build_structure_packet(
    context,
    props,
    output_dir: str | Path,
    width: int,
    height: int,
    *,
    camera_base_path: str | Path,
    render_mode: str = "STANDARD",
) -> dict:
    """Build the Wondful 3.0 deterministic Structure Packet.

    Native Blender render passes are the primary path.  The legacy projected-mesh
    builder is invoked only as a fallback, so dense production car meshes do not pay
    the old wireframe/triangle rasterization cost when native passes work.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene = context.scene
    camera = scene.camera
    objects, source_mode = resolve_structure_objects(context, props)
    if camera is None:
        return {"enabled": False, "source_mode": "NO_CAMERA", "object_names": []}
    if not objects:
        return {
            "enabled": False,
            "source_mode": source_mode,
            "product_collection_name": getattr(getattr(props, "product_collection", None), "name", ""),
            "object_names": [],
            "reason": "PRODUCT_COLLECTION_EMPTY_OR_HIDDEN",
        }

    camera_base = out / "01_camera_base.png"
    try:
        shutil.copy2(Path(camera_base_path), camera_base)
    except Exception:
        camera_base = Path(camera_base_path)

    wheel_points: list[list[float]] = []
    depsgraph = context.evaluated_depsgraph_get()
    for obj in objects:
        if not any(token in obj.name.lower() for token in WHEEL_KEYWORDS):
            continue
        try:
            ndc = world_to_camera_view(scene, camera, geometry_center_world(obj.evaluated_get(depsgraph)))
            if ndc.z > 0 and 0 <= ndc.x <= 1 and 0 <= ndc.y <= 1:
                wheel_points.append([round(float(ndc.x), 6), round(1.0 - float(ndc.y), 6)])
        except Exception:
            pass

    packet = {
        "enabled": True,
        "packet_version": PACKET_VERSION,
        "source_mode": source_mode,
        "camera_name": camera.name,
        "product_collection_name": getattr(getattr(props, "product_collection", None), "name", ""),
        "object_names": [obj.name for obj in objects],
        "product_object_names": [obj.name for obj in objects],
        "camera_visible_mesh_count": len(objects),
        "product_collection_mesh_count": len(objects),
        "camera_base_path": str(camera_base),
        "camera_output_width": int(width),
        "camera_output_height": int(height),
        "camera_aspect": round(float(width) / float(height), 8) if height else 1.0,
        "scene_bbox_normalized": [0.0, 0.0, 1.0, 1.0],
        "wheel_points_normalized": wheel_points,
        "wheel_anchor_source": "NAMED_MESH_BOUNDING_BOX_CENTER_ESTIMATE",
        "render_mode": str(render_mode),
        "native_passes": False,
    }

    native_error = ""
    try:
        native = _native_data_passes(context, props, out, width, height, render_mode=render_mode)
        packet.update(native)
    except Exception as exc:
        native_error = f"{type(exc).__name__}: {exc}"
        # Fallback is deliberately isolated to failure cases.  It keeps 3.0 usable
        # on scenes/render builds where compositor data-pass export is unavailable.
        fallback = build_structure_guides(context, props, out, width, height)
        if not fallback.get("enabled"):
            return {
                "enabled": False,
                "source_mode": source_mode,
                "error": native_error,
                "reason": fallback.get("reason") or fallback.get("source_mode") or "NATIVE_AND_FALLBACK_FAILED",
            }
        packet.update(fallback)
        packet["packet_version"] = PACKET_VERSION
        packet["camera_base_path"] = str(camera_base)
        packet["camera_output_width"] = int(width)
        packet["camera_output_height"] = int(height)
        packet["camera_aspect"] = round(float(width) / float(height), 8) if height else 1.0
        packet["render_mode"] = str(render_mode)
        packet["native_passes"] = False
        packet["native_pass_error"] = native_error
        legacy_mask = packet.get("mask_path", "")
        if legacy_mask and Path(legacy_mask).exists():
            target_mask = out / "02_product_mask.png"
            try:
                if Path(legacy_mask).resolve() != target_mask.resolve():
                    shutil.copy2(legacy_mask, target_mask)
                packet["mask_path"] = str(target_mask)
            except Exception:
                pass
        packet["silhouette_path"] = _fallback_silhouette(packet.get("mask_path", ""), out)
        packet.setdefault("depth_path", "")
        packet.setdefault("normal_path", "")
        packet.setdefault("part_id_path", "")
        packet.setdefault("part_index_path", "")

    # In 3.0 the clean silhouette is the visual audit guide.  Dense topology
    # wireframes are diagnostic only and are not sent to the generator by default.
    if packet.get("silhouette_path"):
        packet["guide_path"] = packet["silhouette_path"]

    _select_generation_refs(packet, render_mode)
    _validate_packet_canvas(packet, width, height)
    packet["canvas_verified"] = True
    packet["structure_quality"] = "NATIVE_PASSES" if packet["native_passes"] else "PROJECTED_FALLBACK"
    packet["control_priority"] = [
        "camera_base",
        "product_mask",
        "scene_depth",
        "scene_normal",
        "product_silhouette",
        "part_id",
        "product_identity",
        "person_identity",
        "style",
        "prompt",
    ]
    packet["roles"] = {
        "camera_base": "Camera / Composition / Perspective / whole-frame spatial relationship",
        "product_mask": "product Position / Scale / visible footprint",
        "scene_depth": "foreground-background depth ordering",
        "scene_normal": "surface orientation and curvature guidance",
        "product_silhouette": "product outer contour and boundary",
        "part_id": "product Mesh-object semantic regions only; never final colors",
    }

    manifest_path = out / "07_structure.json"
    serializable = dict(packet)
    manifest_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    packet["manifest_path"] = str(manifest_path)
    try:
        (out / "structure_control.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass
    return packet


def _resample_rgba(rgba, width, height):
    src_h, src_w = rgba.shape[:2]
    # Pixel-center mapping, shared by every control; no align-corners offset.
    xs = np.minimum(src_w - 1, ((np.arange(width) + 0.5) * src_w / width).astype(np.intp))
    ys = np.minimum(src_h - 1, ((np.arange(height) + 0.5) * src_h / height).astype(np.intp))
    return rgba[ys[:, None], xs[None, :]].copy()


def _validate_packet_canvas(packet, width, height):
    expected = (int(width), int(height))
    paths = [packet.get("camera_base_path"), *packet.get("generation_reference_paths", [])]
    if packet.get("edit_mask_path"):
        paths.append(packet["edit_mask_path"])
    for path in paths:
        if not path or image_dimensions(path) != expected:
            raise RuntimeError(f"Structure Packet 图像尺寸不一致：{Path(path).name if path else 'missing'}，期望 {width}×{height}")
