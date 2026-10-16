from __future__ import annotations

import math
import time
import os
from pathlib import Path
from typing import Tuple

import bpy

from .composition import image_dimensions


def find_view3d(context):
    if context.area and context.area.type == "VIEW_3D":
        area = context.area
        region = next((r for r in area.regions if r.type == "WINDOW"), None)
        space = area.spaces.active
        if region:
            return context.window, area, region, space

    window = context.window
    if window and window.screen:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                region = next((r for r in area.regions if r.type == "WINDOW"), None)
                if region:
                    return window, area, region, area.spaces.active
    raise RuntimeError("未找到可用的 Blender 3D Viewport。请确保当前窗口中存在 3D View。")


def viewport_dimensions(context) -> Tuple[int, int]:
    _window, _area, region, _space = find_view3d(context)
    return int(region.width), int(region.height)


def camera_output_dimensions(context) -> Tuple[int, int]:
    """Return the scene's effective camera/output dimensions.

    The final AI canvas follows Render Properties resolution rather than the
    3D Viewport/N-panel size.
    """
    render = context.scene.render
    pct = max(1, int(render.resolution_percentage)) / 100.0
    width = max(1, int(round(int(render.resolution_x) * pct)))
    height = max(1, int(round(int(render.resolution_y) * pct)))
    return width, height


def require_scene_camera(context):
    camera = getattr(context.scene, "camera", None)
    if camera is None:
        raise RuntimeError("当前 Scene 没有活动 Camera。请先设置 Scene Camera 后再进行 AI 渲染。")
    return camera


def _capture_framebuffer(window, region, filepath: str) -> bool:
    """Compatibility helper for the legacy viewport-capture path."""
    if not hasattr(window, "screenshot"):
        return False
    try:
        import imbuf

        rect = ((region.x, region.y), (region.x + region.width, region.y + region.height))
        pixels = window.screenshot(region=rect, use_alpha=False)
        height, width = pixels.shape[0], pixels.shape[1]
        ibuf = imbuf.new((width, height))
        ibuf.file_type = "PNG"
        if not hasattr(ibuf, "with_buffer"):
            return False
        with ibuf.with_buffer(write=True) as buf:
            buf.cast("B")[:] = pixels.cast("B")
        imbuf.write(ibuf, filepath=filepath)
        ibuf.free()
        return Path(filepath).exists()
    except Exception:
        return False


def _snapshot_image_format(settings) -> dict:
    """Blender 5.x adds ``media_type``; EXR multilayer lives under MULTI_LAYER_IMAGE
    and PNG cannot be assigned until media_type is IMAGE.  Save both so the user's
    Output Properties are restored exactly."""
    return {
        "media_type": getattr(settings, "media_type", None),
        "file_format": settings.file_format,
    }


def _set_png(settings) -> None:
    if getattr(settings, "media_type", None) not in (None, "IMAGE"):
        settings.media_type = "IMAGE"
    settings.file_format = "PNG"


def _restore_image_format(settings, snap: dict) -> None:
    try:
        if snap.get("media_type") is not None and getattr(settings, "media_type", None) != snap["media_type"]:
            settings.media_type = snap["media_type"]
        settings.file_format = snap["file_format"]
    except Exception:
        pass


def _capture_opengl_view(context, window, area, region, filepath: str) -> bool:
    """Legacy active-view capture retained as a fallback utility."""
    scene = context.scene
    render = scene.render
    old_path = render.filepath
    old_x, old_y, old_pct = render.resolution_x, render.resolution_y, render.resolution_percentage
    old_format = _snapshot_image_format(render.image_settings)
    try:
        render.filepath = filepath
        render.resolution_x = max(1, int(region.width))
        render.resolution_y = max(1, int(region.height))
        render.resolution_percentage = 100
        _set_png(render.image_settings)
        with context.temp_override(window=window, area=area, region=region, scene=scene):
            result = bpy.ops.render.opengl(animation=False, sequencer=False, write_still=True, view_context=True)
        return "FINISHED" in result and Path(filepath).exists()
    finally:
        render.filepath = old_path
        render.resolution_x, render.resolution_y = old_x, old_y
        render.resolution_percentage = old_pct
        _restore_image_format(render.image_settings, old_format)


def _capture_camera_opengl(context, filepath: str) -> bool:
    """Capture exactly the active Scene Camera frame using Blender's OpenGL render.

    `view_context=False` makes the OpenGL render use the Scene Camera rather than
    the arbitrary interactive viewport framing. We temporarily normalize the
    effective render size so the output bitmap has the same X/Y ratio and pixel
    dimensions the user configured in Output Properties.
    """
    scene = context.scene
    require_scene_camera(context)
    width, height = camera_output_dimensions(context)
    window, area, region, _space = find_view3d(context)

    render = scene.render
    old_path = render.filepath
    old_x, old_y, old_pct = render.resolution_x, render.resolution_y, render.resolution_percentage
    old_format = _snapshot_image_format(render.image_settings)
    old_color_mode = getattr(render.image_settings, "color_mode", None)
    frame_options = {key: getattr(render, key) for key in ("use_border", "use_crop_to_border", "use_compositing", "use_sequencer", "use_multiview") if hasattr(render, key)}
    try:
        for key in frame_options:
            setattr(render, key, False)
        render.filepath = filepath
        render.resolution_x = width
        render.resolution_y = height
        render.resolution_percentage = 100
        _set_png(render.image_settings)
        try:
            render.image_settings.color_mode = "RGB"
        except Exception:
            pass

        with context.temp_override(window=window, area=area, region=region, scene=scene):
            result = bpy.ops.render.opengl(animation=False, sequencer=False, write_still=True, view_context=False)
        return "FINISHED" in result and Path(filepath).exists()
    finally:
        render.filepath = old_path
        render.resolution_x, render.resolution_y = old_x, old_y
        render.resolution_percentage = old_pct
        _restore_image_format(render.image_settings, old_format)
        for key, value in frame_options.items():
            setattr(render, key, value)
        if old_color_mode is not None:
            try:
                render.image_settings.color_mode = old_color_mode
            except Exception:
                pass


def capture_camera_reference(context, filepath: str) -> tuple[int, int, str]:
    """Capture the current Scene Camera frame as the sole composition reference.

    Returns (width, height, method). The camera/output frame is intentionally
    independent from the visible 3D Viewport or sidebar dimensions.
    """
    filepath = str(Path(filepath))
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    width, height = camera_output_dimensions(context)
    require_scene_camera(context)
    if context.scene.camera.data.type not in {"PERSP", "ORTHO"}:
        raise RuntimeError("当前构图控制支持透视或正交相机，请使用 PERSP / ORTHO 相机。")
    render = context.scene.render
    if abs(float(render.pixel_aspect_x) / float(render.pixel_aspect_y) - 1.0) > 1e-6:
        raise RuntimeError("AI 图像使用方形像素，请先把输出的像素宽高比设为 1:1。")
    if _capture_camera_opengl(context, filepath):
        actual = image_dimensions(filepath)
        if actual != (width, height):
            raise RuntimeError(f"相机捕获尺寸异常：期望 {width}×{height}，实际 {actual}；已停止提交，避免结构图错位。")
        return width, height, "CAMERA_OPENGL_FULL_FRAME"
    raise RuntimeError("无法捕获当前 Scene Camera Frame。请确认 Scene Camera 有效，并在 3D Viewport 中重试。")


def capture_active_viewport(context, filepath: str) -> tuple[int, int, str]:
    """Legacy capture helper kept for compatibility with older sessions/tools."""
    filepath = str(Path(filepath))
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    window, area, region, _space = find_view3d(context)
    width, height = int(region.width), int(region.height)

    if _capture_framebuffer(window, region, filepath):
        return width, height, "FRAMEBUFFER"
    if _capture_opengl_view(context, window, area, region, filepath):
        return width, height, "OPENGL"
    raise RuntimeError("无法捕获当前 3D Viewport。")


def _product_floor_z(collection):
    from mathutils import Vector
    zs = []
    objs = list(collection.all_objects) if collection else [o for o in bpy.context.scene.objects]
    for obj in objs:
        if obj.type != "MESH" or obj.hide_render or _excluded(obj):
            continue
        for corner in obj.bound_box:
            zs.append((obj.matrix_world @ Vector(corner)).z)
    return min(zs) if zs else 0.0


LAST_CLAY_INFO = {}
# 1.2.0 环境体块: massing objects never count as product (bounds, ground, terrain pose)
_MASSING_EXCLUDE = set()


def _excluded(obj) -> bool:
    return obj in _MASSING_EXCLUDE


def _product_up_axis(collection):
    """World-space 'up' of the product: Z axis of its root object(s).

    When the user tilts the car onto a slope by rotating its root/parent, the
    temporary floor follows that tilt so the wheels still sit on the ground.
    """
    from mathutils import Vector
    if not collection:
        return Vector((0.0, 0.0, 1.0))
    objs = list(collection.all_objects)
    members = set(objs)
    roots = [o for o in objs if o.parent is None or o.parent not in members]
    axes = []
    for root in roots:
        top = root
        while top.parent is not None:  # parented to an empty outside the collection
            top = top.parent
        axis = (top.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0)))
        if axis.length > 1e-6:
            axes.append(axis.normalized())
    if not axes:
        return Vector((0.0, 0.0, 1.0))
    ref = axes[0]
    agree = [a for a in axes if a.dot(ref) > 0.995]
    if len(agree) < max(1, len(axes) // 2):
        return Vector((0.0, 0.0, 1.0))  # roots disagree: keep world up
    up = sum(agree, Vector()) / len(agree)
    up.normalize()
    return up if up.z > 0.2 else Vector((0.0, 0.0, 1.0))


def _product_bounds(collection):
    from mathutils import Vector
    pts = []
    objs = list(collection.all_objects) if collection else [o for o in bpy.context.scene.objects]
    for obj in objs:
        if obj.type != "MESH" or obj.hide_render or _excluded(obj):
            continue
        pts += [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return pts


def _user_ground(context, collection, product_pts):
    """Return a mesh object of the user's own that forms ground under the product, else None."""
    from mathutils import Vector
    if not product_pts:
        return None
    members = set(collection.all_objects) if collection else set()
    xs = [p.x for p in product_pts]; ys = [p.y for p in product_pts]; zs = [p.z for p in product_pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    foot = max(max(xs) - min(xs), max(ys) - min(ys), 1e-3)
    bottom = min(zs)
    for obj in context.scene.objects:
        if obj in members or obj.type != "MESH" or obj.hide_render or obj.name.startswith("_wondful_") \
                or _excluded(obj):
            continue
        try:
            if not obj.visible_get():
                continue
        except Exception:
            pass
        bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        bx = [p.x for p in bb]; by = [p.y for p in bb]; bz = [p.z for p in bb]
        if not (min(bx) <= cx <= max(bx) and min(by) <= cy <= max(by)):
            continue
        if max(max(bx) - min(bx), max(by) - min(by)) < 1.5 * foot:
            continue
        if min(bz) > bottom + 0.05 * foot:
            continue  # starts above the wheels: not a ground
        return obj
    return None


def _crest_mesh(name, center, up, forward, half_len, half_wid, angle_deg, size=300.0, segments=64):
    """A hilltop mound: an elliptical flat top just larger than the product's footprint, then a
    rounded shoulder and ground that falls away by ``angle_deg`` in every direction, so the
    product reads as parked on the very top of a hill from any camera angle."""
    side = forward.cross(up).normalized()
    a = math.radians(angle_deg)
    shoulder = max(half_len, half_wid) * 0.5
    offsets = [0.0]
    drops = [0.0]
    for t in (0.2, 0.4, 0.6, 0.8, 1.0):  # rounded shoulder: slope eases from 0 to angle
        offsets.append(shoulder * t)
        drops.append(math.tan(a) * shoulder * t * t / 2.0)
    for e in (shoulder + 4.0, shoulder + 15.0, shoulder + 50.0, shoulder + size):
        offsets.append(e)
        drops.append(drops[5] + math.tan(a) * (e - shoulder))
    verts = [tuple(center)]
    ring_n = segments
    for e, h in zip(offsets, drops):
        for k in range(ring_n):
            th = 2.0 * math.pi * k / ring_n
            p = (center + forward * ((half_len + e) * math.cos(th)) + side * ((half_wid + e) * math.sin(th))
                 - up * h)
            verts.append((p.x, p.y, p.z))
    faces = []
    for k in range(ring_n):
        faces.append((0, 1 + k, 1 + (k + 1) % ring_n))
    for r in range(len(offsets) - 1):
        o0, o1 = 1 + r * ring_n, 1 + (r + 1) * ring_n
        for k in range(ring_n):
            k1 = (k + 1) % ring_n
            faces.append((o0 + k, o1 + k, o1 + k1, o0 + k1))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth = True
    return mesh


def _clay_sky_gradient(tree, bg):
    """Outdoor-style environment for paint reflections: dark ground below the horizon, a bright
    horizon band and a deeper sky above, so the clear coat shows a real horizon line, sky
    gradient and dark lower reflections instead of a flat, weightless gray."""
    nodes, links = tree.nodes, tree.links
    coord = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    ramp = nodes.new("ShaderNodeValToRGB")
    links.new(coord.outputs["Generated"], sep.inputs[0])
    remap = nodes.new("ShaderNodeMapRange")
    remap.inputs["From Min"].default_value = -1.0
    remap.inputs["From Max"].default_value = 1.0
    links.new(sep.outputs["Z"], remap.inputs["Value"])
    links.new(remap.outputs["Result"], ramp.inputs["Fac"])
    # World "Generated" is the view direction; remap its Z from [-1, 1] to [0, 1].
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (0.03, 0.03, 0.035, 1.0)
    els[1].position, els[1].color = 1.0, (0.10, 0.18, 0.38, 1.0)
    for pos, col in ((0.47, (0.10, 0.10, 0.11, 1.0)), (0.5, (1.2, 1.18, 1.12, 1.0)),
                     (0.55, (0.6, 0.7, 0.85, 1.0)), (0.7, (0.25, 0.36, 0.58, 1.0))):
        e = els.new(pos)
        e.color = col
    links.new(ramp.outputs["Color"], bg.inputs[0])


def _product_forward(pts, up):
    """Horizontal (perpendicular to ``up``) axis along which the product is longest."""
    from mathutils import Vector
    ref = Vector((1.0, 0.0, 0.0)) if abs(up.x) < 0.9 else Vector((0.0, 1.0, 0.0))
    a = (ref - up * ref.dot(up)).normalized()
    b = up.cross(a).normalized()
    best, best_len = a, -1.0
    for k in range(18):  # 10° steps over 180°
        ang = math.pi * k / 18
        axis = a * math.cos(ang) + b * math.sin(ang)
        proj = [p.dot(axis) for p in pts]
        length = max(proj) - min(proj)
        if length > best_len:
            best, best_len = axis, length
    return best, best_len




def _collection_roots(collection):
    """Objects of the collection whose parent is outside it (cameras/lights excluded)."""
    if not collection:
        return []
    objs = list(collection.all_objects)
    members = set(objs)
    return [o for o in objs if (o.parent is None or o.parent not in members) and o.type not in {"CAMERA", "LIGHT"}]


def _product_vertices(context, collection):
    """World-space vertices (N×3 numpy array) of the product's evaluated, renderable meshes."""
    import numpy as np
    depsgraph = context.evaluated_depsgraph_get()
    chunks = []
    objs = list(collection.all_objects) if collection else list(context.scene.objects)
    for obj in objs:
        if obj.type != "MESH" or obj.hide_render or obj.name.startswith("_wondful_") or _excluded(obj):
            continue
        ev = obj.evaluated_get(depsgraph)
        try:
            me = ev.to_mesh()
        except Exception:
            continue
        try:
            n = len(me.vertices)
            if not n:
                continue
            co = np.empty(n * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            m = np.array(ev.matrix_world, dtype=np.float64)
            chunks.append(co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3])
        finally:
            ev.to_mesh_clear()
    return np.concatenate(chunks) if chunks else None


def _looks_like_vehicle(collection) -> bool:
    from .product_kind import names_look_like_vehicle, collection_names
    return names_look_like_vehicle(collection_names(collection))


_LOCAL_AXES = {"NEG_Y": (0.0, -1.0, 0.0), "POS_Y": (0.0, 1.0, 0.0), "NEG_X": (-1.0, 0.0, 0.0), "POS_X": (1.0, 0.0, 0.0)}


def _terrain_forward(scene, collection, P, axis_mode="AUTO"):
    """Horizontal forward of the product and how it was chosen.

    Manual: the chosen local axis of the collection root(s).  AUTO: Blender/glTF convention
    (front faces local -Y); for a vehicle whose -Y is across its length, the long axis is used,
    pointing to the camera's side (3/4 front shots)."""
    import numpy as np
    from mathutils import Vector
    up = Vector((0.0, 0.0, 1.0))
    roots = _collection_roots(collection)

    def horiz(v):
        v = Vector((v.x, v.y, 0.0))
        return v.normalized() if v.length > 1e-6 else None

    local = Vector(_LOCAL_AXES.get(axis_mode, _LOCAL_AXES["NEG_Y"]))
    cands = [horiz(r.matrix_world.to_3x3() @ local) for r in roots]
    cands = [c for c in cands if c is not None]
    cand = (sum(cands, Vector()) / len(cands)) if cands else horiz(local)
    cand = horiz(cand) if cand is not None and cand.length > 1e-6 else Vector((0.0, -1.0, 0.0))
    if axis_mode in _LOCAL_AXES:
        return cand, "manual:" + axis_mode
    xy = P[:, :2]
    # Minimum-area footprint rectangle (1° steps over 90°): its long side is the length axis.
    # (The longest projection would point along the footprint diagonal instead.)
    best, best_len, width, best_area = None, -1.0, 0.0, None
    for k in range(90):
        a = math.pi * k / 180
        ax, pe = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        pa, pp = xy @ ax, xy @ pe
        la, lp = float(pa.max() - pa.min()), float(pp.max() - pp.min())
        if best_area is None or la * lp < best_area - 1e-12:
            best_area = la * lp
            if la >= lp:
                best, best_len, width = Vector((ax[0], ax[1], 0.0)), la, lp
            else:
                best, best_len, width = Vector((pe[0], pe[1], 0.0)), lp, la
    if abs(cand.dot(best)) > 0.7:
        return (best if cand.dot(best) > 0 else -best), "auto:-Y"
    if best_len > 1.3 * max(width, 1e-6) and _looks_like_vehicle(collection):
        # vehicles are longest front-to-back; furniture is often widest side-to-side, so only
        # vehicles use the long axis
        cam = scene.camera
        if cam is not None:
            centre = Vector((float(xy[:, 0].mean()), float(xy[:, 1].mean()), 0.0))
            to_cam = Vector((cam.matrix_world.translation.x, cam.matrix_world.translation.y, 0.0)) - centre
            return (best if best.dot(to_cam) >= 0 else -best), "auto:long-axis-to-camera"
        return best, "auto:long-axis"
    return cand, "auto:-Y"


def _basis_snapshot(obj):
    if obj is None:
        return None
    return (obj.location.copy(), obj.rotation_mode, obj.rotation_euler.copy(), obj.rotation_quaternion.copy(),
            tuple(obj.rotation_axis_angle), obj.scale.copy())


def _basis_restore(obj, snap):
    if obj is None or snap is None:
        return
    loc, mode, eul, quat, aa, scale = snap
    obj.rotation_mode = mode
    obj.location = loc
    obj.rotation_euler = eul
    obj.rotation_quaternion = quat
    obj.rotation_axis_angle = aa
    obj.scale = scale


def _pose_product(context, collection, W, created, restore):
    """Apply world matrix ``W`` to the product temporarily without touching its transforms.

    Each root is re-parented under a temporary empty that carries the tilt, so the user's
    location/rotation/scale (even animated ones) stay untouched; ``restore`` puts the original
    parent and parent-inverse back.  The scene camera never moves."""
    from mathutils import Matrix
    scene = context.scene
    cam = scene.camera
    cam_world = cam.matrix_world.copy() if cam else None
    cam_snap = _basis_snapshot(cam)
    for r in _collection_roots(collection):
        parent = r.parent
        if parent is not None and r.parent_type != "OBJECT":
            snap = _basis_snapshot(r)
            restore.append(lambda r=r, snap=snap: _basis_restore(r, snap))
            r.matrix_world = W @ r.matrix_world
            continue
        mpi = r.matrix_parent_inverse.copy()
        e = bpy.data.objects.new("_wondful_terrain_pose", None)
        scene.collection.objects.link(e)
        created.append(("obj", e))
        if parent is not None:
            e.parent = parent
            e.matrix_parent_inverse = mpi
            base = parent.matrix_world @ mpi
        else:
            base = Matrix.Identity(4)
        e.matrix_basis = base.inverted() @ W @ base
        restore.append(lambda r=r, parent=parent, mpi=mpi: (setattr(r, "parent", parent),
                                                            setattr(r, "matrix_parent_inverse", mpi)))
        r.parent = e
        r.matrix_parent_inverse = Matrix.Identity(4)
    context.view_layer.update()
    if cam is not None:
        diff = max(abs(a - b) for ra, rb in zip(cam.matrix_world, cam_world) for a, b in zip(ra, rb))
        if diff > 1e-7:  # camera was parented inside the product: keep the shot fixed
            restore.append(lambda: _basis_restore(cam, cam_snap))
            cam.matrix_world = cam_world
            context.view_layer.update()


def _mass_centre(P, cells=24):
    """Centre of the occupied voxels of the product's vertices: a centre-of-mass stand-in that a
    densely tessellated part (wheel spokes, stitching) cannot drag toward itself."""
    import numpy as np
    lo, hi = P.min(0), P.max(0)
    ext = np.maximum(hi - lo, 1e-9)
    idx = np.clip(((P - lo) / ext * cells).astype(int), 0, cells - 1)
    occ = np.unique(idx[:, 0] * cells * cells + idx[:, 1] * cells + idx[:, 2])
    ijk = np.stack([occ // (cells * cells), (occ // cells) % cells, occ % cells], axis=1)
    return lo + (ijk.mean(0) + 0.5) / cells * ext


def _custom_terrain(context, collection, settings, created, restore):
    """Build the 自定义 terrain mesh and (temporarily) rest the product on it.

    The product is rigid: its stable pose (pitch, roll, height) on the height field comes from
    terrain.rest_pose (lowest centre of mass with nothing below the ground: usually three support
    points, the fourth corner of a twisted ground lifted).  Returns (mesh, info) or (None, info)."""
    import numpy as np
    from mathutils import Matrix, Vector
    from . import terrain as TR
    s = TR.normalize(settings)
    info = {"mode": "CUSTOM"}
    if TR.is_flat(s):
        info["flat"] = True
        return None, info
    P = _product_vertices(context, collection)
    if P is None or not len(P):
        return None, info
    scene = context.scene
    up = Vector((0.0, 0.0, 1.0))
    fwd, how = _terrain_forward(scene, collection, P, s["front_axis"])
    side = fwd.cross(up).normalized()
    F, S = np.array(fwd), np.array(side)
    u, v, z = P @ F, P @ S, P[:, 2]
    zmin, height = float(z.min()), float(z.max() - z.min())
    uc, vc = float(u.max() + u.min()) / 2.0, float(v.max() + v.min()) / 2.0
    half_len, half_wid = float(u.max() - u.min()) / 2.0, float(v.max() - v.min()) / 2.0
    C = fwd * uc + side * vc
    C.z = zmin
    Cn = np.array(C)
    footprint = max(2 * half_len, 2 * half_wid, 1e-3)
    span = max(footprint, height, 1e-3)
    margin = 0.06 * footprint
    U, V = half_len + margin, half_wid + margin
    shoulder = 0.5 * max(U, V)
    hf = TR.HeightField(s, U, V, shoulder, prod_height=max(height, 1e-3))
    rel = P - Cn
    com = _mass_centre(rel)
    sup = TR.support_points(rel, F, S)
    Rn, tn, pitch, roll = TR.rest_pose(sup, com, hf, F, S, follow=s["follow"])
    Q = rel @ Rn.T + tn
    extra = float(np.max(hf.height(Q @ F, Q @ S) - Q[:, 2]))  # every vertex, not just the underside
    Q[:, 2] += extra
    tn = tn + np.array((0.0, 0.0, extra))
    lift = float(tn[2])
    eps = max(0.004 * span, 1e-4)
    rep = TR.contact_report(Q, Rn @ com + tn, hf, F, S, eps, footprint)
    lifted, gap = TR.lifted_corner(rep, footprint)
    M4 = Matrix([list(map(float, Rn[0])) + [float(tn[0])], list(map(float, Rn[1])) + [float(tn[1])],
                 list(map(float, Rn[2])) + [float(tn[2])], [0.0, 0.0, 0.0, 1.0]])
    W = Matrix.Translation(C) @ M4 @ Matrix.Translation(-C)
    posed = bool(np.abs(Rn - np.eye(3)).max() > 1e-7 or np.abs(tn).max() > 1e-6)
    if posed:
        _pose_product(context, collection, W, created, restore)
    us = TR.grid_offsets(U, shoulder, span)
    vs = TR.grid_offsets(V, shoulder, span)
    ug, vg = np.meshgrid(np.array(us), np.array(vs), indexing="ij")
    hg = hf.height(ug, vg)
    cap = None
    carved = 0.0
    cam = scene.camera
    if cam is not None:
        # The camera never moves: if rising ground would bury it, level the terrain off just
        # below the lens (outside the pad only), like a slope that flattens out where the
        # photographer stands.
        rel_cam = np.array(cam.matrix_world.translation) - Cn
        cu, cv, cz = float(rel_cam @ F), float(rel_cam @ S), float(rel_cam[2])
        room = max(0.15 * height, 0.05 * span)
        outside = (np.abs(ug) > U) | (np.abs(vg) > V)
        if float(hf.height(cu, cv)) > cz - room or float(hg.max()) > cz - room:
            near_cam = np.hypot(ug - cu, vg - cv) < 2.0 * span
            if float(hf.height(cu, cv)) > cz - room or bool((hg[near_cam] > cz - room).any()):
                cap = cz - room
                band = 0.1 * span
                soft = hg - np.square(np.clip(hg - cap + band, 0.0, 2 * band)) / (4.0 * band)
                capped = np.where(hg >= cap + band, cap, np.where(hg > cap - band, soft, hg))
                hg = np.where(outside, capped, hg)
        # Keep the line of sight from the lens to the product's base clear: ground between the
        # camera and the product (within the product's bearing, eased at the edges) is kept just
        # under the sight line, so rising ground toward the camera reads as a crest instead of a
        # wall that hides the product.
        near_u, near_v = max(-U, min(U, cu)), max(-V, min(V, cv))
        d_t = math.hypot(cu - near_u, cv - near_v)
        if d_t > 1e-3:
            z_t = float(hf.height(near_u, near_v))
            du, dv = ug - cu, vg - cv
            dist = np.hypot(du, dv)
            to_c = math.atan2(-cv, -cu)
            corners = [math.atan2(sv * V - cv, su * U - cu) for su in (-1, 1) for sv in (-1, 1)]
            wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
            half = max(abs(wrap(a - to_c)) for a in corners)
            ang = np.abs(np.angle(np.exp(1j * (np.arctan2(dv, du) - to_c))))
            fall = math.radians(15.0)
            wa = np.clip(1.0 - (ang - half) / fall, 0.0, 1.0)
            wd = np.clip((d_t - dist) / (0.2 * d_t), 0.0, 1.0)
            w = wa * wd
            w = w * w * (3 - 2 * w)  # smoothstep
            sight = cz + (z_t - cz) * dist / d_t - 0.02 * span
            excess = np.clip(hg - sight, 0.0, None)
            carve = np.where(outside, w * excess, 0.0)
            if float(carve.max()) > 1e-4:
                hg = hg - carve
                carved = round(float(carve.max()), 4)
    pts = Cn[None, None, :] + ug[..., None] * F + vg[..., None] * S + hg[..., None] * np.array((0.0, 0.0, 1.0))
    nu, nv = len(us), len(vs)
    verts = pts.reshape(-1, 3)
    ii, jj = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1), indexing="ij")
    a = (ii * nv + jj).ravel()
    quads = np.stack([a, a + nv, a + nv + 1, a + 1], axis=1)
    mesh = bpy.data.meshes.new("_wondful_clay_floor")
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    mesh.loops.add(quads.size)
    mesh.loops.foreach_set("vertex_index", quads.astype(np.int32).ravel())
    mesh.polygons.add(len(quads))
    mesh.polygons.foreach_set("loop_start", (np.arange(len(quads)) * 4).astype(np.int32))
    mesh.polygons.foreach_set("loop_total", np.full(len(quads), 4, np.int32))
    mesh.update(calc_edges=True)
    mesh.validate()
    mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), bool))
    vehicle = s.get("vehicle")
    if vehicle is None:
        vehicle = _looks_like_vehicle(collection)
    hints = {}
    if cam is not None:
        inv = cam.matrix_world.to_3x3().inverted()
        for d, vec in (("front", fwd), ("right", side)):
            c = inv @ vec
            opp = {"front": "back", "right": "left"}[d]
            if abs(c.x) >= 0.35:
                a1, b1 = ("画面右侧", "画面左侧") if c.x > 0 else ("画面左侧", "画面右侧")
                if d == "front":
                    a1, b1 = "朝" + a1, "朝" + b1
            else:
                if d == "front":
                    a1, b1 = ("朝向镜头", "背向镜头") if c.z > 0 else ("背向镜头", "朝向镜头")
                else:
                    a1, b1 = ("靠近镜头一侧", "远离镜头一侧") if c.z > 0 else ("远离镜头一侧", "靠近镜头一侧")
            hints[d], hints[opp] = a1, b1
        try:
            from bpy_extras.object_utils import world_to_camera_view
            proj = {}
            for k in TR.CORNERS:
                su = 1.0 if k[0] == "f" else -1.0
                sv = 1.0 if k[1] == "r" else -1.0
                p = C + fwd * (su * U) + side * (sv * V)
                p.z += float(hf.height(su * U, sv * V))
                proj[k] = world_to_camera_view(scene, cam, p)
            # four distinct labels: nearest / farthest from the lens, the other two by picture side
            by_depth = sorted(TR.CORNERS, key=lambda k: proj[k].z)
            hints[by_depth[0]] = "离镜头最近"
            hints[by_depth[-1]] = "离镜头最远"
            mid = sorted(by_depth[1:3], key=lambda k: proj[k].x)
            hints[mid[0]], hints[mid[1]] = "画面左侧", "画面右侧"
        except Exception:
            pass
    pose = {"pitch": pitch, "roll": roll, "lifted": lifted, "gap": gap if lifted else 0.0}
    text = TR.describe(s, vehicle=bool(vehicle), hints=hints, pose=pose)
    comp = TR.components(s)
    info.update({
        "forward": [round(x, 4) for x in fwd], "forward_how": how, "pitch_deg": round(pitch, 2),
        "roll_deg": round(roll, 2), "lift": round(lift, 5), "posed": posed, "follow": s["follow"],
        "contact_points": rep["contact_points"], "contact_groups": rep["contact_groups"],
        "group_corners": rep["group_corners"], "com_inside": rep["com_inside"],
        "corner_clearance": {k: round(x, 5) for k, x in rep["corner_clearance"].items()},
        "lifted_corner": lifted, "lifted_gap": round(gap, 5), "contact_eps": round(eps, 5),
        "min_clearance": round(rep["min_clearance"], 6), "pivot": [round(x, 4) for x in C],
        "vehicle": bool(vehicle), "hints": hints, "camera_cap": None if cap is None else round(cap, 4),
        "sightline_carve": carved, "support_points": int(len(sup)),
        "settings": {k: s[k] for k in TR.CORNERS + ("crest", "bump")},
        "components": {k: round(x, 3) for k, x in comp.items()},
        "text": text,
    })
    return mesh, info


def _render_still(scene):
    """``render.render(write_still=True)``; a Cycles build without OpenImageDenoise raises
    "Failed to denoise" when denoising is on: retry once without denoising."""
    try:
        bpy.ops.render.render(write_still=True)
    except RuntimeError as exc:
        cycles = getattr(scene, "cycles", None)
        if "denois" not in str(exc).lower() or cycles is None or not cycles.use_denoising:
            raise
        print("[Wondful] 当前 Blender 不支持降噪，改为不降噪渲染")
        cycles.use_denoising = False
        bpy.ops.render.render(write_still=True)


_GEOMETRY_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES", "POINTCLOUD", "VOLUME",
                   "GPENCIL", "GREASEPENCIL"}


def _render_mask_now(context, filepath: str, collection, holdout=()) -> bool:
    """Render only the product collection, opaque, on a transparent film: its alpha is the
    product silhouette from the scene camera (1.2.0 分层输出). Caller restores render settings
    it already snapshots; this restores what it changes itself.

    ``holdout`` (1.2.0 环境体块): massing objects stay in the render as holdouts, so a block in
    front of the product cuts its silhouette and no block ever enters the subject layer."""
    scene = context.scene
    render = scene.render
    product = set(collection.all_objects) if collection is not None else set()
    if not product:
        return False
    hidden = []
    held = []
    holdout = [o for o in holdout or () if o not in product and not o.hide_render]
    view_layer = context.view_layer
    old_override = view_layer.material_override
    old_engine = render.engine
    old_film = getattr(render, "film_transparent", False)
    old_path = render.filepath
    old_mode = render.image_settings.color_mode
    cycles = getattr(scene, "cycles", None)
    old_cycles = (cycles.samples, cycles.use_denoising, cycles.max_bounces) if cycles is not None else None
    mat = bpy.data.materials.new("_wondful_mask_mat")
    try:
        hold = set(holdout)
        for obj in scene.objects:
            if obj in hold:
                if not getattr(obj, "is_holdout", True):
                    obj.is_holdout = True
                    held.append(obj)
                continue
            if obj.type in _GEOMETRY_TYPES and obj not in product and not obj.hide_render:
                obj.hide_render = True
                hidden.append(obj)
        view_layer.material_override = mat  # glass and alpha-blended parts count as solid
        render.film_transparent = True
        render.filepath = filepath
        _set_png(render.image_settings)
        render.image_settings.color_mode = "RGBA"
        # Workbench is instant with a GPU context; background Blender may have none (no EGL)
        # and aborts instead of raising, so it goes straight to Cycles with camera rays only.
        engines = [os.environ.get("WONDFUL_MASK_ENGINE", "").strip().upper()]
        engines += ["CYCLES"] if bpy.app.background else ["BLENDER_WORKBENCH", "CYCLES"]
        for engine in [e for e in engines if e]:
            try:
                render.engine = engine
                if engine == "CYCLES" and cycles is not None:
                    cycles.samples = 8
                    cycles.use_denoising = False
                    cycles.max_bounces = 0
                bpy.ops.render.render(write_still=True)
                if Path(filepath).exists():
                    return True
            except Exception as exc:
                print(f"[Wondful] 产品遮罩用 {engine} 渲染失败：{exc}")
        return False
    finally:
        for obj in hidden:
            try:
                obj.hide_render = False
            except Exception:
                pass
        for obj in held:
            try:
                obj.is_holdout = False
            except Exception:
                pass
        view_layer.material_override = old_override
        try:
            bpy.data.materials.remove(mat)
        except Exception:
            pass
        render.engine = old_engine
        if hasattr(render, "film_transparent"):
            render.film_transparent = old_film
        render.filepath = old_path
        try:
            render.image_settings.color_mode = old_mode
        except Exception:
            pass
        if old_cycles is not None:
            cycles.samples, cycles.use_denoising, cycles.max_bounces = old_cycles


def keep_exact_slots(collection, with_glass=False):
    """1.2.0 细节回贴: {(object, slot index or -1): 'explicit' | 'auto' | 'glass' | 'matte'} for the
    product's keep-exact parts (and, with ``with_glass``, its glass / transparent and matte /
    textile parts).

    Explicit = custom property ``wondful_keep_exact`` on the object or its material
    (「标为原样保留」); automatic = names of plates / badges / logos (plate, license, logo,
    badge, emblem, 车牌, 标 …). An explicit False on the object switches the automatic
    detection off for it."""
    from .detail_paste import KEEP_PROP, classify_slot
    out = {}
    if collection is None:
        return out
    for obj in collection.all_objects:
        if obj.type not in _GEOMETRY_TYPES or obj in _MASSING_EXCLUDE:
            continue
        oprop = obj.get(KEEP_PROP) if hasattr(obj, "get") else None
        slots = list(getattr(obj, "material_slots", []) or [])
        if not slots:
            kind = classify_slot(obj.name, oprop, "", None)
            if kind:
                out[(obj, -1)] = kind
            continue
        for i, slot in enumerate(slots):
            mat = slot.material
            kind = classify_slot(obj.name, oprop, mat.name if mat else "", mat.get(KEEP_PROP) if mat else None)
            if not kind and with_glass and _material_is_glass(mat):
                kind = "glass"
            if not kind and with_glass and _material_is_matte(mat):
                kind = "matte"
            if kind:
                out[(obj, i)] = kind
    return out


def mirrored_keep_parts(collection, camera, threshold: float = 0.6) -> list:
    """1.2.2: keep-exact parts (plates / badges / logos) whose texture reads mirrored from the
    scene camera in Blender itself (mirrored UVs or a negative object scale). 细节回贴 pastes
    these exactly as Blender shows them, so a mirrored model plate stays mirrored in the
    result — the panel warns instead of silently "fixing" the user's model."""
    from .detail_paste import mirrored_text_fraction
    if collection is None or camera is None:
        return []
    cam_pos = tuple(camera.matrix_world.translation)
    found = []
    for (obj, slot), kind in keep_exact_slots(collection, with_glass=False).items():
        if kind not in {"explicit", "auto"} or obj.type != "MESH" or obj.data is None:
            continue
        me = obj.data
        uv = me.uv_layers.active
        if uv is None:
            continue
        try:
            me.calc_loop_triangles()
        except Exception:
            continue
        mw = obj.matrix_world
        pos, uvs = [], []
        for tri in me.loop_triangles:
            if slot >= 0 and tri.material_index != slot:
                continue
            pos.append([tuple(mw @ me.vertices[v].co) for v in tri.vertices])
            uvs.append([tuple(uv.data[l].uv) for l in tri.loops])
        if not pos:
            continue
        frac = mirrored_text_fraction(pos, uvs, cam_pos, negative_scale=mw.determinant() < 0)
        if frac >= threshold:
            mat = obj.material_slots[slot].material if 0 <= slot < len(obj.material_slots) else None
            found.append({"object": obj.name, "material": mat.name if mat else "", "fraction": round(frac, 2)})
    return found


def _material_is_matte(mat) -> bool:
    """Matte / textile material (fabric, velvet, wood, leather, rubber): it shows no sharp
    reflections, so any fine texture the AI draws on it is invented; 细节回贴 uses the
    Blender texture there even where it is smooth."""
    if mat is None:
        return False
    from .detail_paste import name_looks_matte
    if name_looks_matte(mat.name):
        return True
    try:
        tree = mat.node_tree if mat.use_nodes else None
        for node in (tree.nodes if tree else ()):
            if node.type != "BSDF_PRINCIPLED":
                continue
            sheen = node.inputs.get("Sheen Weight") or node.inputs.get("Sheen")
            if sheen is not None and not sheen.is_linked and float(sheen.default_value) > 0.2:
                return True
            rough, metal = node.inputs.get("Roughness"), node.inputs.get("Metallic")
            if rough is not None and not rough.is_linked and float(rough.default_value) >= 0.6 \
                    and (metal is None or (not metal.is_linked and float(metal.default_value) < 0.5)):
                return True
    except Exception:
        return False
    return False


def _material_is_glass(mat) -> bool:
    """Transparent / transmissive material (glass, windows, light covers): the AI renders its
    reflections and what shows through, so 细节回贴 leaves it to the AI."""
    if mat is None:
        return False
    from .detail_paste import name_looks_glass, name_looks_lamp
    if name_looks_lamp(mat.name):
        return False  # lamp covers: the lamp graphics behind them are identity detail, keep pasting
    if name_looks_glass(mat.name):
        return True
    try:
        if getattr(mat, "blend_method", "OPAQUE") in {"BLEND"}:
            return True
        tree = mat.node_tree if mat.use_nodes else None
        for node in (tree.nodes if tree else ()):
            if node.type == "BSDF_PRINCIPLED":
                for key in ("Transmission Weight", "Transmission"):
                    sock = node.inputs.get(key)
                    if sock is not None and not sock.is_linked and float(sock.default_value) > 0.3:
                        return True
                alpha = node.inputs.get("Alpha")
                if alpha is not None and not alpha.is_linked and float(alpha.default_value) < 0.6:
                    return True
            elif node.type in {"BSDF_GLASS", "BSDF_TRANSPARENT", "BSDF_REFRACTION"}:
                out = node.outputs[0]
                if out.is_linked:
                    return True
    except Exception:
        return False
    return False


def keep_exact_texts(collection) -> list:
    """Texts the user typed in ``wondful_keep_text`` on product objects (for the prompt)."""
    from .detail_paste import KEEP_TEXT_PROP
    texts = []
    for obj in (collection.all_objects if collection is not None else ()):
        value = obj.get(KEEP_TEXT_PROP) if hasattr(obj, "get") else None
        if isinstance(value, str) and value.strip() and value.strip() not in texts:
            texts.append(value.strip())
    return texts


def _emission_material(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*rgb, 1.0)
    em.inputs["Strength"].default_value = 1.0
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], outn.inputs["Surface"])
    return mat


def _holdout_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    ho = nt.nodes.new("ShaderNodeHoldout")
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(ho.outputs[0], outn.inputs["Surface"])
    return mat


def _isolate_product(scene, product, holdout, hidden, held, hide_lights=False):
    hold = set(holdout)
    for obj in scene.objects:
        if obj in hold:
            if not getattr(obj, "is_holdout", True):
                obj.is_holdout = True
                held.append(obj)
            continue
        if obj in product or obj.hide_render:
            continue
        if obj.type in _GEOMETRY_TYPES or (hide_lights and obj.type == "LIGHT"):
            obj.hide_render = True
            hidden.append(obj)


def _unisolate(hidden, held):
    for obj in hidden:
        try:
            obj.hide_render = False
        except Exception:
            pass
    for obj in held:
        try:
            obj.is_holdout = False
        except Exception:
            pass


def material_color_kind(mat) -> str:
    """按材质校色 kind of one material (see ``material_color.material_kind``)."""
    from .material_color import material_kind
    from .detail_paste import name_looks_lamp
    if mat is None:
        return "default"
    name = mat.name
    vals = {"metallic": 0.0, "roughness": 0.5, "coat": 0.0}
    emissive = name_looks_lamp(name)
    try:
        tree = mat.node_tree if mat.use_nodes else None
        out = None
        for node in (tree.nodes if tree else ()):
            if node.type == "OUTPUT_MATERIAL" and (out is None or getattr(node, "is_active_output", False)):
                out = node
        for node in (tree.nodes if tree else ()):
            if node.type == "BSDF_PRINCIPLED":
                for key, sock_names in (("metallic", ("Metallic",)), ("roughness", ("Roughness",)),
                                        ("coat", ("Coat Weight", "Clearcoat"))):
                    for sn in sock_names:
                        sock = node.inputs.get(sn)
                        if sock is not None and not sock.is_linked:
                            vals[key] = max(vals[key], float(sock.default_value)) if key != "roughness" \
                                else float(sock.default_value)
                            break
                es = node.inputs.get("Emission Strength")
                ec = node.inputs.get("Emission Color") or node.inputs.get("Emission")
                if es is not None and (es.is_linked or float(es.default_value) > 0.05) and ec is not None and \
                        (ec.is_linked or max(ec.default_value[:3]) > 0.05):
                    emissive = True
            elif node.type == "EMISSION" and node.outputs[0].is_linked:
                emissive = True
        if out is not None and out.inputs["Surface"].is_linked:
            src = out.inputs["Surface"].links[0].from_node
            if src.type == "EMISSION":
                emissive = True
    except Exception:
        pass
    return material_kind(name, glass=_material_is_glass(mat), emissive=emissive,
                         matte=_material_is_matte(mat), **vals)


def _render_detail_now(context, filepath: str, collection, holdout=(), keep_path: str = "",
                       matid_path: str = "") -> dict:
    """1.2.0 细节回贴: clean detail pass of the product only, with its own materials, under a
    flat even light (uniform white world, no lamps), transparent film, Standard view
    transform. Optionally also the keep-exact mask (R = explicit, G = automatic, A =
    coverage, B = glass / transparent parts left to the AI, black = matte / textile parts
    whose texture always comes from Blender; other product parts are holdouts so they still
    occlude). Optionally the material-ID pass for 按材质校色 (``matid_path``: every product
    material as one flat palette emission colour, no anti-aliasing, plus ``<path>.json`` with
    the material list and kinds). Restores every
    setting it changes. Returns {"detail": bool, "keep": bool, "matid": bool, "keep_slots": n, "engine": str}."""
    scene = context.scene
    render = scene.render
    product = set(collection.all_objects) if collection is not None else set()
    info = {"detail": False, "keep": False, "matid": False, "keep_slots": 0, "engine": ""}
    if not product:
        return info
    hidden, held, created, slot_undo = [], [], [], []
    holdout = [o for o in holdout or () if o not in product and not o.hide_render]
    view_layer = context.view_layer
    old = {"override": view_layer.material_override, "engine": render.engine,
           "film": getattr(render, "film_transparent", False), "path": render.filepath,
           "mode": render.image_settings.color_mode, "world": scene.world}
    vs = scene.view_settings
    old_cm = (vs.view_transform, vs.look, vs.exposure, vs.gamma)
    cycles = getattr(scene, "cycles", None)
    old_cycles = (cycles.samples, cycles.use_denoising, cycles.max_bounces) if cycles is not None else None
    eevee = getattr(scene, "eevee", None)
    old_eevee = getattr(eevee, "taa_render_samples", None) if eevee is not None else None
    old_filter = getattr(cycles, "filter_width", None) if cycles is not None else None
    old_dither = getattr(render, "dither_intensity", None)
    orig_mats = {}
    for obj in product:
        for i, slot in enumerate(getattr(obj, "material_slots", []) or []):
            orig_mats[(obj, i)] = slot.material
    try:
        _isolate_product(scene, product, holdout, hidden, held, hide_lights=True)
        view_layer.material_override = None
        world = bpy.data.worlds.new("_wondful_detail_world")
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
            bg.inputs[1].default_value = float(os.environ.get("WONDFUL_DETAIL_LIGHT", "1.0"))
        scene.world = world
        created.append(("world", world))
        render.film_transparent = True
        _set_png(render.image_settings)
        render.image_settings.color_mode = "RGBA"
        try:
            vs.view_transform, vs.look, vs.exposure, vs.gamma = "Standard", "None", 0.0, 1.0
        except Exception:
            pass
        forced = os.environ.get("WONDFUL_DETAIL_ENGINE", "").strip().upper()
        # Eevee is fast in the UI; background Blender may have no GPU context (no EGL) and
        # Eevee/Workbench abort there instead of raising, so it uses Cycles + denoise.
        engines = [forced] if forced else (["CYCLES"] if bpy.app.background
                                           else ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"])
        for engine in engines:
            try:
                render.engine = engine
                break
            except Exception:
                continue
        if render.engine == "CYCLES" and cycles is not None:
            cycles.samples = int(os.environ.get("WONDFUL_DETAIL_SAMPLES", "128"))
            cycles.use_denoising = os.environ.get("WONDFUL_DETAIL_DENOISE", "1") != "0"
            cycles.max_bounces = max(4, int(old_cycles[2] if old_cycles else 8))
        elif eevee is not None and old_eevee is not None:
            eevee.taa_render_samples = max(64, int(old_eevee))
        info["engine"] = render.engine
        render.filepath = filepath
        try:
            _render_still(scene)
            info["detail"] = Path(filepath).exists()
        except Exception as exc:
            if render.engine != "CYCLES":   # one more try on Cycles
                print(f"[Wondful] 细节底图 {render.engine} 失败：{exc}，改用 Cycles")
                render.engine = "CYCLES"
                info["engine"] = "CYCLES"
                if cycles is not None:
                    cycles.samples = int(os.environ.get("WONDFUL_DETAIL_SAMPLES", "128"))
                    cycles.use_denoising = True
                _render_still(scene)
                info["detail"] = Path(filepath).exists()
            else:
                raise
        slots = keep_exact_slots(collection, with_glass=True) if keep_path else {}
        info["keep_slots"] = sum(1 for k in slots.values() if k in {"explicit", "auto"})
        info["glass_slots"] = sum(1 for k in slots.values() if k == "glass")
        info["matte_slots"] = sum(1 for k in slots.values() if k == "matte")
        if slots:
            explicit = _emission_material("_wondful_keep_explicit", (1.0, 0.0, 0.0))
            auto = _emission_material("_wondful_keep_auto", (0.0, 1.0, 0.0))
            glass = _emission_material("_wondful_keep_glass", (0.0, 0.0, 1.0))
            matte = _emission_material("_wondful_keep_matte", (0.0, 0.0, 0.0))
            hold_mat = _holdout_material("_wondful_keep_holdout")
            created += [("mat", explicit), ("mat", auto), ("mat", glass), ("mat", matte), ("mat", hold_mat)]
            for obj in product:
                if obj.type not in _GEOMETRY_TYPES or obj.hide_render:
                    continue
                mats = getattr(getattr(obj, "data", None), "materials", None)
                if mats is None:
                    continue
                if len(obj.material_slots) == 0:
                    mats.append(None)
                    slot_undo.append(lambda mats=mats: mats.clear() if len(mats) <= 1 else mats.pop())
                for i, slot in enumerate(list(obj.material_slots)):
                    kind = slots.get((obj, i)) or slots.get((obj, -1))
                    old_link = slot.link
                    if old_link != "OBJECT":
                        slot.link = "OBJECT"
                    old_mat = slot.material

                    def undo(slot=slot, link=old_link, mat=old_mat):
                        if slot.link != "OBJECT":
                            slot.link = "OBJECT"
                        slot.material = mat
                        if link != "OBJECT":
                            slot.link = link
                    slot_undo.append(undo)
                    slot.material = {"explicit": explicit, "auto": auto, "glass": glass, "matte": matte}.get(kind, hold_mat)
            render.engine = "CYCLES"
            if cycles is not None:
                cycles.samples = 16
                cycles.use_denoising = False
                cycles.max_bounces = 0
            render.filepath = keep_path
            bpy.ops.render.render(write_still=True)
            info["keep"] = Path(keep_path).exists()
        if matid_path:
            try:
                t_id = time.time()
                info["matid"] = _render_matid(scene, product, matid_path, orig_mats, created, slot_undo)
                info["matid_seconds"] = round(time.time() - t_id, 1)
            except Exception as exc:
                print(f"[Wondful] 材质 ID 渲染失败（跳过按材质校色）：{exc}")
        return info
    finally:
        for undo in reversed(slot_undo):
            try:
                undo()
            except Exception as exc:
                print(f"[Wondful] 恢复材质槽失败：{exc}")
        _unisolate(hidden, held)
        view_layer.material_override = old["override"]
        render.engine = old["engine"]
        if hasattr(render, "film_transparent"):
            render.film_transparent = old["film"]
        render.filepath = old["path"]
        try:
            render.image_settings.color_mode = old["mode"]
        except Exception:
            pass
        try:
            vs.view_transform, vs.look, vs.exposure, vs.gamma = old_cm
        except Exception:
            pass
        scene.world = old["world"]
        if old_cycles is not None:
            cycles.samples, cycles.use_denoising, cycles.max_bounces = old_cycles
        if old_filter is not None:
            try:
                cycles.filter_width = old_filter
            except Exception:
                pass
        if old_dither is not None:
            try:
                render.dither_intensity = old_dither
            except Exception:
                pass
        if old_eevee is not None:
            try:
                eevee.taa_render_samples = old_eevee
            except Exception:
                pass
        for kind, block in created:
            try:
                {"mat": bpy.data.materials, "world": bpy.data.worlds}[kind].remove(block)
            except Exception:
                pass


def _render_matid(scene, product, matid_path, orig_mats, created, slot_undo) -> bool:
    """Material-ID pass (按材质校色): each distinct product material -> one palette emission
    colour (``material_color.palette_colour``), Cycles, 4 samples, pixel filter 0.01 px (no
    anti-aliasing mixes between IDs), no dither; writes ``matid_path`` and ``<path>.json``.
    Slot overrides are appended to ``slot_undo`` (restored to ``orig_mats`` by the caller)."""
    import json
    from .material_color import palette_colour, srgb_to_linear_scalar, sidecar_path, MAX_MATERIALS
    render = scene.render
    cycles = getattr(scene, "cycles", None)
    ids, mats_out, id_mats = {}, [], {}

    def id_material(mat):
        key = mat.name if mat is not None else None
        if key not in ids:
            if len(ids) >= MAX_MATERIALS:
                return None
            idx = len(ids)
            ids[key] = idx
            rgb = palette_colour(idx)
            em = _emission_material(f"_wondful_matid_{idx}", tuple(srgb_to_linear_scalar(c / 255.0) for c in rgb))
            created.append(("mat", em))
            id_mats[key] = em
            mats_out.append({"index": idx, "name": key or "(无材质)", "kind": material_color_kind(mat),
                             "rgb": list(rgb)})
        return id_mats[key]

    for obj in sorted(product, key=lambda o: o.name):     # stable material order
        if obj.type not in _GEOMETRY_TYPES or obj.hide_render:
            continue
        mats = getattr(getattr(obj, "data", None), "materials", None)
        if mats is None:
            continue
        if len(obj.material_slots) == 0:
            mats.append(None)
            slot_undo.append(lambda mats=mats: mats.clear() if len(mats) <= 1 else mats.pop())
        for i, slot in enumerate(list(obj.material_slots)):
            old_link, old_mat = slot.link, slot.material
            if old_link != "OBJECT":
                slot.link = "OBJECT"

            def undo(slot=slot, link=old_link, mat=old_mat):
                if slot.link != "OBJECT":
                    slot.link = "OBJECT"
                slot.material = mat
                if link != "OBJECT":
                    slot.link = link
            slot_undo.append(undo)
            slot.material = id_material(orig_mats.get((obj, i), old_mat if old_mat is None or
                                                      not old_mat.name.startswith("_wondful_") else None))
    render.engine = "CYCLES"
    if cycles is not None:
        cycles.samples = 4
        cycles.use_denoising = False
        cycles.max_bounces = 0
        if hasattr(cycles, "filter_width"):
            cycles.filter_width = 0.01
    if hasattr(render, "dither_intensity"):
        render.dither_intensity = 0.0
    render.filepath = matid_path
    bpy.ops.render.render(write_still=True)
    with open(sidecar_path(matid_path), "w", encoding="utf-8") as fh:
        json.dump({"materials": mats_out}, fh, ensure_ascii=False, indent=1)
    return Path(matid_path).exists()


def render_detail_pass(context, filepath: str, collection, keep_path: str = "", massing=None,
                       matid_path: str = "") -> dict:
    """Standalone 细节回贴 detail pass (when no clay base was rendered): scene camera, output size."""
    scene = context.scene
    require_scene_camera(context)
    width, height = camera_output_dimensions(context)
    render = scene.render
    snap = {"x": render.resolution_x, "y": render.resolution_y, "pct": render.resolution_percentage,
            "fmt": _snapshot_image_format(render.image_settings)}
    frame_options = {k: getattr(render, k) for k in ("use_border", "use_crop_to_border", "use_compositing", "use_sequencer") if hasattr(render, k)}
    try:
        for k in frame_options:
            setattr(render, k, False)
        render.resolution_x, render.resolution_y, render.resolution_percentage = width, height, 100
        from . import massing as MS
        holdout = MS.massing_objects(massing, collection) if massing is not None else []
        info = _render_detail_now(context, filepath, collection, holdout, keep_path, matid_path)
        info["detail"] = bool(info.get("detail")) and image_dimensions(filepath) == (width, height)
        return info
    finally:
        render.resolution_x, render.resolution_y, render.resolution_percentage = snap["x"], snap["y"], snap["pct"]
        _restore_image_format(render.image_settings, snap["fmt"])
        for k, v in frame_options.items():
            setattr(render, k, v)


def render_product_mask(context, filepath: str, collection, massing=None) -> bool:
    """Product silhouette from the scene camera at the camera output size (no clay base)."""
    scene = context.scene
    require_scene_camera(context)
    width, height = camera_output_dimensions(context)
    render = scene.render
    snap = {"x": render.resolution_x, "y": render.resolution_y, "pct": render.resolution_percentage,
            "fmt": _snapshot_image_format(render.image_settings)}
    frame_options = {k: getattr(render, k) for k in ("use_border", "use_crop_to_border", "use_compositing", "use_sequencer") if hasattr(render, k)}
    try:
        for k in frame_options:
            setattr(render, k, False)
        render.resolution_x, render.resolution_y, render.resolution_percentage = width, height, 100
        from . import massing as MS
        holdout = MS.massing_objects(massing, collection) if massing is not None else []
        return _render_mask_now(context, filepath, collection, holdout) and image_dimensions(filepath) == (width, height)
    finally:
        render.resolution_x, render.resolution_y, render.resolution_percentage = snap["x"], snap["y"], snap["pct"]
        _restore_image_format(render.image_settings, snap["fmt"])
        for k, v in frame_options.items():
            setattr(render, k, v)


def render_clay_base(context, filepath: str, collection=None, ground="AUTO", crest_angle=10.0,
                     terrain=None, mask_path: str = "", massing=None, detail_path: str = "",
                     keep_path: str = "", matid_path: str = "") -> bool:
    """Render the Scene Camera with a temporary floor, sun and light-gray world.

    This is the edit canvas for the simple direct-edit path: a lit clay render
    with ground contact gives the image model a photographic starting point, so
    it repaints materials and environment instead of re-composing the shot.
    Everything temporary is removed afterwards; the user's scene is unchanged.

    ``massing`` (1.2.0 环境体块): a collection of context blocks.  They render with a temporary
    matte grey-blue material (their own materials come back afterwards), cast shadows, rest on a
    sculpted floor, and never count as product; ``LAST_CLAY_INFO["massing_text"]`` describes
    where each one sits in the frame for the prompt.
    """
    scene = context.scene
    require_scene_camera(context)
    width, height = camera_output_dimensions(context)
    render = scene.render
    snap = {
        "engine": render.engine, "filepath": render.filepath, "x": render.resolution_x, "y": render.resolution_y,
        "pct": render.resolution_percentage, "fmt": _snapshot_image_format(render.image_settings),
        "world": scene.world, "film": getattr(render, "film_transparent", False),
    }
    frame_options = {k: getattr(render, k) for k in ("use_border", "use_crop_to_border", "use_compositing", "use_sequencer") if hasattr(render, k)}
    created = []
    restore = []  # undo steps for the temporary product pose (自定义地形), run before cleanup
    from . import massing as MS
    blocks = MS.massing_objects(massing, collection) if massing is not None else []
    _MASSING_EXCLUDE.clear()
    _MASSING_EXCLUDE.update(blocks)
    try:
        from mathutils import Vector
        ref_ground = _product_floor_z(collection)  # the level the user modelled on, before any pose
        pts = _product_bounds(collection)
        ground_mode = (ground or "AUTO").upper()
        ground = _user_ground(context, collection, pts)
        up = _product_up_axis(collection)
        slope = math.degrees(math.acos(max(-1.0, min(1.0, up.z))))
        LAST_CLAY_INFO.clear()
        LAST_CLAY_INFO.update({"user_ground": ground.name if ground else "", "slope_deg": round(slope, 1)})
        mesh = bpy.data.meshes.new("_wondful_clay_floor")
        s = 500.0
        mesh.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
        floor = bpy.data.objects.new("_wondful_clay_floor", mesh)
        if pts:
            # Floor plane perpendicular to the product's up axis, touching its lowest point
            # (a car tilted onto a slope gets a matching sloped floor).
            low = min(pts, key=lambda p: p.dot(up))
            floor.rotation_mode = "QUATERNION"
            floor.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(up)
            floor.location = low
        else:
            floor.location.z = _product_floor_z(collection)
        if ground is not None:
            floor.hide_render = True  # the user's own terrain (ramp, curb, road) is the ground
        elif ground_mode == "CREST" and pts:
            up = Vector((0.0, 0.0, 1.0))  # a hilltop is level: the product sits flat on the very top
            fwd, length = _product_forward(pts, up)
            low = min(pts, key=lambda p: p.dot(up))
            centre = sum(pts, Vector()) / len(pts)
            centre = centre - up * (centre.dot(up) - low.dot(up))
            side_ax = fwd.cross(up).normalized()
            sproj = [p.dot(side_ax) for p in pts]
            prod_width = max(sproj) - min(sproj)
            fproj = [p.dot(fwd) for p in pts]
            centre = centre + fwd * ((max(fproj) + min(fproj)) / 2.0 - centre.dot(fwd)) \
                + side_ax * ((max(sproj) + min(sproj)) / 2.0 - centre.dot(side_ax))
            crest = _crest_mesh("_wondful_clay_floor", centre, up, fwd, length * 0.5 + 0.3, prod_width * 0.5 + 0.3,
                                crest_angle)
            old_mesh = floor.data
            floor.data = crest
            floor.rotation_mode = "QUATERNION"
            floor.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            floor.location = (0.0, 0.0, 0.0)
            bpy.data.meshes.remove(old_mesh)
            mesh = crest
            LAST_CLAY_INFO["crest_deg"] = round(float(crest_angle), 1)
        elif ground_mode == "CUSTOM" and pts:
            terrain_mesh, tinfo = _custom_terrain(context, collection, terrain, created, restore)
            LAST_CLAY_INFO["terrain"] = tinfo
            LAST_CLAY_INFO["terrain_text"] = tinfo.get("text", "")
            if terrain_mesh is not None:
                old_mesh = floor.data
                floor.data = terrain_mesh
                floor.rotation_mode = "QUATERNION"
                floor.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
                floor.location = (0.0, 0.0, 0.0)
                bpy.data.meshes.remove(old_mesh)
                mesh = terrain_mesh
        mat = bpy.data.materials.new("_wondful_clay_floor_mat")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.2, 0.2, 0.21, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.8
        mesh.materials.append(mat)
        scene.collection.objects.link(floor)
        created += [("obj", floor), ("mesh", mesh), ("mat", mat)]
        if blocks:
            massing_mat = MS.make_material(bpy.data.materials.new)
            created.append(("mat", massing_mat))
            if ground_mode in {"CREST", "CUSTOM"} and ground is None:
                LAST_CLAY_INFO["massing_snapped"] = MS.snap_to_ground(context, blocks, floor, ref_ground, restore)
            MS.swap_materials(blocks, massing_mat, restore)
            context.view_layer.update()
            items = MS.describe_blocks(context, blocks, _product_bounds(collection))
            from .product_kind import names_look_like_vehicle, collection_names
            veh = names_look_like_vehicle(collection_names(collection)) if collection is not None else False
            LAST_CLAY_INFO["massing"] = items
            LAST_CLAY_INFO["massing_text"] = MS.build_massing_text(items, vehicle=veh)
        sun_data = bpy.data.lights.new("_wondful_clay_sun", type="SUN")
        sun_data.energy = 4.0
        try:
            sun_data.angle = 0.12
        except Exception:
            pass
        sun = bpy.data.objects.new("_wondful_clay_sun", sun_data)
        sun.rotation_euler = (0.85, 0.0, 0.75)
        scene.collection.objects.link(sun)
        created += [("obj", sun), ("light", sun_data)]
        if pts:  # two long softbox strips: crisp highlight bands across the clear coat
            from mathutils import Vector as _V
            c = sum(pts, _V()) / len(pts)
            span = max((p - c).length for p in pts)
            for i, (off, rot, energy) in enumerate((((0.0, 0.0, 2.2), (0.0, 0.0, 0.6), 900.0),
                                                     ((-1.6, 1.4, 0.9), (1.2, 0.0, -0.9), 500.0))):
                ld = bpy.data.lights.new(f"_wondful_clay_strip{i}", type="AREA")
                ld.shape = "RECTANGLE"
                ld.size, ld.size_y = span * 2.2, span * 0.18
                ld.energy = energy * (span / 2.5) ** 2
                lo = bpy.data.objects.new(f"_wondful_clay_strip{i}", ld)
                lo.location = c + _V(off) * span
                lo.rotation_euler = rot
                scene.collection.objects.link(lo)
                created += [("obj", lo), ("light", ld)]
        world = bpy.data.worlds.new("_wondful_clay_world")
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[1].default_value = 1.0
            _clay_sky_gradient(world.node_tree, bg)
        scene.world = world
        created.append(("world", world))
        for k in frame_options:
            setattr(render, k, False)
        vs = scene.view_settings
        snap["cm"] = (vs.view_transform, vs.look, vs.exposure)
        try:
            vs.view_transform = "AgX"
            vs.look = "AgX - Medium High Contrast"
        except Exception:
            try:
                vs.view_transform = "Filmic"
                vs.look = "Medium High Contrast"
            except Exception:
                pass
        if hasattr(render, "film_transparent"):
            render.film_transparent = False
        engine = os.environ.get("WONDFUL_CLAY_ENGINE", "").strip().upper()
        if not engine:
            engine = render.engine if render.engine in {"CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"} else "BLENDER_EEVEE_NEXT"
        for candidate in (engine, "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
            try:
                render.engine = candidate
                break
            except Exception:
                continue
        if render.engine == "CYCLES":
            try:
                scene.cycles.samples = int(os.environ.get("WONDFUL_CLAY_SAMPLES", "48"))
                scene.cycles.use_denoising = os.environ.get("WONDFUL_CLAY_DENOISE", "1") != "0"
            except Exception:
                pass
        render.filepath = filepath
        render.resolution_x, render.resolution_y, render.resolution_percentage = width, height, 100
        _set_png(render.image_settings)
        try:
            render.image_settings.color_mode = "RGB"
        except Exception:
            pass
        _render_still(scene)
        ok = Path(filepath).exists() and image_dimensions(filepath) == (width, height)
        if ok and mask_path and collection is not None:
            # 1.2.0 分层输出: same camera, same temporary product pose as the clay base.
            try:
                LAST_CLAY_INFO["mask"] = bool(_render_mask_now(context, mask_path, collection, blocks)
                                              and image_dimensions(mask_path) == (width, height))
            except Exception as exc:
                LAST_CLAY_INFO["mask"] = False
                print(f"[Wondful] 产品遮罩渲染失败：{exc}")
        if ok and detail_path and collection is not None:
            # 1.2.0 细节回贴: same camera and temporary pose; clay lamps / floor are hidden for it.
            try:
                import time as _time
                t_detail = _time.time()
                for kind, block in created:
                    if kind == "obj" and not block.hide_render:
                        block.hide_render = True
                info = _render_detail_now(context, detail_path, collection, blocks, keep_path, matid_path)
                info["detail"] = bool(info.get("detail")) and image_dimensions(detail_path) == (width, height)
                info["seconds"] = round(_time.time() - t_detail, 1)
                LAST_CLAY_INFO["detail"] = info
            except Exception as exc:
                LAST_CLAY_INFO["detail"] = {"detail": False, "error": str(exc)[:200]}
                print(f"[Wondful] 细节底图渲染失败：{exc}")
        return ok
    finally:
        _MASSING_EXCLUDE.clear()
        for undo in reversed(restore):
            try:
                undo()
            except Exception as exc:
                print(f"[Wondful] 恢复产品姿态 / 体块材质失败：{exc}")
        if restore:
            try:
                context.view_layer.update()
            except Exception:
                pass
        if "cm" in snap:
            try:
                scene.view_settings.view_transform, scene.view_settings.look, scene.view_settings.exposure = snap["cm"]
            except Exception:
                pass
        render.engine = snap["engine"]
        render.filepath = snap["filepath"]
        render.resolution_x, render.resolution_y, render.resolution_percentage = snap["x"], snap["y"], snap["pct"]
        _restore_image_format(render.image_settings, snap["fmt"])
        if hasattr(render, "film_transparent"):
            render.film_transparent = snap["film"]
        for k, v in frame_options.items():
            setattr(render, k, v)
        scene.world = snap["world"]
        for kind, block in created:
            try:
                coll = {"obj": bpy.data.objects, "mesh": bpy.data.meshes, "mat": bpy.data.materials,
                        "light": bpy.data.lights, "world": bpy.data.worlds}[kind]
                coll.remove(block)
            except Exception:
                pass
