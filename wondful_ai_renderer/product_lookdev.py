"""Lit product reference and placed-product canvas for the single-source edit.

The 2026-11 review showed two failures of the source-only path:

* the product reference was a flat, uniformly lit RGBA pass, so the image model received no
  paint response at all (no sky / ground reflections, no horizon line, no clear-coat specular)
  and painted a flat solid green; panel gaps, handles and lamp internals had almost no shading
  contrast and were smoothed away;
* the source photograph was the edit canvas, so the model kept the *old* vehicle's position and
  size instead of the projection's.

This module renders a *look-development* pass of the product from the scene camera — the
product's own materials under a sky/ground gradient sampled from the source photograph, a
temporary clear coat on painted body materials, a shadow catcher for ground contact, AgX — and
composites it over the source photograph at the exact camera position. The composite becomes the
edit canvas, so geometry is pinned by pixels instead of by a second reference image, and the
model re-renders an already placed car rather than inventing one.

The lit pass is a structure / paint-response reference only: the final illumination still comes
from the photograph (the prompt says so). Every temporary change to the user's scene is undone.

The colour / compositing helpers are pure numpy (worker-thread and plain-Python test safe);
``render_lookdev`` needs ``bpy`` and the main thread.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np

LOOKDEV_VERSION = 2
# Temporary clear coat for painted body materials (only when the material has none of its own).
COAT_WEIGHT = 1.0
COAT_ROUGHNESS = 0.10      # 0.03 made mirror-hard sun glints the image model redrew as lamps
HIGHLIGHT_CLAMP = 2.0      # Cycles direct/indirect sample clamp: no hot specular patches in the reference
MAX_SHADOW = 0.72          # shadow-catcher opacity cap in the composite
SKY_ROWS = 0.22            # top fraction of the photo sampled as "sky"
GROUND_ROWS = 0.22         # bottom fraction sampled as "ground"

_PAINT = re.compile(r"paint|body|车漆|车身|漆", re.IGNORECASE)
_NOT_PAINT = re.compile(r"interior|内饰|seat|座椅|glass|玻璃|window|tyre|tire|轮胎|rubber|橡胶|chrome|镀铬|"
                        r"light|lamp|灯|caliper|卡钳|brake|刹车|plate|车牌|logo|badge|标|rim|轮毂|wheel",
                        re.IGNORECASE)


# --------------------------------------------------------------------------- pure numpy

def srgb_to_linear(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def environment_colours(source_rgb: np.ndarray) -> dict:
    """Robust sky / horizon / ground colours (scene-linear RGB) from a photograph.

    Medians of the top and bottom bands, so a bright sun disc or a dark tyre does not dominate.
    ``sun_x`` is the horizontal position (0 = left, 1 = right) of the brightest part of the upper
    half, ``sun_in_frame`` whether it is clearly inside the frame (backlight)."""
    a = np.asarray(source_rgb)
    if a.ndim != 3 or a.shape[2] < 3 or a.shape[0] < 8 or a.shape[1] < 8:
        raise ValueError("来源图无法采样环境颜色。")
    img = a[..., :3].astype(np.float32) / (255.0 if a.dtype == np.uint8 else 1.0)
    h, w = img.shape[:2]
    top = img[: max(1, int(h * SKY_ROWS))].reshape(-1, 3)
    bottom = img[h - max(1, int(h * GROUND_ROWS)):].reshape(-1, 3)
    mid = img[int(h * 0.40): int(h * 0.60)].reshape(-1, 3)
    sky = srgb_to_linear(np.median(top, axis=0))
    ground = srgb_to_linear(np.median(bottom, axis=0))
    horizon = srgb_to_linear(np.percentile(mid, 75, axis=0))
    upper = img[: max(2, h // 2)]
    lum = upper @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    cols = lum.mean(axis=0)
    k = max(1, w // 32)
    smooth = np.convolve(cols, np.ones(k, np.float32) / k, mode="same")
    peak = int(np.argmax(smooth))
    contrast = float(smooth.max() - np.median(smooth))
    return {
        "sky": [float(x) for x in np.clip(sky, 0.0, 1.0)],
        "horizon": [float(x) for x in np.clip(horizon, 0.0, 1.0)],
        "ground": [float(x) for x in np.clip(ground, 0.0, 1.0)],
        "sun_x": round(peak / max(1, w - 1), 3),
        "sun_in_frame": bool(contrast > 0.06 and 0.08 < peak / max(1, w - 1) < 0.92),
        "sun_contrast": round(contrast, 4),
    }


def _resize_rgba(rgba: np.ndarray, w: int, h: int) -> np.ndarray:
    """Premultiplied resize so transparent edges do not bleed dark fringes."""
    from .layered_output import resize
    a = rgba.astype(np.float32) / 255.0
    alpha = a[..., 3:4]
    pre = np.concatenate([a[..., :3] * alpha, alpha], axis=2)
    out = np.clip(resize(pre, w, h), 0.0, 1.0)
    al = out[..., 3:4]
    rgb = np.where(al > 1e-4, out[..., :3] / np.maximum(al, 1e-4), 0.0)
    return np.concatenate([np.clip(rgb, 0, 1), al], axis=2)


def composite(source_rgb: np.ndarray, lookdev_rgba: np.ndarray, product_alpha: np.ndarray | None = None,
              max_shadow: float = MAX_SHADOW) -> np.ndarray:
    """Place the lit product over the photograph (straight alpha in, uint8 RGB out).

    ``lookdev_rgba`` carries the product *and* the shadow catcher's darkening in its alpha. Where
    ``product_alpha`` (the flat pass coverage) is given, pixels outside the product are treated as
    shadow only and capped at ``max_shadow`` so the composite never paints an opaque floor."""
    src = np.asarray(source_rgb)[..., :3].astype(np.float32) / 255.0
    h, w = src.shape[:2]
    look = _resize_rgba(np.asarray(lookdev_rgba), w, h) if np.asarray(lookdev_rgba).shape[:2] != (h, w) else \
        np.asarray(lookdev_rgba).astype(np.float32) / 255.0
    rgb, alpha = look[..., :3], look[..., 3:4]
    if product_alpha is not None:
        pa = np.asarray(product_alpha, np.float32)
        if pa.max() > 1.0:
            pa = pa / 255.0
        if pa.shape[:2] != (h, w):
            from .layered_output import resize
            pa = np.clip(resize(pa[..., None], w, h)[..., 0], 0, 1)
        shadow_only = alpha[..., 0] * (1.0 - pa)
        alpha = np.maximum(pa, np.minimum(shadow_only, max_shadow))[..., None]
        # inside the shadow region the catcher writes (near) black: keep it, but never brighter
        # than the lit product where both overlap
    out = src * (1.0 - alpha) + rgb * alpha
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def build_canvas(source_path, lookdev_path, projection_path, output_path) -> dict:
    """File-level ``composite``: photograph + lit product → RGB PNG at the photograph's size."""
    from .png_io import read_png, write_png
    from .reference_collage import read_rgb
    src = read_rgb(source_path)
    look = read_png(lookdev_path)
    flat = read_png(projection_path)
    if look.ndim != 3 or look.shape[2] != 4 or flat.ndim != 3 or flat.shape[2] != 4:
        raise ValueError("产品光照参考或投影缺少 RGBA 通道。")
    if look.shape[:2] != flat.shape[:2]:
        raise ValueError("产品光照参考与投影尺寸不一致。")
    out = composite(src, look, flat[..., 3])
    write_png(output_path, out)
    return {"path": str(output_path), "size": [int(out.shape[1]), int(out.shape[0])]}


WORLD_TARGET = 0.6           # brightest band of the gradient after normalisation (scene-linear)


def world_strength(env: dict) -> float:
    """Exposure of the gradient world: with no sun the environment is the only light, so a dusk or
    night photo (sky median ~0.05) would leave the product black. Normalise the brightest band
    to ``WORLD_TARGET`` and keep the colour ratios; clamp so a white sky is not dimmed to grey."""
    peak = max(max(env.get("sky", [0])), max(env.get("horizon", [0])), 1e-4)
    return float(min(20.0, max(1.0, WORLD_TARGET / peak)))


def is_paint_name(name: str) -> bool:
    return bool(_PAINT.search(name or "")) and not _NOT_PAINT.search(name or "")


def paint_materials(rows) -> list[str]:
    """Names of body-paint materials among ``appearance_lock.collect_product_materials`` rows.

    Name convention first; without a paint name, the heaviest coloured, opaque, non-metal-chrome
    material (the body shell is almost always the largest surface)."""
    named = [r["name"] for r in rows if is_paint_name(r.get("name", ""))]
    if named:
        return named
    best = None
    for r in rows:
        name = r.get("name", "")
        if _NOT_PAINT.search(name or ""):
            continue
        if float(r.get("transmission", 0.0) or 0.0) > 0.3:
            continue
        rgb = [float(x) for x in (r.get("base_color") or [0, 0, 0])[:3]]
        if max(rgb) - min(rgb) < 0.04 and max(rgb) < 0.05:   # near-black trim
            continue
        if best is None or r.get("weight", 0) > best.get("weight", 0):
            best = r
    return [best["name"]] if best else []


def sun_rotation(camera_matrix, sun_x: float, backlit: bool, elevation_deg: float = 28.0):
    """Euler (XYZ) for a sun lamp roughly matching the photo: azimuth from the bright side of the
    frame relative to the camera's view direction; behind the subject when ``backlit``."""
    fwd = [-camera_matrix[0][2], -camera_matrix[1][2]]   # camera looks down its -Z
    yaw_cam = math.atan2(fwd[1], fwd[0])
    side = (float(sun_x) - 0.5) * 2.0                 # -1 left … +1 right of frame
    # light *travels* from the sun towards the scene; a sun in front of the camera (backlight)
    # shines towards the camera
    if backlit:
        travel = yaw_cam + math.pi - side * math.radians(35)
    else:
        travel = yaw_cam - side * math.radians(70) + math.radians(20)
    el = math.radians(elevation_deg)
    # Blender sun shines along its local -Z; rotate so -Z = (cos el*cos t, cos el*sin t, -sin el)
    return (math.pi / 2 - el, 0.0, travel + math.pi / 2)


# --------------------------------------------------------------------------- bpy

def _principled(material):
    tree = getattr(material, "node_tree", None) if getattr(material, "use_nodes", False) else None
    if tree is None:
        return []
    return [n for n in tree.nodes if getattr(n, "type", "") == "BSDF_PRINCIPLED"]


def _apply_coat(materials, undo):
    """Temporary clear coat on the named paint materials (library-linked ones are skipped)."""
    import bpy
    done = []
    for name in materials:
        mat = bpy.data.materials.get(name)
        if mat is None or getattr(mat, "library", None) is not None:
            continue
        for node in _principled(mat):
            weight = node.inputs.get("Coat Weight") or node.inputs.get("Clearcoat")
            rough = node.inputs.get("Coat Roughness") or node.inputs.get("Clearcoat Roughness")
            if weight is None or getattr(weight, "is_linked", False):
                continue
            if float(weight.default_value) >= 0.5:
                continue                                    # the user already modelled a coat
            old_w = weight.default_value
            undo.append(lambda s=weight, v=old_w: setattr(s, "default_value", v))
            weight.default_value = COAT_WEIGHT
            if rough is not None and not getattr(rough, "is_linked", False):
                old_r = rough.default_value
                undo.append(lambda s=rough, v=old_r: setattr(s, "default_value", v))
                rough.default_value = COAT_ROUGHNESS
            done.append(mat.name)
    return sorted(set(done))


def _gradient_world(env, created):
    """Sky above, bright horizon band, ground below — reflections read as a real outdoor scene."""
    import bpy
    world = bpy.data.worlds.new("_wondful_lookdev_world")
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    mapr = nt.nodes.new("ShaderNodeMapRange")
    # Generated world coordinates are the view direction; Z in [-1, 1] → [0, 1] on the ramp
    nt.links.new(coord.outputs["Generated"], sep.inputs[0])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    nt.links.new(sep.outputs["Z"], mapr.inputs[0])
    mapr.inputs[1].default_value, mapr.inputs[2].default_value = -1.0, 1.0
    mapr.inputs[3].default_value, mapr.inputs[4].default_value = 0.0, 1.0
    nt.links.new(mapr.outputs[0], ramp.inputs["Fac"])
    sky, hor, gnd = env["sky"], env["horizon"], env["ground"]
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (*[g * 0.55 for g in gnd], 1.0)
    els[1].position, els[1].color = 1.0, (*[s * 0.85 for s in sky], 1.0)
    for pos, col in ((0.47, [g * 0.9 for g in gnd]), (0.499, gnd), (0.502, [max(h, s) for h, s in zip(hor, sky)]),
                     (0.56, [(h + s) / 2 for h, s in zip(hor, sky)])):
        e = els.new(pos)
        e.color = (*[float(c) for c in col], 1.0)
    bg.inputs["Strength"].default_value = float(os.environ.get("WONDFUL_LOOKDEV_WORLD", "0")) or world_strength(env)
    created.append(("world", world))
    return world


def render_lookdev(context, filepath: str, collection, env: dict, paint_rows=None) -> dict:
    """Render the product's own materials under ``env`` with ground contact, RGBA, at the camera
    output size. Restores the scene. Returns {"ok", "coated", "engine", "sun"}."""
    import bpy
    from mathutils import Vector
    from . import viewport_capture as VC
    scene = context.scene
    VC.require_scene_camera(context)
    width, height = VC.camera_output_dimensions(context)
    render = scene.render
    product = set(collection.all_objects) if collection is not None else set()
    info = {"ok": False, "coated": [], "engine": "", "sun": None}
    if not product:
        return info
    vs = scene.view_settings
    cycles = getattr(scene, "cycles", None)
    snap = {"engine": render.engine, "path": render.filepath, "x": render.resolution_x, "y": render.resolution_y,
            "pct": render.resolution_percentage, "fmt": VC._snapshot_image_format(render.image_settings),
            "mode": render.image_settings.color_mode, "world": scene.world,
            "film": getattr(render, "film_transparent", False),
            "cm": (vs.view_transform, vs.look, vs.exposure, vs.gamma),
            "cycles": (cycles.samples, cycles.use_denoising, cycles.max_bounces) if cycles is not None else None,
            "clamp": (getattr(cycles, "sample_clamp_direct", 0.0), getattr(cycles, "sample_clamp_indirect", 0.0)) if cycles is not None else None,
            "override": context.view_layer.material_override}
    frame_options = {k: getattr(render, k) for k in ("use_border", "use_crop_to_border", "use_compositing",
                                                     "use_sequencer") if hasattr(render, k)}
    hidden, held, created, undo = [], [], [], []
    try:
        for k in frame_options:
            setattr(render, k, False)
        VC._isolate_product(scene, product, (), hidden, held, hide_lights=True)
        context.view_layer.material_override = None
        info["coated"] = _apply_coat(paint_materials(paint_rows or []), undo)
        _gradient_world(env, created)
        scene.world = created[-1][1]
        # ground contact: shadow catcher plane under the product, perpendicular to its up axis
        pts = VC._product_bounds(collection)
        up = VC._product_up_axis(collection)
        mesh = bpy.data.meshes.new("_wondful_lookdev_floor")
        s = 200.0
        mesh.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
        floor = bpy.data.objects.new("_wondful_lookdev_floor", mesh)
        if pts:
            low = min(pts, key=lambda p: p.dot(up))
            floor.rotation_mode = "QUATERNION"
            floor.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(up)
            floor.location = low
        floor.is_shadow_catcher = True
        scene.collection.objects.link(floor)
        created += [("obj", floor), ("mesh", mesh)]
        # v2: no guessed sun by default. A sun placed from the frame's bright side put a hard
        # glint into a concave panel (the YU7 front vent) and the model turned it into a light
        # bar. The key light comes from the photograph itself; the lit pass only shows form.
        if os.environ.get("WONDFUL_LOOKDEV_SUN", "0") == "1":
            sun_data = bpy.data.lights.new("_wondful_lookdev_sun", type="SUN")
            sun_data.energy = float(os.environ.get("WONDFUL_LOOKDEV_SUN_ENERGY", "2.0"))
            sun_data.angle = math.radians(4.0)
            sun = bpy.data.objects.new("_wondful_lookdev_sun", sun_data)
            sun.rotation_euler = sun_rotation(scene.camera.matrix_world, env.get("sun_x", 0.35),
                                              bool(env.get("sun_in_frame")))
            scene.collection.objects.link(sun)
            created += [("obj", sun), ("light", sun_data)]
            info["sun"] = {"x": env.get("sun_x"), "backlit": bool(env.get("sun_in_frame"))}
        render.engine = "CYCLES"
        info["engine"] = "CYCLES"
        if cycles is not None:
            cycles.samples = int(os.environ.get("WONDFUL_LOOKDEV_SAMPLES", "96"))
            cycles.use_denoising = os.environ.get("WONDFUL_LOOKDEV_DENOISE", "1") != "0"
            cycles.max_bounces = max(6, int(snap["cycles"][2] if snap["cycles"] else 8))
            if hasattr(cycles, "sample_clamp_direct"):
                cycles.sample_clamp_direct = HIGHLIGHT_CLAMP
                cycles.sample_clamp_indirect = HIGHLIGHT_CLAMP
        render.film_transparent = True
        VC._set_png(render.image_settings)
        render.image_settings.color_mode = "RGBA"
        try:
            vs.view_transform, vs.look, vs.exposure, vs.gamma = "AgX", "AgX - Medium High Contrast", 0.0, 1.0
        except Exception:
            try:
                vs.view_transform, vs.look, vs.exposure, vs.gamma = "Filmic", "Medium High Contrast", 0.0, 1.0
            except Exception:
                pass
        render.resolution_x, render.resolution_y, render.resolution_percentage = width, height, 100
        render.filepath = filepath
        VC._render_still(scene)
        info["ok"] = Path(filepath).is_file() and VC.image_dimensions(filepath) == (width, height)
        return info
    finally:
        for fn in reversed(undo):
            try:
                fn()
            except Exception:
                pass
        VC._unisolate(hidden, held)
        try:
            context.view_layer.material_override = snap["override"]
        except Exception:
            pass
        scene.world = snap["world"]
        for kind, block in reversed(created):
            try:
                {"obj": bpy.data.objects, "mesh": bpy.data.meshes, "light": bpy.data.lights,
                 "world": bpy.data.worlds}[kind].remove(block)
            except Exception:
                pass
        render.engine = snap["engine"]
        render.filepath = snap["path"]
        render.resolution_x, render.resolution_y, render.resolution_percentage = snap["x"], snap["y"], snap["pct"]
        VC._restore_image_format(render.image_settings, snap["fmt"])
        try:
            render.image_settings.color_mode = snap["mode"]
        except Exception:
            pass
        if hasattr(render, "film_transparent"):
            render.film_transparent = snap["film"]
        try:
            vs.view_transform, vs.look, vs.exposure, vs.gamma = snap["cm"]
        except Exception:
            pass
        if cycles is not None and snap["cycles"]:
            cycles.samples, cycles.use_denoising, cycles.max_bounces = snap["cycles"]
        if cycles is not None and snap.get("clamp") and hasattr(cycles, "sample_clamp_direct"):
            cycles.sample_clamp_direct, cycles.sample_clamp_indirect = snap["clamp"]
        for k, v in frame_options.items():
            setattr(render, k, v)
