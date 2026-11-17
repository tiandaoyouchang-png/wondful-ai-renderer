"""Reference-photo projection for 「参考图建模重渲染」 (mode 2, 2026-11).

Mode 2 renders the user's car from the user's own camera in the reference photograph's world and
style. The environment for that new view is not invented by the image model: the reference photo
(the clean plate, so the old vehicle is gone) is projected from the reference camera onto the
scene blockout built from that photo, plus a large dome for sky and far terrain, and rendered from
the user's camera. Pixels the reference camera never saw come out transparent ("holes"); the image
model fills only those and turns the whole frame into one photograph.

Per-pixel exact projection in the shader (no UV unwrap, no modifiers): the reference camera's
clip-space matrix rows are baked into a node group as constants; u, v = clip.xy / clip.w.
Every temporary datablock is removed and every touched setting restored.
"""
from __future__ import annotations

import math
from pathlib import Path

PROJECTION_VERSION = 1
DOME_RADIUS = 3000.0


def clip_matrix(scene, camera, width: int, height: int):
    """4×4 world → clip matrix of ``camera`` for a ``width``×``height`` frame."""
    depsgraph = __import__("bpy").context.evaluated_depsgraph_get()
    proj = camera.calc_matrix_camera(depsgraph, x=int(width), y=int(height),
                                     scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y)
    return proj @ camera.matrix_world.inverted()


def _projection_material(image, M, uv_mode=False):
    import bpy
    mat = bpy.data.materials.new("_wondful_projection")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")

    def row(i):
        dot = nt.nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        nt.links.new(geo.outputs["Position"], dot.inputs[0])
        dot.inputs[1].default_value = (M[i][0], M[i][1], M[i][2])
        add = nt.nodes.new("ShaderNodeMath")
        add.operation = "ADD"
        nt.links.new(dot.outputs["Value"], add.inputs[0])
        add.inputs[1].default_value = M[i][3]
        return add.outputs[0]

    cx, cy, cw = row(0), row(1), row(3)

    def ndc_to_uv(c):
        div = nt.nodes.new("ShaderNodeMath"); div.operation = "DIVIDE"
        nt.links.new(c, div.inputs[0]); nt.links.new(cw, div.inputs[1])
        mad = nt.nodes.new("ShaderNodeMath"); mad.operation = "MULTIPLY_ADD"
        nt.links.new(div.outputs[0], mad.inputs[0]); mad.inputs[1].default_value = 0.5; mad.inputs[2].default_value = 0.5
        return mad.outputs[0]

    u, v = ndc_to_uv(cx), ndc_to_uv(cy)
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(u, comb.inputs[0]); nt.links.new(v, comb.inputs[1])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.extension = "EXTEND"
    tex.interpolation = "Linear"
    nt.links.new(comb.outputs[0], tex.inputs["Vector"])
    emit = nt.nodes.new("ShaderNodeEmission")
    if uv_mode:
        nt.links.new(comb.outputs[0], emit.inputs["Color"])
    else:
        nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    emit.inputs["Strength"].default_value = 1.0

    # valid = w > 0 and 0 ≤ u ≤ 1 and 0 ≤ v ≤ 1
    def inside(val):
        a = nt.nodes.new("ShaderNodeMath"); a.operation = "GREATER_THAN"
        nt.links.new(val, a.inputs[0]); a.inputs[1].default_value = 0.0
        b = nt.nodes.new("ShaderNodeMath"); b.operation = "LESS_THAN"
        nt.links.new(val, b.inputs[0]); b.inputs[1].default_value = 1.0
        m = nt.nodes.new("ShaderNodeMath"); m.operation = "MULTIPLY"
        nt.links.new(a.outputs[0], m.inputs[0]); nt.links.new(b.outputs[0], m.inputs[1])
        return m.outputs[0]

    wpos = nt.nodes.new("ShaderNodeMath"); wpos.operation = "GREATER_THAN"
    nt.links.new(cw, wpos.inputs[0]); wpos.inputs[1].default_value = 1e-6
    m1 = nt.nodes.new("ShaderNodeMath"); m1.operation = "MULTIPLY"
    nt.links.new(inside(u), m1.inputs[0]); nt.links.new(inside(v), m1.inputs[1])
    m2 = nt.nodes.new("ShaderNodeMath"); m2.operation = "MULTIPLY"
    nt.links.new(m1.outputs[0], m2.inputs[0]); nt.links.new(wpos.outputs[0], m2.inputs[1])
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(m2.outputs[0], mix.inputs["Fac"])
    nt.links.new(transp.outputs[0], mix.inputs[1])
    nt.links.new(emit.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def render_projection(context, filepath: str, environment, reference_camera, plate_path: str,
                      plate_size, product_collection=None, uv_filepath: str = "") -> dict:
    """Render ``environment`` (a collection) textured by ``plate_path`` projected from
    ``reference_camera`` through the scene camera; RGBA, holes transparent. Restores the scene."""
    import bpy
    from mathutils import Vector
    from . import viewport_capture as VC
    scene = context.scene
    VC.require_scene_camera(context)
    width, height = VC.camera_output_dimensions(context)
    render = scene.render
    vs = scene.view_settings
    cycles = getattr(scene, "cycles", None)
    M = clip_matrix(scene, reference_camera, *plate_size)
    snap = {"engine": render.engine, "path": render.filepath, "x": render.resolution_x, "y": render.resolution_y,
            "pct": render.resolution_percentage, "fmt": VC._snapshot_image_format(render.image_settings),
            "mode": render.image_settings.color_mode, "film": render.film_transparent, "world": scene.world,
            "cm": (vs.view_transform, vs.look, vs.exposure, vs.gamma),
            "override": context.view_layer.material_override,
            "cycles": (cycles.samples, cycles.use_denoising) if cycles is not None else None}
    frame_options = {k: getattr(render, k) for k in ("use_border", "use_crop_to_border", "use_compositing",
                                                     "use_sequencer") if hasattr(render, k)}
    env_objs = set(environment.all_objects)
    hidden, held, created = [], [], []
    image = bpy.data.images.load(str(plate_path), check_existing=False)
    created.append(("image", image))
    info = {"ok": False, "version": PROJECTION_VERSION}
    try:
        for k in frame_options:
            setattr(render, k, False)
        # dome for sky and far terrain, centred on the reference camera
        mesh = bpy.data.meshes.new("_wondful_projection_dome")
        dome = bpy.data.objects.new("_wondful_projection_dome", mesh)
        import bmesh
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=DOME_RADIUS)
        bm.to_mesh(mesh); bm.free()
        dome.location = reference_camera.matrix_world.translation
        scene.collection.objects.link(dome)
        created += [("obj", dome), ("mesh", mesh)]
        keep = env_objs | {dome}
        VC._isolate_product(scene, keep, (), hidden, held, hide_lights=True)
        mat = _projection_material(image, M)
        created.append(("mat", mat))
        context.view_layer.material_override = mat
        world = bpy.data.worlds.new("_wondful_projection_world")
        world.use_nodes = True
        wnt = world.node_tree
        for n in list(wnt.nodes):
            wnt.nodes.remove(n)
        bg = wnt.nodes.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = 0.0
        wout = wnt.nodes.new("ShaderNodeOutputWorld")
        wnt.links.new(bg.outputs[0], wout.inputs["Surface"])
        created.append(("world", world))
        scene.world = world
        render.engine = "CYCLES"
        if cycles is not None:
            cycles.samples = 16
            cycles.use_denoising = False
        render.film_transparent = True
        VC._set_png(render.image_settings)
        render.image_settings.color_mode = "RGBA"
        vs.view_transform, vs.look, vs.exposure, vs.gamma = "Standard", "None", 0.0, 1.0
        render.resolution_x, render.resolution_y, render.resolution_percentage = width, height, 100
        render.filepath = filepath
        VC._render_still(scene)
        info["ok"] = Path(filepath).is_file()
        if uv_filepath:
            uvmat = _projection_material(image, M, uv_mode=True)
            created.append(("mat", uvmat))
            context.view_layer.material_override = uvmat
            vs.view_transform = "Raw" if "Raw" in [i.identifier for i in vs.bl_rna.properties["view_transform"].enum_items] else "Standard"
            render.image_settings.file_format = "OPEN_EXR"
            render.image_settings.color_depth = "32"
            render.image_settings.color_mode = "RGBA"
            render.filepath = uv_filepath
            VC._render_still(scene)
            info["uv_ok"] = Path(uv_filepath).is_file()
        return info
    finally:
        VC._unisolate(hidden, held)
        try:
            context.view_layer.material_override = snap["override"]
        except Exception:
            pass
        scene.world = snap["world"]
        for kind, block in reversed(created):
            try:
                {"obj": bpy.data.objects, "mesh": bpy.data.meshes, "mat": bpy.data.materials,
                 "world": bpy.data.worlds, "image": bpy.data.images}[kind].remove(block, do_unlink=True)
            except Exception:
                pass
        render.engine = snap["engine"]; render.filepath = snap["path"]
        render.resolution_x, render.resolution_y, render.resolution_percentage = snap["x"], snap["y"], snap["pct"]
        VC._restore_image_format(render.image_settings, snap["fmt"])
        try:
            render.image_settings.color_mode = snap["mode"]
        except Exception:
            pass
        render.film_transparent = snap["film"]
        try:
            vs.view_transform, vs.look, vs.exposure, vs.gamma = snap["cm"]
        except Exception:
            pass
        if cycles is not None and snap["cycles"]:
            cycles.samples, cycles.use_denoising = snap["cycles"]
        for k, v in frame_options.items():
            setattr(render, k, v)


def coverage(rgba) -> float:
    """Share of the frame the reference photo covers (alpha > 0.5)."""
    import numpy as np
    a = np.asarray(rgba)
    return float((a[..., 3] > 127).mean()) if a.ndim == 3 and a.shape[2] == 4 else 1.0


MAX_STRETCH = 2.0  # one reference pixel may cover at most this many target pixels (linear)


def confidence(uv_rgba, plate_size, max_stretch: float = MAX_STRETCH):
    """Per-pixel 0..1 trust in the projected photo from the UV pass (u, v, -, alpha).

    Holes are 0. Where the reference is magnified more than ``max_stretch`` (grazing angles,
    ground right under the new camera) trust falls off, so those areas are regenerated too.
    """
    import numpy as np
    a = np.asarray(uv_rgba, dtype=np.float32)
    u = a[..., 0] * float(plate_size[0])
    v = a[..., 1] * float(plate_size[1])
    valid = a[..., 3] > 0.5
    du_y, du_x = np.gradient(u)
    dv_y, dv_x = np.gradient(v)
    area = np.abs(du_x * dv_y - du_y * dv_x)  # reference px² per target px²
    linear = np.sqrt(np.maximum(area, 1e-9))  # reference px per target px
    # magnification = 1/linear; conf 1 below max_stretch, 0 at 2×max_stretch
    mag = 1.0 / linear
    conf = np.clip((2.0 * max_stretch - mag) / max_stretch, 0.0, 1.0)
    conf[~valid] = 0.0
    # a seam (huge jump in uv between neighbours) is not trustworthy either
    conf[linear > 50.0] = 0.0
    return conf


# ---------------------------------------------------------------- view plate (worker-safe, no bpy)

VIEW_PLATE_VERSION = 1
TRUST = 0.5            # below this the projected pixel is regenerated
HOLE_GROW = 6          # px the regenerated region reaches into trusted pixels (seam room)

VIEW_PLATE_PROMPT = (
    "【新机位空镜补全】图 1 是把图 2 这张实拍照片投影到新机位后得到的画面，透明遮罩区域是原照片没有拍到或被拉伸模糊的部分。"
    "只补全透明遮罩区域：按遮罩外的真实画面和图 2 的场景，自然延续地形、路面、植被、山体和天空，"
    "使整张图看起来是同一地点、同一时间、同一光线和同一相机风格下从新机位拍的一张空镜照片。"
    "保持图 2 的色调、光线方向、曝光、颗粒和景深。不要放入任何车辆、人物或新物体，不要改变遮罩外的内容。只输出一张图。"
)


def push_pull_fill(rgb: np.ndarray, weight: np.ndarray, levels: int = 9) -> np.ndarray:
    """Fill low-weight pixels from the surrounding trusted ones (pyramid push-pull).

    Gives the image model a neutral, colour-plausible start inside holes instead of black or
    stretched texture; the masked edit replaces it anyway."""
    import numpy as np
    img = rgb.astype(np.float32)
    w = np.clip(weight.astype(np.float32), 0, 1)
    pyr = []
    cur_c, cur_w = img * w[..., None], w.copy()
    for _ in range(levels):
        pyr.append((cur_c, cur_w))
        h, wd = cur_w.shape
        if h < 4 or wd < 4:
            break
        h2, w2 = h // 2, wd // 2
        c = cur_c[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, 3).sum(axis=(1, 3))
        ww = cur_w[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).sum(axis=(1, 3))
        cur_c, cur_w = c, ww
    filled = cur_c / np.maximum(cur_w, 1e-6)[..., None]
    for c, ww in reversed(pyr):
        h, wd = ww.shape
        up = np.repeat(np.repeat(filled, 2, axis=0), 2, axis=1)
        up = np.pad(up, ((0, max(0, h - up.shape[0])), (0, max(0, wd - up.shape[1])), (0, 0)), mode="edge")[:h, :wd]
        own = c / np.maximum(ww, 1e-6)[..., None]
        a = np.clip(ww, 0, 1)[..., None]
        filled = own * a + up * (1 - a)
    return filled


def make_view_plate(projection_path, conf, reference_path, work_dir, edit_fn, *, cancel_check=None) -> dict:
    """Projected reference + AI fill of the untrusted pixels → the empty photograph the user's
    camera would have taken there. Trusted pixels come back byte-for-byte from the projection.

    ``edit_fn(base_png, mask_png, reference_png, out_png)`` performs one masked edit with
    ``VIEW_PLATE_PROMPT`` (image 1 = base, image 2 = the reference photo)."""
    import numpy as np
    from .png_io import read_png, write_png
    from .reference_collage import read_rgb
    from .layered_output import dilate, blur, edit_mask_rgba
    from .source_plate import _resize_exact
    proj = read_png(projection_path)
    if proj.ndim != 3 or proj.shape[2] != 4:
        raise ValueError("新机位投影缺少 RGBA 通道。")
    h, w = proj.shape[:2]
    conf = np.asarray(conf, np.float32)
    if conf.shape != (h, w):
        raise ValueError("投影可信度图与投影尺寸不一致。")
    rgb = proj[..., :3].astype(np.float32)
    regen = dilate(conf < TRUST, HOLE_GROW)
    info = {"version": VIEW_PLATE_VERSION, "regen_frac": round(float(regen.mean()), 4),
            "trusted_frac": round(float((conf >= 0.99).mean()), 4)}
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    base = push_pull_fill(rgb, conf * (~regen))
    base_path = work / "view_plate_base.png"
    write_png(base_path, np.clip(base + 0.5, 0, 255).astype(np.uint8))
    out_path = work / "view_plate.png"
    if not regen.any():
        write_png(out_path, np.clip(rgb + 0.5, 0, 255).astype(np.uint8))
        return {**info, "path": str(out_path), "edited": False}
    mask_path = work / "view_plate_editmask.png"
    write_png(mask_path, edit_mask_rgba(regen))
    if cancel_check and cancel_check():
        raise RuntimeError("任务已取消")
    raw = work / "view_plate_raw.png"
    produced = edit_fn(str(base_path), str(mask_path), str(reference_path), str(raw)) or str(raw)
    edited = _resize_exact(read_rgb(produced), w, h).astype(np.float32)
    # trusted pixels back from the projection, soft seam into the regenerated ones
    keep = np.clip(blur((conf * (~regen)).astype(np.float32), 3), 0, 1) * (~regen)
    keep = np.maximum(keep, np.where(regen, 0.0, conf))
    out = rgb * keep[..., None] + edited * (1 - keep[..., None])
    write_png(out_path, np.clip(out + 0.5, 0, 255).astype(np.uint8))
    return {**info, "path": str(out_path), "edited": True, "raw": str(produced)}


def codex_view_plate_fn(*, model="", client_version="", long_edge=2048, timeout=600, log_dir="", cancel_check=None):
    def fn(base, mask, reference, out):
        from . import codex_direct as CD
        from .composition import image_dimensions
        w, h = image_dimensions(base)
        r = CD.edit_image(prompt=VIEW_PLATE_PROMPT, edit_base_path=base, reference_paths=[reference], output_path=out,
                          size=CD.choose_edit_size(w, h, int(long_edge)), mask_path=mask, model=model,
                          client_version=client_version, timeout=timeout, log_dir=log_dir or str(Path(out).parent),
                          tag="view_plate", cancel_check=cancel_check, exact_prompt=True)
        return (r or {}).get("source") or out
    return fn
