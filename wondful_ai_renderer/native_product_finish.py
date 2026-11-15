"""Main-thread-only native product finish; never mutates the caller's scene or data."""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np

from .layered_output import dilate, fit_mask, inpaint_pushpull, luminance, resize, warp_mask
from .png_io import read_png, write_png


def _filled(mask):
    """Fill only enclosed transparent openings; boundary-connected canvas remains background."""
    outside = ~np.asarray(mask, bool)
    seed = np.zeros_like(outside); seed[0] = outside[0]; seed[-1] |= outside[-1]; seed[:, 0] |= outside[:, 0]; seed[:, -1] |= outside[:, -1]
    seen = seed
    while True:
        grown = dilate(seen, 1) & outside
        if np.array_equal(grown, seen): return ~seen
        seen = grown


def _srgb_to_linear(x):
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    return np.where(x <= .0031308, x * 12.92, 1.055 * np.maximum(x, 0) ** (1 / 2.4) - .055)


def _removal_mask(ai, alpha):
    """Fit only an AI-footprint removal bound; native geometry is never warped."""
    native_bound = dilate(_filled(alpha), 2)
    try:
        # Fit at quarter scale: this estimates only a conservative removable
        # footprint and must not make the final native geometry expensive or warped.
        h, w = alpha.shape; sw, sh = max(1, w // 4), max(1, h // 4)
        fit = fit_mask(resize(ai[..., :3].astype(np.float32) / 255, sw, sh), resize(alpha.astype(np.float32), sw, sh), scale_range=(.88, 1.16))
        ys, xs = np.nonzero(alpha)
        scale_x, scale_y = w / sw, h / sh
        fitted = warp_mask(alpha.astype(np.float32), (fit["scale_x"], fit["scale_y"]),
                           fit["dx"] * scale_x, fit["dy"] * scale_y,
                           (float(xs.mean()), float(ys.mean()))) > .35
        fitted = dilate(fitted, 3)
        return native_bound | fitted, {"fit_used": True, "fit_search": fit["search"], "fit_chamfer_before": fit["chamfer_before"], "fit_chamfer_after": fit["chamfer_after"], "fit_pixels": int(fitted.sum())}
    except (ValueError, FloatingPointError, KeyError):
        return native_bound, {"fit_used": False, "fit_fallback": "native alpha bound", "fit_pixels": 0}


def compose_native(ai_rgba, native_rgba, shadowed=None, receiver=None, paint_mask=None):
    """Pure compositor. It never reads AI pixels into the native product interior."""
    ai = np.asarray(ai_rgba, np.uint8); native = np.asarray(native_rgba, np.uint8)
    if ai.shape != native.shape or ai.ndim != 3 or ai.shape[2] != 4:
        raise ValueError("AI/native RGBA canvas mismatch")
    alpha = native[..., 3] > 0
    if not alpha.any(): raise ValueError("native product alpha is empty")
    # The generated vehicle must not remain behind the unwarped native model.
    # Include enclosed openings (wheel spokes/windows) and only a narrow bound
    # around the actual same-camera render: this is removal, never a model fit.
    removable, fit_report = _removal_mask(ai, alpha)
    plate = ai[..., :3].copy()
    filled = np.clip(inpaint_pushpull(plate.astype(np.float32) / 255, removable) * 255 + .5, 0, 255).astype(np.uint8)
    plate[removable] = filled[removable]
    contact = np.zeros_like(alpha)
    factor = np.ones(alpha.shape, np.float32)
    if shadowed is not None or receiver is not None:
        if shadowed is None or receiver is None: raise ValueError("both receiver passes are required")
        s, r = np.asarray(shadowed, np.uint8), np.asarray(receiver, np.uint8)
        if s.shape != ai.shape or r.shape != ai.shape: raise ValueError("receiver canvas mismatch")
        ys, xs = np.nonzero(alpha); radius = max(4, int((xs.max() - xs.min() + 1) * .04))
        rows = np.indices(alpha.shape)[0]
        contact = dilate(alpha, radius) & (rows >= ys.max() - int((ys.max() - ys.min() + 1) * .22)) & ~alpha
        valid = contact & (s[..., 3] > 2) & (r[..., 3] > 2)
        ratio = np.ones(alpha.shape, np.float32)
        ratio[valid] = np.clip(luminance(s[..., :3][valid] / 255.) / np.maximum(luminance(r[..., :3][valid] / 255.), 1e-4), 0, 1)
        dark = valid & (ratio < .93)
        factor[dark] = 1 - .08 * np.clip((.93 - ratio[dark]) / .93, 0, 1)
        plate = np.rint(plate * factor[..., None]).astype(np.uint8)
    rgb = native[..., :3].astype(np.float32) / 255
    if paint_mask is not None:
        p = np.asarray(paint_mask, bool) & (native[..., 3] > 240)
        if p.any():
            target = np.median(luminance(_srgb_to_linear(ai[..., :3] / 255.))[p])
            current = np.median(luminance(_srgb_to_linear(rgb))[p])
            rgb = np.clip(_linear_to_srgb(_srgb_to_linear(rgb) * np.clip(target / max(current, 1e-6), .75, 1.25)), 0, 1)
    a = native[..., 3:4].astype(np.float32) / 255
    out = np.rint((plate / 255 * (1 - a) + rgb * a) * 255).clip(0, 255).astype(np.uint8)
    result = np.dstack([out, ai[..., 3]])
    report = {"removal_pixels": int(removable.sum()), "inpaint_pixels": int(removable.sum()), "shadow_ratio_pixels": int((factor < 1).sum()), **fit_report,
              "shadow_factor_min": float(factor.min()), "outside_domain_bitexact": bool(np.array_equal(result[..., :3][~(removable | contact)], ai[..., :3][~(removable | contact)])),
              "limitations": ["hidden background is local push-pull reconstruction", "scene lighting is approximate"]}
    return result, report


def _palette(ai):
    x = ai[..., :3].astype(np.float32) / 255; v = x.max(-1); s = v - x.min(-1); good = (.08 < v) & (v < .72) & (.04 < s) & (s < .42)
    cool = x[good & (x[..., 2] >= x[..., 0])]; warm = x[good & (x[..., 0] > x[..., 2])]
    def soft(a, fallback):
        c = a.mean(0) if len(a) else np.array(fallback, np.float32); return tuple(.55 * c.mean() + .45 * c)
    return soft(cool, (.23, .30, .36)), soft(warm, (.34, .24, .27))


def _apply_values(material, values, protection, report):
    """Apply approved values to a temporary copy, preserving protected source data."""
    # ``linked`` from the catalog means at least one node socket is linked; it
    # must not lock unrelated PBR controls (for example a normal map).  Socket
    # links are checked below.  Linked-library materials remain immutable.
    protected = bool(protection.get("keep") or protection.get("library"))
    node = next((n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None) if material.use_nodes else None
    entry = report.setdefault(protection["name"], {"applied": [], "skipped": []})
    if protected or node is None:
        entry["skipped"] = sorted(values)
        return
    for key, value in values.items():
        socket = node.inputs.get({"coat_weight":"Coat Weight", "coat_roughness":"Coat Roughness", "base_color":"Base Color", "roughness":"Roughness", "metallic":"Metallic", "transmission_weight":"Transmission Weight"}.get(key, ""))
        if socket is None or socket.is_linked or (key == "base_color" and (protection.get("paint") or protection.get("base_color_locked"))):
            entry["skipped"].append(key)
            continue
        socket.default_value = (*value, 1.0) if key == "base_color" else value
        entry["applied"].append(key)


def _paint_mask(helpers, shape):
    """Read an optional frozen material-ID helper; no product-specific colours."""
    try:
        meta = json.loads(Path(helpers["matid_json"]).read_text())
        paint = {tuple(row["rgb"]) for row in meta["materials"]
                 if "paint" in str(row.get("name", "")).lower() or "body" in str(row.get("name", "")).lower()}
        ids = read_png(helpers["matid"])
        if ids.shape[:2] != shape[:2] or not paint:
            return None
        return np.any(np.stack([np.all(ids[..., :3] == rgb, axis=2) for rgb in paint]), axis=0)
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _checked_plan(plan, protections):
    allowed = {"roughness", "metallic", "coat_weight", "coat_roughness", "transmission_weight", "base_color"}
    result = {}
    for row in plan or []:
        if not isinstance(row, dict) or set(row) != {"name", "values"} or row["name"] in result:
            raise ValueError("native material plan is invalid")
        name, values = row["name"], row["values"]
        if name not in protections or not isinstance(values, dict) or not values or set(values) - allowed:
            raise ValueError("native material plan is invalid or references an unknown material")
        clean = {}
        for key, value in values.items():
            seq = value if key == "base_color" else (value,)
            if (key == "base_color" and (not isinstance(value, (list, tuple)) or len(value) != 3)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in seq)):
                raise ValueError("native material plan has invalid values")
            clean[key] = list(value) if key == "base_color" else float(value)
        result[name] = clean
    return result


def _clean_environment(ai, projection_path):
    projection = read_png(projection_path)
    if projection.shape != ai.shape or not (projection[..., 3] > 0).any():
        raise ValueError("product projection does not match AI canvas for environment cleanup")
    removal, evidence = _removal_mask(ai, projection[..., 3] > 0)
    plate = ai.copy()
    filled = np.clip(inpaint_pushpull(ai[..., :3].astype(np.float32) / 255, removal) * 255 + .5, 0, 255).astype(np.uint8)
    plate[..., :3][removal] = filled[removal]
    return plate, evidence


def _render_modifier_mismatch(source):
    for modifier in source.modifiers:
        if bool(getattr(modifier, "show_render", False)) != bool(getattr(modifier, "show_viewport", False)):
            return modifier.name
        if hasattr(modifier, "levels") and hasattr(modifier, "render_levels") and modifier.levels != modifier.render_levels:
            return modifier.name
    return None


def _alpha_agreement(native_rgba, projection_rgba, threshold=127):
    native = np.asarray(native_rgba)[..., 3] > threshold
    approved = np.asarray(projection_rgba)[..., 3] > threshold
    if native.shape != approved.shape or not approved.any():
        raise ValueError("native/projection alpha canvas mismatch")
    union = native | approved
    iou = float((native & approved).sum() / max(1, union.sum()))
    area_delta = abs(int(native.sum()) - int(approved.sum())) / max(1, int(approved.sum()))
    evidence = {"threshold": threshold, "native_pixels": int(native.sum()), "projection_pixels": int(approved.sum()),
                "iou": iou, "area_delta": area_delta, "minimum_iou": .985, "maximum_area_delta": .015}
    if iou < .985 or area_delta > .015:
        raise ValueError("native alpha drifted from approved projection: IoU %.4f, area delta %.4f" % (iou, area_delta))
    return evidence


def finish(context, props, ai_path, projection_record, helpers, directory, material_plan):
    """Render evaluated product copies in a disposable scene and compose their exact same-camera alpha."""
    import bpy
    from mathutils import Vector
    from . import product_projection
    if threading.current_thread() is not threading.main_thread(): raise RuntimeError("native product finish must run on Blender main thread")
    if not product_projection.current(context, props, projection_record): raise ValueError("product projection is stale")
    if not isinstance(helpers, dict) or not all(Path(path).is_file() for path in helpers.values()):
        raise ValueError("frozen product helpers are incomplete")
    ai = read_png(ai_path); root = Path(directory); root.mkdir(parents=True, exist_ok=True); write_png(root / "native_raw_ai.png", ai)
    clean_environment, environment_evidence = _clean_environment(ai, projection_record["path"])
    clean_path = root / "native_clean_ai_background.png"; write_png(clean_path, clean_environment)
    collection = getattr(props, "product_collection", None); camera = context.scene.camera
    if collection is None or camera is None: raise ValueError("product collection/camera missing")
    scene = bpy.data.scenes.new("__wondful_native_finish"); temp = bpy.data.collections.new("__wondful_native_finish"); scene.collection.children.link(temp)
    owned = [scene, temp]; objects=[]; copied={}; status={}; paths={}; result = None
    try:
        from .reference_material_plan import catalog
        graph = context.evaluated_depsgraph_get()
        protections = {row["name"]: row for row in catalog(context, props)}
        mats = _checked_plan(material_plan, protections)
        for source in collection.all_objects:
            if source.hide_render or not getattr(source, "visible_camera", True):
                continue
            if source.type in {"VOLUME", "GREASEPENCIL", "GPENCIL"}:
                raise ValueError("unsupported render-visible product object: %s (%s)" % (source.name, source.type))
            if source.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
                continue  # EMPTY/ARMATURE/lights do not contribute visible geometry.
            mismatch = _render_modifier_mismatch(source)
            if mismatch:
                raise ValueError("render/viewport modifier mismatch for %s: %s" % (source.name, mismatch))
            evaluated = source.evaluated_get(graph)
            try:
                mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=graph)
            except RuntimeError as exc:
                raise ValueError("could not evaluate render-visible product object: %s" % source.name) from exc
            owned.append(mesh)
            obj = bpy.data.objects.new(source.name + "__native", mesh); temp.objects.link(obj)
            obj.matrix_world = evaluated.matrix_world.copy(); objects.append(obj); owned.append(obj)
            # new_from_object already owns slots.  Replace them index-for-index
            # so polygon material IDs and OBJECT-linked slots stay aligned.
            for index, slot in enumerate(evaluated.material_slots):
                original = slot.material
                if original is None:
                    continue
                copy = copied.get(original)
                if copy is None:
                    copy = original.copy(); copied[original] = copy; owned.append(copy)
                if index >= len(mesh.materials):
                    raise ValueError("evaluated material slots changed for %s" % source.name)
                mesh.materials[index] = copy
        if not objects: raise ValueError("selected product has no mesh")
        for original, mat in copied.items():
            if original.name in mats:
                protection = dict(protections[original.name])
                protection["library"] = getattr(original, "library", None) is not None
                _apply_values(mat, mats[original.name], protection, status)
        evaluated_camera = camera.evaluated_get(graph)
        cam_data = evaluated_camera.data.copy(); cam = bpy.data.objects.new(camera.name + "__native", cam_data); temp.objects.link(cam); cam.matrix_world = evaluated_camera.matrix_world.copy(); owned += [cam_data, cam]; scene.camera = cam
        # Projection freshness does not yet fingerprint DOF dependencies.  Disable
        # it rather than silently rendering a different blurred silhouette.
        cam_data.dof.use_dof = False
        scene.render.engine='CYCLES'; scene.cycles.device='CPU'; scene.cycles.samples=56; scene.cycles.use_denoising=True; scene.cycles.max_bounces=8; scene.render.film_transparent=True; scene.render.resolution_x=ai.shape[1]; scene.render.resolution_y=ai.shape[0]; scene.render.resolution_percentage=100; scene.render.pixel_aspect_x=context.scene.render.pixel_aspect_x; scene.render.pixel_aspect_y=context.scene.render.pixel_aspect_y; scene.render.image_settings.file_format='PNG'; scene.render.image_settings.color_mode='RGBA'; scene.view_settings.view_transform='Standard'; scene.view_settings.look='Medium High Contrast'; scene.view_settings.exposure=0; scene.view_settings.gamma=1
        world=bpy.data.worlds.new('__wondful_native_world'); owned.append(world); world.use_nodes=True; scene.world=world; nodes=world.node_tree.nodes; nodes.clear(); out=nodes.new('ShaderNodeOutputWorld'); bg=nodes.new('ShaderNodeBackground'); env=nodes.new('ShaderNodeTexEnvironment'); image=bpy.data.images.load(str(clean_path), check_existing=False); owned.append(image); env.image=image; bg.inputs['Strength'].default_value=.10; world.node_tree.links.new(env.outputs['Color'],bg.inputs['Color']); world.node_tree.links.new(bg.outputs['Background'],out.inputs['Surface'])
        pts=[obj.matrix_world @ Vector(v) for obj in objects for v in obj.bound_box]; lo,hi=Vector(map(min,zip(*pts))),Vector(map(max,zip(*pts))); centre=(lo+hi)/2; extent=max(hi-lo); q=cam.matrix_world.to_quaternion(); forward=q@Vector((0,0,-1)); right=q@Vector((1,0,0)); up=q@Vector((0,1,0)); cool,warm=_palette(ai)
        for name,pos,col,ratio in (("key",centre-right*extent*1.1+up*extent*.65-forward*extent*.9,warm,2.2),("fill",centre+right*extent*1.1+up*extent*.28-forward*extent*.55,cool,.55),("rim",centre+up*extent*1.45-forward*extent*.15,cool,.65)):
            data=bpy.data.lights.new('__native_'+name,'AREA'); data.shape='RECTANGLE'; data.size=extent*1.7; data.size_y=extent*.37; data.color=col; data.energy=55*ratio*((pos-centre).length/1.5)**2; light=bpy.data.objects.new(data.name,data); temp.objects.link(light); light.location=pos; light.rotation_euler=(centre-pos).to_track_quat('-Z','Y').to_euler(); owned += [data,light]
        paths['native']=str(root/'native_product_rgba.png'); scene.render.filepath=paths['native']; bpy.ops.render.render(scene=scene.name, write_still=True)
        native=read_png(paths['native']); alpha_evidence = _alpha_agreement(native, read_png(projection_record['path']))
        half=extent*1.5; plane_mesh=bpy.data.meshes.new('__native_receiver_mesh'); plane_mesh.from_pydata([(-half,-half,0),(half,-half,0),(half,half,0),(-half,half,0)],[],[(0,1,2,3)]); owned.append(plane_mesh)
        receiver=bpy.data.objects.new('__native_receiver',plane_mesh); temp.objects.link(receiver); receiver.location=(centre.x,centre.y,lo.z-.002); owned.append(receiver)
        receiver_mat=bpy.data.materials.new('__native_receiver_mat'); receiver_mat.use_nodes=True; node=next(n for n in receiver_mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED'); node.inputs['Base Color'].default_value=(.5,.5,.5,1); node.inputs['Roughness'].default_value=.65; plane_mesh.materials.append(receiver_mat); owned.append(receiver_mat)
        for obj in objects: obj.visible_camera=False
        paths['shadowed']=str(root/'native_receiver_shadowed.png'); scene.render.filepath=paths['shadowed']; bpy.ops.render.render(scene=scene.name, write_still=True)
        for obj in objects: obj.hide_render=True
        paths['receiver']=str(root/'native_receiver_unshadowed.png'); scene.render.filepath=paths['receiver']; bpy.ops.render.render(scene=scene.name, write_still=True)
        output, comp=compose_native(ai,native,read_png(paths['shadowed']),read_png(paths['receiver']),_paint_mask(helpers, ai.shape)); paths['output']=str(root/'native_product_finish.png'); write_png(paths['output'],output)
        applied = sum(len(row['applied']) for row in status.values()); skipped = sum(len(row['skipped']) for row in status.values())
        report={**comp,"output":paths['output'],"raw_ai":str(root/'native_raw_ai.png'),"clean_environment":str(clean_path),"environment_cleanup":environment_evidence,"alpha_agreement":alpha_evidence,"material_status":status,"material_count":len(mats),"material_plan_applied":{"materials":sum(bool(row['applied']) for row in status.values()),"values":applied,"skipped_values":skipped},"no_plan":not bool(mats),"camera_dof":"disabled: projection signature does not include DOF dependencies","unsupported":["lighting/material calibration is approximate; no automatic three-view PBR extraction"],"helpers_frozen":bool(helpers)}
        result = (paths['output'], report)
    finally:
        for item in reversed(owned):
            try:
                if isinstance(item,bpy.types.Object): bpy.data.objects.remove(item, do_unlink=True)
                elif isinstance(item,bpy.types.Mesh): bpy.data.meshes.remove(item, do_unlink=True)
                elif isinstance(item,bpy.types.Material): bpy.data.materials.remove(item, do_unlink=True)
                elif isinstance(item,bpy.types.Camera): bpy.data.cameras.remove(item, do_unlink=True)
                elif isinstance(item,bpy.types.Light): bpy.data.lights.remove(item, do_unlink=True)
                elif isinstance(item,bpy.types.World): bpy.data.worlds.remove(item, do_unlink=True)
                elif isinstance(item,bpy.types.Image): bpy.data.images.remove(item, do_unlink=True)
                elif isinstance(item,bpy.types.Collection): bpy.data.collections.remove(item, do_unlink=True)
                elif isinstance(item,bpy.types.Scene): bpy.data.scenes.remove(item, do_unlink=True)
            except (ReferenceError, RuntimeError): pass
    if not product_projection.current(context, props, projection_record):
        raise ValueError("product projection changed during native finish cleanup")
    result[1]["native_scene_state_preserved"] = True
    (root/'native_product_finish.json').write_text(json.dumps(result[1],indent=2)+'\n')
    return result
