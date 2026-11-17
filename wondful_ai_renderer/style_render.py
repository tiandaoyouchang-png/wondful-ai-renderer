"""Mode 2 「按参考风格渲染」 (2026-11).

The user supplies the car (its model and materials) and the camera; the reference photograph
supplies only the rendering style: light, colour grade, atmosphere, kind of place, lens feel.
No scene blockout and no copy of the reference layout.

Pipeline:
  1. the same lit product pass as mode 1 (own materials, temporary clear coat, shadow catcher),
     rendered from the user's camera under a gradient world sampled from the reference;
  2. a placeholder background in the reference's colours (sky → horizon → ground, horizon at the
     camera's true horizon line) with the lit product composited on it = the canvas;
  3. one image edit: canvas (image 1) + reference (image 2, style only) + identity sheet
     (image 3, materials only), with the product's geometry, pose, position and size locked;
  4. the result is fitted to the canvas and the silhouette checked against the Blender mask
     (``result_worker.source_detail_step``, shared with mode 1).

Worker-safe parts (numpy only) are separate from the bpy helpers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

STYLE_RENDER_VERSION = 1


def _linear_to_srgb(c):
    c = np.clip(np.asarray(c, np.float32), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def gradient_background(env: dict, width: int, height: int, horizon: float = 0.45) -> np.ndarray:
    """uint8 RGB placeholder: sky at the top, the horizon colour at ``horizon`` (0 = top),
    ground at the bottom. Smooth, no texture: the model is told it is only a colour placeholder."""
    sky = _linear_to_srgb(env["sky"])
    hor = _linear_to_srgb(env["horizon"])
    gnd = _linear_to_srgb(env["ground"])
    hz = float(np.clip(horizon, 0.05, 0.95))
    y = (np.arange(height, dtype=np.float32) + 0.5) / height
    t_up = np.clip(y / hz, 0, 1)[:, None]
    t_dn = np.clip((y - hz) / (1 - hz), 0, 1)[:, None]
    col = np.where((y < hz)[:, None], sky * (1 - t_up) + hor * t_up, hor * (1 - t_dn) + gnd * t_dn)
    img = np.repeat(col[:, None, :], width, axis=1)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def build_style_canvas(reference_path, lookdev_path, projection_path, output_path, horizon: float) -> dict:
    """Placeholder background in the reference's colours + the lit product → RGB canvas."""
    from .png_io import read_png, write_png
    from .reference_collage import read_rgb
    from .product_lookdev import environment_colours, composite
    env = environment_colours(read_rgb(reference_path))
    look = read_png(lookdev_path)
    flat = read_png(projection_path)
    if look.ndim != 3 or look.shape[2] != 4 or flat.ndim != 3 or flat.shape[2] != 4:
        raise ValueError("产品光照参考或投影缺少 RGBA 通道。")
    h, w = flat.shape[:2]
    bg = gradient_background(env, w, h, horizon)
    background_path = Path(output_path).with_name("style_background.png")
    write_png(background_path, bg)
    out = composite(bg, look, flat[..., 3])
    write_png(output_path, out)
    return {"path": str(output_path), "background": str(background_path), "environment": env,
            "horizon": round(float(horizon), 4), "size": [int(w), int(h)]}


def style_prompt() -> str:
    from .source_reference import PAINT_FINISH_POLICY, DETAIL_FIDELITY_POLICY
    return "\n".join([
        "【按参考风格渲染】图 1 是我的汽车三维模型在指定机位下的光照渲染，车外的背景只是按参考图色调铺的占位渐变，不是要保留的内容。"
        "图 2 是风格参考照片。图 3 是该车型的三视图或身份图。",
        "1. 车辆的外形、比例、姿态、在画面中的位置和大小、机位、透视和焦距必须与图 1 完全一致；"
        "车身轮廓、灯组、格栅、进气口、轮毂、车窗、后视镜等所有造型细节都按图 1，不得增删或改形，不得用身份图或其裁切重设计。"
        "三视图或身份图只确定各部件的材质属性。",
        "2. 按图 2 的风格生成整个画面：环境类型与地貌、时间与天气、主光方向与软硬、色调与调色、对比度、雾气与空气感、景深、镜头质感与颗粒都与图 2 一致；"
        "环境按图 1 的机位和地平线自然展开，不需要复制图 2 的构图和布局。",
        "3. 让车辆真实地处于这个环境中：接地阴影、轮胎与地面的接触、环境反射、车漆清漆高光和车窗反射都符合图 2 的光线；车漆颜色保持图 1。",
        PAINT_FINISH_POLICY,
        DETAIL_FIDELITY_POLICY,
        "4. 画面中不出现其他车辆、人物、文字、标志或水印。只输出一张照片级图像。",
    ])


def codex_style_fn(*, model="", client_version="", long_edge=2048, timeout=900, log_dir="", cancel_check=None):
    """``fn(canvas, reference, identity, out)`` → path of the generated image (Codex direct)."""
    def fn(canvas, reference, identity, out):
        from . import codex_direct as CD
        from .composition import image_dimensions
        w, h = image_dimensions(canvas)
        refs = [reference] + ([identity] if identity else [])
        manifest = [{"path": reference, "members": [{"role": "environment"}]}]
        if identity:
            manifest.append({"path": identity, "members": [{"role": "product"}]})
        r = CD.edit_image(prompt=style_prompt(), edit_base_path=canvas, reference_paths=refs, output_path=out,
                          manifest=manifest,
                          size=CD.choose_edit_size(w, h, int(long_edge)), model=model,
                          client_version=client_version, timeout=timeout, log_dir=log_dir or str(Path(out).parent),
                          tag="style_render", cancel_check=cancel_check, exact_prompt=True)
        return (r or {}).get("source") or out
    return fn


# ---------------------------------------------------------------- bpy helpers

def horizon_fraction(scene, camera) -> float:
    """Where the true horizon crosses the frame (0 = top, 1 = bottom), clamped to the frame."""
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    m = camera.matrix_world
    loc = m.translation
    fwd = m.to_3x3() @ Vector((0.0, 0.0, -1.0))
    flat = Vector((fwd.x, fwd.y, 0.0))
    if flat.length < 1e-6:
        return 0.45
    far = loc + flat.normalized() * 10000.0
    co = world_to_camera_view(scene, camera, far)
    return float(min(0.98, max(0.02, 1.0 - co.y)))


# ---------------------------------------------------------------- whole-frame registration

MAX_WHOLE_SCALE = 0.15   # the whole frame is generated, so it may be rescaled this much …
MAX_WHOLE_SHIFT = 0.08   # … and shifted this fraction of the width to put the car on the model


def register_whole(ai: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, dict]:
    """Mode 2 only: the background is generated, not a photograph to preserve, so the model's
    drift (a car drawn ~10 % larger, a little lower) is corrected by one uniform similarity of the
    whole frame instead of a local warp. Edges are clamped; the fitted car sits on the Blender
    silhouette."""
    from .layered_output import fit_mask
    info = {"applied": False}
    hard = alpha > 0.5
    if not hard.any():
        return ai, info
    fit = fit_mask(ai, alpha.astype(np.float32), scale_range=(1 - MAX_WHOLE_SCALE, 1 + MAX_WHOLE_SCALE),
                   shift_frac=MAX_WHOLE_SHIFT)
    s = (float(fit["scale_x"]) + float(fit["scale_y"])) / 2.0
    dx, dy = float(fit["dx"]), float(fit["dy"])
    info.update({k: fit[k] for k in ("scale_x", "scale_y", "dx", "dy", "search", "chamfer_before", "chamfer_after")})
    if fit["search"] == "identity" or (abs(s - 1) < 0.003 and max(abs(dx), abs(dy)) < 0.75):
        return ai, info
    ys, xs = np.nonzero(hard)
    centre = (float(xs.mean()), float(ys.mean()))
    info.update({"applied": True, "scale": round(s, 4)})
    info["exposed"] = exposed_mask(ai.shape[:2], s, dx, dy, centre)
    return sample_mirrored(ai.astype(np.float32), s, dx, dy, centre), info


def exposed_mask(shape, s, dx, dy, centre) -> np.ndarray:
    """Pixels whose source position after the similarity lies outside the generated frame
    (they were mirrored in, not generated)."""
    h, w = shape
    cx, cy = centre
    xs = (np.arange(w, dtype=np.float32) - cx) * s + cx + dx
    ys = (np.arange(h, dtype=np.float32) - cy) * s + cy + dy
    bad_x = (xs < 0) | (xs > w - 1)
    bad_y = (ys < 0) | (ys > h - 1)
    return bad_y[:, None] | bad_x[None, :]


def _reflect(v, n):
    if n <= 1:
        return np.zeros_like(v)
    period = 2.0 * (n - 1)
    v = np.mod(np.abs(v), period)
    return np.where(v > n - 1, period - v, v)


def sample_mirrored(img: np.ndarray, s: float, dx: float, dy: float, centre) -> np.ndarray:
    """``source_crop.sample_similarity`` with mirrored instead of clamped borders, so the strip the
    shift exposes continues the picture instead of smearing its edge pixels."""
    h, w = img.shape[:2]
    cx, cy = centre
    xs = _reflect((np.arange(w, dtype=np.float32) - cx) * s + cx + dx, w)
    ys = _reflect((np.arange(h, dtype=np.float32) - cy) * s + cy + dy, h)
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    fx = (xs - x0)[None, :, None]; fy = (ys - y0)[:, None, None]
    a = img[y0][:, x0]; b = img[y0][:, x1]; c = img[y1][:, x0]; d = img[y1][:, x1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def register_file(fitted_path, projection_path, out_path) -> dict:
    from .png_io import read_png, write_png
    from .reference_collage import read_rgb
    ai = read_rgb(fitted_path).astype(np.float32)
    alpha = read_png(projection_path)[..., 3].astype(np.float32) / 255.0
    if alpha.shape != ai.shape[:2]:
        raise ValueError("结果与产品遮罩尺寸不一致。")
    out, info = register_whole(ai, alpha)
    write_png(out_path, np.clip(out + 0.5, 0, 255).astype(np.uint8))
    exposed = info.pop("exposed", None)
    if exposed is not None and exposed.any():
        info["exposed_frac"] = round(float(exposed.mean()), 4)
        info["exposed_px"] = int(max(exposed.all(axis=1).sum(), exposed.all(axis=0).sum()))
        mask_path = Path(out_path).with_name(Path(out_path).stem + "_exposed.png")
        write_png(mask_path, np.repeat((exposed.astype(np.uint8) * 255)[..., None], 3, axis=2))
        info["exposed_path"] = str(mask_path)
    return {**info, "path": str(out_path)}


def cover_fit(src, dst, width: int, height: int) -> bool:
    """Scale ``src`` to cover width×height and crop the centre (mode 2: the model may return a
    slightly different aspect; the frame is generated, so a thin crop is better than bars)."""
    from .png_io import write_png
    from .reference_collage import read_rgb
    from .layered_output import resize
    try:
        img = read_rgb(src).astype(np.float32)
    except Exception:
        return False
    sh, sw = img.shape[:2]
    scale = max(width / sw, height / sh)
    fw, fh = max(width, int(round(sw * scale))), max(height, int(round(sh * scale)))
    img = resize(img, fw, fh) if (fw, fh) != (sw, sh) else img
    ox, oy = (fw - width) // 2, (fh - height) // 2
    write_png(dst, np.clip(img[oy:oy + height, ox:ox + width] + 0.5, 0, 255).astype(np.uint8))
    return True


BORDER_GROW = 12          # px the regenerated border reaches into the generated picture
BORDER_MIN_PX = 16        # a mirrored strip narrower than this is invisible; no extra call

BORDER_PROMPT = (
    "【补全画面边缘】图 1 是一张照片，透明遮罩区域是画面边缘缺失的部分。"
    "只补全透明遮罩区域：按遮罩外的画面自然延续地面、环境、天空和光线，与原图同一光线、色调、颗粒和景深，接缝不可见。"
    "不要改变遮罩外的任何内容，不要放入车辆、人物、文字或新物体。只输出一张图。"
)


def fill_border(image_path, exposed_path, out_path, edit_fn) -> dict:
    """One masked edit that regenerates the mirrored strip; everything else is kept byte-for-byte."""
    from .png_io import read_png, write_png
    from .reference_collage import read_rgb
    from .layered_output import dilate, edit_mask_rgba
    from .source_plate import restore_outside, _resize_exact
    base = read_rgb(image_path)
    h, w = base.shape[:2]
    exposed = read_png(exposed_path)
    exposed = (exposed[..., 0] if exposed.ndim == 3 else exposed) > 127
    region = dilate(exposed, BORDER_GROW)
    mask_path = Path(out_path).with_name(Path(out_path).stem + "_editmask.png")
    write_png(mask_path, edit_mask_rgba(region))
    produced = edit_fn(str(image_path), str(mask_path), str(Path(out_path).with_name(Path(out_path).stem + "_raw.png")))
    edited = _resize_exact(read_rgb(produced), w, h)
    write_png(out_path, restore_outside(base, edited, region))
    return {"path": str(out_path), "region_frac": round(float(region.mean()), 4)}


def codex_border_fn(*, model="", client_version="", long_edge=2048, timeout=600, log_dir="", cancel_check=None):
    def fn(base, mask, out):
        from . import codex_direct as CD
        from .composition import image_dimensions
        w, h = image_dimensions(base)
        r = CD.edit_image(prompt=BORDER_PROMPT, edit_base_path=base, reference_paths=[], output_path=out,
                          size=CD.choose_edit_size(w, h, int(long_edge)), mask_path=mask, model=model,
                          client_version=client_version, timeout=timeout, log_dir=log_dir or str(Path(out).parent),
                          tag="style_border", cancel_check=cancel_check, exact_prompt=True)
        return (r or {}).get("source") or out
    return fn


def finish_variant(raw, index, helpers, width, height, session_dir, *, cancel_check=None, on_status=None,
                   border_fn=None) -> dict:
    """Worker thread: fit → whole-frame registration → silhouette check + detail paste (shared with
    mode 1). Returns {"final", "fitted", "registered", "register", "iou", "note"}; never raises
    except cancellation."""
    from .result_worker import source_detail_step, Cancelled
    out = {"final": str(raw), "fitted": "", "registered": "", "register": None, "iou": None, "note": ""}
    try:
        fitted = Path(session_dir) / f"style_fit_{index:02d}.png"
        if not cover_fit(raw, fitted, width, height):
            out["note"] = "结果格式无法读取，保留 AI 原图"
            return out
        out["fitted"] = out["final"] = str(fitted)
        reg = register_file(fitted, helpers["detail"], Path(session_dir) / f"style_reg_{index:02d}.png")
        out["register"] = {k: reg.get(k) for k in ("applied", "scale", "dx", "dy", "search",
                                                   "chamfer_before", "chamfer_after")}
        out["registered"] = out["final"] = reg["path"]
        if reg.get("exposed_path"):
            out["register"]["exposed_frac"] = reg.get("exposed_frac")
            if border_fn is not None and reg.get("exposed_px", 0) >= BORDER_MIN_PX:
                try:
                    filled = fill_border(reg["path"], reg["exposed_path"],
                                         Path(session_dir) / f"style_border_{index:02d}.png", border_fn)
                    out["border"] = filled
                    out["registered"] = out["final"] = reg["path"] = filled["path"]
                except Cancelled:
                    raise
                except Exception as exc:
                    out["border"] = {"error": str(exc)[:200]}
        step = source_detail_step(reg["path"], index, helpers, width, height, session_dir, edit_fn=None,
                                  cancel_check=cancel_check, on_status=on_status, vehicle=True)
        out["iou"] = step.get("iou")
        out["note"] = step.get("note", "")
        if step.get("refined"):
            out["final"] = step["refined"]
    except Cancelled:
        raise
    except Exception as exc:
        out["note"] = f"对齐或细节检查失败（保留 AI 原图）：{str(exc)[:80]}"
    return out


def rank(variant: dict) -> float:
    iou = (variant.get("iou") or {}).get("iou")
    return float(iou) if iou is not None else -1.0
