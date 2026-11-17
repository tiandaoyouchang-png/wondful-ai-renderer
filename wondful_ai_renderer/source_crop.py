"""Local canvas for the single-source edit (2026-11 review, "局部画布").

In a wide photograph the car is a few hundred pixels wide, and the image endpoint re-renders the
whole frame at its own size (often medium quality): lamp internals, spokes and badges lose
resolution and the silhouette drifts by several pixels (IoU 0.91–0.93 in the YU7 tests).

So the edit is made on a crop around the product instead of the whole frame:

* the crop has the endpoint's landscape aspect (3:2), the product fills ~62 % of its width, and
  it reaches down far enough for the contact shadow;
* the provider's result is scaled back to the crop box and pasted into the photograph (the clean
  plate when there is one), with a per-channel gain matched on a border ring and a feathered seam;
* outside the crop every pixel is the photograph's.

Pure numpy, worker-thread safe.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CROP_VERSION = 2
ASPECT = 1.5              # 1536×1024, the endpoint's landscape size
FILL = 0.70               # product width / crop width (0.62 → 0.70 after the YU7 round 4)
MAX_ALIGN_SCALE = 0.05    # registration accepted only for small drift: |scale−1| ≤ 5 %
MAX_ALIGN_SHIFT = 0.04    # … and a shift ≤ 4 % of the crop width
MIN_GAIN, MAX_GAIN = 0.8, 1.25
SKIP_IF_WIDER = 0.85      # crop would cover most of the frame anyway → whole-frame edit


def crop_box(product_alpha: np.ndarray, aspect: float = ASPECT, fill: float = FILL):
    """(x0, y0, x1, y1) of the local canvas, inside the frame; None when cropping gains nothing."""
    from .layered_output import bbox
    H, W = product_alpha.shape[:2]
    b = bbox(np.asarray(product_alpha) > 0.5)
    if b is None:
        return None
    y0, y1, x0, x1 = b
    cw, ch = x1 - x0 + 1, y1 - y0 + 1
    w = cw / fill
    h = w / aspect
    if h < ch / 0.66:                 # tall product: height decides
        h = ch / 0.66
        w = h * aspect
    if w > W or h > H:                # cannot keep the aspect inside the frame
        scale = min(W / w, H / h)
        w, h = w * scale, h * scale
    if w >= SKIP_IF_WIDER * W:
        return None
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0 + 0.10 * ch  # a little ground below for the contact shadow
    bx0 = int(round(min(max(cx - w / 2, 0), W - w)))
    by0 = int(round(min(max(cy - h / 2, 0), H - h)))
    bw, bh = int(round(w)), int(round(h))
    return bx0, by0, min(W, bx0 + bw), min(H, by0 + bh)


def crop_file(path, box, out_path, keep_alpha=False):
    from .png_io import read_png, write_png
    from .reference_collage import read_rgb
    x0, y0, x1, y1 = box
    a = read_png(path) if keep_alpha else read_rgb(path)
    write_png(out_path, np.ascontiguousarray(a[y0:y1, x0:x1]))
    return str(out_path)


def _feather(h, w, f):
    yy = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]
    xx = np.minimum(np.arange(w), np.arange(w)[::-1])[None, :]
    return np.clip(np.minimum(yy, xx) / float(max(1, f)), 0, 1).astype(np.float32)


def sample_similarity(img: np.ndarray, s, dx: float, dy: float, centre) -> np.ndarray:
    """img'(x) = img(T(x)), T(x) = (x − c)·s + c + d: brings content that the model drew at T(x)
    back to x (the inverse of ``layered_output.warp_mask``'s mapping). Bilinear, edge-clamped."""
    h, w = img.shape[:2]
    cx, cy = centre
    sx, sy = (s, s) if np.isscalar(s) else (float(s[0]), float(s[1]))
    xs = np.clip((np.arange(w, dtype=np.float32) - cx) * sx + cx + dx, 0, w - 1)
    ys = np.clip((np.arange(h, dtype=np.float32) - cy) * sy + cy + dy, 0, h - 1)
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    fx = (xs - x0)[None, :, None]; fy = (ys - y0)[:, None, None]
    a = img[y0][:, x0]; b = img[y0][:, x1]; c = img[y1][:, x0]; d = img[y1][:, x1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def register_product(ai: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, dict]:
    """Snap the model's car onto the Blender silhouette: fit the mask to the image's edges
    (``layered_output.fit_mask``), and when the drift is small, resample the car region so it
    sits exactly on the model; the background a little away from the car is left in place."""
    from .layered_output import fit_mask, bbox, dilate, blur
    info = {"applied": False}
    hard = alpha > 0.5
    b = bbox(hard)
    if b is None:
        return ai, info
    fit = fit_mask(ai, alpha.astype(np.float32))
    sx, sy, dx, dy = float(fit["scale_x"]), float(fit["scale_y"]), float(fit["dx"]), float(fit["dy"])
    info.update({k: fit[k] for k in ("scale_x", "scale_y", "dx", "dy", "search", "chamfer_before", "chamfer_after")})
    w = ai.shape[1]
    if fit["search"] == "identity" or (max(abs(sx - 1), abs(sy - 1)) < 0.003 and max(abs(dx), abs(dy)) < 0.75):
        return ai, info
    if max(abs(sx - 1), abs(sy - 1)) > MAX_ALIGN_SCALE or max(abs(dx), abs(dy)) > MAX_ALIGN_SHIFT * w:
        info["reason"] = "drift too large to correct by resampling"
        return ai, info
    ys, xs = np.nonzero(hard)
    centre = (float(xs.mean()), float(ys.mean()))
    warped = sample_similarity(ai, (sx, sy), dx, dy, centre)
    y0, y1, x0, x1 = b
    grow = max(4, int(round(0.06 * max(y1 - y0, x1 - x0))))
    region = dilate(hard, grow).astype(np.float32)
    soft = np.clip(blur(region, max(2, grow // 2)), 0, 1)[..., None]
    info["applied"] = True
    return ai * (1 - soft) + warped * soft, info


def paste_back(ai_crop_path, base_path, box, out_path, product_alpha=None) -> dict:
    """Scale the provider's crop back into ``box`` of ``base_path`` (photograph / plate), snap the car
    to the Blender silhouette when ``product_alpha`` (crop-space, 0..1) is given, match exposure on
    the border ring, feather the seam; write a full-frame RGB PNG."""
    from .png_io import write_png
    from .reference_collage import read_rgb
    from .layered_output import resize
    base = read_rgb(base_path).astype(np.float32)
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    ai = read_rgb(ai_crop_path).astype(np.float32)
    if ai.shape[:2] != (bh, bw):
        ai = np.clip(resize(ai, bw, bh), 0, 255)
    align = {"applied": False}
    if product_alpha is not None:
        a = np.asarray(product_alpha, np.float32)
        if a.shape != (bh, bw):
            a = np.clip(resize(a[..., None], bw, bh)[..., 0], 0, 1)
        try:
            ai, align = register_product(ai, a)
        except Exception as exc:
            align = {"applied": False, "error": str(exc)[:120]}
    f = max(8, int(round(0.04 * bw)))
    ring = _feather(bh, bw, 2 * f) < 1.0
    ref = base[y0:y1, x0:x1]
    gain = np.ones(3, np.float32)
    if ring.any():
        num = ref[ring].mean(axis=0) + 1.0
        den = ai[ring].mean(axis=0) + 1.0
        gain = np.clip(num / den, MIN_GAIN, MAX_GAIN)
    ai = np.clip(ai * gain, 0, 255)
    m = _feather(bh, bw, f)[..., None]
    out = base.copy()
    out[y0:y1, x0:x1] = ref * (1 - m) + ai * m
    write_png(out_path, np.clip(out + 0.5, 0, 255).astype(np.uint8))
    return {"version": CROP_VERSION, "box": [int(x0), int(y0), int(x1), int(y1)],
            "gain": [round(float(g), 4) for g in gain], "feather": f, "align": align}
