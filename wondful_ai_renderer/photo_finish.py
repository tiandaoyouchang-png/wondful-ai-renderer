"""1.2.4 照片质感 (高级 › 成片): a deterministic camera-like finish on the final image.

Why: the generated images have almost no sensor noise (flat-area texture floor 0.25–0.35 vs 0.8–1.1 for
real full-frame photos delivered at 2560 px), bright paint highlights clip hard, and city scenes are
over-saturated. This runs LAST, at the final canvas size, after 细节修复 / 按材质校色 (and after 多角度
统一调色). ``_raw.png`` is never touched. Pure numpy (no bpy, Pillow or OpenCV): safe on worker threads.

Order (research prototype ``ai_look/postfx.py``, 2026-10-19):
1. highlight shoulder — soft tanh roll-off of the max channel above ``shoulder`` (hue-preserving, slope 1 at
   the knee; the prototype's curve had a slope of 1.4 there and brightened the 0.82–0.97 band instead);
2. saturation compression on the ENVIRONMENT only (the product mask keeps its saturation, so the
   material colour lock is not undone);
3. vignette (corner −``vignette`` EV, linear light);
4. lateral chromatic aberration (R/B scaled about the centre, ``ca`` px at the corner);
5. signal-dependent sensor grain (shot + read noise in linear light, ~0.55 px grain, 30 % chroma),
   seeded from the image content so the same input always gives the same output.

The fixed-coordinate IoU is computed on the raw AI image before this runs; the product colour shifts
by < 1 ΔE on average (measured 0.4 mean / 0.9 p95 at 细微).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

LEVEL_OFF, LEVEL_SUBTLE, LEVEL_STANDARD = "OFF", "SUBTLE", "STANDARD"
LEVELS = {
    # 细微 = the report's calibrated values (grain 0.35 lands at real-photo noise 0.8–1.1)
    LEVEL_SUBTLE: {"shoulder": 0.82, "sat_knee": 0.55, "sat_amt": 0.35, "vignette": 0.20, "ca": 0.40, "grain": 0.35},
    # 标准 = a little more film-like (grain 0.7 was measured as too coarse; 0.6 stays below it)
    LEVEL_STANDARD: {"shoulder": 0.80, "sat_knee": 0.55, "sat_amt": 0.40, "vignette": 0.25, "ca": 0.50, "grain": 0.60},
}
LABELS = {LEVEL_OFF: "关", LEVEL_SUBTLE: "细微", LEVEL_STANDARD: "标准"}
CLEAN_SUFFIX = "_clean.png"     # the exported image before this finish, in <output>/.wondful/sidecars/
_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


# --------------------------------------------------------------------------- helpers

def srgb_to_linear(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    x = np.maximum(np.asarray(x, np.float32), 0.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def gaussian(img, sigma):
    """Separable Gaussian blur (reflect border), float32; works on (H, W) or (H, W, C)."""
    a = np.asarray(img, np.float32)
    if sigma <= 0:
        return a.copy()
    r = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = a
    for axis in (0, 1):
        n = out.shape[axis]
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="reflect" if n > r else "edge")
        acc = np.zeros_like(out)
        for i, kv in enumerate(k):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(i, i + n)
            acc += kv * p[tuple(sl)]
        out = acc
    return out


def seed_of(img) -> int:
    """Content seed (md5 of a sparse sample of the 8-bit image) → deterministic grain."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(np.rint(np.asarray(a, np.float32) * 255.0), 0, 255).astype(np.uint8)
    return int(hashlib.md5(np.ascontiguousarray(a[::37, ::41]).tobytes()).hexdigest()[:8], 16)


def protect_mask(mask, shape):
    """Product protection 0..1 at ``shape`` (H, W): the product mask grown by 2 px and feathered, so the
    whole product interior is fully protected and only the outline blends."""
    if mask is None:
        return None
    m = np.asarray(mask, np.float32)
    if m.shape[:2] != tuple(shape[:2]):
        from .layered_output import resize
        m = resize(m, int(shape[1]), int(shape[0]))
    from .layered_output import dilate
    grown = dilate(m > 0.5, 2).astype(np.float32)
    return np.where(m > 0.5, 1.0, np.clip(np.maximum(gaussian(grown, 1.5), m), 0.0, 1.0)).astype(np.float32)


def _bilinear_channel(ch, u, v):
    H, W = ch.shape
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    fx, fy = u - x0, v - y0
    a, b = ch[y0, x0], ch[y0, x0 + 1]
    c, d = ch[y0 + 1, x0], ch[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


# --------------------------------------------------------------------------- steps (float RGB 0..1)

def tone(x, protect=None, shoulder=0.82, sat_knee=0.55, sat_amt=0.35):
    """Highlight shoulder on the max channel (hue kept) + saturation compression above ``sat_knee``
    outside the product (``protect`` = 1 keeps the saturation)."""
    x = np.asarray(x, np.float32)
    m = x.max(-1, keepdims=True)
    t = float(shoulder)
    k = 0.8      # slope 1 at the knee (no kink), white 1.0 → t + 0.83·(1 − t) ≈ 0.97 at 细微
    mm = np.where(m > t, t + (1 - t) * np.tanh((m - t) / (1 - t) * k) / k, m)
    x = x * (mm / np.maximum(m, 1e-6))
    mx = x.max(-1, keepdims=True)
    mn = x.min(-1, keepdims=True)
    s = (mx - mn) / np.maximum(mx, 1e-6)
    sn = np.where(s > sat_knee, sat_knee + (s - sat_knee) * (1 - sat_amt), s)
    if protect is not None:
        p = protect[..., None]
        sn = p * s + (1 - p) * sn
    ratio = np.where(s > 1e-6, sn / np.maximum(s, 1e-6), 1.0)
    out = mx - (mx - x) * ratio
    return np.clip(out, 0, 1).astype(np.float32)


def vignette(x, stops=0.2):
    H, W = x.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot((xx + 0.5 - W / 2) / (W / 2), (yy + 0.5 - H / 2) / (W / 2))
    rc = float(np.hypot(1.0, H / W))
    g = (2.0 ** (-stops * np.clip(r / rc, 0, 1) ** 2.5)).astype(np.float32)
    return np.clip(linear_to_srgb(srgb_to_linear(x) * g[..., None]), 0, 1)


def lateral_ca(x, px_at_corner=0.4):
    """Red slightly magnified, blue slightly shrunk about the centre (``px_at_corner`` at the corner)."""
    H, W = x.shape[:2]
    cx, cy = W / 2 - 0.5, H / 2 - 0.5
    k = float(px_at_corner) / float(np.hypot(W / 2, H / 2))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = x.copy()
    for c, s in ((0, 1 + k), (2, 1 - k)):
        out[..., c] = _bilinear_channel(x[..., c], cx + (xx - cx) / s, cy + (yy - cy) / s)
    return out


def grain(x, strength=0.35, size_px=0.55, chroma=0.30, seed=0):
    """Signal-dependent sensor noise in linear light (shot + read), slightly correlated, weak chroma."""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    H, W = x.shape[:2]
    lin = srgb_to_linear(x)
    Y = lin @ _LUMA
    sigma = (strength * np.sqrt(0.00055 * Y + 0.0000020)).astype(np.float32)
    n = rng.standard_normal((H, W), dtype=np.float32)
    if size_px > 0:
        n = gaussian(n, size_px)
        n /= n.std() + 1e-8
    c = rng.standard_normal((H, W, 2), dtype=np.float32)
    c = gaussian(c, 1.6)
    c /= c.std() + 1e-8
    cn = np.stack([c[..., 0], -0.5 * (c[..., 0] + c[..., 1]), c[..., 1]], -1) * (chroma * sigma)[..., None]
    out = lin + (sigma * n)[..., None] + cn
    return np.clip(linear_to_srgb(np.maximum(out, 0)), 0, 1).astype(np.float32)


def apply(rgb, mask=None, level=LEVEL_SUBTLE, seed=None):
    """Full finish of one float RGB image (0..1). ``mask``: product mask 0..1 (same size or resampled).
    Returns float32 RGB; ``level`` OFF returns the input unchanged."""
    p = LEVELS.get(level)
    x = np.asarray(rgb, np.float32)[..., :3]
    if p is None:
        return x.copy()
    if seed is None:
        seed = seed_of(x)
    prot = protect_mask(mask, x.shape) if mask is not None else None
    x = tone(x, prot, p["shoulder"], p["sat_knee"], p["sat_amt"])
    if p["vignette"]:
        x = vignette(x, p["vignette"])
    if p["ca"]:
        x = lateral_ca(x, p["ca"])
    if p["grain"]:
        x = grain(x, p["grain"], seed=seed)
    return x


def apply_file(src, dst, mask_path="", level=LEVEL_SUBTLE) -> dict:
    """PNG in → PNG out (alpha kept; fully transparent padding left as it is). Never raises: on any
    error the source is copied and the report says why."""
    t0 = time.time()
    rep = {"level": level, "applied": False}
    try:
        from .png_io import read_png, write_png
        rgba = read_png(src)
        if level not in LEVELS:
            rep["reason"] = "off"
            if str(src) != str(dst):
                write_png(dst, rgba)
            return rep
        H, W = rgba.shape[:2]
        mask = None
        if mask_path and Path(mask_path).is_file():
            from .layered_output import load_mask
            mask = load_mask(mask_path, (W, H))
        seed = seed_of(rgba[..., :3])
        out = apply(rgba[..., :3].astype(np.float32) / 255.0, mask, level, seed=seed)
        res = rgba.copy()
        res[..., :3] = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
        pad = rgba[..., 3] == 0
        if pad.any():
            res[pad] = rgba[pad]
        write_png(dst, res)
        rep.update({"applied": True, "seed": seed, "protected": mask is not None,
                    "seconds": round(time.time() - t0, 2), "size": [W, H]})
    except Exception as exc:     # never lose the image over a finish
        rep["error"] = str(exc)[:200]
        try:
            if str(src) != str(dst):
                import shutil
                shutil.copy2(src, dst)
        except Exception:
            pass
    return rep


def clean_path(exported) -> Path:
    """``<output>/.wondful/sidecars/<stem>_clean.png`` (= output_manager.sidecar_path; no bpy here)."""
    p = Path(exported)
    return p.parent / ".wondful" / "sidecars" / (p.stem + CLEAN_SUFFIX)


def clean_source(image_path) -> str:
    """The pre-finish copy of an exported image when there is one, else the image itself (多角度 uses
    it as the 主视角 texture so the grain is never projected and stretched)."""
    try:
        c = clean_path(image_path)
        return str(c) if c.is_file() else str(image_path)
    except Exception:
        return str(image_path)


def note(rep) -> str:
    if not rep:
        return ""
    if rep.get("error"):
        return f"照片质感失败（已保留原图）：{rep['error'][:60]}"
    if not rep.get("applied"):
        return ""
    return f"照片质感：{LABELS.get(rep.get('level'), '')}"
