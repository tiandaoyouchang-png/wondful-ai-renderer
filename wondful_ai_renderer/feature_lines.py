"""1.2.3 结构线 — Blender feature lines (normal / depth discontinuities) for detail validation.

The geometry pass (``viewport_capture._render_geometry``) stores, per product pixel, the
camera-space normal (x, y) and the camera distance (normalised to the product's depth range) as
an emission colour (Standard view transform, so sRGB-encoded in the PNG). Feature lines are the
pixels where the surface normal turns by more than ``ANGLE_DEG`` or the depth jumps by more than
``DEPTH_JUMP`` between neighbours: creases, panel gaps, wheel spokes, lamp outlines, folds.

These lines come from the geometry alone (no materials, no light), so the detail score that uses
them is independent of the detail pass the paste copies in: pasting a texture that the product
does not have cannot raise the score any more (GPT-6 review, checkerboard counterexample).
Pure numpy, no bpy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ANGLE_DEG = 28.0       # normal turn between neighbours that counts as a crease
DEPTH_JUMP = 0.025     # depth jump (share of the product's depth range) that counts as an edge
MIN_LINE_PX = 6        # isolated specks smaller than this are dropped


def geometry_path_for(detail_path) -> str:
    """The geometry (feature-line) pass sits next to the detail pass."""
    p = Path(str(detail_path))
    if "detail_pass" in p.stem:
        return str(p.with_name(p.stem.replace("detail_pass", "geometry_pass") + p.suffix))
    return str(p.with_name(p.stem + "_geometry" + (p.suffix or ".png")))


def sidecar_path(path) -> str:
    return str(path) + ".json"


def _srgb_to_linear(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def decode(rgba: np.ndarray, encoded_srgb: bool = True):
    """(normal HxWx3, depth HxW in 0..1, product HxW bool) from the geometry pass pixels."""
    a = rgba.astype(np.float32)
    if rgba.dtype == np.uint8:
        a = a / 255.0
    rgb = a[..., :3]
    if encoded_srgb:
        rgb = _srgb_to_linear(rgb)
    alpha = a[..., 3] if a.shape[-1] == 4 else np.ones(a.shape[:2], np.float32)
    nx = rgb[..., 0] * 2.0 - 1.0
    ny = rgb[..., 1] * 2.0 - 1.0
    nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
    n = np.stack([nx, ny, nz], -1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    return n, rgb[..., 2], alpha > 0.5


def feature_lines(normal: np.ndarray, depth: np.ndarray, product: np.ndarray,
                  angle_deg: float = ANGLE_DEG, depth_jump: float = DEPTH_JUMP) -> np.ndarray:
    """Boolean map of inner feature lines (the product's outer outline is excluded)."""
    cos_t = float(np.cos(np.radians(angle_deg)))
    line = np.zeros(product.shape, bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        h, w = product.shape
        ys = slice(0, h - dy)
        yt = slice(dy, h)
        if dx >= 0:
            xs, xt = slice(0, w - dx), slice(dx, w)
        else:
            xs, xt = slice(-dx, w), slice(0, w + dx)
        both = product[ys, xs] & product[yt, xt]
        dot = (normal[ys, xs] * normal[yt, xt]).sum(-1)
        jump = np.abs(depth[ys, xs] - depth[yt, xt])
        hit = both & ((dot < cos_t) | (jump > depth_jump))
        line[ys, xs] |= hit
    if MIN_LINE_PX > 1 and line.any():
        from .layered_output import label_components
        labels, areas = label_components(line)
        small = [lab for lab, a in areas.items() if a < MIN_LINE_PX]
        if small:
            line &= ~np.isin(labels, small)
    return line


def load(path, w: int, h: int):
    """Feature lines (bool HxW) of a geometry pass file resampled to w×h; None if absent/broken."""
    try:
        if not path or not Path(path).is_file():
            return None
        from .png_io import read_png
        rgba = read_png(path)
        meta = {}
        side = Path(sidecar_path(path))
        if side.is_file():
            meta = json.loads(side.read_text(encoding="utf-8"))
        n, d, prod = decode(rgba, encoded_srgb=bool(meta.get("srgb", True)))
        lines = feature_lines(n, d, prod)
        if lines.shape != (h, w):
            ys = (np.arange(h) * lines.shape[0] // h)
            xs = (np.arange(w) * lines.shape[1] // w)
            lines = lines[ys][:, xs]
        return lines
    except Exception as exc:
        print(f"[Wondful] 结构线读取失败：{exc}")
        return None
