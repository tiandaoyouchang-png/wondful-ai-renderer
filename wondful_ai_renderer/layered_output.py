"""分层输出 (1.2.0): split a generated image into Subject / Shadow / Background layers.

Pure numpy + stdlib, no bpy, so everything here runs on a worker thread and in
plain-Python tests. Blender's bundled Python has numpy but no OpenCV / SciPy /
psd-tools, so the morphology, matting, inpainting and the PSD writer are local.

Pipeline (works for any product, not only cars):
1. ``fit_mask``: the Blender alpha mask of the product (same camera as the clay
   base) is fitted to the AI image with a similarity transform (scale + shift),
   scored by chamfer distance of the mask outline to the image's edges. Image
   models sometimes enlarge or nudge the product by a few percent.
2. ``refine_alpha`` (1.2.2): the fitted Blender mask's own anti-aliased edge is the
   matte; only a 2–3 px band is refined by a local colour model, used only where it
   agrees with the mask; detached far-away mask pieces are dropped; edge colours are
   unmixed against the local background and lean on the local product colour.
   (With a detail pass the mask is placed by the inner-structure fit, as 产品还原度.)
3. Plate (background without the product):
   - SHADOW mode: ``inpaint_pushpull`` fills the product region from the
     surrounding pixels (pyramid push-pull diffusion, no model call).
   - FULL mode: the caller supplies a clean plate from a masked image edit.
4. ``estimate_shadow``: luminance ratio image/plate, restricted to the ground band
   below the product's mid-line, as a black layer with alpha.
5. Layers are saved as PNGs plus a layered PSD (``write_psd``).
"""
from __future__ import annotations

import json
import struct
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from .png_io import read_png, write_png

MODE_SHADOW = "SHADOW"   # 主体 + 投影 (no extra model call)
MODE_FULL = "FULL"       # 完整（含干净背景）: + masked image edit for a clean plate

SCALE_PRIOR = 0.3
IDENTITY_FIT_GAIN = 0.85
WIDE_FIT_GAIN = 0.8   # a wide-range fit must beat the near fit by 20 %
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# --------------------------------------------------------------------------- basic ops

def box_sum(img: np.ndarray, r: int) -> np.ndarray:
    """Sum over a (2r+1)² window with zero padding (integral image)."""
    r = int(max(0, r))
    if r == 0:
        return img.astype(np.float32, copy=True)
    a = img.astype(np.float64)
    pad = [(r + 1, r), (r + 1, r)] + [(0, 0)] * (a.ndim - 2)
    s = np.pad(a, pad).cumsum(0).cumsum(1)
    k = 2 * r + 1
    out = s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
    return out.astype(np.float32)


def box_mean(img: np.ndarray, r: int) -> np.ndarray:
    h, w = img.shape[:2]
    n = box_sum(np.ones((h, w), np.float32), r)
    if img.ndim == 3:
        n = n[..., None]
    return box_sum(img, r) / n


def blur(img: np.ndarray, r: int, passes: int = 2) -> np.ndarray:
    out = img.astype(np.float32)
    for _ in range(passes):
        out = box_mean(out, r)
    return out


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation. Small radii: octagon via alternating cross/square steps; large: square."""
    m = np.asarray(mask, bool)
    r = int(max(0, r))
    if r == 0:
        return m.copy()
    if r > 12:
        return box_sum(m.astype(np.float32), r) > 0.5
    out = m.copy()
    for i in range(r):
        p = np.pad(out, 1)
        nxt = out | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        if i % 2:
            nxt |= p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
        out = nxt
    return out


def erode(mask: np.ndarray, r: int) -> np.ndarray:
    return ~dilate(~np.asarray(mask, bool), r)


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Bilinear resize (area-averaged first when shrinking by 2× or more)."""
    a = img.astype(np.float32)
    sh, sw = a.shape[:2]
    while sw >= 2 * w and sh >= 2 * h:
        a = a[: sh // 2 * 2, : sw // 2 * 2]
        a = 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])
        sh, sw = a.shape[:2]
    if (sw, sh) == (w, h):
        return a
    ys = np.clip((np.arange(h) + 0.5) * sh / h - 0.5, 0, sh - 1)
    xs = np.clip((np.arange(w) + 0.5) * sw / w - 0.5, 0, sw - 1)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, sh - 1); x1 = np.minimum(x0 + 1, sw - 1)
    fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
    if a.ndim == 3:
        fy = fy[..., None]; fx = fx[..., None]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def _lanczos_taps(n_src: int, n_dst: int, a: int = 3):
    """Lanczos-a taps for enlarging: (2a, n_dst) source indices (edge-clamped) and normalised weights."""
    x = (np.arange(n_dst, dtype=np.float64) + 0.5) * n_src / n_dst - 0.5
    base = np.floor(x).astype(int)
    offs = np.arange(-a + 1, a + 1)[:, None]
    idx = base[None, :] + offs
    d = x[None, :] - idx
    k = np.sinc(d) * np.sinc(d / a)
    k[np.abs(d) >= a] = 0.0
    k /= np.maximum(k.sum(0, keepdims=True), 1e-12)
    return np.clip(idx, 0, n_src - 1), k.astype(np.float32)


def upscale(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """1.2.4 (AI 感): Lanczos-3 when the image is enlarged (the model returns at most ~1672 px wide,
    the camera canvas is often 2560); shrinking keeps ``resize``. No sharpening is added: an unsharp
    mask would also amplify the AI's smeared texture. Values are clipped to the input range."""
    a = img.astype(np.float32)
    sh, sw = a.shape[:2]
    if (sw, sh) == (w, h):
        return a
    if w < sw or h < sh:
        return resize(a, w, h)
    lo, hi = float(a.min()), float(a.max())
    ix, kx = _lanczos_taps(sw, w)
    tmp = np.zeros((sh, w) + a.shape[2:], np.float32)
    for t in range(ix.shape[0]):
        wt = kx[t].reshape((1, w) + (1,) * (a.ndim - 2))
        tmp += a[:, ix[t]] * wt
    iy, ky = _lanczos_taps(sh, h)
    out = np.zeros((h, w) + a.shape[2:], np.float32)
    for t in range(iy.shape[0]):
        wt = ky[t].reshape((h, 1) + (1,) * (a.ndim - 2))
        out += tmp[iy[t]] * wt
    return np.clip(out, lo, hi)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., :3] @ LUMA


def bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


# --------------------------------------------------------------------------- 1. mask fit

def edge_map(gray: np.ndarray, frac: float = 0.12) -> np.ndarray:
    """Binary edges: Sobel magnitude above the (1-frac) quantile (no OpenCV)."""
    g = blur(gray, 1, 1)
    p = np.pad(g, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    mag = np.hypot(gx, gy)
    thr = max(float(np.quantile(mag, 1.0 - frac)), 0.04)
    return mag > thr


def truncated_distance(edges: np.ndarray, cap: int) -> np.ndarray:
    """Approximate distance (in px, capped) to the nearest edge pixel."""
    dist = np.full(edges.shape, float(cap), np.float32)
    cur = edges.copy()
    dist[cur] = 0
    for k in range(1, cap):
        cur = dilate(cur, 1)
        newly = cur & (dist == cap)
        dist[newly] = k
        if cur.all():
            break
    return dist


def _outline_points(mask: np.ndarray, limit: int = 5000) -> np.ndarray:
    m = mask > 0.5
    ys, xs = np.nonzero(m & ~erode(m, 1))
    pts = np.stack([xs, ys], 1).astype(np.float32)
    if len(pts) > limit:
        pts = pts[np.linspace(0, len(pts) - 1, limit).astype(int)]
    return pts


def _chamfer(dt: np.ndarray, pts: np.ndarray, centre, s, shifts: np.ndarray) -> np.ndarray:
    """Mean capped edge distance of the outline scaled by ``s`` (scalar or (sx, sy)) per shift."""
    h, w = dt.shape
    base = (pts - centre) * np.asarray(s, np.float32) + centre
    xs = np.rint(base[:, 0][None, :] + shifts[:, 0][:, None]).astype(int).clip(0, w - 1)
    ys = np.rint(base[:, 1][None, :] + shifts[:, 1][:, None]).astype(int).clip(0, h - 1)
    raw = dt[ys, xs].mean(1)
    # Scale prior: an outline shrunk onto interior detail (or blown up onto background
    # clutter) can score well on edges alone; real model drift is mostly a few percent.
    sx, sy = (s, s) if np.isscalar(s) else (float(s[0]), float(s[1]))
    return raw * (1.0 + SCALE_PRIOR * (abs(sx - 1.0) + abs(sy - 1.0)))


def warp_mask(mask: np.ndarray, s: float, dx: float, dy: float, centre) -> np.ndarray:
    """Scale ``mask`` by ``s`` (scalar or (sx, sy)) about ``centre`` then shift by (dx, dy)."""
    h, w = mask.shape
    cx, cy = centre
    sx, sy = (s, s) if np.isscalar(s) else (float(s[0]), float(s[1]))
    xs = (np.arange(w, dtype=np.float32) - cx - dx) / sx + cx
    ys = (np.arange(h, dtype=np.float32) - cy - dy) / sy + cy
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    fx = (xs - x0)[None, :]; fy = (ys - y0)[:, None]
    p = np.pad(mask.astype(np.float32), 1)

    def at(yy, xx):
        return p[(yy + 1).clip(0, h + 1)][:, (xx + 1).clip(0, w + 1)]
    top = at(y0, x0) * (1 - fx) + at(y0, x0 + 1) * fx
    bot = at(y0 + 1, x0) * (1 - fx) + at(y0 + 1, x0 + 1) * fx
    return np.clip(top * (1 - fy) + bot * fy, 0, 1)


def _fit_search(gray, mask, scale_range, shift_frac, aniso):
    """Coarse-to-fine similarity search; returns (cost, sx, sy, dx, dy, cost_identity, centre)."""
    h, w = mask.shape
    f = max(1, int(round(max(h, w) / 640)))
    sw, sh = max(1, w // f), max(1, h // f)
    g_small = resize(gray, sw, sh)
    m_small = resize(mask, sw, sh)
    dt_s = truncated_distance(edge_map(g_small), 12)
    pts = _outline_points(m_small, 3000)
    ys, xs = np.nonzero(m_small > 0.5)
    c_s = np.array([xs.mean(), ys.mean()], np.float32)
    rng = int(round(shift_frac * max(sw, sh)))
    grid = np.arange(-rng, rng + 1, 2, dtype=np.float32)
    shifts = np.stack(np.meshgrid(grid, grid), -1).reshape(-1, 2)
    zero = np.zeros((1, 2), np.float32)
    best = (float(_chamfer(dt_s, pts, c_s, 1.0, zero)[0]), 1.0, 0.0, 0.0)
    per_scale = []
    for s in np.arange(scale_range[0], scale_range[1] + 1e-6, 0.01):
        costs = _chamfer(dt_s, pts, c_s, float(s), shifts)
        i = int(np.argmin(costs))
        per_scale.append((float(costs[i]), float(s), float(shifts[i, 0]), float(shifts[i, 1])))
    # The 2-px grid can miss a narrow optimum: re-search the best few scales at 1 px.
    near = np.arange(-3, 4, 1, dtype=np.float32)
    local = np.stack(np.meshgrid(near, near), -1).reshape(-1, 2)
    for _cost, s, dx, dy in sorted(per_scale)[:4]:
        for s2 in (s - 0.005, s, s + 0.005):
            if not scale_range[0] - 1e-6 <= s2 <= scale_range[1] + 1e-6:
                continue
            cand = local + np.array([dx, dy], np.float32)
            costs = _chamfer(dt_s, pts, c_s, s2, cand)
            i = int(np.argmin(costs))
            if costs[i] < best[0] - 1e-6:
                best = (float(costs[i]), float(s2), float(cand[i, 0]), float(cand[i, 1]))
    # Full resolution: refine scale/shift, then a small independent width/height scale
    # (the image model redraws perspective a little differently from the clay render).
    dt = truncated_distance(edge_map(gray), 12)
    ptsf = _outline_points(mask, 6000)
    ys, xs = np.nonzero(mask > 0.5)
    c = np.array([xs.mean(), ys.mean()], np.float32)
    s0, dx0, dy0 = best[1], best[2] * w / sw, best[3] * h / sh
    identity = float(_chamfer(dt, ptsf, c, 1.0, zero)[0])
    fine = np.arange(-f - 2, f + 3, 1, dtype=np.float32)
    fshift = np.stack(np.meshgrid(fine, fine), -1).reshape(-1, 2) + np.array([dx0, dy0], np.float32)
    bestf = (identity, 1.0, 1.0, 0.0, 0.0)
    for s in np.arange(s0 - 0.01, s0 + 0.0101, 0.0025):
        costs = _chamfer(dt, ptsf, c, float(s), fshift)
        i = int(np.argmin(costs))
        if costs[i] < bestf[0] - 1e-6:
            bestf = (float(costs[i]), float(s), float(s), float(fshift[i, 0]), float(fshift[i, 1]))
    _, s1, _, dx1, dy1 = bestf
    loc = np.arange(-3, 4, 1, dtype=np.float32)
    ashift = np.stack(np.meshgrid(loc, loc), -1).reshape(-1, 2) + np.array([dx1, dy1], np.float32)
    steps = np.arange(-aniso, aniso + 1e-6, 0.005)
    for ax in steps:
        for ay in steps:
            sxy = (float(s1 + ax), float(s1 + ay))
            costs = _chamfer(dt, ptsf, c, sxy, ashift)
            i = int(np.argmin(costs))
            if costs[i] < bestf[0] - 1e-6:
                bestf = (float(costs[i]), sxy[0], sxy[1], float(ashift[i, 0]), float(ashift[i, 1]))
    return bestf + (identity, c)


def fit_mask(image: np.ndarray, mask: np.ndarray, scale_range=(0.88, 1.16), shift_frac=0.035) -> dict:
    """Fit ``mask`` (H×W float 0..1, same canvas as ``image``) to the image's edges.

    Returns dict(mask=fitted float mask, scale, scale_x, scale_y, dx, dy, chamfer_before,
    chamfer_after). Two searches: a near one (scale ±2 %) and a wide one; the wide result
    (e.g. a model that enlarged the product ~12 %) wins only when it explains the edges
    clearly better, because an outline shrunk onto interior detail can also score well.
    """
    if bbox(mask > 0.5) is None:
        raise ValueError("产品遮罩为空")
    gray = luminance(image)
    near = _fit_search(gray, mask, (0.98, 1.02), shift_frac, 0.01)
    wide = _fit_search(gray, mask, scale_range, shift_frac, 0.015)
    identity, c = near[5], near[6]
    pick, which = near, "near"
    if wide[0] < near[0] * WIDE_FIT_GAIN:
        pick, which = wide, "wide"
    cost, sx, sy, dx, dy = pick[:5]
    # Same camera, so the Blender mask is right unless the model visibly moved the product:
    # keep it as-is unless the fit explains the edges at least 15 % better.
    if cost > identity * IDENTITY_FIT_GAIN:
        cost, sx, sy, dx, dy, which = identity, 1.0, 1.0, 0.0, 0.0, "identity"
    is_identity = (sx, sy, dx, dy) == (1.0, 1.0, 0.0, 0.0)
    fitted = mask.astype(np.float32) if is_identity else warp_mask(mask, (sx, sy), dx, dy, c)
    return {"mask": fitted, "scale": round((sx + sy) / 2, 4), "scale_x": round(sx, 4), "scale_y": round(sy, 4),
            "dx": round(dx, 1), "dy": round(dy, 1), "search": which,
            "chamfer_before": round(identity, 3), "chamfer_after": round(cost, 3)}


# --------------------------------------------------------------------------- 2. alpha

def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter, gray guide."""
    mi = box_mean(guide, r); mp = box_mean(src, r)
    cov = box_mean(guide * src, r) - mi * mp
    var = box_mean(guide * guide, r) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    return box_mean(a, r) * guide + box_mean(b, r)


def _sample_offsets(radius: int, count: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, count)
    rad = np.sqrt(rng.uniform(0.04, 1.0, count)) * radius
    return np.stack([np.rint(np.sin(ang) * rad), np.rint(np.cos(ang) * rad)], 1).astype(int)


EDGE_BAND_MAX_PX = 3     # 1.2.2: refinement only this close to the Blender mask edge
AGREE_TOL = 0.35         # colour-model alpha is used only within this of the mask's alpha
STRAY_KEEP_FRAC = 0.10   # a detached mask piece this big (vs the main body) is product
STRAY_NEAR_FRAC = 0.02   # …or one this close (fraction of the product size) to the main body


def edge_band_px(long_edge: int) -> int:
    """Refinement band half-width: 2 px up to ~1750 px long edge, 3 px above."""
    return int(min(EDGE_BAND_MAX_PX, max(2, round(long_edge / 700.0))))


def label_components(mask: np.ndarray):
    """8-connected components of a boolean mask (run-length union-find, no SciPy).
    Returns (labels int32 H×W with 0 = background, areas dict label -> px)."""
    m = np.asarray(mask, bool)
    h, w = m.shape
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    runs_prev = []
    rows = []
    for y in range(h):
        row = m[y]
        if not row.any():
            rows.append([]); runs_prev = []
            continue
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]   # [s, e)
        cur = []
        j = 0
        for s0, e0 in zip(starts.tolist(), ends.tolist()):
            lab = 0
            while j < len(runs_prev) and runs_prev[j][1] < s0:   # prev run ends before s0-1
                j += 1
            k = j
            while k < len(runs_prev) and runs_prev[k][0] <= e0:   # overlaps [s0-1, e0]
                pl = find(runs_prev[k][2])
                if lab == 0:
                    lab = pl
                elif pl != lab:
                    parent[max(pl, lab)] = min(pl, lab)
                    lab = min(pl, lab)
                k += 1
            if lab == 0:
                parent.append(len(parent))
                lab = len(parent) - 1
            cur.append((s0, e0, lab))
        rows.append(cur)
        runs_prev = cur
    labels = np.zeros((h, w), np.int32)
    areas = {}
    for y, cur in enumerate(rows):
        for s0, e0, lab in cur:
            r = find(lab)
            labels[y, s0:e0] = r
            areas[r] = areas.get(r, 0) + (e0 - s0)
    return labels, areas


def product_pieces(mask: np.ndarray) -> np.ndarray:
    """The product's own silhouette pieces: the main body plus detached pieces that are big
    or sit right next to it. Small far-away pieces (a stray mesh in the product collection,
    a speck) are dropped — they caused a lone blurred blob in the subject layer."""
    hard = np.asarray(mask) > 0.5
    labels, areas = label_components(hard)
    if not areas:
        return hard
    main = max(areas, key=areas.get)
    keep = labels == main
    b = bbox(keep)
    size = max(b[1] - b[0] + 1, b[3] - b[2] + 1)
    near = dilate(keep, max(4, int(round(STRAY_NEAR_FRAC * size))))
    for lab, area in areas.items():
        if lab == main:
            continue
        piece = labels == lab
        if area >= STRAY_KEEP_FRAC * areas[main] or (piece & near).any():
            keep |= piece
    return keep


def refine_alpha(image: np.ndarray, fitted: np.ndarray, samples: int = 48) -> dict:
    """Subject alpha that trusts the (fitted) Blender product mask. Returns dict(alpha, core,
    band, bg_local, fg_local, dropped_px).

    1.2.2: the AI product sits within 0–2 px of Blender (产品还原度 readouts), so the mask's
    own anti-aliased edge is the matte. Only a 2–3 px band around the mask edge is refined:
    a shared-sampling colour model (closest foreground/background colour pair) is used only
    where it agrees with the mask (|Δα| ≤ AGREE_TOL) and the pair explains the pixel; the
    old 10–16 px band with a guided filter followed grass blades and mud into the matte.
    Detached far-away mask pieces are dropped (``product_pieces``).
    """
    h, w = fitted.shape
    L = max(h, w)
    prior = np.clip(np.asarray(fitted, np.float32), 0, 1)
    pieces = product_pieces(prior)
    dropped = int(((prior > 0.5) & ~pieces).sum())
    near_pieces = dilate(pieces, EDGE_BAND_MAX_PX + 1)
    prior = np.where(near_pieces, prior, 0.0).astype(np.float32)
    hard = prior > 0.5
    rb = edge_band_px(L)
    core = erode(hard, rb)
    outside = ~dilate(hard, rb)
    band = ~core & ~outside
    I = image[..., :3].astype(np.float32)
    a = np.where(core, 1.0, np.where(outside, 0.0, prior)).astype(np.float32)
    # Local foreground / background colours (normalised box means of core / outside pixels).
    rr = 3 * rb + 3
    fo = outside.astype(np.float32); fc = core.astype(np.float32)
    nb = box_sum(fo, rr); nf = box_sum(fc, rr)
    B_mean = box_sum(I * fo[..., None], rr) / np.maximum(nb, 1e-6)[..., None]
    F_mean = box_sum(I * fc[..., None], rr) / np.maximum(nf, 1e-6)[..., None]
    by, bx = np.nonzero(band)
    bg_local = np.where((nb > 0.5)[..., None], B_mean, I)
    fg_local = np.where((nf > 0.5)[..., None], F_mean, I)
    if len(by):
        offs = _sample_offsets(rr, samples)
        chunk = 40000                                     # bounds memory on 4K canvases
        for lo in range(0, len(by), chunk):
            cy, cx = by[lo:lo + chunk], bx[lo:lo + chunk]
            sy = np.clip(cy[:, None] + offs[None, :, 0], 0, h - 1)
            sx = np.clip(cx[:, None] + offs[None, :, 1], 0, w - 1)
            col = I[sy, sx]                                   # (N, K, 3)
            p = I[cy, cx]
            dist = np.sqrt(((col - p[:, None, :]) ** 2).sum(-1))
            big = 1e3
            dF = np.where(core[sy, sx], dist, big)
            dB = np.where(outside[sy, sx], dist, big)
            iF = dF.argmin(1); iB = dB.argmin(1)
            n = np.arange(len(cy))
            hasF = dF[n, iF] < big; hasB = dB[n, iB] < big
            F = col[n, iF]; B = col[n, iB]
            d = F - B
            dd = (d * d).sum(-1)
            a_col = np.clip(((p - B) * d).sum(-1) / np.maximum(dd, 1e-6), 0, 1)
            resid = np.sqrt(((p - (a_col[:, None] * F + (1 - a_col[:, None]) * B)) ** 2).sum(-1))
            conf = np.clip(np.sqrt(dd) / 0.12, 0, 1) * np.exp(-resid / 0.04)
            pr = prior[cy, cx]
            agree = np.abs(a_col - pr) <= AGREE_TOL
            conf = np.where(hasF & hasB & agree, conf, 0.0)
            a[cy, cx] = conf * a_col + (1 - conf) * pr
            bg_local[cy, cx] = np.where((hasB & (conf > 0.2))[:, None], B, bg_local[cy, cx])
    a = np.where(core, 1.0, np.where(outside | (prior < 0.01), 0.0, a)).astype(np.float32)   # never grows past the mask
    a[a < 0.03] = 0.0
    a[a > 0.97] = 1.0
    return {"alpha": a, "core": core, "band": band, "bg_local": bg_local, "fg_local": fg_local,
            "dropped_px": dropped, "band_px": rb}


def decontaminate(image: np.ndarray, alpha: np.ndarray, background: np.ndarray,
                  foreground: np.ndarray | None = None) -> np.ndarray:
    """Foreground colour F with I = a·F + (1-a)·B at soft-edge pixels (clamped).

    1.2.2: with ``foreground`` (local product colour), low-alpha pixels lean on it instead
    of the noisy unmixed estimate, so no grass / mud tint survives on the edge."""
    a = alpha[..., None]
    I = image[..., :3]
    F = np.clip((I - (1 - a) * background) / np.maximum(a, 1e-3), 0, 1)
    if foreground is not None:
        t = np.clip((alpha - 0.15) / 0.6, 0, 1)[..., None]     # a ≤ .15 → local product colour
        F = foreground * (1 - t) + F * t
    soft = (alpha > 0.0) & (alpha < 0.98)
    out = I.copy()
    out[soft] = np.clip(F[soft], 0, 1)
    return out


# --------------------------------------------------------------------------- 3. plate / inpaint

def inpaint_pushpull(image: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Fill ``hole`` from surrounding pixels with pyramid push-pull diffusion."""
    img = image[..., :3].astype(np.float32)
    wgt = (~np.asarray(hole, bool)).astype(np.float32)
    if wgt.sum() == 0:
        return img.copy()
    levels = []
    cur_i, cur_w = img * wgt[..., None], wgt
    while True:
        levels.append((cur_i, cur_w))
        h, w = cur_w.shape
        if (cur_w > 0.999).all() or min(h, w) <= 2:
            break
        ph, pw = h % 2, w % 2
        ci = np.pad(cur_i, ((0, ph), (0, pw), (0, 0)), mode="edge")
        cw = np.pad(cur_w, ((0, ph), (0, pw)), mode="edge")
        si = ci[0::2, 0::2] + ci[1::2, 0::2] + ci[0::2, 1::2] + ci[1::2, 1::2]
        sw = cw[0::2, 0::2] + cw[1::2, 0::2] + cw[0::2, 1::2] + cw[1::2, 1::2]
        nw = np.minimum(sw, 1.0)
        cur_i = np.where(sw[..., None] > 0, si / np.maximum(sw, 1e-6)[..., None], 0) * nw[..., None]
        cur_w = nw
    # Pull: coarse → fine; premultiplied colour, unknown filled from the coarser level.
    ci, cw = levels[-1]
    filled = ci / np.maximum(cw, 1e-6)[..., None]
    for li, lw in reversed(levels[:-1]):
        h, w = lw.shape
        up = resize(filled, w, h)
        colour = li / np.maximum(lw, 1e-6)[..., None]
        filled = np.where(lw[..., None] > 0, colour * lw[..., None] + up * (1 - lw[..., None]), up)
    out = img.copy()
    hole = np.asarray(hole, bool)
    out[hole] = filled[hole]
    return out


def product_region(mask: np.ndarray) -> dict:
    """Edit/fill region: product dilated + the ground band below it (where its shadow falls)."""
    h, w = mask.shape
    hard = mask > 0.5
    b = bbox(hard)
    if b is None:
        raise ValueError("产品遮罩为空")
    y0, y1, x0, x1 = b
    bh, bw = y1 - y0 + 1, x1 - x0 + 1
    size = max(bh, bw)
    region = dilate(hard, max(3, int(round(0.045 * size))))
    gy0 = y0 + int(0.55 * bh)
    gy1 = min(h, y1 + 1 + int(round(0.16 * bh)))
    gx0 = max(0, x0 - int(round(0.10 * bw)))
    gx1 = min(w, x1 + 1 + int(round(0.10 * bw)))
    ground = np.zeros((h, w), bool)
    ground[gy0:gy1, gx0:gx1] = True
    region |= ground
    return {"region": region, "mid_y": gy0, "bbox": b}


def edit_mask_rgba(region: np.ndarray) -> np.ndarray:
    """OpenAI-style edit mask: transparent = editable."""
    h, w = region.shape
    out = np.zeros((h, w, 4), np.uint8)
    out[..., 3] = np.where(region, 0, 255)
    return out


# --------------------------------------------------------------------------- 4. shadow

def estimate_shadow(image: np.ndarray, plate: np.ndarray, region: np.ndarray, mid_y: int,
                    alpha: np.ndarray) -> np.ndarray:
    """Shadow strength k (0..1): image = plate·(1-k) below the product mid-line."""
    h, w = alpha.shape
    L = max(h, w)
    # Low-pass both luminances first: ground texture (grass, sand ripples, snow crust) must not
    # read as shadow; shadows are smooth at this scale, contact shadows only soften slightly.
    r_lp = max(1, int(round(L / 256)))
    yi = blur(luminance(image[..., :3]), r_lp) + 1e-3
    yp = blur(luminance(plate[..., :3]), r_lp) + 1e-3
    ratio = np.clip(yi / yp, 0, 1)
    k = 1.0 - ratio
    k = k * np.clip((k - 0.04) / 0.06, 0, 1)             # ignore residual texture (soft threshold)
    rows = np.arange(h, dtype=np.float32)[:, None]
    ramp = max(2.0, 0.01 * L)
    ground = np.clip((rows - mid_y) / ramp, 0, 1)
    reg = blur(region.astype(np.float32), max(1, int(round(L / 320))))
    k = k * ground * reg
    k = blur(k, max(1, int(round(L / 900))), 1)
    k = k * (1.0 - alpha)                                   # no shadow under the subject itself
    # The soft subject outline darkens the ratio too: drop shadow on a thin band around the
    # subject except below its lowest point in each column (contact shadow stays attached).
    sub = alpha > 0.03
    any_col = sub.any(0)
    if not any_col.any():
        return np.clip(k, 0, 1).astype(np.float32)
    idx = np.arange(w)
    lowest = np.where(any_col, h - 1 - np.argmax(sub[::-1], 0), -1)
    left = np.maximum.accumulate(np.where(any_col, idx, -1))
    right = np.minimum.accumulate(np.where(any_col, idx, w)[::-1])[::-1]
    near = np.where(left < 0, right, np.where(right >= w, left,
                    np.where(idx - left <= right - idx, left, right)))
    lowest_eff = lowest[near]                    # uncovered columns borrow the nearest covered one
    edge = dilate(sub, max(2, int(round(0.005 * L)))) & ~sub
    below = rows >= (lowest_eff[None, :] - max(1, int(round(0.002 * L))))
    k = np.where((edge | sub) & ~below, 0.0, k)
    # Beside the product (columns it does not cover) only the ground near its contact line
    # can hold its shadow: start a little above the local ground line.
    bh = float(np.ptp(np.nonzero(sub.any(1))[0]) + 1)
    bw = float(np.ptp(np.nonzero(any_col)[0]) + 1)
    rr = max(1, int(round(0.15 * bw)))
    ground_line = np.lib.stride_tricks.sliding_window_view(
        np.pad(lowest_eff.astype(np.float32), rr, mode="edge"), 2 * rr + 1).max(1)
    start = np.where(any_col, float(mid_y), ground_line - 0.12 * bh)
    sr = max(1, int(round(0.04 * bw)))
    start = np.convolve(np.pad(start, sr, mode="edge"), np.ones(2 * sr + 1) / (2 * sr + 1), mode="valid")
    side = np.clip((rows - start[None, :]) / ramp, 0, 1)
    k = k * side
    return np.clip(k, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- 5. PSD writer

def _packbits_row(row: np.ndarray) -> bytes:
    """PackBits one row: runs ≥ 3 as repeat packets, the rest as literal packets."""
    n = len(row)
    if n == 0:
        return b""
    if (row == row[0]).all():
        out = bytearray()
        left = n
        while left:
            k = min(128, left)
            out += bytes(((257 - k) & 0xFF, int(row[0]))) if k > 1 else bytes((0, int(row[0])))
            left -= k
        return bytes(out)
    change = np.flatnonzero(np.diff(row.astype(np.int16)) != 0) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [n])))
    long_runs = np.flatnonzero(lengths >= 3)
    out = bytearray()
    raw = row.tobytes()

    def literal(a, b):
        while a < b:
            k = min(128, b - a)
            out.append(k - 1)
            out.extend(raw[a:a + k])
            a += k
    pos = 0
    for idx in long_runs:
        s, ln = int(starts[idx]), int(lengths[idx])
        literal(pos, s)
        v = raw[s]
        while ln:
            k = min(128, ln)
            if k == 1:
                out.append(0); out.append(v)
            else:
                out.append((257 - k) & 0xFF); out.append(v)
            ln -= k
        pos = int(starts[idx] + lengths[idx])
    literal(pos, n)
    return bytes(out)


def _rle_plane(plane: np.ndarray):
    rows = [_packbits_row(plane[y]) for y in range(plane.shape[0])]
    counts = b"".join(struct.pack(">H", len(r)) for r in rows)
    return counts, b"".join(rows)


def _pascal_name(name: str) -> bytes:
    raw = name.encode("ascii", "replace")[:255]
    data = bytes([len(raw)]) + raw
    return data + b"\x00" * ((4 - len(data) % 4) % 4)


def write_psd(path, layers, composite: np.ndarray) -> str:
    """Write an 8-bit RGB PSD. ``layers``: bottom-first list of dicts
    {name (ASCII), rgba (H×W×4 uint8), visible (bool, default True)}. ``composite``: H×W×3/4 uint8."""
    comp = np.asarray(composite, np.uint8)
    h, w = comp.shape[:2]
    out = bytearray()
    has_alpha = comp.shape[2] == 4
    out += b"8BPS" + struct.pack(">H6xHIIHH", 1, 4 if has_alpha else 3, h, w, 8, 3)
    out += struct.pack(">I", 0)          # colour mode data
    out += struct.pack(">I", 0)          # image resources
    records = bytearray()
    channel_data = bytearray()
    for layer in layers:
        rgba = np.asarray(layer["rgba"], np.uint8)
        if rgba.shape[:2] != (h, w):
            raise ValueError("layer size differs from canvas")
        chans = []
        for cid, plane in ((-1, rgba[..., 3]), (0, rgba[..., 0]), (1, rgba[..., 1]), (2, rgba[..., 2])):
            counts, data = _rle_plane(np.ascontiguousarray(plane))
            blob = struct.pack(">H", 1) + counts + data
            chans.append((cid, blob))
        records += struct.pack(">iiiiH", 0, 0, h, w, len(chans))
        for cid, blob in chans:
            records += struct.pack(">hI", cid, len(blob))
        flags = 0x08 | (0x02 if not layer.get("visible", True) else 0)
        records += b"8BIMnorm" + struct.pack(">BBBB", 255, 0, flags, 0)
        extra = struct.pack(">I", 0) + struct.pack(">I", 0) + _pascal_name(layer["name"])
        records += struct.pack(">I", len(extra)) + extra
        for _cid, blob in chans:
            channel_data += blob
    count = len(layers)
    layer_info = struct.pack(">h", -count if has_alpha else count) + records + channel_data
    if len(layer_info) % 2:
        layer_info += b"\x00"
    lmi = struct.pack(">I", len(layer_info)) + layer_info + struct.pack(">I", 0)
    out += struct.pack(">I", len(lmi)) + lmi
    planes = [comp[..., i] for i in range(comp.shape[2])]
    counts_all, data_all = b"", b""
    for plane in planes:
        c, d = _rle_plane(np.ascontiguousarray(plane))
        counts_all += c
        data_all += d
    out += struct.pack(">H", 1) + counts_all + data_all
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bytes(out))
    return str(target)


# --------------------------------------------------------------------------- driver

def _to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def layer_paths(image_path) -> dict:
    p = Path(image_path)
    stem = p.with_suffix("")
    return {"subject": f"{stem}_1_subject.png", "shadow": f"{stem}_2_shadow.png",
            "background": f"{stem}_3_background.png", "psd": f"{stem}_layers.psd"}


def load_mask(mask_path, size) -> np.ndarray:
    """Product mask from a Blender RGBA render (alpha channel), resampled to ``size`` (w, h)."""
    rgba = read_png(mask_path)
    m = rgba[..., 3].astype(np.float32) / 255.0
    if (m.shape[1], m.shape[0]) != tuple(size):
        m = np.clip(resize(m, int(size[0]), int(size[1])), 0, 1)
    return m


def _structure_fitted(I, raw_mask, detail_path):
    """1.2.2: place the mask with the inner-structure fit (same as 产品还原度) when the Blender
    detail pass exists; None when unavailable or it does not beat the identity."""
    if not detail_path or not Path(detail_path).is_file():
        return None
    from .identity_score import structure_fit
    from .detail_paste import warp_image
    h, w = raw_mask.shape
    D = read_png(detail_path).astype(np.float32) / 255.0
    if D.shape[-1] == 3:
        D = np.dstack([D, np.ones(D.shape[:2], np.float32)])
    if D.shape[:2] != (h, w):
        D = np.clip(resize(D, w, h), 0, 1)
    f = structure_fit(I, D[..., :3], D[..., 3], raw_mask)
    ys, xs = np.nonzero(raw_mask > 0.5)
    c = (float(xs.mean()), float(ys.mean()))
    moved = (f["scale"], f["dx"], f["dy"]) != (1.0, 0.0, 0.0)
    m = warp_image(raw_mask, (f["scale"], f["scale"]), f["dx"], f["dy"], c) if moved else raw_mask
    return {"mask": np.clip(m, 0, 1).astype(np.float32), "scale": f["scale"], "scale_x": f["scale"],
            "scale_y": f["scale"], "dx": f["dx"], "dy": f["dy"], "search": "structure", "ncc": f["ncc"]}


def build_layers(image_path, mask_path, *, mode=MODE_SHADOW, plate_fn=None, write=True, log=None,
                 work_dir="", detail_path="") -> dict:
    """Make the layers for one generated image. ``plate_fn(image_path, mask_rgba_path, region)``
    returns a path to a clean plate (FULL mode). Temporary files go to ``work_dir``
    (default: next to the image). Returns a report (``_arrays`` holds the float layers)."""
    t0 = time.time()
    timings = {}
    rgba = read_png(image_path)
    h, w = rgba.shape[:2]
    I = rgba[..., :3].astype(np.float32) / 255.0
    canvas_alpha = rgba[..., 3].astype(np.float32) / 255.0   # aspect-fit padding stays transparent
    raw_mask = load_mask(mask_path, (w, h))
    timings["load"] = time.time() - t0
    t = time.time()
    fit = None
    try:
        fit = _structure_fitted(I, raw_mask, detail_path)
    except Exception as exc:  # the outline fit below still works
        if log:
            log(f"结构对位失败，改用轮廓对位：{exc}")
    if fit is None:
        fit = fit_mask(I, raw_mask)
    timings["fit"] = time.time() - t
    t = time.time()
    matte = refine_alpha(I, fit["mask"])
    alpha = matte["alpha"] * canvas_alpha
    timings["alpha"] = time.time() - t
    t = time.time()
    reg = product_region(np.maximum(fit["mask"], (alpha > 0.5).astype(np.float32)))
    region = reg["region"] & (canvas_alpha > 0.5)
    report = {"mode": mode, "size": [w, h], "fit": {k: v for k, v in fit.items() if k != "mask"},
              "matte": {"band_px": matte["band_px"], "dropped_px": matte["dropped_px"]}}
    paths = layer_paths(image_path)
    plate = None
    if mode == MODE_FULL and plate_fn is not None:
        mask_file = str(Path(work_dir or Path(image_path).parent) / (Path(image_path).stem + "_editmask.png"))
        try:
            write_png(mask_file, edit_mask_rgba(region))
            plate_path = plate_fn(str(image_path), mask_file, region)
            prgba = read_png(plate_path)
            P = prgba[..., :3].astype(np.float32) / 255.0
            if P.shape[:2] != (h, w):
                P = resize(P, w, h)
            # The edit can shift colour/brightness inside the mask: carry the low-frequency
            # difference at the region border into the region so the plate joins seamlessly.
            correction = inpaint_pushpull(I - P, region)          # = I - P outside the region
            P = np.clip(np.where(region[..., None], P + correction, I), 0, 1)
            ed = blur(region.astype(np.float32), max(2, int(round(max(h, w) / 220))))[..., None]
            plate = P * ed + I * (1 - ed)
            report["plate"] = "codex_edit"
        except Exception as exc:  # fall back to the local plate, still produce layers
            report["plate_error"] = str(exc)[:300]
            if log:
                log(f"干净背景失败，改用本地估算：{exc}")
        finally:
            try:
                Path(mask_file).unlink()
            except OSError:
                pass
        timings["plate"] = time.time() - t
        t = time.time()
    if plate is None:
        plate = inpaint_pushpull(I, region)
        report.setdefault("plate", "local_inpaint")
        timings["inpaint"] = time.time() - t
        t = time.time()
    shadow = estimate_shadow(I, plate, region, reg["mid_y"], alpha)
    timings["shadow"] = time.time() - t
    t = time.time()
    fg = decontaminate(I, alpha, plate if report["plate"] == "codex_edit" else matte["bg_local"],
                       matte["fg_local"])
    # Recomposite: subject over shadow over background vs the original.
    a3 = alpha[..., None]
    rec = fg * a3 + (plate * (1 - shadow[..., None])) * (1 - a3)
    valid = canvas_alpha > 0.5
    err = np.abs(rec - I)[valid]
    report["recomposite_err"] = round(float(err.mean() * 255), 3)
    report["recomposite_err_region"] = round(float(np.abs(rec - I)[region & valid].mean() * 255), 3) if (region & valid).any() else 0.0
    report["subject_px"] = int((alpha > 0.5).sum())
    report["shadow_max"] = round(float(shadow.max()), 3)
    files = {}
    if write:
        subj = np.dstack([_to_u8(fg), _to_u8(alpha)])
        shad = np.zeros((h, w, 4), np.uint8)
        shad[..., 3] = _to_u8(shadow)
        files["subject"] = write_png(paths["subject"], subj)
        files["shadow"] = write_png(paths["shadow"], shad)
        layers = []
        if report["plate"] == "codex_edit":
            bg = np.dstack([_to_u8(plate), _to_u8(canvas_alpha)])
            files["background"] = write_png(paths["background"], bg)
            layers.append({"name": "Background", "rgba": bg})
        else:
            # 主体 + 投影: the original sits hidden at the bottom as a reference.
            layers.append({"name": "Original", "rgba": rgba, "visible": False})
        layers += [{"name": "Shadow", "rgba": shad}, {"name": "Subject", "rgba": subj}]
        if report["plate"] == "codex_edit":
            composite = _to_u8(rec)
        else:
            ca = np.maximum(alpha, shadow)
            col = np.where(ca[..., None] > 0, fg * a3 / np.maximum(ca, 1e-6)[..., None], 0)
            composite = np.dstack([_to_u8(col), _to_u8(ca)])
        files["psd"] = write_psd(paths["psd"], layers, composite)
    timings["write"] = time.time() - t
    report["files"] = files
    report["timings"] = {k: round(v, 2) for k, v in timings.items()}
    report["seconds"] = round(time.time() - t0, 2)
    report["_arrays"] = {"alpha": alpha, "shadow": shadow, "plate": plate, "fg": fg, "rec": rec}
    return report


# --------------------------------------------------------------------------- clean plate (FULL)

PLATE_PROMPT_VEHICLE = (
    "【背景补全】图 1 是一张成片。把画面中的汽车完全移除，包括车身、车轮、车底阴影和车身在地面上的倒影；"
    "按周围环境自然补全被遮挡的地面、远景和天空，让画面看起来像同一机位、同一光线下没有车时拍的空镜。"
    "车轮扬起的尘土/雪雾/水花如果脱离车身存在可以保留。不得改变 Mask 以外的任何像素、构图、色调和机位。只输出一张图。")
PLATE_PROMPT_GENERIC = (
    "【背景补全】图 1 是一张成片。把画面中的主体产品完全移除，包括它的全部部件、它投在地面或桌面上的阴影和倒影；"
    "按周围环境自然补全被遮挡的地面、墙面、远景和天空，让画面看起来像同一机位、同一光线下没有这件产品时拍的空镜。"
    "不要放入任何新物体。不得改变 Mask 以外的任何像素、构图、色调和机位。只输出一张图。")


def plate_prompt(vehicle: bool) -> str:
    return PLATE_PROMPT_VEHICLE if vehicle else PLATE_PROMPT_GENERIC


def make_plate_fn(*, model="", client_version="", log_dir="", vehicle=False, long_edge=2048, timeout=300):
    """Clean plate via a masked Codex image edit (codex_direct, worker-thread safe)."""
    def plate(image_path, mask_path, region):
        from . import codex_direct
        h, w = region.shape
        root = Path(log_dir or Path(image_path).parent)
        out = root / (Path(image_path).stem + "_plate.png")
        codex_direct.edit_image(
            prompt=plate_prompt(vehicle), edit_base_path=str(image_path), reference_paths=[],
            output_path=str(out), size=codex_direct.choose_edit_size(w, h, int(long_edge)),
            mask_path=str(mask_path), model=model, client_version=client_version, timeout=timeout,
            log_dir=str(root), tag="plate_" + Path(image_path).stem[-24:], exact_prompt=True)
        return str(out)
    return plate


# --------------------------------------------------------------------------- background runner

class LayerRunner:
    """One worker thread; jobs never raise into the caller. Results are polled on the main thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs = []
        self._results = []
        self._thread = None
        self.pending = 0

    def submit(self, image_path, mask_path, mode, plate_fn=None, work_dir="", detail_path=""):
        with self._lock:
            self._jobs.append((str(image_path), str(mask_path), mode, plate_fn, str(work_dir or ""),
                               str(detail_path or "")))
            self.pending += 1
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True, name="wondful-layers")
                self._thread.start()

    def _run(self):
        while True:
            with self._lock:
                if not self._jobs:
                    self._thread = None
                    return
                image_path, mask_path, mode, plate_fn, work_dir, detail_path = self._jobs.pop(0)
            try:
                rep = build_layers(image_path, mask_path, mode=mode, plate_fn=plate_fn, work_dir=work_dir,
                                   detail_path=detail_path)
                rep.pop("_arrays", None)
                rep["ok"] = True
                try:
                    Path(work_dir or Path(image_path).parent, Path(image_path).stem + "_layers.json").write_text(
                        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
                except OSError:
                    pass
            except Exception as exc:
                print(f"[Wondful] 分层输出失败：{exc}\n{traceback.format_exc()}")
                rep = {"ok": False, "error": str(exc)[:300]}
            rep["image"] = image_path
            with self._lock:
                self._results.append(rep)
                self.pending -= 1

    def drain(self):
        with self._lock:
            out, self._results = self._results, []
            return out


RUNNER = LayerRunner()


def summarize(results) -> str:
    """Panel note for finished layer jobs."""
    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    parts = []
    if ok:
        last = ok[-1]
        kind = "主体 + 投影 + 背景" if last.get("plate") == "codex_edit" else "主体 + 投影"
        note = f"分层输出完成 {len(ok)} 张（{kind}，PSD + PNG），用时 {last.get('seconds', 0):.0f}s"
        if last.get("plate_error"):
            note += "；干净背景失败，已用本地估算"
        parts.append(note)
    if bad:
        parts.append("分层输出失败：" + bad[-1].get("error", "")[:80])
    return "；".join(parts)
