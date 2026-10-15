"""草图模式: a rough product proxy from three-view images (visual hull), numpy + stdlib only.

No bpy here, so everything is unit-testable and safe on worker threads.

Axes (Blender convention, product forward = -Y):
  X = 宽 (left/right as seen in the front view), Y = 长 (front→back), Z = 高.
  front view : seen from -Y; image right = +X, image up = +Z
  side view  : default "front of product on the image LEFT" (a right-side view seen from +X)
  top view   : default "front of product at the image BOTTOM" (seen from +Z, image right = +X)

Pipeline: image → silhouette (alpha, or background colour + threshold, holes filled,
largest blob kept) → crop to bbox → carve a voxel grid (front ∩ side ∩ top) →
light blur → naive surface nets (closed quad mesh, outward winding) → a few
Laplacian passes.  The result is a visual hull: outline and proportions are about
right, concavities that no view can see (cups, the gap under a table top seen from
nowhere) stay filled; the image model is told to correct those.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

VIEWS = ("front", "side", "top")
VIEW_LABELS = {"front": "正视图", "side": "侧视图", "top": "俯视图"}


# --------------------------------------------------------------------------- image io

def load_rgba(path, max_side: int = 1024) -> np.ndarray:
    """Float RGBA (h, w, 4) in 0..1, top row first, for a PNG file (png_io).  Inside
    Blender the add-on decodes any format (JPG too) through bpy: see
    sketch_proxy.image_to_rgba, which hands the pixels to to_rgba_float."""
    p = Path(path)
    if not p.read_bytes()[:8].startswith(b"\x89PNG"):
        raise ValueError(f"{p.name} 不是 PNG：请在 Blender 中读取（sketch_proxy.image_to_rgba）")
    from .png_io import read_png
    return to_rgba_float(read_png(str(p)), max_side)


def to_rgba_float(arr, max_side: int = 1024) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    else:
        a = a.astype(np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.shape[2] == 1:
        a = np.concatenate([a, a, a, np.ones_like(a)], axis=2)
    elif a.shape[2] == 2:
        a = np.concatenate([a[:, :, :1]] * 3 + [a[:, :, 1:2]], axis=2)
    elif a.shape[2] == 3:
        a = np.concatenate([a, np.ones_like(a[:, :, :1])], axis=2)
    a = a[:, :, :4]
    h, w = a.shape[:2]
    step = int(math.ceil(max(h, w) / float(max_side))) if max_side else 1
    if step > 1:  # box downsample by an integer factor (fast, no resampler needed)
        hh, ww = h // step * step, w // step * step
        a = a[:hh, :ww].reshape(hh // step, step, ww // step, step, 4).mean(axis=(1, 3))
    return np.ascontiguousarray(a, dtype=np.float32)


# --------------------------------------------------------------------------- connected components

def label_components(mask: np.ndarray):
    """4-connected labels via row runs + union-find.  Returns (labels int32, sizes list),
    label 0 = background, components numbered 1..n."""
    mask = np.asarray(mask, bool)
    h, w = mask.shape
    labels = np.zeros((h, w), np.int32)
    if not mask.any():
        return labels, [0]
    padded = np.zeros((h, w + 2), np.int8)
    padded[:, 1:-1] = mask
    diff = np.diff(padded, axis=1)
    starts_r, starts_c = np.nonzero(diff == 1)
    ends_r, ends_c = np.nonzero(diff == -1)  # exclusive end column
    n = len(starts_r)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    row_first = np.searchsorted(starts_r, np.arange(h + 1))
    for r in range(1, h):
        a0, a1 = row_first[r - 1], row_first[r]
        b0, b1 = row_first[r], row_first[r + 1]
        i, j = a0, b0
        while i < a1 and j < b1:
            if starts_c[i] < ends_c[j] and starts_c[j] < ends_c[i]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
            if ends_c[i] < ends_c[j]:
                i += 1
            else:
                j += 1
    roots = np.array([find(i) for i in range(n)])
    uniq, run_label = np.unique(roots, return_inverse=True)
    run_label = run_label + 1
    sizes = [0] * (len(uniq) + 1)
    for k in range(n):
        labels[starts_r[k], starts_c[k]:ends_c[k]] = run_label[k]
        sizes[run_label[k]] += int(ends_c[k] - starts_c[k])
    return labels, sizes


def fill_holes(mask: np.ndarray, max_hole: float = 1.0) -> np.ndarray:
    """Fill background regions not connected (4-conn) to the image border.

    ``max_hole`` = largest hole filled, as a fraction of the filled object area
    (1.0 = every hole).  Big enclosed openings (the gap in a chair back, a handle)
    are real shape and stay open with a smaller value."""
    mask = np.asarray(mask, bool)
    bg = ~mask
    labels, sizes = label_components(bg)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    border = border[border > 0]
    outside = np.isin(labels, border)
    filled = ~outside
    if max_hole >= 1.0:
        return filled
    limit = max_hole * float(filled.sum())
    big = [k for k in range(1, len(sizes)) if sizes[k] > limit and k not in set(border.tolist())]
    if big:
        filled &= ~np.isin(labels, big)
    return filled


def largest_component(mask: np.ndarray, keep_ratio: float = 0.0) -> np.ndarray:
    labels, sizes = label_components(mask)
    if len(sizes) <= 1:
        return np.zeros_like(mask, bool)
    order = np.argsort(sizes[1:])[::-1] + 1
    best = sizes[order[0]]
    keep = [int(k) for k in order if sizes[k] >= max(1, keep_ratio * best)] if keep_ratio > 0 else [int(order[0])]
    return np.isin(labels, keep)


def dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    m = np.asarray(mask, bool).copy()
    for _ in range(max(0, int(r))):
        n = m.copy()
        n[1:] |= m[:-1]; n[:-1] |= m[1:]; n[:, 1:] |= m[:, :-1]; n[:, :-1] |= m[:, 1:]
        m = n
    return m


def erode(mask: np.ndarray, r: int = 1) -> np.ndarray:
    return ~dilate(~np.asarray(mask, bool), r)


# --------------------------------------------------------------------------- silhouette

def background_colour(rgb: np.ndarray, border: int = 0) -> np.ndarray:
    h, w = rgb.shape[:2]
    b = border or max(2, min(h, w) // 50)
    ring = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                           rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    return np.median(ring, axis=0)


def silhouette_mask(rgba: np.ndarray, threshold: float = 0.0, keep_all: bool = False):
    """Boolean product mask + how it was found ("alpha" or "background").

    Transparent PNG → alpha > 0.5.  Otherwise the background colour is the median of
    the border ring; pixels farther than ``threshold`` (auto: 0.10 RGB distance, a
    bit more on noisy JPG backgrounds) are product.  Line drawings work too: closed
    outlines are filled by the hole filling.  ``keep_all`` keeps every sizeable blob
    (used by the sheet splitter), otherwise only the largest one.
    """
    a = np.asarray(rgba, np.float32)
    alpha = a[:, :, 3]
    if float(alpha.min()) < 0.5 and 0.002 < float((alpha > 0.5).mean()) < 0.98:
        mask, how = alpha > 0.5, "alpha"
    else:
        rgb = a[:, :, :3]
        bg = background_colour(rgb)
        dist = np.sqrt(((rgb - bg) ** 2).sum(axis=2))
        if threshold <= 0:
            h, w = dist.shape
            b = max(2, min(h, w) // 50)
            ring = np.concatenate([dist[:b].ravel(), dist[-b:].ravel(), dist[:, :b].ravel(), dist[:, -b:].ravel()])
            threshold = max(0.10, float(np.percentile(ring, 99)) * 1.5 + 0.02)
        mask, how = dist > threshold, "background"
        # close 1-px gaps in anti-aliased outlines before filling
        mask = erode(dilate(mask, 1), 1) | mask
    raw = int(mask.sum())
    full = fill_holes(mask)
    if how == "background" and raw < 0.35 * max(1, int(full.sum())):
        mask = full          # line drawing: the outline encloses the product, fill it all
    else:
        mask = fill_holes(mask, max_hole=0.02)  # specks / small gaps; big openings stay open
    if keep_all:
        mask = largest_component(mask, keep_ratio=0.002) if mask.any() else mask
    else:
        mask = largest_component(mask)
    return mask, how


def crop_to_mask(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError("没有找到产品轮廓（图片是否是白底 / 透明底？）")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return mask[y0:y1, x0:x1], (int(x0), int(y0), int(x1), int(y1))


def view_silhouette(rgba, threshold: float = 0.0):
    """Cropped silhouette for one view: (mask, info)."""
    mask, how = silhouette_mask(rgba, threshold)
    crop, box = crop_to_mask(mask)
    return crop, {"how": how, "bbox": box, "size": [int(crop.shape[1]), int(crop.shape[0])],
                  "fill": round(float(crop.mean()), 3)}


# --------------------------------------------------------------------------- three-view sheet

def split_sheet(rgba, threshold: float = 0.0, gap: int = 0):
    """Split one three-view sheet into front / side / top crops.

    The three biggest silhouette blobs (blobs closer than ``gap`` px merged, so a
    wheel or a detached handle stays with its view) are matched to views by shared
    dimensions: front.height ≈ side.height, front.width ≈ top.width, side.width ≈
    top.height.  Returns (dict view → (mask_crop, rgba_crop)), info) or raises
    ValueError when the sheet does not look like 2–3 separated views.  With only two
    blobs the top view is left out.
    """
    a = np.asarray(rgba, np.float32)
    h, w = a.shape[:2]
    full, how = silhouette_mask(a, threshold, keep_all=True)
    g = gap or max(3, int(round(min(h, w) * 0.012)))
    grouped, _ = label_components(dilate(full, g))
    _l, sizes = label_components(full)
    groups = []
    for lab in range(1, int(grouped.max()) + 1):
        sel = (grouped == lab) & full
        cnt = int(sel.sum())
        if cnt == 0:
            continue
        ys, xs = np.nonzero(sel)
        groups.append({"label": lab, "area": cnt, "box": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)})
    groups.sort(key=lambda g_: -g_["area"])
    if len(groups) < 2:
        raise ValueError("三视图拼图里只找到 1 个轮廓：请把正视 / 侧视 / 俯视分开排列，中间留白，或分别导入三张图")
    big = [g_ for g_ in groups[:3] if g_["area"] >= 0.08 * groups[0]["area"]]
    if len(big) < 2:
        raise ValueError("三视图拼图里只有一个主要轮廓，其他都太小：请分别导入三张图")
    extra = groups[len(big):]
    if extra and extra[0]["area"] > 0.25 * big[-1]["area"]:
        raise ValueError("三视图拼图里有超过 3 个大小相近的轮廓（文字、尺寸标注或多余视图？），无法可靠拆分：请分别导入三张图")
    dims = [(g_["box"][2] - g_["box"][0], g_["box"][3] - g_["box"][1]) for g_ in big]

    def rel(p, q):
        return abs(p - q) / float(max(p, q, 1))

    import itertools
    names = ("front", "side", "top")[:len(big)]
    scored = []
    for perm in itertools.permutations(range(len(big))):
        d = {names[k]: dims[perm[k]] for k in range(len(big))}
        cost = rel(d["front"][1], d["side"][1])
        rotated = False
        if "top" in d:
            straight = rel(d["front"][0], d["top"][0]) + rel(d["side"][0], d["top"][1])
            turned = rel(d["front"][0], d["top"][1]) + rel(d["side"][0], d["top"][0])
            rotated = turned < straight
            cost += min(straight, turned)
        fx0, fy0 = big[perm[0]]["box"][:2]
        scored.append((cost, fx0 + fy0, perm, rotated))
    # Dimensions alone cannot tell front from side when the top view fits both readings
    # (e.g. a wide product vs a long one with a sideways top view): among equally good
    # readings the front view is the first one in reading order (top-left-most).
    best_cost = min(s_[0] for s_ in scored)
    cost, _pos, perm, rotated = min((s_ for s_ in scored if s_[0] <= best_cost + 0.05), key=lambda s_: (s_[1], s_[0]))
    if len(big) == 2:
        # front vs side is ambiguous on heights alone: the left blob is the front view
        boxes = [big[i]["box"] for i in perm]
        if boxes[0][0] > boxes[1][0]:
            perm = perm[::-1]
    limit = 0.35 if len(big) == 3 else 0.2
    if cost > limit:
        raise ValueError(f"三个轮廓的尺寸对不上（误差 {cost:.2f}），无法判断哪张是正视 / 侧视 / 俯视：请分别导入三张图")
    out = {}
    for k, name in enumerate(names):
        g_ = big[perm[k]]
        x0, y0, x1, y1 = g_["box"]
        m = (grouped[y0:y1, x0:x1] == g_["label"]) & full[y0:y1, x0:x1]
        out[name] = (m, a[y0:y1, x0:x1].copy())
    info = {"how": how, "views": len(big), "cost": round(cost, 3), "top_rotated": bool(rotated),
            "boxes": {names[k]: big[perm[k]]["box"] for k in range(len(big))}}
    return out, info


# --------------------------------------------------------------------------- orientation / size

def orient_side(mask, front_on: str = "LEFT"):
    """Side crop → array indexed [row=up..down, col=front..back]."""
    return mask if (front_on or "LEFT").upper() == "LEFT" else mask[:, ::-1]


def orient_top(mask, front_on: str = "BOTTOM"):
    """Top crop → array indexed [row=back..front, col=-X..+X] (front at the bottom)."""
    f = (front_on or "BOTTOM").upper()
    if f == "TOP":
        return mask[::-1, ::-1]
    if f == "LEFT":   # product nose points left in the image: rotate 90° counter-clockwise
        return np.rot90(mask, 1)
    if f == "RIGHT":
        return np.rot90(mask, -1)
    return mask


def auto_top_front(top, expected_w_over_l: float, top_front: str = "BOTTOM", side_front: str = "LEFT"):
    """Top views are often drawn lying sideways (a car pointing left, like its side view).
    When the top crop's aspect fits the front/side proportions only after a 90° turn,
    switch to the matching sideways orientation.  Returns (top_front, turned)."""
    f = (top_front or "BOTTOM").upper()
    h, w = np.asarray(top).shape
    upright = (w / float(max(h, 1))) if f in ("BOTTOM", "TOP") else (h / float(max(w, 1)))
    err_up = abs(math.log(max(upright, 1e-6) / max(expected_w_over_l, 1e-6)))
    err_turn = abs(math.log(max(1.0 / max(upright, 1e-6), 1e-6) / max(expected_w_over_l, 1e-6)))
    if err_turn + 0.15 < err_up:
        if f in ("BOTTOM", "TOP"):
            return ("LEFT" if (side_front or "LEFT").upper() == "LEFT" else "RIGHT"), True
        return "BOTTOM", True
    return f, False


def solve_dimensions(front, side, top=None, mode="AUTO", known_axis="LENGTH", known_value=1.0,
                     length=1.0, width=1.0, height=1.0):
    """Real size (width X, length Y, height Z) in metres.

    AUTO: width/height from the front view aspect, length/height from the side view
    aspect, scaled so ``known_axis`` (LENGTH/WIDTH/HEIGHT) equals ``known_value``.
    MANUAL: the three values as given."""
    if (mode or "AUTO").upper() == "MANUAL":
        return float(width), float(length), float(height)
    fh, fw = front.shape
    sh, sw = side.shape
    W = fw / float(fh)        # in units of height
    L = sw / float(sh)
    H = 1.0
    if top is not None and top.shape[0] > 0:
        # keep front/side, top only cross-checks (reported by caller)
        pass
    ref = {"LENGTH": L, "WIDTH": W, "HEIGHT": H}.get((known_axis or "LENGTH").upper(), L)
    s = float(known_value) / max(ref, 1e-9)
    return W * s, L * s, H * s


# --------------------------------------------------------------------------- carving

def resample_mask(mask, rows: int, cols: int) -> np.ndarray:
    """Area-average resample of a boolean crop to (rows, cols) → float coverage 0..1."""
    m = np.asarray(mask, np.float64)
    h, w = m.shape
    ii = np.zeros((h + 1, w + 1))
    ii[1:, 1:] = m.cumsum(0).cumsum(1)
    ry = np.linspace(0, h, rows + 1)
    rx = np.linspace(0, w, cols + 1)

    def sample(y, x):  # bilinear lookup into the integral image (fractional edges)
        y0 = np.clip(np.floor(y).astype(int), 0, h - 1)
        x0 = np.clip(np.floor(x).astype(int), 0, w - 1)
        fy = (y - y0)[:, None]
        fx = (x - x0)[None, :]
        a = ii[y0][:, x0]; b = ii[y0][:, x0 + 1]; c = ii[y0 + 1][:, x0]; d = ii[y0 + 1][:, x0 + 1]
        return a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + c * fy * (1 - fx) + d * fy * fx

    S = sample(ry, rx)
    area = np.outer(np.diff(ry), np.diff(rx))
    cov = (S[1:, 1:] - S[:-1, 1:] - S[1:, :-1] + S[:-1, :-1]) / np.maximum(area, 1e-12)
    return np.clip(cov, 0.0, 1.0)


def grid_shape(dims, resolution: int = 160):
    W, L, H = dims
    m = max(W, L, H)
    return tuple(max(8, int(round(resolution * d / m))) for d in (W, L, H))


def carve(front, side, top=None, dims=(1.0, 1.0, 1.0), resolution: int = 160):
    """Visual hull occupancy, shape (nx, ny, nz), float 0..1 (soft at the silhouette edges).

    front: [z down..up rows, x cols]; side: oriented [z rows, y front..back cols];
    top: oriented [y back..front rows, x cols] or None."""
    nx, ny, nz = grid_shape(dims, resolution)
    F = resample_mask(front, nz, nx)[::-1].T          # (nx, nz), z up
    S = resample_mask(side, nz, ny)[::-1].T           # (ny, nz), y = front(-Y)→back(+Y)
    occ = np.minimum(F[:, None, :], S[None, :, :])
    if top is not None:
        T = resample_mask(top, ny, nx)[::-1].T        # (nx, ny): rows back..front → flip so y front→back
        occ = np.minimum(occ, T[:, :, None])
    return occ.astype(np.float32)


def blur3(field: np.ndarray, passes: int = 1) -> np.ndarray:
    f = field.astype(np.float32)
    for _ in range(max(0, passes)):
        for ax in range(3):
            p = np.pad(f, [(1, 1) if a == ax else (0, 0) for a in range(3)], mode="edge")
            lo = [slice(None)] * 3; mid = [slice(None)] * 3; hi = [slice(None)] * 3
            lo[ax] = slice(0, -2); mid[ax] = slice(1, -1); hi[ax] = slice(2, None)
            f = 0.25 * p[tuple(lo)] + 0.5 * p[tuple(mid)] + 0.25 * p[tuple(hi)]
    return f


# --------------------------------------------------------------------------- surface nets

def surface_nets(field: np.ndarray, iso: float = 0.5):
    """Naive surface nets on a scalar field (inside = value > iso).

    Returns (verts (n,3) float in grid-point units, quads (m,4) int) — a closed,
    outward-wound quad mesh (the field is zero-padded first).  Grid point (i,j,k) of
    the input sits at coordinate (i, j, k)."""
    f = np.pad(np.asarray(field, np.float32), 1, mode="constant", constant_values=0.0)
    inside = f > iso
    N = f.shape
    cells = tuple(n - 1 for n in N)
    acc = np.zeros(cells + (3,), np.float64)
    cnt = np.zeros(cells, np.int32)
    edges = []
    for a in range(3):
        sl0 = [slice(None)] * 3; sl1 = [slice(None)] * 3
        sl0[a] = slice(0, -1); sl1[a] = slice(1, None)
        f0, f1 = f[tuple(sl0)], f[tuple(sl1)]
        cross = inside[tuple(sl0)] != inside[tuple(sl1)]
        t = np.where(cross, (iso - f0) / np.where(f1 - f0 == 0, 1e-9, f1 - f0), 0.0)
        edges.append((cross, inside[tuple(sl0)]))
        idx = np.nonzero(cross)
        if len(idx[0]) == 0:
            continue
        pos = np.stack(idx, axis=1).astype(np.float64)
        pos[:, a] += t[idx]
        b, c = (a + 1) % 3, (a + 2) % 3
        for db in (0, 1):
            for dc in (0, 1):
                cell = [idx[0].copy(), idx[1].copy(), idx[2].copy()]
                cell[b] = cell[b] - db
                cell[c] = cell[c] - dc
                ok = (cell[b] >= 0) & (cell[b] < cells[b]) & (cell[c] >= 0) & (cell[c] < cells[c])
                ci = tuple(x[ok] for x in cell)
                np.add.at(acc, ci, pos[ok])
                np.add.at(cnt, ci, 1)
    active = cnt > 0
    vid = -np.ones(cells, np.int64)
    vid[active] = np.arange(int(active.sum()))
    verts = acc[active] / cnt[active][:, None] - 1.0  # undo padding
    quads = []
    for a, (cross, low_in) in enumerate(edges):
        idx = np.nonzero(cross)
        if len(idx[0]) == 0:
            continue
        b, c = (a + 1) % 3, (a + 2) % 3
        corners = []
        for db, dc in ((1, 1), (0, 1), (0, 0), (1, 0)):
            cell = [idx[0].copy(), idx[1].copy(), idx[2].copy()]
            cell[b] = cell[b] - db
            cell[c] = cell[c] - dc
            corners.append(vid[tuple(cell)])
        q = np.stack(corners, axis=1)
        flip = ~low_in[idx]
        q[flip] = q[flip][:, ::-1]
        quads.append(q)
    quads = np.concatenate(quads) if quads else np.zeros((0, 4), np.int64)
    return verts.astype(np.float64), quads.astype(np.int64)


def laplacian_smooth(verts, quads, iterations: int = 4, lam: float = 0.5, mu: float = -0.53):
    """Taubin smoothing (shrink-free) over quad edges."""
    v = np.asarray(verts, np.float64).copy()
    if len(quads) == 0 or iterations <= 0:
        return v
    e = np.concatenate([quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    deg = np.bincount(e[:, 0], minlength=len(v)).astype(np.float64)
    deg[deg == 0] = 1
    for _ in range(iterations):
        for factor in (lam, mu):
            s = np.zeros_like(v)
            np.add.at(s, e[:, 0], v[e[:, 1]])
            v = v + factor * (s / deg[:, None] - v)
    return v


def mesh_volume(verts, quads) -> float:
    v = np.asarray(verts, np.float64)
    q = np.asarray(quads)
    if len(q) == 0:
        return 0.0
    tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    a, b, c = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


# --------------------------------------------------------------------------- top level

def build_hull(front, side, top=None, *, dims=None, size_mode="AUTO", known_axis="LENGTH", known_value=1.0,
               length=1.0, width=1.0, height=1.0, side_front="LEFT", top_front="BOTTOM",
               resolution: int = 160, smooth: int = 4):
    """Silhouette crops → (verts in metres, quads, info).

    Mesh is centred on X/Y, sitting on z = 0, product front towards -Y."""
    side_o = orient_side(side, side_front)
    if dims is None:
        dims = solve_dimensions(front, side_o, None, size_mode, known_axis, known_value, length, width, height)
    W, L, H = (max(1e-4, float(d)) for d in dims)
    top_turned = False
    if top is not None:
        top_front, top_turned = auto_top_front(top, W / L, top_front, side_front)
    top_o = orient_top(top, top_front) if top is not None else None
    occ = carve(front, side_o, top_o, (W, L, H), resolution)
    field = blur3(occ, 1)
    verts, quads = surface_nets(field, 0.5)
    if len(quads) == 0:
        raise ValueError("三视图没有交集：请检查各视图的朝向（侧视图车头 / 正面朝哪边、俯视图正面朝哪边）")
    verts = laplacian_smooth(verts, quads, smooth)
    nx, ny, nz = occ.shape
    # grid point i sits at the centre of voxel i: x = (i + 0.5) * W / nx
    scale = np.array([W / nx, L / ny, H / nz])
    verts = (verts + 0.5) * scale
    verts[:, 0] -= W / 2.0
    verts[:, 1] -= L / 2.0
    verts[:, 2] -= verts[:, 2].min()
    info = {"dims": [round(W, 4), round(L, 4), round(H, 4)], "grid": [nx, ny, nz],
            "occupied": round(float((occ > 0.5).mean()), 4), "verts": int(len(verts)), "faces": int(len(quads)),
            "volume_m3": round(mesh_volume(verts, quads), 6)}
    info["top_front"] = top_front if top is not None else ""
    info["top_rotated"] = bool(top_turned)
    if top_o is not None:
        tw = top_o.shape[1] / float(max(top_o.shape[0], 1))
        info["top_aspect_check"] = round((W / L) / max(tw, 1e-6), 3)  # 1.0 = top view agrees with front+side
    return verts, quads, info


def mask_to_png(mask, path):
    """Debug helper: write a boolean mask as an 8-bit gray PNG (uses png_io)."""
    from .png_io import write_png
    m = (np.asarray(mask, bool) * 255).astype(np.uint8)
    return write_png(path, np.dstack([m, m, m, np.full_like(m, 255)]))
