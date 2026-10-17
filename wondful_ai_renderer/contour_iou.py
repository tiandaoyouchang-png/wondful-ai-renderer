"""1.2.3 固定坐标验收 — fixed-coordinate silhouette IoU, the official pass/fail of one AI result.

Ground truth is the Blender product mask at the camera canvas, *at fixed coordinates*: the AI
image is only resampled to the canvas size (same aspect), never scaled, shifted or fitted.

The AI-side product region comes from a deterministic contour extraction (no learned
segmentation, pure numpy, no bpy) with rules fixed in advance:

1. Contours. Every boundary of the Blender mask (each product piece and each enclosed hole) is
   traced. Each node may move along its normal by at most ``r`` = 3 % of the product diagonal
   (6–48 px at the scoring size), and never past the medial line of the region next to it.
2. Snapping. Dynamic programming along each contour picks, per node, the displacement with the
   strongest colour edge *across the normal* (Lab Sobel, orientation-consistent), with a small
   pull towards zero and ±1 px smoothness between neighbours. The snapped contours are filled.
3. Holes = background. Enclosed holes of the Blender mask (handle gaps, the space between chair
   legs) stay background inside their snapped boundary; their eroded core is always background
   (codified holefix: a learned matte fills them, a product never does). Secondary check: a
   hole whose core looks like the product rather than the scene counts as filled in the lower
   IoU bound, so a closed handle gap ends up 待复核 / 未通过, never silently 通过.
4. Thin rods = product. Mask parts thinner than ``2 r`` (antennas, legs, rods) are snapped like
   every other contour and never eroded away (a learned matte tends to drop them). As a
   secondary check, rod pixels that look like the nearby background (closest background sample
   nearer than ``ROD_TAU``) count as missing in the lower IoU bound.
5. Verdict. IoU against the Blender mask. Nodes whose edge support is weak may sit
   ±``UNCERTAIN_SPAN`` of the band elsewhere: their worst / best placement gives IoU bounds.
   通过 if the lower bound ≥ 0.95, 未通过 if the upper bound < 0.95, otherwise 待复核.

The bar is fixed: ``PASS_IOU`` = 0.95 (user, never lowered). The fitted readouts in
``identity_score`` stay diagnostics only.
"""
from __future__ import annotations

import time

import numpy as np

from .layered_output import dilate, erode, label_components, resize, _sample_offsets

PASS_IOU = 0.95
SCORE_LONG_EDGE = 1600     # scoring resolution cap (long edge, px)
BAND_FRAC = 0.03           # max node displacement as a share of the product diagonal
BAND_MIN, BAND_MAX = 6, 48
HOLE_MIN_FRAC = 0.0015     # enclosed holes smaller than this share of the canvas are ignored
HOLE_ERODE_FRAC = 0.006    # hole core (eroded by 0.6 % of the long edge, ≥2 px) is always background
PRIOR_W = 0.30             # pull of each contour node towards d = 0 (per band width)
SMOOTH_W = 0.04            # cost of a 1 px step between neighbouring nodes
CHROMA_W = 0.6             # weight of a/b against L in the edge strength
EDGE_CAP = 3.0             # soft cap of edge strength (in units of the band's 90th percentile)
WEAK_EDGE = 0.35           # node edge support below this = uncertain node
UNCERTAIN_SPAN = 0.25      # an uncertain node may sit ±25 % of the band away (IoU bounds)
WEAK_MAX = 0.5             # more uncertain nodes than this share = 待复核 at best
HOLE_FILLED = 0.5          # a hole whose core looks like the product on more than this share is "filled"
STEP_K = 2.0               # px each side of a candidate for the product/background test
STEP_NORM = 0.12           # Lab-scaled distance that counts as a clear product/background step
STEP_FLOOR = 0.25          # an edge with no product/background step keeps this share of its strength
REGION_W = 0.25           # cost per px of moving the outline across product-like / bg-like pixels
RIVAL_GAP = 3              # px: a second edge this far from the chosen one …
RIVAL_TOL = 0.05           # … whose total cost is within 5 % of the chosen edge makes the node uncertain
SAMPLES = 96
ROD_TAU = 0.10             # thin-rod pixel looks present when farther than this from every bg sample
ROD_PRESENT = 0.3          # a rod with fewer present-looking pixels than this is uncertain
OUT_OF_BAND_FRAC = 0.15    # snapped contour at the band limit on this share of nodes = drifted past it
VERDICT_PASS, VERDICT_FAIL, VERDICT_REVIEW = "通过", "未通过", "待复核"


def _holes(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Enclosed background components of the mask (not touching the canvas border)."""
    labels, areas = label_components(~mask)
    out = np.zeros_like(mask)
    if not areas:
        return out
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])).tolist())
    for lab, area in areas.items():
        if lab and lab not in border and area >= min_px:
            out |= labels == lab
    return out


def _dist(mask: np.ndarray, cap: int) -> np.ndarray:
    """Capped octagonal distance (px) to the nearest True pixel."""
    d = np.full(mask.shape, float(cap), np.float32)
    cur = np.asarray(mask, bool).copy()
    d[cur] = 0
    for k in range(1, cap):
        p = np.pad(cur, 1)
        nxt = cur | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        if k % 2 == 0:
            nxt |= p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
        new = nxt & ~cur
        if not new.any():
            break
        d[new] = k
        cur = nxt
    return d


_MOORE = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]   # clockwise from N


_DIR_OF = {d: i for i, d in enumerate(_MOORE)}


def trace_boundary(region: np.ndarray) -> np.ndarray:
    """Ordered outer boundary pixels (y, x) of one 8-connected region (Moore-neighbour tracing,
    stops when the start state repeats)."""
    R = np.pad(np.asarray(region, bool), 1)
    ys, xs = np.nonzero(R)
    if not len(ys):
        return np.zeros((0, 2), int)
    i0 = np.lexsort((xs, ys))[0]
    start = (int(ys[i0]), int(xs[i0]))
    cur, bdir = start, 6                       # backtrack = west neighbour (background)
    pts = [start]
    first_state = None
    for _ in range(8 * int(R.sum()) + 16):
        nxt = None
        for k in range(1, 9):
            dirn = (bdir + k) % 8
            cy, cx = cur[0] + _MOORE[dirn][0], cur[1] + _MOORE[dirn][1]
            if R[cy, cx]:
                pb = _MOORE[(dirn - 1) % 8]
                by, bx = cur[0] + pb[0], cur[1] + pb[1]
                nxt = (cy, cx)
                bdir = _DIR_OF[(by - cy, bx - cx)]
                break
        if nxt is None:
            break                              # isolated pixel
        state = (nxt, bdir)
        if first_state is None:
            first_state = state
        elif state == first_state:
            break
        cur = nxt
        pts.append(cur)
    if len(pts) > 1 and pts[-1] == start:
        pts.pop()
    return np.array(pts, int) - 1


def _bilinear(img: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    y = np.clip(y, 0, h - 1.001)
    x = np.clip(x, 0, w - 1.001)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    fy = y - y0
    fx = x - x0
    for _ in range(img.ndim - 2):
        fy = fy[..., None]
        fx = fx[..., None]
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy)
            + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


def fill_polygon(pts_yx: np.ndarray, shape) -> np.ndarray:
    """Even-odd scanline fill of a closed polygon (float y, x vertices) at pixel centres."""
    h, w = shape
    if len(pts_yx) < 3:
        return np.zeros(shape, bool)
    y0, x0 = pts_yx[:, 0].astype(np.float64), pts_yx[:, 1].astype(np.float64)
    y1, x1 = np.roll(y0, -1), np.roll(x0, -1)
    lo, hi = np.minimum(y0, y1), np.maximum(y0, y1)
    r_lo = np.ceil(lo - 0.5).astype(int)
    r_hi = np.ceil(hi - 0.5).astype(int)        # rows r with lo <= r + 0.5 < hi
    n = np.maximum(r_hi - r_lo, 0)
    keep = np.nonzero(n > 0)[0]
    if not len(keep):
        return np.zeros(shape, bool)
    nk = n[keep]
    e = np.repeat(keep, nk)
    off = np.arange(len(e)) - np.repeat(np.cumsum(nk) - nk, nk)
    r = r_lo[e] + off
    t = (r + 0.5 - y0[e]) / (y1[e] - y0[e])
    xc = x0[e] + t * (x1[e] - x0[e])
    col = np.clip(np.ceil(xc - 0.5).astype(int), 0, w)
    ok = (r >= 0) & (r < h)
    tog = np.zeros((h, w + 1), np.int32)
    np.add.at(tog, (r[ok], col[ok]), 1)
    return (np.cumsum(tog, 1)[:, :w] % 2).astype(bool)


def _snap(pts, product, grad, Din, Dout, r, scale_e, lab=None, look=None):
    """DP along the normals of one traced boundary. Returns (disp, support, normals, points,
    lim_in, lim_out); positive displacement always grows the product."""
    n = len(pts)
    P = pts.astype(np.float32)
    idx = np.arange(n)
    k = 3 if n > 24 else 1
    sm = sum(P[(idx + o) % n] for o in range(-k, k + 1)) / (2 * k + 1)
    tang = sm[(idx + 2) % n] - sm[(idx - 2) % n]
    tn = np.hypot(tang[:, 0], tang[:, 1]) + 1e-6
    nrm = np.stack([tang[:, 1], -tang[:, 0]], 1) / tn[:, None]
    h, w = product.shape
    probe = np.rint(P + 2.5 * nrm).astype(int)
    probe[:, 0] = np.clip(probe[:, 0], 0, h - 1)
    probe[:, 1] = np.clip(probe[:, 1], 0, w - 1)
    nrm[product[probe[:, 0], probe[:, 1]]] *= -1        # normals point product -> background
    offs = np.arange(-r, r + 1, dtype=np.float32)
    yy = P[:, 0:1] + offs[None, :] * nrm[:, 0:1]
    xx = P[:, 1:2] + offs[None, :] * nrm[:, 1:2]
    g = _bilinear(grad, yy, xx)                          # (n, K, C, 2)
    proj = g[..., 0] * nrm[:, None, None, 0] + g[..., 1] * nrm[:, None, None, 1]
    E = np.sqrt((proj ** 2).sum(-1)) / scale_e
    E = EDGE_CAP * E / (EDGE_CAP + E)                    # soft cap: keeps the peak where it is
    din = _bilinear(Din, yy, xx)
    dout = _bilinear(Dout, yy, xx)
    lim_in = np.maximum(1.0, 0.85 * din[:, :r + 1].max(1))
    lim_out = np.maximum(1.0, 0.85 * dout[:, r:].max(1))
    ok = (offs[None, :] >= -lim_in[:, None]) & (offs[None, :] <= lim_out[:, None])
    gain = E
    if lab is not None:
        # an outline separates product from *background*: just outside the candidate looks like
        # the background further out along the normal, just inside does not (internal edges
        # between two product colours score low here)
        far = np.clip(np.linspace(0.5, 1.0, 5)[None, :] * np.maximum(lim_out, 2.0)[:, None] + r * 0.25, 1, 2 * r)
        Bf = _bilinear(lab, P[:, 0:1] + far * nrm[:, 0:1], P[:, 1:2] + far * nrm[:, 1:2]).mean(1)   # (n, 3)
        ins = _bilinear(lab, yy - STEP_K * nrm[:, 0:1], xx - STEP_K * nrm[:, 1:2])
        outs = _bilinear(lab, yy + STEP_K * nrm[:, 0:1], xx + STEP_K * nrm[:, 1:2])
        d_in = np.sqrt(((ins - Bf[:, None, :]) ** 2).sum(-1))
        d_out = np.sqrt(((outs - Bf[:, None, :]) ** 2).sum(-1))
        S = np.clip((d_in - d_out) / STEP_NORM, 0, None)
        S = EDGE_CAP * S / (EDGE_CAP + S)
        gain = E * (STEP_FLOOR + S)
    region = 0.0
    if lab is not None and look is not None:
        # region term (Blender appearance at fixed coordinates, colour-normalised to the AI):
        # moving the outline inward across pixels that look like *this product here* rather than
        # like the background costs; so does moving it outward across background-looking pixels.
        A = _bilinear(lab, yy, xx)                                           # (n, K, 3)
        Bl = _bilinear(look, yy, xx)
        A_in = _bilinear(lab, P[:, 0] - 3 * nrm[:, 0], P[:, 1] - 3 * nrm[:, 1])
        d_bg = np.sqrt(((A - Bf[:, None, :]) ** 2).sum(-1))
        d_look = np.sqrt(((A - Bl) ** 2).sum(-1))
        d_in = np.sqrt(((A - A_in[:, None, :]) ** 2).sum(-1))
        prod_like = np.clip((d_bg - d_look) / STEP_NORM, 0, 1)             # inside Blender's outline
        bg_like = np.clip((d_in - d_bg) / STEP_NORM, 0, 1)                 # outside it
        K = len(offs)
        zero = r
        inner = np.where(np.arange(K)[None, :] < zero, prod_like, 0.0)
        outer = np.where(np.arange(K)[None, :] > zero, bg_like, 0.0)
        # cost(d<0) = sum of prod_like over offsets d..-1; cost(d>0) = sum of bg_like over 1..d
        cin = np.cumsum(inner[:, ::-1], 1)[:, ::-1]                       # sum from o to the end
        cout = np.cumsum(outer, 1)
        region = REGION_W * np.where(np.arange(K)[None, :] < zero, cin, cout)
    unary = np.where(ok, -gain + region + PRIOR_W * np.abs(offs)[None, :] / max(r, 1), 1e6)
    K = len(offs)
    cost = unary[0].copy()
    back = np.zeros((n, K), np.int8)
    big = np.array([1e9])
    for i in range(1, n):
        stack = np.stack([np.concatenate((big, cost[:-1])) + SMOOTH_W, cost,
                          np.concatenate((cost[1:], big)) + SMOOTH_W])
        a = stack.argmin(0)
        cost = stack[a, np.arange(K)] + unary[i]
        back[i] = a - 1
    d = np.zeros(n, int)
    d[-1] = int(cost.argmin())
    for i in range(n - 1, 0, -1):
        d[i - 1] = d[i] + int(back[i, d[i]])
    # a rival edge: another displacement (≥ RIVAL_GAP px away) nearly as strong as the chosen one
    # rivals are judged on the full node cost (edge, region term and prior), not the edge alone
    sc = np.where(ok, -unary, -1e9)
    far_mask = np.abs(np.arange(K)[None, :] - d[:, None]) >= max(RIVAL_GAP, int(round(0.2 * r)))
    rival = np.where(far_mask, sc, -1e9).argmax(1)
    chosen = gain[idx, d]
    has_rival = sc[idx, rival] >= sc[idx, d] - RIVAL_TOL * np.maximum(chosen, 1e-6)
    return offs[d], chosen, nrm, P, lim_in, lim_out, np.where(has_rival, offs[rival], np.nan)


def _filled_holes(I, M, holes, hole_core, near, r):
    """Secondary check of the holes = background rule: a hole whose core pixels are closer to the
    product colours around it than to the scene background is reported as possibly filled."""
    out = np.zeros_like(M)
    exterior = ~near & ~dilate(holes, r)
    ey, ex = np.nonzero(exterior)
    if len(ey) < 50:
        return out
    rng = np.random.default_rng(5)
    pick = rng.choice(len(ey), size=min(600, len(ey)), replace=False)
    bg = I[ey[pick], ex[pick]]                                      # (B, 3)
    hl, ha = label_components(holes)
    ring_all = M & dilate(holes, r) & ~holes
    for lab, area in ha.items():
        hole = hl == lab
        core = hole & hole_core
        cy, cx = np.nonzero(core)
        if len(cy) < 30:
            continue
        ring = ring_all & dilate(hole, r)
        ry, rx = np.nonzero(ring)
        if len(ry) < 30:
            continue
        sel = rng.choice(len(ry), size=min(400, len(ry)), replace=False)
        fg = I[ry[sel], rx[sel]]
        take = rng.choice(len(cy), size=min(800, len(cy)), replace=False)
        px = I[cy[take], cx[take]]
        dF = np.sqrt(((px[:, None, :] - fg[None]) ** 2).sum(-1)).min(1)
        dB = np.sqrt(((px[:, None, :] - bg[None]) ** 2).sum(-1)).min(1)
        if float((dF + 0.02 < dB).mean()) > HOLE_FILLED:
            out |= hole
    return out


def _rod_rule(I, zone, outside, r):
    """Thin-rod pixels: product iff farther than ROD_TAU from every nearby background sample."""
    hh, ww = zone.shape
    by, bx = np.nonzero(zone)
    res = np.zeros_like(zone)
    if not len(by):
        return res
    offs = _sample_offsets(2 * r + 2, SAMPLES, seed=11)
    for lo in range(0, len(by), 30000):
        cy, cx = by[lo:lo + 30000], bx[lo:lo + 30000]
        sy = np.clip(cy[:, None] + offs[None, :, 0], 0, hh - 1)
        sx = np.clip(cx[:, None] + offs[None, :, 1], 0, ww - 1)
        dist = np.sqrt(((I[sy, sx] - I[cy, cx][:, None, :]) ** 2).sum(-1))
        dB = np.where(outside[sy, sx], dist, 1e3).min(1)
        hit = dB > ROD_TAU
        res[cy[hit], cx[hit]] = True
    return res


def _look(ai_lab: np.ndarray, detail_rgb: np.ndarray, core: np.ndarray):
    """Blender detail colours mapped into the AI's colour space (per-channel robust linear fit
    over the product core, Lab-scaled like ``ai_lab``), so lighting differences cancel."""
    from .material_color import srgb_to_lab
    d = srgb_to_lab(np.clip(detail_rgb, 0, 1)) / np.array([100.0, 128.0, 128.0], np.float32)
    d[..., 1:] *= CHROMA_W
    out = np.zeros_like(d)
    sel = core
    if sel.sum() < 100:
        return None
    for c in range(3):
        a, b = ai_lab[..., c][sel], d[..., c][sel]
        a5, a95 = np.percentile(a, [5, 95])
        b5, b95 = np.percentile(b, [5, 95])
        k = float(np.clip((a95 - a5 + 1e-3) / (b95 - b5 + 1e-3), 0.3, 3.0))
        out[..., c] = (d[..., c] - np.median(b)) * k + np.median(a)
    return out


def extract_region(ai_rgb: np.ndarray, product_mask: np.ndarray, bounds: bool = True, detail_rgb=None) -> dict:
    """Deterministic AI-side product region on the canvas of ``product_mask``.

    Returns dict(region, region_lo, region_hi, band_px, thin_px, hole_px, weak_frac, out_of_band)."""
    from .material_color import srgb_to_lab
    from .layered_output import blur
    M = np.asarray(product_mask) > 0.5
    h, w = M.shape
    I = np.clip(ai_rgb[..., :3].astype(np.float32), 0, 1)
    ys, xs = np.nonzero(M)
    if len(xs) < 50:
        raise ValueError("产品遮罩为空")
    diag = float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    r = int(np.clip(round(BAND_FRAC * diag), BAND_MIN, BAND_MAX))
    y0, y1 = max(0, ys.min() - 2 * r - 4), min(h, ys.max() + 2 * r + 5)
    x0, x1 = max(0, xs.min() - 2 * r - 4), min(w, xs.max() + 2 * r + 5)
    Mc, Ic = M[y0:y1, x0:x1], I[y0:y1, x0:x1]
    hh, ww = Mc.shape
    holes = _holes(Mc, int(HOLE_MIN_FRAC * h * w))
    hole_core = erode(holes, max(2, int(round(HOLE_ERODE_FRAC * max(h, w)))))
    lab = srgb_to_lab(Ic) / np.array([100.0, 128.0, 128.0], np.float32)
    lab[..., 1:] *= CHROMA_W
    grad = np.zeros((hh, ww, 3, 2), np.float32)
    for c in range(3):
        p = np.pad(blur(lab[..., c], 1, 1), 1, mode="edge")
        grad[..., c, 1] = ((p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])) / 8
        grad[..., c, 0] = ((p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])) / 8
    near = dilate(Mc, r)
    band = near & ~erode(Mc, r)
    gmag = np.sqrt((grad ** 2).sum((-1, -2)))
    scale_e = max(float(np.quantile(gmag[band], 0.90)), 1e-3)
    Din = _dist(~Mc, 2 * r + 2)
    Dout = _dist(Mc, 2 * r + 2)
    look = None
    if detail_rgb is not None:
        try:
            look = _look(lab, detail_rgb[y0:y1, x0:x1, :3].astype(np.float32), erode(Mc, 2))
        except Exception:
            look = None
    labels, areas = label_components(Mc)
    boundaries = [(labels == i, True) for i, a in areas.items() if a >= 30]
    hl, ha = label_components(holes)
    boundaries += [(hl == i, False) for i, a in ha.items() if a >= 30]
    keys = ("d", "lo", "hi") if bounds else ("d",)
    prod = {k: np.zeros_like(Mc) for k in keys}
    hole = {k: np.zeros_like(Mc) for k in keys}
    weak_n = tot_n = at_lim = 0
    for reg, is_product in boundaries:
        pts = trace_boundary(reg)
        if len(pts) < 8:
            if is_product:
                for k in keys:
                    prod[k] |= reg
            continue
        if len(pts) > 600:
            pts = pts[::2]
        disp, sup, nrm, P, lim_in, lim_out, alt = _snap(pts, Mc, grad, Din, Dout, r, scale_e, lab, look)
        weak = sup < WEAK_EDGE
        rivalled = ~np.isnan(alt) & ~weak
        weak_n += int(weak.sum())
        tot_n += len(weak)
        at_lim += int((np.abs(disp) >= r - 0.5).sum())
        U = max(1.0, UNCERTAIN_SPAN * r)
        cand = {"d": disp}
        if bounds:
            worst = np.where(weak, np.where(disp >= 0, disp + U, disp - U), disp)
            best = np.where(weak, np.sign(disp) * np.maximum(np.abs(disp) - U, 0), disp)
            a = np.nan_to_num(alt)
            worst = np.where(rivalled & (np.abs(a) > np.abs(disp)), a, worst)
            best = np.where(rivalled & (np.abs(a) < np.abs(disp)), a, best)
            weak_n += int(rivalled.sum())
            cand["lo"] = np.clip(worst, -lim_in, lim_out)
            cand["hi"] = np.clip(best, -lim_in, lim_out)
        # traced pixel centres sit half a pixel inside their region
        half = 0.5 if is_product else -0.5
        for k in keys:
            poly = P + (cand[k] + half)[:, None] * nrm
            filled = fill_polygon(poly, Mc.shape)
            if is_product:
                prod[k] |= filled
            else:
                hole[k] |= filled
    outside = ~near | hole_core
    hole_filled = _filled_holes(Ic, Mc, holes, hole_core, near, r) if holes.any() else np.zeros_like(Mc)
    thin = Mc & ~dilate(erode(Mc, r), r + 1)
    rod = _rod_rule(Ic, thin, outside, r) if thin.any() else thin
    rod_missing = np.zeros_like(Mc)
    if thin.any():
        tl, ta = label_components(thin)
        for i, a in ta.items():
            comp = tl == i
            if a >= 20 and (rod & comp).sum() < ROD_PRESENT * a:
                rod_missing |= comp
    out = {}
    for k in keys:
        R = prod[k] & ~hole[k] & ~hole_core
        if k == "lo" and rod_missing.any():
            R = R & ~rod_missing        # worst case: rods that mostly look like background are missing
        if k == "lo" and hole_filled.any():
            R = R | hole_filled         # worst case: a hole the AI seems to have closed is product
        z = np.zeros_like(M)
        z[y0:y1, x0:x1] = R
        out[k] = z
    return {"region": out["d"], "region_lo": out.get("lo"), "region_hi": out.get("hi"), "band_px": r,
            "thin_px": int(thin.sum()), "hole_px": int(holes.sum()),
            "weak_frac": round(weak_n / max(tot_n, 1), 3), "holes_filled_px": int(hole_filled.sum()),
            "out_of_band": round(at_lim / max(tot_n, 1), 3)}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def verdict(iou_lo: float, iou_hi: float, out_of_band: float, weak_frac: float = 0.0) -> str:
    if out_of_band > OUT_OF_BAND_FRAC or weak_frac > WEAK_MAX:
        # ran past the band, or most of the outline has no edge to judge by: never 通过
        return VERDICT_FAIL if iou_hi < PASS_IOU else VERDICT_REVIEW
    if iou_lo >= PASS_IOU:
        return VERDICT_PASS
    if iou_hi < PASS_IOU:
        return VERDICT_FAIL
    return VERDICT_REVIEW


def score(ai_rgb: np.ndarray, product_mask: np.ndarray, detail_rgb=None) -> dict:
    """Fixed-coordinate IoU of one AI image (already on the canvas) vs the Blender mask.
    ``detail_rgb`` (optional, same canvas): the Blender detail pass, used as the product's own
    appearance in the contour's region term — never as the outline itself."""
    h, w = product_mask.shape[:2]
    f = max(h, w) / float(SCORE_LONG_EDGE)
    if f > 1.0:
        sw, sh = int(round(w / f)), int(round(h / f))
        ai_rgb = resize(ai_rgb, sw, sh)
        product_mask = resize(np.asarray(product_mask, np.float32), sw, sh)
        if detail_rgb is not None:
            detail_rgb = resize(detail_rgb, sw, sh)
    M = np.asarray(product_mask) > 0.5
    ex = extract_region(ai_rgb, M, detail_rgb=detail_rgb)
    R = ex["region"]
    iou = _iou(R, M)
    lo = min(iou, _iou(ex["region_lo"], M))
    hi = max(iou, _iou(ex["region_hi"], M))
    v = verdict(lo, hi, ex["out_of_band"], ex["weak_frac"])
    return {"iou": round(iou, 4), "iou_lo": round(lo, 4), "iou_hi": round(hi, 4), "verdict": v,
            "passed": v == VERDICT_PASS, "threshold": PASS_IOU,
            "weak_frac": ex["weak_frac"], "out_of_band": ex["out_of_band"],
            "missing_px": int((M & ~R).sum()), "extra_px": int((R & ~M).sum()),
            "band_px": ex["band_px"], "thin_px": ex["thin_px"], "hole_px": ex["hole_px"],
            "holes_filled_px": ex.get("holes_filled_px", 0),
            "size": [int(M.shape[1]), int(M.shape[0])], "method": "fixed-coordinate contour v2 (DP snap)",
            "_region": R, "_ambiguous": ex["region_lo"] ^ ex["region_hi"]}


def score_file(image_path, mask_path, region_out="", detail_path="") -> dict:
    """File entry: the image is resampled to the mask (canvas) size, never fitted. Never raises."""
    t0 = time.time()
    try:
        from .png_io import read_png
        rgba = read_png(mask_path).astype(np.float32) / 255.0
        Mf = rgba[..., 3] if rgba.ndim == 3 and rgba.shape[-1] == 4 else (rgba if rgba.ndim == 2 else rgba[..., 0])
        h, w = Mf.shape
        img = read_png(image_path).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.dstack([img] * 3)
        if img.shape[:2] != (h, w):
            img = np.clip(resize(img[..., :3], w, h), 0, 1)
        det = None
        if detail_path:
            try:
                from pathlib import Path as _P
                if _P(detail_path).is_file():
                    det = read_png(detail_path).astype(np.float32) / 255.0
                    if det.shape[:2] != (h, w):
                        det = np.clip(resize(det, w, h), 0, 1)
                    det = det[..., :3]
            except Exception:
                det = None
        rep = score(img[..., :3], Mf, detail_rgb=det)
        rep["appearance"] = det is not None
        if region_out:
            try:
                from .png_io import write_png
                R, A = rep["_region"], rep["_ambiguous"]
                vis = np.dstack([R * 255, R * 255, np.where(A, 255, R * 255)]).astype(np.uint8)
                write_png(region_out, vis)
            except Exception:
                pass
    except Exception as exc:
        rep = {"error": str(exc)[:200], "verdict": VERDICT_REVIEW, "passed": False, "iou": None}
    rep.pop("_region", None)
    rep.pop("_ambiguous", None)
    rep["seconds"] = round(time.time() - t0, 2)
    return rep


def summarize(rep: dict) -> str:
    """Panel line, e.g. 「轮廓 IoU 0.962 · 通过（≥0.95，固定坐标）」."""
    if not rep or rep.get("iou") is None:
        return "轮廓 IoU：无法计算（待复核）" if rep else ""
    line = f"轮廓 IoU {rep['iou']:.3f} · {rep['verdict']}（≥{PASS_IOU:.2f}，固定坐标）"
    if rep.get("verdict") == VERDICT_REVIEW:
        line += f" 范围 {rep.get('iou_lo', 0):.3f}–{rep.get('iou_hi', 0):.3f}"
    return line
