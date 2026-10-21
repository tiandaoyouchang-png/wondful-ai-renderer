"""④ 场景粗模 · 细节粗模 (V3, phase 1): measured near / mid-ground geometry from the 精确模式 depth.

Built only when 精确模式 measured the reference photo (MoGe-2, MIT). Pure numpy (Blender on Windows has no
scipy); the ``bpy`` part at the end imports lazily.

What it builds (analysis frame of ``precise_depth``: camera at (0, 0, h), looking +Y, Z up, ground z = 0):
- **ground**: a 360° polar heightfield. Measured ground (< ``R_SPLIT``) is binned by azimuth / range, occluded
  ground is interpolated along range, azimuths outside the photo fade to the ring median, and beyond
  ``R_SPLIT`` the surface blends into the LLM terrain height (``far_z``) out to ``R_OUT`` — no holes from any
  orbit camera, and the far LLM bands still stand on the ground they were built on;
- **structures** (buildings / walls): a measured thin skin (closed shell ≤ 1 m thick, torn at depth jumps and
  grazing quads) plus closed prisms behind it with the measured width / eaves height and a rule depth;
  procedural windows (thin closed frames) only on prism faces the reference camera cannot see;
- **vehicles / props**: closed chamfered boxes at the measured position and size; **poles**: cylinders;
  floating slabs (awnings, signs) at their measured height;
- **trees / bushes**: closed lumpy crown volumes + trunks (never a depth skin: foliage depth is noise);
- the **far field** is not measured geometry: the LLM bands stay as they are (user, 2026-10-20: 「重要的是近景
  和中景 最远处的山还好」).

Ported from the 2026-10-20 prototype (``scene_blockout/vb.py``, V3 = V1 volumes + measured skin), with the
scipy calls replaced (``label`` / ``find_objects`` / ``gaussian`` / ``median`` / ``max`` filters) and a
triangle budget (``TRI_BUDGET``): skin stride and window count adapt so a scene stays ≲ 80 k triangles.
"""
from __future__ import annotations

import math
import time

import numpy as np

SKY, GROUND, STRUCT, VEG, FAR = 0, 1, 2, 3, 4
R_SPLIT = 130.0            # near / mid ground limit (m): beyond this nothing measured is built
R_BLEND = 1.45             # ground blends into the LLM terrain between R_SPLIT and R_SPLIT × R_BLEND
R_OUT = 1500.0             # outer ground radius (same as the LLM 地面_外圈)
TRI_BUDGET = 80000
SKIN_BUDGET = 40000
TERRAIN_AZ_IN = 1.0        # azimuth step inside the photo (deg); outside 6°
TERRAIN_AZ_OUT = 6.0
SKIN_T = 1.0               # skin thickness cap (m)
# 近密远疏 (distance bands, idea from Building Rome arXiv 2610.08790): near field small / detailed, far field coarse
BAND_NEAR = 25.0           # m from the product: full detail (windows, front hedges, fine skin)
BAND_MID = 55.0            # beyond this from the product: merged lots, no procedural windows / front hedges
SKIN_NEAR_R = 30.0         # m from the camera: measured skin at the fine stride; beyond it at twice the stride
SKIN_OVERLAP = 1.5         # the coarse skin starts this much nearer (overlap band, no crack between strides)
SKIN_BUDGET_BANDS = 52000  # with bands the near skin may use what the coarse far skin saves …
SKIN_RESERVE = 6000        # … but this much of the total budget stays for 场景补全
CARVE_MIN_D = 0.8          # 空/可见/被挡住: a carved prism keeps at least this depth
CARRIER_TAG = "wondful_detail_carrier"
FAR_TAG = "wondful_sb_far"
COLLECTION_NAME = "Wondful 细节粗模（投影）"
GROUND_TYPES = {"terrain", "road", "road_branch", "path", "sidewalk", "plaza", "river", "water_plane", "rug",
                "zone_shrub_slope", "zone_grass", "zone_rock", "zone_scree", "zone_sand", "zone_snow", "zone_paving",
                "zone_field", "terrain_steep"}
FAR_TYPE_PREFIX = ("band_",)
FAR_TYPES = {"cliff", "band_sea"}
# indoor / implausible ground fit → keep the normal blockout (interior research run: h 0.61 m, roll −9.6°)
CAM_H_RANGE = (0.9, 25.0)
MAX_ROLL = 6.0
MAX_PITCH = 35.0
MIN_GROUND_SHARE = 0.04

# part classes (stored per face as ``C``)
C_GROUND, C_BUILDING, C_PROP, C_TREE, C_POLE, C_SKIN, C_WINDOW, C_WALL = 0, 1, 2, 3, 4, 5, 6, 7
C_NAME = {C_GROUND: "地面", C_BUILDING: "建筑体积", C_PROP: "车辆道具", C_TREE: "树冠", C_POLE: "杆",
          C_SKIN: "实测表皮", C_WINDOW: "窗", C_WALL: "墙栏",
          # 场景补全 (scene_complete.C_INF_*): inferred parts, separate objects tagged INFER_TAG
          8: "推测建筑", 9: "推测绿植", 10: "推测电杆", 11: "推测路沿", 12: "推测窗"}
C_INFERRED = (8, 9, 10, 11, 12)
INFER_TAG = "wondful_detail_inferred"
LAYOUT_KEY = "wondful_completion"


# ============================================================================== numpy stand-ins for scipy.ndimage

def label(mask, eight=True):
    """Connected components of a 2-D bool array (run-length union-find). Returns (labels int32, n)."""
    m = np.asarray(mask, bool)
    H, W = m.shape
    lab = np.zeros((H, W), np.int32)
    runs = []                    # (row, x0, x1) inclusive-exclusive
    row_runs = []
    for y in range(H):
        r = m[y].astype(np.int8)
        d = np.diff(np.r_[0, r, 0])
        s = np.nonzero(d == 1)[0]
        e = np.nonzero(d == -1)[0]
        base = len(runs)
        runs.extend((y, int(a), int(b)) for a, b in zip(s, e))
        row_runs.append((base, len(runs)))
    n = len(runs)
    if n == 0:
        return lab, 0
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    k = 1 if eight else 0
    for y in range(1, H):
        a0, a1 = row_runs[y - 1]
        b0, b1 = row_runs[y]
        i, j = a0, b0
        while i < a1 and j < b1:
            _, s1, e1 = runs[i]
            _, s2, e2 = runs[j]
            if s1 < e2 + k and s2 < e1 + k:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
            if e1 < e2:
                i += 1
            else:
                j += 1
    roots = {}
    for idx, (y, s, e) in enumerate(runs):
        r = find(idx)
        if r not in roots:
            roots[r] = len(roots) + 1
        lab[y, s:e] = roots[r]
    return lab, len(roots)


def find_objects(lab, n):
    """Bounding slices of labels 1..n (like ``ndimage.find_objects``)."""
    out = [None] * n
    ys, xs = np.nonzero(lab)
    if len(ys) == 0:
        return out
    v = lab[ys, xs] - 1
    y0 = np.full(n, 1 << 30); y1 = np.full(n, -1); x0 = np.full(n, 1 << 30); x1 = np.full(n, -1)
    np.minimum.at(y0, v, ys); np.maximum.at(y1, v, ys); np.minimum.at(x0, v, xs); np.maximum.at(x1, v, xs)
    for k in range(n):
        if y1[k] >= 0:
            out[k] = (slice(int(y0[k]), int(y1[k]) + 1), slice(int(x0[k]), int(x1[k]) + 1))
    return out


def _gauss_kernel(sigma):
    r = max(1, int(4.0 * sigma + 0.5))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    return k / k.sum(), r


def gaussian_filter(a, sigma):
    """Separable Gaussian, reflect padding (``ndimage`` default mode), per axis sigma."""
    a = np.asarray(a, np.float64)
    sig = (sigma,) * a.ndim if np.isscalar(sigma) else sigma
    out = a
    for ax, s in enumerate(sig):
        if s <= 0:
            continue
        k, r = _gauss_kernel(s)
        pad = [(0, 0)] * a.ndim
        pad[ax] = (r, r)
        p = np.pad(out, pad, mode="symmetric")
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            sl = [slice(None)] * a.ndim
            sl[ax] = slice(i, i + out.shape[ax])
            acc += w * p[tuple(sl)]
        out = acc
    return out


def _window1d(a, size):
    a = np.asarray(a, np.float64)
    r = size // 2
    p = np.pad(a, (r, size - 1 - r), mode="symmetric")
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(p, size)


def median_filter1d(a, size):
    return np.median(_window1d(a, size), -1)


def maximum_filter1d(a, size):
    return _window1d(a, size).max(-1)


# ============================================================================== mesh container

class Mesh:
    def __init__(self):
        self.V, self.F, self.C, self.parts = [], [], [], []
        self.n = 0

    def add(self, V, F, c, info=None):
        V = np.asarray(V, np.float64).reshape(-1, 3)
        F = np.asarray(F, np.int64)
        if len(F) == 0:
            return
        if F.shape[1] == 4:
            F = np.r_[F[:, [0, 1, 2]], F[:, [0, 2, 3]]]
        self.V.append(V)
        self.F.append(F + self.n)
        self.C.append(np.full(len(F), c, np.uint8))
        self.n += len(V)
        if info is not None:
            info = dict(info)
            info["tris"] = int(len(F))
            self.parts.append(info)

    def tris(self):
        return int(sum(len(f) for f in self.F))

    def arrays(self):
        if not self.F:
            return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0, np.uint8)
        return np.concatenate(self.V), np.concatenate(self.F), np.concatenate(self.C)


def weld(V, F, tol=1e-4):
    """Merge coincident vertices (for closure checks)."""
    key = np.round(np.asarray(V) / tol).astype(np.int64)
    _u, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return np.asarray(V)[idx], inv.ravel()[np.asarray(F)]


def open_edges(F):
    """Edges used by exactly one triangle (0 for a closed, edge-manifold solid)."""
    F = np.asarray(F)
    E = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
    E = E[E[:, 0] != E[:, 1]]
    key = np.sort(E, 1)
    _u, cnt = np.unique(key, axis=0, return_counts=True)
    return _u[cnt == 1]


# ============================================================================== primitives (all closed)

def box(c, u, n, W, D, z0, z1):
    """Closed box: front face centre c (xy), u along the facade, n toward the camera, width W, depth D (along −n)."""
    c = np.asarray(c, float); u = np.asarray(u, float); n = np.asarray(n, float)
    corners = []
    for z in (z0, z1):
        for a, b in ((-.5, 0), (.5, 0), (.5, -1), (-.5, -1)):
            p = c + u * a * W + n * b * D
            corners.append((p[0], p[1], z))
    F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return np.array(corners), np.array(F)


def window_frames(o, a, b, la, lb, nrm, floor_h=3.0, bay=3.2, win_w=1.2, win_h=1.5, proud=0.08, limit=10 ** 9):
    """Thin closed window frames on one face (origin o, unit axes a/b, lengths, outward normal)."""
    nf = int((lb - 0.6) // floor_h)
    nb = int(la // bay)
    if lb < 3.2 or la < 2.5 or nb == 0 or nf == 0:
        return [], []
    mx = (la - nb * bay) / 2
    V, F = [], []
    count = 0
    for j in range(nf):
        s_v = 0.6 + j * floor_h + (floor_h - win_h) * 0.55
        if s_v + win_h > lb - 0.3:
            break
        for k in range(nb):
            if count >= limit:
                return V, F
            s_u = mx + k * bay + (bay - win_w) / 2
            p0 = o + a * s_u + b * s_v - nrm * 0.02
            base = len(V)
            for dz in (0.0, proud + 0.02):
                q = p0 + nrm * dz
                V.extend([q, q + a * win_w, q + a * win_w + b * win_h, q + b * win_h])
            F.extend([(base + 0, base + 3, base + 2, base + 1), (base + 4, base + 5, base + 6, base + 7),
                      (base + 0, base + 1, base + 5, base + 4), (base + 1, base + 2, base + 6, base + 5),
                      (base + 2, base + 3, base + 7, base + 6), (base + 3, base + 0, base + 4, base + 7)])
            count += 1
    return V, F


def prism_with_windows(M, c, u, n, W, D, z0, z1, cam_xy, info, windows=True, win_limit=10 ** 9,
                       cls=C_BUILDING, win_cls=C_WINDOW):
    """Closed prism (``box``) + window frames only on its vertical faces the reference camera cannot see."""
    V, F = box(c, u, n, W, D, z0, z1)
    M.add(V, F, cls, info)
    if not windows:
        return 0
    return add_windows(M, c, u, n, W, D, z0, z1, cam_xy, win_limit=win_limit, win_cls=win_cls)


def add_windows(M, c, u, n, W, D, z0, z1, cam_xy, win_limit=10 ** 9, win_cls=C_WINDOW):
    """Window frames on the vertical faces of box(c, u, n, W, D, z0, z1) the reference camera cannot see."""
    if z1 - z0 < 4.0 or win_limit <= 0:
        return 0
    c = np.asarray(c, float); u = np.asarray(u, float); n = np.asarray(n, float)
    up = np.array([0.0, 0.0, 1.0]); u3 = np.r_[u, 0.0]; n3 = np.r_[n, 0.0]
    faces = [  # (origin, axis a, length, outward normal, face centre xy)
        (np.r_[c - u * W / 2, z0], u3, W, n3, c),                                         # front (+n)
        (np.r_[c + u * W / 2, z0], -n3, D, u3, c + u * W / 2 - n * D / 2),                 # right (+u)
        (np.r_[c - u * W / 2 - n * D, z0], n3, D, -u3, c - u * W / 2 - n * D / 2),         # left (−u)
        (np.r_[c + u * W / 2 - n * D, z0], -u3, W, -n3, c - n * D),                         # back (−n)
    ]
    made = 0
    for o, a, la, nrm, fc in faces:
        to_cam = np.asarray(cam_xy, float) - fc
        if float(nrm[:2] @ to_cam) > 0.0:          # the ref camera sees this face: the photo has the real windows
            continue
        Vw, Fw = window_frames(o, a, up, la, z1 - z0, nrm, limit=win_limit - made)
        if Fw:
            M.add(Vw, Fw, win_cls)
            made += len(Fw) // 6
    return made


def rounded_box(c, u, n, W, D, z0, z1, bev=0.25):
    """Closed chamfered box (vehicles / props): octagonal footprint, top edge chamfer."""
    c = np.asarray(c, float); u = np.asarray(u, float); n = np.asarray(n, float)
    b = min(bev, W / 3, D / 3, (z1 - z0) / 3)
    ctr = c - n * D / 2
    hw, hd = W / 2, D / 2
    ring = [(-hw + b, -hd), (hw - b, -hd), (hw, -hd + b), (hw, hd - b), (hw - b, hd), (-hw + b, hd), (-hw, hd - b),
            (-hw, -hd + b)]

    def lay(z, s):
        return [np.r_[ctr + u * x * s + n * y * s, z] for x, y in ring]
    sh = 1 - 2 * b / max(W, D)
    V = lay(z0, 1) + lay(z1 - b, 1) + lay(z1, sh)
    F = []
    for L in range(2):
        for k in range(8):
            a0, a1 = L * 8 + k, L * 8 + (k + 1) % 8
            F.append((a0, a1, a1 + 8, a0 + 8))
    F.append(tuple(range(7, -1, -1)))
    F.append(tuple(range(16, 24)))
    tri = []
    for f in F:
        for k in range(1, len(f) - 1):
            tri.append((f[0], f[k], f[k + 1]))
    return np.array(V), np.array(tri)


def ellipsoid(c, rx, ry, rz, seed=0, nu=12, nv=8, lump=0.12):
    """Closed lumpy crown."""
    rng = np.random.default_rng(seed)
    ph = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    th = np.linspace(0, np.pi, nv)
    k1, k2, k3 = rng.uniform(0, 2 * np.pi, 3)
    V = [np.r_[c[0], c[1], c[2] + rz]]
    for t in th[1:-1]:
        for p in ph:
            s = 1 + lump * (np.sin(3 * p + k1) * np.sin(2 * t + k2) + 0.5 * np.sin(5 * p + k3))
            V.append(np.r_[c[0] + rx * s * np.sin(t) * np.cos(p), c[1] + ry * s * np.sin(t) * np.sin(p),
                           c[2] + rz * np.cos(t) * (0.8 + 0.2 * s)])
    V.append(np.r_[c[0], c[1], c[2] - rz * 0.85])
    F = []
    rows = nv - 2
    for i in range(nu):
        F.append((0, 1 + (i + 1) % nu, 1 + i))
    for rj in range(rows - 1):
        for i in range(nu):
            a = 1 + rj * nu + i
            b = 1 + rj * nu + (i + 1) % nu
            F.append((a, b, b + nu))
            F.append((a, b + nu, a + nu))
    last = len(V) - 1
    base = 1 + (rows - 1) * nu
    for i in range(nu):
        F.append((last, base + i, base + (i + 1) % nu))
    return np.array(V), np.array(F)


def cylinder(x, y, r, z0, z1, k=8):
    a = np.linspace(0, 2 * np.pi, k, endpoint=False)
    V = [(x + r * np.cos(t), y + r * np.sin(t), z0) for t in a] + [(x + r * np.cos(t), y + r * np.sin(t), z1) for t in a]
    F = [(i, (i + 1) % k, k + (i + 1) % k) for i in range(k)] + [(i, k + (i + 1) % k, k + i) for i in range(k)]
    F += [(0, k - 1 - i, k - 2 - i) for i in range(k - 2)] + [(k, k + 1 + i, k + 2 + i) for i in range(k - 2)]
    return np.array(V), np.array(F)


# ============================================================================== ground

def plausible(geo, scene_kind="outdoor"):
    """(ok, reason): is the measured ground fit good enough for a measured blockout?"""
    cam = geo.get("cam", {})
    if str(scene_kind or "outdoor") == "interior":
        return False, "室内场景"
    h = float(cam.get("height_m", 0) or 0)
    if not (CAM_H_RANGE[0] <= h <= CAM_H_RANGE[1]):
        return False, f"测得机位高 {h:.2f} m 不可信"
    if abs(float(cam.get("roll_deg", 0) or 0)) > MAX_ROLL:
        return False, f"测得滚转 {float(cam.get('roll_deg', 0)):.1f}° 不可信"
    if abs(float(cam.get("pitch_deg", 0) or 0)) > MAX_PITCH:
        return False, f"测得俯仰 {float(cam.get('pitch_deg', 0)):.1f}° 不可信"
    cls = np.asarray(geo["cls"])
    share = float((cls == GROUND).mean())
    if share < MIN_GROUND_SHARE:
        return False, f"地面像素只有 {share * 100:.1f}%"
    return True, ""


def terrain(P, mask, cls, hfov, far_z=None, pad=None, r_split=R_SPLIT, r_out=R_OUT):
    """360° polar heightfield (V, F, (ab, rb, Hm)). ``far_z(x, y)`` → LLM terrain height (arrays) used beyond
    ``r_split``; ``pad`` = (x, y, radius, z) flattens the product spot. Rows are closed across the 360° seam and
    start at r = 0, so the only open edge is the outer rim."""
    r = np.hypot(P[..., 0], P[..., 1])
    A = np.degrees(np.arctan2(P[..., 0], P[..., 1]))
    g = mask & (cls == GROUND) & (r < r_split)
    half = hfov / 2 + 4
    inner = np.arange(-half, half + 1e-6, TERRAIN_AZ_IN)
    outer_r = np.arange(half + TERRAIN_AZ_OUT, 180.0, TERRAIN_AZ_OUT)
    ab = np.r_[-outer_r[::-1], inner, outer_r]
    rb = np.r_[np.linspace(0.0, 3.0, 4)[:-1], np.geomspace(3.0, r_split * R_BLEND, 46),
               np.geomspace(r_split * R_BLEND * 1.25, r_out, 8)]
    nA, nR = len(ab), len(rb)
    edges_a = np.r_[ab[0] - 1.5, (ab[1:] + ab[:-1]) / 2, ab[-1] + 1.5]
    edges_r = np.r_[-1, (rb[1:] + rb[:-1]) / 2, rb[-1] * 1.1]
    Hm = np.full((nA, nR), np.nan)
    if g.any():
        ia = np.clip(np.searchsorted(edges_a, A[g]) - 1, 0, nA - 1)
        ir = np.clip(np.searchsorted(edges_r, r[g]) - 1, 0, nR - 1)
        Z = P[..., 2][g]
        key = ia * nR + ir
        o = np.lexsort((Z, key))
        ks, Zs = key[o], Z[o]
        u, st, cnt = np.unique(ks, return_index=True, return_counts=True)
        flat = np.full(nA * nR, np.nan)
        flat[u] = Zs[st + cnt // 2]
        Hm = flat.reshape(nA, nR)
    near = rb <= r_split
    for i in range(nA):
        v = ~np.isnan(Hm[i]) & near
        if not v.any():
            continue
        idx = np.nonzero(v)[0]
        row = np.interp(np.arange(nR), idx, Hm[i, idx])          # holds the last value beyond the last sample
        k = idx[0]
        if rb[k] > 2:
            pre = np.arange(k)
            row[pre] = Hm[i, k] * np.clip((rb[pre] - 2.0) / max(rb[k] - 2.0, 1e-3), 0, 1)
        Hm[i] = row
    good = ~np.isnan(Hm).all(1)
    gi = np.nonzero(good)[0]
    ring = np.nanmedian(Hm[good], 0) if len(gi) else np.zeros(nR)
    for i in range(nA):
        if not good[i]:
            if len(gi):
                j = gi[np.argmin(np.abs(gi - i))]
                t = min(1.0, abs(ab[i] - ab[j]) / 40.0)
                Hm[i] = (1 - t) * Hm[j] + t * ring
            else:
                Hm[i] = 0.0
    Hm[:, rb < 2.0] = 0.0
    Hm = gaussian_filter(Hm, (0.8, 0.8))
    AA, RR = np.meshgrid(np.radians(ab), rb, indexing="ij")
    X, Y = RR * np.sin(AA), RR * np.cos(AA)
    if far_z is not None:
        fz = np.asarray(far_z(X, Y), np.float64)
        t = np.clip((RR - r_split) / (r_split * (R_BLEND - 1.0)), 0.0, 1.0)
        t = t * t * (3 - 2 * t)
        Hm = Hm * (1 - t) + fz * t
    else:
        t = np.clip((RR - r_split) / (r_split * (R_BLEND - 1.0)), 0.0, 1.0)
        Hm = Hm * (1 - t)
    if pad is not None:
        px, py, pr, pz = pad
        d = np.hypot(X - px, Y - py)
        w = np.clip((d - pr) / max(1.5 * pr, 1.0), 0.0, 1.0)
        w = w * w * (3 - 2 * w)
        Hm = Hm * w + pz * (1 - w)
    V = np.stack([X, Y, Hm], -1).reshape(-1, 3)
    idx = np.arange(nA * nR).reshape(nA, nR)
    F = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    F2 = np.stack([idx[-1, :-1], idx[0, :-1], idx[0, 1:], idx[-1, 1:]], -1)
    F = np.r_[F, F2]
    # quads touching r = 0 collapse to triangles (shared centre); drop the degenerate one
    F = np.r_[F[:, [0, 1, 2]], F[:, [0, 2, 3]]]
    Vr = V[F]
    area = np.linalg.norm(np.cross(Vr[:, 1] - Vr[:, 0], Vr[:, 2] - Vr[:, 0]), axis=1)
    F = F[area > 1e-9]
    return V, F, (ab, rb, Hm)


def terrain_z(T, x, y):
    ab, rb, Hm = T
    a = math.degrees(math.atan2(x, y))
    r = math.hypot(x, y)
    i = int(np.clip(np.searchsorted(ab, a), 1, len(ab) - 1))
    j = int(np.clip(np.searchsorted(rb, r), 1, len(rb) - 1))
    return float(Hm[i, j])


# ============================================================================== components

def components(P, mask, cls, classes, st=2, grad=0.06, min_px=40, N=None, r_split=R_SPLIT):
    Nq = N[::st, ::st] if N is not None else None
    Q = P[::st, ::st]
    m = mask[::st, ::st] & np.isin(cls[::st, ::st], classes)
    r = np.hypot(Q[..., 0], Q[..., 1])
    m &= (r < r_split) & (Q[..., 2] > 0.2)
    lr = np.log(np.clip(r, 0.3, None))
    gy, gx = np.gradient(lr)
    m2 = m & (np.hypot(gx, gy) < grad)
    lab, n = label(m2, eight=True)
    out = []
    for k, sl in enumerate(find_objects(lab, n), 1):
        if sl is None:
            continue
        cm = lab[sl] == k
        if cm.sum() < min_px:
            continue
        pts = Q[sl][cm]
        lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
        keep = ((pts >= lo - 0.05) & (pts <= hi + 0.05)).all(1)
        full = np.zeros(m.shape, bool)
        full[sl] = cm
        out.append({"pts": pts[keep], "N": Nq[sl][cm][keep] if Nq is not None else None, "sl": sl, "cm": cm,
                    "px": int(cm.sum()), "full": full, "fill": cm.sum() / float(cm.size)})
    return out, Q


def facade_axes(pts, Nc, cen):
    """Dominant facade normal n (horizontal, facing the camera) from the component's normals; u along it."""
    view = -cen / max(np.linalg.norm(cen), 1e-6)
    n = view
    if Nc is not None and len(Nc):
        nh = Nc[:, :2]
        L = np.linalg.norm(nh, axis=1)
        ok = L > 0.6
        if ok.mean() >= 0.25:
            ang = np.arctan2(nh[ok, 1], nh[ok, 0]) % np.pi
            h, e = np.histogram(ang, 36, (0, np.pi), weights=L[ok])
            h = h + np.roll(h, 1) + np.roll(h, -1)
            k = int(np.argmax(h))
            a = (e[k] + e[k + 1]) / 2
            n = np.array([np.cos(a), np.sin(a)])
    if n @ view < 0:
        n = -n
    u = np.array([-n[1], n[0]])
    return u, n


def quad_mask(Q, pm, cam_h, tear=0.08):
    rr = np.linalg.norm(Q - np.array([0, 0, cam_h]), axis=-1)
    q_ok = pm[:-1, :-1] & pm[:-1, 1:] & pm[1:, :-1] & pm[1:, 1:]
    R4 = np.stack([rr[:-1, :-1], rr[:-1, 1:], rr[1:, 1:], rr[1:, :-1]], -1)
    q_ok &= (R4.max(-1) / np.maximum(R4.min(-1), 1e-3)) < (1 + tear)
    a = Q[1:, :-1] - Q[:-1, :-1]
    b = Q[:-1, 1:] - Q[:-1, :-1]
    nrm = np.cross(a, b)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-9)
    vw = Q[:-1, :-1] - np.array([0, 0, cam_h])
    vw /= np.maximum(np.linalg.norm(vw, axis=-1, keepdims=True), 1e-9)
    q_ok &= np.abs((nrm * vw).sum(-1)) > 0.08
    return q_ok


def shell_piece(Q, qm, cam_h, M, c_out, tmax):
    """Closed thin solid from a quad mask on the strided grid: front = measured surface, back pushed along the
    (mostly horizontal) view ray by ≤ ``tmax``, side walls on the boundary edges."""
    h, w = Q.shape[:2]
    idx = np.arange(h * w).reshape(h, w)
    ys, xs = np.nonzero(qm)
    if len(ys) == 0:
        return 0
    quads = np.stack([idx[ys, xs], idx[ys, xs + 1], idx[ys + 1, xs + 1], idx[ys + 1, xs]], -1)
    used = np.unique(quads)
    remap = -np.ones(h * w, np.int64)
    remap[used] = np.arange(len(used))
    Vf = Q.reshape(-1, 3)[used].astype(np.float64)
    q = remap[quads]
    ext = float(np.linalg.norm(np.ptp(Vf[:, :2], 0)))
    t = float(np.clip(0.8 * ext, 0.3, tmax))
    dirs = Vf - np.array([0, 0, cam_h])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs[:, 2] *= 0.3
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    Vb = Vf + dirs * t
    nv = len(Vf)
    V = np.r_[Vf, Vb]
    E = np.r_[q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 3]], q[:, [3, 0]]]
    key = np.sort(E, 1)
    _u, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bnd = E[cnt[inv.ravel()] == 1]
    Fs = np.stack([bnd[:, 0], bnd[:, 1], bnd[:, 1] + nv, bnd[:, 0] + nv], -1)
    M.add(V, np.r_[q[:, ::-1], q + nv, Fs], c_out, {"kind": "skin", "quads": int(len(q)), "t": round(t, 2)})
    return int(len(q))


def skin_quads(P, mask, cls, cam_h, st, r_split=R_SPLIT, tear=0.08, min_q=24):
    """Structure-only skin pieces at stride ``st``: list of (Q, quad mask)."""
    Q = P[::st, ::st].astype(np.float64)
    c = cls[::st, ::st]
    m = mask[::st, ::st]
    r = np.hypot(Q[..., 0], Q[..., 1])
    m = m & (c == STRUCT) & (r < r_split) & (Q[..., 2] > 0.05)
    q_ok = quad_mask(Q, m, cam_h, tear)
    lab, n = label(q_ok, eight=False)
    pieces = []
    for k, sl in enumerate(find_objects(lab, n), 1):
        if sl is None:
            continue
        qm = np.zeros_like(q_ok)
        qm[sl] = lab[sl] == k
        if qm.sum() < max(4, min_q // (st * st // 4 or 1)):
            continue
        pieces.append(qm)
    return Q, pieces


def _skin_tris(pieces):
    tot = 0
    for qm in pieces:
        q = int(qm.sum())
        tot += 4 * q + 4 * int(math.sqrt(q) * 4)
    return tot


# ============================================================================== volumes (V1 rules)

def build_volumes(P, N, mask, cls, T, cam_h, M, st=2, top_q=20, shrink=0.94, windows=True, win_limit=400,
                  r_split=R_SPLIT, avoid=None, occ=None, bands=False):
    """Closed proxies per measured component. ``avoid`` = (x, y, radius) product spot: nothing is built there.
    ``occ`` (``scene_complete.Occupancy``): rule depths are cut back where they would reach free space.
    ``bands`` (近密远疏): components are processed nearest-to-the-product first (the window budget goes to the
    near field) and prisms beyond ``BAND_MID`` m from the product get no procedural windows."""
    cam_xy = np.zeros(2)
    stats = {"buildings": 0, "props": 0, "walls": 0, "slabs": 0, "poles": 0, "trees": 0, "windows": 0, "thin": 0,
             "carved": 0, "carved_m": 0.0}
    ref_xy = np.zeros(2) if avoid is None else np.array([float(avoid[0]), float(avoid[1])])

    def carve(c0, u, n, W, D, z0, z1, d_min=CARVE_MIN_D):
        if occ is None:
            return D
        d = occ.clip_depth(c0, u, n, W, D, z0, z1, d_min=min(d_min, D))
        if d < D - 0.25:
            stats["carved"] += 1
            stats["carved_m"] = round(stats["carved_m"] + (D - d), 1)
        return d

    def blocked(xy, rad=0.0):
        return avoid is not None and math.hypot(xy[0] - avoid[0], xy[1] - avoid[1]) < avoid[2] + rad

    def overlaps(pts_xy):
        # the part's own measured points reach into the product spot (a facade 20 m away never does)
        if avoid is None or not len(pts_xy):
            return False
        return float(np.min(np.hypot(pts_xy[:, 0] - avoid[0], pts_xy[:, 1] - avoid[1]))) < avoid[2]

    comps, Q = components(P, mask, cls, [STRUCT], st, N=N, r_split=r_split)
    if bands:
        comps.sort(key=lambda c: float(np.linalg.norm(np.median(c["pts"][:, :2], 0) - ref_xy)) if len(c["pts"]) else 1e9)
    thin = []
    for c in comps:
        pts = c["pts"]
        if len(pts) < 8:
            continue
        Nc = c["N"]
        cen = np.median(pts[:, :2], 0)
        dist = float(np.linalg.norm(cen))
        z1 = float(np.percentile(pts[:, 2], 97))
        zb = float(np.percentile(pts[:, 2], 3))
        u, n = facade_axes(pts, Nc, cen)
        a = pts[:, :2] @ u
        b = pts[:, :2] @ n
        W = float(a.max() - a.min())
        Dm = float(np.percentile(b, 95) - np.percentile(b, 5))
        g0 = terrain_z(T, *cen)
        if overlaps(pts[:, :2]):
            continue
        if W < 0.7 and z1 - zb > 2.5:
            V, F = cylinder(cen[0], cen[1], max(0.12, W / 2), g0 - 0.3, z1)
            M.add(V, F, C_POLE, {"kind": "pole", "dist": round(dist, 1), "h": round(z1 - g0, 2),
                                 "xy": [round(float(cen[0]), 2), round(float(cen[1]), 2)]})
            stats["poles"] += 1
            continue
        if z1 - g0 < 0.4:
            continue
        hh0 = z1 - g0
        if c["fill"] < 0.42 or (hh0 > 2.5 and W > 2.5 and c["fill"] < 0.6
                                and abs(float(np.median(np.abs(Nc[:, :2] @ n)))) < 0.5):
            thin.append((c, 0.6 if hh0 < 2.5 else 1.0))          # sparse / irregular: measured thin shell only
            stats["thin"] += 1
            continue
        if zb - g0 > 1.5 and z1 - zb < 2.2:                         # floating slab (awning, sign)
            front = float(np.percentile(b, 90))
            c0 = u * (a.min() + a.max()) / 2 + n * front
            V, F = box(c0, u, n, W, max(Dm, 0.3), zb, z1)
            M.add(V, F, C_PROP, {"kind": "slab", "dist": round(dist, 1), "W": round(W, 2), "z": [round(zb, 2), round(z1, 2)]})
            stats["slabs"] += 1
            continue
        if z1 - g0 >= 3.0 and W >= 3.0:                            # building: stepped eaves line of prisms
            binw = max(2.0, W / 14)
            nb = max(1, int(round(W / binw)))
            e = np.linspace(a.min(), a.max(), nb + 1)
            sub = Q[c["sl"]]
            cm = c["cm"]
            colt = []
            for xcol in range(cm.shape[1]):
                rows = np.nonzero(cm[:, xcol])[0]
                if len(rows) < 3:
                    continue
                p = sub[rows[min(1, len(rows) - 1)], xcol]
                colt.append((p[:2] @ u, p[2]))
            colt = np.array(colt) if colt else np.zeros((0, 2))
            segs = []
            for k in range(nb):
                s = (a >= e[k]) & (a <= e[k + 1])
                if s.sum() < 6:
                    continue
                ct = colt[(colt[:, 0] >= e[k]) & (colt[:, 0] <= e[k + 1]), 1] if len(colt) else []
                top = float(np.percentile(ct, top_q)) if len(ct) >= 2 else float(np.percentile(pts[s, 2], 90))
                segs.append([e[k], e[k + 1], top, float(np.percentile(b[s], 30))])
            merged = []
            for sg in segs:
                if merged and abs(merged[-1][2] - sg[2]) < 0.9 and abs(merged[-1][3] - sg[3]) < 1.0 \
                        and abs(merged[-1][1] - sg[0]) < binw * 1.6:
                    m_ = merged[-1]
                    w0, w1 = m_[1] - m_[0], sg[1] - sg[0]
                    m_[2] = (m_[2] * w0 + sg[2] * w1) / (w0 + w1)
                    m_[3] = (m_[3] * w0 + sg[3] * w1) / (w0 + w1)
                    m_[1] = sg[1]
                else:
                    merged.append(list(sg))
            for a0, a1, top, front in merged:
                Ws = a1 - a0
                if top - g0 < 1.0 or Ws < 0.8:
                    continue
                D = float(np.clip(max(Dm, 0.7 * W), 6.0, 18.0)) if top - g0 > 4 else float(np.clip(max(Dm, 0.5 * Ws), 2.0, 10.0))
                c0 = u * (a0 + a1) / 2 + n * front - n * 0.35
                D = carve(c0, u, n, Ws * shrink + 0.05, D, g0, top)
                far = bands and float(np.linalg.norm(c0 - ref_xy)) > BAND_MID
                stats["windows"] += prism_with_windows(
                    M, c0, u, n, Ws * shrink + 0.05, D, g0 - 0.5, top, cam_xy,
                    {"kind": "building", "dist": round(dist, 1), "W": round(Ws, 2), "D": round(D, 1), "top": round(top, 2),
                     "xy": [round(float(c0[0]), 2), round(float(c0[1]), 2)], "u": [round(float(u[0]), 4), round(float(u[1]), 4)],
                     "n": [round(float(n[0]), 4), round(float(n[1]), 4)], "g0": round(g0, 2)},
                    windows=windows and not far, win_limit=max(0, win_limit - stats["windows"]))
                stats["buildings"] += 1
            continue
        hh = z1 - g0
        front = float(np.percentile(b, 85))
        c0 = u * (a.min() + a.max()) / 2 + n * front
        if W >= 6.0 and hh < 2.2:
            V, F = box(c0, u, n, W, carve(c0, u, n, W, max(Dm, 0.3), g0, z1, d_min=0.3), g0 - 0.3, z1)
            M.add(V, F, C_WALL, {"kind": "wall/fence", "dist": round(dist, 1), "W": round(W, 2), "h": round(hh, 2),
                                 "xy": [round(float(c0[0]), 2), round(float(c0[1]), 2)]})
            stats["walls"] += 1
            continue
        dyn = _static_scene().is_dynamic_prop(W, hh)
        if dyn:                                             # 1.2.6 只做场景相关物体: pedestrians / parked cars
            stats["dynamic_skipped"] = stats.get("dynamic_skipped", 0) + 1
            continue
        if 1.4 <= W <= 2.8 and 1.2 <= hh <= 4.0:            # vehicle seen end-on → length assumed
            D = max(Dm, 4.5 if hh < 2.2 else 7.0)
        elif 3.2 <= W <= 6.5 and 1.0 <= hh <= 2.2:          # vehicle side-on
            D = max(Dm, 1.8)
        else:
            D = max(Dm, 0.5 * W, 0.4)
        D = carve(c0, u, n, W, D, g0, z1, d_min=max(0.4, min(Dm, 1.2)))
        V, F = rounded_box(c0, u, n, W, D, g0 - 0.2, z1)
        M.add(V, F, C_PROP, {"kind": "vehicle/prop", "dist": round(dist, 1), "W": round(W, 2), "D": round(D, 2), "h": round(hh, 2),
                                  "xy": [round(float(c0[0]), 2), round(float(c0[1]), 2)]})
        stats["props"] += 1
    vcomps, _ = components(P, mask, cls, [VEG], st, min_px=25, r_split=r_split)
    for ci, c in enumerate(vcomps):
        pts = c["pts"]
        if len(pts) < 5:
            continue
        cen = np.median(pts[:, :2], 0)
        dist = float(np.linalg.norm(cen))
        view = cen / max(dist, 1e-6)
        u = np.array([view[1], -view[0]])
        a = pts[:, :2] @ u
        b = pts[:, :2] @ view
        z1 = float(np.percentile(pts[:, 2], 98))
        zb = float(np.percentile(pts[:, 2], 4))
        g0 = terrain_z(T, *cen)
        Ht = z1 - max(zb, g0)
        if Ht < 0.25:
            continue
        W = float(a.max() - a.min())
        crown_w = max(1.0, min(Ht * 1.1, 9.0))
        nb = max(1, int(round(W / crown_w)))
        e = np.linspace(a.min(), a.max(), nb + 1)
        for k in range(nb):
            s = (a >= e[k]) & (a <= e[k + 1])
            if s.sum() < 5:
                continue
            top = float(np.percentile(pts[s, 2], 97))
            bot = float(np.percentile(pts[s, 2], 5))
            ww = e[k + 1] - e[k]
            rx = max(0.4, ww / 2 * 1.05)
            rz = max(0.3, (top - bot) / 2)
            front = float(np.percentile(b[s], 15))
            ry = rx
            if occ is not None:
                # crown depth along the view ray: the back half must not reach free space (ground seen behind it)
                f0 = u * (e[k] + e[k + 1]) / 2 + view * front
                dd = carve(f0, u, -view, 2 * rx * 0.8, 2 * rx * 0.9 + 0.2, bot, top, d_min=0.6)
                ry = float(np.clip(dd / 1.8, 0.35, rx))
            cxy = u * (e[k] + e[k + 1]) / 2 + view * (front + ry * 0.9)
            if blocked(cxy, rx):
                continue
            V, F = ellipsoid(np.r_[cxy, (top + bot) / 2], rx, rx, rz, seed=ci * 31 + k)
            if ry < rx - 1e-6:          # squash along the view ray only (the measured width stays)
                dxy = V[:, :2] - cxy
                along = dxy @ view
                V[:, :2] = cxy + dxy + np.outer(along * (ry / rx - 1.0), view)
            M.add(V, F, C_TREE, {"kind": "tree crown" if bot - g0 > 0.8 else "bush/hedge", "dist": round(dist, 1),
                                 "r": round(rx, 2), "z": [round(bot, 2), round(top, 2)],
                                 "xy": [round(float(cxy[0]), 2), round(float(cxy[1]), 2)], "g0": round(g0, 2)})
            stats["trees"] += 1
            if bot - g0 > 0.8:
                V, F = cylinder(cxy[0], cxy[1], max(0.12, min(0.35, rx * 0.12)), g0 - 0.2, bot + rz * 0.5, 6)
                M.add(V, F, C_POLE)
    return stats, thin, Q



def _static_scene():
    """1.2.6 只做场景相关物体 helpers (works when this module is loaded by path, as in the tests)."""
    try:
        from . import static_scene as SS
    except ImportError:
        import importlib.util as _ilu
        import os as _os
        import sys as _sys
        SS = _sys.modules.get("wondful_static_scene")
        if SS is None:
            _sp = _ilu.spec_from_file_location("wondful_static_scene",
                                               _os.path.join(_os.path.dirname(__file__), "static_scene.py"))
            SS = _ilu.module_from_spec(_sp)
            _sp.loader.exec_module(SS)
            _sys.modules["wondful_static_scene"] = SS
    return SS


# ============================================================================== top level

def _scene_complete():
    try:
        from . import scene_complete as SC
    except ImportError:          # loaded by path (tests)
        import importlib.util as _ilu
        import os as _os
        _sp = _ilu.spec_from_file_location("scene_complete", _os.path.join(_os.path.dirname(__file__), "scene_complete.py"))
        SC = _ilu.module_from_spec(_sp)
        _sp.loader.exec_module(SC)
    return SC


def build(geo, *, far_z=None, pad=None, scene_kind="outdoor", r_split=R_SPLIT, budget=TRI_BUDGET, z_offset=0.0,
          complete=True, analysis=None, obstacles=(), free_space=False, bands=True, seam=True):
    """Detail blockout from one 精确模式 measurement → dict(ok, reason, V, F, C, parts, stats).

    ``far_z(x, y)``: LLM terrain height in the analysis frame (already offset like the result), used beyond
    ``r_split``. ``pad`` = (x, y, radius, z) product spot in the same frame (ground flattened to z; no part
    built inside). ``z_offset`` lifts every measured height (LLM terrain height under the camera).
    ``complete``: 场景补全 — infer the street the photo never saw (``scene_complete``; ``analysis`` = the ④ analysis
    JSON for road direction / lot widths / poles); the result then carries ``layout`` (analysis frame).
    1.2.5 (ideas from Building Rome, arXiv 2610.08790): ``free_space`` = 空/可见/被挡住 three-state test (rule
    depths cut back out of free space, inferred parts only in hidden / unknown space); ``bands`` = 近密远疏 (fine
    skin / windows near, coarse far); ``seam`` = 重叠衔接 (inferred rows blend into the measured neighbours)."""
    t0 = time.time()
    ok, reason = plausible(geo, scene_kind)
    if not ok:
        return {"ok": False, "reason": reason, "seconds": round(time.time() - t0, 3)}
    P = np.asarray(geo["P"], np.float64).copy()
    P[..., 2] += z_offset
    N = np.asarray(geo["N"], np.float64)
    mask = np.asarray(geo["mask"], bool)
    cls = np.asarray(geo["cls"])
    cam = geo["cam"]
    cam_h = float(cam["height_m"]) + z_offset
    Pg = P.copy()
    Pg[..., 2] -= z_offset                       # the heightfield is built around 0 and lifted afterwards
    fz = None
    if far_z is not None:
        fz = lambda X, Y: np.asarray(far_z(X, Y), np.float64) - z_offset   # noqa: E731
    padg = None if pad is None else (pad[0], pad[1], pad[2], pad[3] - z_offset)
    VT, FT, T = terrain(Pg, mask, cls, float(cam["hfov_deg"]), far_z=fz, pad=padg, r_split=r_split)
    VT = VT.copy()
    VT[:, 2] += z_offset
    T = (T[0], T[1], T[2] + z_offset)
    t_ter = time.time() - t0
    M = Mesh()
    avoid = None if pad is None else (pad[0], pad[1], pad[2])
    st_vol = 2 if max(P.shape[:2]) <= 1000 else 4
    win_limit = 400
    SC = _scene_complete()
    occ = None
    if free_space:
        try:
            occ = SC.Occupancy(geo, z_offset)
        except Exception as exc:     # never lose the blockout over the free-space test
            print(f"[Wondful] 空间三态不可用：{exc}")
    stats, thin, Qv = build_volumes(P, N, mask, cls, T, cam_h, M, st=st_vol, win_limit=win_limit, r_split=r_split,
                                    avoid=avoid, occ=occ, bands=bands)
    vol_tris = M.tris()
    room = budget - len(FT) - vol_tris
    if room < SKIN_BUDGET // 2 and stats["windows"]:
        # over budget: drop the procedural windows first
        M = Mesh()
        stats, thin, Qv = build_volumes(P, N, mask, cls, T, cam_h, M, st=st_vol, windows=False, r_split=r_split,
                                        avoid=avoid, occ=occ, bands=bands)
        vol_tris = M.tris()
        room = budget - len(FT) - vol_tris
    skin_budget = max(4000, min(SKIN_BUDGET, room))
    st = 2
    if bands:
        skin_budget = max(4000, min(SKIN_BUDGET_BANDS, room - (SKIN_RESERVE if complete else 0)))
        # 近密远疏: fine stride for the near skin, twice as coarse beyond SKIN_NEAR_R (overlapping band, no crack)
        rr = np.hypot(P[..., 0], P[..., 1])
        while True:
            Qn, pn = skin_quads(P, mask & (rr < SKIN_NEAR_R), cls, cam_h, st, r_split=r_split)
            Qf, pf = skin_quads(P, mask & (rr >= SKIN_NEAR_R - SKIN_OVERLAP), cls, cam_h, 2 * st, r_split=r_split)
            est = _skin_tris(pn) + _skin_tris(pf)
            if est <= skin_budget or st >= 8:
                break
            st += 1
        skin_sets = [(Qn, pn), (Qf, pf)]
    else:
        while True:
            Q, pieces = skin_quads(P, mask, cls, cam_h, st, r_split=r_split)
            est = _skin_tris(pieces)
            if est <= skin_budget or st >= 8:
                break
            st += 1
        skin_sets = [(Q, pieces)]
    nq = 0
    for Q, pieces in skin_sets:
        for qm in pieces:
            if avoid is not None:
                ys, xs = np.nonzero(qm)
                cxy = np.median(Q[ys, xs, :2], 0)
                if math.hypot(cxy[0] - avoid[0], cxy[1] - avoid[1]) < avoid[2]:
                    continue
            nq += shell_piece(Q, qm, cam_h, M, C_SKIN, SKIN_T)
    # thin irregular structures (fences, railings, palm trunks): measured thin shells at the volume stride
    for c, tm in thin:
        left = budget - len(FT) - M.tris()
        k = 1
        while True:
            qm = quad_mask(Qv[::k, ::k], c["full"][::k, ::k], cam_h)
            if 5 * int(qm.sum()) <= max(0, left) // 2 or k >= 4:
                break
            k += 1
        if 5 * int(qm.sum()) > left:
            continue
        shell_piece(Qv[::k, ::k], qm, cam_h, M, C_SKIN, tm)
    layout = None
    if complete:
        # 场景补全: continue the measured street where the photo did not look (closed, tagged, lower confidence)
        try:
            mview = occ if occ is not None else SC.MasterView(geo, z_offset)
            cst, layout = SC.complete(_Prims(), M, list(M.parts), lambda x, y: terrain_z(T, x, y), mview.visible,
                                      pad=pad, analysis=analysis, budget_left=max(0, budget - len(FT) - M.tris()),
                                      obstacles=obstacles, occupancy=occ, bands=bands, seam=seam)
        except Exception as exc:     # never lose the measured blockout over the inferred part
            cst, layout = {"ok": False, "reason": f"出错：{str(exc)[:60]}"}, None
        stats["complete"] = cst
    M.add(VT, FT, C_GROUND, {"kind": "ground", "tris": int(len(FT))})
    V, F, C = M.arrays()
    stats.update({"skin_stride": st, "skin_quads": nq, "free_space": occ is not None, "bands": bool(bands), "terrain_tris": int(len(FT)), "tris": int(len(F)),
                  "terrain_s": round(t_ter, 3), "r_split": r_split})
    return {"ok": True, "reason": "", "V": V, "F": F, "C": C, "parts": M.parts, "stats": stats,
            "layout": layout, "seconds": round(time.time() - t0, 3)}


class _Prims:
    """This module's primitives for ``scene_complete`` (works the same when the module was exec'd by path)."""

    def __getattr__(self, name):
        return globals()[name]


# ============================================================================== far skyline silhouette (远山剪影)
# User 2026-10-20: 「远山可以相对粗糙一些 只保留形即可」. Only the SHAPE of the far range comes from the
# measurement: the sky boundary's elevation angle per azimuth (direction only, so MoGe's noisy far depth never
# reaches the surface), strongly smoothed, at one constant distance, as a low-poly closed ridge with a crest
# thickness and a back slope. Unreliable measurement (near objects hide the sky boundary, big jumps) → the LLM
# 远景带 stay. Optional: 高级 › 多角度 「远山剪影」 (default off until the real-generation test decides).

SKY_TAG = "wondful_detail_skyline"
SKY_BIN_DEG = 1.0          # azimuth bin of the raw silhouette
SKY_SMOOTH_DEG = 2.5       # gaussian sigma (deg of azimuth): bumps narrower than ~2–3° are removed
SKY_MIN_VALID = 0.5        # share of the photo's azimuth bins whose sky boundary is far-field
SKY_JUMP_DEG = 3.0         # adjacent-bin elevation jump counted as a break
SKY_MAX_JUMP_SHARE = 0.06
SKY_TRI_BUDGET = 2000
SKY_STEP_OUT = 6.0         # azimuth step outside the photo (mirrored wrap)
SKY_DEPTH = 0.35           # ridge depth (front foot → back foot) as a share of the distance
# cross-section (range offset × depth, height share): front foot, front shoulder, crest front, crest back, back foot
SKY_SECTION = ((-0.45, 0.0), (-0.18, 0.72), (0.0, 1.0), (0.08, 1.0), (0.55, 0.0))


def measured_skyline(geo, r_split=R_SPLIT, z_offset=0.0):
    """Sky-boundary silhouette of one measurement → dict(ok, reason, az, el (deg, smoothed), raw, dist, ...).

    Per column the first non-sky pixel under sky gives a direction (azimuth, elevation); it counts only when that
    pixel is far field (horizontal range ≥ ``r_split``), otherwise the near object hides the far range there."""
    P = np.asarray(geo["P"], np.float64)
    mask = np.asarray(geo["mask"], bool)
    cam_h = float(geo["cam"]["height_m"])
    H, W = mask.shape
    out = {"ok": False, "reason": ""}
    top_sky = ~mask[0]
    first = np.argmax(mask, axis=0)
    has = mask.any(axis=0) & top_sky
    cols = np.nonzero(has)[0]
    if len(cols) < W * 0.2:
        out["reason"] = "画面顶部几乎没有天空"
        return out
    rows = first[cols]
    top = P[rows, cols]                          # the boundary pixel: its DIRECTION only (depth-free)
    rng = np.median(np.stack([np.hypot(*P[np.clip(rows + k, 0, H - 1), cols][:, :2].T) for k in range(3)]), 0)
    rt = np.maximum(np.hypot(top[:, 0], top[:, 1]), 1e-6)
    az = np.degrees(np.arctan2(top[:, 0], top[:, 1]))
    el = np.degrees(np.arctan2(top[:, 2] - cam_h, rt))
    far = rng >= float(r_split)
    # photo azimuth span from every column's ray direction (sky or not)
    mid = np.nonzero(mask[H // 2])[0]
    a_all = np.degrees(np.arctan2(P[H // 2, :, 0], P[H // 2, :, 1]))
    a0, a1 = float(np.min(a_all[mid])) if len(mid) else float(az.min()), float(np.max(a_all[mid])) if len(mid) else float(az.max())
    a0, a1 = min(a0, float(az.min())), max(a1, float(az.max()))
    edges = np.arange(a0, a1 + SKY_BIN_DEG, SKY_BIN_DEG)
    nb = max(1, len(edges) - 1)
    raw = np.full(nb, np.nan)
    rr = np.full(nb, np.nan)
    b = np.clip(np.searchsorted(edges, az, side="right") - 1, 0, nb - 1)
    for i in range(nb):
        sel = (b == i) & far
        if sel.sum():
            raw[i] = float(np.median(el[sel]))
            rr[i] = float(np.median(rng[sel]))
    valid = np.isfinite(raw)
    share = float(valid.mean())
    out.update({"valid_share": round(share, 3), "span_deg": [round(a0, 2), round(a1, 2)], "bins": int(nb)})
    if share < SKY_MIN_VALID:
        out["reason"] = f"天际线被近景挡住太多（远景只占 {share:.0%}）"
        return out
    v = raw[valid]
    vm = median_filter1d(v, 3) if len(v) >= 3 else v
    jumps = int((np.abs(np.diff(vm)) > SKY_JUMP_DEG).sum()) if len(vm) > 1 else 0
    out["jumps"] = jumps
    if jumps > max(2, int(SKY_MAX_JUMP_SHARE * len(vm))):
        out["reason"] = f"实测天际线跳变太多（{jumps} 处 > {SKY_JUMP_DEG:.0f}°）"
        return out
    x = np.arange(nb, dtype=np.float64)
    filled = np.interp(x, x[valid], vm)
    sig = SKY_SMOOTH_DEG / SKY_BIN_DEG
    k = np.arange(-int(3 * sig) - 1, int(3 * sig) + 2)
    g = np.exp(-0.5 * (k / sig) ** 2)
    g /= g.sum()
    pad = len(k) // 2
    sm = np.convolve(np.pad(filled, pad, mode="edge"), g, mode="valid")[:nb]
    dist = float(np.clip(np.nanmedian(rr), 1.5 * float(r_split), 3000.0))
    out.update({"ok": True, "az": (edges[:-1] + SKY_BIN_DEG / 2).tolist(), "el": sm.tolist(),
                "raw": [None if not np.isfinite(t) else round(float(t), 3) for t in raw],
                "dist": round(dist, 1), "cam_h": cam_h + float(z_offset),
                "residual_deg": round(float(np.sqrt(np.mean((sm[valid] - vm) ** 2))), 3),
                "el_range_deg": [round(float(sm.min()), 2), round(float(sm.max()), 2)]})
    return out


def _reflect(d, span):
    t = np.mod(d, 2 * span)
    return np.where(t > span, 2 * span - t, t)


def skyline_profile(sk, az_query):
    """Smoothed elevation (deg) at any azimuth: measured inside the photo, mirrored and cross-faded outside (the
    same 360° wrap idea as the LLM 远景带), continuous all the way round."""
    az = np.asarray(sk["az"], np.float64)
    el = np.asarray(sk["el"], np.float64)
    a0, a1 = float(az[0]), float(az[-1])
    span = max(a1 - a0, 1.0)
    q = np.asarray(az_query, np.float64)
    rel = np.mod(q - a0, 360.0)                  # 0 … 360 from the left edge going right
    inside = rel <= span
    res = np.interp(a0 + np.minimum(rel, span), az, el)
    gap = 360.0 - span
    if gap > 1e-6:
        d_r = rel - span                         # distance past the right edge
        d_l = 360.0 - rel                        # distance before the left edge
        right = np.interp(a1 - _reflect(d_r, span), az, el)
        left = np.interp(a0 + _reflect(d_l, span), az, el)
        w = np.clip(1.0 - d_r / gap, 0.0, 1.0)
        out = np.where(inside, res, w * right + (1 - w) * left)
        return out
    return res


def skyline_ridge(sk, foot_z=None, budget=SKY_TRI_BUDGET):
    """Closed low-poly ridge ring (local analysis frame, camera at x = y = 0) → (V, F tris)."""
    az = np.asarray(sk["az"], np.float64)
    a0, a1 = float(az[0]), float(az[-1])
    span = max(a1 - a0, 1.0)
    D = float(sk["dist"])
    cam_h = float(sk["cam_h"])
    ns = len(SKY_SECTION)
    per_seg = 2 * ns
    n_out = int(math.ceil(max(0.0, 360.0 - span) / SKY_STEP_OUT))
    n_in = max(8, min(int(math.ceil(span / SKY_BIN_DEG)), budget // per_seg - n_out))
    a_in = np.linspace(a0, a1, n_in + 1)
    a_out = np.linspace(a1, a0 + 360.0, n_out + 1)[1:-1] if n_out > 1 else np.zeros(0)
    ring = np.r_[a_in, a_out]
    el = skyline_profile(sk, ring)
    crest = cam_h + D * np.tan(np.radians(el))
    if foot_z is None:
        foot_z = min(0.0, float(crest.min())) - 30.0
    crest = np.maximum(crest, foot_z + 5.0)
    dep = SKY_DEPTH * D
    V = []
    th = np.radians(ring)
    for (dr, f) in SKY_SECTION:
        r = D + dr * dep
        z = foot_z + (crest - foot_z) * f
        V.append(np.stack([r * np.sin(th), r * np.cos(th), z], -1))
    V = np.stack(V, 1).reshape(-1, 3)            # vertex (i, j) = i * ns + j
    n = len(ring)
    F = []
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(ns):
            j2 = (j + 1) % ns                    # j = ns-1 → 0 closes the bottom
            a, b_, c, d = i * ns + j, i * ns + j2, i2 * ns + j2, i2 * ns + j
            F.append((a, b_, c))
            F.append((a, c, d))
    return V, np.asarray(F, np.int64)


# ============================================================================== near / far split of the LLM blocks

def llm_role(env_type, dist_xy, r_split=R_SPLIT):
    """What happens to one LLM block in a pass that uses the detail blockout:
    ``keep`` (far field: bands, cliffs, sea, anything beyond ``r_split``) or ``replace`` (ground / near-mid block,
    which the measured geometry stands in for)."""
    t = str(env_type or "")
    if t.startswith(FAR_TYPE_PREFIX) or t in FAR_TYPES:
        return "keep"
    if t in GROUND_TYPES:
        return "replace"
    return "replace" if float(dist_xy) < r_split else "keep"


def is_far_band(env_type) -> bool:
    """LLM 远景带 replaced by the 远山剪影 (mountain / hill / dune bands and their 360° wraps; not the sea)."""
    t = str(env_type or "")
    return t.startswith(FAR_TYPE_PREFIX) and t != "band_sea"


def summary_text(res) -> str:
    if not res:
        return ""
    if not res.get("ok"):
        return f"细节粗模未生成（{res.get('reason', '')}，多角度沿用普通粗模）"
    s = res["stats"]
    sk = res.get("skyline") or {}
    far = "远景沿用 AI 远景带" + ("（远山剪影可用）" if sk.get("ok") else "")
    line = (f"细节粗模 {s['tris'] // 1000}k 面：建筑 {s['buildings']} · 车辆道具 {s['props'] + s['slabs']} · "
            f"树 {s['trees']} · {far}")
    cst = s.get("complete")
    if cst:
        line += " · " + completion_text(cst)
    return line


def completion_text(cst) -> str:
    if not cst:
        return ""
    if not cst.get("ok"):
        return f"场景补全：{cst.get('reason', '未补全')}"
    return (f"场景补全：推测建筑 {cst.get('buildings', 0)} · 绿植 {cst.get('hedges', 0) + cst.get('trees', 0)} · "
            f"电杆 {cst.get('poles', 0)} · 路沿 {cst.get('curbs', 0)}"
            + (f"（{'、'.join(cst['mirrored'])}侧按对侧推测）" if cst.get("mirrored") else ""))


# ============================================================================== Blender side (main thread)

def _world_tris(objs, depsgraph=None):
    import bpy  # noqa: F401
    acc = []
    for ob in objs:
        if ob.type != "MESH":
            continue
        me = ob.data
        me.calc_loop_triangles()
        n = len(me.vertices)
        if n == 0 or not len(me.loop_triangles):
            continue
        co = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3).astype(np.float64)
        lt = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", lt)
        Mw = np.array(ob.matrix_world)
        cw = co @ Mw[:3, :3].T + Mw[:3, 3]
        acc.append(cw[lt.reshape(-1, 3)])
    return np.concatenate(acc) if acc else np.zeros((0, 3, 3))


def remove_carrier():
    import bpy
    for coll in [c for c in bpy.data.collections if c.get(CARRIER_TAG) or c.get(INFER_TAG)]:
        for ob in list(coll.all_objects):
            me = ob.data if ob.type == "MESH" else None
            bpy.data.objects.remove(ob, do_unlink=True)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
        bpy.data.collections.remove(coll)


def find_carrier(scene=None):
    import bpy
    for coll in bpy.data.collections:
        if coll.get(CARRIER_TAG) and len(coll.objects):
            return coll
    return None


def build_in_blender(context, info, geo, scene_kind="outdoor", r_split=R_SPLIT, analysis=None, complete=True,
                     free_space=False, bands=True, seam=True):
    """Build the detail blockout for a just-built ④ blockout (``scene_blockout_build.build`` info dict).

    Objects go to a hidden child collection 「Wondful 细节粗模（投影）」 of the blockout collection (tagged
    ``CARRIER_TAG``; hidden in viewport and render, so the normal flow never sees them), one mesh per class.
    The LLM blocks get ``FAR_TAG`` = 1 when they stay visible in a detail pass (far field). Returns a
    JSON-safe summary (``ok``, ``reason``, ``stats``, ``seconds``)."""
    import bpy
    from mathutils import Matrix, Vector
    from . import massing as MS
    from . import scene_blockout as SB
    remove_carrier()
    t0 = time.time()
    coll = bpy.data.collections.get(info["collection"])
    if coll is None:
        return {"ok": False, "reason": "场景粗模集合不存在"}
    world = Matrix(info["world"])
    winv = world.inverted()
    cam_local_z = float(info.get("cam_local_z", 0.0))
    z_offset = cam_local_z - float(geo["cam"]["height_m"])      # LLM terrain height under the camera
    # LLM terrain sampler (local frame): ray-cast the LLM ground objects (terrain + 外圈 + zones), else z_offset
    ground_objs = [o for o in coll.all_objects if o.get("wondful_env_type") in ("terrain",) and o.type == "MESH"]
    tris = _world_tris(ground_objs)
    sampler = None
    if len(tris):
        loc = tris.reshape(-1, 3) @ np.array(winv)[:3, :3].T + np.array(winv)[:3, 3]
        loc = loc.reshape(-1, 3, 3)
        sampler = _height_sampler(loc)

    def far_z(X, Y):
        if sampler is None:
            return np.full(np.shape(X), z_offset)
        return sampler(X, Y, z_offset)

    sp = info.get("spot_local")
    pad = None
    if sp:
        pad = (float(sp[0]), float(sp[1]), float(info.get("pad_radius", 2.5)), float(sp[2]))
    obstacles = []
    if complete:
        try:
            obstacles = [f[:5] for f in llm_footprints(coll, winv, r_split)]
        except Exception as exc:
            print(f"[Wondful] 场景补全：读取 AI 体块失败：{exc}")
    res = build(geo, far_z=far_z, pad=pad, scene_kind=scene_kind, r_split=r_split, z_offset=z_offset,
                complete=complete, analysis=analysis, obstacles=obstacles, free_space=free_space, bands=bands, seam=seam)
    out = {k: res.get(k) for k in ("ok", "reason", "stats", "seconds")}
    if not res["ok"]:
        print(f"[Wondful] 细节粗模未生成：{res['reason']}（多角度沿用普通粗模）")
        return out
    sub = bpy.data.collections.new(COLLECTION_NAME)
    sub[CARRIER_TAG] = 1
    coll.children.link(sub)
    mat = bpy.data.materials.get(MS.MAT_NAME + "_细节") or MS.make_material(bpy.data.materials.new)
    mat.name = MS.MAT_NAME + "_细节"
    V, F, C = res["V"], res["F"], res["C"]
    Mw = np.array(world)
    Vw = V @ Mw[:3, :3].T + Mw[:3, 3]
    inf_coll = None
    for c in np.unique(C):
        sel = C == c
        if int(c) in C_INFERRED:
            # 场景补全: part of the visible 场景粗模 (the saved blockout holds the completed street)
            if inf_coll is None:
                inf_coll = _inferred_collection(coll)
            _link_mesh(inf_coll, f"补全_{C_NAME.get(int(c), str(c))}", Vw, F[sel], mat, hidden=False,
                       tags=(INFER_TAG,), hint=inferred_hint(int(c), analysis))
            continue
        ob = _link_mesh(sub, f"细节_{C_NAME.get(int(c), str(c))}", Vw, F[sel], mat, hidden=True, tags=(CARRIER_TAG,),
                        hint=C_NAME.get(int(c), ""))
    # 远山剪影 (optional at render time): measured far silhouette ridge, hidden like the rest of the carrier
    try:
        sk = measured_skyline(geo, r_split=r_split, z_offset=z_offset)
    except Exception as exc:
        sk = {"ok": False, "reason": f"出错：{str(exc)[:60]}"}
    if sk.get("ok"):
        Vs, Fs = skyline_ridge(sk)
        Vsw = (Vs @ Mw[:3, :3].T + Mw[:3, 3]).astype(np.float32)
        Fs32 = Fs.astype(np.int32)
        me = bpy.data.meshes.new("细节_远山剪影")
        me.vertices.add(len(Vsw))
        me.vertices.foreach_set("co", Vsw.ravel())
        me.loops.add(Fs32.size)
        me.loops.foreach_set("vertex_index", Fs32.ravel())
        me.polygons.add(len(Fs32))
        me.polygons.foreach_set("loop_start", np.arange(0, Fs32.size, 3, dtype=np.int32))
        me.polygons.foreach_set("loop_total", np.full(len(Fs32), 3, np.int32))
        me.update()
        me.validate()
        me.materials.append(mat)
        ob = bpy.data.objects.new("细节_远山剪影", me)
        ob[CARRIER_TAG] = 1
        ob[SKY_TAG] = 1
        ob["wondful_hint"] = "远山剪影"
        sub.objects.link(ob)
        ob.hide_render = True
        ob.hide_viewport = True
        sk["tris"] = int(len(Fs))
    out["skyline"] = {k: v for k, v in sk.items() if k not in ("az", "el", "raw")}
    print("[Wondful] 远山剪影：" + (f"可用（{sk['tris']} 面，距离 {sk['dist']:.0f} m，远景占 {sk['valid_share']:.0%}）"
                                if sk.get("ok") else f"不可用，沿用 AI 远景带（{sk.get('reason', '')}）"))
    # mark the far LLM blocks (kept in a detail pass)
    kept = 0
    for ob in coll.all_objects:
        if ob.get(CARRIER_TAG) or ob.get(INFER_TAG):
            continue
        try:
            pts = [winv @ (ob.matrix_world @ Vector(c)) for c in ob.bound_box]
        except Exception:
            continue
        cx = sum(p.x for p in pts) / 8.0
        cy = sum(p.y for p in pts) / 8.0
        dmin = min(math.hypot(p.x, p.y) for p in pts)
        dist = max(dmin, math.hypot(cx, cy) * 0.5)
        role = llm_role(ob.get("wondful_env_type"), math.hypot(cx, cy) if dist < r_split else dist, r_split)
        if ob.get("wondful_sb_wrap") and ob.get("wondful_env_type") in GROUND_TYPES:
            role = "replace"
        ob[FAR_TAG] = 1 if role == "keep" else 0
        kept += role == "keep"
    sub["wondful_detail_stats"] = str(res["stats"])
    if res.get("layout"):
        try:
            from . import scene_complete as SC
            import json as _json
            lw = SC.layout_to_world(res["layout"], Mw)
            lw["stats"] = (res["stats"].get("complete") or {})
            (inf_coll if inf_coll is not None else sub)[LAYOUT_KEY] = _json.dumps(lw, ensure_ascii=False, default=float)
        except Exception as exc:
            print(f"[Wondful] 场景补全布局保存失败：{exc}")
    out["stats"]["llm_kept"] = kept
    out["seconds_total"] = round(time.time() - t0, 3)
    print(f"[Wondful] 细节粗模：{res['stats']['tris']} 面（地面 {res['stats']['terrain_tris']}，表皮步长 "
          f"{res['stats']['skin_stride']}），生成 {res['seconds']:.2f} s，保留远景体块 {kept} 个")
    return out


INFER_COLLECTION = "Wondful 场景补全（推测）"
INFER_HINT = {8: "与主视角同一条街的房屋（主视角以外，按看到的街道推测补全）：同样的类型、颜色范围、层数、窗户和屋顶，不要画成空白墙",
              9: "与主视角同样的修剪树篱 / 行道树（推测补全）", 10: "与主视角同样的电线杆（推测补全）",
              11: "与主视角同样的路沿（推测补全）", 12: "房屋窗户（推测补全）"}


def inferred_hint(c, analysis=None) -> str:
    """Design hint of a 场景补全 part: the generic class hint + (1.2.6) the scene's look from the ④ analysis
    (画风 for a voxel / cartoon reference, lit signs for a night street) so a pole or an inferred house is not drawn
    as a plain photoreal one."""
    from .scene_style import hint_prefix
    base = INFER_HINT.get(int(c), C_NAME.get(int(c), ""))
    try:
        return hint_prefix(analysis, building=int(c) in (8, 12)) + base
    except Exception:
        return base


def _inferred_collection(parent):
    import bpy
    sub = bpy.data.collections.new(INFER_COLLECTION)
    sub[INFER_TAG] = 1
    parent.children.link(sub)
    return sub


def remove_inferred():
    """Drop every 场景补全 collection (a new 场景粗模 is being finished)."""
    import bpy
    for old in [c for c in bpy.data.collections if c.get(INFER_TAG)]:
        for ob in list(old.all_objects):
            me = ob.data if ob.type == "MESH" else None
            bpy.data.objects.remove(ob, do_unlink=True)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
        bpy.data.collections.remove(old)


def find_inferred(scene=None):
    """The 场景补全 collection (visible part of the 场景粗模), or None."""
    import bpy
    for coll in bpy.data.collections:
        if coll.get(INFER_TAG) and len(coll.objects):
            return coll
    return None


def completion_layout(scene=None):
    """World-frame 场景补全 layout (dict) of the current 场景粗模, or None."""
    import json as _json
    import bpy
    for coll in bpy.data.collections:
        raw = coll.get(LAYOUT_KEY) if (coll.get(INFER_TAG) or coll.get(CARRIER_TAG)) else None
        if raw:
            try:
                return _json.loads(raw)
            except ValueError:
                return None
    return None


def _link_mesh(coll, name, Vw, F, mat, hidden=True, tags=(), hint=""):
    import bpy
    used = np.unique(F)
    remap = -np.ones(len(Vw), np.int64)
    remap[used] = np.arange(len(used))
    Vc = np.asarray(Vw)[used].astype(np.float32)
    Fl = remap[F].astype(np.int32)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(Vc))
    me.vertices.foreach_set("co", Vc.ravel())
    me.loops.add(Fl.size)
    me.loops.foreach_set("vertex_index", Fl.ravel())
    me.polygons.add(len(Fl))
    me.polygons.foreach_set("loop_start", np.arange(0, Fl.size, 3, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(len(Fl), 3, np.int32))
    try:      # flat faces: the normal pass of a box must not be smoothed across its corners
        me.polygons.foreach_set("use_smooth", np.zeros(len(Fl), bool))
    except Exception:
        pass
    me.update()
    me.validate()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    for t in tags:
        ob[t] = 1
    ob["wondful_hint"] = hint
    coll.objects.link(ob)
    ob.hide_render = bool(hidden)
    ob.hide_viewport = bool(hidden)
    return ob


BUILDING_ENV = ("building", "house", "tower", "shop", "wall", "kiosk", "pavilion")
NON_OBSTACLE_ENV = GROUND_TYPES | {"band_far", "band_mid", "band_near", "band_sea", "cliff", "sky"}


def llm_footprints(coll, winv, r_max=R_SPLIT):
    """LLM blocks of the 场景粗模 as local-frame footprints [(c, u, n, W, D, ob)] (front = the −y face)."""
    from mathutils import Vector
    out = []
    for ob in coll.all_objects:
        if ob.type != "MESH" or ob.get(CARRIER_TAG) or ob.get(INFER_TAG):
            continue
        et = str(ob.get("wondful_env_type") or "")
        if not et or et in NON_OBSTACLE_ENV or et.startswith(FAR_TYPE_PREFIX):
            continue
        bb = [winv @ (ob.matrix_world @ Vector(c)) for c in ob.bound_box]
        c0, c3, c4 = np.array(bb[0][:2]), np.array(bb[3][:2]), np.array(bb[4][:2])
        W = float(np.linalg.norm(c4 - c0))
        D = float(np.linalg.norm(c3 - c0))
        if W < 0.2 or D < 0.05:
            continue
        mid = (c0 + c4) / 2
        if math.hypot(*mid) > r_max:
            continue
        u = (c4 - c0) / W
        n = -(c3 - c0) / D
        zs = [p.z for p in bb]
        out.append((mid, u, n, W, D, ob, min(zs), max(zs), et))
    return out


def complete_llm_in_blender(context, info, analysis=None, bands=True, seam=True):
    """场景补全 for a 场景粗模 without a measurement (精确模式 off / failed): building rows / curbs / greenery
    inferred from the LLM blocks, visible in the blockout. Visibility = the reference camera's frustum + a ray
    test against the blockout. Returns a JSON-safe summary."""
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Matrix, Vector
    from . import massing as MS
    from . import scene_complete as SC
    t0 = time.time()
    coll = bpy.data.collections.get(info.get("collection", ""))
    if coll is None or not info.get("world"):
        return {"ok": False, "reason": "场景粗模集合不存在"}
    remove_inferred()
    world = Matrix(info["world"])
    winv = world.inverted()
    scene = context.scene
    cam = scene.objects.get(info.get("camera_name", "")) or scene.camera
    if cam is None:
        return {"ok": False, "reason": "没有参考相机"}
    feet = llm_footprints(coll, winv)
    camxy = np.array((winv @ cam.matrix_world.translation)[:2])
    parts = []
    for mid, u, n, W, D, ob, z0, z1, et in feet:
        if not any(w in et for w in BUILDING_ENV) or "wall" in et:
            continue
        # front = the long face turned most toward the camera
        best = None
        for cc, uu, nn, ww, dd in ((mid, u, n, W, D), (mid - n * D + 0 * u, -u, -n, W, D),
                                   (mid - n * D / 2 + u * W / 2, -n, u, D, W), (mid - n * D / 2 - u * W / 2, n, -u, D, W)):
            to = camxy - cc
            sc = float(nn @ to) / max(1e-6, float(np.linalg.norm(to)))
            if best is None or sc > best[0]:
                best = (sc, cc, uu, nn, ww, dd)
        _sc, cc, uu, nn, ww, dd = best
        parts.append({"kind": "building", "xy": [float(cc[0]), float(cc[1])], "u": [float(uu[0]), float(uu[1])],
                      "n": [float(nn[0]), float(nn[1])], "W": ww, "D": dd, "top": z1, "g0": z0 + 0.0,
                      "dist": round(float(np.hypot(*cc)), 1)})
    ground_objs = [o for o in coll.all_objects if o.get("wondful_env_type") in ("terrain",) and o.type == "MESH"]
    tris = _world_tris(ground_objs)
    sampler = None
    if len(tris):
        loc = tris.reshape(-1, 3) @ np.array(winv)[:3, :3].T + np.array(winv)[:3, 3]
        sampler = _height_sampler(loc.reshape(-1, 3, 3))
    zcam = float(info.get("cam_local_z", 1.6))

    def ground(x, y):
        return float(sampler(np.array([x]), np.array([y]), 0.0)[0]) if sampler is not None else 0.0
    dg = context.evaluated_depsgraph_get()
    origin = cam.matrix_world.translation

    def visible(Q):
        out = []
        for q in np.asarray(Q).reshape(-1, 3):
            pw = world @ Vector((float(q[0]), float(q[1]), float(q[2])))
            c = world_to_camera_view(scene, cam, pw)
            if not (c.z > 0.05 and -0.02 <= c.x <= 1.02 and -0.02 <= c.y <= 1.02):
                out.append(False)
                continue
            d = pw - origin
            dist = d.length
            hit, loc, _n, _i, _o, _m = scene.ray_cast(dg, origin, d.normalized(), distance=dist)
            out.append(not (hit and (loc - origin).length < dist * 0.97 - 0.3))
        return np.array(out, bool)
    sp = info.get("spot_local")
    pad = (float(sp[0]), float(sp[1]), float(info.get("pad_radius", 2.5)), float(sp[2])) if sp else None
    M = Mesh()
    obstacles = [(f[0], f[1], f[2], f[3], f[4]) for f in feet]
    try:
        cst, layout = SC.complete(_Prims(), M, parts, ground, visible, pad=pad, analysis=analysis, obstacles=obstacles,
                                  bands=bands, seam=seam)
    except Exception as exc:
        return {"ok": False, "reason": f"出错：{str(exc)[:60]}"}
    if not cst.get("ok"):
        return cst
    V, F, C = M.arrays()
    Mw = np.array(world)
    Vw = V @ Mw[:3, :3].T + Mw[:3, 3]
    mat = bpy.data.materials.get(MS.MAT_NAME) or MS.make_material(bpy.data.materials.new)
    sub = _inferred_collection(coll)
    for c in np.unique(C):
        sel = C == c
        _link_mesh(sub, f"补全_{C_NAME.get(int(c), str(c))}", Vw, F[sel], mat, hidden=False, tags=(INFER_TAG,),
                   hint=inferred_hint(int(c), analysis))
    import json as _json
    lw = SC.layout_to_world(layout, Mw)
    lw["stats"] = cst
    sub[LAYOUT_KEY] = _json.dumps(lw, ensure_ascii=False, default=float)
    cst["seconds"] = round(time.time() - t0, 2)
    print("[Wondful] " + SC.summary_text(cst))
    return cst


def _height_sampler(tris):
    """Vectorised top-down height lookup over triangles (local frame): bins triangles on a coarse polar-free
    xy grid and tests barycentrics. Returns ``f(X, Y, default)``."""
    t = np.asarray(tris, np.float64)
    if len(t) > 60000:
        t = t[np.linspace(0, len(t) - 1, 60000).astype(int)]
    mn = t[..., :2].min(1)
    mx = t[..., :2].max(1)

    def f(X, Y, default):
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        out = np.full(X.shape, np.nan)
        xs, ys = X.ravel(), Y.ravel()
        res = np.full(xs.shape, np.nan)
        # chunk over points; candidate triangles via bbox test
        for i0 in range(0, len(xs), 256):
            px, py = xs[i0:i0 + 256], ys[i0:i0 + 256]
            cand = (mn[None, :, 0] <= px[:, None]) & (mx[None, :, 0] >= px[:, None]) & \
                   (mn[None, :, 1] <= py[:, None]) & (mx[None, :, 1] >= py[:, None])
            pi, ti = np.nonzero(cand)
            if not len(pi):
                continue
            a, b, c = t[ti, 0], t[ti, 1], t[ti, 2]
            v0 = b[:, :2] - a[:, :2]
            v1 = c[:, :2] - a[:, :2]
            v2 = np.stack([px[pi], py[pi]], -1) - a[:, :2]
            den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            ok = np.abs(den) > 1e-12
            den = np.where(ok, den, 1.0)
            l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
            l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
            ins = ok & (l1 >= -1e-6) & (l2 >= -1e-6) & (l1 + l2 <= 1 + 1e-6)
            z = a[:, 2] + l1 * (b[:, 2] - a[:, 2]) + l2 * (c[:, 2] - a[:, 2])
            z = np.where(ins, z, -np.inf)
            best = np.full(len(px), -np.inf)
            np.maximum.at(best, pi, z)
            best[~np.isfinite(best)] = np.nan
            res[i0:i0 + 256] = best
        res = np.where(np.isnan(res), default, res)
        out = res.reshape(X.shape)
        return out
    return f


class detail_pass:
    """Context manager for one render pass that uses the detail blockout: the carrier objects render, the LLM
    blocks it replaces (and ``extra_hide``, e.g. the temporary clay floor) are hidden; everything is restored on
    exit. A no-op when no carrier exists."""

    def __init__(self, massing_coll, extra_hide=(), skyline=False):
        self.massing_coll = massing_coll
        self.extra = list(extra_hide or ())
        self.skyline = bool(skyline)     # 远山剪影: the measured ridge stands in for the LLM 远景带 (not the sea)
        self.undo = []
        self.active = False
        self.skyline_used = False

    def __enter__(self):
        if self.massing_coll is None:
            return self
        carrier = find_carrier()
        if carrier is None:
            return self
        try:
            objs = list(self.massing_coll.all_objects)
        except Exception:
            return self
        if not any(o.get(CARRIER_TAG) for o in objs):
            # the carrier belongs to a blockout outside this massing collection
            return self
        sky = self.skyline and any(o.get(SKY_TAG) for o in objs)
        for o in objs:
            if o.get(CARRIER_TAG):
                if o.get(SKY_TAG) and not sky:
                    continue                    # 远山剪影 off: the ridge stays hidden
                self.undo.append((o, o.hide_render, o.hide_viewport))
                o.hide_render = False
                o.hide_viewport = False
            elif not o.hide_render and o.get("wondful_env_type") is not None and (
                    int(o.get(FAR_TAG, 0) or 0) == 0 or (sky and is_far_band(o.get("wondful_env_type")))):
                self.undo.append((o, o.hide_render, o.hide_viewport))
                o.hide_render = True
        self.skyline_used = sky
        for o in self.extra:
            try:
                self.undo.append((o, o.hide_render, o.hide_viewport))
                o.hide_render = True
            except Exception:
                pass
        self.active = True
        return self

    def __exit__(self, *exc):
        for o, hr, hv in reversed(self.undo):
            try:
                o.hide_render = hr
                o.hide_viewport = hv
            except Exception:
                pass
        self.undo = []
        return False
