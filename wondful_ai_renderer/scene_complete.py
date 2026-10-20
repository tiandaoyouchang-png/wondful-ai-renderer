"""④ 场景粗模 · 场景补全 (1.2.5, 高级 › 多角度): infer the parts of the scene the reference photo never saw.

The measured 细节粗模 only covers what the photo shows; an orbit camera that turns away from it used to see bare
ground and the plain backs of measured prisms, which the image model painted as blank white walls — and later
views re-projected those inventions (user, 2026-10-20: 「要按照当前能看到的场景推测看不到的场景里有什么 把场景
模型补全」). This module continues the street the photo shows:

- **street frame**: the road direction / width from the analysis road nearest the product (fallback: the measured
  facade direction), origin on the road next to the product;
- **building rows**: per street side the measured facade setback, eaves heights, depth (and the analysis lot
  widths) continue at the measured rhythm along the whole street, behind the camera included; a side with no
  measured building copies the other side (lower confidence) when the scene is clearly a street;
- **curbs, hedges, trees, poles** continue at the measured / analysed rhythm;
- nothing is placed where the reference camera sees (projection test against the measured depth), inside a
  measured prism, or inside the clear circle around the product (orbit cameras stand there).

Every inferred part is a closed primitive tagged with a confidence (``conf``); the layout (boxes, street frame,
facade references for texture patches) is returned JSON-safe so 多角度场景一致 can tell, per pixel, what an
uncovered area should look like and fill it with a style patch sampled from the 主视角.

Pure numpy, analysis frame of ``precise_depth`` (camera at (0, 0, h), looking +Y, Z up). ``prims`` is the
``detail_blockout`` module (box / rounded_box / ellipsoid / cylinder / prism_with_windows / Mesh).
"""
from __future__ import annotations

import math

import numpy as np

# part classes for inferred geometry (continue detail_blockout's C_* numbering; separate objects in Blender)
C_INF_BUILDING, C_INF_VEG, C_INF_POLE, C_INF_CURB, C_INF_WINDOW = 8, 9, 10, 11, 12
C_INF_NAME = {C_INF_BUILDING: "推测建筑", C_INF_VEG: "推测绿植", C_INF_POLE: "推测电杆", C_INF_CURB: "推测路沿",
              C_INF_WINDOW: "推测窗"}
INFER_CLASSES = tuple(C_INF_NAME)

S_BACK = 90.0              # how far the rows continue along the street (m, both directions from the product)
CLEAR_EXTRA = 3.0          # clear circle around the product = master camera distance + this (orbit cameras)
NEAR_STATS = 70.0          # measured parts within this distance of the product drive the side statistics
MIN_BUILDINGS = 2          # fewer measured building prisms (and < MIN_FRONTAGE m of facade) → not built up, no rows
MIN_FRONTAGE = 15.0
CONF_SIDE = 0.55           # inferred from the same side's measurement
CONF_MIRROR = 0.35         # copied from the opposite side
CONF_CURB = 0.5
WINDOW_LIMIT = 260         # procedural windows on inferred facades (triangle budget)
LOT_DEFAULT = 7.5          # lot width when the analysis gives none (m)
# 近密远疏 / 重叠衔接 (ideas from Building Rome, arXiv 2610.08790)
BAND_NEAR = 25.0           # m from the product (along the street): full detail
BAND_MID = 55.0            # beyond: lots merged ×FAR_LOT_SCALE, no windows / front hedges, long curb pieces
FAR_LOT_SCALE = 1.8
CURB_LEN = (8.0, 16.0, 24.0)   # curb piece length near / mid / far
SEAM_LEN = 14.0            # an inferred lot within this distance (along the street) of a measured prism blends into it
SEAM_TOP_MAX = 6.0         # … only when the eaves step is plausible for neighbours (else the rhythm stays)


# ============================================================================== helpers

def _hash01(*k) -> float:
    """Deterministic pseudo random in [0, 1) (no RNG state: the same scene always gets the same layout)."""
    x = 0.0
    for i, v in enumerate(k):
        x += (float(v) + 0.618 * i) * (12.9898 + 7.233 * i)
    return (math.sin(x) * 43758.5453) % 1.0


def _unit(v):
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class MasterView:
    """Can the reference camera see a point? Projection into the measured frame + depth test against MoGe."""

    def __init__(self, geo, z_offset=0.0):
        cam = geo["cam"]
        self.P = np.asarray(geo["P"], np.float64)
        self.mask = np.asarray(geo["mask"], bool)
        self.H, self.W = self.mask.shape
        self.h = float(cam["height_m"]) + float(z_offset)
        self.z_offset = float(z_offset)
        p = math.radians(float(cam.get("pitch_deg", 0) or 0))
        r = -math.radians(float(cam.get("roll_deg", 0) or 0))       # sign checked against the measured pixels
        self.th = math.tan(math.radians(float(cam["hfov_deg"])) / 2)
        self.tv = self.th * self.H / self.W
        fwd = np.array([0.0, math.cos(p), math.sin(p)])
        up0 = np.array([0.0, -math.sin(p), math.cos(p)])
        right0 = np.array([1.0, 0.0, 0.0])
        self.fwd = fwd
        self.right = right0 * math.cos(r) + up0 * math.sin(r)
        self.up = -right0 * math.sin(r) + up0 * math.cos(r)
        rng = np.linalg.norm(self.P - np.array([0.0, 0.0, float(cam["height_m"])]), axis=-1)
        self.range = np.where(self.mask, rng, np.inf)

    def pixel(self, Q):
        Q = np.asarray(Q, np.float64).reshape(-1, 3)
        D = Q - np.array([0.0, 0.0, self.h])
        zc = D @ self.fwd
        zs = np.where(np.abs(zc) < 1e-6, 1e-6, zc)
        u = ((D @ self.right) / zs / self.th + 1) / 2 * self.W - 0.5
        v = (1 - (D @ self.up) / zs / self.tv) / 2 * self.H - 0.5
        return u, v, zc, np.linalg.norm(D, axis=-1)

    def visible(self, Q, margin=0.02):
        """Bool per point: inside the photo frame and not behind the measured surface (sky counts as seen)."""
        u, v, zc, rng = self.pixel(Q)
        mw, mh = margin * self.W, margin * self.H
        inside = (zc > 0.05) & (u >= -mw) & (u <= self.W - 1 + mw) & (v >= -mh) & (v <= self.H - 1 + mh)
        ui = np.clip(np.round(u).astype(int), 0, self.W - 1)
        vi = np.clip(np.round(v).astype(int), 0, self.H - 1)
        # horizontal neighbourhood minimum: a block peeking past a facade's side edge counts as seen (not vertical:
        # on grazing ground the pixel below is much nearer and would hide everything standing on it)
        best = np.full(len(ui), np.inf)
        for dx in (-2, 0, 2):
            best = np.minimum(best, self.range[vi, np.clip(ui + dx, 0, self.W - 1)])
        return inside & (rng < best * 0.97 + 0.3)


# ------------------------------------------------------------------------------ 空 / 可见 / 被挡住 (three states)
# Idea borrowed from Building Rome (arXiv 2610.08790; idea only, no code / weights): every point of space the
# reference camera looks at is one of free (a camera ray passed through it before hitting the measured surface),
# visible (on the measured surface, within a tolerance) or hidden (behind it). Outside the photo it is unknown.
# Inferred parts may only go to hidden / unknown space, and measured volumes whose rule depth (a building prism's
# 6–18 m, a car seen end-on's 4.5 m) pokes into free space are cut back — no walls / hedges growing where the photo
# shows open ground.

S_UNKNOWN, S_FREE, S_VISIBLE, S_HIDDEN = 0, 1, 2, 3
FREE_REL = 0.05            # a point is free when it is nearer than the measured surface by this share …
FREE_ABS = 0.3             # … plus this (m): MoGe noise / edge bleeding never counts as free
CARVE_STEP = 0.5           # depth sampling step behind a measured front (m)
CARVE_MIN_D = 0.8          # a carved prism keeps at least this depth (closed, still reads as a volume)
CARVE_Q = 25               # percentile over the prism's sample columns where free space begins


class Occupancy(MasterView):
    """Three-state test against the 精确模式 measurement. Sky pixels count as free (the ray went to the sky)."""

    def __init__(self, geo, z_offset=0.0):
        super().__init__(geo, z_offset)
        R = self.range
        Rn, Rx = R.copy(), np.where(np.isfinite(R), R, -np.inf)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                sh = np.roll(np.roll(R, dy, 0), dx, 1)
                Rn = np.minimum(Rn, sh)
                Rx = np.maximum(Rx, np.where(np.isfinite(sh), sh, -np.inf))
        self.r_min = Rn                    # nearest measured range in the 3×3 neighbourhood (inf = all sky)
        self.r_max = np.where(np.isfinite(Rx), Rx, np.inf)

    def state(self, Q):
        """S_* per point (n,)."""
        u, v, zc, rng = self.pixel(Q)
        inside = (zc > 0.05) & (u >= 0) & (u <= self.W - 1) & (v >= 0) & (v <= self.H - 1)
        ui = np.clip(np.round(u).astype(int), 0, self.W - 1)
        vi = np.clip(np.round(v).astype(int), 0, self.H - 1)
        rn, rx = self.r_min[vi, ui], self.r_max[vi, ui]
        out = np.full(len(rng), S_UNKNOWN, np.int8)
        free = inside & (rng < rn * (1 - FREE_REL) - FREE_ABS)
        hidden = inside & ~free & np.isfinite(rx) & (rng > rx * (1 + FREE_REL) + FREE_ABS)
        out[inside] = S_VISIBLE
        out[free] = S_FREE
        out[hidden] = S_HIDDEN
        return out

    def free(self, Q):
        return self.state(Q) == S_FREE

    def clip_depth(self, c, u, n, W, D, z0, z1, d_min=CARVE_MIN_D):
        """Largest depth ≤ D (behind the front face at ``c``, along −n) that stays out of free space."""
        if D <= d_min:
            return float(D)
        c = np.asarray(c, np.float64)
        u = np.asarray(u, np.float64)
        n = np.asarray(n, np.float64)
        bs = np.arange(CARVE_STEP, D + 1e-6, CARVE_STEP)
        A = np.linspace(-0.45, 0.45, 5) * W
        Z = np.linspace(z0 + min(0.35, (z1 - z0) / 3), z1 - 0.05, 4)
        AA, ZZ, BB = np.meshgrid(A, Z, bs, indexing="ij")
        xy = c[None] + u[None] * AA.reshape(-1, 1) - n[None] * BB.reshape(-1, 1)
        Q = np.c_[xy, ZZ.reshape(-1)]
        fr = self.free(Q).reshape(len(A) * len(Z), len(bs))
        first = np.where(fr.any(1), bs[np.argmax(fr, 1)] - CARVE_STEP, D)
        d = float(np.percentile(first, CARVE_Q))
        return float(np.clip(d, d_min, D))


def _box_samples(c, u, n, W, D, z0, z1):
    """Corners, edge midpoints and face centres of a box (front centre c, along u, depth along −n)."""
    pts = []
    for a in (-0.5, 0.0, 0.5):
        for b in (0.0, -0.5, -1.0):
            for z in (min(z0 + 0.3, (z0 + z1) / 2), (z0 + z1) / 2, z1):
                p = np.asarray(c) + np.asarray(u) * a * W + np.asarray(n) * b * D
                pts.append((p[0], p[1], z))
    return np.array(pts)


def _dense_samples(c, u, n, W, D, z0, z1, k=5, kz=4):
    """k×k×kz grid through the whole box volume (the three-state test: no part may touch free space)."""
    a = np.linspace(-0.5, 0.5, k)
    b = np.linspace(0.0, -1.0, k)
    z = np.linspace(min(z0 + 0.3, (z0 + z1) / 2), z1, kz)
    A, B, Z = np.meshgrid(a, b, z, indexing="ij")
    xy = np.asarray(c)[None] + np.asarray(u)[None] * (A.reshape(-1, 1) * W) + np.asarray(n)[None] * (B.reshape(-1, 1) * D)
    return np.c_[xy, Z.reshape(-1)]


def _footprint_pts(c, u, n, W, D, k=4):
    a = np.linspace(-0.5, 0.5, k)
    b = np.linspace(0.0, -1.0, k)
    A, B = np.meshgrid(a, b)
    return np.asarray(c)[None] + np.asarray(u)[None] * (A.ravel()[:, None] * W) + np.asarray(n)[None] * (B.ravel()[:, None] * D)


def _in_rects(pts, rects, pad=0.3):
    """Any footprint point inside any oriented rectangle (c, u, n, W, D)?"""
    for c, u, n, W, D in rects:
        d = pts - np.asarray(c)[None]
        a = d @ np.asarray(u)
        b = d @ np.asarray(n)
        if np.any((np.abs(a) <= W / 2 + pad) & (b <= pad) & (b >= -D - pad)):
            return True
    return False


# ============================================================================== street frame

def street_frame(parts, pad, analysis=None):
    """(origin a0 (2,), direction t (2,), road half width, source) of the street through the product."""
    px, py = (float(pad[0]), float(pad[1])) if pad is not None else (0.0, 8.0)
    best = None
    for rd in (analysis or {}).get("roads", []) or []:
        if str(rd.get("kind", "road")) not in ("road", "road_branch", "street", "path"):
            continue
        pts = np.array([[float(p.get("x", 0)), float(p.get("y", 0))] for p in rd.get("points", []) or []], np.float64)
        if len(pts) < 2:
            continue
        # nearest point on the polyline
        dmin, foot, seg = 1e9, None, None
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            ab = b - a
            t = float(np.clip(((np.array([px, py]) - a) @ ab) / max(ab @ ab, 1e-9), 0, 1))
            q = a + ab * t
            d = float(np.hypot(q[0] - px, q[1] - py))
            if d < dmin:
                dmin, foot, seg = d, q, i
        width = float(rd.get("width_m", 8) or 8)
        if dmin > max(25.0, width):
            continue
        # direction: principal axis of the polyline points within 60 m of the foot
        near = pts[np.hypot(pts[:, 0] - foot[0], pts[:, 1] - foot[1]) < 60]
        if len(near) < 2:
            near = pts[max(0, seg): seg + 2]
        d = near - near.mean(0)
        w, v = np.linalg.eigh(d.T @ d)
        t = _unit(v[:, -1])
        score = dmin - 0.2 * width
        if best is None or score < best[0]:
            best = (score, foot, t, width / 2)
    if best is not None:
        return np.asarray(best[1]), np.asarray(best[2]), float(np.clip(best[3], 2.5, 12.0)), "analysis"
    # fallback: mean measured facade direction (sign-aligned), road edge 2 m in front of the nearest facade
    us, ws = [], []
    for p in parts:
        if p.get("kind") == "building" and p.get("u") is not None:
            us.append(np.asarray(p["u"], np.float64))
            ws.append(float(p.get("W", 1.0)) / max(1.0, float(p.get("dist", 10)) / 20.0))
    if not us:
        return np.array([px, py]), np.array([0.0, 1.0]), 4.0, "default"
    ref = us[int(np.argmax(ws))]
    acc = sum(w * (u if u @ ref >= 0 else -u) for u, w in zip(us, ws))
    return np.array([px, py]), _unit(acc), 4.0, "facades"


# ============================================================================== side statistics

def side_stats(parts, a0, t, pad, analysis=None):
    """Per side (+1 right of the street direction, −1 left): measured setback / heights / depth / rhythm."""
    nperp = np.array([t[1], -t[0]])
    px, py = (float(pad[0]), float(pad[1])) if pad is not None else (0.0, 8.0)
    out = {1: {"off": [], "tops": [], "D": [], "s": [], "hedges": 0, "trees": [], "poles": [], "lots": []},
           -1: {"off": [], "tops": [], "D": [], "s": [], "hedges": 0, "trees": [], "poles": [], "lots": []}}
    for p in parts:
        if p.get("xy") is None:
            continue
        xy = np.asarray(p["xy"], np.float64)
        if math.hypot(xy[0] - px, xy[1] - py) > NEAR_STATS:
            continue
        off = float((xy - a0) @ nperp)
        s = float((xy - a0) @ t)
        sd = 1 if off >= 0 else -1
        k = p.get("kind")
        if k == "building":
            out[sd]["off"].append(abs(off))
            out[sd]["tops"].append(float(p["top"]) - float(p.get("g0", 0.0)))
            out[sd]["D"].append(float(p.get("D", 10)))
            out[sd]["s"].append(s)
        elif k == "bush/hedge":
            out[sd]["hedges"] += 1
        elif k == "tree crown":
            out[sd]["trees"].append(float(p.get("r", 2.0)))
        elif k == "pole":
            out[sd]["poles"].append((s, float(p.get("h", 8.0))))
    an = analysis or {}
    for pr in an.get("primitives", []) or []:
        try:
            xy = np.array([float(pr.get("x", 0)), float(pr.get("y", 0))])
        except (TypeError, ValueError):
            continue
        if math.hypot(xy[0] - px, xy[1] - py) > NEAR_STATS:
            continue
        off = float((xy - a0) @ nperp)
        sd = 1 if off >= 0 else -1
        ty = str(pr.get("type", ""))
        if ty == "building":
            w = float(pr.get("w", 0) or 0)
            d = float(pr.get("d", 0) or 0)
            front = max(w, d) if abs(abs(float(pr.get("rot_z_deg", 0) or 0)) % 180 - 90) > 45 else min(w, d)
            if 3.0 <= max(w, d) <= 30.0:
                out[sd]["lots"].append(float(np.clip(w if w > 0 else front, 4.0, 16.0)))
        elif ty in ("pole", "lamp_post", "utility_pole") or "电杆" in str(pr.get("label_zh", "")) \
                or "杆" == str(pr.get("label_zh", ""))[-1:]:
            out[sd]["poles"].append((float((xy - a0) @ t), float(pr.get("h", 8.0) or 8.0)))
    for fm in an.get("foliage_masses", []) or []:
        try:
            xy = np.array([float(fm.get("x", 0)), float(fm.get("y", 0))])
        except (TypeError, ValueError):
            continue
        if math.hypot(xy[0] - px, xy[1] - py) > NEAR_STATS:
            continue
        sd = 1 if float((xy - a0) @ nperp) >= 0 else -1
        if str(fm.get("kind", "")) in ("hedge", "shrub", "bush"):
            out[sd]["hedges"] += 1
        elif str(fm.get("kind", "")) in ("deciduous", "conifer", "palm", "tree"):
            out[sd]["trees"].append(float(np.clip(float(fm.get("crown_w", 5) or 5) / 2, 1.2, 4.0)))
    return out


def _summ(st):
    if not st["off"]:
        return None
    off = np.array(st["off"])
    tops = np.clip(np.array(st["tops"]), 2.6, 30.0)
    return {"setback": float(np.percentile(off, 25)), "top_med": float(np.median(tops)),
            "tops": [float(x) for x in np.percentile(tops, [20, 40, 60, 80])],
            "D": float(np.clip(np.median(st["D"]), 6.0, 14.0)),
            "lot": float(np.clip(np.median(st["lots"]), 5.0, 14.0)) if st["lots"] else LOT_DEFAULT,
            "s": [float(min(st["s"])), float(max(st["s"]))], "n": len(st["off"])}


# ============================================================================== completion

def complete(prims, M, parts, ground, visible, *, pad=None, analysis=None, budget_left=10 ** 9, windows=True,
             obstacles=(), occupancy=None, bands=False, seam=False):
    """Add inferred closed parts to ``M`` (a ``prims.Mesh``); returns (stats, layout) — layout in the analysis
    frame, JSON-safe. ``ground(x, y)`` → ground height; ``visible(points (n,3))`` → bool per point (the reference
    camera sees it: ``MasterView.visible`` with a measurement, a frustum test without); ``obstacles`` = extra
    footprints (c, u, n, W, D) nothing may overlap (the LLM blocks). Never raises for a scene it cannot
    complete: stats["ok"] False + reason.
    ``occupancy`` (``Occupancy``): a part is refused when any of its dense samples is free or on the measured
    surface (only hidden / unknown space is filled). ``bands``: 近密远疏 (merged far lots, windows near-first, no
    far windows / front hedges, longer far curbs). ``seam``: 重叠衔接 — the lots next to a measured prism take
    its eaves height and setback and blend into the street rhythm over ``SEAM_LEN``."""
    stats = {"ok": False, "reason": "", "buildings": 0, "hedges": 0, "trees": 0, "poles": 0, "curbs": 0,
             "windows": 0, "tris": 0, "mirrored": [], "rejected_seen": 0, "rejected_overlap": 0, "rejected_free": 0,
             "seam_blended": 0}
    layout = {"frame": "analysis", "boxes": [], "facades": [], "veg_refs": [], "street": None}
    builds = [p for p in parts if p.get("kind") == "building" and p.get("xy") is not None]
    # facade references (measured, seen by the master): grouped prism fronts → texture patches later
    groups = {}
    for p in builds:
        key = (round(p["u"][0], 2), round(p["u"][1], 2), round(p.get("dist", 0)))
        groups.setdefault(key, []).append(p)
    for g in groups.values():
        u = np.asarray(g[0]["u"], np.float64)
        n = np.asarray(g[0]["n"], np.float64)
        ss = []
        for p in g:
            c = np.asarray(p["xy"], np.float64)
            ss += [float(c @ u) - float(p["W"]) / 2, float(c @ u) + float(p["W"]) / 2]
        cen = np.mean([np.asarray(p["xy"], np.float64) for p in g], 0)
        b = float(cen @ n)
        s0, s1 = min(ss), max(ss)
        mid = u * (s0 + s1) / 2 + n * b
        layout["facades"].append({"c": [float(mid[0]), float(mid[1])], "u": [float(u[0]), float(u[1])],
                                  "n": [float(n[0]), float(n[1])], "W": round(s1 - s0, 2),
                                  "z": [round(float(np.median([p.get("g0", 0.0) for p in g])), 2),
                                        round(float(np.percentile([p["top"] for p in g], 30)), 2)],
                                  "dist": float(g[0].get("dist", 0)), "n_parts": len(g)})
    for p in parts:
        if p.get("kind") in ("bush/hedge", "tree crown") and p.get("xy") is not None:
            layout["veg_refs"].append({"xy": p["xy"], "r": float(p.get("r", 1.0)), "z": p.get("z", [0, 1]),
                                       "kind": p["kind"]})
    if len(builds) < MIN_BUILDINGS and sum(float(p.get("W", 0)) for p in builds) < MIN_FRONTAGE:
        stats["reason"] = "实测建筑太少，不是街区场景（只补地面）"
        return stats, layout
    a0, t, hw, src = street_frame(parts, pad, analysis)
    nperp = np.array([t[1], -t[0]])
    layout["street"] = {"a0": [float(a0[0]), float(a0[1])], "t": [float(t[0]), float(t[1])], "half_width": round(hw, 2),
                        "source": src}
    sides = side_stats(parts, a0, t, pad, analysis)
    summ = {sd: _summ(sides[sd]) for sd in (1, -1)}
    street_like = src == "analysis" or len(builds) >= 6
    conf = {}
    for sd in (1, -1):
        if summ[sd] is not None:
            conf[sd] = CONF_SIDE
        elif summ[-sd] is not None and street_like:
            summ[sd] = dict(summ[-sd])
            conf[sd] = CONF_MIRROR
            stats["mirrored"].append("右" if sd > 0 else "左")
    if not conf:
        stats["reason"] = "两侧都没有实测建筑"
        return stats, layout
    px, py = (float(pad[0]), float(pad[1])) if pad is not None else (0.0, 8.0)
    pz = float(pad[3]) if pad is not None and len(pad) > 3 else 0.0
    prod = np.array([px, py])
    clear_r = max((float(pad[2]) if pad is not None else 2.5) + 2.0, math.hypot(px, py) + CLEAR_EXTRA)
    layout["clear_r"] = round(clear_r, 2)
    # occupied footprints: measured prisms (front centre, u, n, W, D)
    rects = [(np.asarray(p["xy"]), np.asarray(p["u"]), np.asarray(p["n"]), float(p["W"]) + 0.5, float(p["D"]))
             for p in builds] + [tuple(o) for o in obstacles or ()]
    plazas = []
    for rd in (analysis or {}).get("roads", []) or []:
        if str(rd.get("kind", "")) == "plaza":
            pts = [(float(q.get("x", 0)), float(q.get("y", 0))) for q in rd.get("points", []) or []]
            if pts:
                plazas.append((np.array(pts), float(rd.get("width_m", 20) or 20) / 2))

    def in_plaza(xy):
        for pts, r in plazas:
            if float(np.min(np.hypot(pts[:, 0] - xy[0], pts[:, 1] - xy[1]))) < r:
                return True
        return False

    def clear_of_product(fp):
        return float(np.min(np.hypot(fp[:, 0] - px, fp[:, 1] - py))) >= clear_r

    tri0 = M.tris()
    win_left = WINDOW_LIMIT if windows else 0
    boxes = layout["boxes"]

    def place_box(kind, cls, c, u, n, W, D, z0, z1, cf, style=0, builder=None):
        nonlocal win_left
        fp = _footprint_pts(c, u, n, W, D)
        if not clear_of_product(fp):
            return False
        if _in_rects(fp, rects):
            stats["rejected_overlap"] += 1
            return False
        if np.asarray(visible(_box_samples(c, u, n, W, D, z0, z1))).any():
            stats["rejected_seen"] += 1
            return False
        if occupancy is not None:
            stt = occupancy.state(_dense_samples(c, u, n, W, D, z0, z1))
            if (stt == S_FREE).any() or (stt == S_VISIBLE).any():
                stats["rejected_free"] += 1
                return False
        if budget_left - (M.tris() - tri0) < 60:
            return False
        info = {"kind": "inferred " + kind, "inferred": True, "conf": cf,
                "xy": [round(float(c[0]), 2), round(float(c[1]), 2)], "W": round(W, 2), "D": round(D, 2),
                "z": [round(z0, 2), round(z1, 2)]}
        if builder is not None:
            builder(info)
        else:
            V, F = prims.box(c, u, n, W, D, z0, z1)
            M.add(V, F, cls, info)
        rects.append((np.asarray(c), np.asarray(u), np.asarray(n), W, D))
        boxes.append({"kind": kind, "c": [float(c[0]), float(c[1])], "u": [float(u[0]), float(u[1])],
                      "n": [float(n[0]), float(n[1])], "a": [-W / 2, W / 2], "b": [-D, 0.0],
                      "z": [round(z0, 2), round(z1, 2)], "conf": cf, "style": int(style)})
        return True

    hedge_any = any(sides[sd]["hedges"] for sd in (1, -1))
    tree_r = [r for sd in (1, -1) for r in sides[sd]["trees"] if r >= 1.0]
    s_prod = float((prod - a0) @ t)
    deferred = []                         # (distance, c, u, n, W, D, z0, z1): windows added near-first afterwards
    # measured fronts per side (s along the street, setback, eaves above ground, width) for the seam blend
    meas = {1: [], -1: []}
    for p in builds:
        xy = np.asarray(p["xy"], np.float64)
        off = float((xy - a0) @ nperp)
        meas[1 if off >= 0 else -1].append((float((xy - a0) @ t), abs(off),
                                             float(p["top"]) - float(p.get("g0", 0.0)), float(p["W"])))

    def seam_blend(sd, smid, ww, top, setback):
        """(top, setback, weight) of a lot next to a measured prism: its eaves / front, fading over SEAM_LEN."""
        best = None
        for s_m, off_m, top_m, w_m in meas[sd]:
            gap = abs(s_m - smid) - (w_m + ww) / 2
            if gap < SEAM_LEN and abs(off_m - setback) < 8.0 and abs(top_m - top) < SEAM_TOP_MAX \
                    and (best is None or gap < best[0]):
                best = (max(gap, 0.0), top_m, off_m)
        if best is None:
            return top, setback, 0.0
        x = float(np.clip(best[0] / SEAM_LEN, 0.0, 1.0))
        wgt = 1.0 - x * x * (3 - 2 * x)
        return top + (best[1] - top) * wgt, setback + (best[2] - setback) * wgt, wgt

    for sd, cf in conf.items():
        sm = summ[sd]
        setback0 = max(sm["setback"], hw + 1.5)
        n_in = -sd * nperp                    # facade normal → toward the street
        lot = sm["lot"]
        hed = sides[sd]["hedges"] or sides[-sd]["hedges"]
        k = 0
        s = -S_BACK
        while s < S_BACK:
            r = _hash01(sd, k)
            far_lot = bands and abs(s - s_prod) > BAND_MID
            w = float(np.clip(lot * (0.8 + 0.45 * r), 4.0, 16.0))
            if far_lot:
                w = float(np.clip(w * FAR_LOT_SCALE, 4.0, 28.0))
            setback = setback0
            gap = 0.6 + 1.4 * _hash01(sd, k, 2)
            alley = (k % 4 == 3) and (hed or tree_r)
            if alley:
                gap += 3.5
            top = sm["tops"][int(_hash01(sd, k, 3) * 4) % 4] * (0.92 + 0.16 * _hash01(sd, k, 4))
            if seam:
                top, setback, wgt = seam_blend(sd, s + w / 2, w, top, setback)
                setback = max(setback, hw + 1.0)
                if wgt > 0.05:
                    stats["seam_blended"] += 1
            # lots may be shortened from either end to stay out of the photo / measured prisms
            placed = False
            for trim in (0.0, 0.35, 0.6):
                for end in ((1, -1) if trim else (0,)):
                    ww = w * (1 - trim)
                    smid = s + w / 2 + end * (w - ww) / 2
                    c = a0 + t * smid + sd * nperp * setback
                    g0 = float(ground(float(c[0]), float(c[1])))
                    D = sm["D"] * (0.85 + 0.3 * _hash01(sd, k, 5))
                    style = int(_hash01(sd, k, 6) * 997)

                    def build_prism(info, c=c, ww=ww, D=D, g0=g0, top=top):
                        nonlocal win_left
                        if bands:
                            # 近密远疏: the box now, its windows later nearest-first (none beyond BAND_MID)
                            M.add(*prims.box(c, t, n_in, ww, D, g0 - 0.5, g0 + top), C_INF_BUILDING, info)
                            deferred.append((float(np.hypot(c[0] - px, c[1] - py)), c, t.copy(), n_in.copy(), ww, D,
                                             g0 - 0.5, g0 + top))
                            return
                        room = max(0, (budget_left - (M.tris() - tri0) - 12) // 12)     # a window frame = 12 tris
                        lim = int(min(win_left, room))
                        made = prims.prism_with_windows(
                            M, c, t, n_in, ww, D, g0 - 0.5, g0 + top, np.asarray(c) - n_in * 1e4, info,
                            windows=lim > 0, win_limit=lim, cls=C_INF_BUILDING, win_cls=C_INF_WINDOW)
                        stats["windows"] += made
                        win_left -= made
                    if place_box("building", C_INF_BUILDING, c, t, n_in, ww, D, g0 - 0.5, g0 + top, cf, style,
                                 builder=build_prism):
                        stats["buildings"] += 1
                        placed = True
                        break
                if placed:
                    break
            if alley:
                ac = a0 + t * (s + w + gap / 2) + sd * nperp * (setback + 0.6)
                g0 = float(ground(float(ac[0]), float(ac[1])))
                if tree_r and (k // 4) % 2 == 0:
                    rr = float(np.clip(np.median(tree_r), 1.5, 3.2))
                    cc = ac + n_in * 0.0

                    def build_tree(info, cc=cc, rr=rr, g0=g0):
                        V, F = prims.ellipsoid(np.r_[cc, g0 + 2.2 + rr], rr, rr, rr * 0.9, seed=k)
                        M.add(V, F, C_INF_VEG, info)
                        V, F = prims.cylinder(cc[0], cc[1], 0.18, g0 - 0.2, g0 + 2.4 + rr * 0.5, 6)
                        M.add(V, F, C_INF_POLE)
                    if place_box("tree", C_INF_VEG, cc + n_in * rr, t, n_in, 2 * rr, 2 * rr, g0, g0 + 2.2 + 2 * rr,
                                 cf * 0.9, builder=build_tree):
                        stats["trees"] += 1
                elif hed:
                    hc = a0 + t * (s + w + gap / 2) + sd * nperp * (setback - 0.2)
                    hwid = gap - 0.6

                    def build_hedge(info, hc=hc, hwid=hwid, g0=g0):
                        V, F = prims.rounded_box(hc, t, n_in, hwid, 1.0, g0 - 0.1, g0 + 1.25, bev=0.3)
                        M.add(V, F, C_INF_VEG, info)
                    if place_box("hedge", C_INF_VEG, hc, t, n_in, hwid, 1.0, g0 - 0.1, g0 + 1.25, cf * 0.9,
                                 builder=build_hedge):
                        stats["hedges"] += 1
            # hedges in front of some lots (front gardens) at the measured rhythm (near / mid band only)
            if hed and placed and not far_lot and _hash01(sd, k, 7) < 0.4:
                hc = a0 + t * (s + w / 2) + sd * nperp * (setback - 1.6)
                g0 = float(ground(float(hc[0]), float(hc[1])))
                hwid = w * 0.7

                def build_fh(info, hc=hc, hwid=hwid, g0=g0):
                    V, F = prims.rounded_box(hc, t, n_in, hwid, 0.9, g0 - 0.1, g0 + 1.15, bev=0.3)
                    M.add(V, F, C_INF_VEG, info)
                if place_box("hedge", C_INF_VEG, hc, t, n_in, hwid, 0.9, g0 - 0.1, g0 + 1.15, cf * 0.8,
                             builder=build_fh):
                    stats["hedges"] += 1
            s += w + gap
            k += 1
        # poles along the curb at the measured / analysed spacing
        poles = sorted(sides[sd]["poles"] or sides[-sd]["poles"])
        if poles:
            sp = float(np.clip(np.median(np.diff([p[0] for p in poles])) if len(poles) > 1 else 28.0, 18.0, 45.0))
            ph = float(np.clip(np.median([p[1] for p in poles]), 5.0, 11.0))
            s = poles[0][0] - sp * math.floor((poles[0][0] + S_BACK) / sp)
            while s < S_BACK:
                c = a0 + t * s + sd * nperp * (hw + 0.6)
                g0 = float(ground(float(c[0]), float(c[1])))

                def build_pole(info, c=c, g0=g0, ph=ph):
                    V, F = prims.cylinder(c[0], c[1], 0.16, g0 - 0.3, g0 + ph, 8)
                    M.add(V, F, C_INF_POLE, info)
                if place_box("pole", C_INF_POLE, c + n_in * 0.2, t, n_in, 0.4, 0.4, g0, g0 + ph, cf * 0.8,
                             builder=build_pole):
                    stats["poles"] += 1
                s += sp
    # windows of the inferred rows, nearest first, within the remaining triangle budget (近密远疏)
    for dist, c, uu, nn, ww, D, z0, z1 in sorted(deferred, key=lambda e: e[0]):
        if not windows or win_left <= 0 or dist > BAND_MID:
            break
        room = max(0, (budget_left - (M.tris() - tri0) - 12) // 12)
        lim = int(min(win_left, room))
        if lim <= 0:
            break
        made = prims.add_windows(M, c, uu, nn, ww, D, z0, z1, np.asarray(c) - nn * 1e4, win_limit=lim,
                                 win_cls=C_INF_WINDOW)
        stats["windows"] += made
        win_left -= made
    # curbs on both road edges (outside plazas / junctions)
    for sd in (1, -1):
        s = -S_BACK
        while s < S_BACK:
            ds = abs(s + 4.0 - s_prod)
            L = 8.0 if not bands or ds < BAND_NEAR else (CURB_LEN[1] if ds < BAND_MID else CURB_LEN[2])
            # a long far piece that cannot be placed whole falls back to 8 m pieces (no curb gaps from merging)
            for L_try in ((L,) if L <= CURB_LEN[0] else (L, CURB_LEN[0])):
                ok_any = False
                for s1 in np.arange(s, s + L - 1e-6, L_try):
                    c = a0 + t * (s1 + L_try / 2) + sd * nperp * (hw + 0.15)
                    if in_plaza(c):
                        continue
                    g0 = float(ground(float(c[0]), float(c[1])))
                    if place_box("curb", C_INF_CURB, c, t, -sd * nperp, L_try, 0.3, g0 - 0.1, g0 + 0.15, CONF_CURB):
                        stats["curbs"] += 1
                        ok_any = True
                    elif L_try > CURB_LEN[0]:
                        break
                if ok_any or L_try == CURB_LEN[0]:
                    break
            s += L
    stats["tris"] = M.tris() - tri0
    stats["ok"] = stats["buildings"] + stats["curbs"] + stats["hedges"] + stats["trees"] > 0
    stats["reason"] = "" if stats["ok"] else "没有可放推测体块的位置"
    stats["street_source"] = src
    stats["setback"] = {("右" if sd > 0 else "左"): round(max(summ[sd]["setback"], hw + 1.5), 1) for sd in conf}
    stats["product_z"] = round(pz, 3)
    layout["ground_z"] = round(pz, 3)
    return stats, layout


def layout_to_world(layout, world):
    """Analysis-frame layout → world (``world`` = 4×4 nested list / array: rotation about Z + translation)."""
    Wm = np.asarray(world, np.float64)
    R = Wm[:2, :2]
    tr = Wm[:3, 3]

    def P(xy):
        q = R @ np.asarray(xy, np.float64)[:2] + tr[:2]
        return [round(float(q[0]), 3), round(float(q[1]), 3)]

    def Vv(v):
        q = R @ np.asarray(v, np.float64)[:2]
        return [round(float(q[0]), 5), round(float(q[1]), 5)]
    dz = float(tr[2])
    out = {"frame": "world", "clear_r": layout.get("clear_r"), "boxes": [], "facades": [], "veg_refs": [], "street": None,
           "ground_z": (float(layout["ground_z"]) + dz) if layout.get("ground_z") is not None else None}
    for b in layout.get("boxes", []):
        out["boxes"].append(dict(b, c=P(b["c"]), u=Vv(b["u"]), n=Vv(b["n"]), z=[b["z"][0] + dz, b["z"][1] + dz]))
    for f in layout.get("facades", []):
        out["facades"].append(dict(f, c=P(f["c"]), u=Vv(f["u"]), n=Vv(f["n"]), z=[f["z"][0] + dz, f["z"][1] + dz]))
    for v in layout.get("veg_refs", []):
        out["veg_refs"].append(dict(v, xy=P(v["xy"]), z=[v["z"][0] + dz, v["z"][1] + dz]))
    st = layout.get("street")
    if st:
        out["street"] = dict(st, a0=P(st["a0"]), t=Vv(st["t"]))
    return out


def describe(stats) -> str:
    """Prompt sentence: what the completion added (Chinese, for the image model)."""
    if not stats or not stats.get("ok"):
        return ""
    bits = []
    if stats.get("curbs"):
        bits.append("街道沿原方向继续延伸（同样的路面和路沿）")
    if stats.get("buildings"):
        bits.append("两侧继续是与主视角同类型的房屋")
    if stats.get("hedges"):
        bits.append("房前有同样的修剪树篱")
    if stats.get("trees"):
        bits.append("间隔有同样的行道树")
    if stats.get("poles"):
        bits.append("路边有同样的电线杆")
    return "，".join(bits)


def summary_text(stats) -> str:
    if not stats:
        return ""
    if not stats.get("ok"):
        return f"场景补全：{stats.get('reason', '未补全')}"
    return (f"场景补全：推测建筑 {stats['buildings']} · 绿植 {stats['hedges'] + stats['trees']} · 电杆 {stats['poles']}"
            f" · 路沿 {stats['curbs']}" + (f"（{'、'.join(stats['mirrored'])}侧按对侧推测）" if stats.get("mirrored") else ""))
