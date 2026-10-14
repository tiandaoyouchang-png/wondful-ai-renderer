"""1.1.0: custom terrain for the clay base (高级 › 地面 › 自定义).

Pure math + prompt text, no ``bpy``: the Blender side lives in viewport_capture.

The terrain is a height field in the product's own frame: ``u`` along its forward
axis, ``v`` along its right axis, height along world up.  It is set by

* four CORNER slopes (左前 / 右前 / 左后 / 右后, degrees, -35..+35): how steeply the
  ground rises (+) or falls (-) toward that corner of the product.  They split into
  four components:

      pitch  = (左前 + 右前 - 左后 - 右后) / 4    front/back slope (爬坡 / 下坡)
      roll   = (右前 + 右后 - 左前 - 左后) / 4    left/right slope (侧坡)
      twist  = (左前 + 右后 - 右前 - 左后) / 4    diagonal twist (对角扭曲)
      mean   = (左前 + 右前 + 左后 + 右后) / 4    all four down = 坡顶, all up = 低洼

  Pitch and roll are a plane under the product.  The twist is a saddle that is gentle
  right under the product (so a rigid product rests on three points with one corner a
  little in the air, like on a real twisted track) and steepens to the full angle
  beyond it.  The mean (and the 坡顶 strength) is a crest: the ground falls away all
  round from a plateau just larger than the product.
* 坡顶 strength (0-1): extra fall-away, up to ``CREST_MAX_DEG`` all round.
* 凹凸 strength (0-1): smooth, low-frequency, deterministic bumps (颠簸路面).

The product is a rigid body: ``rest_pose`` finds its resting pitch/roll/height on the
field by minimising the height of its centre of mass with no point below the ground
— the stable pose, typically three support points with the centre of mass inside
their triangle.  With 姿态跟随地形 off the product stays level and is only lifted
so nothing penetrates.
"""
from __future__ import annotations

import math

try:  # numpy is bundled with Blender; tests run with it too
    import numpy as np
except Exception:  # pragma: no cover
    np = None

CORNERS = ("fl", "fr", "bl", "br")
CORNER_LABELS = {"fl": "左前", "fr": "右前", "bl": "左后", "br": "右后"}
MAX_ANGLE = 35.0
CREST_MAX_DEG = 30.0
BUMP_REL_AMPLITUDE = 0.035      # bump amplitude at strength 1, × product footprint size
TWIST_PAD_GAP = 0.6             # pad twist corner height = this × product height × tan(twist)
TWIST_PAD_CAP = 0.24            # ... at most this × pad half-diagonal (keeps the rigid product's tilt modest)
LEGACY_DIRECTIONS = ("front", "back", "left", "right")

# corners (左前, 右前, 左后, 右后), crest, bump
PRESETS = {
    "CREST": ("坡顶", {"fl": 0.0, "fr": 0.0, "bl": 0.0, "br": 0.0, "crest": 0.65, "bump": 0.0}),
    "TWIST": ("对角扭曲", {"fl": 12.0, "fr": -12.0, "bl": -12.0, "br": 12.0, "crest": 0.0, "bump": 0.0}),
    "TWIST_MIRROR": ("对角扭曲（镜像）", {"fl": -12.0, "fr": 12.0, "bl": 12.0, "br": -12.0, "crest": 0.0, "bump": 0.0}),
    "BUMPY": ("颠簸路面", {"fl": 0.0, "fr": 0.0, "bl": 0.0, "br": 0.0, "crest": 0.0, "bump": 0.7}),
    "CLIMB": ("爬坡", {"fl": 15.0, "fr": 15.0, "bl": -15.0, "br": -15.0, "crest": 0.0, "bump": 0.0}),
    "DESCENT": ("下坡", {"fl": -15.0, "fr": -15.0, "bl": 15.0, "br": 15.0, "crest": 0.0, "bump": 0.0}),
    "SIDE": ("侧坡", {"fl": 12.0, "fr": -12.0, "bl": 12.0, "br": -12.0, "crest": 0.0, "bump": 0.0}),
}
PRESET_ORDER = ("CREST", "TWIST", "TWIST_MIRROR", "BUMPY", "CLIMB", "DESCENT", "SIDE")
# 1.1.0 candidate presets that no longer exist: 山脊 → 坡顶, 山谷 → (all corners up) handled by mean
LEGACY_PRESETS = {"RIDGE": "CREST", "VALLEY": "CREST"}


def _clamp(x, lo, hi):
    try:
        x = float(x)
    except Exception:
        x = 0.0
    if x != x:  # NaN
        x = 0.0
    return max(lo, min(hi, x))


def normalize(settings: dict | None) -> dict:
    """Clamp a settings dict: {fl, fr, bl, br: deg, crest, bump: 0-1, follow, front_axis, vehicle, seed}."""
    settings = dict(settings or {})
    if any(d in settings for d in LEGACY_DIRECTIONS) and not any(c in settings for c in CORNERS):
        settings.update(corners_from_legacy(settings))
    out = {c: round(_clamp(settings.get(c, 0.0), -MAX_ANGLE, MAX_ANGLE), 4) for c in CORNERS}
    out["crest"] = _clamp(settings.get("crest", 0.0), 0.0, 1.0)
    out["bump"] = _clamp(settings.get("bump", 0.0), 0.0, 1.0)
    out["follow"] = bool(settings.get("follow", True))
    out["front_axis"] = str(settings.get("front_axis") or "AUTO").upper()
    out["vehicle"] = settings.get("vehicle")  # None = detect
    try:
        out["seed"] = int(settings.get("seed", 7)) & 0x7FFFFFFF
    except Exception:
        out["seed"] = 7
    return out


def corners_from_legacy(old: dict) -> dict:
    """Old 4-direction values {front: (mode, deg), ...} → corner slopes (each corner = sum of its
    two directions' signed angles, so 爬坡/侧坡 keep their pitch/roll and 山脊 becomes a crest)."""
    sig = {}
    for d in LEGACY_DIRECTIONS:
        mode, ang = (old.get(d) or ("FLAT", 0.0))
        mode = str(mode or "FLAT").upper()
        a = _clamp(ang, 0.0, MAX_ANGLE)
        sig[d] = a if mode == "UP" else (-a if mode == "DOWN" else 0.0)
    return {"fl": sig["front"] + sig["left"], "fr": sig["front"] + sig["right"],
            "bl": sig["back"] + sig["left"], "br": sig["back"] + sig["right"]}


def preset_settings(name: str) -> dict:
    name = LEGACY_PRESETS.get(name, name)
    label, values = PRESETS[name]
    return normalize(dict(values))


def settings_from_props(props) -> dict:
    migrate_legacy_props(props)
    s = {c: getattr(props, f"clay_terrain_{c}", 0.0) for c in CORNERS}
    s["crest"] = getattr(props, "clay_terrain_crest", 0.0)
    s["bump"] = getattr(props, "clay_terrain_bump", 0.0)
    s["follow"] = getattr(props, "clay_terrain_follow", True)
    s["front_axis"] = getattr(props, "clay_terrain_front_axis", "AUTO")
    from .product_kind import hint_says_vehicle, props_hint
    if hint_says_vehicle(props_hint(props)):
        s["vehicle"] = True  # otherwise detected from object / material names
    return normalize(s)


def migrate_legacy_props(props) -> bool:
    """A .blend saved with the 1.1.0 candidate's 前/后/左/右 controls keeps those values as raw
    ID properties; turn them into corner slopes once and drop them.  Returns True if migrated."""
    getter = getattr(props, "get", None)
    if getter is None:
        return False
    keys = [f"clay_terrain_{d}" for d in LEGACY_DIRECTIONS] + [f"clay_terrain_{d}_angle" for d in LEGACY_DIRECTIONS]
    try:
        present = [k for k in keys if getter(k) is not None]
    except Exception:
        return False
    if not present:
        return False
    modes = ("FLAT", "UP", "DOWN")
    old = {}
    for d in LEGACY_DIRECTIONS:
        m = getter(f"clay_terrain_{d}")
        if isinstance(m, int):
            m = modes[m] if 0 <= m < len(modes) else "FLAT"
        a = getter(f"clay_terrain_{d}_angle")
        old[d] = (m or "FLAT", 15.0 if a is None else a)
    corners = corners_from_legacy(old)
    try:
        for c in CORNERS:
            setattr(props, f"clay_terrain_{c}", _clamp(corners[c], -MAX_ANGLE, MAX_ANGLE))
        for k in present:
            try:
                del props[k]
            except Exception:
                pass
    except Exception:
        return False
    return True


def apply_preset_to_props(props, name: str) -> None:
    values = preset_settings(name)
    for c in CORNERS:
        setattr(props, f"clay_terrain_{c}", values[c])
    setattr(props, "clay_terrain_crest", values["crest"])
    setattr(props, "clay_terrain_bump", values["bump"])


def components(settings: dict) -> dict:
    """pitch / roll / twist / dome in degrees (dome > 0 = ground falls away all round)."""
    s = normalize(settings)
    fl, fr, bl, br = (s[c] for c in CORNERS)
    mean = (fl + fr + bl + br) / 4.0
    return {"pitch": (fl + fr - bl - br) / 4.0, "roll": (fr + br - fl - bl) / 4.0,
            "twist": (fl + br - fr - bl) / 4.0, "dome": s["crest"] * CREST_MAX_DEG - mean,
            "bump": s["bump"]}


def is_flat(settings: dict) -> bool:
    c = components(settings)
    return all(abs(c[k]) < 0.05 for k in ("pitch", "roll", "twist", "dome")) and c["bump"] < 0.005


def pose_angles(gu: float, gv: float) -> tuple[float, float]:
    """(pitch, roll) in degrees of a plane with gradients gu, gv: pitch > 0 = front up, roll > 0 = right up."""
    return math.degrees(math.atan(gu)), math.degrees(math.atan(gv))


def _ease(x, w):
    """0 for x<=0, x²/2w up to w, then x - w/2 (C¹ soft ramp).  numpy arrays."""
    x = np.clip(x, 0.0, None)
    return np.where(x < w, x * x / (2.0 * w), x - w / 2.0)


class HeightField:
    """Ground height h(u, v) around the product (u forward, v right, origin under its centre).

    half_u / half_v: half size of the pad (product footprint + margin); ``prod_height``
    sets how gentle the twist is right under the product."""

    def __init__(self, settings, half_u, half_v, shoulder, prod_height=None):
        self.settings = normalize(settings)
        self.comp = components(self.settings)
        self.half_u, self.half_v, self.shoulder = float(half_u), float(half_v), float(max(shoulder, 1e-6))
        self.diag = math.hypot(self.half_u, self.half_v)
        ph = float(prod_height if prod_height else max(self.half_u, self.half_v))
        self.gu = math.tan(math.radians(self.comp["pitch"]))
        self.gv = math.tan(math.radians(self.comp["roll"]))
        self.dome = math.tan(math.radians(self.comp["dome"]))
        self.twist_far = math.tan(math.radians(self.comp["twist"]))
        # twist height at the pad corners: gentle (one corner of a rigid product lifts by ~4×)
        ht = TWIST_PAD_GAP * ph * abs(self.twist_far)
        ht = min(ht, TWIST_PAD_CAP * self.diag, abs(self.twist_far) * self.diag)
        self.twist_pad = math.copysign(ht, self.twist_far) if self.twist_far else 0.0
        self.twist_run = 1.5 * 2.0 * max(self.half_u, self.half_v)   # full twist slope this far past the shoulder
        self.twist_fade = 2.0 * max(self.half_u, self.half_v)        # then levels off over this distance
        size = 2.0 * max(self.half_u, self.half_v)
        self.bump_amp = BUMP_REL_AMPLITUDE * size * self.settings["bump"]
        rng = np.random.default_rng(self.settings["seed"])
        waves = []
        for k, rel in enumerate((1.7, 1.15, 0.85, 0.6, 0.42)):
            ang = rng.uniform(0, math.pi)
            lam = rel * size * rng.uniform(0.9, 1.1)
            amp = (1.0, 0.8, 0.6, 0.45, 0.3)[k]
            waves.append((math.cos(ang) * 2 * math.pi / lam, math.sin(ang) * 2 * math.pi / lam,
                          rng.uniform(0, 2 * math.pi), amp))
        norm = math.sqrt(sum(w[3] ** 2 for w in waves) / 2.0)
        self.waves = [(a, b, p, amp / norm) for a, b, p, amp in waves]

    # ---- parts (numpy in, numpy out)
    def _twist(self, u, v):
        if not self.twist_far:
            return np.zeros_like(u)
        a, b = u / self.half_u, v / self.half_v
        r2 = a * a + b * b
        # -sin 2φ in the normalized frame: + toward 左前 (a>0, b<0) and 右后 (a<0, b>0)
        ang = np.where(r2 > 1e-12, -2.0 * a * b / np.maximum(r2, 1e-12), 0.0)
        r = np.sqrt(r2) * self.diag / math.sqrt(2.0)  # physical radius, = diag at a corner
        D, H = self.diag, self.twist_pad
        inner = H * (r / D) ** 2
        s0 = 2.0 * H / D  # radial slope at the pad corner
        w = self.shoulder
        x = np.clip(r - D, 0.0, None)
        xe = np.minimum(x, w)
        outer = H + s0 * xe + (self.twist_far - s0) * xe * xe / (2.0 * w)
        # full slope for a while beyond the shoulder, then easing back to level ground, so the
        # twisted slopes read around the product and the horizon stays in the picture
        run, fade = self.twist_run, self.twist_fade
        y = np.clip(x - w, 0.0, None)
        yr = np.minimum(y, run)
        yf = np.clip(y - run, 0.0, fade)
        outer = outer + self.twist_far * (yr + yf - yf * yf / (2.0 * fade))
        return ang * np.where(r <= D, inner, outer)

    def _dome(self, u, v):
        if not self.dome:
            return np.zeros_like(u)
        du = np.clip(np.abs(u) - self.half_u, 0.0, None)
        dv = np.clip(np.abs(v) - self.half_v, 0.0, None)
        return -self.dome * _ease(np.hypot(du, dv), self.shoulder)

    def _bump(self, u, v):
        if self.bump_amp <= 0:
            return np.zeros_like(u)
        h = np.zeros_like(u)
        for a, b, p, amp in self.waves:
            h = h + amp * np.cos(a * u + b * v + p)
        return self.bump_amp * h

    def height(self, u, v):
        scalar = not (isinstance(u, np.ndarray) or isinstance(v, np.ndarray))
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        u, v = np.broadcast_arrays(u, v)
        h = self.gu * u + self.gv * v + self._twist(u, v) + self._dome(u, v) + self._bump(u, v)
        return float(h) if scalar else h


def grid_offsets(half, shoulder, size, far=300.0):
    """Sorted 1-D vertex coordinates: fine near the product (bumps, twist, contact), coarse far away."""
    step = size / 40.0
    near = half + shoulder + 3.0 * size
    n = max(8, int(math.ceil(near / step)))
    pos = [near * k / n for k in range(n + 1)]
    x, d = near, step
    while x < max(far, size * 60.0):
        d *= 1.35
        x += d
        pos.append(x)
    pos += [half, half + shoulder]
    neg = [-p for p in pos if p > 0]
    return sorted(set(round(p, 9) for p in neg + pos))


# --------------------------------------------------------------------------- rigid rest pose

def _rot(axis, ang):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def pose_matrix(fwd, side, pitch_deg, roll_deg):
    """3×3 world rotation: pitch > 0 lifts the front (about the right axis), roll > 0 lifts the
    right side (about -forward)."""
    return _rot(-np.asarray(fwd, float), math.radians(roll_deg)) @ _rot(side, math.radians(pitch_deg))


def support_points(P, fwd, side, cells=48, band=0.35):
    """The product's underside: the lowest vertex in each cell of a footprint grid, among the
    vertices in the lowest ``band`` of its height.  P: N×3 world points relative to the pivot."""
    F, S = np.asarray(fwd, float), np.asarray(side, float)
    z = P[:, 2]
    h = float(z.max() - z.min()) or 1.0
    sel = P[z < z.min() + band * h]
    u, v = sel @ F, sel @ S
    iu = np.clip(((u - u.min()) / max(np.ptp(u), 1e-9) * cells).astype(int), 0, cells - 1)
    iv = np.clip(((v - v.min()) / max(np.ptp(v), 1e-9) * cells).astype(int), 0, cells - 1)
    key = iu * cells + iv
    order = np.lexsort((sel[:, 2], key))
    first = np.ones(len(order), bool)
    first[1:] = key[order][1:] != key[order][:-1]
    return sel[order[first]]


def _lift(points, R, field, F, S):
    q = points @ R.T
    return np.max(field.height(q @ F, q @ S) - q[:, 2])


def _groups(xy, link):
    """Cluster 2-D points (single linkage, ``link`` distance) → list of index arrays."""
    n = len(xy)
    if not n:
        return []
    label = -np.ones(n, int)
    k = 0
    for i in range(n):
        if label[i] >= 0:
            continue
        label[i] = k
        stack = [i]
        while stack:
            j = stack.pop()
            near = np.where((label < 0) & (np.hypot(xy[:, 0] - xy[j, 0], xy[:, 1] - xy[j, 1]) < link))[0]
            label[near] = k
            stack.extend(near.tolist())
        k += 1
    return [np.where(label == g)[0] for g in range(k)]


def _hull(pts):
    pts = sorted(set((float(a), float(b)) for a, b in pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def pose_angles_of(R, fwd, side):
    """(pitch, roll) in degrees of a world rotation: pitch > 0 = front up, roll > 0 = right up."""
    f = R @ np.asarray(fwd, float)
    s = R @ np.asarray(side, float)
    return (math.degrees(math.asin(max(-1.0, min(1.0, f[2])))),
            math.degrees(math.asin(max(-1.0, min(1.0, s[2])))))


def _settle(X, c, field, size, iters=8):
    """Let a posed rigid point set tip over contact lines until it is stable: >= 3 separate
    contacts with the centre of mass above their support polygon.  Returns (R, t) with
    X_new = X @ R.T + t (rotation about the world origin)."""
    R_acc, t_acc = np.eye(3), np.zeros(3)
    tol = 2e-4 * size
    link = 0.08 * size
    zv = np.array((0.0, 0.0, 1.0))
    for _ in range(iters):
        clear = X[:, 2] - field.height(X[:, 0], X[:, 1])
        touch = np.where(clear < tol)[0]
        gs = [touch[g] for g in _groups(X[touch, :2], link)]
        cxy = c[:2]
        axis_pts = None
        if len(gs) == 1:
            a = X[gs[0]].mean(0)
            d = c - a
            d[2] = 0.0
            if np.linalg.norm(d) < 1e-9 * size:
                break  # centre of mass straight above the single contact
            axis = np.cross(zv, d / np.linalg.norm(d))
            axis_pts = (a, gs[0])
        else:
            hull = _hull(X[touch, :2])
            if len(gs) >= 3 and len(hull) >= 3 and _inside(cxy, hull):
                break
            if len(gs) == 2 or len(hull) < 3:
                ga, gb = gs[0], gs[1]
                if len(gs) > 2:  # collinear contacts: use the two farthest groups
                    cen = [X[g].mean(0) for g in gs]
                    pairs = [(np.linalg.norm(cen[i][:2] - cen[j][:2]), i, j) for i in range(len(gs)) for j in range(i + 1, len(gs))]
                    _, i, j = max(pairs)
                    ga, gb = gs[i], gs[j]
                # tip over the two lowest-lever points: the extreme points of each group along the line
                ca, cb = X[ga].mean(0), X[gb].mean(0)
                a, b = ca, cb
                idx = np.concatenate([ga, gb])
            else:
                # centre of mass outside the polygon: tip over the hull edge it is beyond
                best = None
                for i in range(len(hull)):
                    p, q = np.array(hull[i]), np.array(hull[(i + 1) % len(hull)])
                    e = q - p
                    side_ = e[0] * (cxy[1] - p[1]) - e[1] * (cxy[0] - p[0])
                    if side_ < 0 and (best is None or side_ < best[0]):
                        best = (side_, p, q)
                if best is None:
                    break
                _, p, q = best
                ia = touch[np.argmin(np.hypot(X[touch, 0] - p[0], X[touch, 1] - p[1]))]
                ib = touch[np.argmin(np.hypot(X[touch, 0] - q[0], X[touch, 1] - q[1]))]
                a, b = X[ia], X[ib]
                idx = touch
            axis = b - a
            if np.linalg.norm(axis) < 1e-9:
                break
            axis = axis / np.linalg.norm(axis)
            axis_pts = (a, idx)
        a, fixed = axis_pts
        # direction that lowers the centre of mass
        if np.cross(axis, c - a)[2] > 0:
            axis = -axis
        # the points of the supporting feet themselves (and their neighbours) stay on the axis
        others = np.ones(len(X), bool)
        others[fixed] = False
        ref = X[fixed][:: max(1, len(fixed) // 256), :2]
        for q in ref:
            others &= np.hypot(X[:, 0] - q[0], X[:, 1] - q[1]) >= link
        if not others.any():
            break

        def rotated(ang):
            Rm = _rot(axis, ang)
            return (X - a) @ Rm.T + a, Rm

        def gap(ang):
            Y, _ = rotated(ang)
            Yo = Y[others]
            return float(np.min(Yo[:, 2] - field.height(Yo[:, 0], Yo[:, 1])))

        lo, hi, step = 0.0, None, math.radians(0.1)
        ang = step
        while ang < math.radians(30.0):
            if gap(ang) <= 0.0:
                hi = ang
                break
            lo = ang
            ang += step
        if hi is None:
            break
        for _ in range(30):
            mid = (lo + hi) / 2.0
            if gap(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        X, Rm = rotated(hi)
        t_step = a - Rm @ a
        lift = float(np.max(field.height(X[:, 0], X[:, 1]) - X[:, 2]))
        X[:, 2] += lift
        t_step = t_step + np.array((0.0, 0.0, lift))
        c = Rm @ c + t_step
        R_acc = Rm @ R_acc
        t_acc = Rm @ t_acc + t_step
    return R_acc, t_acc


def _inside(p, hull):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    return all(cross(hull[i], hull[(i + 1) % len(hull)], p) >= -1e-12 for i in range(len(hull)))


def rest_pose(points, com, field, fwd, side, follow=True, limit=40.0):
    """Stable resting pose of a rigid product on ``field``: (R, t, pitch, roll).

    ``points``: candidate contact points (world, relative to the pivot on the ground under the
    product's centre); ``com``: centre of mass, same frame.  Posed point = R @ p + t.  A coarse
    search finds the pitch/roll with the lowest centre of mass (product lowered until its first
    point touches, nothing below the ground); then it tips over its contact line(s) until it
    rests on >= 3 separate contacts with the centre of mass above their support polygon."""
    F, S = np.asarray(fwd, float), np.asarray(side, float)
    com = np.asarray(com, float)
    points = np.asarray(points, float)
    # the field is defined in (u, v) = (p·F, p·S); work in that horizontal frame
    B = np.array([F, S, (0.0, 0.0, 1.0)])  # world → (u, v, z)

    def energy(p, r):
        R = pose_matrix(F, S, p, r)
        lift = _lift(points, R, field, F, S)
        return float((R @ com)[2] + lift), lift

    if not follow:
        return np.eye(3), np.array((0.0, 0.0, energy(0.0, 0.0)[1])), 0.0, 0.0
    best = (energy(0.0, 0.0)[0], 0.0, 0.0)
    grid = np.arange(-limit, limit + 1e-9, 2.0)
    for p in grid:
        for r in grid:
            e = energy(p, r)[0]
            if e < best[0] - 1e-12:
                best = (e, p, r)
    for step in (0.5, 0.125, 0.03125):
        e0, p0, r0 = best
        for dp in (-3, -2, -1, 0, 1, 2, 3):
            for dr in (-3, -2, -1, 0, 1, 2, 3):
                p, r = p0 + dp * step, r0 + dr * step
                if abs(p) > limit or abs(r) > limit:
                    continue
                e = energy(p, r)[0]
                if e < best[0] - 1e-12:
                    best = (e, p, r)
    _, p, r = best
    R0 = pose_matrix(F, S, p, r)
    lift = energy(p, r)[1]
    t0 = np.array((0.0, 0.0, lift))
    # settle in the (u, v, z) frame, where the field is a plain function of the first two coordinates
    X = (points @ R0.T + t0) @ B.T
    c = B @ (R0 @ com + t0)
    size = float(max(np.ptp(X[:, 0]), np.ptp(X[:, 1]), 1e-6))
    Rs, ts = _settle(X, c, _UVField(field), size)
    R = B.T @ Rs @ B @ R0
    t = B.T @ (Rs @ (B @ t0) + ts)
    pitch, roll = pose_angles_of(R, F, S)
    return R, t, pitch, roll


class _UVField:
    def __init__(self, field):
        self.field = field

    def height(self, u, v):
        return self.field.height(u, v)


def contact_report(points, com, field, fwd, side, eps, size):
    """Contacts of posed points (already rotated + lifted, relative to the pivot).

    Returns dict: clearance min, contact groups (clustered), quadrant labels, whether the
    centre of mass projects inside the support polygon, and each corner's lowest clearance."""
    F, S = np.asarray(fwd, float), np.asarray(side, float)
    u, v = points @ F, points @ S
    clear = points[:, 2] - field.height(u, v)
    touch = clear < eps
    tu, tv = u[touch], v[touch]
    link = 0.08 * size
    groups = [[(tu[i], tv[i]) for i in g] for g in _groups(np.stack([tu, tv], axis=1), link)]
    cu, cv = float(com @ F), float(com @ S)
    uc, vc = (float(u.max() + u.min()) / 2.0, float(v.max() + v.min()) / 2.0)
    gl = []
    for g in groups:
        gu = sum(a for a, _ in g) / len(g)
        gv = sum(b for _, b in g) / len(g)
        gl.append(("f" if gu >= uc else "b") + ("r" if gv >= vc else "l"))
    hull = _hull(list(zip(tu, tv)))
    inside = len(hull) >= 3 and _inside((cu, cv), hull)
    corner_clear = {}
    for c in CORNERS:
        m = ((u >= uc) if c[0] == "f" else (u < uc)) & ((v >= vc) if c[1] == "r" else (v < vc))
        if m.any():
            corner_clear[c] = float(clear[m].min())
    return {"min_clearance": float(clear.min()), "contact_points": int(touch.sum()),
            "contact_groups": len(groups), "group_corners": gl, "com_inside": inside,
            "corner_clearance": corner_clear}


def lifted_corner(report, size, min_gap=None):
    """The corner hanging in the air (largest corner clearance above a small threshold), or None."""
    cc = report.get("corner_clearance") or {}
    if not cc:
        return None, 0.0
    c, gap = max(cc.items(), key=lambda kv: kv[1])
    thr = min_gap if min_gap is not None else 0.006 * size
    return (c, gap) if gap > thr else (None, 0.0)


# --------------------------------------------------------------------------- prompt text

def _names(vehicle: bool):
    if vehicle:
        return {"subj": "车身", "it": "车", "front": "车头", "back": "车尾", "left": "车身左侧", "right": "车身右侧",
                "corner": {"fl": "左前轮", "fr": "右前轮", "bl": "左后轮", "br": "右后轮"},
                "ground_corner": {"fl": "左前方", "fr": "右前方", "bl": "左后方", "br": "右后方"},
                "contact": "四个车轮都实实在在压在地面上", "three": "三个车轮着地"}
    return {"subj": "产品", "it": "产品", "front": "产品正面", "back": "产品背面", "left": "产品左侧", "right": "产品右侧",
            "corner": {"fl": "左前角", "fr": "右前角", "bl": "左后角", "br": "右后角"},
            "ground_corner": {"fl": "左前方", "fr": "右前方", "bl": "左后方", "br": "右后方"},
            "contact": "产品底部的支撑点（脚、轮或底座）全部贴住地面", "three": "产品三点着地"}


def describe(settings, vehicle=False, hints=None, pose=None) -> str:
    """Chinese terrain sentence for the image prompt.

    ``hints`` maps a direction (front/back/left/right) or corner (fl/fr/bl/br) to where it is in
    the picture (e.g. {"right": "画面左侧", "fr": "画面左下方"}).  ``pose`` is the rest-pose
    result {"pitch", "roll", "lifted": corner or None, "gap": metres under the lifted corner} (defaults to the plane components)."""
    s = normalize(settings)
    if is_flat(s):
        return ""
    hints = hints or {}
    n = _names(bool(vehicle))
    comp = components(s)
    pose = pose or {}
    pitch = float(pose.get("pitch", comp["pitch"] if s["follow"] else 0.0))
    roll = float(pose.get("roll", comp["roll"] if s["follow"] else 0.0))
    lifted = pose.get("lifted")

    def name(d):
        h = hints.get(d)
        return f"{n[d]}（{h}）" if h else n[d]

    def cname(c, ground=False):
        base = (n["ground_corner"] if ground else n["corner"])[c]
        h = hints.get(c)
        return f"{base}（{h}）" if h else base

    parts = []
    dome = comp["dome"]
    if dome >= 1.0:
        parts.append(f"{n['it']}停在坡顶最高处，四周地面从{'车轮' if vehicle else '产品脚'}下向外、向下掉落约{round(dome)}°，"
                     f"坡顶的轮廓线清楚可见")
    elif dome <= -1.0:
        parts.append(f"{n['it']}停在低洼处，四周地面向外抬起约{round(-dome)}°")
    # on twisted / bumpy ground the small tilt of the rigid product is described by the lifted corner
    tilt_min = 5.0 if (abs(comp["twist"]) >= 1.0 or s["bump"] >= 0.05) else 0.5
    if abs(pitch) >= tilt_min:
        p = round(abs(pitch))
        if vehicle:
            parts.append(f"车头朝上爬坡约{p}°（车头高、车尾低）" if pitch > 0 else f"车头朝下下坡约{p}°（车头低、车尾高）")
        else:
            parts.append(f"产品立在约{p}°的斜坡上，正面朝上坡方向（前高后低）" if pitch > 0
                         else f"产品立在约{p}°的斜坡上，正面朝下坡方向（前低后高）")
    if abs(roll) >= tilt_min:
        r = round(abs(roll))
        low, high = ("right", "left") if roll < 0 else ("left", "right")
        parts.append(f"{n['subj']}向{n[low][-2:]}倾斜约{r}°（{name(low)}低、{name(high)}高）")
    tw = comp["twist"]
    if abs(tw) >= 1.0:
        hi, lo = (("fl", "br"), ("fr", "bl")) if tw > 0 else (("fr", "bl"), ("fl", "br"))
        parts.append(f"地面呈对角扭曲：{cname(hi[0], True)}与{cname(hi[1], True)}地面较高、"
                     f"{cname(lo[0], True)}与{cname(lo[1], True)}地面较低，坡面像被拧过一样起伏约{round(abs(tw))}°")
    b = s["bump"]
    if b >= 0.05:
        word = "轻微" if b < 0.35 else ("明显" if b < 0.75 else "强烈")
        parts.append(f"地面{word}凹凸不平、高低起伏，像颠簸的土路/坑洼地，不是平整路面")
    if lifted in CORNERS:
        gap = float(pose.get("gap") or 0.0)
        cm = f"约{max(1, round(gap * 100))}厘米" if gap > 0 else "一段"
        parts.append(f"刚性{n['subj']}不随地面扭曲：{n['three']}，{cname(lifted)}明显悬空、离地{cm}，"
                     f"能清楚看到它和下方地面之间的空隙")
    if not any(("倾斜" in x or "坡上" in x or "爬坡" in x or "下坡" in x) for x in parts) and not lifted \
            and tilt_min < 1.0:
        parts.append(f"{n['subj']}姿态基本水平")
    text = "本次地形：" + "；".join(parts)
    if lifted in CORNERS:
        text += (f"。地形与{n['subj']}姿态以白模为准：着地的支撑点贴住地面、不陷进地面，"
                 f"{cname(lifted)}保持悬空（和白模一样留出空隙并有对应的阴影），不要让它落地，"
                 f"也不要把{n['subj']}摆正；不要把地面改平，地面的起伏、扭曲方向和陡缓都与白模一致。")
    else:
        text += (f"。地形与{n['subj']}姿态以白模为准：{n['contact']}，不悬空、不陷进地面；"
                 "不要把地面改平，也不要改变坡的方向、起伏和陡缓。")
    return text
