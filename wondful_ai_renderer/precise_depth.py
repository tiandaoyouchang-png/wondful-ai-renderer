"""1.2.5 ④ 场景粗模 ·「精确模式」— measure the reference photo, then pin the LLM layout to it (no bpy).

Ported from the 2026-10-19 research prototypes (blockout_research/moge_infer.py, analyze_D.py, depth_snap.py),
numpy only (no scipy / PIL / SegFormer):

1. ``infer(rgb)``: MoGe-2 ViT-S on the CPU → metric point map; focal + shift recovered like MoGe's own
   post-processing; ground plane by RANSAC on upward-normal pixels → camera height / pitch / roll / hFOV in the
   plugin's frame (camera at x=0, y=0, z=h, looking +Y, Z up; ground z=0).
2. ``depth_block(geo)``: the MEASURED GEOMETRY text appended to the analysis prompt (variant D): fixed camera,
   an 8×6 depth grid and the biggest depth blobs.
3. ``apply_camera(data, cam)``: the measured camera overrides whatever the LLM returned (fixed values).
4. ``snap(data, geo)``: every LLM block is moved to the measured distance of its own screen footprint, scaling
   horizontal positions and dimensions while keeping its terrain-relative base. This is not a camera-centred
   3D scale: vertical screen position can change when ground contact is retained. Optionally (P1,
   ``SIZE_SNAP_ENABLED``, off) its height and width are set from that depth region's measured points (clamped).
5. ``range_error(data, geo)``: the 3D fit number: median |distance error| of blocks vs the measured depth.

Classes come from the depth itself (no segmentation model): sky = MoGe mask off; ground = upward normal and
near the fitted plane; vegetation = green-dominant pixels; the rest = structure.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

INFER_LONG = 924            # MoGe input long edge (multiple of 14; 5 s on the research CPU)
NUM_TOKENS = 1800
MAXR = 100.0                # only snap what monocular depth measures well
SKY, GROUND, STRUCT, VEG, FAR = 0, 1, 2, 3, 4
FAR_M = 120.0              # non-green, non-ground beyond this = far terrain / mountains (not a building)
CLS_NAME = {SKY: "sky", GROUND: "ground", STRUCT: "structure", VEG: "vegetation", FAR: "far terrain/mountain"}
CLS_SHORT = {SKY: "sky", GROUND: "groun", STRUCT: "struc", VEG: "veget", FAR: "far"}
THIN = ("ceiling", "rug", "water_plane", "pole", "lamp", "sign", "window", "curtain", "person")
SIZE_SNAP = ("building", "wall", "vehicle", "furniture", "awning", "fence", "hedge", "rock")
CLASS_OF = {"building": (STRUCT,), "wall": (STRUCT,), "vehicle": (STRUCT,), "fence": (STRUCT, VEG), "awning": (STRUCT,),
            "furniture": (STRUCT,), "tree": (VEG,), "bush": (VEG,), "hedge": (VEG,), "plant": (VEG,), "rock": (STRUCT, FAR)}
NONGROUND = (STRUCT, VEG, FAR)
FOV_WARN_DEG = 25.0
SIZE_PAD_U, SIZE_PAD_V, SIZE_GROW_MAX = 0.15, 0.4, 1.5   # P1 width/height snap window + growth cap
# P1 width/height snap: implemented + unit-tested, but OFF — on the street reference it lowered the 2D fit
# (skyIoU 0.774 → 0.665; a tighter window 0.726) while distance-only raised it (0.725 → 0.738). 2026-10-20.
SIZE_SNAP_ENABLED = False


# ---------------------------------------------------------------------------------------- image helpers

def resize_bilinear(img, w, h) -> np.ndarray:
    """float32 HxWxC bilinear resize (align-corners false)."""
    a = np.asarray(img, np.float32)
    H, W = a.shape[:2]
    ys = np.clip((np.arange(h) + 0.5) * H / h - 0.5, 0, H - 1)
    xs = np.clip((np.arange(w) + 0.5) * W / w - 0.5, 0, W - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    wy = (ys - y0)[:, None, None] if a.ndim == 3 else (ys - y0)[:, None]
    wx = (xs - x0)[None, :, None] if a.ndim == 3 else (xs - x0)[None, :]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def infer_size(fw, fh, long=INFER_LONG):
    """MoGe input size for a frame: long edge ``long``, both sides multiples of 14."""
    if fw >= fh:
        W = long
        H = max(14, int(round(long * fh / float(fw) / 14.0)) * 14)
    else:
        H = long
        W = max(14, int(round(long * fw / float(fh) / 14.0)) * 14)
    return W, H


# ---------------------------------------------------------------------------------------- MoGe post-processing

def uv_grid(w, h):
    a = w / float(h)
    sx, sy = a / (1 + a * a) ** .5, 1 / (1 + a * a) ** .5
    u = np.linspace(-sx * (w - 1) / w, sx * (w - 1) / w, w, dtype=np.float32)
    v = np.linspace(-sy * (h - 1) / h, sy * (h - 1) / h, h, dtype=np.float32)
    return np.stack(np.meshgrid(u, v, indexing="xy"), -1)


def focal_shift(pts, mask):
    """Recover (focal in uv units, z shift) of an affine-invariant point map (MoGe), scipy-free:
    least squares over the shift with the focal in closed form (coarse grid + golden-section refine)."""
    h, w = mask.shape
    uv = uv_grid(w, h)
    st = max(1, h // 64)
    m = mask[::st, ::st]
    P = pts[::st, ::st][m].astype(np.float64)
    U = uv[::st, ::st][m].astype(np.float64)
    xy, z = P[:, :2], P[:, 2]

    def cost_f(s):
        zz = z + s
        if np.any(zz <= 1e-6):
            return np.inf, 0.0
        pr = xy / zz[:, None]
        f = (pr * U).sum() / max(np.square(pr).sum(), 1e-12)
        return float(np.square(f * pr - U).sum()), float(f)
    zmin = float(z.min())
    lo = -zmin + 1e-3 * max(1.0, abs(zmin))
    hi = max(1.0, 4.0 * float(np.median(np.abs(z))))
    grid = np.linspace(lo, hi, 61)
    costs = [cost_f(s)[0] for s in grid]
    i = int(np.argmin(costs))
    a, b = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(40):
        if cost_f(c)[0] < cost_f(d)[0]:
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    s = (a + b) / 2
    return cost_f(s)[1], float(s)


def fit_ground(Pw, Nw, valid, rng=None, iters=300, tol=0.15):
    """RANSAC plane on upward-facing near/mid pixels (Z-up camera frame). Returns (n, d, inlier count)."""
    H = valid.shape[0]
    rows = np.zeros_like(valid)
    rows[int(H * 0.35):] = True
    cand = valid & rows & (Nw[..., 2] > 0.8) & (Pw[..., 1] > 0.3) & (Pw[..., 1] < 80)
    G = Pw[cand].astype(np.float64)
    if len(G) < 50:
        cand = valid & rows & (Pw[..., 1] > 0.3) & (Pw[..., 1] < 80)
        G = Pw[cand].astype(np.float64)
    if len(G) < 3:
        raise ValueError("没有找到地面")
    if len(G) > 60000:
        G = G[np.random.default_rng(1).choice(len(G), 60000, replace=False)]
    rng = rng or np.random.default_rng(0)
    best = None
    for _ in range(iters):
        a, b, c = G[rng.choice(len(G), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n[2] < 0:
            n = -n
        if n[2] < 0.8:
            continue
        d = -n @ a
        cnt = int((np.abs(G @ n + d) < tol).sum())
        if best is None or cnt > best[2]:
            best = (n, float(d), cnt)
    if best is None:
        raise ValueError("没有找到地面")
    n, d, _ = best
    inl = np.abs(G @ n + d) < tol
    Gi = G[inl]
    if len(Gi) >= 3:      # least-squares refine
        cen = Gi.mean(0)
        _, _, vt = np.linalg.svd(Gi - cen, full_matrices=False)
        n2 = vt[-1]
        if n2[2] < 0:
            n2 = -n2
        if n2[2] > 0.8:
            n, d = n2, float(-n2 @ cen)
    return n, d, int(inl.sum())


def camera_from_plane(n, d):
    """Rotation (rows = world axes in the Z-up camera frame), camera height, pitch, roll (plugin convention:
    the builder's camera Euler is (90° + pitch, roll, 0), so a roll that raises the camera's right side is negative)."""
    z = np.asarray(n, float) / np.linalg.norm(n)
    xax = np.array([1.0, 0, 0]) - z * z[0]
    xax /= np.linalg.norm(xax)
    yax = np.cross(z, xax)
    R = np.stack([xax, yax, z])
    # For the builder's XYZ Euler, the ground normal in this camera frame is
    # (-sin(roll), cos(roll)*sin(pitch), cos(roll)*cos(pitch)).  asin(fwd.z)
    # measures elevation, not the Euler pitch, and is biased when roll != 0.
    pitch = math.degrees(math.atan2(z[1], z[2]))
    roll_up = math.degrees(math.asin(max(-1.0, min(1.0, z[0]))))
    return R, float(d), pitch, -roll_up


def postprocess(pts, nrm, msk, scale, rgb_small):
    """MoGe raw outputs (H×W×3 points, normals, mask, scale) → geo dict in the plugin frame."""
    H, W = msk.shape
    f, s = focal_shift(pts, msk)
    P = pts.astype(np.float32).copy()
    P[..., 2] += s
    P *= float(scale)                                # metric, OpenCV cam (x right, y down, z fwd)
    diag = (W * W + H * H) ** .5
    hfov = math.degrees(2 * math.atan(W / diag / f))
    Pw = np.stack([P[..., 0], P[..., 2], -P[..., 1]], -1)
    Nw = np.stack([nrm[..., 0], nrm[..., 2], -nrm[..., 1]], -1)
    n, d, inl = fit_ground(Pw, Nw, msk)
    R, cam_h, pitch, roll = camera_from_plane(n, d)
    Pa = Pw @ R.T
    Pa[..., 2] += cam_h
    Na = Nw @ R.T
    cls = classify(Pa, Na, msk, rgb_small)
    cam = {"height_m": round(cam_h, 3), "pitch_deg": round(pitch, 2), "roll_deg": round(roll, 2),
           "hfov_deg": round(hfov, 2), "metric_scale": round(float(scale), 3), "ground_inliers": inl,
           "infer_res": [W, H]}
    return {"P": Pa.astype(np.float32), "N": Na.astype(np.float32), "mask": msk, "cls": cls, "cam": cam}


def classify(P, N, mask, rgb_small) -> np.ndarray:
    cls = np.full(mask.shape, STRUCT, np.uint8)
    rgb = np.asarray(rgb_small, np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    green = (g > r * 1.04) & (g > b * 1.04) & (rgb.max(-1) - rgb.min(-1) > 0.05)
    cls[green] = VEG
    far = ~green & (np.hypot(P[..., 0], P[..., 1]) > FAR_M)
    cls[far] = FAR
    ground = (N[..., 2] > 0.75) & (np.abs(P[..., 2]) < np.maximum(0.35, 0.02 * np.hypot(P[..., 0], P[..., 1])))
    cls[ground] = GROUND
    cls[~mask] = SKY
    return cls


def infer(rgb, model_path, *, ort=None, cancel=None):
    """Run MoGe-2 on an RGB uint8/float image (any size; the analysis copy). Returns geo dict (+ ``cam.infer_s``)."""
    if ort is None:
        from . import precise_setup as PS
        ort = PS.import_onnxruntime()
    a = np.asarray(rgb)[..., :3].astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    fh, fw = a.shape[:2]
    W, H = infer_size(fw, fh)
    x = resize_bilinear(a, W, H).clip(0, 1)
    opts = ort.SessionOptions()
    try:
        opts.log_severity_level = 3
    except Exception:
        pass
    try:
        sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    except Exception:
        # non-ASCII user folders on Windows: hand the model over as bytes instead of a path
        with open(model_path, "rb") as fh_:
            sess = ort.InferenceSession(fh_.read(), sess_options=opts, providers=["CPUExecutionProvider"])
    if cancel and cancel():
        raise RuntimeError("已取消")
    t = time.time()
    pts, nrm, msk, scale = sess.run(None, {"image": np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32),
                                           "num_tokens": np.array(NUM_TOKENS, np.int64)})
    t_inf = time.time() - t
    geo = postprocess(pts[0], nrm[0], msk[0] > 0.5, float(np.ravel(scale)[0]), x)
    geo["cam"]["infer_s"] = round(t_inf, 2)
    geo["frame"] = (fw, fh)
    return geo


def save_geo(path, geo):
    import json
    np.savez_compressed(path, P=geo["P"], N=geo["N"], mask=geo["mask"], cls=geo["cls"],
                        cam=np.array(json.dumps(geo["cam"])), frame=np.array(geo.get("frame", (0, 0))))


def load_geo(path, frame=None):
    """A recorded geo (``save_geo``; or the research ``*_geo.npz`` + its ``*_moge_cam.json``)."""
    import json
    from pathlib import Path
    z = np.load(path)
    P, N, mask = z["P"], z["N"], z["mask"].astype(bool)
    if "cam" in z.files:
        cam = json.loads(str(z["cam"]))
    else:
        cam = json.loads(Path(str(path).replace("_geo.npz", "_moge_cam.json")).read_text())
        cam.pop("R_world_from_cam", None)
    if "cls" in z.files:
        cls = z["cls"]
    else:
        cls = classify(P, N, mask, np.full(mask.shape + (3,), 0.5, np.float32))
        if "seg" in z.files:            # research npz: SegFormer classes (0 sky 1 ground 2 struct 3 veg 4 rock)
            seg = z["seg"]
            cls = np.where(seg == 3, VEG, np.where(seg == 1, GROUND, np.where(seg == 0, SKY, np.where(seg == 4, FAR, STRUCT)))).astype(np.uint8)
            cls[~mask] = SKY
    fr = tuple(int(v) for v in z["frame"]) if "frame" in z.files else None
    return {"P": P, "N": N, "mask": mask, "cls": cls, "cam": cam, "frame": fr if fr and fr[0] else frame}


# ---------------------------------------------------------------------------------------- camera / projection

def horizon(cam, frame) -> float:
    hw = frame[1] / float(frame[0])
    tv = math.tan(math.radians(cam["hfov_deg"]) / 2.0) * hw
    return 0.5 + math.tan(math.radians(cam["pitch_deg"])) / (2.0 * tv)


def apply_camera(data, cam, frame) -> dict:
    """The measured camera replaces the LLM camera (height, pitch, roll, hFOV, horizon). Returns the LLM camera."""
    c = data.setdefault("camera", {})
    old = {k: c.get(k) for k in ("height_m", "pitch_deg", "roll_deg", "hfov_deg", "horizon_y_frac")}
    for k in ("height_m", "pitch_deg", "roll_deg", "hfov_deg"):
        c[k] = float(cam[k])
    c["horizon_y_frac"] = round(horizon(cam, frame), 4)
    return old


class Projector:
    """Project the ground-aligned local frame with the builder's camera convention.

    Horizontal sensor fit, square pixels, zero shift; (u, v) have a top-left
    origin. The rotation is Blender's Euler((90 + pitch), roll, 0), 'XYZ'.
    """

    def __init__(self, cam, frame):
        self.h = float(cam["height_m"])
        p = math.radians(float(cam["pitch_deg"]))
        r = math.radians(float(cam.get("roll_deg", 0.0)))
        sp, cp, sr, cr = math.sin(p), math.cos(p), math.sin(r), math.cos(r)
        self.tx = math.tan(math.radians(float(cam["hfov_deg"])) / 2)
        self.ty = self.tx * frame[1] / float(frame[0])
        # Ry(roll) @ Rx(90 + pitch): local -Z is forward, +Y up, +X right.
        # A 2D rotation after projection is not equivalent at nonzero pitch.
        self.f = np.array([sr * sp, cp, cr * sp])
        self.up = np.array([sr * cp, -sp, cr * cp])
        self.rt = np.array([cr, 0.0, -sr])

    def __call__(self, pt):
        d = np.asarray(pt, float) - np.array([0, 0, self.h])
        zc = d @ self.f
        if zc <= 0.1:
            return None
        return 0.5 + (d @ self.rt) / (zc * 2 * self.tx), 0.5 - (d @ self.up) / (zc * 2 * self.ty)


def prim_corners(pr):
    a = math.radians(pr.get("rot_z_deg", 0))
    ca, sa = math.cos(a), math.sin(a)
    z0 = pr.get("z_offset", 0)
    out = []
    for sx in (-.5, .5):
        for sy in (-.5, .5):
            for z in (z0, z0 + pr["h"] + pr.get("roof_h", 0)):
                out.append((pr["x"] + sx * pr["w"] * ca - sy * pr["d"] * sa,
                            pr["y"] + sx * pr["w"] * sa + sy * pr["d"] * ca, z))
    return out


def _window(geo, u0, u1, v0, v1):
    H, W = geo["mask"].shape
    u0, u1 = max(0.0, u0), min(1.0, u1)
    v0, v1 = max(0.0, v0), min(1.0, v1)
    if u1 - u0 < 1e-3 or v1 - v0 < 1e-3:
        return None
    return (slice(int(v0 * H), max(int(v0 * H) + 1, int(v1 * H))),
            slice(int(u0 * W), max(int(u0 * W) + 1, int(u1 * W))))


def measured(geo, us, vs, classes):
    sl = _window(geo, min(us), max(us), min(vs), max(vs))
    if sl is None:
        return None
    P = geo["P"][sl]
    sel = np.isin(geo["cls"][sl], classes)
    if sel.sum() < 6:
        sel = np.isin(geo["cls"][sl], NONGROUND)
        if sel.sum() < 6:
            return None
    r = np.hypot(P[..., 0], P[..., 1])[sel]
    return float(np.median(r)), int(sel.sum())


# ---------------------------------------------------------------------------------------- prompt block (variant D)

def _label(mask):
    """4-connected components of a small boolean grid (pure numpy/python; the grid is ≤ ~200×120)."""
    lab = np.zeros(mask.shape, np.int32)
    n = 0
    H, W = mask.shape
    for y0 in range(H):
        for x0 in range(W):
            if mask[y0, x0] and not lab[y0, x0]:
                n += 1
                stack = [(y0, x0)]
                lab[y0, x0] = n
                while stack:
                    y, x = stack.pop()
                    for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                            lab[yy, xx] = n
                            stack.append((yy, xx))
    return lab, n


def depth_block(geo, frame) -> str:
    P, cls = geo["P"], geo["cls"]
    cam = geo["cam"]
    H, W = cls.shape
    R = np.hypot(P[..., 0], P[..., 1])
    Z = P[..., 2]
    valid = cls != SKY
    hz = horizon(cam, frame)
    lines = ["MEASURED GEOMETRY (from a local monocular metric depth model run on this photo; trust it for the camera and",
             "for distances up to ~100 m; beyond ~150 m it underestimates, so for far mountains/dunes keep your own judgement):",
             f"camera: height_m {cam['height_m']:.2f}, pitch_deg {cam['pitch_deg']:.2f}, roll_deg {cam['roll_deg']:.2f}, "
             f"hfov_deg {cam['hfov_deg']:.1f}, horizon_y_frac {hz:.3f}. USE THESE CAMERA VALUES EXACTLY.",
             "depth grid (8 columns left->right x 6 rows top->bottom; each cell = dominant class : median horizontal distance m "
             "/ median height above ground m; '-' = sky):"]
    for r in range(6):
        row = []
        for c in range(8):
            sl = (slice(r * H // 6, (r + 1) * H // 6), slice(c * W // 8, (c + 1) * W // 8))
            v = valid[sl]
            if v.mean() < 0.25:
                row.append("-")
                continue
            k = int(np.bincount(cls[sl][v], minlength=5).argmax())
            row.append(f"{CLS_SHORT[k]}:{np.median(R[sl][v]):.0f}/{np.median(Z[sl][v]):.1f}")
        lines.append(" | ".join(row))
    lines.append("depth blobs (connected objects; screen bbox x0-x1,y0-y1 fractions; distance range m; top height m):")
    st = max(1, W // 160)
    Rs, Zs, Cs = R[::st, ::st], Z[::st, ::st], cls[::st, ::st]
    hs, ws = Cs.shape
    lr = np.log(np.clip(Rs, 0.5, None))
    gy, gx = np.gradient(lr)
    blobs = []
    for k in (STRUCT, VEG, FAR):
        sel = (Cs == k) & (np.hypot(gx, gy) < 0.08 * st) & (Zs > 0.3)
        lab, n = _label(sel)
        for i in range(1, n + 1):
            m = lab == i
            cnt = int(m.sum())
            if cnt < 0.002 * hs * ws:
                continue
            ys, xs = np.nonzero(m)
            rr, zz = Rs[m], Zs[m]
            blobs.append((cnt, f"{CLS_NAME[k]} x {xs.min() / ws:.2f}-{(xs.max() + 1) / ws:.2f}, y {ys.min() / hs:.2f}-"
                               f"{(ys.max() + 1) / hs:.2f}: {np.percentile(rr, 10):.0f}-{np.percentile(rr, 90):.0f} m, "
                               f"top {np.percentile(zz, 97):.1f} m"))
    for _, t in sorted(blobs, key=lambda b: -b[0])[:18]:
        lines.append("- " + t)
    lines.append("Place every primitive / foliage mass at the measured distance of its blob (x,y in metres from these numbers), "
                 "with heights from the measured tops; shape the terrain so its heights match the ground cells.")
    return "\n".join(lines)


def fov_warning(llm_cam, cam):
    try:
        diff = abs(float(llm_cam.get("hfov_deg")) - float(cam["hfov_deg"]))
    except (TypeError, ValueError):
        return ""
    return f"深度模型测得的视角与 AI 估计相差 {diff:.0f}°，玻璃 / 镜面 / 夜景可能测不准" if diff > FOV_WARN_DEG else ""


# ---------------------------------------------------------------------------------------- snap (P0 distance + P1 size)

def _scale_about_camera(item, ratio, keys_size):
    """Rescale a grounded layout in XY (legacy name; not a full 3D homothety).

    ``z_offset`` is clearance above terrain, not an absolute world-space Z.
    The builder re-anchors the base on the terrain at the new XY. Inserting a
    camera-height correction here would float or bury grounded buildings.
    Keep that clearance; vertical projection is allowed to change as depth is
    corrected. A screen-invariant adjustment requires a terrain-aware solve.
    """
    item["x"] = item["x"] * ratio
    item["y"] = item["y"] * ratio
    for k in keys_size:
        if k in item:
            item[k] = item[k] * ratio


def _size_snap(pr, geo, proj, r_med, classes):
    """P1: height (top of the measured points) and lateral width of the depth region behind the block."""
    pj = [q for q in map(proj, prim_corners(pr)) if q]
    if len(pj) < 4:
        return None
    us, vs = [q[0] for q in pj], [q[1] for q in pj]
    du, dv = max(us) - min(us), max(vs) - min(vs)
    sl = _window(geo, min(us) - SIZE_PAD_U * du, max(us) + SIZE_PAD_U * du, min(vs) - SIZE_PAD_V * dv, max(vs))
    if sl is None:
        return None
    P = geo["P"][sl]
    rr = np.hypot(P[..., 0], P[..., 1])
    sel = np.isin(geo["cls"][sl], classes) & (rr > r_med * 0.75) & (rr < r_med * 1.33)
    if sel.sum() < 30:
        return None
    pts = P[sel]
    out = {}
    z0 = pr.get("z_offset", 0.0)
    old_tot = pr["h"] + pr.get("roof_h", 0.0)
    top = float(np.percentile(pts[:, 2], 95))
    new_tot = float(np.clip(top - z0, 0.7 * old_tot, SIZE_GROW_MAX * old_tot))
    if old_tot > 0.05 and abs(new_tot / old_tot - 1) > 0.03:
        k = new_tot / old_tot
        pr["h"] *= k
        pr["roof_h"] = pr.get("roof_h", 0.0) * k
        out["h_ratio"] = round(k, 2)
    if int(pr.get("repeat", 1) or 1) > 1:
        return out
    az = math.atan2(pr["x"], pr["y"])                    # lateral axis ⟂ line of sight
    lat = np.array([math.cos(az), -math.sin(az)])
    lm = pts[:, :2] @ lat
    m0, m1 = np.percentile(lm, [5, 95])
    corners = np.array([c[:2] for c in prim_corners(pr)])
    le = corners @ lat
    e0, e1 = float(le.min()), float(le.max())
    L_e, L_m = e1 - e0, float(m1 - m0)
    if L_e < 0.2 or L_m < 0.2:
        return out
    k = float(np.clip(L_m / L_e, 0.7, 1.4))
    a = math.radians(pr.get("rot_z_deg", 0))
    ax_w = np.array([math.cos(a), math.sin(a)])          # local X (w) in world
    key = "w" if abs(ax_w @ lat) >= 0.7071 else "d"
    if abs(k - 1) > 0.03:
        pr[key] *= k
        out["w_ratio" if key == "w" else "d_ratio"] = round(k, 2)
    shift = float(np.clip((m0 + m1) / 2 - (e0 + e1) / 2, -0.3 * L_e, 0.3 * L_e))
    if abs(shift) > 0.05:
        pr["x"] += shift * lat[0]
        pr["y"] += shift * lat[1]
        out["shift_m"] = round(shift, 2)
    return out


def snap(data, geo, frame, size=True) -> dict:
    """Pin the LLM layout to the measured depth (in place). Returns {"items": [...], "snapped": n}."""
    proj = Projector(data["camera"], frame)
    rep = []
    n = 0
    for pr in data.get("primitives", []):
        if pr.get("type") in THIN:
            continue
        pj = [q for q in map(proj, prim_corners(pr)) if q]
        if len(pj) < 4:
            continue
        classes = CLASS_OF.get(pr.get("type"), NONGROUND)
        m = measured(geo, [q[0] for q in pj], [q[1] for q in pj], classes)
        if not m:
            continue
        r_llm = math.hypot(pr["x"], pr["y"])
        ratio = float(np.clip(m[0] / max(r_llm, 0.1), 0.4, 2.5))
        row = {"el": pr.get("label_zh") or pr.get("id"), "type": pr.get("type"), "llm_m": round(r_llm, 1),
               "depth_m": round(m[0], 1), "ratio": round(ratio, 2)}
        rep.append(row)
        if max(r_llm, m[0]) > MAXR:
            continue
        _scale_about_camera(pr, ratio, ("w", "d", "h", "roof_h", "repeat_dx", "repeat_dy"))
        n += 1
        if size and pr.get("type") in SIZE_SNAP:
            s = _size_snap(pr, geo, proj, m[0], classes)
            if s:
                row.update(s)
    for fm in data.get("foliage_masses", []):
        if fm.get("outline"):
            us = [q["x_frac"] for q in fm["outline"]]
            vs = [q["y_frac"] for q in fm["outline"]]
        else:
            pj = [q for q in map(proj, [(fm["x"] + dx, fm["y"], z) for dx in (-fm["crown_w"] / 2, fm["crown_w"] / 2)
                                        for z in (0, fm["crown_base_h"] + fm["crown_h"])]) if q]
            if len(pj) < 2:
                continue
            us, vs = [q[0] for q in pj], [q[1] for q in pj]
        m = measured(geo, us, vs, (VEG,))
        if not m:
            continue
        r_llm = math.hypot(fm["x"], fm["y"])
        ratio = float(np.clip(m[0] / max(r_llm, 0.1), 0.4, 2.5))
        rep.append({"el": fm.get("label_zh") or fm.get("id"), "type": "foliage", "llm_m": round(r_llm, 1),
                    "depth_m": round(m[0], 1), "ratio": round(ratio, 2)})
        if max(r_llm, m[0]) > MAXR:
            continue
        keys = ("crown_w", "crown_h", "crown_base_h", "trunk_w") if fm.get("outline") else \
            ("crown_w", "crown_d", "crown_h", "crown_base_h", "trunk_w")
        _scale_about_camera(fm, ratio, keys)
        n += 1
    return {"items": rep, "snapped": n}


def range_error(data, geo, frame):
    """3D fit: median |distance error| (%) of non-thin blocks and foliage within MAXR vs the measured depth, using the
    data's own camera for the screen footprint. Returns (pct or None, n, worst list)."""
    proj = Projector(data["camera"], frame)
    errs = []
    for pr in data.get("primitives", []):
        if pr.get("type") in THIN:
            continue
        pj = [q for q in map(proj, prim_corners(pr)) if q]
        if len(pj) < 4:
            continue
        m = measured(geo, [q[0] for q in pj], [q[1] for q in pj], CLASS_OF.get(pr.get("type"), NONGROUND))
        r = math.hypot(pr["x"], pr["y"])
        if m and max(r, m[0]) <= MAXR:
            errs.append((abs(math.log(m[0] / max(r, 0.1))), pr.get("label_zh") or pr.get("id"), round(r, 1), round(m[0], 1)))
    for fm in data.get("foliage_masses", []):
        if fm.get("outline"):
            us, vs = [q["x_frac"] for q in fm["outline"]], [q["y_frac"] for q in fm["outline"]]
        else:
            pj = [q for q in map(proj, [(fm["x"] + dx, fm["y"], z) for dx in (-fm["crown_w"] / 2, fm["crown_w"] / 2)
                                        for z in (0, fm["crown_base_h"] + fm["crown_h"])]) if q]
            if len(pj) < 2:
                continue
            us, vs = [q[0] for q in pj], [q[1] for q in pj]
        m = measured(geo, us, vs, (VEG,))
        r = math.hypot(fm["x"], fm["y"])
        if m and max(r, m[0]) <= MAXR:
            errs.append((abs(math.log(m[0] / max(r, 0.1))), fm.get("label_zh") or fm.get("id"), round(r, 1), round(m[0], 1)))
    if not errs:
        return None, 0, []
    med = float(np.median([e[0] for e in errs]))
    worst = [{"el": e[1], "placed_m": e[2], "depth_m": e[3]} for e in sorted(errs, key=lambda e: -e[0])[:4]]
    return round(100 * (math.exp(med) - 1), 1), len(errs), worst


def precise_pass(data, geo, frame, size=None) -> dict:
    """apply_camera + snap on a deep copy; returns (new_data, info)."""
    size = SIZE_SNAP_ENABLED if size is None else size
    d = copy.deepcopy(data)
    llm_cam = apply_camera(d, geo["cam"], frame)
    before, _n, _w = range_error(d, geo, frame)
    rep = snap(d, geo, frame, size=size)
    after, n, worst = range_error(d, geo, frame)
    info = {"llm_camera": llm_cam, "camera": {k: geo["cam"][k] for k in ("height_m", "pitch_deg", "roll_deg", "hfov_deg")},
            "err_before_pct": before, "err_after_pct": after, "n": n, "worst": worst, "snapped": rep["snapped"],
            "items": rep["items"], "warning": fov_warning(llm_cam, geo["cam"])}
    d["_precise"] = {k: v for k, v in info.items() if k != "items"}
    return d, info
