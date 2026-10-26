"""Synthetic depth measurements for the 精确模式 tests (numpy only): ray-cast a flat ground + axis-aligned boxes
from a plugin camera (x=0, y=0, z=h, looking +Y), returning a geo dict like precise_depth.postprocess."""
import math

import numpy as np


def ray_dirs(cam, W, H):
    tx = math.tan(math.radians(cam["hfov_deg"]) / 2)
    ty = tx * H / float(W)
    p = math.radians(cam["pitch_deg"])
    f = np.array([0, math.cos(p), math.sin(p)])
    up = np.array([0, -math.sin(p), math.cos(p)])
    rt = np.array([1.0, 0, 0])
    u = (np.arange(W) + 0.5) / W
    v = (np.arange(H) + 0.5) / H
    U, V = np.meshgrid(u, v)
    d = f[None, None] + ((U - 0.5) * 2 * tx)[..., None] * rt + ((0.5 - V) * 2 * ty)[..., None] * up
    return d


def geo(cam, W=320, H=180, boxes=(), far=400.0):
    """boxes: (x0, x1, y0, y1, z0, z1, cls). Returns geo dict (P, N, mask, cls, cam, frame)."""
    D = ray_dirs(cam, W, H)
    O = np.array([0, 0, cam["height_m"]])
    t = np.full((H, W), np.inf)
    N = np.zeros((H, W, 3), np.float32)
    C = np.zeros((H, W), np.uint8)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(D[..., 2] < -1e-6, -O[2] / D[..., 2], np.inf)
    hit = tg < t
    t[hit] = tg[hit]
    N[hit] = (0, 0, 1)
    C[hit] = 1
    for (x0, x1, y0, y1, z0, z1, k) in boxes:
        lo, hi = np.array([x0, y0, z0]), np.array([x1, y1, z1])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - O) / D
            t2 = (hi - O) / D
        tn = np.nanmax(np.minimum(t1, t2), -1)
        tf = np.nanmin(np.maximum(t1, t2), -1)
        hb = (tn <= tf) & (tn > 0) & (tn < t)
        t[hb] = tn[hb]
        N[hb] = (0, -1, 0)
        C[hb] = k
    P = (O + D * np.where(np.isfinite(t), t, 0)[..., None]).astype(np.float32)
    mask = np.isfinite(t) & (np.hypot(P[..., 0], P[..., 1]) < far)
    C[~mask] = 0
    return {"P": P, "N": N, "mask": mask, "cls": C, "cam": dict(cam), "frame": (W, H)}
