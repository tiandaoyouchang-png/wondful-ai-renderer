"""1.2.3 多角度场景一致 (高级): render more cameras of one scene so they agree with an accepted master view.

The user renders and accepts one camera (主视角) with the normal flow, then marks it. Every other
selected camera is rendered nearest-angle first. Its edit canvas is the clay base with the
environment part replaced by a camera projection of every view accepted so far (master first).
The projection uses Blender position / normal passes: a world point takes the colour from the
source view that sees it most head-on, and sky directions take the colour of a source's sky.
The text adds a light sentence locked in world space (from one analysis call on the master
image), the per-view block positions (環境体块 placement for this camera), and a note about the
projected base. The reference photo never reaches the image model: it is read as text only,
like in the main flow.

Retry policy (user, 2026-10-17; 1.2.4 adds the low tier): one image call per camera, at most one retry,
and only when the backend reported the ``low`` quality tier (2026-10-20), or when
- the product is out of spec: detail edge > 6 px, product colour ΔE > 15, offset > 4 px or
  scale > 1.5 % (产品还原度 of the raw AI image); or
- the environment grossly disagrees with the master on geometry pixels only (sky excluded):
  reprojection ΔE > 15 while the two views share at least half of the frame (covis ≥ 0.5).
  Calibrated on the prototype runs: no first try of 12 would have fired.

Pure numpy except the ``bpy`` helpers at the end (geometry passes, obstacle check, wider
blockout), which import ``bpy`` lazily. The worker thread only uses the numpy part.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from .layered_output import blur, dilate, erode, resize

WORK_LONG = 960            # projection works at this long edge (seconds instead of tens)
EDGE_PX = 6.0              # retry when the raw AI detail edge (vs the detail pass) is above this
COLOUR_DE = 15.0           # … or the product colour ΔE
OFFSET_PX = 4.0            # … or the product moved
SCALE_PCT = 1.5            # … or was scaled
ENV_GEOM_DE = 15.0         # environment: geometry-pixel reprojection ΔE vs the master
ENV_MIN_COVIS = 0.4        # … only judged when the views share at least this much of the frame (1.2.5: 0.5 → 0.4;
                           #     the 2026-10-20 orbit's broken 216° view had covis 0.44 and never retried)
MAX_CALLS = 2              # one call + at most one retry per camera


def camera_calls(pref_max) -> int:
    """Image calls per 多角度 camera: MAX_CALLS, capped by 偏好设置「最多生成次数」 (1.2.5)."""
    try:
        return max(1, min(MAX_CALLS, int(pref_max)))
    except (TypeError, ValueError):
        return MAX_CALLS
MASTER_BIAS = 0.30         # the master wins over AI views unless they see the point much more head-on (1.2.5: 0.10)
HOP_PENALTY = 0.20         # per generation of AI-from-AI (a view rendered mostly from other AI views)
MAX_SOURCE_HOP = 1         # 1.2.5: only the master and views rendered mostly from it are projection sources
HOP_SHARE = 0.15           # a source counts for a view's hop when it coloured at least this share of it
STRETCH_MAX = 4.0          # source pixel footprint / target footprint above this = stretched streak → not used
STRETCH_MAX_MASTER = 64.0  # the master is the truth: only extreme smears (> 8× linear) are refused (real orbit test
                           #     2026-10-20: 9.0 dropped 17 % of the master's coverage at 72° and its ΔE rose 5.1 → 8.0)
AUX_TAG = "wondful_mv_aux"
POOL_FILE = "pool.json"
LIGHT_FILE = "light.json"

FILLER_HINT = "与本场景同风格的普通建筑（参考图以外补充的周边体块）：比主体建筑朴素、不要比它们更显眼"
CLOSURE_HINT = "与房间其他墙同样的平整墙面：颜色、材质与其他墙一致，不要在这里新加家具、门窗或装饰"
BUILDING_WORDS = ("建筑", "楼", "房", "店", "塔", "厦", "亭", "building", "house", "tower", "shop")
WALL_WORDS = ("墙", "wall", "窗", "window", "帘", "curtain", "天花", "ceiling")

BASE_NOTE = ("这张底图的环境部分已经按同一场景其他角度（{sources}）的照片投影着色（可能有拉伸、接缝、重影和灰色未覆盖区）："
             "保留已着色部分的颜色、材质、天空和光照，修正拉伸和接缝，灰色未覆盖区按灰蓝色体块和场景描述补成同样风格、"
             "同样光线的真实环境；产品部分仍以白模为准，严格锁定。")


# --------------------------------------------------------------------------- camera math

def camera_record(name, matrix, lens_angle, sensor_fit, width, height, ortho=False) -> dict:
    """JSON-safe camera: 4×4 world matrix, tan of the half field of view per axis, canvas size.
    ``lens_angle`` is Blender's ``camera.data.angle`` (the fitted axis); AUTO fits the longer side."""
    t = math.tan(float(lens_angle) / 2.0)
    fit = (sensor_fit or "AUTO").upper()
    horizontal = fit == "HORIZONTAL" or (fit == "AUTO" and width >= height)
    tx, ty = (t, t * height / width) if horizontal else (t * width / height, t)
    return {"name": str(name), "matrix": [list(map(float, r)) for r in matrix], "tx": tx, "ty": ty,
            "res": [int(width), int(height)], "ortho": bool(ortho)}


def work_size(width, height, long_edge=WORK_LONG):
    s = min(1.0, float(long_edge) / max(width, height))
    return max(8, int(round(width * s))), max(8, int(round(height * s)))


def _mat(rec):
    return np.asarray(rec["matrix"], np.float64)


def project(rec, P, W, H):
    """World points (...,3) → pixel u, v (float, top-left origin) and depth (>0 in front)."""
    M = _mat(rec)
    pl = (np.asarray(P, np.float64) - M[:3, 3]) @ M[:3, :3]
    z = -pl[..., 2]
    zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
    nx = pl[..., 0] / zs / rec["tx"]
    ny = pl[..., 1] / zs / rec["ty"]
    return (nx + 1) / 2 * W - 0.5, (1 - ny) / 2 * H - 0.5, z


def rays(rec, W, H):
    """World-space view ray direction of every pixel centre."""
    M = _mat(rec)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u / W * 2 - 1) * rec["tx"], (1 - v / H * 2) * rec["ty"], -np.ones_like(u)], -1)
    return d @ M[:3, :3].T


def forward(rec):
    return -_mat(rec)[:3, 2]


def yaw_deg(rec) -> float:
    """Heading of the camera's view direction, 0 = +Y, clockwise seen from above (+X = 90)."""
    f = forward(rec)
    return math.degrees(math.atan2(f[0], f[1]))


def orbit_angle(rec_a, rec_b, centre) -> float:
    """Horizontal angle (deg, 0..180) between two cameras seen from the product centre."""
    c = np.asarray(centre, np.float64)
    a = _mat(rec_a)[:3, 3] - c
    b = _mat(rec_b)[:3, 3] - c
    a2, b2 = math.atan2(a[0], a[1]), math.atan2(b[0], b[1])
    d = abs((math.degrees(a2 - b2) + 180.0) % 360.0 - 180.0)
    return d


def pick_cameras(slots, selected, master, is_present=lambda obj: True):
    """Which other cameras 多角度场景一致 renders (no bpy; ``obj`` only needs ``name`` and ``type``).

    ``slots`` are the panel's camera rows as (camera_or_None, enabled).  With at least one row the list is the
    input: enabled rows with a camera present in the scene, minus the master and duplicates (first enabled row wins).
    With no rows at all the old behaviour stays: the selected cameras except the master.
    Returns {"cameras", "source": "list" | "selected", "missing": [row numbers], "master_rows", "duplicate_rows",
    "disabled_rows"} (row numbers 1-based) so the panel can say why a row is skipped."""
    mname = getattr(master, "name", None)
    out = {"cameras": [], "source": "list" if slots else "selected", "missing": [], "master_rows": [],
           "duplicate_rows": [], "disabled_rows": []}
    if not slots:
        seen = set()
        for o in selected or []:
            if getattr(o, "type", "") == "CAMERA" and o.name != mname and o.name not in seen:
                seen.add(o.name)
                out["cameras"].append(o)
        return out
    seen = set()
    for row, (cam, enabled) in enumerate(slots, 1):
        if cam is None or getattr(cam, "type", "") != "CAMERA" or not is_present(cam):
            out["missing"].append(row)
            continue
        if cam.name == mname:
            out["master_rows"].append(row)
            continue
        if not enabled:
            out["disabled_rows"].append(row)
            continue
        if cam.name in seen:
            out["duplicate_rows"].append(row)
            continue
        seen.add(cam.name)
        out["cameras"].append(cam)
    return out


# --------------------------------------------------------------------------- 1.2.3 ⑥ 渲染视角 (shared 「相机 N」 list)

def checked_views(slots, is_present=lambda obj: True):
    """⑥ 渲染视角: the checked rows' cameras in list order (no bpy).  Rows without a camera, non-cameras,
    cameras gone from the scene, unchecked rows and repeats (first row wins) are dropped.  The first one is
    the 主视角; the rest are rendered by 多角度场景一致 against it."""
    out, seen = [], set()
    for cam, enabled in slots or []:
        if not enabled or cam is None or getattr(cam, "type", "") != "CAMERA" or not is_present(cam):
            continue
        if cam.name in seen:
            continue
        seen.add(cam.name)
        out.append(cam)
    return out


def cameras_to_append(existing_names, candidates):
    """Cameras to add to the list for a drop: cameras only, not already listed, each once, in drop order."""
    seen = set(n for n in existing_names or [] if n)
    out = []
    for obj in candidates or []:
        if obj is None or getattr(obj, "type", "") != "CAMERA" or obj.name in seen:
            continue
        seen.add(obj.name)
        out.append(obj)
    return out


def cameras_in_collection(coll):
    """Every camera in ``coll`` and its child collections (depth first, collection order, each once).
    Duck-typed: ``coll.objects`` and ``coll.children``; a collection reached twice is walked once."""
    out, names, walked = [], set(), set()

    def walk(c, depth):
        if c is None or depth > 64 or id(c) in walked:
            return
        walked.add(id(c))
        for o in list(getattr(c, "objects", []) or []):
            if getattr(o, "type", "") == "CAMERA" and o.name not in names:
                names.add(o.name)
                out.append(o)
        for child in list(getattr(c, "children", []) or []):
            walk(child, depth + 1)
    walk(coll, 0)
    return out


def drop_note(added, skipped, from_collection=""):
    """One line for the 拖入 fields: what a drop did."""
    if not added and not skipped:
        return f"集合「{from_collection}」里没有相机" if from_collection else ""
    parts = []
    if added:
        parts.append(f"已加入 {len(added)} 个相机" if len(added) > 1 else f"已加入 {added[0]}")
    if skipped:
        parts.append(f"跳过已在列表的 {len(skipped)} 个" if len(skipped) > 1 else f"{skipped[0]} 已在列表")
    return " · ".join(parts)


def move_index(n, index, direction):
    """New index after moving row ``index`` up (-1) or down (+1) in a list of ``n`` rows, or None if it can't move."""
    if not 0 <= index < n:
        return None
    to = index + (1 if direction > 0 else -1)
    return to if 0 <= to < n else None


def scene_blockout_exists(scene):
    """④ ran: a 「从参考图生成场景粗模」 blockout (tagged collection) with objects is in the scene."""
    try:
        for coll in scene.collection.children_recursive:
            if coll.get("wondful_scene_blockout") and len(coll.all_objects):
                return True
    except Exception:
        pass
    return False


def view_phase(index, total, name, master=False):
    """Progress text of one view in a ⑥ multi-view run: 「视角 i/N · 名称（主视角）」."""
    return f"视角 {index}/{total} · {name}" + ("（主视角）" if master else "")


def next_view_name(existing, prefix="Wondful 视角"):
    """「Wondful 视角 N」 with the smallest N ≥ 1 not in ``existing``."""
    names = set(existing)
    n = 1
    while f"{prefix} {n}" in names:
        n += 1
    return f"{prefix} {n}"


def order_views(master, others, centre):
    """Nearest angle to the master first (each view can reuse every view accepted before it)."""
    return sorted(others, key=lambda r: (orbit_angle(master, r, centre), r["name"]))


def same_camera(a, b, tol=1e-4) -> bool:
    try:
        return (np.allclose(_mat(a), _mat(b), atol=tol) and abs(a["tx"] - b["tx"]) < tol
                and list(a["res"]) == list(b["res"]))
    except Exception:
        return False


# --------------------------------------------------------------------------- image helpers

def bilinear(img, u, v):
    H, W = img.shape[:2]
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = (u - x0)[..., None], (v - y0)[..., None]
    a, b, c, d = img[y0, x0], img[y0, x0 + 1], img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def lab(rgb255):
    from .material_color import srgb_to_lab
    return srgb_to_lab(np.asarray(rgb255, np.float32) / 255.0)


def sky_mask(pos, img=None):
    """Sky pixels of one view without a segmentation model (the 110 MB SegFormer is not bundled):
    blockout sky (no geometry in the position pass), shrunk a little, and — when the AI image is
    given — only where the image is smooth (the AI often draws trees / poles wider than their
    blocks; those carry texture)."""
    sky = np.asarray(pos)[..., 3] < 0.5
    sky = erode(sky, 3)
    if img is not None:
        from .layered_output import box_mean
        lum = np.asarray(img, np.float32)[..., :3].mean(-1) / 255.0
        var = box_mean(lum * lum, 3) - box_mean(lum, 3) ** 2
        textured = dilate(np.sqrt(np.maximum(var, 0.0)) > 0.03, 2)
        sky &= ~textured
    return sky


def skyfill(img, sky):
    """The image with every non-sky pixel replaced by the sky colour of its row (linear
    interpolation between sky pixels, the row's median outside them); rows without sky copy the
    nearest row above that has some."""
    H, W = img.shape[:2]
    out = img.astype(np.float32).copy()
    xs = np.arange(W)
    last = None
    first = None
    for y in range(H):
        sx = np.nonzero(sky[y])[0]
        if len(sx) >= 3:
            for c in range(3):
                out[y, :, c] = np.interp(xs, sx, img[y, sx, c])
            med = np.median(img[y, sx, :3], 0)
            out[y, :sx[0], :3] = med
            out[y, sx[-1] + 1:, :3] = med
            last = out[y].copy()
            if first is None:
                first = y
        elif last is not None:
            out[y] = last
    if first is not None:
        out[:first] = out[first]
    return out


# --------------------------------------------------------------------------- projection

def correspond(vk, vj):
    """For every pixel of view k: where it lands in view j and whether view j saw that point.

    ``vk`` / ``vj``: dicts with ``cam`` (camera_record), ``pos`` (H,W,4 world position + hit
    alpha), ``mask`` (H,W product mask 0..1). Returns U, V, vis (geometry seen by j, depth test,
    not behind j's product), sky (view-k background pixel in front of camera j), hit, sky_in."""
    Pk, Pj = vk["pos"], vj["pos"]
    H, W = Pk.shape[:2]
    hit = Pk[..., 3] > 0.5
    u, v, z = project(vj["cam"], Pk[..., :3], W, H)
    inside = (z > 0) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)
    Q = Pj[vi, ui]
    dist = np.linalg.norm(Pk[..., :3] - Q[..., :3], axis=-1)
    mj = dilate(np.asarray(vj["mask"]) > 0.5, max(2, int(round(W / 240))))
    vis = hit & inside & (Q[..., 3] > 0.5) & (dist < 0.03 * np.abs(z) + 0.08) & ~mj[vi, ui]
    D = rays(vk["cam"], W, H)
    Mj = _mat(vj["cam"])
    su, sv, sz = project(vj["cam"], Mj[:3, 3] + D, W, H)
    sky = (~hit) & (sz > 0)
    sky_in = sky & (su >= 0) & (su <= W - 1) & (sv >= 0) & (sv <= H - 1)
    return np.where(hit, u, su), np.where(hit, v, sv), vis, sky, hit, sky_in


def fj2(rec, W):
    """Squared focal length in working pixels (pixel footprint ∝ distance² / (cos · f²))."""
    f = (W / 2.0) / max(float(rec.get("tx", 1.0)), 1e-6)
    return f * f


def view_hop(stats, sources) -> int:
    """Generation of a new view: 1 + the highest hop among the sources that coloured ≥ HOP_SHARE of it
    (master = 0). A view rendered mostly from AI views is hop 2 and is no longer a projection source."""
    share = (stats or {}).get("share") or {}
    hops = [int(v.get("hop", 0 if v.get("master") else 1)) for v in sources
            if share.get(v["name"], 0.0) >= HOP_SHARE]
    return 1 + (max(hops) if hops else 0)


def usable_source(entry) -> bool:
    """1.2.5: only the master and views at most MAX_SOURCE_HOP generations from it are re-projected."""
    return bool(entry.get("master")) or int(entry.get("hop", 1)) <= MAX_SOURCE_HOP


def build_base(vk, sources, clay):
    """Accumulated projection of every accepted source view onto view k (working resolution).

    ``sources``: list of view dicts with ``img`` (H,W,3 float 0..255), ``sky`` (bool) and
    ``master`` (bool), master first. ``clay``: view k's clay base at working size (H,W,3).
    Returns (env image, coverage 0..1, source map, stats). Product pixels stay clay."""
    H, W = vk["pos"].shape[:2]
    Pk = vk["pos"]
    Nk = vk["nrm"][..., :3]
    Nk = Nk / np.maximum(np.linalg.norm(Nk, axis=-1, keepdims=True), 1e-6)
    mk = np.asarray(vk["mask"], np.float32)
    out = np.asarray(clay, np.float32)[..., :3].copy()
    ok_ = _mat(vk["cam"])[:3, 3] - Pk[..., :3]
    dk = np.maximum(np.linalg.norm(ok_, axis=-1), 1e-6)
    cos_k = np.abs((Nk * (ok_ / dk[..., None])).sum(-1))
    foot_k = (dk ** 2 / np.maximum(cos_k, 0.05)) / (fj2(vk["cam"], W) + 1e-12)
    best = np.full((H, W), -1.0, np.float32)
    srcmap = np.zeros((H, W), np.int16)
    skydone = np.zeros((H, W), bool)
    cached = []
    for n, vj in enumerate(sources, start=1):
        U, V, vis, sky, hit, sky_in = correspond(vk, vj)
        cached.append((U, V, sky, sky_in))
        dvec = _mat(vj["cam"])[:3, 3] - Pk[..., :3]
        dj = np.maximum(np.linalg.norm(dvec, axis=-1, keepdims=True), 1e-6)
        dvec /= dj
        cos_j = np.abs((Nk * dvec).sum(-1))
        face = cos_j + (MASTER_BIAS if vj.get("master") else -HOP_PENALTY * max(0, int(vj.get("hop", 1)) - 1))
        # 1.2.5 stretch clamp: a source that saw the surface much coarser (far / grazing) than this view smears
        # it into streaks (the 216° hedge-on-the-ground) → refused; the pixel stays for the style fill
        ratio = ((dj[..., 0] ** 2 / np.maximum(cos_j, 0.05)) / (fj2(vj["cam"], W) + 1e-12)) / foot_k
        vis = vis & (ratio <= (STRETCH_MAX_MASTER if vj.get("master") else STRETCH_MAX))
        take = vis & (face > best)
        if take.any():
            col = bilinear(vj["img"], U, V)
            out[take] = col[take]
            best[take] = face[take]
            srcmap[take] = n
    geom = srcmap > 0
    for n, (vj, (U, V, sky, sky_in)) in enumerate(zip(sources, cached), start=1):
        ss = vj["sky"]
        ui = np.clip(np.round(U).astype(int), 0, W - 1)
        vi = np.clip(np.round(V).astype(int), 0, H - 1)
        ok = sky_in & ss[vi, ui] & ~skydone
        if ok.any():
            out[ok] = bilinear(vj["img"], U, V)[ok]
            skydone |= ok
            srcmap[ok] = 100 + n
    for n, (vj, (U, V, sky, sky_in)) in enumerate(zip(sources, cached), start=1):
        ok = sky & ~skydone
        if ok.any() and vj["sky"].sum() > 30:
            out[ok] = bilinear(skyfill(vj["img"], vj["sky"]), U, V)[ok]
            skydone |= ok
            srcmap[ok] = 200 + n
    cover = blur((geom | skydone).astype(np.float32), 1) * (1.0 - mk)
    env = mk < 0.5
    n_env = max(1, int(env.sum()))
    stats = {"coverage": round(float((geom | skydone)[env].sum()) / n_env, 3),
             "coverage_geometry": round(float(geom[env].sum()) / n_env, 3),
             "sources": [v["name"] for v in sources],
             "share": {v["name"]: round(float(((srcmap == n) | (srcmap == 100 + n) | (srcmap == 200 + n))[env].sum()) / n_env, 3)
                       for n, v in enumerate(sources, start=1)}}
    return out, cover, srcmap, stats


# --------------------------------------------------------------------------- 1.2.5 场景补全 style fill
# Pixels no accepted view covers (the parts the reference photo never saw) used to stay clay grey, and the image
# model painted them as blank white walls. They now get a style patch rectified from the 主视角: facades of the
# measured buildings, the road / sidewalk surface, the hedges — tiled in world units on the 场景补全 layout.

FILL_MPP_FACADE = 0.06     # metres per texel of a rectified facade patch
FILL_MPP_GROUND = 0.03
FILL_ALPHA = 0.9           # how strongly an uncovered pixel takes the fill (the rest stays clay)
MAX_FACADE_PATCHES = 4
CLS_NONE, CLS_ROAD, CLS_SIDE, CLS_FACADE, CLS_VEG, CLS_CURB, CLS_POLE = range(7)
KIND_CLASS = {"building": CLS_FACADE, "hedge": CLS_VEG, "tree": CLS_VEG, "pole": CLS_POLE, "curb": CLS_CURB}
CLASS_WORD = {CLS_ROAD: "路面", CLS_SIDE: "人行道", CLS_FACADE: "房屋立面", CLS_VEG: "树篱 / 树", CLS_CURB: "路沿",
              CLS_POLE: "电杆"}


def _mirror(x, n):
    x = np.floor(x).astype(np.int64) % (2 * n)
    return np.where(x < n, x, 2 * n - 1 - x)


def _unit_n(N):
    N = np.asarray(N, np.float64)[..., :3]
    return N / np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-6)


def facade_tangent(P, N):
    """Horizontal unit tangent of vertical surfaces: from the position pass along the image row (render normals
    can be smoothed across box corners), the normal's perpendicular where the row has a depth jump."""
    P = np.asarray(P, np.float64)
    hit = P[..., 3] > 0.5
    g = np.zeros(P.shape[:2] + (2,))
    g[:, 1:-1] = P[:, 2:, :2] - P[:, :-2, :2]
    both = np.zeros(hit.shape, bool)
    both[:, 1:-1] = hit[:, 2:] & hit[:, :-2]
    ln = np.linalg.norm(g, axis=-1)
    dz = np.zeros(hit.shape)
    dz[:, 1:-1] = np.abs(P[:, 2:, 2] - P[:, :-2, 2])
    rng_ok = both & (ln > 1e-4) & (ln < 3.0) & (dz < 0.5 * ln + 0.05)
    tan = np.stack([-N[..., 1], N[..., 0]], -1)
    tan[rng_ok] = g[rng_ok]
    return tan / np.maximum(np.linalg.norm(tan, axis=-1, keepdims=True), 1e-6)


def in_box(P, b, tol=0.15):
    c = np.asarray(b["c"], np.float64)
    u = np.asarray(b["u"], np.float64)
    n = np.asarray(b["n"], np.float64)
    d = np.asarray(P, np.float64)[..., :2] - c
    a = d @ u
    bb = d @ n
    z = np.asarray(P, np.float64)[..., 2]
    return ((a >= b["a"][0] - tol) & (a <= b["a"][1] + tol) & (bb >= b["b"][0] - tol) & (bb <= b["b"][1] + tol)
            & (z >= b["z"][0] - tol) & (z <= b["z"][1] + tol))


def class_map(vk, layout):
    """Per pixel of view k: what the surface is (CLS_*) and a style index (which facade patch)."""
    P = np.asarray(vk["pos"], np.float64)
    H, W = P.shape[:2]
    hit = P[..., 3] > 0.5
    N = _unit_n(vk["nrm"])
    cls = np.zeros((H, W), np.int8)
    style = np.zeros((H, W), np.int64)
    up = np.abs(N[..., 2]) > 0.7
    cls[hit & up] = CLS_ROAD
    st = (layout or {}).get("street")
    if st:
        a0 = np.asarray(st["a0"], np.float64)
        t = np.asarray(st["t"], np.float64)
        nperp = np.array([t[1], -t[0]])
        off = (P[..., :2] - a0) @ nperp
        cls[hit & up & (np.abs(off) > float(st["half_width"]) + 0.3)] = CLS_SIDE
    vert = hit & ~up
    cls[vert] = CLS_FACADE
    tan = facade_tangent(P, N)
    s = (P[..., :2] * tan).sum(-1)
    plane = (P[..., :2] * np.stack([tan[..., 1], -tan[..., 0]], -1)).sum(-1)
    style[vert] = (np.floor(s[vert] / 9.0).astype(np.int64) * 7 + np.round(plane[vert] / 3.0).astype(np.int64) * 13) % 997
    for b in (layout or {}).get("boxes", []):
        k = KIND_CLASS.get(b.get("kind"), CLS_FACADE)
        inside = hit & in_box(P, b)
        if k == CLS_FACADE:
            inside &= ~(up & (P[..., 2] < b["z"][0] + 0.7))       # the ground at its foot stays ground
        cls[inside] = k
        style[inside] = int(b.get("style", 0))
    return cls, style


def _rectify(mv, origin, ax_u, ax_v, nu, nv, mpp, tol, rel=0.03):
    """Sample the master image on a world grid (origin + ax_u·i·mpp + ax_v·j·mpp): colour + validity
    (the master sees that point: its position pass agrees within ``tol`` and it is not product / sky)."""
    P = mv["pos"]
    H, W = P.shape[:2]
    i = (np.arange(nu) + 0.5) * mpp
    j = (np.arange(nv) + 0.5) * mpp
    Q = (np.asarray(origin, np.float64)[None, None] + np.asarray(ax_u, np.float64)[None, None] * i[None, :, None]
         + np.asarray(ax_v, np.float64)[None, None] * j[:, None, None])
    u, v, z = project(mv["cam"], Q, W, H)
    inside = (z > 0) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)
    S = P[vi, ui]
    rng = np.linalg.norm(Q - _mat(mv["cam"])[:3, 3], axis=-1)
    prod = dilate(np.asarray(mv["mask"]) > 0.3, 3)
    ok = inside & (S[..., 3] > 0.5) & (np.linalg.norm(S[..., :3] - Q, axis=-1) < tol + rel * rng) & ~prod[vi, ui]
    if mv.get("sky") is not None:
        ok &= ~np.asarray(mv["sky"])[vi, ui]
    col = bilinear(np.asarray(mv["img"], np.float32), u, v)
    return col, ok


def _crop_facade(col, ok, min_px=24, need=0.75):
    """Facade patch: rows anchored at the foot (bottom) — the tallest band that still has a long run of mostly
    valid columns wins (largest area). None if nothing big enough."""
    nv = ok.shape[0]
    best = None
    for frac in (1.0, 0.8, 0.62, 0.48, 0.36):
        r0 = int(nv * (1 - frac))
        if nv - r0 < min_px:
            continue
        C = _crop_patch(col[r0:], ok[r0:], min_px=min_px, need=need, rows_fixed=True)
        if C is not None and (best is None or C.size > best.size):
            best = C
    return best


def _crop_patch(col, ok, min_px=24, need=0.75, rows_fixed=False, min_valid=0.8):
    """Largest block of rows / columns that is mostly valid; holes filled down each column. None if too small."""
    cf = ok.mean(0)
    good = cf >= need
    best, run, start = (0, 0, 0), 0, 0
    for k, g in enumerate(list(good) + [False]):
        if g:
            if run == 0:
                start = k
            run += 1
        else:
            if run > best[0]:
                best = (run, start, start + run)
            run = 0
    if best[0] < min_px:
        return None
    c0, c1 = best[1], best[2]
    if rows_fixed:
        r0, r1 = 0, ok.shape[0]
    else:
        rf = ok[:, c0:c1].mean(1)
        rows = np.nonzero(rf >= need)[0]
        if len(rows) < min_px:
            return None
        r0, r1 = rows[0], rows[-1] + 1
    C = col[r0:r1, c0:c1].astype(np.float32).copy()
    O = ok[r0:r1, c0:c1].copy()
    if O.mean() < min_valid:
        return None
    # holes: normalised convolution at growing radii (no column streaks)
    for r in (1, 2, 4, 8, 16, 32, 64):
        if O.all():
            break
        w = blur(O.astype(np.float32), r, passes=1)
        num = blur(C * O[..., None], r, passes=1)
        fill = (~O) & (w > 0.05)
        C[fill] = num[fill] / w[fill][:, None]
        O = O | fill
    if not O.all():
        C[~O] = np.median(C[O], 0) if O.any() else 128
    return C


PATCH_MAX_ANISO = 0.45       # structure-tensor anisotropy above this = a smear (rectified across a depth step)
PATCH_MAX_LOWSTD = 14.0      # grey std of the patch's low frequencies (a shadow / a kerb in it)
PATCH_MAX_DE = 12.0          # colour distance (ΔE) of an extra patch to the best one
PATCH_MIN_DETAIL = 0.5       # an extra patch keeps ≥ this share of the best one's fine detail (else a blur)


def patch_quality(C, ref=None):
    """Is ``C`` a usable texture sample (and, given the best patch ``ref``, a consistent extra one)?"""
    C = np.asarray(C, np.float32)[..., :3]
    g = C.mean(-1)
    r = max(2, min(g.shape) // 5)
    low = blur(g, r)
    hi = g - low
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1] = g[2:] - g[:-2]
    jxx, jyy, jxy = float((gx * gx).mean()), float((gy * gy).mean()), float((gx * gy).mean())
    tr = jxx + jyy
    aniso = math.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / tr if tr > 1e-6 else 0.0
    q = {"aniso": aniso, "lowstd": float(low.std()), "detail": float(hi.std()), "lab": lab(C.reshape(-1, 3).mean(0))}
    ok = aniso <= PATCH_MAX_ANISO and q["lowstd"] <= PATCH_MAX_LOWSTD
    if ok and ref is not None:
        qr = patch_quality(ref)
        ok = (float(np.linalg.norm(q["lab"] - qr["lab"])) <= PATCH_MAX_DE
              and q["detail"] >= PATCH_MIN_DETAIL * qr["detail"])
    q["ok"] = bool(ok)
    return q


def _greenish(C):
    """Foliage-like sample: green-leaning and darker than a painted wall (a pale green house is not a hedge)."""
    L, a, _b = (float(x) for x in lab(np.asarray(C, np.float32)[..., :3].reshape(-1, 3).mean(0)))
    return L < 62.0 and a < -5.0


def master_patches(mv, layout):
    """Style patches rectified from the master view: facades (≤ 4, distinct colours), road, sidewalk, greenery."""
    out = {"facade": [], "road": None, "side": None, "veg": None, "road_all": [], "side_all": [], "veg_all": []}
    if mv.get("img") is None or not layout:
        return out
    cam_o = _mat(mv["cam"])[:3, 3]
    cands = []
    for f in layout.get("facades", []):
        c = np.asarray(f["c"], np.float64)
        u = np.asarray(f["u"], np.float64)
        n = np.asarray(f["n"], np.float64)
        z0, z1 = float(f["z"][0]) + 0.2, float(f["z"][1])
        Wd = float(f["W"])
        if Wd < 2.0 or z1 - z0 < 2.0:
            continue
        # the master must have seen this facade at least about as finely as the patch texel (else it is a blur)
        dist = float(np.linalg.norm(np.r_[c, (z0 + z1) / 2] - cam_o))
        if dist * 2.0 * float(mv["cam"].get("tx", 1.0)) / mv["pos"].shape[1] > 2.0 * FILL_MPP_FACADE:
            continue
        nu = int(min(600, Wd / FILL_MPP_FACADE))
        nv = int(min(400, (z1 - z0) / FILL_MPP_FACADE))
        origin = np.r_[c - u * Wd / 2 + n * 0.35, z1]
        # the measured facade is not one plane (recesses, awnings, hedges in front): loose depth agreement — what
        # matters is that the master pixel shows that street front, not the sky or the product
        col, ok = _rectify(mv, origin, np.r_[u, 0.0], np.array([0.0, 0.0, -1.0]), nu, nv, FILL_MPP_FACADE, 1.5, rel=0.25)
        C = _crop_facade(col, ok, min_px=int(2.5 / FILL_MPP_FACADE))
        if C is not None:
            cands.append((C.shape[0] * C.shape[1], C))
    cands.sort(key=lambda x: -x[0])
    for _a, C in cands:
        mean = lab(C.reshape(-1, 3).mean(0))
        if all(float(np.linalg.norm(mean - lab(D.reshape(-1, 3).mean(0)))) > 6.0 for D in out["facade"]):
            out["facade"].append(C)
        if len(out["facade"]) >= MAX_FACADE_PATCHES:
            break
    # ground: 4 m squares around master ground pixels near the camera, away from the product
    P = mv["pos"]
    N = _unit_n(mv["nrm"])
    H, W = P.shape[:2]
    rng = np.linalg.norm(P[..., :3] - cam_o, axis=-1)
    prod = dilate(np.asarray(mv["mask"]) > 0.3, max(3, W // 60))
    g = (P[..., 3] > 0.5) & (N[..., 2] > 0.8) & (rng < 30) & ~prod
    st = layout.get("street")
    ys, xs = np.nonzero(g)
    if len(ys):
        rs = np.random.default_rng(7)
        pick = rs.choice(len(ys), min(60, len(ys)), replace=False)
        side = 2.0
        n_g = int(side / FILL_MPP_GROUND)
        cands = {"road": [], "side": []}
        for k in pick:
            q = P[ys[k], xs[k], :3]
            key = "road"
            if st:
                t = np.asarray(st["t"], np.float64)
                off = float((q[:2] - np.asarray(st["a0"])) @ np.array([t[1], -t[0]]))
                key = "road" if abs(off) <= float(st["half_width"]) - 0.5 else (
                    "side" if abs(off) >= float(st["half_width"]) + 0.5 else "")
            if not key:
                continue
            origin = q - np.array([side / 2, side / 2, 0.0])
            col, ok = _rectify(mv, origin, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), n_g, n_g, FILL_MPP_GROUND,
                               0.25)
            sc = float(ok.mean()) - 0.004 * float(rng[ys[k], xs[k]])
            if ok.mean() > 0.85:
                cands[key].append((sc, q, col, ok))
        for key in ("road", "side"):
            # 1.2.6: up to MAX_GROUND_PATCHES patches from distinct spots (best first) instead of one; the extra
            # ones must look like the best one (no smeared curb / step rectifications, shadows, blur, other colour)
            got, at = [], []
            for sc, q, col, ok in sorted(cands[key], key=lambda c: -c[0]):
                if any(float(np.linalg.norm(q[:2] - a[:2])) < side * 0.9 for a in at):
                    continue
                C = _crop_patch(col, ok, min_px=n_g // 2, need=0.7)
                # the best one is kept as before (a road with no sample would stay clay)
                if C is not None and (not got or patch_quality(C, got[0])["ok"]):
                    got.append(C)
                    at.append(q)
                if len(got) >= MAX_GROUND_PATCHES:
                    break
            out[key] = got[0] if got else None
            out[key + "_all"] = got
    for v in layout.get("veg_refs", []):
        xy = np.asarray(v["xy"], np.float64)
        r = float(v["r"])
        bot, top = float(v["z"][0]), float(v["z"][1])
        if r < 0.4 or top - bot < 0.4:
            continue
        d = cam_o[:2] - xy
        d /= max(np.linalg.norm(d), 1e-6)
        tg = np.array([d[1], -d[0]])
        origin = np.r_[xy - tg * r + d * r * 0.6, top]
        nu = max(8, int(2 * r / FILL_MPP_FACADE))
        nv = max(8, int((top - bot) / FILL_MPP_FACADE))
        col, ok = _rectify(mv, origin, np.r_[tg, 0.0], np.array([0.0, 0.0, -1.0]), nu, nv, FILL_MPP_FACADE, r + 0.8)
        C = _crop_patch(col, ok, min_px=8, need=0.6)
        if C is not None and min(C.shape[:2]) >= 12 and _greenish(C):
            out["veg_all"].append(C)
    out["veg_all"].sort(key=lambda C: -C.size)
    out["veg_all"] = out["veg_all"][:MAX_GROUND_PATCHES]
    out["veg"] = out["veg_all"][0] if out["veg_all"] else None
    return out


# ------------------------------------------------- 1.2.6 de-tiling: histogram-preserving stochastic texture fill
# 1.2.5 mirrored one small master patch per class across the whole uncovered area: a visible checkerboard on the
# road (2026-10-20 orbit 144° / 216°). Now every class is a stochastic texture: up to MAX_GROUND_PATCHES master
# patches, flattened (their light gradient removed), Gaussianised per decorrelated colour channel, and blended on a
# world-space triangle grid where each vertex takes a random patch, position, quarter turn and flip (Heitz & Neyret
# 2018, "High-performance by-example noise using a histogram-preserving blending operator"). The variance-
# preserving blend keeps the patch's contrast and colour histogram, so there is no blur and no seam; a faint low-
# frequency colour drift (world metres) replaces the flat look. Facades blend 1D strips along the wall with a random
# offset / flip per strip and a per-building tint, so two buildings of one patch differ. Flat shading, no lighting.

FILL_CELL_M = 0.55          # triangle-grid edge on the ground (m); a patch needs ≥ 2 cells + margin
FILL_VEG_CELL_M = 0.45
FILL_STRIP_M = 2.6          # facade strip length (m); blended over FILL_STRIP_BAND of it
FILL_STRIP_BAND = 0.25
FILL_LOWFREQ_M = 7.0        # low-frequency colour drift: wavelength (m) …
FILL_LOWFREQ_AMP = 0.045    # … and amplitude (fraction of the colour)
FILL_TINT = (0.06, 0.025)   # per-building brightness / per-channel tint (±)
MAX_GROUND_PATCHES = 6
FILL_LOD_R = (1, 2, 4, 8, 16)   # box radii of the prefiltered levels 1…5
_NQ = 256


def _ppf(p):
    """Inverse standard normal CDF (scalar, Newton on math.erf)."""
    x = 0.0
    for _ in range(60):
        c = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        d = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        step = (c - p) / max(d, 1e-12)
        x -= max(-2.0, min(2.0, step))
        if abs(step) < 1e-10:
            break
    return x


_PQ = (np.arange(_NQ) + 0.5) / _NQ
_GQ = np.array([_ppf(float(p)) for p in _PQ])


def _rnd(i, j, salt, k=0):
    """Deterministic uniform [0, 1) per integer lattice point (i, j), salt and stream k."""
    m = np.uint64(0xFFFFFFFF)
    x = (np.asarray(i, np.int64).astype(np.uint64) * np.uint64(0x9E3779B1)) & m
    x ^= (np.asarray(j, np.int64).astype(np.uint64) * np.uint64(0x85EBCA77)) & m
    sv = np.asarray(salt, np.int64).astype(np.uint64)
    x ^= (sv * np.uint64(0x27D4EB2F) + np.uint64((int(k) * 0x165667B1 + 0x5BD1E995) & 0xFFFFFFFF)) & m
    for mul in (0x2C1B3C6D, 0x297A2D39, 0x7FEB352D):
        x ^= x >> np.uint64(15)
        x = (x * np.uint64(mul)) & m
    x ^= x >> np.uint64(16)
    return (x & np.uint64(0xFFFFFF)).astype(np.float64) / float(1 << 24)


def value_noise(x, y, salt=0):
    """Smooth value noise in [-1, 1] at lattice spacing 1 (two octaves)."""
    out = 0.0
    for amp, f, k in ((0.7, 1.0, 0), (0.3, 2.3, 1)):
        xs, ys = np.asarray(x, np.float64) * f, np.asarray(y, np.float64) * f
        i0, j0 = np.floor(xs), np.floor(ys)
        fx, fy = xs - i0, ys - j0
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        i0, j0 = i0.astype(np.int64), j0.astype(np.int64)
        r = [[_rnd(i0 + a, j0 + b, salt, 7 + k) for a in (0, 1)] for b in (0, 1)]
        v = (r[0][0] * (1 - fx) + r[0][1] * fx) * (1 - fy) + (r[1][0] * (1 - fx) + r[1][1] * fx) * fy
        out = out + amp * (2.0 * v - 1.0)
    return out


def _flatten(p):
    """Remove a patch's low-frequency light gradient (mirror-tiled it became the checkerboard)."""
    p = np.asarray(p, np.float32)[..., :3]
    r = max(2, min(p.shape[:2]) // 5)
    return p - blur(p, r) + p.reshape(-1, 3).mean(0)


def make_synth(patches, mpp, cell_m=FILL_CELL_M, rotate=True, flatten=True):
    """Stochastic texture from master patches (list of H×W×3), or None. Texel size ``mpp`` metres."""
    ps = [np.asarray(p, np.float32)[..., :3] for p in (patches or []) if p is not None and min(np.shape(p)[:2]) >= 12]
    if not ps:
        return None
    if flatten:
        # each patch loses its light gradient and takes the pooled mean colour: patches from a sunnier and a
        # shadier spot would otherwise mottle the ground at the grid-cell scale (the slow drift is added later)
        ps = [_flatten(p) for p in ps]
        pm = np.concatenate([p.reshape(-1, 3) for p in ps]).mean(0)
        ps = [p - p.reshape(-1, 3).mean(0) + pm for p in ps]
    allpx = np.concatenate([p.reshape(-1, 3) for p in ps]).astype(np.float64)
    mu = allpx.mean(0)
    _ev, E = np.linalg.eigh(np.cov((allpx - mu).T) + np.eye(3) * 1e-3)
    Y = (allpx - mu) @ E
    qx = np.stack([np.quantile(Y[:, c], _PQ) for c in range(3)])
    qx = qx + np.arange(_NQ)[None] * 1e-6                  # strictly increasing (flat patches)
    G = []
    for p in ps:
        y = (p.reshape(-1, 3).astype(np.float64) - mu) @ E
        g = np.stack([np.interp(y[:, c], qx[c], _GQ) for c in range(3)], -1)
        G.append(g.reshape(p.shape).astype(np.float32))
    # one atlas (patches side by side, 4 px apart) × prefiltered levels (footprint 2^l texels): far pixels average
    # the texture instead of aliasing into grit; one trilinear gather serves every patch
    Ha = max(p.shape[0] for p in G)
    xo = np.cumsum([0] + [p.shape[1] + 4 for p in G])
    atlas = np.zeros((len(FILL_LOD_R) + 1, Ha, int(xo[-1]), 3), np.float32)
    for k, Gp in enumerate(G):
        lv = [Gp] + [blur(Gp, r) for r in FILL_LOD_R]
        for li, a in enumerate(lv):
            atlas[li, :Gp.shape[0], xo[k]:xo[k] + Gp.shape[1]] = a
    small = min(min(p.shape[:2]) for p in ps)
    cell = max(3.0, min(cell_m / mpp, (small - 4) / 2.0))
    return {"G": G, "atlas": atlas, "xo": xo[:-1].astype(np.float64),
            "pw": np.array([p.shape[1] for p in G], np.float64), "ph": np.array([p.shape[0] for p in G], np.float64),
            "mu": mu, "E": E, "qx": qx, "cell": cell, "rotate": bool(rotate), "mpp": float(mpp),
            "mean": mu.copy()}


def _degauss(S, g):
    y = np.stack([np.interp(g[:, c], _GQ, S["qx"][c]) for c in range(3)], -1)
    return y @ S["E"].T + S["mu"]


def _tri(S, x, y, lod):
    """Trilinear texel of the atlas: atlas pixel (x, y), fractional level ``lod`` (None = level 0)."""
    A = S["atlas"]
    L, Ha, Wa = A.shape[:3]
    x = np.clip(x, 0, Wa - 1.001)
    y = np.clip(y, 0, Ha - 1.001)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]

    def at(li):
        return ((A[li, y0, x0] * (1 - fx) + A[li, y0, x0 + 1] * fx) * (1 - fy)
                + (A[li, y0 + 1, x0] * (1 - fx) + A[li, y0 + 1, x0 + 1] * fx) * fy)
    if lod is None:
        return at(0)
    lod = np.clip(lod, 0, L - 1)
    l0 = np.minimum(np.floor(lod).astype(np.int64), L - 2)
    fl = (lod - l0)[:, None]
    return at(l0) * (1 - fl) + at(l0 + 1) * fl


def _sample_g(S, pidx, cu, cv, du, dv, rot, flip, lod=None):
    """Gaussianised texel of patch pidx at centre (cu, cv) + local (du, dv) turned by rot·90° / flipped."""
    du = np.where(flip, -du, du)
    for k in range(4):
        sel = rot == k
        if k == 1:
            du[sel], dv[sel] = -dv[sel], du[sel].copy()
        elif k == 2:
            du[sel], dv[sel] = -du[sel], -dv[sel]
        elif k == 3:
            du[sel], dv[sel] = dv[sel], -du[sel].copy()
    w, h = S["pw"][pidx], S["ph"][pidx]
    x = S["xo"][pidx] + np.clip(np.clip(cu * (w - 1), 1, w - 2) + du, 0, w - 1.001)
    y = np.clip(np.clip(cv * (h - 1), 1, h - 2) + dv, 0, h - 1.001)
    return _tri(S, x, y, lod)


def synth_ground(S, u, v, salt=0, lod=None):
    """Texture S at texel coordinates (u, v) (any shape → shape + (3,)): triangle-grid histogram-preserving blend.
    ``lod``: per point log2 of the pixel footprint in texels (prefiltered level)."""
    u = np.asarray(u, np.float64).ravel()
    v = np.asarray(v, np.float64).ravel()
    lod = None if lod is None else np.asarray(lod, np.float64).ravel()
    c = S["cell"]
    b = v / (0.8660254 * c)
    a = u / c - 0.5 * b
    a0, b0 = np.floor(a), np.floor(b)
    fa, fb = a - a0, b - b0
    upper = fa + fb > 1.0
    # three lattice vertices + barycentric weights of the equilateral triangle the point lies in
    verts = [(np.where(upper, a0 + 1, a0), np.where(upper, b0 + 1, b0)), (a0 + 1, b0), (a0, b0 + 1)]
    w = [np.where(upper, fa + fb - 1, 1 - fa - fb), np.where(upper, 1 - fb, fa), np.where(upper, 1 - fa, fb)]
    n = len(S["G"])
    acc = np.zeros(u.shape + (3,), np.float64)
    wsum = np.zeros(u.shape, np.float64)
    for (va, vb), wi in zip(verts, w):
        ia, ib = va.astype(np.int64), vb.astype(np.int64)
        vu = (va + 0.5 * vb) * c
        vv = vb * 0.8660254 * c
        pidx = np.minimum((_rnd(ia, ib, salt, 0) * n).astype(np.int64), n - 1)
        rot = (_rnd(ia, ib, salt, 1) * 4).astype(np.int64) if S["rotate"] else np.zeros(ia.shape, np.int64)
        flip = _rnd(ia, ib, salt, 2) < 0.5
        cu, cv = _margin(S, pidx, _rnd(ia, ib, salt, 3), _rnd(ia, ib, salt, 4))
        g = _sample_g(S, pidx, cu, cv, u - vu, v - vv, rot, flip, lod)
        acc += wi[:, None] * g
        wsum += wi * wi
    g = acc / np.sqrt(np.maximum(wsum, 1e-6))[:, None]
    return _degauss(S, g)


def _margin(S, pidx, ru, rv):
    """Random patch position (fractions) that keeps a full cell of margin from every edge."""
    m = S["cell"] + 1.5
    w, h = S["pw"][pidx] - 1, S["ph"][pidx] - 1
    lo_u, hi_u = np.minimum(m / w, 0.5), np.maximum(1 - m / w, 0.5)
    lo_v, hi_v = np.minimum(m / h, 0.5), np.maximum(1 - m / h, 0.5)
    return lo_u + ru * (hi_u - lo_u), lo_v + rv * (hi_v - lo_v)


def synth_strips(S, s, v, salt=0, strip_px=None, band=FILL_STRIP_BAND, lod=None):
    """Facade texture: patch rows fixed (v = patch row, mirrored above the top), columns from 1D strips along the
    wall (s texels), each with a random patch column offset and flip, blended histogram-preserving at the joints."""
    s = np.asarray(s, np.float64).ravel()
    v = np.asarray(v, np.float64).ravel()
    lod = None if lod is None else np.asarray(lod, np.float64).ravel()
    Gp = S["G"][0]
    h, w = Gp.shape[:2]
    L = float(strip_px or FILL_STRIP_M / S["mpp"])
    # a strip is at most 40 % of the patch, so its random column offset spans more than a strip (a narrow patch
    # would otherwise repeat every strip)
    L = max(4.0, min(L, (w - 2) / 2.5))
    x = s / L
    k0 = np.floor(x)
    f = x - k0
    vm = _mirror(v, h).astype(np.float64)
    acc = np.zeros(s.shape + (3,), np.float64)
    wsum = np.zeros(s.shape, np.float64)
    # strip k covers [k·L, (k+1)·L); strip k−1 runs on over the first band·L of it and fades out there
    wc = np.clip(f / band, 0, 1)
    for dk, wi in ((0, wc), (-1, 1.0 - wc)):
        k = (k0 + dk).astype(np.int64)
        loc = s - k * L
        off = _rnd(k, 0, salt, 5) * max(0.0, w - 2 - L * (1 + band))
        flip = _rnd(k, 0, salt, 6) < 0.5
        col = np.where(flip, (w - 2) - (off + loc), off + loc)
        g = _tri(S, np.clip(col, 0, w - 1.001), vm, lod)
        acc += wi[:, None] * g
        wsum += wi * wi
    g = acc / np.sqrt(np.maximum(wsum, 1e-6))[:, None]
    return _degauss(S, g)


def tile_periodicity(gray, mask=None, min_lag=8, hp=None, min_overlap=0.25):
    """How strongly a texture repeats: the highest normalised autocorrelation at lags ≥ ``min_lag`` px whose
    overlap is ≥ ``min_overlap`` of the valid area (≈1 exact tiling / mirror tiling, ≈0.1–0.3 natural texture).
    ``hp`` (px) high-passes first so slow light / colour drift does not count."""
    X = np.asarray(gray, np.float64)
    M = np.ones(X.shape) if mask is None else np.asarray(mask, np.float64)
    if M.sum() < 4 * min_lag * min_lag:
        return float("nan")
    if hp:
        X = X - blur((X * M).astype(np.float32), int(hp)) / np.maximum(blur(M.astype(np.float32), int(hp)), 1e-3)
    X = (X - (X * M).sum() / M.sum()) * M
    sh = (2 * X.shape[0], 2 * X.shape[1])
    F, FM, E = np.fft.rfft2(X, sh), np.fft.rfft2(M, sh), np.fft.rfft2(X * X, sh)
    num = np.fft.irfft2(F * np.conj(F), sh)
    e1 = np.fft.irfft2(E * np.conj(FM), sh)
    e2 = np.fft.irfft2(FM * np.conj(E), sh)
    cnt = np.fft.irfft2(FM * np.conj(FM), sh)
    ncc = num / np.sqrt(np.maximum(e1 * e2, 1e-9))
    dy = np.fft.fftfreq(sh[0]) * sh[0]
    dx = np.fft.fftfreq(sh[1]) * sh[1]
    lag = np.hypot(dy[:, None], dx[None, :])
    ok = (lag >= min_lag) & (cnt >= min_overlap * M.sum())
    return float(np.clip(ncc[ok].max(), -1, 1)) if ok.any() else float("nan")


def ground_resample(vk, img, region, layout, mpp=0.1, extent_m=30.0, max_foot=2.0):
    """Perspective-corrected view of a base's ground: a world grid (street-aligned, ``mpp`` m, ±extent around the
    camera) at the height of the region's dominant horizontal plane, each point projected into the view and sampled from ``img`` (grey) where the
    view really sees that point (position pass within 0.2 m), it lies in ``region`` and the pixel is no coarser
    than ``max_foot`` grid cells. → (grey grid, valid mask)."""
    P = np.asarray(vk["pos"], np.float64)
    H, W = P.shape[:2]
    up = (P[..., 3] > 0.5) & (_unit_n(vk["nrm"])[..., 2] > 0.8) & np.asarray(region, bool)
    gz = (layout or {}).get("ground_z")
    if up.any():                               # the region's dominant horizontal plane (usually the ground)
        gz = float(np.median(P[..., 2][up]))
    gz = float(gz or 0.0)
    st = (layout or {}).get("street")
    t = np.asarray(st["t"], np.float64) if st else np.array([1.0, 0.0])
    t = t / max(np.linalg.norm(t), 1e-9)
    nrm2 = np.array([t[1], -t[0]])
    cam_o = _mat(vk["cam"])[:3, 3]
    n = int(2 * extent_m / mpp)
    a = (np.arange(n) - n / 2 + 0.5) * mpp
    Q = cam_o[:2] + a[None, :, None] * t + a[:, None, None] * nrm2
    Q = np.dstack([Q, np.full(Q.shape[:2], float(gz))])
    u, v, z = project(vk["cam"], Q, W, H)
    inside = (z > 0) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)
    S = P[vi, ui]
    foot = pixel_footprint(P)
    ok = (inside & (S[..., 3] > 0.5) & (np.linalg.norm(S[..., :3] - Q, axis=-1) < 0.2)
          & np.asarray(region, bool)[vi, ui] & (foot[vi, ui] <= max_foot * mpp))
    g = bilinear(np.asarray(img, np.float32)[..., :3].mean(-1, keepdims=True), u, v)[..., 0]
    return np.where(ok, g, 0.0), ok


def fill_periodicity(vk, base, region, layout, mpp=0.1):
    """Repetition of a base's ground region (H×W bool): ``tile_periodicity`` of the perspective-corrected ground
    (``ground_resample``), high-passed at 2 m, lags ≥ 1 m. NaN when the region is too small to tell."""
    g, ok = ground_resample(vk, base, region, layout, mpp)
    if ok.sum() < 400:
        return float("nan")
    return tile_periodicity(g, ok, min_lag=int(round(1.0 / mpp)), hp=int(round(2.0 / mpp)), min_overlap=0.2)


def pixel_footprint(P, cap=2.0):
    """World size (m) of each pixel: the smaller of its right / down neighbour steps on the same surface (a depth
    jump on one side does not count), capped at ``cap``."""
    P = np.asarray(P, np.float64)
    hit = P[..., 3] > 0.5
    big = np.full(hit.shape, cap)
    dx, dy = big.copy(), big.copy()
    sx = np.linalg.norm(P[:, 1:, :3] - P[:, :-1, :3], axis=-1)
    sy = np.linalg.norm(P[1:, :, :3] - P[:-1, :, :3], axis=-1)
    okx = hit[:, 1:] & hit[:, :-1]
    oky = hit[1:] & hit[:-1]
    dx[:, :-1] = np.where(okx, sx, cap)
    dx[:, -1] = dx[:, -2]
    dy[:-1] = np.where(oky, sy, cap)
    dy[-1] = dy[-2]
    # a pixel's own footprint is the larger axis, but an edge (one side jumps) takes the other side's step
    lx = np.minimum(dx, np.roll(dx, 1, 1))
    ly = np.minimum(dy, np.roll(dy, 1, 0))
    return np.minimum(np.maximum(lx, ly), cap)


def fill_synths(patches):
    """The stochastic textures of a master patch set (built once, cached in the dict)."""
    if patches.get("_synth") is not None:
        return patches["_synth"]

    def lst(key):
        got = [p for p in (patches.get(key + "_all") or []) if p is not None]
        return got or ([patches[key]] if patches.get(key) is not None else [])
    road, side, veg = lst("road"), lst("side"), lst("veg")
    road, side = road or side, side or road
    if not veg:                 # no greenery sample: a dark green with leaf-sized noise (deterministic)
        rs = np.random.default_rng(5)
        n = blur(rs.normal(0, 1, (48, 48)).astype(np.float32), 1, passes=1)
        veg = [np.clip(np.array((58, 78, 46), np.float32) * (1 + 0.35 * n / max(float(n.std()), 1e-6))[..., None], 0, 255)]
    S = {"road": make_synth(road, FILL_MPP_GROUND), "side": make_synth(side, FILL_MPP_GROUND),
         "curb": make_synth([np.clip(np.asarray(p, np.float32) * 1.12 + 6, 0, 255) for p in side], FILL_MPP_GROUND),
         "veg": make_synth(veg, FILL_MPP_FACADE, cell_m=FILL_VEG_CELL_M, rotate=False),
         # facades keep their rows (shop front at the foot, windows above): no flattening, no turning
         "facade": [make_synth([p], FILL_MPP_FACADE, rotate=False, flatten=False) for p in (patches.get("facade") or [])]}
    patches["_synth"] = S
    return S


def style_fill(vk, cls, style, patches, layout):
    """(fill image H×W×3, fill mask) for view k from the master patches: stochastic, world-space textures."""
    P = np.asarray(vk["pos"], np.float64)
    H, W = P.shape[:2]
    N = _unit_n(vk["nrm"])
    out = np.zeros((H, W, 3), np.float32)
    have = np.zeros((H, W), bool)
    gz = (layout or {}).get("ground_z")
    gz = float(gz) if gz is not None else float(np.percentile(P[..., 2][(P[..., 3] > 0.5) & (N[..., 2] > 0.8)], 30)) \
        if ((P[..., 3] > 0.5) & (N[..., 2] > 0.8)).any() else 0.0
    st = (layout or {}).get("street")
    if st:
        t = np.asarray(st["t"], np.float64)
        a0 = np.asarray(st["a0"], np.float64)
        gs = (P[..., :2] - a0) @ t
        go = (P[..., :2] - a0) @ np.array([t[1], -t[0]])
    else:
        gs, go = P[..., 0], P[..., 1]
    S = fill_synths(patches)
    foot = pixel_footprint(P)
    lod_g = np.log2(np.maximum(foot / FILL_MPP_GROUND, 1.0))
    lod_f = np.log2(np.maximum(foot / FILL_MPP_FACADE, 1.0))
    drift = 1.0 + FILL_LOWFREQ_AMP * value_noise(gs / FILL_LOWFREQ_M, go / FILL_LOWFREQ_M, 11)

    def put(sel, col, mod=None):
        if col is None or not sel.any():
            return
        c = col if mod is None else col * mod[sel][:, None]
        out[sel] = np.clip(c, 0, 255)
        have[sel] = True

    g = FILL_MPP_GROUND
    for cl, key, salt in ((CLS_ROAD, "road", 1), (CLS_SIDE, "side", 2)):
        sel = cls == cl
        if sel.any() and S[key] is not None:
            put(sel, synth_ground(S[key], gs[sel] / g, go[sel] / g, salt, lod_g[sel]), drift)
    sel = cls == CLS_CURB
    if sel.any() and S["curb"] is not None:
        put(sel, synth_ground(S["curb"], gs[sel] / g, (P[..., 2] - gz)[sel] / g, 3, lod_g[sel]), drift)
    tan = facade_tangent(P, N)
    s = (P[..., :2] * tan).sum(-1)
    fac = [f for f in S["facade"] if f is not None]
    zbase = np.full((H, W), gz)
    for b in (layout or {}).get("boxes", []):
        if b.get("kind") == "building":
            inside = (cls == CLS_FACADE) & in_box(P, b)
            zbase[inside] = float(b["z"][0]) + 0.5
    zr = P[..., 2] - zbase
    for k, Sk in enumerate(fac):
        sel = (cls == CLS_FACADE) & (style % len(fac) == k)
        if not sel.any():
            continue
        sty = style[sel]
        h = Sk["G"][0].shape[0]
        # bottom row of the patch = the facade's foot; above the patch height the upper storeys repeat. Each
        # building (style id) gets its own strip offsets / flips and a small tint, so neighbours differ
        col = synth_strips(Sk, s[sel] / FILL_MPP_FACADE, (h - 1) - zr[sel] / FILL_MPP_FACADE, salt=sty, lod=lod_f[sel])
        tint = (1.0 + FILL_TINT[0] * (2 * _rnd(sty, 0, 0, 8) - 1))[:, None] \
            + FILL_TINT[1] * (2 * np.stack([_rnd(sty, 0, 0, 9 + c) for c in range(3)], -1) - 1)
        vn = 1.0 + 0.6 * FILL_LOWFREQ_AMP * value_noise(s[sel] / FILL_LOWFREQ_M, P[..., 2][sel] / FILL_LOWFREQ_M, 12)
        put(sel, col * tint * vn[:, None])
    sel = cls == CLS_VEG
    if sel.any() and S["veg"] is not None:
        put(sel, synth_ground(S["veg"], s[sel] / FILL_MPP_FACADE, -P[..., 2][sel] / FILL_MPP_FACADE, 4, lod_f[sel]),
            drift)
    pole = cls == CLS_POLE
    out[pole] = (96, 96, 98)
    have |= pole
    have &= np.asarray(vk["mask"]) < 0.5
    return out, have


def apply_fill(env, cover, fill, have, alpha=FILL_ALPHA):
    """Uncovered pixels take the style fill: returns (env', cover', filled share mask)."""
    need = np.clip(1.0 - cover, 0.0, 1.0) * have.astype(np.float32) * float(alpha)
    env2 = env * (1.0 - need[..., None]) + fill * need[..., None]
    cover2 = np.clip(cover + need, 0.0, 1.0)
    return env2, cover2, need > 0.25


def fill_note(cls, filled, stats_text="") -> str:
    """Prompt note for a view whose base carries style fill (empty when it is negligible)."""
    n = max(1, int(filled.size))
    share = float(filled.sum()) / n
    if share < 0.02:
        return ""
    kinds = [CLASS_WORD[k] for k in (CLS_FACADE, CLS_ROAD, CLS_SIDE, CLS_VEG, CLS_CURB, CLS_POLE)
             if float((filled & (cls == k)).sum()) / n > 0.004]
    return ("底图里主视角没拍到的部分（约 {:.0%}）已按看到的街道推测补全，并铺了从主视角取样的同类材质示意纹理（{}）："
            "请把这些部分画成主视角同一条街的自然延续{}——同样类型、相近颜色和层数的房屋（有窗户、门、屋顶 / 雨棚），"
            "同样的路面、路沿、树篱和电杆；示意纹理只表示材质和颜色范围，可以重画细节，"
            "不要画成空白墙面、大面积纯色墙或空地。").format(share, "、".join(kinds) or "环境",
                                                        f"（{stats_text}）" if stats_text else "")


def compose_canvas(env_work, cover_work, clay_full, mask_full):
    """Full-size edit canvas: projected environment (upsampled) over the clay base; the product
    and every uncovered pixel stay clay."""
    Hf, Wf = clay_full.shape[:2]
    env = resize(env_work, Wf, Hf)
    cov = np.clip(resize(cover_work, Wf, Hf), 0, 1)[..., None]
    cov = cov * (1.0 - np.clip(np.asarray(mask_full, np.float32), 0, 1)[..., None])
    clay = np.asarray(clay_full, np.float32)[..., :3]
    return np.clip(env * cov + clay * (1 - cov), 0, 255)


def reproj_de(master, vk, img_k):
    """Mean ΔE76 vs the master at corresponding pixels (both blurred). ``geom`` = buildings /
    ground / room pixels only (sky excluded); ``covis`` = share of view k's environment the
    master also sees."""
    H, W = vk["pos"].shape[:2]
    B1 = lab(blur(np.asarray(master["img"], np.float32)[..., :3], 2))
    Bk = lab(blur(np.asarray(img_k, np.float32)[..., :3], 2))
    U, V, vis, sky, hit, sky_in = correspond(vk, master)
    env = np.asarray(vk["mask"]) < 0.5
    ref = bilinear(B1, U, V)
    if master["sky"].sum() > 30:
        sf = lab(blur(skyfill(np.asarray(master["img"], np.float32), master["sky"]), 2))
        ref = np.where(sky[..., None], bilinear(sf, U, V), ref)
    de = np.linalg.norm(ref - Bk, axis=-1)
    co = (vis | sky) & env
    g = vis & env
    n_env = max(1, int(env.sum()))
    return {"reproj_dE": round(float(de[co].mean()), 1) if co.sum() > 0.01 * n_env else None,
            "reproj_dE_geom": round(float(de[g].mean()), 1) if g.sum() > 0.01 * n_env else None,
            "covis": round(float(co.sum()) / n_env, 3)}


# --------------------------------------------------------------------------- 1.2.4 统一调色

# --------------------------------------------------------------------------- 1.2.6 风格与光线锁定
# Real cyber orbit test 2026-10-20: the new views lost the neon signs and the night lighting and turned brown / grey
# (emissive signs live only in master pixels, low projection coverage; the colour match pulled toward the master mean;
# the prompt had no style lock; the low-tier retry fired on dark walls). The 主视角 look is measured here and locked.

EMISSIVE_V = 0.8           # brightest channel ≥ this share with saturation ≥ EMISSIVE_S (saturated neon: pure red is only
EMISSIVE_S = 0.45          #     L 53), or Lab L ≥ EMISSIVE_L_WHITE (white / warm-white lightboxes) = a light source / lit sign
EMISSIVE_L_WHITE = 85.0
NIGHT_MEDIAN_L = 34.0      # environment median L below this (and a dark or absent sky) = a night / dark scene
NIGHT_SKY_L = 30.0
HUE_NAMES = ((15, "红"), (40, "橙"), (70, "黄"), (160, "绿"), (200, "青"), (260, "蓝"), (300, "紫"), (345, "粉红"),
             (361, "红"))
SIGN_WORDS = (("neon", "霓虹灯"), ("lightbox", "灯箱"), ("light box", "灯箱"), ("lantern", "灯笼"),
              ("billboard", "广告牌"), ("advertis", "广告牌"), ("sign", "招牌"), ("shop window", "橱窗"),
              ("storefront", "店面灯光"), ("street lamp", "路灯"), ("streetlight", "路灯"), ("street light", "路灯"),
              ("霓虹", "霓虹灯"), ("灯箱", "灯箱"), ("灯笼", "灯笼"), ("招牌", "招牌"), ("广告", "广告牌"), ("路灯", "路灯"))
WET_WORDS = ("wet", "puddle", "rain", "湿", "积水", "雨")


def emissive_mask(img):
    """Pixels that are light sources / lit signs (bright and saturated, or near white) of an RGB 0..255 image."""
    rgb = np.asarray(img, np.float32)[..., :3]
    mx = rgb.max(-1)
    sat = (mx - rgb.min(-1)) / np.maximum(mx, 1e-3)
    return ((mx >= EMISSIVE_V * 255.0) & (sat >= EMISSIVE_S)) | (lab(rgb)[..., 0] >= EMISSIVE_L_WHITE)


def _hue_deg(rgb):
    rgb = np.asarray(rgb, np.float32)[..., :3] / 255.0
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = np.maximum(mx - mn, 1e-6)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
    return h, (mx - mn) / np.maximum(mx, 1e-6)


def master_style(img, sky=None, mask=None) -> dict:
    """Measured look of the 主视角 environment: night (dark median + dark / no sky), emissive share and
    the dominant colours of its light sources. ``img`` RGB 0..255, ``sky`` / ``mask`` (product) optional."""
    img = np.asarray(img, np.float32)[..., :3]
    env = np.ones(img.shape[:2], bool) if mask is None else np.asarray(mask) < 0.5
    L = lab(img)[..., 0]
    med = float(np.median(L[env])) if env.any() else 50.0
    sk = None if sky is None else (np.asarray(sky, bool) & env)
    sky_l = float(L[sk].mean()) if sk is not None and sk.sum() > 50 else None
    night = med < NIGHT_MEDIAN_L and (sky_l is None or sky_l < NIGHT_SKY_L)
    em = emissive_mask(img) & env
    if sk is not None:
        em &= ~sk
    frac = float(em.sum()) / max(1, int(env.sum()))
    colours = []
    if em.sum() >= 30:
        h, sat = _hue_deg(img[em])
        col = sat > 0.35
        if col.sum() >= 20:
            names = []
            for hv in h[col]:
                for lim, nm in HUE_NAMES:
                    if hv < lim:
                        names.append(nm)
                        break
            uniq, cnt = np.unique(np.array(names), return_counts=True)
            order = np.argsort(-cnt)
            colours = [str(uniq[i]) for i in order if cnt[i] >= 0.08 * len(names)][:4]
        if (~col).sum() >= 0.2 * len(col):
            colours.append("暖白" if float(np.median(img[em][~col][:, 0] - img[em][~col][:, 2])) > 8 else "白")
    return {"night": bool(night), "median_L": round(med, 1), "sky_L": None if sky_l is None else round(sky_l, 1),
            "emissive": round(frac, 4), "colours": colours}


def style_lock_text(style, analysis=None, light=None, brief="") -> str:
    """风格与光线锁定 paragraph for every new view (must keep the 主视角's look). Generic: night scenes get the
    light-source / sign lock, day scenes only the time-of-day / colour lock. ``analysis``: the ④ analysis JSON
    (optional), ``light``: the master light estimate (optional)."""
    style = style or {}
    an = analysis or {}
    pal = an.get("palette") or {}
    texts = " ".join(str(x or "") for x in (an.get("environment_description"), an.get("scene_summary"),
                                             an.get("environment_zh"), pal.get("lock_zh"), pal.get("ground_words"),
                                             (light or {}).get("sky_words"))).lower()
    night = bool(style.get("night")) or (light_kind(light or {}, True) == "night_artificial"
                                        and float(style.get("median_L", 50) or 50) < 45)
    parts = []
    from .scene_style import analysis_text, art_style, style_phrase
    art = art_style(analysis_text(an), brief)
    lock = str(pal.get("lock_zh") or an.get("environment_zh") or "").strip()
    if night:
        signs = []
        for en, zh in SIGN_WORDS:
            if en in texts and zh not in signs:
                signs.append(zh)
        cols = "、".join(style.get("colours") or [])
        src = "、".join(signs[:5]) or "招牌、灯箱、橱窗和路灯"
        parts.append(
            "风格与光线锁定（必须保持，与主视角同一场景同一时刻）：这是夜景，天空保持夜色，画面以人工光为主。"
            f"主视角里的发光{src}" + (f"（主要颜色：{cols}）" if cols else "")
            + "在这个角度同样要有：两侧建筑立面、店面上继续布满同类型、同样密度、真的在发光的招牌和灯光，"
            "保留它们周围的局部亮光、光晕和强烈的明暗对比；暗处保持深色但看得清结构。"
            "不要画成白天、黄昏或阴天，不要把招牌画成不发光的普通墙面或空白墙，不要整体提亮成灰色，也不要压成单一的棕灰色调。")
    else:
        parts.append("风格与光线锁定（必须保持，与主视角同一场景同一时刻）：时段、天气、色温、对比度和整体色调与主视角一致，"
                     "不要改变时段或天气，不要整体偏色。")
    if art:      # 1.2.6: a voxel / cartoon reference keeps its 画风 in every new view (never drawn photoreal)
        parts.append(f"画风：整个环境是{style_phrase(art)}风格，与主视角一致（造型语言、材质和纹理表现同主视角），"
                     "不要画成写实照片或普通写实建筑。")
    if lock:
        parts.append(f"场景基调：{lock}")
    if any(w in texts for w in WET_WORDS):
        parts.append("路面是湿的：保留灯光和招牌在路面上的倒影，倒影颜色与上方光源一致。")
    elif night:
        parts.append("路面干湿与主视角一致（主视角不是湿路就不要画成湿路或积水），只保留少量灯光反光。")
    return "".join(parts)


GRADE_STRENGTH = 0.75      # share of the fitted transform that is applied (research: 0.75)
GRADE_MIN_PIXELS = 400     # co-visible environment pixels needed to fit at all
GRADE_MAX_SHIFT = 12.0     # ΔE: a fit that would move the environment mean further is not trusted


GRADE_NIGHT_STRENGTH = 0.4   # 1.2.6: night scenes — a weaker pull toward the master (local light differs per view)
GRADE_MIN_GAIN = 0.95        # night: the L / a / b gains never flatten contrast or saturation below this


def fit_grade(master, vk, img_k, strength=GRADE_STRENGTH, night=False):
    """Colour transform of view k onto the 主视角, fitted on CO-VISIBLE ENVIRONMENT pixels only.

    Not a whole-image statistics match (that made v4 worse in the research: each view sees different
    content): ``correspond`` says which pixel of view k shows the same world point (geometry, depth
    tested) or the same sky direction as a master pixel; both images are blurred, converted to Lab,
    and a 4×3 affine Lab transform is least-squares fitted, refitted without the worst 20 %, and
    blended with the identity (``strength``). ``img_k``: view k at working size (H, W, 3, 0..255).
    Returns {"A": 4×3 list, "pixels", "dE_before", "dE_after", "shift"} or None (too little overlap,
    no improvement or an implausible shift).

    ``night`` (1.2.6, 风格与光线锁定): light sources / lit signs (``emissive_mask``) of either image are left out of
    the fit, the strength drops to GRADE_NIGHT_STRENGTH, the L / a / b gains stay ≥ GRADE_MIN_GAIN (no contrast /
    saturation flattening toward the master mean) and ``apply_grade`` keeps local contrast and emissive pixels."""
    H, W = vk["pos"].shape[:2]
    B1 = lab(blur(np.asarray(master["img"], np.float32)[..., :3], 2))
    Bk = lab(blur(np.asarray(img_k, np.float32)[..., :3], 2))
    U, V, vis, sky, hit, sky_in = correspond(vk, master)
    ui = np.clip(np.round(U).astype(int), 0, W - 1)
    vi = np.clip(np.round(V).astype(int), 0, H - 1)
    msky = master.get("sky")
    sky_ok = sky_in & (msky[vi, ui] if msky is not None else False)
    env = ~dilate(np.asarray(vk["mask"]) > 0.5, 3)
    cov = (vis | sky_ok) & env
    ref = bilinear(B1, U, V)
    if night:
        em_m = emissive_mask(np.asarray(master["img"], np.float32)[..., :3])
        em_k = emissive_mask(img_k)
        em_r = bilinear(em_m.astype(np.float32)[..., None], U, V)[..., 0] > 0.2
        cov &= ~dilate(em_k, 2) & ~em_r
        strength = min(strength, GRADE_NIGHT_STRENGTH)
    n = int(cov.sum())
    if n < GRADE_MIN_PIXELS:
        return None
    X = Bk[cov].astype(np.float64)
    Y = ref[cov].astype(np.float64)
    if n > 60000:      # deterministic subsample
        sel = np.linspace(0, n - 1, 60000).astype(int)
        X, Y = X[sel], Y[sel]
    Xa = np.c_[X, np.ones(len(X))]
    try:
        A = np.linalg.lstsq(Xa, Y, rcond=None)[0]
        e = np.linalg.norm(Xa @ A - Y, axis=1)
        keep = e < np.percentile(e, 80)
        A = np.linalg.lstsq(Xa[keep], Y[keep], rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    ident = np.vstack([np.eye(3), np.zeros((1, 3))])
    A = strength * A + (1 - strength) * ident
    if night:
        mu = X.mean(0)
        for c in range(3):
            if A[c, c] < GRADE_MIN_GAIN:      # keep the fitted mean, refuse the flattening
                A[3, c] += (A[c, c] - GRADE_MIN_GAIN) * mu[c]
                A[c, c] = GRADE_MIN_GAIN
    before = float(np.linalg.norm(X - Y, axis=1).mean())
    after = float(np.linalg.norm(Xa @ A - Y, axis=1).mean())
    shift = float(np.linalg.norm((Xa @ A).mean(0) - X.mean(0)))
    if not np.all(np.isfinite(A)) or after >= before or shift > GRADE_MAX_SHIFT:
        return None
    out = {"A": A.tolist(), "pixels": n, "dE_before": round(before, 2), "dE_after": round(after, 2),
           "shift": round(shift, 2)}
    if night:
        out["night"] = True
    return out


def apply_grade(img, grade, mask):
    """Apply a ``fit_grade`` transform to the ENVIRONMENT of a full-size image (H, W, 3, 0..255).
    ``mask``: product mask 0..1 at the same size; the product (grown 2 px, feathered) is untouched."""
    from .material_color import lab_to_srgb
    img = np.asarray(img, np.float32)[..., :3]
    if not grade:
        return img.copy()
    A = np.asarray(grade["A"], np.float32)
    L = lab(img)
    if grade.get("night"):
        # 1.2.6: grade the low frequencies only and add the local detail back (lit edges, glows and highlights keep
        # their contrast); light sources / lit signs are left as they are
        r = max(2, min(img.shape[:2]) // 80)
        Lb = blur(L, r)
        Lt = (Lb.reshape(-1, 3) @ A[:3] + A[3]).reshape(L.shape) + (L - Lb)
        out = lab_to_srgb(Lt) * 255.0
    else:
        Lt = L.reshape(-1, 3) @ A[:3] + A[3]
        out = lab_to_srgb(Lt.reshape(L.shape)) * 255.0
    m = np.asarray(mask, np.float32)
    if m.shape[:2] != img.shape[:2]:
        m = resize(m, img.shape[1], img.shape[0])
    prot = np.where(m > 0.5, 1.0, np.clip(np.maximum(blur(dilate(m > 0.5, 2).astype(np.float32), 1), m), 0, 1))
    if grade.get("night"):
        em = dilate(emissive_mask(img), 1)
        prot = np.maximum(prot, np.maximum(em.astype(np.float32), blur(dilate(em, 2).astype(np.float32), 2)))
    prot = prot[..., None]
    return np.clip(out * (1 - prot) + img * prot, 0, 255)


def grade_note(results) -> str:
    graded = [r for r in results if (r.get("grade") or {}).get("applied")]
    if not graded:
        return ""
    return "统一调色 " + "、".join(f"{r['name']}（ΔE {r['grade']['dE_before']:.1f}→{r['grade']['dE_after']:.1f}）"
                               for r in graded)


# --------------------------------------------------------------------------- policy

def product_violations(rep, iou_verdict=None) -> list:
    """Out-of-spec readings of the raw AI image (产品还原度). The edge reading is the one against
    the detail pass (``edge_px_detail``), the reading the thresholds were calibrated on.

    ``iou_verdict``: the fixed-coordinate silhouette acceptance (轮廓验收) of the same image. When
    it passes (IoU ≥ 0.95 without any shift or scale) the product did not move, so the fitted
    offset / scale readings are not trusted: on fine-textured fabric (velvet) the inner-structure
    fit can lock onto texture and read 10 px / −7.5 % on a product that sits on its outline."""
    rep = rep or {}
    if rep.get("error"):
        return []
    out = []
    placement_ok = iou_verdict == "通过"
    edge = rep.get("edge_px_detail", rep.get("edge_px"))
    if edge is not None and edge > EDGE_PX:
        out.append(f"细节边缘 {edge:.1f}px")
    if (rep.get("colour_dE") or 0) > COLOUR_DE:
        out.append(f"颜色 ΔE {rep['colour_dE']:.0f}")
    if abs(rep.get("shift_px") or 0) > OFFSET_PX and not placement_ok:
        out.append(f"偏移 {rep['shift_px']:.1f}px")
    if abs(rep.get("scale_pct") or 0) > SCALE_PCT and not placement_ok:
        out.append(f"缩放 {rep['scale_pct']:+.1f}%")
    return out


def env_violations(rp, enabled=True) -> list:
    if not enabled or not rp:
        return []
    g, c = rp.get("reproj_dE_geom"), rp.get("covis") or 0.0
    if g is not None and c >= ENV_MIN_COVIS and g > ENV_GEOM_DE:
        return [f"环境与主视角色差 {g:.1f}"]
    return []


def choose(tries):
    """Fewest product violations, then fewest environment violations, then (1.2.4) not the ``low``
    quality tier, then the lower edge."""
    def key(t):
        rep = t.get("identity") or {}
        edge = rep.get("edge_px_detail", rep.get("edge_px"))
        low = str(t.get("image_quality") or "").strip().lower() == "low"
        return (len(t.get("product_violations") or []), len(t.get("env_violations") or []), int(low),
                99.0 if edge is None else float(edge))
    return min(tries, key=key)


# --------------------------------------------------------------------------- text

LIGHT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["light_type", "key_rel_azimuth_deg", "key_elevation_deg", "sky_words"],
    "properties": {
        "light_type": {"type": "string", "enum": ["direct_sun", "hazy_sun", "overcast", "night_artificial",
                                                          "interior_window", "interior_mixed"]},
        "key_rel_azimuth_deg": {"type": "number", "description":
                                "horizontal direction FROM the scene TOWARD the sun / brightest sky / main window, relative to "
                                "the camera: 0 = straight ahead (backlight), +90 = camera right, -90 = camera left, 180 = behind"},
        "key_elevation_deg": {"type": "number"},
        "sky_words": {"type": "string", "description": "short Chinese description of the light: time of day, colour "
                                                       "temperature, cloud, haze, shadow softness"}}}

LIGHT_PROMPT = ("This is a rendered photo of a scene. Estimate the main light: the sun (or the brightest sky region if "
                "overcast, or the main window if it is an interior). Use cast shadows, highlights, shading and the sky "
                "gradient. An OUTDOOR scene at night lit by signs, shop windows or street lamps is night_artificial (its main "
                "light is the brightest artificial source); interior_window / interior_mixed are only for indoor rooms. "
                "Give the horizontal direction from the scene toward the light relative to the camera (0 ahead / "
                "backlight, +90 right, -90 left, 180 behind the camera), its elevation, and a short Chinese description.")


def world_light(light: dict, master_cam) -> dict:
    """Make the master-relative estimate a world direction (fixed for every other camera)."""
    out = dict(light or {})
    out["world_azimuth_deg"] = round(((float(out.get("key_rel_azimuth_deg", 0.0)) + yaw_deg(master_cam)) + 180.0) % 360.0
                                     - 180.0, 1)
    return out


NIGHT_WORDS = ("夜", "night", "neon", "霓虹", "dark sky", "近黑")


def light_kind(light: dict, outdoor=None) -> str:
    """The light type to word the lock with. 1.2.6: when the ④ analysis says the scene is outdoor, an
    ``interior_*`` estimate (the light model read a lit night street as 'interior_mixed') is worded as a
    night street (night words in the estimate) or soft daylight — never 「同一个房间」 / 「窗户（主光）」."""
    kind = str((light or {}).get("light_type") or "")
    if outdoor and kind.startswith("interior"):
        words = str((light or {}).get("sky_words") or "").lower()
        return "night_artificial" if any(w in words for w in NIGHT_WORDS) else "overcast"
    return kind


def light_sentence(light: dict, cam, outdoor=None) -> str:
    """Where the world-locked light sits for this camera, in the words the prototype used.
    ``outdoor``: the ④ analysis scene kind is outdoor (None = unknown)."""
    if not light or "world_azimuth_deg" not in light:
        return ""
    rel = (float(light["world_azimuth_deg"]) - yaw_deg(cam) + 180.0) % 360.0 - 180.0
    a = abs(rel)
    lr = "右" if rel > 0 else "左"
    other = "左" if lr == "右" else "右"
    words = str(light.get("sky_words") or "").strip()
    kind = light_kind(light, outdoor)
    if kind == "night_artificial":
        if a <= 35:
            pos = f"最亮的人工光源在画面前方{'偏' + lr if a > 10 else ''}：主体朝镜头一面偏暗，轮廓有彩色亮边"
        elif a <= 120:
            pos = f"最亮的人工光源在画面{lr}侧：受光面朝画面{lr}侧，{other}侧更暗"
        else:
            pos = f"最亮的人工光源在镜头身后{'偏' + lr if a < 165 else ''}：朝镜头一面受光"
        return (f"光线锁定（与主视角是同一个夜晚同一时刻，光源在世界里固定）：夜景，人工照明为主（招牌、灯箱、橱窗、路灯），{pos}。"
                "天空保持夜色，光源颜色、亮暗对比、局部光斑和光晕与主视角一致" + (f"（{words}）" if words else "")
                + "，不要改成白天、黄昏或阴天，不要去掉或熄灭灯光。")
    if kind.startswith("interior"):
        if a <= 35:
            pos = f"窗户（主光）在画面前方{'偏' + lr if a > 10 else ''}：逆光，产品和家具朝镜头一面略暗，亮边朝窗"
        elif a <= 120:
            pos = f"窗户（主光）在画面{lr}侧{'略偏前' if a < 80 else '略偏后'}：侧光，受光面朝画面{lr}侧，阴影投向画面{other}侧"
        else:
            pos = f"窗户（主光）在镜头身后{'偏' + lr if a < 165 else ''}，画面里看不到窗：顺光，朝镜头一面受光，阴影朝画面深处"
        return (f"光线锁定（与主视角是同一个房间同一时刻）：{pos}。窗外光的亮度、色温、室内补光和阴影软硬与主视角一致"
                + (f"（{words}）" if words else "") + "，不要改变时段、不要新增灯具光斑。")
    if kind == "overcast":
        if a <= 60:
            pos = f"最亮的云隙 / 天空在画面前方{'偏' + lr if a > 12 else ''}"
        elif a <= 120:
            pos = f"最亮的天空在画面{lr}侧"
        else:
            pos = "最亮的天空在镜头身后（画面里看不到），画面里的天空是较暗的云层"
        return (f"光线锁定（与主视角是同一时刻同一天气，光源方向在世界里固定）：阴天柔和散射光，{pos}；没有明显的硬阴影。"
                "天空颜色、云量、湿度和色温与主视角一致" + (f"（{words}）" if words else "") + "，不要改成晴天、日落或别的时段。")
    el = float(light.get("key_elevation_deg", 30.0) or 30.0)
    if a <= 35:
        pos = f"逆光：太阳在画面前方{'偏' + lr if a > 10 else ''}，主体背光、轮廓有亮边，阴影朝镜头方向（画面下方）投下"
    elif a <= 110:
        pos = (f"侧光：阳光从画面{lr}侧{'略偏前' if a < 75 else ('略偏后' if a > 95 else '')}照来，物体朝画面{lr}侧的面受光，"
               f"阴影投向画面{other}侧{'并朝向镜头' if a < 75 else ('并朝画面深处' if a > 95 else '')}")
    else:
        pos = f"顺光：太阳在镜头身后{'偏' + lr if a < 165 else ''}，物体朝镜头一面受光，阴影朝画面深处投下"
    return (f"光线锁定（与主视角是同一时刻，太阳在世界里固定不动）：{pos}，太阳高度约 {el:.0f}°。天空颜色、云量、雾霾、色温和"
            "阴影软硬与主视角一致" + (f"（{words}）" if words else "") + "，不要改变时段或天气。")


def view_text(massing_text, light_line, source_names, fill_line="", style_line="") -> str:
    """Per-view addition to the simple-path prompt (goes in as ``massing_text``). ``style_line``: the
    1.2.6 风格与光线锁定 from ``style_lock_text`` (must-keep look of the 主视角)."""
    parts = [t.strip() for t in (massing_text or "",) if t and t.strip()]
    if light_line:
        parts.append(light_line)
    if style_line and style_line.strip():
        parts.append(style_line.strip())
    names = [n for n in source_names or () if n]
    parts.append(BASE_NOTE.format(sources="主视角" + ("和" + "、".join(names[1:]) if len(names) > 1 else "")))
    if fill_line:
        parts.append(fill_line)
    return "\n".join(parts)


def estimate_light(image_path, *, client_version="", model="", cancel_check=None, timeout=240) -> dict:
    """One strict-JSON analysis call on the master AI image (never the reference photo)."""
    from . import codex_direct as CD
    auth_path = CD.find_auth_file()
    if auth_path is None:
        raise CD.DirectEditError("未找到 Codex 登录文件", kind="auth_local", fatal=True)
    auth = CD.load_auth(auth_path)
    if CD.token_expiring(CD.access_token(auth)):
        auth = CD.refresh_auth(auth, auth_path)
    url_img, _mime, _n = CD.data_url(image_path)
    import os
    base = os.environ.get(CD.BASE_URL_ENV, "").strip() or CD.DEFAULT_BASE_URL
    payload = {"model": (model or "").strip() or CD.DEFAULT_MODEL,
               "instructions": "Estimate lighting as strict JSON. Do not use any tool.",
               "input": [{"type": "message", "role": "user", "content": [
                   {"type": "input_text", "text": LIGHT_PROMPT}, {"type": "input_image", "image_url": url_img}]}],
               "tools": [], "parallel_tool_calls": False, "store": False, "stream": True, "include": [],
               "text": {"format": {"type": "json_schema", "name": "light", "strict": True, "schema": LIGHT_SCHEMA}}}
    refreshed = False
    while True:
        try:
            text = CD.post_responses(base.rstrip("/") + "/responses", CD.auth_headers(auth, client_version), payload,
                                     timeout=timeout, cancel_check=cancel_check, want="text")
            break
        except CD.DirectEditError as exc:
            if exc.kind == "auth" and exc.status == 401 and not refreshed:
                refreshed = True
                auth = CD.refresh_auth(auth, auth_path)
                continue
            raise
    m = re.search(r"\{.*\}", text or "", re.S)
    data = json.loads(m.group(0) if m else text)
    return {k: data.get(k) for k in LIGHT_SCHEMA["required"]}


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name) -> str:
    return re.sub(r"[^\w\u3400-\u9fff.-]+", "_", str(name or "camera")).strip("_")[:60] or "camera"


# --------------------------------------------------------------------------- texture pool

def load_view(pool, entry, with_image=True):
    """Pool entry → view dict (numpy arrays at working size)."""
    from .png_io import read_png
    d = Path(pool)
    v = {"name": entry["name"], "cam": entry["cam"], "master": bool(entry.get("master")),
         "hop": 0 if entry.get("master") else int(entry.get("hop", 1)),
         "pos": np.load(d / entry["pos"]), "nrm": np.load(d / entry["nrm"]),
         "mask": np.load(d / entry["mask"]).astype(np.float32)}
    H, W = v["pos"].shape[:2]
    if with_image and entry.get("image") and (d / entry["image"]).is_file():
        img = read_png(d / entry["image"]).astype(np.float32)[..., :3]
        if img.shape[:2] != (H, W):
            img = resize(img, W, H)
        v["img"] = img
        v["sky"] = sky_mask(v["pos"], img)
    return v


def read_pool(pool) -> dict:
    p = Path(pool) / POOL_FILE
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {"views": []}


def write_pool(pool, data) -> None:
    Path(pool).mkdir(parents=True, exist_ok=True)
    (Path(pool) / POOL_FILE).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def summary_line(results) -> str:
    """Panel line: cameras done, image calls used, retries and why."""
    if not results:
        return ""
    done = [r for r in results if r.get("ok")]
    calls = sum(int(r.get("calls") or 0) for r in results)
    retried = [r for r in results if (r.get("calls") or 0) > 1]
    line = f"多角度场景一致：{len(done)}/{len(results)} 个相机完成 · 图像调用 {calls} 次"
    if retried:
        line += " · 重试：" + "；".join(f"{r['name']}（{'、'.join(r.get('retry_reason') or [])}）" for r in retried)
    bad = [r["name"] for r in done if r.get("product_violations")]
    if bad:
        line += " · 产品仍超标：" + "、".join(bad)
    g = grade_note(done)
    if g:
        line += " · " + g
    o = occlusion_note(results)
    if o:
        line += " · " + o
    return line


def occlusion_note(results) -> str:
    """1.2.6 被遮挡就调视角: cameras moved for the occlusion and cameras skipped (no clear pose)."""
    moved, blocked = [], []
    for r in results or ():
        occ = r.get("occlusion") or {}
        if occ.get("status") == "adjusted":
            moved.append(f"{r['name']}（遮挡 {occ['occ_before'] * 100:.0f}%→{occ['occ_after'] * 100:.0f}%，"
                         f"距离 {occ['dist']:.1f}→{occ['dist_after']:.1f} m）")
        elif occ.get("status") == "blocked":
            blocked.append(r["name"])
    parts = []
    if moved:
        parts.append("被遮挡已临时调整机位（原相机未改）：" + "、".join(moved))
    if blocked:
        parts.append("产品被遮挡，已跳过/请调整相机：" + "、".join(blocked))
    return " · ".join(parts)


# --------------------------------------------------------------------------- Blender helpers (main thread)

def render_geometry_now(context, prefix, width, height) -> bool:
    """World-position (+hit alpha) and normal passes of the scene camera at width×height, as
    ``<prefix>_pos.npy`` / ``<prefix>_nrm.npy`` (float32, top row first). Called from
    ``render_clay_base`` while its temporary floor and product pose are in place. Restores every
    setting it changes."""
    import os
    import bpy
    scene = context.scene
    render = scene.render
    vl = context.view_layer
    cyc = getattr(scene, "cycles", None)
    snap = {"engine": render.engine, "x": render.resolution_x, "y": render.resolution_y,
            "pct": render.resolution_percentage, "film": render.film_transparent, "path": render.filepath,
            "fmt": render.image_settings.file_format, "depth": render.image_settings.color_depth,
            "mode": render.image_settings.color_mode, "override": vl.material_override,
            "filter": getattr(render, "filter_size", None)}
    csnap = None
    if cyc is not None:
        csnap = {k: getattr(cyc, k) for k in ("samples", "max_bounces", "use_denoising", "use_adaptive_sampling")
                 if hasattr(cyc, k)}
    mats = []
    ok = True
    try:
        render.engine = "CYCLES"
        if cyc is not None:
            for k, v in (("samples", 1), ("max_bounces", 0), ("use_denoising", False), ("use_adaptive_sampling", False)):
                if hasattr(cyc, k):
                    setattr(cyc, k, v)
        if snap["filter"] is not None:
            render.filter_size = 0.01
        render.resolution_x, render.resolution_y, render.resolution_percentage = int(width), int(height), 100
        render.film_transparent = True
        render.image_settings.file_format = "OPEN_EXR"
        render.image_settings.color_depth = "32"
        render.image_settings.color_mode = "RGBA"
        for kind, sock in (("pos", "Position"), ("nrm", "Normal")):
            mat = bpy.data.materials.new(f"_wondful_mv_{kind}")
            mats.append(mat)
            mat.use_nodes = True
            nt = mat.node_tree
            nt.nodes.clear()
            g = nt.nodes.new("ShaderNodeNewGeometry")
            em = nt.nodes.new("ShaderNodeEmission")
            oo = nt.nodes.new("ShaderNodeOutputMaterial")
            nt.links.new(g.outputs[sock], em.inputs[0])
            nt.links.new(em.outputs[0], oo.inputs[0])
            vl.material_override = mat
            exr = f"{prefix}_{kind}.exr"
            render.filepath = exr
            bpy.ops.render.render(write_still=True)
            im = bpy.data.images.load(exr)
            try:
                a = np.array(im.pixels[:], np.float32).reshape(im.size[1], im.size[0], 4)[::-1]
            finally:
                bpy.data.images.remove(im)
            try:
                os.remove(exr)
            except OSError:
                pass
            np.save(f"{prefix}_{kind}.npy", np.ascontiguousarray(a if kind == "pos" else a[..., :3]))
    except Exception as exc:
        print(f"[Wondful] 多角度场景一致：位置 / 法线通道渲染失败：{exc}")
        ok = False
    finally:
        vl.material_override = snap["override"]
        for mat in mats:
            try:
                bpy.data.materials.remove(mat)
            except Exception:
                pass
        render.engine = snap["engine"]
        render.resolution_x, render.resolution_y, render.resolution_percentage = snap["x"], snap["y"], snap["pct"]
        render.film_transparent = snap["film"]
        render.filepath = snap["path"]
        try:
            render.image_settings.file_format = snap["fmt"]
            render.image_settings.color_depth = snap["depth"]
            render.image_settings.color_mode = snap["mode"]
        except Exception:
            pass
        if snap["filter"] is not None:
            render.filter_size = snap["filter"]
        if cyc is not None and csnap:
            for k, v in csnap.items():
                try:
                    setattr(cyc, k, v)
                except Exception:
                    pass
    return ok


def scene_camera_record(scene, cam_obj) -> dict:
    from .viewport_capture import camera_output_dimensions

    class _Ctx:
        pass
    ctx = _Ctx()
    ctx.scene = scene
    w, h = camera_output_dimensions(ctx)
    d = cam_obj.data
    return camera_record(cam_obj.name, cam_obj.matrix_world, d.angle, getattr(d, "sensor_fit", "AUTO"), w, h,
                         ortho=getattr(d, "type", "PERSP") != "PERSP")


def product_bounds(collection):
    from mathutils import Vector
    pts = [o.matrix_world @ Vector(c) for o in collection.all_objects
           if o.type == "MESH" and not o.hide_render for c in o.bound_box]
    if not pts:
        return None
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return mn, mx


def obstacle(context, cam_obj, product_coll, nx=20, ny=12):
    """(blocked, hidden) for one camera: share of view rays in the product's screen column that hit
    a standing environment object in front of the product, and share of product sample points
    hidden behind one. Ground-level objects (below 15 % of the product height) never count."""
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    scene = context.scene
    b = product_bounds(product_coll)
    if b is None:
        return 0.0, 0.0
    mn, mx = b
    P = (mn + mx) / 2
    ph = max(1e-3, mx.z - mn.z)
    prod = set(product_coll.all_objects)
    dg = context.evaluated_depsgraph_get()
    o = cam_obj.matrix_world.translation
    dP = (P - o).length

    def standing(hit_obj, loc):
        return hit_obj not in prod and loc.z > mn.z + 0.15 * ph

    uv = [world_to_camera_view(scene, cam_obj, Vector((x, y, z))) for x in (mn.x, mx.x) for y in (mn.y, mx.y)
          for z in (mn.z, mx.z)]
    x0 = max(0.0, min(p.x for p in uv) - 0.1)
    x1 = min(1.0, max(p.x for p in uv) + 0.1)
    y1 = min(1.0, max(p.y for p in uv) + 0.05)
    R = cam_obj.matrix_world.to_3x3()
    rec = scene_camera_record(scene, cam_obj)
    bad = n = 0
    for iy in range(ny):
        for ix in range(nx):
            fx = x0 + (x1 - x0) * (ix + 0.5) / nx
            fy = y1 * (iy + 0.5) / ny
            d = (R @ Vector(((fx * 2 - 1) * rec["tx"], (fy * 2 - 1) * rec["ty"], -1.0))).normalized()
            n += 1
            hit, loc, _nrm, _i, ob, _m = scene.ray_cast(dg, o, d, distance=dP * 0.85)
            if hit and standing(ob, loc):
                bad += 1
    hid = tot = 0
    for fx in (0.15, 0.5, 0.85):
        for fy in (0.15, 0.5, 0.85):
            for fz in (0.1, 0.45, 0.8):
                q = Vector((mn.x + (mx.x - mn.x) * fx, mn.y + (mx.y - mn.y) * fy, mn.z + ph * fz))
                v = q - o
                tot += 1
                hit, loc, _nrm, _i, ob, _m = scene.ray_cast(dg, o, v.normalized(), distance=max(0.0, v.length - 0.05))
                if hit and ob not in prod:
                    hid += 1
    return bad / max(1, n), hid / max(1, tot)


def decide_interior(kind, ray_hits, shape_room, wall_like, rays=72) -> bool:
    """Whether the other cameras get a closed room (1.2.6 室内误判 fix).

    ``kind``: the ④ analysis scene kind ("outdoor" / "interior" / "" unknown) — the authority when known: an
    outdoor street is never closed in, whatever its blocks are called (场景补全's 「推测建筑」「推测窗」 used to
    trip the old ``wall_like >= 2`` name rule and walled in outdoor scenes since cfc6ced). Unknown (a hand-made
    massing collection): shape first — horizontal rays from the product mostly hit blocks, or ``massing.is_interior``
    (a slab overhead / thin walls on ≥ 2 sides); wall-like NAMES count only together with partial enclosure
    (≥ a quarter of the rays hit something). ``wall_like`` must not include 场景补全 or 多角度补充 objects."""
    kind = str(kind or "")
    if kind == "outdoor":
        return False
    if kind == "interior":
        return True
    if ray_hits >= rays // 2 or shape_room:
        return True
    return wall_like >= 2 and ray_hits >= rays // 4


# --------------------------------------------------------------------------- 1.2.6 被遮挡就调视角
# Real cyber orbit test 2026-10-20: narrow street + 12.6 m orbit put 4 of 4 cameras behind buildings (product 100 %
# hidden) and every one still spent 1–2 image calls. Now each 多角度 camera is ray-checked through the blockout
# BEFORE any image call; an occluded camera is moved (temporarily — the user's camera is restored afterwards and the
# pose used is recorded) or skipped.

OCCLUDE_MAX = 0.20                 # share of the product silhouette hidden by the environment that is acceptable
POSE_RADII = (0.9, 0.8, 0.7, 0.6, 0.5, 0.42, 0.35, 0.3, 0.25)      # step 1: closer along the line to the product
POSE_AZ = (5, -5, 10, -10, 15, -15, 20, -20, 30, -30)               # step 2: small azimuth nudges (degrees)
POSE_DH = (0.0, 1.0, -0.5, 2.0)                                      # … and height nudges (metres)
POSE_AZ_SUGGEST = (40, -40, 50, -50, 60, -60, 75, -75, 90, -90)      # blocked: the nearest clear azimuth, as a hint only
POSE_MIN_LENS = 0.6                # the lens may widen to this share of its focal length to keep the product framed
POSE_MIN_MM = 12.0
POSE_FRAME = 0.94                  # product extent must stay inside this share of the frame (both axes)


def pose_candidates(cam_xyz, aim_xyz, min_dist, radii=POSE_RADII, az=POSE_AZ, dh=POSE_DH):
    """Ordered camera positions to try when the original one is occluded: first straight in along the horizontal line
    to the aim point (azimuth and height kept), then azimuth / height nudges (closest to the original first), each
    from the original distance inward. Every candidate: {"pos", "r", "d_az", "dh"}; ``min_dist`` = the closest
    horizontal distance allowed (product half diagonal + clearance)."""
    cx, cy, cz = (float(v) for v in cam_xyz)
    ax, ay, _az = (float(v) for v in aim_xyz)
    dx, dy = cx - ax, cy - ay
    R = math.hypot(dx, dy)
    if R < 1e-6:
        return []
    base = math.atan2(dx, dy)
    out = []

    def add(r, d_az, d_h):
        dist = R * r
        if dist < min_dist - 1e-9:
            return
        a = base + math.radians(d_az)
        out.append({"pos": (ax + math.sin(a) * dist, ay + math.cos(a) * dist, cz + d_h), "r": round(r, 3),
                    "d_az": float(d_az), "dh": float(d_h), "dist": round(dist, 3)})
    for r in radii:
        add(r, 0.0, 0.0)
    nudges = sorted(((d_az, d_h) for d_az in (0.0,) + tuple(az) for d_h in dh if (d_az, d_h) != (0.0, 0.0)),
                    key=lambda t: (abs(t[0]) / 5.0 + abs(t[1]), abs(t[0])))
    for d_az, d_h in nudges:
        for r in (1.0,) + tuple(radii):
            add(r, d_az, d_h)
    return out


def plan_clear_pose(candidates, evaluate, max_occ=OCCLUDE_MAX):
    """First candidate ``evaluate(c)`` accepts: it returns {"occ", "free", "lens_scale"} (lens_scale None = the
    product cannot be framed). Returns (candidate + its evaluation, tried count) or (None, tried)."""
    tried = 0
    for c in candidates:
        tried += 1
        ev = evaluate(c) or {}
        if ev.get("free") and ev.get("lens_scale") is not None and float(ev.get("occ", 1.0)) <= max_occ:
            return dict(c, **ev), tried
    return None, tried


def _look_at(pos, aim):
    from mathutils import Vector
    return (Vector(aim) - Vector(pos)).to_track_quat("-Z", "Y").to_matrix()


def _product_samples(context, product_coll, limit=1500):
    """(world vertex samples, BVH of the product meshes) — the silhouette the occlusion is measured on."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    dg = context.evaluated_depsgraph_get()
    verts, polys, pts = [], [], []
    for o in product_coll.all_objects:
        if o.type != "MESH" or o.hide_render:
            continue
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        try:
            M = ev.matrix_world
            off = len(verts)
            vv = [M @ v.co for v in me.vertices]
            verts.extend(vv)
            polys.extend([tuple(off + i for i in p.vertices) for p in me.polygons])
        finally:
            ev.to_mesh_clear()
    if not verts:
        return [], None
    step = max(1, len(verts) // limit)
    pts = [Vector(v) for v in verts[::step]]
    return pts, BVHTree.FromPolygons(verts, polys, all_triangles=False)


def product_occlusion(context, pos, R, tx, ty, product_coll, samples=None, grid=(32, 20)):
    """Share of the product silhouette (pixels whose ray hits the product) that an environment object hides,
    for a camera at ``pos`` with rotation ``R`` (3×3, -Z forward) and half-FOV tangents ``tx`` / ``ty``. Also
    returns the product's max |NDC| extent (framing) and the silhouette sample count."""
    from mathutils import Vector
    scene = context.scene
    dg = context.evaluated_depsgraph_get()
    pts, bvh = samples if samples is not None else _product_samples(context, product_coll)
    if bvh is None:
        return {"occ": 0.0, "extent": 0.0, "n": 0}
    Ri = R.transposed()
    o = Vector(pos)
    xs, ys, ext = [], [], 0.0
    for p in pts:
        q = Ri @ (p - o)
        if q.z > -1e-3:
            return {"occ": 1.0, "extent": 9.0, "n": 0, "behind": True}
        x, y = (q.x / -q.z) / tx, (q.y / -q.z) / ty
        xs.append(x)
        ys.append(y)
        ext = max(ext, abs(x), abs(y))
    x0, x1 = max(-1.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(-1.0, min(ys)), min(1.0, max(ys))
    prod = set(product_coll.all_objects)
    nx, ny = grid
    sil = hid = 0
    for iy in range(ny):
        for ix in range(nx):
            u = x0 + (x1 - x0) * (ix + 0.5) / nx
            v = y0 + (y1 - y0) * (iy + 0.5) / ny
            d = (R @ Vector((u * tx, v * ty, -1.0))).normalized()
            loc, _n, _i, tP = bvh.ray_cast(o, d)
            if loc is None:
                continue
            sil += 1
            hit, _l, _nn, _ii, ob, _m = scene.ray_cast(dg, o, d, distance=max(0.0, tP - 0.03))
            if hit and ob not in prod:
                hid += 1
    return {"occ": hid / sil if sil else 1.0, "extent": ext, "n": sil}


def camera_free(context, pos, product_coll, clearance=0.35):
    """The camera stands in open space: not inside a block (most axis rays hit back faces) and nothing within
    ``clearance`` metres around it horizontally."""
    from mathutils import Vector
    scene = context.scene
    dg = context.evaluated_depsgraph_get()
    prod = set(product_coll.all_objects)
    o = Vector(pos)
    back = 0
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        dv = Vector(d)
        hit, _l, n, _i, ob, _m = scene.ray_cast(dg, o, dv, distance=500.0)
        if hit and ob not in prod and n.dot(dv) > 0:
            back += 1
    if back >= 3:
        return False
    for k in range(8):
        a = 2 * math.pi * k / 8
        hit, _l, _n, _i, ob, _m = scene.ray_cast(dg, o, Vector((math.sin(a), math.cos(a), 0.0)), distance=clearance)
        if hit and ob not in prod:
            return False
    return True


def adjust_camera_for_occlusion(context, cam_obj, product_coll, max_occ=OCCLUDE_MAX, apply=True):
    """Ray-check the product through the blockout from ``cam_obj``; when more than ``max_occ`` of its silhouette is
    hidden, find the nearest clear pose (``pose_candidates`` order: closer first, then azimuth / height nudges; the
    lens may widen to keep the product framed) and — with ``apply`` — move the camera there. The caller restores the
    camera (``snapshot``). Returns a JSON-safe record: before / after occlusion, original and adjusted pose, status
    "clear" (no change), "adjusted" or "blocked" (no clear pose: skip, never spend an image call)."""
    from mathutils import Matrix, Vector
    scene = context.scene
    b = product_bounds(product_coll)
    rec0 = scene_camera_record(scene, cam_obj)
    M0 = cam_obj.matrix_world.copy()
    pos0 = M0.translation.copy()
    R0 = M0.to_3x3().normalized()
    out = {"status": "clear", "camera": cam_obj.name, "lens": round(float(cam_obj.data.lens), 3),
           "pos": [round(v, 3) for v in pos0]}
    if b is None:
        out["status"] = "no_product"
        return out
    mn, mx = b
    C = (mn + mx) / 2
    samples = _product_samples(context, product_coll)
    tx, ty = float(rec0["tx"]), float(rec0["ty"])
    first = product_occlusion(context, pos0, R0, tx, ty, product_coll, samples)
    out["occ_before"] = round(first["occ"], 3)
    out["occ_after"] = out["occ_before"]
    rel = pos0 - C
    out.update({"dist": round(rel.xy.length, 3), "height": round(rel.z, 3),
                "azimuth_deg": round(math.degrees(math.atan2(rel.x, rel.y)), 2)})
    if first["occ"] <= max_occ:
        return out
    fwd = -(R0 @ Vector((0, 0, 1)))
    aim = pos0 + fwd * max(0.5, (C - pos0).dot(fwd))          # the point on the view axis nearest the product
    half = 0.5 * math.hypot(mx.x - mn.x, mx.y - mn.y)
    cands = pose_candidates(tuple(pos0), tuple(aim), half + 1.2)
    lens0 = float(cam_obj.data.lens)

    def evaluate(c):
        pos = Vector(c["pos"])
        if pos.z < mn.z + 0.3:
            return {"free": False}
        R = _look_at(pos, aim)
        ev = product_occlusion(context, pos, R, tx, ty, product_coll, samples, grid=(24, 14))
        if ev.get("behind"):
            return {"free": False}
        need = 1.0 if ev["extent"] <= POSE_FRAME else POSE_FRAME / ev["extent"]
        scale = need if need >= max(POSE_MIN_LENS, POSE_MIN_MM / max(lens0, 1e-3)) else None
        if ev["occ"] > max_occ or scale is None:
            return {"occ": ev["occ"], "lens_scale": scale, "free": True}
        if scale < 1.0:      # framed with a wider lens: occlusion re-measured with that field of view
            ev = product_occlusion(context, pos, R, tx / scale, ty / scale, product_coll, samples, grid=(24, 14))
        return {"occ": ev["occ"], "lens_scale": scale, "free": camera_free(context, pos, product_coll)}

    best, tried = plan_clear_pose(cands, evaluate, max_occ)
    out["tried"] = tried
    if best is None:
        out["status"] = "blocked"
        out["message"] = f"{cam_obj.name}：产品被遮挡 {first['occ'] * 100:.0f}%，附近没有找到不被遮挡的机位，已跳过（请调整相机）"
        # a hint for the user (never applied: a pose this far from the camera is a different view)
        far, _n = plan_clear_pose([c for c in pose_candidates(tuple(pos0), tuple(aim), half + 1.2, az=POSE_AZ_SUGGEST,
                                                              dh=(0.0, 1.0)) if abs(c["d_az"]) > max(POSE_AZ)],
                                  evaluate, max_occ)
        if far is not None:
            out["suggest"] = {"d_az_deg": far["d_az"], "dist": far["dist"], "d_height": far["dh"],
                              "lens": round(lens0 * float(far["lens_scale"]), 1)}
            out["message"] += (f"；最近的可用机位：方位 {far['d_az']:+.0f}°、距离 {far['dist']:.1f} m"
                               + (f"、焦距 {lens0 * float(far['lens_scale']):.0f} mm" if far["lens_scale"] < 0.999 else ""))
        return out
    pos = Vector(best["pos"])
    R = _look_at(pos, aim)
    scale = float(best["lens_scale"])
    final = product_occlusion(context, pos, R, tx / scale, ty / scale, product_coll, samples)
    rel = pos - C
    out.update({"status": "adjusted", "occ_after": round(final["occ"], 3), "r_factor": best["r"],
                "d_az_deg": best["d_az"], "d_height": best["dh"], "lens_after": round(lens0 * scale, 3),
                "pos_after": [round(v, 3) for v in pos], "dist_after": round(rel.xy.length, 3),
                "height_after": round(rel.z, 3),
                "azimuth_after_deg": round(math.degrees(math.atan2(rel.x, rel.y)), 2)})
    out["message"] = (f"{cam_obj.name}：产品被遮挡 {first['occ'] * 100:.0f}% → {final['occ'] * 100:.0f}%，机位已临时调整"
                      f"（距离 {out['dist']:.1f}→{out['dist_after']:.1f} m，方位 {best['d_az']:+.0f}°，高度 {best['dh']:+.1f} m"
                      + (f"，焦距 {lens0:.0f}→{lens0 * scale:.0f} mm" if scale < 0.999 else "") + "；原相机未改）")
    if apply:
        cam_obj.matrix_world = Matrix.Translation(pos) @ R.to_4x4()
        if scale < 0.999:
            cam_obj.data.lens = lens0 * scale
        context.view_layer.update()
    return out


def snapshot_camera(cam_obj) -> dict:
    return {"obj": cam_obj, "matrix": cam_obj.matrix_world.copy(), "lens": float(cam_obj.data.lens)}


def restore_camera(snap) -> None:
    try:
        snap["obj"].matrix_world = snap["matrix"]
        snap["obj"].data.lens = snap["lens"]
    except Exception:
        pass


def _block_words(obj) -> str:
    """Name words of one block for the interior / built-up checks (1.2.3: the same naming as the
    prompt — hint, object / material name, analysis label — without the size guess)."""
    from .massing import group_hint
    hint, _src = group_hint([obj], shape=False)
    return (hint + " " + obj.name).lower()


def widen_blockout(context, massing_coll, product_coll, master_cam, fillers=True):
    """Temporary wider blockout for the other cameras (removed again by ``remove_aux``):
    - interiors (horizontal rays from the product mostly hit blocks, or wall-like blocks exist):
      flat walls closing every side where rays escape, never on a side the master camera sees;
    - built-up outdoor scenes (≥ 2 building-like blocks): plain filler buildings in the 36 sectors
      around the product with nothing within 70 m, at 30 / 42 / 55 m, outside the master frame.
    Tagged ``wondful_mv_aux`` with a design hint, linked into the massing collection."""
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Matrix, Vector
    from . import massing as MS
    made, log = [], {"closure": [], "fillers": 0}
    if massing_coll is None or product_coll is None:
        return made, log
    scene = context.scene
    b = product_bounds(product_coll)
    blocks = MS.massing_objects(massing_coll, product_coll)
    if b is None or not blocks:
        return made, log
    mn, mx = b
    Pg = Vector(((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mn.z))
    dg = context.evaluated_depsgraph_get()
    prod = set(product_coll.all_objects)
    words = [_block_words(o) for o in blocks]
    hits = 0
    miss = {"x-": 0, "x+": 0, "y-": 0, "y+": 0}
    for j in range(72):
        a = 2 * math.pi * j / 72
        d = Vector((math.sin(a), math.cos(a), 0.0))
        hit, _loc, _n, _i, ob, _m = scene.ray_cast(dg, Pg + Vector((0, 0, 1.3)), d, distance=40.0)
        if hit and ob not in prod:
            hits += 1
        else:
            key = ("x+" if d.x > 0 else "x-") if abs(d.x) > abs(d.y) else ("y+" if d.y > 0 else "y-")
            miss[key] += 1
    kind = MS.blockout_kind(blocks)
    own = [o for o in blocks if not o.get(MS.INFERRED_PROP) and not o.get(AUX_TAG)]
    boxes = [MS._box_of([o.matrix_world @ Vector(c) for c in o.bound_box]) for o in own]
    pbox = (mn.x, mn.y, mn.z, mx.x, mx.y, mx.z)
    wall_like = sum(any(w in _block_words(o) for w in WALL_WORDS) for o in own)
    interior = decide_interior(kind, hits, MS.is_interior(boxes, pbox), wall_like)
    log.update({"interior": interior, "kind": kind or "unknown", "ray_hits": hits, "wall_like": wall_like})
    mat = MS.make_material(bpy.data.materials.new)
    mat.name = "_wondful_mv_aux_mat"
    made.append(mat)

    def add(name, verts, faces, hint):
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        me.update()
        ob = bpy.data.objects.new(name, me)
        me.materials.append(mat)
        ob[AUX_TAG] = 1
        ob["wondful_hint"] = hint
        massing_coll.objects.link(ob)
        made.append(ob)
        return ob

    def seen_by_master(pts):
        for p in pts:
            c = world_to_camera_view(scene, master_cam, Vector(p))
            if c.z > 0 and -0.02 <= c.x <= 1.02 and -0.02 <= c.y <= 1.02:
                return True
        return False

    allpts = [o.matrix_world @ Vector(c) for o in blocks for c in o.bound_box]
    bmn = Vector((min(p.x for p in allpts), min(p.y for p in allpts), min(p.z for p in allpts)))
    bmx = Vector((max(p.x for p in allpts), max(p.y for p in allpts), max(p.z for p in allpts)))
    if interior:
        z0, zt = mn.z, max(bmx.z, mn.z + 2.6)
        sides = {"y-": [(bmn.x, bmn.y, z0), (bmx.x, bmn.y, z0), (bmx.x, bmn.y, zt), (bmn.x, bmn.y, zt)],
                 "y+": [(bmn.x, bmx.y, z0), (bmx.x, bmx.y, z0), (bmx.x, bmx.y, zt), (bmn.x, bmx.y, zt)],
                 "x-": [(bmn.x, bmn.y, z0), (bmn.x, bmx.y, z0), (bmn.x, bmx.y, zt), (bmn.x, bmn.y, zt)],
                 "x+": [(bmx.x, bmn.y, z0), (bmx.x, bmx.y, z0), (bmx.x, bmx.y, zt), (bmx.x, bmn.y, zt)]}
        for key, n in miss.items():
            if n == 0 or seen_by_master(sides[key]):
                continue
            add(f"多角度补充_墙_{key}", sides[key], [(0, 1, 2, 3)], CLOSURE_HINT)
            log["closure"].append(key)
        return made, log
    builds = [o for o, t in zip(blocks, words) if any(w in t for w in BUILDING_WORDS)]
    if not fillers:          # 1.2.5: the ④ 场景补全 already continues the street
        log["fillers_skipped"] = "场景补全"
        return made, log
    if len(builds) < 2:
        return made, log
    dims = sorted((o.dimensions.x, o.dimensions.y, o.dimensions.z) for o in builds)
    bw = float(np.median([min(max(d[0], 4), 14) for d in dims]))
    bd = float(np.median([min(max(d[1], 4), 14) for d in dims]))
    bh = float(np.median([min(max(d[2], 3.5), 10) for d in dims]))
    for j in range(36):
        a = 2 * math.pi * (j + 0.5) / 36
        dv = Vector((math.sin(a), math.cos(a), 0.0))
        hit, _loc, _n, _i, ob, _m = scene.ray_cast(dg, Pg + Vector((0, 0, 3.0)), dv, distance=70.0)
        if hit and ob not in prod:
            continue
        for dist in (30.0, 42.0, 55.0):
            c = Pg + dv * dist
            fr = (math.sin(j * 12.9898) * 43758.5453) % 1.0
            w_, h_ = bw * (0.8 + 0.5 * fr), bh * (0.7 + 0.7 * ((fr * 7.31) % 1.0))
            R = Matrix.Rotation(-a, 3, "Z")
            corners = [c + R @ Vector((sx * w_ / 2, sy * bd / 2, 0)) for sx in (-1, 1) for sy in (-1, 1)]
            pts = [tuple(q + Vector((0, 0, z))) for q in corners for z in (0.0, h_)]
            if seen_by_master(pts):
                continue
            v = [tuple(q + Vector((0, 0, -0.1))) for q in corners] + [tuple(q + Vector((0, 0, h_))) for q in corners]
            f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (1, 5, 7, 3), (3, 7, 6, 2), (2, 6, 4, 0)]
            add(f"多角度补充_建筑_{j}", v, f, FILLER_HINT)
            log["fillers"] += 1
            break
    return made, log


def remove_aux(made) -> None:
    import bpy
    for block in reversed(made or []):
        try:
            if isinstance(block, bpy.types.Object):
                me = block.data
                bpy.data.objects.remove(block, do_unlink=True)
                if me is not None and me.users == 0:
                    bpy.data.meshes.remove(me)
            elif isinstance(block, bpy.types.Material):
                bpy.data.materials.remove(block)
        except Exception:
            pass
