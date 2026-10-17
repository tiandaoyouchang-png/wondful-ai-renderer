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

Retry policy (user, 2026-10-17): one image call per camera, at most one retry, and only when
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
ENV_MIN_COVIS = 0.5        # … only judged when the views share at least this much of the frame
MAX_CALLS = 2              # one call + at most one retry per camera
MASTER_BIAS = 0.10         # the master wins ties between equally head-on sources
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
    best = np.full((H, W), -1.0, np.float32)
    srcmap = np.zeros((H, W), np.int16)
    skydone = np.zeros((H, W), bool)
    cached = []
    for n, vj in enumerate(sources, start=1):
        U, V, vis, sky, hit, sky_in = correspond(vk, vj)
        cached.append((U, V, sky, sky_in))
        dvec = _mat(vj["cam"])[:3, 3] - Pk[..., :3]
        dvec /= np.maximum(np.linalg.norm(dvec, axis=-1, keepdims=True), 1e-6)
        face = np.abs((Nk * dvec).sum(-1)) + (MASTER_BIAS if vj.get("master") else 0.0)
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
    """Fewest product violations, then fewest environment violations, then the lower edge."""
    def key(t):
        rep = t.get("identity") or {}
        edge = rep.get("edge_px_detail", rep.get("edge_px"))
        return (len(t.get("product_violations") or []), len(t.get("env_violations") or []),
                99.0 if edge is None else float(edge))
    return min(tries, key=key)


# --------------------------------------------------------------------------- text

LIGHT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["light_type", "key_rel_azimuth_deg", "key_elevation_deg", "sky_words"],
    "properties": {
        "light_type": {"type": "string", "enum": ["direct_sun", "hazy_sun", "overcast", "interior_window", "interior_mixed"]},
        "key_rel_azimuth_deg": {"type": "number", "description":
                                "horizontal direction FROM the scene TOWARD the sun / brightest sky / main window, relative to "
                                "the camera: 0 = straight ahead (backlight), +90 = camera right, -90 = camera left, 180 = behind"},
        "key_elevation_deg": {"type": "number"},
        "sky_words": {"type": "string", "description": "short Chinese description of the light: time of day, colour "
                                                       "temperature, cloud, haze, shadow softness"}}}

LIGHT_PROMPT = ("This is a rendered photo of a scene. Estimate the main light: the sun (or the brightest sky region if "
                "overcast, or the main window if it is an interior). Use cast shadows, highlights, shading and the sky "
                "gradient. Give the horizontal direction from the scene toward the light relative to the camera (0 ahead / "
                "backlight, +90 right, -90 left, 180 behind the camera), its elevation, and a short Chinese description.")


def world_light(light: dict, master_cam) -> dict:
    """Make the master-relative estimate a world direction (fixed for every other camera)."""
    out = dict(light or {})
    out["world_azimuth_deg"] = round(((float(out.get("key_rel_azimuth_deg", 0.0)) + yaw_deg(master_cam)) + 180.0) % 360.0
                                     - 180.0, 1)
    return out


def light_sentence(light: dict, cam) -> str:
    """Where the world-locked light sits for this camera, in the words the prototype used."""
    if not light or "world_azimuth_deg" not in light:
        return ""
    rel = (float(light["world_azimuth_deg"]) - yaw_deg(cam) + 180.0) % 360.0 - 180.0
    a = abs(rel)
    lr = "右" if rel > 0 else "左"
    other = "左" if lr == "右" else "右"
    words = str(light.get("sky_words") or "").strip()
    kind = str(light.get("light_type") or "")
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


def view_text(massing_text, light_line, source_names) -> str:
    """Per-view addition to the simple-path prompt (goes in as ``massing_text``)."""
    parts = [t.strip() for t in (massing_text or "",) if t and t.strip()]
    if light_line:
        parts.append(light_line)
    names = [n for n in source_names or () if n]
    parts.append(BASE_NOTE.format(sources="主视角" + ("和" + "、".join(names[1:]) if len(names) > 1 else "")))
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
    return line


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


def _block_words(obj) -> str:
    """Name words of one block for the interior / built-up checks (1.2.3: the same naming as the
    prompt — hint, object / material name, analysis label — without the size guess)."""
    from .massing import group_hint
    hint, _src = group_hint([obj], shape=False)
    return (hint + " " + obj.name).lower()


def widen_blockout(context, massing_coll, product_coll, master_cam):
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
    wall_like = sum(any(w in t for w in WALL_WORDS) for t in words)
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
    interior = hits >= 36 or wall_like >= 2
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
