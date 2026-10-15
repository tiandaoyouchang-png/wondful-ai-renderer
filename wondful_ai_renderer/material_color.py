"""按材质校色 (1.2.0, part of 细节回贴): pull each material's colour back to the model's.

Pure numpy + stdlib, no bpy.

细节回贴 takes the low frequencies (lighting, reflections, colour) from the AI result, so a
material the image model painted in the wrong hue (wood legs drawn too red, a fabric drawn
too orange) stays wrong. Blender knows which pixel belongs to which material: a *material-ID
pass* (each product material rendered as one flat emission colour, no anti-aliasing, same
camera and size as the detail pass) gives one region per material. For every region:

1. robust colour statistics in CIE Lab: medians over the mid-tone pixels only (highlights and
   deep shadows excluded) of the AI result and of the detail pass (real materials, flat
   white light = close to albedo);
2. the scene's global colour cast (sunset warmth, neon tint) is estimated as the
   area-weighted median offset AI − model over the matte / ordinary materials and kept: only
   the part of a material's error that differs from the cast is corrected;
3. the AI's chroma is scaled and its hue rotated (and, limited, its L* scaled) so the region's
   median moves towards the model's; applied per pixel as factors / rotations, so shadows,
   near-neutral highlights and every pixel's variation keep their relation (no pixel is pushed
   through grey into the complementary hue), reduced further on specular highlights:
   lighting gradients, shadows and reflections stay the AI's. Shifts are capped (no garish results) and feathered across region borders.

Skipped: glass / transparent, emissive, mirror-like chrome, regions below a pixel minimum.
Reflective car paint and metals get a gentle strength (their AI colour is mostly reflected
environment). The weight is multiplied by the paste weight, so the outer ~4 px AI edge and
the background are never touched."""
from __future__ import annotations

import json
import re

import numpy as np

from .layered_output import blur, erode

DEFAULT_STRENGTH = 0.7
MAX_AB_SHIFT = 10.0          # Lab a*/b* units, before strength, when the shift adds chroma
MAX_AB_DESAT = 24.0          # … when it lowers chroma (calming an over-saturated part is never garish)
MAX_L_SHIFT = 6.0            # Lab L* units, before strength
MAX_L_RATIO = (0.8, 1.25)    # and at most this lightness factor on the region
L_AMOUNT = 0.5               # share of the relative lightness error that is corrected
MIN_AREA_FRAC = 0.0002       # of the canvas (≈184 px at 1280×720), and at least MIN_AREA_PX
MIN_AREA_PX = 120
KIND_GAIN = {"default": 1.0, "matte": 1.0, "reflective": 0.35, "glass": 0.0, "emissive": 0.0, "mirror": 0.0}
SKIP_KINDS = {"glass", "emissive", "mirror"}
CAST_KINDS = {"default", "matte"}
PALETTE_LEVELS = (0, 51, 102, 153, 204, 255)

_PAINT_RE = re.compile(r"(paint|lacquer|clear ?coat|carpaint|车漆|烤漆|漆)", re.IGNORECASE)
_MIRROR_RE = re.compile(r"(mirror|chrome|镜|镀铬)", re.IGNORECASE)


def name_looks_paint(name) -> bool:
    return bool(_PAINT_RE.search(str(name or "")))


def name_looks_mirror(name) -> bool:
    return bool(_MIRROR_RE.search(str(name or "")))


def material_kind(name="", glass=False, emissive=False, matte=False, metallic=0.0, roughness=0.5,
                  coat=0.0) -> str:
    """'glass' | 'emissive' | 'mirror' | 'reflective' | 'matte' | 'default' (pure, from values
    the Blender side reads off the material; unknown / linked inputs come in as defaults)."""
    if glass:
        return "glass"
    if emissive:
        return "emissive"
    if matte:
        return "matte"
    if name_looks_mirror(name) or (metallic >= 0.9 and roughness <= 0.1):
        return "mirror"
    if name_looks_paint(name) or metallic >= 0.5 or coat >= 0.3:
        return "reflective"
    return "default"


# --------------------------------------------------------------------------- ID palette

def palette_colour(index: int):
    """sRGB 8-bit colour of material ``index`` (0-based) in the ID pass; never black."""
    n = len(PALETTE_LEVELS)
    code = int(index) + 1
    if code >= n ** 3:
        raise ValueError("too many materials for the ID palette")
    r, g, b = code % n, (code // n) % n, (code // (n * n)) % n
    return (PALETTE_LEVELS[r], PALETTE_LEVELS[g], PALETTE_LEVELS[b])


MAX_MATERIALS = len(PALETTE_LEVELS) ** 3 - 1


def srgb_to_linear_scalar(v: float) -> float:
    v = float(v)
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def decode_ids(rgba_u8: np.ndarray, count: int, tol: int = 20) -> np.ndarray:
    """ID pass (uint8 RGBA) -> int label map: material index, or -1 (background / no match)."""
    rgb = rgba_u8[..., :3].astype(np.int32)
    alpha = rgba_u8[..., 3] if rgba_u8.shape[-1] == 4 else np.full(rgba_u8.shape[:2], 255, np.uint8)
    lv = np.array(PALETTE_LEVELS, np.int32)
    q = np.abs(rgb[..., None] - lv[None, None, None, :]).argmin(-1)           # nearest level per channel
    err = np.abs(rgb - lv[q]).max(-1)
    n = len(PALETTE_LEVELS)
    code = q[..., 0] + q[..., 1] * n + q[..., 2] * n * n
    labels = code - 1
    ok = (alpha >= 250) & (err <= tol) & (labels >= 0) & (labels < int(count))
    return np.where(ok, labels, -1).astype(np.int32)


def resize_labels(labels: np.ndarray, w: int, h: int) -> np.ndarray:
    """Nearest-neighbour resize of a label map."""
    H, W = labels.shape
    if (W, H) == (w, h):
        return labels
    xs = np.clip(((np.arange(w) + 0.5) * W / w).astype(int), 0, W - 1)
    ys = np.clip(((np.arange(h) + 0.5) * H / h).astype(int), 0, H - 1)
    return labels[ys][:, xs]


def warp_labels(labels: np.ndarray, s, dx: float, dy: float, centre) -> np.ndarray:
    """Nearest-neighbour version of ``detail_paste.warp_image`` for label maps."""
    h, w = labels.shape
    cx, cy = centre
    sx, sy = (s, s) if np.isscalar(s) else (float(s[0]), float(s[1]))
    xs = np.round((np.arange(w, dtype=np.float32) - cx - dx) / sx + cx).astype(int)
    ys = np.round((np.arange(h, dtype=np.float32) - cy - dy) / sy + cy).astype(int)
    inside = ((xs >= 0) & (xs < w))[None, :] & ((ys >= 0) & (ys < h))[:, None]
    out = labels[ys.clip(0, h - 1)][:, xs.clip(0, w - 1)]
    return np.where(inside, out, -1).astype(np.int32)


def read_sidecar(path) -> list:
    """Material list written next to the ID pass: [{"index", "name", "kind"}, ...]."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    mats = data.get("materials") if isinstance(data, dict) else data
    return [dict(m) for m in mats or []]


def sidecar_path(matid_path) -> str:
    return str(matid_path) + ".json"


# --------------------------------------------------------------------------- colour space

_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]], np.float32)
_MI = np.linalg.inv(_M).astype(np.float32)
_WHITE = np.array([0.95047, 1.0, 1.08883], np.float32)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = np.clip(rgb.astype(np.float32), 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1).astype(np.float32)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f > 0.206893, f ** 3, (f - 16.0 / 116.0) / 7.787) * _WHITE
    lin = np.clip(xyz @ _MI.T, 0, None)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(srgb, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------- statistics

def robust_lab(lab: np.ndarray, sel: np.ndarray, lo: float = 15, hi: float = 85):
    """Median L*, a*, b* over the mid-tone pixels of ``sel`` (lightness percentiles lo..hi):
    specular highlights and the darkest shadow pixels do not count."""
    px = lab[sel]
    if len(px) == 0:
        return None
    l0, l1 = np.percentile(px[:, 0], [lo, hi])
    mid = px[(px[:, 0] >= l0) & (px[:, 0] <= l1)]
    if len(mid) < 10:
        mid = px
    return np.median(mid, 0).astype(np.float64)


def _wmedian(values, weights):
    v = np.asarray(values, np.float64); w = np.asarray(weights, np.float64)
    order = np.argsort(v)
    c = np.cumsum(w[order])
    return float(v[order][np.searchsorted(c, 0.5 * c[-1])])


def region_stats(ai_lab: np.ndarray, model_lab: np.ndarray, labels: np.ndarray, materials, valid: np.ndarray,
                 min_area: int) -> list:
    """Per material: core pixels (1-px eroded label ∩ ``valid``), AI and model robust Lab."""
    out = []
    kinds = {int(m.get("index", i)): (str(m.get("kind") or "default"), str(m.get("name") or ""))
             for i, m in enumerate(materials)}
    for idx in np.unique(labels[labels >= 0]):
        idx = int(idx)
        kind, name = kinds.get(idx, ("default", ""))
        core = erode(labels == idx, 1) & valid
        n = int(core.sum())
        rec = {"index": idx, "name": name, "kind": kind, "px": n}
        if n >= min_area:
            rec["ai"] = robust_lab(ai_lab, core)
            rec["model"] = robust_lab(model_lab, core, 10, 90)
        out.append(rec)
    return out


def _expected(model, ai):
    """Model colour seen at the AI's lightness: a*/b* scale with L* (a darker-lit surface of
    the same albedo has proportionally less chroma)."""
    s = float(np.clip((ai[0] + 2.0) / (model[0] + 2.0), 0.5, 1.5))
    return np.array([model[0], model[1] * s, model[2] * s])


def estimate_cast(stats) -> dict:
    """Scene colour cast: area-weighted median of (AI − expected model) over matte / ordinary
    materials (reflective ones if there is nothing else). dL, da, db in Lab units."""
    use = [r for r in stats if "ai" in r and r["kind"] in CAST_KINDS]
    if not use:
        use = [r for r in stats if "ai" in r and r["kind"] not in SKIP_KINDS]
    if not use:
        return {"dL": 0.0, "da": 0.0, "db": 0.0, "n": 0}
    offs = [r["ai"] - _expected(r["model"], r["ai"]) for r in use]
    w = [r["px"] for r in use]
    return {"dL": _wmedian([o[0] for o in offs], w), "da": _wmedian([o[1] for o in offs], w),
            "db": _wmedian([o[2] for o in offs], w), "n": len(use)}


def relative_error(rec, cast) -> dict:
    """Colour error of one material after removing the scene cast: ΔE76 (L*a*b*) and ΔEab
    (a*b* only: hue + chroma). Same definition for raw / pasted / corrected images."""
    exp = _expected(rec["model"], rec["ai"])
    d = rec["ai"] - exp - np.array([cast["dL"], cast["da"], cast["db"]])
    return {"dE": float(np.sqrt((d ** 2).sum())), "dEab": float(np.hypot(d[1], d[2])), "dL": float(d[0]),
            "da": float(d[1]), "db": float(d[2])}


# --------------------------------------------------------------------------- correction

MAX_HUE_DEG = 30.0
CHROMA_RANGE = (0.35, 1.35)


def plan_shift(ai, target, max_ab=MAX_AB_SHIFT, max_desat=MAX_AB_DESAT, max_l=MAX_L_SHIFT, l_amount=L_AMOUNT) -> dict:
    """Region correction as a chroma factor, a hue rotation and a lightness factor (applied per
    pixel, so a pixel's colour never crosses neutral and its variation is kept). Capped: chroma
    ×0.35–1.35, hue ±30° (less for near-neutral colours, whose hue is noise), median a*b*
    displacement ≤ ``max_ab`` when adding chroma / ``max_desat`` when removing it, L* ≤
    ``max_l`` and ×0.8–1.25. Returns dict(log_c, rot, log_l, shift=[dL, da, db] at the median)."""
    ai = np.asarray(ai, np.float64); target = np.asarray(target, np.float64)
    c0, c1 = float(np.hypot(ai[1], ai[2])), float(np.hypot(target[1], target[2]))
    f = float(np.clip((c1 + 0.5) / (c0 + 0.5), *CHROMA_RANGE))
    rot = 0.0
    if min(c0, c1) > 1.0:
        dh = float(np.angle(complex(target[1], target[2]) / complex(ai[1], ai[2])))
        conf = float(np.clip((min(c0, c1) - 1.0) / 6.0, 0, 1))          # near-neutral: hue unreliable
        rot = float(np.clip(dh, -np.radians(MAX_HUE_DEG), np.radians(MAX_HUE_DEG))) * conf
    z0 = complex(ai[1], ai[2])
    z1 = z0 * f * complex(np.cos(rot), np.sin(rot))
    disp = abs(z1 - z0)
    cap = max_ab if f > 1.0 else max_desat
    t = min(1.0, cap / disp) if disp > 1e-6 else 1.0
    log_c, rot = np.log(f) * t, rot * t
    lm = max(float(ai[0]), 5.0)
    dl = float(np.clip((target[0] - ai[0]) * l_amount, -max_l, max_l))
    dl = float(np.clip(dl, (MAX_L_RATIO[0] - 1) * lm, (MAX_L_RATIO[1] - 1) * lm))
    log_l = float(np.log((lm + dl) / lm))
    z2 = z0 * np.exp(log_c) * complex(np.cos(rot), np.sin(rot))
    return {"log_c": float(log_c), "rot": float(rot), "log_l": log_l,
            "shift": [dl, float(z2.real - ai[1]), float(z2.imag - ai[2])]}


def correct_colours(image: np.ndarray, model: np.ndarray, labels: np.ndarray, materials, weight: np.ndarray,
                    strength: float = DEFAULT_STRENGTH, model_alpha=None, protect=None,
                    max_ab: float = MAX_AB_SHIFT, max_desat: float = MAX_AB_DESAT, max_l: float = MAX_L_SHIFT,
                    l_amount: float = L_AMOUNT,
                    cast=None) -> dict:
    """Shift each material region of ``image`` (AI or pasted result, float RGB 0..1) towards the
    model colour of ``model`` (detail pass, same canvas). ``labels``: int map (-1 = none);
    ``materials``: [{"index", "name", "kind"}]; ``weight``: where the image may change at all
    (paste weight: 0 on the outer AI edge and the background); ``protect``: optional 0..1 map
    left untouched (pasted keep-exact parts). Returns dict(image, regions, cast, changed_px)."""
    strength = float(np.clip(strength, 0.0, 1.0))
    h, w = labels.shape
    img = image[..., :3].astype(np.float32)
    if strength <= 0 or not (labels >= 0).any():
        return {"image": img.copy(), "regions": [], "cast": None, "changed_px": 0}
    L = max(h, w)
    lab = srgb_to_lab(img)
    mlab = srgb_to_lab(model[..., :3])
    valid = weight > 0.5
    if model_alpha is not None:
        valid &= model_alpha > 0.5
    if protect is not None:
        valid &= protect < 0.5
    min_area = max(MIN_AREA_PX, int(MIN_AREA_FRAC * h * w))
    stats = region_stats(lab, mlab, labels, materials, valid, min_area)
    if cast is None:
        cast = estimate_cast(stats)
    cvec = np.array([cast["dL"], cast["da"], cast["db"]])
    rf = max(1, int(round(L / 640)))
    # per-region fields, blended by feathered masks: log chroma factor, hue rotation, log L factor
    num = np.zeros((h, w, 3), np.float32)
    lmed = np.zeros((h, w), np.float32)
    used = np.zeros((h, w), np.float32)
    regions = []
    for rec in stats:
        gain = KIND_GAIN.get(rec["kind"], 1.0)
        info = {"index": rec["index"], "name": rec["name"], "kind": rec["kind"], "px": rec["px"]}
        if "ai" not in rec:
            info["skipped"] = "too_small"
        elif gain <= 0:
            info["skipped"] = rec["kind"]
        if "skipped" in info:
            regions.append(info)
            continue
        plan = plan_shift(rec["ai"], _expected(rec["model"], rec["ai"]) + cvec, max_ab, max_desat, max_l, l_amount)
        g = gain * strength
        m = blur((labels == rec["index"]).astype(np.float32), rf, 1)
        num[..., 0] += m * (g * plan["log_c"])
        num[..., 1] += m * (g * plan["rot"])
        num[..., 2] += m * (g * plan["log_l"])
        lmed += m * max(float(rec["ai"][0]), 5.0)
        used += m
        info.update({"ai": [round(float(x), 2) for x in rec["ai"]], "model": [round(float(x), 2) for x in rec["model"]],
                     "chroma_x": round(float(np.exp(g * plan["log_c"])), 3),
                     "hue_deg": round(float(np.degrees(g * plan["rot"])), 1),
                     "light_x": round(float(np.exp(g * plan["log_l"])), 3),
                     "shift": [round(float(x) * g, 2) for x in plan["shift"]], "gain": gain})
        regions.append(info)
    allm = blur((labels >= 0).astype(np.float32), rf, 1)
    field = num / np.maximum(allm, 1e-3)[..., None]
    lmed_f = lmed / np.maximum(used, 1e-3)
    Lp = lab[..., 0]
    hl = np.clip((Lp - 1.35 * lmed_f) / (0.5 * lmed_f + 10.0), 0, 1)   # specular highlights: keep
    wpx = np.clip(weight, 0, 1) * (1.0 - 0.75 * hl)
    if protect is not None:
        wpx = wpx * (1.0 - np.clip(protect, 0, 1))
    wpx = np.where(labels >= 0, wpx, wpx * (allm > 0.5))               # never past the product
    a_, b_ = lab[..., 1], lab[..., 2]
    C = np.hypot(a_, b_)
    H = np.arctan2(b_, a_)
    C2 = C * np.exp(field[..., 0] * wpx)
    H2 = H + field[..., 1] * wpx
    out_lab = np.stack([Lp * np.exp(field[..., 2] * wpx), C2 * np.cos(H2), C2 * np.sin(H2)], -1)
    out = lab_to_srgb(out_lab)
    change = (wpx > 1e-4) & (np.abs(field).max(-1) > 1e-5)            # no Lab round-trip where nothing moves
    out = np.where(change[..., None], out, img)                        # exact identity elsewhere
    changed = int((np.abs(out - img).max(-1) > (1.5 / 255)).sum())
    return {"image": out, "regions": regions, "changed_px": changed,
            "cast": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in cast.items()}}


def measure(image: np.ndarray, model: np.ndarray, labels: np.ndarray, materials, valid: np.ndarray,
            cast=None, min_area=None) -> dict:
    """Per-material relative colour error (see ``relative_error``) of ``image`` against the model;
    ``cast`` defaults to the one estimated on this image (pass the raw image's cast to compare
    several versions on the same footing)."""
    h, w = labels.shape
    min_area = min_area or max(MIN_AREA_PX, int(MIN_AREA_FRAC * h * w))
    stats = region_stats(srgb_to_lab(image[..., :3]), srgb_to_lab(model[..., :3]), labels, materials, valid, min_area)
    cast = cast or estimate_cast(stats)
    res = {}
    for rec in stats:
        if "ai" in rec:
            res[rec["name"] or str(rec["index"])] = dict(relative_error(rec, cast), kind=rec["kind"], px=rec["px"])
    return {"cast": cast, "materials": res}
