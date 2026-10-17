"""细节回贴 (1.2.0): paste the product's real details back onto the AI result.

Pure numpy + stdlib, no bpy (worker-thread and plain-Python test safe).

The image model keeps the product's silhouette and position but redraws its details:
plate text, logos, wheel spokes, fabric weave. Blender knows them exactly, so after the
generation a clean *detail pass* (product only, real materials, flat even light, same
camera, same temporary pose as the clay base) is blended back:

1. ``fit_mask`` (shared with 分层输出): the product mask is fitted to the AI image with
   a small similarity transform; the detail pass and the keep-exact mask are warped
   the same way (``warp_image``).
2. Frequency separation inside a feathered, eroded product mask: low frequencies
   (lighting, reflections, colour cast, shadow) come from the AI image, high frequencies
   (texture, seams, edges, small text) from the detail pass, scaled to the AI's local
   brightness. Where the detail pass is flat (smooth paint) the AI keeps its own high
   frequencies, so sharp reflections (neon streaks, horizon lines) survive. The
   silhouette itself always stays AI pixels: no halo, no seam against the background.
3. 标识原样贴回: keep-exact regions (plates, badges, logos) are replaced outright by the
   detail pass, with light per-channel colour / brightness matching to the AI result.
4. 按材质校色 (optional, ``matid`` + ``colour_strength``): each material region from the
   material-ID pass is pulled towards the model's colour (``material_color``), keeping the
   scene's colour cast, the AI's lighting and the outer AI edge.
"""
from __future__ import annotations

import re
import time

import numpy as np

from .layered_output import blur, dilate, erode, fit_mask, guided_filter, luminance, resize

DEFAULT_STRENGTH = 0.8
HEURISTIC_MAX_FRAC = 0.25
MIN_KEEP_DENSITY = 0.35     # keep-exact blobs must fill this share of a ~L/100 window to be replaced outright   # auto-detected keep-exact parts larger than this share of the product are ignored
KEEP_PROP = "wondful_keep_exact"
KEEP_TEXT_PROP = "wondful_keep_text"
KEEP_PROMPT_VEHICLE = "车牌、车标和标识上的文字与图案保持原样，一字不改，不要重写、替换或模糊。"
KEEP_PROMPT_GENERIC = "产品上的 Logo、铭牌和标识文字与图案保持原样，一字不改，不要重写、替换或模糊。"

_KEEP_RE = re.compile(r"(plate|licen[cs]e|logo|badge|emblem|decal|车牌|牌照|车标|标识|标志|徽标|铭牌|字标|标)",
                      re.IGNORECASE)
_GLASS_RE = re.compile(r"(glass|window|windshield|windscreen|玻璃|车窗|透明)", re.IGNORECASE)
_KEEP_NOT = re.compile(r"(template|plateau|platen)", re.IGNORECASE)


# --------------------------------------------------------------------------- keep-exact names

def name_looks_keep_exact(name) -> bool:
    """Automatic heuristic: object / material names of plates, badges and logos."""
    text = str(name or "")
    if not text:
        return False
    return bool(_KEEP_RE.search(_KEEP_NOT.sub("", text)))


def name_looks_glass(name) -> bool:
    return bool(_GLASS_RE.search(str(name or "")))


_MATTE_RE = re.compile(r"(fabric|velvet|cloth|textile|linen|wool|felt|suede|leather|wood|"
                       r"布|绒|毛|麻|皮|木)", re.IGNORECASE)
_LAMP_RE = re.compile(r"(light|lamp|led|灯)", re.IGNORECASE)


def name_looks_matte(name) -> bool:
    return bool(_MATTE_RE.search(str(name or "")))


def name_looks_lamp(name) -> bool:
    return bool(_LAMP_RE.search(str(name or "")))


def prop_state(value):
    """Custom-property value -> True (marked) / False (explicitly unmarked) / None (not set)."""
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"", "0", "false", "no", "off", "否", "不"}:
            return False
        return True
    try:
        return bool(value)
    except Exception:
        return None


def classify_slot(obj_name, obj_prop, mat_name, mat_prop):
    """'explicit' | 'auto' | '' for one material slot of one object.

    An explicit mark on the object or the material wins; an explicit False on the object
    (「取消原样保留」) switches the heuristic off for that object."""
    o, m = prop_state(obj_prop), prop_state(mat_prop)
    if o is True or m is True:
        return "explicit"
    if o is False or m is False:
        return ""
    if name_looks_keep_exact(obj_name) or name_looks_keep_exact(mat_name):
        return "auto"
    return ""


def keep_prompt(vehicle: bool, texts=()) -> str:
    """Prompt sentence for products with keep-exact text (minor; the paste is what enforces it)."""
    base = KEEP_PROMPT_VEHICLE if vehicle else KEEP_PROMPT_GENERIC
    texts = [str(t).strip() for t in texts or () if str(t or "").strip()]
    if texts:
        base += "需要原样保留的文字：" + "、".join(f"「{t}」" for t in texts[:6]) + "。"
    return base


def mirrored_text_fraction(tri_pos: np.ndarray, tri_uv: np.ndarray, cam_pos, negative_scale: bool = False) -> float:
    """1.2.2: share (by screen-facing area) of a textured part whose image texture reads
    mirrored from the camera. ``tri_pos`` (N, 3, 3) world-space triangles (object transform
    applied, so a negative scale counts), ``tri_uv`` (N, 3, 2). Seen from the side where the
    vertices run counter-clockwise, a texture reads correctly iff its UV triangle is
    counter-clockwise too. Only triangles whose (outward) normal faces the camera count —
    the far plate of a car seen from the front is hidden behind the body anyway. With a
    negative object scale the world-space winding flips while Blender keeps the shading
    normal outward, so ``negative_scale`` flips which side is the outside."""
    P = np.asarray(tri_pos, np.float64)
    U = np.asarray(tri_uv, np.float64)
    if P.size == 0:
        return 0.0
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    view = ((np.asarray(cam_pos, np.float64)[None, :] - P.mean(1)) * n).sum(-1)
    e1, e2 = U[:, 1] - U[:, 0], U[:, 2] - U[:, 0]
    uv_area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    to_cam = np.asarray(cam_pos, np.float64)[None, :] - P.mean(1)
    facing = np.abs(view) / np.maximum(np.linalg.norm(to_cam, axis=1), 1e-9)   # projected area
    outward = -view if negative_scale else view
    ok = (np.abs(uv_area) > 1e-12) & (facing > 1e-12) & (outward > 0)
    if not ok.any():
        return 0.0
    mirrored = (uv_area * view < 0) & ok
    return float(facing[mirrored].sum() / facing[ok].sum())


def merge_keep_masks(rgba: np.ndarray, product: np.ndarray, max_auto_frac: float = HEURISTIC_MAX_FRAC):
    """Part masks from the keep render: R = explicit keep-exact, G = automatic keep-exact,
    B = glass / transparent, black with alpha = matte / textile; A = coverage.

    1.2.3: the decision is made per automatic region (8-connected piece), not all-or-nothing:
    a piece covering more than ``max_auto_frac`` of the product (a product that *is* a "plate",
    a mis-named body panel) is dropped on its own, real small logos / plates next to it stay.
    Returns (keep, glass, matte, info)."""
    scale = 255.0 if rgba.dtype == np.uint8 else 1.0
    a = rgba[..., 3].astype(np.float32) / scale
    rgb = rgba[..., :3].astype(np.float32) / scale
    explicit = np.clip(a * rgb[..., 0], 0, 1)
    auto = np.clip(a * rgb[..., 1], 0, 1)
    glass = np.clip(a * rgb[..., 2], 0, 1)
    matte = np.clip(a * (1.0 - rgb.max(-1)), 0, 1)
    prod_px = float((product > 0.5).sum())
    auto_px = float((auto > 0.5).sum())
    frac = auto_px / prod_px if prod_px > 0 else 0.0
    dropped_px, dropped_n, kept_n = 0, 0, 0
    if auto_px:
        from .layered_output import label_components
        labels, areas = label_components(auto > 0.5)
        drop = np.zeros(auto.shape, bool)
        for lab, area in areas.items():
            if prod_px > 0 and area > max_auto_frac * prod_px:
                drop |= labels == lab
                dropped_px += int(area)
                dropped_n += 1
            else:
                kept_n += 1
        if dropped_n:
            # the soft rim of a dropped piece goes with it
            from .layered_output import dilate as _dil
            auto = np.where(_dil(drop, 2) & ~(labels > 0) | drop, 0.0, auto)
    use_auto = kept_n > 0
    keep = np.maximum(explicit, auto) if use_auto else explicit
    if product.shape == keep.shape:
        keep = np.minimum(keep, np.clip(product, 0, 1))
    return keep.astype(np.float32), glass.astype(np.float32), matte.astype(np.float32), {
        "explicit_px": int((explicit > 0.5).sum()), "auto_px": int(auto_px), "glass_px": int((glass > 0.5).sum()),
        "matte_px": int((matte > 0.5).sum()), "auto_frac": round(frac, 4), "auto_used": bool(use_auto),
        "auto_dropped_px": dropped_px, "auto_dropped_regions": dropped_n, "auto_kept_regions": kept_n}


# --------------------------------------------------------------------------- geometry

def warp_image(img: np.ndarray, s, dx: float, dy: float, centre) -> np.ndarray:
    """``layered_output.warp_mask`` for any channel count (bilinear, edge-clamped)."""
    h, w = img.shape[:2]
    cx, cy = centre
    sx, sy = (s, s) if np.isscalar(s) else (float(s[0]), float(s[1]))
    xs = (np.arange(w, dtype=np.float32) - cx - dx) / sx + cx
    ys = (np.arange(h, dtype=np.float32) - cy - dy) / sy + cy
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    fx = (xs - x0)[None, :]; fy = (ys - y0)[:, None]
    x0c, x1c = x0.clip(0, w - 1), (x0 + 1).clip(0, w - 1)
    y0c, y1c = y0.clip(0, h - 1), (y0 + 1).clip(0, h - 1)
    src = img.astype(np.float32)
    if src.ndim == 3:
        fx = fx[..., None]; fy = fy[..., None]
    top = src[y0c][:, x0c] * (1 - fx) + src[y0c][:, x1c] * fx
    bot = src[y1c][:, x0c] * (1 - fx) + src[y1c][:, x1c] * fx
    out = top * (1 - fy) + bot * fy
    # outside the source canvas -> 0 (transparent detail, empty mask)
    inside = ((xs >= -0.5) & (xs <= w - 0.5))[None, :] & ((ys >= -0.5) & (ys <= h - 0.5))[:, None]
    return out * (inside[..., None] if out.ndim == 3 else inside)


def _grad(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1] = gray[2:] - gray[:-2]
    return np.sqrt(gx * gx + gy * gy)


def _ncc(a: np.ndarray, b: np.ndarray, sel: np.ndarray) -> float:
    if sel.sum() < 50:
        return -1.0
    x = a[sel] - a[sel].mean(); y = b[sel] - b[sel].mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))


MIN_ALIGN_NCC = 0.2       # below this the detail pass does not match the AI image: skip the paste
IDENTITY_MARGIN = 1.03    # a moved fit must explain the interior edges 3 % better than identity


def verify_alignment(ai: np.ndarray, detail: np.ndarray, detail_alpha: np.ndarray, mask: np.ndarray,
                     fit: dict) -> dict:
    """Check (and if needed correct) the alignment with the detail pass itself.

    The mask-outline fit only sees the silhouette; an outline can snap onto background
    clutter. The detail pass has the product's *inner* edges, so the transform is chosen by
    the correlation of edge maps (AI vs warped detail pass) among: identity, the outline
    fit, and a small scale / shift grid around identity, then refined by ±1 px at full size.
    Identity wins unless another candidate is clearly better. Returns dict(scale_x, scale_y,
    dx, dy, ncc, ncc_identity, ncc_outline, source, ok)."""
    h, w = mask.shape
    f = max(1, int(round(max(h, w) / 640)))
    sw, sh = max(8, w // f), max(8, h // f)
    ga = _grad(luminance(resize(ai, sw, sh)))
    dl = resize(detail * detail_alpha[..., None], sw, sh)
    da = resize(detail_alpha, sw, sh)
    gd = _grad(luminance(dl)) * (da > 0.5)
    ms = resize(mask, sw, sh)
    ys, xs = np.nonzero(ms > 0.5)
    if not len(xs):
        return {"scale_x": 1.0, "scale_y": 1.0, "dx": 0.0, "dy": 0.0, "ncc": -1.0, "ok": False, "source": "empty"}
    c = (float(xs.mean()), float(ys.mean()))

    def score(sx, sy, dx, dy):
        g = warp_image(gd, (sx, sy), dx / f, dy / f, c) if (sx, sy, dx, dy) != (1.0, 1.0, 0.0, 0.0) else gd
        m = warp_image(ms, (sx, sy), dx / f, dy / f, c) if (sx, sy, dx, dy) != (1.0, 1.0, 0.0, 0.0) else ms
        return _ncc(ga, g, erode(m > 0.5, 2))

    ident = score(1.0, 1.0, 0.0, 0.0)
    best = (ident, 1.0, 1.0, 0.0, 0.0, "identity")
    outline = None
    if fit and (fit.get("scale_x", 1.0), fit.get("scale_y", 1.0), fit.get("dx", 0.0), fit.get("dy", 0.0)) != (1.0, 1.0, 0.0, 0.0):
        outline = score(fit["scale_x"], fit["scale_y"], fit["dx"], fit["dy"])
        if outline > ident * IDENTITY_MARGIN:
            best = (outline, fit["scale_x"], fit["scale_y"], fit["dx"], fit["dy"], "outline")
    step = 2 * f
    for s in (0.98, 0.99, 1.0, 1.01, 1.02):
        for dx in range(-3 * step, 3 * step + 1, step):
            for dy in range(-3 * step, 3 * step + 1, step):
                if (s, dx, dy) == (1.0, 0, 0):
                    continue
                v = score(s, s, float(dx), float(dy))
                if v > best[0] and v > ident * IDENTITY_MARGIN:
                    best = (v, s, s, float(dx), float(dy), "grid")
    if best[5] != "identity" and f > 1:   # full-resolution ±1 px refinement (sub-sampling step)
        v0, sx, sy, dx, dy, src = best
        for ddx in range(-f, f + 1):
            for ddy in range(-f, f + 1):
                v = score(sx, sy, dx + ddx, dy + ddy)
                if v > best[0]:
                    best = (v, sx, sy, dx + ddx, dy + ddy, src)
    v, sx, sy, dx, dy, src = best
    return {"scale_x": round(float(sx), 4), "scale_y": round(float(sy), 4), "dx": round(float(dx), 1),
            "dy": round(float(dy), 1), "ncc": round(float(v), 3), "ncc_identity": round(float(ident), 3),
            "ncc_outline": None if outline is None else round(float(outline), 3), "source": src,
            "ok": bool(v >= MIN_ALIGN_NCC)}


def product_size(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask > 0.5)
    if not len(xs):
        return 0.0
    return float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def blur_radius(diag: float) -> int:
    """Frequency split radius: ~1 % of the product's bounding-box diagonal (2–24 px)."""
    return int(np.clip(round(0.01 * diag), 2, 24))


def paste_weight(mask: np.ndarray, L: int):
    """Feathered interior weight: 0 on and near the silhouette, 1 deep inside.

    Erode first, then feather inwards, so the outermost ``r_e`` pixels stay pure AI
    (no halo from the detail pass's background or a 1-px misalignment)."""
    hard = mask > 0.5
    r_e = max(2, int(round(0.003 * L)))
    r_f = max(2, int(round(0.004 * L)))
    core = erode(hard, r_e + r_f)
    w = np.clip(blur(core.astype(np.float32), r_f, 2), 0, 1)
    return np.where(erode(hard, r_e), w, 0.0).astype(np.float32), r_e, r_f


def _norm_blur(img: np.ndarray, weight: np.ndarray, r: int) -> np.ndarray:
    """Normalised convolution: blur of ``img`` using only pixels where ``weight`` > 0."""
    num = blur(img * weight[..., None], r, 2)
    den = blur(weight, r, 2)[..., None]
    return num / np.maximum(den, 1e-4)


# --------------------------------------------------------------------------- blend

def denoise_detail(detail: np.ndarray, L: int) -> np.ndarray:
    """Edge-preserving clean-up of the detail pass (guided filter, luminance guide): removes
    render grain (few samples, no denoiser) so grain is never mistaken for detail, while
    edges, seams and text with real contrast pass through."""
    g = luminance(detail)
    r = max(1, int(round(L / 640)))
    return np.clip(np.dstack([guided_filter(g, detail[..., c], r, 4e-4) for c in range(3)]), 0, 1)


def frequency_blend(ai: np.ndarray, detail: np.ndarray, detail_alpha: np.ndarray, weight: np.ndarray,
                    r: int, strength: float = DEFAULT_STRENGTH, flat=(0.012, 0.04), matte=None,
                    occluder=None, expected=None):
    """Low frequencies of ``ai`` + high frequencies of ``detail`` inside ``weight``.

    ``detail`` is brightness-normalised to the AI locally (ratio of low-pass luminance,
    clamped), so a flat-lit detail pass carries its texture into a dark night scene at
    the right contrast. Per pixel the detail's high frequencies replace the AI's only as
    far as the detail pass actually has structure there (local high-frequency energy,
    ramping from ``flat[0]`` to ``flat[1]``): on flat paint the AI keeps its own sharp
    reflections and grain-level noise in the detail pass is ignored. On ``matte`` parts
    (fabric, wood, leather, rubber: no sharp reflections) the detail pass wins everywhere,
    so texture the AI invented there (crumpled velvet, fake grain) is replaced."""
    strength = float(np.clip(strength, 0.0, 1.0))
    ai = ai.astype(np.float32)
    detail = detail.astype(np.float32)
    support = np.clip(detail_alpha, 0, 1).astype(np.float32)
    inside = (support > 0.5).astype(np.float32)
    a_low = blur(ai, r, 2)
    d_low = _norm_blur(detail, np.maximum(support, 1e-3), r)
    hf_a = ai - a_low
    hf_d = (detail - d_low) * inside[..., None]
    la, ld = luminance(a_low), luminance(d_low)
    # 1.2.2 "in shadow": the AI is much darker here than the (levels-matched) detail pass.
    if expected is not None:
        le = luminance(_norm_blur(expected.astype(np.float32), np.maximum(support, 1e-3), r))
        shade = np.clip((SHADE[0] - (la + 0.02) / (le + 0.02)) / (SHADE[0] - SHADE[1]), 0, 1)
    else:
        shade = np.zeros_like(la)
    occ = np.zeros_like(la) if occluder is None else np.clip(occluder, 0, 1).astype(np.float32)
    calm = np.maximum(shade, occ)                     # shadowed or occluded: no extra punch
    gain = np.clip((la + 0.02) / (ld + 0.02), 0.25, 3.0)
    gain = np.minimum(gain, 3.0 - 2.0 * calm)[..., None]
    hf_d = hf_d * gain
    er = max(1, r // 3)
    energy = blur(np.abs(luminance(hf_d)) * inside, er, 2) / np.maximum(blur(inside, er, 2), 1e-3)
    structure = np.clip((energy - flat[0]) / max(flat[1] - flat[0], 1e-4), 0, 1)
    if matte is not None:
        structure = np.maximum(structure, np.clip(matte, 0, 1))
    # A flat-lit pass has less micro-contrast than a photo: lift the detail's high frequencies
    # towards the AI's local contrast (never lower, at most ×2.5) so edges keep their punch.
    e_ai = blur(np.abs(luminance(hf_a)), r, 2)
    e_d = blur(np.abs(luminance(hf_d)) * inside, r, 2) / np.maximum(blur(inside, r, 2), 1e-3)
    lift = np.clip((e_ai + 1e-3) / (e_d + 1e-3), 1.0, 2.5)
    hf_d = hf_d * (1.0 + (lift - 1.0) * (1.0 - calm))[..., None]   # mud speckle energy is not "contrast"
    wd = (strength * structure * np.clip(weight, 0, 1) * inside * (1.0 - occ))[..., None]
    out = a_low + wd * hf_d + (1 - wd) * hf_a
    # Never brighten past the AI's own shading where it is shadowed / occluded: the paste only
    # adds detail there, it does not wash dirt or shadow out.
    up = np.maximum(luminance(out) - luminance(ai), 0.0)
    cap = up * calm
    out = out - (cap / np.maximum(luminance(out), 1e-4))[..., None] * out * (up > 0)[..., None]
    sel = weight > 0.5
    return np.clip(out, 0, 1), {"r": int(r), "occluded_frac": round(float((occ[sel] > 0.5).mean()) if sel.any() else 0.0, 3),
                                "shade_frac": round(float((shade[sel] > 0.5).mean()) if sel.any() else 0.0, 3),
                                "structure_mean": round(float(structure[sel].mean()) if sel.any() else 0.0, 3),
                                "wd_mean": round(float(wd[..., 0][sel].mean()) if sel.any() else 0.0, 3)}


def colour_match(src: np.ndarray, ref: np.ndarray, region: np.ndarray, amount: float = 0.85,
                 levels=(0.7, 2.5), cast=(0.6, 1.6)) -> np.ndarray:
    """Light colour / brightness / contrast match of ``src`` (detail pass) to ``ref`` (AI) over
    ``region``. Levels: the 2nd / 98th luminance percentiles of the detail pass are mapped
    onto the AI's (a flat-lit plate has grey letters, a photo of it near-black ones), scale
    clamped to ``levels``. Colour cast: per-channel ratio of the AI's chroma to the detail's,
    clamped to ``cast``. Both applied at ``amount``; the pixels' structure stays the detail's."""
    sel = region > 0.5
    if sel.sum() < 4:
        return src
    ls, lr = luminance(src)[sel], luminance(ref)[sel]
    s5, s95 = np.percentile(ls, [2, 98])
    r5, r95 = np.percentile(lr, [2, 98])
    k = float(np.clip((r95 - r5 + 1e-3) / (s95 - s5 + 1e-3), levels[0], levels[1]))
    out = (src - s95) * k + r95
    ms = src[sel].mean(0) / max(float(ls.mean()), 1e-3)
    mr = ref[sel].mean(0) / max(float(lr.mean()), 1e-3)
    g = np.clip((mr + 0.01) / (ms + 0.01), cast[0], cast[1])
    out = out * g[None, None, :]
    return np.clip(src + amount * (out - src), 0, 1)


OCC_HUE = (25.0, 50.0)       # hue departure ramp (degrees) vs the material's expected hue
OCC_DESAT = (0.40, 0.20)     # chroma ratio ramp (AI / expected) for colour that is "lost" (grey mud, spray)
OCC_DARK = (0.55, 0.35)      # lightness ratio ramp (AI / material median), lower body only
OCC_LOWER = (0.40, 0.65)     # lower-body ramp: fraction of the product's height from the top
SHADE = (0.62, 0.32)         # AI / expected low-pass luminance ratio ramp → "in shadow"


def occluder_mask(ai: np.ndarray, detail: np.ndarray, region: np.ndarray, L: int, labels=None) -> np.ndarray:
    """1.2.2: soft 0..1 map of AI pixels that sit *in front of* / *on* the product (mud, dust,
    water, particles, contact shadow) — where the paste and the colour correction fade to 0.

    Per material (material-ID labels when available, else the colour-matched detail pass)
    the expected hue / chroma / lightness is the material's typical value. A pixel is an
    occluder when its hue departs strongly (brown mud on green paint), when it lost most of
    its colour without getting brighter (grey spray / dust; a sky reflection is brighter and
    does not count), or — in the lower body / ground-contact band — when it is much darker
    than the material's typical lightness (dirt, splash shadow)."""
    from .material_color import srgb_to_lab
    sel = region > 0.5
    h, w = region.shape
    if sel.sum() < 50:
        return np.zeros((h, w), np.float32)
    li = srgb_to_lab(ai)
    Li, Ci, Hi = li[..., 0], np.hypot(li[..., 1], li[..., 2]), np.arctan2(li[..., 2], li[..., 1])
    le = srgb_to_lab(colour_match(detail, ai, region, amount=1.0))
    Ce, He = np.hypot(le[..., 1], le[..., 2]), np.arctan2(le[..., 2], le[..., 1])
    Lm = np.full((h, w), float(np.median(Li[sel])), np.float32)
    groups = []
    if labels is not None:
        groups = [sel & (labels == k) for k in np.unique(labels[sel & (labels >= 0)])]
    if not groups:
        groups = [sel]
    for m in groups:
        if m.sum() < 30:
            continue
        lm = float(np.median(Li[m]))
        Lm[m] = lm
        if labels is not None:
            mid = m & (np.abs(Li - lm) < 0.35 * lm + 5)        # typical tones of this material
            mid = mid if mid.sum() >= 20 else m
            a, b = float(np.median(li[..., 1][mid])), float(np.median(li[..., 2][mid]))
            Ce[m], He[m] = np.hypot(a, b), np.arctan2(b, a)
    dh = np.degrees(np.abs(np.angle(np.exp(1j * (Hi - He)))))
    chromatic = np.clip((np.minimum(Ci, Ce) - 10.0) / 8.0, 0, 1)
    hue = np.clip((dh - OCC_HUE[0]) / (OCC_HUE[1] - OCC_HUE[0]), 0, 1) * chromatic
    not_brighter = np.clip((1.05 * Lm + 5.0 - Li) / 6.0, 0, 1)
    desat = (np.clip((OCC_DESAT[0] - Ci / np.maximum(Ce, 1.0)) / (OCC_DESAT[0] - OCC_DESAT[1]), 0, 1)
             * np.clip((Ce - 20.0) / 10.0, 0, 1) * not_brighter)       # only colourful materials (paint)
    ratio = (Li + 2.0) / (Lm + 2.0)
    dark = np.clip((OCC_DARK[0] - ratio) / (OCC_DARK[0] - OCC_DARK[1]), 0, 1)
    # only on light / colourful materials: rims and black trim are dark by design
    dark *= np.clip((Lm - 30.0) / 10.0, 0, 1) * np.clip((Ce - 12.0) / 10.0, 0, 1)
    ys = np.nonzero(sel.any(1))[0]
    y0, y1 = int(ys.min()), int(ys.max())
    t = (np.arange(h, dtype=np.float32) - y0) / max(1.0, y1 - y0)
    lower = np.clip((t - OCC_LOWER[0]) / (OCC_LOWER[1] - OCC_LOWER[0]), 0, 1)[:, None]
    occ = np.maximum(np.maximum(hue, desat), dark * lower) * sel
    occ = blur(occ.astype(np.float32), 1, 1)                          # single noisy pixels do not count
    hard = dilate(occ > 0.4, max(1, int(round(L / 900))))
    soft = np.clip(blur(hard.astype(np.float32), max(2, int(round(L / 450))), 2) * 1.6, 0, 1)
    return (np.maximum(occ, soft) * sel).astype(np.float32)


def paste_details(ai_rgb: np.ndarray, detail_rgba: np.ndarray, product_mask: np.ndarray,
                  keep_rgba=None, strength: float = DEFAULT_STRENGTH, canvas_alpha=None, fit=None,
                  matid=None, colour_strength: float = 0.0) -> dict:
    """Paste the detail pass onto the AI image. All inputs on the same canvas (H×W).

    ``ai_rgb`` float 0..1; ``detail_rgba`` float 0..1 (straight alpha); ``product_mask``
    float 0..1; ``keep_rgba`` optional keep render (see ``merge_keep_masks``).
    ``matid``: optional (labels, materials) from the material-ID pass; with ``colour_strength``
    > 0 each material is pulled towards the model's colour after the blend (按材质校色,
    ``material_color.correct_colours``; pasted keep-exact parts are left alone).
    Returns dict(image, fit, keep, colour, timings, ...)."""
    t0 = time.time()
    h, w = ai_rgb.shape[:2]
    L = max(h, w)
    I = ai_rgb[..., :3].astype(np.float32)
    D = detail_rgba[..., :3].astype(np.float32)
    Da = detail_rgba[..., 3].astype(np.float32) if detail_rgba.shape[-1] == 4 else np.ones((h, w), np.float32)
    timings = {}
    align = None
    if fit is None:
        fit = fit_mask(I, product_mask)
        align = verify_alignment(I, D, Da, product_mask, fit)
        if not align["ok"]:
            return {"image": I.copy(), "skipped": "对齐不可靠（细节底图与成片不匹配）",
                    "fit": {k: v for k, v in fit.items() if k != "mask"}, "align": align, "keep": {},
                    "blend": {}, "changed_px": 0, "weight": np.zeros((h, w), np.float32), "keep_mask": None,
                    "timings": {"fit": round(time.time() - t0, 2), "total": round(time.time() - t0, 2)}}
        if (align["scale_x"], align["scale_y"], align["dx"], align["dy"]) != (
                fit.get("scale_x"), fit.get("scale_y"), fit.get("dx"), fit.get("dy")):
            fit = {"scale": round((align["scale_x"] + align["scale_y"]) / 2, 4), "scale_x": align["scale_x"],
                   "scale_y": align["scale_y"], "dx": align["dx"], "dy": align["dy"], "search": align["source"],
                   "outline_fit": {k: v for k, v in fit.items() if k != "mask"}}
    timings["fit"] = round(time.time() - t0, 2)
    t = time.time()
    sx, sy, dx, dy = fit.get("scale_x", 1.0), fit.get("scale_y", 1.0), fit.get("dx", 0.0), fit.get("dy", 0.0)
    identity = (sx, sy, dx, dy) == (1.0, 1.0, 0.0, 0.0)
    ys, xs = np.nonzero(product_mask > 0.5)
    centre = (float(xs.mean()), float(ys.mean())) if len(xs) else (w / 2, h / 2)
    keep = glass = matte = None
    keep_info = {}
    if keep_rgba is not None:
        keep, glass, matte, keep_info = merge_keep_masks(keep_rgba, product_mask)
    labels = materials = None
    if matid is not None and colour_strength > 0:
        labels, materials = matid
    if not identity:
        if labels is not None:
            from .material_color import warp_labels
            labels = warp_labels(labels, (sx, sy), dx, dy, centre)
        # warp premultiplied colour so the transparent background never bleeds in
        Dp = warp_image(np.dstack([D * Da[..., None], Da]), (sx, sy), dx, dy, centre)
        Da = Dp[..., 3]
        D = Dp[..., :3] / np.maximum(Da, 1e-4)[..., None]
        if keep is not None:
            keep = warp_image(keep, (sx, sy), dx, dy, centre)
            glass = warp_image(glass, (sx, sy), dx, dy, centre)
            matte = warp_image(matte, (sx, sy), dx, dy, centre)
    fitted = fit.get("mask")
    if fitted is None:
        fitted = product_mask if identity else warp_image(product_mask, (sx, sy), dx, dy, centre)
    timings["warp"] = round(time.time() - t, 2)
    t = time.time()
    weight, r_e, r_f = paste_weight(fitted, L)
    weight = weight * np.clip((Da - 0.5) * 2.0, 0, 1)   # holes in the detail pass stay AI
    if glass is not None and (glass > 0.5).any():          # glass / windows: AI reflections win
        weight = weight * (1.0 - np.clip(blur(dilate(glass > 0.5, 1).astype(np.float32), 2, 1), 0, 1))
    if canvas_alpha is not None:
        weight = weight * (canvas_alpha > 0.5)
    diag = product_size(fitted)
    r = blur_radius(diag)
    D = denoise_detail(D, L)
    occ_region = (fitted > 0.5) & (Da > 0.5)
    if canvas_alpha is not None:
        occ_region &= canvas_alpha > 0.5
    occ = occluder_mask(I, D, occ_region.astype(np.float32), L, labels=labels)
    expected = colour_match(D, I, occ_region.astype(np.float32), amount=1.0)
    blended, binfo = frequency_blend(I, D, Da, weight, r, strength, matte=matte, occluder=occ, expected=expected)
    out = I * (1 - weight[..., None]) + blended * weight[..., None]
    timings["blend"] = round(time.time() - t, 2)
    t = time.time()
    keep_px = 0
    if keep is not None and (keep > 0.5).any():
        # replace keep-exact regions outright, colour-matched to the AI result; a slightly
        # grown hard core with a 1–2 px feather, but never past the product silhouette.
        k_hard = keep > 0.5
        # Badges too small to carry legible text (a few pixels: a distant bonnet logo) are
        # left to the frequency blend: replacing them outright only swaps a dot for a dot
        # and their colour statistics are unreliable.
        rk = max(4, int(round(L / 200)))
        k_hard = k_hard & (blur(k_hard.astype(np.float32), rk, 1) >= MIN_KEEP_DENSITY)
        k_hard = dilate(k_hard, rk) & (keep > 0.5)
        grow = max(1, int(round(L / 1400)))
        k_core = dilate(k_hard, grow) & (fitted > 0.5)
        k = np.clip(blur(k_core.astype(np.float32), max(1, grow), 1), 0, 1) * (Da > 0.5)
        k = np.where(erode(fitted > 0.5, max(1, r_e // 2)), k, 0.0)
        if canvas_alpha is not None:
            k = k * (canvas_alpha > 0.5)
        k = k * (1.0 - occ)          # 1.2.2: a plate / scuff plate under mud stays muddy
        # 1.2.3: keep-exact means exact — the Blender pixels go in unchanged (no colour_match:
        # it turned a pure red logo grey-red under a warm scene cast); 按材质校色 protects them too.
        out = out * (1 - k[..., None]) + D * k[..., None]
        keep_px = int((k > 0.5).sum())
    timings["keep"] = round(time.time() - t, 2)
    colour = {}
    if labels is not None:
        t = time.time()
        from .material_color import correct_colours
        protect = None
        if keep_px:
            protect = k
        # own weight: the outer r_e px stay AI (same edge rule as the paste), then a short ramp;
        # no wide feather, so thin parts (chair legs) are corrected across their width.
        hard_f = fitted > 0.5
        cw = np.clip(blur(erode(hard_f, r_e + 1).astype(np.float32), 2, 1), 0, 1) * erode(hard_f, r_e)
        cw = cw * np.clip((Da - 0.5) * 2.0, 0, 1)
        if canvas_alpha is not None:
            cw = cw * (canvas_alpha > 0.5)
        cw = cw * (1.0 - occ)        # 1.2.2: never recolour mud / spray / dust towards clean paint
        cres = correct_colours(np.clip(out, 0, 1), D, labels, materials, cw.astype(np.float32), colour_strength,
                               model_alpha=Da, protect=protect, occluder=occ)
        out = cres["image"]
        colour = {"cast": cres["cast"], "regions": cres["regions"], "changed_px": cres["changed_px"],
                  "strength": round(float(colour_strength), 3)}
        timings["colour"] = round(time.time() - t, 2)
    out = np.clip(out, 0, 1)
    changed = np.abs(out - I).max(-1) > (1.5 / 255)
    return {"image": out, "fit": {k: v for k, v in fit.items() if k != "mask"}, "align": align,
            "keep": dict(keep_info, pasted_px=keep_px), "colour": colour,
            "blend": dict(binfo, erode=r_e, feather=r_f, strength=round(float(strength), 3), diag=round(diag, 1)),
            "changed_px": int(changed.sum()), "weight": weight, "keep_mask": keep, "occluder": occ,
            "timings": dict(timings, total=round(time.time() - t0, 2))}


# --------------------------------------------------------------------------- file driver

def raw_path(path) -> str:
    from pathlib import Path
    p = Path(path)
    return str(p.with_name(p.stem + "_raw" + (p.suffix or ".png")))


def load_matid(matid_path, w: int, h: int):
    """(labels, materials) from the material-ID pass and its JSON sidecar, at w×h; None if absent."""
    from pathlib import Path
    from .png_io import read_png
    from . import material_color as MC
    side = MC.sidecar_path(matid_path)
    if not matid_path or not Path(matid_path).is_file() or not Path(side).is_file():
        return None
    materials = MC.read_sidecar(side)
    if not materials:
        return None
    arr = read_png(matid_path)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr.astype(np.float32) / 255.0, 0, 1) * 255 + 0.5).astype(np.uint8)
    if arr.shape[-1] == 3:
        arr = np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
    labels = MC.decode_ids(arr, len(materials))
    return MC.resize_labels(labels, w, h), materials


def paste_file(image_path, detail_path, mask_path, output_path, *, keep_path="", strength=DEFAULT_STRENGTH,
               debug_dir="", matid_path="", colour_strength=0.0) -> dict:
    """File-level entry: read PNGs (any size; Blender passes are resampled to the image),
    write the pasted PNG (alpha of the input image preserved). Returns a JSON-able report."""
    from pathlib import Path
    from .png_io import read_png, write_png
    t0 = time.time()
    rgba = read_png(image_path)
    h, w = rgba.shape[:2]
    I = rgba[..., :3].astype(np.float32) / 255.0
    ca = rgba[..., 3].astype(np.float32) / 255.0 if rgba.shape[-1] == 4 else None

    def load(path, channels):
        arr = read_png(path).astype(np.float32) / 255.0
        if arr.shape[-1] == 3 and channels == 4:
            arr = np.dstack([arr, np.ones(arr.shape[:2], np.float32)])
        if arr.shape[:2] != (h, w):
            arr = np.clip(resize(arr, w, h), 0, 1)
        return arr

    D = load(detail_path, 4)
    M = load(mask_path, 4)[..., 3]
    K = load(keep_path, 4) if keep_path and Path(keep_path).is_file() else None
    matid = load_matid(matid_path, w, h) if matid_path and colour_strength > 0 else None
    res = paste_details(I, D, M, K, strength=strength, canvas_alpha=ca, matid=matid,
                        colour_strength=colour_strength if matid is not None else 0.0)
    out = (np.clip(res["image"], 0, 1) * 255 + 0.5).astype(np.uint8)
    if ca is not None:
        out = np.dstack([out, rgba[..., 3]])
    write_png(output_path, out)
    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        stem = Path(output_path).stem
        gray = lambda x: np.repeat((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)[..., None], 3, -1)
        write_png(str(Path(debug_dir) / f"{stem}_weight.png"), gray(res["weight"]))
        if res["keep_mask"] is not None:
            write_png(str(Path(debug_dir) / f"{stem}_keep.png"), gray(res["keep_mask"]))
    rep = {k: v for k, v in res.items() if k not in {"image", "weight", "keep_mask", "occluder"}}
    rep["seconds"] = round(time.time() - t0, 2)
    rep["output"] = str(output_path)
    return rep


def mirrored_note(parts) -> str:
    """Panel warning for keep-exact parts whose texture reads mirrored in Blender itself."""
    if not parts:
        return ""
    names = "、".join(str(p.get("object", ""))[:24] for p in parts[:2])
    return (f" · 注意：{names} 的贴图在模型里就是镜像的（从相机看文字是反的，UV 或缩放为负），"
            "细节回贴会照模型贴回；请在 Blender 里修正该物体的 UV / 缩放")


def summarize(report) -> str:
    if not report:
        return ""
    if report.get("error"):
        return "细节回贴失败：" + str(report["error"])[:80]
    if report.get("skipped"):
        return "细节回贴已跳过：" + str(report["skipped"])[:60] + "，保留 AI 原图"
    keep = (report.get("keep") or {}).get("pasted_px", 0)
    fit = report.get("fit") or {}
    moved = fit.get("search") not in (None, "identity")
    colour = (report.get("colour") or {}).get("regions") or []
    fixed = sum(1 for r in colour if "shift" in r and max(abs(x) for x in r["shift"]) >= 0.5)
    return ("细节回贴完成" + ("（含标识原样贴回）" if keep else "")
            + (f" · 按材质校色 {fixed} 种材质" if fixed else "")
            + (f" · 对齐 缩放 {fit.get('scale', 1):.3f}" if moved else "")
            + f" · {report.get('seconds', 0):.1f} 秒")
