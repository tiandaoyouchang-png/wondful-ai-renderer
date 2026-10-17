"""Product identity score (1.2.1): how faithfully one AI result keeps the Blender product.

Silhouette IoU alone cannot tell a result whose wheels or lamps were redrawn from a faithful
one. The passes the detail paste already renders give us more: the clean detail pass (the
product's real inner edges), the product mask and the material-ID labels. Four sub-scores,
each 0–100, computed on the *raw* AI image (before the paste), all on the Blender canvas:

- 构图 composition: how far the AI moved / scaled the product (outline fit vs identity).
- 轮廓 silhouette: how well the outline matches after that best fit (chamfer distance).
- 细节 detail: per-material correlation of inner edges (AI vs detail pass) after alignment;
  glass and emissive parts are skipped (their look belongs to the scene). The weakest
  parts are reported by material name, e.g. 轮毂.
- 材质色 colour: per-material colour error after removing the scene's overall colour cast
  (``material_color.measure``).

The panel shows pixel / ΔE readings at the output size (user, 2026-10-16: 用像素计量，好换算):
偏移 px, 缩放 %, 轮廓 px (mean outline distance after the fit), 细节边缘 px (symmetric mean
distance between Blender and AI inner edges per part), 颜色 ΔE. Placement comes from the inner
structure (``structure_fit``). The 0–100 sub-scores and ``overall`` stay in the JSON only.
Pure numpy; never raises from ``score_file``.
"""
from __future__ import annotations

import time

import numpy as np

from .layered_output import blur, erode, luminance, resize, fit_mask, truncated_distance

# score curves (calibrated on the 1.2.0 chair / YU7 test sets, see tests/test_121_identity_score.py)
COMP_SCALE_K = 500.0      # 1 % scale error  -> -5
COMP_SHIFT_K = 1000.0     # shift of 1 % of the product diagonal -> -10
SIL_K = 8.0               # chamfer (px per 1000 px of long edge) -> -8 each
DETAIL_LO, DETAIL_HI = 0.10, 0.70   # edge NCC mapped to 0..100
COLOUR_K = 2.0            # ΔE76 (cast removed) -> -2 each
MIN_EDGE = 0.5            # parts with less than half the product's mean inner-edge energy are too flat to judge
MIN_PART_FRAC = 0.002     # parts smaller than this share of the canvas are not scored on their own
SKIP_KINDS = {"glass", "emissive", "mirror"}


def _clip(v: float) -> float:
    return float(np.clip(v, 0.0, 100.0))


def _grad(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    return np.sqrt(gx * gx + gy * gy)


def _ncc(a: np.ndarray, b: np.ndarray, sel: np.ndarray) -> float | None:
    if sel.sum() < 40:
        return None
    x = a[sel] - a[sel].mean()
    y = b[sel] - b[sel].mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    if d < 1e-9:
        return None
    return float((x * y).sum() / d)


def _diag(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask > 0.5)
    if not len(xs):
        return 0.0
    return float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


EDGE_CAP = 12             # px: an edge further than this counts as missing (capped distance)
EDGE_FRAC = 0.15          # strongest 15 % of gradients inside a part count as its edges


def _sobel(gray: np.ndarray) -> np.ndarray:
    g = blur(gray, 1, 1)
    p = np.pad(g, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return np.hypot(gx, gy)


def _edges(mag: np.ndarray, sel: np.ndarray, frac: float = EDGE_FRAC, floor: float = 0.04) -> np.ndarray:
    if sel.sum() < 20:
        return np.zeros_like(sel)
    thr = max(float(np.quantile(mag[sel], 1.0 - frac)), floor)
    return (mag > thr) & sel


def edge_offset_px(e_ai: np.ndarray, dt_ai: np.ndarray, e_bl: np.ndarray, dt_bl: np.ndarray, sel: np.ndarray):
    """Symmetric mean edge distance (px, full resolution) inside ``sel``: Blender edges the AI
    lost or moved, and AI edges Blender does not have (redrawn structure). None if too few edges."""
    a = e_bl & sel
    b = e_ai & sel
    if a.sum() < 15 or b.sum() < 15:
        return None
    return float((dt_ai[a].mean() + dt_bl[b].mean()) / 2.0)


def _part_edge_px(mag_ai: np.ndarray, mag_bl: np.ndarray, sel: np.ndarray):
    """Edge offset (px) of one part, computed on its bounding box (plus the distance cap)."""
    ys, xs = np.nonzero(sel)
    if len(xs) < 40:
        return None
    h, w = sel.shape
    y0, y1 = max(0, ys.min() - EDGE_CAP), min(h, ys.max() + EDGE_CAP + 1)
    x0, x1 = max(0, xs.min() - EDGE_CAP), min(w, xs.max() + EDGE_CAP + 1)
    sc = sel[y0:y1, x0:x1]
    ea = _edges(mag_ai[y0:y1, x0:x1], sc)
    eb = _edges(mag_bl[y0:y1, x0:x1], sc)
    return edge_offset_px(ea, truncated_distance(ea, EDGE_CAP), eb, truncated_distance(eb, EDGE_CAP), sc)


def structure_fit(ai: np.ndarray, detail: np.ndarray, detail_alpha: np.ndarray, mask: np.ndarray,
                  scale_range=(0.88, 1.16), shift_frac=0.06) -> dict:
    """Where did the AI put the product? Coarse-to-fine search (scale about the product centre +
    shift) maximising the correlation of inner-structure edges (detail pass vs AI). The outline
    fit alone snaps onto scene clutter; wheels, lamps and seams do not. Returns full-res
    dict(scale, dx, dy, ncc, ncc_identity)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask > 0.5)
    if not len(xs):
        return {"scale": 1.0, "dx": 0.0, "dy": 0.0, "ncc": -1.0, "ncc_identity": -1.0}
    cx, cy = float(xs.mean()), float(ys.mean())
    lum_a, lum_d = luminance(ai), luminance(detail * detail_alpha[..., None])

    def level(long_edge):
        f = max(h, w) / float(long_edge)
        sw, sh = max(8, int(round(w / f))), max(8, int(round(h / f)))
        ga = _grad(blur(resize(lum_a, sw, sh), 1, 1))
        gd = _grad(blur(resize(lum_d, sw, sh), 1, 1))
        ms = resize(mask * (detail_alpha > 0.5), sw, sh)
        return f, ga, gd, ms

    from .detail_paste import warp_image

    def ncc_at(lv, s, dx, dy):
        f, ga, gd, ms = lv
        c = (cx / f, cy / f)
        g = warp_image(gd, (s, s), dx / f, dy / f, c)
        m = erode(warp_image(ms, (s, s), dx / f, dy / f, c) > 0.5, 1)
        v = _ncc(ga, g, m)
        return -1.0 if v is None else v

    lv = level(200)
    f = lv[0]
    rng = int(round(shift_frac * 200))
    best = (ncc_at(lv, 1.0, 0.0, 0.0), 1.0, 0.0, 0.0)
    ident = best[0]
    for s in np.arange(scale_range[0], scale_range[1] + 1e-6, 0.02):
        for ddx in range(-rng, rng + 1, 2):
            for ddy in range(-rng, rng + 1, 2):
                v = ncc_at(lv, float(s), ddx * f, ddy * f)
                if v > best[0]:
                    best = (v, float(s), ddx * f, ddy * f)
    for long_edge, ds, step in ((200, 0.01, 1), (400, 0.01, 1), (800, 0.005, 1)):
        lv = level(min(long_edge, max(h, w)))
        f = lv[0]
        v0, s0, x0, y0 = best
        best = (ncc_at(lv, s0, x0, y0), s0, x0, y0)
        for s in (s0 - ds, s0, s0 + ds):
            for ddx in range(-2, 3, step):
                for ddy in range(-2, 3, step):
                    v = ncc_at(lv, s, x0 + ddx * f, y0 + ddy * f)
                    if v > best[0]:
                        best = (v, s, x0 + ddx * f, y0 + ddy * f)
        if long_edge == 800 or long_edge >= max(h, w):
            ident = ncc_at(lv, 1.0, 0.0, 0.0)
    v, s, dx, dy = best
    if v < ident * 1.03:     # identity unless clearly better (same rule as the paste)
        v, s, dx, dy = ident, 1.0, 0.0, 0.0
    return {"scale": round(float(s), 4), "dx": round(float(dx), 1), "dy": round(float(dy), 1),
            "ncc": round(float(v), 3), "ncc_identity": round(float(ident), 3)}


def composition_score(fit: dict, diag: float) -> float:
    s = abs(float(fit.get("scale", 1.0)) - 1.0)
    shift = float(np.hypot(fit.get("dx", 0.0), fit.get("dy", 0.0))) / max(diag, 1.0)
    return _clip(100.0 - COMP_SCALE_K * s - COMP_SHIFT_K * shift)


def silhouette_score(fit: dict, long_edge: int) -> float | None:
    ch = fit.get("chamfer_after")
    if ch is None:
        return None
    return _clip(100.0 - SIL_K * float(ch) * 1000.0 / max(long_edge, 1))


FAKE_DENSITY = 1.5          # AI-only strong-edge px per √(part px) on a flat part → named 可疑线条
FAKE_STRONG = 0.45          # Sobel strength of a "strong" edge (≈ a 0.11 luminance step)
FAKE_FAR_PX = 4             # … an AI edge this far from every Blender structure / detail edge is AI-only
FLAT_FEATURE = 0.004        # a part with fewer feature-line pixels than this share is "flat"


def _feature_edge_px(mag_ai, F, sel):
    """1.2.3 细节边缘 vs the geometry feature lines: mean distance (px, capped) from each Blender
    structure line pixel of the part to the nearest AI edge (structure the AI lost or moved)."""
    ys, xs = np.nonzero(sel)
    if len(xs) < 40:
        return None
    h, w = sel.shape
    y0, y1 = max(0, ys.min() - EDGE_CAP), min(h, ys.max() + EDGE_CAP + 1)
    x0, x1 = max(0, xs.min() - EDGE_CAP), min(w, xs.max() + EDGE_CAP + 1)
    sc = sel[y0:y1, x0:x1]
    f = F[y0:y1, x0:x1] & sc
    if f.sum() < 15:
        return None
    ea = _edges(mag_ai[y0:y1, x0:x1], sc)
    return float(truncated_distance(ea, EDGE_CAP)[f].mean())


def _fake_line_density(mag_ai, F, mag_bl, sel):
    """AI-only line amount of one part: strong AI edge pixels farther than ``FAKE_FAR_PX`` from
    every Blender structure line and detail-pass edge, per √(part area) — about the share of a
    line crossing the whole part times its width. Flagged by name, never suppressed."""
    ys, xs = np.nonzero(sel)
    if len(xs) < 200:
        return None
    h, w = sel.shape
    y0, y1 = max(0, ys.min() - 8), min(h, ys.max() + 9)
    x0, x1 = max(0, xs.min() - 8), min(w, xs.max() + 9)
    sc = erode(sel[y0:y1, x0:x1], 2)
    if sc.sum() < 100:
        return None
    strong = (mag_ai[y0:y1, x0:x1] > FAKE_STRONG) & sc
    ref = F[y0:y1, x0:x1] | (mag_bl[y0:y1, x0:x1] > FAKE_STRONG * 0.5)
    far = truncated_distance(ref, FAKE_FAR_PX + 1) > FAKE_FAR_PX
    return float((strong & far).sum()) / float(np.sqrt(sc.sum()))


def score(ai_rgb: np.ndarray, detail_rgba: np.ndarray, product_mask: np.ndarray, matid=None,
          placement_of=None, features=None) -> dict:
    """All arrays on the same H×W canvas. ``ai_rgb``/``detail_rgba`` float 0..1, ``product_mask``
    float 0..1, ``matid`` optional (labels, materials) from ``detail_paste.load_matid``.
    ``features`` (1.2.3): Blender feature lines (bool, ``feature_lines.load``) — when given, 细节边缘
    is measured against them (independent of the pasted detail pass); the detail-pass reading is
    kept as ``edge_px_detail`` (diagnostic)."""
    from .detail_paste import verify_alignment, warp_image
    from . import material_color as MC
    h, w = product_mask.shape
    L = max(h, w)
    I = ai_rgb[..., :3].astype(np.float32)
    D = detail_rgba[..., :3].astype(np.float32)
    Da = detail_rgba[..., 3].astype(np.float32) if detail_rgba.shape[-1] == 4 else np.ones((h, w), np.float32)
    diag = _diag(product_mask)
    fit = fit_mask(I, product_mask)
    if placement_of and placement_of.get("fit"):
        # the pasted image: placement is the raw image's (the paste does not move the product)
        pf = placement_of["fit"]
        place = {"scale": pf.get("scale", 1.0), "dx": pf.get("dx", 0.0), "dy": pf.get("dy", 0.0), "ncc": 1.0}
    else:
        try:
            place = structure_fit(I, D, Da, product_mask)
        except Exception:
            place = None
    if place is not None and place["ncc"] > 0.15:
        # placement (偏移/缩放) from inner structure; the outline fit keeps the 轮廓 reading
        fit = dict(fit, scale=place["scale"], scale_x=place["scale"], scale_y=place["scale"],
                   dx=place["dx"], dy=place["dy"], placement="structure", placement_ncc=place["ncc"])
    out = {"composition": composition_score(fit, diag), "silhouette": silhouette_score(fit, L),
           "fit": {k: v for k, v in fit.items() if k != "mask"}, "parts": [], "detail": None, "colour": None}

    # align the passes onto the AI image (same rule as the paste) before the inner comparisons
    align = verify_alignment(I, D, Da, product_mask, fit)
    out["align"] = align
    sx, sy, dx, dy = align["scale_x"], align["scale_y"], align["dx"], align["dy"]
    ys, xs = np.nonzero(product_mask > 0.5)
    centre = (float(xs.mean()), float(ys.mean()))
    labels = materials = None
    if matid is not None:
        labels, materials = matid
    if (sx, sy, dx, dy) != (1.0, 1.0, 0.0, 0.0):
        Dp = warp_image(np.dstack([D * Da[..., None], Da]), (sx, sy), dx, dy, centre)
        Da = Dp[..., 3]
        D = Dp[..., :3] / np.maximum(Da, 1e-4)[..., None]
        M = warp_image(product_mask, (sx, sy), dx, dy, centre)
        if labels is not None:
            labels = MC.warp_labels(labels, (sx, sy), dx, dy, centre)
        if features is not None:
            features = warp_image(features.astype(np.float32), (sx, sy), dx, dy, centre) > 0.35
    else:
        M = product_mask
    inner = erode((M > 0.5) & (Da > 0.5), max(2, int(round(L / 400))))

    # detail: inner-edge correlation per material (or over the whole product without labels)
    # structural edges only: compare at ~640 px with a light blur, so generated micro-texture
    # (velvet, paint noise) does not count as a changed part
    f = max(1, int(round(L / 640)))
    sw, sh = max(8, w // f), max(8, h // f)
    ga = _grad(blur(luminance(resize(I, sw, sh)), 1, 1))
    gd = _grad(blur(luminance(resize(D, sw, sh)), 1, 1))
    inner_s = resize(inner.astype(np.float32), sw, sh) > 0.5
    mag_ai = _sobel(luminance(I))
    mag_bl = _sobel(luminance(D))
    parts = []
    if labels is not None and materials:
        lab_s = labels[(np.arange(sh) * h // sh)][:, (np.arange(sw) * w // sw)]
        min_px = MIN_PART_FRAC * sw * sh
        for m in materials:
            idx = int(m.get("index", 0))
            kind = str(m.get("kind") or "default")
            sel = inner_s & (lab_s == idx)
            if sel.sum() < min_px:
                continue
            energy = float(gd[sel].mean())
            rec = {"name": m.get("name") or str(idx), "kind": kind, "px": int(sel.sum()), "edge": round(energy, 4)}
            if kind not in SKIP_KINDS and energy >= MIN_EDGE * float(gd[inner_s].mean() + 1e-9):
                v = _ncc(ga, gd, sel)
                if v is not None:
                    rec["edge_ncc"] = round(v, 3)
                    rec["detail"] = round(_clip(100.0 * (v - DETAIL_LO) / (DETAIL_HI - DETAIL_LO)), 1)
                px_off = _part_edge_px(mag_ai, mag_bl, inner & (labels == idx))
                if px_off is not None:
                    rec["edge_px_detail"] = round(px_off, 2)
                    if features is None:
                        rec["edge_px"] = rec["edge_px_detail"]
            if features is not None and kind not in SKIP_KINDS:
                part_sel = inner & (labels == idx)
                fo = _feature_edge_px(mag_ai, features, part_sel)
                if fo is not None:
                    rec["edge_px"] = round(fo, 2)
                    rec["feature_px"] = int((features & part_sel).sum())
                flat = float((features & part_sel).sum()) < FLAT_FEATURE * max(int(part_sel.sum()), 1)
                if flat and kind in {"reflective", "default", "matte"}:
                    fd = _fake_line_density(mag_ai, features, mag_bl, part_sel)
                    if fd is not None:
                        rec["ai_only_lines"] = round(fd, 4)
                        if fd > FAKE_DENSITY:
                            rec["fake_lines"] = True
            parts.append(rec)
    scored = [p for p in parts if "detail" in p]
    if scored:
        # weighted by how much real structure each part carries (wheels, lamps >> smooth paint)
        wsum = sum(p["px"] * p["edge"] for p in scored)
        out["detail"] = round(sum(p["detail"] * p["px"] * p["edge"] for p in scored) / max(wsum, 1e-9), 1)
    else:
        v = _ncc(ga, gd, inner_s)
        if v is not None:
            out["detail"] = round(_clip(100.0 * (v - DETAIL_LO) / (DETAIL_HI - DETAIL_LO)), 1)

    # colour: per-material ΔE with the scene cast removed
    if labels is not None and materials:
        try:
            meas = MC.measure(I, D, labels, materials, inner)
            by_name = {p["name"]: p for p in parts}
            vals = []
            for name, rec in meas["materials"].items():
                if rec.get("kind") in SKIP_KINDS:
                    continue
                c = _clip(100.0 - COLOUR_K * rec["dE"])
                vals.append((c, rec["px"]))
                p = by_name.setdefault(name, {"name": name, "kind": rec.get("kind", ""), "px": int(rec["px"])})
                if p not in parts:
                    parts.append(p)
                p["dE"] = round(rec["dE"], 1)
                p["colour"] = round(c, 1)
            if vals:
                out["colour"] = round(sum(c * px for c, px in vals) / sum(px for _c, px in vals), 1)
        except Exception as exc:   # colour is optional; never fail the score for it
            out["colour_error"] = str(exc)[:120]

    out["parts"] = parts
    # pixel / ΔE readings (what the panel shows; the 0–100 sub-scores stay in the JSON)
    out["shift_px"] = round(float(np.hypot(fit.get("dx", 0.0), fit.get("dy", 0.0))), 1)
    out["scale_pct"] = round((float(fit.get("scale", 1.0)) - 1.0) * 100.0, 1)
    out["outline_px"] = None if fit.get("chamfer_after") is None else round(float(fit["chamfer_after"]), 2)
    out["edge_ref"] = "geometry" if features is not None else "detail"
    timed = [p for p in parts if p.get("edge_px_detail") is not None]
    if timed:
        wsum = sum(p["px"] * p["edge"] for p in timed)
        out["edge_px_detail"] = round(sum(p["edge_px_detail"] * p["px"] * p["edge"] for p in timed) / max(wsum, 1e-9), 2)
    else:
        v = _part_edge_px(mag_ai, mag_bl, inner)
        out["edge_px_detail"] = None if v is None else round(v, 2)
    if features is not None:
        fp = [p for p in parts if p.get("edge_px") is not None and p.get("feature_px")]
        if fp:
            n = sum(p["feature_px"] for p in fp)
            out["edge_px"] = round(sum(p["edge_px"] * p["feature_px"] for p in fp) / max(n, 1), 2)
        else:
            v = _feature_edge_px(mag_ai, features, inner)
            out["edge_px"] = None if v is None else round(v, 2)
        if labels is None and features is not None:
            fd = _fake_line_density(mag_ai, features, mag_bl, inner)
            if fd is not None and fd > FAKE_DENSITY:
                parts.append({"name": "产品", "kind": "default", "px": int(inner.sum()), "fake_lines": True,
                              "ai_only_lines": round(fd, 4)})
    else:
        out["edge_px"] = out["edge_px_detail"]
    out["fake_lines"] = [p["name"] for p in parts if p.get("fake_lines")]
    des = [(p["dE"], p["px"]) for p in parts if p.get("dE") is not None]
    out["colour_dE"] = round(sum(d * n for d, n in des) / sum(n for _d, n in des), 1) if des else None
    out["size"] = [int(w), int(h)]
    subs = [out[k] for k in ("composition", "silhouette", "detail", "colour") if out.get(k) is not None]
    out["overall"] = round(float(np.mean(subs)), 1) if subs else None
    for k in ("composition", "silhouette"):
        if out[k] is not None:
            out[k] = round(out[k], 1)
    return out


WEAK_EDGE_PX = 4.0         # a part whose edges sit more than 4 px off is named
WEAK_DE = 25.0             # ... or whose colour is off by more than ΔE 25


def weakest_parts(report: dict, n: int = 3) -> list:
    """The n worst parts past the pixel / ΔE limits, as 'tyre_3 4.1px' / 'wood ΔE 28'."""
    rows = []
    for p in report.get("parts") or []:
        e = p.get("edge_px")
        if e is not None and e > WEAK_EDGE_PX:
            rows.append((e / WEAK_EDGE_PX, f"{p['name']} {e:.1f}px"))
        d = p.get("dE")
        if d is not None and d > WEAK_DE:
            rows.append((d / WEAK_DE, f"{p['name']} ΔE {d:.0f}"))
        if p.get("fake_lines"):
            # 1.2.3: AI-only lines on a flat part (fake seams): named only, never auto-suppressed
            rows.append((1.0 + float(p.get("ai_only_lines") or 0), f"{p['name']} 可疑线条"))
    return [name for _v, name in sorted(rows, reverse=True)[:n]]


def score_file(image_path, detail_path, mask_path, matid_path="", placement_of=None, geometry_path="") -> dict:
    """File-level entry (passes are resampled to the image). Never raises: errors land in 'error'."""
    t0 = time.time()
    try:
        from pathlib import Path
        from .png_io import read_png
        from .detail_paste import load_matid
        rgba = read_png(image_path)
        h, w = rgba.shape[:2]
        I = rgba[..., :3].astype(np.float32) / 255.0

        def load(path):
            arr = read_png(path).astype(np.float32) / 255.0
            if arr.shape[-1] == 3:
                arr = np.dstack([arr, np.ones(arr.shape[:2], np.float32)])
            if arr.shape[:2] != (h, w):
                arr = np.clip(resize(arr, w, h), 0, 1)
            return arr

        D = load(detail_path)
        M = load(mask_path)[..., 3]
        matid = load_matid(matid_path, w, h) if matid_path and Path(matid_path).is_file() else None
        features = None
        if geometry_path and Path(geometry_path).is_file():
            from .feature_lines import load as load_features
            features = load_features(geometry_path, w, h)
        rep = score(I, D, M, matid, placement_of=placement_of, features=features)
    except Exception as exc:
        rep = {"error": str(exc)[:200]}
    rep["seconds"] = round(time.time() - t0, 2)
    return rep


def _px(v) -> str:
    return f"{v:.1f}px" if v is not None else "—"


def summarize(report, final=None) -> str:
    """One panel line in pixels at the output size, e.g.
    1672×941 · 偏移 2px · 缩放 +0.5% · 轮廓 1.8px · 细节边缘 2.1px（回贴后 1.2px）· 颜色 ΔE 6 · 待查：tyre_3 4.1px"""
    if not report or report.get("shift_px") is None:
        return ""
    parts = []
    if report.get("size"):
        parts.append(f"{report['size'][0]}×{report['size'][1]}")
    parts.append(f"偏移 {report['shift_px']:.0f}px")
    parts.append(f"缩放 {report.get('scale_pct', 0):+.1f}%")
    if report.get("outline_px") is not None:
        parts.append(f"轮廓 {_px(report['outline_px'])}")
    if report.get("edge_px") is not None:
        line = f"细节边缘 {_px(report['edge_px'])}"
        if final and final.get("edge_px") is not None:
            line += f"（回贴后 {_px(final['edge_px'])}）"
        parts.append(line)
    if report.get("colour_dE") is not None:
        line = f"颜色 ΔE {report['colour_dE']:.0f}"
        if final and final.get("colour_dE") is not None:
            line += f"（回贴后 {final['colour_dE']:.0f}）"
        parts.append(line)
    weak = weakest_parts(report)
    if weak:
        parts.append("待查：" + "、".join(weak))
    return " · ".join(parts)
