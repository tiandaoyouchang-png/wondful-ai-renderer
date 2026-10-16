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

Overall = mean of the available sub-scores. Pure numpy; never raises from ``score_file``.
"""
from __future__ import annotations

import time

import numpy as np

from .layered_output import blur, erode, luminance, resize, fit_mask

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


def composition_score(fit: dict, diag: float) -> float:
    s = abs(float(fit.get("scale", 1.0)) - 1.0)
    shift = float(np.hypot(fit.get("dx", 0.0), fit.get("dy", 0.0))) / max(diag, 1.0)
    return _clip(100.0 - COMP_SCALE_K * s - COMP_SHIFT_K * shift)


def silhouette_score(fit: dict, long_edge: int) -> float | None:
    ch = fit.get("chamfer_after")
    if ch is None:
        return None
    return _clip(100.0 - SIL_K * float(ch) * 1000.0 / max(long_edge, 1))


def score(ai_rgb: np.ndarray, detail_rgba: np.ndarray, product_mask: np.ndarray, matid=None) -> dict:
    """All arrays on the same H×W canvas. ``ai_rgb``/``detail_rgba`` float 0..1, ``product_mask``
    float 0..1, ``matid`` optional (labels, materials) from ``detail_paste.load_matid``."""
    from .detail_paste import verify_alignment, warp_image
    from . import material_color as MC
    h, w = product_mask.shape
    L = max(h, w)
    I = ai_rgb[..., :3].astype(np.float32)
    D = detail_rgba[..., :3].astype(np.float32)
    Da = detail_rgba[..., 3].astype(np.float32) if detail_rgba.shape[-1] == 4 else np.ones((h, w), np.float32)
    diag = _diag(product_mask)
    fit = fit_mask(I, product_mask)
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
    subs = [out[k] for k in ("composition", "silhouette", "detail", "colour") if out.get(k) is not None]
    out["overall"] = round(float(np.mean(subs)), 1) if subs else None
    for k in ("composition", "silhouette"):
        if out[k] is not None:
            out[k] = round(out[k], 1)
    return out


def weakest_parts(report: dict, n: int = 2, below: float = 60.0) -> list:
    """The n weakest scored parts under ``below``, as 'name 细节' / 'name 颜色'."""
    rows = []
    for p in report.get("parts") or []:
        for key, label in (("detail", "细节"), ("colour", "颜色")):
            v = p.get(key)
            if v is not None and v < below:
                rows.append((v, f"{p['name']} {label}"))
    return [name for _v, name in sorted(rows)[:n]]


def score_file(image_path, detail_path, mask_path, matid_path="") -> dict:
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
        rep = score(I, D, M, matid)
    except Exception as exc:
        rep = {"error": str(exc)[:200]}
    rep["seconds"] = round(time.time() - t0, 2)
    return rep


def summarize(report, final=None) -> str:
    """One panel line: 识别分 88（回贴后 95）· 构图 100 · 轮廓 86 · 细节 91 · 材质色 84 · 待查：tyre_3 细节."""
    if not report or report.get("overall") is None:
        return ""
    names = (("构图", "composition"), ("轮廓", "silhouette"), ("细节", "detail"), ("材质色", "colour"))
    line = f"识别分 {report['overall']:.0f}"
    if final and final.get("overall") is not None:
        line += f"（回贴后 {final['overall']:.0f}）"
    line += "".join(f" · {label} {report[key]:.0f}" for label, key in names if report.get(key) is not None)
    weak = weakest_parts(report)
    if weak:
        line += " · 待查：" + "、".join(weak)
    return line
