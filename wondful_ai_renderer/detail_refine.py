"""细节修复 (1.2.3): check the product's small details on the AI result and regenerate only the
ones that came out wrong.

Pure numpy + stdlib, no bpy (worker-thread and plain-Python test safe).

After the main image of a variant:

1. 细节检查 — every small detail region (``detail_paste.detail_candidates``: plates, badges /
   logos, lamps, rims / spokes, calipers, grille; same classes and size caps as 细节回贴) is
   compared with the Blender detail pass *structurally* (``structure_score``: edge NCC + SSIM
   on luminance-gradient magnitude, plus an OCR-free stroke-profile check for plates / badges),
   with a ±2 px shift tolerance. Regions that pass are left alone: no work, no cost.
2. 细节修复 — failed regions only. Nearby ones are merged into one padded crop (context margin
   ``PAD`` of the part on every side); the crop of the AI image (lighting / colour / reflections)
   and the matching crop of the Blender detail pass (structure only, product imagery only —
   the environment reference is never sent) go to the image model through ``edit_fn`` (same
   provider path as the main render), upscaled to its working size, with a prompt that fixes
   only the part. At most ``MAX_CALLS`` calls per variant within ``TIME_BUDGET`` seconds;
   cancel is checked before every call.
3. The result is downscaled back, aligned to the untouched context (edge NCC, ±``SHIFT`` px),
   colour / lightness matched to the surrounding ring in Lab and pasted with an inward-only
   feather inside the part's own mask (no pixel outside the part changes → no halo).
4. Gate — kept only if the structure score improved by ``IMPROVE`` and naturalness is not worse
   (``natural_check``: border seam energy, ring ΔE, micro-contrast / flatness, lightness shift);
   otherwise the AI region stays.

Without an image-editing provider (``edit_fn`` None: no Codex login / provider cannot edit) the
failed regions fall back to the a7055cd small-region Blender paste + its A/B gate.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from . import detail_paste as DP
from .layered_output import blur, dilate, erode, luminance, resize

STRUCT_PASS = {"keep": 0.55, "wheel": 0.50, "detail": 0.45}   # structure score at/above = 合格
FLAT_DETAIL = 0.035        # Blender part with less interior edge energy (p95) than this has no structure to check
IMPROVE = 0.05             # a fix must raise the structure score by at least this much
MIN_AFTER = 0.0            # … and end at least here (0 = only "better" is required)
MIN_FIX_PX = 120           # failed parts smaller than this carry nothing legible at output size: not sent
PAD = 0.5                  # context margin around a failed part, × its size, every side
MIN_PAD_PX = 24
MAX_CALLS = 3              # image-model calls per variant
TIME_BUDGET = 420.0        # s per variant for all fix calls together
WORK_EDGE = 1024           # long edge the crop is upscaled to for the image model
SHIFT = 3                  # px (output resolution): alignment search of the returned crop
INTERIOR = 0.20           # plates / badges: border band (× short side) left out of the check
CHECK_SHIFT = 2            # px: shift tolerance of the structure check
CONTEXT_MIN_NCC = 0.45     # the returned crop must still match the AI crop outside the parts
MAX_L_GAIN = (0.8, 1.25)   # Lab L std ratio clamp of the ring match
MAX_AB = 8.0               # Lab a/b shift clamp of the ring match
REF_BG = 0.5               # background of the Blender reference crop (neutral grey, no environment)


class Cancelled(RuntimeError):
    pass


# --------------------------------------------------------------------------- structure check

def _shift(img, dx, dy):
    """Shift by integer pixels with edge padding (no wrap)."""
    if dx == 0 and dy == 0:
        return img
    p = max(abs(dx), abs(dy))
    pad = ((p, p), (p, p)) + ((0, 0),) * (img.ndim - 2)
    big = np.pad(img, pad, mode="edge")
    h, w = img.shape[:2]
    return big[p - dy:p - dy + h, p - dx:p - dx + w]


def _gmag(rgb):
    return DP._grad(blur(luminance(rgb), 1, 1))


def _gssim(a, b, sel):
    """SSIM of two gradient-magnitude maps over ``sel``, each normalised by its own p95
    (contrast-invariant: the Blender pass is lit flat, the AI photo is not)."""
    if sel.sum() < 30:
        return 0.0
    x = np.clip(a[sel] / (np.percentile(a[sel], 95) + 1e-4), 0, 1.5)
    y = np.clip(b[sel] / (np.percentile(b[sel], 95) + 1e-4), 0, 1.5)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cxy = ((x - mx) * (y - my)).mean()
    c1, c2 = 1e-3, 3e-3
    return float(((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2)))


def _profile_ncc(a, b, R):
    """OCR-free text-structure check: column / row profiles of horizontal / vertical edge energy
    inside the part (letters and stroke rhythm), correlated. Mean of both, -1..1."""
    ys, xs = np.nonzero(R)
    if len(xs) < 30:
        return 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    out = []
    for axis in (0, 1):
        def prof(g):
            gy = np.zeros_like(g); gx = np.zeros_like(g)
            gx[:, 1:-1] = np.abs(g[:, 2:] - g[:, :-2])
            gy[1:-1] = np.abs(g[2:] - g[:-2])
            e = (gx if axis == 0 else gy)[y0:y1, x0:x1] * R[y0:y1, x0:x1]
            p = e.sum(axis=axis)
            k = np.ones(3) / 3
            return np.convolve(p, k, mode="same")
        pa, pb = prof(a), prof(b)
        if len(pa) < 6:
            continue
        pa, pb = pa - pa.mean(), pb - pb.mean()
        den = np.sqrt((pa * pa).sum() * (pb * pb).sum())
        out.append(float((pa * pb).sum() / den) if den > 1e-9 else 0.0)
    return float(np.mean(out)) if out else 0.0


def structure_score(A, D, R, kind="detail", shift=CHECK_SHIFT) -> dict:
    """How well the AI crop ``A`` reproduces the Blender detail crop ``D`` inside part mask ``R``.

    score = mean(edge NCC, gradient SSIM) (+ stroke-profile NCC for plates / badges), best over
    integer shifts of the Blender crop within ±``shift`` px. ``flat`` = the Blender part has no
    interior structure to compare (score 1, always 合格)."""
    core = erode(R, 1) if erode(R, 1).sum() >= 30 else R
    sel = dilate(R, 1)
    band = None
    if kind == "keep":
        # plates / badges are solid shapes: their outline always matches (it is the silhouette
        # the AI kept), so judge only the interior — the letters / emblem
        ys, xs = np.nonzero(R)
        b = max(3, int(round(INTERIOR * min(np.ptp(ys) + 1, np.ptp(xs) + 1)))) if len(xs) else 3
        inner = erode(R, b)
        if inner.sum() >= 30:
            band = dilate(R, 2) & ~inner
            core = sel = inner
    ga, gd0 = _gmag(A), _gmag(D)
    if (float(np.percentile(gd0[core], 95)) < FLAT_DETAIL) if core.any() else True:
        return {"score": 1.0, "ncc": 1.0, "gssim": 1.0, "text": None, "flat": True, "dx": 0, "dy": 0}
    yA, yD0 = blur(luminance(A), 1, 1), blur(luminance(D), 1, 1)
    best = None
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            gd = _shift(gd0, dx, dy)
            ncc = DP._ncc(ga, gd, sel)
            if best is None or ncc > best[0]:
                best = (ncc, dx, dy)
    ncc, dx, dy = best
    gd = _shift(gd0, dx, dy)
    gs = _gssim(ga, gd, core)
    text = energy = None
    parts = [max(ncc, -1.0), gs]
    if kind == "keep":
        text = _profile_ncc(yA, _shift(yD0, dx, dy), core)
        parts.append(text)
    score = float(np.mean(parts))
    if band is not None and band.any():
        # interior edge energy relative to the part's own outline contrast: a blank or smeared
        # plate keeps its outline but loses the letters (lighting-invariant ratio, AI vs Blender)
        ra, rd = DP._grad(luminance(A)), DP._grad(luminance(D))
        rel = lambda g: float(np.percentile(g[core], 90)) / (float(np.percentile(g[band], 90)) + 1e-4)
        rel_d = rel(rd)
        if rel_d > 0.05:
            energy = float(np.clip(rel(ra) / rel_d, 0, 1))
            score = min(score, energy)
    return {"score": round(score, 3), "ncc": round(float(ncc), 3), "gssim": round(float(gs), 3),
            "text": None if text is None else round(float(text), 3),
            "energy": None if energy is None else round(energy, 3), "flat": False, "dx": int(dx), "dy": int(dy)}


def check_regions(I, D, regions):
    """细节检查: score every live candidate (``decision`` not "drop"); sets rec["check"] and
    rec["decision"] = "pass" / "fail". Returns (passed, failed) lists."""
    passed, failed = [], []
    for rec in regions:
        if rec.get("decision") == "drop":
            continue
        sl, R = rec["_sl"], rec["_R"]
        chk = structure_score(I[sl], D[sl], R, rec["kind"])
        thr = STRUCT_PASS.get(rec["kind"], STRUCT_PASS["detail"])
        chk["threshold"] = thr
        rec["check"] = chk
        if chk["flat"] or chk["score"] >= thr:
            rec["decision"] = "pass"
            passed.append(rec)
        else:
            rec["decision"] = "fail"
            failed.append(rec)
    return passed, failed


# --------------------------------------------------------------------------- grouping / crops

def _pad_box(bbox, shape, pad=PAD):
    x0, y0, x1, y1 = bbox
    h, w = shape[:2]
    px = max(MIN_PAD_PX, int(round(pad * (x1 - x0))))
    py = max(MIN_PAD_PX, int(round(pad * (y1 - y0))))
    return [max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)]


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _priority(rec):
    return (0 if rec["kind"] == "keep" else 1, -rec["px"])


def group_regions(failed, shape, pad=PAD, max_calls=MAX_CALLS):
    """Merge failed regions whose padded boxes overlap into one crop each; most important
    groups first (plates / badges, then larger parts). Returns (groups, over_cap) where a group
    is dict(box, members) and ``over_cap`` the regions beyond ``max_calls`` crops."""
    groups = [{"box": _pad_box(r["bbox"], shape, pad), "members": [r]} for r in failed]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i]["box"], groups[j]["box"]
                if _overlap(a, b):
                    groups[i] = {"box": [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])],
                                 "members": groups[i]["members"] + groups[j]["members"]}
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    groups.sort(key=lambda g: min(_priority(m) for m in g["members"]))
    over = [m for g in groups[max_calls:] for m in g["members"]]
    return groups[:max_calls], over


def edit_size(cw, ch, long_edge=WORK_EDGE):
    """(w, h) the crop is upscaled to: the image model's working size closest to the crop aspect."""
    from .codex_direct import choose_edit_size
    tw, th = (int(v) for v in choose_edit_size(cw, ch, long_edge).split("x"))
    return tw, th


def fit_box_aspect(box, shape, aspect):
    """Grow ``box`` (x0, y0, x1, y1) to the given w/h aspect around its centre, inside the image
    (shifted inwards at the border); never shrinks."""
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bw / max(bh, 1) < aspect:
        bw = min(w, int(round(bh * aspect)))
    else:
        bh = min(h, int(round(bw / aspect)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    nx0 = int(round(min(max(0, cx - bw / 2), w - bw)))
    ny0 = int(round(min(max(0, cy - bh / 2), h - bh)))
    return [nx0, ny0, nx0 + bw, ny0 + bh]


_PART_WORDS = (
    (re.compile(r"(plate|licen[cs]e|车牌|牌照)", re.I), "车牌（文字与图案）", "铭牌（文字与图案）"),
    (re.compile(r"(logo|badge|emblem|decal|车标|标识|字标|徽标|铭牌|alum)", re.I), "车标 / 标识（文字与图案）", "Logo / 标识（文字与图案）"),
    (re.compile(r"(light|lamp|led|灯)", re.I), "车灯（灯组形状与分格）", "灯（形状与分格）"),
    (re.compile(r"(caliper|brake|卡钳|刹车)", re.I), "刹车卡钳（形状）", "部件（形状）"),
    (re.compile(r"(grille|grill|格栅)", re.I), "格栅（网格图案）", "格栅（网格图案）"),
)


def part_words(rec, vehicle=True) -> str:
    name = str(rec.get("material", ""))
    for rx, car, generic in _PART_WORDS:
        if rx.search(name):
            return car if vehicle else generic
    if rec["kind"] == "keep":
        return "车牌 / 标识（文字与图案）" if vehicle else "Logo / 铭牌（文字与图案）"
    if rec["kind"] == "wheel":
        return "轮毂（辐条的数量、形状与间距）" if vehicle else "轮状部件（辐条形状与间距）"
    return "细节部件（形状与图案）"


def fix_prompt(members, vehicle=True, names=("图 1", "图 2")) -> str:
    """Exact prompt of one crop fix: ``names[0]`` = AI crop (edit canvas), ``names[1]`` = Blender
    crop (Codex: 图 1 / 图 2; Antigravity: Edit Base / Reference 1)."""
    a, b = names
    parts = "、".join(dict.fromkeys(part_words(m, vehicle) for m in members))
    paint = "车漆颜色" if vehicle else "表面颜色"
    return (f"【局部细节修复】{a} 是一张商业摄影成片的局部放大图，是编辑画布。"
            f"{b} 是同一位置、同一视角的 Blender 产品模型渲染，只作为部件形状、文字和图案的参考；"
            "它的灯光、颜色、灰色背景和平淡质感都不要参考。\n"
            f"只修正 {a} 中的：{parts}，让它的形状、文字和图案与 {b} 一致（文字一字不改，不镜像）。\n"
            f"其他一切与 {a} 完全一致：光照、明暗、{paint}、反射与高光、景深与噪点、透视和画面里的位置都不变；"
            "这些部件以外的像素不要改动；不平移、不缩放、不裁切、不旋转、不重新构图，不添加任何东西。只输出一张图。")


# --------------------------------------------------------------------------- merge back

def _lab_ring_match(P, A, ring):
    """Lab mean / std of the fixed crop pulled onto the AI crop over the surrounding ring."""
    from .material_color import lab_to_srgb, srgb_to_lab
    if ring.sum() < 8:
        return P
    lp, la = srgb_to_lab(P), srgb_to_lab(A)
    out = lp.copy()
    mp, ma = lp[..., 0][ring].mean(), la[..., 0][ring].mean()
    k = float(np.clip((la[..., 0][ring].std() + 0.5) / (lp[..., 0][ring].std() + 0.5), *MAX_L_GAIN))
    out[..., 0] = np.clip((lp[..., 0] - mp) * k + ma, 0, 100)
    ab = np.clip(la[..., 1:][ring].mean(0) - lp[..., 1:][ring].mean(0), -MAX_AB, MAX_AB)
    out[..., 1:] = lp[..., 1:] + ab[None, None, :]
    return np.clip(lab_to_srgb(out), 0, 1).astype(np.float32)


def align_crop(P, A, context, shift=SHIFT):
    """Integer shift of the returned crop ``P`` that best matches the AI crop ``A`` on the
    untouched ``context`` (edge NCC). Returns (shifted P, dx, dy, ncc)."""
    ga = _gmag(A)
    gp0 = _gmag(P)
    best = None
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            ncc = DP._ncc(ga, _shift(gp0, dx, dy), context)
            if best is None or ncc > best[0]:
                best = (ncc, dx, dy)
    ncc, dx, dy = best
    return _shift(P, dx, dy), int(dx), int(dy), float(ncc)


LIGHT_SHIFT = 25.0         # Lab L: mean lightness of the part may move at most this far (a smeared rim → crisp spokes moved 15.6 in the 2026-10-19 real test)


def natural_check(A, C, P, R, ring) -> dict:
    """Naturalness of a fixed region ``C`` (= AI crop ``A`` with the colour-matched model crop
    ``P`` pasted inside ``R``) vs the AI original. Unlike the Blender paste, the reference for
    edges is the model's own photo: a soft / smeared AI part has almost no border energy, a
    correctly redrawn one legitimately has more, so a *seam* is extra border energy beyond what
    both the AI and the model crop have there (a visible cut), and a *step* is a ring ΔE beyond
    both. flat = micro-contrast below ``GATE_FLAT`` × the AI's; light = mean lightness moved
    more than ``LIGHT_SHIFT`` (relit / wrong exposure). Returns dict(ok, reason, metrics)."""
    border = dilate(R, 1) & ~erode(R, 1)
    inner = R & ~erode(R, 2)
    if not inner.any():
        inner = R
    la, lc, lp = DP._lab(A), DP._lab(C), DP._lab(P)
    e = lambda lab: float(DP._grad(lab[..., 0] / 100.0)[border].mean()) if border.any() else 0.0
    e_a, e_c, e_p = e(la), e(lc), e(lp)
    ring_sel = ring if ring.any() else inner

    def ring_de(lab):
        return float(np.linalg.norm(lab[inner].mean(0) - lab[ring_sel].mean(0)))
    de_a, de_c, de_p = ring_de(la), ring_de(lc), ring_de(lp)
    core = erode(R, 1) if erode(R, 1).sum() >= 8 else R
    hf = lambda y: y - blur(y, 1, 1)
    ea = float(np.abs(hf(luminance(A)))[core].mean())
    ec = float(np.abs(hf(luminance(C)))[core].mean())
    flat = (ec + 1e-4) / (ea + 1e-4)
    light = abs(float(lc[..., 0][R].mean()) - float(la[..., 0][R].mean()))
    m = {"seam_ai": round(e_a, 4), "seam_fix": round(e_c, 4), "seam_model": round(e_p, 4),
         "ring_de_ai": round(de_a, 2), "ring_de_fix": round(de_c, 2), "ring_de_model": round(de_p, 2),
         "contrast": round(flat, 3), "light_shift": round(light, 2)}
    reason = ""
    if flat < DP.GATE_FLAT:
        reason = "比原图更平"
    elif e_c > max(e_a, e_p) * DP.GATE_SEAM_RATIO + DP.GATE_SEAM_ABS:
        reason = "接缝明显"
    elif de_c > max(de_a, de_p) + DP.GATE_STEP_L:
        reason = "明暗 / 颜色不匹配"
    elif light > LIGHT_SHIFT:
        reason = "光照被改变"
    return {"ok": not reason, "reason": reason, "metrics": m}


def merge_fix(I, D, group, fixed_crop, box, L):
    """Paste one returned crop back. ``fixed_crop`` is at box size already. For each member:
    ring colour match, inward-only feather inside its own mask, structure + naturalness gate.
    Returns list of (rec, C or None, weight or None, info)."""
    x0, y0, x1, y1 = box
    A_box = I[y0:y1, x0:x1]
    parts = np.zeros(A_box.shape[:2], bool)
    g = max(3, int(round(DP.RING * L)))
    for rec in group["members"]:
        sl, R = rec["_sl"], rec["_R"]
        parts[sl[0].start - y0:sl[0].stop - y0, sl[1].start - x0:sl[1].stop - x0] |= dilate(R, g)
    context = ~parts
    if context.sum() < 50:
        context = np.ones_like(parts)
    P_box, dx, dy, ctx_ncc = align_crop(fixed_crop, A_box, context)
    results = []
    f = max(1, int(round(L / 900)))
    for rec in group["members"]:
        sl, R, ring = rec["_sl"], rec["_R"], rec["_ring"]
        info = {"align": [dx, dy], "context_ncc": round(ctx_ncc, 3)}
        if ctx_ncc < CONTEXT_MIN_NCC:
            results.append((rec, None, None, dict(info, reason="修复图对不上")))
            continue
        oy, ox = sl[0].start - y0, sl[1].start - x0
        Pc = P_box[oy:oy + R.shape[0], ox:ox + R.shape[1]]
        Ac, Dc = I[sl], D[sl]
        Pc = _lab_ring_match(Pc, Ac, ring)
        wr = np.clip((blur(R.astype(np.float32), f, 1) - 0.5) * 2.0, 0, 1) * R
        C = Ac * (1 - wr[..., None]) + Pc * wr[..., None]
        after = structure_score(C, Dc, R, rec["kind"])
        before = rec["check"]["score"]
        gate = natural_check(Ac, C, Pc, R, ring)
        info.update(after=after, metrics=gate["metrics"])
        ok, why = gate["ok"], gate["reason"]
        if after["score"] < before + IMPROVE or after["score"] < MIN_AFTER:
            ok, why = False, "结构没有改善"
        results.append((rec, C if ok else None, wr if ok else None, dict(info, reason=why)))
    return results


# --------------------------------------------------------------------------- pipeline

STOP_KINDS = ("auth", "auth_local", "quota", "rate_limit_long")


def _save(path, rgb):
    from .png_io import write_png
    write_png(str(path), (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))


def _load_rgb(path):
    from .png_io import read_png
    a = read_png(str(path))
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a[..., :3].astype(np.float32) / 255.0


def refine_details(ai_rgb, detail_rgba, product_mask, keep_rgba=None, *, strength=DP.DEFAULT_STRENGTH,
                   canvas_alpha=None, fit=None, matid=None, colour_strength=0.0, edit_fn=None, work_dir="",
                   tag="v01", vehicle=True, on_status=None, cancel_check=None, max_calls=MAX_CALLS,
                   time_budget=TIME_BUDGET) -> dict:
    """细节检查 → 细节修复 (crop regeneration) → gate, for one canvas-sized result.

    ``edit_fn(base_path, ref_path, out_path, size, prompt_fn) -> path`` regenerates one crop
    (``size`` "WxH", ``prompt_fn(names)`` the exact prompt); None = no image-edit provider →
    failed regions get the a7055cd Blender paste + A/B gate instead. Raises ``Cancelled``.
    Returns dict(image, weight, report fields: mode, regions, checked, passed, fixed, pasted,
    dropped, calls, skipped?, timings)."""
    t0 = time.time()

    def status(text):
        if on_status is not None:
            try:
                on_status(text)
            except Exception:
                pass

    def check_cancel():
        if cancel_check is not None and cancel_check():
            raise Cancelled("任务已取消")

    h, w = ai_rgb.shape[:2]
    L = max(h, w)
    I = ai_rgb[..., :3].astype(np.float32)
    res = {"image": I.copy(), "weight": np.zeros((h, w), np.float32), "mode": "check", "regions": [],
           "checked": 0, "passed": 0, "fixed": 0, "pasted": 0, "dropped": 0, "calls": [], "colour": {}}
    check_cancel()
    status("细节检查")
    ctx = DP.detail_candidates(ai_rgb, detail_rgba, product_mask, keep_rgba, fit, matid, canvas_alpha)
    if "skipped" in ctx:
        res.update(skipped=ctx["skipped"], fit=ctx["fit"], align=ctx["align"],
                   timings={"total": round(time.time() - t0, 2)})
        return res
    D, regions = ctx["D"], ctx["regions"]
    passed, failed = check_regions(I, D, regions)
    timings = {"check": round(time.time() - t0, 2)}
    res.update(fit=ctx["fit"], align=ctx["align"], checked=len(passed) + len(failed), passed=len(passed))
    out = I.copy()
    weight = res["weight"]
    fallback = []
    if failed and edit_fn is None:
        res["mode"] = "paste"
        fallback = list(failed)
    elif failed:
        res["mode"] = "refine"
        tiny = [r for r in failed if r["px"] < MIN_FIX_PX]
        for rec in tiny:     # a 16×6 px badge: regeneration cannot add anything legible (real test 2026-10-19)
            rec.update(decision="drop", reason="太小，不重新生成")
        groups, over = group_regions([r for r in failed if r["px"] >= MIN_FIX_PX], I.shape, max_calls=max_calls)
        for rec in over:
            rec.update(decision="drop", reason="超出每张调用上限")
        stop = ""
        work = Path(work_dir or ".")
        work.mkdir(parents=True, exist_ok=True)
        t_fix = time.time()
        for gi, grp in enumerate(groups):
            check_cancel()
            if stop:
                fallback += grp["members"]
                continue
            if time.time() - t_fix > time_budget:
                for rec in grp["members"]:
                    rec.update(decision="drop", reason="超出时间上限")
                continue
            status(f"细节修复 {gi + 1}/{len(groups)}")
            bx = grp["box"]
            tw, th = edit_size(bx[2] - bx[0], bx[3] - bx[1])
            box = fit_box_aspect(bx, I.shape, tw / th)
            x0, y0, x1, y1 = box
            bw, bh = x1 - x0, y1 - y0
            base_p = work / f"detail_fix_{tag}_{gi + 1}_ai.png"
            ref_p = work / f"detail_fix_{tag}_{gi + 1}_ref.png"
            out_p = work / f"detail_fix_{tag}_{gi + 1}_out.png"
            _save(base_p, resize(I[y0:y1, x0:x1], tw, th))
            Da = ctx["Da"][y0:y1, x0:x1, None]
            ref = D[y0:y1, x0:x1] * Da + REF_BG * (1 - Da)      # product only, neutral grey around
            _save(ref_p, resize(ref, tw, th))
            members = grp["members"]
            call = {"group": gi + 1, "box": box, "size": f"{tw}x{th}",
                    "regions": [i for i, r in enumerate(regions) if any(r is m for m in members)]}
            t = time.time()
            try:
                got = edit_fn(str(base_p), str(ref_p), str(out_p), f"{tw}x{th}",
                              lambda names=("图 1", "图 2"): fix_prompt(members, vehicle, names))
                call["seconds"] = round(time.time() - t, 1)
                P = _load_rgb(got or out_p)
            except Cancelled:
                raise
            except Exception as exc:
                call.update(seconds=round(time.time() - t, 1), ok=False, error=str(exc)[:200])
                res["calls"].append(call)
                kind = str(getattr(exc, "kind", "") or "")
                if kind == "cancelled" or (cancel_check is not None and cancel_check()):
                    raise Cancelled("任务已取消")
                if kind in STOP_KINDS or getattr(exc, "fatal", False):
                    stop = str(exc)[:120]
                    fallback += members
                else:
                    for rec in members:
                        rec.update(decision="drop", reason="生图失败")
                continue
            call["ok"] = True
            res["calls"].append(call)
            if P.shape[:2] != (bh, bw):
                P = np.clip(resize(P, bw, bh), 0, 1)
            for rec, C, wr, info in merge_fix(I, D, grp, P, box, L):
                rec["fix"] = info
                if C is None:
                    rec.update(decision="drop", reason=info["reason"])
                    continue
                sl, R = rec["_sl"], rec["_R"]
                out[sl] = C * R[..., None] + out[sl] * ~R[..., None]
                weight[sl] = np.maximum(weight[sl], wr)
                rec["decision"] = "fixed"
        if stop:
            res["stopped"] = stop
        timings["fix"] = round(time.time() - t_fix, 2)
    pasted = []
    seam_a = seam_c = 0.0
    for rec in fallback:
        C, wr, gate = DP.blender_paste_region(I, D, rec, strength, L)
        rec["metrics"] = gate["metrics"]
        if not gate["ok"]:
            rec.update(decision="drop", reason=gate["reason"])
            continue
        rec["decision"] = "pasted"
        pasted.append((rec, C, wr))
        seam_a += gate["metrics"]["seam_ai"] * rec["px"]
        seam_c += gate["metrics"]["seam_paste"] * rec["px"]
    if pasted and seam_c > seam_a * DP.GATE_TOTAL_SEAM + DP.GATE_SEAM_ABS * sum(r["px"] for r, _c, _w in pasted):
        for rec, _c, _w in pasted:
            rec.update(decision="drop", reason="原图更自然")
        pasted = []
    for rec, C, wr in pasted:
        sl, R = rec["_sl"], rec["_R"]
        out[sl] = C * R[..., None] + out[sl] * ~R[..., None]
        weight[sl] = np.maximum(weight[sl], wr)
    labels, materials = ctx["labels"], ctx["materials"]
    if pasted and labels is not None and colour_strength > 0:
        out, _occ, res["colour"] = DP.colour_fix_regions(out, I, D, ctx["Da"], ctx["hard"], labels, materials,
                                                         weight, ctx["excl_mat"], colour_strength, canvas_alpha)
    res["fixed"] = sum(1 for r in regions if r.get("decision") == "fixed")
    res["pasted"] = sum(1 for r in regions if r.get("decision") == "pasted")
    res["dropped"] = sum(1 for r in regions if r.get("decision") == "drop" and r.get("reason") not in ("太小", "区域过大"))
    res["regions"] = DP.strip_private(regions)
    if not (res["fixed"] or res["pasted"]):
        res["skipped"] = ("全部合格" if not failed else "原图更自然") if res["checked"] else "没有可检查的小细节"
        res["image"] = I.copy()
    else:
        res["image"] = np.clip(out, 0, 1)
    res["keep_mask"] = ctx["keep"]
    res["changed_px"] = int((np.abs(res["image"] - I).max(-1) > (1.5 / 255)).sum())
    res["timings"] = dict(timings, total=round(time.time() - t0, 2))
    return res


def refine_file(image_path, detail_path, mask_path, output_path, *, keep_path="", strength=DP.DEFAULT_STRENGTH,
                matid_path="", colour_strength=0.0, edit_fn=None, work_dir="", tag="v01", vehicle=True,
                on_status=None, cancel_check=None, max_calls=MAX_CALLS, time_budget=TIME_BUDGET) -> dict:
    """File-level 细节修复 (same inputs as ``detail_paste.paste_file``); writes ``output_path`` only
    when something changed (``skipped`` otherwise). Returns a JSON-able report."""
    from .png_io import read_png, write_png
    t0 = time.time()
    rgba = read_png(image_path)
    h, w = rgba.shape[:2]
    I = rgba[..., :3].astype(np.float32) / 255.0
    ca = rgba[..., 3].astype(np.float32) / 255.0 if rgba.shape[-1] == 4 else None

    def load(path):
        arr = read_png(path).astype(np.float32) / 255.0
        if arr.shape[-1] == 3:
            arr = np.dstack([arr, np.ones(arr.shape[:2], np.float32)])
        if arr.shape[:2] != (h, w):
            arr = np.clip(resize(arr, w, h), 0, 1)
        return arr

    D = load(detail_path)
    M = load(mask_path)[..., 3]
    K = load(keep_path) if keep_path and Path(keep_path).is_file() else None
    matid = DP.load_matid(matid_path, w, h) if matid_path else None
    res = refine_details(I, D, M, K, strength=strength, canvas_alpha=ca, matid=matid,
                         colour_strength=colour_strength if matid is not None else 0.0, edit_fn=edit_fn,
                         work_dir=work_dir or str(Path(output_path).parent), tag=tag, vehicle=vehicle,
                         on_status=on_status, cancel_check=cancel_check, max_calls=max_calls,
                         time_budget=time_budget)
    if not res.get("skipped"):
        out = (np.clip(res["image"], 0, 1) * 255 + 0.5).astype(np.uint8)
        if ca is not None:
            out = np.dstack([out, rgba[..., 3]])
        write_png(output_path, out)
    rep = {k: v for k, v in res.items() if k not in {"image", "weight", "keep_mask", "occluder"}}
    rep["seconds"] = round(time.time() - t0, 2)
    rep["output"] = "" if res.get("skipped") else str(output_path)
    try:
        Path(work_dir or Path(output_path).parent, f"detail_fix_{tag}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    except OSError:
        pass
    return rep


def summarize(report) -> str:
    """Panel line: 「细节：5 处合格 · 修复 2 处 · 放弃 0」 / 「细节：全部合格（6 处）」."""
    if not report:
        return ""
    if report.get("error"):
        return "细节修复失败：" + str(report["error"])[:80]
    checked = int(report.get("checked", 0) or 0)
    if not checked:
        why = str(report.get("skipped") or "没有可检查的小细节")
        if why == "没有可检查的小细节":
            return "细节：没有可检查的小细节"
        return "细节检查已跳过：" + why[:50] + "，保留 AI 原图"
    passed = int(report.get("passed", 0) or 0)
    if passed >= checked:
        return f"细节：全部合格（{checked} 处）"
    text = f"细节：{passed} 处合格"
    if report.get("mode") == "paste":
        text += f" · Blender 回贴 {report.get('pasted', 0)} 处 · 放弃 {report.get('dropped', 0)}（无可用生图编辑）"
    else:
        text += f" · 修复 {report.get('fixed', 0)} 处"
        if report.get("pasted"):
            text += f" · Blender 回贴 {report['pasted']} 处"
        text += f" · 放弃 {report.get('dropped', 0)}"
        calls = report.get("calls") or []
        if calls:
            text += f" · 生图 {len(calls)} 次"
        if report.get("stopped"):
            text += " · 生图已停止：" + str(report["stopped"])[:30]
    return text + f" · {float(report.get('seconds', 0) or 0):.0f} 秒"


# --------------------------------------------------------------------------- provider

def make_edit_fn(provider_id, *, model="", client_version="", cli_path="", timeout=300, log_dir="",
                 cancel_check=None, direct_edit=True, imagegen_enabled=True):
    """``edit_fn`` for ``refine_details`` on the same provider path as the main render, or None
    when no crop edit is possible here (no Codex login / direct edit off / AGY without image
    generation) — the caller then falls back to the Blender paste. Only the two crops are sent."""
    if provider_id == "codex":
        if not direct_edit:
            return None
        from . import codex_direct as CD
        try:
            if CD.find_auth_file() is None:
                return None
        except Exception:
            return None

        def codex_fn(base, ref, out, size, prompt_fn):
            r = CD.edit_image(prompt=prompt_fn(("图 1", "图 2")), edit_base_path=base, reference_paths=[ref],
                              output_path=out, size=size, model=model, client_version=client_version,
                              timeout=timeout, log_dir=log_dir or str(Path(out).parent),
                              tag=Path(out).stem[-28:], cancel_check=cancel_check, exact_prompt=True)
            return r.get("source") or out
        return codex_fn
    if provider_id == "antigravity":
        if not imagegen_enabled:
            return None
        from . import antigravity_client as AG
        try:
            if not AG.discover_antigravity_cli(cli_path):
                return None
        except Exception:
            return None

        def agy_fn(base, ref, out, size, prompt_fn):
            AG.generate_image(prompt=prompt_fn(("Edit Base", "Reference 1")), reference_paths=[ref],
                              output_path=out, cwd=log_dir or str(Path(out).parent), explicit_path=cli_path,
                              model=model, timeout=int(timeout), edit_base_path=base)
            return out
        return agy_fn
    return None
