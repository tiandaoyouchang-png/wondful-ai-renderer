"""1.2.8 地面分带 (PC feedback 2026-10-21: 多角度路面不一致).

The 主视角 (v01) showed a compacted dirt road, a dark stone curb in the foreground and a coarse gravel shoulder;
the 216° view turned all of it into one even fine sand — no curb, no banding. The ground was locked by words
(colour, dry) only: the blockout had no ground bands and the 主视角 projection does not reach the ground behind
the product.

Three parts, all generic (any image whose ground has parallel bands; nothing about this scene is hard-coded):

1. ``measure`` — ground bands from the 主视角 image + its measured position / normal pass: the ground pixels'
   colour / grain profile across the dominant band direction (searched, or the measured street direction when it
   agrees) is cut at sharp steps into bands (width, order, colour, grain, height step). A narrow band that is darker
   or lighter than both neighbours is a 路沿 (curb). No clear banding → no bands (everything stays as before).
2. ``build_geometry`` (bpy) — the bands as strips in the 场景粗模 (the curb raised by its measured / default step),
   along the measured direction out to what every camera sees; ``band_patches`` / ``band_fill`` — uncovered ground
   of another view takes the texture rectified from the 主视角 for *its own band* (world-space, so perspective and
   scale follow the geometry; the curb keeps its across-profile; seams blended by ``multiview.apply_fill``).
3. ``bands_text`` / ``view_sentence`` — every view's prompt describes the bands («压实土路面 + 深色石路沿 +
   粗砂石路肩，干燥») and where they run in that view, merged into the 地面锁定; ``analysis_bands_text`` gives the
   same description from the ④ measured reference for the 主视角 request itself (张数 2 → v02 gets it too).

Pure numpy except ``build_geometry`` (imports ``bpy`` lazily).
"""
from __future__ import annotations

import math
import re

import numpy as np

from .layered_output import blur, dilate

BIN_M = 0.1               # profile bin across the bands (m)
MAX_FOOT_M = 0.15         # ground pixels coarser than this (grazing / far) do not measure bands
MAX_RANGE_M = 45.0
BAND_MAX_FOOT_M = 0.09    # a band (other than the product's own) needs this median pixel footprint to count
PROD_CLEAR_M = 2.0        # ground this close to the product's footprint is skipped (its shadow is not a band)
MIN_BIN_PX = 12
EDGE_WIN = 3              # bins on each side of a candidate band edge
STEP_DE = 10.0            # colour step (ΔE76 of the medians) that makes a band edge
TEX_STEP = 0.5            # … or a grain step (relative) with at least 0.8 × STEP_DE colour step
MERGE_DE = 9.0            # neighbouring bands closer than this (and of similar grain) are one band
MAX_BANDS = 5             # more → the weakest neighbouring pair (never a curb) is merged
CHUNK_M = 1.5             # profile bins: per-chunk medians along the bands …
LIGHT_Q = 0.7             # … the chunk at this lightness quantile (a cast shadow on part of the band is skipped)
MIN_BAND_M = 0.3
CURB_MAX_M = 0.9          # a band at most this wide, darker / lighter than both neighbours = curb
CURB_TONE_L = 8.0
CURB_H = 0.12             # default curb height step when the depth does not show one (m)
CURB_H_MAX = 0.3
OPEN_EXTEND_M = 8.0       # outer bands run on this far past what the 主视角 saw (their outer edge is the frame)
MIN_GROUND_PX = 1500
ANGLE_STEP_DEG = 3
PRIOR_KEEP = 0.6          # the measured street / analysed road direction is kept when it scores ≥ this share of
                          # the best (a light gradient across the frame can out-score the true bands)
PATCH_T_M = 3.0           # rectified band patch length along the bands (m)
PATCH_S_MAX_M = 2.4       # … and at most this across
MAX_BAND_PATCHES = 4
BAND_CLS0 = 20            # class ids of band fill pixels (BAND_CLS0 + band index) for the seam colour match
GEOM_STEP_M = 2.0
GEOM_EXTENT_M = (40.0, 160.0)
GEOM_LIFT = 0.02          # flat band strips sit this far above the measured ground plane
BAND_TAG = "wondful_ground_band"


def _mv():
    from . import multiview as MV
    return MV


# ------------------------------------------------------------------------------------------- measurement

def _de(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64)))


def grain_map(img):
    """Per pixel grain: |grey − blur(grey, 2)| (gravel ≫ compacted earth / asphalt)."""
    g = np.asarray(img, np.float32)[..., :3].mean(-1)
    return np.abs(g - blur(g, 2))


def ground_selection(view):
    """(ground mask, product centre xy or None, product half extents) of a view dict (pos / nrm / mask / img)."""
    MV = _mv()
    P = np.asarray(view["pos"], np.float64)
    N = MV._unit_n(view["nrm"])
    hit = P[..., 3] > 0.5 if P.shape[-1] > 3 else np.isfinite(P[..., 2])
    mask = np.asarray(view["mask"], np.float32)
    H, W = mask.shape
    prod = mask > 0.5
    if view.get("ground") is not None:       # a measured ground class (④ precise geo) instead of the normal test
        g = hit & np.asarray(view["ground"], bool) & ~dilate(mask > 0.3, max(2, W // 160))
    else:
        g = hit & (N[..., 2] > 0.85) & ~dilate(mask > 0.3, max(2, W // 160))
    if view.get("sky") is not None:
        g &= ~np.asarray(view["sky"], bool)
    cam_o = MV._mat(view["cam"])[:3, 3] if view.get("cam") else None
    if cam_o is not None:
        rng = np.linalg.norm(P[..., :3] - cam_o, axis=-1)
        g &= rng < MAX_RANGE_M
    foot = MV.pixel_footprint(P)
    g &= foot <= MAX_FOOT_M
    centre, half = None, None
    pp = prod & hit
    if view.get("product_box") is not None:          # the product's real footprint (not only its visible side)
        lo, hi = (np.asarray(c, np.float64)[:2] for c in view["product_box"])
        centre, half = (lo + hi) / 2, (hi - lo) / 2
    elif pp.sum() >= 30:
        xy = P[..., :2][pp]
        lo, hi = np.percentile(xy, 2, 0), np.percentile(xy, 98, 0)
        centre = (lo + hi) / 2
        half = (hi - lo) / 2
    if centre is not None:
        near = np.all(np.abs(P[..., :2] - centre) <= half + float(view.get("product_clear", PROD_CLEAR_M)), -1)
        g &= ~near
    if g.sum() >= 50:      # the ground level only (roofs / plinth tops are up-facing too)
        z = P[..., 2]
        zlo = float(np.percentile(z[g], 10))
        g &= z < zlo + 0.5
    return g, centre, half


def _profile(s, labs, tex, z, tt=None, bin_m=BIN_M, chunk_m=CHUNK_M, light_q=LIGHT_Q):
    """Across-band profile: per bin the median colour / grain / height. With ``tt`` (along-band offsets) each bin
    is the median of its per-chunk medians (chunks of ``chunk_m`` along the bands), so a shadow or a stone that
    covers a minority of the band's length does not shift it."""
    lo = math.floor(float(s.min()) / bin_m) * bin_m
    nb = int(math.ceil((float(s.max()) - lo) / bin_m)) + 1
    idx = np.clip(((s - lo) / bin_m).astype(np.int64), 0, nb - 1)
    order = np.argsort(idx, kind="stable")
    idx_s = idx[order]
    cuts = np.searchsorted(idx_s, np.arange(nb + 1))
    out = {"lo": lo, "bin": bin_m, "n": np.zeros(nb, np.int64), "lab": np.full((nb, 3), np.nan),
           "tex": np.full(nb, np.nan), "z": np.full(nb, np.nan)}
    for k in range(nb):
        a, b = cuts[k], cuts[k + 1]
        out["n"][k] = b - a
        if b - a < MIN_BIN_PX:
            continue
        sel = order[a:b]
        if tt is not None:
            ch = np.floor(tt[sel] / chunk_m).astype(np.int64)
            meds, texs = [], []
            for c in np.unique(ch):
                m = sel[ch == c]
                if len(m) >= max(4, MIN_BIN_PX // 3):
                    meds.append(np.median(labs[m], 0))
                    texs.append(np.median(tex[m]))
            if len(meds) >= 2:
                # shadows only darken: the chunk at the LIGHT_Q lightness quantile stands for the bin
                meds = np.asarray(meds)
                if light_q is None:
                    out["lab"][k] = np.median(meds, 0)
                else:
                    j = int(np.argsort(meds[:, 0])[min(len(meds) - 1, int(round(light_q * (len(meds) - 1))))])
                    out["lab"][k] = meds[j]
                out["tex"][k] = float(np.median(texs))
                out["z"][k] = np.median(z[sel])
                continue
        out["lab"][k] = np.median(labs[sel], 0)
        out["tex"][k] = np.median(tex[sel])
        out["z"][k] = np.median(z[sel])
    return out


def _direction_score(xy, L, theta, bin_m=0.2):
    """Share of the lightness variance the binned profile across ``theta`` explains (η²)."""
    n = np.array([math.cos(theta), math.sin(theta)])
    s = xy @ n
    idx = ((s - s.min()) / bin_m).astype(np.int64)
    cnt = np.bincount(idx)
    tot = np.bincount(idx, L)
    ok = cnt >= MIN_BIN_PX
    if ok.sum() < 6:
        return 0.0
    mean = tot[ok] / cnt[ok]
    grand = float(L.mean())
    var_b = float((cnt[ok] * (mean - grand) ** 2).sum()) / float(cnt[ok].sum())
    var_t = float(L.var()) + 1e-6
    return var_b / var_t


def band_direction(xy, L, prior_t=None):
    """Unit direction along the bands (2D): the across-axis whose binned profile explains most of the lightness
    variance (searched every 3°, refined to 1°). ``prior_t`` (measured street direction) is kept when it scores
    ≥ PRIOR_KEEP of the best."""
    best, bt = -1.0, 0.0
    for d in range(0, 180, ANGLE_STEP_DEG):
        sc = _direction_score(xy, L, math.radians(d))
        if sc > best:
            best, bt = sc, math.radians(d)
    for dd in (-2, -1, 1, 2):
        th = bt + math.radians(dd)
        sc = _direction_score(xy, L, th)
        if sc > best:
            best, bt = sc, th
    t = np.array([-math.sin(bt), math.cos(bt)])
    if prior_t is not None:
        pt = np.asarray(prior_t, np.float64)[:2]
        if np.linalg.norm(pt) > 1e-9:
            pt = pt / np.linalg.norm(pt)
            ps = _direction_score(xy, L, math.atan2(-pt[0], pt[1]))
            if ps >= PRIOR_KEEP * best:
                return pt, ps, "street"
    return t, best, "measured"


def _segments(prof):
    """Band cuts of a profile: (list of (k0, k1) bin ranges, list of edge strengths)."""
    labv = prof["lab"]
    valid = np.isfinite(labv[:, 0])
    nb = len(valid)
    # fill short gaps (≤ 3 bins) by the nearest valid bin so a thin unmeasured stripe does not split a band
    idx = np.nonzero(valid)[0]
    if len(idx) < 2 * EDGE_WIN + 2:
        return [], []
    filled = labv.copy()
    tex = prof["tex"].copy()
    runs = []
    k = idx[0]
    start = k
    last = k
    for j in idx[1:]:
        if j - last > 4:
            runs.append((start, last + 1))
            start = j
        else:
            for m in range(last + 1, j):
                src = last if m - last <= j - m else j
                filled[m] = labv[src]
                tex[m] = prof["tex"][src]
        last = j
    runs.append((start, last + 1))
    segs, strengths = [], []
    for r0, r1 in runs:
        if r1 - r0 < 2 * EDGE_WIN:
            if r1 - r0 >= 3:
                segs.append((r0, r1))
            continue
        sc = np.zeros(r1 - r0)
        for k in range(r0 + EDGE_WIN - 1, r1 - EDGE_WIN + 1):
            if k <= r0:
                continue
            left = filled[max(r0, k - EDGE_WIN):k]
            right = filled[k:min(r1, k + EDGE_WIN)]
            if len(left) < 2 or len(right) < 2:
                continue
            de = _de(np.median(left, 0), np.median(right, 0))
            tl = float(np.median(tex[max(r0, k - EDGE_WIN):k]))
            tr = float(np.median(tex[k:min(r1, k + EDGE_WIN)]))
            tstep = abs(tl - tr) / max(tl, tr, 1e-6)
            if de >= STEP_DE or (tstep >= TEX_STEP and de >= 0.8 * STEP_DE):
                sc[k - r0] = de + 10.0 * tstep
        cuts = []
        for i in range(len(sc)):
            if sc[i] <= 0:
                continue
            lo, hi = max(0, i - 2), min(len(sc), i + 3)
            if sc[i] >= sc[lo:hi].max() and not any(abs(i - c) <= 1 for c in cuts):
                cuts.append(i)
        bounds = [r0] + [r0 + c for c in cuts] + [r1]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b > a:
                segs.append((a, b))
                strengths.append(0.0)
    return segs, strengths


def _seg_stats(prof, a, b):
    labv = prof["lab"][a:b]
    ok = np.isfinite(labv[:, 0])
    if not ok.any():
        return None
    n = prof["n"][a:b][ok].astype(np.float64)
    return {"k0": a, "k1": b, "lab": np.median(labv[ok], 0), "tex": float(np.median(prof["tex"][a:b][ok])),
            "z": float(np.median(prof["z"][a:b][ok])), "n": int(n.sum())}


def _merge(prof, segs):
    st = [x for x in (_seg_stats(prof, a, b) for a, b in segs) if x is not None]
    # contiguous only: a gap between two segments (no measurement) keeps them apart
    changed = True
    while changed and len(st) > 1:
        changed = False
        # 1) too narrow → into the closer neighbour (only when it touches)
        for i, s in enumerate(st):
            w = (s["k1"] - s["k0"]) * prof["bin"]
            if w >= MIN_BAND_M - 1e-9:
                continue
            cands = []
            if i > 0 and st[i - 1]["k1"] == s["k0"]:
                cands.append((_de(st[i - 1]["lab"], s["lab"]), i - 1))
            if i + 1 < len(st) and st[i + 1]["k0"] == s["k1"]:
                cands.append((_de(st[i + 1]["lab"], s["lab"]), i + 1))
            if not cands:
                continue
            j = min(cands)[1]
            a, b = min(st[i]["k0"], st[j]["k0"]), max(st[i]["k1"], st[j]["k1"])
            new = _seg_stats(prof, a, b)
            st[min(i, j)] = new
            del st[max(i, j)]
            changed = True
            break
        if changed:
            continue
        # 2) neighbours that look the same
        for i in range(len(st) - 1):
            a, b = st[i], st[i + 1]
            if a["k1"] != b["k0"]:
                continue
            tr = max(a["tex"], b["tex"]) / max(min(a["tex"], b["tex"]), 1e-6)
            if _de(a["lab"], b["lab"]) < MERGE_DE and tr < 1.6:
                st[i] = _seg_stats(prof, a["k0"], b["k1"])
                del st[i + 1]
                changed = True
                break
    return st


def _main_run(st, prof, prod_s, prod_r=0.0):
    """Only the contiguous run of bands that holds (or is nearest to) the product: bands seen past a gap (far
    ground, roofs) are not measured reliably."""
    if not st:
        return st
    lo, bm = prof["lo"], prof["bin"]
    runs, cur = [], [st[0]]
    for x in st[1:]:
        g0, g1 = lo + cur[-1]["k1"] * bm, lo + x["k0"] * bm
        # a gap is the product's own cleared footprint when it lies within prod_r of the product
        in_prod = prod_r > 0 and g0 >= prod_s - prod_r - bm and g1 <= prod_s + prod_r + bm
        if x["k0"] - cur[-1]["k1"] <= 4 or in_prod:
            cur.append(x)
        else:
            runs.append(cur)
            cur = [x]
    runs.append(cur)

    def dist(r):
        a, b = lo + r[0]["k0"] * bm, lo + r[-1]["k1"] * bm
        return 0.0 if a <= prod_s <= b else min(abs(a - prod_s), abs(b - prod_s))
    run = list(min(runs, key=lambda r: (dist(r), -len(r))))
    # the same surface on both sides of the product's cleared footprint is one band
    i = 0
    while i < len(run) - 1:
        a, b = run[i], run[i + 1]
        if b["k0"] > a["k1"] and _de(a["lab"], b["lab"]) < MERGE_DE:
            run[i] = _seg_stats(prof, a["k0"], b["k1"])
            del run[i + 1]
            continue
        i += 1
    return run


def _is_curbish(prof, st, i):
    x = st[i]
    if (x["k1"] - x["k0"]) * prof["bin"] > CURB_MAX_M or i == 0 or i == len(st) - 1:
        return False
    L = x["lab"][0]
    d = [L - st[i - 1]["lab"][0], L - st[i + 1]["lab"][0]]
    return all(v <= -CURB_TONE_L for v in d) or all(v >= CURB_TONE_L for v in d)


def _limit(prof, st):
    """At most MAX_BANDS: merge the most similar neighbouring pair (never a curb-like band) until it fits."""
    st = list(st)
    while len(st) > MAX_BANDS:
        best = None
        for i in range(len(st) - 1):
            if _is_curbish(prof, st, i) or _is_curbish(prof, st, i + 1):
                continue
            d = _de(st[i]["lab"], st[i + 1]["lab"])
            if best is None or d < best[0]:
                best = (d, i)
        if best is None:
            break
        i = best[1]
        st[i] = _seg_stats(prof, st[i]["k0"], st[i + 1]["k1"])
        del st[i + 1]
    return st


_MATERIALS = (  # (regex, zh, en) — first match wins
    (r"asphalt|tarmac|bitumen|沥青|柏油", "沥青", "asphalt"),
    (r"concrete|cement|水泥|混凝土", "水泥", "concrete"),
    (r"cobble|paver|paving stone|flagstone|石板|铺石|卵石", "石板", "stone paving"),
    (r"brick|砖", "砖", "brick"),
    (r"gravel|grit|pebble|crushed stone|砂石|碎石|砾", "砂石", "gravel"),
    (r"earth|dirt|soil|dust|mud|土", "土", "dirt"),
    (r"sand|沙", "沙", "sand"),
    (r"grass|lawn|turf|草", "草", "grass"),
)


_GRAIN_PREF = {"coarse": ("砂石", "沙", "土"), "fine": ("土", "沙", "砂石")}


def _material(text, default=("", ""), grain=""):
    """Material words of a surface description; with a grain the loose materials are preferred by it (a fine band
    of «dust and fine gravel» is earth, a coarse one gravel)."""
    t = (text or "").lower()
    found = [(zh, en) for rx, zh, en in _MATERIALS if re.search(rx, t)]
    if not found:
        return default
    for pref in _GRAIN_PREF.get(grain, ()):
        if found[0][0] in ("沥青", "水泥", "石板", "砖", "草"):
            break
        for zh, en in found:
            if zh == pref:
                return zh, en
    return found[0]


def _analysis_words(analysis):
    an = analysis or {}
    roads = [r for r in an.get("roads", []) or [] if isinstance(r, dict)]
    road_txt = " ".join(str(r.get("surface") or "") + " " + str(r.get("label_zh") or "") for r in roads
                        if str(r.get("kind") or "road") in ("road", "street", "lane", "path", ""))
    zones = [z for z in an.get("surface_zones", []) or [] if isinstance(z, dict)]
    zone_txt = " ".join(str(z.get("surface") or "") for z in zones)
    terr = an.get("terrain") or {}
    pal = an.get("palette") or {}
    ground_txt = " ".join((str(terr.get("surface") or ""), str(terr.get("label_zh") or ""),
                           str(pal.get("ground_words") or ""), zone_txt))
    curb_txt = " ".join(str(p.get("label_zh") or "") + " " + str(p.get("description") or "")
                        for p in an.get("primitives", []) or [] if isinstance(p, dict)
                        and re.search(r"curb|kerb|路沿|路缘|路牙", str(p.get("id") or "") + str(p.get("label_zh") or "")))
    return {"road": road_txt, "ground": ground_txt, "curb": curb_txt, "has_road": bool(roads)}


def _classify(bands, prod_s, words):
    """kind (main / curb / shoulder / strip) + zh / en words per band (in place)."""
    n = len(bands)
    main = None
    for i, b in enumerate(bands):
        if b["s0"] - 1e-9 <= prod_s <= b["s1"] + 1e-9:
            main = i
    if main is None and n:
        main = int(np.argmin([min(abs(b["s0"] - prod_s), abs(b["s1"] - prod_s)) for b in bands]))
    wide = [b for b in bands if b["width"] > CURB_MAX_M]
    texs = [b["tex"] for b in (wide or bands)]
    tmin, tmax = (min(texs), max(texs)) if texs else (0.0, 0.0)
    for i, b in enumerate(bands):
        L = float(b["lab"][0])
        nb = [bands[j] for j in (i - 1, i + 1) if 0 <= j < n and abs(bands[j]["s1" if j < i else "s0"]
                                                                       - b["s0" if j < i else "s1"]) < 1e-6]
        dl = [L - float(x["lab"][0]) for x in nb]
        tone = ""
        if dl and all(d <= -CURB_TONE_L for d in dl):
            tone = "dark"
        elif dl and all(d >= CURB_TONE_L for d in dl):
            tone = "light"
        is_curb = (b["width"] <= CURB_MAX_M and len(nb) == 2 and tone != "" and i != main)
        if not is_curb and main is not None and i != main:    # a wide band's tone is judged against the main band
            d = L - float(bands[main]["lab"][0])
            tone = "dark" if d <= -12.0 else ("light" if d >= 12.0 else "")
        elif i == main:
            tone = ""
        b["tone"] = tone
        if is_curb:
            b["kind"] = "curb"
        elif i == main:
            b["kind"] = "main"
        elif b["width"] <= CURB_MAX_M:
            b["kind"] = "strip"
        else:
            b["kind"] = "shoulder"
        coarse = len(texs) > 1 and b["tex"] >= 1.25 * tmin and b["tex"] >= tmax - 1e-9 and tmax > tmin * 1.25
        fine = len(texs) > 1 and b["tex"] <= tmin + 1e-9 and tmax > tmin * 1.25
        b["grain"] = "coarse" if coarse else ("fine" if fine else "")
    road_like = words.get("has_road")
    for b in bands:
        tone_zh = {"dark": "深色", "light": "浅色"}.get(b["tone"], "")
        tone_en = {"dark": "dark ", "light": "light "}.get(b["tone"], "")
        if b["kind"] == "curb":
            m_zh, m_en = _material(words.get("curb"), ("石", "stone"))
            if m_zh in ("土", "沙", "草", "砂石"):
                m_zh, m_en = "石", "stone"
            b["zh"] = f"{tone_zh}{m_zh}路沿"
            b["en"] = f"{tone_en}{m_en} curb"
            continue
        src = words.get("road") if (b["kind"] == "main" and road_like) else words.get("ground")
        m_zh, m_en = _material(src, _material(words.get("ground"), ("", ""), b["grain"]), b["grain"])
        if b["grain"] == "coarse" and m_zh in ("土", "沙"):
            m_zh, m_en = "砂石", "gravel"
        q_zh, q_en = "", ""
        if b["grain"] == "coarse":
            q_zh, q_en = "粗", "coarse "
        elif b["grain"] == "fine" and m_zh in ("土", "沙", "砂石", ""):
            q_zh, q_en = "压实", "compacted "
        if b["kind"] == "main":
            noun_zh, noun_en = ("路面", "road surface") if road_like else ("地面", "ground")
        elif b["kind"] == "strip":
            noun_zh, noun_en = "条带", "strip"
        else:
            noun_zh, noun_en = ("路肩", "shoulder") if road_like else ("地面", "ground")
        if not m_zh:
            m_zh, m_en = ("", "")
        b["zh"] = f"{tone_zh if b['kind'] != 'main' else ''}{q_zh}{m_zh}{noun_zh}"
        b["en"] = f"{tone_en if b['kind'] != 'main' else ''}{q_en}{m_en + ' ' if m_en else ''}{noun_en}"


def measure(view, prior_t=None, analysis=None, source="master"):
    """Ground bands of one view (the 主视角, or the ④ reference with its measured depth). → dict, or {} when the
    ground shows no clear banding (then nothing changes downstream)."""
    MV = _mv()
    if view.get("img") is None:
        return {}
    P = np.asarray(view["pos"], np.float64)
    g, centre, half = ground_selection(view)
    if int(g.sum()) < MIN_GROUND_PX:
        return {}
    img = np.asarray(view["img"], np.float32)[..., :3]
    labs = MV.lab(img[g])
    tex = grain_map(img)[g]
    xyz = P[..., :3][g]
    if centre is None:
        cam_o = MV._mat(view["cam"])[:3, 3]
        fw = MV.forward(view["cam"])
        f2 = fw[:2] / max(np.linalg.norm(fw[:2]), 1e-9)
        centre = cam_o[:2] + f2 * 10.0
    a0 = np.asarray(centre, np.float64)
    xy = xyz[:, :2] - a0
    t, score, how = band_direction(xy, labs[:, 0], prior_t)
    if score < 0.05:
        return {}
    n = np.array([t[1], -t[0]])
    s = xy @ n
    tt = xy @ t
    prof = _profile(s, labs, tex, xyz[:, 2], tt=tt, light_q=view.get("light_q", LIGHT_Q))
    segs, _st = _segments(prof)
    st = _merge(prof, segs)
    prod_r = 0.0
    if half is not None:
        prod_r = float(np.linalg.norm(half)) + float(view.get("product_clear", PROD_CLEAR_M))
    st = _main_run(st, prof, 0.0, prod_r)
    st = _limit(prof, st)
    if len(st) < 2:
        return {}
    lo, bm = prof["lo"], prof["bin"]
    foot = MV.pixel_footprint(P)[g]
    bands = []
    for x in st:
        s0, s1 = lo + x["k0"] * bm, lo + x["k1"] * bm
        sel = (s >= s0) & (s < s1)
        if sel.sum() < MIN_BIN_PX:
            continue
        if float(np.median(foot[sel])) > BAND_MAX_FOOT_M and not (s0 - 1e-6 <= 0.0 <= s1 + 1e-6):
            continue          # seen only at grazing angles (far / behind the product): not measured reliably
        rgb = np.median(img[g][sel], 0)
        bands.append({"s0": round(float(s0), 3), "s1": round(float(s1), 3), "width": round(float(s1 - s0), 3),
                      "lab": [round(float(c), 2) for c in x["lab"]], "rgb": [int(round(float(c))) for c in rgb],
                      "hex": "#%02X%02X%02X" % tuple(int(round(float(c))) for c in rgb),
                      "tex": round(float(x["tex"]), 2), "z": round(float(x["z"]), 3), "px": int(sel.sum()),
                      "t_span": [round(float(np.percentile(tt[sel], 2)), 2), round(float(np.percentile(tt[sel], 98)), 2)]})
    # dropped bands leave a hole: keep only the chain around the product's band
    if bands:
        k = min(range(len(bands)), key=lambda i: 0.0 if bands[i]["s0"] <= 0.0 <= bands[i]["s1"]
                else min(abs(bands[i]["s0"]), abs(bands[i]["s1"])))
        a = k
        while a > 0 and bands[a]["s0"] - bands[a - 1]["s1"] <= 0.5:
            a -= 1
        b = k
        while b + 1 < len(bands) and bands[b + 1]["s0"] - bands[b]["s1"] <= 0.5:
            b += 1
        bands = bands[a:b + 1]
    if len(bands) < 2:
        return {}
    # bands touching on both sides keep contiguous edges; the outermost edges are where the view stopped seeing
    bands[0]["open_lo"] = True
    bands[-1]["open_hi"] = True
    for i in range(len(bands) - 1):
        if abs(bands[i]["s1"] - bands[i + 1]["s0"]) > 0.5:
            bands[i]["open_hi"] = True
            bands[i + 1]["open_lo"] = True
    prod_s = 0.0
    _classify(bands, prod_s, _analysis_words(analysis))
    # ground plane z = c0 + ct·t + cs·s (least squares on the measured ground), gentle slopes only
    A = np.stack([np.ones_like(s), tt, s], -1)
    try:
        coef = np.linalg.lstsq(A, xyz[:, 2], rcond=None)[0]
    except np.linalg.LinAlgError:
        coef = np.array([float(np.median(xyz[:, 2])), 0.0, 0.0])
    coef[1:] = np.clip(coef[1:], -0.15, 0.15)
    for i, b in enumerate(bands):
        b["height"] = 0.0
        if b["kind"] == "curb":
            nbz = [bands[j]["z"] for j in (i - 1, i + 1) if 0 <= j < len(bands)]
            dz = b["z"] - (float(np.median(nbz)) if nbz else b["z"])
            b["height"] = round(float(min(CURB_H_MAX, dz)) if dz > 0.03 else CURB_H, 3)
            b["height_measured"] = bool(dz > 0.03)
    if not any(b["kind"] in ("curb", "shoulder", "strip") for b in bands):
        return {}
    cam_o = MV._mat(view["cam"])[:3, 3] if view.get("cam") else None
    near_side = None
    if cam_o is not None:
        near_side = "lo" if float((cam_o[:2] - a0) @ n) < 0 else "hi"
    return {"t": [round(float(c), 6) for c in t], "n": [round(float(c), 6) for c in n],
            "a0": [round(float(c), 4) for c in a0], "plane": [round(float(c), 5) for c in coef],
            "bands": bands, "score": round(float(score), 3), "direction": how, "source": source,
            "near_side": near_side, "t_span": [round(float(np.percentile(tt, 1)), 2),
                                               round(float(np.percentile(tt, 99)), 2)]}


def band_index(info, s):
    """Band index per across-offset ``s`` (−1 outside). Outer (open) edges run on OPEN_EXTEND_M."""
    s = np.asarray(s, np.float64)
    out = np.full(s.shape, -1, np.int64)
    for i, b in enumerate(info.get("bands") or []):
        lo = b["s0"] - (OPEN_EXTEND_M if b.get("open_lo") else 0.0)
        hi = b["s1"] + (OPEN_EXTEND_M if b.get("open_hi") else 0.0)
        out[(s >= lo) & (s < hi) & (out < 0)] = i
    return out


def frame(info, xy):
    """(along t, across s) of world xy points in the band frame."""
    a0 = np.asarray(info["a0"], np.float64)
    t = np.asarray(info["t"], np.float64)
    n = np.asarray(info["n"], np.float64)
    d = np.asarray(xy, np.float64)[..., :2] - a0
    return d @ t, d @ n


def plane_z(info, tt, s):
    c = info.get("plane") or [0.0, 0.0, 0.0]
    return c[0] + c[1] * np.asarray(tt) + c[2] * np.asarray(s)


# ------------------------------------------------------------------------------------------- texture fill

def band_patches(mv, info, mpp=None):
    """Per band: texture patches rectified from the 主视角 on a grid along the bands (u = along, v = across), at
    the band's own texel size (≥ the 主视角's pixel footprint there, so the patch is not a depth-direction smear)."""
    MV = _mv()
    base_mpp = mpp or MV.FILL_MPP_GROUND
    t3 = np.r_[np.asarray(info["t"], np.float64), 0.0]
    n3 = np.r_[np.asarray(info["n"], np.float64), 0.0]
    a0 = np.asarray(info["a0"], np.float64)
    P = np.asarray(mv["pos"], np.float64)
    hit = P[..., 3] > 0.5
    foot = MV.pixel_footprint(P)
    tt_m, s_m = frame(info, P[..., :2])
    out = []
    for b in info.get("bands") or []:
        narrow = b["width"] <= CURB_MAX_M
        if narrow:
            sa, sb = b["s0"], b["s1"]
        else:
            m = min(0.15, b["width"] * 0.1)
            sa, sb = b["s0"] + m, b["s1"] - m
            if sb - sa > PATCH_S_MAX_M:          # the stretch of the band nearest the camera (finest pixels)
                if info.get("near_side", "lo") == "hi":
                    sa = sb - PATCH_S_MAX_M
                else:
                    sb = sa + PATCH_S_MAX_M
        sel = hit & (s_m >= sa) & (s_m < sb) & (np.asarray(mv["mask"]) < 0.5)
        fb = float(np.median(foot[sel])) if sel.sum() > 20 else base_mpp
        bm = float(np.clip(fb, base_mpp, 3 * base_mpp)) if not narrow else base_mpp
        nv = max(4, int(round((sb - sa) / bm)))
        nu = int(round(PATCH_T_M / bm))
        t0, t1 = b.get("t_span", [-10, 10])
        got = []
        if t1 - t0 >= PATCH_T_M * 0.6:
            centres = np.linspace(t0 + PATCH_T_M / 2, max(t0 + PATCH_T_M / 2, t1 - PATCH_T_M / 2), 9)
            cands = []
            for tc in centres:
                o2 = a0 + t3[:2] * (tc - PATCH_T_M / 2) + n3[:2] * sa
                # the 主视角's geometry is the flat blockout ground (no curb step): rectify on the band plane
                z = float(plane_z(info, tc, 0.5 * (sa + sb)))
                col, ok = MV._rectify(mv, np.r_[o2, z], t3, n3, nu, nv, bm, 0.3)
                cands.append((float(ok.mean()), tc, col, ok))
            cands.sort(key=lambda c: -c[0])
            used = []
            for frac, tc, col, ok in cands:
                if frac < 0.6 or any(abs(tc - u) < PATCH_T_M * 0.8 for u in used):
                    continue
                C = MV._crop_patch(col, ok, min_px=max(6, nv // 2) if not narrow else 12, need=0.6,
                                   rows_fixed=narrow, min_valid=0.7)
                if C is not None and min(C.shape[:2]) >= (4 if narrow else 12):
                    got.append(C)
                    used.append(tc)
                if len(got) >= MAX_BAND_PATCHES:
                    break
        if not got:   # nothing rectifiable: a flat patch of the band's measured colour with fine noise
            rs = np.random.default_rng(17)
            nz = blur(rs.normal(0, 1, (max(nv, 12), 64)).astype(np.float32), 1, passes=1)
            got = [np.clip(np.asarray(b["rgb"] if "rgb" in b else (128, 118, 104), np.float32)[None, None]
                           * (1 + 0.08 * nz[..., None]), 0, 255)]
        out.append({"narrow": narrow, "patches": got, "s0": sa, "s1": sb, "mpp": bm})
    return out


def band_synths(patch_sets):
    MV = _mv()
    out = []
    for ps in patch_sets:
        if ps["narrow"] or min(min(p.shape[:2]) for p in ps["patches"]) < 12:
            S = MV.make_synth([p for p in ps["patches"] if min(p.shape[:2]) >= 4] or ps["patches"],
                              MV.FILL_MPP_GROUND, rotate=False, flatten=False) \
                if min(min(p.shape[:2]) for p in ps["patches"]) >= 12 else _strip_synth(ps["patches"])
            out.append({"mode": "strip", "S": S, **{k: ps[k] for k in ("s0", "s1", "mpp") if k in ps}})
        else:
            out.append({"mode": "area", "S": MV.make_synth(ps["patches"], ps.get("mpp", MV.FILL_MPP_GROUND),
                                                           rotate=False),
                        **{k: ps[k] for k in ("s0", "s1", "mpp") if k in ps}})
    return out


def _strip_synth(patches):
    """Very narrow band (< 12 texels across): repeat the rows (make_synth needs 12 px) — still a strip texture."""
    MV = _mv()
    ps = []
    for p in patches:
        p = np.asarray(p, np.float32)
        rep = int(math.ceil(12 / max(1, p.shape[0])))
        ps.append(np.repeat(p, rep, axis=0))
    S = MV.make_synth(ps, MV.FILL_MPP_GROUND, rotate=False, flatten=False)
    if S is not None:
        S["row_scale"] = float(ps[0].shape[0]) / float(np.asarray(patches[0]).shape[0])
    return S


def ground_level(vk, info, extra=0.45):
    """Pixels of view k on the banded ground: hit, not product, at most ``extra`` m above the band plane (curb
    faces included), up-facing or within the curb height."""
    MV = _mv()
    P = np.asarray(vk["pos"], np.float64)
    hit = P[..., 3] > 0.5
    N = MV._unit_n(vk["nrm"])
    tt, s = frame(info, P[..., :2])
    dz = P[..., 2] - plane_z(info, tt, s)
    hmax = max([b.get("height", 0.0) for b in info.get("bands") or []] + [0.0])
    gl = hit & (np.asarray(vk["mask"]) < 0.5) & (dz > -0.6) & (
        ((N[..., 2] > 0.7) & (dz < extra)) | (dz < hmax + 0.06))
    return gl, tt, s, dz


def band_fill(vk, info, synths, fill=None, have=None, cls=None):
    """Uncovered-ground texture of view k from the band synths. Returns (fill, have, cls, band index map); band
    pixels override the generic road / sidewalk fill (they carry the measured banding), other pixels are kept."""
    MV = _mv()
    P = np.asarray(vk["pos"], np.float64)
    H, W = P.shape[:2]
    fill = np.zeros((H, W, 3), np.float32) if fill is None else np.asarray(fill, np.float32).copy()
    have = np.zeros((H, W), bool) if have is None else np.asarray(have, bool).copy()
    cls = np.zeros((H, W), np.int64) if cls is None else np.asarray(cls).astype(np.int64).copy()
    gl, tt, s, _dz = ground_level(vk, info)
    bi = band_index(info, s)
    bi[~gl] = -1
    if not (bi >= 0).any():
        return fill, have, cls, bi
    foot = MV.pixel_footprint(P)
    drift = 1.0 + MV.FILL_LOWFREQ_AMP * MV.value_noise(tt / MV.FILL_LOWFREQ_M, s / MV.FILL_LOWFREQ_M, 21)
    for i, sy in enumerate(synths):
        sel = bi == i
        if not sel.any() or sy.get("S") is None:
            continue
        S = sy["S"]
        g = float(sy.get("mpp") or MV.FILL_MPP_GROUND)
        lod = np.log2(np.maximum(foot / g, 1.0))
        if sy["mode"] == "strip":
            h = S["G"][0].shape[0]
            v = (s[sel] - sy["s0"]) / g * float(S.get("row_scale", 1.0))
            v = np.clip(v, 0, h - 1.001)
            col = MV.synth_strips(S, tt[sel] / g, v, salt=31 + i, lod=lod[sel])
        else:
            col = MV.synth_ground(S, tt[sel] / g, (s[sel] - sy["s0"]) / g, 41 + i, lod[sel])
        fill[sel] = np.clip(col * drift[sel][:, None], 0, 255)
        have[sel] = True
        cls[sel] = BAND_CLS0 + i
    return fill, have, cls, bi


# ------------------------------------------------------------------------------------------- prompt text

def _w(x):
    return f"{x:.1f}".rstrip("0").rstrip(".")


def order_from_camera(info):
    """Bands ordered from the 主视角 camera side outward (near → far)."""
    bands = list(info.get("bands") or [])
    return bands if info.get("near_side", "lo") != "hi" else bands[::-1]


def bands_text(info, dry=True):
    """One sentence (zh) + one line (en): the 主视角's ground bands, near → far, with widths and the curb step."""
    bands = order_from_camera(info)
    if len(bands) < 2:
        return "", ""
    zh, en = [], []
    for b in bands:
        open_ = b.get("open_lo") or b.get("open_hi")
        w = ("≥ " if open_ else "约 ") + _w(b["width"]) + " m 宽"
        extra = ""
        if b["kind"] == "curb" and b.get("height", 0) > 0:
            extra = f"，略高于两侧（约 {int(round(b['height'] * 100))} cm）"
        if b["kind"] == "main":
            extra += "，产品所在"
        zh.append(f"{b['zh']}（{w}{extra}）")
        en.append(b["en"] + (f" (~{int(round(b['height'] * 100))} cm high)" if b["kind"] == "curb" and b.get("height") else ""))
    names = " + ".join(b["zh"] for b in bands) + ("，干燥" if dry else "")
    s_zh = ("地面分带（主视角实测，从主视角近处到远处）：" + " → ".join(zh) + f"。即「{names}」。"
            "这些分带沿道路方向笔直连续延伸，宽度、先后次序、颜色和颗粒粗细保持不变；路沿是一道连续的条石，不要断开、"
            "消失或改成别的高度；不要把这些分带合并成一整片均匀的细沙或同一种地面")
    s_en = ("[GROUND BANDS] measured from the master view: " + " | ".join(en)
            + "; every band continues straight along the road with the same width, order, colour and grain — keep the "
              "curb continuous, do not merge the bands into one uniform sand surface.")
    return s_zh, s_en


def view_stats(vk, info, bi=None):
    """Where each band shows in view k: share of the frame, nearer / farther than the product, rows."""
    P = np.asarray(vk["pos"], np.float64)
    H, W = P.shape[:2]
    if bi is None:
        gl, _tt, s, _dz = ground_level(vk, info)
        bi = band_index(info, s)
        bi[~gl] = -1
    MV = _mv()
    cam_o = MV._mat(vk["cam"])[:3, 3]
    depth = np.linalg.norm(P[..., :3] - cam_o, axis=-1)
    m = np.asarray(vk["mask"]) > 0.5
    pd = float(np.median(depth[m & (P[..., 3] > 0.5)])) if (m & (P[..., 3] > 0.5)).sum() > 20 else None
    out = []
    rows = np.arange(H)[:, None] * np.ones((1, W))
    cols = np.ones((H, 1)) * np.arange(W)[None]
    for i, b in enumerate(info.get("bands") or []):
        sel = bi == i
        share = float(sel.mean())
        e = {"i": i, "share": round(share, 4)}
        if sel.sum() >= 20:
            dd = float(np.median(depth[sel]))
            e["rel"] = "" if pd is None else ("behind" if dd > pd + 1.0 else ("front" if dd < pd - 1.0 else "at"))
            e["row"] = round(float(np.median(rows[sel])) / H, 3)
            e["span"] = round(float(np.percentile(cols[sel], 97) - np.percentile(cols[sel], 3)) / W, 3)
        out.append(e)
    return out


def view_sentence(info, stats):
    """Where the bands run in this view (only bands that show)."""
    bands = info.get("bands") or []
    bits = []
    for e in stats or []:
        if e.get("share", 0) < 0.0015 or "rel" not in e:
            continue
        b = bands[e["i"]]
        where = {"behind": "在产品后方（离镜头更远处）", "front": "在产品前方（离镜头更近处）",
                 "at": "在产品所在的位置"}.get(e["rel"], "")
        across = "横贯画面" if e.get("span", 0) > 0.6 else ""
        pos = "画面上部" if e.get("row", 0.5) < 0.42 else ("画面下部" if e.get("row", 0.5) > 0.7 else "画面中部")
        bits.append(f"{b['zh']}{where}{pos}{('、' + across) if across else ''}")
    if not bits:
        return ""
    return "这个视角里：" + "；".join(bits) + "（白模里的条带 / 路沿就是它们的位置，按白模画）"


def lock_addition(info, stats=None, dry=True):
    """Text merged into the 地面锁定 line (zh sentence, en line)."""
    zh, en = bands_text(info, dry=dry)
    if not zh:
        return "", ""
    vs = view_sentence(info, stats) if stats is not None else ""
    return zh + ("；" + vs if vs else ""), en


def analysis_bands_text(info):
    """The 主视角 request's own band line (from the ④ measured reference): every image of a 张数 > 1 request gets it."""
    zh, en = bands_text(info)
    if not zh:
        return ""
    return "【地面分带】" + zh + "。\n" + en


# ------------------------------------------------------------------------------------------- geometry (bpy)

def geometry_extent(info, cam_xy):
    """Half length (m) of the band strips along t: past every camera by GEOM_EXTENT_M[0], capped."""
    if not cam_xy:
        return GEOM_EXTENT_M[0]
    tt, _s = frame(info, np.asarray(cam_xy, np.float64))
    return float(min(GEOM_EXTENT_M[1], max(GEOM_EXTENT_M[0], float(np.abs(tt).max()) + GEOM_EXTENT_M[0])))


def strip_mesh(info, b, half_len, step=GEOM_STEP_M):
    """(verts, faces) of one band strip in world space: flat bands a thin quad strip just above the band plane,
    the curb a raised box (top + both side faces + end caps)."""
    t = np.asarray(info["t"], np.float64)
    n = np.asarray(info["n"], np.float64)
    a0 = np.asarray(info["a0"], np.float64)
    s0 = b["s0"] - (min(OPEN_EXTEND_M, 3.0) if b.get("open_lo") else 0.0)
    s1 = b["s1"] + (min(OPEN_EXTEND_M, 3.0) if b.get("open_hi") else 0.0)
    ts = np.arange(-half_len, half_len + 1e-6, step)
    verts, faces = [], []
    curb = b.get("kind") == "curb" and b.get("height", 0) > 0
    for k, tv in enumerate(ts):
        for sv in (s0, s1):
            xy = a0 + t * tv + n * sv
            z = float(plane_z(info, tv, sv))
            verts.append((float(xy[0]), float(xy[1]), z + (b["height"] if curb else GEOM_LIFT)))
        if curb:
            for sv in (s0, s1):
                xy = a0 + t * tv + n * sv
                verts.append((float(xy[0]), float(xy[1]), float(plane_z(info, tv, sv)) - 0.05))
    per = 4 if curb else 2
    for k in range(len(ts) - 1):
        a, c = k * per, (k + 1) * per
        faces.append((a, a + 1, c + 1, c))                  # top
        if curb:
            faces.append((a + 2, a, c, c + 2))              # side at s0
            faces.append((a + 1, a + 3, c + 3, c + 1))      # side at s1
    if curb and len(ts) > 1:
        faces.append((0, 2, 3, 1))
        e = (len(ts) - 1) * per
        faces.append((e, e + 1, e + 3, e + 2))
    return verts, faces


def raised(b) -> bool:
    return b.get("kind") == "curb" and float(b.get("height", 0) or 0) > 0


def build_geometry(context, massing_coll, info, cam_xy=()):
    """The raised bands (curbs) as objects in the 场景粗模, along the measured direction past every camera (names
    start with ``_wondful_`` so the block list / prompt never counts them; tagged ``BAND_TAG``). Returns the created
    datablocks (for ``multiview.remove_aux``)."""
    import bpy
    made = []
    if massing_coll is None or not info or not info.get("bands"):
        return made
    half = geometry_extent(info, cam_xy)
    mats = {}

    def mat_for(kind):
        if kind in mats:
            return mats[kind]
        m = bpy.data.materials.new(f"_wondful_ground_band_{kind}")
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        grey = {"curb": 0.16, "main": 0.24, "shoulder": 0.2, "strip": 0.19}.get(kind, 0.21)
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (grey, grey, grey * 1.04, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.85
        mats[kind] = m
        made.append(m)
        return m
    for i, b in enumerate(info["bands"]):
        # flat bands lie on the measured ground itself (the fill and the prompt carry them); a strip just above it
        # would only add a fake step edge where it ends — only bands with a height step become geometry
        if not raised(b):
            continue
        v, f = strip_mesh(info, b, half)
        if not f:
            continue
        name = f"_wondful_ground_band_{i:02d}_{b.get('kind', 'band')}"
        me = bpy.data.meshes.new(name)
        me.from_pydata(v, [], f)
        me.update()
        ob = bpy.data.objects.new(name, me)
        me.materials.append(mat_for(b.get("kind", "band")))
        ob[BAND_TAG] = i
        ob["wondful_mv_aux"] = 1
        ob["wondful_hint"] = b.get("zh", "")
        massing_coll.objects.link(ob)
        made.append(ob)
    return made


def analysis_direction(analysis):
    """Unit xy direction of the longest analysed road / long surface zone (analysis frame), or None."""
    best = None
    for r in (analysis or {}).get("roads", []) or []:
        pts = [p for p in (r.get("points") or []) if isinstance(p, dict)]
        if len(pts) < 2:
            continue
        d = np.array([float(pts[-1].get("x", 0)) - float(pts[0].get("x", 0)),
                      float(pts[-1].get("y", 0)) - float(pts[0].get("y", 0))])
        L = float(np.linalg.norm(d))
        if L > 1.0 and (best is None or L > best[0]):
            best = (L, d / L)
    return None if best is None else best[1]


_CURB_RX = re.compile(r"curb|kerb|路沿|路缘|路牙|道牙", re.I)


def _grain_words(text):
    t = (text or "").lower()
    if re.search(r"coarse|rough|angular|rubble|粗|碎石", t):
        return "coarse"
    if re.search(r"compacted|packed|smooth|level|fine|压实|平整|细", t):
        return "fine"
    return ""


def analysis_bands(analysis):
    """Ground bands from the ④ analysis alone (roads with widths, surface zones, long thin curb primitives) for
    the 主视角 request itself — no image yet. Same dict shape as ``measure`` (frame: the analysis camera at the
    origin); {} when the analysis has fewer than two ground bands."""
    an = analysis or {}
    t = analysis_direction(an)
    if t is None:
        t = np.array([1.0, 0.0])
    n = np.array([t[1], -t[0]])
    if n[1] < 0:                       # across axis points away from the camera (+y forward)
        n = -n
    items = []
    for r in an.get("roads", []) or []:
        pts = [p for p in (r.get("points") or []) if isinstance(p, dict)]
        if not pts:
            continue
        c = np.mean([[float(p.get("x", 0)), float(p.get("y", 0))] for p in pts], 0) @ n
        w = float(r.get("width_m") or 0)
        if w <= 0:
            continue
        items.append({"s0": c - w / 2, "s1": c + w / 2, "text": f"{r.get('surface', '')} {r.get('label_zh', '')}",
                      "road": str(r.get("kind") or "road") in ("road", "street", "lane", "path")})
    for z in an.get("surface_zones", []) or []:
        pts = [p for p in (z.get("points") or []) if isinstance(p, dict)]
        if len(pts) < 3:
            continue
        ss = [np.array([float(p.get("x", 0)), float(p.get("y", 0))]) @ n for p in pts]
        items.append({"s0": min(ss), "s1": max(ss), "text": str(z.get("surface") or ""),
                      "road": bool(re.search(r"road|street|paving|路", str(z.get("kind", "")) + str(z.get("id", ""))))})
    curbs = []
    for p in an.get("primitives", []) or []:
        if not isinstance(p, dict) or not _CURB_RX.search(f"{p.get('id', '')} {p.get('label_zh', '')}"):
            continue
        w, d = float(p.get("w") or 0), float(p.get("d") or 0)
        if max(w, d) < 5 or min(w, d) > 0.8:
            continue
        c = np.array([float(p.get("x", 0)), float(p.get("y", 0))]) @ n
        th = min(w, d)
        curbs.append({"s0": c - th / 2, "s1": c + th / 2, "h": float(p.get("h") or CURB_H),
                      "text": str(p.get("material") or "")})
    if not items:
        return {}
    # surface pieces → contiguous non-overlapping bands (a later / zone item overrides where they overlap)
    edges = sorted({round(v, 2) for it in items for v in (it["s0"], it["s1"])}
                   | {round(v, 2) for c in curbs for v in (c["s0"], c["s1"])})
    bands = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 0.05:
            continue
        mid = 0.5 * (a + b)
        cb = [c for c in curbs if c["s0"] - 1e-6 <= mid <= c["s1"] + 1e-6]
        if cb:
            src = {"kind": "curb", "text": cb[0]["text"], "h": cb[0]["h"]}
        else:
            cov = [it for it in items if it["s0"] - 1e-6 <= mid <= it["s1"] + 1e-6]
            if not cov:
                continue
            it = min(cov, key=lambda x: x["s1"] - x["s0"])     # the most specific piece
            src = {"kind": "area", "text": it["text"], "road": it["road"]}
        if bands and bands[-1]["_src"]["text"] == src["text"] and abs(bands[-1]["s1"] - a) < 0.3:
            bands[-1]["s1"] = b
            continue
        bands.append({"s0": a, "s1": b, "_src": src})
    if len(bands) < 2:
        return {}
    spot = an.get("product_spot") or {}
    ps = float(np.array([float(spot.get("x", 0)), float(spot.get("y", 0))]) @ n) if spot else None
    words = _analysis_words(an)
    out = []
    for i, b in enumerate(bands):
        src = b.pop("_src")
        b["s0"], b["s1"] = round(float(b["s0"]), 2), round(float(b["s1"]), 2)
        b["width"] = round(b["s1"] - b["s0"], 2)
        if src["kind"] == "curb":
            m_zh, m_en = ("石", "stone") if re.search(r"stone|石", src["text"], re.I) else \
                _material(src["text"], ("石", "stone"))
            if m_zh in ("土", "沙", "草", "砂石"):
                m_zh, m_en = "石", "stone"
            dark = bool(re.search(r"dark|深", src["text"], re.I))
            b.update(kind="curb", height=round(min(CURB_H_MAX, src["h"] or CURB_H), 2),
                     zh=("深色" if dark else "") + m_zh + "路沿", en=("dark " if dark else "") + m_en + " curb")
        else:
            gr = _grain_words(src["text"])
            m_zh, m_en = _material(src["text"], ("", ""), gr)
            main = ps is not None and b["s0"] - 1e-6 <= ps <= b["s1"] + 1e-6
            q_zh, q_en = {"coarse": ("粗", "coarse "), "fine": ("压实", "compacted ")}.get(gr, ("", ""))
            noun = ("路面", "road surface") if (src.get("road") or main and words.get("has_road")) else \
                ("路肩", "shoulder") if words.get("has_road") else ("地面", "ground")
            b.update(kind="main" if main else "shoulder", height=0.0, zh=f"{q_zh}{m_zh}{noun[0]}",
                     en=f"{q_en}{m_en + ' ' if m_en else ''}{noun[1]}")
        if out and out[-1]["zh"] == b["zh"] and out[-1]["kind"] != "curb" and abs(out[-1]["s1"] - b["s0"]) < 0.3:
            out[-1]["s1"] = b["s1"]
            out[-1]["width"] = round(out[-1]["s1"] - out[-1]["s0"], 2)
            if b["kind"] == "main":
                out[-1]["kind"] = "main"
            continue
        out.append(b)
    while len(out) > 2 and out[-1]["kind"] != "curb" and out[-1]["width"] < 1.5 and out[-1]["kind"] != "main":
        out.pop()                      # a sliver where two analysed pieces end at different places
    while len(out) > 2 and out[0]["kind"] != "curb" and out[0]["width"] < 1.5 and out[0]["kind"] != "main":
        out.pop(0)
    if len(out) < 2:
        return {}
    out[0]["open_lo"] = True
    out[-1]["open_hi"] = True
    return {"t": [float(t[0]), float(t[1])], "n": [float(n[0]), float(n[1])], "a0": [0.0, 0.0],
            "plane": [0.0, 0.0, 0.0], "bands": out, "source": "analysis", "near_side": "lo"}


def analysis_bands_text(analysis):
    """The 主视角 request's band line (from the ④ analysis): every image of a 张数 > 1 request gets the same one."""
    info = analysis_bands(analysis)
    zh, en = bands_text(info)
    if not zh:
        return ""
    return "【地面分带】" + zh.replace("（主视角实测，从主视角近处到远处）", "（来自环境参考图分析，从镜头近处到远处）") + "。\n" + en.replace(
        "measured from the master view", "from the reference analysis")


# ------------------------------------------------------------------------------------------- operator glue

def measure_master(master_img, pool, prefix, rec, W, H, ww, wh, product_box=None, completion=None, analysis=None):
    """Bands of the 主视角 right after its geometry pass: the 主视角 image (fitted to the render canvas, written to
    ``pool/master.png`` like the job does) + ``<prefix>_pos/_nrm/_maskw.npy``. ``product_box`` = (min, max) world
    corners of the product; the measured street direction of the ④ 场景补全 (``completion``) is the prior."""
    from pathlib import Path
    from .layered_output import resize
    from .png_io import read_png
    from . import result_worker as RW
    from .photo_finish import clean_source
    MV = _mv()
    mp = Path(pool) / "master.png"
    if not RW.fit_to_canvas(clean_source(master_img), mp, W, H):
        return {}
    prefix = str(prefix)
    img = read_png(mp).astype(np.float32)[..., :3]
    pos = np.load(prefix + "_pos.npy")
    if img.shape[:2] != pos.shape[:2]:
        img = resize(img, pos.shape[1], pos.shape[0])
    view = {"cam": rec, "pos": pos, "nrm": np.load(prefix + "_nrm.npy"),
            "mask": np.load(prefix + "_maskw.npy").astype(np.float32), "img": img}
    view["sky"] = MV.sky_mask(view["pos"], img)
    if product_box is not None:
        mn, mx = product_box
        view["product_box"] = ([float(mn[0]), float(mn[1])], [float(mx[0]), float(mx[1])])
    st = (completion or {}).get("street") or {}
    prior = st.get("t") if st else None
    return measure(view, prior_t=prior, analysis=analysis, source="master")


def summary(info) -> str:
    bands = order_from_camera(info)
    if not bands:
        return ""
    geo = info.get("geometry")
    return ("地面分带：" + " + ".join(b["zh"] for b in bands)
            + (f"（路沿几何 {geo} 条）" if geo else ""))


def fill_note(share) -> str:
    """Prompt note when a view's base carries band texture where the 主视角 did not see the ground."""
    if share < 0.02:
        return ""
    return ("底图里主视角没拍到的地面（约 {:.0%}）已按主视角的地面分带铺了取样纹理（每条带用它自己的路面 / 路沿 / 路肩材质）："
            "请画成同样分带的真实地面，分带的位置、宽度和走向保持底图里的样子，纹理只表示材质和颜色范围，可以重画细节。").format(share)
