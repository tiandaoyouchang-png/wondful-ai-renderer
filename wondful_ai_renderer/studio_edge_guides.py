"""Optional long bright-edge hints from a reference PNG, without OpenCV or PIL.

These image measurements are deliberately *candidates*, not geometry truth. A
reasoning model still decides whether they represent architecture or highlights
on a car. No files, endpoints or geometry are ever written by this module.
"""
from __future__ import annotations

from collections import deque
import math

import numpy as np


def bright_bar_hints(rgba, limit=8):
    """Return large elongated pale components as image-space line candidates.

    Works in stock Blender's numpy environment. The detection is conservative:
    it cannot infer dark parts, hidden corners or correct 3D depth.
    """
    rgb = np.asarray(rgba)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return []
    h, w = rgb.shape[:2]
    if min(h, w) < 80:
        return []
    step = max(1, math.ceil(max(h, w) / 420))
    nh, nw = h // step, w // step
    # Block averaging, not point sampling: narrow strips remain detectable.
    patch = rgb[:nh*step, :nw*step, :3].astype(np.float32).reshape(nh, step, nw, step, 3).mean(axis=(1, 3))
    pale = (patch.min(axis=2) > 174) & (patch.mean(axis=2) > 199)
    # On black automotive stages a thin, long RED stripe often defines the
    # floor seam. Large bright patches below it are usually reflections, not
    # another architectural portal. Derive this cutoff from the image itself,
    # rather than assuming that the floor is always at a fixed screen height.
    red = (patch[..., 0] > 85) & (patch[..., 0] > patch[..., 1] * 1.55) & (patch[..., 0] > patch[..., 2] * 1.2)
    red_rows = red.sum(axis=1)
    floor_rows = np.flatnonzero((red_rows > max(22, int(.1*nw))) &
                                (np.arange(nh) > nh * .48))
    if len(floor_rows):
        seam = int(floor_rows[np.argmax(red_rows[floor_rows])])
        pale[min(nh, seam + 3):] = False
    visited = np.zeros((nh, nw), bool)
    candidates = []
    for yi, xi in zip(*np.nonzero(pale)):
        if visited[yi, xi]:
            continue
        visited[yi, xi] = True
        todo = deque([(int(yi), int(xi))])
        pixels = []
        while todo:
            y, x = todo.popleft()
            pixels.append((x, y))
            for yy, xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1),
                           (y-1,x-1),(y-1,x+1),(y+1,x-1),(y+1,x+1)):
                if 0 <= yy < nh and 0 <= xx < nw and pale[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    todo.append((yy, xx))
        if len(pixels) < max(40, int(nw*nh*0.0004)):
            continue
        coords = np.asarray(pixels, dtype=np.float64)
        center = coords.mean(axis=0)
        delta = coords-center
        eigenvalues, eigenvectors = np.linalg.eigh(delta.T @ delta / len(coords))
        if eigenvalues[-1] <= 1e-9:
            continue
        direction = eigenvectors[:, -1]
        length_projection = delta @ direction
        l0, l1 = np.percentile(length_projection, [2, 98])
        length = l1 - l0
        thickness = len(coords) / max(length, 1)
        # Large straight bars only; a round highlight or car body is not a frame.
        if length < max(42.0, min(nw,nh)*0.24) or thickness/length > 0.28:
            continue
        p0, p1 = center + l0*direction, center + l1*direction
        if p0[1] > p1[1]:
            p0, p1 = p1, p0
        candidates.append({
            "start": (round(float((p0[0]+.5)*step/w), 3), round(float((p0[1]+.5)*step/h), 3)),
            "end": (round(float((p1[0]+.5)*step/w), 3), round(float((p1[1]+.5)*step/h), 3)),
            "approx_width_frac": round(thickness*step/w, 3),
            "relative_length": round(length*step/max(w,h), 3),
        })
    candidates.sort(key=lambda a: a["relative_length"], reverse=True)
    return candidates[:max(0, min(12, int(limit)))]


def guide_text(candidates):
    if not candidates:
        return ""
    lines = ["COMPUTER-VISION HINTS (bright elongated pixel components, only candidates):"]
    for i, row in enumerate(candidates, 1):
        lines.append(f"{i}. screen centreline {row['start']} -> {row['end']}, approx width of full image "
                     f"{row['approx_width_frac']:.3f}; may be a light bar OR a car highlight/reflection.")
    portals = infer_open_v_portals(candidates)
    for portal in portals:
        p = portal['points']
        lines.append('PAIRED V candidate (verify in the image, not guaranteed): '
                     f"floor-left {p[0]}, EXTRAPOLATED apex {p[1]}, floor-right {p[2]}, "
                     f"beam width frac {portal['width_frac']:.4f}. The apex may lie ABOVE the crop. "
                     'Use one open light_frames polyline, not a filled triangular wall.')
    lines.append("Use the original photo to identify REAL rigid architectural beams, discard car reflections "
                 "and floor reflections. Extrapolate visible beam centre lines to an off-screen V apex where needed; "
                 "preserve multiple independent near/far frames. Do not treat these hints as proven 3D geometry.")
    return "\n".join(lines)


def infer_open_v_portals(candidates, limit=2):
    """Offer high-confidence *image-space* V outlines when two long limbs agree.

    Intersection outside the photograph is allowed, because an automotive LED
    portal is frequently cropped at its apex. These are hypotheses, not a 3D
    scene reconstruction; downstream must validate material/occlusion/depth.
    """
    left = []
    right = []
    for c in candidates:
        try:
            x0, y0 = (float(v) for v in c['start'])
            x1, y1 = (float(v) for v in c['end'])
            dy = y1 - y0
            if not (-.12 <= y0 <= .27 and .56 <= y1 <= 1.05
                    and dy >= .35 and float(c['relative_length']) >= .30):
                continue
            slope = (x1-x0)/dy
            if -.9 < slope < -.12 and x1 < .68:
                left.append((c, x0, y0, x1, y1, slope))
            if .12 < slope < .9 and x1 > .48:
                right.append((c, x0, y0, x1, y1, slope))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
    pairs = []
    for l in left:
        for r in right:
            if l[0] is r[0] or r[3]-l[3] < .25:
                continue
            # x(y) = intercept + slope*y
            li = l[1]-l[-1]*l[2]
            ri = r[1]-r[-1]*r[2]
            if abs(l[-1]-r[-1]) < .2:
                continue
            apex_y = (ri-li)/(l[-1]-r[-1])
            apex_x = li+l[-1]*apex_y
            if not (-.85 < apex_y <= .30 and .2 <= apex_x <= .8):
                continue
            if apex_y >= min(l[2],r[2])+.05:
                continue
            score = min(float(l[0]['relative_length']),float(r[0]['relative_length']))
            score *= .8 + .2*min(1.0, r[3]-l[3])
            if score < .22:
                continue
            pairs.append({'points':[
                {'x_frac':round(l[3],4),'y_frac':round(l[4],4)},
                {'x_frac':round(apex_x,4),'y_frac':round(apex_y,4)},
                {'x_frac':round(r[3],4),'y_frac':round(r[4],4)}],
                'width_frac':round(max(.003,min(.12,(float(l[0]['approx_width_frac'])+float(r[0]['approx_width_frac']))/2)),4),
                'hint_strength':round(score,3), 'source':'paired bright-line image hypothesis'})
    pairs.sort(key=lambda p:p['hint_strength'],reverse=True)
    distinct=[]
    for pair in pairs:
        if any(abs(pair['points'][1]['x_frac']-p['points'][1]['x_frac']) < .09 and
               abs(pair['points'][1]['y_frac']-p['points'][1]['y_frac']) < .15 for p in distinct):
            continue
        distinct.append(pair)
    return distinct[:max(0,min(4,int(limit)))]


def rescue_outer_portal(data, candidates, *, dark_studio):
    """Conservatively rescue ONE missing outer V after online AI analysis.

    Never overwrite an existing V or a broad connected ribbon. Only for a dark
    indoor studio with a strong two-limb match. The resulting geometry is
    explicitly tagged as a pixel hypothesis in warnings for human review.
    """
    if not dark_studio or data.get('scene_kind') != 'interior' or data.get('ribbons'):
        return data, False
    already = data.get('light_frames') or []
    if any(len(f.get('points') or []) >= 3 for f in already):
        return data, False
    portals = infer_open_v_portals(candidates, limit=1)
    if not portals or portals[0]['hint_strength'] < .28:
        return data, False
    hint = portals[0]
    row = {
        'id':'detected_outer_v','label_zh':'待复核外层发光门架',
        'points':hint['points'], 'width_frac':hint['width_frac'],
        'depth_m':12., 'beam_depth_m':.4, 'color_hex':'#F0FAFF',
        'emission':2.5, 'material':'image-guided unconfirmed luminous V beams',
        'grounded':True,
    }
    from .studio_frames import normalize_frames
    valid,_ = normalize_frames([row])
    if not valid:
        return data, False
    out=dict(data)
    out['light_frames']=list(already)+valid
    out['_geometry_warnings']=list(data.get('_geometry_warnings',[]))+[
        '外层发光V梁由双直线像素候选补建，仅用于初步占位；请检查是否属于真实建筑、端点与遮挡层级']
    return out,True
