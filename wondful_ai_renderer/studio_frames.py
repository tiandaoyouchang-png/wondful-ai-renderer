"""Screen-anchored luminous architectural frames for reference-replica blockouts.

Frames are real, closed 3D beam meshes (not an image/texture overlay). A single
photograph constrains their projection, not their unique hidden depth. The user
can subsequently adjust the generated Blender geometry in the scene.

This module deliberately has no Blender, OpenCV, PIL or network dependency.
"""
from __future__ import annotations

import math
import re

import numpy as np

MAX_FRAMES = 12
MAX_POINTS = 16
_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _finite(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} 不是有效数值")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{label} 不是有限数值")
    return num


def normalize_frames(items):
    """Validate bounds, detect impossible strokes and preserve ordered screen anchors."""
    valid, warnings = [], []
    if items is None:
        return valid, warnings
    if not isinstance(items, list):
        return valid, ["发光框架不是列表，已忽略"]
    if len(items) > MAX_FRAMES:
        warnings.append(f"发光框架超过 {MAX_FRAMES} 组，已忽略多余项")
    ids = set()
    for i, raw in enumerate(items[:MAX_FRAMES]):
        name = f"发光框架{i + 1}"
        try:
            if not isinstance(raw, dict):
                raise ValueError("结构不是对象")
            name = str(raw.get("label_zh") or name).strip()[:36] or name
            identity = str(raw.get("id") or f"light_frame_{i + 1}").strip()[:80]
            if not identity or identity in ids:
                raise ValueError("重复或空白的结构 id")
            pts = raw.get("points")
            if not isinstance(pts, list) or not 2 <= len(pts) <= MAX_POINTS:
                raise ValueError("轮廓点数量必须为 2–16")
            coords = []
            for point in pts:
                if not isinstance(point, dict):
                    raise ValueError("缺少屏幕坐标")
                x, y = (_finite(point.get(k), k) for k in ("x_frac", "y_frac"))
                if not (-1.5 <= x <= 2.5 and -1.5 <= y <= 2.5):
                    raise ValueError("屏幕端点超出允许画幅范围")
                if not coords or math.hypot(x - coords[-1][0], y - coords[-1][1]) >= 0.001:
                    coords.append((x, y))
            if len(coords) < 2:
                raise ValueError("有效轮廓点不足")
            depth = _finite(raw.get("depth_m", 15.0), "depth_m")
            width = _finite(raw.get("width_frac", 0.018), "width_frac")
            thickness = _finite(raw.get("beam_depth_m", 0.25), "beam_depth_m")
            emission = _finite(raw.get("emission", 1.5), "emission")
            if not (2.0 <= depth <= 200.0 and 0.001 <= width <= 0.14
                    and 0.02 <= thickness <= 5.0 and 0.0 <= emission <= 30.0):
                raise ValueError("结构深度、梁宽、厚度或发光强度超限")
            color = str(raw.get("color_hex") or "#EAF4FF").strip().upper()
            if not _HEX.fullmatch(color):
                raise ValueError("颜色须为 #RRGGBB")
            grounded = raw.get("grounded", False)
            if type(grounded) is not bool:
                raise ValueError("grounded 必须为布尔值")
            ids.add(identity)
            valid.append({"id": identity, "label_zh": name,
                          "points": [{"x_frac": x, "y_frac": y} for x, y in coords],
                          "depth_m": depth, "width_frac": width, "beam_depth_m": thickness,
                          "color_hex": color, "emission": emission,
                          "material": str(raw.get("material") or "")[:400],
                          "grounded": grounded})
        except (TypeError, ValueError, OverflowError) as exc:
            warnings.append(f"{name}：{exc}，已跳过")
    return valid, warnings


def _miter_normals(points_px):
    """Offset around the visible stroke, with a bounded joint at an acute V."""
    segments = np.diff(points_px, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths < 1e-3):
        raise ValueError("发光框架有重合端点")
    tangents = segments / lengths[:, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    out = []
    for i in range(len(points_px)):
        if i == 0:
            out.append(normals[0]); continue
        if i == len(points_px) - 1:
            out.append(normals[-1]); continue
        joint = normals[i - 1] + normals[i]
        norm = np.linalg.norm(joint)
        if norm < 0.1:  # a hairpin cannot have a stable miter
            out.append(normals[i]); continue
        joint /= norm
        proj = float(np.dot(joint, normals[i]))
        out.append(joint * min(2.5, 1.0 / max(0.15, proj)))
    return np.asarray(out)


def beam_mesh(points, frame, width_frac, depth_m, beam_depth_m, project):
    """Build a manifold rectangular sweep. ``project(u,v,y)`` returns XYZ.

    Each front/back vertex projects onto the same normalized image anchor.
    The close (camera-facing) skin is face 0..n-2; other faces form the depth.
    """
    w, h = frame
    if w <= 0 or h <= 0:
        raise ValueError("参考画幅无效")
    xy = np.asarray([[p["x_frac"] * w, p["y_frac"] * h] for p in points], float)
    if len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError("发光框架缺少端点")
    off = _miter_normals(xy) * (float(width_frac) * w * 0.5)
    near = max(0.2, float(depth_m) - float(beam_depth_m) / 2.0)
    far = float(depth_m) + float(beam_depth_m) / 2.0
    verts = []
    for i, (x, y) in enumerate(xy):
        for depth, direction in ((near, +1), (near, -1), (far, +1), (far, -1)):
            px, py = (xy[i] + direction * off[i]) / (w, h)
            verts.append(tuple(map(float, project(float(px), float(py), depth))))
    faces = []
    n = len(xy)
    for j in range(n - 1):
        a, b = 4*j, 4*(j+1)
        faces.append((a, b, b+1, a+1))  # illuminated front skin
    front_faces = len(faces)
    for j in range(n - 1):
        a, b = 4*j, 4*(j+1)
        faces.extend(((a+2, a+3, b+3, b+2),
                      (a, a+2, b+2, b),
                      (a+1, b+1, b+3, a+3)))
    faces.extend(((0, 1, 3, 2), (4*(n-1), 4*(n-1)+2, 4*(n-1)+3, 4*(n-1)+1)))
    arr = np.asarray(verts, float)
    if not np.isfinite(arr).all():
        raise ValueError("发光框架投影产生非有限坐标")
    for face in faces:
        v = arr[list(face)]
        area = np.linalg.norm(np.cross(v[1]-v[0], v[2]-v[0]))
        if area < 1e-9:
            raise ValueError("发光框架包含退化面")
    return arr, faces, {"front_faces": front_faces, "segments": n-1,
                        "projection_points": [tuple(p) for p in xy], "width_px": float(width_frac)*w}


def projected_stroke_mask(frame, rows, *, scale=1):
    """Deterministic raster of scene-frame centerlines (for reference diagnostics)."""
    w, h = frame
    w = max(1, int(round(w * scale)))
    h = max(1, int(round(h * scale)))
    mask = np.zeros((h, w), bool)
    for row in rows:
        pts = np.asarray([[p["x_frac"]*w, p["y_frac"]*h] for p in row["points"]], float)
        radius = max(0.7, row["width_frac"] * w / 2.0)
        for a, b in zip(pts[:-1], pts[1:]):
            x0, x1 = max(0, int(math.floor(min(a[0], b[0]) - radius))), min(w, int(math.ceil(max(a[0], b[0]) + radius + 1)))
            y0, y1 = max(0, int(math.floor(min(a[1], b[1]) - radius))), min(h, int(math.ceil(max(a[1], b[1]) + radius + 1)))
            if x0 >= x1 or y0 >= y1:
                continue
            xx, yy = np.meshgrid(np.arange(x0, x1)+0.5, np.arange(y0, y1)+0.5)
            dx, dy = b-a
            t = np.clip(((xx-a[0])*dx + (yy-a[1])*dy) / max(float(dx*dx+dy*dy), 1e-12), 0.0, 1.0)
            mask[y0:y1, x0:x1] |= (xx-a[0]-t*dx)**2 + (yy-a[1]-t*dy)**2 <= radius**2
    return mask


def frame_alignment_score(photo_rgb, rows, exclude_bbox=None, working_width=320):
    """Compare bright frame strokes to photograph pixels, not sky IoU.

    Designed for artificial studios: the black background is *not* sky, and a
    better pair of white LED beams must not be rejected for reducing sky IoU.
    The photographed product is excluded so silver car highlights do not count
    as architectural beams. This is one reference-view metric, not 3D truth.
    """
    if not rows:
        return None
    photo = np.asarray(photo_rgb, dtype=np.float32)
    h0, w0 = photo.shape[:2]
    w = min(512, max(64, int(working_width)))
    h = max(8, round(w*h0/w0))
    yy = np.minimum((np.arange(h)*h0/h).astype(int), h0-1)
    xx = np.minimum((np.arange(w)*w0/w).astype(int), w0-1)
    image = photo[yy[:,None], xx[None,:], :3]
    if image.max() > 1.5:
        image /= 255.0
    valid = np.ones((h,w),bool)
    if exclude_bbox is not None:
        try:
            a,b,c,d = [float(exclude_bbox[k]) for k in ('x_min','y_min','x_max','y_max')]
            if 0<=a<c<=1 and 0<=b<d<=1:
                valid[max(0,int(b*h)-3):min(h,math.ceil(d*h)+3),
                      max(0,int(a*w)-3):min(w,math.ceil(c*w)+3)] = False
        except (KeyError,TypeError,ValueError):
            pass
    # Mirror images / bright diffuse floor reflections are not structural bars.
    visible_bottom = min(1., max(0.75, max(max(p['y_frac'] for p in r['points'])
                                             for r in rows if r['points']) + 0.03))
    if visible_bottom <= 0.94:
        valid[min(h,int(visible_bottom*h)):] = False
    rgbmin = image.min(axis=2)
    pale = (rgbmin > .54) & (image.mean(axis=2) > .69)
    red = (image[...,0] > .35) & (image[...,0] > 1.4*image[...,1]) & (image[...,0] > 1.35*image[...,2])

    def dilate(mask, radius=2):
        pad = np.pad(mask, radius)
        result = np.zeros_like(mask)
        for dy in range(radius*2+1):
            for dx in range(radius*2+1):
                result |= pad[dy:dy+h,dx:dx+w]
        return result

    outcomes, hints = [], []
    for row in rows:
        mask = projected_stroke_mask((w,h), [row]) & valid
        if not np.any(mask):
            continue
        col = row['color_hex'].lstrip('#')
        c = [int(col[i:i+2],16) for i in (0,2,4)]
        target = red if c[0] > 1.3 * max(c[1],c[2]) else pale
        target &= valid
        matches = mask & dilate(target)
        # Recall within the frame's vicinity, not among every white reflection.
        vicinity = dilate(mask, 12)
        relevant = target & vicinity
        precision = float(matches.sum()) / max(1,int(mask.sum()))
        recall = float((relevant & dilate(mask)).sum()) / max(1,int(relevant.sum()))
        f = 2*precision*recall/max(1e-8,precision+recall)
        weight = max(1.0, math.sqrt(float(mask.sum())))
        outcomes.append((f,precision,recall,weight,row['label_zh']))
        if precision < 0.35:
            hints.append(f"发光框架「{row['label_zh']}」中心线与目标亮条不匹配，优先检查端点/梁宽")
    if not outcomes:
        return {'frame_f':0.0,'frame_precision':0.0,'frame_recall':0.0,'frame_hints':['亮框全部不在可见画幅内']}
    total=sum(r[3] for r in outcomes)
    return {'frame_f': round(sum(r[0]*r[3] for r in outcomes)/total,3),
            'frame_precision':round(sum(r[1]*r[3] for r in outcomes)/total,3),
            'frame_recall':round(sum(r[2]*r[3] for r in outcomes)/total,3),
            'frame_hints':hints}
