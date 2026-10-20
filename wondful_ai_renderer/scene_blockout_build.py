"""1.2.3 从参考图生成场景粗模 — Blender side: analysis JSON → blocks in 「Wondful 场景粗模」.

Port of the accepted v4 prototype builder (env_blockout/build_blockout_v4.py) plus the 360° band wrap and
outer ground disc of the multi-view prototype (mv2_render.py prep), restructured to run inside the add-on:
- builds in analysis coordinates (camera foot = origin, looking +Y), then moves the whole blockout so the
  analysed product spot lands under the user's product (the product never moves);
- yaw: the user's scene camera direction when the camera is not set from the analysis, else the product's
  long axis along the analysed spot heading;
- every block group is named in Chinese (``label_zh`` / type word) and carries the same auto label
  (``massing.store_label``), so 多角度场景一致 and the normal prompt can name it;
- no rendering, no file IO except the optional preview (``render_preview``) used by the fit score;
- re-running replaces the collection; the user's own camera objects are never modified.
"""
from __future__ import annotations

import json
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector
from mathutils.bvhtree import BVHTree

from . import scene_blockout as SB

MAT_PREFIX = "Wondful粗模_"
TINT = {
    "terrain": (0.50, 0.55, 0.50), "road": (0.22, 0.23, 0.25), "path": (0.45, 0.42, 0.38), "sidewalk": (0.62, 0.62, 0.60),
    "river": (0.35, 0.45, 0.60), "building": (0.66, 0.67, 0.70), "wall": (0.60, 0.61, 0.64), "fence": (0.55, 0.56, 0.60),
    "awning": (0.58, 0.58, 0.62), "pole": (0.40, 0.40, 0.44), "vehicle": (0.70, 0.72, 0.78), "person": (0.7, 0.6, 0.55),
    "tree": (0.36, 0.50, 0.38), "bush": (0.42, 0.56, 0.42), "hedge": (0.40, 0.54, 0.40), "rock": (0.55, 0.54, 0.56),
    "water_plane": (0.35, 0.45, 0.60),
    "band_rocky": (0.62, 0.62, 0.68), "band_snowy": (0.80, 0.82, 0.86), "band_forested": (0.38, 0.48, 0.42),
    "band_grassy": (0.52, 0.58, 0.48), "band_dune": (0.70, 0.62, 0.50), "band_sea": (0.35, 0.45, 0.60),
    "plaza": (0.58, 0.57, 0.55), "sign": (0.50, 0.50, 0.55), "ceiling": (0.78, 0.78, 0.80), "furniture": (0.66, 0.58, 0.52),
    "window": (0.88, 0.91, 0.95), "rug": (0.55, 0.52, 0.50), "lamp": (0.62, 0.55, 0.50), "plant": (0.40, 0.55, 0.40),
    "curtain": (0.62, 0.62, 0.66),
    "foliage_deciduous": (0.33, 0.48, 0.33), "foliage_conifer": (0.26, 0.40, 0.30), "foliage_palm": (0.38, 0.52, 0.36),
    "foliage_shrub": (0.40, 0.54, 0.38), "foliage_hedge": (0.37, 0.51, 0.36), "foliage_forest_edge": (0.30, 0.44, 0.32),
    "trunk": (0.40, 0.35, 0.30), "road_branch": (0.13, 0.13, 0.15), "cliff": (0.60, 0.58, 0.60),
    "zone_shrub_slope": (0.42, 0.50, 0.36), "zone_grass": (0.50, 0.58, 0.42), "zone_rock": (0.58, 0.57, 0.58),
    "zone_scree": (0.62, 0.60, 0.58), "zone_sand": (0.72, 0.65, 0.52), "zone_snow": (0.85, 0.87, 0.90),
    "zone_paving": (0.62, 0.62, 0.60), "zone_field": (0.56, 0.58, 0.44), "terrain_steep": (0.55, 0.54, 0.56),
}
FLAT = {"terrain", "road", "road_branch", "path", "sidewalk", "plaza", "river", "rug", "water_plane", "band_sea"}
NO_SHADOW_INTERIOR = {"ceiling", "wall", "window", "curtain"}


def _hexrgb(h):
    import re
    m = re.search(r"#?([0-9a-fA-F]{6})", h or "")
    if not m:
        return None
    v = m.group(1)
    return tuple((int(v[i:i + 2], 16) / 255.0) ** 2.2 for i in (0, 2, 4))


def _mix(a, b, f=0.5):
    return tuple(a[i] * (1 - f) + b[i] * f for i in range(3)) if b else a


def _cosfall(d, r):
    r = max(r, 0.5)
    return np.where(d < r, 0.5 * (1 + np.cos(np.pi * np.minimum(d, r) / r)), 0.0)


def _seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy + 1e-9
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / L2, 0, 1)
    return np.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _catmull_xyz(pts):
    P = [np.array([p["x"], p["y"], p.get("z", 0.0)], float) for p in pts]
    if len(P) == 2:
        n = int(max(2, min(400, np.linalg.norm(P[1][:2] - P[0][:2]) / 1.0)))
        return np.array([P[0] + (P[1] - P[0]) * t for t in np.linspace(0, 1, n)])
    P = [2 * P[0] - P[1]] + P + [2 * P[-1] - P[-2]]
    out = []
    for k in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[k - 1], P[k], P[k + 1], P[k + 2]
        n = int(max(4, min(120, np.linalg.norm(p2[:2] - p1[:2]) / 1.0)))
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out)


def _inpoly(X, Y, poly):
    ins = np.zeros(X.shape, bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi, xj, yj = poly[i]["x"], poly[i]["y"], poly[j]["x"], poly[j]["y"]
        ins ^= ((yi > Y) != (yj > Y)) & (X < (xj - xi) * (Y - yi) / (yj - yi + 1e-12) + xi)
        j = i
    return ins


def _rect_dist(px, py, p):
    return SB.rect_dist(px, py, p)


# ------------------------------------------------------------------------------------------- scene helpers

def remove_existing():
    """Delete the previous 「Wondful 场景粗模」 collection, its objects and their meshes."""
    removed = 0
    for coll in [c for c in bpy.data.collections if c.name == SB.COLLECTION_NAME or c.get("wondful_scene_blockout")]:
        for ob in list(coll.all_objects):
            me = ob.data if ob.type == "MESH" else None
            bpy.data.objects.remove(ob, do_unlink=True)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
            removed += 1
        for child in list(coll.children_recursive):
            bpy.data.collections.remove(child)
        bpy.data.collections.remove(coll)
    for m in [m for m in bpy.data.materials if m.name.startswith(MAT_PREFIX) and m.users == 0]:
        bpy.data.materials.remove(m)
    return removed


def product_info(product_collection, exclude=()):
    """(base point, dims (w, d, h), long-axis angle or None) of the product in world space."""
    if product_collection is None:
        return None
    ex = set(exclude or ())
    pts = []
    for o in product_collection.all_objects:
        if o.type != "MESH" or o in ex or o.hide_render:
            continue
        pts += [o.matrix_world @ Vector(c) for c in o.bound_box]
    if not pts:
        return None
    xs = np.array([[p.x, p.y] for p in pts])
    zmin, zmax = min(p.z for p in pts), max(p.z for p in pts)
    c = xs.mean(0)
    cov = np.cov((xs - c).T) if len(xs) > 2 else np.eye(2)
    w_, v_ = np.linalg.eigh(cov)
    major = v_[:, int(np.argmax(w_))]
    minor = np.array([-major[1], major[0]])
    lmaj = float(np.ptp((xs - c) @ major))
    lmin = float(np.ptp((xs - c) @ minor))
    angle = SB.yaw_to(float(major[0]), float(major[1])) if lmaj >= 1.3 * max(lmin, 1e-6) else None
    mn, mx = xs.min(0), xs.max(0)
    base = Vector((float((mn[0] + mx[0]) / 2), float((mn[1] + mx[1]) / 2), float(zmin)))
    return {"base": base, "dims": (max(lmin, 0.05), max(lmaj, 0.05), max(zmax - zmin, 0.05)), "axis": angle,
            "half_diag": float(np.max(np.hypot(xs[:, 0] - c[0], xs[:, 1] - c[1])))}


def camera_yaw(cam):
    if cam is None or cam.type != "CAMERA":
        return None
    f = cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
    if math.hypot(f.x, f.y) < 1e-4:
        return None
    return SB.yaw_to(f.x, f.y)


# ------------------------------------------------------------------------------------------- builder

class _Builder:
    def __init__(self, data, coll, frame, pad=None):
        self.S = SB.normalize(data)
        self.coll = coll
        self.fw, self.fh = frame
        self.hw = self.fh / float(self.fw)
        self.warn = []
        self.mats = {}
        self.groups = []           # [(label, [objects])]
        self.interior = self.S.get("scene_kind") == "interior"
        self.pad = pad              # (x, y, radius) flatten pad for the product, local coordinates
        self._tints()

    # ---------------------------------------------------------------- materials / objects
    def _tints(self):
        P = self.S.get("palette", {}) or {}
        self.T = dict(TINT)
        surf = (str(self.S["terrain"].get("surface", "")) + " " + str(P.get("ground_words", ""))).lower()
        g = _hexrgb(P.get("ground_hex")) if (self.interior or P.get("snow_cover") or any(
            k in surf for k in ("sand", "dune", "snow", "desert", "paving", "floor", "concrete", "cobble"))) \
            else _hexrgb(P.get("grass_hex"))
        self.T["terrain"] = _mix(self.T["terrain"], g, 0.5)
        f = _hexrgb(P.get("foliage_hex"))
        for k in list(self.T):
            if k.startswith("foliage_") or k in ("tree", "bush", "hedge", "plant", "band_forested"):
                self.T[k] = _mix(self.T[k], f, 0.45)
        lum = 0.2126 * self.T["terrain"][0] + 0.7152 * self.T["terrain"][1] + 0.0722 * self.T["terrain"][2]
        if P.get("snow_cover"):
            self.T["road"], self.T["road_branch"] = (0.52, 0.54, 0.58), (0.46, 0.48, 0.52)
        elif lum < 0.16:
            self.T["road"], self.T["road_branch"] = (0.42, 0.42, 0.44), (0.36, 0.36, 0.38)
        else:
            self.T["road"], self.T["road_branch"] = (0.085, 0.088, 0.095), (0.12, 0.115, 0.11)

    def mat(self, t):
        if t not in self.mats:
            name = MAT_PREFIX + t
            m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
            c = self.T.get(t, (0.55, 0.58, 0.62))
            m.diffuse_color = (*c, 1.0)
            m.use_nodes = True
            bsdf = m.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (*c, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.85
            self.mats[t] = m
        return self.mats[t]

    def obj(self, name, me, t, smooth=False):
        o = bpy.data.objects.new(name, me)
        self.coll.objects.link(o)
        me.materials.append(self.mat(t))
        o["wondful_env_type"] = t
        if smooth:
            for p in me.polygons:
                p.use_smooth = True
        return o

    def mesh(self, name, verts, faces, t, smooth=False):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(int(i) for i in f) for f in faces])
        me.update()
        return self.obj(name, me, t, smooth)

    def group(self, label, objs):
        objs = [o for o in objs if o is not None]
        if objs:
            self.groups.append((label, objs))
        return objs

    # ---------------------------------------------------------------- terrain
    def height_raw(self, X, Y):
        T = self.S["terrain"]
        sl = math.tan(math.radians(T.get("base_slope_deg", 0.0)))
        sd = math.radians(T.get("base_slope_dir_deg", 0.0))
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        H = sl * (X * math.sin(sd) + Y * math.cos(sd))
        for b in T.get("bumps", []):
            H = H + b["height_m"] * _cosfall(np.hypot(X - b["x"], Y - b["y"]), b["radius_m"])
        for r in T.get("ridges", []):
            pts = r["points"]
            d = np.full(X.shape, 1e12)
            for a, b2 in zip(pts[:-1], pts[1:]):
                d = np.minimum(d, _seg_dist(X, Y, a["x"], a["y"], b2["x"], b2["y"]))
            H = H + r["height_m"] * _cosfall(d, r["width_m"] / 2)
        return H

    def terrain(self):
        S, T = self.S, self.S["terrain"]
        c = S["camera"]
        self.hfov = math.radians(c["hfov_deg"])
        self.tanh = math.tan(self.hfov / 2)
        far = float(max(50.0, min(20000.0, T.get("far_m", 500.0))))
        self.far = far
        NR, NC = 128, 128
        ys = np.geomspace(0.3, far, NR)
        extra = []
        for rd in S.get("roads", []):
            py = [p["y"] for p in rd.get("points", [])]
            if len(py) < 2:
                continue
            y0r, y1r = max(0.3, min(py) - 10), min(far, max(py) + 10)
            if y1r <= y0r:
                continue
            stp = max(0.8 * max(rd["width_m"], 1.0), (y1r - y0r) / 260.0)
            extra.append(np.arange(y0r, y1r, stp))
        for cf in S.get("cliffs", []):
            py = [p["y"] for p in cf.get("base_points", [])]
            if len(py) >= 2:
                extra.append(np.linspace(max(0.3, min(py) - 10), min(far, max(py) + 10), 60))
        if self.pad is not None:
            extra.append(np.linspace(max(0.3, self.pad[1] - 3 * self.pad[2]), min(far, self.pad[1] + 3 * self.pad[2]), 24))
        if extra:
            ys = np.unique(np.concatenate([ys] + extra))
            ys = ys[np.concatenate([[True], np.diff(ys) > 0.25])]
        ys = np.concatenate([[-15.0, -3.0], ys])
        tanh = self.tanh

        def half(y):
            return max(abs(y), 3.0) * tanh * 1.5 + 8.0
        grid = np.array([np.stack([np.linspace(-half(y), half(y), NC), np.full(NC, y)], 1) for y in ys])
        ysb = np.concatenate([[-15.0, -3.0], np.geomspace(0.3, far, NR)])
        gb = np.array([np.stack([np.linspace(-half(y), half(y), NC), np.full(NC, y)], 1) for y in ysb])
        Zb = self.height_raw(gb[..., 0], gb[..., 1])
        Zb_raw = Zb.copy()
        it = int(round(max(0.0, min(1.0, T.get("smoothness", 0.5))) * 8))
        for _ in range(it):
            Zp = np.pad(Zb, 1, mode="edge")
            Zb = 0.5 * Zb + 0.125 * (Zp[:-2, 1:-1] + Zp[2:, 1:-1] + Zp[1:-1, :-2] + Zp[1:-1, 2:])

        def rows(A):
            # interpolate base rows onto the final rows column by column (base / final share the x layout per row)
            return np.stack([np.interp(grid[:, 0, 1], ysb, A[:, j]) for j in range(NC)], 1)
        Z = rows(Zb)
        Z = Z + (self.height_raw(grid[..., 0], grid[..., 1]) - rows(Zb_raw)) * (1.0 - it / 8.0)
        foot = float(self.height_raw(np.array([0.0]), np.array([0.0]))[0])
        if self.interior:
            Z[:] = 0.0
            foot = 0.0
        self.foot = foot
        Z_RAW = Z.copy()
        R = grid.shape[0]
        ROAD_D = np.full(grid.shape[:2], 1e12)
        self.ROADS_C = []
        for rd in S.get("roads", []):
            if len(rd.get("points", [])) >= 2:
                cc = _catmull_xyz(rd["points"])[:, :2]
                self.ROADS_C.append((cc, max(0.5, rd["width_m"]) / 2))
                for a, b2 in zip(cc[:-1], cc[1:]):
                    ROAD_D = np.minimum(ROAD_D, _seg_dist(grid[..., 0], grid[..., 1], a[0], a[1], b2[0], b2[1])
                                        - max(0.5, rd["width_m"]) / 2)
        for rd in S.get("roads", []):
            if rd.get("follow_terrain", True) or len(rd.get("points", [])) < 2:
                continue
            C3 = _catmull_xyz(rd["points"])
            C3[:, 2] += foot
            w2 = max(0.5, rd["width_m"]) / 2 + 0.6
            sh = max(3.0, rd["width_m"] * 1.2)
            GX, GY = grid[..., 0], grid[..., 1]
            best = np.full(GX.shape, 1e12)
            bz = np.zeros(GX.shape)
            for a, b2 in zip(C3[:-1], C3[1:]):
                vx, vy = b2[0] - a[0], b2[1] - a[1]
                L2 = vx * vx + vy * vy + 1e-9
                t = np.clip(((GX - a[0]) * vx + (GY - a[1]) * vy) / L2, 0, 1)
                d = np.hypot(GX - (a[0] + t * vx), GY - (a[1] + t * vy))
                m = d < best
                best = np.where(m, d, best)
                bz = np.where(m, a[2] + t * (b2[2] - a[2]), bz)
            wgt = np.where(best <= w2, 1.0, np.where(best < w2 + sh, 0.5 * (1 + np.cos(np.pi * (best - w2) / sh)), 0.0))
            Z = Z * (1 - wgt) + bz * wgt
            rd["_graded"] = True
        # product pad: the ground under the product is level (no dune / slope through the product)
        if self.pad is not None and not self.interior:
            px, py, pr = self.pad[:3]
            zs = self._grid_h(grid, ys, Z, px, py)
            d = np.hypot(grid[..., 0] - px, grid[..., 1] - py)
            w = np.where(d <= pr, 1.0, np.where(d < pr * 2.5, 0.5 * (1 + np.cos(np.pi * (d - pr) / (pr * 1.5))), 0.0))
            Z = Z * (1 - w) + zs * w
        self.grid, self.ys, self.Z, self.R, self.NC = grid, ys, Z, R, NC
        edge = Z[-1]
        verts = [(grid[i, j, 0], grid[i, j, 1], Z[i, j]) for i in range(R) for j in range(NC)]
        for yk, wk in ((far * 4, 6.0), (far * 30, 40.0)):
            for j in range(NC):
                verts.append((grid[-1, j, 0] * wk, yk, edge[j]))
        RR = R + 2
        faces = [(i * NC + j, i * NC + j + 1, (i + 1) * NC + j + 1, (i + 1) * NC + j)
                 for i in range(RR - 1) for j in range(NC - 1)]
        label = SB.terrain_label(S)
        ter = self.mesh(f"{label}_地形", verts, faces, "terrain", smooth=True)
        ter.data.materials.append(self.mat("terrain_steep"))
        gv = np.array(verts[:R * NC]).reshape(R, NC, 3)
        a, b, c4, d4 = gv[:-1, :-1], gv[:-1, 1:], gv[1:, 1:], gv[1:, :-1]
        n = np.cross(c4 - a, d4 - b)
        nz = np.abs(n[..., 2]) / (np.linalg.norm(n, axis=-1) + 1e-12)
        cx, cy = (a[..., 0] + c4[..., 0]) / 2, (a[..., 1] + c4[..., 1]) / 2
        carved = np.abs(Z - Z_RAW) > 0.3
        cf = carved[:-1, :-1] | carved[:-1, 1:] | carved[1:, 1:] | carved[1:, :-1]
        gr = np.concatenate([grid, Z_RAW[..., None]], -1)
        ar, br, cr, dr = gr[:-1, :-1], gr[:-1, 1:], gr[1:, 1:], gr[1:, :-1]
        nr = np.cross(cr - ar, dr - br)
        nzr = np.abs(nr[..., 2]) / (np.linalg.norm(nr, axis=-1) + 1e-12)
        steep = nz < math.cos(math.radians(40))
        steep_raw = nzr < math.cos(math.radians(40))
        mi = np.where(steep & (steep_raw | ~cf), 1, 0)
        for zn in S.get("surface_zones", []):
            if len(zn.get("points", [])) < 3:
                continue
            ter.data.materials.append(self.mat("zone_" + zn.get("kind", "grass")))
            mi = np.where(_inpoly(cx, cy, zn["points"]), len(ter.data.materials) - 1, mi)
        allm = np.zeros(len(ter.data.polygons), int)
        allm[:mi.size] = mi.ravel()
        ter.data.polygons.foreach_set("material_index", allm.tolist())
        ter["wondful_terrain"] = True
        ter.data.update()
        self.TBVH = BVHTree.FromPolygons([tuple(v) for v in verts], faces)
        self.terrain_obj = ter
        self.group(label, [ter])
        return ter

    @staticmethod
    def _grid_h(grid, ys, Z, x, y):
        R = grid.shape[0]
        i = int(np.clip(np.searchsorted(ys, y) - 1, 0, R - 2))
        ty = min(max((y - ys[i]) / max(ys[i + 1] - ys[i], 1e-6), 0.0), 1.0)
        r0 = float(np.interp(x, grid[i, :, 0], Z[i]))
        r1 = float(np.interp(x, grid[i + 1, :, 0], Z[i + 1]))
        return (1 - ty) * r0 + ty * r1

    def th(self, x, y):
        h = self.TBVH.ray_cast(Vector((float(x), float(y), 1e5)), Vector((0, 0, -1)))
        if h[0] is not None:
            return float(h[0].z)
        return self._grid_h(self.grid, self.ys, self.Z, float(x), float(y))

    # ---------------------------------------------------------------- camera (local)
    def camera(self):
        c = self.S["camera"]
        loc = Vector((0.0, 0.0, self.th(0.0, 0.0) + c["height_m"]))
        rot = Euler((math.radians(90 + c["pitch_deg"]), math.radians(c.get("roll_deg", 0.0)), 0.0), "XYZ")
        self.cam_local = Matrix.Translation(loc) @ rot.to_matrix().to_4x4()
        self.CAMPOS = loc.copy()
        self.R3 = rot.to_matrix()
        self.lens = SB.lens_from_hfov(c["hfov_deg"])

    def ray(self, u, v):
        sx = (2 * u - 1) * self.tanh
        sy = (1 - 2 * v) * self.tanh * self.hw
        return (self.R3 @ Vector((sx, sy, -1.0))).normalized()

    # ---------------------------------------------------------------- roads
    def roads(self):
        S = self.S
        nroad = 0
        for rd in S.get("roads", []):
            if len(rd.get("points", [])) < 2:
                continue
            C = _catmull_xyz(rd["points"])[:, :2]
            w = max(0.5, rd["width_m"])
            k = rd.get("kind", "road")
            mk = k
            if k == "road":
                mk = "road" if nroad == 0 else "road_branch"
                nroad += 1
            lift = {"road": 0.06, "path": 0.05, "sidewalk": 0.15, "river": 0.02, "plaza": 0.03}.get(k, 0.05)
            across = [-0.5, -0.25, 0, 0.25, 0.5]
            GZ = None
            if rd.get("_graded"):
                GZ = _catmull_xyz(rd["points"])[:, 2] + self.foot
            vs, fs = [], []
            for i, p in enumerate(C):
                a = C[max(i - 1, 0)]
                b = C[min(i + 1, len(C) - 1)]
                t = b - a
                t = t / (np.linalg.norm(t) + 1e-9)
                nrm = np.array([t[1], -t[0]])
                for s in across:
                    q = p + nrm * s * w
                    th = self.th(q[0], q[1])
                    zz = th if GZ is None else max(GZ[min(i, len(GZ) - 1)], th)
                    dc = math.sqrt((q[0] - self.CAMPOS.x) ** 2 + (q[1] - self.CAMPOS.y) ** 2 + (zz - self.CAMPOS.z) ** 2)
                    vs.append((q[0], q[1], zz + lift + 0.0025 * dc))
            na = len(across)
            for i in range(len(C) - 1):
                for j in range(na - 1):
                    fs.append((i * na + j, i * na + j + 1, (i + 1) * na + j + 1, (i + 1) * na + j))
            label = SB.element_label("road", rd)
            o = self.mesh(f"{label}_路面", vs, fs, mk, smooth=True)
            o["wondful_material"] = rd.get("surface", "")
            self.group(label, [o])

    # ---------------------------------------------------------------- sea
    def sea(self):
        T = self.S["terrain"]
        self.HAS_SEA = bool(T.get("has_sea")) and not self.interior
        if self.HAS_SEA:
            zsea = self.foot + float(T.get("sea_level_m", -2.0))
            Wd = self.far * 60
            o = self.mesh("海面_水面", [(-Wd, -60, zsea), (Wd, -60, zsea), (Wd, self.far * 40, zsea),
                                     (-Wd, self.far * 40, zsea)], [(0, 1, 2, 3)], "water_plane")
            self.group("海面", [o])

    # ---------------------------------------------------------------- distant bands (+ 360° wrap)
    def bands(self):
        for bd in self.S.get("mountain_bands", []):
            prof = sorted(bd.get("profile", []), key=lambda p: p["x_frac"])
            if len(prof) < 2:
                continue
            D = max(50.0, bd.get("distance_m", 500.0))
            dep = max(D * 0.05, bd.get("depth_m", D * 0.3))
            kind = bd.get("kind", "rocky")
            label = SB.element_label("band", bd)
            if kind == "sea":
                if self.HAS_SEA:
                    continue
                zs = []
                for p in prof:
                    d = self.ray(p["x_frac"], max(p["y_frac"], 0.0))
                    if d.y > 1e-6:
                        zs.append((self.CAMPOS + d * ((D - self.CAMPOS.y) / d.y)).z)
                zsea = float(np.median(zs)) if zs else self.foot - 2.0
                zsea = min(zsea, self.CAMPOS.z - 0.5)
                Wd = self.far * 60
                o = self.mesh(f"{label}_远景", [(-Wd, D * 0.6, zsea), (Wd, D * 0.6, zsea), (Wd, self.far * 40, zsea),
                                               (-Wd, self.far * 40, zsea)], [(0, 1, 2, 3)], "band_sea")
                self.group(label, [o])
                continue
            if prof[0]["x_frac"] > -0.05:
                prof = [{"x_frac": -0.1, "y_frac": prof[0]["y_frac"]}] + prof
            if prof[-1]["x_frac"] < 1.05:
                prof = prof + [{"x_frac": 1.1, "y_frac": prof[-1]["y_frac"]}]
            xf = np.array([p["x_frac"] for p in prof])
            yf = np.array([p["y_frac"] for p in prof])
            us = np.linspace(xf[0], xf[-1], 160)
            vv = np.interp(us, xf, yf)
            if kind not in ("rocky", "snowy"):
                vv = np.convolve(np.pad(vv, 4, mode="edge"), np.ones(9) / 9, mode="valid")
            crest = []
            for u, v in zip(us, vv):
                d = self.ray(u, v)
                if abs(d.y) < 1e-6:
                    continue
                t = (D - self.CAMPOS.y) / d.y
                if t <= 0:
                    continue
                crest.append(self.CAMPOS + d * t)
            if len(crest) < 2:
                self.warn.append(f"远景 {label} 不在画面内")
                continue
            foot_z = min(float(np.min(self.Z)), 0.0) - 30.0
            rows = [(-dep * 0.5, 0.0), (-dep * 0.3, 0.55), (-dep * 0.12, 0.88), (0.0, 1.0), (dep * 0.15, 0.85), (dep * 0.5, 0.0)]
            vs, fs = [], []
            for (dy, f) in rows:
                for p in crest:
                    sx = (p.y + dy - self.CAMPOS.y) / max(p.y - self.CAMPOS.y, 1e-6)
                    vs.append(Vector((p.x * sx, p.y + dy, foot_z + (p.z - foot_z) * f)))
            n = len(crest)
            for i in range(len(rows) - 1):
                for j in range(n - 1):
                    fs.append((i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j))
            # curve the shell: every vertex slides along its camera ray to horizontal range = forward depth (the
            # reference view is unchanged), then mirrored copies wrap it to 360° for other camera angles
            C1 = self.CAMPOS
            out = []
            for p in vs:
                d = p - C1
                hr = math.hypot(d.x, d.y)
                s = (d.y / hr) if (hr > 1e-6 and d.y > 0.5) else 1.0
                out.append(C1 + d * s)
            smooth = kind not in ("rocky", "snowy")
            t_ = "band_" + kind
            main = self.mesh(f"{label}_远景", out, fs, t_, smooth=smooth)
            objs = [main]
            bear = [math.atan2(p.x - C1.x, p.y - C1.y) for p in out]
            b0, b1 = min(bear), max(bear)
            Dw = b1 - b0
            rng = [math.hypot(p.x - C1.x, p.y - C1.y) for p in out]
            n_side = int(math.ceil((math.pi - Dw / 2) / Dw)) if Dw > 0.05 else 0
            for side in (1, -1):
                for k in range(1, min(n_side, 12) + 1):
                    vv2 = []
                    for p, b, r in zip(out, bear, rng):
                        if side > 0:
                            nb = b1 + (k - 1) * Dw + ((b1 - b) if k % 2 else (b - b0))
                        else:
                            nb = b0 - (k - 1) * Dw - ((b - b0) if k % 2 else (b1 - b))
                        vv2.append((C1.x + r * math.sin(nb), C1.y + r * math.cos(nb), p.z))
                    fcs = [tuple(reversed(f)) for f in fs] if k % 2 else fs
                    o = self.mesh(f"{label}_环绕", vv2, fcs, t_, smooth=smooth)
                    o["wondful_sb_wrap"] = 1
                    objs.append(o)
            self.group(label, objs)

    # ---------------------------------------------------------------- cliffs
    def cliff(self, cf):
        pts = cf.get("base_points", [])
        if len(pts) < 2:
            return
        C3 = _catmull_xyz(pts)
        C3[:, 2] += self.foot
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C3[:, :2], axis=0), axis=1))])
        n = int(np.clip(L[-1] / 1.5, 8, 400))
        tt = np.linspace(0, L[-1], n)
        C3 = np.stack([np.interp(tt, L, C3[:, k]) for k in range(3)], 1)
        top = self.foot + float(cf.get("top_z_m", 30.0))
        side = 1.0 if cf.get("side", "left") == "right" else -1.0
        thick = max(4.0, float(cf.get("thickness_m", 25.0)))
        lean = math.tan(math.radians(max(-15.0, min(45.0, cf.get("lean_deg", 5.0)))))
        rough = max(0.0, min(1.0, cf.get("roughness", 0.6)))
        rng = np.random.default_rng(len(cf.get("id", "c")) * 17 + 3)
        NH = 24
        ph = rng.uniform(0, 6.28, 4)
        front, back = [], []
        for i in range(n):
            a, b = C3[max(i - 1, 0)], C3[min(i + 1, n - 1)]
            t = (b - a)[:2]
            t = t / (np.linalg.norm(t) + 1e-9)
            nr = np.array([t[1], -t[0]]) * side
            zb = C3[i, 2] - 1.5
            colf, colb = [], []
            for k in range(NH + 1):
                f = k / NH
                z = zb + f * (top - zb)
                hz = z - zb
                nz = rough * (0.9 * (0.5 + 0.5 * math.sin(z * 0.55 + ph[0]))
                              + 0.7 * (0.5 + 0.5 * math.sin(tt[i] * 0.21 + z * 0.13 + ph[1])) + 0.5 * rng.uniform(0, 1))
                off = lean * hz + nz + (0.0 if k > 0 else 0.4)
                colf.append((C3[i, 0] + nr[0] * off, C3[i, 1] + nr[1] * off, z))
                colb.append((C3[i, 0] + nr[0] * (off + thick), C3[i, 1] + nr[1] * (off + thick), z))
            front.append(colf)
            back.append(colb)
        vs = [v for col in front for v in col] + [v for col in back for v in col]
        m = NH + 1
        B = n * m
        fs = []
        for i in range(n - 1):
            for k in range(NH):
                fs.append((i * m + k, (i + 1) * m + k, (i + 1) * m + k + 1, i * m + k + 1))
                fs.append((B + i * m + k, B + i * m + k + 1, B + (i + 1) * m + k + 1, B + (i + 1) * m + k))
            fs.append((i * m + NH, (i + 1) * m + NH, B + (i + 1) * m + NH, B + i * m + NH))
        for i in (0, n - 1):
            for k in range(NH):
                fs.append((i * m + k, i * m + k + 1, B + i * m + k + 1, B + i * m + k))
        label = SB.element_label("cliff", cf)
        o = self.mesh(f"{label}_崖", vs, fs, "cliff")
        o["wondful_material"] = cf.get("surface", "")
        self.group(label, [o])

    def cliffs(self):
        for cf in self.S.get("cliffs", []):
            try:
                self.cliff(cf)
            except Exception as exc:
                self.warn.append(f"崖壁 {cf.get('id')}: {exc}")

    # ---------------------------------------------------------------- primitives
    def _shape_mesh(self, name, sh, w, d, h, rh):
        bm = bmesh.new()
        if sh == "box":
            bmesh.ops.create_cube(bm, size=1.0)
            bmesh.ops.scale(bm, vec=(w, d, h), verts=bm.verts)
            bmesh.ops.translate(bm, vec=(0, 0, h / 2), verts=bm.verts)
        elif sh == "gable":
            v = [(-w / 2, -d / 2, 0), (w / 2, -d / 2, 0), (w / 2, d / 2, 0), (-w / 2, d / 2, 0),
                 (-w / 2, -d / 2, h), (w / 2, -d / 2, h), (w / 2, d / 2, h), (-w / 2, d / 2, h),
                 (-w / 2, 0, h + rh), (w / 2, 0, h + rh)]
            f = [(0, 3, 2, 1), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 9, 5), (3, 0, 4, 8, 7), (4, 5, 9, 8), (6, 7, 8, 9)]
            bm.free()
            me = bpy.data.meshes.new(name)
            me.from_pydata(v, [], f)
            me.update()
            return me
        elif sh in ("cylinder", "cone"):
            bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=20, radius1=0.5,
                                  radius2=0.5 if sh == "cylinder" else 0.0, depth=1.0)
            bmesh.ops.scale(bm, vec=(w, d, h), verts=bm.verts)
            bmesh.ops.translate(bm, vec=(0, 0, h / 2), verts=bm.verts)
        elif sh == "sphere":
            bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=10, radius=0.5)
            bmesh.ops.scale(bm, vec=(w, d, h), verts=bm.verts)
            bmesh.ops.translate(bm, vec=(0, 0, h / 2), verts=bm.verts)
        else:  # dome
            bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1.0)
            bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -1e-4], context="VERTS")
            bmesh.ops.scale(bm, vec=(w / 2, d / 2, h), verts=bm.verts)
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        return me

    def make(self, p, x, y, label, hscale=1.0):
        t, sh = p["type"], p["shape"]
        if sh == "cone" and t != "tree":
            sh = "box"
        w, d = max(p["w"], 0.05), max(p["d"], 0.05)
        h = max(p["h"], 0.05) * hscale
        rz = math.radians(p.get("rot_z_deg", 0.0))
        cs, sn = math.cos(rz), math.sin(rz)
        samp = [(x + a * w / 2 * cs - b * d / 2 * sn, y + a * w / 2 * sn + b * d / 2 * cs)
                for a in (-1, 0, 1) for b in (-1, 0, 1)]
        z = min(self.th(sx, sy) for sx, sy in samp) + p.get("z_offset", 0.0)
        if t in ("wall", "fence", "pole", "sign") and min(w, d) < 1.0:
            z = self.th(x, y) + p.get("z_offset", 0.0)
        me = self._shape_mesh(f"{label}", sh, w, d, h, max(0.0, p.get("roof_h", 0.0)) * hscale)
        o = self.obj(label, me, t, smooth=sh in ("sphere", "dome", "cylinder", "cone"))
        o.location = (x, y, z)
        o.rotation_euler = (0, 0, rz)
        o["wondful_material"] = p.get("material", "")
        return o

    def primitives(self):
        for p in self.S.get("primitives", []):
            if p["type"] == "water_plane" and self.HAS_SEA:
                continue
            if p["type"] == "pole" and p["shape"] == "box" and max(p["w"], p["d"]) < 1.0 and p["h"] > 1.0:
                p["shape"] = "cylinder"
                p["w"] = p["d"] = max(0.12, min(p["w"], p["d"], 0.4))
            if p["type"] == "ceiling" and p.get("z_offset", 0) < 1.0:
                if p["h"] > 1.5:
                    p["z_offset"], p["h"] = p["h"], 0.2
                else:
                    p["z_offset"] = 2.7
            label = SB.element_label("primitive", p, self.interior)
            n = int(max(1, min(40, p.get("repeat", 1))))
            objs = []
            skipped = 0
            for k in range(n):
                x = p["x"] + k * p.get("repeat_dx", 0.0)
                y = p["y"] + k * p.get("repeat_dy", 0.0)
                if self.pad is not None and p["type"] not in ("rug", "ceiling", "water_plane") \
                        and p.get("z_offset", 0.0) < self.pad[4] \
                        and _rect_dist(self.pad[0], self.pad[1], dict(p, x=x, y=y)) < self.pad[3]:
                    skipped += 1
                    continue
                jit = 1.0 + 0.15 * math.sin(k * 2.399 + len(p["id"])) if (n > 1 and p["type"] in ("tree", "bush")) else 1.0
                objs.append(self.make(p, x, y, label, jit))
            if skipped:
                self.warn.append(f"移除了与产品重叠的{label}（{p.get('id', '')}）")
            self.group(label, objs)

    # ---------------------------------------------------------------- foliage
    def _lumpy(self, name, lump, seed):
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
        rng = np.random.default_rng(seed)
        ph = rng.uniform(0, 6.28, 3)
        for v in bm.verts:
            c = v.co
            n = math.sin(3.1 * c.x + ph[0]) * math.sin(2.7 * c.y + ph[1]) * math.sin(3.4 * c.z + ph[2])
            v.co = c * (1.0 + lump * 0.28 * n + lump * 0.10 * rng.uniform(-1, 1))
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        return me

    def _trunk(self, label, x, y, z0, height, tw, lean=0.0):
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=12, radius1=tw / 2, radius2=tw / 2, depth=height)
        me = bpy.data.meshes.new(label)
        bm.to_mesh(me)
        bm.free()
        o = self.obj(f"{label}_树干", me, "trunk", smooth=True)
        o.location = (x + lean * height / 2, y, z0 + height / 2)
        o.rotation_euler = (0, math.atan(lean), 0)
        return o

    def foliage_volume(self, f, label, rng):
        x, y = f["x"], f["y"]
        cw, cd_, ch = max(0.5, f["crown_w"]), max(0.5, f["crown_d"]), max(0.5, f["crown_h"])
        base_h = max(0.0, f.get("crown_base_h", 0.0))
        lump = max(0.0, min(1.0, f.get("lumpiness", 0.5)))
        lean = math.tan(math.radians(max(-35.0, min(35.0, f.get("lean_deg", 0.0)))))
        kind = f["kind"]
        z0 = self.th(x, y)
        objs = []
        n = int(6 + lump * 8) if kind != "hedge" else int(5 + cw / max(cd_, 1.0) * 2)
        for i in range(min(n, 18)):
            if kind == "conifer":
                hz = rng.uniform(0.05, 0.95)
                taper = (1 - hz) * 0.95 + 0.08
                r = 0.5 * min(cw, cd_) * taper * rng.uniform(0.75, 1.0)
                px, py = rng.uniform(-0.12, 0.12) * cw * taper, rng.uniform(-0.12, 0.12) * cd_ * taper
                pz = base_h + hz * ch
                sx, sy, sz = r, r * cd_ / max(cw, 1e-3), r * 0.9
            else:
                while True:
                    u = rng.uniform(-1, 1, 3)
                    if (u ** 2).sum() <= 1:
                        break
                rr = rng.uniform(0.32, 0.5)
                px, py = u[0] * cw * (0.5 - rr * 0.5), u[1] * cd_ * (0.5 - rr * 0.5)
                pz = base_h + ch * (0.5 + u[2] * (0.5 - rr * 0.5))
                sx, sy, sz = rr * cw, rr * cd_, rr * ch
                if kind == "palm":
                    pz = base_h + ch * rng.uniform(0.7, 1.0)
                    sz = rr * ch * 0.5
            o = self.obj(f"{label}_树冠", self._lumpy(label, lump, int(rng.integers(1e6))), "foliage_" + kind, smooth=True)
            o.location = (x + px + lean * pz, y + py, z0 + pz)
            o.scale = (sx, sy, sz)
            o["wondful_material"] = f.get("material", "")
            objs.append(o)
        if base_h > 0.3 and kind not in ("hedge", "shrub", "forest_edge"):
            objs.append(self._trunk(label, x, y, z0, base_h + ch * 0.35, max(0.15, f.get("trunk_w", 0.4)), lean))
        return objs

    def _raster(self, P, su, sv):
        u0, u1 = P[:, 0].min(), P[:, 0].max()
        v0, v1 = P[:, 1].min(), P[:, 1].max()
        nu = int(np.clip((u1 - u0) / su, 4, 400)) + 1
        nv = int(np.clip((v1 - v0) / sv, 4, 400)) + 1
        U, Vv = np.meshgrid(np.linspace(u0, u1, nu), np.linspace(v0, v1, nv))
        inside = np.zeros(U.shape, bool)
        xp, yp = P[:, 0], P[:, 1]
        j = len(P) - 1
        for i in range(len(P)):
            inside ^= ((yp[i] > Vv) != (yp[j] > Vv)) & (U < (xp[j] - xp[i]) * (Vv - yp[i]) / (yp[j] - yp[i] + 1e-12) + xp[i])
            j = i
        return U, Vv, inside, (u0, u1)

    @staticmethod
    def _dmap(m, cap=80):
        dist = np.zeros(m.shape)
        cur = m.copy()
        k = 0
        while cur.any() and k < cap:
            dist[cur] += 1
            pad = np.pad(cur, 1)
            cur = cur & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
            k += 1
        return dist

    def foliage_outline(self, f, label):
        """Traced silhouette → one lumpy shell at depth y (projected outline = the traced one)."""
        P = np.array([(p["x_frac"], p["y_frac"]) for p in f["outline"]], float)
        D = max(1.0, f["y"])
        dep = max(0.5, min(f.get("crown_d", 4.0), D * 0.8))
        yb = D + dep / 2
        U, Vv, inside, (u0, u1) = self._raster(P, 0.005, 0.005)
        if inside.sum() < 4:
            return []
        dist = self._dmap(inside, 60)
        lump = max(0.0, min(1.0, f.get("lumpiness", 0.5)))
        rng = np.random.default_rng(len(f.get("id", "f")) * 31 + 5)
        ph = rng.uniform(0, 6.28, 6)
        sc = (u1 - u0) + 1e-3
        noise = (np.sin(U / sc * 9 + ph[0]) * np.sin(Vv / sc * 8 + ph[1])
                 + 0.6 * np.sin(U / sc * 21 + ph[2]) * np.sin(Vv / sc * 19 + ph[3]))
        edge = np.clip(dist / 14.0, 0, 1) ** 0.6
        push = dep * (0.85 * edge + 0.18 * lump * noise * edge)
        vid = -np.ones(U.shape, int)
        verts = []
        for (r, c) in zip(*np.nonzero(inside)):
            d = self.ray(float(U[r, c]), float(Vv[r, c]))
            if d.y < 1e-4:
                d = Vector((d.x, 1e-4, d.z)).normalized()
            p = self.CAMPOS + d * ((yb - self.CAMPOS.y) / d.y)
            p = p - d * float(push[r, c])
            vid[r, c] = len(verts)
            verts.append(tuple(p))
        faces = []
        nv, nu = U.shape
        for r in range(nv - 1):
            for c in range(nu - 1):
                q = (vid[r, c], vid[r, c + 1], vid[r + 1, c + 1], vid[r + 1, c])
                if min(q) >= 0:
                    faces.append(q)
        if not faces:
            return []
        o = self.mesh(f"{label}_树冠", verts, faces, "foliage_" + f["kind"], smooth=True)
        o["wondful_material"] = f.get("material", "")
        return [o]

    def _on_road(self, x, y, margin=0.4):
        for c2, hw in self.ROADS_C:
            if np.min(np.hypot(c2[:, 0] - x, c2[:, 1] - y)) < hw + margin:
                return True
        return False

    def foliage_crowns(self, f, label):
        """Wide traced outline → separate lumpy crowns packed inside the silhouette, each with its own trunk."""
        P = np.array([(p["x_frac"], p["y_frac"]) for p in f["outline"]], float)
        D = max(1.0, f["y"])
        su = 0.005
        sv = su / self.hw
        U, Vv, inside, _ = self._raster(P, su, sv)
        inside &= (U > -0.05) & (U < 1.05) & (Vv > -0.05) & (Vv < 1.05)
        if inside.sum() < 8:
            return []
        avail = inside.copy()
        covered = np.zeros(inside.shape, bool)
        crowns = []
        cap = 0.08 / su
        RR, CC = np.mgrid[0:inside.shape[0], 0:inside.shape[1]]
        while len(crowns) < 18:
            dist = self._dmap(avail)
            if dist.max() < 3:
                break
            r, c = np.unravel_index(np.argmax(dist), dist.shape)
            rad = min(dist[r, c], cap)
            crowns.append((float(U[r, c]), float(Vv[r, c]), rad * su))
            dd = np.hypot(RR - r, CC - c)
            covered |= (dd <= rad) & inside
            avail &= dd > rad * 1.12
            if covered.sum() > 0.86 * inside.sum():
                break
        lump = max(0.3, min(1.0, f.get("lumpiness", 0.6)))
        objs = []
        for i, (u, v, rfr) in enumerate(crowns):
            d = self.ray(u, v)
            if d.y < 1e-4:
                continue
            p = self.CAMPOS + d * ((D - self.CAMPOS.y) / d.y)
            rw = rfr * 2 * self.tanh * (D - self.CAMPOS.y)
            o = self.obj(f"{label}_树冠", self._lumpy(label, lump, 100 + i), "foliage_" + f["kind"], smooth=True)
            o.location = tuple(p)
            o.scale = (rw * 1.04, rw * 0.8, rw * 0.8)
            o["wondful_material"] = f.get("material", "")
            objs.append(o)
            gz = self.th(p.x, p.y)
            top = p.z - rw * 0.6
            if top - gz > 0.8 and not self._on_road(p.x, p.y, 0.6):
                objs.append(self._trunk(label, p.x, p.y, gz, top - gz, max(0.15, min(0.8, f.get("trunk_w", 0.35)))))
        return objs

    def foliage(self):
        rng = np.random.default_rng(7)
        for f in self.S.get("foliage_masses", []):
            label = SB.element_label("foliage", f)
            f.setdefault("kind", "deciduous")
            for k, dv in (("x", 0.0), ("y", 10.0), ("crown_w", 4.0), ("crown_d", 4.0), ("crown_h", 4.0)):
                f[k] = float(f.get(k, dv) or dv)
            try:
                out = f.get("outline") or []
                if len(out) >= 4 and f["kind"] in ("deciduous", "conifer", "palm") and (
                        max(p["x_frac"] for p in out) - min(p["x_frac"] for p in out) > 0.15):
                    objs = self.foliage_crowns(f, label) or self.foliage_outline(f, label)
                elif len(out) >= 4:
                    objs = self.foliage_outline(f, label) or self.foliage_volume(f, label, rng)
                else:
                    objs = self.foliage_volume(f, label, rng)
                self.group(label, objs)
            except Exception as exc:
                self.warn.append(f"{label}: {exc}")

    # ---------------------------------------------------------------- outer ground disc (other camera angles)
    def ground_disc(self, cx, cy, zc):
        if self.interior:
            return
        radii = [0.0] + [2.0 * (1.16 ** i) for i in range(48)]
        radii = [r for r in radii if r < 1500] + [1500.0]
        nseg = 96
        verts, faces = [], []
        for r in radii:
            for j in range(nseg if r > 0 else 1):
                a = 2 * math.pi * j / nseg
                x, y = cx + r * math.sin(a), cy + r * math.cos(a)
                h = self.TBVH.ray_cast(Vector((x, y, 1e5)), Vector((0, 0, -1)))
                verts.append((x, y, (h[0].z - 0.05) if h[0] is not None else (zc - 0.05)))
        for j in range(nseg):
            faces.append((0, 1 + j, 1 + (j + 1) % nseg))
        for i in range(1, len(radii) - 1):
            b0 = 1 + (i - 1) * nseg
            b1 = 1 + i * nseg
            for j in range(nseg):
                faces.append((b0 + j, b1 + j, b1 + (j + 1) % nseg, b0 + (j + 1) % nseg))
        o = self.mesh("地面_外圈", verts, faces, "terrain", smooth=True)
        o["wondful_sb_wrap"] = 1
        # part of the terrain block (same name in the prompt)
        for label, objs in self.groups:
            if objs and objs[0] is getattr(self, "terrain_obj", None):
                objs.append(o)
                return
        self.group(SB.terrain_label(self.S), [o])

    def surface_at(self, x, y):
        """Top ground surface (terrain or road ribbon) at a local point."""
        best = self.th(x, y)
        for label, objs in self.groups:
            for o in objs:
                if o.get("wondful_env_type") in ("road", "road_branch", "path", "sidewalk", "plaza") and o.type == "MESH":
                    me = o.data
                    bvh = BVHTree.FromPolygons([tuple(v.co) for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
                    h = bvh.ray_cast(Vector((float(x), float(y), 1e5)), Vector((0, 0, -1)))
                    if h[0] is not None:
                        best = max(best, float(h[0].z))
        return best

    def run(self):
        self.HAS_SEA = False
        self.terrain()
        self.camera()
        self.roads()
        self.sea()
        self.bands()
        self.cliffs()
        self.primitives()
        self.foliage()
        return self


def _assemble(b, world):
    """Move every block to world space, parent each group to its first object, store names and labels."""
    from . import massing as MS
    bpy.context.view_layer.update()      # location / rotation / scale set above → matrix_world
    for label, objs in b.groups:
        for o in objs:
            o.matrix_world = world @ o.matrix_world
    bpy.context.view_layer.update()
    roots = []
    n = 0
    for label, objs in b.groups:
        n += 1
        root = objs[0]
        for i, o in enumerate(objs):
            o.name = f"{label}_{n:02d}" if i == 0 else f"{label}_{n:02d}_{i:02d}"
            o[SB.GROUP_PROP] = n
            MS.store_label(o, label)
            if b.interior and o.get("wondful_env_type") in NO_SHADOW_INTERIOR:
                try:
                    o.visible_shadow = False   # the clay sun still lights the room
                except Exception:
                    pass
            if o is not root:
                mw = o.matrix_world.copy()
                o.parent = root
                o.matrix_world = mw
        roots.append(root)
    return roots


def build(context, data, *, product_collection=None, parent_collection=None, set_camera=True, frame=None,
          massing_objects=()):
    """Build (replacing any earlier one) and return an info dict for the panel note / tests."""
    scene = context.scene
    S = SB.normalize(data)
    frame = frame or SB.frame_of(data)
    remove_existing()
    coll = bpy.data.collections.new(SB.COLLECTION_NAME)
    coll["wondful_scene_blockout"] = 1
    (parent_collection or scene.collection).children.link(coll)
    prod = product_info(product_collection, exclude=massing_objects)
    spot = S["product_spot"]
    pad = None
    moved = ""
    if prod is not None:
        # the analysed spot can be too tight for this product; nudge it (≤ 2.5 × its radius, still in frame) off standing blocks
        # before anything is removed
        free = SB.free_spot(S.get("primitives", []), spot["x"], spot["y"], prod["half_diag"] * 0.85, prod["dims"][2],
                           hfov_deg=S["camera"]["hfov_deg"])
        if free is not None and free[2] > 0:
            spot["x"], spot["y"] = free[0], free[1]
            moved = f"产品位置挪了 {free[2]:.2f} m，避开周围体块"
        # (x, y, level-ground radius, remove-obstacle radius, product height), analysis coordinates
        pad = (spot["x"], spot["y"], prod["half_diag"] + 0.6, prod["half_diag"] * 0.85, prod["dims"][2])
    b = _Builder(S, coll, frame, pad=pad).run()
    sx, sy = spot["x"], spot["y"]
    zs = b.surface_at(sx, sy)
    # yaw: the user's own camera direction when the camera is not set from the analysis, else the product's long
    # axis along the analysed spot heading
    yaw = 0.0
    own = camera_yaw(scene.camera) if (scene.camera is not None and scene.camera.name != SB.CAMERA_NAME) else None
    if not set_camera and own is not None:
        yaw = own
    elif prod is not None and prod["axis"] is not None:
        yaw = SB.align_yaw(prod["axis"], spot.get("rot_z_deg", 0.0))
    elif own is not None:
        yaw = own
    Rz = Matrix.Rotation(yaw, 4, "Z")
    base = prod["base"] if prod is not None else Vector((0.0, 0.0, 0.0))
    local_spot = Vector((sx, sy, zs))
    world = Matrix.Translation(base - (Rz @ local_spot)) @ Rz
    b.ground_disc(sx, sy, zs)
    roots = _assemble(b, world)
    note = SB.scene_note(S)
    if note and getattr(b, "terrain_obj", None) is not None:
        b.terrain_obj[SB.SCENE_NOTE_PROP] = note[:400]
    cam_world = world @ b.cam_local
    cam_obj = None
    if set_camera:
        cam_obj = bpy.data.objects.get(SB.CAMERA_NAME)
        if cam_obj is None or cam_obj.type != "CAMERA":
            cd = bpy.data.cameras.new(SB.CAMERA_NAME)
            cam_obj = bpy.data.objects.new(SB.CAMERA_NAME, cd)
            scene.collection.objects.link(cam_obj)
        elif cam_obj.name not in scene.objects:
            scene.collection.objects.link(cam_obj)
        cd = cam_obj.data
        cd.type = "PERSP"
        cd.sensor_fit = "HORIZONTAL"
        cd.sensor_width = 36.0
        cd.lens = b.lens
        cd.shift_x = cd.shift_y = 0.0
        cd.clip_start = 0.1
        cd.clip_end = 100000.0
        cam_obj.matrix_world = cam_world
        cam_obj["wondful_sb_camera"] = 1
        scene.camera = cam_obj
    stored = dict(S)
    stored["_frame"] = {"w": frame[0], "h": frame[1]}
    txt = bpy.data.texts.get(SB.TEXT_NAME) or bpy.data.texts.new(SB.TEXT_NAME)
    txt.clear()
    txt.write(json.dumps({k: v for k, v in stored.items() if not str(k).startswith("_graded")}, ensure_ascii=False,
                         indent=1, default=str))
    coll["wondful_sb_yaw_deg"] = round(math.degrees(yaw), 2)
    context.view_layer.update()
    hw = frame[1] / float(frame[0])
    rx, ry = scene.render.resolution_x, scene.render.resolution_y
    notes = ([moved] if moved else []) + list(b.warn)
    if set_camera and rx and ry and abs((ry / float(rx)) / hw - 1.0) > 0.05:
        notes.append(f"渲染画幅 {rx}x{ry} 与参考图 {frame[0]}x{frame[1]} 不同，上下取景会有差异")
    c = S["camera"]
    hz = SB.horizon_from_pitch(c["pitch_deg"], c["hfov_deg"], hw)
    return {"collection": coll.name, "blocks": len(roots), "objects": len(coll.all_objects), "kind": S["scene_kind"],
            "camera": bool(cam_obj), "camera_name": cam_obj.name if cam_obj else "", "lens": round(b.lens, 3),
            "cam_world": [list(r) for r in cam_world], "yaw_deg": round(math.degrees(yaw), 2), "frame": list(frame),
            "labels": [lab for lab, _o in b.groups], "notes": notes, "scene_note": note,
            "horizon_check": round(hz - c["horizon_y_frac"], 3), "spot_world": list(world @ local_spot),
            # 1.2.5 细节粗模: analysis → world matrix, camera height in the local frame, product spot + pad
            "world": [list(r) for r in world], "cam_local_z": float(b.CAMPOS.z), "spot_local": [sx, sy, zs],
            "pad_radius": float(pad[2]) if pad else 2.5}


# ------------------------------------------------------------------------------------------- preview render

def render_preview(context, info, path, width=480):
    """Render the blockout alone from the analysis camera (own tints, background transparent) → RGBA PNG.

    A temporary scene shares only the blockout collection, so the user's scene, product and render settings are
    untouched.  Background Blender uses Cycles (no GL); the UI uses Workbench (about a second)."""
    coll = bpy.data.collections.get(info["collection"])
    if coll is None:
        raise RuntimeError("场景粗模集合不存在")
    fw, fh = info["frame"]
    tmp = bpy.data.scenes.new("_wondful_sb_preview")
    made = []
    try:
        tmp.collection.children.link(coll)
        cd = bpy.data.cameras.new("_wondful_sb_preview_cam")
        cd.sensor_fit, cd.sensor_width, cd.lens = "HORIZONTAL", 36.0, info["lens"]
        cd.clip_start, cd.clip_end = 0.1, 100000.0
        cam = bpy.data.objects.new("_wondful_sb_preview_cam", cd)
        made += [cam, cd]
        tmp.collection.objects.link(cam)
        cam.matrix_world = Matrix(info["cam_world"])
        tmp.camera = cam
        world = bpy.data.worlds.new("_wondful_sb_preview_world")
        made.append(world)
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (0.80, 0.84, 0.90, 1)
            bg.inputs[1].default_value = 0.6
        tmp.world = world
        sd = bpy.data.lights.new("_wondful_sb_preview_sun", "SUN")
        sd.energy = 2.5
        sun = bpy.data.objects.new("_wondful_sb_preview_sun", sd)
        made += [sun, sd]
        tmp.collection.objects.link(sun)
        sun.matrix_world = Matrix(info["cam_world"]).to_3x3().to_4x4() @ Euler((math.radians(-35), math.radians(25), 0)).to_matrix().to_4x4()
        r = tmp.render
        r.resolution_x = int(width)
        r.resolution_y = max(8, int(round(width * fh / float(fw))))
        r.resolution_percentage = 100
        r.film_transparent = True
        r.image_settings.file_format = "PNG"
        r.image_settings.color_mode = "RGBA"
        tmp.view_settings.view_transform = "Standard"
        engine = os.environ.get("WONDFUL_SB_PREVIEW_ENGINE", "").strip().upper() or (
            "CYCLES" if bpy.app.background else "BLENDER_WORKBENCH")
        r.engine = engine
        if engine == "CYCLES":
            tmp.cycles.device = "CPU"
            tmp.cycles.samples = 8
            tmp.cycles.use_denoising = False
            tmp.cycles.max_bounces = 2
            if info.get("kind") == "interior":
                ad = bpy.data.lights.new("_wondful_sb_preview_room", "POINT")
                ad.energy = 120
                ad.shadow_soft_size = 1.5
                lo = bpy.data.objects.new("_wondful_sb_preview_room", ad)
                made += [lo, ad]
                tmp.collection.objects.link(lo)
                lo.matrix_world = Matrix(info["cam_world"]) @ Matrix.Translation((0, 0.3, -1.5))
        else:
            sh = tmp.display.shading
            sh.light = "STUDIO"
            sh.color_type = "MATERIAL"
        r.filepath = str(path)
        bpy.ops.render.render(write_still=True, scene=tmp.name)
        return os.path.isfile(path)
    finally:
        try:
            tmp.collection.children.unlink(coll)
        except Exception:
            pass
        for item in made:
            try:
                if isinstance(item, bpy.types.Object):
                    bpy.data.objects.remove(item, do_unlink=True)
                elif isinstance(item, bpy.types.Camera):
                    bpy.data.cameras.remove(item)
                elif isinstance(item, bpy.types.Light):
                    bpy.data.lights.remove(item)
                elif isinstance(item, bpy.types.World):
                    bpy.data.worlds.remove(item)
            except Exception:
                pass
        bpy.data.scenes.remove(tmp)
