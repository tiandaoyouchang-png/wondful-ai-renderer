"""Camera-to-reference-photo alignment (Fable 5.2 design, 2026-11) (numpy only, no bpy in this file).

Car frame: origin = centre of the wheel rectangle on the ground, front -Y, car's left +X, Z up.
Image "u-space": x in [0,1] left->right, y in [0,h/w] top->bottom (isotropic, unit = image width).
Params p = [az, el, log d, dyaw, dpitch, roll, log f];  f = focal / image width (lens_mm = 36*f).
az = 0 front view, +90deg camera on the car's LEFT (front then points to image-left), 180 rear view.
"""
import numpy as np

D = np.pi / 180.0
SIG_KP, SIG_BB, SIG_HZ = 0.03, 0.04, 0.03     # 1-sigma of vision-LLM points / bbox / horizon (u-space)
F_GRID = np.array([20, 28, 40, 60, 90, 135]) / 36.0
F_PRIOR, F_SIG = 45 / 36.0, 0.5               # log-normal focal prior


def car_template(wheelbase, track, wheel_r, length, width, height, y_off=0.0):
    """3D keypoints + 16-point 'body + cabin' proxy of a car with the given metric dimensions."""
    hx, hy = track / 2, wheelbase / 2
    xy = {'FL': (hx, -hy), 'FR': (-hx, -hy), 'RL': (hx, hy), 'RR': (-hx, hy)}
    kp = {k + '_contact': np.array([x, y, 0.0]) for k, (x, y) in xy.items()}
    kp.update({k + '_center': np.array([x, y, wheel_r]) for k, (x, y) in xy.items()})
    def box(x, y0, y1, z0, z1):
        return [[sx * x, y, z] for sx in (-1, 1) for y in (y0, y1) for z in (z0, z1)]
    L, W, H = length, width, height
    proxy = np.array(box(W / 2, y_off - L / 2, y_off + L / 2, 0, 0.60 * H) +
                     box(0.42 * W, y_off - 0.18 * L, y_off + 0.42 * L, 0.60 * H, H))
    return {'kp': kp, 'proxy': proxy, 'H': H, 'anchor_z': H / 2}


def camera(p, anchor_z):
    """Return (C, R, f): camera position, 3x3 world-from-camera (columns right, up, back), focal."""
    az, el, logd, dyaw, dpitch, roll, logf = p
    u = np.array([np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az), np.sin(el)])
    C = np.array([0, 0, anchor_z]) + np.exp(logd) * u
    yaw, pit = np.arctan2(-u[1], -u[0]) + dyaw, -el + dpitch
    fwd = np.array([np.cos(pit) * np.cos(yaw), np.cos(pit) * np.sin(yaw), np.sin(pit)])
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    right, up = np.cos(roll) * right + np.sin(roll) * up, np.cos(roll) * up - np.sin(roll) * right
    return C, np.stack([right, up, -fwd], 1), np.exp(logf)


def project(p, X, aspect, anchor_z):
    """Project Nx3 car-frame points to u-space. Returns (Nx2, depth N)."""
    C, R, f = camera(p, anchor_z)
    Xc = (np.atleast_2d(X) - C) @ R
    depth = -Xc[:, 2]
    z = np.maximum(depth, 1e-3)
    return np.stack([0.5 + f * Xc[:, 0] / z, 0.5 / aspect - f * Xc[:, 1] / z], 1), depth


def _huber(r, k=1.5):
    a = np.abs(r)
    return np.where(a <= k, r, np.sign(r) * np.sqrt(np.maximum(2 * k * a - k * k, 0)))


def _az_from_facing(facing):
    """'front-left', 'rear', 'left side', '左前' ... -> azimuth hint (rad) or None."""
    t = str(facing or '').lower()
    v = np.zeros(2)
    for keys, ang in ((('front', '前'), 0), (('rear', 'back', '后', '後'), 180),
                      (('left', '左'), 90), (('right', '右'), -90)):
        if any(k in t for k in keys):
            v += [np.cos(ang * D), np.sin(ang * D)]
    return float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 0.5 else None


def from_locate_json(js, aspect):
    """Adapter: vision-locate JSON (normalised 0-1, origin top-left) -> internal obs dict (u-space).
    Wheel labels may be 'front_left', 'FL', 'rear-right', '左前' ...; wheels with visible=false are dropped.
    heading_deg is taken as the signed view azimuth (0 front view, +90 car's left side toward camera,
    180 rear, -90 right side); it is only used when it agrees with `facing` within 60 deg."""
    s = 1.0 / aspect
    o = {'kp': {}, 'bbox': None, 'trunc': set(), 'horizon': None, 'az_hint': None}
    for w in js.get('wheels') or []:
        if w.get('visible') is False:
            continue
        t = str(w.get('label', '')).lower().replace('-', '_')
        ax = 'F' if (t.startswith('f') or 'front' in t or '前' in t) else 'R'
        sd = 'L' if ('left' in t or '左' in t or t in ('fl', 'rl')) else 'R'
        for key in ('contact', 'center'):
            if w.get(key) and len(w[key]) == 2:
                o['kp'][ax + sd + '_' + key] = np.array([w[key][0], w[key][1] * s])
    if js.get('bbox'):
        x0, y0, x1, y1 = js['bbox']
        o['bbox'] = np.array([x0, y0 * s, x1, y1 * s])
        o['trunc'] = {e for e, hit in zip(('left', 'top', 'right', 'bottom'),
                                          (x0 < .01, y0 < .01, x1 > .99, y1 > .99)) if hit}
    co = js.get('cut_off')
    if isinstance(co, (list, tuple, set)): o['trunc'] |= {str(e).lower() for e in co}
    elif isinstance(co, dict): o['trunc'] |= {k for k, v in co.items() if v}
    if js.get('horizon_y') is not None:
        o['horizon'] = float(js['horizon_y']) * s
    fa, hd = _az_from_facing(js.get('facing')), js.get('heading_deg')
    if hd is not None and (fa is None or abs(np.arctan2(np.sin(hd * D - fa), np.cos(hd * D - fa))) < 60 * D):
        o['az_hint'] = float(hd) * D
    else:
        o['az_hint'] = fa
    return o


def _swap_lr(kp):
    m = {'L': 'R', 'R': 'L'}
    return {k[0] + m[k[1]] + k[2:]: v for k, v in kp.items()}


def residuals(p, tpl, obs, aspect, dims_trusted=True, sig_bb=SIG_BB):
    """Stacked, sigma-normalised, Huberised residual vector (data terms + soft priors)."""
    az, el, logd, dyaw, dpitch, roll, logf = p
    a0, r = tpl['anchor_z'], []
    names = [k for k in obs['kp'] if k in tpl['kp']]
    if names:
        uv, dep = project(p, np.array([tpl['kp'][k] for k in names]), aspect, a0)
        e = (uv - np.array([obs['kp'][k] for k in names])) / SIG_KP
        e[:, 1] *= np.where([k.endswith('contact') for k in names], 1.5, 1.0)  # ground contact y matters most
        r += [_huber(e.ravel()), 10 * np.minimum(dep, 0)]
    if obs['bbox'] is not None:
        uv, _ = project(p, tpl['proxy'], aspect, a0)
        x0, y0, x1, y1 = obs['bbox']
        px0, py0, px1, py1 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
        t, wk = obs['trunc'], (1.0 if dims_trusted else 0.35)
        if not ({'left', 'right'} & t):          # centre strong, width weak when dims are a guess
            r.append(_huber(np.array([(px0 + px1 - x0 - x1) / 2, wk * (px1 - px0 - x1 + x0)]) / sig_bb))
        else:
            r.append(_huber(np.array([px0 - x0 if 'left' not in t else 0.0,
                                      px1 - x1 if 'right' not in t else 0.0]) / sig_bb))
        if 'bottom' not in t: r.append(_huber(np.array([(py1 - y1) / sig_bb])))
        if 'top' not in t:    r.append(_huber(np.array([wk * (py0 - y0) / sig_bb])))
    C, R, f = camera(p, a0)
    if obs['horizon'] is not None:               # horizon = image of the horizontal view direction
        h = -R[:, 2] * [1, 1, 0]; h /= np.linalg.norm(h) + 1e-9
        r.append(_huber(np.array([(project(p, C + 1e5 * h, aspect, a0)[0][0, 1] - obs['horizon']) / SIG_HZ])))
    pri = [(logf - np.log(F_PRIOR * max(1.0, 1 / aspect))) / F_SIG,   # 45 mm-equiv on the long side
           (C[2] - 1.5) / 2.0, roll / (2 * D),
           20 * min(0.0, C[2] - 0.15), 20 * min(0.0, el + 3 * D), 20 * max(0.0, el - 75 * D),
           5 * max(0.0, abs(logf - np.log(1.5)) - np.log(3.0))]       # keep ~18..160 mm
    if obs.get('az_hint') is not None:
        dz = az - obs['az_hint']
        pri.append(float(_huber(np.arctan2(np.sin(dz), np.cos(dz)) / (30 * D))))
    sides, axles = {k[1] for k in names}, {k[0] for k in names}
    if len(sides) == 1:      # only one side's wheels are visible -> that side must face the camera
        pri.append(min(0.0, (1 if 'L' in sides else -1) * C[0]) / 0.3)
    if len(axles) == 1:      # only one axle visible -> that end must face the camera
        pri.append(min(0.0, (-1 if 'F' in axles else 1) * C[1]) / 0.3)
    r.append(np.array(pri))
    return np.concatenate(r)


def lm(fun, p0, free, iters=40):
    """Levenberg-Marquardt with forward-difference Jacobian over the `free` parameter indices."""
    p, lam = np.array(p0, float), 1e-2
    r = fun(p); c = r @ r
    for _ in range(iters):
        J = np.empty((r.size, len(free)))
        for j, k in enumerate(free):
            q = p.copy(); q[k] += 1e-5
            J[:, j] = (fun(q) - r) / 1e-5
        g, Hm, ok = J.T @ r, J.T @ J, False
        for _ in range(8):
            dp = np.linalg.lstsq(Hm + lam * np.diag(np.diag(Hm) + 1e-9), -g, rcond=None)[0]
            q = p.copy(); q[free] += dp
            rq = fun(q); cq = rq @ rq
            if np.isfinite(cq) and cq < c:
                ok = (c - cq) > 1e-9 * c
                p, r, c, lam = q, rq, cq, max(lam / 3, 1e-8)
                break
            lam *= 4
        if not ok:
            break
    return p, c


def initialise(tpl, obs, aspect, top_k=4):
    """Coarse closed-form seeds: grid over (focal, azimuth, elevation); distance from bbox height,
    pan/tilt from bbox centre. Returns the top_k seeds sorted by cost."""
    if obs['bbox'] is not None:
        x0, y0, x1, y1 = obs['bbox']
    else:
        q = np.array(list(obs['kp'].values())); x0, x1 = q[:, 0].min(), q[:, 0].max()
        y1 = q[:, 1].max(); y0 = y1 - 0.5 * (x1 - x0)
    bx, by, bh, cy = (x0 + x1) / 2, (y0 + y1) / 2, max(y1 - y0, 1e-3), 0.5 / aspect
    seeds = []
    for f in F_GRID:
        els = [3 * D, 10 * D, 22 * D]
        if obs['horizon'] is not None:           # depression angle of the car's ground line below the horizon
            els = [float(np.clip(np.arctan((y1 - obs['horizon']) / f), 0, 60 * D))]
        for el in els:
            for az in np.arange(-180, 180, 22.5) * D:
                p = np.array([az, el, np.log(f * tpl['H'] / bh), np.arctan((bx - 0.5) / f),
                              np.arctan((by - cy) / f), 0.0, np.log(f)])
                r = residuals(p, tpl, obs, aspect)
                seeds.append((r @ r, p))
    seeds.sort(key=lambda s: s[0])
    return [s[1] for s in seeds[:top_k]]


def refine_keypoints(tpl, obs, aspect, dims_trusted=True):
    """MAP fit. Tries the labels as given and with left/right swapped (vision L/R is unreliable),
    for each: best seeds -> LM with fixed focal -> LM with free focal.
    Returns dict(p, cost, swapped, alt_cost)."""
    out = []
    for swapped in (False, True):
        o = dict(obs, kp=_swap_lr(obs['kp']) if swapped else obs['kp'])
        fun = lambda p: residuals(p, tpl, o, aspect, dims_trusted)
        best = min((lm(fun, s, [0, 1, 2, 3, 4], 25) for s in initialise(tpl, o, aspect)), key=lambda t: t[1])
        p, c = lm(fun, best[0], [0, 1, 2, 3, 4, 5, 6], 40)
        out.append({'p': p, 'cost': c, 'swapped': swapped, 'obs': o})
        if not obs['kp']:
            break
    out.sort(key=lambda d: d['cost'])
    out[0]['alt_cost'] = out[1]['cost'] if len(out) > 1 else np.inf
    return out[0]


def convex_hull(pts):
    """Andrew monotone chain; returns hull vertices counter-clockwise (in x right / y down: clockwise)."""
    P = sorted(map(tuple, pts))
    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) -
                                   (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h[:-1]
    return np.array(half(P) + half(P[::-1]))


def hull_mask(uv, gw, gh, aspect):
    """Rasterise the convex hull of u-space points on a gw x gh grid covering the image."""
    h = convex_hull(uv)
    xs, ys = np.meshgrid((np.arange(gw) + 0.5) / gw, (np.arange(gh) + 0.5) / gh / aspect)
    m = np.ones((gh, gw), bool)
    for a, b in zip(h, np.roll(h, -1, 0)):
        m &= ((b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])) >= 0
    return m


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / u if u else 0.0


def mask_bbox(mask, aspect):
    """Tight bbox (u-space) of a boolean mask, ignoring 1 % speckle on each side."""
    gh, gw = mask.shape
    cx, cy = np.cumsum(mask.sum(0)) / mask.sum(), np.cumsum(mask.sum(1)) / mask.sum()
    x0, x1 = np.searchsorted(cx, .01), np.searchsorted(cx, .99) + 1
    y0, y1 = np.searchsorted(cy, .01), np.searchsorted(cy, .99) + 1
    return np.array([x0 / gw, y0 / gh / aspect, x1 / gw, y1 / gh / aspect])


def refine_silhouette(fit, tpl, mask, aspect, dims_trusted=True, mu=0.02):
    """mask: 2D bool array (old-vehicle foreground, any low resolution, e.g. 128 px wide).
    1) replace the LLM bbox by the mask bbox (3x tighter sigma) and re-run LM;
    2) pattern search maximising IoU(proxy hull, mask), tethered to the keypoint cost.
    With untrusted dims only translation/distance move (shape mismatch must not rotate the camera)."""
    gh, gw = mask.shape
    bb = mask_bbox(mask, aspect)
    lim = (bb[0] < .02, bb[1] < .02, bb[2] > .98, bb[3] > 1 / aspect - .02)      # mask touches the frame
    o = dict(fit['obs'], bbox=bb, trunc=fit['obs']['trunc'] | {e for e, h in zip(('left', 'top', 'right', 'bottom'), lim) if h})
    fun = lambda p: residuals(p, tpl, o, aspect, dims_trusted, sig_bb=SIG_BB / 3)
    p, c0 = lm(fun, fit['p'], [0, 1, 2, 3, 4, 5, 6], 30)
    n = fun(p).size
    def J(q):
        r = fun(q)
        return 1 - iou(hull_mask(project(q, tpl['proxy'], aspect, tpl['anchor_z'])[0], gw, gh, aspect), mask) \
            + mu * (r @ r) / n
    free = [0, 1, 2, 3, 4] if dims_trusted else [2, 3, 4]
    step0 = np.array([4 * D, 3 * D, 0.06, 1.5 * D, 1.5 * D, 0, 0])
    best = J(p)
    for scale in (1, .5, .25, .125):
        moved = True
        while moved:
            moved = False
            for k in free:
                for s in (1, -1):
                    q = p.copy(); q[k] += s * scale * step0[k]
                    j = J(q)
                    if j < best - 1e-5:
                        p, best, moved = q, j, True
    return dict(fit, p=p, cost=float(fun(p) @ fun(p)), obs=o)


def footprint_metrics(uv, bbox, mask=None, aspect=1.0):
    """Compare projected 3D points `uv` (u-space; proxy OR the real model's vertices) with the photo
    vehicle bbox / mask. Shared by score() and the mode-1 pre-flight guard."""
    x0, y0, x1, y1 = bbox
    (px0, py0), (px1, py1) = np.clip(uv.min(0), 0, [1, 1 / aspect]), np.clip(uv.max(0), 0, [1, 1 / aspect])
    iw, ih = min(x1, px1) - max(x0, px0), min(y1, py1) - max(y0, py0)
    inter = max(iw, 0) * max(ih, 0)
    m = {'bbox_iou': inter / ((x1 - x0) * (y1 - y0) + (px1 - px0) * (py1 - py0) - inter),
         'dx': ((px0 + px1) - (x0 + x1)) / 2 / (x1 - x0),      # centre offset / bbox width
         'dbottom': (py1 - y1) / (y1 - y0),                    # ground-line offset / bbox height
         'h_ratio': (py1 - py0) / (y1 - y0), 'w_ratio': (px1 - px0) / (x1 - x0)}
    if mask is not None:
        m['sil_iou'] = iou(hull_mask(uv, mask.shape[1], mask.shape[0], aspect), mask)
    return m


def score(fit, tpl, aspect, mask=None):
    """Quality report: status in {'aligned','touch_up','failed'} + metrics + reasons."""
    p, o = fit['p'], fit['obs']
    C, R, f = camera(p, tpl['anchor_z'])
    uv, _ = project(p, tpl['proxy'], aspect, tpl['anchor_z'])
    bbox = o['bbox'] if o['bbox'] is not None else np.r_[uv.min(0), uv.max(0)]
    m = footprint_metrics(uv, bbox, mask, aspect)
    names = [k for k in o['kp'] if k in tpl['kp']]
    diag = np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])
    if names:
        e = project(p, np.array([tpl['kp'][k] for k in names]), aspect, tpl['anchor_z'])[0] \
            - np.array([o['kp'][k] for k in names])
        m['kp_rms'] = float(np.sqrt((e ** 2).sum(1).mean()) / diag)
        m['kp_max'] = float(np.sqrt((e ** 2).sum(1)).max() / diag)
    m.update(n_kp=len(names), lens_mm=36 * f, cam_height=C[2], elev_deg=p[1] / D, roll_deg=p[5] / D,
             az_deg=p[0] / D, ambiguity=fit['cost'] / max(fit.get('alt_cost', np.inf), 1e-9))
    bad, warn = [], []
    def chk(name, v, ok, lim):                  # ok / lim are (lo, hi) ranges
        if not (lim[0] <= v <= lim[1]): bad.append(name)
        elif not (ok[0] <= v <= ok[1]): warn.append(name)
    if names: chk('kp_rms', m['kp_rms'], (0, .06), (0, .12))
    k = 1.0 if mask is not None else 2.0        # LLM bbox is ~2x noisier than the mask bbox
    chk('bbox_iou', m['bbox_iou'], (1 - .20 * k, 1), (1 - .40 * k if mask is not None else .45, 1))
    chk('dbottom', abs(m['dbottom']), (0, .08 * k), (0, .15 * k))
    chk('dx', abs(m['dx']), (0, .08 * k), (0, .20 * k))
    if mask is not None: chk('sil_iou', m['sil_iou'], (.72, 1), (.55, 1))
    chk('lens_mm', m['lens_mm'], (20, 120), (14, 200))
    chk('cam_height', m['cam_height'], (0.3, 6.0), (0.1, 15.0))
    chk('roll_deg', abs(m['roll_deg']), (0, 3), (0, 8))
    if len({k[:2] for k in names}) < 2: bad.append('too_few_wheels')
    if m['ambiguity'] > 0.7 and names and o.get('az_hint') is None: warn.append('mirror_ambiguous')
    m.update(status='failed' if bad else ('touch_up' if warn else 'aligned'), bad=bad, warn=warn)
    return m


def align(obs_norm, old_dims, img_w, img_h, mask=None, dims_trusted=True):
    """One-call entry. old_dims: dict(wheelbase, track, wheel_r, length, width, height) of the PHOTO's
    vehicle (vision estimate, or the user's model as fallback with dims_trusted=False)."""
    aspect = img_w / img_h
    tpl, obs = car_template(**old_dims), from_locate_json(obs_norm, aspect)
    fit = refine_keypoints(tpl, obs, aspect, dims_trusted)
    if mask is not None:
        fit = refine_silhouette(fit, tpl, mask, aspect, dims_trusted)
    return fit, score(fit, tpl, aspect, mask), tpl
