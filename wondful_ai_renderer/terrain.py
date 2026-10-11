"""1.1.0: custom terrain for the clay base (高级 › 地面 › 自定义).

Pure math + prompt text, no ``bpy``: the Blender side lives in viewport_capture.

The terrain is a height field in the product's own frame: ``u`` along its forward
axis, ``v`` along its right axis, height along world up.  Under the product there
is a planar "pad" (just larger than its footprint); outside the pad each of the
four directions (前/后/左/右) continues flat, uphill or downhill by its angle,
with a rounded shoulder.  When the pose follows the terrain the pad is tilted to
the average of the two opposite sides, so the product pitches with a front/back
slope and rolls with a left/right slope and still sits flush on a plane.
"""
from __future__ import annotations

import math

try:  # numpy is bundled with Blender; tests run with it too
    import numpy as np
except Exception:  # pragma: no cover
    np = None

DIRECTIONS = ("front", "back", "left", "right")
MODES = ("FLAT", "UP", "DOWN")
MAX_ANGLE = 35.0

PRESETS = {
    "CLIMB": ("爬坡", {"front": ("UP", 15.0), "back": ("DOWN", 15.0), "left": ("FLAT", 0.0), "right": ("FLAT", 0.0)}),
    "DESCENT": ("下坡", {"front": ("DOWN", 15.0), "back": ("UP", 15.0), "left": ("FLAT", 0.0), "right": ("FLAT", 0.0)}),
    "SIDE": ("侧坡", {"front": ("FLAT", 0.0), "back": ("FLAT", 0.0), "left": ("UP", 12.0), "right": ("DOWN", 12.0)}),
    "RIDGE": ("山脊", {"front": ("FLAT", 0.0), "back": ("FLAT", 0.0), "left": ("DOWN", 20.0), "right": ("DOWN", 20.0)}),
    "VALLEY": ("山谷", {"front": ("FLAT", 0.0), "back": ("FLAT", 0.0), "left": ("UP", 20.0), "right": ("UP", 20.0)}),
}
PRESET_ORDER = ("CLIMB", "DESCENT", "SIDE", "RIDGE", "VALLEY")


def normalize(settings: dict | None) -> dict:
    """Clamp a settings dict: {front: (mode, deg), ..., follow: bool, front_axis: str}."""
    settings = dict(settings or {})
    out = {}
    for d in DIRECTIONS:
        mode, ang = settings.get(d) or ("FLAT", 0.0)
        mode = str(mode or "FLAT").upper()
        if mode not in MODES:
            mode = "FLAT"
        ang = max(0.0, min(MAX_ANGLE, float(ang or 0.0)))
        if mode == "FLAT" or ang < 0.05:
            mode, ang = "FLAT", 0.0
        out[d] = (mode, ang)
    out["follow"] = bool(settings.get("follow", True))
    out["front_axis"] = str(settings.get("front_axis") or "AUTO").upper()
    out["vehicle"] = settings.get("vehicle")  # None = detect
    return out


def preset_settings(name: str) -> dict:
    label, values = PRESETS[name]
    return normalize(dict(values))


def settings_from_props(props) -> dict:
    s = {}
    for d in DIRECTIONS:
        s[d] = (getattr(props, f"clay_terrain_{d}", "FLAT"), getattr(props, f"clay_terrain_{d}_angle", 0.0))
    s["follow"] = getattr(props, "clay_terrain_follow", True)
    s["front_axis"] = getattr(props, "clay_terrain_front_axis", "AUTO")
    hint = " ".join(str(getattr(props, k, "") or "") for k in ("identity_details", "product_look_prompt", "scene_brief"))
    if any(w in hint for w in ("车", "轮毂", "轮胎")):
        s["vehicle"] = True  # otherwise detected from object / material names
    return normalize(s)


def apply_preset_to_props(props, name: str) -> None:
    values = preset_settings(name)
    for d in DIRECTIONS:
        mode, ang = values[d]
        setattr(props, f"clay_terrain_{d}", mode)
        if mode != "FLAT":
            setattr(props, f"clay_terrain_{d}_angle", ang)


def is_flat(settings: dict) -> bool:
    s = normalize(settings)
    return all(s[d][0] == "FLAT" for d in DIRECTIONS)


def outward_slope(mode_angle) -> float:
    """dh/dt going outward from the product in that direction (UP > 0, DOWN < 0)."""
    mode, ang = mode_angle
    if mode == "FLAT":
        return 0.0
    t = math.tan(math.radians(ang))
    return t if mode == "UP" else -t


def slopes(settings: dict) -> dict:
    s = normalize(settings)
    return {d: outward_slope(s[d]) for d in DIRECTIONS}


def pad_gradients(settings: dict) -> tuple[float, float]:
    """(gu, gv): dh/du and dh/dv of the pad under the product (0, 0 when not following)."""
    s = normalize(settings)
    if not s["follow"]:
        return 0.0, 0.0
    sl = slopes(s)
    gu = (sl["front"] - sl["back"]) / 2.0   # back outward = -u direction
    gv = (sl["right"] - sl["left"]) / 2.0   # left outward = -v direction
    return gu, gv


def pose_angles(gu: float, gv: float) -> tuple[float, float]:
    """(pitch, roll) in degrees: pitch > 0 = front up, roll > 0 = right side up."""
    return math.degrees(math.atan(gu)), math.degrees(math.atan(gv))


def profile(x, half, g, s_pos, s_neg, shoulder):
    """Height along one axis. ``x`` scalar or numpy array.

    |x| <= half: pad plane (gradient g).  Beyond, the gradient eases from the pad's
    over ``shoulder`` to the outward slope (s_pos for +x, s_neg for -x), then stays.
    """
    w = max(float(shoulder), 1e-6)
    scalar = not (np is not None and isinstance(x, np.ndarray))
    xa = np.asarray(x, dtype=float) if np is not None else None
    if scalar:
        return _profile_scalar(float(x), half, g, s_pos, s_neg, w)
    t_pos = np.clip(xa - half, 0.0, None)
    t_neg = np.clip(-xa - half, 0.0, None)
    inside = np.clip(xa, -half, half) * g

    def side(t, g_out, s_out):
        te = np.minimum(t, w)
        h = g_out * te + (s_out - g_out) * te * te / (2.0 * w)
        return h + np.clip(t - w, 0.0, None) * s_out

    return inside + side(t_pos, g, s_pos) + side(t_neg, -g, s_neg)


def _profile_scalar(x, half, g, s_pos, s_neg, w):
    inside = max(-half, min(half, x)) * g

    def side(t, g_out, s_out):
        if t <= 0:
            return 0.0
        te = min(t, w)
        return g_out * te + (s_out - g_out) * te * te / (2.0 * w) + max(0.0, t - w) * s_out

    return inside + side(x - half, g, s_pos) + side(-x - half, -g, s_neg)


class HeightField:
    def __init__(self, settings, half_u, half_v, shoulder):
        self.settings = normalize(settings)
        self.gu, self.gv = pad_gradients(self.settings)
        self.sl = slopes(self.settings)
        self.half_u, self.half_v, self.shoulder = float(half_u), float(half_v), float(shoulder)

    def height(self, u, v):
        return (profile(u, self.half_u, self.gu, self.sl["front"], self.sl["back"], self.shoulder)
                + profile(v, self.half_v, self.gv, self.sl["right"], self.sl["left"], self.shoulder))


def grid_offsets(half, shoulder, size, far=300.0):
    """Sorted 1-D vertex coordinates: fine across pad + shoulder, coarse far away."""
    w = shoulder
    pos = [half * k / 4.0 for k in range(5)]
    pos += [half + w * t for t in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)]
    for e in (0.8, 3.0, 10.0):
        pos.append(half + w + size * e)
    pos.append(half + w + max(far, size * 60.0))
    neg = [-p for p in pos[1:]]
    return sorted(set(neg + pos))


# --------------------------------------------------------------------------- prompt text

def _names(vehicle: bool):
    if vehicle:
        return {"subj": "车身", "front": "车头", "back": "车尾", "left": "车身左侧", "right": "车身右侧",
                "contact": "四个车轮都实实在在压在坡面上"}
    return {"subj": "产品", "front": "产品正面", "back": "产品背面", "left": "产品左侧", "right": "产品右侧",
            "contact": "产品底部的支撑点（脚、轮或底座）全部贴住地面"}


def describe(settings, vehicle=False, hints=None) -> str:
    """Chinese terrain sentence for the image prompt, e.g.
    「车头朝上爬坡约15°（车头高、车尾低）；…」.  ``hints`` maps a direction to the
    image side it appears on (e.g. {"right": "画面左侧"})."""
    s = normalize(settings)
    if is_flat(s):
        return ""
    hints = hints or {}
    n = _names(bool(vehicle))

    def name(d):
        h = hints.get(d)
        return f"{n[d]}（{h}）" if h else n[d]

    gu, gv = pad_gradients(s)
    pitch, roll = pose_angles(gu, gv)
    pose = []
    if abs(pitch) >= 0.5:
        p = round(abs(pitch))
        if vehicle:
            pose.append(f"车头朝上爬坡约{p}°（车头高、车尾低）" if pitch > 0 else f"车头朝下下坡约{p}°（车头低、车尾高）")
        else:
            pose.append(f"产品立在约{p}°的斜坡上，正面朝上坡方向（前高后低）" if pitch > 0
                        else f"产品立在约{p}°的斜坡上，正面朝下坡方向（前低后高）")
    if abs(roll) >= 0.5:
        r = round(abs(roll))
        low, high = ("right", "left") if roll < 0 else ("left", "right")
        pose.append(f"{n['subj']}向{n[low][-2:]}倾斜约{r}°（{name(low)}低、{name(high)}高）")
    if not pose:
        pose.append(f"{n['subj']}保持水平")
    around = []
    words = {"front": "前方", "back": "后方", "left": "", "right": ""}
    for d in DIRECTIONS:
        mode, ang = s[d]
        a = round(ang)
        where = name(d) + words[d]
        if mode == "UP":
            around.append(f"{where}地面向上抬升约{a}°")
        elif mode == "DOWN":
            around.append(f"{where}地面向下倾约{a}°")
    form = ""
    if s["left"][0] == "DOWN" and s["right"][0] == "DOWN":
        form = f"（{'车' if vehicle else '产品'}停在山脊上，两侧地面向下掉落）"
    elif s["left"][0] == "UP" and s["right"][0] == "UP":
        form = f"（{'车' if vehicle else '产品'}停在山谷底部，两侧地面抬起）"
    elif s["front"][0] == "DOWN" and s["back"][0] == "DOWN":
        form = f"（{'车' if vehicle else '产品'}停在坡顶，前后地面都向下掉落）"
    elif s["front"][0] == "UP" and s["back"][0] == "UP":
        form = f"（{'车' if vehicle else '产品'}停在两道坡之间的低处）"
    text = "本次地形：" + "；".join(x for x in pose if x)
    if around:
        text += "；" + "，".join(around) + form
    text += (f"。地形与{n['subj']}姿态以白模为准：{n['contact']}，不悬空、不陷进地面；"
             "不要把坡面改平，也不要改变坡的方向和陡缓。")
    return text
