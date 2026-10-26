"""1.2.6 只做场景相关物体 · no pedestrians / other vehicles / animals (user rule 2026-10-21).

User: 「不要加行人」「只做场景相关物体」. The ④ 场景粗模 and every image request keep only the static scene
(buildings, road surface, trees, poles, signs, railings …):

- ``strip_dynamic`` drops people / other vehicles / bicycles / animals from a ④ analysis before any block is
  built (decided by the analysis category first, then by the element's own words);
- ``is_dynamic_prop`` tells a measured 细节粗模 cluster that is person- or vehicle-sized and not part of a
  building / wall (those are only ever moving things in a street photo);
- ``apply`` appends one Chinese + English negative line to every image prompt (主视角, 多角度, 全景, 细节修复,
  分层输出), idempotent;
- the user re-enables a category only by asking for it in the 一句话需求 (``wanted``): 「街上有行人」 /
  "with pedestrians" → people allowed; 「不要行人」 never counts as asking.

No bpy here (pure helpers, unit-tested).
"""
from __future__ import annotations

import math
import re

PERSON, VEHICLE, ANIMAL = "person", "vehicle", "animal"
CATEGORIES = (PERSON, VEHICLE, ANIMAL)
MARKER = "【只保留场景静态物体】"
MARKER_EN = "[STATIC SCENE ONLY]"

# analysis categories (scene_blockout.PRIM_TYPES) that are moving things
TYPE_CATEGORY = {"person": PERSON, "pedestrian": PERSON, "people": PERSON, "crowd": PERSON,
                 "vehicle": VEHICLE, "car": VEHICLE, "bicycle": VEHICLE, "bike": VEHICLE, "motorcycle": VEHICLE,
                 "animal": ANIMAL}
# categories that are always static: their words never make them dynamic (a 人物广告牌 is a sign)
STATIC_TYPES = {"building", "wall", "fence", "pole", "sign", "ceiling", "window", "rug", "curtain", "water_plane",
                "awning", "tree", "bush", "hedge", "plant", "lamp", "rock"}

# words (lower-case English / Chinese) that name a moving thing …
_WORDS = {
    PERSON: ("pedestrian", "person", "people", "crowd", "man", "woman", "men", "women", "child", "children", "kid",
             "boy", "girl", "passerby", "passer-by", "commuter", "tourist", "walker", "cyclist", "rider", "figure",
             "human", "行人", "路人", "人群", "人物", "人们", "人流", "游客", "男人", "女人", "男子", "女子", "男行人",
             "女行人", "小孩", "孩子", "儿童", "情侣", "骑手", "骑车人", "乘客", "上班族", "人影"),
    VEHICLE: ("car", "cars", "taxi", "cab", "bus", "truck", "van", "suv", "lorry", "tram", "bicycle", "bike", "bikes",
              "motorcycle", "motorbike", "scooter", "moped", "vehicle", "vehicles", "traffic", "车辆", "汽车", "轿车",
              "出租车", "的士", "公交", "巴士", "卡车", "货车", "面包车", "电车", "自行车", "单车", "摩托", "电动车",
              "电瓶车", "三轮车", "车流"),
    ANIMAL: ("dog", "dogs", "cat", "cats", "bird", "birds", "pigeon", "pigeons", "horse", "animal", "animals", "pet",
             "狗", "猫", "鸟", "鸽子", "马匹", "动物", "宠物"),
}
# … unless the same name says it is a static thing (signs, road markings, places, statues)
_STATIC_WORDS = ("billboard", "poster", "sign", "screen", "display", "advert", "statue", "sculpture", "mural",
                 "crosswalk", "crossing", "zebra", "sidewalk", "pavement", "parking", "station", "stop", "lane",
                 "garage", "shop", "store", "building", "rack", "stand", "shelter", "rental", "广告", "海报", "招牌", "灯箱", "屏", "雕像", "雕塑", "壁画",
                 "人行道", "人行横道", "斑马线", "停车", "车站", "车道", "站台", "车库", "商店", "店", "楼", "牌", "架", "棚", "停放点")

NEGATIVE_ZH = {PERSON: "行人、人物或人群", VEHICLE: "其他车辆、自行车或摩托车", ANIMAL: "动物"}
NEGATIVE_EN = {PERSON: "pedestrians, people, human figures or crowds", VEHICLE: "other vehicles, cars, bicycles or "
               "motorcycles", ANIMAL: "animals"}

_ALLOW: frozenset = frozenset()
_BRIEF = ""


def _en(word: str) -> str:
    """English whole-word pattern, plural allowed (pedestrian → pedestrians, bus → buses)."""
    return r"(?<![a-z])" + re.escape(word) + r"(?:s|es)?(?![a-z])"


def _pat(word: str) -> str:
    return re.escape(word) if re.search(r"[\u3400-\u9fff]", word) else _en(word)


def _has_word(text: str, word: str) -> bool:
    if re.search(r"[\u3400-\u9fff]", word):
        return word in text
    return re.search(_en(word), text) is not None


def _text_of(item) -> str:
    item = item or {}
    raw = " ".join(str(item.get(k, "") or "") for k in ("id", "label_zh", "material", "kind"))
    return re.sub(r"[_\-]+", " ", raw).lower()


def category_of(item) -> str:
    """Moving-thing category of one analysis element ('' = static scene object).

    The analysis category decides first (type person / vehicle / animal …); a static category (building, sign,
    pole …) is never dynamic whatever it is called; an unknown / generic category (furniture, rock …) falls back
    to its own name and material words, and a static word in the same name wins (人物广告牌, 停车场)."""
    t = str((item or {}).get("type", "") or "").strip().lower()
    if t in TYPE_CATEGORY:
        return TYPE_CATEGORY[t]
    if t in STATIC_TYPES:
        return ""
    text = _text_of(item)
    if any(_has_word(text, w) for w in _STATIC_WORDS):
        return ""
    for cat in CATEGORIES:
        if any(_has_word(text, w) for w in _WORDS[cat]):
            return cat
    return ""


_NEG = re.compile(r"(不要|不需要|不想要|不能有|不出现|不含|没有|别有|别加|别放|无|去掉|去除|删掉|禁止|拒绝|"
                  r"\bno\b|\bnot\b|\bwithout\b|\bnever\b|\bremove\b|\bavoid\b|\bempty of\b)[^，。；,.;!！?？]{0,8}$")
# a lone 人 only counts in a few unambiguous forms (一个人, 有人, 几个人, 人来人往) — 人行道 / 无人 / 人工 never do
_PERSON_LONE = re.compile(r"(有人|[一两几些多群个位名]+人(?![行工造民])|人来人往|人山人海|熙熙攘攘)")


def wanted(brief: str) -> frozenset:
    """Categories the user explicitly asks for in the 一句话需求 (negated mentions never count)."""
    text = str(brief or "").lower()
    if not text.strip():
        return frozenset()
    out = set()
    for cat in CATEGORIES:
        words = [w for w in _WORDS[cat] if w not in ("figure", "pet", "stop", "traffic")]
        if cat == VEHICLE:
            # the product itself is often a car: only plural / other / traffic words ask for more vehicles
            words = ["其他车", "别的车", "车流", "车辆", "出租车", "公交", "巴士", "卡车", "自行车", "单车", "摩托",
                     "电动车", "traffic", "taxis", "taxi", "buses", "bus", "bicycle", "bicycles", "bike", "bikes",
                     "motorcycle", "motorcycles", "other cars", "other vehicles", "parked cars", "scooter"]
        for w in words:
            if cat == PERSON and w in ("man", "men", "rider"):
                continue
            for m in re.finditer(_pat(w), text):
                if not _NEG.search(text[:m.start()]):
                    out.add(cat)
                    break
            if cat in out:
                break
        if cat == PERSON and PERSON not in out:
            for m in _PERSON_LONE.finditer(text):
                if not _NEG.search(text[:m.start()]):
                    out.add(PERSON)
                    break
    return frozenset(out)


def set_brief(brief: str) -> frozenset:
    """Called when a task starts (operators._start_thread): the categories the current 一句话需求 allows."""
    global _ALLOW, _BRIEF
    _BRIEF = str(brief or "")
    _ALLOW = wanted(_BRIEF)
    return _ALLOW


def allowed() -> frozenset:
    return _ALLOW


def strip_dynamic(data, allow=None):
    """Remove moving things from a ④ analysis (primitives; foliage / roads are static by kind).

    Returns ``(new_data, dropped)``; ``dropped`` = [(category, label_or_id), …]. ``allow`` = categories the user
    asked for (None → the current task's 一句话需求)."""
    allow = allowed() if allow is None else frozenset(allow or ())
    if not isinstance(data, dict):
        return data, []
    keep, dropped = [], []
    for p in data.get("primitives", []) or []:
        cat = category_of(p) if isinstance(p, dict) else ""
        if cat and cat not in allow:
            dropped.append((cat, str(p.get("label_zh") or p.get("id") or "")))
        else:
            keep.append(p)
    if not dropped:
        return data, []
    out = dict(data)
    out["primitives"] = keep
    return out, dropped


def dropped_note(dropped) -> str:
    if not dropped:
        return ""
    n = {c: sum(1 for k, _ in dropped if k == c) for c in CATEGORIES}
    bits = [f"{lab} {n[c]}" for c, lab in ((PERSON, "行人"), (VEHICLE, "车辆"), (ANIMAL, "动物")) if n[c]]
    return "只保留场景静态物体：未生成" + " · ".join(bits)


# 1.2.7 (PC test 2026-10-21: a 2.0 m wide x 3.0 m tall static kiosk / pillar was deleted as an end-on car):
# size-only vehicle detection is limited to real passenger-car proportions.
CAR_LEN = (3.5, 5.5)       # m, length (side-on: the visible width)
CAR_WID = (1.6, 2.2)       # m, width (end-on: the visible width)
CAR_HGT = (1.2, 2.0)       # m, height


def vehicle_pose(W: float, h: float, D=None) -> str:
    """'end' / 'side' when a free-standing measured cluster has passenger-car proportions, else ''.

    Seen end-on the visible width is the car's width and its depth (when measured) must not exceed a car
    length; side-on the visible width is its length and its depth must not exceed a car width (the two are
    swapped). A measured depth below the car range is fine (one view only sees the front face)."""
    if not (CAR_HGT[0] <= h <= CAR_HGT[1]):
        return ""
    d = None if D is None else float(D)
    if CAR_WID[0] <= W <= CAR_WID[1] and (d is None or d <= CAR_LEN[1] + 0.5):
        return "end"
    if CAR_LEN[0] <= W <= CAR_LEN[1] and (d is None or d <= CAR_WID[1] + 0.8):
        return "side"
    return ""


def is_dynamic_prop(W: float, h: float, allow=None, D=None, static=False) -> str:
    """细节粗模 measured cluster (free-standing, not building / wall / slab): its category by size, '' = keep.

    Person-sized (≤ 0.9 m wide, 1.0–2.2 m tall; a vending machine / kiosk is wider) → person; passenger-car
    proportions only (``vehicle_pose``: length 3.5–5.5 m, width 1.6–2.2 m, height 1.2–2.0 m, end-on swaps
    length and width) → vehicle. ``static``: the ④ analysis put a static element (sign, kiosk, pillar …) here
    → never deleted. Low props (benches, planters, bollards) stay."""
    if static:
        return ""
    allow = allowed() if allow is None else frozenset(allow or ())
    cat = ""
    if W <= 0.9 and 1.0 <= h <= 2.2:
        cat = PERSON
    elif vehicle_pose(W, h, D):
        cat = VEHICLE
    return "" if (not cat or cat in allow) else cat


def static_footprints(analysis) -> list:
    """Footprints of the ④ analysis' static elements (category '' — signs, kiosks, pillars, shelters …) that
    are small enough to be a free-standing prop: [(x, y, half_w, half_d, rot_rad)], repeats expanded. A measured
    cluster inside one is a static thing and is never removed as a person / vehicle."""
    out = []
    for p in (analysis or {}).get("primitives", []) or []:
        if not isinstance(p, dict) or category_of(p):
            continue
        if str(p.get("type", "")).lower() in ("building", "window", "ceiling", "rug", "water_plane"):
            continue
        try:
            x, y = float(p.get("x", 0)), float(p.get("y", 0))
            w, d = float(p.get("w", 0) or 0), float(p.get("d", 0) or 0)
            rot = math.radians(float(p.get("rot_z_deg", 0) or 0))
            n = max(1, min(40, int(p.get("repeat", 1) or 1)))
            dx, dy = float(p.get("repeat_dx", 0) or 0), float(p.get("repeat_dy", 0) or 0)
        except (TypeError, ValueError):
            continue
        if max(w, d) > 15.0 or max(w, d) <= 0:
            continue
        for k in range(n):
            out.append((x + k * dx, y + k * dy, w / 2, d / 2, rot))
    return out


def in_static_footprint(x: float, y: float, fps, margin: float = 0.8) -> bool:
    for cx, cy, hw, hd, rot in fps or ():
        c, s = math.cos(rot), math.sin(rot)
        lx = (x - cx) * c + (y - cy) * s
        ly = -(x - cx) * s + (y - cy) * c
        if abs(lx) <= hw + margin and abs(ly) <= hd + margin:
            return True
    return False


def negative_text(allow=None) -> str:
    """One Chinese + English line forbidding every category the user did not ask for ('' when all allowed)."""
    allow = allowed() if allow is None else frozenset(allow or ())
    cats = [c for c in CATEGORIES if c not in allow]
    if not cats:
        return ""
    zh = "，也不要出现".join(NEGATIVE_ZH[c] for c in cats)
    en = "; no ".join(NEGATIVE_EN[c] for c in cats)
    return (f"{MARKER}画面里任何位置都不要出现{zh}（主体产品本身除外，不要画进玻璃反射、远处或背景里）；"
            f"只保留场景本身的静态物体：建筑、路面、树木、灯杆、招牌、护栏等。\n"
            f"{MARKER_EN} Absolutely no {en} anywhere in the image (the product itself excepted; not in the "
            f"background, the distance or reflections either); show only the static scene: buildings, road "
            f"surface, trees, poles, signs, railings.")


def apply(prompt: str, allow=None) -> str:
    """Append the negative line to an image prompt once (idempotent; '' prompt stays '' + line)."""
    text = str(prompt or "")
    if MARKER in text or MARKER_EN in text:
        return text
    neg = negative_text(allow)
    if not neg:
        return text
    return (text.rstrip() + "\n" + neg) if text.strip() else neg


# static compounds that contain a moving-thing word (masked before scrubbing a description)
_STATIC_COMPOUNDS = re.compile(
    r"(人物(?=广告|海报|招牌|灯箱|画|像|大屏|屏)|人行道|人行横道|人行天桥|步行街|车道|车站|停车场|停车位|车库|公交站|巴士站|"
    r"自行车道|非机动车道|pedestrian (?:crossing|bridge|street|zone|area|walkway|precinct)|car ?park|parking|"
    r"bus (?:stop|lane|shelter|station)|bike lane|cycle lane|taxi rank|crosswalk)", re.I)
_CLAUSE = re.compile(r"(?:[^。；;!！?？\n,，.]|\.(?!\s|$))+(?:[。；;!！?？\n,，]|\.(?=\s|$))?\s*")


def _clause_category(clause: str) -> str:
    text = _STATIC_COMPOUNDS.sub(" ", clause.lower())
    for cat in CATEGORIES:
        for w in _WORDS[cat]:
            if w in ("figure", "stop", "sign"):
                continue
            for m in re.finditer(_pat(w), text):
                if not _NEG.search(text[:m.start()]):
                    return cat
    return ""


def scrub(text: str, allow=None) -> str:
    """Drop the clauses of a scene description that mention a moving thing the user did not ask for
    (「行人穿过斑马线」, "taxis wait at the kerb"); static compounds (人行道, 人物广告牌, pedestrian crossing) stay."""
    allow = allowed() if allow is None else frozenset(allow or ())
    src = str(text or "")
    if not src.strip() or set(CATEGORIES) <= set(allow):
        return src
    out = []
    for m in _CLAUSE.finditer(src):
        c = m.group(0)
        cat = _clause_category(c)
        if cat and cat not in allow:
            continue
        out.append(c)
    res = "".join(out).strip()
    return re.sub(r"[,，]\s*$", "。", res) if res else ""
