"""1.2.0 环境体块: simple boxes / cylinders / ramps blocked out around the product.

In the clay base they get a temporary matte grey-blue material (restored afterwards) and
the image prompt asks the model to turn each block into a real building / structure at the
same position, height and silhouette, in the scene's style, while the product stays locked.

Pure-text helpers (name hints, placement words, prompt paragraph, material swap bookkeeping)
need no ``bpy`` so they are unit-tested directly; Blender-only parts import ``bpy`` lazily.
"""
from __future__ import annotations

import json
import math
import re

MASSING_HEX = "#9FB4C8"
MAT_NAME = "_wondful_massing_mat"
AUTO_KEYS = ("环境", "体块", "context", "massing")
HINT_PROPS = ("wondful_hint", "hint", "描述", "提示")
# 建筑参考: per-block reference images (JSON list of paths) stored on the block object
REF_PROP = "wondful_ref_paths"
MAX_BLOCK_REFS = 3        # per block
MAX_BUILDING_IMAGES = 4   # per generation, on top of the clay canvas (image 1)
DEFAULT_COLLECTION = "环境体块"
HULL_TAG = "wondful_hull_block"
# a block built outside the massing collection (at a placeholder in the user's own collection) still counts
ENV_EXTRA_TAG = "wondful_env_extra"
PLACEHOLDER_PROP = "wondful_placeholder"          # on the block: the hidden placeholder's name
PLACEHOLDER_BOX_PROP = "wondful_placeholder_box"  # on the block: the placeholder box it was fitted to (JSON)
PLACEHOLDER_FOR_PROP = "wondful_placeholder_for"  # on the placeholder: the block built in its place
SB_GROUP_PROP = "wondful_sb_group"                # = scene_blockout.GROUP_PROP (kept bpy-free here)

# English name words → Chinese hint (Chinese names are used as written)
_EN_HINTS = {
    "glass": "玻璃幕墙", "curtainwall": "玻璃幕墙", "brick": "砖墙", "stone": "石材", "concrete": "混凝土",
    "wood": "木结构", "timber": "木结构", "tree": "树", "trees": "树", "forest": "树林", "hill": "山",
    "mountain": "山", "rock": "岩石", "wall": "墙面", "fence": "围栏", "ramp": "坡道", "stairs": "台阶",
    "steps": "台阶", "tower": "高楼", "building": "建筑", "house": "房屋", "shop": "店铺", "bridge": "桥",
    "pillar": "柱子", "column": "柱子", "hedge": "绿篱", "planter": "花坛", "billboard": "广告牌",
    "canopy": "雨棚", "pavilion": "亭子", "kiosk": "售货亭", "metal": "金属", "neon": "霓虹招牌",
    # 1.2.3 interior / street vocabulary (多角度场景一致 names every block)
    "sofa": "沙发", "couch": "沙发", "armchair": "扶手椅", "chair": "椅子", "stool": "凳子", "bench": "长椅",
    "table": "桌子", "desk": "书桌", "cabinet": "柜子", "cupboard": "柜子", "wardrobe": "衣柜", "console": "矮柜",
    "sideboard": "餐边柜", "dresser": "斗柜", "drawer": "抽屉柜", "bookshelf": "书架", "bookcase": "书架",
    "shelf": "搁板", "shelves": "搁板", "shelving": "置物架", "rack": "置物架", "bed": "床", "nightstand": "床头柜",
    "tv": "电视", "television": "电视", "monitor": "显示器", "lamp": "灯", "light": "灯", "chandelier": "吊灯",
    "plant": "绿植", "houseplant": "盆栽绿植", "pot": "花盆", "vase": "花瓶", "cup": "杯子", "mug": "杯子",
    "book": "书", "books": "书", "clock": "钟", "frame": "画框", "painting": "挂画", "picture": "挂画",
    "mirror": "镜子", "rug": "地毯", "carpet": "地毯", "floor": "地面", "ceiling": "天花板", "soffit": "吊顶",
    "beam": "梁", "window": "窗", "windows": "窗", "glazing": "玻璃门窗", "door": "门", "mullion": "门窗竖框",
    "curtain": "窗帘", "curtains": "窗帘", "drape": "窗帘", "drapes": "窗帘", "sheer": "纱帘", "blind": "百叶帘",
    "blinds": "百叶帘", "panel": "墙板", "cladding": "饰面墙板", "counter": "台面", "island": "岛台",
    "fireplace": "壁炉", "radiator": "暖气片", "cushion": "靠垫", "pillow": "枕头", "road": "道路",
    "street": "街道", "sidewalk": "人行道", "pavement": "人行道", "path": "小路", "curb": "路沿", "kerb": "路沿",
    "grass": "草地", "lawn": "草地", "bush": "灌木", "shrub": "灌木", "foliage": "树冠", "trunk": "树干",
    "cliff": "崖壁", "dune": "沙丘", "sand": "沙地", "water": "水面", "pool": "水池", "pole": "杆子",
    "streetlight": "路灯", "lamppost": "路灯", "sign": "指示牌", "awning": "遮阳篷", "roof": "屋顶",
    "balcony": "阳台", "railing": "栏杆", "terrain": "地面", "ground": "地面", "heightfield": "地面",
}
# two-word names (checked before single words)
_EN_PAIRS = {
    ("coffee", "table"): "茶几", ("side", "table"): "边几", ("end", "table"): "边几", ("dining", "table"): "餐桌",
    ("media", "console"): "电视柜", ("tv", "console"): "电视柜", ("tv", "stand"): "电视柜", ("tv", "cabinet"): "电视柜",
    ("book", "shelf"): "书架", ("wall", "shelf"): "墙面搁板", ("floor", "lamp"): "落地灯", ("table", "lamp"): "台灯",
    ("street", "light"): "路灯", ("wood", "panel"): "木饰面墙板", ("back", "panel"): "背板",
    ("glass", "door"): "玻璃门", ("sheer", "curtain"): "纱帘", ("sheer", "curtains"): "纱帘", ("sliding", "door"): "推拉门", ("ceiling", "light"): "吸顶灯",
}
# parts: attached to the noun before them ("sofa_back" → 沙发靠背)
_EN_PARTS = {"back": "靠背", "arm": "扶手", "seat": "坐垫", "base": "底座", "leg": "腿", "legs": "腿",
             "fronts": "柜门", "doors": "柜门", "top": "台面", "frame": "框"}
# colours: prefix the next noun ("white_sheer_curtains" → 白色纱帘)
_EN_COLOURS = {"white": "白色", "black": "黑色", "grey": "灰色", "gray": "灰色", "red": "红色", "green": "绿色",
               "blue": "蓝色", "yellow": "黄色", "brown": "棕色", "beige": "米色", "dark": "深色", "pale": "浅色"}
# category words (analysis builders prefix names with them): only used when nothing more specific
_CATEGORY = {"furniture": "家具", "prop": "摆件", "props": "摆件", "decor": "装饰", "fixture": "固定装置",
             "structure": "结构", "env": "", "environment": ""}
# words that carry no design hint in a multi-word name
_STOP = {"and", "of", "the", "with", "on", "in", "at", "near", "far", "left", "right", "upper", "lower", "middle",
         "lowest", "highest", "central", "centre", "center", "small", "large", "big", "main", "rear", "front",
         "gathered", "decorative", "inner", "outer", "a", "x", "y", "z"}
# analysis-JSON element types (wondful_env_type) → Chinese
ENV_TYPE_WORDS = {"wall": "墙面", "ceiling": "天花板", "window": "门窗", "door": "门", "curtain": "窗帘",
                  "furniture": "家具", "plant": "绿植", "terrain": "地面", "floor": "地面", "building": "建筑",
                  "road": "道路", "tree": "树", "cliff": "崖壁", "rock": "岩石", "foliage": "树木", "trunk": "树干",
                  "water": "水面", "sky": "", "prop": "摆件"}
# 1.2.3: auto-stored hints (set when a block is created) rank below the user's own object name
AUTO_HINT_FLAG = "wondful_hint_auto"
LABEL_PROP = "wondful_label"
SHAPE_NOTE = "（按尺寸推测）"
INTERIOR_MARK = "室内物体"
# generic Blender / massing words that carry no design hint
_GENERIC = {
    "cube", "box", "cylinder", "cone", "plane", "mesh", "object", "obj", "block", "mass", "massing",
    "context", "sphere", "uvsphere", "icosphere", "torus", "立方体", "长方体", "圆柱", "圆柱体", "圆锥",
    "平面", "球体", "体块", "块", "环境", "物体", "网格", "立方", "柱体",
}
_CJK = re.compile(r"[\u3400-\u9fff]")


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def massing_rgba(hex_color: str = MASSING_HEX) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(srgb_to_linear(int(h[i:i + 2], 16) / 255.0) for i in (0, 2, 4)) + (1.0,)


def looks_like_massing_name(name) -> bool:
    low = str(name or "").lower()
    return any(k in low for k in AUTO_KEYS)


def auto_pick(names, exclude=()):
    """First collection name that looks like 环境 / 体块 / Context / Massing (excluded names skipped).

    Exact-ish names win over names that merely contain a keyword."""
    exclude = {str(n) for n in exclude or () if n}
    cands = [str(n) for n in names or () if n and str(n) not in exclude and looks_like_massing_name(n)]
    if not cands:
        return ""
    def rank(n):
        low = n.lower().strip()
        return (0 if low in AUTO_KEYS or low in ("环境体块", "context massing", "massing blocks") else 1, len(n))
    return sorted(cands, key=rank)[0]


_STYLE = {"glass", "curtainwall", "brick", "stone", "concrete", "wood", "timber", "metal", "neon"}
# a noun in front of another noun reads as a modifier ("floor_houseplant" → 落地盆栽绿植)
_MODIFIER = {"floor": "落地", "wall": "墙面", "shelf": "搁板上的", "table": "桌上的", "ceiling": "",
             "window": "窗边", "tv": "电视", "television": "电视", "door": "门", "balcony": "阳台",
             "street": "街边", "road": "路边", "book": "书"}


def _name_words(name):
    """(specific words, category words) from a block / material name.

    Consecutive English nouns merge into one compound with the last one as the head; a leading
    element-type word from an analysis builder ("furniture_", "wall_", "plant_") is dropped when
    more follows; style words (glass, brick…) stay separate."""
    base = re.sub(r"\.\d+$", "", str(name or ""))
    toks = [t.strip() for t in re.split(r"[_\-\s.·/|]+", base) if t.strip() and not t.strip().isdigit()]
    lows = [re.sub(r"\d+$", "", t.lower()) for t in toks]
    out, cats = [], []
    group = []          # [(word, modifier form)] of the current English noun run
    colour = ""
    on_target = False   # "book_on_table": what follows "on" is where the thing sits

    def flush():
        nonlocal group
        if not group:
            return
        head = group[-1][0]
        mods = ""
        for word, mod in group[:-1]:
            if word in head or (mod and mod in head):
                continue
            mods += mod
        w = mods + head
        if w and w not in out:
            out.append(w)
        group = []

    start = 0
    if len(lows) > 1 and not _CJK.search(toks[0]) and (lows[0] in ENV_TYPE_WORDS or lows[0] in _CATEGORY):
        c = _CATEGORY.get(lows[0]) or ENV_TYPE_WORDS.get(lows[0], "")
        if c:
            cats.append(c)
        start = 1
    i = start
    while i < len(toks):
        t, low = toks[i], lows[i]
        nxt = lows[i + 1] if i + 1 < len(toks) else ""
        if _CJK.search(t):
            flush()
            word = re.sub(r"\d+$", "", t)
            word = re.sub(r"(体块|块)$", "", word) or word
            word = {"地形": "地面"}.get(word, word)
            if word and word not in _GENERIC and re.sub(r"\d+", "", t) not in _GENERIC and word not in out:
                out.append(word)
            i += 1
            continue
        if low == "on" and group:
            on_target = True
            i += 1
            continue
        if low == "and":
            i += 1
            continue
        if (low, nxt) in _EN_PAIRS:
            word, mod, i = _EN_PAIRS[(low, nxt)], _EN_PAIRS[(low, nxt)], i + 2
        elif low in _EN_COLOURS:
            colour, i = _EN_COLOURS[low], i + 1
            continue
        elif low in _EN_PARTS and group:
            word, mod = group[-1]
            part = _EN_PARTS[low]
            if not word.endswith(part):
                group[-1] = (word + part, mod)
            i += 1
            continue
        elif low in _CATEGORY:
            if _CATEGORY[low] and _CATEGORY[low] not in cats:
                cats.append(_CATEGORY[low])
            i += 1
            continue
        elif not low or low in _GENERIC or low in _STOP:
            i += 1
            continue
        elif low in _STYLE:
            flush()
            if _EN_HINTS[low] not in out:
                out.append(_EN_HINTS[low])
            i += 1
            continue
        elif low in _EN_HINTS or (low.endswith("s") and low[:-1] in _EN_HINTS):
            key = low if low in _EN_HINTS else low[:-1]
            word, mod, i = _EN_HINTS[key], _MODIFIER.get(key, _EN_HINTS[key]), i + 1
        else:
            i += 1  # unknown latin word (Cube, Tower01x…): no hint
            continue
        if colour:
            word, mod, colour = colour + word, colour + mod, ""
        if on_target and group:
            # "X on Y" → Y上的X
            thing = group[-1][0]
            group[-1] = (word + "上的" + thing, word + "上的" + thing)
            on_target = False
            continue
        group.append((word, mod))
    flush()
    return out, cats


def _drop_covered(words):
    """Drop a word another word already contains ("墙" next to "墙面搁板"; "窗" next to "窗帘" stays)."""
    out = []
    for w in words:
        if any(w != o and len(w) == 1 and o.startswith(w) and o[1:] in ("面", "板", "体") for o in words):
            continue
        if w == "墙面" and any(o != w and o.endswith("墙") for o in words):
            continue
        out.append(w)
    return out


def hint_from_name(name, custom="") -> str:
    """Design hint of one block: a custom property wins, else the meaningful words of its name.

    "楼_玻璃幕墙" → "楼，玻璃幕墙"; "Tower_glass.001" → "高楼，玻璃幕墙"; "Cube.003" → "";
    "furniture_sofa_back" → "沙发靠背"; "furniture_coffee_table" → "茶几"."""
    custom = str(custom or "").strip()
    if custom:
        return custom[:60]
    words, cats = _name_words(name)
    words = _drop_covered(words)
    return "，".join(words or [])[:60]


def is_interior(boxes, product_box) -> bool:
    """Room-like blockout: a horizontal slab over the product, or tall thin walls on ≥ 2 sides of it.

    ``boxes`` / ``product_box``: world boxes (x0, y0, z0, x1, y1, z1)."""
    if not product_box:
        return False
    px0, py0, pz0, px1, py1, pz1 = product_box
    cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
    sides = set()
    for b in boxes or ():
        x0, y0, z0, x1, y1, z1 = b
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        if x0 <= cx <= x1 and y0 <= cy <= y1 and z0 >= pz1 and dz < 0.6 and min(dx, dy) > 1.0 and z0 - pz0 < 6.0:
            return True
        if dz >= 2.0 and min(dx, dy) <= 0.4 and max(dx, dy) >= 1.0 and z0 - pz0 < 0.5:
            if dx <= dy:
                sides.add("x-" if x1 <= cx else "x+")
            else:
                sides.add("y-" if y1 <= cy else "y+")
    return len(sides) >= 2


def shape_hint(box, floor_z=0.0, interior=False, room_top=None) -> str:
    """Fallback name from size / height alone (metres), e.g. "沙发", "墙面", "建筑"; "" when nothing fits."""
    x0, y0, z0, x1, y1, z1 = box
    dx, dy, dz = max(0.0, x1 - x0), max(0.0, y1 - y0), max(0.0, z1 - z0)
    long_, short = max(dx, dy), min(dx, dy)
    bottom, top = z0 - floor_z, z1 - floor_z
    if interior:
        if dz < 0.08 and long_ >= 2.0 and bottom < 0.1:
            return "地面（地板 / 地毯）"
        if room_top is not None and dz < 0.35 and long_ >= 0.8 and z1 >= room_top - 0.35 and bottom > 1.8:
            return "天花板" if short >= 1.5 else "吊顶 / 梁"
        if short < 0.3 and dz >= 1.8:
            return "墙面" if long_ >= 1.0 else "门框 / 窗帘 / 柱子"
        if short < 0.15 and 0.3 <= dz < 1.8 and bottom >= 0.5:
            return "墙上的装饰（挂画 / 电视 / 墙板）"
        if dz < 0.08 and bottom >= 0.5 and long_ < 2.5:
            return "墙面搁板"
        if long_ < 0.5 and top <= 0.3:
            return "地上的小摆件"
        if long_ < 0.5 and bottom >= 0.3:
            return "桌面 / 柜面上的小摆件"
        if long_ < 0.6 and dz >= 0.4 and top <= 1.6:
            return "盆栽 / 落地灯 / 小边几"
        if top <= 0.55 and long_ >= 0.6:
            return "茶几 / 矮柜"
        if top <= 1.1 and long_ >= 1.4 and short >= 0.6:
            return "沙发"
        if 0.65 <= top <= 0.9 and long_ >= 0.8 and short >= 0.6:
            return "桌子"
        if top <= 1.1 and long_ >= 1.4:
            return "矮柜 / 电视柜"
        if top <= 1.1:
            return "椅子 / 边柜"
        if dz >= 1.1 and short <= 0.7:
            return "柜子 / 书架"
        return "高柜 / 大型家具"
    if dz < 0.3 and long_ >= 3.0 and bottom < 0.3:
        return "道路 / 人行道" if long_ >= 4 * max(short, 1e-6) else "地面（广场 / 草地）"
    if dz >= 3.0 and short >= 3.0:
        return "高楼" if dz >= 24.0 else "建筑"
    if short < 1.5 and dz >= 2.5:
        return "树 / 路灯 / 柱子"
    if dz < 1.5 and long_ >= 2.0 and long_ >= 4 * max(short, 1e-6):
        return "矮墙 / 围栏 / 绿篱"
    if dz < 1.5:
        return "花坛 / 台阶 / 低矮构筑物"
    return "小型构筑物（亭子 / 售货亭）"


def derive_hint(name="", custom="", auto=False, materials=(), label="", env_type="", box=None,
                floor_z=0.0, interior=False, room_top=None):
    """(hint, source) for one block, first match wins:
    the user's own hint (``custom``, not ``auto``) › object-name words › material-name words ›
    an auto-stored hint (created by 环境体块) › analysis label / element type › category words
    of the names › a size / height guess (marked 按尺寸推测).  source: custom / name / material /
    auto / label / category / shape / ""."""
    custom = str(custom or "").strip()
    if custom and not auto:
        return custom[:60], "custom"
    words, cats = _name_words(name)
    if words:
        return "，".join(_drop_covered(words))[:60], "name"
    for m in materials or ():
        if not m or str(m).startswith("_wondful"):
            continue
        mw, mc = _name_words(m)
        if mw:
            return "，".join(_drop_covered(mw))[:60], "material"
        cats += [c for c in mc if c not in cats]
    if custom:
        return custom[:60], "auto"
    label = str(label or "").strip()
    if label:
        lw, lc = _name_words(label)
        if lw:
            return "，".join(_drop_covered(lw))[:60], "label"
        if _CJK.search(label):
            return label[:60], "label"
    et = ENV_TYPE_WORDS.get(str(env_type or "").strip().lower(), "")
    if et:
        return et, "label"
    if cats:
        return "，".join(cats)[:60], "category"
    if box is not None:
        guess = shape_hint(box, floor_z, interior, room_top)
        if guess:
            return guess + SHAPE_NOTE, "shape"
    return "", ""


def placement_words(x0, x1, y0, y1, depth="SAME", height_ratio=0.0, product_x=None) -> str:
    """Where one block sits in the frame.

    ``x0..x1`` / ``y0..y1``: its projected box in normalised image coords (x: 0 left → 1 right,
    y: 0 bottom → 1 top), may extend past 0..1.  ``depth``: FRONT / SAME / BACK relative to the
    product.  ``height_ratio``: block height / product height (0 = unknown).  ``product_x``:
    the product's projected (x0, x1), for 产品左侧 / 右侧 / 正后方."""
    vx0, vx1 = max(0.0, x0), min(1.0, x1)
    cx = (vx0 + vx1) / 2.0
    width = vx1 - vx0
    if width > 0.7:
        side = "横跨画面"
    elif cx < 1.0 / 3.0:
        side = "画面左侧"
    elif cx > 2.0 / 3.0:
        side = "画面右侧"
    else:
        side = "画面中间"
    parts = [side]
    if product_x is not None:
        px0, px1 = product_x
        if cx < px0:
            parts.append("在产品左边")
        elif cx > px1:
            parts.append("在产品右边")
        else:
            parts.append({"BACK": "在产品正后方", "FRONT": "在产品正前方"}.get(depth, "与产品重叠"))
    parts.append({"BACK": "背景（产品后方）", "FRONT": "前景（产品前方，离镜头更近）"}.get(
        depth, "与产品大致同一深度（产品旁边）"))
    if y1 > 1.02:
        parts.append("顶部超出画面")
    elif y1 > 0.66:
        parts.append("高至画面上部")
    elif y1 < 0.45:
        parts.append("较低矮，顶部在画面中下部")
    if x0 < -0.02 or x1 > 1.02:
        parts.append("一侧延伸出画面")
    if height_ratio >= 1.0:
        parts.append(f"高约为产品的 {height_ratio:.1f} 倍")
    elif height_ratio > 0:
        parts.append(f"高约为产品的 {height_ratio:.1f} 倍（比产品矮）")
    return "，".join(parts)


def depth_word(near, far, product_depth) -> str:
    """FRONT / SAME / BACK from camera-space distances (block nearest / farthest, product centre)."""
    if product_depth <= 0:
        return "SAME"
    if near > product_depth * 1.08:
        return "BACK"
    if far < product_depth * 0.92:
        return "FRONT"
    return "SAME"


def build_massing_text(items, vehicle=False, interior=None) -> str:
    """Prompt paragraph for the blocks. ``items``: dicts with ``place`` and optional ``hint``.

    ``interior`` (None = read from the items' ``interior`` key): room blockouts are described
    as 室内物体 (walls, furniture, windows…) instead of buildings."""
    items = [i for i in items or () if i.get("place")]
    if not items:
        return ""
    if interior is None:
        interior = any(bool(i.get("interior")) for i in items)
    subj = "车" if vehicle else "产品"
    named = any((i.get("hint") or "").strip() for i in items)
    if interior:
        lines = [
            f"画面里的灰蓝色哑光体块（共 {len(items)} 个）= 房间里的{INTERIOR_MARK}（墙面、地面、天花、门窗、窗帘、家具、摆件等）的体量占位，不是产品："
            "按下面每个体块「」里的名称把它画成符合场景风格的真实物体（名称写的是墙就画成墙面，写的是沙发就画成沙发，以此类推），"
            "保持它在画面中的位置、高度、轮廓和透视（与地面、墙面的接触位置不变），材质、颜色、纹理和细节按场景风格设计；"
            "不要画成纯色方块，不要保留灰蓝色，不要把体块当成产品，不要把体块画成建筑或别的东西，物体上不要出现任何品牌 Logo 或文字，"
            "也不要删除体块或改变它们的前后遮挡关系。",
        ]
    else:
        lines = [
            f"画面里的灰蓝色哑光体块（共 {len(items)} 个）= 环境里的建筑 / 结构的体量占位，不是产品："
            "把每个体块画成符合场景风格的真实建筑或结构" + ("（「」里的名称写的是什么就画成什么）" if named else "")
            + "，保持它在画面中的位置、高度、轮廓和透视（地面接触位置不变），"
            "材质、立面、窗户、招牌、细节可以按场景风格自由设计，体量允许略有退台、檐口、凹凸等变化；"
            "不要画成纯色方块，不要保留灰蓝色，不要把体块当成产品，建筑上不要出现任何品牌 Logo、招牌文字或产品品牌名，也不要删除体块或改变它们的前后遮挡关系。",
        ]
    lines.append(f"{subj}是保持自身材质颜色的主体，不属于体块，{subj}继续严格锁定（位置、大小、角度、轮廓不变），"
                 f"体块变成的{'物体' if interior else '建筑'}不得遮挡、穿插或改变{subj}。")
    if named:
        lines.append("各体块（「」里是体块的名称 / 设计提示，按它设计该体块；标「按尺寸推测」的是按大小估计的，可按场景合理调整）："
                     if any(SHAPE_NOTE in (i.get("hint") or "") for i in items)
                     else "各体块（「」里是体块的名称 / 设计提示，按提示设计该体块）：")
    else:
        lines.append("各体块：")
    for n, it in enumerate(items, start=1):
        hint = (it.get("hint") or "").strip()
        lines.append(f"- 体块 {n}" + (f"「{hint}」" if hint else "") + "：" + it["place"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- 建筑参考 (per-block refs)

def parse_refs(value) -> list:
    """Paths stored in a block's ``wondful_ref_paths`` (JSON list; a plain string = one path)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except ValueError:
            raw = [text]
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
    out = []
    for p in raw:
        p = str(p or "").strip()
        if p and p not in out:
            out.append(p)
    return out


def dump_refs(paths) -> str:
    return json.dumps(parse_refs(list(paths or [])), ensure_ascii=False)


def merge_refs(existing, new, limit: int = MAX_BLOCK_REFS):
    """(kept list, dropped list): existing refs first, duplicates ignored, at most ``limit``."""
    kept = parse_refs(existing)
    dropped = []
    for p in parse_refs(list(new or [])):
        if p in kept:
            continue
        if len(kept) >= limit:
            dropped.append(p)
        else:
            kept.append(p)
    return kept, dropped


def visible_area(box) -> float:
    """Frame area (0..1) of a block's projected box [x0, x1, y0, y1], clipped to the frame."""
    try:
        x0, x1, y0, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return 0.0
    w = max(0.0, min(1.0, x1) - max(0.0, x0))
    h = max(0.0, min(1.0, y1) - max(0.0, y0))
    return w * h


def short_place(place) -> str:
    """First two parts of a placement phrase ("画面左侧，在产品左边")."""
    parts = [p for p in str(place or "").split("，") if p]
    return "，".join(parts[:2])


def select_building_refs(items, cap: int = MAX_BUILDING_IMAGES, exists=None) -> list:
    """Which block reference images go with this generation (after the clay canvas).

    ``items``: the prompt's block list (same order / numbering as ``build_massing_text``), each
    with ``refs`` (paths), ``box`` and ``hint``.  Over the cap, blocks larger in frame win: one
    image per block in that order first, then second images, …  Returned grouped by block (in
    priority order) as dicts ``{block, path, hint, place, area}``; ``block`` is 1-based."""
    cands = []
    for n, it in enumerate(items or (), start=1):
        refs = [p for p in parse_refs(it.get("refs")) if exists is None or exists(p)][:MAX_BLOCK_REFS]
        if refs:
            cands.append((visible_area(it.get("box")), n, it, refs))
    cands.sort(key=lambda c: (-c[0], c[1]))
    take = {n: 0 for _a, n, _i, _r in cands}
    left = max(0, int(cap))
    rnd = 0
    while left > 0 and any(take[n] < len(r) for _a, n, _i, r in cands):
        for _a, n, _i, refs in cands:
            if left > 0 and rnd < len(refs):
                take[n] += 1
                left -= 1
        rnd += 1
    out = []
    for area, n, it, refs in cands:
        for p in refs[:take[n]]:
            out.append({"block": n, "path": p, "hint": str(it.get("hint") or "").strip(),
                        "place": str(it.get("place") or ""), "area": round(area, 4)})
    return out


def building_ref_text(selection, first_index: int = 2, vehicle: bool = False, names=None) -> str:
    """Prompt paragraph mapping each attached reference image to its block.

    ``selection``: ``select_building_refs`` output in attach order.  Images are called
    图 ``first_index``…, or by ``names`` (one label per image, e.g. file names for the codex exec
    fallback, whose canvas is the camera capture without grey-blue blocks)."""
    sel = [s for s in selection or () if s.get("path")]
    if not sel:
        return ""
    subj = "车" if vehicle else "产品"
    groups = []
    for i, s in enumerate(sel):
        label = (f"参考图「{names[i]}」" if names else f"图{first_index + i}")
        if groups and groups[-1][0]["block"] == s["block"]:
            groups[-1][1].append(label)
        else:
            groups.append((s, [label]))
    grey = "" if names else "灰蓝"
    lines = [f"建筑参考图（每张只管对应的那一个{grey}体块）："]
    for s, labels in groups:
        where = short_place(s.get("place")) or "画面中"
        name = f"体块 {s['block']}" + (f"「{s['hint']}」" if s.get("hint") else "")
        lines.append(f"- {'、'.join(labels)} 是{where}的{name}（{grey}体块）的参考：按它的建筑风格、立面、材质、窗户和细节来画这栋建筑，"
                     "但保持体块的位置、高度、轮廓和透视；参考图里的机位、构图、天空、其他物体和车辆一律不要照搬。")
    lines.append("没有参考图的体块仍按场景风格自由设计。参考图只影响对应的体块，不改变画面构图、地面和其他元素。")
    who = f"相机底图里的{subj}" if names else f"图 1 里不是灰蓝色的主体就是我们的{subj}，它"
    lines.append(f"{who}完全锁定：位置、大小、角度、透视、轮廓、颜色都与{'底图' if names else '图 1 '}一致，"
                 f"参考图不影响{subj}；不要按参考图的机位重新构图，也不要放大、缩小或移动{subj}。")
    return "\n".join(lines)


# --------------------------------------------------------------------------- material swap

def swap_materials(objects, material, restore) -> int:
    """Show ``material`` on every slot of ``objects`` through object-level links.

    The meshes' own material lists are not edited (a shared mesh keeps its look on other
    objects); objects without slots get one temporary slot.  ``restore`` receives undo steps
    (run in reverse order).  Returns the number of objects changed."""
    count = 0
    for obj in objects or ():
        data = getattr(obj, "data", None)
        mats = getattr(data, "materials", None)
        if mats is None:
            continue
        added = False
        if len(obj.material_slots) == 0:
            mats.append(None)
            added = True
            # clear() (not pop()) also drops the object's own slot; the list was empty before
            restore.append(lambda mats=mats: mats.clear() if len(mats) <= 1 else mats.pop())
        for slot in list(obj.material_slots):
            old_link, old_obj_mat = slot.link, None
            if old_link == "OBJECT":
                old_obj_mat = slot.material
            else:
                slot.link = "OBJECT"
                old_obj_mat = slot.material  # the object-level material behind the slot
            slot.material = material

            def undo(slot=slot, link=old_link, mat=old_obj_mat):
                if slot.link != "OBJECT":
                    slot.link = "OBJECT"
                slot.material = mat
                slot.link = link
            restore.append(undo)
        count += 1
    return count


def make_material(new_material):
    """Configure a fresh material (``new_material(name)``) as matte grey-blue."""
    mat = new_material(MAT_NAME)
    rgba = massing_rgba()
    try:
        mat.diffuse_color = rgba
    except Exception:
        pass
    try:
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = rgba
            bsdf.inputs["Roughness"].default_value = 0.9
            for key in ("Specular IOR Level", "Specular"):
                if key in bsdf.inputs:
                    bsdf.inputs[key].default_value = 0.15
                    break
    except Exception:
        pass
    return mat


# --------------------------------------------------------------------------- Blender side

def find_collection(scene, props=None):
    """The massing collection: the one picked in 高级, else an auto-detected one by name.

    Never the product collection (or one inside / containing it)."""
    import bpy
    if props is not None and not bool(getattr(props, "massing_enabled", True)):
        return None
    product = getattr(props, "product_collection", None) if props is not None else None
    picked = getattr(props, "massing_collection", None) if props is not None else None

    def clashes(c):
        if product is None or c is None:
            return False
        if c == product:
            return True
        return c.name in {x.name for x in product.children_recursive} or \
            product.name in {x.name for x in c.children_recursive}
    if picked is not None:
        return None if clashes(picked) else picked
    in_scene = {c.name: c for c in scene.collection.children_recursive}
    name = auto_pick(in_scene.keys(), exclude=[c.name for c in in_scene.values() if clashes(c)])
    return in_scene.get(name) if name else None


def is_scene_blockout_block(obj) -> bool:
    """A block made by 「从参考图生成场景粗模」 (tagged per group by scene_blockout_build, or in its collection)."""
    try:
        if obj.get(SB_GROUP_PROP) or obj.get("wondful_sb_wrap"):
            return True
        return any(c.get("wondful_scene_blockout") for c in getattr(obj, "users_collection", ()) or ())
    except Exception:
        return False


def block_ref_label(name, n_refs, from_scene_ref=False) -> str:
    """建筑参考 list line: user refs win; a 场景粗模 block without its own refs came from the scene photo."""
    if n_refs:
        return f"{name} · {n_refs} 张参考"
    return f"{name} · 来自场景参考图" if from_scene_ref else f"{name} · 无参考"


def yaw_of(matrix) -> float:
    """Z rotation (rad) of a 4x4 / 3x3 world matrix (rows), scale-independent; tilt is ignored."""
    ax, ay = float(matrix[0][0]), float(matrix[1][0])
    if math.hypot(ax, ay) > 1e-9:
        return math.atan2(ay, ax)
    return math.atan2(-float(matrix[0][1]), float(matrix[1][1]))   # X axis points straight up / down


def placeholder_box(corners, yaw) -> dict:
    """The placeholder's box in its own Z-rotated frame: centre (x, y), bottom z, dims (w along its X,
    d along its Y, h), yaw.  ``corners`` = its 8 world-space bound-box corners."""
    c, s = math.cos(-yaw), math.sin(-yaw)
    loc = [(c * p[0] - s * p[1], s * p[0] + c * p[1], float(p[2])) for p in corners]
    xs, ys, zs = [p[0] for p in loc], [p[1] for p in loc], [p[2] for p in loc]
    mx, my = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    cw, sw = math.cos(yaw), math.sin(yaw)
    return {"centre": [cw * mx - sw * my, sw * mx + cw * my], "bottom": min(zs),
            "dims": [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)], "yaw": yaw}


def fit_to_placeholder(block_min, block_max, box, fit=True) -> dict:
    """Transform for a block (local bound box ``block_min`` / ``block_max``, scale 1) standing in
    ``box`` (``placeholder_box``): same yaw, bottom on the box bottom, centred in XY, and with
    ``fit`` scaled uniformly (three-view proportions kept) to the largest size inside the box;
    without ``fit`` (user-given 长宽高) scale 1, only position / rotation.  → location, yaw, scale, dims."""
    bd = [max(float(block_max[i]) - float(block_min[i]), 1e-9) for i in range(3)]
    s = 1.0
    if fit:
        pd = box["dims"]
        s = min(pd[0] / bd[0], pd[1] / bd[1], pd[2] / bd[2])
        if not (s > 1e-6):
            raise ValueError("占位体太小或是扁平的，无法按它的大小缩放")
    yaw = float(box["yaw"])
    cx = (float(block_min[0]) + float(block_max[0])) / 2.0 * s
    cy = (float(block_min[1]) + float(block_max[1])) / 2.0 * s
    c, si = math.cos(yaw), math.sin(yaw)
    loc = [box["centre"][0] - (c * cx - si * cy), box["centre"][1] - (si * cx + c * cy),
           box["bottom"] - float(block_min[2]) * s]
    return {"location": loc, "yaw": yaw, "scale": s, "dims": [d * s for d in bd], "fitted": bool(fit)}


def _extra_objects(collection):
    """Blocks tagged ENV_EXTRA_TAG in a scene that contains ``collection`` (Blender only)."""
    try:
        import bpy
    except ImportError:
        return []
    try:
        scenes = [sc for sc in bpy.data.scenes
                  if sc.collection == collection or collection in sc.collection.children_recursive]
        if not scenes:
            return []
        return [o for o in bpy.data.objects
                if o.get(ENV_EXTRA_TAG) and any(sc in scenes for sc in o.users_scene)]
    except Exception:
        return []


def massing_objects(collection, product_collection=None):
    """Renderable geometry of the massing collection (product objects never included), plus blocks tagged
    ``ENV_EXTRA_TAG`` (built at a placeholder in a collection outside it) in the same scene."""
    if collection is None:
        return []
    prod = set(product_collection.all_objects) if product_collection is not None else set()
    out = []
    seen = set()
    for o in list(collection.all_objects) + _extra_objects(collection):
        if id(o) in seen:
            continue
        seen.add(id(o))
        if o in prod or o.hide_render or o.name.startswith("_wondful_"):
            continue
        if getattr(getattr(o, "data", None), "materials", None) is None:
            continue
        out.append(o)
    return out


def _roots(objs):
    s = set(objs)
    return [o for o in objs if o.parent is None or o.parent not in s]


def _world_bbox(objs):
    from mathutils import Vector
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    return pts


def snap_to_ground(context, objs, floor, ref_z, restore) -> list:
    """Temporarily rest ground-standing blocks on a sculpted clay floor (坡顶 / 自定义 terrain).

    A block counts as ground-standing when its bottom is near ``ref_z`` (the level the user
    modelled on); it moves straight down/up to the lowest floor point under its footprint so
    it never floats.  Raised blocks (stacked, on a podium) keep their place."""
    from mathutils import Vector, Matrix
    moved = []
    if floor is None:
        return moved
    context.view_layer.update()
    inv = floor.matrix_world.inverted()
    kids = {}
    for o in objs:
        top = o
        while top.parent is not None and top.parent in objs:
            top = top.parent
        kids.setdefault(top, []).append(o)
    for root, members in kids.items():
        pts = _world_bbox(members)
        if not pts:
            continue
        zs = [p.z for p in pts]
        bottom, height = min(zs), max(zs) - min(zs)
        if abs(bottom - ref_z) > max(0.3, 0.15 * height):
            continue
        xs = [p.x for p in pts]; ys = [p.y for p in pts]
        samples = [(x, y) for x in (min(xs), (min(xs) + max(xs)) / 2, max(xs))
                   for y in (min(ys), (min(ys) + max(ys)) / 2, max(ys))]
        hits = []
        for x, y in samples:
            o = inv @ Vector((x, y, bottom + 1000.0))
            d = (inv.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
            ok, loc, _n, _i = floor.ray_cast(o, d, distance=1e5)
            if ok:
                hits.append((floor.matrix_world @ loc).z)
        if not hits:
            continue
        dz = min(hits) - bottom
        if abs(dz) < 1e-4:
            continue
        from .viewport_capture import _basis_snapshot, _basis_restore
        snap = _basis_snapshot(root)
        restore.append(lambda r=root, s=snap: _basis_restore(r, s))
        root.matrix_world = Matrix.Translation((0.0, 0.0, dz)) @ root.matrix_world
        moved.append((root.name, round(dz, 3)))
    context.view_layer.update()
    return moved


def _box_of(pts):
    if not pts:
        return None
    return (min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts),
            max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts))


def scene_context(point_groups, product_pts):
    """(interior, floor_z, room_top) of a blockout from its blocks' world points and the product's."""
    boxes = [b for b in (_box_of(p) for p in point_groups or ()) if b is not None]
    pbox = _box_of(list(product_pts or ()))
    floor_z = pbox[2] if pbox else (min(b[2] for b in boxes) if boxes else 0.0)
    interior = is_interior(boxes, pbox)
    room_top = max(b[5] for b in boxes) if (interior and boxes) else None
    return interior, floor_z, room_top


def object_hint_fields(obj) -> dict:
    """What a Blender block carries for naming: custom hint (+ auto flag), material names, analysis label / type."""
    out = {"name": getattr(obj, "name", ""), "custom": "", "auto": False, "materials": [], "label": "", "env_type": ""}
    try:
        keys = set(obj.keys())
    except Exception:
        keys = set()
    for key in HINT_PROPS:
        if key in keys:
            try:
                val = str(obj[key] or "").strip()
            except Exception:
                val = ""
            if val:
                out["custom"] = val
                break
    out["auto"] = bool(out["custom"]) and AUTO_HINT_FLAG in keys and bool(obj.get(AUTO_HINT_FLAG))
    for key, field in ((LABEL_PROP, "label"), ("wondful_env_type", "env_type")):
        if key in keys:
            try:
                out[field] = str(obj[key] or "")
            except Exception:
                pass
    try:
        out["materials"] = [s.material.name for s in obj.material_slots if s.material is not None]
    except Exception:
        out["materials"] = []
    return out


def group_hint(members, box=None, floor_z=0.0, interior=False, room_top=None, shape=True):
    """(hint, source) of a block group (root first): the best-ranked source over its members."""
    rank = {"custom": 0, "name": 1, "material": 2, "auto": 3, "label": 4, "category": 5}
    best = ("", "")
    for m in members or ():
        f = object_hint_fields(m)
        hint, src = derive_hint(f["name"], f["custom"], f["auto"], f["materials"], f["label"], f["env_type"])
        if hint and (not best[0] or rank.get(src, 9) < rank.get(best[1], 9)):
            best = (hint, src)
            if src == "custom":
                break
    if not best[0] and shape and box is not None:
        guess = shape_hint(box, floor_z, interior, room_top)
        if guess:
            best = (guess + SHAPE_NOTE, "shape")
    return best


def store_label(obj, label) -> None:
    """Remember a creator's label (环境体块 / 按参考图搭建) as the block's auto hint: the user's own
    object name and an edited hint still win over it."""
    label = str(label or "").strip()
    if not label:
        return
    obj["wondful_hint"] = label[:60]
    obj[AUTO_HINT_FLAG] = 1


def scene_note_of(objs) -> str:
    """1.2.3 场景粗模: the season / colour line the analysis stored on its terrain block ("" when none)."""
    for o in objs or ():
        try:
            note = str(o.get("wondful_sb_scene_note", "") or "").strip()
        except Exception:
            note = ""
        if note:
            return note
    return ""


SB_KIND_PROP = "wondful_sb_kind"          # ④ 场景粗模: the analysis scene_kind ("outdoor" / "interior") on its collection
INFERRED_PROP = "wondful_detail_inferred"  # 场景补全 objects (detail_blockout.INFER_TAG): only built for outdoor analyses
SB_COLLECTION_PROP = "wondful_scene_blockout"
SB_TEXT = "wondful_scene_blockout.json"    # scene_blockout.TEXT_NAME (the analysis the blockout was built from)


def blockout_kind(objs) -> str:
    """Scene kind the ④ 场景粗模 analysis gave the blockout these blocks belong to: "outdoor", "interior",
    or "" when unknown (a hand-made massing collection — then the shape / ray checks decide).

    1.2.6 (室内误判): the analysis is the authority; 场景补全 objects exist only for outdoor analyses, so one
    of them means outdoor. Older files without the collection tag fall back to the stored analysis text."""
    from_sb = False
    for o in objs or ():
        try:
            if o.get(INFERRED_PROP):
                return "outdoor"
            for c in getattr(o, "users_collection", ()) or ():
                k = str(c.get(SB_KIND_PROP, "") or "")
                if k in ("outdoor", "interior"):
                    return k
                from_sb = from_sb or bool(c.get(SB_COLLECTION_PROP))
        except Exception:
            continue
    if from_sb:
        try:
            import bpy
            txt = bpy.data.texts.get(SB_TEXT)
            if txt is not None:
                k = str(json.loads(txt.as_string()).get("scene_kind", "") or "")
                if k in ("outdoor", "interior"):
                    return k
        except Exception:
            pass
    return ""


def describe_blocks(context, objs, product_pts, kind=None):
    """Placement items (camera projection) for the prompt; blocks fully off-screen are skipped."""
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    scene = context.scene
    cam = scene.camera
    if cam is None:
        return []
    pdepth, pheight, px = 0.0, 0.0, None
    if product_pts:
        pp = [world_to_camera_view(scene, cam, p) for p in product_pts]
        pp = [p for p in pp if p.z > 1e-4]
        if pp:
            px = (min(p.x for p in pp), max(p.x for p in pp))
        c = sum(product_pts, Vector()) / len(product_pts)
        pdepth = world_to_camera_view(scene, cam, c).z
        zs = [p.z for p in product_pts]
        pheight = max(zs) - min(zs)
    groups = {}
    for o in objs:
        top = o
        while top.parent is not None and top.parent in objs:
            top = top.parent
        groups.setdefault(top, []).append(o)
    interior, floor_z, room_top = scene_context([_world_bbox(m) for m in groups.values()], product_pts)
    # 1.2.6: the ④ analysis decides when it exists (an outdoor street is never described as 室内物体, even when
    # temporary walls or 场景补全 rows close it in on two sides); unknown → the shape check above
    kind = blockout_kind(objs) if kind is None else str(kind or "")
    if kind in ("outdoor", "interior") and interior != (kind == "interior"):
        interior = kind == "interior"
        if interior and room_top is None:
            boxes = [b for b in (_box_of(_world_bbox(m)) for m in groups.values()) if b is not None]
            room_top = max(b[5] for b in boxes) if boxes else None
    items = []
    for root, members in groups.items():
        pts = _world_bbox(members)
        proj = [world_to_camera_view(scene, cam, p) for p in pts]
        front = [p for p in proj if p.z > 1e-4]
        if not front:
            continue
        xs = [p.x for p in front]; ys = [p.y for p in front]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if x1 < 0 or x0 > 1 or y1 < 0 or y0 > 1:
            continue
        zs = [p.z for p in pts]
        ratio = (max(zs) - min(zs)) / pheight if pheight > 1e-6 else 0.0
        depth = depth_word(min(p.z for p in front), max(p.z for p in front), pdepth)
        names = [root.name] + [m.name for m in members if m is not root]
        gbox = _box_of(pts)
        hint, source = group_hint([root] + [m for m in members if m is not root], gbox, floor_z, interior, room_top)
        refs = []
        for m in [root] + members:
            try:
                if REF_PROP in m.keys():
                    refs = [_abspath(p) for p in parse_refs(m[REF_PROP])]
                    break
            except Exception:
                pass
        items.append({"name": names[0], "hint": hint, "hint_source": source, "interior": interior,
                      "depth": depth, "refs": refs,
                      "box": [round(v, 3) for v in (x0, x1, y0, y1)], "height_ratio": round(ratio, 2),
                      "place": placement_words(x0, x1, y0, y1, depth, ratio, px), "_sort": (x0 + x1) / 2})
    items.sort(key=lambda i: i["_sort"])
    for i in items:
        i.pop("_sort", None)
    return items


def _abspath(p):
    try:
        import bpy
        return bpy.path.abspath(p)
    except Exception:
        return p


def block_root(obj, objs):
    """The block (a root of ``objs``, where its 建筑参考 are stored) that ``obj`` belongs to:
    ``obj`` itself or its nearest ancestor that is a root block; None when outside the blocks."""
    roots = set(_roots(list(objs or ())))
    while obj is not None and obj not in roots:
        obj = getattr(obj, "parent", None)
    return obj


def block_roots(objs):
    return _roots(list(objs or ()))


def get_refs(obj) -> list:
    try:
        return parse_refs(obj.get(REF_PROP)) if obj is not None else []
    except Exception:
        return []


def set_refs(obj, paths) -> None:
    paths = parse_refs(list(paths or []))
    if paths:
        obj[REF_PROP] = dump_refs(paths)
    elif REF_PROP in obj.keys():
        del obj[REF_PROP]


def ensure_collection(scene, props=None):
    """(collection, created): the massing collection, or a new 「环境体块」 one linked to the scene."""
    import bpy
    coll = find_collection(scene, props)
    if coll is not None:
        return coll, False
    coll = bpy.data.collections.new(DEFAULT_COLLECTION)
    scene.collection.children.link(coll)
    return coll, True
