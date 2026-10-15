"""1.2.0 环境体块: simple boxes / cylinders / ramps blocked out around the product.

In the clay base they get a temporary matte grey-blue material (restored afterwards) and
the image prompt asks the model to turn each block into a real building / structure at the
same position, height and silhouette, in the scene's style, while the product stays locked.

Pure-text helpers (name hints, placement words, prompt paragraph, material swap bookkeeping)
need no ``bpy`` so they are unit-tested directly; Blender-only parts import ``bpy`` lazily.
"""
from __future__ import annotations

import json
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

# English name words → Chinese hint (Chinese names are used as written)
_EN_HINTS = {
    "glass": "玻璃幕墙", "curtainwall": "玻璃幕墙", "brick": "砖墙", "stone": "石材", "concrete": "混凝土",
    "wood": "木结构", "timber": "木结构", "tree": "树", "trees": "树", "forest": "树林", "hill": "山",
    "mountain": "山", "rock": "岩石", "wall": "墙", "fence": "围栏", "ramp": "坡道", "stairs": "台阶",
    "steps": "台阶", "tower": "高楼", "building": "建筑", "house": "房屋", "shop": "店铺", "bridge": "桥",
    "pillar": "柱子", "column": "柱子", "hedge": "绿篱", "planter": "花坛", "billboard": "广告牌",
    "canopy": "雨棚", "pavilion": "亭子", "kiosk": "售货亭", "metal": "金属", "neon": "霓虹招牌",
}
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


def hint_from_name(name, custom="") -> str:
    """Design hint of one block: a custom property wins, else the meaningful words of its name.

    "楼_玻璃幕墙" → "楼，玻璃幕墙"; "Tower_glass.001" → "高楼，玻璃幕墙"; "Cube.003" → ""."""
    custom = str(custom or "").strip()
    if custom:
        return custom[:60]
    base = re.sub(r"\.\d+$", "", str(name or ""))
    out = []
    for tok in re.split(r"[_\-\s.·/|]+", base):
        t = tok.strip()
        if not t or t.isdigit():
            continue
        low = re.sub(r"\d+$", "", t.lower())
        if not low or low in _GENERIC or re.sub(r"\d+", "", t) in _GENERIC:
            continue
        if _CJK.search(t):
            word = re.sub(r"\d+$", "", t)
        elif low in _EN_HINTS:
            word = _EN_HINTS[low]
        else:
            continue  # unknown latin word (Cube, Tower01x…): no hint
        if word and word not in out:
            out.append(word)
    return "，".join(out)[:60]


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


def build_massing_text(items, vehicle=False) -> str:
    """Prompt paragraph for the blocks. ``items``: dicts with ``place`` and optional ``hint``."""
    items = [i for i in items or () if i.get("place")]
    if not items:
        return ""
    subj = "车" if vehicle else "产品"
    lines = [
        f"画面里的灰蓝色哑光体块（共 {len(items)} 个）= 环境里的建筑 / 结构的体量占位，不是产品："
        "把每个体块画成符合场景风格的真实建筑或结构，保持它在画面中的位置、高度、轮廓和透视（地面接触位置不变），"
        "材质、立面、窗户、招牌、细节可以按场景风格自由设计，体量允许略有退台、檐口、凹凸等变化；"
        "不要画成纯色方块，不要保留灰蓝色，不要把体块当成产品，建筑上不要出现任何品牌 Logo、招牌文字或产品品牌名，也不要删除体块或改变它们的前后遮挡关系。",
        f"{subj}是保持自身材质颜色的主体，不属于体块，{subj}继续严格锁定（位置、大小、角度、轮廓不变），"
        f"体块变成的建筑不得遮挡、穿插或改变{subj}。",
        "各体块（「」里是体块名称给出的设计提示，按提示设计该体块）：" if any((i.get("hint") or "").strip() for i in items)
        else "各体块：",
    ]
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


def massing_objects(collection, product_collection=None):
    """Renderable geometry of the massing collection (product objects never included)."""
    if collection is None:
        return []
    prod = set(product_collection.all_objects) if product_collection is not None else set()
    out = []
    for o in collection.all_objects:
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


def describe_blocks(context, objs, product_pts):
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
        hint = ""
        for m in [root] + members:
            custom = ""
            for key in HINT_PROPS:
                try:
                    if key in m.keys():
                        custom = str(m[key])
                        break
                except Exception:
                    pass
            hint = hint_from_name(m.name, custom)
            if hint:
                break
        refs = []
        for m in [root] + members:
            try:
                if REF_PROP in m.keys():
                    refs = [_abspath(p) for p in parse_refs(m[REF_PROP])]
                    break
            except Exception:
                pass
        items.append({"name": names[0], "hint": hint, "depth": depth, "refs": refs,
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
