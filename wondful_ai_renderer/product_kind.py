"""1.1.0: is the product a vehicle?  Pure text logic, no ``bpy``.

Car wording (车 / 车身 / 车轮 / 车漆 / 车牌) belongs in the image prompt only when
the product really is a vehicle; a chair or a sofa gets generic product wording
(产品 / 产品表面材质 / 底部接地点).  Detection uses the product Collection's
object, mesh and material names plus the user's own text (保留细节 / 产品外观 /
场景要求).
"""
from __future__ import annotations

import re

_NAME_RE = re.compile(
    r"(wheel|tire|tyre|vehicle|brake|caliper|license|plate_num|(^|[^a-z])car([^a-z]|$)"
    r"|车|轮胎|轮毂|轮子)"
)
HINT_WORDS = ("车", "轮毂", "轮胎")


def names_look_like_vehicle(names) -> bool:
    """True when any object / mesh / material / collection name looks like a vehicle part."""
    return any(_NAME_RE.search(str(n).lower()) for n in names or () if n)


def hint_says_vehicle(text) -> bool:
    """True when the user's own text talks about a car (车标、轮毂…)."""
    text = str(text or "")
    return any(w in text for w in HINT_WORDS)


def collection_names(collection) -> list:
    """Names to test for a Blender collection (works on any object with the same attributes)."""
    if not collection:
        return []
    names = [getattr(collection, "name", "")]
    for o in getattr(collection, "all_objects", []) or []:
        names.append(o.name)
        data = getattr(o, "data", None)
        if getattr(o, "type", "") == "MESH" and data:
            names.append(data.name)
            names += [m.name for m in getattr(data, "materials", []) if m]
    return names


def props_hint(props) -> str:
    return " ".join(str(getattr(props, k, "") or "")
                    for k in ("identity_details", "product_look_prompt", "scene_brief"))


def is_vehicle(props=None, collection=None) -> bool:
    if props is not None and hint_says_vehicle(props_hint(props)):
        return True
    if collection is None and props is not None:
        collection = getattr(props, "product_collection", None)
    return names_look_like_vehicle(collection_names(collection))


def words(vehicle: bool) -> dict:
    """Wording used by the prompt builders."""
    if vehicle:
        return {
            "subj": "车", "body": "车身", "surface": "车漆",
            "parts": "车轮、灯组、Logo、字标和车牌",
            "contact": "车轮", "contact_full": "四个车轮",
            "texture_tag": "车身质感",
            "identity": "Logo、品牌/车型字标、车牌文字或可读 UI",
            "kind": "汽车",
        }
    return {
        "subj": "产品", "body": "产品", "surface": "产品表面材质",
        "parts": "结构细节、部件、Logo、字标和铭牌",
        "contact": "底部接地点（脚、轮或底座）", "contact_full": "所有底部接地点",
        "texture_tag": "表面质感",
        "identity": "Logo、品牌/型号字标、铭牌文字或可读 UI",
        "kind": "产品",
    }
