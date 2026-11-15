"""Validated, source-only material finishing plans.

The catalog is read on Blender's main thread.  Parsing is deliberately pure so
the response from the existing prompt-polish request cannot execute code or
touch RNA data.
"""
from __future__ import annotations

import json
import math
import re

from .detail_paste import KEEP_PROP, classify_slot


_PAINT = re.compile(r"paint|body|车漆|车身", re.IGNORECASE)
_INTERIOR = re.compile(r"\binterior\b|内饰", re.IGNORECASE)
_SCALARS = {"roughness", "metallic", "coat_weight", "coat_roughness", "transmission_weight"}
_COLOUR = "base_color"
_VALUE_KEYS = _SCALARS | {_COLOUR}
_SOCKET_NAMES = {
    "roughness": "Roughness", "metallic": "Metallic", "coat_weight": "Coat Weight",
    "coat_roughness": "Coat Roughness", "transmission_weight": "Transmission Weight",
    "base_color": "Base Color",
}


def _linked_values(material):
    tree = getattr(material, "node_tree", None) if getattr(material, "use_nodes", False) else None
    if tree is None:
        return set()
    linked = set()
    for node in getattr(tree, "nodes", ()):
        if getattr(node, "type", "") != "BSDF_PRINCIPLED":
            continue
        for key, socket_name in _SOCKET_NAMES.items():
            inputs = getattr(node, "inputs", ())
            socket = inputs.get(socket_name) if hasattr(inputs, "get") else next(
                (item for item in inputs if getattr(item, "name", "") == socket_name), None)
            if socket is not None and getattr(socket, "is_linked", False):
                linked.add(key)
    return linked


def _current_values(material):
    """JSON-safe Principled defaults; linked values are deliberately omitted."""
    tree = getattr(material, "node_tree", None) if getattr(material, "use_nodes", False) else None
    node = next((item for item in getattr(tree, "nodes", ()) if getattr(item, "type", "") == "BSDF_PRINCIPLED"), None)
    if node is None:
        return {}
    values = {}
    inputs = getattr(node, "inputs", ())
    for key, socket_name in _SOCKET_NAMES.items():
        socket = inputs.get(socket_name) if hasattr(inputs, "get") else next(
            (item for item in inputs if getattr(item, "name", "") == socket_name), None)
        if socket is None or getattr(socket, "is_linked", False):
            continue
        value = getattr(socket, "default_value", None)
        if key == _COLOUR:
            try:
                rgb = [float(channel) for channel in value[:3]]
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(channel) and 0.0 <= channel <= 1.0 for channel in rgb):
                values[key] = rgb
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and 0.0 <= number <= 1.0:
                values[key] = number
    return values


def catalog(context, props):
    """Return JSON-safe material protection metadata for the selected product."""
    collection = getattr(props, "product_collection", None)
    rows = {}
    for obj in getattr(collection, "all_objects", ()):
        if (getattr(obj, "type", "") not in {"MESH", "CURVE", "FONT", "SURFACE"}
                or getattr(obj, "hide_render", False)):
            continue
        visible_get = getattr(obj, "visible_get", None)
        if callable(visible_get):
            try:
                if not visible_get(view_layer=getattr(context, "view_layer", None)):
                    continue
            except (RuntimeError, TypeError):
                continue
        object_keep = obj.get(KEEP_PROP) if hasattr(obj, "get") else None
        for slot in getattr(obj, "material_slots", ()):
            material = getattr(slot, "material", None)
            if material is None:
                continue
            name = str(getattr(material, "name", "") or "").strip()
            if not name:
                continue
            material_keep = material.get(KEEP_PROP) if hasattr(material, "get") else None
            keep = bool(classify_slot(getattr(obj, "name", ""), object_keep, name, material_keep))
            paint = bool(_PAINT.search(name) and not _INTERIOR.search(name))
            row = rows.setdefault(name, {"name": name, "paint": False, "keep": False,
                                         "linked": False, "linked_values": set(),
                                         "current_values": _current_values(material)})
            row["paint"] = row["paint"] or paint
            row["keep"] = row["keep"] or keep
            row["linked"] = row["linked"] or getattr(material, "library", None) is not None
            row["linked_values"].update(_linked_values(material))
    # If the naming convention cannot identify paint, the safe fallback is to
    # leave every Base Color alone while still permitting non-colour PBR polish.
    has_paint = any(row["paint"] for row in rows.values())
    for row in rows.values():
        row["base_color_locked"] = bool(row["paint"] or not has_paint)
        row["linked_values"] = sorted(row["linked_values"])
        row["editable_values"] = ([] if row["keep"] or row["linked"] else [
            key for key in sorted(_VALUE_KEYS)
            if key not in row["linked_values"] and not (key == _COLOUR and row["base_color_locked"])
        ])
    return [rows[name] for name in sorted(rows)]


def polish_contract(materials):
    """Instructions appended only to the source final-polish system message."""
    return (
        "\n\nMATERIAL FINISH PLAN (source-only, strict JSON response): return exactly one JSON object "
        "with keys `prompt` and `materials`. `prompt` is the user-visible Chinese image-edit prompt. "
        "`materials` is an array of `{name, values}`. Use only catalog names and only values "
        "`roughness`, `metallic`, `coat_weight`, `coat_roughness`, `transmission_weight`, and `base_color` "
        "(three finite scene-linear RGB numbers in [0,1]). Do not include keep or library-linked materials, "
        "and use only each material's `editable_values`; linked sockets in `linked_values` must remain intact. "
        "For paint or base_color_locked materials, omit only `base_color`; other listed PBR values are allowed. "
        "Infer PBR adjustments only from input image 3 (the identity reference). `current_values` are Blender "
        "starting values for material-name disambiguation, not image-derived PBR evidence or permission to recolor. "
        "An empty materials array is valid: it means retain native existing materials. "
        "Do not put JSON in prompt, do not invent materials, and do not propose a multi-view match.\n"
        "CATALOG:\n" + json.dumps(materials, ensure_ascii=False, separators=(",", ":"))
    )


def _number(value, key):
    if isinstance(value, bool):
        raise ValueError(f"材质 {key} 必须是有限数字")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"材质 {key} 必须是有限数字") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"材质 {key} 必须位于 [0, 1]")
    return number


def parse_reply(reply, materials):
    """Parse one strict JSON provider reply into ``(visible_prompt, plan)``."""
    try:
        data = json.loads(reply)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("最终润色必须返回严格的材质计划 JSON") from exc
    if not isinstance(data, dict) or set(data) != {"prompt", "materials"}:
        raise ValueError("最终润色 JSON 只能包含 prompt 和 materials")
    prompt = data["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("最终润色 JSON 缺少可见提示词")
    requested = data["materials"]
    if not isinstance(requested, list):
        raise ValueError("最终润色 JSON 的 materials 必须是列表")
    known = {row["name"]: row for row in materials if isinstance(row, dict) and isinstance(row.get("name"), str)}
    plan, seen = [], set()
    for item in requested:
        if not isinstance(item, dict) or set(item) != {"name", "values"}:
            raise ValueError("每个材质计划项只能包含 name 和 values")
        name, values = item["name"], item["values"]
        if not isinstance(name, str) or name not in known:
            raise ValueError("材质计划引用了未知材质")
        if name in seen:
            raise ValueError("材质计划包含重复材质")
        if known[name].get("keep") or known[name].get("linked"):
            raise ValueError("材质计划尝试修改受保护材质")
        if not isinstance(values, dict) or not values or set(values) - _VALUE_KEYS:
            raise ValueError("材质计划包含不支持的数值")
        if set(values) - set(known[name].get("editable_values", ())):
            raise ValueError("材质计划尝试修改受保护或已连接的参数")
        cleaned = {}
        for key, value in values.items():
            if key == _COLOUR:
                if not isinstance(value, (list, tuple)) or len(value) != 3:
                    raise ValueError("材质 base_color 必须是 RGB 三元组")
                cleaned[key] = [_number(channel, key) for channel in value]
            else:
                cleaned[key] = _number(value, key)
        seen.add(name)
        plan.append({"name": name, "values": cleaned})
    return prompt.strip(), plan
