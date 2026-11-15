"""3.1.7 product appearance lock: colour/material source, locked product look and
identity-detail checklist.

Pure Python so it can be unit-tested outside Blender. Blender material reading
lives in ``collect_product_materials`` and only touches ``bpy`` objects passed in.
"""
from __future__ import annotations

import colorsys
from typing import Any, Iterable

COLOR_SOURCES = ("BLENDER", "PRODUCT_REF", "TEXT")

_HUE_NAMES = (
    (8, "红"), (22, "橙红"), (40, "橙"), (65, "黄"), (90, "黄绿"), (160, "绿"),
    (190, "青"), (250, "蓝"), (290, "紫"), (335, "品红"), (361, "红"),
)


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _linear_to_srgb(c: float) -> float:
    c = _clamp01(c)
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def color_name(rgb: Iterable[float], linear: bool = True) -> str:
    """Human Chinese colour name plus hex, e.g. ``橙色 #D2541E``."""
    vals = [float(x) for x in list(rgb)[:3]]
    if len(vals) < 3:
        return ""
    if linear:
        vals = [_linear_to_srgb(v) for v in vals]
    r, g, b = (_clamp01(v) for v in vals)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    hex_code = "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))
    if v < 0.12:
        name = "黑色"
    elif s < 0.12:
        name = "白色" if v > 0.85 else ("浅灰色" if v > 0.6 else ("灰色" if v > 0.3 else "深灰色"))
    else:
        deg = h * 360
        hue = next(n for limit, n in _HUE_NAMES if deg < limit)
        if hue == "橙" and v < 0.55:
            hue = "棕"
        prefix = "深" if v < 0.4 else ("浅" if v > 0.85 and s < 0.45 else ("亮" if s > 0.75 and v > 0.75 else ""))
        name = f"{prefix}{hue}色"
    return f"{name} {hex_code}"


def describe_material(mat: dict) -> str:
    """One Chinese clause for a material dict (see collect_product_materials)."""
    name = str(mat.get("name", "") or "材质").strip()
    parts = []
    base = mat.get("base_color")
    if base is not None:
        cn = color_name(base, linear=True)
        if cn:
            parts.append(cn)
    metallic = _clamp01(mat.get("metallic", 0.0))
    rough = _clamp01(mat.get("roughness", 0.5))
    transmission = _clamp01(mat.get("transmission", 0.0))
    sheen = _clamp01(mat.get("sheen", 0.0))
    coat = _clamp01(mat.get("coat", 0.0))
    if transmission > 0.5:
        parts.append("透明/玻璃")
    elif metallic > 0.6:
        parts.append("金属")
    if coat > 0.3:
        parts.append("带清漆层")
    if sheen > 0.3:
        tint = mat.get("sheen_tint")
        tint_name = color_name(tint, linear=True).split(" ")[0] if tint is not None else ""
        if tint_name and tint_name not in ("白色", "浅灰色"):
            parts.append(f"织物绒面（{tint_name}绒光）")
        else:
            parts.append("织物绒面")
    if rough < 0.2:
        parts.append("高光泽")
    elif rough > 0.7:
        parts.append("哑光")
    else:
        parts.append("半哑光")
    if mat.get("textured"):
        parts.append("带贴图纹理")
    return f"{name}：" + "，".join(parts)


def material_summary(materials: list[dict], limit: int = 8) -> str:
    """Summarise the largest product materials (sorted by ``weight``)."""
    rows = [m for m in (materials or []) if isinstance(m, dict)]
    rows.sort(key=lambda m: -float(m.get("weight", 0) or 0))
    lines = [describe_material(m) for m in rows[:limit]]
    return "；".join(x for x in lines if x)


def _image_average(img):
    try:
        import numpy as np
        w, h = img.size
        if w * h == 0:
            return None
        buf = np.empty(w * h * img.channels, dtype=np.float32)
        img.pixels.foreach_get(buf)
        px = buf.reshape(-1, img.channels)[:, :3]
        step = max(1, len(px) // 20000)
        mean = px[::step].mean(axis=0)
        # Byte images are exposed in their own colour space; convert sRGB to linear
        # so every value returned here is scene-linear like socket defaults.
        def lin(c):
            c = float(c)
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        cs = getattr(getattr(img, "colorspace_settings", None), "name", "sRGB")
        if getattr(img, "is_float", False) or cs not in ("sRGB",):
            return [float(x) for x in mean]
        return [lin(x) for x in mean]
    except Exception:
        return None


def _socket_color(socket, depth: int = 0):
    """Approximate scene-linear colour flowing into ``socket`` (textures averaged)."""
    if socket is None:
        return None
    if not getattr(socket, "is_linked", False):
        try:
            return [float(x) for x in list(socket.default_value)[:3]]
        except Exception:
            return None
    if depth > 6:
        return None
    node = socket.links[0].from_node
    kind = getattr(node, "type", "")
    if kind == "TEX_IMAGE" and getattr(node, "image", None) is not None:
        return _image_average(node.image)
    if kind in ("MIX", "MIX_RGB"):
        ins = [i for i in node.inputs if getattr(i, "type", "") == "RGBA" and getattr(i, "enabled", True)]
        if len(ins) >= 2:
            a = _socket_color(ins[0], depth + 1)
            b = _socket_color(ins[1], depth + 1)
            if a is None or b is None:
                return a or b
            blend = getattr(node, "blend_type", "MIX")
            fac_sock = next((i for i in node.inputs if i.name in ("Factor", "Fac") and getattr(i, "enabled", True)), None)
            try:
                fac = 1.0 if fac_sock is None or fac_sock.is_linked else float(fac_sock.default_value if not hasattr(fac_sock.default_value, "__len__") else fac_sock.default_value[0])
            except Exception:
                fac = 1.0
            if blend == "MULTIPLY":
                mixed = [x * y for x, y in zip(a, b)]
            else:
                mixed = b
            return [x * (1 - fac) + y * fac for x, y in zip(a, mixed)]
    for inp in getattr(node, "inputs", ()):
        if getattr(inp, "type", "") == "RGBA" and getattr(inp, "is_linked", False):
            found = _socket_color(inp, depth + 1)
            if found is not None:
                return found
    return None


def collect_product_materials(objects) -> list[dict]:
    """Read Principled BSDF values from Blender mesh objects (main thread only)."""
    found: dict[str, dict] = {}
    for obj in objects or []:
        if getattr(obj, "type", "") != "MESH":
            continue
        data = getattr(obj, "data", None)
        poly_count = len(getattr(data, "polygons", ()) or ()) if data else 0
        for slot in getattr(obj, "material_slots", ()) or ():
            mat = getattr(slot, "material", None)
            if mat is None:
                continue
            entry = found.get(mat.name)
            if entry is None:
                entry = {"name": mat.name, "weight": 0, "base_color": list(getattr(mat, "diffuse_color", (0.8, 0.8, 0.8, 1)))[:3],
                         "metallic": getattr(mat, "metallic", 0.0), "roughness": getattr(mat, "roughness", 0.5)}
                tree = getattr(mat, "node_tree", None) if getattr(mat, "use_nodes", False) else None
                if tree is not None:
                    for node in tree.nodes:
                        if getattr(node, "type", "") != "BSDF_PRINCIPLED":
                            continue
                        def _inp(*names):
                            for n in names:
                                sock = node.inputs.get(n)
                                if sock is not None:
                                    return sock
                            return None
                        bc = _inp("Base Color")
                        if bc is not None:
                            entry["base_color"] = list(bc.default_value)[:3]
                            entry["textured"] = bool(bc.is_linked)
                            if bc.is_linked:
                                avg = _socket_color(bc)
                                if avg is not None:
                                    entry["base_color"] = avg
                        tint = _inp("Sheen Tint")
                        if tint is not None and not tint.is_linked:
                            entry["sheen_tint"] = list(tint.default_value)[:3]
                        for key, names in (("metallic", ("Metallic",)), ("roughness", ("Roughness",)),
                                           ("transmission", ("Transmission Weight", "Transmission")),
                                           ("sheen", ("Sheen Weight", "Sheen")),
                                           ("coat", ("Coat Weight", "Clearcoat"))):
                            sock = _inp(*names)
                            if sock is not None and not getattr(sock, "is_linked", False):
                                try:
                                    entry[key] = float(sock.default_value)
                                except Exception:
                                    pass
                        break
                found[mat.name] = entry
            entry["weight"] += max(1, poly_count // max(1, len(obj.material_slots)))
    return list(found.values())


def identity_checklist(
    product_instructions: Iterable[str] | None = None,
    jev_assets: str | Iterable[str] | None = None,
    user_details: str = "",
    limit: int = 12,
) -> list[str]:
    """Merge user-listed details, product-reference notes and Jev identity assets."""
    items: list[str] = []

    def add_many(text):
        if not text:
            return
        if isinstance(text, str):
            for sep in ("，", ",", "、", ";", "；", "\n"):
                text = text.replace(sep, "|")
            pieces = text.split("|")
        else:
            pieces = list(text)
        for piece in pieces:
            p = str(piece).strip().strip("。.")
            if p and p not in items:
                items.append(p)

    add_many(user_details)
    for note in product_instructions or []:
        add_many(note)
    add_many(jev_assets)
    return items[:limit]


def appearance_lock_block(color_source: str, material_text: str, product_look: str) -> str:
    """Lock Blender paint/geometry while allowing reference material properties.

    ``PRODUCT_REF`` remains a legacy UI value: it never authorizes a body repaint.
    """
    lines = []
    look = (product_look or "").strip()
    if look:
        lines.append("产品外观（锁定，所有场景保持一致，只随环境受光变化）：" + look)
    if color_source == "BLENDER" and (material_text or "").strip():
        lines.append("Blender 当前材质提供车身车漆色相与基础材质基线：" + material_text.strip()
                     + "；三视图可在不冲突时补充其他部件材质属性，不得重涂车漆或改几何。")
    elif color_source == "PRODUCT_REF":
        lines.append("产品参考图仅用于各部件材质属性；车身车漆色相仍以 Blender 当前模型为准，不取参考图颜色、打光、阴影或背景。")
    if not lines:
        return ""
    return "【产品外观锁定】\n" + "\n".join(lines)


def identity_block(items: list[str]) -> str:
    if not items:
        return ""
    return "【必须清晰保留的产品细节】" + "、".join(items) + "。这些细节不得省略、模糊或替换。"
