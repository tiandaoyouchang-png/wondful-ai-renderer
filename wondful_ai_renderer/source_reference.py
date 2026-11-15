"""Verified inputs and explicit image roles for a single-source edit."""
import hashlib
import tempfile
from pathlib import Path

from .composition import image_dimensions


POLISH_SYSTEM = (
    "你是参考图编辑提示词导演。第一张是原始目标图，决定场景、构图、光线和成像；"
    "第二张是目标产品身份图，只辅助识别可见材质；当前 Blender 模型决定产品外形、细节与车漆色相。"
    "保留用户需求和原图可见的光线及摄影效果，不臆造未出现的细节。"
    "用户明确提供的产品外观或材质信息优先。只输出完整的中文生图提示词正文。"
)

EDIT_INSTRUCTIONS = (
    "Use the image_generation tool exactly once to edit the first input image. "
    "The first image is the original canvas and determines the scene, framing and lighting. "
    "The current Blender model determines target shape, visible topology and body-paint hue. The second image only identifies visible component materials. "
    "Replace the original product with the target product, respecting any explicit user appearance instructions. "
    "Keep the original environment and photographic appearance. "
    "Do not let the identity image redesign lamp internals, wheel spokes or any modeled feature, or repaint the body. "
    "Additional product_detail images, if listed, are spatial crops of that same identity, not new views. "
    "Do not invent hidden or unreadable parts or copy identity lighting/background. "
    "Return one image; use no other tool."
)

CAMERA_EDIT_INSTRUCTIONS = (
    "Use the image_generation tool exactly once to edit the first input image and return one PNG. "
    "The captured Blender camera canvas controls projection, silhouette, position, scale and composition. "
    "The current Blender model controls visible geometry, visible fine-feature topology, lamp internals, wheel structures and body-paint hue. "
    "Product identity references only control component material properties, subject to an explicit concrete user paint-colour request. "
    "Explicit user appearance customization takes priority only when it does not override modeled geometry; body paint requires a concrete user paint-colour request. "
    "Additional product_detail images, if listed, are original-pixel spatial crops, not inferred parts or new views. "
    "Do not infer geometry from identity pixels; do not invent hidden or unreadable material details. "
    "Relight for the requested environment; do not copy identity lighting/background. Use no other tool."
)

VISIBLE_IDENTITY_POLICY = (
    "当前 Blender 模型的投影是可见细部拓扑的唯一依据：保留已建模的内部布局、分支、开口、灯组和轮辐。"
    "身份图及其裁切只核对各部件可见材质与表面处理，不得推断、替换或重设计任何几何，不臆造隐藏面。"
    "车身车漆色相保持 Blender 当前模型；环境可改变受光和反射，只有用户明确指定具体车漆颜色时才可覆盖。"
    "用户明确的外观定制优先，但不得覆盖已建模几何；车漆覆盖必须是具体颜色请求。"
)

PAINT_FINISH_POLICY = (
    "仅针对涂漆/喷涂表面：Blender 模型的几何与基础色相不代表预览打光或哑光质感；"
    "应从身份图中推断真实车漆光泽与清漆层响应，而不复制身份图的银灰偏色或摄影棚高光形状；"
    "严格遵从用户明确指定的哑光或高光要求；不确定时不得凭空臆造金属闪粉或珠光颗粒。"
    "图 1 旧产品的漆面颜色、高光条纹、反射形状与表面涂装都不属于环境依据，随旧产品一并舍弃；环境证据只来自天空、地面、可见建筑或光源。"
    "按可见环境及身份漆面的反射柔化程度重建受控清漆响应；不要靠增加反射锐度或能量来制造真实感；无对应场景来源时不凭空加入棚拍硬边亮带、闭合色带或镜面镀铬。"
    "场景天空与地面的真实反射、高光滚降须随表面曲率与视线方向连续过渡，局部 RGB 亮度随受光自然变化，而非整体重涂色相。"
    "严禁将车身处理为单调统一的绿色色块平涂、烘焙平坦的预览着色、塑料玩具感或镜面镀铬过度饱和等投机效果；"
    "不要强行将所有车漆一律设为高光镜面。"
)

PROJECTED_IDENTITY_POLICY = (
    "图 1 是原始目标画布，控制环境、光照与摄影效果；图 2 是当前相机下真实产品的独立 RGBA 投影，"
    "替换原图中的旧产品，并控制可见几何、部件排列、位置、尺度、透视与轮廓，包括已建模的灯具内部和轮毂结构。"
    "图 3 是完整产品身份参考，三视图或身份图只确定各部件的材质属性，不能决定车身车漆色相或任何几何；"
    "不得用身份图或其裁切重设计投影中已建模的灯组、轮辐或其他可见部件。"
    "车漆色相保持当前 Blender 模型，除非用户明确指定具体车漆颜色；环境只能改变受光和反射，不能给车漆重上色。"
    "不臆造隐藏结构，不复制身份图的背景或灯光。"
    + PAINT_FINISH_POLICY
)
PROJECTED_POLISH_SYSTEM = (
    "你是产品结构约束下的参考图编辑提示词导演。" + PROJECTED_IDENTITY_POLICY
    + "保留完整用户需求，输出完整中文生图提示词正文。不要描述成粗模、环境白模或自动几何验收。"
)
PROJECTED_EDIT_INSTRUCTIONS = (
    "Use image_generation exactly once to edit input 1 and return one image. "
    "Input 1 is the original environment/lighting/photographic canvas. Input 2 is a fresh same-camera "
    "product-only RGBA projection of the actual model: replace the source's old product with this model; "
    "the projection controls visible geometry, part arrangement, "
    "position, scale, perspective and silhouette, including modeled lamp internals and wheel structures. "
    "Input 3 is complete product identity for component material properties only; identity and its spatial crops "
    "must not redesign any visible modeled parts or change body-paint hue. Preserve only an explicit concrete user paint-colour request. "
    "Do not invent hidden geometry or copy identity lighting/background. No mask or other tool. "
    "Discard the old source product's paint, highlights and reflection shapes as material or lighting evidence. "
    "Match reflection blur to visible identity finish and environment; do not invent hard-edged reflection bands or closed graphic swirls without a corresponding scene source. "
    "For painted surfaces: model geometry/base hue do not imply preview lighting or matte finish; "
    "infer evidenced paint gloss/clearcoat response from identity without copying silver hue or studio highlight shapes; "
    "honor explicit user matte/gloss; do not invent unevidenced metallic flakes or pearl. "
    "Curvature reflections and lighting rolloff vary continuously with environment; avoid flat solid fills, baked preview shading, toy plastic, or chrome over-saturation shortcuts without forcing all paint glossy."
)


def append_identity_crops(images, directory, limit=5):
    """Append native-resolution spatial tiles of a wide, standalone identity.

    Only the unused input slots are used; originals are never resized or replaced.
    Bounds are half-open RGB pixel coordinates, with the origin at the top left.
    """
    images = list(images)
    info = {"status": "skipped", "reason": "no_capacity_or_standalone_identity", "crops": []}
    count = min(3, max(0, int(limit) - len(images)))
    identity_index = next((i for i, item in enumerate(images) if item[1] == "product"), None)
    if not count or identity_index is None:
        return images, info
    path = images[identity_index][0]
    identity_number = identity_index + 1
    size = image_dimensions(path)
    if not size or size[0] < 1024 or size[1] < 256 or size[0] < 2 * size[1]:
        info["reason"] = "not_sufficiently_wide"
        return images, info
    from .reference_collage import read_rgb
    from .png_io import write_png
    before = fingerprint(path)
    pixels = read_rgb(path)
    if not matches(before, path):
        raise ValueError("产品身份图在裁切时发生变化，请重试。")
    height, width = pixels.shape[:2]
    if (width, height) != tuple(size):
        raise ValueError("产品身份图解码尺寸不一致，请重新选择图片。")
    # Full-height overlapping windows, evenly spaced across the raster.
    crop_width = max(height, (width * 6 + 5 * count - 1) // (5 * count)
                     if count > 1 else (width + 1) // 2)
    starts = ([int(round((width - crop_width) * i / (count - 1))) for i in range(count)]
              if count > 1 else [(width - crop_width) // 2])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="identity_pixels_", dir=str(directory)))
    info = {"status": "attached", "source": before, "source_size": [width, height],
            "source_image_number": identity_number, "bounds_convention": "RGB top-left, half-open xyxy", "crops": []}
    for i, left in enumerate(starts, 1):
        bounds = [left, 0, left + crop_width, height]
        crop_path = output / f"spatial_{i}.png"
        write_png(crop_path, pixels[:, left:left + crop_width].copy())
        label = f"图 {identity_number} 原始像素空间裁切 {bounds}（非独立视角；仅辅助材质核对，不改几何或车漆）"
        images.append((str(crop_path), "product_detail", label))
        info["crops"].append({"path": str(crop_path), "role": "product_detail", "bounds": bounds,
                              "size": [crop_width, height], "image_number": len(images),
                              "sha256": fingerprint(crop_path)["sha256"]})
    return images, info


def detail_prompt(info):
    if not info.get("crops"):
        return ""
    number = info["source_image_number"]
    lines = [f"补充图像职责（以上传顺序为准）：图 {number} 保留完整产品身份原图；以下仅为它的原始 RGB 像素空间裁切，"
             "没有放大、补画或新增视角，只辅助核对部件材质，不能据空间位置推断部件名称、几何或不可见细节。"]
    lines.extend(f"图 {crop['image_number']}：图 {number} 的像素区域 {crop['bounds']}，尺寸 {crop['size']}；仅辅助核对部件材质。"
                 for crop in info["crops"])
    return "\n".join(lines)


def fingerprint(path):
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    if not data:
        raise ValueError("参考图为空，请重新选择。")
    return {"path": str(path), "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def matches(snapshot, path):
    try:
        return fingerprint(path) == snapshot
    except (OSError, ValueError, TypeError):
        return False


def input_key(target, identity, target_note=""):
    """Bind AI-authored text to both images and the target's reference note."""
    images = ":".join(str(value) for item in (target, identity)
                      for value in (item["size"], item["sha256"]))
    return images + ":" + hashlib.sha256((target_note or "").strip().encode("utf-8")).hexdigest()


def copy_verified(path, directory, name):
    """Copy the exact bytes and reject a source changed while it was read."""
    before = fingerprint(path)
    original = Path(before["path"])
    data = original.read_bytes()
    if len(data) != before["size"] or hashlib.sha256(data).hexdigest() != before["sha256"]:
        raise ValueError("参考图在读取时发生变化，请重试。")
    dest = Path(directory) / (name + (original.suffix.lower() or ".png"))
    dest.write_bytes(data)
    if not matches(before, original):
        raise ValueError("参考图在复制时发生变化，请重试。")
    dimensions = image_dimensions(dest)
    if not dimensions or min(dimensions) <= 0:
        raise ValueError("参考图无法读取尺寸，请重新选择图片。")
    return str(dest), before, dimensions


def validate_inputs(source, identity, limit, projection=""):
    if int(limit) < (3 if projection else 2):
        raise ValueError("当前服务商无法完整接收原图、产品结构参考和身份图。" if projection else
                         "当前服务商无法同时接收原图和产品身份图。")
    images = []
    primary = [(source, "source_canvas", "原始目标图，编辑画布")]
    if projection:
        primary.append((projection, "product_projection", "当前相机真实产品独立 RGBA 投影（可见几何及部件排列权威）"))
    primary.append((identity, "product", "目标产品身份图：仅核对部件材质；不改模型几何或车漆色相"))
    for path, role, label in primary:
        p = str(Path(path).expanduser().resolve())
        if not Path(p).is_file() or not image_dimensions(p):
            raise ValueError(f"{label}不可用，请重新选择。")
        images.append((p, role, label))
    if len({p for p, _r, _l in images}) != len(images):
        raise ValueError("原图、产品结构参考与身份图需要分别作为独立输入。")
    return images


def edit_prompt(polished, identity_text="", target_note="", projection=False):
    if not (polished or "").strip():
        raise ValueError("AI 润色没有返回有效内容，已停止生图。")
    lines = ["图 1 是原始目标图及编辑画布：保留原有环境、构图、光线和成像。",
             "图 2 是目标产品身份参考：仅核对部件材质；当前 Blender 模型决定造型、可见细部与车漆色相。"
             "按原图环境自然受光，不复制身份图的背景或灯光。", VISIBLE_IDENTITY_POLICY]
    if projection:
        lines = [PROJECTED_IDENTITY_POLICY]
    if (identity_text or "").strip():
        lines.append("用户指定的产品信息：\n" + identity_text.strip())
    note = (target_note or "").strip()
    if note and note not in (polished or "") and note not in (identity_text or ""):
        lines.append("目标图参考重点：\n" + note)
    lines.append("完整润色要求：\n" + polished.strip())
    return "\n".join(lines)
