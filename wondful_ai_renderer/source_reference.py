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
    "车漆按真实汽车清漆漆面渲染：基础色相保持 Blender 模型颜色，表面是一层光滑透明清漆。"
    "朝上的引擎盖、车顶和肩线映出来源照片的天空颜色与亮度，朝下的门板下半部和裙边映出地面颜色，"
    "侧面沿车身腰线有一条随曲面弯曲的地平线反射分界；太阳或主光在曲面最高点形成小而锐利的镜面高光，"
    "掠射角处反射增强（菲涅尔），背光面保留深色但仍带环境反射。反射随曲率连续变化，亮度在同一块漆面内自然渐变。"
    "图 2 光照参考里的反射只说明曲面走向和漆面的光泽程度，最终反射内容、颜色和光线方向以来源照片为准。"
    "不采用来源照片中旧车的漆色或高光形状。用户明确要求哑光或缎面时按用户要求处理；未有依据时不加金属闪粉或珠光。"
)

DETAIL_FIDELITY_POLICY = (
    "车身细节逐项保留并清晰可见：门缝和板件分缝是细而暗的连续线，门把手、充电口盖、后视镜、"
    "前后灯组的内部灯条与透镜层次、进气口格栅、轮毂每根轮辐、刹车卡钳、轮胎胎侧、车标与车牌文字"
    "都保持图 1 中的数量、位置、形状与比例，以照片级清晰度重新受光，不得简化、平滑、合并或删除。"
)

PROJECTED_IDENTITY_POLICY = (
    "图 1 是原始来源照片，目标产品已由 Blender 按同一相机渲染并放置在最终位置：保留该产品的位置、尺度、透视、"
    "朝向、轮廓、车轮位置与接地点，以及全部已建模部件；照片中的环境、地面、天空、光线方向、色温、曝光与镜头效果保持不变。"
    "把图 1 中的产品重新渲染成与照片同一光照下拍摄的真实车辆：替换掉 Blender 的预览光照与阴影，使用照片的主光方向、"
    "环境光、反射和接触阴影，轮胎压在地面上并有贴合的接触阴影。"
    "若图 1 中新产品轮廓之外还残留旧车的任何部分（车顶、车尾、车轮、阴影或反光），将其清除并按周围环境补全背景。"
    "图 2 是同一产品在同一相机下的独立光照参考（透明背景），用于核对几何、部件排列和漆面光泽，不决定环境或光线方向。"
    "图 3 是完整产品身份参考，三视图或身份图只确定各部件的材质属性，不能决定车身车漆色相或任何几何；不得用身份图或其裁切重设计投影中已建模的灯组、轮辐或其他可见部件。"
    "车漆色相保持当前 Blender 模型，除非用户明确指定具体车漆颜色。不臆造隐藏结构，不复制身份图的背景或灯光。"
    + DETAIL_FIDELITY_POLICY + PAINT_FINISH_POLICY
)
PROJECTED_POLISH_SYSTEM = (
    "你是产品结构约束下的参考图编辑提示词导演。" + PROJECTED_IDENTITY_POLICY
    + "保留完整用户需求，输出完整中文生图提示词正文。用正面、具体的光线与材质描述写作，"
    "说明来源照片的主光方向、天空与地面颜色以及它们在车漆上的反射位置。不要描述成粗模、环境白模或自动几何验收。"
)
PROJECTED_EDIT_INSTRUCTIONS = (
    "Use image_generation exactly once to edit input 1 and return one image. "
    "Input 1 is the original photograph with the target product already rendered into it by Blender from the same "
    "camera: keep that product's exact position, scale, perspective, silhouette, wheel positions and ground contact, "
    "and keep every modeled part (panel gaps, door handles, mirrors, lamp internals, grille, every wheel spoke, brake "
    "calipers, badges and plate text) with the same count, shape and placement. Keep the photograph's environment, "
    "light direction, colour temperature, exposure and lens effects. Re-render the product photographically under the "
    "photograph's light: replace Blender's preview shading with the scene's key light, sky and ground reflections and a "
    "tight contact shadow under each tyre. Remove any leftover part of the photograph's original vehicle outside the "
    "new silhouette and restore the background there. "
    "Input 2 is the same product alone on transparency (lighting reference for geometry and paint glossiness only, "
    "not for environment or light direction). Input 3 is product identity for component material properties only; "
    "it must not redesign modeled parts or change body-paint hue. Preserve only an explicit concrete user paint colour. "
    "Body paint is glossy automotive clearcoat over the model's base hue: upward surfaces reflect the photograph's "
    "sky, lower body sides reflect the ground, a curved horizon reflection line follows the body side, the sun gives "
    "small sharp specular highlights, reflections strengthen at grazing angles and vary continuously with curvature. "
    "Honour an explicit user matte/satin request; do not add unevidenced metallic flake or pearl. "
    "Do not copy the old vehicle's paint or highlight shapes. No mask or other tool."
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
    primary = [(source, "source_canvas", "来源照片编辑画布（目标产品已按同一相机放置在最终位置）" if projection
                else "原始目标图，编辑画布")]
    if projection:
        primary.append((projection, "product_projection", "同一相机下目标产品的独立光照参考（几何、部件排列与漆面光泽）"))
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
