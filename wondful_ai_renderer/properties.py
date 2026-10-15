# RNA Property declarations must evaluate at class creation (Blender 4.3+).


import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import AddonPreferences, PropertyGroup


MAX_REFERENCES_PER_KIND = 8
# 3.1.8: the product-shape reference is a single image (ideally a three-view sheet).
MAX_PRODUCT_REFERENCES = 1


def reference_limit(kind):
    return MAX_PRODUCT_REFERENCES if kind == "PRODUCT" else MAX_REFERENCES_PER_KIND


_PROMPT_SYNCING = False



def _split_prompt_for_editor(text: str, width: int = 22) -> list[tuple[str, bool]]:
    """Split one logical prompt into visual editor rows without changing content.

    Each row stores whether the original prompt had a real newline after that row.
    Soft wrapping never inserts a newline into the canonical prompt. This keeps the
    4.3-5.1 scrollable fallback visually wrapped while preserving exact text.
    """
    value = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if value == "":
        return [("", False)]

    width = max(8, int(width))
    rows: list[tuple[str, bool]] = []
    logical_lines = value.split("\n")
    last_line_index = len(logical_lines) - 1
    for line_index, logical_line in enumerate(logical_lines):
        hard_break = line_index < last_line_index
        if logical_line == "":
            rows.append(("", hard_break))
            continue

        chunks = [logical_line[i:i + width] for i in range(0, len(logical_line), width)] or [""]
        for chunk_index, chunk in enumerate(chunks):
            rows.append((chunk, bool(hard_break and chunk_index == len(chunks) - 1)))
    return rows or [("", False)]


def _rebuild_prompt_from_rows(rows) -> str:
    """Rebuild the canonical prompt exactly from visual fallback rows."""
    parts: list[str] = []
    for item in rows:
        parts.append(str(getattr(item, "text", "")))
        if bool(getattr(item, "hard_break_after", False)):
            parts.append("\n")
    return "".join(parts)


def sync_prompt_editor(props, text: str | None = None) -> None:
    """Populate the scrollable fallback editor from the single canonical prompt."""
    global _PROMPT_SYNCING
    if _PROMPT_SYNCING:
        return
    _PROMPT_SYNCING = True
    try:
        value = props.prompt if text is None else str(text or "")
        props.prompt_lines.clear()
        for line, hard_break_after in _split_prompt_for_editor(value):
            item = props.prompt_lines.add()
            item.text = line
            item.hard_break_after = hard_break_after
        props.prompt_line_index = min(
            max(0, int(getattr(props, "prompt_line_index", 0))),
            max(0, len(props.prompt_lines) - 1),
        )
    finally:
        _PROMPT_SYNCING = False


def _prompt_value_update(self, context):
    # Canonical edits (including AI writeback) update the open Text Editor directly.
    # Legacy prompt_lines are migration-only; never rebuild them on each keystroke.
    from .prompt_editor import sync_prompt_editors
    sync_prompt_editors(self)


def _prompt_line_update(self, context):
    global _PROMPT_SYNCING
    if _PROMPT_SYNCING or context is None or getattr(context, "scene", None) is None:
        return
    props = getattr(context.scene, "wondful_ai", None)
    if props is None:
        return
    _PROMPT_SYNCING = True
    try:
        props.prompt = _rebuild_prompt_from_rows(props.prompt_lines)
    finally:
        _PROMPT_SYNCING = False



def _reference_instruction_update(self, context):
    """Changing a style-reference note invalidates the style-derived prompt.

    Product/person notes are consumed directly by both prompt generation and final
    render mapping, so they do not need to block rendering. Style notes affect the
    dedicated style-analysis pass and therefore require prompt regeneration.
    """
    try:
        if context is None or getattr(context, "scene", None) is None:
            return
        props = getattr(context.scene, "wondful_ai", None)
        if props is None:
            return
        self_ptr = self.as_pointer()
        for item in props.style_images:
            if item.as_pointer() == self_ptr:
                props.style_reference_revision = int(getattr(props, "style_reference_revision", 0)) + 1
                props.style_prompt_dirty = True
                break
    except Exception:
        pass

def _refresh_compare_update(self, context):
    try:
        if context and getattr(context, "scene", None) and getattr(context.scene, "wondful_ai", None):
            from .comparison_view import refresh_compare_image
            refresh_compare_image(context)
    except Exception:
        # UI changes must never make Blender unusable. The explicit Refresh button
        # in Wondful Compare reports errors if the source images are invalid.
        pass


class WONDFUL_AddonPreferences(AddonPreferences):
    bl_idname = __package__

    codex_cli_path: StringProperty(
        name="Codex CLI 路径",
        description="留空时自动检测 PATH、Homebrew、~/.local/bin 等常见位置",
        subtype="FILE_PATH",
        default="",
    )
    codex_model: StringProperty(
        name="Codex 推理模型",
        description="留空使用当前 Codex 默认模型；仅在你明确知道模型 ID 时填写",
        default="",
    )
    antigravity_cli_path: StringProperty(
        name="Antigravity CLI 路径",
        description="留空时自动检测 PATH、Homebrew、~/.local/bin 等常见位置",
        subtype="FILE_PATH",
        default="",
    )
    antigravity_model: StringProperty(
        name="Antigravity 推理模型",
        description="只选择代理的推理模型，不是生图模型；生图由 AGY 内置工具决定",
        default="",
    )
    output_directory: StringProperty(
        name="渲染输出目录",
        description="AI 最终渲染图导出目录；留空时优先使用 .blend 同目录下的 Wondful_Renders，未保存工程则使用用户 Pictures/Wondful_AI_Renderer",
        subtype="DIR_PATH",
        default="",
    )
    long_edge: IntProperty(
        name="目标长边",
        description="用于告诉 Codex 期望尺寸并做最终比例校正；ImageGen 实际支持的尺寸由 Codex 当前工具决定",
        default=2048,
        min=1024,
        max=4096,
        step=256,
    )
    codex_imagegen_enabled: BoolProperty(
        name="启用 Codex ImageGen（实验）",
        description="通过 ChatGPT OAuth 的 Codex 会话调用内置 image generation；当前 Codex 版本/账号可能不暴露该工具",
        default=True,
    )
    codex_direct_edit: BoolProperty(
        name="Codex 图像编辑（直连）",
        description="在 Blender 白模画布上直接调用 Codex 图像编辑（复用 Codex 的 ChatGPT 登录，无需 API Key），"
                    "产品形状、位置和尺度更稳；失败时自动改用 Codex 代理生成",
        default=True,
    )
    codex_direct_simple: BoolProperty(
        name="简洁输入（只发白模底图）",
        description="直连编辑时先渲一张带地面和灯光的白模底图，只发这一张图和一段短提示词；"
                    "不附结构图、三视图和环境照，车的位置和大小最稳",
        default=True,
    )
    codex_polish_timeout: IntProperty(
        name="Codex 润色超时（秒）",
        description="Codex AI 润色允许的最长总运行时间；复杂多图任务可能需要数分钟。0 表示不设置插件侧硬超时",
        default=900,
        min=0,
        max=7200,
        step=60,
    )
    codex_audit_timeout: IntProperty(
        name="Codex 构图验收超时（秒）",
        description="Codex 视觉构图验收允许的最长总运行时间。0 表示不设置插件侧硬超时",
        default=600,
        min=0,
        max=7200,
        step=60,
    )
    codex_render_timeout: IntProperty(
        name="Codex ImageGen 超时（秒）",
        description="单次 Codex ImageGen 允许的最长总运行时间；旧版固定 420 秒会误杀正常长任务。0 表示不设置插件侧硬超时",
        default=1800,
        min=0,
        max=14400,
        step=60,
    )
    antigravity_imagegen_enabled: BoolProperty(
        name="启用 Antigravity 生图",
        description="使用 Antigravity CLI 内置 generate_image 工具生成或编辑图片；实际可用性取决于当前账号、区域和 Antigravity 版本",
        default=True,
    )
    antigravity_polish_timeout: IntProperty(
        name="Antigravity 润色超时（秒）",
        description="Antigravity AI 润色允许的最长总运行时间；0 表示不设置插件侧硬超时",
        default=900,
        min=0,
        max=7200,
        step=60,
    )
    antigravity_audit_timeout: IntProperty(
        name="Antigravity 构图验收超时（秒）",
        description="Antigravity 构图验收允许的最长总运行时间；0 表示不设置插件侧硬超时",
        default=600,
        min=0,
        max=7200,
        step=60,
    )
    antigravity_render_timeout: IntProperty(
        name="Antigravity 生图超时（秒）",
        description="单次 Antigravity generate_image 允许的最长总运行时间；0 表示不设置插件侧硬超时",
        default=1800,
        min=0,
        max=14400,
        step=60,
    )
    strict_composition_lock: BoolProperty(
        name="严格构图锁定",
        description="以 Blender Camera Reference 为编辑底图，优先保持主体位置、尺度、视角和透视；生成后仍需检查对齐",
        default=True,
    )
    structure_guides_enabled: BoolProperty(
        name="结构控制图",
        description="从 Blender 相机生成 Mask、深度、法线和轮廓；原生通道不可用时降级为投影近似",
        default=True,
    )
    masked_alignment_repair: BoolProperty(
        name="局部结构修复",
        description="局部偏移且位置估计可信时，修复范围覆盖当前位置与白模目标；比例或视角错误时回到原始白模重生成",
        default=True,
    )
    identity_preserve_enabled: BoolProperty(
        name="身份资产空间保护",
        description="根据 Jev + Part Index 为 Logo、字标、车牌文字和可读 UI 生成 Preserve Mask，优先阻止生图模型改写这些区域",
        default=True,
    )
    identity_preserve_padding: IntProperty(
        name="身份保护扩边（像素）",
        description="在身份资产可见像素周围增加少量保护边界，避免 Logo 边缘被生图模型侵蚀",
        default=2,
        min=0,
        max=12,
    )
    identity_hard_restore: BoolProperty(
        name="硬恢复身份像素（实验）",
        description="最终导出前把 Camera Base 的身份区域像素精确回贴。仅当 Camera Base 已含正确 Logo/字标外观时启用；白模场景建议关闭",
        default=False,
    )
    auto_alignment_retry: BoolProperty(
        name="严格模式自动构图校验 / 重试",
        description="仅在主界面选择“严格对齐”时生效：生成后由当前 AI Provider 对比 Blender Camera Reference 与结果；未达阈值时自动带纠偏要求再生成一次",
        default=True,
    )
    alignment_threshold: IntProperty(
        name="构图通过分数",
        description="0-100。低于该分数时触发一次自动纠偏重试",
        default=88,
        min=60,
        max=100,
    )
    alignment_max_attempts: IntProperty(
        name="最多生成次数",
        description="包含第一次生成。建议 2，避免无意中过多消耗当前 Provider 的生图额度",
        default=2,
        min=1,
        max=3,
    )

    def draw(self, context):
        layout = self.layout
        info = layout.box()
        info.label(text="无需 OpenAI / Google / DeepSeek / Seedream API Key。", icon="INFO")
        info.label(text="Codex 复用 ChatGPT OAuth；Antigravity 复用 Google Keychain / OAuth。插件不保存、不记录 Token。")
        info.label(text="图像编辑直连会读取 Codex 的 auth.json 登录信息，令牌过期时按 Codex 方式刷新并写回。")

        codex = layout.box()
        codex.label(text="Codex", icon="CONSOLE")
        codex.prop(self, "codex_cli_path")
        codex.prop(self, "codex_model")

        antigravity = layout.box()
        antigravity.label(text="Antigravity", icon="WORLD")
        antigravity.prop(self, "antigravity_cli_path")
        antigravity.prop(self, "antigravity_model")

        output = layout.box()
        output.label(text="输出", icon="FILE_FOLDER")
        output.prop(self, "output_directory")

        col = layout.column(align=True)
        col.prop(self, "long_edge")
        col.prop(self, "codex_imagegen_enabled")
        col.prop(self, "codex_direct_edit")
        col.prop(self, "codex_direct_simple")
        col.prop(self, "antigravity_imagegen_enabled")
        col.prop(self, "codex_polish_timeout")
        col.prop(self, "codex_audit_timeout")
        col.prop(self, "codex_render_timeout")
        col.prop(self, "antigravity_polish_timeout")
        col.prop(self, "antigravity_audit_timeout")
        col.prop(self, "antigravity_render_timeout")
        col.separator()
        col.prop(self, "strict_composition_lock")
        col.prop(self, "structure_guides_enabled")
        col.prop(self, "masked_alignment_repair")
        col.prop(self, "identity_preserve_enabled")
        if self.identity_preserve_enabled:
            col.prop(self, "identity_preserve_padding")
            col.prop(self, "identity_hard_restore")
        col.prop(self, "auto_alignment_retry")
        if self.auto_alignment_retry:
            col.prop(self, "alignment_threshold")
            col.prop(self, "alignment_max_attempts")

        box = layout.box()
        box.label(text="运行依赖")
        box.label(text="Codex：官方 Codex CLI + ChatGPT OAuth；Antigravity：官方 Antigravity CLI + Google OAuth。")
        box.label(text="统一 Provider：选择 Codex 时润色/验收/生图都走 Codex；选择 Antigravity 时全部走 Antigravity。")
        box.label(text=f"参考图：产品造型 1 张（建议三视图）；人物 / 环境风格每类最多 {MAX_REFERENCES_PER_KIND} 张。")


class WONDFUL_PromptLine(PropertyGroup):
    # Blender 4.3-5.1 compatibility row. Rows are only a visual viewport into
    # one canonical prompt; hard_break_after preserves real user newlines.
    text: StringProperty(name="提示词", default="", update=_prompt_line_update)
    hard_break_after: BoolProperty(default=False, options={"HIDDEN"})


class WONDFUL_ReferenceItem(PropertyGroup):
    image: PointerProperty(name="参考图", type=bpy.types.Image)
    source_path: StringProperty(name="原始路径", subtype="FILE_PATH", default="")
    instruction: StringProperty(
        name="参考什么",
        description="告诉 AI 这张图具体参考什么，产品图仅填写造型细节（如前脸、轮毂）；环境／风格图可填写灯光、色调、材质表现",
        default="",
        update=_reference_instruction_update,
    )


class WONDFUL_ModelItem(PropertyGroup):
    name: StringProperty(name="模型 ID", default="")
    label: StringProperty(default="")
    description: StringProperty(default="")
    is_default: BoolProperty(default=False)


class WONDFUL_AISettings(PropertyGroup):
    codex_models: CollectionProperty(type=WONDFUL_ModelItem)
    antigravity_models: CollectionProperty(type=WONDFUL_ModelItem)
    codex_models_message: StringProperty(default="点击刷新，从当前 CLI 读取模型列表", options={"SKIP_SAVE"})
    antigravity_models_message: StringProperty(default="点击刷新，从当前 CLI 读取模型列表", options={"SKIP_SAVE"})
    # Never store OAuth codes/tokens here. The code field lives on WindowManager
    # with SKIP_SAVE, while URLs and process state stay in memory only.
    auth_stage: StringProperty(default="", options={"SKIP_SAVE"})
    auth_detail: StringProperty(default="", options={"SKIP_SAVE"})
    auth_has_url: BoolProperty(default=False, options={"SKIP_SAVE"})
    agy_image_capability: StringProperty(default="UNKNOWN", options={"SKIP_SAVE"})
    last_reference_bundle: StringProperty(default="", options={"SKIP_SAVE"})
    task_phase: StringProperty(default="", options={"SKIP_SAVE"})
    prompt_target_key: StringProperty(default="", options={"HIDDEN"})
    style_summary_cache_key: StringProperty(default="", options={"HIDDEN"})
    style_summary_cache: StringProperty(
        name="环境分析",
        description="AI 从当前环境／风格参考图读出的主光方向、色温、软硬与氛围。可直接修改，下次润色会用修改后的内容",
        default="",
    )
    ui_show_style_summary: BoolProperty(name="环境分析结果", default=False)

    # 3.1.7 product appearance lock
    color_source: EnumProperty(
        name="配色来源",
        description="产品固有色与材质从哪里来；任何来源都不会采用产品照片的打光",
        items=[
            ("BLENDER", "Blender 材质", "读取产品集合的材质颜色、金属度、粗糙度，写入外观锁定"),
            ("PRODUCT_REF", "产品参考图", "允许从产品参考图提取配色与表面材质（仍不取其打光）"),
            ("TEXT", "仅文字", "旧行为：配色与材质只来自提示词文字"),
        ],
        default="BLENDER",
    )
    product_look_prompt: StringProperty(
        name="产品外观",
        description="锁定的产品质感描述，例如“厚清漆、长而柔的渐变高光”。换环境参考时不会被重写，所有场景一致",
        default="",
    )
    identity_details: StringProperty(
        name="必须保留的细节",
        description="用逗号分隔，例如“车标，前车牌，日行灯”。会写进生图指令，并在验收时逐项检查",
        default="",
    )
    # 3.1.8: auto-fill of the two fields above from the product reference.
    product_autofill_key: StringProperty(default="", options={"HIDDEN"})
    product_look_auto: StringProperty(default="", options={"HIDDEN"})
    identity_details_auto: StringProperty(default="", options={"HIDDEN"})
    product_autofill_message: StringProperty(default="", options={"SKIP_SAVE"})
    # 3.1.9 one-button flow
    clay_ground: EnumProperty(
        name="地面",
        description="白模底图的临时地面形状。场景里有你自己的地面/地形时始终用你的",
        items=[
            ("AUTO", "自动", "平地；车身倾斜时自动铺成同坡度的斜坡"),
            ("CREST", "山头", "车停在山头最高处：四周地面向下倾斜，配合低机位仰拍效果最好"),
            ("CUSTOM", "自定义", "四个角（左前/右前/左后/右后）分别设地面高低，可加坡顶和凹凸；产品像刚体一样稳稳落在地面上"),
        ],
        default="AUTO",
    )
    # 1.1.0 custom terrain (高级 › 地面 › 自定义). Four corners of the product's own footprint:
    # signed slope (°) of the ground toward that corner, + rises, - falls.  Plus 坡顶 / 凹凸 strengths.
    clay_terrain_fl: FloatProperty(name="左前", description="产品左前角方向的地面坡度：正值=该角地面抬高，负值=降低（°）",
                                   default=15.0, min=-35.0, max=35.0, step=100, precision=0)
    clay_terrain_fr: FloatProperty(name="右前", description="产品右前角方向的地面坡度：正值=该角地面抬高，负值=降低（°）",
                                   default=15.0, min=-35.0, max=35.0, step=100, precision=0)
    clay_terrain_bl: FloatProperty(name="左后", description="产品左后角方向的地面坡度：正值=该角地面抬高，负值=降低（°）",
                                   default=-15.0, min=-35.0, max=35.0, step=100, precision=0)
    clay_terrain_br: FloatProperty(name="右后", description="产品右后角方向的地面坡度：正值=该角地面抬高，负值=降低（°）",
                                   default=-15.0, min=-35.0, max=35.0, step=100, precision=0)
    clay_terrain_crest: FloatProperty(
        name="坡顶", description="产品停在坡顶：四周地面从产品脚下向外掉落（0 = 无，1 = 约30°）。可与四角坡度叠加",
        default=0.0, min=0.0, max=1.0, subtype="FACTOR",
    )
    clay_terrain_bump: FloatProperty(
        name="凹凸", description="地面高低起伏、坑洼颠簸的强度（0 = 平整，1 = 强烈）。起伏平缓、每次相同",
        default=0.0, min=0.0, max=1.0, subtype="FACTOR",
    )
    clay_terrain_follow: BoolProperty(
        name="姿态跟随地形",
        description="白模渲染时产品像刚体一样落在地面上：随坡俯仰、侧倾，扭曲或颠簸的地面上三点着地、一角可能悬空离地；渲染完立即恢复你的原始变换。关闭则产品保持水平，只抬高到不穿地",
        default=True,
    )
    clay_terrain_front_axis: EnumProperty(
        name="产品正面",
        description="产品的“前”是哪个方向（按产品根物体的本地轴）。自动：-Y 为正面；识别为车辆且 -Y 不在车长方向时取车长方向",
        items=[
            ("AUTO", "自动", "-Y 为正面（Blender / glTF 惯例）；车辆取车长方向并朝向相机一侧"),
            ("NEG_Y", "-Y", "本地 -Y 是正面"),
            ("POS_Y", "+Y", "本地 +Y 是正面"),
            ("NEG_X", "-X", "本地 -X 是正面"),
            ("POS_X", "+X", "本地 +X 是正面"),
        ],
        default="AUTO",
    )
    clay_crest_angle: FloatProperty(
        name="坡顶下坡角度", description="山头四周向下倾斜的角度", default=18.0, min=5.0, max=35.0,
        subtype="NONE", unit="NONE",
    )
    scene_brief: StringProperty(
        name="画面需求",
        description="一句话说想要的场景，例如“雪地清晨，冷色调”。可留空，只按环境参考图来",
        default="",
    )
    scene_brief_used: StringProperty(default="", options={"HIDDEN"})
    pending_generate: BoolProperty(default=False, options={"SKIP_SAVE"})
    two_step: BoolProperty(
        name="分两步",
        description="打开后「生成」只写提示词，确认后再点「渲染」",
        default=False,
    )
    prompt_removed_notice: StringProperty(default="", options={"SKIP_SAVE"})
    ui_show_appearance_lock: BoolProperty(name="产品外观锁定", default=True)

    analysis_provider: EnumProperty(
        name="AI Provider",
        description="统一控制 AI 润色、构图验收与最终生图；保留旧属性名以兼容已有 .blend",
        items=[
            ("CODEX", "Codex", "使用 ChatGPT OAuth 的 Codex 完成润色、验收和最终生图"),
            ("ANTIGRAVITY", "Antigravity", "使用 Google OAuth 的 Antigravity CLI 完成润色、验收和 generate_image 生图"),
        ],
        default="CODEX",
    )
    prompt: StringProperty(name="提示词", description="唯一提示词内容：AI润色后直接覆盖写回，并可继续手动修改", default="", update=_prompt_value_update)
    # 2.18: keep a stable creative brief separate from the visible AI-polished prompt.
    # This prevents a previous style description from being recursively fed back when
    # the user replaces the style reference. The UI still exposes only one prompt box.
    prompt_notice: StringProperty(default="", options={"SKIP_SAVE"})
    prompt_source: StringProperty(name="原始创意需求", default="", options={"HIDDEN"})
    last_ai_prompt: StringProperty(name="上次AI提示词", default="", options={"HIDDEN"})
    prompt_reference_policy_version: IntProperty(default=0, min=0, options={"HIDDEN"})
    prompt_style_fingerprint: StringProperty(name="已同步风格指纹", default="", options={"HIDDEN"})
    prompt_expanded: BoolProperty(name="展开提示词", description="展开自动换行全文或收起为 6 行；完整编辑按钮支持多行输入", default=False)
    render_mode: EnumProperty(
        name="渲染模式",
        description="控制单次渲染的远程调用数量。Codex 每次生成 4 张，Antigravity 每次生成 2 张；严格对齐才执行远程构图验收和自动修复",
        items=[
            ("FAST", "快速预览", "Camera Base + Product Mask/Silhouette，最多 1280px 长边；Codex 生成 4 张，Antigravity 生成 2 张；用于快速看方向"),
            ("STANDARD", "标准", "自动生成 Structure Packet（Mask/Depth/Normal/Silhouette），使用目标分辨率；Codex 生成 4 张，Antigravity 生成 2 张；推荐默认"),
            ("STRICT", "严格对齐", "使用完整 Structure Packet（含 Part ID），生成后远程验收；未达阈值时按设置自动修复重试"),
        ],
        default="STANDARD",
    )
    output_directory: StringProperty(
        name="本工程输出目录",
        description="当前 .blend 工程的 AI 渲染输出位置；留空时使用插件全局输出目录或默认 Wondful_Renders",
        subtype="DIR_PATH",
        default="",
    )
    prompt_lines: CollectionProperty(type=WONDFUL_PromptLine)
    prompt_line_index: IntProperty(default=0, min=0)

    # 2.14: changing style references invalidates any prompt that was generated
    # from the previous style set. Rendering stays blocked until prompt polish
    # succeeds against the new references. The revision guard also prevents a
    # polish started on one style set from accidentally validating a newer set.
    style_reference_revision: IntProperty(default=0, min=0)
    prompt_style_revision: IntProperty(default=0, min=0)
    style_prompt_dirty: BoolProperty(
        name="风格提示词待刷新",
        description="风格参考图已变化；必须先重新生成提示词，再允许最终生图",
        default=False,
    )
    style_sync_initialized: BoolProperty(
        name="风格提示词已同步",
        description="当前提示词已经由 2.14+ 流程基于当前风格参考图重新生成过",
        default=False,
    )

    product_images: CollectionProperty(type=WONDFUL_ReferenceItem)
    product_image_index: IntProperty(default=0, min=0)
    person_images: CollectionProperty(type=WONDFUL_ReferenceItem)
    person_image_index: IntProperty(default=0, min=0)
    style_images: CollectionProperty(type=WONDFUL_ReferenceItem)
    style_image_index: IntProperty(default=0, min=0)

    # 2.19: one Collection is the explicit product structure source.
    # The Scene Camera still controls the whole-frame composition; every visible
    # Mesh inside this Collection (including child Collections) is treated as product.
    product_collection: PointerProperty(
        name="产品集合",
        description="Structure Lock 的产品来源。集合及其子集合内所有可见 Mesh 都视为产品，并按当前 Scene Camera 精确投影生成 Guide / Mask",
        type=bpy.types.Collection,
    )

    viewport_image: PointerProperty(name="Viewport 参考", type=bpy.types.Image)
    result_image: PointerProperty(name="AI 结果", type=bpy.types.Image)
    structure_mask_image: PointerProperty(name="当前结果的白模轮廓", type=bpy.types.Image)

    codex_account_state: EnumProperty(
        name="Codex账号",
        items=[
            ("UNKNOWN", "未检测", "尚未检测 Codex"),
            ("NOT_INSTALLED", "未安装", "未找到 Codex CLI"),
            ("LOGGED_OUT", "未登录", "Codex 未登录 ChatGPT"),
            ("LOGGED_IN", "已登录", "Codex 已通过 ChatGPT OAuth 登录"),
            ("ERROR", "错误", "Codex 状态检测失败"),
        ],
        default="UNKNOWN",
    )
    codex_account_message: StringProperty(default="")
    codex_cli_version: StringProperty(default="")
    codex_cli_resolved: StringProperty(default="")

    antigravity_account_state: EnumProperty(
        name="Antigravity账号",
        items=[
            ("UNKNOWN", "未检测", "尚未检测 Antigravity CLI"),
            ("NOT_INSTALLED", "未安装", "未找到 Antigravity CLI"),
            ("LOGGED_OUT", "未登录", "Antigravity 未通过 Google 账号登录"),
            ("LOGGED_IN", "已登录", "Antigravity 已通过 Google 账号登录"),
            ("ERROR", "错误", "Antigravity 状态检测失败"),
        ],
        default="UNKNOWN",
    )
    antigravity_account_message: StringProperty(default="")
    antigravity_cli_version: StringProperty(default="")
    antigravity_cli_resolved: StringProperty(default="")

    status: EnumProperty(
        name="状态",
        items=[
            ("IDLE", "Ready", "等待操作"),
            ("AUTHENTICATING", "Authenticating", "正在进行账号认证"),
            ("REFRESHING_MODELS", "Refreshing Models", "正在从当前 CLI 读取推理模型列表"),
            ("CLASSIFYING", "Classifying References", "正在通过 Jev 整理参考图语义角色"),
            ("POLISHING", "Polishing Prompt", "正在分析 Blender 构图并润色提示词"),
            ("RENDERING", "Rendering", "正在通过当前 AI Provider 生成最终图片"),
            ("SUCCESS", "Success", "生成成功"),
            ("ERROR", "Error", "发生错误"),
        ],
        default="IDLE",
    )
    progress: FloatProperty(name="进度", default=0.0, min=0.0, max=1.0, subtype="PERCENTAGE")
    eta_seconds: IntProperty(name="预计剩余", default=0, min=0)
    last_error: StringProperty(name="错误", default="")
    last_session_id: StringProperty(name="Session", default="")
    last_viewport_path: StringProperty(default="")
    last_result_path: StringProperty(default="")
    last_export_path: StringProperty(name="导出结果", subtype="FILE_PATH", default="")
    variant_count: IntProperty(
        name="生成张数",
        description="本次 AI 渲染实际导出的独立结果数量",
        default=1,
        min=1,
        max=4,
        options={"SKIP_SAVE"},
    )
    last_export_paths: StringProperty(
        name="导出结果列表",
        description="本次 AI 渲染导出的全部结果路径（JSON）",
        default="",
        options={"SKIP_SAVE"},
    )
    active_conversation_id: StringProperty(
        name="当前会话 ID",
        description="本次由当前 AI Provider 分配的远程会话 ID；可用于终端 agy -c 或客户端追溯",
        default="",
        options={"SKIP_SAVE"},
    )
    typesafe_auto_classify: BoolProperty(
        name="TypeSafe 智能识别分类",
        description="使用 TypeSafe 语义原语自动推断参考图角色（产品造型、环境风格或人物）",
        default=True,
    )
    typesafe_aesthetic_theme: EnumProperty(
        name="美学风格基调",
        description="TypeSafe 参数化美学主题",
        items=[
            ("AUTO", "智能自动推荐", "由 TypeSafe 根据提示词与参考图自动分析判断"),
            ("COMMERCIAL_STUDIO", "商业高级摄影棚拍", "大面积柔光箱，通透漫射与精密高光"),
            ("CINEMATIC_DRAMATIC", "电影级戏剧光影", "强反差与侧逆光，深邃阴影与环境氛围"),
            ("MINIMALIST_INDUSTRIAL", "极简现代工业", "自然漫射天光，克制中性色调与精密材质"),
            ("CYBER_TECH", "赛博未来科技", "暗场发光，冷暖对撞与未来主义科技质感"),
            ("NATURAL_LIFESTYLE", "真实户外与生活", "自然日光或黄金时刻，丰富环境反射"),
        ],
        default="AUTO",
    )
    typesafe_contrast_level: IntProperty(
        name="光影反差等级",
        description="1=全柔光漫射，3=标准商业光比，5=极端戏剧性强反差",
        default=3,
        min=1,
        max=5,
    )
    typesafe_material_gloss: IntProperty(
        name="材质高光等级",
        description="1=完全哑光，3=半光泽微光，5=光学镜面/电镀铬",
        default=3,
        min=1,
        max=5,
    )
    # Kept for .blend compatibility. 3.1.3 no longer fabricates a visual score
    # from file size; real visual ranking comes from the provider alignment audit.
    best_candidate_score: FloatProperty(
        name="旧版候选评分（停用）",
        description="3.1.3 已停用；文件大小不能代表视觉质量",
        default=0.0,
        options={"SKIP_SAVE"},
    )
    jev_status: StringProperty(
        name="Jev 状态",
        default="",
        options={"SKIP_SAVE"},
    )
    jev_status_message: StringProperty(
        name="Jev 详情",
        default="",
        options={"SKIP_SAVE"},
    )
    jev_semantic_summary: StringProperty(
        name="Jev 语义摘要",
        default="",
        options={"SKIP_SAVE"},
    )
    jev_identity_assets: StringProperty(
        name="身份关键资产",
        default="",
        options={"SKIP_SAVE"},
    )
    last_capture_method: StringProperty(default="")
    last_canvas_warning: StringProperty(default="")
    last_engine_note: StringProperty(
        name="生图通道",
        description="上次渲染实际使用的生图通道（直连图像编辑 / Codex 代理生成）",
        default="",
    )
    layered_output: BoolProperty(
        name="分层输出",
        description="生成后在结果图旁另存分层文件：主体（透明 PNG）、投影、可选干净背景，以及 PSD",
        default=False,
    )
    layered_mode: EnumProperty(
        name="分层方式",
        items=[
            ("SHADOW", "主体 + 投影", "不额外调用模型：按 Blender 产品遮罩抠出主体，从成片估算地面投影，几秒完成"),
            ("FULL", "完整（含干净背景）", "额外调用一次图像编辑生成去掉产品的干净背景（每张约多 30–50 秒），投影更准"),
        ],
        default="SHADOW",
    )
    last_layers_note: StringProperty(
        name="分层输出状态",
        default="",
        options={"SKIP_SAVE"},
    )
    # 1.2.0 环境体块: context blocks rendered grey-blue in the clay base, turned into buildings
    massing_enabled: BoolProperty(
        name="环境体块",
        description="白模底图里把环境体块集合（方块、圆柱、坡道等）临时显示为灰蓝色体块，"
                    "AI 按场景风格把它们画成真实建筑 / 结构，产品保持锁定",
        default=True,
    )
    massing_collection: PointerProperty(
        name="体块集合",
        description="环境体块所在的集合；留空 = 自动识别名称含 环境 / 体块 / Context / Massing 的集合",
        type=bpy.types.Collection,
        poll=lambda self, coll: coll != getattr(self, "product_collection", None),
    )
    last_massing_note: StringProperty(
        name="环境体块状态",
        default="",
        options={"SKIP_SAVE"},
    )
    # 1.2.0 草图模式: no real 3D model — a rough proxy carved from three-view images
    sketch_mode: BoolProperty(
        name="草图模式",
        description="没有真实 3D 模型时用：从正视 / 侧视 / 俯视图生成粗略草模代替产品，摆好位置和相机后照常生成；"
                    "外形细节、材质、颜色和 Logo 由 AI 按三视图补全（概念图，轮廓不是严格锁定）",
        default=False,
    )
    sketch_input: EnumProperty(
        name="三视图输入",
        items=[
            ("SEPARATE", "分开的三张图", "正视、侧视、俯视（可选）各一张，白底或透明底的 JPG / PNG"),
            ("SHEET", "一张三视图拼图", "三个视图排在一张图里、彼此留白，自动拆分（视图之间没有空隙或有尺寸标注时请改用分开的图）"),
        ],
        default="SEPARATE",
    )
    sketch_front_path: StringProperty(name="正视图", description="从产品正面看（白底或透明底）", subtype="FILE_PATH", default="")
    sketch_side_path: StringProperty(name="侧视图", description="从产品侧面看（白底或透明底）", subtype="FILE_PATH", default="")
    sketch_top_path: StringProperty(name="俯视图", description="从正上方看（可选，白底或透明底）", subtype="FILE_PATH", default="")
    sketch_sheet_path: StringProperty(name="三视图拼图", description="正视 / 侧视 / 俯视排在一张图里", subtype="FILE_PATH", default="")
    sketch_side_front: EnumProperty(
        name="侧视图正面朝向",
        items=[("LEFT", "正面朝左", "侧视图里产品正面（车头）在左边"),
               ("RIGHT", "正面朝右", "侧视图里产品正面（车头）在右边")],
        default="LEFT",
    )
    sketch_top_front: EnumProperty(
        name="俯视图正面朝向",
        items=[("BOTTOM", "正面朝下", "俯视图里产品正面在下方（常见画法）"),
               ("TOP", "正面朝上", "俯视图里产品正面在上方"),
               ("LEFT", "正面朝左", "俯视图横放，产品正面在左边"),
               ("RIGHT", "正面朝右", "俯视图横放，产品正面在右边")],
        default="BOTTOM",
    )
    sketch_size_mode: EnumProperty(
        name="尺寸",
        items=[("AUTO", "按比例", "按正视 / 侧视图的长宽比推算，只需填一个已知尺寸"),
               ("MANUAL", "长宽高", "直接填写真实的长、宽、高")],
        default="AUTO",
    )
    sketch_known_axis: EnumProperty(
        name="已知尺寸",
        items=[("LENGTH", "长", "前后方向的长度"), ("WIDTH", "宽", "左右方向的宽度"), ("HEIGHT", "高", "高度")],
        default="LENGTH",
    )
    sketch_known_value: FloatProperty(name="已知尺寸", default=1.0, min=0.01, max=100.0, unit="LENGTH", precision=3)
    sketch_length: FloatProperty(name="长", description="前后方向（Y）", default=1.0, min=0.01, max=100.0, unit="LENGTH", precision=3)
    sketch_width: FloatProperty(name="宽", description="左右方向（X）", default=1.0, min=0.01, max=100.0, unit="LENGTH", precision=3)
    sketch_height: FloatProperty(name="高", description="高度（Z）", default=1.0, min=0.01, max=100.0, unit="LENGTH", precision=3)
    sketch_resolution: IntProperty(name="精度", description="雕刻网格在最长方向上的格数（越高越细、越慢）",
                                   default=160, min=64, max=256)
    last_sketch_note: StringProperty(name="草模状态", default="", options={"SKIP_SAVE"})
    last_canvas_fit: BoolProperty(
        name="比例已适配",
        description="结果已按比例缩放并留边到白模画布，未裁切、未拉伸",
        default=False,
        options={"SKIP_SAVE"},
    )
    last_structure_note: StringProperty(default="")
    compare_mode: EnumProperty(
        name="对比模式",
        items=[
            ("SPLIT", "滑动", "左右分割对比 Camera Reference 与 AI Render"),
            ("OVERLAY", "叠加", "半透明叠加检查轮心、车身边界、地平线和透视是否重合"),
            ("DIFFERENCE", "差异", "像素差异，包含灯光和材质变化"),
            ("OUTLINE", "轮廓", "把白模产品轮廓画在结果上，检查位置、尺度与边界"),
        ],
        default="SPLIT",
        update=_refresh_compare_update,
    )
    compare_factor: FloatProperty(
        name="对比位置",
        default=0.5,
        min=0.0,
        max=1.0,
        update=_refresh_compare_update,
    )
    compare_return_workspace: StringProperty(name="返回工作区", default="")
    compare_drag_active: BoolProperty(name="鼠标拖动已启用", default=False)
    ui_show_account: BoolProperty(name="账户详情", default=False)
    ui_show_references: BoolProperty(name="参考图", default=True)
    ui_show_product_refs: BoolProperty(name="产品参考图", default=True)
    ui_show_person_refs: BoolProperty(name="人物参考图", default=False)
    ui_show_style_refs: BoolProperty(name="环境／风格参考图", default=True)
    ui_show_advanced: BoolProperty(name="高级设置", default=False)
    ui_show_prompt: BoolProperty(name="提示词", default=False)
    alignment_score: FloatProperty(name="构图对齐分数", default=-1.0, min=-1.0, max=100.0)
    alignment_message: StringProperty(name="构图对齐说明", default="")
    alignment_attempts: IntProperty(name="生成尝试次数", default=0, min=0, max=40)
    last_preparation_seconds: FloatProperty(name="本地准备耗时", default=0.0, min=0.0)
    last_total_seconds: FloatProperty(name="端到端耗时", default=0.0, min=0.0)
    last_generation_seconds: FloatProperty(name="远程生图耗时", default=0.0, min=0.0)
    last_audit_seconds: FloatProperty(name="远程验收耗时", default=0.0, min=0.0)
