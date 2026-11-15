from __future__ import annotations

from .composition import canvas_contract

REFERENCE_POLICY_VERSION = 4

PRODUCT_REFERENCE_RULE = (
    "产品身份图或三视图只用于确定各部件的材质属性；不提取其车漆颜色、灯光、高光、反射、阴影、曝光、色温或背景。"
    "Blender 当前模型唯一决定产品几何、车型造型、灯组内部和轮辐结构，以及车身车漆色相。"
    "环境只改变受光和反射，不能改变固有车漆色相；只有用户明确指定具体车漆颜色时才可覆盖。"
)
PRODUCT_REFERENCE_RULE_WITH_COLOR = (
    "兼容旧 PRODUCT_REF 设置：产品身份图或三视图仍只确定各部件材质属性，绝不提取或覆盖车身车漆颜色。"
    "不提取产品图的灯光、高光位置、反射亮条、阴影、曝光、色温或背景。"
    "Blender 当前模型唯一决定产品几何、灯组内部、轮辐结构和车漆色相；只有用户明确指定具体车漆颜色时才可覆盖。"
)

THREE_VIEW_NOTE = (
    "产品参考图可能是三视图（同一产品的正面／侧面／背面拼在一张图里）："
    "把各分格当作同一产品的部件材质样本；不得从中理解或改写造型，不要把拼版、分格线、标注文字或多个产品画进结果。"
)


def product_reference_rule(color_source: str = "BLENDER") -> str:
    base = PRODUCT_REFERENCE_RULE_WITH_COLOR if color_source == "PRODUCT_REF" else PRODUCT_REFERENCE_RULE
    return base + THREE_VIEW_NOTE


ENVIRONMENT_REFERENCE_RULE = (
    "环境主参考是产品与场景照明的主要依据：提取光源方向、色温、软硬、明暗关系、"
    "环境反射和接触阴影，让产品处在该环境中自然受光；"
    "其他环境图仅按说明补充兼容信息，不叠加互相矛盾的主光。"
    "可以提取色调、材质表现、环境质感与后期风格，不复制参考图的主体或构图。"
)
REFERENCE_LIGHTING_POLICY = (
    "【参考职责与受光规则】\n"
    "三视图／产品身份图只用于各部件材质属性，不提供车漆颜色、几何或打光依据。"
    "产品光照必须由当前环境／风格参考图推导，包含主光方向、色温、软硬、"
    "亮暗分布、环境反射、高光形态和接触阴影，使产品与环境具有一致的受光关系。"
    "不得保留或照搬产品图原有的棚拍光、轮廓光、反射亮条、阴影、曝光或背景；"
    "也不得把 Blender Camera Base 的临时照明或结构控制图的明暗当作最终打光。"
    "环境参考存在时，正文或旧 AI 提示词中与该环境冲突的产品打光描述必须重建。"
    "没有环境参考时，使用用户文字中的灯光要求；未指定时采用自然中性的统一环境光，"
    "仍不得借用产品图打光。车漆色相依据当前 Blender 模型，只有用户明确指定具体颜色时可覆盖；材质属性可由三视图逐部件核对。"
    "逐图说明只能在各自类别职责内收窄参考范围，不能把产品图升级为车漆、几何或打光参考。"
)


_COLOR_POLICY = {
    "BLENDER": "车身车漆色相以【产品外观锁定】中的 Blender 当前材质为准；环境只改变受光、反射与阴影，不改变固有色。",
    "PRODUCT_REF": "兼容旧 PRODUCT_REF：三视图只补充各部件材质属性，绝不允许参考图重涂车身车漆色相。",
    "TEXT": "车身车漆色相仍以 Blender 当前模型为准；只有用户明确指定具体车漆颜色时才可覆盖。",
}


def reference_lighting_policy(color_source: str = "BLENDER") -> str:
    extra = _COLOR_POLICY.get(color_source, "")
    return REFERENCE_LIGHTING_POLICY + (("\n" + extra) if extra else "")


def needs_reference_policy_refresh(props) -> bool:
    """One-time refresh for previously AI-polished references, preserving manual drafts."""
    return bool(
        int(getattr(props, "prompt_reference_policy_version", 0)) < REFERENCE_POLICY_VERSION
        and (getattr(props, "last_ai_prompt", "") or getattr(props, "style_sync_initialized", False))
        and (len(getattr(props, "product_images", ())) or len(getattr(props, "style_images", ())))
    )

BASE_SYSTEM_PROMPT = r"""
你是 Wondful AI 渲染器的商业视觉 Appearance 提示词导演。Blender 的确定性 Structure Packet 已经独立负责构图、相机、透视、几何、位置、尺度、旋转、遮挡和空间关系；你的输出提示词不得再承担这些结构控制职责。

你的任务是读取用户创意需求，以及产品 / 人物 / 风格参考图中与外观有关的信息，输出一段适合高质量图像生成模型直接执行的专业中文 Appearance Prompt。

必须遵守：
1. 不要在最终提示词中描述或约束构图、机位、焦段、裁切、主体二维位置、大小、角度、前后关系、遮挡、bbox、轮心坐标、地平线、消失点、画幅比例等 Structure 信息。
2. 不要输出“严格保持构图”“不要重新构图”“禁止 zoom / pan / crop”“保持 Reference 1 位置”等结构性负面约束。结构由 Blender Structure Packet 本身提供，不靠文字重复约束。
3. 当前 Blender 模型决定产品几何、车型造型、灯组、轮辐与车身车漆色相；产品身份图／三视图只提供各部件材质属性。人物图提供人物身份；环境／风格图提供整图环境受光、色调、摄影和氛围。任何参考图都不能接管 Blender 构图。
4. 每张图的“参考什么”只在本类别职责内生效；冲突内容必须忽略，不能用产品图重写几何、车漆颜色或打光。
5. Blender Camera Base 可以帮助理解当前场景是什么，但不要把它的空间结构翻译成文字提示词。
6. 如果旧提示词中残留 2.x / 3.0 早期版本生成的构图锁定、坐标、相机、bbox、轮心、裁切、透视等文字，必须在本次润色中删除。

最终提示词只应自然覆盖：
- 图片类型 / 商业用途 / 创意主题
- 主体身份和需要表现的外观特征
- 材质、表面质感、反射、透明、玻璃、金属、织物等 Appearance
- 灯光、色调、对比度、曝光、氛围
- 摄影质感与镜头成像风格（仅视觉质感，不描述机位和构图）
- 场景的视觉气氛与环境细节
- 后期、色彩管理、商业广告质感
- 画质、真实感、细节等级

输出只给最终提示词正文，不解释分析过程或参考职责，不输出 Markdown 标题、编号、项目符号或代码块；尽量写成连续自然的一段正文。产品受光描述须来自环境参考的实际光照分析。
""".strip() + "\n\n" + REFERENCE_LIGHTING_POLICY

REFERENCE_ATMOSPHERE_POLICY = (
    "【原始目标图外观优先】原始目标图是环境、相机、曝光与成像氛围的依据；白模仅控制目标产品的局部投影与几何。"
    "具体描述并匹配产品相对背景的曝光、暗部与亮部各自的面积占比、阴影覆盖、负补光、局部反差和高光面积；"
    "固有色和材质类别不变，但呈现亮度与饱和度必须随环境受光，不因产品展示而自动提亮暗部或增加补光。"
    "区分主体运动、相机追随的方向性快门拖影与景深虚化，匹配实际可见的类型、方向和主观强度；"
    "在白模投影不变的前提下保留这些成像效果，不把强烈动感改成静止或仅用浅景深替代。"
    "空气、雾尘、扬尘、辉光和颗粒只按原图实际有无、浓度、覆盖范围与受光表现还原，"
    "不统一稀释雾尘、压低反差或把曝光改成明亮通透。白天保持白天，静止保持静止，"
    "没有的运动、尘雾或光效不要添加。几何风格与镜头成像分别判断，风格化几何也可以保留原图的摄影光学。"
    "这些观察优先于通用的柔化、克制、增亮、细节清晰或画风模板；不得借此移动相机、改变产品轮廓、身份或固有色。"
)

REPLICA_POLISH_SYSTEM_PROMPT = r"""
你是 Wondful AI 渲染器的参考图复刻提示词导演。你会收到：原始目标图、当前白模粗模预览，以及产品身份参考图。

原始目标图定义必须保留的环境、相机与视觉意图：仅限本图实际可见的机位感、静止或运动状态、景深或模糊、环境纵深、照明、色调、材质、氛围和镜头成像语言。不要臆造未出现的运动、机位感、空气效果或时段。白模粗模预览仅控制目标产品的局部几何、透视、尺度和位置；不以粗模重建环境，也不要为了参考图里的另一辆车而改动真实产品身份。

输出一段可直接用于生图的中文 Appearance Prompt。自然保留原图的视觉意图和摄影语言，保留用户手写需求与产品身份；不要输出坐标、bbox、尺寸或要求改变白模几何的指令。不要解释分析过程、不要标题或列表。
""".strip() + "\n\n" + REFERENCE_LIGHTING_POLICY + "\n\n" + REFERENCE_ATMOSPHERE_POLICY



# 3.1.7: hard markers are unambiguous framing/camera vocabulary. Soft markers
# (位置/透视/地平线…) also appear in normal lighting or scenery descriptions, so a
# clause using them is only removed when it carries a constraint verb AND no
# appearance vocabulary. Removal works per clause, never the whole sentence.
_HARD_STRUCTURE_MARKERS = (
    "构图", "机位", "镜头位置", "焦段", "裁切", "取景", "消失点",
    "bbox", "轮心", "坐标", "画面比例", "占画面", "前后遮挡", "前后层级",
    "三分法", "景别", "俯仰", "偏航", "滚转", "zoom", "pan", "crop", "reframe",
)
_SOFT_STRUCTURE_MARKERS = ("位置", "尺度", "透视", "地平线")
_STRUCTURE_MARKERS = _HARD_STRUCTURE_MARKERS + _SOFT_STRUCTURE_MARKERS
_STRUCTURE_CONSTRAINT_MARKERS = (
    "保持", "锁定", "严格", "禁止", "不得", "不要", "必须", "固定", "不变", "一致",
    "贴合", "控制", "约束", "改变", "重新找", "重新构", "不得改", "不能改",
)
_APPEARANCE_MARKERS = (
    "光", "影", "高光", "反射", "阴影", "色", "质感", "材质", "纹理", "漆", "金属", "玻璃",
    "织物", "皮革", "木", "镀铬", "雾", "天空", "云", "氛围", "曝光", "对比", "饱和",
    "清晰", "细节", "车标", "logo", "标志", "车牌", "灯组",
)


def _is_structure_clause(clause: str) -> bool:
    import re

    lower = clause.lower()
    if re.search(r"(?:^|[^a-z])(?:x|y)\s*=\s*-?\d", lower):
        return True
    has_hard = any(m.lower() in lower for m in _HARD_STRUCTURE_MARKERS)
    has_soft = any(m in lower for m in _SOFT_STRUCTURE_MARKERS)
    if ("reference" in lower or "blender" in lower) and (has_hard or has_soft):
        return True
    if not (has_hard or has_soft):
        return False
    has_constraint = any(m.lower() in lower for m in _STRUCTURE_CONSTRAINT_MARKERS)
    if not has_constraint:
        return False
    if has_hard:
        return True
    has_appearance = any(m.lower() in lower for m in _APPEARANCE_MARKERS)
    return not has_appearance


def sanitize_appearance_prompt_report(text: str, *, preserve_visual_intent: bool = False) -> tuple[str, list[str]]:
    """Return (clean_prompt, removed_fragments) for an Appearance prompt."""
    import re

    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return "", []
    kept: list[str] = []
    removed: list[str] = []
    for sentence in re.split(r"(?<=[。！？!?;；])\s*|\n+", raw):
        sentence = sentence.strip()
        if not sentence:
            continue
        end = sentence[-1] if sentence[-1] in "。！？!?;；" else ""
        body = sentence[:-1] if end else sentence
        # A bbox list is one control clause, including its internal commas.
        separator = r"(?<=[，,、])(?![^\[\]]*\])" if preserve_visual_intent else r"(?<=[，,、])"
        clauses = [c for c in re.split(separator, body) if c.strip()]
        def keep(clause):
            if preserve_visual_intent and re.search(
                r"(?:^|[^a-z])(?:x|y|z)\s*=|bbox|坐标|轮心|重新构图|重新取景|裁切|裁剪|"
                r"\b(?:move|reposition|reframe|crop|zoom|pan)\b|"
                r"(?:移动|挪动|调整|改变|重设|切换).{0,8}(?:相机|机位|镜头|构图|焦段|位置|尺度)|"
                r"(?:相机|机位|镜头).{0,8}(?:移动|挪动|改为|改成|拉近|拉远|抬高|降低)",
                clause, re.IGNORECASE,
            ):
                return False
            if not _is_structure_clause(clause):
                return True
            # Replica prompts legitimately describe the visual language present
            # in the supplied target.  The white blockout remains the geometry
            # authority; do not erase that visual intent while cleaning legacy
            # generic composition instructions.
            visual = ("机位", "俯拍", "仰拍", "平视", "鸟瞰", "倾斜", "俯仰", "滚转",
                      "静态", "静止", "跟拍", "运动", "模糊", "焦段", "长焦", "广角", "透视", "景深")
            return preserve_visual_intent and any(word in clause for word in visual)
        good = [c for c in clauses if keep(c)]
        removed.extend(c.strip("，,、 ") for c in clauses if not keep(c))
        if len(good) == len(clauses):
            kept.append(sentence)
        elif good:
            joined = "".join(good).rstrip("，,、 ")
            kept.append(joined + (end or "。"))
    return "".join(kept).strip(), [r for r in removed if r]


def sanitize_appearance_prompt(text: str, *, preserve_visual_intent: bool = False) -> str:
    """Remove structure-control clauses from an Appearance prompt.

    Deterministic guard after AI polish and again before final render. Since
    3.1.7 it removes clauses, not whole sentences, and keeps clauses that are
    about light, colour or material even if they mention 位置/透视/地平线.
    """
    return sanitize_appearance_prompt_report(text, preserve_visual_intent=preserve_visual_intent)[0]

STYLE_ANALYSIS_SYSTEM_PROMPT = r"""
你是 Wondful AI 渲染器的视觉风格分析器。你只负责读取本轮提供的环境／风格参考图，提取可迁移到其他构图中的视觉语言。必须实际观察当前图片，不得沿用此前会话、首次上传图片或旧提示词中的风格记忆。

首先提取环境照明：主光方向、光源色温、光源软硬、明暗关系、环境反射和接触阴影；说明产品置于这个环境中应如何受光。不得混入产品参考图原有的打光。再分析并描述：色调、材质表现、布光、对比度、摄影语言、商业广告风格、场景质感、环境氛围、后期风格、镜头成像质感。不要描述或复制参考图中的具体产品、人物、Logo、文字、主体位置、相机、景别和构图。

环境主参考首先确定统一照明。逐图“参考什么”用于限定其他风格特征或辅助环境信息；多张图不叠加互相冲突的主光。不复制图中产品自身的独立棚拍打光。

输出一段紧凑、明确、可直接嵌入最终图像生成提示词的中文风格摘要，不要解释分析过程。
""".strip()

SOURCE_STYLE_ANALYSIS_SYSTEM_PROMPT = r"""
你是 Wondful 原图编辑的独立风格分析器。只看本轮原始来源图，反推出整张图的环境、光线、曝光、色调、氛围、摄影和后期语言；不得复制来源图中旧产品的车型、产品几何、灯组、轮辐、Logo、材质或车漆颜色。最终合并时，整图氛围来自该摘要，当前 Blender 模型决定目标产品几何和车漆色相，三视图／身份图只决定各部件材质属性。环境反射可以改变车漆的可见反射，不能改变其固有色。输出一段紧凑中文风格摘要，不解释过程。
""".strip()


PRODUCT_ANALYSIS_SYSTEM_PROMPT = r"""
你是 Wondful AI 渲染器的产品外观分析器。只看本轮附带的产品参考图（可能是三视图：同一产品的多个角度拼在一张图里），
为“产品外观锁定”写两项内容：
1. appearance：各非车漆部件的表面质感，一句中文短语串，例如“哑光黑饰条，熏黑玻璃，拉丝金属铭牌，轮胎橡胶纹理”。
   描述材质与表面处理（光泽度、清漆、拉丝、磨砂、织物纹理等）。车身车漆颜色和车漆表面基线来自当前 Blender 模型，不得从本图推断或覆盖；不写产品照片自己的打光、背景、阴影、曝光，也不写构图与相机。
2. details：仅列可由材质区分的可见部件，3 到 8 项名词短语，例如车标金属表面、前车牌材质、日行灯透镜、轮毂涂层，或家具的品牌铭牌、扶手缝线材质、椅背面料、金属脚套。
   不把这些项目当作几何或结构指令；灯组布局、轮辐数量和任何造型由当前 Blender 模型决定。
只输出一个 JSON 对象，不要任何解释：{"appearance": "...", "details": ["...", "..."]}
""".strip()


def product_analysis_user_text(color_source: str = "BLENDER", note: str = "") -> str:
    color = "只写各部件材质与表面处理，不写或推断车身车漆颜色；车漆色相由当前 Blender 模型决定，除非用户明确指定具体颜色。"
    text = "请分析附带的产品参考图，填写产品外观锁定。" + color
    if (note or "").strip():
        text += "\n用户对这张图的部件材质重点说明：" + note.strip()
    return text


def product_analysis_cache_key(fingerprint, note: str = "") -> str:
    """Role-versioned key for product material autofill; callers retain manual drafts."""
    import hashlib
    return hashlib.sha256((str(REFERENCE_POLICY_VERSION) + "\n" + str(fingerprint) + "\n" + (note or "").strip()).encode("utf-8")).hexdigest()


def parse_product_analysis(text: str) -> tuple[str, list[str]]:
    """Parse the analyser reply into (appearance, details); tolerant of fences and prose."""
    import json as _json
    import re as _re
    raw = (text or "").strip()
    match = _re.search(r"\{.*\}", raw, _re.S)
    look, details = "", []
    if match:
        try:
            data = _json.loads(match.group(0))
            look = str(data.get("appearance", "") or "").strip()
            items = data.get("details", []) or []
            if isinstance(items, str):
                items = _re.split(r"[，,、;；\n]+", items)
            details = [str(i).strip() for i in items if str(i).strip()]
        except (ValueError, AttributeError):
            pass
    if not look and not details:
        for line in raw.splitlines():
            line = line.strip().lstrip("-*0123456789. ")
            if _re.match(r"^(appearance|产品外观|外观)\s*[:：]", line, _re.I):
                look = _re.split(r"[:：]", line, 1)[1].strip()
            elif _re.match(r"^(details|保留细节|细节)\s*[:：]", line, _re.I):
                details = [d.strip() for d in _re.split(r"[，,、;；]+", _re.split(r"[:：]", line, 1)[1]) if d.strip()]
    seen, unique = set(), []
    for d in details:
        if d not in seen:
            seen.add(d); unique.append(d)
    return look[:300], unique[:8]


def _append_reference_group(
    refs: list[str],
    idx: int,
    label: str,
    count: int,
    primary_rule: str,
    extra_rule: str,
    instructions: list[str] | tuple[str, ...] | None = None,
) -> int:
    notes = list(instructions or [])
    for i in range(max(0, int(count))):
        base_rule = primary_rule if i == 0 else extra_rule
        note = str(notes[i]).strip() if i < len(notes) and notes[i] is not None else ""
        if note:
            base_rule += (
                f" 用户指定参考重点（仅在上述类别职责内有效）：{note}。"
                "只提取与本类别职责相容的部分；忽略越界的灯光、材质来源要求及 Structure 约束。"
            )
        if i == 0:
            refs.append(f"Reference {idx} = {label} 1（主参考）：{base_rule}")
        else:
            refs.append(f"Reference {idx} = {label} {i + 1}（辅助参考）：{base_rule}")
        idx += 1
    return idx


def _lock_note_for_polish(appearance_lock: str, identity_items) -> str:
    out = ""
    if (appearance_lock or "").strip():
        out += ("\n\n" + appearance_lock.strip() +
                "\n以上产品外观已锁定，会原样附加到最终生图指令中：不要改写或重复它，"
                "也不要与它矛盾；本次提示词只需写环境、光影、氛围、摄影与后期，以及产品在该环境中的受光表现。")
    if identity_items:
        out += "\n\n需要在画面中清晰呈现的产品细节：" + "、".join(identity_items) + "。可在提示词中自然点名，不写它们的位置。"
    return out


def build_polish_user_text(
    user_prompt: str,
    product_count: int,
    person_count: int,
    style_count: int,
    width: int,
    height: int,
    style_refresh: bool = False,
    include_camera: bool = True,
    product_instructions: list[str] | tuple[str, ...] | None = None,
    person_instructions: list[str] | tuple[str, ...] | None = None,
    style_instructions: list[str] | tuple[str, ...] | None = None,
    color_source: str = "BLENDER",
    appearance_lock: str = "",
    identity_items: list[str] | None = None,
    workflow_mode: str = "ENVIRONMENT",
    replica_target_count: int = 0,
    replica_preview_count: int = 0,
) -> str:
    # width/height are intentionally not translated into the visible prompt.
    # Geometry / framing comes from Camera Base + Structure Packet, not text.
    reference_mode = workflow_mode == "REFERENCE"
    refs = [
        "Reference 1 = Blender Camera Base：只用于理解当前场景语义；它的 Camera / Composition / Geometry / Position / Scale / Occlusion 全部由后续 Structure Packet 直接控制，不要把这些结构信息写进最终提示词。"
    ] if include_camera and not reference_mode else []
    idx = 2 if refs else 1
    if reference_mode:
        if replica_target_count:
            refs.append(f"Reference {idx} = 复刻原始目标图：只提取本图实际可见的视觉意图（例如画面确有的机位感、静止或运动状态、环境纵深、光线、气氛与镜头成像语言）；不得臆造未出现的特征、复制其中车辆身份或用文字改写白模决定的几何。")
            idx += replica_target_count
        if replica_preview_count:
            refs.append(f"Reference {idx} = 当前白模粗模预览：仅控制目标产品局部几何、透视、遮挡和尺度；不覆盖来源图的环境、相机与曝光。")
            idx += replica_preview_count
    idx = _append_reference_group(
        refs,
        idx,
        "产品参考图",
        product_count,
        product_reference_rule(color_source),
        product_reference_rule(color_source),
        instructions=product_instructions,
    )
    idx = _append_reference_group(
        refs,
        idx,
        "人物参考图",
        person_count,
        "默认只提取人物身份、人脸、发型、服装和主要外观，不参考其构图。",
        "补充同一人物的身份与外观细节，不参考其构图。",
        instructions=person_instructions,
    )
    _append_reference_group(
        refs,
        idx,
        "环境／风格参考图",
        style_count,
        ENVIRONMENT_REFERENCE_RULE,
        "辅助环境参考：" + ENVIRONMENT_REFERENCE_RULE,
        instructions=style_instructions,
    )

    summary = f"本次 Appearance 参考：产品 {product_count} 张，人物 {person_count} 张，环境／风格 {style_count} 张。"
    refresh_note = ""
    if style_refresh:
        refresh_note = (
            "\n\n【风格参考已更新】\n"
            "必须重新读取本轮最新风格图，从零重建色调、材质、灯光、摄影、环境氛围和后期语言。"
            "旧提示词中来自上一组风格图的视觉描述全部视为旧缓存；保留用户真正的创意主题与主体需求即可。"
            "同时清除旧提示词中任何构图、相机、位置、尺度、bbox、轮心、裁切、透视等 Structure 描述。"
        )

    return (
        summary
        + "\n"
        + "\n".join(refs)
        + refresh_note
        + "\n\n用户当前创意需求 / 当前可见提示词：\n"
        + (user_prompt.strip() or "生成高品质商业视觉画面。")
        + _lock_note_for_polish(appearance_lock, identity_items)
        + "\n\n" + reference_lighting_policy(color_source)
        + ("\n\n" + REFERENCE_ATMOSPHERE_POLICY if reference_mode else "")
        + ("\n\n请直接输出最终复刻 Appearance Prompt。保留原始目标图实际可见的环境、相机、光线、场景氛围、静止或运动状态与镜头成像语言；白模粗模预览仅控制目标产品局部投影，不重建环境。"
           if reference_mode else "\n\n请直接输出最终 Appearance Prompt。不要加入任何构图或几何约束；Structure 全部交给 Blender 参考图与 Structure Packet。")
    )


def build_render_prompt(
    polished_prompt: str,
    width: int,
    height: int,
    product_count: int,
    person_count: int,
    style_count: int,
    target_size: str = "",
    strict_lock: bool = True,
    structure_guide: bool = False,
    structure_packet: dict | None = None,
    product_instructions: list[str] | tuple[str, ...] | None = None,
    person_instructions: list[str] | tuple[str, ...] | None = None,
    style_instructions: list[str] | tuple[str, ...] | None = None,
    color_source: str = "BLENDER",
    appearance_lock: str = "",
    identity_items: list[str] | None = None,
    workflow_mode: str = "ENVIRONMENT",
) -> str:
    """Build an appearance-only generation instruction.

    Structure is not restated as textual geometry constraints. The only hidden
    text about Structure is role metadata so the provider knows what each
    attached raster means. Actual composition comes from the rasters themselves.
    """
    mapping = [
        ("Reference 1 = Blender Camera Base：目标产品局部投影依据，不接管来源图的环境、相机或曝光。"
         if workflow_mode == "REFERENCE" else
         "Reference 1 = Blender Camera Base：结构源图。其空间关系直接作为生成基础，不从文字提示词推导构图。")
    ]
    idx = 2
    packet = structure_packet or {}
    role_text = {
        "product_mask": "产品可见区域 Mask；只是一张结构控制图，不是最终黑白设计。",
        "scene_depth": "Scene Depth；只表达可见表面的深度层级，控制图明暗不是最终画面明暗。",
        "scene_normal": "Scene Normal；只表达表面朝向与曲面结构，RGB 不是最终材质颜色。",
        "product_silhouette": "产品可见外轮廓控制图，不是最终线稿。",
        "part_id": "产品部件 ID 分区，只表达部件边界，伪彩色绝不能进入最终画面。",
    }
    packet_roles = list(packet.get("generation_reference_roles", []))
    if packet_roles:
        for role in packet_roles:
            mapping.append(
                f"Reference {idx} = Blender Structure Packet / {role}：{role_text.get(role, '确定性结构控制数据，不是最终外观。')}"
            )
            idx += 1
    elif structure_guide:
        mapping.append(f"Reference {idx} = Blender Structure Guide：结构控制数据，不是最终外观。")
        idx += 1
        mapping.append(f"Reference {idx} = Blender Product Mask：产品结构控制数据，不是最终外观。")
        idx += 1

    idx = _append_reference_group(
        mapping,
        idx,
        "产品参考图",
        product_count,
        product_reference_rule(color_source),
        product_reference_rule(color_source),
        instructions=product_instructions,
    )
    idx = _append_reference_group(
        mapping,
        idx,
        "人物参考图",
        person_count,
        "默认只提供人物身份、人脸、发型、服装与主要外观，不提供构图。",
        "补充同一人物的身份与外观细节，不提供构图。",
        instructions=person_instructions,
    )
    _append_reference_group(
        mapping,
        idx,
        "原始复刻目标图" if workflow_mode == "REFERENCE" else "环境／风格参考图",
        style_count,
        ("保留该图环境、相机、曝光和摄影语言；目标产品局部投影由白模控制，旧产品身份不保留。"
         if workflow_mode == "REFERENCE" else ENVIRONMENT_REFERENCE_RULE),
        ("辅助原始复刻目标图：补充本图实际可见的环境与成像；不得改变目标产品局部投影。"
         if workflow_mode == "REFERENCE" else "辅助环境参考：" + ENVIRONMENT_REFERENCE_RULE),
        instructions=style_instructions,
    )

    # Keep target size out of the text as well. The provider adapter / output
    # pipeline already receives the base image and target dimensions separately.
    appearance_rule = (
        "任务：只做 Appearance Re-render。构图与几何完全由已附带的 Blender Camera Base / Structure Packet 决定；"
        "不要从下方文字中寻找或生成任何新的构图方案。下方正文只控制材质、灯光、色调、身份、风格、环境质感与后期。"
    )
    if workflow_mode == "REFERENCE":
        appearance_rule = ("任务：在原始来源画布上替换目标产品，保留来源环境、相机、曝光与镜头成像。"
                           "白模仅控制目标产品局部投影；不按环境体块或地形粗模重建场景。实际附件角色清单决定图像顺序。")
    control_color_rule = (
        "结构控制图仅用于条件控制：不要把 Mask 的黑白、Depth 的灰度、Normal 的 RGB、Part ID 的伪彩色、Silhouette 的线条复制成最终视觉元素。"
    )
    from .appearance_lock import identity_block
    blocks = [
        appearance_rule,
        canvas_contract(width, height, target_size),
        "\n".join(mapping),
        appearance_lock.strip(),
        sanitize_appearance_prompt(polished_prompt, preserve_visual_intent=workflow_mode == "REFERENCE"),
        identity_block(list(identity_items or [])),
        control_color_rule,
        reference_lighting_policy(color_source),
        REFERENCE_ATMOSPHERE_POLICY if workflow_mode == "REFERENCE" else "",
    ]
    return "\n\n".join(b for b in blocks if b)


def select_polish_source(props, visible_prompt: str, style_refresh: bool) -> str:
    """Keep fresh user edits authoritative while avoiding repeated old AI style."""
    visible = (visible_prompt or "").strip()
    last_ai = (getattr(props, "last_ai_prompt", "") or "").strip()
    stored = (getattr(props, "prompt_source", "") or "").strip()
    if visible != last_ai or not last_ai:
        props.prompt_source = visible
        return visible
    if style_refresh and stored:
        return stored
    return visible or stored


def enrich_prompt_with_typesafe(
    canonical_prompt: str,
    style_notes: list[str] | None = None,
) -> tuple[str, dict]:
    """Compatibility entrypoint backed by real Jev semantics in 3.1.3."""
    from . import jev_semantics
    params = jev_semantics.analyze_appearance(canonical_prompt, style_notes)
    synthesized = str(params.get("synthesized", "") or "").strip()
    base = sanitize_appearance_prompt(canonical_prompt)
    combined = base if not synthesized or synthesized in base else (f"{base}。{synthesized}" if base else synthesized)
    return sanitize_appearance_prompt(combined), params

