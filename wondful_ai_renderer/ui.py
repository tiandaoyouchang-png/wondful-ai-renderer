from __future__ import annotations

from pathlib import Path

from textwrap import wrap
import unicodedata

import bpy
from bpy.types import Panel, UIList

from .properties import MAX_REFERENCES_PER_KIND
from . import prompt_profiles
from .prompt_engine import needs_reference_policy_refresh
from .viewport_capture import camera_output_dimensions
from .operators import provider_variant_cap, render_variant_count

# Text built from several parts is not a single key in Blender's translation table,
# so it is translated here (see i18n.py).
_t = bpy.app.translations.pgettext_iface


def _prefs(context):
    return context.preferences.addons[__package__].preferences


def _disclosure(row, props, attr, label, icon="NONE", suffix=""):
    opened = bool(getattr(props, attr))
    row.prop(props, attr, text="", icon="TRIA_DOWN" if opened else "TRIA_RIGHT", emboss=False)
    row.label(text=f"{label}{suffix}", icon=icon)
    return opened


PROMPT_EDITOR_VISIBLE_ROWS = 6


def _prompt_display_lines(text, columns=36):
    """Wrap CJK and Latin display cells without modifying the canonical prompt."""
    result = []
    for paragraph in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line, width = "", 0
        for char in paragraph:
            cells = 0 if unicodedata.combining(char) else (2 if unicodedata.east_asian_width(char) in "WF" else 1)
            if line and width + cells > columns:
                result.append(line)
                line, width = "", 0
            line += char
            width += cells
        result.append(line)
    return result


def _prompt_char_count(text: str) -> int:
    """Count visible prompt characters; line breaks are not counted as 字."""
    return len((text or "").replace("\r", "").replace("\n", ""))


def _draw_prompt_editor(layout, props, columns=36):
    """Draw one canonical prompt editor.

    Prefer Blender's native multi-line textbox whenever the running build exposes
    it (feature detection, not a hard version check). Older Blender builds do not
    provide an inline multi-line RNA editor in panels, so never fake one with
    per-line StringProperty fields: expose a single editable StringProperty and
    a full Text Editor in the current area with a return button.
    """
    expanded = bool(getattr(props, "prompt_expanded", False))
    lines = _prompt_display_lines(props.prompt, columns)
    if callable(getattr(layout, "textbox", None)):
        try:
            layout.textbox(
                props, "prompt",
                initial_visible_lines=max(12, min(30, len(lines))) if expanded else PROMPT_EDITOR_VISIBLE_ROWS,
                placeholder="输入创意需求，或点击 AI 润色生成完整提示词…",
            )
            return
        except (TypeError, AttributeError):
            # A backport may expose a different signature; keep the panel usable.
            pass

    field = layout.box()
    # Always begin with one real editable field. Both routes edit props.prompt.
    entry = field.row()
    entry.scale_y = 1.4
    entry.prop(props, "prompt", text="")
    if props.prompt:
        preview = field.column(align=True)
        preview.scale_y = 0.95
        visible = lines[:120] if expanded else lines[:PROMPT_EDITOR_VISIBLE_ROWS]
        for line in visible:
            preview.label(text=line or " ")
        if len(lines) > len(visible):
            preview.label(text=f"还有 {len(lines) - len(visible)} 行，" + ("可在完整编辑器中读取" if expanded else "点击展开全文"))
    field.operator("wondful.open_prompt_text_editor", text="编辑完整提示词", icon="GREASEPENCIL")


def _draw_progress_action(row, props, *, active_status: str, idle_text: str, icon: str, operator: str, enabled: bool = True):
    """Use the action button's own slot as its live progress indicator."""
    if props.status == active_status:
        pct = max(0, min(100, int(round(float(props.progress) * 100))))
        phase = getattr(props, "task_phase", "")
        text = f"{phase or idle_text} · 估算 {pct}%"
        row.enabled = False
        if hasattr(row, "progress"):
            row.progress(factor=max(0.0, min(1.0, float(props.progress))), text=text, type="BAR")
        else:
            # Older Blender fallback: slider occupies the same action slot.
            row.prop(props, "progress", text=text, slider=True)
        return None
    row.enabled = bool(enabled)
    return row.operator(operator, text=idle_text, icon=icon)


class WONDFUL_UL_prompt_lines(UIList):
    bl_idname = "WONDFUL_UL_prompt_lines"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.prop(item, "text", text="", emboss=False)


class WONDFUL_UL_reference_items(UIList):
    """Legacy compatibility only; 2.7 renders references as image cards instead."""
    bl_idname = "WONDFUL_UL_reference_items"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        image = getattr(item, "image", None)
        layout.label(text=(image.name if image else "无效图片"), icon="IMAGE_DATA" if image else "ERROR")


def _draw_reference_card(layout, props, item, index, kind):
    card = layout.box()
    split = card.split(factor=0.36)
    preview = split.column(align=True)
    if item.image:
        icon_value = bpy.types.UILayout.icon(item.image)
        preview.template_icon(icon_value=icon_value, scale=5.0)
    else:
        preview.label(text="图片无效", icon="ERROR")

    meta = split.column(align=True)
    if kind == "STYLE":
        primary_text = "主光来源"
        icon0 = "LIGHT_SUN"
    else:
        primary_text = "主参考"
        icon0 = "CHECKMARK"
    if kind == "PRODUCT" and index > 0:
        meta.label(text="未使用（只用第一张）", icon="ERROR")
    else:
        meta.label(text=primary_text if index == 0 else f"辅助 {index + 1}", icon=icon0 if index == 0 else "IMAGE_DATA")
    if item.image:
        name = item.image.name
        meta.label(text=name if len(name) <= 24 else name[:21] + "…")
    meta.prop(item, "instruction", text="造型重点" if kind == "PRODUCT" else "参考什么")
    controls = meta.row(align=True)
    if kind != "PRODUCT":
        up = controls.operator("wondful.move_reference", text="", icon="TRIA_UP")
        up.ref_kind, up.direction, up.item_index = kind, "UP", index
        down = controls.operator("wondful.move_reference", text="", icon="TRIA_DOWN")
        down.ref_kind, down.direction, down.item_index = kind, "DOWN", index
    replace = controls.operator("wondful.replace_reference", text="", icon="FILE_REFRESH")
    replace.ref_kind, replace.item_index = kind, index
    remove = controls.operator("wondful.clear_reference", text="", icon="X")
    remove.ref_kind, remove.item_index = kind, index


def _draw_reference_group(layout, props, title, collection_attr, index_attr, kind, open_attr):
    collection = getattr(props, collection_attr)
    box = layout.box()
    header = box.row(align=True)
    opened = _disclosure(header, props, open_attr, title, suffix=f"  {len(collection)}")
    header.separator()
    replaces = kind in {"STYLE", "PRODUCT"} and bool(collection)
    add_text = "换图" if replaces else "文件"
    add_icon = "FILE_REFRESH" if replaces else "ADD"
    add = header.operator("wondful.load_reference", text=add_text, icon=add_icon)
    add.ref_kind = kind
    paste = header.operator("wondful.paste_reference", text=("替换粘贴" if replaces else "粘贴"))
    paste.ref_kind = kind
    if collection:
        clear_all = header.operator("wondful.clear_all_references", text="", icon="TRASH")
        clear_all.ref_kind = kind
    if not opened:
        return

    if kind == "PRODUCT":
        box.label(text="只用 1 张，建议放三视图（正/侧/后拼成一张）。", icon="INFO")
        box.label(text="仅参考产品造型，不参考原图打光。")
    elif kind == "STYLE":
        box.label(text="决定产品受光、反射、阴影与环境氛围。", icon="LIGHT")
    if not collection:
        empty = box.column(align=True)
        empty.label(text="拖入文件，或复制图片后点“粘贴”", icon="IMAGE_DATA")
        if kind == "STYLE":
            empty.label(text="第一张决定主光，可用箭头调整顺序；最多 8 张。")
        elif kind == "PRODUCT":
            empty.label(text="只放 1 张；再上传会直接替换。")
        else:
            empty.label(text="第一张自动作为主参考；最多 8 张。")
        return

    # The preview and its identity/controls now live in one card. There is no
    # detached name list + selected-image preview anymore.
    for index, item in enumerate(collection):
        _draw_reference_card(box, props, item, index, kind)


# ---------------------------------------------------------------------------
# 3.2.0 five-step card layout helpers
# ---------------------------------------------------------------------------

STEP_NUMBERS = ("①", "②", "③", "④", "⑤")
GENERATE_STAGES = ("读产品图", "分析环境", "写提示词", "渲染验收")
BUSY_STATUSES = {"AUTHENTICATING", "CLASSIFYING", "POLISHING", "RENDERING", "REFRESHING_MODELS"}


def _step_card(layout, number, title, *, done, ready=True):
    """One step = one box. The number turns into a CHECKMARK once the step is done.

    ``ready`` only greys the card (``active``); it never disables buttons, so
    a user can always prepare a later step first.
    """
    card = layout.box()
    card.active = bool(ready or done)
    header = card.row(align=True)
    title_col = header.row(align=True)
    if done:
        title_col.label(text=title, icon="CHECKMARK")
    else:
        title_col.label(text=f"{STEP_NUMBERS[number - 1]}  {_t(title)}")
    tools = header.row(align=True)
    tools.alignment = "RIGHT"
    return card, tools


def _fit(tools, units):
    """Give the header's right-hand tools a fixed width so the step title never truncates first."""
    try:
        tools.ui_units_x = units
    except AttributeError:
        pass


def _thumb_scale(context, fraction=1.0, lo=4.0, hi=16.0):
    """template_icon scale is in units of ~20 UI pixels; fit the sidebar width."""
    region = getattr(context, "region", None)
    width = float(getattr(region, "width", 0) or 320)
    ui_scale = float(getattr(getattr(context.preferences, "system", None), "ui_scale", 1.0) or 1.0)
    usable = width / max(0.5, ui_scale) - 44.0
    return max(lo, min(hi, usable * fraction / 20.0))


def _draw_thumbnail(layout, image, scale):
    """Large preview of an image datablock (works in Blender 4.x sidebars)."""
    if image is None:
        layout.label(text="图片无效", icon="ERROR")
        return
    try:
        image.preview_ensure()
    except Exception:
        pass
    row = layout.row()
    row.alignment = "CENTER"
    row.template_icon(icon_value=bpy.types.UILayout.icon(image), scale=scale)


def _short(text, limit=26):
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _reference_header_tools(tools, collection, kind):
    """Compact icon actions on the card header (only once images exist; the empty
    state has its own big buttons). Product keeps one image, so its replace /
    paste-replace / remove live on the image itself and the header stays clean."""
    if not collection or kind == "PRODUCT":
        return
    _fit(tools, 3.3)
    add = tools.operator("wondful.load_reference", text="", icon="ADD")
    add.ref_kind = kind
    paste = tools.operator("wondful.paste_reference", text="", icon="PASTEDOWN")
    paste.ref_kind = kind
    clear_all = tools.operator("wondful.clear_all_references", text="", icon="TRASH")
    clear_all.ref_kind = kind


def _draw_reference_empty(card, kind, hint_lines):
    drop = card.box()
    actions = drop.row(align=True)
    actions.scale_y = 1.6
    add = actions.operator("wondful.load_reference", text="选择图片", icon="FILEBROWSER")
    add.ref_kind = kind
    paste = actions.operator("wondful.paste_reference", text="粘贴", icon="PASTEDOWN")
    paste.ref_kind = kind
    hints = drop.column(align=True)
    hints.scale_y = 0.85
    for index, line in enumerate(hint_lines):
        hints.label(text=line, icon="INFO" if index == 0 else "BLANK1")


def _draw_item_controls(row, item, index, kind, *, movable):
    if movable:
        up = row.operator("wondful.move_reference", text="", icon="TRIA_UP")
        up.ref_kind, up.direction, up.item_index = kind, "UP", index
        down = row.operator("wondful.move_reference", text="", icon="TRIA_DOWN")
        down.ref_kind, down.direction, down.item_index = kind, "DOWN", index
    replace = row.operator("wondful.replace_reference", text="", icon="FILE_REFRESH")
    replace.ref_kind, replace.item_index = kind, index
    if kind == "PRODUCT":
        # Product holds one image: pasting replaces it (operator behaviour).
        paste = row.operator("wondful.paste_reference", text="", icon="PASTEDOWN")
        paste.ref_kind = kind
    remove = row.operator("wondful.clear_reference", text="", icon="X")
    remove.ref_kind, remove.item_index = kind, index


def _draw_instruction(layout, item, placeholder):
    if _supports_placeholder():
        layout.prop(item, "instruction", text="", icon="GREASEPENCIL", placeholder=placeholder)
    else:
        layout.prop(item, "instruction", text=placeholder)


def _draw_primary_reference(card, context, item, index, kind, *, caption, caption_icon, placeholder, movable):
    """First reference: a large centred thumbnail, caption + compact controls under it."""
    _draw_thumbnail(card, item.image, _thumb_scale(context))
    meta = card.row(align=True)
    label = meta.row(align=True)
    label.label(text=caption, icon=caption_icon)
    controls = meta.row(align=True)
    controls.alignment = "RIGHT"
    _fit(controls, 4.4 if movable else (3.3 if kind == "PRODUCT" else 2.2))
    _draw_item_controls(controls, item, index, kind, movable=movable)
    _draw_instruction(card, item, placeholder)


def _draw_secondary_reference(card, context, item, index, kind, *, caption, caption_icon, placeholder, movable):
    """Further references: small thumbnail on the left, caption, controls and note on the right."""
    sub = card.box()
    row = sub.row()
    thumb = row.column()
    _draw_thumbnail(thumb, item.image, 2.6)
    meta = row.column(align=True)
    head = meta.row(align=True)
    head.label(text=caption, icon=caption_icon)
    controls = meta.row(align=True)
    _draw_item_controls(controls, item, index, kind, movable=movable)
    _draw_instruction(meta, item, placeholder)


def _stage_index(props):
    """Map the live task phase onto the four 「生成」 stages (display only)."""
    if props.status == "RENDERING":
        return 3
    phase = getattr(props, "task_phase", "") or ""
    if props.status == "POLISHING":
        if "风格" in phase or "环境" in phase:
            return 1
        if "产品" in phase or "准备" in phase:
            return 0
        return 2
    return -1


def _draw_progress_bar(layout, props, text):
    bar = layout.row(align=True)
    factor = max(0.0, min(1.0, float(props.progress)))
    if hasattr(bar, "progress"):
        bar.progress(factor=factor, text=text, type="BAR")
    else:
        bar.enabled = False
        bar.prop(props, "progress", text=text, slider=True)


def _draw_iou_gate(adv, context, props, busy):
    """高级 › 轮廓验收 (1.2.3): fixed-coordinate IoU ≥ 0.95, bounded regeneration (≤ 3)."""
    box = adv.box()
    row = box.row(align=True)
    row.enabled = not busy
    row.prop(props, "iou_gate", text="轮廓验收 · 不过线自动重生成", icon="CHECKMARK")
    if getattr(props, "iou_gate", True):
        row = box.row(align=True)
        row.enabled = not busy
        row.prop(props, "iou_gate_max", text="最多生成（次）")
        box.label(text="IoU ≥ 0.95，固定坐标；通过即停，每次都从原始白模重画", icon="INFO")
    note = getattr(props, "last_iou_note", "")
    if note:
        icon = "CHECKMARK" if "通过（" in note and "未通过" not in note else "ERROR"
        for index, line in enumerate(wrap(note, width=44)[:3]):
            box.label(text=line, icon=icon if index == 0 else "BLANK1")


def _draw_detail_paste(adv, context, props, busy):
    """高级 › 细节回贴 (1.2.0): toggle, strength, 「标为原样保留」 on selected parts."""
    box = adv.box()
    box.prop(props, "detail_paste", text="细节回贴", icon="BRUSH_DATA")
    if getattr(props, "detail_paste", True):
        row = box.row(align=True)
        row.enabled = not busy
        row.prop(props, "detail_paste_strength", text="细节强度", slider=True)
        row = box.row(align=True)
        row.enabled = not busy
        row.prop(props, "detail_colour_fix", text="按材质校色")
        if getattr(props, "detail_colour_fix", True):
            row.prop(props, "detail_colour_strength", text="强度", slider=True)
        box.label(text="光照随 AI，文字 / Logo / 纹理用 Blender 的；原图存为 _raw", icon="INFO")
        coll = getattr(props, "product_collection", None)
        if coll is None:
            box.label(text="需要先在 ② 选择产品集合", icon="ERROR")
        else:
            try:
                from .viewport_capture import keep_exact_slots
                slots = keep_exact_slots(coll)
                objs = {o for (o, _i) in slots}
                auto = {o for (o, _i), k in slots.items() if k == "auto"} - {o for (o, _i), k in slots.items() if k == "explicit"}
                text = f"原样保留：{len(objs)} 个部件" + (f"（自动识别 {len(auto)}）" if auto else "") if objs else "原样保留：无（可选中车牌 / 标识后标记）"
            except Exception:
                text = "原样保留：—"
            box.label(text=text, icon="LOCKED")
        try:
            from .operators import keep_exact_state
            selected, marked = keep_exact_state(context, props)
        except Exception:
            selected, marked = [], 0
        mark = box.row(align=True)
        mark.enabled = bool(selected) and not busy
        all_marked = bool(selected) and marked == len(selected)
        mark.operator("wondful.mark_keep_exact", text="取消原样保留" if all_marked else "标为原样保留",
                      icon="UNLOCKED" if all_marked else "LOCKED")
    note = getattr(props, "last_detail_note", "")
    if note:
        main, _sep, warn = note.partition(" · 注意：")
        box.label(text=_short(main, 44), icon="ERROR" if ("失败" in main or "跳过" in main) else "CHECKMARK")
        if warn:  # 1.2.2: a plate / logo that reads mirrored in the model itself
            for index, line in enumerate(wrap(warn, width=42)[:4]):
                box.label(text=line, icon="ERROR" if index == 0 else "BLANK1")


def _draw_massing(adv, context, props, busy):
    """高级 › 环境体块 (1.2.0): context-block collection, per-block 建筑参考, blocks from three views."""
    box = adv.box()
    box.prop(props, "massing_enabled", text="环境体块", icon="MESH_CUBE")
    if not getattr(props, "massing_enabled", True):
        return
    row = box.row(align=True)
    row.enabled = not busy
    row.prop(props, "massing_collection", text="集合")
    coll, blocks = None, []
    try:
        from . import massing as MS
        coll = MS.find_collection(context.scene, props)
        if coll is not None:
            blocks = MS.block_roots(MS.massing_objects(coll, getattr(props, "product_collection", None)))
    except Exception:
        coll, blocks = None, []
    if coll is not None:
        how = "已选择" if getattr(props, "massing_collection", None) else "自动识别"
        box.label(text=f"{how}：{coll.name} · {len(blocks)} 个体块", icon="CHECKMARK")
        box.label(text="白模里显示为灰蓝色，AI 画成真实建筑；产品保持锁定", icon="INFO")
    elif getattr(props, "massing_collection", None) is not None:
        box.label(text="该集合与产品集合重叠，已忽略", icon="ERROR")
    else:
        box.label(text="未找到：集合名含 环境 / 体块 / Context / Massing 即自动识别", icon="INFO")
    if getattr(props, "last_massing_note", ""):
        box.label(text=props.last_massing_note[:60], icon="BLANK1")
    if coll is not None:
        _draw_block_refs(box, context, props, busy, blocks)
    _draw_massing_hull(box, props, busy)
    _draw_scene_blockout(box, props, busy)


def _draw_scene_blockout(box, props, busy):
    """1.2.3 从参考图生成场景粗模: one environment photo → a whole blockout (Wondful 场景粗模)."""
    sub = box.box()
    if not _disclosure(sub.row(align=True), props, "ui_show_scene_blockout", "从参考图生成场景粗模", icon="WORLD"):
        return
    col = sub.column()
    col.enabled = not busy
    col.label(text="一张环境照片 → 地形 / 道路 / 建筑 / 树 / 远山体块", icon="INFO")
    col.prop(props, "sb_image_path", text="参考图")
    custom = (getattr(props, "sb_image_path", "") or "").strip()
    first = ""
    try:
        items = list(getattr(props, "style_images", []) or [])
        if items:
            first = Path(getattr(items[0], "source_path", "") or getattr(getattr(items[0], "image", None), "filepath", "")
                         or "").name
    except Exception:
        first = ""
    if not custom:
        col.label(text=f"留空：用 ③ 第一张「{first[:28]}」" if first else "留空：用 ③ 第一张环境参考图（现在还没有）",
                  icon="BLANK1" if first else "ERROR")
    row = col.row(align=True)
    row.prop(props, "sb_set_camera", text="同时设置相机")
    row.prop(props, "sb_correct", text="自动校正一轮")
    col.label(text="参考图只用于分析，不发给生图模型；产品位置不变", icon="BLANK1")
    build = col.row(align=True)
    build.scale_y = 1.2
    build.enabled = bool(custom or first)
    build.operator("wondful.scene_blockout_build", text="从参考图生成场景粗模", icon="MOD_BUILD")
    note = getattr(props, "last_sb_note", "")
    if note:
        bad = "失败" in note or "请先" in note or "不存在" in note or "需要" in note
        for index, line in enumerate(wrap(note, width=44)[:4]):
            col.label(text=line, icon=("ERROR" if bad else "CHECKMARK") if index == 0 else "BLANK1")
    else:
        col.label(text="约 3 分钟；重新生成会替换上一次的场景粗模", icon="BLANK1")


def _draw_multiview(adv, context, props, busy):
    """高级 › 多角度一致 (1.2.3): master camera + its accepted image, render the other selected cameras."""
    box = adv.box()
    if not _disclosure(box.row(align=True), props, "ui_show_multiview", "多角度一致", icon="OUTLINER_OB_CAMERA"):
        return
    row = box.row(align=True)
    row.enabled = not busy
    row.prop(props, "mv_master_camera", text="主视角")
    row.operator("wondful.mv_set_master", text="设为主视角", icon="CAMERA_DATA")
    row = box.row(align=True)
    row.enabled = not busy
    row.prop(props, "mv_master_image", text="主视角图")
    master = getattr(props, "mv_master_camera", None)
    try:
        others = [o for o in (getattr(context, "selected_objects", None) or [])
                  if getattr(o, "type", "") == "CAMERA" and o != master]
    except Exception:
        others = []
    if master is None:
        box.label(text="先用 ⑤ 渲染一个满意的相机，再点「设为主视角」", icon="INFO")
    box.label(text=f"选中的其他相机：{len(others)} 个" + ("（在视图里选中要渲染的相机）" if not others else ""),
              icon="CHECKMARK" if others else "INFO")
    _draw_mv_block_name(box, context, props, busy)
    row = box.row(align=True)
    row.enabled = not busy
    row.prop(props, "mv_widen", text="加宽体块")
    row.prop(props, "mv_env_check", text="环境一致性检查")
    box.label(text="离主视角最近的先渲染；每个相机 1 次生图，产品超标最多重试 1 次", icon="INFO")
    act = box.row(align=True)
    act.enabled = bool(master is not None and others) and not busy
    act.operator("wondful.multiview_render", text=f"渲染 {len(others)} 个相机" if others else "渲染选中的相机",
                 icon="RENDER_STILL")
    note = getattr(props, "last_mv_note", "")
    if note:
        bad = ("失败" in note) or ("未完成" in note) or ("超标" in note) or ("注意" in note)
        for index, line in enumerate(wrap(note, width=44)[:5]):
            box.label(text=line, icon=("ERROR" if bad else "CHECKMARK") if index == 0 else "BLANK1")


def _draw_mv_block_name(box, context, props, busy):
    """多角度一致 › the active block's name as the prompt uses it (editable; empty = automatic)."""
    active = getattr(context, "active_object", None)
    if active is None or getattr(active, "type", "") != "MESH" or not hasattr(active, "wondful_hint"):
        return
    try:
        from . import massing as MS
        coll = MS.find_collection(context.scene, props)
        blocks = MS.massing_objects(coll, getattr(props, "product_collection", None)) if coll is not None else []
        root = MS.block_root(active, blocks)
    except Exception:
        return
    if root is None:
        return
    row = box.row(align=True)
    row.enabled = not busy
    row.prop(root, "wondful_hint", text="体块名称")
    if not str(getattr(root, "wondful_hint", "") or "").strip() or root.get(MS.AUTO_HINT_FLAG):
        try:
            hint, src = MS.group_hint([root])
        except Exception:
            hint, src = "", ""
        if hint and src != "custom":
            box.label(text=f"自动：{hint}"[:48], icon="BLANK1")
        elif not hint:
            box.label(text="留空：生图时按尺寸和位置自动命名", icon="BLANK1")


def _draw_block_refs(box, context, props, busy, blocks):
    """建筑参考: the active block's 1–3 reference images, then every block with its ref count."""
    from . import massing as MS
    refs_box = box.box()
    refs_box.label(text="建筑参考（真实建筑照片 / 效果图 / 概念草图）", icon="IMAGE_REFERENCE")
    active = getattr(context, "active_object", None)
    root = MS.block_root(active, blocks)
    col = refs_box.column()
    col.enabled = not busy
    if root is None:
        col.label(text="选中一个体块后可添加 1–3 张参考图", icon="INFO")
    else:
        refs = MS.get_refs(root)
        col.label(text=f"当前体块：{root.name} · {len(refs)}/{MS.MAX_BLOCK_REFS} 张", icon="OBJECT_DATA")
        for index, path in enumerate(refs):
            line = col.row(align=True)
            missing = not Path(bpy.path.abspath(path)).is_file()
            line.label(text=Path(path).name[:40], icon="ERROR" if missing else "FILE_IMAGE")
            op = line.operator("wondful.block_ref_clear", text="", icon="X")
            op.index = index
        actions = col.row(align=True)
        add = actions.row(align=True)
        add.enabled = len(refs) < MS.MAX_BLOCK_REFS
        add.operator("wondful.block_ref_add", text="添加参考图", icon="ADD")
        clear = actions.row(align=True)
        clear.enabled = bool(refs)
        clear.operator("wondful.block_ref_clear", text="清空", icon="TRASH").index = -1
    if blocks:
        lst = refs_box.column(align=True)
        for obj in blocks[:8]:
            n = len(MS.get_refs(obj))
            text = MS.block_ref_label(obj.name, n, MS.is_scene_blockout_block(obj))
            op = lst.operator("wondful.block_select", text=text,
                              icon="IMAGE_REFERENCE" if n else "MESH_CUBE", emboss=(obj == root))
            op.name = obj.name
        if len(blocks) > 8:
            lst.label(text=f"… 另有 {len(blocks) - 8} 个体块", icon="BLANK1")
    refs_box.label(text=f"每次生图最多带 {MS.MAX_BUILDING_IMAGES} 张，画面里大的体块优先", icon="BLANK1")


def _draw_massing_hull(box, props, busy):
    """从三视图生成建筑体块: visual hull of a building into the massing collection."""
    sub = box.box()
    if not _disclosure(sub.row(align=True), props, "ui_show_massing_hull", "从三视图生成建筑体块", icon="HOME"):
        return
    col = sub.column()
    col.enabled = not busy
    col.label(text="建筑立面 / 侧面 / 平面轮廓 → 灰蓝体块，三视图自动作为它的参考", icon="INFO")
    col.row(align=True).prop(props, "hull_input", expand=True)
    if getattr(props, "hull_input", "SEPARATE") == "SHEET":
        col.prop(props, "hull_sheet_path", text="拼图")
        col.label(text="三个视图之间要留白；拆不开时请改用分开的图", icon="BLANK1")
    else:
        col.prop(props, "hull_front_path", text="正视")
        col.prop(props, "hull_side_path", text="侧视")
        col.prop(props, "hull_top_path", text="俯视")
        col.label(text="白底或透明底 JPG / PNG；俯视图可不填", icon="BLANK1")
    row = col.row(align=True)
    row.label(text="侧视")
    row.prop(props, "hull_side_front", text="")
    row.label(text="俯视")
    row.prop(props, "hull_top_front", text="")
    col.row(align=True).prop(props, "hull_size_mode", expand=True)
    if getattr(props, "hull_size_mode", "AUTO") == "MANUAL":
        dims = col.row(align=True)
        dims.prop(props, "hull_width", text="宽")
        dims.prop(props, "hull_length", text="长")
        dims.prop(props, "hull_height", text="高")
    else:
        known = col.row(align=True)
        known.prop(props, "hull_known_axis", text="")
        known.prop(props, "hull_known_value", text="")
    col.prop(props, "hull_resolution", text="精度")
    build = col.row(align=True)
    build.scale_y = 1.2
    build.operator("wondful.massing_hull_build", text="从三视图生成建筑体块", icon="MOD_BUILD")
    col.label(text="选中一个占位立方体，新体块会放到它的位置并按它大小缩放", icon="INFO")
    note = getattr(props, "last_hull_note", "")
    if note:
        for index, line in enumerate(wrap(note, width=44)[:7 if "失败" in note else 3]):
            col.label(text=line, icon=("ERROR" if "失败" in note else "CHECKMARK") if index == 0 else "BLANK1")
    else:
        col.label(text="没选占位体时放在 3D 游标处、落在地面上，正立面朝 -Y", icon="BLANK1")


def _draw_terrain(layout, props):
    """1.1.0 自定义地形: presets + four product-relative corners + 坡顶 / 凹凸 + pose toggle."""
    box = layout.box()
    box.label(text="自定义地形（相对产品自身的前后左右）", icon="MOD_DISPLACE")
    for keys in (("CREST", "坡顶"), ("TWIST", "对角扭曲"), ("TWIST_MIRROR", "镜像扭曲"), ("BUMPY", "颠簸路面")), \
            (("CLIMB", "爬坡"), ("DESCENT", "下坡"), ("SIDE", "侧坡")):
        presets = box.row(align=True)
        for key, label in keys:
            presets.operator("wondful.terrain_preset", text=label).preset = key
    col = box.column(align=True)
    col.label(text="四角地面坡度（°）：正 = 该角抬高，负 = 降低")
    for pair in (("fl", "fr"), ("bl", "br")):
        row = col.row(align=True)
        for key in pair:
            row.prop(props, f"clay_terrain_{key}")
    box.prop(props, "clay_terrain_crest", text="坡顶", slider=True)
    box.prop(props, "clay_terrain_bump", text="凹凸", slider=True)
    box.prop(props, "clay_terrain_follow", text="姿态跟随地形")
    axis = box.row(align=True)
    axis.label(text="产品正面")
    axis.prop(props, "clay_terrain_front_axis", text="")


class WONDFUL_PT_main(Panel):
    bl_label = "Wondful AI 渲染器 · 1.2"
    bl_idname = "WONDFUL_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Wondful AI"

    def draw(self, context):
        """3.2.0: five step cards (连接 / 产品三视图（可选） / 环境参考图 / 提示词 / 生成); the rest in 高级."""
        from .operators import generate_needs_polish
        layout = self.layout
        props = context.scene.wondful_ai
        prefs = _prefs(context)
        busy = props.status in BUSY_STATUSES

        active_is_antigravity = props.analysis_provider == "ANTIGRAVITY"
        state = props.antigravity_account_state if active_is_antigravity else props.codex_account_state
        account_message = props.antigravity_account_message if active_is_antigravity else props.codex_account_message
        cli_version = props.antigravity_cli_version if active_is_antigravity else props.codex_cli_version
        provider_label = "Antigravity" if active_is_antigravity else "Codex"
        refresh_op = "wondful.antigravity_check" if active_is_antigravity else "wondful.codex_check"
        camera = context.scene.camera
        camera_w, camera_h = camera_output_dimensions(context)

        imagegen_enabled = prefs.codex_imagegen_enabled if not active_is_antigravity else prefs.antigravity_imagegen_enabled
        structure_ready = bool((not prefs.structure_guides_enabled) or (not prefs.strict_composition_lock) or props.product_collection)
        needs_polish = generate_needs_polish(props, prefs)

        connected = state == "LOGGED_IN"
        has_product = len(props.product_images) > 0
        has_style = len(props.style_images) > 0
        prompt_ready = bool((props.prompt or "").strip()) and not needs_polish
        can_generate = bool(not busy and connected and camera and structure_ready and imagegen_enabled)

        # ---- ① 连接 -------------------------------------------------------------
        step1, tools = _step_card(layout, 1, "连接", done=connected)
        if connected:
            _fit(tools, 4.6)
            tools.label(text=("Google" if active_is_antigravity else "ChatGPT"))
            tools.operator(refresh_op, text="", icon="FILE_REFRESH")
        else:
            account_text = {
                "UNKNOWN": "未检测",
                "NOT_INSTALLED": _t("找不到 {provider}").format(provider=provider_label),
                "LOGGED_OUT": "未登录",
                "ERROR": "异常",
            }.get(state, state)
            if active_is_antigravity and props.status == "AUTHENTICATING":
                account_text = "正在连接"
            _fit(tools, 6 if len(account_text) > 4 else 4)
            tools.label(text=account_text, icon="QUESTION" if state == "UNKNOWN" else "ERROR")
            tools.operator(refresh_op, text="", icon="FILE_REFRESH")
            connect = step1.row(align=True)
            connect.scale_y = 1.5
            connect.enabled = not busy
            if active_is_antigravity:
                connect.operator("wondful.antigravity_terminal", text="Google 登录", icon="USER")
                connect.operator("wondful.antigravity_login", text="验证登录", icon="CHECKMARK")
                step1.label(text="终端和浏览器里登录完，回来点验证登录。", icon="INFO")
            else:
                connect.operator("wondful.codex_login", text="使用 ChatGPT 登录", icon="USER")
            if state == "NOT_INSTALLED":
                step1.label(text="已装 Codex 仍找不到？在高级 › 插件偏好设置里填路径。", icon="INFO")
            elif account_message:
                step1.label(text=_short(account_message, 60), icon="BLANK1")
        if props.status == "AUTHENTICATING":
            auth = step1.box()
            if active_is_antigravity:
                for line in wrap(props.auth_detail or "正在连接 AGY", width=38):
                    auth.label(text=line)
                if props.auth_has_url:
                    auth.operator("wondful.antigravity_auth_browser", text="打开本次登录页", icon="URL")
                if props.auth_stage == "WAITING_CODE":
                    auth.prop(context.window_manager, "wondful_auth_code", text="授权码")
                    auth.operator("wondful.antigravity_submit_code", text="提交授权码", icon="CHECKMARK")
                auth.operator("wondful.antigravity_auth_cancel", text="取消连接", icon="CANCEL")
            else:
                auth.prop(props, "progress", text="等待浏览器授权", slider=True)

        # ---- ② 产品三视图 -------------------------------------------------------
        # 1.1: the 三视图 is optional — the step is done once the product collection is set.
        product_done = structure_ready and (has_product or bool(props.product_collection))
        step2, tools = _step_card(layout, 2, "产品三视图（可选）", done=product_done, ready=connected)
        _reference_header_tools(tools, props.product_images, "PRODUCT")
        if not has_product:
            _draw_reference_empty(step2, "PRODUCT", (
                "可选：放 1 张三视图（正 / 侧 / 后拼成一张）",
                "不放也能生成，产品以场景里的模型为准",
            ))
        else:
            for index, item in enumerate(props.product_images):
                if index == 0:
                    _draw_primary_reference(step2, context, item, index, "PRODUCT",
                                            caption="三视图", caption_icon="IMAGE_DATA",
                                            placeholder="造型重点（可选）", movable=False)
                else:
                    extra = step2.row(align=True)
                    extra.label(text=f"未使用 · {_short(item.image.name if item.image else '无效图片', 16)}", icon="ERROR")
                    remove = extra.operator("wondful.clear_reference", text="", icon="X")
                    remove.ref_kind, remove.item_index = "PRODUCT", index
        if not props.product_collection:
            step2.label(text="产品所在集合（锁定构图用）：" if not structure_ready else "产品所在集合：")
        target = step2.row(align=True)
        target.prop(props, "product_collection", text="", icon="OUTLINER_COLLECTION")
        if not camera:
            step2.label(text="场景里还没有活动相机", icon="ERROR")

        # ---- ③ 环境参考图 -------------------------------------------------------
        step3, tools = _step_card(layout, 3, "环境参考图", done=has_style, ready=connected and product_done)
        _reference_header_tools(tools, props.style_images, "STYLE")
        if not has_style:
            _draw_reference_empty(step3, "STYLE", (
                "决定受光、反射、阴影与氛围（可选）",
                "第一张作为主光来源，最多 8 张",
            ))
        else:
            for index, item in enumerate(props.style_images):
                if index == 0:
                    _draw_primary_reference(step3, context, item, index, "STYLE",
                                            caption="主光来源", caption_icon="LIGHT_SUN",
                                            placeholder="参考什么（可选）", movable=len(props.style_images) > 1)
                else:
                    _draw_secondary_reference(step3, context, item, index, "STYLE",
                                              caption=f"辅助 {index + 1}", caption_icon="IMAGE_DATA",
                                              placeholder="参考什么", movable=True)

        # ---- ④ 提示词 -----------------------------------------------------------
        step4, tools = _step_card(layout, 4, "提示词", done=prompt_ready, ready=connected and product_done)
        _fit(tools, 3)
        tools.label(text=f"{_prompt_char_count(props.prompt)} 字")
        brief = step4.row(align=True)
        brief.scale_y = 1.25
        if _supports_placeholder():
            brief.prop(props, "scene_brief", text="", icon="GREASEPENCIL", placeholder="一句话需求（可留空）：雪地清晨，冷色调")
        else:
            step4.label(text="一句话需求（可留空）：")
            brief = step4.row(align=True)
            brief.scale_y = 1.25
            brief.prop(props, "scene_brief", text="", icon="GREASEPENCIL")
        status_row = step4.row(align=True)
        status_row.scale_y = 0.85
        if prompt_ready:
            status_row.label(text="提示词已就绪，生成时直接使用", icon="CHECKMARK")
        else:
            status_row.label(text="点「生成」时自动读图并写提示词", icon="INFO")
        prompt_box = step4.column()
        if _disclosure(prompt_box.row(align=True), props, "ui_show_prompt", "查看 / 编辑完整提示词", icon="TEXT"):
            region_width = getattr(getattr(context, "region", None), "width", 330)
            ui_scale = getattr(getattr(context.preferences, "system", None), "ui_scale", 1.0)
            columns = max(16, int((region_width / max(.5, ui_scale) - 65) / 7.5))
            prompt_box.prop(props, "prompt_expanded", text="收起" if props.prompt_expanded else "展开全文", emboss=False)
            _draw_prompt_editor(prompt_box, props, columns)
            if props.prompt_notice:
                prompt_box.label(text=props.prompt_notice[:100], icon="INFO")
            if getattr(props, "prompt_removed_notice", ""):
                prompt_box.label(text=props.prompt_removed_notice[:100], icon="TRASH")
            polish = prompt_box.row(align=True)
            polish.enabled = bool(not busy and connected)
            polish.operator("wondful.polish_prompt", text="只重写提示词", icon="TEXT")

        # ---- ⑤ 生成 -------------------------------------------------------------
        result_done = bool(props.result_image) and not busy
        step5, tools = _step_card(layout, 5, "生成", done=result_done, ready=can_generate or busy)
        if props.two_step and not busy:
            _fit(tools, 3)
            tools.label(text="分两步")
        generating = props.status in {"POLISHING", "RENDERING"}
        if generating:
            current = _stage_index(props)
            stages = step5.column(align=True)
            stages.scale_y = 0.8
            for index, name in enumerate(GENERATE_STAGES):
                icon = "CHECKMARK" if index < current else ("PLAY" if index == current else "RADIOBUT_OFF")
                cell = stages.row(align=True)
                cell.active = index <= current
                cell.label(text=name + ("  ·  进行中" if index == current else ""), icon=icon)
            pct = max(0, min(100, int(round(float(props.progress) * 100))))
            phase = getattr(props, "task_phase", "") or GENERATE_STAGES[max(0, current)]
            step5.label(text=phase, icon="TIME")
            bar = step5.column()
            bar.scale_y = 1.4
            _draw_progress_bar(bar, props, f"估算 {pct}%")
        else:
            if props.two_step:
                go_text = "生成提示词" if needs_polish else "渲染"
            else:
                go_text = "生成"
            # 1.2.2: images per generation, right above 生成.
            count_row = step5.row(align=True)
            count_row.enabled = not busy
            count_row.label(text="张数")
            count_row.prop(props, "render_count", expand=True)
            cap = provider_variant_cap("antigravity" if active_is_antigravity else "codex")
            if int(props.render_count) > cap:
                step5.label(text=f"Antigravity 每次最多 {cap} 张，将生成 {cap} 张", icon="INFO")
            go = step5.row(align=True)
            go.scale_y = 2.0
            go.enabled = can_generate
            go.operator("wondful.generate", text=go_text, icon="RENDER_STILL")
        if busy:
            if props.status not in {"POLISHING", "RENDERING"}:
                step5.label(text=_short(getattr(props, "task_phase", "") or "后台任务进行中", 40), icon="TIME")
            cancel_row = step5.row(align=True)
            cancel_row.scale_y = 1.2
            cancel_row.operator("wondful.cancel_task", text="取消", icon="CANCEL")
        else:
            next_step = (
                "先完成 ① 连接" if not connected else
                "先在场景里设置活动相机" if not camera else
                "先在 ② 选择产品集合" if not structure_ready else
                "ImageGen 已在偏好设置里关闭" if not imagegen_enabled else "")
            if next_step:
                step5.label(text=next_step, icon="INFO")
        if getattr(props, "model_notice", ""):
            for index, line in enumerate(wrap(props.model_notice, width=42)[:3]):
                step5.label(text=line, icon="ERROR" if index == 0 else "BLANK1")
        if props.status == "ERROR" and props.last_error:
            err = step5.box()
            lines = wrap(props.last_error, width=42)[:6]
            for index, line in enumerate(lines):
                err.label(text=line, icon="ERROR" if index == 0 else "BLANK1")
            err.operator("wondful.copy_diagnostics", text="复制诊断信息", icon="COPYDOWN")

        # ---- 结果（属于 ⑤）------------------------------------------------------
        provider_variant_count = render_variant_count("antigravity" if active_is_antigravity else "codex", props)
        if props.result_image:
            result = step5.box()
            title = result.row(align=True)
            title.label(text="结果", icon="IMAGE_DATA")
            if props.alignment_score >= 0:
                ok = props.alignment_score >= prefs.alignment_threshold
                score = title.row(align=True)
                score.alignment = "RIGHT"
                _fit(score, 3.6)
                score.label(text=f"对齐 {props.alignment_score:.0f}", icon="CHECKMARK" if ok else "ERROR")
            _draw_thumbnail(result, props.result_image, _thumb_scale(context, 1.0, hi=18.0))
            exported_count = max(0, int(getattr(props, "variant_count", 0)))
            if props.status == "RENDERING" and props.last_export_path:
                result.label(text=f"已完成 {exported_count}/{provider_variant_count} 张，继续生成中", icon="TIME")
            elif exported_count > 1:
                result.label(text=f"共 {exported_count} 张，预览为最佳一张", icon="INFO")
            engine_note = getattr(props, "last_engine_note", "")
            if engine_note:
                fell_back = "改用" in engine_note
                for index, line in enumerate(wrap(engine_note, width=42)[:2]):
                    result.label(text=line, icon=("ERROR" if fell_back else "CHECKMARK") if index == 0 else "BLANK1")
            layers_note = getattr(props, "last_layers_note", "")
            if layers_note:
                result.label(text=_short(layers_note, 42), icon="ERROR" if "失败" in layers_note else "RENDERLAYERS")
            detail_note = getattr(props, "last_detail_note", "")
            if "镜像" in detail_note:
                result.label(text="车牌 / 标识在模型里是镜像的，见 高级 › 细节回贴", icon="ERROR")
            if detail_note and ("完成" in detail_note or "失败" in detail_note or "跳过" in detail_note):
                result.label(text=_short(detail_note, 42),
                             icon="ERROR" if ("失败" in detail_note or "跳过" in detail_note) else "BRUSH_DATA")
            again = result.row(align=True)
            again.scale_y = 1.4
            again.enabled = not busy
            again.operator("wondful.ai_render", text="再来一张", icon="FILE_REFRESH")
            actions = result.row(align=True)
            actions.scale_y = 1.2
            actions.operator("wondful.open_compare_workspace", text="对比", icon="ARROW_LEFTRIGHT")
            actions.operator("wondful.open_output_directory", text="打开文件夹", icon="FILE_FOLDER")

        # ---- 高级 ---------------------------------------------------------------
        layout.separator()
        advanced = layout.box()
        if not _disclosure(advanced.row(align=True), props, "ui_show_advanced", "高级", icon="PREFERENCES"):
            return
        adv = advanced.column()

        ai = adv.box()
        ai.label(text="AI 与模型", icon="TOOL_SETTINGS")
        prov = ai.row(align=True)
        prov.enabled = not busy
        prov.prop(props, "analysis_provider", expand=True)
        model_provider = "antigravity" if active_is_antigravity else "codex"
        model_field = model_provider + "_model"
        models = getattr(props, model_provider + "_models")
        model_row = ai.row(align=True)
        model_row.enabled = not busy
        if models:
            try:
                model_row.prop_search(prefs, model_field, props, model_provider + "_models", text="推理模型", results_are_suggestions=True)
            except TypeError:
                model_row.prop_search(prefs, model_field, props, model_provider + "_models", text="推理模型")
        else:
            model_row.prop(prefs, model_field, text="推理模型")
        model_row.operator("wondful.refresh_models", text="", icon="FILE_REFRESH")
        model_row.operator("wondful.model_default", text="默认")
        msg = getattr(props, model_provider + "_models_message")
        if msg:
            ai.label(text=msg[:90])
        if active_is_antigravity:
            ai.label(text="生图：AGY generate_image（由 CLI 管理）")
        else:
            ai.prop(prefs, "codex_direct_edit", text="图像编辑直连：在白模画布上直接编辑（推荐）")
            if getattr(prefs, "codex_direct_edit", True):
                ai.prop(prefs, "codex_direct_simple", text="简洁输入：只发带地面灯光的白模底图（推荐）")
                if getattr(prefs, "codex_direct_simple", True):
                    ground = ai.row(align=True)
                    ground.prop(props, "clay_ground", expand=True)
                    if getattr(props, "clay_ground", "AUTO") == "CREST":
                        ground.prop(props, "clay_crest_angle", text="下坡°")
                    elif getattr(props, "clay_ground", "AUTO") == "CUSTOM":
                        _draw_terrain(ai, props)
            ai.label(text=("生图：Codex 图像编辑（直连，失败自动改用 CLI）" if getattr(prefs, "codex_direct_edit", True)
                           else "生图：Codex ImageGen（由 CLI 管理）"))
        acct = ai.row(align=True)
        acct.operator(refresh_op, text="检测登录", icon="FILE_REFRESH")
        if connected and not active_is_antigravity:
            acct.operator("wondful.codex_logout", text="退出登录", icon="X")
        if active_is_antigravity:
            acct.operator("wondful.antigravity_terminal", text="Google 登录", icon="CONSOLE")
        if cli_version:
            ai.label(text=cli_version[:90])
        if connected and account_message:
            ai.label(text=account_message[:90])

        flow = adv.box()
        flow.label(text="流程", icon="SETTINGS")
        flow.prop(props, "two_step", text="分两步：先写提示词，确认后再渲染")
        mode_row = flow.row(align=True)
        mode_row.label(text="模式")
        mode_row.prop(props, "render_mode", expand=True)
        out_row = flow.row(align=True)
        out_row.prop(props, "output_directory", text="输出")
        out_row.operator("wondful.choose_output_directory", text="", icon="FILEBROWSER")
        if not (props.output_directory or "").strip():
            flow.label(text="留空：保存过的工程输出到 .blend 旁边的 Wondful_Renders", icon="INFO")
        if camera:
            flow.label(text=f"相机：{camera.name} · {camera_w} × {camera_h}", icon="CAMERA_DATA")
        if not prefs.structure_guides_enabled:
            flow.label(text="结构控制图已在偏好设置中关闭", icon="ERROR")

        layers = adv.box()
        layers.prop(props, "layered_output", text="分层输出", icon="RENDERLAYERS")
        if getattr(props, "layered_output", False):
            mode_row = layers.row(align=True)
            mode_row.enabled = not busy
            mode_row.prop(props, "layered_mode", expand=True)
            if getattr(props, "layered_mode", "SHADOW") == "FULL":
                layers.label(text="额外调用一次图像编辑生成干净背景，每张约多 30–50 秒", icon="TIME")
            else:
                layers.label(text="不额外调用模型：主体抠图 + 估算投影，几秒完成", icon="INFO")
            layers.label(text="存到结果图旁：主体 / 投影 / 背景 PNG + PSD", icon="FILE_IMAGE")
            if not getattr(props, "product_collection", None):
                layers.label(text="需要先在 ② 选择产品集合", icon="ERROR")
        if getattr(props, "last_layers_note", ""):
            for index, line in enumerate(wrap(props.last_layers_note, width=44)[:2]):
                layers.label(text=line, icon="CHECKMARK" if index == 0 and "失败" not in props.last_layers_note
                             else ("ERROR" if index == 0 else "BLANK1"))

        _draw_iou_gate(adv, context, props, busy)

        _draw_detail_paste(adv, context, props, busy)

        _draw_massing(adv, context, props, busy)

        _draw_multiview(adv, context, props, busy)

        lock_box = adv.box()
        if _disclosure(lock_box.row(align=True), props, "ui_show_appearance_lock", "产品外观锁定（自动填写）", icon="LOCKED"):
            lock_box.prop(props, "color_source", text="配色来源")
            lock_box.prop(props, "product_look_prompt", text="产品外观")
            lock_box.prop(props, "identity_details", text="保留细节")
            fill = lock_box.row(align=True)
            fill.enabled = has_product and not busy
            fill.operator("wondful.autofill_product_look", text="重新根据参考填写", icon="EYEDROPPER")
            if getattr(props, "product_autofill_message", ""):
                lock_box.label(text=props.product_autofill_message, icon="CHECKMARK")
            else:
                lock_box.label(text="留空即可，生成时会根据产品参考自动填写。", icon="INFO")
        if getattr(props, "style_summary_cache", "") and has_style:
            summary_box = adv.box()
            if _disclosure(summary_box.row(align=True), props, "ui_show_style_summary", "环境分析（可编辑）", icon="LIGHT_SUN"):
                summary_box.prop(props, "style_summary_cache", text="")

        more_refs = adv.box()
        _draw_reference_group(more_refs, props, "人物参考", "person_images", "person_image_index", "PERSON", "ui_show_person_refs")
        total = len(props.product_images) + len(props.person_images) + len(props.style_images)
        if total > 0 and not busy:
            more_refs.operator("wondful.auto_classify_references", text="Jev 语义整理参考图", icon="LIGHT")
        if getattr(props, "jev_status_message", ""):
            more_refs.label(text=props.jev_status_message[:100], icon="INFO")

        if props.result_image:
            info = adv.box()
            info.label(text="上次渲染", icon="INFO")
            info.template_ID_preview(props, "result_image", rows=4, cols=7)
            if props.last_export_path:
                info.label(text=f"输出：{props.last_export_path[:100]}")
            if props.alignment_attempts:
                info.label(text=f"尝试 {props.alignment_attempts} 次 · Session {props.last_session_id}")
            if props.last_canvas_warning:
                for line in wrap(props.last_canvas_warning, width=44)[:3]:
                    info.label(text=line)
            iou_note = getattr(props, "last_iou_note", "")
            if iou_note:
                info.label(text="轮廓验收（固定坐标）", icon="CHECKMARK" if "未通过" not in iou_note and "待复核" not in iou_note else "ERROR")
                for line in wrap(iou_note, width=44)[:3]:
                    info.label(text=line, icon="BLANK1")
            score_note = getattr(props, "last_score_note", "")
            if score_note:
                info.label(text="产品还原度（按输出像素）", icon="CHECKMARK")
                for line in wrap(score_note, width=44)[:3]:
                    info.label(text=line, icon="BLANK1")
            if props.last_structure_note:
                info.label(text=props.last_structure_note[:100])
            if getattr(props, "last_engine_note", ""):
                info.label(text=props.last_engine_note[:100])
            if props.last_total_seconds > 0:
                info.label(text=f"总用时 {props.last_total_seconds:.0f}s · 生图 {props.last_generation_seconds:.0f}s · 验收 {props.last_audit_seconds:.0f}s")
            if getattr(props, "active_conversation_id", ""):
                info.operator("wondful.copy_conversation_id", text="复制远程会话恢复命令", icon="COPYDOWN")

        tools = adv.row(align=True)
        tools.operator("wondful.copy_diagnostics", text="复制诊断信息", icon="COPYDOWN")
        tools.operator("preferences.addon_show", text="插件偏好设置").module = __package__


def _supports_placeholder():
    """UILayout.prop gained ``placeholder`` in Blender 4.1+; feature-detect once."""
    global _PLACEHOLDER_OK
    if _PLACEHOLDER_OK is None:
        try:
            params = bpy.types.UILayout.bl_rna.functions["prop"].parameters
            _PLACEHOLDER_OK = "placeholder" in params.keys()
        except Exception:
            _PLACEHOLDER_OK = False
    return _PLACEHOLDER_OK


_PLACEHOLDER_OK = None
