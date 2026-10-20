from __future__ import annotations

from pathlib import Path

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


class WONDFUL_UL_render_views(UIList):
    """1.2.3 ⑥ 渲染视角: one row per 「相机 N」 (the 高级 › 多角度 list): ☑ + camera; the first checked row is 主视角."""
    bl_idname = "WONDFUL_UL_render_views"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        scene = context.scene
        master = _first_checked_row(data, scene)
        cam = item.camera
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        if cam is None:
            row.alert = True
            row.prop(item, "camera", text="")
            return
        present = cam.name in scene.objects
        row.alert = not present
        row.label(text=cam.name if present else f"{cam.name}（不在场景里）", icon="OUTLINER_OB_CAMERA")
        if index == master:
            tag = row.row(align=True)
            tag.alignment = "RIGHT"
            tag.label(text="主视角", icon="SOLO_ON")


def _first_checked_row(props, scene):
    """Index of the 主视角 row (first checked camera present in the scene, repeats ignored), or -1."""
    for i, item in enumerate(getattr(props, "mv_cameras", []) or []):
        cam = item.camera
        if item.enabled and cam is not None and getattr(cam, "type", "") == "CAMERA" and cam.name in scene.objects:
            return i
    return -1


def _draw_render_views(layout, context, props, busy, cells):
    """⑥ 渲染视角 (only once ④ built a 场景粗模): the shared camera list with ☑ per row, ↑ ↓ ✕, 拖入相机 /
    拖入集合 fields (drag from the Outliner), 从当前视图新建相机. Returns the checked cameras (list order)."""
    from .operators import generate_views
    views = generate_views(context, props)
    box = layout.box()
    box.enabled = not busy
    head = box.row(align=True)
    head.label(text="渲染视角", icon="OUTLINER_OB_CAMERA")
    count = head.row(align=True)
    count.alignment = "RIGHT"
    count.label(text=f"已选 {len(views)} 个" if views else "当前相机")
    rows = len(props.mv_cameras)
    if rows:
        body = box.row()
        body.template_list("WONDFUL_UL_render_views", "", props, "mv_cameras", props, "mv_camera_index",
                           rows=min(max(rows, 2), 5))
        side = body.column(align=True)
        side.operator("wondful.mv_camera_move", text="", icon="TRIA_UP").direction = "UP"
        side.operator("wondful.mv_camera_move", text="", icon="TRIA_DOWN").direction = "DOWN"
        side.separator()
        rm = side.operator("wondful.mv_camera_remove", text="", icon="X")
        rm.index = int(props.mv_camera_index)
    drop = box.column(align=True)
    drop.prop(props, "mv_drop_camera", text="拖入相机", icon="OUTLINER_OB_CAMERA")
    drop.prop(props, "mv_drop_collection", text="拖入集合", icon="OUTLINER_COLLECTION")
    box.operator("wondful.mv_camera_from_view", text="从当前视图新建相机", icon="VIEW_CAMERA")
    note = getattr(props, "mv_drop_note", "")
    if note:
        _note(box, note, cells, "info" if ("没有相机" in note or "已在列表" in note) else "ok", 1)
    if not views:
        _hint(box, "从大纲视图把相机拖进来并勾选；不选 = 用当前场景相机", cells)
    elif len(views) == 1:
        _hint(box, f"只渲染 {views[0].name}（和单视角一样）", cells)
    else:
        _hint(box, f"先渲染主视角 {views[0].name}，其余 {len(views) - 1} 个按场景粗模保持一致", cells)
    return views


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
# 3.2.0 step card layout helpers (1.2.3: six steps, ④ 场景粗模 optional)
# ---------------------------------------------------------------------------

STEP_NUMBERS = ("①", "②", "③", "④", "⑤", "⑥")
GENERATE_STAGES = ("读产品图", "分析环境", "写提示词", "渲染验收")
BUSY_STATUSES = {"AUTHENTICATING", "CLASSIFYING", "POLISHING", "RENDERING", "REFRESHING_MODELS", "BLOCKOUT"}


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


# ---------------------------------------------------------------------------
# 1.2.3 UI tidy: one text / header style for the whole panel
#   level 0  step card ①–⑥            box · number (✓ when done) · title · right-hand tools
#   level 1  高级                      box · disclosure · PREFERENCES · tab row (one group at a time)
#   level 2  section inside a tab      box · icon + title, or switch (checkbox) + icon + title
#   level 3  tool inside a section     nested box · disclosure · icon + title
#   hints    ≤ 1 INFO line per section (long explanations live in the tooltips)
#   notes    ✓ CHECKMARK done · ERROR failed · INFO advice · TIME running; wrapped by display cells
# ---------------------------------------------------------------------------

SEP = " · "
ADV_TABS = ("MODEL", "OUTPUT", "PRODUCT", "SCENE", "MULTIVIEW", "DIAG")
NOTE_ICONS = {"ok": "CHECKMARK", "fail": "ERROR", "info": "INFO", "run": "TIME", "plain": "NONE"}
_BREAK_AFTER = set(" ，。；：、）」》)/·→=")
_NO_LINE_START = set("，。；：、）」》)·")


def _cells(char):
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in "WF" else 1


def _width(text):
    return sum(_cells(c) for c in text or "")


def _wrap(text, cells):
    """Wrap by display cells (CJK = 2). Breaks after CJK characters, spaces and punctuation, so Latin
    words, paths and numbers are never cut in the middle; closing punctuation never starts a line."""
    cells = max(8, int(cells))
    out = []
    for para in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line, brk = "", 0
        for ch in para:
            if line and _width(line) + _cells(ch) > cells and ch not in _NO_LINE_START and ch != " ":
                cut = brk if 0 < brk < len(line) else len(line)
                out.append(line[:cut].rstrip())
                line = line[cut:].lstrip()
                brk = 0
                for i, c in enumerate(line):
                    if c in _BREAK_AFTER or _cells(c) == 2:
                        brk = i + 1
            if not line and ch == " ":
                continue
            line += ch
            if ch in _BREAK_AFTER or _cells(ch) == 2:
                brk = len(line)
        out.append(line)
    if len(out) > 1 and _width(out[-1]) < cells // 4:
        # avoid a one-word last line: re-wrap the whole text into balanced lines
        total = _width(text)
        balanced = _wrap(text, -(-total // len(out)) + 4) if -(-total // len(out)) + 4 < cells else out
        if len(balanced) == len(out):
            return balanced
    return out or [""]


def _cols(context, depth=2):
    """Display cells that fit one label at this nesting depth (box padding + icon taken off)."""
    region = getattr(context, "region", None)
    width = float(getattr(region, "width", 0) or 330)
    ui_scale = float(getattr(getattr(getattr(context, "preferences", None), "system", None), "ui_scale", 1.0) or 1.0)
    usable = width / max(0.5, ui_scale) - 16.0 - 10.0 * depth - 20.0
    return max(14, int(usable / 5.6))      # one cell ≈ half an 11 px CJK glyph


def _lines(layout, text, icon, cells, max_lines=3):
    """Draw ``text`` wrapped to ``cells``; the icon only on the first line, the rest aligned under it."""
    lines = _wrap(text, cells)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and _width(last) + 2 > cells:
            last = last[:-1]
        lines[-1] = last.rstrip() + "…"
    col = layout.column(align=True)
    for index, line in enumerate(lines):
        col.label(text=line, icon=icon if index == 0 else ("NONE" if icon == "NONE" else "BLANK1"))
    return col


def _note(layout, text, cells, state="ok", max_lines=3):
    """A status note: ✓ done · ERROR failed · INFO advice · TIME running · plain."""
    if text:
        _lines(layout, text, NOTE_ICONS.get(state, "INFO"), cells, max_lines)


def _hint(layout, text, cells, max_lines=2):
    """The one short explanation a section may carry (INFO)."""
    if text:
        _lines(layout, text, "INFO", cells, max_lines)


def _state(note, fail=("失败",), info=()):
    """ok / fail / info for a result note, by the words the pipeline writes into it."""
    note = note or ""
    if any(word in note for word in fail):
        return "fail"
    if any(word in note for word in info):
        return "info"
    return "ok"


def _section(layout, title, icon, props=None, toggle=None, *, enabled=True):
    """Level 2: a section inside an 高级 tab. Returns (box, header_row, on)."""
    box = layout.box()
    head = box.row(align=True)
    if toggle:
        switch = head.row(align=True)
        switch.enabled = bool(enabled)
        switch.prop(props, toggle, text=title, icon=icon)
        return box, head, bool(getattr(props, toggle, False))
    head.label(text=title, icon=icon)
    return box, head, True


def _subsection(layout, props, attr, title, icon):
    """Level 3: a tool inside a section (nested box with a disclosure). Returns (box, opened)."""
    box = layout.box()
    return box, _disclosure(box.row(align=True), props, attr, title, icon=icon)


def _draw_task_progress(layout, props, cells):
    """⑥ while a side task runs: 场景粗模 (④) as one compact line (its stages + bar are in ④), 多角度 (高级)
    as what it does now + a bar; visible whatever 高级 tab is open."""
    pct = max(0, min(100, int(round(float(props.progress) * 100))))
    phase = getattr(props, "task_phase", "") or ""
    if props.status == "BLOCKOUT":
        from . import sb_progress as SP
        stage = getattr(props, "sb_stage", "") or SP.READ
        text = ("④ 场景粗模" + SEP + "正在取消…" if "取消" in phase
                else "④ 场景粗模" + SEP + SP.status_text(stage, getattr(props, "sb_elapsed", 0)) + f"{SEP}{pct}%")
        _note(layout, text, cells, "run", 1)
        return
    _note(layout, phase or "后台任务进行中", cells, "run", 2)
    bar = layout.column()
    bar.scale_y = 1.4
    _draw_progress_bar(bar, props, f"估算 {pct}%")


def _view_phase(index, total, name, master=False):
    from .multiview import view_phase
    return view_phase(index, total, name, master)


def _draw_view_flip(layout, props):
    """⑥ result card after a multi-view run: 〈 视角 i/N · 名称 〉."""
    import json
    try:
        views = json.loads(getattr(props, "view_results", "") or "[]")
    except ValueError:
        views = []
    if len(views) < 2:
        return
    i = max(0, min(int(props.view_result_index), len(views) - 1))
    row = layout.row(align=True)
    row.operator("wondful.view_result_step", text="", icon="TRIA_LEFT").step = -1
    mid = row.row(align=True)
    mid.alignment = "CENTER"
    mid.label(text=_view_phase(i + 1, len(views), views[i].get("camera", ""), bool(views[i].get("master"))))
    row.operator("wondful.view_result_step", text="", icon="TRIA_RIGHT").step = 1


# ---------------------------------------------------------------------------
# 高级 › 产品
# ---------------------------------------------------------------------------

def _draw_iou_gate(adv, context, props, busy):
    """高级 › 产品 › 轮廓验收 (1.2.3): fixed-coordinate IoU ≥ 0.95, bounded regeneration (≤ 3)."""
    box, _head, on = _section(adv, "轮廓验收", "MOD_MASK", props, "iou_gate", enabled=not busy)
    cells = _cols(context, 2)
    if on:
        row = box.row(align=True)
        row.enabled = not busy
        row.prop(props, "iou_gate_max", text="最多生成次数")
        _hint(box, "IoU ≥ 0.95，固定坐标；不过线自动重新生成", cells)
        try:    # 1.2.5: the preference is the hard cap — say so when it is lower than this setting
            cap = int(context.preferences.addons[__package__].preferences.alignment_max_attempts)
        except Exception:
            cap = 0
        if 0 < cap < int(getattr(props, "iou_gate_max", 3) or 3):
            _hint(box, f"偏好设置「最多生成次数」= {cap}，实际每张最多 {cap} 次", cells)
    note = getattr(props, "last_iou_note", "")
    if note:
        ok = "通过（" in note and "未通过" not in note
        _note(box, note, cells, "ok" if ok else ("info" if "待复核" in note else "fail"))


def _draw_detail_paste(adv, context, props, busy):
    """高级 › 产品 › 细节修复 (1.2.3, was 细节回贴): toggle, Blender-paste fallback strength + 按材质校色,
    「标为原样保留」 on selected parts."""
    box, _head, on = _section(adv, "细节修复", "BRUSH_DATA", props, "detail_paste")
    cells = _cols(context, 2)
    if on:
        _hint(box, "先检查车牌 / 标识 / 灯 / 轮毂，不合格的只放大该处重新生成；无法调用生图时按下面的强度改用 Blender 回贴", cells)
        row = box.row(align=True)
        row.enabled = not busy
        row.prop(props, "detail_paste_strength", text="回贴强度", slider=True)
        row = box.row(align=True)
        row.enabled = not busy
        row.prop(props, "detail_colour_fix", text="按材质校色")
        if getattr(props, "detail_colour_fix", True):
            row.prop(props, "detail_colour_strength", text="强度", slider=True)
        coll = getattr(props, "product_collection", None)
        if coll is None:
            _hint(box, "需要先在 ② 选择产品集合", cells)
        else:
            try:
                from .viewport_capture import keep_exact_slots
                slots = keep_exact_slots(coll)
                objs = {o for (o, _i) in slots}
                auto = {o for (o, _i), k in slots.items() if k == "auto"} - {o for (o, _i), k in slots.items() if k == "explicit"}
                text = (f"原样保留：{len(objs)} 个部件" + (f"（自动识别 {len(auto)}）" if auto else "")) if objs \
                    else "原样保留：无 · 可选中车牌 / 标识后标记"
            except Exception:
                text = "原样保留：—"
            _lines(box, text, "LOCKED", cells, 2)
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
        _note(box, main, cells, _state(main, info=("跳过", "已放弃", "无可用生图")), 2)
        if warn:  # 1.2.2: a plate / logo that reads mirrored in the model itself
            _note(box, "注意：" + warn, cells, "info", 4)


def _draw_appearance_lock(adv, context, props, busy, has_product):
    """高级 › 产品 › 产品外观锁定 (filled automatically from the product reference)."""
    box, _head, _on = _section(adv, "产品外观锁定", "MATERIAL")
    cells = _cols(context, 2)
    box.prop(props, "color_source", text="配色来源")
    box.prop(props, "product_look_prompt", text="产品外观")
    box.prop(props, "identity_details", text="保留细节")
    fill = box.row(align=True)
    fill.enabled = has_product and not busy
    fill.operator("wondful.autofill_product_look", text="按产品参考重新填写", icon="EYEDROPPER")
    if getattr(props, "product_autofill_message", ""):
        _note(box, props.product_autofill_message, cells, "ok", 2)
    else:
        _hint(box, "留空即可，生成时按产品参考自动填写", cells)


# ---------------------------------------------------------------------------
# 高级 › 场景
# ---------------------------------------------------------------------------

def _draw_ground(adv, context, props, prefs, is_antigravity):
    """高级 › 场景 › 地面 (was hidden inside 「AI 与模型」): same conditions as before, only moved."""
    box, _head, _on = _section(adv, "地面", "MESH_PLANE")
    cells = _cols(context, 2)
    used = (not is_antigravity) and getattr(prefs, "codex_direct_edit", True) and getattr(prefs, "codex_direct_simple", True)
    if not used:
        _hint(box, "只在 模型 › 图像编辑直连 + 简洁输入 时使用", cells)
        return
    ground = box.row(align=True)
    ground.prop(props, "clay_ground", expand=True)
    if getattr(props, "clay_ground", "AUTO") == "CREST":
        ground.prop(props, "clay_crest_angle", text="下坡 °")
    elif getattr(props, "clay_ground", "AUTO") == "CUSTOM":
        _draw_terrain(box, props)


def _draw_terrain(layout, props):
    """1.1.0 自定义地形: presets + four product-relative corners + 坡顶 / 凹凸 + pose toggle."""
    box, opened = _subsection(layout, props, "ui_show_terrain", "自定义地形", "MOD_DISPLACE")
    if not opened:
        return
    for keys in (("CREST", "坡顶"), ("TWIST", "对角扭曲"), ("TWIST_MIRROR", "镜像扭曲"), ("BUMPY", "颠簸路面")), \
            (("CLIMB", "爬坡"), ("DESCENT", "下坡"), ("SIDE", "侧坡")):
        presets = box.row(align=True)
        for key, label in keys:
            presets.operator("wondful.terrain_preset", text=label).preset = key
    col = box.column(align=True)
    col.label(text="四角坡度 (°)：正 = 抬高 · 负 = 降低", icon="INFO")
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


def _draw_massing(adv, context, props, busy):
    """高级 › 场景 › 环境体块 (1.2.0): collection, per-block 建筑参考, blocks from three views
    (1.2.3: 从参考图生成场景粗模 moved to the main flow as ④)."""
    box, _head, on = _section(adv, "环境体块", "MESH_CUBE", props, "massing_enabled")
    if not on:
        return
    cells = _cols(context, 2)
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
        _note(box, f"{how}：{coll.name}{SEP}{len(blocks)} 个体块", cells, "ok", 2)
        _hint(box, "白模里显示为灰蓝色，AI 画成真实建筑", cells)
    elif getattr(props, "massing_collection", None) is not None:
        _note(box, "该集合与产品集合重叠，已忽略", cells, "info")
    else:
        _hint(box, "未找到 · 集合名含「环境」「体块」即自动识别", cells)
    if getattr(props, "last_massing_note", ""):
        _note(box, props.last_massing_note, cells, "plain", 2)
    if coll is not None:
        _draw_block_refs(box, context, props, busy, blocks)
    _draw_massing_hull(box, props, busy)


def _draw_block_refs(box, context, props, busy, blocks):
    """建筑参考: the active block's 1–3 reference images, then every block with its ref count."""
    from . import massing as MS
    refs_box, opened = _subsection(box, props, "ui_show_block_refs", "建筑参考", "IMAGE_REFERENCE")
    if not opened:
        return
    cells = _cols(context, 3)
    active = getattr(context, "active_object", None)
    root = MS.block_root(active, blocks)
    col = refs_box.column()
    col.enabled = not busy
    if root is None:
        _hint(col, "选中一个体块，添加 1–3 张建筑照片 / 效果图 / 草图", cells)
    else:
        refs = MS.get_refs(root)
        _lines(col, f"当前体块：{root.name}{SEP}{len(refs)}/{MS.MAX_BLOCK_REFS} 张", "OBJECT_DATA", cells, 2)
        for index, path in enumerate(refs):
            line = col.row(align=True)
            missing = not Path(bpy.path.abspath(path)).is_file()
            line.label(text=_short(Path(path).name, 36), icon="ERROR" if missing else "FILE_IMAGE")
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
            lst.label(text=f"另有 {len(blocks) - 8} 个体块…", icon="BLANK1")
    _note(refs_box, f"每次生图最多带 {MS.MAX_BUILDING_IMAGES} 张，画面里大的体块优先", cells, "plain", 2)


def _draw_massing_hull(box, props, busy):
    """从三视图生成建筑体块: visual hull of a building into the massing collection."""
    sub, opened = _subsection(box, props, "ui_show_massing_hull", "从三视图生成建筑体块", "HOME")
    if not opened:
        return
    cells = _cols(bpy.context, 3)
    col = sub.column()
    col.enabled = not busy
    col.row(align=True).prop(props, "hull_input", expand=True)
    if getattr(props, "hull_input", "SEPARATE") == "SHEET":
        col.prop(props, "hull_sheet_path", text="拼图")
    else:
        col.prop(props, "hull_front_path", text="正视")
        col.prop(props, "hull_side_path", text="侧视")
        col.prop(props, "hull_top_path", text="俯视")
    col.prop(props, "hull_side_front", text="侧视朝向")
    col.prop(props, "hull_top_front", text="俯视朝向")
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
    _hint(col, "选中占位立方体：新体块放到它的位置并按它缩放", cells)
    build = col.row(align=True)
    build.scale_y = 1.2
    build.operator("wondful.massing_hull_build", text="从三视图生成建筑体块", icon="MOD_BUILD")
    note = getattr(props, "last_hull_note", "")
    if note:
        _note(col, note, cells, _state(note), 5 if "失败" in note else 3)


def _scene_blockout_exists(context):
    """④ is done once a 「从参考图生成场景粗模」 blockout is in the scene: its tagged collection
    (the same marker as the 「· 来自场景参考图」 block label) still holds objects."""
    from .multiview import scene_blockout_exists
    return scene_blockout_exists(context.scene)


def _sb_first_reference_name(props):
    try:
        items = list(getattr(props, "style_images", []) or [])
        if items:
            return Path(getattr(items[0], "source_path", "") or getattr(getattr(items[0], "image", None), "filepath", "")
                        or "").name or getattr(getattr(items[0], "image", None), "name", "") or "③"
    except Exception:
        pass
    return ""


def _draw_blockout_step(layout, context, props, busy, connected, is_antigravity, cells):
    """④ 场景粗模（可选）(1.2.3, was 高级 › 场景 › 环境体块): one environment photo → a whole blockout
    (Wondful 场景粗模) + optional camera. Compact: hint · 参考图 · two toggles · one button; stages + bar while running."""
    from . import sb_progress as SP
    running = props.status == "BLOCKOUT"
    done = (not running) and _scene_blockout_exists(context)
    card, tools = _step_card(layout, 4, "场景粗模", done=done, ready=connected)
    _fit(tools, 2.4)
    tools.label(text="可选")
    custom = (getattr(props, "sb_image_path", "") or "").strip()
    first = _sb_first_reference_name(props)
    col = card.column()
    col.enabled = not busy
    if not running:
        _hint(col, "用 ③ 的参考图搭灰模和相机，约 3–5 分钟", cells, 1)
    if _supports_placeholder():
        col.prop(props, "sb_image_path", text="参考图", placeholder="留空 = 用 ③ 第一张")
    else:
        col.prop(props, "sb_image_path", text="参考图")
        if not custom and first:
            _note(col, "留空 = 用 ③ 第一张", cells, "plain", 1)
    if not custom and not first and not running:
        _note(col, "③ 还没有环境参考图：先放一张，或在这里选图", cells, "info", 2)
    row = col.row(align=True)
    row.prop(props, "sb_set_camera", text="同时设置相机")
    row.prop(props, "sb_correct", text="自动校正一轮")
    # 1.2.5 精确模式 (default off): local depth model measures camera + distances
    row = col.row(align=True)
    row.prop(props, "sb_precise", text="精确模式（本机测深度）")
    # 1.2.5 场景补全 (default on): the blockout continues the street the photo does not show
    row.prop(props, "sb_complete", text="场景补全")
    if getattr(props, "sb_precise", False) and not running:
        from . import precise_setup as PS
        if not PS.cached_status()["ready"]:
            _hint(col, f"首次使用下载约 {PS.APPROX_MB} MB；参考图不上传", cells, 1)
    failed = (getattr(props, "sb_failed_stage", "") or "") if not running else ""
    if running or failed:
        _draw_sb_progress(card, props, running, failed, SP)
    if running:
        # the only live control while it runs: 取消 takes the build button's place
        cancel = card.row(align=True)
        cancel.scale_y = 1.3
        cancel.operator("wondful.cancel_task", text="取消", icon="CANCEL")
        return
    build = card.row(align=True)
    build.scale_y = 1.3
    build.enabled = bool(custom or first) and not busy
    build.operator("wondful.scene_blockout_build", text="生成场景粗模", icon="MOD_BUILD")
    if is_antigravity:
        _hint(card, "需要 Codex 登录（高级 › 模型 › AI 服务）", cells, 1)
    note = getattr(props, "last_sb_note", "")
    if note:
        _note(card, note, cells, _state(note, fail=("失败", "不存在"), info=("请先", "需要", "取消")), 4)
    elif done:
        _note(card, "已有场景粗模；重新生成会替换它", cells, "plain", 1)


def _draw_sb_progress(layout, props, running, failed, SP):
    """④ running: 「分析参考图 · 已用 1:42」 + typical-time hint + bar 「第 2/5 步 · 24%」 (compact, no stage list);
    failed: the stage list with the failed stage marked."""
    ids = SP.decode(getattr(props, "sb_stage_ids", ""))
    stage = getattr(props, "sb_stage", "") or SP.READ
    elapsed = int(getattr(props, "sb_elapsed", 0) or 0)
    if not running:
        stages = layout.column(align=True)
        stages.scale_y = 0.8
        for name, icon, active in SP.row_icons(ids, stage, failed):
            cell = stages.row(align=True)
            cell.active = active
            cell.label(text=name + (SEP + "失败" if icon == "ERROR" else ""), icon=icon)
        return
    phase = getattr(props, "task_phase", "") or ""
    if "取消" in phase:
        layout.label(text="正在取消，等待当前请求结束…", icon="TIME")
    else:
        layout.label(text=SP.status_text(stage, elapsed), icon="TIME")
    tip = SP.hint(stage)
    if tip:
        _hint(layout, tip, _cols(bpy.context, 2))
    bar = layout.column()
    bar.scale_y = 1.2
    pct = max(0, min(100, int(round(float(props.progress) * 100))))
    i = SP.index(ids, stage)
    step = f"第 {i + 1}/{len(ids)} 步{SEP}" if i >= 0 and ids else ""
    _draw_progress_bar(bar, props, f"{step}{pct}%")


def _draw_style_summary(adv, context, props):
    """高级 › 场景 › 环境分析 (editable; used the next time the prompt is written)."""
    box, _head, _on = _section(adv, "环境分析", "LIGHT_SUN")
    box.prop(props, "style_summary_cache", text="")
    _hint(box, "可直接修改，下次写提示词时使用", _cols(context, 2))


def _draw_person_refs(adv, context, props, busy):
    """高级 › 场景 › 人物参考: same cards and tools as ③; Jev sorts all references by content."""
    box, head, _on = _section(adv, "人物参考", "ARMATURE_DATA")
    cells = _cols(context, 2)
    images = props.person_images
    tools = head.row(align=True)
    tools.alignment = "RIGHT"
    _reference_header_tools(tools, images, "PERSON")
    if not images:
        _draw_reference_empty(box, "PERSON", ("可选 · 第一张作为主参考，最多 8 张",))
    else:
        for index, item in enumerate(images):
            _draw_secondary_reference(box, context, item, index, "PERSON",
                                      caption=f"人物 {index + 1}" + ("（主参考）" if index == 0 else ""),
                                      caption_icon="IMAGE_DATA", placeholder="参考什么", movable=True)
    total = len(props.product_images) + len(props.person_images) + len(props.style_images)
    if total > 0 and not busy:
        box.operator("wondful.auto_classify_references", text="按内容整理参考图（Jev）", icon="SORTALPHA")
    if getattr(props, "jev_status_message", ""):
        _note(box, props.jev_status_message, cells, "info", 2)


# ---------------------------------------------------------------------------
# 高级 › 成片
# ---------------------------------------------------------------------------

def _draw_photo_finish(adv, context, props, busy):
    """高级 › 成片 › 照片质感 (1.2.4): 关 / 细微 / 标准, the last step on the final image."""
    box, _head, _on = _section(adv, "照片质感", "IMAGE_DATA")
    cells = _cols(context, 2)
    row = box.row(align=True)
    row.enabled = not busy
    row.prop(props, "photo_finish", expand=True)
    level = getattr(props, "photo_finish", "SUBTLE")
    if level == "OFF":
        _hint(box, "不加后期，输出与以前相同", cells)
    else:
        _hint(box, "高光滚降 · 环境饱和度 · 暗角 · 色差 · 颗粒；产品轮廓和颜色不变", cells)
    note = getattr(props, "last_finish_note", "")
    if note:
        _note(box, note, cells, _state(note), 2)


# ---------------------------------------------------------------------------
# 高级 › 多角度
# ---------------------------------------------------------------------------

def _draw_multiview(adv, context, props, busy):
    """高级 › 多角度 › 多角度场景一致 (1.2.3): master camera + its accepted image, render the other cameras."""
    box, _head, _on = _section(adv, "多角度场景一致", "OUTLINER_OB_CAMERA")
    cells = _cols(context, 2)
    row = box.row(align=True)
    row.enabled = not busy
    row.prop(props, "mv_master_camera", text="主视角")
    row.operator("wondful.mv_set_master", text="设为主视角", icon="CAMERA_DATA")
    row = box.row(align=True)
    row.enabled = not busy
    row.prop(props, "mv_master_image", text="主视角图")
    master = getattr(props, "mv_master_camera", None)
    try:
        from .operators import mv_pick
        picked = mv_pick(context, props, master)
    except Exception:
        picked = {"cameras": [], "source": "list", "missing": [], "master_rows": [], "duplicate_rows": [],
                  "disabled_rows": []}
    others = picked["cameras"]
    _draw_mv_cameras(box, context, props, busy, picked)
    _draw_mv_block_name(box, context, props, busy)
    row = box.row(align=True)
    row.enabled = not busy
    row.prop(props, "mv_widen", text="加宽体块")
    row.prop(props, "mv_env_check", text="环境一致性检查")
    row = box.row(align=True)
    row.enabled = not busy
    row.prop(props, "mv_colour_match", text="统一调色")
    # 1.2.5 细节粗模 (needs ④ 精确模式)
    row = box.row(align=True)
    row.enabled = not busy
    row.prop(props, "sb_detail_carrier", text="细节粗模投影")
    sub = row.row(align=True)
    sub.enabled = not busy and bool(getattr(props, "sb_detail_carrier", True))
    sub.prop(props, "sb_detail_clay", text="白模也用")
    sub.prop(props, "sb_detail_skyline", text="远山剪影")
    if master is None:
        _hint(box, "先用 ⑥ 渲染满意的一张，再点「设为主视角」", cells)
    else:
        _hint(box, "每个相机 1 次生图；产品超标或 low 画质档最多重试 1 次", cells)
    act = box.row(align=True)
    act.scale_y = 1.2
    act.enabled = bool(master is not None and others) and not busy
    act.operator("wondful.multiview_render", text=f"渲染 {len(others)} 个相机", icon="RENDER_STILL")
    note = getattr(props, "last_mv_note", "")
    if note:
        _note(box, note, cells, _state(note, fail=("失败", "未完成"), info=("超标", "注意", "生成中")), 5)


def _draw_mv_cameras(box, context, props, busy, picked):
    """「相机 1/2/3…」 rows (camera picker + 启用 + ✕), 添加相机槽 / 从当前视图新建相机; no rows = selected cameras."""
    cells = _cols(context, 2)
    cams = box.column(align=True)
    cams.enabled = not busy
    slots = list(getattr(props, "mv_cameras", []) or [])
    why = {}
    for key, text in (("missing", "未选相机或相机已不在场景里"), ("master_rows", "与主视角相同，跳过"),
                      ("duplicate_rows", "与上面重复，跳过")):
        for n in picked.get(key, []):
            why[n] = text
    for index, item in enumerate(slots):
        row = cams.row(align=True)
        n = index + 1
        row.alert = n in picked.get("missing", [])
        row.label(text=f"相机 {n}")
        row.prop(item, "camera", text="")
        row.prop(item, "enabled", text="启用")
        row.operator("wondful.mv_camera_remove", text="", icon="X").index = index
        if n in why:
            cams.label(text=f"相机 {n}：{why[n]}", icon="ERROR" if n in picked.get("missing", []) else "INFO")
    add = cams.row(align=True)
    add.operator("wondful.mv_camera_add", text="添加相机槽", icon="ADD")
    add.operator("wondful.mv_camera_from_view", text="从当前视图新建相机", icon="VIEW_CAMERA")
    count = len(picked.get("cameras", []))
    if picked.get("source") == "selected":
        _note(box, f"列表为空：用视图里选中的相机（{count} 个）" + ("，或添加相机槽" if not count else ""),
              cells, "ok" if count else "info", 2)
    elif not count:
        _hint(box, "在「相机 N」里选相机并勾选启用", cells)


def _draw_mv_block_name(box, context, props, busy):
    """多角度场景一致 › the active block's name as the prompt uses it (editable; empty = automatic)."""
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
            _note(box, f"自动：{hint}", _cols(context, 2), "plain", 1)
        elif not hint:
            _note(box, "留空：生图时按尺寸和位置自动命名", _cols(context, 2), "plain", 1)


# ---------------------------------------------------------------------------
# 更新 (1.2.3): compact banner on top of the main panel + 高级 › 诊断 › 更新
# ---------------------------------------------------------------------------

def _draw_update_banner(layout, context):
    """「发现新版本 X.Y.Z」 [更新] [忽略此版本]; while installing the stage + bar; then the result."""
    from . import update_ops as UO
    manifest = UO.banner_manifest()
    if not manifest:
        return
    phase = UO.STATE["phase"]
    cells = _cols(context, 2)
    box = layout.box()
    if phase in ("DOWNLOADING", "VERIFYING", "INSTALLING"):
        _note(box, f"更新到 {manifest['version']}{SEP}{UO.STATE['message']}", cells, "run", 1)
        bar = box.column()
        bar.progress(factor=float(UO.STATE["progress"]), type="BAR", text=UO.STATE["message"])
        return
    if phase == "DONE":
        _note(box, UO.STATE["message"], cells, "ok", 2)
        for line in UO.STATE.get("notes") or ():
            _note(box, line, cells, "plain", 2)
        return
    if phase == "FAILED":
        _note(box, UO.STATE["message"], cells, "fail", 4)
        row = box.row(align=True)
        row.operator("wondful.update_install", text="重试", icon="FILE_REFRESH")
        row.operator("wondful.update_open_page", text="GitHub 下载", icon="URL")
        return
    box.label(text=f"发现新版本 {manifest['version']}", icon="INFO")
    if manifest.get("notes"):
        _note(box, manifest["notes"], cells, "plain", 2)
    row = box.row(align=True)
    row.operator("wondful.update_install", text="更新", icon="IMPORT")
    row.operator("wondful.update_ignore", text="忽略此版本", icon="X").version = manifest["version"]


def _draw_update_diag(adv, context, prefs):
    """高级 › 诊断 › 更新: version, 自动检查更新, 检查更新, last check result."""
    from . import update_ops as UO
    acells = _cols(context, 2)
    box, _head, _on = _section(adv, "更新", "FILE_REFRESH")
    _note(box, f"当前版本 {UO.local_version()}", acells, "plain", 1)
    row = box.row(align=True)
    row.prop(prefs, "auto_check_updates")
    check = row.row(align=True)
    check.enabled = not UO.busy()
    check.operator("wondful.update_check", text="检查更新", icon="FILE_REFRESH")
    phase, message = UO.STATE["phase"], UO.STATE["message"]
    if not message:
        return
    state = {"CHECKING": "run", "DOWNLOADING": "run", "VERIFYING": "run", "INSTALLING": "run",
             "CHECK_FAILED": "fail", "FAILED": "fail", "AVAILABLE": "info"}.get(phase, "ok")
    _note(box, message, acells, state, 4)
    if phase == "AVAILABLE" and UO.STATE.get("manifest"):
        box.operator("wondful.update_install", text=f"更新到 {UO.STATE['manifest']['version']}", icon="IMPORT")
    elif phase in ("CHECK_FAILED", "FAILED"):
        box.operator("wondful.update_open_page", text="GitHub 手动下载", icon="URL")


GITHUB_URL = "https://github.com/tiandaoyouchang-png/wondful-ai-renderer"


def draw_github_footer(layout):
    """1.2.3: one small row at the very bottom (main panel + add-on preferences): the GitHub page, 欢迎点 Star."""
    layout.separator()
    row = layout.row(align=True)
    row.scale_y = 0.9
    row.alignment = "CENTER"
    op = row.operator("wm.url_open", text="GitHub · 欢迎点 Star", icon="SOLO_ON", emboss=False)
    op.url = GITHUB_URL


class WONDFUL_PT_main(Panel):
    bl_label = "Wondful AI 渲染器 · 1.2"
    bl_idname = "WONDFUL_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Wondful AI"

    def draw(self, context):
        WONDFUL_PT_main._draw_main(self, context)
        draw_github_footer(self.layout)

    def _draw_main(self, context):
        """Six step cards (连接 / 产品三视图（可选） / 环境参考图 / 场景粗模（可选） / 提示词 / 生成); the rest in 高级 (tabs, 1.2.3)."""
        from .operators import generate_needs_polish
        layout = self.layout
        props = context.scene.wondful_ai
        prefs = _prefs(context)
        busy = props.status in BUSY_STATUSES
        cells = _cols(context, 1)

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

        _draw_update_banner(layout, context)

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
                _hint(step1, _t("在终端和浏览器里登录后，回来点「验证登录」"), cells)
            else:
                connect.operator("wondful.codex_login", text="使用 ChatGPT 登录", icon="USER")
            if state == "NOT_INSTALLED":
                _hint(step1, _t("已装 Codex 仍找不到？在 高级 › 诊断 › 偏好设置 里填路径"), cells)
            elif account_message:
                _note(step1, account_message, cells, "plain", 2)
        if props.status == "AUTHENTICATING":
            auth = step1.box()
            if active_is_antigravity:
                _lines(auth, props.auth_detail or _t("正在连接 AGY"), "TIME", _cols(context, 2), 4)
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
                    extra.label(text=f"未使用{SEP}{_short(item.image.name if item.image else '无效图片', 16)}", icon="INFO")
                    remove = extra.operator("wondful.clear_reference", text="", icon="X")
                    remove.ref_kind, remove.item_index = "PRODUCT", index
        if not props.product_collection:
            step2.label(text="产品所在集合（锁定构图用）：" if not structure_ready else "产品所在集合：")
        target = step2.row(align=True)
        target.prop(props, "product_collection", text="", icon="OUTLINER_COLLECTION")
        if not camera:
            _hint(step2, "场景里还没有活动相机", cells)

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

        # ---- ④ 场景粗模（可选）--------------------------------------------------
        _draw_blockout_step(layout, context, props, busy, connected, active_is_antigravity, cells)

        # ---- ⑤ 提示词 -----------------------------------------------------------
        step4, tools = _step_card(layout, 5, "提示词", done=prompt_ready, ready=connected and product_done)
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
                _note(prompt_box, props.prompt_notice, cells, "info", 2)
            if getattr(props, "prompt_removed_notice", ""):
                _lines(prompt_box, props.prompt_removed_notice, "TRASH", cells, 2)
            polish = prompt_box.row(align=True)
            polish.enabled = bool(not busy and connected)
            polish.operator("wondful.polish_prompt", text="只重写提示词", icon="TEXT")

        # ---- ⑥ 生成 -------------------------------------------------------------
        result_done = bool(props.result_image) and not busy
        step5, tools = _step_card(layout, 6, "生成", done=result_done, ready=can_generate or busy)
        if props.two_step and not busy:
            _fit(tools, 3)
            tools.label(text="分两步")
        generating = props.status in {"POLISHING", "RENDERING"}
        # 1.2.3: side tasks report here too (场景粗模 from ④ as one line, 多角度 from 高级), whatever tab is open
        side_task = props.status == "BLOCKOUT" or (generating and (props.task_phase or "").startswith("多角度"))
        if side_task:
            _draw_task_progress(step5, props, cells)
        elif generating:
            from .operators import views_plan
            plan = views_plan(props)
            if plan:
                _note(step5, _view_phase(1, int(plan.get("total") or len(plan["cameras"])), plan["cameras"][0], True),
                      cells, "run", 1)
            current = _stage_index(props)
            stages = step5.column(align=True)
            stages.scale_y = 0.8
            for index, name in enumerate(GENERATE_STAGES):
                icon = "CHECKMARK" if index < current else ("PLAY" if index == current else "RADIOBUT_OFF")
                cell = stages.row(align=True)
                cell.active = index <= current
                cell.label(text=name + (SEP + "进行中" if index == current else ""), icon=icon)
            pct = max(0, min(100, int(round(float(props.progress) * 100))))
            phase = getattr(props, "task_phase", "") or GENERATE_STAGES[max(0, current)]
            _note(step5, phase, cells, "run", 2)
            bar = step5.column()
            bar.scale_y = 1.4
            _draw_progress_bar(bar, props, f"估算 {pct}%")
        else:
            views = []
            if _scene_blockout_exists(context):      # 1.2.3: ④ built a 场景粗模 → choose the views to render
                views = _draw_render_views(step5, context, props, busy, cells)
            if props.two_step:
                go_text = "生成提示词" if needs_polish else "渲染"
            else:
                go_text = "生成"
            if len(views) > 1 and not (props.two_step and needs_polish):
                go_text += f" {len(views)} 个视角"
            # 1.2.2: images per generation, right above 生成.
            count_row = step5.row(align=True)
            count_row.enabled = not busy
            count_row.label(text="张数")
            count_row.prop(props, "render_count", expand=True)
            cap = provider_variant_cap("antigravity" if active_is_antigravity else "codex")
            if int(props.render_count) > cap:
                _hint(step5, f"Antigravity 每次最多 {cap} 张，将生成 {cap} 张", cells)
            go = step5.row(align=True)
            go.scale_y = 2.0
            go.enabled = can_generate
            go.operator("wondful.generate", text=go_text, icon="RENDER_STILL")
        if busy:
            if props.status not in {"POLISHING", "RENDERING"} and not side_task:
                _note(step5, getattr(props, "task_phase", "") or "后台任务进行中", cells, "run", 2)
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
                _hint(step5, next_step, cells)
        if getattr(props, "model_notice", ""):
            _note(step5, props.model_notice, cells, "info", 3)
        if props.status == "ERROR" and props.last_error:
            err = step5.box()
            _note(err, props.last_error, _cols(context, 2), "fail", 6)
            err.operator("wondful.copy_diagnostics", text="复制诊断信息", icon="COPYDOWN")

        # ---- 结果（属于 ⑥）------------------------------------------------------
        provider_variant_count = render_variant_count("antigravity" if active_is_antigravity else "codex", props)
        if props.result_image:
            rcells = _cols(context, 2)
            result = step5.box()
            title = result.row(align=True)
            title.label(text="结果", icon="IMAGE_DATA")
            if props.alignment_score >= 0:
                ok = props.alignment_score >= prefs.alignment_threshold
                score = title.row(align=True)
                score.alignment = "RIGHT"
                _fit(score, 3.6)
                score.label(text=f"对齐 {props.alignment_score:.0f}", icon="CHECKMARK" if ok else "INFO")
            _draw_view_flip(result, props)
            _draw_thumbnail(result, props.result_image, _thumb_scale(context, 1.0, hi=18.0))
            exported_count = max(0, int(getattr(props, "variant_count", 0)))
            if props.status == "RENDERING" and props.last_export_path:
                _note(result, f"已完成 {exported_count}/{provider_variant_count} 张，继续生成中", rcells, "run", 1)
            elif exported_count > 1:
                _note(result, f"共 {exported_count} 张，预览为最佳一张", rcells, "plain", 1)
            engine_note = getattr(props, "last_engine_note", "")
            if engine_note:
                _note(result, engine_note, rcells, "info" if "改用" in engine_note else "ok", 2)
            layers_note = getattr(props, "last_layers_note", "")
            if layers_note:
                _note(result, layers_note, rcells, _state(layers_note), 2)
            detail_note = getattr(props, "last_detail_note", "")
            if "镜像" in detail_note:
                _note(result, "车牌 / 标识在模型里是镜像的，见 高级 › 产品 › 细节修复", rcells, "info", 2)
            if detail_note and detail_note.startswith("细节"):
                _note(result, detail_note.partition(" · 注意：")[0], rcells,
                      _state(detail_note, info=("跳过", "已放弃", "无可用生图")), 2)
            again = result.row(align=True)
            again.scale_y = 1.4
            again.enabled = not busy
            again.operator("wondful.ai_render", text="再来一张", icon="FILE_REFRESH")
            actions = result.row(align=True)
            actions.scale_y = 1.2
            actions.operator("wondful.open_compare_workspace", text="对比原图", icon="ARROW_LEFTRIGHT")
            actions.operator("wondful.open_output_directory", text="打开文件夹", icon="FILE_FOLDER")

        # ---- 高级 ---------------------------------------------------------------
        # 1.2.3: one group at a time (模型 · 输出 · 成片 · 产品 · 场景 · 多角度 · 诊断; 成片 1.2.4); running progress stays in ⑥ (场景粗模: ④ + one line in ⑥).
        layout.separator()
        advanced = layout.box()
        if not _disclosure(advanced.row(align=True), props, "ui_show_advanced", "高级", icon="PREFERENCES"):
            return
        tabs = advanced.row(align=True)
        tabs.scale_y = 1.2
        tabs.prop(props, "ui_adv_tab", expand=True)
        adv = advanced.column()
        tab = getattr(props, "ui_adv_tab", "MODEL")
        acells = _cols(context, 2)

        if tab == "MODEL":
            ai, _head, _on = _section(adv, "AI 服务", "TOOL_SETTINGS")
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
            model_row.operator("wondful.model_default", text="恢复默认")
            msg = getattr(props, model_provider + "_models_message")
            if msg:
                _note(ai, msg, acells, "plain", 2)
            if active_is_antigravity:
                ai.label(text="生图：AGY generate_image（由 CLI 管理）", icon="RENDER_STILL")
            else:
                ai.prop(prefs, "codex_direct_edit", text="图像编辑直连（推荐）")
                if getattr(prefs, "codex_direct_edit", True):
                    simple = ai.row(align=True)
                    simple.separator(factor=2.0)
                    simple.prop(prefs, "codex_direct_simple", text="简洁输入（推荐）")
                ai.label(text=("生图：Codex 图像编辑 · 失败自动改用 CLI" if getattr(prefs, "codex_direct_edit", True)
                               else "生图：Codex ImageGen（由 CLI 管理）"), icon="RENDER_STILL")
                if getattr(prefs, "codex_direct_edit", True) and getattr(prefs, "codex_direct_simple", True):
                    _hint(ai, "地面 / 地形在 场景 › 地面", acells)
            acct, _head, _on = _section(adv, "账户", "USER")
            row = acct.row(align=True)
            row.operator(refresh_op, text="检测登录", icon="FILE_REFRESH")
            if connected and not active_is_antigravity:
                row.operator("wondful.codex_logout", text="退出登录", icon="X")
            if active_is_antigravity:
                row.operator("wondful.antigravity_terminal", text="Google 登录", icon="USER")
            if cli_version:
                _note(acct, cli_version, acells, "plain", 2)
            if connected and account_message:
                _note(acct, account_message, acells, "plain", 2)

        elif tab == "OUTPUT":
            flow, _head, _on = _section(adv, "流程", "SEQUENCE")
            flow.prop(props, "two_step", text="分两步：先写提示词，确认后再渲染")
            mode_row = flow.row(align=True)
            mode_row.label(text="模式")
            mode_row.prop(props, "render_mode", expand=True)
            out_row = flow.row(align=True)
            out_row.prop(props, "output_directory", text="输出目录")
            out_row.operator("wondful.choose_output_directory", text="", icon="FILEBROWSER")
            if not (props.output_directory or "").strip():
                _hint(flow, "留空：存到 .blend 旁的 Wondful_Renders", acells)
            if camera:
                flow.label(text=f"相机：{camera.name}{SEP}{camera_w} × {camera_h}", icon="CAMERA_DATA")
            if not prefs.structure_guides_enabled:
                _note(flow, "结构控制图已在偏好设置中关闭", acells, "info")

            layers, _head, on = _section(adv, "分层输出", "RENDERLAYERS", props, "layered_output")
            if on:
                mode_row = layers.row(align=True)
                mode_row.enabled = not busy
                mode_row.prop(props, "layered_mode", expand=True)
                full = getattr(props, "layered_mode", "SHADOW") == "FULL"
                _hint(layers, ("每张约多 30–50 秒" if full else "几秒完成") + SEP + "存到结果图旁（PNG + PSD）", acells)
                if not getattr(props, "product_collection", None):
                    _hint(layers, "需要先在 ② 选择产品集合", acells)
            if getattr(props, "last_layers_note", ""):
                _note(layers, props.last_layers_note, acells, _state(props.last_layers_note), 2)

        elif tab == "FINISH":
            _draw_photo_finish(adv, context, props, busy)

        elif tab == "PRODUCT":
            _draw_iou_gate(adv, context, props, busy)
            _draw_detail_paste(adv, context, props, busy)
            _draw_appearance_lock(adv, context, props, busy, has_product)

        elif tab == "SCENE":
            _draw_ground(adv, context, props, prefs, active_is_antigravity)
            _draw_massing(adv, context, props, busy)
            if getattr(props, "style_summary_cache", "") and has_style:
                _draw_style_summary(adv, context, props)
            _draw_person_refs(adv, context, props, busy)

        elif tab == "MULTIVIEW":
            _draw_multiview(adv, context, props, busy)

        else:  # DIAG
            info, _head, _on = _section(adv, "上次渲染", "RENDER_RESULT")
            if not props.result_image:
                _hint(info, "还没有渲染结果", acells)
            else:
                info.template_ID_preview(props, "result_image", rows=4, cols=7)
                if props.last_export_path:
                    _note(info, f"输出：{props.last_export_path}", acells, "plain", 2)
                if props.alignment_attempts:
                    session = f"{SEP}会话 {props.last_session_id}" if props.last_session_id else ""
                    _note(info, f"尝试 {props.alignment_attempts} 次{session}", acells, "plain", 1)
                if props.last_canvas_warning:
                    _note(info, props.last_canvas_warning, acells, "info", 3)
                iou_note = getattr(props, "last_iou_note", "")
                if iou_note:
                    ok = "未通过" not in iou_note and "待复核" not in iou_note
                    _note(info, iou_note if iou_note.startswith("轮廓") else "轮廓验收：" + iou_note, acells,
                          "ok" if ok else ("info" if "待复核" in iou_note else "fail"), 3)
                score_note = getattr(props, "last_score_note", "")
                if score_note:
                    _note(info, "产品还原度：" + score_note, acells, "ok", 3)
                detail_note = getattr(props, "last_detail_note", "")
                if detail_note:   # 1.2.3: 细节：5 处合格 · 修复 2 处 · 放弃 0 / 细节：全部合格
                    _note(info, detail_note.partition(" · 注意：")[0], acells,
                          _state(detail_note, info=("跳过", "已放弃", "无可用生图")), 3)
                if props.last_structure_note:
                    _note(info, props.last_structure_note, acells, "plain", 2)
                if getattr(props, "last_engine_note", ""):
                    _note(info, props.last_engine_note, acells, "plain", 2)
                if props.last_total_seconds > 0:
                    _note(info, f"总用时 {props.last_total_seconds:.0f} 秒{SEP}生图 {props.last_generation_seconds:.0f} 秒"
                                f"{SEP}验收 {props.last_audit_seconds:.0f} 秒", acells, "plain", 2)
                if getattr(props, "active_conversation_id", ""):
                    info.operator("wondful.copy_conversation_id", text="复制远程会话恢复命令", icon="COPYDOWN")

            _draw_update_diag(adv, context, prefs)

            box, _head, _on = _section(adv, "工具", "CONSOLE")
            tools = box.row(align=True)
            tools.operator("wondful.copy_diagnostics", text="复制诊断信息", icon="COPYDOWN")
            tools.operator("preferences.addon_show", text="打开偏好设置", icon="PREFERENCES").module = __package__


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
