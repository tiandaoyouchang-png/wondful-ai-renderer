# RNA Property declarations must evaluate at class creation (Blender 4.3+).

import time

import bpy

from .i18n import translate_message as _message

_t = bpy.app.translations.pgettext_iface
from bpy.props import FloatProperty
from bpy.types import Operator, Panel

from .compare_core import compose_difference, compose_overlay, compose_split, compose_outline, preview_dimensions, resize_rgba_nearest
from .composition import assess_canvas


COMPARE_WORKSPACE_NAME = "Wondful Compare"
COMPARE_IMAGE_NAME = "Wondful_Compare_Preview"
COMPARE_WORKSPACE_OWNER_KEY = "wondful_compare_owned"
COMPARE_WORKSPACE_OWNER_VALUE = "Wondful AI Renderer"
_ACTIVE_COMPARE_OPERATOR = None
_COMPARE_CACHE = {
    "ref_key": None,
    "result_key": None,
    "width": 0,
    "height": 0,
    "reference": None,
    "result": None,
}


def _image_key(image):
    if image is None:
        return None
    try:
        return int(image.as_pointer())
    except Exception:
        return id(image)


def _image_rgba(image):
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("Wondful Compare 需要 Blender 自带的 NumPy。") from exc

    w, h = int(image.size[0]), int(image.size[1])
    if w <= 0 or h <= 0:
        raise RuntimeError(_t('图片 {v0} 尚未加载完成。').format(v0=image.name))
    data = np.asarray(image.pixels[:], dtype=np.float32)
    expected = w * h * 4
    if data.size != expected:
        raise RuntimeError(_t('图片 {v0} 像素数据异常。').format(v0=image.name))
    return data.reshape((h, w, 4))


def _ensure_cached_arrays(props, force=False):
    ref = props.viewport_image
    result = props.result_image
    if not ref or not result:
        raise RuntimeError("当前 Render Session 没有同时包含 Camera Reference 和 AI Render。")

    ref_w, ref_h = int(ref.size[0]), int(ref.size[1])
    out_w, out_h = int(result.size[0]), int(result.size[1])
    if ref_w <= 0 or ref_h <= 0 or out_w <= 0 or out_h <= 0:
        raise RuntimeError("对比图片尚未加载完成。")
    if not assess_canvas((out_w, out_h), (ref_w, ref_h))["matches"]:
        raise RuntimeError("AI 结果与 Camera Reference 宽高比不一致，无法精确对比。")

    ref_key = _image_key(ref)
    result_key = _image_key(result)
    target_w, target_h = preview_dimensions(ref_w, ref_h, 1400)
    cache_hit = (
        not force
        and _COMPARE_CACHE["ref_key"] == ref_key
        and _COMPARE_CACHE["result_key"] == result_key
        and _COMPARE_CACHE["width"] == target_w
        and _COMPARE_CACHE["height"] == target_h
        and _COMPARE_CACHE["reference"] is not None
        and _COMPARE_CACHE["result"] is not None
    )
    if cache_hit:
        return target_w, target_h, _COMPARE_CACHE["reference"], _COMPARE_CACHE["result"]

    ref_arr = resize_rgba_nearest(_image_rgba(ref), target_w, target_h)
    out_arr = resize_rgba_nearest(_image_rgba(result), target_w, target_h)
    _COMPARE_CACHE.update(
        {
            "ref_key": ref_key,
            "result_key": result_key,
            "width": target_w,
            "height": target_h,
            "reference": ref_arr,
            "result": out_arr,
        }
    )
    return target_w, target_h, ref_arr, out_arr


def _ensure_compare_image(width: int, height: int):
    image = bpy.data.images.get(COMPARE_IMAGE_NAME)
    if image is None:
        image = bpy.data.images.new(COMPARE_IMAGE_NAME, width=width, height=height, alpha=True)
    elif (int(image.size[0]), int(image.size[1])) != (width, height):
        try:
            bpy.data.images.remove(image)
        except Exception:
            pass
        image = bpy.data.images.new(COMPARE_IMAGE_NAME, width=width, height=height, alpha=True)
    return image


def refresh_compare_image(context, force_cache=False):
    """Rebuild the visible comparison preview in Blender's Image Editor."""
    if context is None or not getattr(context, "scene", None) or not hasattr(context.scene, "wondful_ai"):
        return None
    props = context.scene.wondful_ai
    if not props.viewport_image or not props.result_image:
        return None

    width, height, ref_arr, out_arr = _ensure_cached_arrays(props, force=force_cache)
    mode = getattr(props, "compare_mode", "SPLIT")
    factor = max(0.0, min(1.0, float(props.compare_factor)))
    if mode == "OVERLAY":
        composed = compose_overlay(ref_arr, out_arr, factor)
    elif mode == "DIFFERENCE":
        composed = compose_difference(ref_arr, out_arr)
    elif mode == "OUTLINE":
        mask_image = getattr(props, "structure_mask_image", None)
        if mask_image is None:
            raise RuntimeError("当前结果没有白模产品轮廓，请启用结构图后重新生成。")
        mask = resize_rgba_nearest(_image_rgba(mask_image), width, height)
        composed = compose_outline(out_arr, mask)
    else:
        composed = compose_split(ref_arr, out_arr, factor)

    image = _ensure_compare_image(width, height)
    try:
        image.pixels.foreach_set(composed.ravel())
    except Exception:
        image.pixels[:] = composed.ravel().tolist()
    image.update()
    return image


def _find_image_editor_area(window):
    if not window or not window.screen:
        return None, None
    for area in window.screen.areas:
        if area.type == "IMAGE_EDITOR":
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region:
                return area, region
    return None, None


def _configure_compare_screen(window, compare_image=None):
    if not window or not window.screen or not window.screen.areas:
        raise RuntimeError("无法配置 Wondful Compare 工作区。")

    area, region = _find_image_editor_area(window)
    if area is None:
        candidates = [a for a in window.screen.areas if a.type == "VIEW_3D"] or list(window.screen.areas)
        area = max(candidates, key=lambda a: max(1, a.width) * max(1, a.height))
        area.type = "IMAGE_EDITOR"
        region = next((r for r in area.regions if r.type == "WINDOW"), None)

    space = area.spaces.active
    try:
        space.image = compare_image
    except Exception:
        pass
    try:
        space.show_region_ui = True
    except Exception:
        pass
    try:
        space.show_region_toolbar = False
    except Exception:
        pass
    return area, region


def _fit_image_view(window, area, region):
    if not window or not area or not region:
        return
    try:
        with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=region):
            bpy.ops.image.view_all(fit_view=True)
    except Exception:
        pass


def _workspace_id(workspace):
    if workspace is None:
        return None
    try:
        return int(workspace.as_pointer())
    except Exception:
        return id(workspace)


def _is_owned_compare_workspace(workspace):
    if workspace is None:
        return False
    try:
        return workspace.get(COMPARE_WORKSPACE_OWNER_KEY) == COMPARE_WORKSPACE_OWNER_VALUE
    except Exception:
        return False


def _find_owned_compare_workspace():
    for workspace in bpy.data.workspaces:
        if _is_owned_compare_workspace(workspace):
            return workspace
    return None


def _unique_workspace_name(base):
    if bpy.data.workspaces.get(base) is None:
        return base
    index = 1
    while bpy.data.workspaces.get(f"{base}.{index:03d}") is not None:
        index += 1
    return f"{base}.{index:03d}"


def _repair_legacy_compare_workspace(context):
    """Repair the specific pre-2.20 duplicate/rename bug when it is unambiguous.

    Older versions assumed workspace.duplicate() switched the current window to the
    duplicate. On Blender builds where it did not, the user's original workspace was
    renamed to Wondful Compare while the untouched duplicate became e.g. 布局.001.

    We only auto-repair the exact signature recorded by compare_return_workspace:
    missing original name + matching .001 duplicate + untagged Wondful Compare.
    Nothing is deleted; the legacy mutated workspace is preserved under a Legacy name.
    """
    window = context.window
    props = context.scene.wondful_ai
    legacy = bpy.data.workspaces.get(COMPARE_WORKSPACE_NAME)
    if legacy is None or _is_owned_compare_workspace(legacy):
        return

    return_name = (getattr(props, "compare_return_workspace", "") or "").strip()
    if not return_name or bpy.data.workspaces.get(return_name) is not None:
        return

    candidate = bpy.data.workspaces.get(return_name + ".001")
    if candidate is None or candidate is legacy:
        return

    # Switch away from the legacy workspace before renaming it. Preserve it instead
    # of deleting user data; users can remove it manually after verifying recovery.
    if window.workspace is legacy:
        window.workspace = candidate
    legacy.name = _unique_workspace_name("Wondful Compare Legacy")
    candidate.name = return_name
    props.compare_return_workspace = return_name


def _duplicate_workspace_safely(context, source_workspace):
    """Duplicate source_workspace and return the actual newly-created Workspace.

    Never assume Blender switches the window to the duplicate. The set-difference is
    the source of truth, preventing the user's original workspace from being renamed
    or reconfigured.
    """
    window = context.window
    if source_workspace is None:
        raise RuntimeError("没有可用于创建对比工作区的源 Workspace。")
    if window.workspace is not source_workspace:
        window.workspace = source_workspace

    before = {_workspace_id(ws) for ws in bpy.data.workspaces}
    try:
        with bpy.context.temp_override(window=window, screen=window.screen):
            bpy.ops.workspace.duplicate()
    except Exception as exc:
        raise RuntimeError(_t('无法创建 {v0} 工作区: {v1}').format(v0=COMPARE_WORKSPACE_NAME, v1=exc)) from exc

    created = [ws for ws in bpy.data.workspaces if _workspace_id(ws) not in before]
    if not created:
        raise RuntimeError("Blender 未返回新建 Workspace；已停止，原布局不会被修改。")
    duplicate = created[-1]
    window.workspace = duplicate
    duplicate.name = COMPARE_WORKSPACE_NAME
    try:
        duplicate[COMPARE_WORKSPACE_OWNER_KEY] = COMPARE_WORKSPACE_OWNER_VALUE
    except Exception:
        pass
    return duplicate


def _ensure_compare_workspace(context):
    window = context.window
    if not window:
        raise RuntimeError("没有可用的 Blender Window。")
    props = context.scene.wondful_ai

    _repair_legacy_compare_workspace(context)

    current = window.workspace
    if current and not _is_owned_compare_workspace(current):
        props.compare_return_workspace = current.name

    existing = _find_owned_compare_workspace()
    if existing is not None:
        window.workspace = existing
    else:
        source = window.workspace
        if _is_owned_compare_workspace(source):
            target_name = (props.compare_return_workspace or "").strip()
            source = bpy.data.workspaces.get(target_name) if target_name else None
        _duplicate_workspace_safely(context, source)

    compare_image = refresh_compare_image(context, force_cache=True)
    area, region = _configure_compare_screen(window, compare_image)
    if not area or not region:
        raise RuntimeError("Wondful Compare 工作区没有可用的 Image Editor 区域。")
    _fit_image_view(window, area, region)
    return window, area, region


def _start_drag_modal(window, area, region):
    """Start the drag operator from the Image Editor WINDOW region.

    Starting it from the sidebar UI region was the reason 2.5/2.6 could show a
    compare image while mouse dragging did not work reliably.
    """
    try:
        with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=region):
            result = bpy.ops.wondful.compare_modal("INVOKE_DEFAULT")
        return "RUNNING_MODAL" in result
    except Exception:
        return False


def _display_rect(region, image):
    """Return fitted-image x/y/w/h in global window coordinates.

    The compare workspace calls Image > View All before drag mode starts, so a
    fitted rectangle is a stable mapping from mouse X to split factor.
    """
    rw = max(1.0, float(region.width))
    rh = max(1.0, float(region.height))
    iw = max(1.0, float(image.size[0]))
    ih = max(1.0, float(image.size[1]))
    image_ratio = iw / ih
    region_ratio = rw / rh
    if region_ratio >= image_ratio:
        h = rh
        w = h * image_ratio
        x = float(region.x) + (rw - w) * 0.5
        y = float(region.y)
    else:
        w = rw
        h = w / image_ratio
        x = float(region.x)
        y = float(region.y) + (rh - h) * 0.5
    return x, y, max(1.0, w), max(1.0, h)


class WONDFUL_OT_open_compare_workspace(Operator):
    bl_idname = "wondful.open_compare_workspace"
    bl_label = "打开渲染对比"
    bl_description = "进入 Wondful Compare 工作区，比较 Blender Camera Reference 与 AI Render"

    def execute(self, context):
        props = context.scene.wondful_ai
        if not props.viewport_image or not props.result_image:
            self.report({"ERROR"}, _message("当前 Render Session 没有同时包含 Camera Reference 和 AI Render。", report=True))
            return {"CANCELLED"}
        try:
            window, area, region = _ensure_compare_workspace(context)
            started = _start_drag_modal(window, area, region)
        except Exception as exc:
            self.report({"ERROR"}, _message(str(exc), report=True))
            return {"CANCELLED"}
        if started:
            self.report({"INFO"}, _message("已进入 Wondful Compare。直接在图片上按住左键左右拖动分割线。", report=True))
        else:
            self.report({"INFO"}, _message("已进入 Wondful Compare。鼠标拖动未自动启动，可用右侧分割位置滑块或“启用鼠标拖动”。", report=True))
        return {"FINISHED"}


class WONDFUL_OT_leave_compare_workspace(Operator):
    bl_idname = "wondful.leave_compare_workspace"
    bl_label = "返回原工作区"

    def execute(self, context):
        props = context.scene.wondful_ai
        props.compare_drag_active = False
        target_name = (props.compare_return_workspace or "").strip()
        target = bpy.data.workspaces.get(target_name) if target_name else None
        if target is None:
            target = next((w for w in bpy.data.workspaces if not _is_owned_compare_workspace(w)), None)
        if target is None:
            self.report({"WARNING"}, _message("没有找到可返回的工作区。", report=True))
            return {"CANCELLED"}
        context.window.workspace = target
        return {"FINISHED"}


class WONDFUL_OT_compare_refresh(Operator):
    bl_idname = "wondful.compare_refresh"
    bl_label = "刷新对比"

    def execute(self, context):
        try:
            image = refresh_compare_image(context, force_cache=True)
            if context.area and context.area.type == "IMAGE_EDITOR" and image:
                context.area.spaces.active.image = image
                region = next((r for r in context.area.regions if r.type == "WINDOW"), None)
                _fit_image_view(context.window, context.area, region)
        except Exception as exc:
            self.report({"ERROR"}, _message(str(exc), report=True))
            return {"CANCELLED"}
        return {"FINISHED"}


class WONDFUL_OT_compare_set_factor(Operator):
    bl_idname = "wondful.compare_set_factor"
    bl_label = "设置对比位置"

    value: FloatProperty(default=0.5, min=0.0, max=1.0)

    def execute(self, context):
        context.scene.wondful_ai.compare_factor = self.value
        return {"FINISHED"}


class WONDFUL_OT_compare_modal(Operator):
    bl_idname = "wondful.compare_modal"
    bl_label = "启用鼠标拖动"
    bl_description = "在对比图片内按住鼠标左键左右拖动分割线；Esc 结束拖动模式"

    _area = None
    _window_region = None
    _dragging = False
    _last_update_time = 0.0

    def invoke(self, context, event):
        global _ACTIVE_COMPARE_OPERATOR
        props = context.scene.wondful_ai
        if not props.viewport_image or not props.result_image:
            self.report({"ERROR"}, _message("没有可对比的 Camera Reference 和 AI Render。", report=True))
            return {"CANCELLED"}
        if context.area.type != "IMAGE_EDITOR":
            self.report({"ERROR"}, _message("请在 Wondful Compare 的 Image Editor 中启用鼠标拖动。", report=True))
            return {"CANCELLED"}
        if props.compare_mode != "SPLIT":
            props.compare_mode = "SPLIT"

        region = next((r for r in context.area.regions if r.type == "WINDOW"), None)
        if region is None:
            self.report({"ERROR"}, _message("Image Editor 没有 WINDOW 区域。", report=True))
            return {"CANCELLED"}

        if _ACTIVE_COMPARE_OPERATOR is not None and _ACTIVE_COMPARE_OPERATOR is not self:
            try:
                _ACTIVE_COMPARE_OPERATOR.finish(context)
            except Exception:
                pass
        _ACTIVE_COMPARE_OPERATOR = self
        self._area = context.area
        self._window_region = region
        self._dragging = False
        self._last_update_time = 0.0
        props.compare_drag_active = True
        _fit_image_view(context.window, self._area, self._window_region)
        context.window_manager.modal_handler_add(self)
        self.report({"INFO"}, _message("鼠标拖动已开启：直接在图片内按住左键左右拖动，Esc 结束。", report=True))
        return {"RUNNING_MODAL"}

    def _mouse_factor(self, event, allow_outside=False):
        region = self._window_region
        image = bpy.data.images.get(COMPARE_IMAGE_NAME)
        if region is None or image is None:
            return False, 0.5
        x0, y0, width, height = _display_rect(region, image)
        mx = float(event.mouse_x)
        my = float(event.mouse_y)

        # Generous grab area fixes intermittent misses on the thin divider and on
        # high-DPI displays. Once dragging starts, x is captured even outside the
        # image and clamped to 0..1 until mouse release.
        margin = 28.0
        inside = (
            (x0 - margin) <= mx <= (x0 + width + margin)
            and (y0 - margin) <= my <= (y0 + height + margin)
        )
        if not inside and not allow_outside:
            return False, 0.5
        factor = (mx - x0) / width
        return True, max(0.0, min(1.0, factor))

    def _apply_mouse(self, context, event, allow_outside=False, force=False):
        ok, factor = self._mouse_factor(event, allow_outside=allow_outside)
        if not ok:
            return False

        # Rebuilding a 1400px NumPy preview on every raw MOUSEMOVE can flood the
        # UI event loop on high polling-rate mice. Keep capture continuous but
        # throttle expensive pixel uploads to roughly 30 FPS. Press/release are
        # always committed immediately.
        now = time.monotonic()
        if not force and (now - self._last_update_time) < (1.0 / 30.0):
            return True
        self._last_update_time = now
        context.scene.wondful_ai.compare_factor = factor
        try:
            self._area.tag_redraw()
        except Exception:
            pass
        return True

    def modal(self, context, event):
        # The operator may stay alive while the user changes workspace. End it
        # cleanly instead of leaving a stale modal handler behind.
        if not context.window or not _is_owned_compare_workspace(context.window.workspace):
            return self.finish(context)
        if event.type in {"ESC", "RIGHTMOUSE"}:
            return self.finish(context)
        if event.type == "LEFTMOUSE":
            if event.value == "PRESS":
                if self._apply_mouse(context, event, allow_outside=False, force=True):
                    self._dragging = True
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}
            if event.value == "RELEASE" and self._dragging:
                self._apply_mouse(context, event, allow_outside=True, force=True)
                self._dragging = False
                return {"RUNNING_MODAL"}
        if event.type == "MOUSEMOVE" and self._dragging:
            self._apply_mouse(context, event, allow_outside=True)
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}

    def cancel(self, context):
        return self.finish(context)

    def finish(self, context):
        global _ACTIVE_COMPARE_OPERATOR
        try:
            if getattr(context, "scene", None) and hasattr(context.scene, "wondful_ai"):
                context.scene.wondful_ai.compare_drag_active = False
        except Exception:
            pass
        self._dragging = False
        if _ACTIVE_COMPARE_OPERATOR is self:
            _ACTIVE_COMPARE_OPERATOR = None
        return {"FINISHED"}


class WONDFUL_PT_compare_workspace(Panel):
    bl_label = "渲染对比"
    bl_idname = "WONDFUL_PT_compare_workspace"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Wondful Compare"

    def draw(self, context):
        layout = self.layout
        props = context.scene.wondful_ai

        hero = layout.box()
        row = hero.row(align=True)
        row.label(text="Camera Reference", icon="CAMERA_DATA")
        row.label(text="⇄")
        row.label(text="AI Render", icon="IMAGE_DATA")

        mode = hero.row(align=True)
        mode.prop(props, "compare_mode", expand=True)

        if props.compare_mode == "SPLIT":
            hero.prop(props, "compare_factor", text="分割位置", slider=True)
            presets = hero.row(align=True)
            op = presets.operator("wondful.compare_set_factor", text="原图")
            op.value = 1.0
            op = presets.operator("wondful.compare_set_factor", text="50 / 50")
            op.value = 0.5
            op = presets.operator("wondful.compare_set_factor", text="AI")
            op.value = 0.0
            if props.compare_drag_active:
                hero.label(text="鼠标拖动已开启 · 在图片内按住左键拖动", icon="CHECKMARK")
            else:
                hero.operator("wondful.compare_modal", text="启用鼠标拖动", icon="ARROW_LEFTRIGHT")
                hero.label(text="若拖动模式不可用，分割位置滑块始终可以直接对比。", icon="INFO")
        elif props.compare_mode == "OVERLAY":
            hero.prop(props, "compare_factor", text="AI 透明度", slider=True)
            hero.label(text="建议 50% 检查轮心、车身边界、地平线是否重合。", icon="INFO")
        elif props.compare_mode == "OUTLINE":
            hero.label(text="青线是白模产品轮廓；检查结果边缘是否与青线重合。", icon="INFO")
            if not getattr(props, "structure_mask_image", None):
                hero.label(text="当前结果缺少轮廓，请启用结构图后重新生成。", icon="ERROR")
        else:
            hero.label(text="亮区包含灯光、材质差异；构图请结合轮廓模式检查。", icon="INFO")

        hero.operator("wondful.compare_refresh", text="重新载入当前 Session", icon="FILE_REFRESH")

        align = layout.box()
        align.label(text="构图校验")
        if props.alignment_score >= 0:
            threshold = context.preferences.addons[__package__].preferences.alignment_threshold
            icon = "CHECKMARK" if props.alignment_score >= threshold else "ERROR"
            align.label(text=_t('构图评分  {v0} / 100').format(v0=format(props.alignment_score, '.0f')), icon=icon)
            if props.alignment_attempts:
                align.label(text=_t('生成尝试  {v0} 次').format(v0=props.alignment_attempts))
        else:
            align.label(text="尚未获得对齐评分", icon="INFO")
        if props.alignment_message:
            align.label(text=_message(props.alignment_message)[:100])

        layout.operator("wondful.leave_compare_workspace", text="返回原工作区", icon="TRIA_LEFT")
