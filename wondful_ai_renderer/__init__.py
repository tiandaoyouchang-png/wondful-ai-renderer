bl_info = {
    "name": "Wondful AI 渲染器 1.2",
    "author": "Wondful AI",
    "version": (1, 2, 3),
    "blender": (4, 3, 0),
    "location": "3D Viewport > Sidebar > Wondful AI",
    "description": "Deterministic Geometry → Structure Packet → Generative Appearance；Camera Base + Mask/Depth/Normal/Silhouette/Part-ID + Jev Semantic Layer",
    "category": "Render",
}

import bpy

try:
    from bpy.app.handlers import persistent
except Exception:
    def persistent(fn):
        return fn

from .i18n import TRANSLATIONS
from .prompt_editor import reset_prompt_editors, draw_prompt_editor_header

from .comparison_view import (
    WONDFUL_OT_compare_modal,
    WONDFUL_OT_compare_refresh,
    WONDFUL_OT_compare_set_factor,
    WONDFUL_OT_open_compare_workspace,
    WONDFUL_OT_leave_compare_workspace,
    WONDFUL_PT_compare_workspace,
)
from .operators import (
    WONDFUL_OT_ai_render,
    WONDFUL_OT_clear_all_references,
    WONDFUL_OT_clear_reference,
    WONDFUL_OT_codex_check,
    WONDFUL_OT_codex_login,
    WONDFUL_OT_codex_logout,
    WONDFUL_OT_antigravity_check,
    WONDFUL_OT_antigravity_login,
    WONDFUL_OT_antigravity_submit_code,
    WONDFUL_OT_antigravity_auth_cancel,
    WONDFUL_OT_antigravity_auth_browser,
    WONDFUL_OT_antigravity_terminal,
    WONDFUL_OT_copy_diagnostics,
    WONDFUL_OT_refresh_models,
    WONDFUL_OT_model_default,
    WONDFUL_OT_terrain_preset,
    WONDFUL_OT_block_ref_add,
    WONDFUL_OT_block_ref_clear,
    WONDFUL_OT_block_select,
    WONDFUL_OT_massing_hull_build,
    WONDFUL_OT_scene_blockout_build,
    WONDFUL_OT_mv_set_master,
    WONDFUL_OT_mv_camera_add,
    WONDFUL_OT_mv_camera_remove,
    WONDFUL_OT_mv_camera_from_view,
    WONDFUL_OT_multiview_render,
    WONDFUL_OT_mark_keep_exact,
    WONDFUL_OT_load_reference,
    WONDFUL_OT_choose_output_directory,
    WONDFUL_OT_open_output_directory,
    WONDFUL_OT_move_reference,
    WONDFUL_OT_paste_reference,
    WONDFUL_OT_polish_prompt,
    WONDFUL_OT_open_prompt_text_editor,
    WONDFUL_OT_close_prompt_text_editor,
    WONDFUL_OT_prompt_expand,
    WONDFUL_OT_prompt_line_add,
    WONDFUL_OT_prompt_line_remove,
    WONDFUL_OT_replace_reference,
    WONDFUL_OT_cancel_task,
    WONDFUL_OT_auto_classify_references,
    WONDFUL_OT_autofill_product_look,
    WONDFUL_OT_generate,
    WONDFUL_OT_copy_conversation_id,
    reset_stale_task_status,
)
from .properties import WONDFUL_AISettings, WONDFUL_AddonPreferences, WONDFUL_PromptLine, WONDFUL_ReferenceItem, WONDFUL_ModelItem, WONDFUL_MVCameraItem, sync_prompt_editor
from .ui import WONDFUL_PT_main, WONDFUL_UL_prompt_lines, WONDFUL_UL_reference_items
from . import update_ops


CLASSES = (
    WONDFUL_AddonPreferences,
    WONDFUL_PromptLine,
    WONDFUL_ReferenceItem,
    WONDFUL_ModelItem,
    WONDFUL_MVCameraItem,
    WONDFUL_AISettings,
    WONDFUL_UL_prompt_lines,
    WONDFUL_UL_reference_items,
    WONDFUL_OT_codex_check,
    WONDFUL_OT_codex_login,
    WONDFUL_OT_codex_logout,
    WONDFUL_OT_antigravity_check,
    WONDFUL_OT_antigravity_login,
    WONDFUL_OT_antigravity_submit_code,
    WONDFUL_OT_antigravity_auth_cancel,
    WONDFUL_OT_antigravity_auth_browser,
    WONDFUL_OT_antigravity_terminal,
    WONDFUL_OT_copy_diagnostics,
    WONDFUL_OT_refresh_models,
    WONDFUL_OT_model_default,
    WONDFUL_OT_terrain_preset,
    WONDFUL_OT_block_ref_add,
    WONDFUL_OT_block_ref_clear,
    WONDFUL_OT_block_select,
    WONDFUL_OT_massing_hull_build,
    WONDFUL_OT_scene_blockout_build,
    WONDFUL_OT_mv_set_master,
    WONDFUL_OT_mv_camera_add,
    WONDFUL_OT_mv_camera_remove,
    WONDFUL_OT_mv_camera_from_view,
    WONDFUL_OT_multiview_render,
    WONDFUL_OT_mark_keep_exact,
    WONDFUL_OT_polish_prompt,
    WONDFUL_OT_ai_render,
    WONDFUL_OT_cancel_task,
    WONDFUL_OT_auto_classify_references,
    WONDFUL_OT_autofill_product_look,
    WONDFUL_OT_generate,
    WONDFUL_OT_copy_conversation_id,
    WONDFUL_OT_load_reference,
    WONDFUL_OT_choose_output_directory,
    WONDFUL_OT_open_output_directory,
    WONDFUL_OT_paste_reference,
    WONDFUL_OT_replace_reference,
    WONDFUL_OT_clear_reference,
    WONDFUL_OT_clear_all_references,
    WONDFUL_OT_move_reference,
    WONDFUL_OT_open_prompt_text_editor,
    WONDFUL_OT_close_prompt_text_editor,
    WONDFUL_OT_prompt_expand,
    WONDFUL_OT_prompt_line_add,
    WONDFUL_OT_prompt_line_remove,
    WONDFUL_OT_open_compare_workspace,
    WONDFUL_OT_leave_compare_workspace,
    WONDFUL_OT_compare_refresh,
    WONDFUL_OT_compare_set_factor,
    WONDFUL_OT_compare_modal,
    WONDFUL_PT_compare_workspace,
    *update_ops.CLASSES,
    WONDFUL_PT_main,
)


def _initialize_prompt_editors():
    """Rebuild compatibility rows from the canonical prompt outside Panel.draw()."""
    for scene in getattr(bpy.data, "scenes", []):
        try:
            props = getattr(scene, "wondful_ai", None)
            if props is not None:
                # Always rebuild after register/load so .blend files saved by older
                # versions cannot keep stale 84-character rows that still ellipsize.
                sync_prompt_editor(props, props.prompt or "")
        except Exception:
            # A malformed legacy scene must never prevent the add-on from registering.
            pass


def _recover_stale_task_states():
    """Clear serialized async status when no worker survived the reload."""
    for scene in getattr(bpy.data, "scenes", []):
        try:
            reset_stale_task_status(getattr(scene, "wondful_ai", None))
        except (AttributeError, ReferenceError, TypeError):
            pass


@persistent
def _wondful_load_post(_dummy):
    from .operators import cancel_auth_session
    cancel_auth_session(detach=True, context=bpy.context)
    for wm in getattr(bpy.data, "window_managers", []):
        if hasattr(wm, "wondful_auth_code"):
            wm.wondful_auth_code = ""
    reset_prompt_editors()
    _initialize_prompt_editors()
    _recover_stale_task_states()
    _migrate_terrain_props()


def _migrate_terrain_props():
    """1.1.0: files saved with the old 前/后/左/右 terrain controls get the matching corner slopes."""
    try:
        from .terrain import migrate_legacy_props
        for scene in getattr(bpy.data, "scenes", []):
            props = getattr(scene, "wondful_ai", None)
            if props is not None and migrate_legacy_props(props):
                print(f"[Wondful] 场景 {scene.name}：旧版前/后/左/右地形设置已换算为四角坡度。")
    except Exception as exc:
        print(f"[Wondful] 地形设置迁移跳过：{exc}")


@persistent
def _wondful_save_pre(_dummy):
    for wm in getattr(bpy.data, "window_managers", []):
        if hasattr(wm, "wondful_auth_code"):
            wm.wondful_auth_code = ""
    # Save canonical text and the user's original layout, never an orphaned
    # temporary editor whose in-memory return button cannot survive file reload.
    reset_prompt_editors(flush=True)


def _block_hint_edited(obj, _context):
    """多角度场景一致 › 体块名称 edited by the user: it now outranks the object name (no longer auto)."""
    try:
        if "wondful_hint_auto" in obj.keys():
            del obj["wondful_hint_auto"]
    except Exception:
        pass


def register():
    bpy.app.translations.register(__name__, TRANSLATIONS)
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.wondful_auth_code = bpy.props.StringProperty(
        name="授权码", subtype="PASSWORD", default="", options={"SKIP_SAVE"})
    bpy.types.Scene.wondful_ai = bpy.props.PointerProperty(type=WONDFUL_AISettings)
    # 1.2.3: a block's name for the prompt; shares storage with the ``obj["wondful_hint"]`` custom property
    bpy.types.Object.wondful_hint = bpy.props.StringProperty(
        name="体块名称", default="", maxlen=60, update=_block_hint_edited,
        description="这个环境体块是什么（如 沙发、墙面、书架、高楼）；留空则按物体名、材质名或尺寸自动命名")
    try:
        if _wondful_load_post not in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.append(_wondful_load_post)
        if _wondful_save_pre not in bpy.app.handlers.save_pre:
            bpy.app.handlers.save_pre.append(_wondful_save_pre)
    except Exception:
        pass
    _initialize_prompt_editors()
    _recover_stale_task_states()
    bpy.types.TEXT_HT_header.append(draw_prompt_editor_header)
    update_ops.schedule_startup_check()


def unregister():
    update_ops.unregister_timers()
    from .operators import cancel_auth_session
    cancel_auth_session(detach=True, context=bpy.context)
    if hasattr(bpy.types.WindowManager, "wondful_auth_code"):
        del bpy.types.WindowManager.wondful_auth_code
    reset_prompt_editors(flush=True)
    try:
        bpy.types.TEXT_HT_header.remove(draw_prompt_editor_header)
    except (ValueError, RuntimeError):
        pass
    try:
        if _wondful_load_post in bpy.app.handlers.load_post:
            bpy.app.handlers.load_post.remove(_wondful_load_post)
        if _wondful_save_pre in bpy.app.handlers.save_pre:
            bpy.app.handlers.save_pre.remove(_wondful_save_pre)
    except Exception:
        pass
    if hasattr(bpy.types.Scene, "wondful_ai"):
        del bpy.types.Scene.wondful_ai
    if hasattr(bpy.types.Object, "wondful_hint"):
        del bpy.types.Object.wondful_hint
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    bpy.app.translations.unregister(__name__)
