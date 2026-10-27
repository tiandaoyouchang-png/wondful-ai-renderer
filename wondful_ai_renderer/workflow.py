"""Explicit workflows and reversible capture isolation; no network or import-time bpy.

ENVIRONMENT keeps the user's camera and appearance references. REFERENCE uses an
explicit target, a valid blockout, and that target's appearance. Switching mode
never changes cameras, geometry, materials, lights or visibility.
"""
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
from pathlib import Path

ENVIRONMENT = "ENVIRONMENT"
REFERENCE = "REFERENCE"
BUILD_ID = "two-workflows-ui-r4"
BUSY = {"POLISHING", "RENDERING", "AUTHENTICATING", "BLOCKOUT", "CHECKING", "AUTOFILLING", "CLASSIFYING", "REFRESHING_MODELS"}
DRAFT_FIELDS = ("prompt", "scene_brief", "scene_brief_used", "prompt_source", "last_ai_prompt",
                "prompt_style_fingerprint", "prompt_style_revision", "prompt_reference_policy_version",
                "style_prompt_dirty", "style_sync_initialized", "style_summary_cache",
                "style_summary_cache_key", "prompt_profile_version", "prompt_target_provider",
                "prompt_target_model", "prompt_target_key")
_switching = set()


def mode(props):
    return REFERENCE if getattr(props, "workflow_mode", ENVIRONMENT) == REFERENCE else ENVIRONMENT


def is_reference(props):
    return mode(props) == REFERENCE


def reference_revision(props):
    """Environment image-list revisions must not invalidate the replica draft."""
    return 0 if is_reference(props) else int(getattr(props, "style_reference_revision", 0))


def label(value):
    return "参考图复刻" if value == REFERENCE else "环境参考渲染"


def switch_mode(props, context=None):
    """RNA update callback: keep each workflow's editable draft, never mutate the scene."""
    key = props.as_pointer() if hasattr(props, "as_pointer") else id(props)
    if key in _switching:
        return
    _switching.add(key)
    try:
        old = getattr(props, "workflow_last_mode", ENVIRONMENT)
        new = mode(props)
        if old == new:
            return
        if getattr(props, "status", "IDLE") in BUSY:
            props.workflow_mode = old
            props.prompt_notice = "任务进行中，请完成或取消后再切换工作模式"
            return
        try:
            drafts = json.loads(getattr(props, "workflow_drafts", "") or "{}")
            if not isinstance(drafts, dict):
                drafts = {}
        except (ValueError, TypeError):
            drafts = {}
        drafts[old] = {k: getattr(props, k) for k in DRAFT_FIELDS if hasattr(props, k)}
        saved = drafts.get(new, {})
        for field in DRAFT_FIELDS:
            if not hasattr(props, field):
                continue
            current = getattr(props, field)
            default = False if isinstance(current, bool) else 0 if isinstance(current, int) else ""
            value = saved.get(field, default) if isinstance(saved, dict) else default
            if isinstance(value, type(default)):
                setattr(props, field, value)
        props.workflow_drafts = json.dumps(drafts, ensure_ascii=False)
        props.workflow_last_mode = new
        props.pending_generate = False
        props.views_plan = ""
        if hasattr(props, "compare_pair"):
            props.compare_pair = "TARGET_BLOCKOUT" if new == REFERENCE else "CAMERA_RESULT"
            props.compare_mode = "SPLIT"
        props.prompt_notice = "已切换到" + label(new) + "；相机和场景未改动"
    finally:
        _switching.discard(key)


def file_hash(path):
    if not path:
        return ""
    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, ValueError):
        return ""


def target_changed(props, context):
    """Loading a target is explicit; errors leave no stale preview or approval."""
    props.replica_reviewed = False  # legacy serialized field; no longer a gate
    props.sb_reference_image = None
    props.sb_preview_image = None
    props.replica_preview_signature = ""
    props.replica_revision_status = ""
    props.replica_can_undo = False
    if is_reference(props):
        props.style_prompt_dirty = True
    raw = (getattr(props, "sb_image_path", "") or "").strip()
    if raw:
        try:
            import bpy
            path = bpy.path.abspath(raw)
            if Path(path).is_file():
                props.sb_reference_image = bpy.data.images.load(path, check_existing=True)
                props.sb_reference_image.reload()
        except (OSError, RuntimeError, ValueError):
            pass


def identity_available(props):
    """Cheap UI readiness only. On-disk / decoder checks belong to action preflight."""
    items = getattr(props, "product_images", ())
    if not items:
        return False
    image = getattr(items[0], "image", None)
    try:
        return image is not None and len(image.size) == 2 and min(image.size) > 0
    except (AttributeError, TypeError, ReferenceError):
        return False


def identity_path(props, *, resolve=None, packed_export=None):
    """Only the actual first identity slot is authoritative; no second-slot fallback."""
    if not identity_available(props):
        return ""
    item = props.product_images[0]
    image = item.image
    resolve = resolve or (lambda p: str(Path(p).expanduser()))
    for raw in (getattr(item, "source_path", ""), getattr(image, "filepath_raw", ""), getattr(image, "filepath", "")):
        if not raw:
            continue
        try:
            path = Path(resolve(raw))
            if path.is_file() and path.stat().st_size > 0:
                return str(path)
        except (OSError, ValueError):
            pass
    if getattr(image, "packed_file", None) and packed_export:
        try:
            path = packed_export(image)
            if path and Path(path).is_file() and Path(path).stat().st_size > 0:
                return str(path)
        except (OSError, RuntimeError, ValueError):
            pass
    return ""


def render_text(props, references_changed=False):
    """Keep manual copy verbatim; do not reuse AI prose tied to replaced references."""
    prompt = str(getattr(props, "prompt", "") or "").strip()
    brief = str(getattr(props, "scene_brief", "") or "").strip()
    ai = str(getattr(props, "last_ai_prompt", "") or "").strip()
    changed_brief = brief != str(getattr(props, "scene_brief_used", "") or "").strip()
    if prompt and not (ai and prompt == ai and (references_changed or changed_brief)):
        return prompt
    return brief or ("按已确认的相机与粗模复刻目标参考图的环境材质和光照，保留真实车模。"
                     if is_reference(props) else "按当前相机渲染真实车模，参考环境图的材质、光照和氛围。")


def contract(props):
    common = ("产品几何、比例、灯组和轮毂以当前真实车模为准；参考图中的另一辆车不提供产品身份。"
              "车漆基础配色保留，受光、反射与接地阴影按目标环境重新匹配。")
    if is_reference(props):
        return ("【参考图复刻】当前相机底图与已确认粗模是最终构图和遮挡依据；原始目标图只提供其中实际可见的光线、场景气氛、"
                "静止或运动状态与镜头成像语言。不得臆造未出现的特征，不为匹配参考车型而拉伸车模，也不在生图阶段重新移动相机。" + common)
    return ("【环境参考渲染】只使用用户当前相机决定视角、主体位置和大小；环境图只提供环境风格、材质与光照，"
            "不得照搬其中的机位或车辆位置；不引用任何旧复刻粗模或其分析。" + common)


def preflight(props, *, has_camera, has_product, has_blockout, target_path="", source_ok=True, identity_ok=None):
    if not has_product:
        return "请先选择车模所在的产品集合"
    if not (identity_available(props) if identity_ok is None else identity_ok):
        return "请先添加有效的产品身份参考图"
    if not has_camera:
        return "请先选择活动相机"
    if not is_reference(props):
        return ""
    target_hash = file_hash(target_path)
    if not target_hash:
        return "请先选择有效的复刻目标图"
    if not has_blockout:
        return "请先生成相机与粗模，再检查预览"
    if not source_ok:
        return "粗模参考来源不可用；请重新生成粗模"
    if target_hash != str(getattr(props, "sb_built_target_hash", "") or ""):
        return "目标图与当前粗模不一致，或旧粗模缺少来源记录；请重新生成粗模"
    return ""


def _tag(obj, key):
    try:
        return bool(obj.get(key))
    except (AttributeError, TypeError, ReferenceError):
        return False


def generated_objects(scene):
    """Owned blockouts and inferred carriers only; never infer ownership by a name."""
    generated = set()
    for coll in getattr(scene.collection, "children_recursive", ()):
        if _tag(coll, "wondful_scene_blockout") or _tag(coll, "wondful_detail_carrier"):
            generated.update(coll.all_objects)
    for obj in scene.objects:
        if any(_tag(obj, key) for key in ("wondful_sb_group", "wondful_sb_wrap", "wondful_detail_carrier",
                                         "wondful_detail_inferred", "wondful_detail_skyline", "wondful_sb_far")):
            generated.add(obj)
    return generated


@contextmanager
def isolate_environment_capture(context):
    """Hide only generated replica geometry for a single synchronous capture.

    Nested captures preserve their caller's state. Visibility is restored even
    on render failure. The product collection is never hidden. No camera changes.
    """
    props = getattr(context.scene, "wondful_ai", None)
    restore = []
    try:
        if not is_reference(props):
            product = getattr(props, "product_collection", None)
            protected = set(getattr(product, "all_objects", ()))
            for obj in generated_objects(context.scene) - protected:
                if getattr(obj, "type", "") in {"CAMERA", "LIGHT"}:
                    continue
                render_hidden = obj.hide_render
                try:
                    viewport_hidden = obj.hide_get(view_layer=context.view_layer)
                except RuntimeError:
                    viewport_hidden = None  # no object base in this view layer; already absent from OpenGL
                restore.append((obj, render_hidden, viewport_hidden))
                obj.hide_render = True
                if viewport_hidden is not None:
                    try:
                        obj.hide_set(True, view_layer=context.view_layer)
                    except RuntimeError:
                        restore[-1] = (obj, render_hidden, None)
        yield
    finally:
        for obj, render_hidden, viewport_hidden in reversed(restore):
            try:
                obj.hide_render = render_hidden
                if viewport_hidden is not None:
                    obj.hide_set(viewport_hidden, view_layer=context.view_layer)
            except (ReferenceError, RuntimeError):
                pass


def isolated_capture(fn):
    @wraps(fn)
    def call(context, *args, **kwargs):
        with isolate_environment_capture(context):
            return fn(context, *args, **kwargs)
    return call


def compare_images(props):
    pair = getattr(props, "compare_pair", "CAMERA_RESULT")
    if is_reference(props) and pair == "TARGET_BLOCKOUT":
        return getattr(props, "sb_reference_image", None), getattr(props, "sb_preview_image", None)
    if is_reference(props) and pair == "TARGET_RESULT":
        return getattr(props, "sb_reference_image", None), getattr(props, "result_image", None)
    return getattr(props, "viewport_image", None), getattr(props, "result_image", None)
