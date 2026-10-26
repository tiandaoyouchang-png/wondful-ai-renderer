"""Small native UI actions with dynamic, hover-only explanatory copy."""
import bpy
from bpy.props import StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper
from pathlib import Path
from . import workflow


def _idle(context):
    from .operators import _ACTIVE_LOCK
    return not _ACTIVE_LOCK.locked() and getattr(context.scene.wondful_ai, 'status', 'IDLE') not in workflow.BUSY


def set_target(context, raw):
    """Load before replacing: failed reads leave the last usable target intact."""
    from .operators import _load_reference_image
    path = bpy.path.abspath(raw)
    image = _load_reference_image(path)
    if not image or min(image.size) <= 0:
        raise RuntimeError('\u76ee\u6807\u56fe\u65e0\u6cd5\u8bfb\u53d6')
    props = context.scene.wondful_ai
    same_path = props.sb_image_path == path
    props.sb_image_path = path
    # RNA need not fire an update for an identical path. A replaced file is new input.
    if same_path:
        workflow.target_changed(props, context)
    props.sb_reference_image = image


class WONDFUL_OT_ui_help(Operator):
    bl_idname = 'wondful.ui_help'
    bl_label = '\u8bf4\u660e'
    bl_options = {'INTERNAL'}
    message: StringProperty(default='', options={'HIDDEN', 'SKIP_SAVE'})

    @classmethod
    def description(cls, context, properties):
        return properties.message or cls.bl_label

    def execute(self, context):
        # Keyboard / click access is retained without permanent panel prose.
        text = self.message
        def draw(menu, _ctx):
            for line in text.splitlines():
                menu.layout.label(text=line)
        context.window_manager.popup_menu(draw, title=self.bl_label, icon='INFO')
        return {'FINISHED'}


class WONDFUL_OT_target_load(Operator, ImportHelper):
    bl_idname = 'wondful.target_load'
    bl_label = '\u9009\u62e9\u590d\u523b\u76ee\u6807\u56fe'
    bl_description = '\u9009\u62e9\u6216\u66ff\u6362\u72ec\u7acb\u76ee\u6807\u56fe\uff0c\u4e0d\u6539\u53d8\u73af\u5883\u53c2\u8003\u3002\u6362\u56fe\u540e\u9700\u91cd\u65b0\u5339\u914d\u7c97\u6a21'
    filter_glob: StringProperty(default='*.png;*.jpg;*.jpeg;*.webp', options={'HIDDEN'})
    filename_ext = ''
    @classmethod
    def poll(cls, context):
        return _idle(context) and workflow.is_reference(context.scene.wondful_ai)
    def execute(self, context):
        if not self.poll(context):
            return {'CANCELLED'}
        try:
            set_target(context, self.filepath)
        except Exception as exc:
            self.report({'ERROR'}, str(exc)); return {'CANCELLED'}
        return {'FINISHED'}


class WONDFUL_OT_target_paste(Operator):
    bl_idname = 'wondful.target_paste'
    bl_label = '\u7c98\u8d34\u590d\u523b\u76ee\u6807\u56fe'
    bl_description = '\u4ece\u526a\u8d34\u677f\u8bfb\u53d6\u56fe\u7247\u5e76\u66ff\u6362\u76ee\u6807\u56fe\uff0c\u5931\u8d25\u65f6\u4fdd\u7559\u539f\u56fe'
    @classmethod
    def poll(cls, context):
        return _idle(context) and workflow.is_reference(context.scene.wondful_ai)
    def execute(self, context):
        if not self.poll(context):
            return {'CANCELLED'}
        import uuid
        from .clipboard_utils import save_clipboard_image
        from .config_manager import CONFIG_DIR
        try:
            folder = Path(CONFIG_DIR) / 'clipboard'; folder.mkdir(parents=True, exist_ok=True)
            path = save_clipboard_image(folder / ('target_' + uuid.uuid4().hex + '.png'))
            set_target(context, str(path))
        except Exception as exc:
            self.report({'ERROR'}, str(exc)); return {'CANCELLED'}
        return {'FINISHED'}


class WONDFUL_OT_target_clear(Operator):
    bl_idname = 'wondful.target_clear'
    bl_label = '\u79fb\u9664\u590d\u523b\u76ee\u6807\u56fe'
    bl_description = '\u53ea\u6e05\u7a7a\u76ee\u6807\u56fe\u9009\u62e9\uff0c\u4e0d\u5220\u9664\u786c\u76d8\u6587\u4ef6\u3001\u8f66\u6a21\u6216\u7c97\u6a21'
    @classmethod
    def poll(cls, context):
        return _idle(context) and workflow.is_reference(context.scene.wondful_ai)
    def execute(self, context):
        if not self.poll(context):
            return {'CANCELLED'}
        context.scene.wondful_ai.sb_image_path = ''
        return {'FINISHED'}


class WONDFUL_OT_set_workflow_mode(Operator):
    bl_idname = 'wondful.set_workflow_mode'
    bl_label = '切换工作模式'
    bl_description = '切换环境参考渲染与参考图复刻；不会改相机或删除场景'
    bl_options = {'INTERNAL'}
    next_mode: StringProperty(default='ENVIRONMENT', options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return _idle(context)

    def execute(self, context):
        if self.next_mode not in (workflow.ENVIRONMENT, workflow.REFERENCE):
            self.report({'ERROR'}, '无效的工作模式')
            return {'CANCELLED'}
        if not self.poll(context):
            return {'CANCELLED'}
        context.scene.wondful_ai.workflow_mode = self.next_mode
        return {'FINISHED'}


class WONDFUL_OT_preview_reference(Operator):
    bl_idname = 'wondful.preview_reference'
    bl_label = '查看完整参考图'
    bl_description = '打开完整图片预览；侧栏缩略图采用填充裁切，原始参考图不会被修改'
    bl_options = {'INTERNAL'}
    image_name: StringProperty(default='', options={'SKIP_SAVE'})
    image_path: StringProperty(default='', options={'SKIP_SAVE'})

    def _image(self):
        image = bpy.data.images.get(self.image_name) if self.image_name else None
        if image is None and self.image_path:
            path = bpy.path.abspath(self.image_path)
            if Path(path).is_file():
                image = bpy.data.images.load(path, check_existing=True)
        return image

    def invoke(self, context, event):
        if self._image() is None:
            self.report({'ERROR'}, '未找到图片')
            return {'CANCELLED'}
        return context.window_manager.invoke_popup(self, width=560)

    def draw(self, context):
        image = self._image()
        if image is None:
            self.layout.label(text='图片不可用', icon='ERROR')
            return
        self.layout.label(text=image.name, icon='IMAGE_DATA')
        # Blender's native popup preview keeps the entire image (FIT).
        self.layout.template_preview(image, show_buttons=False)
        if getattr(image, 'size', None):
            self.layout.label(text=f'{image.size[0]} × {image.size[1]} px')

    def execute(self, context):
        return {'FINISHED'}


CLASSES = (WONDFUL_OT_ui_help, WONDFUL_OT_target_load, WONDFUL_OT_target_paste,
           WONDFUL_OT_target_clear, WONDFUL_OT_set_workflow_mode, WONDFUL_OT_preview_reference)
