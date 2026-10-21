"""Real Blender translation/RNA and headless UI regression check (no AI calls).

blender -b --factory-startup --python-exit-code 1 --python tools/blender_i18n_check.py
For actual pixel screenshots, also use tools/ui_screenshots.py in GUI mode.
"""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import wondful_ai_renderer as addon
from wondful_ai_renderer import i18n, ui, properties, comparison_view, update_ops

CJK = re.compile(r'[\u4e00-\u9fff]')


class RecordingLayout:
    """Use Blender's real translation and RNA lookup, recording displayed strings.

    This does not render pixels. ui_draw_check.py separately validates the actual
    UILayout API surface against Blender RNA across its full scenario matrix.
    """
    def __init__(self, log):
        self.log = log

    def _record(self, text, context='*'):
        if text:
            self.log.append(bpy.app.translations.pgettext_iface(str(text), context))

    def __getattr__(self, name):
        if name in {'row', 'column', 'box', 'split', 'grid_flow', 'column_flow'}:
            return lambda *args, **kwargs: RecordingLayout(self.log)
        if name in {'separator', 'template_icon', 'template_ID_preview', 'template_list'}:
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def label(self, **kw):
        self._record(kw.get('text', ''))

    def operator(self, idname, **kw):
        module, name = idname.split('.')
        rna = getattr(getattr(bpy.ops, module), name).get_rna_type()
        self._record(kw.get('text', rna.name), '*' if 'text' in kw else rna.translation_context)
        return SimpleNamespace()

    def prop(self, data, attr, **kw):
        rna = data.bl_rna.properties[attr]
        self._record(kw.get('text', rna.name), rna.translation_context)
        self._record(kw.get('placeholder', ''))
        if rna.type == 'ENUM' and kw.get('expand'):
            for item in rna.enum_items:
                self._record(item.name, rna.translation_context)

    def prop_search(self, data, attr, search_data, search_attr, **kw):
        self.prop(data, attr, **kw)

    def progress(self, **kw):
        self._record(kw.get('text', ''))


def draw_panel(props):
    log = []
    holder = SimpleNamespace(layout=RecordingLayout(log))
    ui.WONDFUL_PT_main.draw(holder, bpy.context)
    return log


def assert_english(log, scenario):
    remaining = [text for text in log if CJK.search(text)]
    assert not remaining, (scenario, remaining)


def main():
    addon.register()
    try:
        # Prevent any optional updater network request during the test.
        prefs = bpy.context.preferences.addons.get(addon.__name__)
        if prefs is None:
            bpy.context.preferences.addons.new().module = addon.__name__
        prefs = bpy.context.preferences.addons[addon.__name__].preferences
        prefs.auto_check_updates = False
        view = bpy.context.preferences.view
        view.language = 'en_US'
        view.use_translate_interface = True
        view.use_translate_tooltips = True
        view.use_translate_reports = True
        for (context, source), expected in i18n.TRANSLATIONS['en_US'].items():
            assert bpy.app.translations.pgettext_iface(source, context) == expected, (context, source)
            assert bpy.app.translations.pgettext_tip(source, context) == expected, (context, source)
        assert i18n.translate_message('下载中 42%') == 'Downloading 42%'
        assert i18n.translate_message('任务已被取消', report=True) == 'Task cancelled'
        # Check the actual metadata Blender registered, including operator context.
        for cls in (properties.WONDFUL_AddonPreferences, properties.WONDFUL_AISettings):
            for prop in cls.bl_rna.properties:
                for text in (prop.name, prop.description):
                    if CJK.search(text):
                        assert not CJK.search(bpy.app.translations.pgettext_iface(text, prop.translation_context)), text
        for cls in addon.CLASSES:
            if issubclass(cls, bpy.types.Operator):
                title = cls.bl_label
                if CJK.search(title):
                    rna = getattr(getattr(bpy.ops, cls.bl_idname.split('.')[0]), cls.bl_idname.split('.')[1]).get_rna_type()
                    assert not CJK.search(bpy.app.translations.pgettext_iface(title, rna.translation_context)), (cls.__name__, rna.translation_context)
        props = bpy.context.scene.wondful_ai
        props.ui_show_advanced = True
        props.ui_show_prompt = True
        props.ui_show_terrain = True
        props.ui_show_block_refs = True
        props.prompt = 'Studio render, soft light.'
        props.scene_brief = 'Snowy morning.'
        coll = bpy.data.collections.new('Product')
        bpy.context.scene.collection.children.link(coll)
        props.product_collection = coll
        item = props.product_images.add()
        item.image = bpy.data.images.new('Product Reference', width=16, height=16)
        style = props.style_images.add()
        style.image = bpy.data.images.new('Style Reference', width=16, height=16)
        props.codex_account_state = 'LOGGED_IN'
        scenarios = 0
        states = (
            ('IDLE', '', ''),
            ('POLISHING', '分析当前风格 (1/2)', ''),
            ('RENDERING', '生成第 1/4 张 · 尝试 2/3', ''),
            ('BLOCKOUT', '场景粗模：分析参考图 · 已用 1:42', ''),
            ('ERROR', '', '场景粗模失败（分析参考图 · 已用 1:42）：未找到 Codex CLI。'),
        )
        for provider in ('CODEX', 'ANTIGRAVITY'):
            props.analysis_provider = provider
            for state, phase, error in states:
                props.status, props.task_phase, props.last_error = state, phase, error
                for tab in ui.ADV_TABS:
                    props.ui_adv_tab = tab
                    assert_english(draw_panel(props), (provider, state, tab))
                    scenarios += 1
        props.status, props.task_phase, props.last_error = 'SUCCESS', '', ''
        props.result_image = bpy.data.images.new('Result', width=16, height=16)
        props.last_detail_note = '细节：5 处合格 · 修复 2 处 · 放弃 0 · 15 秒'
        props.last_iou_note = '轮廓 IoU 0.962 · 通过（≥0.95，固定坐标）'
        props.last_layers_note = '分层输出完成 1 张（主体 + 投影，PSD + PNG），用时 3s'
        props.last_engine_note = '生图：Codex 图像编辑（直连）'
        for tab in ui.ADV_TABS:
            props.ui_adv_tab = tab
            assert_english(draw_panel(props), ('result', tab))
            scenarios += 1
        log = []
        comparison_view.WONDFUL_PT_compare_workspace.draw(SimpleNamespace(layout=RecordingLayout(log)), bpy.context)
        assert_english(log, 'comparison')
        for phase, message in (('AVAILABLE', '发现新版本 9.9.9'), ('DOWNLOADING', '下载中 42%'),
                               ('DONE', '已更新到 9.9.9，请重启 Blender'), ('FAILED', '安装失败：未找到 Codex CLI。')):
            update_ops.STATE.update(phase=phase, manifest={'version': '9.9.9', 'notes': ''}, message=message, progress=.42)
            assert_english(draw_panel(props), ('update', phase))
            scenarios += 1
        update_ops.STATE.update(phase='IDLE', manifest=None, message='')
        view.language = 'zh_HANS'
        for context, source in i18n.TRANSLATIONS['en_US']:
            assert bpy.app.translations.pgettext_iface(source, context) == source, (context, source)
        raw = '生成第 1/4 张 · 尝试 2/3'
        assert i18n.translate_message(raw) == raw
        view.language = 'en_US'
        view.use_translate_interface = False
        assert i18n.translate_message(raw) == raw
        view.use_translate_reports = False
        assert i18n.translate_message('任务已被取消', report=True) == '任务已被取消'
        view.use_translate_interface = True
        assert props.prompt == 'Studio render, soft light.'
        assert props.last_detail_note == '细节：5 处合格 · 修复 2 处 · 放弃 0 · 15 秒'
        print(f'I18N OK: {len(i18n.TRANSLATIONS["en_US"])} keys; {scenarios} English panel scenarios; Chinese and disabled-translation fallback unchanged')
    finally:
        addon.unregister()
    assert bpy.app.translations.pgettext_iface('生成') == '生成'
    addon.register()
    addon.unregister()
    print('I18N registration/unregistration/re-registration OK')


if __name__ == '__main__':
    main()
