"""Translation coverage, dynamic diagnostics, data preservation and locale regressions."""
import ast
import re
import runpy
import unittest
from pathlib import Path
from string import Formatter

ROOT = Path(__file__).resolve().parents[1]
CHECK = runpy.run_path(str(ROOT / 'tools/check_i18n.py'))
I18N = CHECK['load_catalog']()
ENTRIES = I18N.TRANSLATIONS['en_US']
CJK = re.compile(r'[\u4e00-\u9fff]')


def english(text):
    return ENTRIES.get(('*', text), text)


class WorkflowTranslationTests(unittest.TestCase):
    def test_declared_translation_surface_is_complete(self):
        found, missing = CHECK['audit']()
        self.assertGreater(len(found), 1000)
        self.assertEqual(missing, [])

    def test_coverage_detects_new_untranslated_ui_and_metadata(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ui.py'
            path.write_text('row.label(text="新按钮")\nx = StringProperty(description="新说明")\n')
            messages = list(CHECK['collect_messages'](Path(directory)))
            self.assertEqual({source for _ctx, source, _loc in messages}, {'新按钮', '新说明'})
            self.assertTrue(all((ctx, source) not in ENTRIES for ctx, source, _ in messages))

    def test_placeholder_names_and_counts_are_preserved(self):
        for (context, source), translated in ENTRIES.items():
            with self.subTest(context=context, source=source):
                fields = lambda text: sorted((field, spec, conv) for _, field, spec, conv in Formatter().parse(text) if field)
                self.assertEqual(fields(source), fields(translated))
                self.assertFalse(CJK.search(translated))

    def test_every_legacy_template_matches_a_formatted_message(self):
        # This exercises the actual full-message matcher, not just dictionary membership.
        for pattern, source in I18N._message_patterns():
            values = {field: '23' for _, field, _, _ in Formatter().parse(source) if field}
            raw = source.format(**values)
            self.assertIsNotNone(pattern.fullmatch(raw), source)
            result = I18N.translate_message(raw, translator=english)
            self.assertFalse(CJK.search(result), (source, result))

    def test_nested_error_and_elapsed_time(self):
        raw = '场景粗模失败（分析参考图 · 已用 1:42）：未找到 Codex CLI。'
        self.assertEqual(I18N.translate_message(raw, translator=english),
                         'Scene blockout failed (Analyze reference image · elapsed 1:42): Codex CLI not found.')

    def test_composed_summary_and_numeric_fields_do_not_swallow_clauses(self):
        raw = '细节：5 处合格 · 修复 2 处 · 放弃 0 · 15 秒'
        self.assertEqual(I18N.translate_message(raw, translator=english),
                         'Details: 5 passed · Repaired 2 · Abandoned 0 · 15s')
        for raw in ('下载中 42%', '生成第 1/4 张 · 尝试 2/3',
                    '轮廓 IoU 0.962 · 通过（≥0.95，固定坐标）'):
            self.assertFalse(CJK.search(I18N.translate_message(raw, translator=english)))

    def test_legacy_composed_reference_and_retry_summaries(self):
        cases = {
            '大楼：建筑参考 3 张（每个体块最多 3 张，2 张未添加）':
                '大楼: 3 building references (at most 3 per block; 2 not added)',
            '多角度场景一致：2/2 个相机完成 · 图像调用 3 次 · 重试：产品相机（偏移 5.2px、颜色 ΔE 20）':
                'Consistent views: 2/2 cameras completed · 3 image calls · Retried: 产品相机 (Offset 5.2px, Color ΔE 20)',
            '分层输出完成 1 张（主体 + 投影，PSD + PNG），用时 3s；还有 2 张处理中':
                'Layered output complete: 1 images (Subject + shadow, PSD + PNG), 3s; 2 more images processing',
            '细节边缘 2.1px（回贴后 1.2px）':
                'Detail edges 2.1px (after restoration 1.2px)',
        }
        for raw, expected in cases.items():
            self.assertEqual(I18N.translate_message(raw, translator=english), expected)

    def test_names_paths_and_unknown_external_diagnostics_are_opaque(self):
        path = '/tmp/产品 图像 {v0} · 测试.png'
        self.assertEqual(I18N.translate_message('参考图已不存在：' + path, translator=english),
                         'Reference image no longer exists: ' + path)
        self.assertEqual(I18N.translate_message('视角 1/2 · 产品相机（主视角）', translator=english),
                         'View 1/2 · 产品相机 (main view)')
        raw = '外部服务返回了一段未登记的中文诊断 {some_field}'
        self.assertEqual(I18N.translate_message(raw, translator=english), raw)

    def test_chinese_other_locales_and_disabled_translations_are_exact(self):
        for source in ('生成第 1/4 张 · 尝试 2/3', '细节：5 处合格 · 修复 2 处',
                       '场景粗模失败（分析参考图 · 已用 1:42）：未找到 Codex CLI。'):
            self.assertEqual(I18N.translate_message(source, translator=lambda text: text), source)
        self.assertEqual(set(I18N.TRANSLATIONS), {'en_US'})

    def test_ui_translates_before_wrapping_and_preserves_prompt_preview(self):
        src = (ROOT / 'wondful_ai_renderer/ui.py').read_text()
        self.assertIn('_wrap(_message(text), cells)', src)
        self.assertIn('preview.label(text=line or " ")', src)
        # No Chinese f-string is formatted first in draw code.
        for filename in ('ui.py', 'comparison_view.py'):
            tree = ast.parse((ROOT / 'wondful_ai_renderer' / filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.JoinedStr):
                    self.assertFalse(any(isinstance(value, ast.Constant) and CJK.search(str(value.value))
                                         for value in node.values), (filename, node.lineno))
        operator = (ROOT / 'wondful_ai_renderer/operators.py').read_text()
        report = operator[operator.index('    def report(self, type, message, *, limit=None):'):operator.index('    def _start_thread')]
        self.assertLess(report.index('_LAST_REPORT[self.bl_idname] = str(message)'), report.index('_message(message, report=True)'))

    def test_operator_and_default_contexts(self):
        for source in ('AI渲染', '生成', '编辑完整提示词', '检查更新', '打开渲染对比'):
            self.assertIn(('Operator', source), ENTRIES)
            self.assertIn(('*', source), ENTRIES)
        for source in ('产品集合', '自动', '快速预览', 'Wondful AI 渲染器 · 1.2', '渲染对比'):
            self.assertIn(('*', source), ENTRIES)


if __name__ == '__main__':
    unittest.main()
