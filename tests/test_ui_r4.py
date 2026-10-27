"""Targeted offline contracts for UI-r4. Native Blender visual QA is separate."""
import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import test_two_workflows as T
import test_ui_revision as R

ROOT = T.ROOT
PKG = T.PKG
BV = T.load('blockout_versions')
UIC = T.load('ui_components')
CACHE = T.load('preview_cache')


def setUpModule():
    T.setUpModule()


def tearDownModule():
    T.tearDownModule()


class Tagged(dict):
    def __init__(self, name, *, objects=(), children=(), version=None, **tags):
        super().__init__(tags)
        self.name = name
        self.hide_render = False
        self.hide_viewport = False
        self.hidden = False
        self.type = 'COLLECTION'
        self.objects = list(objects)
        self.children = list(children)
        self.children_recursive = list(children)
        self.all_objects = list(objects)
        for child in children:
            self.children_recursive.extend(child.children_recursive)
            self.all_objects.extend(child.all_objects)
        if version is not None:
            self[BV.TAG] = 1
            self[BV.VERSION] = version
            self[BV.ACTIVE] = 0

    def hide_get(self):
        return self.hidden

    def hide_set(self, value):
        self.hidden = value


class RevisionHistoryTests(unittest.TestCase):
    def make(self):
        old_obj = Tagged('old geometry', type='MESH')
        old_obj.type = 'MESH'
        old_camera = Tagged('old generated camera'); old_camera.type = 'CAMERA'
        old = Tagged(BV.version_name(1), objects=[old_obj, old_camera], version=1)
        old[BV.ACTIVE] = 1
        new_obj = Tagged('new geometry'); new_obj.type = 'MESH'
        new = Tagged(BV.version_name(2), objects=[new_obj], version=2)
        scene = NS(collection=NS(children_recursive=[old, new]))
        props = NS(sb_active_collection=old)
        return scene, props, old, new, old_obj, old_camera, new_obj

    def test_generation_versions_are_numbered_without_reusing_names(self):
        scene, _props, old, new, *_ = self.make()
        self.assertEqual(BV.next_number(scene), 3)
        self.assertEqual(BV.version_name(3), 'Wondful 场景粗模 · 003')
        old[BV.VERSION] = 'broken'
        self.assertEqual(BV.next_number(scene), 3)  # names are fallback

    def test_activate_hides_but_does_not_delete_previous_collection_or_objects(self):
        scene, props, old, new, old_obj, old_camera, new_obj = self.make()
        self.assertIs(BV.activate(scene, new, props), new)
        self.assertIs(props.sb_active_collection, new)
        self.assertEqual(scene.collection.children_recursive, [old, new])
        self.assertEqual(old.all_objects, [old_obj, old_camera])
        self.assertTrue(old.hide_render)
        self.assertTrue(old.hide_viewport)
        self.assertTrue(old_obj.hide_render)
        self.assertTrue(old_obj.hidden)
        self.assertFalse(old_camera.hide_render)
        self.assertTrue(new[BV.ACTIVE])
        self.assertFalse(new_obj.hide_render)
        self.assertIs(BV.active(scene, props), new)

    def test_archived_versions_can_be_activated_again_without_losing_flags(self):
        scene, props, old, new, old_obj, _old_camera, new_obj = self.make()
        old_obj.hide_render = True
        old_obj.hidden = False
        BV.activate(scene, new, props)
        BV.activate(scene, old, props)
        self.assertFalse(old.hide_render)
        self.assertFalse(old.hide_viewport)
        self.assertTrue(old_obj.hide_render)  # original flag, not forced visible
        self.assertFalse(old_obj.hidden)
        self.assertTrue(new.hide_render)
        self.assertTrue(new_obj.hide_render)
        self.assertEqual(len(scene.collection.children_recursive), 2)

    def test_reject_an_unrelated_collection_for_activation(self):
        scene, props, *_ = self.make()
        with self.assertRaises(ValueError):
            BV.activate(scene, Tagged('Manual collection'), props)

    def test_generated_camera_is_grouped_inside_its_own_version(self):
        tree = ast.parse((PKG / 'scene_blockout_build.py').read_text(encoding='utf-8'))
        build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build')
        source = ast.unparse(build)
        self.assertIn('coll.objects.link(cam_obj)', source)
        self.assertNotIn('scene.collection.objects.link(cam_obj)', source)
        self.assertIn('BV.activate(scene, coll', source)

    def test_destruction_is_not_in_any_new_generation_path(self):
        checks = (
            ('scene_blockout_build.py', 'build', 'remove_existing'),
            ('detail_blockout.py', 'build_in_blender', 'remove_carrier'),
            ('detail_blockout.py', 'complete_llm_in_blender', 'remove_inferred'),
            ('operators.py', '_finish', 'remove_inferred'),
        )
        for file, fn, banned in checks:
            tree = ast.parse((PKG / file).read_text(encoding='utf-8'))
            candidates = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn]
            self.assertTrue(candidates, (file, fn))
            for candidate in candidates:
                calls = [ast.unparse(n.func) for n in ast.walk(candidate) if isinstance(n, ast.Call)]
                self.assertNotIn(banned, calls, (file, fn))
                self.assertNotIn('DB.' + banned, calls, (file, fn))

    def test_detail_lookup_prefers_current_version_not_archived_history(self):
        db = T.load('detail_blockout')
        old_detail = Tagged('old carrier', objects=[object()], **{db.CARRIER_TAG:1})
        new_detail = Tagged('new carrier', objects=[object()], **{db.CARRIER_TAG:1})
        old_inferred = Tagged('old inferred', objects=[object()], **{db.INFER_TAG:1,
            db.LAYOUT_KEY:'{"scene":"stale"}'})
        new_inferred = Tagged('new inferred', objects=[object()], **{db.INFER_TAG:1,
            db.LAYOUT_KEY:'{"scene":"current"}'})
        old = Tagged(BV.version_name(1), children=[old_detail, old_inferred], version=1)
        now = Tagged(BV.version_name(2), children=[new_detail, new_inferred], version=2)
        old.hide_render = True
        now[BV.ACTIVE] = 1
        scene = NS(collection=NS(children_recursive=[old, now]), wondful_ai=NS(sb_active_collection=now))
        self.assertIs(db.find_carrier(scene), new_detail)
        self.assertIs(db.find_inferred(scene), new_inferred)
        self.assertEqual(db.completion_layout(scene)['scene'], 'current')
        now.children_recursive = []
        self.assertIsNone(db.find_carrier(scene))
        self.assertIsNone(db.find_inferred(scene))
        self.assertIsNone(db.completion_layout(scene))


class ProgressLayout(R.RecordingLayout):
    def child(self, *args, **kwargs):
        return ProgressLayout(self.log, self)
    row = column = box = split = grid_flow = child
    def progress(self, *args, **kw):
        self.log.append(('progress', kw.get('factor'), kw.get('text'), self.effective_enabled()))


class ModeAndProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ui = T.ui_module()

    def draw(self, props, width=360):
        context = T.context(props)
        context.region.width = width
        layout = ProgressLayout()
        with patch.object(T.bpy, 'context', context, create=True):
            self.ui.WONDFUL_PT_main._draw_main(NS(layout=layout), context)
        return layout.log

    def test_mode_buttons_have_visible_labels_even_in_narrow_sidebar(self):
        for width in (240, 360, 640):
            for mode in ('REFERENCE', 'ENVIRONMENT'):
                log = self.draw(T.props(workflow_mode=mode), width)
                labels = [x[3] for x in log if x[:2] == ('operator', 'wondful.set_workflow_mode')]
                self.assertEqual(labels, ['环境渲染', '参考图复刻'])

    def test_build_has_live_progress_and_enabled_cancel_while_inputs_locked(self):
        props = T.props(workflow_mode='REFERENCE', status='BLOCKOUT', progress=.4,
            task_phase='分析参考图', sb_elapsed=95, sb_image_path='/tmp/reference.png',
            sb_reference_image=T.FakeImage('Ref'),
            codex_account_state='LOGGED_IN', sb_stage='ANALYZE')
        log = self.draw(props)
        self.assertTrue(any(x[0]=='progress' and abs(x[1]-.4)<.001 for x in log))
        cancel = [x for x in log if x[:2]==('operator', 'wondful.cancel_task')]
        self.assertTrue(cancel)
        self.assertTrue(any(x[2] for x in cancel), cancel)
        disabled = [x for x in log if x[:2]==('operator','wondful.target_load')]
        self.assertTrue(disabled)
        self.assertFalse(any(x[2] for x in disabled))
        self.assertTrue(any(x[0]=='label' and '工作模式' in x[1] for x in log))

    def test_compact_toggle_row_and_optional_advanced_settings(self):
        for expanded in (False, True):
            p = T.props(workflow_mode='REFERENCE', ui_show_replica_advanced=expanded)
            log = self.draw(p)
            names = [x[1] for x in log if x[0]=='prop']
            self.assertNotIn('sb_set_camera', names)
            self.assertIn('sb_correct', names)
            self.assertTrue(any(x[0]=='label' and '自动匹配新参考相机' in x[1] for x in log))
            self.assertEqual('sb_precise' in names, expanded)
            self.assertEqual('sb_complete' in names, expanded)


class SeamlessPreviewTests(unittest.TestCase):
    def test_wide_identity_and_environment_use_one_icon_only(self):
        for w, h in ((2400, 900), (1920, 1080), (800, 1600)):
            image = T.FakeImage(width=w, height=h)
            self.assertEqual(UIC.preview_tile_count(image, 'contain'), 1)
            self.assertEqual(UIC.preview_tile_count(image, 'cover'), 1)
        c = T.context()
        for fit in ('contain', 'cover'):
            layout = R.RecordingLayout()
            with patch.object(CACHE, 'icon_for_image', return_value=42) as icon_for:
                UIC.preview(layout, c, T.FakeImage(width=2400, height=900), fit=fit)
            thumbnails = [x for x in layout.log if x[0] == 'thumbnail']
            self.assertEqual(len(thumbnails), 1)
            self.assertEqual(icon_for.call_args.kwargs['fit'], 'seamless')
            self.assertEqual(thumbnails[0][1], UIC.preview_scale(c))

    def test_original_pixel_range_and_both_vehicle_ends_preserved(self):
        import numpy as np
        w, h = 800, 300
        px = np.empty((h, w, 4), dtype=np.float32)
        px[:, :, 0] = np.linspace(.1, .9, w)[None, :]
        px[:, :, 1] = .3
        px[:, :, 2] = .6
        px[:, :, 3] = 1
        before = px.copy()
        image = NS(size=(w, h), pixels=px.ravel())
        thumb = NS()
        CACHE._pixels_thumbnail(image, thumb, 'seamless')
        actual = np.asarray(thumb.image_pixels_float).reshape(256, 256, 4)
        self.assertEqual(thumb.image_size, (256, 256))
        self.assertTrue(np.array_equal(image.pixels.reshape(h, w, 4), before))
        self.assertTrue(np.all(actual[:, :, 3] == 1.0))
        # The full source spans the middle row of the one-piece thumbnail.
        self.assertLess(actual[128, 1, 0], .12)
        self.assertGreater(actual[128, 254, 0], .88)
        # No UI divider or blank black band down the image's center.
        self.assertGreater(actual[128, 127, :3].sum(), .7)
        self.assertGreater(actual[128, 128, :3].sum(), .7)
        self.assertLess(abs(actual[128, 127, 0] - actual[128, 128, 0]), .02)
        # The letterbox area is a blurred extension, not a black frame.
        self.assertGreater(float(actual[10, 128, :3].sum()), .3)

    def test_single_icon_for_all_upload_and_advanced_path_cards(self):
        c = T.context()
        image = T.FakeImage(width=1920, height=1080)
        p = T.props(sb_reference_image=image)
        for kind in ('PRODUCT', 'STYLE', 'TARGET', 'PERSON'):
            layout = R.RecordingLayout()
            with patch.object(CACHE, 'icon_for_image', return_value=42):
                UIC.image_slot(layout, c, p, kind, item=NS(image=image, instruction=''), title=kind)
            self.assertEqual(sum(x[0] == 'thumbnail' for x in layout.log), 1, kind)
        layout = R.RecordingLayout()
        with patch.object(CACHE, 'icon_for_path', return_value=42) as icon_for:
            UIC.path_preview(layout, c, '/tmp/studio.png', '建筑参考')
        self.assertEqual(sum(x[0] == 'thumbnail' for x in layout.log), 1)
        self.assertEqual(icon_for.call_args.kwargs['fit'], 'seamless')
        self.assertTrue(any(x[:2] == ('operator', 'wondful.preview_reference') for x in layout.log))

    def test_path_preview_dimensions_access_is_io_free(self):
        path='/reference.png'
        CACHE._records.clear()
        with patch.object(CACHE.os,'stat',side_effect=AssertionError('read in Panel.draw')):
            self.assertIsNone(CACHE.cached_dimensions_for_path(path))
            CACHE._records['path:seamless:'+path]={'size':(1600,900)}
            self.assertEqual(CACHE.cached_dimensions_for_path(path),(1600,900))
        CACHE._records.clear()

    def test_tiles_are_no_longer_an_accepted_preview_mode(self):
        import numpy as np
        image = NS(size=(3, 2), pixels=np.ones(3 * 2 * 4, dtype=np.float32))
        with self.assertRaises(ValueError):
            CACHE._pixels_thumbnail(image, NS(), 'tile:0:2')


if __name__=='__main__':
    unittest.main()
