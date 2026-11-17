"""Single-reference request contracts; mocked providers, no Blender or network."""
import base64
import json
import hashlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import test_two_workflows as T
from test_316_regressions import load_package_module

CD = load_package_module('codex_direct')
CC = load_package_module('codex_client')
AGY = load_package_module('antigravity_client')
PNG = load_package_module('png_io')


class SourceTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.image('source.png', 25)
        self.identity = self.image('identity.png', 190)
        self.out = str(self.root / 'out.png')
        self.brief = 'KEEP_ALL_POLISHED_TEXT_' * 300
        self.SR = load_package_module("source_reference")
        SR = self.SR
        self.prompt = SR.edit_prompt(self.brief, 'blue metal')

    def image(self, name, value):
        path = self.root / name
        PNG.write_png(str(path), np.full((24, 48, 3), value, np.uint8))
        return str(path)

    def direct(self, reported_quality='', **extra):
        args = dict(prompt=self.prompt, edit_base_path=self.source,
                    reference_paths=[self.source, self.identity], output_path=self.out,
                    size='1024x512', log_dir=str(self.root), source_only=True)
        args.update(extra)
        with ExitStack() as stack:
            for name, value in (('find_auth_file', self.root / 'auth'), ('load_auth', {}),
                                ('token_expiring', False), ('model_chain', ['model']),
                                ('available_models', ['model']), ('auth_headers', {})):
                stack.enter_context(patch.object(CD, name, return_value=value))
            def response(*_args, **kwargs):
                kwargs['meta']['item_quality'] = reported_quality
                return base64.b64encode(Path(self.source).read_bytes()).decode(), {}, 'response'
            post = stack.enter_context(patch.object(CD, 'post_responses', side_effect=response))
            result = CD.edit_image(**args)
            post.assert_called_once()
        return post.call_args.args[2], result

    def wide_identity(self, width=1536, height=512):
        y, x = np.indices((height, width))
        pixels = np.stack((x % 251, y % 253, (x * 7 + y * 11) % 255), axis=-1).astype(np.uint8)
        PNG.write_png(self.identity, pixels)
        return pixels

    def test_spatial_crops_equal_original_pixels_on_worker_thread(self):
        pixels = self.wide_identity()
        originals = [Path(p).read_bytes() for p in (self.source, self.identity)]
        images = self.SR.validate_inputs(self.source, self.identity, 5)
        completed, errors = [], []
        def worker():
            try:
                completed.append(self.SR.append_identity_crops(images, self.root))
            except Exception as exc:
                errors.append(exc)
        with patch.dict(sys.modules, {'bpy': None}):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        appended, info = completed[0]
        self.assertEqual(appended[:2], images)
        self.assertEqual([Path(p).read_bytes() for p in (self.source, self.identity)], originals)
        self.assertEqual(info['source']['sha256'], hashlib.sha256(originals[1]).hexdigest())
        self.assertEqual(info['source_size'], [1536, 512])
        self.assertEqual(len(appended), 5)
        self.assertEqual([crop['image_number'] for crop in info['crops']], [3, 4, 5])
        for crop in info['crops']:
            x0, y0, x1, y1 = crop['bounds']
            actual = PNG.read_png(crop['path'])
            np.testing.assert_array_equal(actual[:, :, :3], pixels[y0:y1, x0:x1])
            self.assertTrue(np.all(actual[:, :, 3] == 255))
            self.assertEqual(crop['size'], [actual.shape[1], actual.shape[0]])
            self.assertEqual(crop['role'], 'product_detail')
            self.assertEqual(crop['sha256'], hashlib.sha256(Path(crop['path']).read_bytes()).hexdigest())
        for left, right in zip(info['crops'], info['crops'][1:]):
            self.assertLess(right['bounds'][0], left['bounds'][2])

    def test_crops_respect_remaining_capacity_without_replacing_any_input(self):
        self.wide_identity()
        originals = self.SR.validate_inputs(self.source, self.identity, 5)
        for occupied in (2, 3, 4, 5):
            with self.subTest(occupied=occupied):
                images = originals + [(self.source, 'environment', 'other')] * (occupied - 2)
                output, info = self.SR.append_identity_crops(images, self.root, limit=5)
                self.assertEqual(output[:occupied], images)
                self.assertEqual(len(output), 5)
                self.assertEqual(len(info['crops']), 5 - occupied)
                self.assertEqual([c['image_number'] for c in info['crops']], list(range(occupied + 1, 6)))
                if len(info['crops']) > 1:
                    self.assertLess(info['crops'][1]['bounds'][0], info['crops'][0]['bounds'][2])

    def test_ordinary_aspects_and_tiny_wide_inputs_get_no_supplement(self):
        for width, height in ((1024, 1024), (1200, 800), (768, 256), (1536, 200)):
            with self.subTest(size=(width, height)):
                PNG.write_png(self.identity, np.zeros((height, width, 3), np.uint8))
                images = self.SR.validate_inputs(self.source, self.identity, 5)
                with patch.object(load_package_module('reference_collage'), 'read_rgb') as reader:
                    output, info = self.SR.append_identity_crops(images, self.root)
                reader.assert_not_called()
                self.assertEqual(output, images)
                self.assertEqual(info['crops'], [])

    def test_identity_change_during_decoding_stops_before_request(self):
        pixels = self.wide_identity()
        def changing(_path):
            PNG.write_png(self.identity, pixels ^ 255)
            return pixels
        with patch.object(load_package_module('reference_collage'), 'read_rgb', side_effect=changing), \
             patch.object(CD, 'post_responses') as post:
            with self.assertRaisesRegex(CD.DirectEditError, '裁切时发生变化'):
                self.direct()
        post.assert_not_called()

    def test_actual_direct_single_modes_keep_originals_roles_and_complete_prompt(self):
        self.wide_identity()
        originals = [Path(p).read_bytes() for p in (self.source, self.identity)]
        manifest = [{'path': self.identity, 'members': [{'path': self.identity, 'role': 'product'}]}]
        for mode in ('source', 'camera_full', 'camera_simple'):
            with self.subTest(mode=mode):
                options = {} if mode == 'source' else dict(source_only=False, camera_capture=True, manifest=manifest)
                if mode == 'camera_simple':
                    options.update(simple_brief=self.brief, identity_text='user-custom turquoise dial')
                payload, result = self.direct(reported_quality='medium', **options)
                if mode == 'source':
                    self.assertEqual(payload['model'], 'gpt-image-2')
                    self.assertEqual(payload['n'], 1)
                    self.assertEqual(payload['quality'], 'high')
                    self.assertEqual(payload['background'], 'opaque')
                    self.assertNotIn('tools', payload)
                    self.assertNotIn('reasoning', payload)
                    self.assertNotIn('complexity', payload)
                    self.assertNotIn('input', payload)
                    self.assertNotIn('mask', payload)
                    images = [base64.b64decode(item['image_url'].split(',', 1)[1]) for item in payload['images']]
                    text = payload['prompt']
                else:
                    images = [base64.b64decode(c['image_url'].split(',', 1)[1]) for c in
                              payload['input'][0]['content'] if c['type'] == 'input_image']
                    self.assertNotIn('input_image_mask', payload['tools'][0])
                    self.assertEqual(payload['tools'][0]['quality'], 'high')
                    text = payload['input'][0]['content'][0]['text']
                self.assertEqual(images[:2], originals)
                self.assertEqual(len(images), 5)
                self.assertEqual(result['image_roles'],
                                 ['source_canvas' if mode == 'source' else 'camera', 'product'] + ['product_detail'] * 3)
                self.assertFalse(result['mask'])
                self.assertEqual(result['image_quality'], 'medium')
                self.assertEqual(text.count(self.brief), 1)
                self.assertIn('用户明确的外观定制优先', text)
                self.assertIn('不臆造隐藏面', text)
                self.assertNotIn('YU7', text)
                for crop, raw in zip(result['identity_details']['crops'], images[2:]):
                    self.assertEqual(raw, Path(crop['path']).read_bytes())
                    self.assertIn(f"图 {crop['image_number']}：图 2 的像素区域 {crop['bounds']}", text)
                if mode != 'source':
                    self.assertEqual(payload['instructions'], self.SR.CAMERA_EDIT_INSTRUCTIONS)
                    self.assertIn('projection, silhouette', payload['instructions'])
                    self.assertIn('visible fine-feature topology', payload['instructions'])
                    self.assertIn('产品投影', text)
                    self.assertNotIn('只允许改变', text)
                log = json.loads((self.root / 'codex_direct.json').read_text())
                self.assertEqual(log['identity_details'], result['identity_details'])

    def test_wide_identity_fallback_still_sends_only_two_originals(self):
        self.wide_identity()
        calls = []
        def exec_turn(**kwargs):
            calls.append(kwargs)
            Path(self.out).write_bytes(Path(self.source).read_bytes())
            return 'WONDFUL_IMAGEGEN_OK'
        args = dict(prompt=self.prompt, reference_paths=[self.source, self.identity],
                    edit_base_path=self.source, output_path=self.out, cwd=str(self.root), timeout=30)
        with ExitStack() as stack:
            for name, value in (('find_auth_file', self.root / 'auth'), ('load_auth', {}),
                                ('token_expiring', False), ('model_chain', ['model']),
                                ('available_models', ['model']), ('auth_headers', {})):
                stack.enter_context(patch.object(CD, name, return_value=value))
            post = stack.enter_context(patch.object(CD, 'post_responses', side_effect=CD.DirectEditError(
                'endpoint', kind='endpoint', fatal=True)))
            stack.enter_context(patch.object(CC, '_exec_turn', side_effect=exec_turn))
            CD.generate_with_fallback(args, fallback=CC.generate_image, state={}, size='1024x512', source_only=True)
        post.assert_called_once()
        payload = post.call_args.args[2]
        self.assertEqual(payload['model'], 'gpt-image-2')
        self.assertEqual(payload['n'], 1)
        self.assertNotIn('tools', payload)
        self.assertNotIn('complexity', payload)
        self.assertNotIn('mask', payload)
        self.assertNotIn('input', payload)
        direct_images = payload['images']
        self.assertEqual(len(direct_images), 5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['image_paths'], [self.source, self.identity])
        self.assertNotIn('像素区域', calls[0]['prompt'])

    def test_camera_capacity_retains_environment_before_detail_supplements(self):
        self.wide_identity()
        environment = self.image('environment.png', 90)
        manifest = [{'path': p, 'members': [{'path': p, 'role': role}]}
                    for p, role in ((self.identity, 'product'), (environment, 'environment'))]
        payload, result = self.direct(source_only=False, camera_capture=True, manifest=manifest,
                                      reference_paths=[self.source, self.identity, environment])
        images = [base64.b64decode(c['image_url'].split(',', 1)[1]) for c in
                  payload['input'][0]['content'] if c['type'] == 'input_image']
        self.assertEqual(images[:3], [Path(p).read_bytes() for p in (self.source, self.identity, environment)])
        self.assertEqual(result['image_roles'], ['camera', 'product', 'environment', 'product_detail', 'product_detail'])
        self.assertEqual(result['dropped_images'], [])
        self.assertEqual([c['image_number'] for c in result['identity_details']['crops']], [4, 5])

    def test_wide_identity_is_untouched_on_hybrid_and_legacy_direct(self):
        self.wide_identity()
        guide = self.image('projection.png', 95)
        manifest = [{'path': self.identity, 'members': [{'path': self.identity, 'role': 'product'}]}]
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid), patch.object(CD, 'append_identity_crops') as crops:
                options = dict(source_only=False, manifest=manifest)
                if hybrid:
                    options.update(original_first=True, scene_reference_path=self.source,
                                   product_projection_path=guide, reference_paths=[guide, self.identity, self.source])
                payload, result = self.direct(**options)
                crops.assert_not_called()
                self.assertEqual(result['identity_details'], {})
                self.assertEqual(result['attached_images'], 3 if hybrid else 2)
                self.assertNotIn('像素区域', payload['input'][0]['content'][0]['text'])

    def test_projected_direct_preserves_primary_three_then_two_exact_crops(self):
        pixels = self.wide_identity()
        guide = self.image('actual_model.png', 95)
        self.prompt = self.SR.edit_prompt(self.brief, 'custom appearance', projection=True)
        payload, result = self.direct(reference_paths=[self.source, guide, self.identity], product_projection_path=guide)
        self.assertEqual(payload['model'], 'gpt-image-2')
        self.assertEqual(payload['n'], 1)
        self.assertEqual(payload['quality'], 'high')
        self.assertNotIn('tools', payload)
        self.assertNotIn('complexity', payload)
        self.assertNotIn('mask', payload)
        self.assertNotIn('input', payload)
        raw = [base64.b64decode(item['image_url'].split(',', 1)[1]) for item in payload['images']]
        self.assertEqual(raw[:3], [Path(p).read_bytes() for p in (self.source, guide, self.identity)])
        self.assertEqual(result['image_roles'], ['source_canvas', 'product_projection', 'product', 'product_detail', 'product_detail'])
        self.assertFalse(result['mask'])
        self.assertEqual(result['identity_details']['source_image_number'], 3)
        text = payload['prompt']
        self.assertIn(self.SR.PROJECTED_EDIT_INSTRUCTIONS, text)
        self.assertIn('must not redesign modeled parts', text)
        self.assertEqual(text.count(self.brief), 1)
        self.assertIn('不得用身份图或其裁切重设计', text)
        for crop in result['identity_details']['crops']:
            x0, y0, x1, y1 = crop['bounds']
            np.testing.assert_array_equal(PNG.read_png(crop['path'])[:, :, :3], pixels[y0:y1, x0:x1])
            self.assertIn(f"图 {crop['image_number']}：图 3 的像素区域", text)

    def test_projected_actual_cli_fallback_keeps_primary_three(self):
        guide=self.image('actual_model.png',95)
        calls=[]
        def exec_turn(**kwargs):
            calls.append(kwargs)
            Path(self.out).write_bytes(Path(self.source).read_bytes())
            return 'WONDFUL_IMAGEGEN_OK'
        args=dict(prompt=self.SR.edit_prompt(self.brief,projection=True),reference_paths=[self.source,guide,self.identity],
                  product_projection_path=guide,edit_base_path=self.source,output_path=self.out,cwd=str(self.root),timeout=30)
        with patch.object(CD,'edit_image',side_effect=CD.DirectEditError('endpoint',kind='request')) as direct, \
             patch.object(CC,'_exec_turn',side_effect=exec_turn):
            CD.generate_with_fallback(args,fallback=CC.generate_image,state={},size='1024x512',source_only=True)
        self.assertEqual(direct.call_args.kwargs['product_projection_path'],guide)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['image_paths'],[self.source,guide,self.identity])
        self.assertIn('不得用身份图或其裁切重设计',calls[0]['prompt'])
        self.assertEqual(calls[0]['prompt'].count(self.brief),1)

    def test_projected_antigravity_stages_all_three_without_role_drop(self):
        guide=self.image('actual_model.png',95)
        calls=[]
        def run(cli,args,**kwargs):
            task=args[args.index('--prompt')+1];calls.append(task)
            paths=__import__('ast').literal_eval(task.split('ImagePaths = ',1)[1].splitlines()[0])
            self.assertEqual([Path(p).read_bytes() for p in paths],
                             [Path(p).read_bytes() for p in (self.source,guide,self.identity)])
            Path(self.out).write_bytes(Path(self.source).read_bytes())
            event={'event':'step_update','step_update':{'step_type':'tool','tool_name':'generate_image',
                   'tool_info':{'output':{'path':self.out}}}}
            return subprocess.CompletedProcess(args,0,stdout=json.dumps(event)+'\n'+json.dumps(
                {'status':'SUCCESS','response':'WONDFUL_ANTIGRAVITY_IMAGEGEN_OK'}),stderr='')
        with patch.object(AGY,'discover_antigravity_cli',return_value='/mock/agy'),patch.object(AGY,'_run',side_effect=run):
            AGY.generate_image(prompt=self.SR.edit_prompt(self.brief,projection=True),reference_paths=[self.source,guide,self.identity],
                               product_projection_path=guide,edit_base_path=self.source,output_path=self.out,cwd=str(self.root),source_only=True)
        self.assertEqual(len(calls),1)
        self.assertIn('不得用身份图或其裁切重设计',calls[0])

    def test_projected_missing_or_reordered_primary_stops_before_provider(self):
        guide=self.image('actual_model.png',95)
        for refs,path in (([self.source,self.identity],guide),([self.source,self.identity,guide],guide),
                          ([self.source,guide,self.identity],str(self.root/'missing.png'))):
            with self.subTest(refs=refs,path=path),patch.object(CD,'find_auth_file') as auth:
                with self.assertRaises(CD.DirectEditError):
                    self.direct(reference_paths=refs,product_projection_path=path)
                auth.assert_not_called()
        with self.assertRaises(ValueError):
            self.SR.validate_inputs(self.source,self.identity,2,projection=guide)

    def test_direct_payload_source_identity_and_complete_prompt(self):
        payload, result = self.direct()
        self.assertEqual(payload['model'], 'gpt-image-2')
        self.assertEqual(payload['n'], 1)
        self.assertEqual(payload['quality'], 'high')
        self.assertNotIn('tools', payload)
        self.assertNotIn('complexity', payload)
        self.assertNotIn('mask', payload)
        self.assertNotIn('input', payload)
        images = [base64.b64decode(item['image_url'].split(',', 1)[1]) for item in payload['images']]
        self.assertEqual(images, [Path(self.source).read_bytes(), Path(self.identity).read_bytes()])
        self.assertEqual(result['image_roles'], ['source_canvas', 'product'])
        self.assertFalse(result['mask'])
        text = payload['prompt']
        self.assertIn(self.SR.EDIT_INSTRUCTIONS, text)
        self.assertEqual(text.count(self.brief), 1)
        self.assertNotIn('product_projection', text)
        self.assertNotIn('白模', text)

    def test_codex_actual_cli_fallback_preserves_inputs_and_prompt(self):
        calls = []
        def exec_turn(**kwargs):
            calls.append(kwargs)
            Path(self.out).write_bytes(Path(self.source).read_bytes())
            return 'WONDFUL_IMAGEGEN_OK'
        args = dict(prompt=self.prompt, reference_paths=[self.source, self.identity],
                    edit_base_path=self.source, edit_mask_path='', output_path=self.out,
                    cwd=str(self.root), timeout=30)
        with patch.object(CD, 'edit_image', side_effect=CD.DirectEditError('endpoint', kind='request')), \
             patch.object(CC, '_exec_turn', side_effect=exec_turn):
            CD.generate_with_fallback(args, fallback=CC.generate_image, state={}, size='1024x512', source_only=True)
        self.assertEqual(calls[0]['image_paths'], [self.source, self.identity])
        text = calls[0]['prompt']
        self.assertEqual(text.count(self.brief), 1)
        for word in ('白模', 'product_projection', 'Camera Base', 'Structure Packet'):
            self.assertNotIn(word, text)

    def test_direct_accepts_original_jpeg_without_reencoding(self):
        self.source=str(T.ROOT/'docs/showcase/watch_rock.jpg')
        payload, _ = self.direct()
        self.assertEqual(payload['model'], 'gpt-image-2')
        self.assertNotIn('input', payload)
        first = payload['images'][0]['image_url']
        self.assertTrue(first.startswith('data:image/jpeg;base64,'))
        self.assertEqual(base64.b64decode(first.split(',',1)[1]),Path(self.source).read_bytes())

    def test_environment_direct_uses_captured_canvas_and_identity(self):
        args = dict(prompt='preserve scene', edit_base_path=self.source,
                    reference_paths=[self.source,self.identity],output_path=self.out,
                    size='1024x512',log_dir=str(self.root),simple_brief='sunset by the sea',
                    manifest=[{'path':self.identity,'members':[{'path':self.identity,'role':'product'}]}],
                    identity_text='BLUE PRODUCT',camera_capture=True)
        with ExitStack() as stack:
            for name,value in (('find_auth_file',self.root/'auth'),('load_auth',{}),('token_expiring',False),
                               ('model_chain',['model']),('available_models',['model']),('auth_headers',{})):
                stack.enter_context(patch.object(CD,name,return_value=value))
            post=stack.enter_context(patch.object(CD,'post_responses',return_value=(
                base64.b64encode(Path(self.source).read_bytes()).decode(),{},'response')))
            CD.edit_image(**args)
        payload=post.call_args.args[2]
        images=[base64.b64decode(c['image_url'].split(',',1)[1]) for c in payload['input'][0]['content'] if c['type']=='input_image']
        self.assertEqual(images,[Path(self.source).read_bytes(),Path(self.identity).read_bytes()])
        text=payload['input'][0]['content'][0]['text']
        self.assertIn('当前 Blender 相机捕获的场景画面',text)
        self.assertIn('sunset by the sea',text)
        self.assertIn('BLUE PRODUCT',text)
        self.assertNotIn('白模',text)

    def test_environment_cli_fallback_keeps_camera_canvas(self):
        args=dict(prompt='keep identity and style',reference_paths=[self.source,self.identity],
                  edit_base_path=self.source,edit_mask_path='',output_path=self.out,cwd=str(self.root),timeout=30)
        calls=[]
        def exec_turn(**kwargs):
            calls.append(kwargs)
            Path(self.out).write_bytes(Path(self.source).read_bytes())
            return 'WONDFUL_IMAGEGEN_OK'
        with patch.object(CD,'edit_image',side_effect=CD.DirectEditError('endpoint',kind='request')), \
             patch.object(CC,'_exec_turn',side_effect=exec_turn):
            CD.generate_with_fallback(args,fallback=CC.generate_image,state={},size='1024x512',
                                      simple_base=self.source,simple_brief='style',camera_capture=True)
        self.assertEqual(calls[0]['image_paths'],[self.source,self.identity])
        self.assertNotIn('白模',calls[0]['prompt'])
        self.assertIn('相机画面',calls[0]['prompt'])

    def test_environment_full_direct_prompt_calls_canvas_a_camera_capture(self):
        args=dict(prompt='complete manual brief',edit_base_path=self.source,
                  reference_paths=[self.source,self.identity],output_path=self.out,size='1024x512',
                  log_dir=str(self.root),camera_capture=True,
                  manifest=[{'path':self.identity,'members':[{'path':self.identity,'role':'product'}]}])
        with ExitStack() as stack:
            for name,value in (('find_auth_file',self.root/'auth'),('load_auth',{}),('token_expiring',False),
                               ('model_chain',['model']),('available_models',['model']),('auth_headers',{})):
                stack.enter_context(patch.object(CD,name,return_value=value))
            post=stack.enter_context(patch.object(CD,'post_responses',return_value=(
                base64.b64encode(Path(self.source).read_bytes()).decode(),{},'response')))
            CD.edit_image(**args)
        text=post.call_args.args[2]['input'][0]['content'][0]['text']
        self.assertIn('当前相机画面',text)
        self.assertIn('complete manual brief',text)
        self.assertNotIn('白模',text)

    def test_environment_antigravity_requests_camera_canvas(self):
        calls=[]
        def run(cli,args,**kwargs):
            calls.append(args[args.index('--prompt')+1])
            Path(self.out).write_bytes(Path(self.source).read_bytes())
            event={'event':'step_update','step_update':{'step_type':'tool','tool_name':'generate_image',
                   'tool_info':{'output':{'path':self.out}}}}
            return subprocess.CompletedProcess(args,0,stdout=json.dumps(event)+'\n'+
                json.dumps({'status':'SUCCESS','response':'WONDFUL_ANTIGRAVITY_IMAGEGEN_OK'}),stderr='')
        with patch.object(AGY,'discover_antigravity_cli',return_value='/mock/agy'),patch.object(AGY,'_run',side_effect=run):
            AGY.generate_image(prompt='identity and environment',reference_paths=[self.source,self.identity],
                edit_base_path=self.source,output_path=self.out,cwd=str(self.root),camera_capture=True)
        self.assertIn('直接编辑捕获的相机画面',calls[0])
        self.assertNotIn('原始白模',calls[0])

    def test_antigravity_actual_staged_request_has_exact_input_order(self):
        seen = []
        def run(cli, args, **kwargs):
            task = args[args.index('--prompt') + 1]
            seen.append(task)
            inputs = json.loads(task.split('ImagePaths = ', 1)[1].splitlines()[0].replace("'", '"'))
            self.assertEqual([Path(p).read_bytes() for p in inputs],
                             [Path(self.source).read_bytes(), Path(self.identity).read_bytes()])
            Path(self.out).write_bytes(Path(self.source).read_bytes())
            event = {'event': 'step_update', 'step_update': {'step_type': 'tool', 'tool_name': 'generate_image',
                     'tool_info': {'output': {'path': self.out}}}}
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(event) + '\n' +
                json.dumps({'status': 'SUCCESS', 'response': 'WONDFUL_ANTIGRAVITY_IMAGEGEN_OK'}), stderr='')
        with patch.object(AGY, 'discover_antigravity_cli', return_value='/mock/agy'), \
             patch.object(AGY, '_run', side_effect=run):
            AGY.generate_image(prompt=self.prompt, reference_paths=[self.source, self.identity],
                edit_base_path=self.source, output_path=self.out, cwd=str(self.root), source_only=True)
        self.assertEqual(seen[0].count(self.brief), 1)
        for word in ('白模', 'product_projection', 'Camera Base', 'Structure Lock'):
            self.assertNotIn(word, seen[0])

    def test_invalid_capacity_mask_or_missing_identity_stop_before_transport(self):
        with self.assertRaises(ValueError):
            self.SR.validate_inputs(self.source, self.identity, 1)
        with patch.object(CD, 'find_auth_file') as auth:
            with self.assertRaises(CD.DirectEditError):
                self.direct(mask_path=self.identity)
            auth.assert_not_called()
        with patch.object(AGY, 'discover_antigravity_cli') as discover:
            with self.assertRaises(AGY.AntigravityError):
                AGY.generate_image(prompt=self.prompt, reference_paths=[self.source, 'missing'],
                    edit_base_path=self.source, output_path=self.out, cwd=str(self.root), source_only=True)
            discover.assert_not_called()

    def test_direct_stop_and_cancellation_never_fallback(self):
        args = dict(prompt=self.prompt, reference_paths=[self.source, self.identity],
                    edit_base_path=self.source, edit_mask_path='', output_path=self.out,
                    cwd=str(self.root), timeout=30)
        fallback = Mock()
        with patch.object(CD, 'edit_image', side_effect=CD.DirectEditError('quota', kind='quota', fatal=True)):
            with self.assertRaises(CD.DirectEditStop):
                CD.generate_with_fallback(args, fallback=fallback, state={}, size='1024x512', source_only=True)
        with self.assertRaisesRegex(RuntimeError, '取消'):
            CD.generate_with_fallback(args, fallback=fallback, state={}, size='1024x512',
                                      source_only=True, cancel_check=lambda: True)
        fallback.assert_not_called()


class SourceOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        T.setUpModule()
        T.load('composition')
        cls.sr = T.load('source_reference')
        T.stub('jev_semantics')
        T.stub('cli_transport', is_cancellation_requested=lambda: False)

    @classmethod
    def tearDownClass(cls):
        T.tearDownModule()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.target = self.image('target.png', 20)
        self.identity = self.image('identity.png', 210)
        image = T.NS(size=(48, 24), filepath=str(self.identity), filepath_raw=str(self.identity), packed_file=None)
        self.props = T.props(workflow_mode=T.WF.REFERENCE, workflow_multiview=False,
                             sb_image_path=str(self.target), product_images=[T.NS(image=image, source_path=str(self.identity), instruction='')],
                             prompt='用户完整文本', scene_brief='蓝色产品', render_count='1')
        self.context = T.context(self.props)
        self.context.scene.name = 'user scene'
        self.context.scene.frame_current = 1
        self.context.scene.frame_subframe = 0.0
        self.captures = Mock(side_effect=AssertionError('capture not allowed'))
        self.sent = []
        self.calls = []
        self.counter = 0
        self.ns = T.function_namespace('_get_prompt_text', '_set_prompt_text', '_reference_fingerprint', '_reference_manifest', '_source_user_info', '_source_paint_lock','_source_style_key',
                                      'workflow_preflight', '_camera_signature',
                                      classes=('WONDFUL_OT_polish_prompt', 'WONDFUL_OT_ai_render'))
        self.ns.update({k:v for k,v in T.OPS.items() if k not in self.ns})
        from queue import Queue, Empty
        self.ns.update(Queue=Queue, Empty=Empty, shutil=shutil, SOURCE_POLISH_SYSTEM=self.sr.POLISH_SYSTEM,
                       image_dimensions=T.package.composition.image_dimensions,
                       assess_canvas=T.package.composition.assess_canvas,
                       _SOURCE_POLISH_APPROVAL={}, image_filepath=lambda im: im.filepath,
                       sync_prompt_editors=lambda props:None,
                       RENDER_VARIANT_COUNT=4,
                       identity_preflight=T.OPS['identity_preflight'],
                       cleanup_old_sessions=lambda:None, create_session=self.session,
                       _addon_prefs=lambda c:T.NS(output_directory='', long_edge=1024, codex_direct_edit=False),
                       _analysis_provider=self.provider, _load_reference_image=lambda p:T.NS(size=(48,24)),
                       _validate_source_image=lambda p:None,
                       copy_references=lambda *a: self.fail('legacy copy'),
                       capture_camera_reference=self.captures, build_structure_packet=self.captures,
                       camera_output_dimensions=lambda c:T.package.composition.image_dimensions(self.target), _render_replica_preview=self.captures,
                       _sb_generation_reference=self.captures,
                       prepare_reference_for_upload=lambda p:p,
                       needs_reference_policy_refresh=lambda p:False,
                       prompt_profiles=T.NS(metadata=lambda *a:{'target_key':'test'}, needs_target_refresh=lambda *a:False,
                           style_cache_key=load_package_module('prompt_profiles').style_cache_key,
                           REFERENCE_ROLE_POLICY_VERSION=load_package_module('prompt_profiles').REFERENCE_ROLE_POLICY_VERSION),
                       select_polish_source=lambda p, visible, refresh:visible,
                       _appearance_context=lambda p:('BLENDER','',[]),
                       _product_autofill_key=lambda *a:'key', _product_autofill_needed=lambda *a:False,
                       appearance_lock_material_text=lambda p:'',
                       generate_needs_polish=lambda *a:True, REFERENCE_POLICY_VERSION=load_package_module('prompt_engine').REFERENCE_POLICY_VERSION,
                       _schedule_operator=lambda *a,**k:self.calls.append('render'),
                       record_timing=lambda *a:None, average_timing=lambda key,default:default,
                       save_metadata=lambda *a:None, resolve_output_directory=lambda *a:self.root,
                       preflight_output_directory=lambda *a:None,
                       render_variant_count=lambda *a:1, safe_diagnostic=str, MAX_IMAGE_PATHS=5,
                       resolve_output_directory_unused=None)
        self.ns['_status'] = lambda p, status, error='': (setattr(p,'status',status), setattr(p,'last_error',error))
        self.polish = Mock(return_value='完整润色内容。' * 50)
        self.style_summary='原图为夜景，暖色侧光与冷色环境反射，保留整体曝光与摄影氛围。'
        source_style_system=load_package_module('prompt_engine').SOURCE_STYLE_ANALYSIS_SYSTEM_PROMPT
        self.polish.side_effect=lambda **kw:(
            self.style_summary if kw['system_prompt']==source_style_system else
            json.dumps({'prompt': self.polish.return_value, 'materials': []}, ensure_ascii=False)
        )
        real_lock=load_package_module('appearance_lock')
        self.isolated_stub('appearance_lock',**{k:v for k,v in vars(real_lock).items() if not k.startswith('__')})
        real_engine=load_package_module('prompt_engine')
        self.isolated_stub('prompt_engine',**{k:v for k,v in vars(real_engine).items() if not k.startswith('__')})
        # the real worker-side steps (source 细节修复, crop-regeneration budget); other suites stub it
        real_worker=load_package_module('result_worker')
        self.isolated_stub('result_worker',**{k:v for k,v in vars(real_worker).items() if not k.startswith('__')})
        self.ns.update(AntigravityNotLoggedInError=AGY.AntigravityNotLoggedInError,
                       AntigravityNotInstalledError=AGY.AntigravityNotInstalledError,
                       CodexNotLoggedInError=CC.CodexNotLoggedInError,CodexNotInstalledError=CC.CodexNotInstalledError)
        self.provider_name='antigravity';self.provider_model=''
        self.provider_calls = []
        self.geometry_revision = 0
        def model_key(context, props):
            camera = context.scene.camera
            collection = props.product_collection
            return repr((id(camera), self.ns['_camera_signature'](camera, context.scene), id(collection),
                         [(id(o), o.name, o.hide_render) for o in collection.all_objects] if collection else None,
                         [[tuple(s.material.diffuse_color) for s in getattr(o,'material_slots',()) if s.material]
                          for o in collection.all_objects] if collection else None,
                         context.scene.frame_current, self.geometry_revision))
        def capture(context, props, directory, dimensions, source_path=""):
            flat = Path(directory)/'product_projection.png'
            rgba = np.zeros((dimensions[1], dimensions[0], 4), np.uint8)
            rgba[3:-3, 5:-5] = (30, 70, 180, 255)
            PNG.write_png(flat, rgba)
            # 2026-11: the AI-facing reference is the lit pass; the flat pass and the placed-product
            # canvas travel as frozen helpers
            path = Path(directory)/'product_lookdev.png'
            lit = rgba.copy(); lit[3:-3, 5:-5] = (40, 120, 60, 255)
            PNG.write_png(path, lit)
            canvas = Path(directory)/'source_product_canvas.png'
            PNG.write_png(canvas, np.dstack([np.full(rgba.shape[:2] + (3,), 90, np.uint8)]))
            helpers={'detail':{'path':str(flat),'snapshot':self.sr.fingerprint(flat)},
                     'canvas':{'path':str(canvas),'snapshot':self.sr.fingerprint(canvas)}}
            for key,name in (('keep','product_keep.png'),('matid','product_matid.png')):
                pixels=np.zeros_like(rgba)
                if key=='matid':pixels[rgba[:,:,3]>0]=(51,0,0,255)
                file=Path(directory)/name;PNG.write_png(file,pixels)
                helpers[key]={'path':str(file),'snapshot':self.sr.fingerprint(file)}
            metadata=Path(directory)/'product_matid.png.json'
            metadata.write_text(json.dumps({'materials':[{'index':0,'name':'surface','kind':'default'}]}))
            helpers['matid_json']={'path':str(metadata),'snapshot':self.sr.fingerprint(metadata)}
            return {'path':str(path), 'snapshot':self.sr.fingerprint(path), 'signature':model_key(context,props),
                    'dimensions':list(dimensions), 'method':'PRODUCT_LIT_CANVAS','helpers':helpers}
        real_projection = load_package_module('product_projection')
        self.projection_capture = Mock(side_effect=capture)
        self.projection = self.isolated_stub('product_projection', capture=self.projection_capture,
            product_objects=real_projection.product_objects, require_matching_aspect=real_projection.require_matching_aspect,
            freeze_helpers=real_projection.freeze_helpers,
            current=lambda c,p,r: bool(r) and model_key(c,p)==r['signature'] and self.sr.matches(r['snapshot'],r['path'])
                                 and all(self.sr.matches(v['snapshot'],v['path']) for v in r['helpers'].values()))
        def native_finish(context, props, ai_path, projection_record, helpers, directory, material_plan):
            output = Path(directory) / 'native_product_finish.png'
            shutil.copy2(ai_path, output)
            return str(output), {'summary': 'native fixture finish', 'material_plan': material_plan,
                                 'helpers_frozen': bool(helpers)}
        self.native_finish = Mock(side_effect=native_finish)
        self.isolated_stub('native_product_finish', finish=self.native_finish)

    def image(self, name, value):
        p=self.root/name
        PNG.write_png(str(p),np.full((24,48,3),value,np.uint8))
        return p

    def isolated_stub(self, name, **items):
        full = T.PACKAGE + '.' + name
        missing = object()
        previous_module = sys.modules.get(full, missing)
        previous_attribute = getattr(T.package, name, missing)
        module = T.stub(name, **items)
        def restore():
            if previous_module is missing:
                sys.modules.pop(full, None)
            else:
                sys.modules[full] = previous_module
            if previous_attribute is missing:
                if hasattr(T.package, name):
                    delattr(T.package, name)
            else:
                setattr(T.package, name, previous_attribute)
        self.addCleanup(restore)
        return module

    def session(self):
        self.counter += 1
        d = self.root / ('session' + str(self.counter)); d.mkdir()
        return T.NS(directory=d, session_id=str(self.counter), viewport_reference=d/'camera.png',
                    prompt_file=d/'prompt.txt', output_raw=d/'raw.png', output_final=d/'final.png')

    def edit_canvas_bytes(self):
        """The placed-product canvas the provider actually edited (the fake provider echoes it)."""
        return Path(self.provider_calls[0]['edit_base_path']).read_bytes()

    def provider(self,*_):
        def generate(**kwargs):
            self.provider_calls.append(kwargs)
            self.sent = [Path(p).read_bytes() for p in kwargs['reference_paths']]
            shutil.copy2(kwargs['reference_paths'][0], kwargs['output_path'])
            return {'status':'ok'}
        return dict(id=self.provider_name, label='Mock Provider', discover=lambda:True, imagegen_enabled=True,
                    cli_path='', model=self.provider_model, polish_timeout=30, render_timeout=30, polish=self.polish,
                    generate=generate, reference_limit=3)

    def run_polish(self, mutate=None, expected=None):
        op = self.ns['WONDFUL_OT_polish_prompt']()
        def start(context, job):
            op._origin_scene=context.scene;op._started_at=time.time();op._thread=None;op._error=None
            try:op._result=job()
            except Exception as exc:op._error=(exc,'');op._result=None
            if mutate: mutate()
            return True
        op._start_thread=start
        op._cleanup=lambda context:None
        self.props.pending_generate=True
        self.assertEqual(op.execute(self.context),{'RUNNING_MODAL'})
        self.assertEqual(op._modal(self.context,T.NS(type='TIMER')),expected or {'FINISHED'})
        return op

    def test_complete_polish_uses_original_real_projection_identity_without_scene_capture(self):
        self.run_polish()
        self.assertEqual(self.polish.call_count,2)
        self.assertEqual(self.calls,['render'])
        sent = self.polish.call_args.kwargs
        first=self.polish.call_args_list[0].kwargs
        self.assertEqual(sent['response_format'], 'json')
        self.assertNotIn('response_format', first)
        self.assertEqual(first['system_prompt'],load_package_module('prompt_engine').SOURCE_STYLE_ANALYSIS_SYSTEM_PROMPT)
        self.assertEqual([Path(p).read_bytes() for p in first['image_paths']],[self.target.read_bytes()])
        self.assertIn(self.style_summary,sent['user_text'])
        self.projection_capture.assert_called_once()
        projection = Path(sent['image_paths'][1]).read_bytes()
        self.assertEqual([Path(p).read_bytes() for p in sent['image_paths']],
                         [self.target.read_bytes(),projection,self.identity.read_bytes()])
        self.assertIn('已建模的灯组', sent['system_prompt'])
        self.assertIn('用户完整文本',sent['user_text'])
        self.assertIn('蓝色产品',sent['user_text'])
        op=self.ns['WONDFUL_OT_ai_render']()
        def start(context, job):
            op._result=job();return True
        op._start_thread=start
        self.assertEqual(op.execute(self.context),{'RUNNING_MODAL'})
        self.assertEqual(self.sent,[Path(op._source_repair['canvas']).read_bytes(),projection,self.identity.read_bytes()])
        self.projection_capture.assert_called_once()
        request=self.provider_calls[0]
        self.assertTrue(request['source_only'])
        self.assertEqual(request['edit_base_path'],request['reference_paths'][0])
        self.assertFalse(request['edit_mask_path'])
        self.assertEqual(request['product_projection_path'], request['reference_paths'][1])
        self.assertEqual(request['prompt'].count(self.polish.return_value),1)
        self.assertEqual(op._camera_output_size,(48,24))
        self.captures.assert_not_called()

    def test_source_style_cache_keeps_mandatory_final_polish_and_tracks_its_own_inputs(self):
        self.run_polish()
        def check(calls):
            self.polish.reset_mock();op=self.run_polish()
            self.assertEqual(self.polish.call_count,calls)
            self.assertEqual(len(self.polish.call_args.kwargs['image_paths']),3)
            return op
        self.assertTrue(check(1)._style_cache_hit)
        self.image(self.identity.name,133)
        self.assertTrue(check(1)._style_cache_hit)  # material reference is not a style input
        self.geometry_revision+=1
        self.assertTrue(check(1)._style_cache_hit)
        self.image(self.target.name,134);self.assertFalse(check(2)._style_cache_hit)
        self.props.sb_target_note='保留逆光';self.assertFalse(check(2)._style_cache_hit)
        self.provider_name='codex';self.assertFalse(check(2)._style_cache_hit)
        self.provider_model='new-analysis-model';self.assertFalse(check(2)._style_cache_hit)
        profiles=load_package_module('prompt_profiles');version=profiles.REFERENCE_ROLE_POLICY_VERSION+1
        with patch.object(profiles,'REFERENCE_ROLE_POLICY_VERSION',version),patch.object(self.ns['prompt_profiles'],'REFERENCE_ROLE_POLICY_VERSION',version):
            self.assertFalse(check(2)._style_cache_hit)
        self.assertFalse(self.provider_calls)

    def test_role_policy_or_provider_change_invalidates_polish_approval(self):
        self.run_polish()
        self.ns['prompt_profiles'].REFERENCE_ROLE_POLICY_VERSION+=1
        op=self.ns['WONDFUL_OT_ai_render']()
        self.assertEqual(op.execute(self.context),{'CANCELLED'})
        self.ns['prompt_profiles'].REFERENCE_ROLE_POLICY_VERSION-=1
        self.run_polish();self.provider_model='changed-after-polish'
        self.assertEqual(op.execute(self.context),{'CANCELLED'})
        self.assertFalse(self.provider_calls)

    def test_current_model_paint_ignores_legacy_identity_color_and_preserves_concrete_user_color(self):
        material=T.NS(name='CustomPaint',diffuse_color=(0.,.4,0.,1.),use_nodes=False,metallic=.7,roughness=.2)
        self.props.product_collection.all_objects[0].material_slots=[T.NS(material=material)]
        self.props.color_source='PRODUCT_REF'
        self.props.product_look_auto=self.props.product_look_prompt='AUTOMATIC_SILVER_FIVE_SPOKES'
        self.props.identity_details_auto=self.props.identity_details='AUTOMATIC_SILVER_FIVE_SPOKES'
        self.props.product_autofill_key='key'
        self.props.prompt='车漆明确改为深红色，保持模型灯组';self.props.scene_brief=''
        self.run_polish()
        text=self.polish.call_args.kwargs['user_text']
        self.assertIn('CustomPaint',text);self.assertIn('#00AA00',text)
        self.assertIn('车漆明确改为深红色',text)
        self.assertNotIn('AUTOMATIC_SILVER_FIVE_SPOKES',text)
        self.assertIn('不锁定其他部件材质',text)
        op=self.ns['WONDFUL_OT_ai_render']();op._start_thread=lambda c,job:(job(),True)[1]
        self.assertEqual(op.execute(self.context),{'RUNNING_MODAL'})
        request=self.provider_calls[0]['prompt']
        self.assertIn('车漆明确改为深红色',request);self.assertIn('#00AA00',request)
        self.assertNotIn('AUTOMATIC_SILVER_FIVE_SPOKES',request)
        self.assertEqual(self.props.color_source,'PRODUCT_REF')

    def test_empty_style_analysis_or_input_change_stops_before_final_polish(self):
        self.polish.side_effect=lambda **kwargs:''
        self.run_polish(expected={'CANCELLED'})
        self.assertEqual(self.polish.call_count,1);self.assertFalse(self.calls)
        self.assertIn('独立风格分析',self.props.last_error)
        def changed(**kwargs):
            self.target.write_bytes(b'changed while analyzing style')
            return self.style_summary
        self.polish.reset_mock();self.polish.side_effect=changed
        self.run_polish(expected={'CANCELLED'})
        self.assertEqual(self.polish.call_count,1);self.assertFalse(self.calls)
        self.assertFalse(self.provider_calls)

    def test_material_revision_during_source_polish_stops_chained_generation(self):
        # Real material/shader/texture fingerprint behavior is separately tested
        # in ProductProjectionTests; this exercises the operator's approval gate.
        material=T.NS(name='BodyPaint',diffuse_color=(0.,.4,0.,1.),use_nodes=False)
        self.props.product_collection.all_objects[0].material_slots=[T.NS(material=material)]
        self.run_polish(mutate=lambda:setattr(material,'diffuse_color',(.4,0.,0.,1.)))
        self.assertFalse(self.calls);self.assertFalse(self.provider_calls)
        self.assertEqual(self.props.status,'ERROR')

    def autofill_functions(self):
        names=('_product_autofill_key','_product_fields_untouched','_product_autofill_needed','_apply_product_autofill','_appearance_context')
        env=T.function_namespace(*names)
        functions={name:env[name] for name in names}
        env.update(self.ns);env.update(functions)
        return functions

    def test_generated_autofill_policy_refresh_never_overwrites_manual_material_draft(self):
        functions=self.autofill_functions()
        p=self.props;p.product_look_prompt=p.product_look_auto='OLD_SILVER_PAINT'
        p.identity_details=p.identity_details_auto='OLD_FIVE_SPOKES';p.product_autofill_policy_version=0
        key=functions['_product_autofill_key'](p,self.provider())
        self.assertTrue(functions['_product_autofill_needed'](p,key))
        _,lock,details=functions['_appearance_context'](p)
        self.assertNotIn('OLD_SILVER_PAINT',lock);self.assertNotIn('OLD_FIVE_SPOKES',str(details))
        self.assertEqual(p.product_look_prompt,'OLD_SILVER_PAINT')
        p.product_look_prompt='手工缎面涂层'
        self.assertFalse(functions['_product_autofill_needed'](p,key))
        self.assertFalse(functions['_apply_product_autofill'](p,key,'新自动结果',['材质']))
        self.assertEqual(p.product_look_prompt,'手工缎面涂层')
        p.product_look_prompt=p.product_look_auto
        self.assertTrue(functions['_apply_product_autofill'](p,key,'哑光饰条',['透镜材质']))
        self.assertEqual(p.product_autofill_policy_version,self.ns['REFERENCE_POLICY_VERSION'])
        self.assertFalse(functions['_product_autofill_needed'](p,key))
        engine=load_package_module('prompt_engine')
        with patch.object(engine,'REFERENCE_POLICY_VERSION',engine.REFERENCE_POLICY_VERSION+1):
            self.assertNotEqual(key,functions['_product_autofill_key'](p,self.provider()))

    def test_environment_polish_drops_legacy_generated_shape_and_paint_even_when_autofill_fails(self):
        engine=load_package_module('prompt_engine')
        self.ns.update(self.autofill_functions())
        self.ns.update(build_polish_user_text=engine.build_polish_user_text,
            PRODUCT_ANALYSIS_SYSTEM_PROMPT=engine.PRODUCT_ANALYSIS_SYSTEM_PROMPT,
            product_analysis_user_text=engine.product_analysis_user_text,parse_product_analysis=engine.parse_product_analysis,
            BASE_SYSTEM_PROMPT=engine.BASE_SYSTEM_PROMPT,sanitize_appearance_prompt_report=engine.sanitize_appearance_prompt_report)
        self.ns['prompt_profiles'].system_prompt=load_package_module('prompt_profiles').system_prompt
        self.isolated_stub('jev_semantics',analyze_appearance=lambda *args:{})
        def copies(paths,session,name):
            result=[]
            for i,path in enumerate(paths):
                dest=session.directory/f'{name}_{i}.png';shutil.copy2(path,dest);result.append(str(dest))
            return result
        self.ns['copy_references']=copies
        self.props.workflow_mode=T.WF.ENVIRONMENT
        for failure in (True,False):
            with self.subTest(analysis_failure=failure):
                p=self.props;p.prompt=p.last_ai_prompt='OLD_AI_SILVER_FIVE_SPOKES'
                p.prompt_source='用户保留真实模型，使用城市夜景'
                p.product_look_prompt=p.product_look_auto='OLD_SILVER_PAINT'
                p.identity_details=p.identity_details_auto='OLD_FIVE_SPOKES'
                p.product_autofill_policy_version=0;p.product_autofill_key='legacy'
                def polish(**kwargs):
                    if kwargs['system_prompt']==engine.PRODUCT_ANALYSIS_SYSTEM_PROMPT:
                        if failure:raise RuntimeError('analysis unavailable')
                        return json.dumps({'appearance':'哑光饰条','details':['灯具透镜材质']})
                    return self.polish.return_value
                self.polish.reset_mock();self.polish.side_effect=polish
                self.run_polish()
                final=self.polish.call_args.kwargs['user_text']
                for old in ('OLD_AI_SILVER_FIVE_SPOKES','OLD_SILVER_PAINT','OLD_FIVE_SPOKES'):self.assertNotIn(old,final)
                self.assertIn('用户保留真实模型',final)
                self.assertEqual(self.polish.call_count,2)
                self.assertEqual(p.product_look_prompt,'OLD_SILVER_PAINT' if failure else '哑光饰条')
        self.assertFalse(self.provider_calls);self.captures.assert_not_called()

    def _check_stale_ai_source(self, image_name):
        prompt_engine = load_package_module('prompt_engine')
        self.ns['select_polish_source'] = prompt_engine.select_polish_source
        self.props.prompt = '手写创意需求'
        self.polish.return_value = '旧产品的 AI 描述'
        self.run_polish()
        self.assertEqual(self.props.prompt_source, '手写创意需求')
        self.assertTrue(self.props.prompt_reference_input_key)
        self.image(getattr(self, image_name).name, 133)
        self.polish.return_value = '新产品的 AI 描述'
        self.run_polish()
        user_text = self.polish.call_args.kwargs['user_text']
        self.assertIn('手写创意需求', user_text)
        self.assertNotIn('旧产品的 AI 描述', user_text)

    def test_identity_bytes_change_discards_old_ai_text_but_keeps_manual_text(self):
        self._check_stale_ai_source('identity')

    def test_target_bytes_change_discards_old_ai_text_but_keeps_manual_text(self):
        self._check_stale_ai_source('target')

    def test_manually_edited_prompt_survives_reference_change(self):
        self.ns['select_polish_source'] = load_package_module('prompt_engine').select_polish_source
        self.polish.return_value = '旧 AI 产品描述'
        self.run_polish()
        self.props.prompt = '用户刚刚亲手修改的需求'
        self.image(self.identity.name, 155)
        self.run_polish()
        text = self.polish.call_args.kwargs['user_text']
        self.assertIn('用户刚刚亲手修改的需求', text)
        self.assertNotIn('旧 AI 产品描述', text)

    def test_unknown_legacy_ai_fingerprint_is_not_reused(self):
        self.ns['select_polish_source'] = load_package_module('prompt_engine').select_polish_source
        self.props.last_ai_prompt = self.props.prompt = '旧 AI 产品描述'
        self.props.prompt_source = ''
        self.props.prompt_reference_input_key = ''
        self.run_polish()
        self.assertNotIn('旧 AI 产品描述', self.polish.call_args.kwargs['user_text'])
        self.assertIn(self.props.scene_brief, self.polish.call_args.kwargs['user_text'])

    def test_changed_target_note_discards_old_ai_text_and_keeps_manual_brief(self):
        self.ns['select_polish_source'] = load_package_module('prompt_engine').select_polish_source
        self.props.prompt = '手写创意需求'
        self.props.sb_target_note = '旧目标备注'
        self.polish.return_value = 'AI 根据旧备注写的内容'
        self.run_polish()
        self.props.sb_target_note = '新目标备注'
        self.run_polish()
        text = self.polish.call_args.kwargs['user_text']
        self.assertIn('手写创意需求', text)
        self.assertIn('新目标备注', text)
        self.assertNotIn('AI 根据旧备注写的内容', text)

    def test_target_note_reaches_polish_and_final_edit_once(self):
        note = '保留独特青色建筑光线'
        self.props.sb_target_note = note
        self.run_polish()
        self.assertEqual(self.polish.call_args.kwargs['user_text'].count(note), 1)
        op = self.ns['WONDFUL_OT_ai_render']()
        op._start_thread = lambda context, job: (job(), True)[1]
        self.assertEqual(op.execute(self.context), {'RUNNING_MODAL'})
        self.assertEqual(self.provider_calls[0]['prompt'].count(note), 1)

    def test_target_note_not_duplicated_when_polish_already_includes_it(self):
        note = '保留独特青色建筑光线'
        self.props.sb_target_note = note
        self.polish.return_value = '产品替换，' + note + '。'
        self.run_polish()
        op = self.ns['WONDFUL_OT_ai_render']()
        op._start_thread = lambda context, job: (job(), True)[1]
        self.assertEqual(op.execute(self.context), {'RUNNING_MODAL'})
        self.assertEqual(self.provider_calls[0]['prompt'].count(note), 1)

    def test_target_note_change_during_polish_or_before_render_stops(self):
        self.props.sb_target_note = '原备注'
        self.run_polish(mutate=lambda: setattr(self.props, 'sb_target_note', '新备注'))
        self.assertFalse(self.calls)
        self.assertEqual(self.props.status, 'ERROR')
        self.run_polish()
        self.props.sb_target_note = '再次修改'
        op = self.ns['WONDFUL_OT_ai_render']()
        self.assertEqual(op.execute(self.context), {'CANCELLED'})
        self.assertFalse(self.provider_calls)

    def test_changed_target_or_identity_stops_automatic_render(self):
        for attr in ('target','identity'):
            with self.subTest(attr=attr):
                self.calls.clear()
                original=getattr(self,attr).read_bytes()
                self.run_polish(mutate=lambda: getattr(self,attr).write_bytes(b'changed'))
                self.assertFalse(self.calls)
                self.assertEqual(self.props.status,'ERROR')
                getattr(self,attr).write_bytes(original)

    def test_changed_input_after_polish_cannot_reach_provider(self):
        self.run_polish()
        self.identity.write_bytes(b'changed')
        op=self.ns['WONDFUL_OT_ai_render']()
        self.assertEqual(op.execute(self.context),{'CANCELLED'})
        self.assertFalse(self.provider_calls)

    def test_structure_and_projection_changes_during_polish_stop_chain(self):
        def alter_projection():
            path=self.projection_capture.call_args.args[2]/'product_projection.png'
            path.write_bytes(b'changed projection')
        for mutate in (lambda:setattr(self.context.scene.camera.data,'lens',60),
                       lambda:setattr(self,'geometry_revision',self.geometry_revision+1),
                       lambda:setattr(self.context.scene,'frame_current',self.context.scene.frame_current+1),alter_projection):
            with self.subTest(mutate=mutate):
                self.calls.clear()
                self.run_polish(mutate=mutate)
                self.assertFalse(self.calls)
                self.assertEqual(self.props.status,'ERROR')

    def test_model_change_after_polish_prevents_image_provider(self):
        self.run_polish()
        self.geometry_revision+=1
        op=self.ns['WONDFUL_OT_ai_render']()
        self.assertEqual(op.execute(self.context),{'CANCELLED'})
        self.assertFalse(self.provider_calls)

    def test_changed_projection_after_polish_and_capture_failure_never_generate(self):
        self.run_polish()
        projection=self.projection_capture.call_args.args[2]/'product_projection.png'
        projection.write_bytes(b'changed cached projection')
        op=self.ns['WONDFUL_OT_ai_render']()
        self.assertEqual(op.execute(self.context),{'CANCELLED'})
        self.assertFalse(self.provider_calls)
        self.polish.reset_mock()
        self.projection_capture.side_effect=RuntimeError('projection capture failed')
        op=self.ns['WONDFUL_OT_polish_prompt']()
        self.assertEqual(op.execute(self.context),{'CANCELLED'})
        self.polish.assert_not_called()
        self.assertFalse(self.provider_calls)

    def test_missing_camera_collection_or_mismatched_aspect_stop_before_polish(self):
        for missing in ('camera','collection','aspect','empty'):
            with self.subTest(missing=missing):
                camera=self.context.scene.camera;collection=self.props.product_collection
                if missing=='camera':self.context.scene.camera=None
                elif missing=='collection':self.props.product_collection=None
                elif missing=='empty':self.props.product_collection=T.Collection('empty')
                else:self.ns['camera_output_dimensions']=lambda context:(100,100)
                provider=Mock()
                with patch.dict(self.ns,{'_analysis_provider':provider}):
                    op=self.ns['WONDFUL_OT_polish_prompt']()
                    self.assertEqual(op.execute(self.context),{'CANCELLED'})
                provider.assert_not_called()
                self.polish.assert_not_called();self.projection_capture.assert_not_called()
                if missing=='aspect':self.assertIn('画幅比例不一致',op.last_report[1])
                self.context.scene.camera=camera;self.props.product_collection=collection
                self.ns['camera_output_dimensions']=lambda context:(48,24)

    def test_slot_replacement_with_same_file_stops_chained_render(self):
        for slot in ('target','identity'):
            with self.subTest(slot=slot):
                self.calls.clear()
                if slot == 'target':
                    mutate=lambda: setattr(self.props,'sb_reference_image',T.NS(filepath=str(self.target),size=(48,24)))
                else:
                    previous=self.props.product_images[0]
                    mutate=lambda: self.props.product_images.__setitem__(0,T.NS(
                        image=T.NS(size=(48,24),filepath=str(self.identity),filepath_raw=str(self.identity)),
                        source_path=previous.source_path,instruction=''))
                self.run_polish(mutate=mutate)
                self.assertFalse(self.calls)
                self.assertEqual(self.props.status,'ERROR')

    def test_non_png_original_controls_actual_canvas_dimensions(self):
        jpeg = self.root/'source.jpg'
        shutil.copy2(T.ROOT/'docs/showcase/watch_rock.jpg',jpeg)
        self.target=jpeg
        self.props.sb_image_path=str(jpeg)
        expected=T.package.composition.image_dimensions(jpeg)
        self.assertTrue(expected and min(expected)>0)
        self.run_polish()
        op=self.ns['WONDFUL_OT_ai_render']()
        op._start_thread=lambda context,job:(job(),True)[1]
        self.assertEqual(op.execute(self.context),{'RUNNING_MODAL'})
        self.assertEqual(op._camera_output_size,expected)
        self.assertEqual(T.package.composition.image_dimensions(self.provider_calls[0]['edit_base_path']),expected)
        self.assertEqual(Path(self.provider_calls[0]['edit_base_path']).name,'source_product_canvas.png')

    def test_toggle_after_polish_cannot_silently_start_multiview_capture(self):
        self.run_polish()
        self.props.workflow_multiview=True
        op=self.ns['WONDFUL_OT_ai_render']()
        self.assertEqual(op.execute(self.context),{'CANCELLED'})
        self.assertFalse(self.provider_calls)
        self.captures.assert_not_called()

    def test_temporary_decode_never_reloads_existing_user_image(self):
        env=T.function_namespace('_validate_source_image')
        image=T.NS(size=(48,24),reload=Mock())
        images=T.NS(load=Mock(return_value=image),remove=Mock())
        env['bpy']=T.NS(path=T.bpy.path,data=T.NS(images=images))
        env['_validate_source_image'](self.target)
        images.load.assert_called_once_with(str(self.target.resolve()),check_existing=False)
        images.remove.assert_called_once_with(image)
        image.reload.assert_not_called()

    def test_source_result_can_be_exported_and_compared(self):
        self.run_polish()
        op=self.ns['WONDFUL_OT_ai_render']()
        def start(context, job):
            op._origin_scene=context.scene;op._thread=None;op._error=None;op._result=job();return True
        op._start_thread=start
        op._cleanup=lambda context:None
        self.ns['load_image']=lambda path, name:T.NS(path=path)
        def export(source, output_dir, scene, camera, session):
            dest=self.root/'exported.png';shutil.copy2(source,dest);return dest
        self.ns['export_result']=export
        self.assertEqual(op.execute(self.context),{'RUNNING_MODAL'})
        variant=op._completed_variants.get_nowait()
        variant['processed']={'final':variant['best_path']}
        op._completed_variants.put(variant)
        self.assertEqual(op._modal_source_only(self.context,T.NS(type='TIMER')),{ 'FINISHED'})
        self.assertEqual(self.props.status,'SUCCESS')
        self.assertEqual(self.props.compare_pair,'TARGET_RESULT')
        self.assertIn('未执行',self.props.last_iou_note)
        self.assertIn('未执行',self.props.last_score_note)
        self.assertEqual(Path(self.props.last_export_path).read_bytes(),self.edit_canvas_bytes())

    def test_model_helpers_survive_polish_generation_and_actual_local_export(self):
        ai = PNG.read_png(self.target)
        detail = ai.copy()
        keep = np.zeros_like(detail)
        labels = np.zeros(detail.shape[:2], np.int32)
        materials = [{'index': 0, 'name': 'body paint', 'kind': 'reflective'}]
        PNG.write_png(self.target,ai)
        MC=load_package_module('material_color')
        base_capture=self.projection_capture.side_effect
        captured={}
        def capture(*args, **kwargs):
            record=base_capture(*args, **kwargs)
            PNG.write_png(record['path'],detail)
            ids=np.zeros_like(detail)
            for material in materials:
                ids[labels==material['index']]=(*MC.palette_colour(material['index']),255)
            PNG.write_png(record['helpers']['keep']['path'],keep)
            PNG.write_png(record['helpers']['matid']['path'],ids)
            Path(record['helpers']['matid_json']['path']).write_text(json.dumps({'materials':materials}))
            record['snapshot']=self.sr.fingerprint(record['path'])
            for key,row in record['helpers'].items():
                row['snapshot']=self.sr.fingerprint(row['path'])
                captured[key]=Path(row['path']).read_bytes()
            return record
        self.projection_capture.side_effect=capture
        self.run_polish()
        self.ns['load_image']=lambda path,name:T.NS(path=path)
        self.ns['export_result']=T.load('output_manager').export_result
        self.ns['enforce_aspect_ratio']=lambda src,dst,w,h:(str(shutil.copy2(src,dst)),False)
        self.isolated_stub('result_worker',**{k:v for k,v in vars(load_package_module('result_worker')).items() if not k.startswith('__')})
        self.ns['save_metadata']=Mock()
        op=self.ns['WONDFUL_OT_ai_render']()
        def start(context,job):
            op._origin_scene=context.scene;op._thread=None;op._error=None;op._result=job();return True
        op._start_thread=start;op._cleanup=lambda context:None
        self.assertEqual(op.execute(self.context),{'RUNNING_MODAL'})
        for key,data in captured.items():
            self.assertEqual(Path(op._source_repair[key]).read_bytes(),data)
        self.assertEqual(op._modal_source_only(self.context,T.NS(type='TIMER')),{'FINISHED'},self.props.last_error)
        self.native_finish.assert_not_called()
        report=op._source_repair_reports[1]
        self.assertEqual(report['method'], 'natural_fusion')
        self.assertFalse(report['native_replacement'])
        # 2026-11: the silhouette is checked against the Blender mask at fixed coordinates; this
        # fixture's provider echoes a flat grey canvas, so the product cannot be aligned and the
        # small-part repair is skipped instead of pasting onto the wrong place
        self.assertTrue(report['geometry_audit'])
        self.assertIsNotNone(report['silhouette_iou'])
        self.assertTrue(report['detail_repair'] is None or report['detail_repair'].get('skipped'))
        self.assertTrue(report['detail_review_required'])
        self.assertEqual(report['material_plan'], [])
        self.assertFalse(report['material_plan_applied'])
        final=PNG.read_png(self.props.last_export_path)
        np.testing.assert_array_equal(final, PNG.read_png(op._source_repair["canvas"]))
        self.assertEqual(Path(self.props.last_export_path).read_bytes(), self.edit_canvas_bytes())
        self.assertEqual(report['raw_provider_image']['sha256'],hashlib.sha256(self.edit_canvas_bytes()).hexdigest())
        self.assertEqual(report['raw_fitted_canvas']['sha256'],hashlib.sha256(self.edit_canvas_bytes()).hexdigest())
        metadata_call = self.ns['save_metadata'].call_args.args[1]
        self.assertEqual(metadata_call['product_finish'], op._source_repair_reports)
        self.assertNotIn('native_product_finish', metadata_call)
        self.assertFalse(metadata_call['geometry_audit'])
        self.assertIn('车灯', self.props.last_detail_note)
        self.assertIn('轮毂', self.props.last_detail_note)
        self.assertIn('标识', self.props.last_detail_note)
        self.assertEqual(len(self.provider_calls),1)
        self.assertEqual(len(self.provider_calls[0]['reference_paths']),3)
        self.assertEqual(self.polish.call_count,2)
        self.projection_capture.assert_called_once();self.captures.assert_not_called()

    def test_changed_detail_helpers_during_polish_or_before_generation_stop(self):
        for name in ('product_keep.png','product_matid.png','product_matid.png.json'):
            with self.subTest(helper=name):
                def mutate():
                    directory=self.projection_capture.call_args.args[2]
                    (directory/name).write_bytes(b'changed helper')
                self.calls.clear()
                self.run_polish(mutate=mutate)
                self.assertFalse(self.calls)
                self.assertEqual(self.props.status,'ERROR')
                self.run_polish()
                mutate()
                op=self.ns['WONDFUL_OT_ai_render']()
                self.assertEqual(op.execute(self.context),{'CANCELLED'})
                self.assertFalse(self.provider_calls)

    def test_frozen_helper_changed_after_generation_stops_export_preserving_raw(self):
        self.run_polish()
        op=self.ns['WONDFUL_OT_ai_render']()
        def start(context,job):
            op._origin_scene=context.scene;op._thread=None;op._error=None;op._result=job();return True
        op._start_thread=start;op._cleanup=lambda context:None
        self.ns['export_result']=Mock()
        self.ns['enforce_aspect_ratio']=lambda src,dst,w,h:(str(shutil.copy2(src,dst)),False)
        self.assertEqual(op.execute(self.context),{'RUNNING_MODAL'})
        Path(op._source_repair['matid_json']).write_text('changed after request')
        raw=Path(self.provider_calls[0]['output_path'])
        self.assertEqual(op._modal_source_only(self.context,T.NS(type='TIMER')),{'CANCELLED'})
        self.ns['export_result'].assert_not_called()
        self.assertEqual(raw.read_bytes(),self.edit_canvas_bytes())
        self.assertIn('辅助文件',self.props.last_error)

    def test_stale_camera_model_or_paint_after_async_generation_stops_export_and_preserves_raw(self):
        material=T.NS(name='BodyPaint',diffuse_color=(0.,.4,0.,1.),use_nodes=False)
        self.props.product_collection.all_objects[0].material_slots=[T.NS(material=material)]
        for stale in ('camera', 'model', 'paint'):
            with self.subTest(stale=stale):
                self.run_polish()
                op=self.ns['WONDFUL_OT_ai_render']()
                def start(context,job):
                    op._origin_scene=context.scene;op._thread=None;op._error=None;op._result=job();return True
                op._start_thread=start;op._cleanup=lambda context:None
                export=Mock();self.ns['export_result']=export
                self.ns['enforce_aspect_ratio']=lambda src,dst,w,h:(str(shutil.copy2(src,dst)),False)
                self.assertEqual(op.execute(self.context),{'RUNNING_MODAL'})
                raw=Path(self.provider_calls[-1]['output_path'])
                if stale == 'camera':
                    self.context.scene.camera.data.lens += 1
                elif stale == 'model':
                    self.geometry_revision += 1
                else:
                    material.diffuse_color=(.9,.1,.1,1)
                self.assertEqual(op._modal_source_only(self.context,T.NS(type='TIMER')),{'CANCELLED'})
                export.assert_not_called()
                self.native_finish.assert_not_called()
                self.assertTrue(raw.is_file())
                self.assertIn('已保留原始生图',self.props.last_error)

    def test_source_only_preserves_photo_finish_preference_without_applying_it(self):
        self.props.photo_finish = 'STANDARD'
        self.run_polish()
        op = self.ns['WONDFUL_OT_ai_render']()
        def start(context, job):
            op._origin_scene = context.scene
            op._thread = None
            op._error = None
            op._result = job()
            return True
        op._start_thread = start
        op._cleanup = lambda context: None
        self.ns['load_image'] = lambda path, name: T.NS(path=path)
        def fit(src, dst, width, height):
            shutil.copy2(src, dst)
            return str(dst), False
        self.ns['enforce_aspect_ratio'] = fit
        def export(src, *args):
            dest = self.root / 'finished_output.png'
            shutil.copy2(src, dest)
            return dest
        self.ns['export_result'] = export
        worker = load_package_module('result_worker')
        finish = load_package_module('photo_finish')
        finish_step = Mock(wraps=worker.photo_finish_step)
        self.isolated_stub('result_worker', **{**{k:v for k,v in vars(worker).items() if not k.startswith('__')}, 'photo_finish_step': finish_step})
        with patch.object(finish, 'apply_file', side_effect=AssertionError('unexpected finish')) as apply_finish:
            self.assertEqual(op.execute(self.context), {'RUNNING_MODAL'})
            self.assertEqual(op._modal_source_only(self.context, T.NS(type='TIMER')), {'FINISHED'})
        finish_step.assert_called_once()
        self.assertIsNone(finish_step.call_args.args[2])
        apply_finish.assert_not_called()
        self.assertEqual(self.props.photo_finish, 'STANDARD')
        self.assertEqual(Path(self.props.last_export_path).read_bytes(), self.edit_canvas_bytes())

    def test_provider_failure_stops_with_error_and_no_result(self):
        self.run_polish()
        op=self.ns['WONDFUL_OT_ai_render']()
        def start(context, job):
            op._origin_scene=context.scene;op._thread=None;op._result=None
            op._error=(RuntimeError('provider failed'), 'trace');return True
        op._start_thread=start;op._cleanup=lambda context:None
        self.assertEqual(op.execute(self.context),{'RUNNING_MODAL'})
        self.assertEqual(op._modal_source_only(self.context,T.NS(type='TIMER')),{ 'CANCELLED'})
        self.assertIn('provider failed',self.props.last_error)
        self.assertFalse(self.props.last_export_path)

    def test_last_queue_item_added_as_thread_exits_is_drained_before_count_check(self):
        self.run_polish()
        op = self.ns['WONDFUL_OT_ai_render']()
        def start(context, job):
            op._origin_scene=context.scene;op._error=None;op._result=job();return True
        op._start_thread=start;op._cleanup=lambda context:None
        self.ns['load_image']=lambda path, name:T.NS(path=path)
        self.ns['export_result']=lambda source, *args:Path(source)
        self.assertEqual(op.execute(self.context), {'RUNNING_MODAL'})
        variant = op._completed_variants.get_nowait()
        variant['processed'] = {'final':variant['best_path']}
        def exits_between_checks():
            op._completed_variants.put(variant)
            return False
        op._thread = T.NS(is_alive=exits_between_checks)
        self.assertEqual(op._modal_source_only(self.context, T.NS(type='TIMER')), {'FINISHED'})
        self.assertEqual(len(op._exported_by_variant), 1)

    def test_missing_selected_identity_does_not_fall_back_to_old_image(self):
        self.props.product_images[0].source_path=str(self.root/'missing.png')
        self.assertEqual(T.WF.identity_path(self.props,resolve=T.bpy.path.abspath,strict_source=True),'')

    def test_environment_execute_captures_camera_without_clay_and_passes_it_to_direct(self):
        props=T.props(workflow_mode=T.WF.ENVIRONMENT,render_count='1',render_mode='STANDARD',
                      detail_paste=False,iou_gate=False,layered_output=False,
                      product_images=[T.NS(image=T.NS(size=(48,24),filepath=str(self.identity)),
                                           source_path=str(self.identity),instruction='')],
                      style_images=[T.NS(image=T.NS(size=(48,24),filepath=str(self.target)),
                                         source_path=str(self.target),instruction='')])
        context=T.context(props);context.scene.name='environment scene'
        prefs=T.defaults('WONDFUL_AddonPreferences')
        prefs.structure_guides_enabled=False;prefs.strict_composition_lock=False
        prefs.codex_direct_edit=True;prefs.codex_direct_simple=True
        prefs.long_edge=1024;prefs.output_directory=''
        calls=[];clay=Mock(side_effect=AssertionError('automatic clay is forbidden'))
        T.stub('viewport_capture',render_clay_base=clay,LAST_CLAY_INFO={})
        T.stub('appearance_lock',identity_block=lambda items:'identity detail')
        T.stub('product_kind',is_vehicle=lambda props:False)
        T.stub('identity_preserve',build_identity_preserve_mask=lambda *a,**k:None,
               write_identity_edit_mask=lambda *a,**k:'')
        T.stub('jev_semantics',collect_product_object_states=lambda *a:[],analyze_appearance=lambda *a:{},
               resolve_object_semantics_batch=lambda *a:[],semantic_prompt_block=lambda *a,**k:'',
               semantic_summary=lambda *a:{'backends':{}})
        T.stub('result_worker',GATE_MAX=3,generation_budget=lambda *a,**k:1,
               call_budget_note=lambda *a,**k:'one',crop_regen_allowed=lambda *a:False,
               finish_variant=lambda path,*a,**k:{'final':path},Cancelled=type('Cancelled',(Exception,),{}))
        def direct(args, **options):
            calls.append((args,options))
            shutil.copy2(options['simple_base'] or args['edit_base_path'],args['output_path'])
            return {'engine':'mock'}
        T.stub('codex_direct',choose_edit_size=lambda *a:'1024x512',generate_with_fallback=direct)
        def capture(_context,path):
            shutil.copy2(self.target,path)
            return (48,24,'CAMERA_OPENGL_FULL_FRAME')
        def copy_refs(paths,session,prefix):
            out=[]
            for index,path in enumerate(paths):
                dest=session.directory/f'{prefix}_{index}.png';shutil.copy2(path,dest);out.append(str(dest))
            return out
        def bundle(groups,*a,**k):
            refs=[p for group in groups for p in group[1]]
            return refs,'',{'groups':[]}
        provider=dict(id='codex',label='Codex',discover=lambda:True,imagegen_enabled=True,
                      cli_path='',model='',polish_timeout=30,render_timeout=30,audit_timeout=30,
                      polish=self.polish,generate=Mock(),audit=Mock(),reference_limit=5)
        self.ns.update(_addon_prefs=lambda c:prefs,_analysis_provider=lambda *a:provider,
            capture_camera_reference=capture,camera_output_dimensions=lambda c:(48,24),
            copy_references=copy_refs,prepare_bundle=bundle,build_structure_packet=self.captures,
            compute_target_size=lambda w,h,e:(w,h,f'{w}x{h}'),build_render_prompt=lambda *a,**k:'environment render',
            _sb_ground_band_line=lambda *a:'',_appearance_context=lambda p:('BLENDER','look',['logo']),
            prompt_profiles=T.NS(metadata=lambda *a:{'target_key':'test'},needs_target_refresh=lambda *a:False,
                                 render_brief=lambda text,*a:text))
        op=self.ns['WONDFUL_OT_ai_render']()
        op._start_thread=lambda context,job:(job(),True)[1]
        self.assertEqual(op.execute(context),{'RUNNING_MODAL'})
        self.assertEqual(len(calls),1)
        args,options=calls[0]
        self.assertTrue(options['camera_capture'])
        self.assertEqual(Path(options['simple_base']),op._session.viewport_reference)
        self.assertEqual(Path(options['simple_base']).read_bytes(),self.target.read_bytes())
        self.assertEqual(Path(args['reference_paths'][0]).read_bytes(),self.target.read_bytes())
        self.assertIn(self.identity.read_bytes(),[Path(p).read_bytes() for p in args['reference_paths']])
        self.assertIn(self.target.read_bytes(),[Path(p).read_bytes() for p in args['reference_paths'][1:]])
        clay.assert_not_called()
        prefs.codex_direct_simple=False
        second=self.ns['WONDFUL_OT_ai_render']()
        second._start_thread=lambda context,job:(job(),True)[1]
        self.assertEqual(second.execute(context),{'RUNNING_MODAL'})
        args,options=calls[-1]
        self.assertTrue(options['camera_capture'])
        self.assertEqual(Path(args['edit_base_path']),second._session.viewport_reference)
        self.assertEqual(Path(args['edit_base_path']).read_bytes(),self.target.read_bytes())
        clay.assert_not_called()

    def _multiview_setup(self, *, provider_id='codex', direct_edit=True, clay_ok=True, bad_hash=False,
                         mutate_identity=False):
        source=self.root/'full_original.png'
        PNG.write_png(str(source),np.full((80,160,3),77,np.uint8))
        snapshot=self.image('resized_snapshot.png',35)
        props=T.props(workflow_mode=T.WF.REFERENCE,workflow_multiview=True,render_count='1',
                      render_mode='STANDARD',detail_paste=False,iou_gate=False,layered_output=False,
                      sb_image_path=str(source),sb_built_target_hash=('wrong' if bad_hash else T.WF.file_hash(source)),
                      product_images=[T.NS(image=T.NS(size=(48,24),filepath=str(self.identity)),
                                           source_path=str(self.identity),instruction='')])
        context=T.context(props);context.scene.name='multiview scene'
        prefs=T.defaults('WONDFUL_AddonPreferences')
        prefs.structure_guides_enabled=False;prefs.strict_composition_lock=False
        prefs.codex_direct_edit=direct_edit;prefs.codex_direct_simple=True
        prefs.long_edge=1024;prefs.output_directory=''
        def clay(_context,path,*args,**kwargs):
            if not clay_ok:
                return False
            if clay_ok == 'invalid':
                Path(path).write_bytes(b'not an image')
                return True
            PNG.write_png(path,np.full((24,48,3),125,np.uint8))
            return True
        clay_mock=Mock(side_effect=clay)
        self.isolated_stub('viewport_capture',render_clay_base=clay_mock,LAST_CLAY_INFO={})
        self.isolated_stub('terrain',settings_from_props=lambda props:{})
        self.isolated_stub('massing',find_collection=lambda *args:None,SB_TEXT='blockout')
        self.isolated_stub('appearance_lock',identity_block=lambda items:'identity detail')
        self.isolated_stub('product_kind',is_vehicle=lambda props:False)
        self.isolated_stub('identity_preserve',build_identity_preserve_mask=lambda *a,**k:None,
               write_identity_edit_mask=lambda *a,**k:'')
        self.isolated_stub('jev_semantics',collect_product_object_states=lambda *a:[],analyze_appearance=lambda *a:{},
               resolve_object_semantics_batch=lambda *a:[],semantic_prompt_block=lambda *a,**k:'',
               semantic_summary=lambda *a:{'backends':{}})
        self.isolated_stub('result_worker',GATE_MAX=3,generation_budget=lambda *a,**k:1,
               call_budget_note=lambda *a,**k:'one',crop_regen_allowed=lambda *a:False,
               finish_variant=lambda path,*a,**k:{'final':path},Cancelled=type('Cancelled',(Exception,),{}))
        self.isolated_stub('codex_direct',choose_edit_size=CD.choose_edit_size,generate_with_fallback=CD.generate_with_fallback)
        def capture(_context,path):
            shutil.copy2(snapshot,path)
            if mutate_identity:
                self.image(self.identity.name,170)
            return (48,24,'CAMERA_OPENGL_FULL_FRAME')
        capture_mock=Mock(side_effect=capture)
        def copy_refs(paths,session,prefix):
            out=[]
            for index,path in enumerate(paths):
                dest=session.directory/f'{prefix}_{index}.png';shutil.copy2(path,dest);out.append(str(dest))
            return out
        def bundle(groups,*args,**kwargs):
            refs=[];manifest=[]
            for role,paths,*_ in groups:
                for path in paths:
                    refs.append(path)
                    manifest.append({'path':path,'members':[{'path':path,'role':role}]})
            return refs,'',{'groups':manifest}
        provider=dict(id=provider_id,label=provider_id,discover=lambda:True,imagegen_enabled=True,
                      cli_path='',model='',polish_timeout=30,render_timeout=30,audit_timeout=30,
                      polish=self.polish,generate=Mock(),audit=Mock(),reference_limit=5)
        self.ns.update(workflow_preflight=lambda *args:'',_addon_prefs=lambda c:prefs,
            _analysis_provider=lambda *args:provider,_sb_generation_reference=lambda *args:(str(snapshot),''),
            capture_camera_reference=capture_mock,camera_output_dimensions=lambda c:(48,24),
            copy_references=copy_refs,prepare_bundle=bundle,build_structure_packet=self.captures,
            compute_target_size=lambda w,h,e:(w,h,f'{w}x{h}'),build_render_prompt=lambda *args,**kw:'multiview render',
            _sb_ground_band_line=lambda *args:'',_appearance_context=lambda p:('BLENDER','look',['logo']),
            prompt_profiles=T.NS(metadata=lambda *args:{'target_key':'test'},needs_target_refresh=lambda *args:False,
                                 render_brief=lambda text,*args:text))
        op=self.ns['WONDFUL_OT_ai_render']()
        op._start_thread=lambda context,job:(job(),True)[1]
        return op,context,props,source,snapshot,clay_mock,capture_mock,provider

    def test_multiview_actual_direct_payload_uses_full_original_not_snapshot(self):
        op,context,props,source,snapshot,clay,_capture,_provider=self._multiview_setup()
        with ExitStack() as stack:
            for name,value in (('find_auth_file',self.root/'auth'),('load_auth',{}),('token_expiring',False),
                               ('model_chain',['model']),('available_models',['model']),('auth_headers',{})):
                stack.enter_context(patch.object(CD,name,return_value=value))
            post=stack.enter_context(patch.object(CD,'post_responses',return_value=(
                base64.b64encode(source.read_bytes()).decode(),{},'response')))
            self.assertEqual(op.execute(context),{'RUNNING_MODAL'})
        images=[base64.b64decode(c['image_url'].split(',',1)[1]) for c in
                post.call_args.args[2]['input'][0]['content'] if c['type']=='input_image']
        self.assertEqual(images[0],source.read_bytes())
        self.assertNotEqual(images[0],snapshot.read_bytes())
        self.assertEqual(images[1],Path(clay.call_args.args[1]).read_bytes())
        self.assertEqual(images[2],self.identity.read_bytes())
        self.assertEqual(op._direct_state['original_first'],True)

    def test_multiview_actual_cli_fallback_keeps_full_original_and_projection(self):
        op,context,_props,source,snapshot,clay,_capture,provider=self._multiview_setup()
        provider['generate']=CC.generate_image
        calls=[]
        def exec_turn(**kwargs):
            calls.append(kwargs)
            (op._session.directory/'output_candidate_v01_a01.png').write_bytes(source.read_bytes())
            return 'WONDFUL_IMAGEGEN_OK'
        with patch.object(CD,'edit_image',side_effect=CD.DirectEditError('direct endpoint',kind='request')), \
             patch.object(CC,'_exec_turn',side_effect=exec_turn):
            self.assertEqual(op.execute(context),{'RUNNING_MODAL'})
        images=[Path(path).read_bytes() for path in calls[0]['image_paths']]
        self.assertEqual(images[0],source.read_bytes())
        self.assertNotEqual(images[0],snapshot.read_bytes())
        self.assertEqual(images[1],Path(clay.call_args.args[1]).read_bytes())
        self.assertIn(self.identity.read_bytes(),images[2:])
        self.assertEqual(op._direct_state['fallbacks'],1)
        self.assertTrue(op._direct_state['original_first'])

    def test_multiview_unsupported_provider_or_disabled_direct_fails_before_capture(self):
        for provider_id, enabled in (('antigravity',True),('codex',False)):
            with self.subTest(provider=provider_id,enabled=enabled):
                op,context,props,_source,_snapshot,clay,capture,provider=self._multiview_setup(
                    provider_id=provider_id,direct_edit=enabled)
                self.assertEqual(op.execute(context),{'CANCELLED'})
                self.assertIn('Codex',props.last_error)
                capture.assert_not_called();clay.assert_not_called();provider['generate'].assert_not_called()

    def test_multiview_missing_projection_stops_instead_of_using_camera_fallback(self):
        op,context,props,_source,_snapshot,clay,_capture,provider=self._multiview_setup(clay_ok=False)
        with patch.object(CD,'post_responses') as post:
            self.assertEqual(op.execute(context),{'CANCELLED'})
        self.assertIn('产品投影',props.last_error)
        clay.assert_called_once();post.assert_not_called();provider['generate'].assert_not_called()

    def test_multiview_projection_exception_reports_stop_without_fallback(self):
        op,context,props,_source,_snapshot,clay,_capture,provider=self._multiview_setup()
        clay.side_effect=RuntimeError('projection failed')
        with patch.object(CD,'post_responses') as post, patch('builtins.print') as printer:
            self.assertEqual(op.execute(context),{'CANCELLED'})
        self.assertIn('产品投影',props.last_error)
        self.assertTrue(any('已停止' in str(call) for call in printer.call_args_list))
        self.assertFalse(any('改用相机截图' in str(call) for call in printer.call_args_list))
        post.assert_not_called();provider['generate'].assert_not_called()

    def test_multiview_invalid_projection_stops_before_provider(self):
        op,context,props,_source,_snapshot,clay,_capture,provider=self._multiview_setup(clay_ok='invalid')
        with patch.object(CD,'post_responses') as post:
            self.assertEqual(op.execute(context),{'CANCELLED'})
        self.assertIn('产品投影',props.last_error)
        clay.assert_called_once();post.assert_not_called();provider['generate'].assert_not_called()

    def test_multiview_changed_original_or_identity_during_capture_stops_before_provider(self):
        for kwargs in ({'bad_hash':True},{'mutate_identity':True}):
            with self.subTest(kwargs=kwargs):
                op,context,props,_source,_snapshot,clay,_capture,provider=self._multiview_setup(**kwargs)
                with patch.object(CD,'post_responses') as post:
                    self.assertEqual(op.execute(context),{'CANCELLED'})
                clay.assert_not_called();post.assert_not_called();provider['generate'].assert_not_called()

    def test_empty_polish_stops_generation(self):
        self.polish.return_value='  '
        op=self.ns['WONDFUL_OT_polish_prompt']()
        def start(context,job):
            op._origin_scene=context.scene;op._started_at=time.time();op._thread=None;op._error=None;op._result=job();return True
        op._start_thread=start;op._cleanup=lambda context:None
        self.props.pending_generate=True
        self.assertEqual(op.execute(self.context),{'RUNNING_MODAL'})
        self.assertEqual(op._modal(self.context,T.NS(type='TIMER')),{ 'CANCELLED'})
        self.assertFalse(self.calls)


class SourceUILayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        T.setUpModule()
        cls.ui=T.ui_module()
    @classmethod
    def tearDownClass(cls):
        T.tearDownModule()
    def draw(self,props, missing=''):
        context=T.context(props)
        if missing == 'camera':
            context.scene.camera=None
        elif missing == 'collection':
            props.product_collection=None
        layout=T.Layout()
        self.ui.WONDFUL_PT_main._draw_main(T.NS(layout=layout),context)
        return layout.log
    def test_single_requires_camera_and_collection_without_requiring_blockout(self):
        props=T.props(workflow_mode=T.WF.REFERENCE,codex_account_state='LOGGED_IN',sb_image_path='target.png')
        log=self.draw(props)
        self.assertTrue(next(item[2] for item in log if item[:2]==('operator','wondful.generate')))
        self.assertFalse(any(item[:2]==('operator','wondful.scene_blockout_build') for item in log))
        for missing in ('camera','collection'):
            log=self.draw(props, missing)
            self.assertFalse(next(item[2] for item in log if item[:2]==('operator','wondful.generate')))
        self.assertTrue(any(item[:2]==('prop','product_collection') for item in log))
        self.assertTrue(any(item[:2]==('prop','camera') for item in log))
    def test_photo_finish_is_disabled_only_for_single_reference(self):
        for mode, multiview, enabled in ((T.WF.REFERENCE, False, False),
                                          (T.WF.REFERENCE, True, True),
                                          (T.WF.ENVIRONMENT, False, True)):
            with self.subTest(mode=mode, multiview=multiview):
                props = T.props(workflow_mode=mode, workflow_multiview=multiview,
                                ui_show_advanced=True, ui_adv_tab='FINISH',
                                photo_finish='STANDARD', last_finish_note='旧效果报告')
                log = self.draw(props)
                self.assertEqual([item for item in log if item[:2] == ('prop', 'photo_finish')],
                                 [('prop', 'photo_finish', enabled)])
                labels = [item[1] for item in log if item[0] == 'label']
                if not enabled:
                    self.assertIn('单图复刻不加照片质感后期', labels)
                    self.assertIn('保留生成结果的原有外观', labels)
                    self.assertFalse(any('旧效果报告' in label for label in labels))
                else:
                    self.assertNotIn('单图复刻不加照片质感后期', labels)
                self.assertEqual(props.photo_finish, 'STANDARD')
    def test_blockout_tools_once_only_in_multiview(self):
        props=T.props(workflow_mode=T.WF.REFERENCE,codex_account_state='LOGGED_IN',sb_image_path='target.png',
                      workflow_multiview=True,ui_show_advanced=True,ui_adv_tab='MULTIVIEW',ui_show_replica_advanced=True)
        log=self.draw(props)
        for name in ('wondful.scene_blockout_build','wondful.replica_preview','wondful.replica_refine','wondful.replica_undo'):
            self.assertEqual(sum(item[:2]==('operator',name) for item in log),1,name)
        for name in ('sb_correct','sb_precise','sb_complete','replica_feedback'):
            self.assertEqual(sum(item[:2]==('prop',name) for item in log),1,name)
    def test_blockout_progress_has_enabled_cancel_when_advanced_was_closed(self):
        props=T.props(workflow_mode=T.WF.REFERENCE,workflow_multiview=True,status='BLOCKOUT',
                      ui_show_advanced=False,ui_adv_tab='MODEL')
        log=self.draw(props)
        self.assertTrue(any(item[:2]==('operator','wondful.cancel_task') and item[2] for item in log))


if __name__ == '__main__':
    unittest.main()
