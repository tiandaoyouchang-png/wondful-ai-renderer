"""UI-r2 behavioural regression. Native RNA/renderer boundaries are simulated.

No screenshots, Blender rendering, provider login or paid model call is performed.
"""
import ast
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import test_two_workflows as T

NS=T.NS
ROOT=T.ROOT
PKG=T.PKG
WF=T.WF
RR=T.load('replica_revision')
RB=T.load('replica_revision_blender')
UIC=T.load('ui_components')
SB=T.load('scene_blockout')


def setUpModule():
    T.setUpModule()
    T.bpy_types.OperatorFileListElement = object


def tearDownModule():
    T.tearDownModule()


def extract(functions=(), classes=()):
    tree=ast.parse((PKG/'operators.py').read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions
              or isinstance(n,ast.ClassDef) and n.name in classes]
    env=dict(T.OPS)
    env.update(ImportHelper=type('ImportHelper',(),{}), IntProperty=lambda **k:k.get('default',0),
               BoolProperty=lambda **k:k.get('default',False), EnumProperty=lambda **k:k.get('default',''),
               CollectionProperty=lambda **k:[],OperatorFileListElement=object,
               StringProperty=lambda **k:k.get('default',''),MAX_REFERENCES_PER_KIND=8,
               MAX_PRODUCT_REFERENCES=1, reference_limit=lambda kind:1 if kind=='PRODUCT' else 8)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(PKG/'operators.py'),'exec'),env)
    return env


class RNACollection(list):
    def add(self):
        item=NS(image=None,source_path='',instruction='')
        self.append(item)
        return item
    def remove(self,index):
        del self[index]
    def move(self,a,b):
        self.insert(b,self.pop(a))


def reply():
    return dict(summary='Adjusted',unresolved=[],camera=dict(apply=False,translation=[0,0,0],
        rotation_deg=[0,0,0],lens_mm=45,shift_x=0,shift_y=0),edits=[],ribbons=[],additions=[])


def scene_snapshot():
    return {'signature':'before','span':50,'objects':[{'id':'o000','kind':'building'},
        {'id':'o001','kind':'ribbon'}]}


def edit(ident='o000'):
    return dict(id=ident,translation=[1,0,0],rotation_deg=[0,0,0],scale=[1,1,1],hidden=False)


def ribbon():
    return dict(id='o001',thickness_m=.2,sections=[
        {'left':{'x':-4,'y':5,'z':1},'right':{'x':4,'y':5,'z':1}},
        {'left':{'x':-4,'y':9,'z':2},'right':{'x':4,'y':9,'z':3}}])


class RequiredIdentityTests(unittest.TestCase):
    def test_no_identity_in_both_modes_is_blocked(self):
        for mode in (WF.ENVIRONMENT,WF.REFERENCE):
            p=T.props(workflow_mode=mode,product_images=[])
            self.assertFalse(WF.identity_available(p))
            self.assertTrue(WF.preflight(p,has_camera=True,has_product=True,has_blockout=False))
    def test_invalid_first_slot_does_not_fall_back_to_second(self):
        p=T.props();valid=p.product_images[0]
        p.product_images=[NS(image=None),valid]
        self.assertFalse(WF.identity_available(p))
        self.assertEqual(WF.identity_path(p),'')
    def test_zero_sized_and_stale_image_handles_are_unavailable(self):
        for image in (NS(size=(0,30)),NS(size=(20,0)),NS(),None):
            self.assertFalse(WF.identity_available(T.props(product_images=[NS(image=image)])))
    def test_missing_or_empty_disk_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.png';path.touch()
            p=T.props(product_images=[NS(image=NS(size=(10,10),filepath=str(path)),source_path=str(path))])
            self.assertEqual(WF.identity_path(p),'')
            path.unlink();self.assertEqual(WF.identity_path(p),'')
    def test_packed_identity_materialized_only_at_action_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'packed.png';path.write_bytes(b'png')
            image=NS(size=(10,10),filepath='',packed_file=object())
            p=T.props(product_images=[NS(image=image,source_path='')]);export=Mock(return_value=str(path))
            self.assertEqual(WF.identity_path(p,packed_export=export),str(path))
            export.assert_called_once_with(image)
    def test_decoder_failure_cannot_reach_remote_request(self):
        p=T.props();c=T.context(p)
        with patch.dict(T.OPS,{'_load_reference_image':Mock(side_effect=ValueError('bad pixels'))}):
            self.assertTrue(T.OPS['identity_preflight'](p))
            op=T.OPS['WONDFUL_OT_generate']()
            self.assertEqual(op.execute(c),{'CANCELLED'})
    def test_manual_approval_value_does_not_gate_environment(self):
        p=T.props(replica_reviewed=False)
        self.assertEqual(WF.preflight(p,has_camera=True,has_product=True,has_blockout=False),'')
        p.replica_reviewed=True
        self.assertEqual(WF.preflight(p,has_camera=True,has_product=True,has_blockout=False),'')
    def test_all_remote_entrypoints_have_preflight(self):
        tree=ast.parse((PKG/'operators.py').read_text())
        for name in ('WONDFUL_OT_generate','WONDFUL_OT_ai_render','WONDFUL_OT_scene_blockout_build',
                     'WONDFUL_OT_polish_prompt','WONDFUL_OT_multiview_render','WONDFUL_OT_replica_refine'):
            node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
            execute=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
            calls=[ast.unparse(n.func) for n in ast.walk(execute) if isinstance(n,ast.Call)]
            self.assertTrue('workflow_preflight' in calls or 'identity_preflight' in calls,name)


class ReferenceActionTests(unittest.TestCase):
    def namespace(self,classes=()):
        ns=extract(('_collection_and_index','_mark_reference_changed','_resolve_reference_index',
                    '_append_reference_from_path','_replace_style_set','_replace_product_reference'),classes)
        ns['_load_reference_image']=lambda path:NS(name=str(path),size=(100,50),filepath=str(path))
        return ns
    def test_append_keeps_existing_style_images_and_notes(self):
        ns=self.namespace();old=NS(image=object(),source_path='old',instruction='keep')
        p=T.props(style_images=RNACollection([old]))
        ns['_append_reference_from_path'](p,'STYLE','new')
        self.assertIs(p.style_images[0],old);self.assertEqual(old.instruction,'keep')
        self.assertEqual(p.style_images[1].source_path,'new')
    def test_failed_append_or_replace_keeps_old_reference(self):
        ns=self.namespace();ns['_load_reference_image']=Mock(side_effect=ValueError('bad image'))
        old=object();p=T.props(style_images=RNACollection([old]))
        with self.assertRaises(ValueError):ns['_append_reference_from_path'](p,'STYLE','bad')
        self.assertEqual(p.style_images,[old])
        count,_=ns['_replace_style_set'](p,['bad'])
        self.assertEqual(count,0);self.assertEqual(p.style_images,[old])
    def test_style_limit_checked_before_decoding(self):
        ns=self.namespace();loader=ns['_load_reference_image']=Mock()
        p=T.props(style_images=RNACollection([object()]*8))
        with self.assertRaises(RuntimeError):ns['_append_reference_from_path'](p,'STYLE','new')
        loader.assert_not_called()
    def test_product_replacement_preserves_note_and_clears_excess_slots(self):
        ns=self.namespace();p=T.props(product_images=RNACollection([
            NS(image=object(),source_path='a',instruction='preserve note'),NS(image=object())]))
        ns['_replace_product_reference'](p,'new')
        self.assertEqual(len(p.product_images),1)
        self.assertEqual(p.product_images[0].instruction,'preserve note')
    def test_stale_card_index_does_not_delete_last_image(self):
        ns=self.namespace();p=T.props(style_images=[object()],style_image_index=0)
        _,_,index=ns['_resolve_reference_index'](p,'STYLE',3)
        self.assertEqual(index,-1)
    def test_paste_replaces_exact_person_slot_even_when_collection_is_full(self):
        ns=self.namespace(('WONDFUL_OT_paste_reference',))
        with tempfile.TemporaryDirectory() as tmp:
            ns.update(CONFIG_DIR=Path(tmp),uuid=__import__('uuid'),save_clipboard_image=lambda p:p,
                      ClipboardImageError=type('ClipboardImageError',(Exception,),{}))
            slots=RNACollection([NS(image=object(),instruction=str(i),source_path=str(i)) for i in range(8)])
            p=T.props(person_images=slots);c=T.context(p);others=[x.image for x in slots]
            op=ns['WONDFUL_OT_paste_reference']();op.ref_kind='PERSON';op.item_index=3;op.append_to_set=False
            self.assertEqual(op.execute(c),{'FINISHED'})
            self.assertEqual(len(slots),8);self.assertEqual(slots[3].instruction,'3')
            for i in (0,1,2,4,5,6,7):self.assertIs(slots[i].image,others[i])
    def test_busy_reference_actions_refuse_direct_execution(self):
        classes=('WONDFUL_OT_load_reference','WONDFUL_OT_paste_reference','WONDFUL_OT_replace_reference',
                 'WONDFUL_OT_clear_reference','WONDFUL_OT_clear_all_references','WONDFUL_OT_move_reference')
        ns=self.namespace(classes);c=T.context(T.props(status='RENDERING'))
        for name in classes:
            with self.subTest(name=name):self.assertEqual(ns[name]().execute(c),{'CANCELLED'})
    def load_actions(self):
        extras=types.ModuleType('bpy_extras');io=types.ModuleType('bpy_extras.io_utils')
        io.ImportHelper=type('ImportHelper',(),{});extras.io_utils=io
        with patch.dict(sys.modules,{'bpy_extras':extras,'bpy_extras.io_utils':io}),\
             patch.object(T.bpy_props,'StringProperty',lambda **kw:kw.get('default',''),create=True):
            return T.load('ui_actions')
    def test_target_reselect_same_path_invalidates_stale_preview(self):
        actions=self.load_actions();p=T.props(workflow_mode=WF.REFERENCE,sb_image_path=str(ROOT/'target.png'))
        p.sb_preview_image=object();c=T.context(p)
        loader=Mock(return_value=NS(size=(50,50)))
        op=types.ModuleType(T.PACKAGE+'.operators');op._load_reference_image=loader
        with patch.dict(sys.modules,{op.__name__:op}),patch.object(T.package,'operators',op),\
             patch.object(WF,'target_changed') as change:
            actions.set_target(c,p.sb_image_path)
        change.assert_called_once_with(p,c)
    def test_target_failure_preserves_path_and_preview(self):
        actions=self.load_actions();p=T.props(sb_image_path='old');old=object();p.sb_preview_image=old
        op=types.ModuleType(T.PACKAGE+'.operators');op._load_reference_image=Mock(side_effect=ValueError('bad'))
        with patch.dict(sys.modules,{op.__name__:op}),patch.object(T.package,'operators',op):
            with self.assertRaises(ValueError):actions.set_target(T.context(p),'bad')
        self.assertEqual(p.sb_image_path,'old');self.assertIs(p.sb_preview_image,old)


class DeclarativeRevisionTests(unittest.TestCase):
    def test_empty_plan_is_honest_no_change(self):
        p=reply();p['unresolved']=['Unsupported lighting request']
        out=RR.validate(p,scene_snapshot());self.assertTrue(out['no_change']);self.assertTrue(out['unresolved'])
    def test_transform_and_camera_supported_without_mutating_input(self):
        p=reply();p['edits']=[edit()];p['camera']['apply']=True;before=copy.deepcopy(p)
        out=RR.validate(p,scene_snapshot());self.assertFalse(out.get('no_change',False));self.assertEqual(p,before)
    def test_unknown_target_rejected(self):
        p=reply();p['edits']=[edit('vehicle')]
        with self.assertRaises(ValueError):RR.validate(p,scene_snapshot())
    def test_duplicate_target_rejected(self):
        p=reply();p['edits']=[edit(),edit()]
        with self.assertRaises(ValueError):RR.validate(p,scene_snapshot())
    def test_unsafe_extra_fields_rejected(self):
        p=reply();p['execute']='import os'
        with self.assertRaises(ValueError):RR.validate(p,scene_snapshot())
    def test_nested_extra_fields_rejected(self):
        p=reply();p['edits']=[dict(edit(),path='/tmp/model.blend')]
        with self.assertRaises(ValueError):RR.validate(p,scene_snapshot())
    def test_non_finite_numbers_and_booleans_rejected(self):
        for bad in (math.inf,math.nan,-math.inf,True,'4',None):
            p=reply();p['edits']=[edit()];p['edits'][0]['translation'][0]=bad
            with self.subTest(value=bad),self.assertRaises(ValueError):RR.validate(p,scene_snapshot())
    def test_excessive_motion_scaling_and_lens_rejected(self):
        for field,value in [('translation',[10000,0,0]),('scale',[0,1,1]),('scale',[-1,1,1]),('rotation_deg',[181,0,0])]:
            p=reply();p['edits']=[edit()];p['edits'][0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):RR.validate(p,scene_snapshot())
        p=reply();p['camera'].update(apply=True,lens_mm=0)
        with self.assertRaises(ValueError):RR.validate(p,scene_snapshot())
    def test_ribbon_geometry_validated_with_real_builder(self):
        p=reply();p['ribbons']=[ribbon()]
        self.assertEqual(len(RR.validate(p,scene_snapshot(),SB.ribbon_mesh)['ribbons']),1)
    def test_degenerate_ribbon_rejected_before_mutation(self):
        p=reply();r=ribbon();r['sections'][0]['right']=r['sections'][0]['left'];p['ribbons']=[r]
        with self.assertRaises(ValueError):RR.validate(p,scene_snapshot(),SB.ribbon_mesh)
    def test_non_ribbon_cannot_accept_surface_replacement(self):
        p=reply();r=ribbon();r['id']='o000';p['ribbons']=[r]
        with self.assertRaises(ValueError):RR.validate(p,scene_snapshot(),SB.ribbon_mesh)
    def test_bounded_box_addition_supported(self):
        p=reply();p['additions']=[dict(label='Wall',location=[0,3,0],dimensions=[2,1,3],rotation_deg=[0,0,0])]
        self.assertEqual(RR.validate(p,scene_snapshot())['additions'][0]['label'],'Wall')
    def test_invalid_boolean_flags_rejected(self):
        p=reply();p['camera']['apply']='true'
        with self.assertRaises(ValueError):RR.validate(p,scene_snapshot())
        p=reply();p['edits']=[dict(edit(),hidden=1)]
        with self.assertRaises(ValueError):RR.validate(p,scene_snapshot())
    def test_limits_on_arrays_and_text(self):
        for key,value in [('edits',[edit()]*25),('unresolved',['x']*9),('summary','x'*1201)]:
            p=reply();p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):RR.validate(p,scene_snapshot())
    def test_feedback_is_in_actual_api_content_with_three_distinct_roles(self):
        c=RR.content('target','preview','identity',scene_snapshot(),'Move left wall outward')
        self.assertEqual([x['image_url'] for x in c if x['type']=='input_image'],['target','preview','identity'])
        self.assertIn('Move left wall outward',c[0]['text']);self.assertIn('WORLD',c[0]['text'])
        self.assertIn('OBJECT-LOCAL',c[0]['text']);self.assertIn('unresolved',c[0]['text'])
    def test_empty_or_long_feedback_rejected(self):
        for feedback in ('','   ','x'*4001):
            with self.assertRaises(ValueError):RR.content('a','b','c',scene_snapshot(),feedback)
    def test_schema_is_strict_at_every_object_level(self):
        def check(v):
            if isinstance(v,dict):
                if v.get('type')=='object':
                    self.assertFalse(v['additionalProperties'])
                    self.assertEqual(set(v['required']),set(v['properties']))
                for sub in v.values():check(sub)
            elif isinstance(v,list):
                for sub in v:check(sub)
        check(RR.SCHEMA)


class RevisionTransactionTests(unittest.TestCase):
    def setUp(self):
        RB._UNDO.clear();self.p=T.props();self.c=T.context(self.p);self.c.scene.as_pointer=lambda:123
        self.tx=Mock(result_signature='after',states=[(None,None,'retained-mesh',False,False)])
    def tearDown(self):RB._UNDO.clear()
    def patching(self,snap=None):
        return patch.object(RB,'snapshot',side_effect=snap or [(scene_snapshot(),{},None),
            (dict(scene_snapshot(),signature='after'),{},None)])
    def test_stale_request_cannot_start_transaction(self):
        with self.patching(),patch.object(RB,'Transaction') as tx:
            with self.assertRaises(RuntimeError):RB.apply_plan(self.c,self.p,reply(),'stale',Mock())
        tx.assert_not_called()
    def test_invalid_plan_cannot_start_transaction(self):
        p=reply();p['edits']=[edit('product')]
        with self.patching(),patch.object(RB,'Transaction') as tx:
            with self.assertRaises(ValueError):RB.apply_plan(self.c,self.p,p,'before',Mock())
        tx.assert_not_called()
    def test_no_change_does_not_create_undo_or_preview(self):
        renderer=Mock()
        with self.patching(),patch.object(RB,'Transaction') as tx:
            out=RB.apply_plan(self.c,self.p,reply(),'before',renderer)
        tx.assert_not_called();renderer.assert_not_called();self.assertTrue(out['no_change'])
        self.assertFalse(self.p.replica_can_undo)
    def test_preview_failure_rolls_back_scene_and_leaves_counter(self):
        p=reply();p['edits']=[edit()]
        with self.patching(),patch.object(RB,'Transaction',return_value=self.tx):
            with self.assertRaises(RuntimeError):RB.apply_plan(self.c,self.p,p,'before',Mock(side_effect=RuntimeError('render failed')))
        self.tx.rollback.assert_called_once();self.assertEqual(self.p.replica_revision_count,0)
        self.assertFalse(RB._UNDO)
    def test_apply_failure_rolls_back_without_running_preview(self):
        p=reply();p['edits']=[edit()];renderer=Mock();self.tx.apply.side_effect=ValueError('bad native geometry')
        with self.patching(),patch.object(RB,'Transaction',return_value=self.tx):
            with self.assertRaises(ValueError):RB.apply_plan(self.c,self.p,p,'before',renderer)
        self.tx.rollback.assert_called_once();renderer.assert_not_called()
    def test_success_keeps_one_step_undo(self):
        p=reply();p['edits']=[edit()];old=Mock();RB._UNDO[123]=old
        with self.patching(),patch.object(RB,'Transaction',return_value=self.tx):
            RB.apply_plan(self.c,self.p,p,'before',Mock())
        old.dispose.assert_called_once_with(retain_meshes=['retained-mesh']);self.assertIs(RB._UNDO[123],self.tx)
        self.assertEqual(self.p.replica_revision_count,1);self.assertTrue(self.p.replica_can_undo)
    def test_undo_refuses_to_destroy_later_manual_edit(self):
        RB._UNDO[123]=self.tx
        with patch.object(RB,'snapshot',return_value=(dict(scene_snapshot(),signature='manual'),{},None)):
            with self.assertRaises(RuntimeError):RB.undo(self.c,self.p)
        self.tx.rollback.assert_not_called();self.assertIn(123,RB._UNDO)
    def test_undo_restores_and_clears_state(self):
        RB._UNDO[123]=self.tx;self.p.replica_can_undo=True;self.p.replica_revision_count=2
        with patch.object(RB,'snapshot',return_value=(dict(scene_snapshot(),signature='after'),{},None)):
            RB.undo(self.c,self.p)
        self.tx.rollback.assert_called_once();self.assertFalse(self.p.replica_can_undo)
        self.assertEqual(self.p.replica_revision_count,1);self.assertNotIn(123,RB._UNDO)
    def test_missing_undo_explicitly_fails(self):
        self.p.replica_can_undo=True
        with self.assertRaises(RuntimeError):RB.undo(self.c,self.p)
        self.assertFalse(self.p.replica_can_undo)
    def test_revision_preview_uses_different_path_for_rollback(self):
        tree=ast.parse((PKG/'operators.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WONDFUL_OT_replica_refine')
        self.assertIn("self._revision_session / 'revised'",ast.unparse(node))
    def test_preview_refresh_does_not_invalidate_target_or_prompt(self):
        tree=ast.parse((PKG/'operators.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_render_replica_preview')
        self.assertNotIn('target_changed',ast.unparse(node))


class RecordingLayout(T.Layout):
    def child(self,*a,**kw):return RecordingLayout(self.log,self)
    row=column=box=split=grid_flow=child
    def template_icon(self,*a,**kw):self.log.append(('thumbnail',kw.get('scale'),self.effective_enabled()))
    def operator(self,idname,**kw):
        op=super().operator(idname,**kw)
        self.log.append(('operator_props',idname,op))
        return op


class ConsistentUILayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.ui=T.ui_module()
    def draw(self,p,width=380):
        c=T.context(p);c.region.width=width;layout=RecordingLayout()
        with patch.object(T.bpy,'context',c,create=True):self.ui.WONDFUL_PT_main._draw_main(NS(layout=layout),c)
        return layout.log
    def test_no_review_checkbox_anywhere_in_panel(self):
        for mode in (WF.ENVIRONMENT,WF.REFERENCE):
            log=self.draw(T.props(workflow_mode=mode))
            self.assertFalse(any(x[:2]==('prop','replica_reviewed') for x in log))
    def test_identity_visible_even_when_legacy_foldout_is_collapsed(self):
        log=self.draw(T.props(ui_show_product_refs=False))
        self.assertTrue(any(x[0]=='label' and '\u4ea7\u54c1\u8eab\u4efd\u53c2\u8003' in x[1] for x in log))
        self.assertTrue(any(x[0]=='thumbnail' for x in log))
    def test_missing_identity_disables_generate_and_build(self):
        for mode in (WF.ENVIRONMENT,WF.REFERENCE):
            log=self.draw(T.props(workflow_mode=mode,product_images=[],codex_account_state='LOGGED_IN',sb_image_path='target.png'))
            for item in log:
                if item[:2] in [('operator','wondful.generate'),('operator','wondful.scene_blockout_build')]:self.assertFalse(item[2])
    def test_explanatory_prose_in_tooltips_not_labels(self):
        for mode in (WF.ENVIRONMENT,WF.REFERENCE):
            log=self.draw(T.props(workflow_mode=mode))
            labels=[x[1] for x in log if x[0]=='label']
            for prose in UIC.HELP.values():self.assertNotIn(prose,labels)
            self.assertTrue(any(x[:2]==('operator','wondful.ui_help') for x in log))
    def test_feedback_field_and_action_present_only_in_replica(self):
        env=self.draw(T.props());ref=self.draw(T.props(workflow_mode=WF.REFERENCE))
        self.assertFalse(any(x[:2]==('prop','replica_feedback') for x in env))
        self.assertTrue(any(x[:2]==('prop','replica_feedback') for x in ref))
        self.assertTrue(any(x[:2]==('operator','wondful.replica_refine') for x in ref))
    def test_add_button_has_append_semantics(self):
        log=self.draw(T.props())
        entries=[x[2] for x in log if x[:2]==('operator_props','wondful.load_reference')]
        self.assertTrue(any(getattr(op,'ref_kind',None)=='STYLE' and getattr(op,'append_to_set',False) for op in entries))
    def test_all_card_types_have_same_preview_and_note_structure(self):
        p=T.props();c=T.context(p);image=T.FakeImage();p.sb_reference_image=image;p.sb_image_path=image.filepath
        counts=[]
        for kind in ('PRODUCT','STYLE','TARGET','PERSON'):
            layout=RecordingLayout();item=NS(image=image,instruction='')
            UIC.image_slot(layout,c,p,kind,item=item,title=kind,required=True)
            counts.append((sum(x[0]=='thumbnail' for x in layout.log),sum(x[0]=='prop' for x in layout.log)))
            self.assertTrue(any(x[0]=='label' and x[1].endswith(' *') for x in layout.log))
        self.assertEqual(counts,[(1,1)]*4)
    def test_empty_cards_keep_note_and_image_well(self):
        p=T.props();c=T.context(p)
        for kind in ('PRODUCT','STYLE','TARGET','PERSON'):
            layout=RecordingLayout();UIC.image_slot(layout,c,p,kind,title=kind)
            self.assertTrue(any(x[0]=='label' and x[1]=='' for x in layout.log))
            self.assertTrue(any(x[0]=='prop' for x in layout.log))
    def test_help_operator_text_is_hover_only(self):
        actions=ReferenceActionTests().load_actions()
        self.assertEqual(actions.WONDFUL_OT_ui_help.description(None,NS(message='hover text')),'hover text')
        layout=RecordingLayout();UIC.help_icon(layout,'hover text')
        self.assertFalse(any(x[0]=='label' for x in layout.log))
        self.assertEqual(next(x[2].message for x in layout.log if x[0]=='operator_props'),'hover text')
    def test_preview_size_bounded_across_sidebar_sizes(self):
        c=T.context()
        for width in (160,240,380,640,1600):
            c.region.width=width
            for scale in (.75,1,2):
                c.preferences.system.ui_scale=scale
                size=UIC.preview_scale(c);self.assertGreaterEqual(size,3);self.assertLessEqual(size,UIC.PREVIEW_MAX)
    def test_all_four_section_numbers_stable(self):
        for mode in (WF.ENVIRONMENT,WF.REFERENCE):
            log=self.draw(T.props(workflow_mode=mode));labels=[x[1] for x in log if x[0]=='label']
            for number in self.ui.STEP_NUMBERS[:4]:self.assertTrue(any(x.startswith(number) for x in labels))
    def test_ui_matrix_populated_references_narrow_and_wide(self):
        count=0
        for mode in (WF.ENVIRONMENT,WF.REFERENCE):
            for provider in ('CODEX','ANTIGRAVITY'):
                for tab in ('MODEL','OUTPUT','FINISH','PRODUCT','SCENE','MULTIVIEW','DIAG'):
                    for state in ('IDLE','RENDERING'):
                        for width in (240,640):
                            p=T.props(workflow_mode=mode,analysis_provider=provider,ui_show_advanced=True,ui_adv_tab=tab,
                                codex_account_state='LOGGED_IN',antigravity_account_state='LOGGED_IN',status=state,
                                style_images=[NS(image=T.FakeImage('Style'),instruction='')],
                                person_images=[NS(image=T.FakeImage('Person'),instruction='')],
                                sb_image_path='/tmp/target.png',sb_reference_image=T.FakeImage('Target'),
                                sb_preview_image=T.FakeImage('Preview'),result_image=T.FakeImage('Result'),
                                replica_feedback='Move wall',replica_can_undo=True)
                            with self.subTest(mode=mode,provider=provider,tab=tab,state=state,width=width):
                                log=self.draw(p,width)
                                if state=='RENDERING':
                                    for row in log:
                                        if row[0]=='operator' and row[1] in ('wondful.generate','wondful.target_load','wondful.replica_refine','wondful.replace_reference','wondful.replica_undo'):
                                            self.assertFalse(row[2])
                            count+=1
        self.assertEqual(count,112)
    def test_every_ui_operator_is_declared(self):
        definitions=set();used=set()
        for f in PKG.glob('*.py'):
            tree=ast.parse(f.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bl_idname' for t in node.targets) and isinstance(node.value,ast.Constant):
                    definitions.add(node.value.value)
                if f.name in ('ui.py','ui_components.py') and isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='operator' and node.args and isinstance(node.args[0],ast.Constant):
                    name=node.args[0].value
                    if isinstance(name,str) and name.startswith('wondful.'):used.add(name)
        self.assertFalse(used-definitions,used-definitions)
    def test_new_operators_registered_and_cleanup_present(self):
        src=(PKG/'__init__.py').read_text()
        for name in ('WONDFUL_OT_replica_refine','WONDFUL_OT_replica_undo','ui_actions.CLASSES','preview_cache.reset()','replica_revision_blender.reset()'):
            self.assertIn(name,src)



class ThumbnailCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache=T.load('preview_cache')
        self.timers=NS(register=Mock(),is_registered=lambda f:True,unregister=Mock())
        self.oldutils=getattr(T.bpy,'utils',None)
        self.utils=types.ModuleType('bpy.utils');self.previews=types.ModuleType('bpy.utils.previews')
        class Collection(dict):
            def __init__(self):super().__init__();self.loads=0
            def load(self,key,path,kind,force_reload=False):
                self.loads+=1;thumb=NS(icon_id=10+self.loads);self[key]=thumb;return thumb
            def new(self,key):
                thumb=NS(icon_id=999);self[key]=thumb;return thumb
        self.collection=Collection()
        self.previews.new=lambda:self.collection;self.previews.remove=Mock()
        self.utils.previews=self.previews
        self.stack=__import__('contextlib').ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(sys.modules,{'bpy.utils':self.utils,'bpy.utils.previews':self.previews}))
        self.stack.enter_context(patch.object(T.bpy,'utils',self.utils,create=True))
        self.stack.enter_context(patch.object(T.bpy.app,'timers',self.timers,create=True))
        self.stack.enter_context(patch.object(T.bpy,'context',NS(window_manager=NS(windows=[])),create=True))
    def test_draw_schedules_once_without_file_io(self):
        with patch.object(self.cache.os,'stat',side_effect=AssertionError('no IO in draw')):
            self.assertEqual(self.cache.icon_for_path('/missing/a.png'),0)
            self.assertEqual(self.cache.icon_for_path('/missing/b.png'),0)
        self.timers.register.assert_called_once()
    def test_tick_refreshes_changed_file_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.png';path.write_bytes(b'a')
            self.cache.icon_for_path(str(path));self.cache._tick();self.assertEqual(self.collection.loads,1)
            self.cache._pending.add(str(path));self.cache._tick();self.assertEqual(self.collection.loads,1)
            path.write_bytes(b'longer')
            self.cache._pending.add(str(path));self.cache._tick();self.assertEqual(self.collection.loads,2)
    def test_missing_file_becomes_empty_preview_without_crash(self):
        self.cache.icon_for_path('/missing/definitely-not-a-file.png');self.cache._tick()
        self.assertEqual(self.cache._records['/missing/definitely-not-a-file.png']['icon'],0)
    def test_lru_bounds_memory_and_unregister_cleans_up(self):
        for i in range(55):self.cache.icon_for_path('/missing/%s.png'%i)
        self.cache._tick();self.assertLessEqual(len(self.cache._records),48)
        self.cache.reset();self.assertFalse(self.cache._records);self.previews.remove.assert_called_once()
    def test_packed_reference_uses_pixels_not_old_disk_path(self):
        image=NS(as_pointer=lambda:3,filepath='/missing/old.png',packed_file=object(),source='FILE',
                 size=(4,2),pixels=[.2,.4,.6,1]*8,is_dirty=False,name_full='Packed')
        self.cache.icon_for_image(image);self.cache._tick()
        self.assertEqual(self.cache._records['image:3']['icon'],999)
        self.assertEqual(self.collection['image:3'].image_size,(4,2))
        self.assertEqual(self.collection.loads,0)
    def test_pixel_thumbnail_is_aspect_preserving_and_non_destructive(self):
        pixels=[.1,.2,.3,1]*40000;before=pixels[:]
        image=NS(size=(400,100),pixels=pixels);thumb=NS()
        self.cache._pixels_thumbnail(image,thumb)
        self.assertEqual(thumb.image_size,(256,64));self.assertEqual(pixels,before)
        self.assertEqual(len(thumb.image_pixels_float),256*64*4)
    def test_invalid_pixel_buffer_fails_explicitly(self):
        with self.assertRaises(ValueError):self.cache._pixels_thumbnail(NS(size=(10,10),pixels=[]),NS())


class MeshAndRollbackTests(unittest.TestCase):
    class DataList(list):
        def foreach_get(self,key,out):
            import numpy as np
            out[:]=np.asarray([getattr(x,key) for x in self]).ravel()
    def mesh(self):
        return NS(vertices=self.DataList([NS(co=(0,0,0)),NS(co=(1,0,0)),NS(co=(1,1,0))]),
                  loops=self.DataList([NS(vertex_index=i) for i in (0,1,2)]),
                  polygons=self.DataList([NS(loop_total=3,material_index=0,use_smooth=False)]),
                  attributes=[],materials=[])
    def test_geometry_fingerprint_detects_same_count_topology_edit(self):
        m=self.mesh();a=RB._mesh_signature(m);m.loops[1].vertex_index=2
        self.assertNotEqual(a,RB._mesh_signature(m))
    def test_geometry_fingerprint_detects_vertex_and_smooth_edits(self):
        m=self.mesh();a=RB._mesh_signature(m);m.vertices[0].co=(0,0,1)
        b=RB._mesh_signature(m);self.assertNotEqual(a,b)
        m.polygons[0].use_smooth=True;self.assertNotEqual(b,RB._mesh_signature(m))
    def test_geometry_fingerprint_detects_uv_edit(self):
        m=self.mesh();uv=NS(name='UVMap',domain='CORNER',data_type='FLOAT2',
            data=self.DataList([NS(vector=(0,0)),NS(vector=(1,0)),NS(vector=(1,1))]))
        m.attributes=[uv];a=RB._mesh_signature(m);uv.data[1].vector=(.5,0)
        self.assertNotEqual(a,RB._mesh_signature(m))
    def test_retained_mesh_is_not_deleted_when_handing_off_undo(self):
        class Mesh:
            users=0
        m0,m1=Mesh(),Mesh()
        tx=object.__new__(RB.Transaction);tx.closed=False;tx.new_meshes=[m1];tx.old_meshes=[m0]
        tx.camera_state=(None,NS(users=0));tx.new_camera_data=[]
        meshes=NS(remove=Mock());cameras=NS(remove=Mock())
        with patch.object(T.bpy,'data',NS(meshes=meshes,cameras=cameras)):
            tx.dispose(retain_meshes=[m1])
        meshes.remove.assert_called_once_with(m0)
        self.assertTrue(tx.closed)
    def test_rollback_restores_mesh_visibility_camera_metadata_and_preview(self):
        class Obj(T.Item):
            def __contains__(self,k):return k in self.tags
            def __setitem__(self,k,v):self.tags[k]=v
            def __delitem__(self,k):del self.tags[k]
        root=Obj('root');root.matrix_world='changed';root.data='newmesh';root.hide_render=True;root.hidden=True
        root.tags={'wondful_sb_width_subdivisions':16,'wondful_sb_skin_faces':100}
        original=T.Item('camera',kind='CAMERA');original.matrix_world='changed-camera';original.data=NS(lens=80)
        saved=NS(lens=40,shift_x=0,shift_y=0,sensor_width=36,sensor_height=24,sensor_fit='HORIZONTAL')
        tx=object.__new__(RB.Transaction);tx.closed=False;tx.new_objects=[]
        tx.states=[(root,'before','oldmesh',False,False)]
        tx.tags={root:{'wondful_sb_width_subdivisions':4,'wondful_sb_skin_faces':None}}
        tx.scene=NS(camera=object());tx.original_camera=original;tx.camera_state=('before-camera',saved)
        tx.props=NS(sb_preview_image='after',replica_preview_signature='after')
        tx.original_preview='before-preview';tx.original_signature='before'
        tx.context=NS(view_layer=NS(update=Mock()));tx.dispose=Mock()
        tx.rollback()
        self.assertEqual(root.matrix_world,'before');self.assertEqual(root.data,'oldmesh')
        self.assertFalse(root.hidden or root.hide_render)
        self.assertEqual(root.tags,{'wondful_sb_width_subdivisions':4})
        self.assertIs(tx.scene.camera,original);self.assertEqual(original.data.lens,40)
        self.assertEqual(tx.props.sb_preview_image,'before-preview');tx.dispose.assert_called_once()


class SecondaryPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ui=T.ui_module()
        tree=ast.parse((PKG/'comparison_view.py').read_text())
        panel=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WONDFUL_PT_compare_workspace')
        env={'__package__':T.PACKAGE,'workflow':WF,'Panel':object}
        exec(compile(ast.Module(body=[panel],type_ignores=[]),'comparison_view.py','exec'),env)
        cls.panel=env[panel.name]
    def compare_draw(self,p):
        layout=RecordingLayout()
        layout.prop_enum=lambda data,attr,value:layout.log.append(('enum',attr,value))
        # All child layouts need prop_enum too.
        with patch.object(RecordingLayout,'prop_enum',lambda self,d,a,v:self.log.append(('enum',a,v)),create=True):
            self.panel.draw(NS(layout=layout),T.context(p))
        return layout.log
    def test_target_pair_does_not_show_old_vehicle_score_or_outline(self):
        p=T.props(workflow_mode=WF.REFERENCE,compare_pair='TARGET_BLOCKOUT',alignment_score=99,
                  sb_reference_image=T.FakeImage('Target'),sb_preview_image=T.FakeImage('Preview'))
        log=self.compare_draw(p)
        self.assertFalse(any(x[0]=='label' and '99' in x[1] for x in log))
        self.assertEqual([x[2] for x in log if x[0]=='enum'],['SPLIT','OVERLAY','DIFFERENCE'])
    def test_camera_pair_preserves_actual_vehicle_score(self):
        p=T.props(alignment_score=98,viewport_image=T.FakeImage('Input'),result_image=T.FakeImage('Output'))
        log=self.compare_draw(p)
        self.assertTrue(any(x[0]=='label' and '98 / 100' in x[1] for x in log))
    def test_compare_mode_instructions_use_tooltips(self):
        for mode in ('SPLIT','OVERLAY','DIFFERENCE','OUTLINE'):
            p=T.props(compare_mode=mode,compare_drag_active=True)
            log=self.compare_draw(p)
            self.assertTrue(any(x[:2]==('operator','wondful.ui_help') for x in log))
            self.assertFalse(any(x[0]=='label' and len(x[1])>35 for x in log))
    def test_target_pair_callback_switches_unsupported_outline(self):
        tree=ast.parse((PKG/'properties.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_refresh_compare_update')
        env={'workflow':WF}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'properties.py','exec'),env)
        p=T.props(workflow_mode=WF.REFERENCE,compare_pair='TARGET_RESULT',compare_mode='OUTLINE')
        env[fn.name](p,None);self.assertEqual(p.compare_mode,'SPLIT')
    def test_preferences_draw_all_fields_through_shared_primitives(self):
        tree=ast.parse((PKG/'properties.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WONDFUL_AddonPreferences')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='draw')
        env={'__package__':T.PACKAGE}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'properties.py','exec'),env)
        p=T.defaults('WONDFUL_AddonPreferences');p.layout=RecordingLayout()
        env['draw'](p,T.context())
        names=[x[1] for x in p.layout.log if x[0]=='prop']
        for key in ('codex_cli_path','codex_model','antigravity_cli_path','output_directory','alignment_max_attempts'):
            self.assertIn(key,names)
        self.assertTrue(any(x[:2]==('operator','wondful.ui_help') for x in p.layout.log))
        self.assertFalse(any(x[0]=='label' and len(x[1])>40 for x in p.layout.log))


if __name__=='__main__':
    unittest.main(verbosity=2)
