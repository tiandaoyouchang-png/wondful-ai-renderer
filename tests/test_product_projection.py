"""Offline checks of real evaluated-product fingerprinting and capture boundaries."""
import sys
import json
import tempfile
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
from test_316_regressions import load_package_module

PP = load_package_module('product_projection')
MV = load_package_module('multiview')
PNG = load_package_module('png_io')


class Buffer:
    def __init__(self, values):
        self.values = np.asarray(values)
    def __len__(self):
        return len(self.values)
    def foreach_get(self, name, target):
        target[:] = self.values.ravel()


class Product:
    def __init__(self, name='product'):
        self.name, self.type = name, 'MESH'
        self.matrix_world = np.eye(4)
        self.hide_render = self.hide_viewport = False
        self.visible_camera = True
        self.coords = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]])
        self.modifiers = []
        self.to_mesh_clear = Mock()
        self.fail = False
    def evaluated_get(self, graph):
        return self
    def to_mesh(self):
        if self.fail:
            raise ValueError('evaluation failed')
        scale = self.modifiers[0].levels if self.modifiers and self.modifiers[0].show_viewport else 1
        return NS(vertices=Buffer(self.coords * scale), loops=Buffer([0,1,2]), polygons=[NS(loop_total=3,material_index=0)])


class SceneObjects(list):
    def __contains__(self, obj):
        return any(o.name == obj for o in self) if isinstance(obj,str) else super().__contains__(obj)


class ProductProjectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.product = Product()
        self.collection = NS(all_objects=[self.product],children_recursive=[],hide_render=False,hide_viewport=False)
        camera = Product('camera');camera.type='CAMERA'
        camera.data = NS(type='PERSP',lens=45.,sensor_width=36.,sensor_height=24.,sensor_fit='HORIZONTAL',
                         shift_x=0.,shift_y=0.,ortho_scale=6.)
        self.props = NS(product_collection=self.collection)
        self.layer = NS(collection=self.collection,exclude=False,hide_viewport=False,children=[])
        self.scene = NS(camera=camera,objects=SceneObjects([camera,self.product]),frame_current=1,frame_subframe=0.,
                        render=NS(resolution_x=48,resolution_y=24,resolution_percentage=100,pixel_aspect_x=1.,pixel_aspect_y=1.))
        self.context = NS(scene=self.scene,view_layer=NS(update=Mock(),layer_collection=self.layer),
                          evaluated_depsgraph_get=lambda:object())
        self.viewport = types.ModuleType(PP.__package__+'.viewport_capture')
        self.viewport.camera_output_dimensions = lambda context:(context.scene.render.resolution_x,context.scene.render.resolution_y)
        self.viewport.render_detail_pass = Mock(side_effect=self.render)
        self.viewport.render_clay_base = Mock(side_effect=AssertionError('clay forbidden'))
        self.viewport.capture_camera_reference = Mock(side_effect=AssertionError('full scene forbidden'))
        self.viewport.material_color_kind=lambda mat:'default'
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules,{self.viewport.__name__:self.viewport}).start()
    def render(self, context, path, collection, keep_path='', matid_path=''):
        self.assertIs(collection,self.collection)
        pixels=np.zeros((24,48,4),np.uint8);pixels[5:20,10:38]=(20,60,190,255)
        PNG.write_png(path,pixels)
        if keep_path:PNG.write_png(keep_path,np.zeros_like(pixels))
        if matid_path:
            from json import dumps
            matid=np.zeros_like(pixels);matid[pixels[:,:,3]>0]=(51,0,0,255)
            PNG.write_png(matid_path,matid)
            Path(matid_path+'.json').write_text(dumps({'materials':[{'index':0,'name':'surface','kind':'default'}]}))
        return {'detail':True,'keep':True,'matid':True,'keep_slots':0}

    def test_capture_uses_only_clean_detail_pass_and_exact_rgba_record(self):
        before=PP.signature(self.context,self.props)
        record=PP.capture(self.context,self.props,self.root,(48,24))
        self.viewport.render_detail_pass.assert_called_once_with(self.context,str(self.root/'product_projection.png'),self.collection,
            keep_path=str(self.root/'product_keep.png'),matid_path=str(self.root/'product_matid.png'))
        self.viewport.render_clay_base.assert_not_called();self.viewport.capture_camera_reference.assert_not_called()
        self.assertEqual(record['signature'],before)
        self.assertEqual(record['method'],'PRODUCT_ONLY_RGBA')
        self.assertEqual(record['dimensions'],[48,24])
        self.assertTrue(PP.current(self.context,self.props,record))
        Path(record['path']).write_bytes(b'changed')
        self.assertFalse(PP.current(self.context,self.props,record))

    def test_aspect_mismatch_fails_before_capture_without_changing_resolution(self):
        with self.assertRaisesRegex(ValueError,'画幅比例不一致'):
            PP.capture(self.context,self.props,self.root,(48,48))
        self.viewport.render_detail_pass.assert_not_called()
        self.assertEqual((self.scene.render.resolution_x,self.scene.render.resolution_y),(48,24))
        PP.require_matching_aspect((1456,816),(1456,817))

    def test_empty_alpha_and_capture_failure_stop(self):
        for failure in ('empty','failed','exception'):
            def render(*args,**kwargs):
                if failure=='exception':raise RuntimeError('render failed')
                PNG.write_png(args[1],np.zeros((24,48,4),np.uint8))
                return {'detail':failure!='failed'}
            self.viewport.render_detail_pass.side_effect=render
            with self.subTest(failure=failure),self.assertRaises((ValueError,RuntimeError)):
                PP.capture(self.context,self.props,self.root,(48,24))
        self.viewport.capture_camera_reference.assert_not_called()
        self.viewport.render_clay_base.assert_not_called()

    def test_selected_geometry_deformation_camera_frame_membership_and_layer_are_bound(self):
        original=PP.signature(self.context,self.props)
        mutations=[lambda:setattr(self.scene.camera.data,'lens',55.),
                   lambda:self.product.coords.__setitem__((0,0),0.25),
                   lambda:setattr(self.scene,'frame_subframe',0.5),
                   lambda:self.collection.all_objects.append(Product('new mesh')),
                   lambda:setattr(self.layer,'exclude',True)]
        for mutate in mutations:
            mutate();current=PP.signature(self.context,self.props)
            self.assertNotEqual(current,original)
            original=current
        self.props.product_collection=NS(**vars(self.collection))
        self.assertNotEqual(PP.signature(self.context,self.props),original)

    def modifier(self):
        modifier=NS(type='SUBSURF',show_viewport=False,show_render=True,levels=1,render_levels=2)
        modifier.bl_rna=NS(properties=[NS(identifier=k,type='BOOLEAN' if isinstance(v,bool) else 'INT' if isinstance(v,int) else 'STRING',is_array=False)
                                     for k,v in vars(modifier).items()])
        return modifier

    def test_render_modifier_evaluation_is_fresh_and_restored_even_on_failure(self):
        modifier=self.modifier();self.product.modifiers=[modifier]
        first=PP.signature(self.context,self.props)
        self.assertFalse(modifier.show_viewport);self.assertEqual(modifier.levels,1)
        modifier.render_levels=3
        self.assertNotEqual(PP.signature(self.context,self.props),first)
        self.assertFalse(modifier.show_viewport);self.assertEqual(modifier.levels,1)
        self.product.fail=True
        with self.assertRaisesRegex(ValueError,'evaluation failed'):
            PP.signature(self.context,self.props)
        self.assertFalse(modifier.show_viewport);self.assertEqual(modifier.levels,1)

    def test_complex_modifier_rna_arrays_are_not_serialized(self):
        modifier=self.modifier()
        modifier.matrix_inverse=np.eye(4)
        class RNA:
            @property
            def properties(self):
                raise AssertionError('Do not enumerate arbitrary modifier RNA')
        modifier.bl_rna=RNA()
        self.product.modifiers=[modifier]
        self.assertEqual(len(PP.signature(self.context,self.props)),64)

    def test_helper_png_and_json_bytes_freeze_and_staleness_is_fatal(self):
        record=PP.capture(self.context,self.props,self.root,(48,24))
        before={key:Path(row['path']).read_bytes() for key,row in record['helpers'].items()}
        destination=self.root/'generation';destination.mkdir()
        frozen=PP.freeze_helpers(record,destination)
        for key,data in before.items():
            self.assertEqual(Path(frozen[key]).read_bytes(),data)
            self.assertEqual(Path(record['helpers'][key]['path']).read_bytes(),data)
        Path(record['helpers']['matid_json']['path']).write_text('corrupt')
        self.assertFalse(PP.current(self.context,self.props,record))
        with self.assertRaisesRegex(ValueError,'变化'):PP.freeze_helpers(record,destination)

    def test_body_paint_keep_flags_material_graph_and_texture_pixels_are_bound(self):
        class Material(dict):pass
        class Pixels:
            def __init__(self):self.values=np.zeros(16,np.float32)
            def __len__(self):return len(self.values)
            def foreach_get(self,target):target[:]=self.values
        pixels=Pixels()
        image=NS(name='identifier image',size=(2,2),filepath='local.png',colorspace_settings=NS(name='sRGB'),pixels=pixels)
        socket=NS(identifier='colour',default_value=[.2,.3,.4,1.])
        node=NS(name='texture',bl_idname='ShaderNodeTexImage',inputs=[socket],outputs=[],image=image)
        mat=Material();mat.name='body paint';mat.diffuse_color=(.5,.5,.5,1.);mat.use_nodes=True;mat.node_tree=NS(nodes=[node],links=[])
        self.product.material_slots=[NS(material=mat,link='DATA')]
        self.product.get=lambda key,default=None:default
        first=PP.signature(self.context,self.props)
        socket.default_value[0]=.7
        second=PP.signature(self.context,self.props);self.assertNotEqual(first,second)
        pixels.values[0]=1.
        third=PP.signature(self.context,self.props);self.assertNotEqual(second,third)
        mat.diffuse_color=(.0,.5,.0,1.)
        fourth=PP.signature(self.context,self.props);self.assertNotEqual(third,fourth)
        mat['wondful_keep_exact']=False
        self.assertNotEqual(fourth,PP.signature(self.context,self.props))

    def test_render_uv_and_material_assignment_change_freshness(self):
        uv=Buffer([[0.,0.],[1.,0.],[0.,1.]])
        polygon=NS(loop_total=3,material_index=0)
        self.product.to_mesh=lambda:NS(vertices=Buffer(self.product.coords),loops=Buffer([0,1,2]),polygons=[polygon],
                                       uv_layers=NS(active=NS(data=uv)))
        first=PP.signature(self.context,self.props)
        uv.values[0,0]=.5
        second=PP.signature(self.context,self.props);self.assertNotEqual(first,second)
        polygon.material_index=1
        self.assertNotEqual(second,PP.signature(self.context,self.props))

    def test_unrelated_scene_mesh_is_never_evaluated(self):
        unrelated=Product('unrelated clocktower');unrelated.evaluated_get=Mock(side_effect=AssertionError('unrelated evaluation'))
        self.scene.objects.append(unrelated)
        with patch.object(MV,'scene_geometry_key',wraps=MV.scene_geometry_key) as geometry:
            PP.signature(self.context,self.props)
        self.assertEqual(geometry.call_args.kwargs['objects'],[self.product])
        self.assertTrue(geometry.call_args.kwargs['render_modifiers'])
        unrelated.evaluated_get.assert_not_called()

    def test_missing_camera_or_empty_collection_and_worker_thread_refuse_snapshot(self):
        errors=[]
        def worker():
            try:PP.signature(self.context,self.props)
            except Exception as exc:errors.append(exc)
        thread=threading.Thread(target=worker);thread.start();thread.join()
        self.assertEqual(len(errors),1);self.assertIn('主线程',str(errors[0]))
        self.scene.camera=None
        with self.assertRaisesRegex(ValueError,'活动相机'):PP.signature(self.context,self.props)
        self.collection.all_objects=[]
        with self.assertRaisesRegex(ValueError,'产品几何'):PP.product_objects(self.context,self.props)


if __name__=='__main__':
    unittest.main()
