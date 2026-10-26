"""Pure-Python geometric and schema acceptance tests for luminous studio portals.

Native Blender lighting, occlusion and model recognition are separate acceptance
requirements. These tests only prove valid geometry, schema and photo anchors.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

PKG=Path(__file__).resolve().parents[1]/'wondful_ai_renderer'
ROOT=PKG.parent
FIXTURE=ROOT/'fixtures/neon_triangle_studio_reference.json'


def module(name):
    key='_wondful_geometry_137.'+name
    import types
    base='_wondful_geometry_137'
    if base not in sys.modules:
        p=types.ModuleType(base)
        p.__path__=[str(PKG)]
        sys.modules[base]=p
    if key not in sys.modules:
        spec=importlib.util.spec_from_file_location(key,PKG/(name+'.py'))
        item=importlib.util.module_from_spec(spec)
        sys.modules[key]=item
        spec.loader.exec_module(item)
    return sys.modules[key]

SF=module('studio_frames')
SB=module('scene_blockout')
RR=module('replica_revision')
EG=module('studio_edge_guides')


def specimen():
    return json.loads(FIXTURE.read_text(encoding='utf8'))


def solver(data, frame):
    camera=data['camera'];theta=math.radians(90+camera['pitch_deg'])
    C=np.array([0.,0.,camera['height_m']])
    R=np.array([[1,0,0],[0,math.cos(theta),-math.sin(theta)],[0,math.sin(theta),math.cos(theta)]])
    tanh=math.tan(math.radians(camera['hfov_deg'])/2);hw=frame[1]/frame[0]
    def project(u,v,depth):
        ray=R@np.array([(2*u-1)*tanh,(1-2*v)*tanh*hw,-1])
        return C+ray*(depth/ray[1])
    def inverse(p):
        local=R.T@(p-C)
        return .5*(1+local[0]/(-local[2])/tanh),.5*(1-local[1]/(-local[2])/(tanh*hw))
    return project,inverse


class StudioFrameSchema(unittest.TestCase):
    def test_photo_fixture_is_schema_valid(self):
        data=specimen()
        try:
            import jsonschema
        except ImportError:
            self.skipTest('jsonschema not available')
        jsonschema.Draft202012Validator(SB.SCHEMA).validate(data)
        result=SB.normalize(data)
        self.assertEqual(len(result['light_frames']),4)
        self.assertEqual(result['light_frames'][0]['id'],'outer_v')
        self.assertEqual(result['light_frames'][0]['grounded'],True)

    def test_backward_compatibility_without_new_field(self):
        data=specimen();data.pop('light_frames')
        self.assertEqual(SB.normalize(data)['light_frames'],[])

    def test_invalid_is_skipped_without_inventing_structure(self):
        data=specimen()
        for frame in data['light_frames']:
            frame['points'][0]['x_frac']=float('nan')
        result=SB.normalize(data)
        self.assertEqual(result['light_frames'],[])
        self.assertEqual(len(result['_geometry_warnings']),4)

    def test_invalid_boolean_or_color_cannot_be_accepted(self):
        for key,value in [('grounded','yes'),('color_hex','#bad'),('width_frac',float('inf'))]:
            data=specimen()
            data['light_frames'][0][key]=value
            actual=SB.normalize(data)
            self.assertEqual(len(actual['light_frames']),3)

    def test_duplicate_ids_rejected(self):
        data=specimen()
        data['light_frames'][1]['id']=data['light_frames'][0]['id']
        self.assertEqual(len(SB.normalize(data)['light_frames']),3)

    def test_frames_have_specific_prompt_and_generation_note(self):
        c=SB.analysis_content('data:image/png;base64,abc',(1456,816))
        self.assertIn('LIGHT_FRAMES',c[0]['text'])
        self.assertIn('grounded=true',c[0]['text'])
        self.assertIn('开放轮廓',SB.scene_note(specimen()))


class StudioBeamTopology(unittest.TestCase):
    def test_all_reference_beams_are_closed_and_pixel_aligned(self):
        data=specimen();w,h=1456,816
        proj,inv=solver(data,(w,h))
        for frame in data['light_frames']:
            verts,faces,layout=SF.beam_mesh(frame['points'],(w,h),frame['width_frac'],frame['depth_m'],frame['beam_depth_m'],proj)
            self.assertEqual(len(verts),len(frame['points'])*4)
            self.assertEqual(layout['front_faces'],len(frame['points'])-1)
            self.assertEqual(len(faces),4*(len(frame['points'])-1)+2)
            edges={}
            for face in faces:
                for a,b in zip(face,face[1:]+face[:1]):
                    edge=tuple(sorted((a,b)))
                    edges[edge]=edges.get(edge,0)+1
            self.assertTrue(all(n==2 for n in edges.values()))
            for i,point in enumerate(frame['points']):
                u,v=inv((verts[i*4]+verts[i*4+1])/2)
                self.assertLess(abs(u-point['x_frac'])*w,.5)
                self.assertLess(abs(v-point['y_frac'])*h,.5)
            self.assertTrue(np.isfinite(verts).all())

    def test_broken_beam_does_not_pass(self):
        data=specimen()['light_frames'][0]
        point=lambda u,v,z:(u,v,z)
        with self.assertRaises(ValueError):
            SF.beam_mesh(data['points'],(0,816),data['width_frac'],data['depth_m'],data['beam_depth_m'],point)
        duplicated=[data['points'][0],data['points'][0],data['points'][2]]
        with self.assertRaises(ValueError):
            SF.beam_mesh(duplicated,(1456,816),data['width_frac'],data['depth_m'],data['beam_depth_m'],point)

    def test_miter_is_bounded_on_hairpin(self):
        points=[{'x_frac':.3,'y_frac':.7},{'x_frac':.5,'y_frac':.2},{'x_frac':.301,'y_frac':.70}]
        def project(u,v,depth):
            return (u*depth,depth,v*depth)
        xyz,faces,meta=SF.beam_mesh(points,(800,400),.01,12,.3,project)
        self.assertTrue(np.isfinite(xyz).all())
        self.assertEqual(len(faces),10)

    def test_shifted_wrong_frames_score_worse_than_aligned(self):
        data=specimen();rows=data['light_frames'];shape=(300,500)
        test_rows=[copy.deepcopy(rows[0]),copy.deepcopy(rows[1])]
        mask=SF.projected_stroke_mask((shape[1],shape[0]),test_rows)
        image=np.full((shape[0],shape[1],3),[15,22,32],np.uint8)
        image[mask]=[245,248,252]
        score=SF.frame_alignment_score(image,test_rows)['frame_f']
        shifted=copy.deepcopy(test_rows)
        for row in shifted:
            for p in row['points']:
                p['x_frac']+=.21
        wrong=SF.frame_alignment_score(image,shifted)['frame_f']
        self.assertGreater(score,0.95)
        self.assertGreater(score-wrong,0.45)


class StudioComputerVision(unittest.TestCase):
    def test_two_long_bars_detected_without_opencv(self):
        shape=(260,420)
        img=np.full((shape[0],shape[1],4),[12,18,24,255],np.uint8)
        def line(x0,y0,x1,y1,width):
            row={'points':[{'x_frac':x0,'y_frac':y0},{'x_frac':x1,'y_frac':y1}],
                 'width_frac':width}
            img[SF.projected_stroke_mask((shape[1],shape[0]),[row])]=[245,248,252,255]
        line(.18,.77,.44,-.08,.035)
        line(.62,-.08,.96,.77,.035)
        img[155:185,120:290]=[110,120,131,255]  # car highlight is not one of the pale beams
        hints=EG.bright_bar_hints(img)
        self.assertGreaterEqual(len(hints),2)
        text=EG.guide_text(hints)
        self.assertIn('candidates',text)
        self.assertIn('car reflections',text)

    def test_cropped_outer_v_is_paired_with_offscreen_apex(self):
        hints=[
            {'start':(.481,.015),'end':(.208,.730),'approx_width_frac':.054,'relative_length':.485},
            {'start':(.673,.016),'end':(.927,.741),'approx_width_frac':.059,'relative_length':.479},
            {'start':(.593,.256),'end':(.509,.497),'approx_width_frac':.015,'relative_length':.159},
        ]
        portal=EG.infer_open_v_portals(hints)
        self.assertEqual(len(portal),1)
        self.assertAlmostEqual(portal[0]['points'][1]['x_frac'],.581,places=2)
        self.assertAlmostEqual(portal[0]['points'][1]['y_frac'],-.247,places=2)
        self.assertIn('EXTRAPOLATED apex',EG.guide_text(hints))

    def test_dark_studio_missing_outer_portal_has_conservative_rescue(self):
        hints=[
            {'start':(.481,.015),'end':(.208,.730),'approx_width_frac':.054,'relative_length':.485},
            {'start':(.673,.016),'end':(.927,.741),'approx_width_frac':.059,'relative_length':.479}]
        base=SB.normalize(specimen());base['light_frames']=[]
        recovered,changed=EG.rescue_outer_portal(base,hints,dark_studio=True)
        self.assertTrue(changed)
        self.assertEqual(len(recovered['light_frames']),1)
        self.assertEqual(recovered['light_frames'][0]['id'],'detected_outer_v')
        self.assertIn('待复核', recovered['light_frames'][0]['label_zh'])
        self.assertEqual(base['light_frames'],[])
        for scene, dark in [('outdoor',True),('interior',False)]:
            updated=dict(base,scene_kind=scene)
            self.assertFalse(EG.rescue_outer_portal(updated,hints,dark_studio=dark)[1])
        with_frames=SB.normalize(specimen())
        self.assertFalse(EG.rescue_outer_portal(with_frames,hints,dark_studio=True)[1])
        with_ribbon=dict(base,ribbons=[{'id':'r'}])
        self.assertFalse(EG.rescue_outer_portal(with_ribbon,hints,dark_studio=True)[1])

    def test_car_highlights_are_not_paired_to_a_v(self):
        unrelated=[
            {'start':(.3,.53),'end':(.51,.61),'approx_width_frac':.02,'relative_length':.2},
            {'start':(.75,.50),'end':(.89,.55),'approx_width_frac':.02,'relative_length':.1},
        ]
        self.assertEqual(EG.infer_open_v_portals(unrelated),[])


class StudioRevision(unittest.TestCase):
    def test_existing_frame_can_replace_its_screen_corners(self):
        frame=specimen()['light_frames'][0]
        snapshot={'span':60,'objects':[{'id':'o000','name':'前景巨型白光门架','kind':'light_frame','frame':frame}]}
        msg={'summary':'拉直外部三角梁','unresolved':[],
             'camera':{'apply':False,'translation':[0,0,0],'rotation_deg':[0,0,0],'lens_mm':45,'shift_x':0,'shift_y':0},
             'edits':[],'ribbons':[],'light_frames':[{
               'id':'o000','points':frame['points'],'width_frac':frame['width_frac'],
               'depth_m':frame['depth_m'],'beam_depth_m':frame['beam_depth_m'],
               'color_hex':frame['color_hex'],'emission':frame['emission'],
               'grounded':True}], 'additions':[]}
        result=RR.validate(msg,snapshot)
        self.assertEqual(result['light_frames'][0]['id'],'o000')
        self.assertNotIn('no_change',result)
        msg['light_frames'][0]['points'][1]['y_frac']=float('nan')
        with self.assertRaises(ValueError):
            RR.validate(msg,snapshot)
        msg['light_frames'][0]['points'][1]['y_frac']=-.24
        snapshot['objects'][0]['kind']='building'
        with self.assertRaises(ValueError):
            RR.validate(msg,snapshot)


if __name__=='__main__':
    unittest.main()
