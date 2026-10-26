"""Reference-space geometry check, NOT a Blender or AI render.

Usage: python tools/neon_triangle_diagnostic.py photo.png fixture.json output-dir
The fixture was manually annotated to test the new geometric representation;
its existence does not imply that a model has automatically recognized it.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'wondful_ai_renderer'


def load(name):
    spec = importlib.util.spec_from_file_location('diagnostic_' + name, PKG / (name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def camera_projector(data, frame):
    c = data['camera']
    theta = math.radians(90 + c['pitch_deg'])
    R = np.array([[1,0,0],[0,math.cos(theta),-math.sin(theta)],
                  [0,math.sin(theta),math.cos(theta)]], dtype=float)
    campos = np.array([0.,0.,c['height_m']])
    tanh = math.tan(math.radians(c['hfov_deg'])/2)
    hw = frame[1]/frame[0]

    def project(u,v,depth):
        local=np.array([(2*u-1)*tanh,(1-2*v)*tanh*hw,-1.],dtype=float)
        ray=R@local
        if ray[1] <= 0:
            raise ValueError('ray behind camera')
        return campos+ray*(depth/ray[1])

    def unproject(p):
        local=R.T@(np.asarray(p)-campos)
        return ((1+(local[0]/(-local[2]))/tanh)/2,
                (1-(local[1]/(-local[2]))/(tanh*hw))/2)
    return project,unproject


def main(photo,fixture,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    image=Image.open(photo).convert('RGB')
    frame=image.size
    data=json.loads(Path(fixture).read_text('utf-8'))
    sf=load('studio_frames')
    png=load('png_io')
    eg=load('studio_edge_guides')
    rows,warnings=sf.normalize_frames(data['light_frames'])
    if warnings:
        raise ValueError(warnings)
    project,unproject=camera_projector(data,frame)
    bg=Image.new('RGB',frame,(11,16,23))
    draw=ImageDraw.Draw(bg)
    check=[]; max_error=0
    for row in reversed(rows):
        verts,faces,layout=sf.beam_mesh(row['points'],frame,row['width_frac'],row['depth_m'],row['beam_depth_m'],project)
        edges={}
        for f in faces:
            for a,b in zip(f,f[1:]+f[:1]):
                e=tuple(sorted((a,b)))
                edges[e]=edges.get(e,0)+1
        assert all(v==2 for v in edges.values()),'not manifold'
        assert len(verts)==len(row['points'])*4
        for face in faces[:layout['front_faces']]:
            pix=[(u*frame[0],v*frame[1]) for u,v in (unproject(verts[i]) for i in face)]
            draw.polygon(pix,fill=row['color_hex'])
        for i,p in enumerate(row['points']):
            for off in (1,-1):
                pos=verts[i*4 + (0 if off==1 else 1)]
                u,v=unproject(pos)
                if not (math.isfinite(u) and math.isfinite(v)):
                    raise ValueError('invalid projection')
            mid=(verts[4*i]+verts[4*i+1])/2
            u,v=unproject(mid)
            error=max(abs(u-p['x_frac'])*frame[0],abs(v-p['y_frac'])*frame[1])
            max_error=max(max_error,error)
        check.append({'id':row['id'],'label':row['label_zh'],'points':len(row['points']),
                      'faces':len(faces),'mesh_manifold':True,'depth_m':row['depth_m'],
                      'width_px':row['width_frac']*frame[0]})
    diag=image.copy()
    d=ImageDraw.Draw(diag)
    colors=['#00E0FF','#FFA72C','#7CFCB4','#FF5E80']
    subject=data['reference_subject']['bbox']
    box=tuple(round(subject[k]*frame[i%2]) for i,k in enumerate(['x_min','y_min','x_max','y_max']))
    for i,row in enumerate(rows):
        points=[(p['x_frac']*frame[0],p['y_frac']*frame[1]) for p in row['points']]
        d.line(points,fill=colors[i%len(colors)],width=3)
        for x,y in points:
            if 0<=x<frame[0] and 0<=y<frame[1]:
                r=6;d.ellipse((x-r,y-r,x+r,y+r),outline=colors[i%len(colors)],width=3)
    # Refused to draw diagnostic geometry over the pictured reference car:
    # in the native 3D flow, occlusion follows physical camera depth.
    d.rectangle(box,outline='#70D7F6',width=2)
    d.text((20,20),'REFERENCE SPACE: manually marked anchors (NOT native Blender render)',fill='#69E3FF')
    display=Image.new('RGB',(frame[0]*2,frame[1]+52),(17,22,29))
    display.paste(diag,(0,52));display.paste(bg,(frame[0],52))
    t=ImageDraw.Draw(display)
    t.text((12,16),'SOURCE + STRUCTURE ANCHORS',fill='white')
    t.text((frame[0]+12,16),'PROJECTED 3D BEAM FACES / NO VEHICLE',fill='white')
    overlay=out/'neon_triangle_structure_diagnostic.png';display.save(overlay)
    cv=eg.bright_bar_hints(png.read_png(photo))
    inferred_v=eg.infer_open_v_portals(cv)
    detected=image.copy()
    adraw=ImageDraw.Draw(detected)
    for limb in cv:
        start=tuple(round(limb['start'][i]*frame[i]) for i in range(2))
        end=tuple(round(limb['end'][i]*frame[i]) for i in range(2))
        adraw.line((start,end),fill='#FFB324',width=4)
    for portal in inferred_v:
        poly=[(round(pt['x_frac']*frame[0]),round(pt['y_frac']*frame[1])) for pt in portal['points']]
        adraw.line(poly,fill='#00F7CF',width=5)
        for pt in (poly[0],poly[-1]):
            x,y=pt
            adraw.ellipse((x-6,y-6,x+6,y+6),outline='#00F7CF',width=4)
        apex=portal['points'][1]
        adraw.text((20,22),f"AUTOMATIC OUTER V CANDIDATE | apex ({apex['x_frac']:.3f}, {apex['y_frac']:.3f}) OUTSIDE PHOTO", fill='#00F7CF')
    adraw.text((20,45),'Detected edge lines are hypotheses; inner frames still need AI / manual review',fill='#E8E8E8')
    detected.save(out/'auto_detected_outer_v_overlay.png')
    scored=sf.frame_alignment_score(np.asarray(image),rows,exclude_bbox=subject)
    result={'source':str(photo),'frame':[frame[0],frame[1]], 'fixture':'manually annotated target, not model output',
            'beams':check,'max_projection_anchor_error_px':max_error,'light_edge_candidates':cv, 'inferred_outer_v':inferred_v,
            'frame_alignment':scored,'warning':'Not a Blender render or online AI inference'}
    (out/'geometry_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'image':str(overlay),'max_proj_error_px':max_error,
                      'frame_alignment':scored,'candidate_count':len(cv)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main(*sys.argv[1:4])
