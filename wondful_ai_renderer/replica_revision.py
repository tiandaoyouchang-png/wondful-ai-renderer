"""Bounded declarative scene corrections. Model replies are data, never code."""
import json
import math

NUM = {'type': 'number'}
VEC = {'type': 'array', 'items': NUM, 'minItems': 3, 'maxItems': 3}
POINT = {'type': 'object', 'additionalProperties': False,
         'required': ['x','y','z'], 'properties': {k: NUM for k in ('x','y','z')}}
SECTION = {'type':'object', 'additionalProperties':False, 'required':['left','right'],
           'properties':{'left':POINT,'right':POINT}}


def obj(properties):
    return {'type':'object','additionalProperties':False, 'required':list(properties), 'properties':properties}


SCHEMA = obj({
    'summary': {'type':'string'},
    'unresolved': {'type':'array','items':{'type':'string'},'maxItems':8},
    'camera': obj({'apply':{'type':'boolean'}, 'translation':VEC,'rotation_deg':VEC,
                   'lens_mm':NUM,'shift_x':NUM,'shift_y':NUM}),
    'edits': {'type':'array','maxItems':24,'items':obj({
        'id':{'type':'string'},'translation':VEC,'rotation_deg':VEC,'scale':VEC,'hidden':{'type':'boolean'}})},
    'ribbons': {'type':'array','maxItems':8,'items':obj({
        'id':{'type':'string'},'sections':{'type':'array','minItems':2,'maxItems':32,'items':SECTION}, 'thickness_m':NUM})},
     'light_frames': {'type':'array','maxItems':8,'items':obj({
        'id':{'type':'string'},
        'points':{'type':'array','minItems':2,'maxItems':16,'items':obj({'x_frac':NUM,'y_frac':NUM})},
        'width_frac':NUM,'depth_m':NUM,'beam_depth_m':NUM,'color_hex':{'type':'string'},
        'emission':NUM,'grounded':{'type':'boolean'}})},
    'additions': {'type':'array','maxItems':8,'items':obj({
        'label':{'type':'string'},'location':VEC,'dimensions':VEC,'rotation_deg':VEC})},
})


def number(v, lo, hi):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise ValueError('Invalid or out-of-range numeric scene correction')
    return float(v)


def vector(v, lo, hi):
    if not isinstance(v, list) or len(v) != 3:
        raise ValueError('Scene correction needs a three-component vector')
    return [number(n, lo, hi) for n in v]


def keys(value, required):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError('Unexpected scene correction fields')


def validate(reply, snapshot, ribbon_builder=None):
    """Validate the entire plan before any mutation. Unknown IDs fail closed."""
    keys(reply, SCHEMA['required'])
    if not isinstance(reply['summary'],str) or len(reply['summary']) > 1200:
        raise ValueError('Invalid correction summary')
    unresolved=reply['unresolved']
    if not isinstance(unresolved,list) or len(unresolved)>8 or any(not isinstance(v,str) or len(v)>1000 for v in unresolved):
        raise ValueError('Invalid unresolved-request list')
    out={'summary':reply['summary'],'unresolved':unresolved[:], 'edits':[], 'ribbons':[], 'light_frames':[], 'additions':[]}
    camera=reply['camera'];keys(camera,SCHEMA['properties']['camera']['required'])
    if type(camera['apply']) is not bool:
        raise ValueError('Invalid camera apply flag')
    # A generous hard cap still prevents a bad response sending objects to infinity.
    limit=min(1000.0,max(20.0,float(snapshot.get('span',20.0))*2))
    out['camera']={'apply':camera['apply'],'translation':vector(camera['translation'],-limit,limit),
                   'rotation_deg':vector(camera['rotation_deg'],-180,180),
                   'lens_mm':number(camera['lens_mm'],10,300) if camera['apply'] else number(camera['lens_mm'],0,300),
                   'shift_x':number(camera['shift_x'],-1,1),'shift_y':number(camera['shift_y'],-1,1)}
    known={v['id']:v for v in snapshot['objects']}
    for kind,maximum in [('edits',24),('ribbons',8),('additions',8)]:
        rows=reply[kind]
        if not isinstance(rows,list) or len(rows)>maximum:
            raise ValueError('Too many scene corrections')
        used=set()
        for row in rows:
            keys(row,SCHEMA['properties'][kind]['items']['required'])
            if kind!='additions':
                ident=row['id']
                if not isinstance(ident,str) or ident not in known or ident in used:
                    raise ValueError('Unknown or duplicate correction target')
                used.add(ident)
            if kind=='edits':
                if type(row['hidden']) is not bool:
                    raise ValueError('Invalid visibility flag')
                out[kind].append(dict(id=ident,translation=vector(row['translation'],-limit,limit),
                    rotation_deg=vector(row['rotation_deg'],-180,180),scale=vector(row['scale'],0.1,10),hidden=row['hidden']))
            elif kind=='ribbons':
                if known[ident].get('kind')!='ribbon':
                    raise ValueError('Only generated ribbons accept a surface replacement')
                thickness=number(row['thickness_m'],0.01,10)
                if not isinstance(row['sections'],list) or not 2<=len(row['sections'])<=32:
                    raise ValueError('Invalid ribbon sections')
                sections=[]
                for section in row['sections']:
                    keys(section,('left','right'))
                    pair={}
                    for edge in ('left','right'):
                        keys(section[edge],('x','y','z'))
                        pair[edge]={axis:number(section[edge][axis],-20000,20000) for axis in ('x','y','z')}
                    sections.append(pair)
                if ribbon_builder:
                    ribbon_builder(sections,thickness)  # catches bow-ties / degenerate geometry before writes
                out[kind].append(dict(id=ident,sections=sections,thickness_m=thickness))
            else:
                label=row['label']
                if not isinstance(label,str) or not label.strip() or len(label)>64:
                    raise ValueError('Invalid new-block label')
                out[kind].append(dict(label=label.strip(),location=vector(row['location'],-20000,20000),
                    dimensions=vector(row['dimensions'],0.02,1000),rotation_deg=vector(row['rotation_deg'],-180,180)))
    from .studio_frames import normalize_frames
    used_frames = set()
    frames = reply['light_frames']
    if not isinstance(frames, list) or len(frames) > 8:
        raise ValueError('Too many luminous-frame corrections')
    for frame in frames:
        keys(frame, SCHEMA['properties']['light_frames']['items']['required'])
        ident = frame['id']
        if not isinstance(ident, str) or ident not in known or ident in used_frames:
            raise ValueError('Unknown or duplicate luminous-frame ID')
        if known[ident].get('kind') != 'light_frame':
            raise ValueError('Only generated light frames can change screen anchors')
        used_frames.add(ident)
        old = known[ident].get('frame') or {}
        full = dict(frame, id=ident, label_zh=old.get('label_zh') or known[ident].get('name') or '发光框架',
                    material=old.get('material', ''))
        parsed, warnings = normalize_frames([full])
        if warnings or len(parsed) != 1:
            raise ValueError('Invalid luminous-frame correction: '+('; '.join(warnings)))
        out['light_frames'].append(parsed[0])
    if not out['camera']['apply'] and not any(out[k] for k in ('edits','ribbons','additions','light_frames')):
        out['no_change']=True
    return out


def content(target_url, preview_url, identity_url, snapshot, feedback):
    note=str(feedback or '').strip()
    if not note or len(note)>4000:
        raise ValueError('\u8bf7\u586b\u5199\u4fee\u6539\u610f\u89c1\uff08\u6700\u591a 4000 \u5b57\uff09')
    rules='''Revise a live Blender reference-replica blockout, not the product model. Return strict JSON only.
Use the user's feedback and current preview, NOT a fresh reconstruction. Only existing IDs in objects are editable.
The image-labelled product identity is mandatory context, but product mesh/materials/transforms are never editable.
Existing edits: translation is WORLD metres, rotation_deg is WORLD XYZ rotation DELTA around the group's bounding-box
centre, scale is a multiplicative XYZ delta around that centre. [0,0,0], [0,0,0], [1,1,1] preserves a transform.
hidden is the final visibility of that group; do not hide a structure unless asked. Group members move together.
Camera translation is WORLD metres and rotation_deg is WORLD XYZ orientation DELTA. lens_mm and shifts are absolute.
Keep the current camera when no camera change is requested; set apply=false and retain current lens and shifts.
Ribbons: only existing kind=ribbon IDs; paired sections are absolute OBJECT-LOCAL coordinates, not screen or world.
Their object transform remains unchanged unless also listed in edits. Preserve side correspondence and thickness.
Light frames: only IDs with kind=light_frame. Use the original normalized PHOTO space coordinates, 0 left/top to
1 right/bottom, and possibly beyond the frame for cropped V apexes. Return the ENTIRE ordered screen centreline of
that frame when a user asks to change its apex, length, angle or beam width. Keep the beam open: never fill the triangle.
Each replacement includes width_frac, depth_m (camera-forward metres), beam_depth_m, RGB hex color, emission
and grounded. Do not change the product or other frames; unchanged light_frames=[].
Additions are coarse boxes only, with WORLD location, full dimensions, and WORLD absolute XYZ rotation. Do not use
boxes to fake twisted surfaces. Never alter unrequested objects. If a request cannot be expressed with these bounded
operations (for example photoreal car-paint relighting), explain it in unresolved, not an unrelated substitute.
summary and unresolved use Chinese. No code, paths, commands, file operations, external references or tool calls.
'''
    return [
        {'type':'input_text','text':rules+'\nUser feedback:\n'+note+'\nCurrent scene snapshot:\n'+json.dumps(snapshot,ensure_ascii=False)},
        {'type':'input_text','text':'Target reference (composition and environment, never product identity):'},
        {'type':'input_image','image_url':target_url},
        {'type':'input_text','text':'Current native Blender preview (the scene to correct):'},
        {'type':'input_image','image_url':preview_url},
        {'type':'input_text','text':'Required product identity (keep this model unchanged):'},
        {'type':'input_image','image_url':identity_url},
    ]
