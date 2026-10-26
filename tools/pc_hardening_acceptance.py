"""Blender CPU checks for PC hardening; no external image generation.
blender -b --factory-startup --python-exit-code 1 --python tools/pc_hardening_acceptance.py -- OUT
"""
import json
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wondful_ai_renderer import ground_bands as GB, multiview as MV, viewport_capture as VC

out = Path(sys.argv[sys.argv.index('--') + 1])
out.mkdir(parents=True, exist_ok=True)
checks = {}
scene = bpy.context.scene
for obj in list(scene.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
collection = bpy.data.collections.new('Environment')
scene.collection.children.link(collection)
bpy.ops.mesh.primitive_plane_add(size=20)
ground = bpy.context.object
for col in list(ground.users_collection):
    col.objects.unlink(ground)
collection.objects.link(ground)
original_mat = bpy.data.materials.new('UserGround')
ground.data.materials.append(original_mat)
bpy.ops.object.camera_add(location=(5, -7, 3))
cam = bpy.context.object
cam.rotation_euler = (Vector((0, 0, 0)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
scene.camera = cam
scene.render.resolution_x, scene.render.resolution_y = 160, 120
scene.render.resolution_percentage = 100
bpy.context.view_layer.update()

key = MV.scene_geometry_key(bpy.context)
checks['signature_stable'] = key == MV.scene_geometry_key(bpy.context)
ground.data.vertices[0].co.x += 0.5
ground.data.update()
bpy.context.view_layer.update()
changed = MV.scene_geometry_key(bpy.context)
checks['signature_mesh_change'] = key != changed
cam.data.lens += 1
checks['signature_camera_change'] = changed != MV.scene_geometry_key(bpy.context)
changed = MV.scene_geometry_key(bpy.context)
mod = ground.modifiers.new('Thickness', 'SOLIDIFY')
mod.thickness = 0.2
bpy.context.view_layer.update()
checks['signature_modifier_change'] = changed != MV.scene_geometry_key(bpy.context)
ground.modifiers.remove(mod)

info = dict(t=[1, 0], n=[0, 1], a0=[0, 0], plane=[0, 0, 0],
            bands=[dict(kind='strip', s0=-1, s1=0, height=0, zh='平面条带'),
                   dict(kind='curb', s0=0, s1=0.3, height=0.12, height_measured=True, zh='路沿')])
made = GB.build_geometry(bpy.context, collection, info)
objects = [ob for ob in made if isinstance(ob, bpy.types.Object)]
checks['only_raised_geometry'] = len(objects) == 1 and objects[0][GB.BAND_TAG] == 1
checks['curb_material_ownership'] = (objects[0].data.materials[0] != original_mat
                                     and list(ground.data.materials) == [original_mat])
mat_name = objects[0].data.materials[0].name
mesh_name = objects[0].data.name
MV.remove_aux(made)
checks['aux_cleanup'] = (not any(ob.get('wondful_mv_aux') for ob in scene.objects)
                         and mat_name not in bpy.data.materials and mesh_name not in bpy.data.meshes)

for name in ('WONDFUL_CLAY_ENGINE', 'WONDFUL_MASK_ENGINE', 'WONDFUL_DETAIL_ENGINE', 'WONDFUL_STRUCTURE_ENGINE'):
    os.environ[name] = 'CYCLES'
os.environ['WONDFUL_CLAY_SAMPLES'] = '2'
before = (scene.render.engine, scene.render.resolution_x, scene.render.resolution_y, scene.camera.name,
          tuple(ground.data.materials), ground.hide_render)
clay = out / 'pure_clay.png'
checks['pure_scene_clay_geometry'] = bool(VC.render_clay_base(
    bpy.context, str(clay), None, ground='OFF', massing=collection,
    geometry_prefix=str(out / 'pure'), geometry_size=(160, 120))) and bool(VC.LAST_CLAY_INFO.get('geometry'))
from wondful_ai_renderer.png_io import write_png
write_png(str(out / 'pure_mask.png'), np.zeros((120, 160, 4), np.uint8))
pos = np.load(out / 'pure_pos.npy')
checks['pure_scene_has_environment_hits'] = bool((pos[..., 3] > 0.5).any())
checks['scene_state_restored'] = before == (
    scene.render.engine, scene.render.resolution_x, scene.render.resolution_y, scene.camera.name,
    tuple(ground.data.materials), ground.hide_render)
report = dict(blender=bpy.app.version_string, checks=checks, ok=all(checks.values()))
(out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
if not report['ok']:
    raise RuntimeError('PC hardening acceptance failed')
