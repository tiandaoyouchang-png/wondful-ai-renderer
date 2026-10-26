"""Blender-only *optional* verification for open luminous studio architecture.

Run outside this distribution environment, in disposable headless Blender:
  blender --background --factory-startup --python-exit-code 1 \
    --python tools/native_studio_frames_smoke.py -- SOURCE_ROOT OUTPUT_DIR

Uses synthetic scene geometry and the hand-annotated fixture; never changes an
existing .blend or calls AI. Verifies real bpy meshes, materials, camera, and PNG.
"""
import json
from pathlib import Path
import sys
import traceback
import bpy


def run():
    if not bpy.app.background:
        raise RuntimeError("Only run in a fresh headless Blender process")
    args = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    if len(args) != 2:
        raise RuntimeError("Expected SOURCE_ROOT and OUTPUT_DIR")
    src, dst = (Path(a).resolve() for a in args)
    dst.mkdir(parents=True,exist_ok=True)
    data = json.loads((src/"fixtures/neon_triangle_studio_reference.json").read_text(encoding="utf-8"))
    sys.path.insert(0,str(src))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    from wondful_ai_renderer import scene_blockout_build as BB
    from wondful_ai_renderer import scene_blockout as SB
    record={"engine":bpy.app.version_string,"reference":"manual fixture", "status":"FAILED", "checks":[]}
    def check(label,condition):
        record['checks'].append({'name':label,'pass':bool(condition)})
        if not condition:
            raise AssertionError(label)
    try:
        info=BB.build(bpy.context,data,frame=(1456,816),product_collection=None,set_camera=True)
        coll=bpy.data.collections.get(info['collection'])
        check("blockout collection",coll is not None)
        built=[ob for ob in coll.all_objects if ob.get("wondful_sb_light_frame")]
        check("four separate LED meshes",len(built)==4)
        check("every mesh has geometry and two materials",all(ob.type=='MESH' and len(ob.data.vertices)>4 and len(ob.data.materials)==2 for ob in built))
        check("every front surface emits", all(
            any(node.type=='EMISSION' for node in ob.data.materials[1].node_tree.nodes) for ob in built))
        check("every frame records editable anchors", all(ob.get('wondful_sb_frame_spec') for ob in built))
        check("scene has a camera",bpy.context.scene.camera is not None)
        terrain = next(ob for ob in coll.all_objects if ob.get("wondful_env_type") == "terrain")
        surface = terrain.data.materials[0].node_tree.nodes.get("Principled BSDF")
        check("studio floor is glossy and dark",surface is not None and
              surface.inputs["Roughness"].default_value <= .15 and
              terrain.data.materials[0].diffuse_color[0] < .1)
        path=dst/'native-neon-triangle-preview.png'
        rendered=BB.render_preview(bpy.context,info,str(path),width=720)
        check("Cycles preview saved",rendered and path.is_file() and path.stat().st_size>1000)
        record['status']='PASSED'
        record['output']=str(path)
        record['frame_labels']=[ob.name for ob in built]
    except Exception:
        record['error']=traceback.format_exc()
        raise
    finally:
        (dst/'native-studio-frames.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    run()
