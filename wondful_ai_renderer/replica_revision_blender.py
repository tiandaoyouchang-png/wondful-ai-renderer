"""Native execution of bounded revision plans with rollback and a one-step undo.

Only owned blockouts in the submitting scene can be edited. Never exec model output.
This module imports bpy only when a native operation is invoked.
"""
import hashlib
import json
import math
from . import workflow
from . import replica_revision as R

_UNDO = {}


def _key(scene):
    return scene.as_pointer()


def owners(scene, props):
    product=set(getattr(getattr(props,'product_collection',None),'all_objects',()))
    found=[c for c in scene.collection.children_recursive if c.get('wondful_scene_blockout')]
    if len(found)!=1:
        raise RuntimeError('\u8bf7\u4f7f\u7528\u5355\u4e00\u6765\u6e90\u7684\u590d\u523b\u7c97\u6a21')
    if any(o in product for o in found[0].all_objects):
        raise RuntimeError('\u4ea7\u54c1\u4e0e\u7c97\u6a21\u96c6\u5408\u91cd\u53e0\uff0c\u4e0d\u6267\u884c\u4fee\u6539')
    return found[0]


def _bounds(obj):
    from mathutils import Vector
    return [list(obj.matrix_world @ Vector(p)) for p in obj.bound_box] if obj.type=='MESH' else []


def _ribbon_sections(obj):
    cols=int(obj.get('wondful_sb_width_subdivisions',1))+1
    verts=obj.data.vertices
    rows=len(verts)//(cols*2)
    if rows<2 or rows*cols*2!=len(verts):
        return []
    steps=sorted(set(round(i*(rows-1)/min(rows-1,31)) for i in range(min(rows,32))))
    out=[]
    for i in steps:
        base=i*cols*2
        left=(verts[base].co+verts[base+cols].co)/2
        right=(verts[base+cols-1].co+verts[base+cols*2-1].co)/2
        out.append({'left':dict(zip(('x','y','z'),left)), 'right':dict(zip(('x','y','z'),right))})
    return out


def _mesh_signature(mesh):
    """Include topology, shading, UV/color attributes: a later manual edit must survive."""
    import numpy as np
    h = hashlib.sha256()
    def feed(collection, key, width=1, dtype=np.float32):
        values = np.empty(len(collection) * width, dtype=dtype)
        collection.foreach_get(key, values)
        h.update(values.tobytes())
    feed(mesh.vertices, 'co', 3)
    feed(mesh.loops, 'vertex_index', dtype=np.int32)
    feed(mesh.polygons, 'loop_total', dtype=np.int32)
    feed(mesh.polygons, 'material_index', dtype=np.int32)
    feed(mesh.polygons, 'use_smooth', dtype=np.bool_)
    attribute_types = {
        'FLOAT': ('value', 1, np.float32), 'INT': ('value', 1, np.int32),
        'BOOLEAN': ('value', 1, np.bool_), 'FLOAT_VECTOR': ('vector', 3, np.float32),
        'FLOAT2': ('vector', 2, np.float32), 'FLOAT_COLOR': ('color', 4, np.float32),
        'BYTE_COLOR': ('color', 4, np.float32), 'INT8': ('value', 1, np.int32),
        'INT32_2D': ('value', 2, np.int32), 'QUATERNION': ('value', 4, np.float32),
        'FLOAT4X4': ('value', 16, np.float32),
    }
    for attribute in sorted(mesh.attributes, key=lambda a: a.name):
        h.update((attribute.name + ':' + attribute.domain + ':' + attribute.data_type).encode())
        spec = attribute_types.get(attribute.data_type)
        if spec is None:
            raise RuntimeError('Unsupported custom mesh attribute: ' + attribute.data_type)
        feed(attribute.data, *spec)
    h.update(str([m.as_pointer() if m else 0 for m in mesh.materials]).encode())
    return h.hexdigest()


def snapshot(context, props):
    """Capture exact state signature plus a bounded, interpretable model summary."""
    import bpy
    import numpy as np
    scene=context.scene; owner=owners(scene,props)
    # Shared blocks would mutate another scene. Require a single-user copy first.
    editable={o for o in owner.all_objects if o.get('wondful_sb_group') or o.get('wondful_sb_revision_created')}
    if any(child not in editable for o in editable for child in o.children_recursive):
        raise RuntimeError('\u7c97\u6a21\u5b50\u7ea7\u5305\u542b\u624b\u52a8\u5bf9\u8c61\uff0c\u4e3a\u907f\u514d\u8bef\u6539\u8bf7\u5148\u5206\u79bb')
    for other in bpy.data.scenes:
        if other is not scene and any(o in set(other.objects) for o in editable):
            raise RuntimeError('\u7c97\u6a21\u88ab\u5176\u4ed6\u573a\u666f\u5171\u7528\uff0c\u8bf7\u5148\u5236\u4f5c\u72ec\u7acb\u526f\u672c')
    roots=[o for o in owner.all_objects if o in editable and o.type=='MESH' and o.parent not in editable]
    if not roots or len(roots)>256:
        raise RuntimeError('\u53ef\u4fee\u6539\u7c97\u6a21\u6570\u91cf\u5fc5\u987b\u4e3a 1\u2013256')
    rows=[]; mapping={}; points=[]; exact=[]
    for i,root in enumerate(sorted(roots,key=lambda o:o.name)):
        ident='o%03d'%i;members=[root]+[o for o in root.children_recursive if o in editable]
        if any(o.type != 'MESH' for o in members):
            raise RuntimeError('\u7c97\u6a21\u7ec4\u5185\u5305\u542b\u975e\u51e0\u4f55\u5bf9\u8c61')
        if any(len(o.modifiers) or len(o.constraints) or o.animation_data or o.data.shape_keys for o in members):
            raise RuntimeError('\u7c97\u6a21\u542b\u4fee\u6539\u5668\u3001\u7ea6\u675f\u6216\u52a8\u753b\uff0c\u4e3a\u4fdd\u62a4\u624b\u52a8\u7f16\u8f91\u672a\u81ea\u52a8\u8986\u76d6')
        mapping[ident]=(root,members)
        bounds=[v for o in members for v in _bounds(o)]
        if not bounds:
            continue
        points.extend(bounds)
        a=np.asarray(bounds); lo,hi=a.min(0),a.max(0)
        row={'id':ident,'name':root.name,'kind':str(root.get('wondful_env_type','building')),
             'center':((lo+hi)/2).tolist(),'dimensions':(hi-lo).tolist(),
             'matrix_world':[list(r) for r in root.matrix_world], 'hidden':bool(root.hide_render),
             'members':[o.name for o in members]}
        if row['kind']=='ribbon':
            row['sections']=_ribbon_sections(root)
        rows.append(row)
        for o in members:
            exact.append((o.name, o.data.as_pointer(), o.parent.name if o.parent else None,
                          [list(r) for r in o.matrix_world], [list(r) for r in o.matrix_parent_inverse],
                          bool(o.hide_render), bool(o.hide_get()), _mesh_signature(o.data)))
    cam=scene.camera
    if cam is None or cam.data.type!='PERSP':
        raise RuntimeError('\u8bf7\u5148\u9009\u62e9\u900f\u89c6\u76f8\u673a')
    camera={'name':cam.name,'matrix_world':[list(r) for r in cam.matrix_world],
            'lens_mm':cam.data.lens,'sensor_fit':cam.data.sensor_fit,'sensor_width':cam.data.sensor_width,
            'sensor_height':cam.data.sensor_height,'shift_x':cam.data.shift_x,'shift_y':cam.data.shift_y}
    product=[(o.name,[list(r) for r in o.matrix_world]) for o in props.product_collection.all_objects]
    target=workflow.file_hash(bpy.path.abspath(props.sb_image_path))
    identity=workflow.file_hash(workflow.identity_path(props,resolve=bpy.path.abspath))
    exact.extend([camera,product,target,identity])
    exact.append([scene.frame_current, owner.name, sorted((o.name,o.data.as_pointer() if o.data else 0) for o in owner.all_objects)])
    digest=hashlib.sha256(json.dumps(exact,sort_keys=True).encode()).hexdigest()
    return {'objects':rows,'camera':camera,'span':float(np.ptp(np.asarray(points),axis=0).max()),
            'signature':digest,'target_hash':target},mapping,owner


class Transaction:
    def __init__(self,context,props,mapping,owner):
        self.context=context;self.props=props;self.mapping=mapping;self.owner=owner
        self.scene=context.scene;self.original_camera=self.scene.camera
        self.objects={o for _,members in mapping.values() for o in members}
        self.states=[(o,o.matrix_world.copy(),o.data,o.hide_render,o.hide_get()) for o in self.objects]
        self.tags={o:{k:o.get(k) for k in ('wondful_sb_width_subdivisions','wondful_sb_skin_faces')} for o in self.objects}
        self.new_objects=[];self.new_meshes=[];self.old_meshes=[];self.new_camera_data=[]
        self.camera_state=(self.original_camera.matrix_world.copy(),self.original_camera.data.copy())
        self.original_preview=props.sb_preview_image
        self.original_signature=getattr(props,'replica_preview_signature','')
        self.result_signature='';self.closed=False

    def rollback(self):
        import bpy
        if self.closed:
            return
        for obj in reversed(self.new_objects):
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj,do_unlink=True)
        # Parent first, then children, so restoring world matrices is deterministic.
        def depth(obj):
            n=0
            while obj.parent is not None:
                n+=1;obj=obj.parent
            return n
        for obj,matrix,mesh,hide,view in sorted(self.states,key=lambda s:depth(s[0])):
            obj.data=mesh;obj.matrix_world=matrix;obj.hide_render=hide;obj.hide_set(view)
            for k,v in self.tags[obj].items():
                if v is None:
                    if k in obj:
                        del obj[k]
                else:
                    obj[k]=v
        self.scene.camera=self.original_camera
        self.original_camera.matrix_world=self.camera_state[0]
        saved=self.camera_state[1]
        for key in ('lens','shift_x','shift_y','sensor_width','sensor_height','sensor_fit'):
            setattr(self.original_camera.data,key,getattr(saved,key))
        self.props.sb_preview_image=self.original_preview
        self.props.replica_preview_signature=self.original_signature
        self.context.view_layer.update()
        self.dispose()

    def dispose(self, retain_meshes=()):
        import bpy
        if self.closed:
            return
        retained = set(retain_meshes)
        for mesh in set(self.new_meshes+self.old_meshes):
            try:
                if mesh not in retained and mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
            except ReferenceError:
                pass  # A later manual operation may already have removed this datablock.
        for data in [self.camera_state[1]]+self.new_camera_data:
            try:
                if data.users == 0:
                    bpy.data.cameras.remove(data)
            except ReferenceError:
                pass
        self.closed=True

    def apply(self,plan):
        import bpy
        from mathutils import Matrix,Vector,Euler
        from . import scene_blockout as SB
        for row in plan['ribbons']:
            root,_=self.mapping[row['id']]
            verts,faces,layout=SB.ribbon_mesh(row['sections'],row['thickness_m'],return_layout=True)
            mesh=bpy.data.meshes.new(root.name+'_revised')
            self.new_meshes.append(mesh)
            mesh.from_pydata(verts.tolist(),[],faces);mesh.update()
            for mat in root.data.materials:
                mesh.materials.append(mat)
            edges={}
            for poly in mesh.polygons:
                skin=poly.index<layout['skin_faces'];poly.use_smooth=skin
                for edge in poly.edge_keys:
                    edges.setdefault(tuple(sorted(edge)),set()).add(skin)
            for edge in mesh.edges:
                edge.use_edge_sharp=len(edges.get(tuple(sorted(edge.vertices)),()))>1
            self.old_meshes.append(root.data);root.data=mesh
            # Layout metadata is used only to recover sections for the next revision.
            root['wondful_sb_width_subdivisions']=layout['width_subdivisions']
            root['wondful_sb_skin_faces']=layout['skin_faces']
        self.context.view_layer.update()
        for row in plan['edits']:
            root,members=self.mapping[row['id']]
            import numpy as np
            a=np.asarray([p for o in members for p in _bounds(o)])
            center=Vector(((a.min(0)+a.max(0))/2).tolist())
            rotation=Euler(tuple(math.radians(x) for x in row['rotation_deg']),'XYZ').to_matrix().to_4x4()
            scaling=Matrix.Diagonal(Vector(tuple(row['scale'])+(1.0,)))
            root.matrix_world=Matrix.Translation(Vector(row['translation'])) @ Matrix.Translation(center) @ rotation @ scaling @ Matrix.Translation(-center) @ root.matrix_world
            for member in members:
                member.hide_render=row['hidden'];member.hide_set(row['hidden'])
        for row in plan['additions']:
            mesh=bpy.data.meshes.new(row['label']);self.new_meshes.append(mesh)
            verts=[(x/2,y/2,z/2) for x in (-1,1) for y in (-1,1) for z in (-1,1)]
            mesh.from_pydata(verts,[],[(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)])
            mesh.update();ob=bpy.data.objects.new(row['label'],mesh);self.new_objects.append(ob)
            self.owner.objects.link(ob)
            ob.location=row['location'];ob.scale=row['dimensions']
            ob.rotation_euler=tuple(math.radians(x) for x in row['rotation_deg'])
            ob['wondful_sb_group']=1+max((int(o.get('wondful_sb_group',0)) for o in self.owner.all_objects),default=0)
            ob['wondful_env_type']='building';ob['wondful_sb_revision_created']=1
            from . import massing
            massing.store_label(ob,row['label'])
        camera=plan['camera']
        if camera['apply']:
            cam=self.scene.camera
            if len(cam.constraints) or getattr(cam,'animation_data',None):
                raise RuntimeError('\u76f8\u673a\u542b\u7ea6\u675f\u6216\u52a8\u753b\uff0c\u672a\u81ea\u52a8\u8986\u76d6\uff1b\u8bf7\u5148\u4f7f\u7528\u65e0\u7ea6\u675f\u526f\u672c')
            if not cam.get('wondful_sb_camera') or cam.data.users > 1 or any(cam in set(s.objects) for s in bpy.data.scenes if s is not self.scene):
                cam=cam.copy();cam.data=cam.data.copy();self.new_camera_data.append(cam.data)
                self.new_objects.append(cam);self.scene.collection.objects.link(cam)
                cam.name='Wondful_Revision_Camera';cam['wondful_sb_camera']=1;self.scene.camera=cam
            rotation=Euler(tuple(math.radians(x) for x in camera['rotation_deg']),'XYZ').to_matrix().to_4x4()
            matrix=cam.matrix_world.copy()
            position=matrix.translation+Vector(camera['translation'])
            matrix=rotation @ matrix;matrix.translation=position;cam.matrix_world=matrix
            cam.data.lens=camera['lens_mm'];cam.data.shift_x=camera['shift_x'];cam.data.shift_y=camera['shift_y']
        self.context.view_layer.update()


def apply_plan(context,props,reply,expected_signature,render_preview):
    from . import scene_blockout as SB
    current,mapping,owner=snapshot(context,props)
    if current['signature']!=expected_signature:
        raise RuntimeError('\u573a\u666f\u6216\u53c2\u8003\u5df2\u53d8\u66f4\uff0c\u672a\u5e94\u7528\u8fc7\u671f\u4fee\u6539\uff1b\u8bf7\u91cd\u65b0\u63d0\u4ea4')
    plan=R.validate(reply,current,ribbon_builder=SB.ribbon_mesh)
    if plan.get('no_change'):
        return plan
    tx=Transaction(context,props,mapping,owner)
    try:
        tx.apply(plan)
        render_preview()  # failed native preview rolls the scene back, not just the UI
        tx.result_signature=snapshot(context,props)[0]['signature']
    except Exception:
        tx.rollback();raise
    old=_UNDO.pop(_key(context.scene),None)
    if old:
        # The previous result may now be the next transaction's undo backup.
        # Never delete its temporarily zero-user mesh while handing off ownership.
        old.dispose(retain_meshes=[state[2] for state in tx.states])
    _UNDO[_key(context.scene)]=tx
    props.replica_can_undo=True
    props.replica_preview_signature=tx.result_signature
    props.replica_revision_count+=1
    return plan


def undo(context,props):
    key=_key(context.scene);tx=_UNDO.get(key)
    if tx is None:
        props.replica_can_undo=False
        raise RuntimeError('\u5f53\u524d\u4f1a\u8bdd\u6ca1\u6709\u53ef\u64a4\u9500\u7684\u8c03\u6574')
    if snapshot(context,props)[0]['signature']!=tx.result_signature:
        raise RuntimeError('\u8c03\u6574\u540e\u573a\u666f\u5df2\u53d8\u66f4\uff0c\u4e3a\u907f\u514d\u8986\u76d6\u624b\u52a8\u7f16\u8f91\u672a\u6267\u884c\u64a4\u9500')
    tx.rollback();_UNDO.pop(key,None);props.replica_can_undo=False
    props.replica_revision_count=max(0,props.replica_revision_count-1)


def reset():
    """Called on unregister / load. No serialized promise of undo survives a reload."""
    for tx in list(_UNDO.values()):
        try:
            tx.dispose()
        except (ReferenceError,RuntimeError):
            pass
    _UNDO.clear()
