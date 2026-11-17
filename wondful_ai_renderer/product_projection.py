"""Main-thread capture and freshness checks for a real product-only reference."""
import hashlib
import json
import threading
from pathlib import Path
import numpy as np

from .source_reference import fingerprint, matches

GEOMETRY_TYPES = {"MESH", "CURVE", "SURFACE", "FONT", "META"}
BASE_HELPERS = frozenset({"keep", "matid", "matid_json"})
LIT_HELPERS = frozenset({"detail", "canvas"})


def require_matching_aspect(source, projection):
    sw, sh = source
    pw, ph = projection
    if min(sw, sh, pw, ph) <= 0 or abs(sw * ph - sh * pw) > max(sw, sh, pw, ph):
        raise ValueError("目标原图与当前相机画幅比例不一致；请手动将相机输出宽高比匹配原图后重新生成，不会自动裁切、拉伸或修改相机。")


def _pointer(value):
    return value.as_pointer() if hasattr(value, "as_pointer") else id(value)


def _appearance(objects):
    """Selected product appearance, including authoritative paint; no other scenes."""
    from .detail_paste import KEEP_PROP, KEEP_TEXT_PROP
    materials, images, trees = {}, {}, set()
    def plain(value):
        if value is None or isinstance(value,(str,bool,int,float)):return value
        if hasattr(value,'as_pointer'):return _pointer(value)
        try:return [plain(v) for v in value]
        except TypeError:return _pointer(value)
    def graph(tree):
        if tree is None or _pointer(tree) in trees:return None
        trees.add(_pointer(tree))
        nodes=[]
        for node in sorted(tree.nodes,key=lambda n:n.name):
            row=[node.name,node.bl_idname,[(direction,s.identifier,plain(s.default_value))
                 for direction,sockets in (('input',node.inputs),('output',node.outputs))
                 for s in sockets if hasattr(s,'default_value')]]
            row.append([(k,plain(getattr(node,k))) for k in ('operation','blend_type','interpolation','extension','projection') if hasattr(node,k)])
            ramp=getattr(node,'color_ramp',None)
            if ramp is not None:row.append([(e.position,list(e.color)) for e in ramp.elements])
            image=getattr(node,'image',None)
            if image is not None:
                key=_pointer(image)
                if key not in images:
                    pixels=np.empty(len(image.pixels),np.float32);image.pixels.foreach_get(pixels)
                    images[key]=[image.name,list(image.size),image.filepath,image.colorspace_settings.name,
                                 hashlib.sha256(pixels.tobytes()).hexdigest()]
                row.append(images[key])
            row.append(graph(getattr(node,'node_tree',None)))
            nodes.append(row)
        return [nodes,sorted((l.from_node.name,l.from_socket.identifier,l.to_node.name,l.to_socket.identifier) for l in tree.links)]
    result=[]
    for obj in sorted(objects,key=lambda o:o.name):
        prop=obj.get(KEEP_PROP) if hasattr(obj,'get') else None
        slots=[]
        for slot in getattr(obj,'material_slots',()):
            material=slot.material
            if material is None:slots.append(None);continue
            mp=material.get(KEEP_PROP)
            key=_pointer(material)
            if key not in materials:
                materials[key]=[list(material.diffuse_color),bool(material.use_nodes),graph(material.node_tree)]
            slots.append([key,material.name,mp,material.get(KEEP_TEXT_PROP),getattr(slot,'link',''),materials.get(key)])
        result.append([_pointer(obj),prop,obj.get(KEEP_TEXT_PROP) if hasattr(obj,'get') else None,slots])
    return result


def product_objects(context, props):
    collection = getattr(props, "product_collection", None)
    if collection is None:
        raise ValueError("请先选择产品集合，单图复刻需要真实产品结构参考。")
    objects = list(collection.all_objects)
    visible = [o for o in objects if o.type in GEOMETRY_TYPES and not o.hide_render
               and getattr(o, "visible_camera", True) and o.name in context.scene.objects]
    if not visible:
        raise ValueError("产品集合没有当前场景中可渲染的产品几何。")
    return objects


def signature(context, props):
    """Hash only selected product geometry; never hash unrelated scene meshes."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("产品结构签名只能在主线程读取。")
    from .multiview import scene_geometry_key
    scene = context.scene
    camera = scene.camera
    if camera is None:
        raise ValueError("请先选择活动相机，单图复刻需要同机位产品结构参考。")
    objects = product_objects(context, props)
    context.view_layer.update()
    evaluated_camera = camera.evaluated_get(context.evaluated_depsgraph_get())
    data, render = evaluated_camera.data, scene.render
    camera_data = {key: getattr(data, key, None) for key in
                   ("type", "lens", "sensor_width", "sensor_height", "sensor_fit", "ortho_scale", "shift_x", "shift_y")}
    camera_data["matrix"] = [list(row) for row in evaluated_camera.matrix_world]
    render_data = {key: getattr(render, key, None) for key in
                   ("resolution_x", "resolution_y", "resolution_percentage", "pixel_aspect_x", "pixel_aspect_y")}
    members = []
    for obj in sorted(objects, key=lambda o: o.name):
        modifiers = [(getattr(m, "name", ""), m.type, bool(m.show_render), bool(m.show_viewport),
                      getattr(m, "levels", None), getattr(m, "render_levels", None))
                     for m in getattr(obj, "modifiers", ())]
        members.append((_pointer(obj), obj.name, obj.type, bool(obj.hide_render),
                        bool(getattr(obj, "hide_viewport", False)), bool(getattr(obj, "visible_camera", True)), modifiers))
    collection = props.product_collection
    collections = [(_pointer(c), bool(c.hide_render), bool(c.hide_viewport))
                   for c in [collection, *collection.children_recursive]]
    layers = []
    owned = {_pointer(c) for c in [collection, *collection.children_recursive]}
    def visit(layer, ancestors):
        chain = ancestors + [(_pointer(layer.collection), bool(layer.exclude), bool(layer.hide_viewport))]
        if _pointer(layer.collection) in owned:
            layers.append(chain)
        for child in layer.children:
            visit(child, chain)
    if getattr(context.view_layer, "layer_collection", None) is not None:
        visit(context.view_layer.layer_collection, [])
    value = {"scene": _pointer(scene), "camera": [_pointer(camera), camera_data], "frame": [scene.frame_current, scene.frame_subframe],
             "render": render_data, "collection": _pointer(collection), "collections": collections,
             "layers": layers, "members": members,
             "appearance": _appearance(objects),
             "geometry": scene_geometry_key(context, objects=objects, render_modifiers=True)}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def capture(context, props, directory, source_dimensions, source_path=""):
    from .viewport_capture import render_detail_pass, camera_output_dimensions
    from .png_io import read_png
    require_matching_aspect(source_dimensions, camera_output_dimensions(context))
    before = signature(context, props)
    path = Path(directory) / "product_projection.png"
    keep_path=Path(directory)/'product_keep.png';matid_path=Path(directory)/'product_matid.png'
    info = render_detail_pass(context, str(path), props.product_collection,
                              keep_path=str(keep_path),matid_path=str(matid_path))
    if not info.get("detail") or not path.is_file():
        raise ValueError("真实产品结构参考捕获失败，已停止；请检查相机和产品集合。")
    pixels = read_png(path)
    if pixels.ndim != 3 or pixels.shape[2] != 4 or not (pixels[:, :, 3] > 0).any():
        raise ValueError("产品结构参考为空或缺少 RGBA 覆盖，请检查产品可见性与相机。")
    if not info.get('matid'):raise ValueError('产品材质 ID 捕获失败，已停止细节修复生成。')
    if not keep_path.exists() and not info.get('keep_slots') and not info.get('keep'):
        from .png_io import write_png
        write_png(keep_path,np.zeros_like(pixels))
    from .source_detail_repair import read_helpers
    metadata_path=Path(str(matid_path)+'.json')
    metadata=json.loads(metadata_path.read_text())
    from .detail_paste import KEEP_PROP, prop_state
    excluded=set()
    for obj in product_objects(context,props):
        for slot in getattr(obj,'material_slots',()):
            mat=slot.material
            if mat is not None and (prop_state(obj.get(KEEP_PROP)) is False or prop_state(mat.get(KEEP_PROP)) is False):
                excluded.add(mat.name)
    for material in metadata['materials']:
        material['repair_excluded']=material['name'] in excluded
    metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    read_helpers(path,keep_path,matid_path)
    lookdev = None
    if source_path:
        lookdev = _capture_lookdev(context, props, Path(directory), source_path, path)
    if signature(context, props) != before:
        raise ValueError("产品或相机在捕获期间发生变化，请重新生成。")
    rows=[('keep',keep_path),('matid',matid_path),('matid_json',metadata_path)]
    if lookdev:
        rows += [('detail', path), ('canvas', Path(lookdev['canvas']))]
    helpers={k:{'path':str(p),'snapshot':fingerprint(p)} for k,p in rows}
    sent = Path(lookdev['path']) if lookdev else path
    record = {"path": str(sent), "snapshot": fingerprint(sent), "signature": before, "helpers":helpers,
              "dimensions": [pixels.shape[1], pixels.shape[0]],
              "method": "PRODUCT_LIT_CANVAS" if lookdev else "PRODUCT_ONLY_RGBA",
              "camera": context.scene.camera.name, "collection": getattr(props.product_collection, "name", "")}
    if lookdev:
        record["lookdev"] = {k: v for k, v in lookdev.items() if k not in {"path", "canvas"}}
    return record


def _capture_lookdev(context, props, directory, source_path, flat_path):
    """Lit product pass (own materials, scene-sampled sky/ground, temporary clear coat, shadow
    catcher) and the placed-product canvas: the photograph with that pass composited at the exact
    camera position. Fails closed: a missing pass stops the edit instead of silently sending the
    flat projection."""
    from .product_lookdev import environment_colours, render_lookdev, build_canvas, LOOKDEV_VERSION
    from .reference_collage import read_rgb
    from .appearance_lock import collect_product_materials
    from .workflow import isolated_capture
    env = environment_colours(read_rgb(source_path))
    rows = collect_product_materials([o for o in product_objects(context, props) if o.type == "MESH"])
    look_path = directory / "product_lookdev.png"
    info = isolated_capture(render_lookdev)(context, str(look_path), props.product_collection, env, paint_rows=rows)
    if not info.get("ok") or not look_path.is_file():
        raise ValueError("产品光照参考渲染失败，已停止；请检查相机和产品集合。")
    canvas_path = directory / "source_product_canvas.png"
    build_canvas(source_path, look_path, flat_path, canvas_path)
    return {"path": str(look_path), "canvas": str(canvas_path), "version": LOOKDEV_VERSION,
            "environment": env, "coated": info.get("coated", []), "sun": info.get("sun")}


def current(context, props, record):
    try:
        return (set(record['helpers']) in (BASE_HELPERS, BASE_HELPERS | LIT_HELPERS)
                and all(matches(r['snapshot'],r['path']) for r in record['helpers'].values())
                and matches(record["snapshot"], record["path"]) and signature(context, props) == record["signature"])
    except (ValueError, OSError, RuntimeError, ReferenceError, AttributeError, TypeError, KeyError):
        return False


def freeze_helpers(record, directory):
    """Copy exact approved PNG/JSON bytes; no render or format conversion."""
    names={'keep':'product_keep.png','matid':'product_matid.png','matid_json':'product_matid.png.json'}
    if set(record.get('helpers',{}))==BASE_HELPERS|LIT_HELPERS:
        names.update({'detail':'product_detail_pass.png','canvas':'source_product_canvas.png'})
    if set(record.get('helpers',{}))!=set(names):raise ValueError('产品细节辅助文件不完整，请重新生成并润色。')
    result={}
    for key,name in names.items():
        row=record['helpers'][key]
        if not matches(row['snapshot'],row['path']):raise ValueError('产品细节辅助文件在润色后已变化。')
        target=Path(directory)/name;data=Path(row['path']).read_bytes()
        if target.resolve()==Path(row['path']).resolve():raise ValueError('必须复制到独立的生成会话，不能覆盖原始辅助文件。')
        if hashlib.sha256(data).hexdigest()!=row['snapshot']['sha256']:raise ValueError('产品细节辅助文件读取期间已变化。')
        target.write_bytes(data)
        if not matches(row['snapshot'],row['path']):raise ValueError('产品细节辅助文件复制期间已变化。')
        result[key]=str(target)
    return result
