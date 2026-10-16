"""Headless panel draw check (3.2.0; 3.2.1 adds direct-edit result states; 1.1.0 custom terrain; 1.2.0 分层输出, 环境体块 + 建筑参考 + 三视图建筑体块).

Installs the release ZIP, enables it, then draws WONDFUL_PT_main against a
strict recording layout in several panel states. Every layout call is checked
against Blender's real RNA: function/keyword names, icon names, property names,
operator idnames and operator property names. Exits non-zero on any problem.

    blender -b --factory-startup --python tools/ui_draw_check.py -- dist/Wondful-AI-Renderer-Blender-3.2.1.zip
"""
import sys
import traceback

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ZIP = argv[0]

UI = bpy.types.UILayout.bl_rna
ICONS = set(UI.functions["prop"].parameters["icon"].enum_items.keys())
problems = []


def _check_kwargs(fn, kwargs):
    rna = UI.functions.get(fn)
    if rna is None:
        problems.append(f"UILayout.{fn} does not exist in this Blender")
        return
    names = set(rna.parameters.keys())
    for key, value in kwargs.items():
        if key not in names:
            problems.append(f"UILayout.{fn}: unknown keyword {key!r}")
        if key == "icon" and value not in ICONS:
            problems.append(f"UILayout.{fn}: unknown icon {value!r}")


class FakeOpProps:
    def __init__(self, idname):
        object.__setattr__(self, "_idname", idname)
        mod, name = idname.split(".")
        rna = getattr(getattr(bpy.ops, mod), name).get_rna_type()
        object.__setattr__(self, "_props", set(rna.properties.keys()))

    def __setattr__(self, key, value):
        if key not in self._props:
            problems.append(f"operator {self._idname}: no property {key!r}")


class FakeLayout:
    """Mirrors the UILayout surface used by ui.py; attributes are plain values."""

    def __init__(self, log, depth=0):
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "_depth", depth)
        for attr in ("enabled", "active", "alert"):
            object.__setattr__(self, attr, True)
        object.__setattr__(self, "scale_x", 1.0)
        object.__setattr__(self, "scale_y", 1.0)
        object.__setattr__(self, "alignment", "EXPAND")

    def __setattr__(self, key, value):
        if key not in UI.properties.keys():
            problems.append(f"UILayout has no attribute {key!r}")
        object.__setattr__(self, key, value)

    def _child(self, kind, **kw):
        _check_kwargs(kind, kw)
        self._log.append(("  " * self._depth) + f"[{kind}]")
        return FakeLayout(self._log, self._depth + 1)

    def row(self, **kw): return self._child("row", **kw)
    def column(self, **kw): return self._child("column", **kw)
    def box(self, **kw): return self._child("box", **kw)
    def split(self, **kw): return self._child("split", **kw)
    def grid_flow(self, **kw): return self._child("grid_flow", **kw)
    def column_flow(self, **kw): return self._child("column_flow", **kw)

    def separator(self, **kw):
        _check_kwargs("separator", kw)

    def label(self, **kw):
        _check_kwargs("label", kw)
        self._log.append(("  " * self._depth) + f"label {kw.get('icon', '')} {kw.get('text', '')}")

    def prop(self, data, attr, **kw):
        _check_kwargs("prop", kw)
        if attr not in data.bl_rna.properties.keys():
            problems.append(f"prop: {type(data).__name__} has no RNA property {attr!r}")
        self._log.append(("  " * self._depth) + f"prop {attr} {kw.get('text', '')}")

    def prop_search(self, data, attr, sdata, sattr, **kw):
        _check_kwargs("prop_search", kw)
        for d, a in ((data, attr), (sdata, sattr)):
            if a not in d.bl_rna.properties.keys():
                problems.append(f"prop_search: missing {a!r}")

    def operator(self, idname, **kw):
        _check_kwargs("operator", kw)
        mod, name = idname.split(".")
        try:
            getattr(getattr(bpy.ops, mod), name).get_rna_type()
        except Exception:
            problems.append(f"operator {idname!r} not registered")
            return type("Dummy", (), {"__setattr__": lambda *a: None})()
        self._log.append(("  " * self._depth) + f"op {idname} {kw.get('icon', '')} {kw.get('text', '')}")
        return FakeOpProps(idname)

    def template_icon(self, **kw):
        _check_kwargs("template_icon", kw)
        self._log.append(("  " * self._depth) + f"template_icon scale={kw.get('scale')}")

    def template_ID_preview(self, data, prop, **kw):
        _check_kwargs("template_ID_preview", kw)
        if prop not in data.bl_rna.properties.keys():
            problems.append(f"template_ID_preview: missing {prop!r}")

    def template_list(self, *a, **kw):
        _check_kwargs("template_list", kw)


class FakePanel:
    pass


def make_image(name, color):
    img = bpy.data.images.new(name, 64, 48)
    img.pixels[:] = list(color) * (64 * 48)
    return img


def draw(state_name):
    from wondful_ai_renderer.ui import WONDFUL_PT_main
    log = []
    panel = FakePanel()
    panel.layout = FakeLayout(log)
    before = len(problems)
    try:
        WONDFUL_PT_main.draw(panel, bpy.context)
    except Exception:
        problems.append(f"[{state_name}] draw raised:\n" + traceback.format_exc())
    new = problems[before:]
    for i in range(before, len(problems)):
        problems[i] = f"[{state_name}] " + problems[i]
    print(f"--- state {state_name}: {len(log)} layout items, {len(new)} problem(s)")
    for line in log:
        if "label" in line or "op " in line or "template_icon" in line:
            print("    " + line)
    return log


def main():
    bpy.ops.preferences.addon_install(filepath=ZIP, overwrite=True)
    bpy.ops.preferences.addon_enable(module="wondful_ai_renderer")
    import wondful_ai_renderer
    print("installed version", wondful_ai_renderer.bl_info["version"], wondful_ai_renderer.bl_info["name"])
    scene = bpy.context.scene
    props = scene.wondful_ai

    # 1. fresh, not connected
    props.codex_account_state = "LOGGED_OUT"
    draw("not_connected")

    # 2. connected + product and environment refs loaded, advanced open
    props.codex_account_state = "LOGGED_IN"
    col = bpy.data.collections.new("Product")
    scene.collection.children.link(col)
    props.product_collection = col
    if scene.camera is None:
        cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
        scene.collection.objects.link(cam)
        scene.camera = cam
    item = props.product_images.add()
    item.image = make_image("three_view_chair.png", (0.8, 0.5, 0.2, 1))
    for i, c in enumerate(((0.2, 0.3, 0.8, 1), (0.9, 0.9, 0.9, 1))):
        s = props.style_images.add()
        s.image = make_image(f"env_{i}.png", c)
    props.scene_brief = "雪地清晨，冷色调"
    props.ui_show_prompt = True
    props.ui_show_advanced = True
    draw("connected_refs")
    # 1.1.0: 高级 › 地面 › 自定义 terrain box (presets, four corners, 坡顶 / 凹凸, pose toggle)
    props.clay_ground = "CUSTOM"
    for preset in ("CREST", "TWIST", "TWIST_MIRROR", "BUMPY", "CLIMB", "DESCENT", "SIDE"):
        bpy.ops.wondful.terrain_preset(preset=preset)
        log = draw("terrain_" + preset.lower())
        if not any("wondful.terrain_preset" in line for line in log):
            problems.append(f"terrain_{preset}: preset buttons missing")
        if not any("clay_terrain_bump" in line for line in log):
            problems.append(f"terrain_{preset}: 凹凸 slider missing")
    bpy.ops.wondful.terrain_preset(preset="TWIST")
    if (props.clay_terrain_fl, props.clay_terrain_fr, props.clay_terrain_bl, props.clay_terrain_br) != (12.0, -12.0, -12.0, 12.0):
        problems.append("terrain_preset TWIST did not fill the corners")
    bpy.ops.wondful.terrain_preset(preset="CREST")
    if not props.clay_terrain_crest > 0.3 or props.clay_terrain_fl != 0.0:
        problems.append("terrain_preset CREST did not set 坡顶")
    # a file saved with the old 前/后/左/右 controls: raw ID props are migrated to corners
    props["clay_terrain_front"] = 1
    props["clay_terrain_front_angle"] = 10.0
    props["clay_terrain_back"] = 2
    props["clay_terrain_back_angle"] = 10.0
    from wondful_ai_renderer.terrain import migrate_legacy_props
    if not migrate_legacy_props(props) or (props.clay_terrain_fl, props.clay_terrain_bl) != (10.0, -10.0) \
            or props.get("clay_terrain_front") is not None:
        problems.append("legacy terrain props were not migrated")
    props.clay_terrain_follow = False
    draw("terrain_custom_level")
    props.clay_ground = "AUTO"

    # 3. generating: polishing then rendering
    props.status = "POLISHING"
    props.task_phase = "分析当前风格 (1/2)"
    props.progress = 0.35
    draw("generating_polish")
    props.status = "RENDERING"
    props.task_phase = "生成第 1/2 张 · 尝试 1/3"
    props.progress = 0.7
    draw("generating_render")

    # 4. result
    props.status = "SUCCESS"
    props.prompt = "清晨雪地，冷色调，柔和侧光。"
    props.result_image = make_image("result.png", (0.5, 0.6, 0.7, 1))
    props.alignment_score = 92
    draw("result")
    props.last_engine_note = "生图：Codex 图像编辑（直连）"
    draw("result_direct_edit")
    props.last_engine_note = "直连编辑不可用（HTTP 404：接口不存在或已变更），已自动改用 Codex 代理生成"
    draw("result_direct_fallback")
    prefs = bpy.context.preferences.addons["wondful_ai_renderer"].preferences
    prefs.codex_direct_edit = False
    draw("result_direct_off")
    prefs.codex_direct_edit = True
    # 1.2.1 识别分 (高级 › 上次渲染)
    props.last_score_note = "识别分 88（回贴后 92） · 构图 100 · 轮廓 80 · 细节 85 · 材质色 86 · 待查：ALUM 颜色、tyre_3 细节"
    draw("result_identity_score")
    props.last_score_note = ""
    # 1.2.0 分层输出 (高级)
    props.layered_output = True
    draw("layers_shadow")
    props.layered_mode = "FULL"
    props.last_layers_note = "分层输出完成 2 张（主体 + 投影 + 背景，PSD + PNG），用时 41s"
    draw("layers_full_done")
    props.last_layers_note = "分层输出失败：产品遮罩未生成"
    draw("layers_failed")
    props.layered_output = False
    props.last_layers_note = ""
    # 1.2.0 环境体块 (高级): no collection yet → hull builder creates 「环境体块」
    props.massing_enabled = True
    props.ui_show_massing_hull = True
    draw("massing_none")
    import tempfile, os
    import numpy as np
    from wondful_ai_renderer import png_io
    from wondful_ai_renderer import massing as MS
    tmp = tempfile.mkdtemp()
    # a stepped tower: front 120w × 300h (narrower top), side 90w × 300h, top 120 × 90
    for name, (h, w) in {"front": (300, 120), "side": (300, 90), "top": (90, 120)}.items():
        a = np.full((h + 40, w + 40, 4), 255, np.uint8)
        a[20:20 + h, 20:20 + w, :3] = 60
        if name in ("front", "side"):
            a[20:20 + h // 3, 20:20 + w // 4, :3] = 255           # set-back upper third
            a[20:20 + h // 3, 20 + w - w // 4:20 + w, :3] = 255
        png_io.write_png(os.path.join(tmp, name + ".png"), a)
        setattr(props, f"hull_{name}_path", os.path.join(tmp, name + ".png"))
    product_before = props.product_collection
    scene.cursor.location = (6.0, 4.0, 3.0)
    props.hull_resolution = 64
    res = bpy.ops.wondful.massing_hull_build()
    coll = bpy.data.collections.get("环境体块")
    obj = bpy.context.view_layer.objects.active
    if res != {"FINISHED"} or coll is None or obj is None or obj.name not in coll.objects:
        problems.append(f"massing_hull_build failed: {res} {props.last_hull_note}")
    else:
        dims = tuple(round(v, 2) for v in obj.dimensions)
        print("hull block", obj.name, dims, tuple(obj.location), props.last_hull_note)
        if abs(dims[2] - 20.0) > 0.6 or abs(dims[0] - 8.0) > 0.6 or abs(dims[1] - 6.0) > 0.6:
            problems.append(f"hull block dims off (want 8 × 6 × 20 m): {dims}")
        if tuple(round(v, 3) for v in obj.location) != (6.0, 4.0, 0.0):
            problems.append(f"hull block not at the cursor on the ground: {tuple(obj.location)}")
        if len(MS.get_refs(obj)) != 3:
            problems.append(f"hull block refs: {MS.get_refs(obj)}")
        if props.product_collection != product_before:
            problems.append("hull build changed the product collection")
        if MS.find_collection(scene, props) != coll:
            problems.append("new 环境体块 collection is not auto-detected as massing")
        bpy.ops.wondful.massing_hull_build()
        if len(coll.objects) != 2:
            problems.append("second hull build did not add a second block")
    draw("massing_hull_built")
    # per-block refs: add to the active block through the operator (files), cap at 3, clear one
    tower = bpy.data.objects.new("高楼", bpy.data.meshes.new("高楼"))
    coll.objects.link(tower)
    bpy.context.view_layer.objects.active = tower
    files = [os.path.join(tmp, n + ".png") for n in ("front", "side", "top")]
    extra = os.path.join(tmp, "extra.png")
    png_io.write_png(extra, np.full((8, 8, 4), 200, np.uint8))
    res = bpy.ops.wondful.block_ref_add(directory=tmp, files=[{"name": "front.png"}, {"name": "side.png"}])
    res2 = bpy.ops.wondful.block_ref_add(directory=tmp, files=[{"name": "top.png"}, {"name": "extra.png"}])
    if MS.get_refs(tower) != files:
        problems.append(f"block_ref_add: {res} {res2} {MS.get_refs(tower)}")
    draw("massing_block_refs")
    bpy.ops.wondful.block_ref_clear(index=1)
    if MS.get_refs(tower) != [files[0], files[2]]:
        problems.append(f"block_ref_clear(index) wrong: {MS.get_refs(tower)}")
    bpy.ops.wondful.block_ref_clear()
    if MS.get_refs(tower) or MS.REF_PROP in tower.keys():
        problems.append("block_ref_clear did not clear")
    bpy.ops.wondful.block_select(name=coll.objects[0].name)
    if bpy.context.view_layer.objects.active != coll.objects[0]:
        problems.append("block_select did not select")
    props.hull_input = "SHEET"
    props.hull_size_mode = "MANUAL"
    props.last_hull_note = "建筑体块生成失败：三视图拼图里只找到 1 个轮廓：请把正视 / 侧视 / 俯视分开排列"
    draw("massing_hull_sheet_failed")
    props.last_hull_note = ""
    props.ui_show_massing_hull = False

    # 5. error + antigravity provider + authenticating
    props.status = "ERROR"
    props.last_error = "测试错误：网络超时，请稍后重试。" * 3
    draw("error")
    props.analysis_provider = "ANTIGRAVITY"
    props.status = "AUTHENTICATING"
    props.auth_stage = "WAITING_CODE"
    draw("antigravity_auth")

    bpy.ops.preferences.addon_disable(module="wondful_ai_renderer")
    print("PROBLEMS", len(problems))
    for p in problems:
        print("  ", p)
    sys.exit(1 if problems else 0)


main()
