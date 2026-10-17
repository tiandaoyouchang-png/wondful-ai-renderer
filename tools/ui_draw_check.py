"""Headless panel draw check (3.2.0; 3.2.1 adds direct-edit result states; 1.1.0 custom terrain; 1.2.0 分层输出, 环境体块 + 建筑参考 + 三视图建筑体块; 1.2.3 从参考图生成场景粗模).

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
ADV_TABS = ("MODEL", "OUTPUT", "PRODUCT", "SCENE", "MULTIVIEW", "DIAG")


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

    def progress(self, **kw):
        _check_kwargs("progress", kw)
        self._log.append(("  " * self._depth) + f"progress {kw.get('text', '')}")

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
    # 1.2.3: 高级 shows one tab at a time; every tab must draw cleanly in this state
    for tab in ADV_TABS:
        props.ui_adv_tab = tab
        draw("connected_refs_tab_" + tab.lower())
    # 1.1.0: 高级 › 场景 › 地面 › 自定义 terrain box (presets, four corners, 坡顶 / 凹凸, pose toggle)
    props.ui_adv_tab = "SCENE"
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

    # 1.2.2: ⑤ 张数 1/2/3/4 right above 生成; Antigravity cap note; 推理 model fallback notice
    log = draw("count_default")
    if not any("render_count" in line for line in log):
        problems.append("count_default: 张数 (render_count) buttons missing")
    else:
        i_count = next(i for i, line in enumerate(log) if "render_count" in line)
        i_go = next((i for i, line in enumerate(log) if "wondful.generate" in line), -1)
        if i_go < i_count:
            problems.append("count_default: 张数 row is not above 生成")
    if props.render_count != "4":
        problems.append(f"render_count default is {props.render_count}, want 4")
    from wondful_ai_renderer.operators import render_variant_count
    for n in ("1", "2", "3", "4"):
        props.render_count = n
        if render_variant_count("codex", props) != int(n):
            problems.append(f"render_variant_count codex N={n}")
        draw("count_" + n)
    props.analysis_provider = "ANTIGRAVITY"
    log = draw("count_antigravity_cap")
    if render_variant_count("antigravity", props) != 2:
        problems.append("antigravity cap not applied")
    if not any("最多 2 张" in line for line in log):
        problems.append("antigravity cap note missing")
    props.analysis_provider = "CODEX"
    props.render_count = "4"
    props.model_notice = "本机 Codex CLI 不支持推理模型 gpt-6.1-sol，已自动改用 gpt-5.6-sol；请更新 Codex CLI（npm i -g @openai/codex）后即可使用新模型"
    log = draw("model_notice")
    if not any("Codex CLI" in line for line in log):
        problems.append("model_notice not drawn")
    props.model_notice = ""

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
    props.last_score_note = "1672×941 · 偏移 18px · 缩放 +8.0% · 轮廓 3.6px · 细节边缘 2.7px（回贴后 2.1px） · 颜色 ΔE 7（回贴后 6） · 待查：Black_Mat 4.8px"
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
    for o in bpy.context.selected_objects:      # factory Cube is selected: it would act as a placeholder
        o.select_set(False)
    bpy.context.view_layer.objects.active = None
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
        obj.select_set(False)                     # with the block selected it would be replaced in place
        bpy.context.view_layer.objects.active = None
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

    # 4a. 1.2.3 环境体块 › 从参考图生成场景粗模: closed / no reference / file reference / after a build / failed
    draw("scene_blockout_closed")
    props.ui_show_scene_blockout = True
    props.style_images.clear()
    props.sb_image_path = ""
    draw("scene_blockout_no_ref")
    try:
        res = bpy.ops.wondful.scene_blockout_build()
    except RuntimeError:          # an ERROR report raises in bpy.ops
        res = {"CANCELLED"}
    if res != {"CANCELLED"} or "③" not in props.last_sb_note:
        problems.append(f"scene_blockout_build without a reference should refuse: {res} {props.last_sb_note}")
    draw("scene_blockout_refused")
    ref_png = os.path.join(tmp, "env_ref.png")
    png_io.write_png(ref_png, np.full((90, 160, 4), 180, np.uint8))
    props.sb_image_path = ref_png
    props.last_sb_note = ""
    draw("scene_blockout_ready")
    import json as _json
    from pathlib import Path as _P
    fixture = _P(__file__).resolve().parents[1] / "tests" / "fixtures" / "scene_blockout" / "desert_v3.json"
    from wondful_ai_renderer import scene_blockout_build as SBB, scene_blockout as SB
    parent, _c = MS.ensure_collection(scene, props)
    info = SBB.build(bpy.context, _json.loads(fixture.read_text(encoding="utf-8")), product_collection=props.product_collection,
                     parent_collection=parent, set_camera=True)
    info["scores"] = [0.612, 0.664]
    info["round"] = 2
    props.last_sb_note = SB.summary(info)
    sbc = bpy.data.collections.get(SB.COLLECTION_NAME)
    if sbc is None or sbc.name not in [c.name for c in parent.children] or MS.find_collection(scene, props) != parent:
        problems.append("场景粗模 collection is not inside the massing collection")
    if bpy.context.scene.camera is None or bpy.context.scene.camera.name != SB.CAMERA_NAME:
        problems.append("场景粗模 did not set its reference camera")
    print("scene blockout", props.last_sb_note)
    draw("scene_blockout_built")
    props.last_sb_note = "场景粗模失败：未找到 Codex 登录文件，请先在 ① 连接 Codex"
    draw("scene_blockout_failed")
    # running: stage list, 「分析参考图 · 已用 1:42」, typical-time hint, bar, 取消 replaces the build button
    from wondful_ai_renderer import sb_progress as SP
    for correct in (False, True):
        ids = SP.stages(correct, True)
        props.sb_stage_ids = SP.encode(ids)
        for stage in (SP.ANALYZE, SP.BUILD) + ((SP.CORRECT, SP.REBUILD) if correct else ()):
            props.status = "BLOCKOUT"
            props.sb_stage = stage
            props.sb_elapsed = 102
            props.progress = SP.fraction(ids, stage, 102)
            props.task_phase = "场景粗模：" + SP.status_text(stage, 102)
            log = draw(f"scene_blockout_running_{'corr_' if correct else ''}{stage.lower()}")
            text = "\n".join(log)
            if f"{SP.label(stage)} · 已用 1:42" not in text:
                problems.append(f"running {stage}: elapsed stage text missing")
            if "op wondful.cancel_task" not in text or "op wondful.scene_blockout_build" in text:
                problems.append(f"running {stage}: 取消 should replace the build button")
            if stage == SP.ANALYZE and "约 3–5 分钟" not in text:
                problems.append("running ANALYZE: typical-time hint missing")
            # 1.2.3: the running stage + bar stay visible in ⑤ whatever 高级 tab is open (or 高级 closed)
            for tab in ADV_TABS + ("CLOSED",):
                props.ui_show_advanced = tab != "CLOSED"
                if tab != "CLOSED":
                    props.ui_adv_tab = tab
                tlog = "\n".join(draw(f"sb_running_{stage.lower()}_tab_{tab.lower()}"))
                if f"场景粗模 · {SP.label(stage)} · 已用 1:42" not in tlog or "progress" not in tlog \
                        or "op wondful.cancel_task" not in tlog:
                    problems.append(f"running {stage}: ⑤ progress missing with tab {tab}")
            props.ui_show_advanced = True
            props.ui_adv_tab = "SCENE"
    props.ui_show_scene_blockout = False
    log = draw("scene_blockout_running_closed")
    if "分析参考图" not in "\n".join(log) and "已用" not in "\n".join(log):
        problems.append("collapsed section should still show the running stage")
    props.ui_show_scene_blockout = True
    props.task_phase = "正在取消，等待当前请求结束"
    draw("scene_blockout_cancelling")
    # failed at a stage: list shows ERROR on that stage, note names it
    props.status = "ERROR"
    props.task_phase = ""
    props.progress = 0.0
    props.sb_stage = ""
    props.sb_stage_ids = SP.encode(SP.stages(True, True))
    props.sb_failed_stage = SP.CORRECT
    props.last_sb_note = SP.failure_note(SP.CORRECT, 260, "网络超时")
    log = draw("scene_blockout_failed_stage")
    if "ERROR 校正分析 · 失败" not in "\n".join(log):
        problems.append("failed stage row missing")
    props.sb_failed_stage = ""
    props.status = "IDLE"
    props.last_error = ""
    props.last_sb_note = "场景粗模已取消（分析参考图 · 已用 0:40）"
    draw("scene_blockout_cancelled")
    props.last_sb_note = ""
    props.sb_image_path = ""
    props.ui_show_scene_blockout = False
    SBB.remove_existing()

    # 4b. 1.2.3 多角度场景一致: closed, open without master, with master + selected cameras, after a run
    props.ui_show_multiview = True
    props.ui_adv_tab = "MULTIVIEW"
    draw("multiview_empty")
    cam_data = bpy.data.cameras.new("mv_cam")
    master = bpy.data.objects.new("MV主视角", cam_data)
    other = bpy.data.objects.new("MV其他", cam_data.copy())
    for o in (master, other):
        bpy.context.scene.collection.objects.link(o)
    bpy.context.scene.camera = master
    res = bpy.ops.wondful.mv_set_master()
    if props.mv_master_camera != master:
        problems.append(f"mv_set_master did not set the master: {res}")
    for o in bpy.context.view_layer.objects:
        o.select_set(o == other)
    log = draw("multiview_ready")          # empty list: falls back to the selected camera
    text = "\n".join(log)
    if "列表为空：用视图里选中的相机（1 个）" not in text or "渲染 1 个相机" not in text \
            or "添加相机槽" not in text or "从当前视图新建相机" not in text or "＋" in text:
        problems.append("multiview empty list: fallback hint / add buttons / count missing")
    # 3 slots: other, a second camera, the master (skipped)
    second = bpy.data.objects.new("MV侧面", cam_data.copy())
    bpy.context.scene.collection.objects.link(second)
    for _ in range(3):
        bpy.ops.wondful.mv_camera_add()
    if len(props.mv_cameras) != 3:
        problems.append(f"mv_camera_add: {len(props.mv_cameras)} rows")
    props.mv_cameras[0].camera, props.mv_cameras[1].camera, props.mv_cameras[2].camera = other, second, master
    log = draw("multiview_three_slots")
    text = "\n".join(log)
    for want in ("相机 1", "相机 2", "相机 3", "渲染 2 个相机", "相机 3：与主视角相同，跳过", "op wondful.mv_camera_remove X"):
        if want not in text:
            problems.append(f"multiview 3 slots: missing {want!r}")
    if "列表为空" in text:
        problems.append("multiview 3 slots: still shows the selection fallback")
    props.mv_cameras[1].enabled = False
    if "渲染 1 个相机" not in "\n".join(draw("multiview_slot_disabled")):
        problems.append("disabled slot still counted")
    props.mv_cameras[1].enabled = True
    # one missing camera: row 2's camera removed from the scene
    bpy.context.scene.collection.objects.unlink(second)
    log = draw("multiview_missing_camera")
    text = "\n".join(log)
    if "相机 2：未选相机或相机已不在场景里" not in text or "渲染 1 个相机" not in text:
        problems.append("multiview missing camera: hint / count wrong")
    bpy.context.scene.collection.objects.link(second)
    bpy.ops.wondful.mv_camera_remove(index=2)
    if len(props.mv_cameras) != 2 or props.mv_cameras[1].camera != second:
        problems.append("mv_camera_remove removed the wrong row")
    from wondful_ai_renderer import operators as _OPS
    from mathutils import Matrix
    cam_new = _OPS.mv_new_view_camera(bpy.context, Matrix.Translation((1, 2, 3)), 25.0)
    if cam_new.name != "Wondful 视角 1" or props.mv_cameras[-1].camera != cam_new or cam_new.type != "CAMERA":
        problems.append(f"mv_new_view_camera: {cam_new.name} {len(props.mv_cameras)}")
    if _OPS.mv_new_view_camera(bpy.context, Matrix.Identity(4), 25.0).name != "Wondful 视角 2":
        problems.append("second view camera not numbered 2")
    rows_before = len(props.mv_cameras)
    try:
        res = bpy.ops.wondful.mv_camera_from_view()   # factory screen has a 3D view; without one it refuses
    except RuntimeError:
        res = {"CANCELLED"}
    made = bpy.data.objects.get("Wondful 视角 3")
    if res == {"FINISHED"}:
        if made is None or made.type != "CAMERA" or len(props.mv_cameras) != rows_before + 1 \
                or props.mv_cameras[-1].camera != made or made.name not in bpy.context.scene.objects:
            problems.append("mv_camera_from_view did not create + list 「Wondful 视角 3」")
        print("camera from view", made.name, tuple(round(v, 2) for v in made.location), round(made.data.lens, 2))
    elif res != {"CANCELLED"} or made is not None:
        problems.append(f"mv_camera_from_view: {res}")
    draw("multiview_view_cameras")
    props.mv_cameras.clear()
    for o in [o for o in bpy.data.objects if o.name.startswith("Wondful 视角")] + [second]:
        bpy.data.objects.remove(o, do_unlink=True)
    props.last_mv_note = ("多角度场景一致：2/2 个相机完成 · 图像调用 3 次 · 重试：MV其他（细节边缘 7.1px） · "
                          "注意：产品前方有遮挡物（MV其他 遮挡 38%），可移动相机")
    draw("multiview_done")
    props.last_mv_note = ""
    # 1.2.3 体块名称: the active block's name field (auto label from 环境体块, then edited by the user)
    block = coll.objects[0]
    bpy.context.view_layer.objects.active = block
    if not block.get(MS.AUTO_HINT_FLAG) or block.wondful_hint != "建筑":
        problems.append(f"hull block auto label missing: {dict(block.items()) if hasattr(block, 'items') else ''}")
    draw("multiview_block_name")
    block.wondful_hint = "红砖老厂房"
    if MS.AUTO_HINT_FLAG in block.keys() or block.get("wondful_hint") != "红砖老厂房" \
            or MS.group_hint([block]) != ("红砖老厂房", "custom"):
        problems.append(f"block name edit did not become the user's hint: {MS.group_hint([block])}")
    block.wondful_hint = ""
    if MS.group_hint([block]) != ("建筑", "name"):
        problems.append(f"empty block name did not fall back to the object name: {MS.group_hint([block])}")
    draw("multiview_block_name_empty")
    props.ui_show_multiview = False

    # 4c. 1.2.3: a 多角度 run reports in ⑤ (phase + bar, not the four 生成 stages) on every tab
    props.status = "RENDERING"
    props.task_phase = "多角度场景一致：MV其他（1/2）"
    props.progress = 0.4
    for tab in ADV_TABS:
        props.ui_adv_tab = tab
        text = "\n".join(draw("mv_running_tab_" + tab.lower()))
        if "多角度场景一致：MV其他（1/2）" not in text or "progress" not in text or "读产品图" in text \
                or "op wondful.cancel_task" not in text:
            problems.append(f"multiview running: ⑤ progress wrong with tab {tab}")
    props.status = "IDLE"
    props.task_phase = ""
    props.progress = 0.0

    # 5. error + antigravity provider + authenticating
    props.status = "ERROR"
    props.last_error = "测试错误：网络超时，请稍后重试。" * 3
    for tab in ADV_TABS:
        props.ui_adv_tab = tab
        draw("error_tab_" + tab.lower())
    draw("error")
    props.analysis_provider = "ANTIGRAVITY"
    props.status = "AUTHENTICATING"
    props.auth_stage = "WAITING_CODE"
    draw("antigravity_auth")
    for tab in ADV_TABS:
        props.ui_adv_tab = tab
        draw("antigravity_auth_tab_" + tab.lower())
    props.status = "IDLE"
    for tab in ADV_TABS:
        props.ui_adv_tab = tab
        draw("antigravity_tab_" + tab.lower())

    bpy.ops.preferences.addon_disable(module="wondful_ai_renderer")
    print("PROBLEMS", len(problems))
    for p in problems:
        print("  ", p)
    sys.exit(1 if problems else 0)


main()
