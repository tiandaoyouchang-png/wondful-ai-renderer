"""Headless panel draw check (3.2.0; 3.2.1 adds direct-edit result states; 1.1.0 custom terrain; 1.2.0 分层输出, 环境体块 + 建筑参考 + 三视图建筑体块; 1.2.3 从参考图生成场景粗模 as main-flow step ④（可选）, six step cards).

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
ADV_TABS = ("MODEL", "OUTPUT", "FINISH", "PRODUCT", "SCENE", "MULTIVIEW", "DIAG")


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

    def template_list(self, listtype, list_id, data, prop, active_data, active_prop, **kw):
        _check_kwargs("template_list", kw)
        for d, a in ((data, prop), (active_data, active_prop)):
            if a not in d.bl_rna.properties.keys():
                problems.append(f"template_list: missing {a!r}")
        cls = getattr(bpy.types, listtype, None)
        if cls is None:
            problems.append(f"template_list: UIList {listtype!r} not registered")
            return
        self._log.append(("  " * self._depth) + f"[template_list {listtype}]")
        for index, item in enumerate(getattr(data, prop)):
            row = FakeLayout(self._log, self._depth + 1)
            self._log.append(("  " * (self._depth + 1)) + f"[item {index}]")
            cls.draw_item(cls, bpy.context, row, data, item, 0, active_data, active_prop, index)


class FakePanel:
    pass


class _PrefsProxy:
    """AddonPreferences.draw with a recording layout: attribute reads go to the real preferences."""

    def __init__(self, prefs, layout):
        object.__setattr__(self, "_prefs", prefs)
        object.__setattr__(self, "layout", layout)

    def __getattr__(self, key):
        return getattr(self._prefs, key)


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
    ops_lines = [l.strip() for l in log if l.strip().startswith("op ")]
    star = "op wm.url_open SOLO_ON GitHub · 欢迎点 Star"
    if not ops_lines or ops_lines[-1] != star or ops_lines.count(star) != 1:
        problems.append("GitHub footer is not the single last row of the panel")
    new = problems[before:]
    for i in range(before, len(problems)):
        problems[i] = f"[{state_name}] " + problems[i]
    print(f"--- state {state_name}: {len(log)} layout items, {len(new)} problem(s)")
    for line in log:
        if "label" in line or "op " in line or "template_icon" in line:
            print("    " + line)
    return log


def main_part(log):
    """The main flow (step cards) part of a log, i.e. everything above 高级."""
    for i, line in enumerate(log):
        if "label" in line and line.rstrip().endswith(" 高级"):
            return log[:i]
    return log


def step4(log):
    """The ④ 场景粗模 card: from its title to the ⑤ 提示词 title."""
    start = next((i for i, l in enumerate(log) if l.rstrip().endswith("④  场景粗模") or l.rstrip().endswith("CHECKMARK 场景粗模")), -1)
    end = next((i for i, l in enumerate(log) if i > start and ("⑤  提示词" in l or "CHECKMARK 提示词" in l)), len(log))
    return log[start:end] if start >= 0 else []


def check_six_steps(name, log):
    """① ② ③ ④ ⑤ ⑥ in order, ④ = 场景粗模 with the 可选 tag; the build button only in ④, never in 高级."""
    titles = []
    for title in ("连接", "产品三视图（可选）", "环境参考图", "场景粗模", "提示词", "生成"):
        idx = next((i for i, l in enumerate(log) if "label" in l and (l.rstrip().endswith("  " + title)
                    or l.rstrip().endswith("CHECKMARK " + title))), -1)
        titles.append(idx)
    if -1 in titles or titles != sorted(titles):
        problems.append(f"[{name}] six step cards missing or out of order: {titles}")
    card = step4(log)
    if not any(l.rstrip().endswith("label  可选") for l in card[:3]):
        problems.append(f"[{name}] ④ has no 可选 tag")
    builds = [i for i, l in enumerate(log) if "op wondful.scene_blockout_build" in l]
    if len(builds) > 1 or (builds and "op wondful.scene_blockout_build" not in "\n".join(card)):
        problems.append(f"[{name}] 生成场景粗模 button outside ④ (or twice)")
    return card


def step6(log):
    """The ⑥ 生成 card up to 高级."""
    main = main_part(log)
    start = next((i for i, l in enumerate(main) if l.rstrip().endswith("⑥  生成") or l.rstrip().endswith("CHECKMARK 生成")), -1)
    return main[start:] if start >= 0 else []


def views_box(log):
    """The 渲染视角 box lines inside ⑥ (empty when absent)."""
    six = step6(log)
    start = next((i for i, l in enumerate(six) if "label OUTLINER_OB_CAMERA 渲染视角" in l), -1)
    if start < 0:
        return []
    end = next((i for i, l in enumerate(six) if i > start and "op wondful.generate" in l), len(six))
    return six[start - 1:end]


def check_render_views(scene, props):
    """1.2.3 ⑥ 渲染视角 (blockout in the scene): empty list, 3 cameras with 2 checked, drop fields, plan / flip."""
    import json as _json
    old_cam = scene.camera
    props.mv_cameras.clear()
    log = draw("views_blockout_empty")
    box = "\n".join(views_box(log))
    print("    ---- 渲染视角 box (blockout, empty list) ----\n" + box)
    for want in ("当前相机", "prop mv_drop_camera 拖入相机", "prop mv_drop_collection 拖入集合",
                 "op wondful.mv_camera_from_view VIEW_CAMERA 从当前视图新建相机", "不选 = 用当前场景相机"):
        if want not in box:
            problems.append(f"views empty: missing {want!r}")
    if "template_list" in box or "op wondful.generate RENDER_STILL 生成\n" not in "\n".join(step6(log)) + "\n":
        problems.append("views empty: list drawn without rows, or 生成 label changed")
    cams = []
    holder = bpy.data.collections.new("视角相机")
    scene.collection.children.link(holder)
    sub = bpy.data.collections.new("视角相机_子")
    holder.children.link(sub)
    for i, name in enumerate(("视角前", "视角侧", "视角后")):
        o = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        (holder if i < 2 else sub).objects.link(o)
        cams.append(o)
    props.mv_drop_camera = cams[1]                       # drag one camera from the Outliner
    if [i.camera for i in props.mv_cameras] != [cams[1]] or props.mv_drop_camera is not None:
        problems.append("drop camera: not appended / field not cleared")
    props.mv_drop_collection = holder                    # drag the collection: the rest, recursive, no repeat
    if [i.camera.name for i in props.mv_cameras] != ["视角侧", "视角前", "视角后"] or props.mv_drop_collection is not None:
        problems.append(f"drop collection: {[i.camera.name for i in props.mv_cameras]}")
    props.mv_camera_index = 1
    bpy.ops.wondful.mv_camera_move(direction="UP")       # 视角前 first
    if [i.camera.name for i in props.mv_cameras] != ["视角前", "视角侧", "视角后"] or props.mv_camera_index != 0:
        problems.append("mv_camera_move UP wrong")
    props.mv_cameras[1].enabled = False                  # 3 cameras, 2 checked
    log = draw("views_blockout_three")
    box = "\n".join(views_box(log))
    print("    ---- 渲染视角 box (blockout, 3 cameras, 2 checked) ----\n" + box)
    for want in ("已选 2 个", "[template_list WONDFUL_UL_render_views]", "label OUTLINER_OB_CAMERA 视角前",
                 "label SOLO_ON 主视角", "op wondful.mv_camera_move TRIA_UP", "op wondful.mv_camera_remove X",
                 "先渲染主视角 视角前，其余 1 个按场景粗模保持一致", "已加入 2 个相机 · 视角侧 已在列表"):
        if want not in box:
            problems.append(f"views three: missing {want!r}")
    if box.count("主视角") != 2:                                       # the tag once + the hint once
        problems.append("views three: 主视角 tag should be on exactly one row")
    if "op wondful.generate RENDER_STILL 生成 2 个视角" not in "\n".join(step6(log)):
        problems.append("views three: 生成 button should read 生成 2 个视角")
    from wondful_ai_renderer.operators import generate_views
    if [o.name for o in generate_views(bpy.context, props)] != ["视角前", "视角后"]:
        problems.append("generate_views: wrong cameras / order")
    props.mv_cameras[0].enabled = False
    if "label SOLO_ON 主视角" not in "\n".join(views_box(draw("views_blockout_one"))):
        problems.append("views one: 主视角 tag missing")
    props.mv_cameras[0].enabled = True
    # running a plan: 视角 1/2 above the stages, box hidden
    props.views_plan = _json.dumps({"cameras": ["视角前", "视角后"], "total": 2}, ensure_ascii=False)
    props.status, props.task_phase, props.progress = "RENDERING", "生成第 1/4 张 · 尝试 1/3", 0.4
    six = "\n".join(step6(draw("views_plan_main")))
    if "视角 1/2 · 视角前（主视角）" not in six or "渲染视角" in six:
        problems.append("views plan main: 视角 1/2 line missing or box shown while running")
    props.task_phase = "多角度场景一致 · 视角 2/2：视角后"
    six = "\n".join(step6(draw("views_plan_rest")))
    if "多角度场景一致 · 视角 2/2：视角后" not in six or "progress" not in six:
        problems.append("views plan rest: phase / bar missing")
    props.views_plan = ""
    props.status, props.task_phase, props.progress = "SUCCESS", "", 0.0
    # result card flips between views
    import tempfile as _tf, os as _os
    import numpy as _np
    from wondful_ai_renderer import png_io as _png
    d = _tf.mkdtemp()
    paths = []
    for i, n in enumerate(("视角前", "视角后")):
        pth = _os.path.join(d, f"{n}.png")
        _png.write_png(pth, _np.full((24, 32, 4), 60 + 90 * i, _np.uint8))
        paths.append(pth)
    props.result_image = make_image("view_result.png", (0.4, 0.4, 0.4, 1))
    props.view_results = _json.dumps([{"camera": "视角前", "path": paths[0], "master": True},
                                      {"camera": "视角后", "path": paths[1], "master": False}], ensure_ascii=False)
    props.view_result_index = 0
    log = "\n".join(step6(draw("views_result_flip")))
    if "视角 1/2 · 视角前（主视角）" not in log or "op wondful.view_result_step TRIA_RIGHT" not in log:
        problems.append("views result: flip row missing")
    bpy.ops.wondful.view_result_step(step=1)
    if props.view_result_index != 1 or props.last_result_path != paths[1] or "视角 2/2 · 视角后" not in "\n".join(draw("views_result_flip2")):
        problems.append("view_result_step did not switch the shown view")
    bpy.ops.wondful.view_result_step(step=1)
    if props.view_result_index != 0:
        problems.append("view_result_step does not wrap")
    props.view_results = ""
    props.view_result_index = 0
    props.result_image = None
    props.status = "IDLE"
    props.mv_drop_note = ""
    props.mv_cameras.clear()
    for o in cams:
        bpy.data.objects.remove(o, do_unlink=True)
    for c in (sub, holder):
        bpy.data.collections.remove(c)
    scene.camera = old_cam


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
    log = draw("connected_refs")
    if views_box(log) or "渲染视角" in "\n".join(main_part(log)):       # no 粗模: ⑥ exactly as before
        problems.append("connected_refs: 渲染视角 box shown without a 场景粗模")
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

    # 1.2.2: ⑥ 张数 (⑤ before 1.2.3) 1/2/3/4 right above 生成; Antigravity cap note; 推理 model fallback notice
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
    # 1.2.3 细节修复 stages in ⑥ (worker phase text)
    for phase in ("处理第 1/2 张 · 细节检查", "处理第 1/2 张 · 细节修复 1/2", "处理第 2/2 张 · 细节修复 2/2"):
        props.task_phase = phase
        log = draw("generating_detail_" + str(len(phase)))
        if not any(phase in line for line in log):
            problems.append(f"detail stage text not drawn: {phase}")
    props.task_phase = "生成第 1/2 张 · 尝试 1/3"

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
    # 1.2.3 细节修复 report: result card + 高级 › 诊断 › 上次渲染 + 高级 › 产品 › 细节修复
    for note in ("细节：5 处合格 · 修复 2 处 · 放弃 0 · 生图 2 次 · 84 秒", "细节：全部合格（6 处）",
                 "细节：4 处合格 · Blender 回贴 1 处 · 放弃 1（无可用生图编辑） · 3 秒",
                 "细节检查已跳过：对齐不可靠（细节底图与成片不匹配），保留 AI 原图"):
        props.last_detail_note = note
        log = draw("result_detail")
        short = note[:12]
        if not any(short in line for line in main_part(log)):
            problems.append(f"detail note not on result card: {note}")
        for tab in ("DIAG", "PRODUCT"):
            props.ui_adv_tab = tab
            log = draw("result_detail_" + tab.lower())
            if not any(short in line for line in log):
                problems.append(f"detail note not in 高级 {tab}: {note}")
    props.last_detail_note = ""
    props.ui_adv_tab = "SCENE"
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

    # 4a. 1.2.3 ④ 场景粗模（可选）: no reference / refused / file reference / built (✓) / failed / running / cancelled
    for tab in ADV_TABS + ("CLOSED",):
        props.ui_show_advanced = tab != "CLOSED"
        if tab != "CLOSED":
            props.ui_adv_tab = tab
        log = draw("sb_idle_tab_" + tab.lower())
        card = check_six_steps("sb_idle_tab_" + tab.lower(), log)
        text = "\n".join(card)
        if "op wondful.scene_blockout_build MOD_BUILD 生成场景粗模" not in text or "prop sb_image_path" not in text \
                or "prop sb_set_camera" not in text or "prop sb_correct" not in text or "约 3–5 分钟" not in text:
            problems.append(f"sb idle ({tab}): ④ compact contents missing")
        if "CHECKMARK 场景粗模" in text:
            problems.append(f"sb idle ({tab}): ④ checked without a blockout")
    props.ui_show_advanced = True
    props.ui_adv_tab = "SCENE"
    props.style_images.clear()
    props.sb_image_path = ""
    log = draw("scene_blockout_no_ref")
    if "③ 还没有环境参考图" not in "\n".join(step4(log)):
        problems.append("sb no ref: ④ should say ③ has no reference")
    try:
        res = bpy.ops.wondful.scene_blockout_build()
    except RuntimeError:          # an ERROR report raises in bpy.ops
        res = {"CANCELLED"}
    if res != {"CANCELLED"} or "③" not in props.last_sb_note or "④" not in props.last_sb_note:
        problems.append(f"scene_blockout_build without a reference should refuse: {res} {props.last_sb_note}")
    draw("scene_blockout_refused")
    ref_png = os.path.join(tmp, "env_ref.png")
    png_io.write_png(ref_png, np.full((90, 160, 4), 180, np.uint8))
    props.sb_image_path = ref_png
    props.last_sb_note = ""
    check_six_steps("scene_blockout_ready", draw("scene_blockout_ready"))
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
    log = draw("scene_blockout_built")
    card = check_six_steps("scene_blockout_built", log)
    if "CHECKMARK 场景粗模" not in "\n".join(card):
        problems.append("sb built: ④ number should turn into ✓")
    props.last_sb_note = ""
    if "已有场景粗模" not in "\n".join(step4(draw("scene_blockout_built_reopened"))):
        problems.append("sb built (no note): ④ should say a blockout exists")
    check_render_views(scene, props)
    props.last_sb_note = "场景粗模失败：未找到 Codex 登录文件，请先在 ① 连接 Codex"
    draw("scene_blockout_failed")
    # running: stage list, 「分析参考图 · 已用 1:42」, typical-time hint, bar, 取消 replaces the build button (all in ④);
    # ⑥ carries one compact 「④ 场景粗模 · … · N%」 line, whatever 高级 tab is open
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
            pct = int(round(props.progress * 100))
            for tab in ADV_TABS + ("CLOSED",):
                props.ui_show_advanced = tab != "CLOSED"
                if tab != "CLOSED":
                    props.ui_adv_tab = tab
                name = f"sb_running_{'corr_' if correct else ''}{stage.lower()}_tab_{tab.lower()}"
                log = draw(name)
                card = "\n".join(check_six_steps(name, log))
                main_text = "\n".join(main_part(log))
                if f"{SP.label(stage)} · 已用 1:42" not in card or "progress" not in card \
                        or "op wondful.cancel_task" not in card or "op wondful.scene_blockout_build" in card:
                    problems.append(f"{name}: ④ stage / bar / 取消 wrong")
                if f"progress 第 {ids.index(stage) + 1}/{len(ids)} 步 · {pct}%" not in card or "PLAY" in card:
                    problems.append(f"{name}: ④ bar should read 第 i/n 步 · N% (no stage list while running)")
                if stage == SP.ANALYZE and "约 3–5 分钟" not in card:
                    problems.append(f"{name}: typical-time hint missing")
                if "CHECKMARK 场景粗模" in card:
                    problems.append(f"{name}: ④ checked while running")
                six = main_text[main_text.index("⑥  生成"):] if "⑥  生成" in main_text else ""
                if f"TIME ④ 场景粗模 · {SP.label(stage)} · 已用 1:42 · {pct}%" not in six or "op wondful.cancel_task" not in six:
                    problems.append(f"{name}: ⑥ one-line status / 取消 missing")
                if six.count("progress") or "读产品图" in six:
                    problems.append(f"{name}: ⑥ should carry one line, not a second bar / the 生成 stages")
            props.ui_show_advanced = True
            props.ui_adv_tab = "SCENE"
    props.task_phase = "正在取消，等待当前请求结束"
    text = "\n".join(draw("scene_blockout_cancelling"))
    if "正在取消，等待当前请求结束…" not in text or "④ 场景粗模 · 正在取消…" not in text:
        problems.append("sb cancelling: ④ / ⑥ texts missing")
    # failed at a stage: list shows ERROR on that stage, note names it
    props.status = "ERROR"
    props.task_phase = ""
    props.progress = 0.0
    props.sb_stage = ""
    props.sb_stage_ids = SP.encode(SP.stages(True, True))
    props.sb_failed_stage = SP.CORRECT
    props.last_sb_note = SP.failure_note(SP.CORRECT, 260, "网络超时")
    props.last_error = props.last_sb_note
    log = draw("scene_blockout_failed_stage")
    card = "\n".join(check_six_steps("scene_blockout_failed_stage", log))
    if "ERROR 校正分析 · 失败" not in card or "op wondful.scene_blockout_build" not in card:
        problems.append("sb error: ④ failed stage row / rebuild button missing")
    props.sb_failed_stage = ""
    props.status = "IDLE"
    props.last_error = ""
    props.last_sb_note = "场景粗模已取消（分析参考图 · 已用 0:40）"
    draw("scene_blockout_cancelled")
    props.last_sb_note = ""
    props.sb_image_path = ""
    SBB.remove_existing()
    log = draw("scene_blockout_removed")
    if "CHECKMARK 场景粗模" in "\n".join(step4(log)):
        problems.append("sb removed: ④ still checked")
    if views_box(log):
        problems.append("sb removed: 渲染视角 box still shown")

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

    # 4c. 1.2.3: a 多角度 run reports in ⑥ (phase + bar, not the four 生成 stages) on every tab
    props.status = "RENDERING"
    props.task_phase = "多角度场景一致：MV其他（1/2）"
    props.progress = 0.4
    for tab in ADV_TABS:
        props.ui_adv_tab = tab
        text = "\n".join(draw("mv_running_tab_" + tab.lower()))
        if "多角度场景一致：MV其他（1/2）" not in text or "progress" not in text or "读产品图" in text \
                or "op wondful.cancel_task" not in text:
            problems.append(f"multiview running: ⑥ progress wrong with tab {tab}")
    props.status = "IDLE"
    props.task_phase = ""
    props.progress = 0.0

    # 4d. 1.2.4 高级 › 成片 › 照片质感 (关 / 细微 / 标准 + last note) and 多角度 › 统一调色
    props.ui_show_advanced = True
    props.ui_adv_tab = "FINISH"
    for level in ("OFF", "SUBTLE", "STANDARD"):
        props.photo_finish = level
        text = "\n".join(draw("finish_" + level.lower()))
        if "photo_finish" not in text:
            problems.append(f"成片: photo_finish selector missing ({level})")
    props.last_finish_note = "照片质感：细微"
    draw("finish_note")
    props.last_finish_note = "照片质感失败（已保留原图）：测试"
    draw("finish_note_fail")
    props.last_finish_note = ""
    props.photo_finish = "SUBTLE"
    props.ui_adv_tab = "MULTIVIEW"
    if "mv_colour_match" not in "\n".join(draw("multiview_colour_match")):
        problems.append("多角度: 统一调色 toggle missing")

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

    # 6. 1.2.3 更新: banner on top of ① (发现新版本 / 下载中 / 完成 / 失败) + 高级 › 诊断 › 更新
    from wondful_ai_renderer import update_ops as UO
    props.analysis_provider = "CODEX"
    props.ui_show_advanced = True
    props.ui_adv_tab = "DIAG"
    fake = {"version": "9.9.9", "zip": "Wondful-AI-Renderer-Blender-9.9.9.zip", "sha256": "0" * 64,
            "size": 1000, "notes": "细节修复更稳；新增一键更新"}
    os.environ["WONDFUL_UPDATE_STATE_DIR"] = tempfile.mkdtemp()
    UO.STATE.update(phase="IDLE", manifest=None, message="", progress=0.0, notes=[])
    log = draw("update_none")
    if any("发现新版本" in l for l in log) or "op wondful.update_check" not in "\n".join(log):
        problems.append("update none: banner shown without an update, or 检查更新 missing in 诊断")
    expect = {
        "AVAILABLE": ("INFO 发现新版本 9.9.9", "op wondful.update_install IMPORT 更新", "op wondful.update_ignore X 忽略此版本"),
        "DOWNLOADING": ("更新到 9.9.9 · 下载中 42%", "progress 下载中 42%"),
        "DONE": ("CHECKMARK 已更新到 9.9.9，请重启 Blender",),
        "FAILED": ("ERROR 校验失败", "op wondful.update_open_page"),
    }
    msgs = {"AVAILABLE": "发现新版本 9.9.9", "DOWNLOADING": "下载中 42%", "DONE": "已更新到 9.9.9，请重启 Blender",
            "FAILED": "校验失败：sha256 不一致（abc…，应为 000…）。当前版本未改动；可到 GitHub 手动下载安装包"}
    for phase, wants in expect.items():
        UO.STATE.update(phase=phase, manifest=fake, message=msgs[phase], progress=0.42,
                        notes=["已删除旧版本副本 1 个"] if phase == "DONE" else [])
        for tab in ("DIAG", "CLOSED"):
            props.ui_show_advanced = tab != "CLOSED"
            log = draw(f"update_{phase.lower()}_{tab.lower()}")
            top = "\n".join(log[:next((i for i, l in enumerate(log) if "连接" in l), len(log))])
            for want in wants:
                if want not in top:
                    problems.append(f"update {phase} ({tab}): banner above ① lacks {want!r}")
            check_six_steps(f"update_{phase.lower()}_{tab.lower()}", log)
    UO.STATE.update(phase="CHECK_FAILED", manifest=None, message="检查失败：无法连接更新服务器（已尝试 3 个地址）。可稍后再试，或到 GitHub 手动下载")
    props.ui_show_advanced = True
    text = "\n".join(draw("update_check_failed_diag"))
    if "ERROR 检查失败" not in text or "发现新版本" in text:
        problems.append("update check failed: 诊断 should show the error, no banner")
    UO.STATE.update(phase="IDLE", manifest=None, message="", progress=0.0, notes=[])
    props.ui_adv_tab = "SCENE"

    # 1.2.3: GitHub link in the add-on preferences too (url checked on the operator properties)
    prefs_log = []
    holder = type("P", (), {})()
    holder.layout = FakeLayout(prefs_log)
    try:
        type(prefs).draw(_PrefsProxy(prefs, holder.layout), bpy.context)
    except Exception:
        problems.append("preferences draw raised:\n" + traceback.format_exc())
    if not any("op wm.url_open SOLO_ON GitHub · 欢迎点 Star" in l for l in prefs_log):
        problems.append("preferences: GitHub link missing")
    from wondful_ai_renderer.ui import GITHUB_URL
    if GITHUB_URL != "https://github.com/tiandaoyouchang-png/wondful-ai-renderer":
        problems.append(f"GitHub url wrong: {GITHUB_URL}")
    print("--- preferences:", [l.strip() for l in prefs_log if "url_open" in l])

    bpy.ops.preferences.addon_disable(module="wondful_ai_renderer")
    print("PROBLEMS", len(problems))
    for p in problems:
        print("  ", p)
    sys.exit(1 if problems else 0)


main()
