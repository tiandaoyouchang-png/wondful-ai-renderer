"""Render real screenshots of the Wondful main panel in several states (GUI mode).

The 3D-view sidebar always shows Blender's own Item/Tool/View tabs first and
its active tab cannot be set from Python, so for screenshots the panel's own
draw() is hosted in a full-screen Text Editor sidebar (all of whose built-in
panels are Python and can be removed). The draw code is exactly
WONDFUL_PT_main.draw. Frames are grabbed from the X display with ImageMagick
`import` and cropped to the sidebar region.

    Xvfb :99 -screen 0 1400x1000x24 &
    DISPLAY=:99 blender --factory-startup <blank.blend> \
        --python tools/ui_screenshots.py -- <addon.zip> <out_dir>
"""
import os
import subprocess
import sys
import traceback

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ZIP, OUT = argv[0], argv[1]
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "screenshot_log.txt"), "w")
STATE = {"area": None, "window": None}


def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


def make_image(name, color, w=120, h=80):
    img = bpy.data.images.new(name, w, h)
    px = []
    for y in range(h):
        for x in range(w):
            t = 0.55 + 0.45 * (x / w)
            band = 0.75 if (w // 3 <= x < w // 3 + 2 or 2 * w // 3 <= x < 2 * w // 3 + 2) else 1.0
            px += [color[0] * t * band, color[1] * t * band, color[2] * t * band, 1.0]
    img.pixels[:] = px
    return img


def setup():
    bpy.ops.preferences.addon_install(filepath=ZIP, overwrite=True)
    bpy.ops.preferences.addon_enable(module="wondful_ai_renderer")
    import wondful_ai_renderer.ui as wui

    for name in dir(bpy.types):
        if name.startswith("TEXT_PT_"):
            try:
                bpy.utils.unregister_class(getattr(bpy.types, name))
            except Exception:
                pass

    class WONDFUL_PT_screenshot_host(bpy.types.Panel):
        bl_label = wui.WONDFUL_PT_main.bl_label
        bl_space_type = "TEXT_EDITOR"
        bl_region_type = "UI"
        bl_category = "Text"
        draw = wui.WONDFUL_PT_main.draw

    bpy.utils.register_class(WONDFUL_PT_screenshot_host)

    window = bpy.context.window_manager.windows[0]
    area = max(window.screen.areas, key=lambda a: a.width * a.height)
    area.type = "TEXT_EDITOR"
    log("setup ok")


def show_sidebar():
    window = bpy.context.window_manager.windows[0]
    for area in window.screen.areas:
        if area.type == "TEXT_EDITOR":
            area.spaces.active.show_region_ui = True
            area.tag_redraw()
    log("sidebar shown")


def widen_sidebar():
    """Drag the sidebar's left edge with xdotool (optional; WONDFUL_SHOT_SIDEBAR_PX)."""
    target = int(os.environ.get("WONDFUL_SHOT_SIDEBAR_PX", "0") or 0)
    window, area, region = _sidebar()
    if not target or region is None:
        return None
    x = region.x
    y = window.height - (region.y + region.height // 2)
    delta = max(0, target - region.width)
    # Run the drag asynchronously so Blender's event loop keeps processing it.
    script = (f"xdotool mousemove {x + 2} {y}; sleep 1; xdotool mousedown 1; sleep 1; "
              f"xdotool mousemove {x - delta // 3} {y}; sleep 1; xdotool mousemove {x - delta} {y}; sleep 1; "
              f"xdotool mouseup 1")
    subprocess.Popen(["/bin/sh", "-c", script])
    log(f"sidebar drag {region.width}px -> target {target}px")
    return None


STATES = []


def state(fn):
    STATES.append(fn)
    return fn


@state
def s1_not_connected(p, scene):
    p.codex_account_state = "LOGGED_OUT"


@state
def s2_connected_refs(p, scene):
    p.codex_account_state = "LOGGED_IN"
    col = bpy.data.collections.new("Product")
    scene.collection.children.link(col)
    p.product_collection = col
    item = p.product_images.add()
    item.image = make_image("three_view_chair.png", (0.95, 0.6, 0.3))
    for i, c in enumerate(((0.35, 0.55, 0.95), (0.85, 0.88, 0.95))):
        s = p.style_images.add()
        s.image = make_image(f"env_snow_{i}.png", c)
    p.scene_brief = "雪地清晨，冷色调"


@state
def s2b_count_2(p, scene):
    """1.2.2: ⑥ 张数 (⑤ before 1.2.3) 1/2/3/4 above 生成 (2 selected)."""
    p.render_count = "2"


@state
def s2c_antigravity_cap(p, scene):
    p.render_count = "4"
    p.analysis_provider = "ANTIGRAVITY"
    p.antigravity_account_state = "LOGGED_IN"


@state
def s2d_model_notice(p, scene):
    p.analysis_provider = "CODEX"
    p.model_notice = ("本机 Codex CLI 不支持推理模型 gpt-6.1-sol，已自动改用 gpt-5.6-sol；"
                      "请更新 Codex CLI（npm i -g @openai/codex）后即可使用新模型")


@state
def s3_generating(p, scene):
    p.model_notice = ""
    p.status = "POLISHING"
    p.task_phase = "分析当前风格 (1/2)"
    p.progress = 0.34


@state
def s4_rendering(p, scene):
    p.status = "RENDERING"
    p.task_phase = "生成第 1/2 张 · 尝试 1/3"
    p.progress = 0.68


@state
def s5_result(p, scene):
    p.status = "SUCCESS"
    p.task_phase = ""
    p.prompt = "清晨雪地，冷色调，柔和侧光，产品居中。"
    p.scene_brief_used = p.scene_brief
    p.result_image = make_image("result.png", (0.6, 0.7, 0.85), 160, 90)
    p.alignment_score = 92


@state
def s6_advanced(p, scene):
    p.ui_show_advanced = True
    p.ui_show_prompt = False


_i = [0]
_phase = [0]


def _sidebar():
    window = bpy.context.window_manager.windows[0]
    for area in window.screen.areas:
        if area.type == "TEXT_EDITOR":
            for region in area.regions:
                if region.type == "UI":
                    return window, area, region
    return window, None, None


def grab(path):
    window, area, region = _sidebar()
    raw = path + ".full.png"
    env = dict(os.environ)
    subprocess.run(["/usr/bin/import", "-window", "root", raw], check=True, env=env)
    if region is not None:
        win_h = window.height
        x, y, w, h = region.x, win_h - (region.y + region.height), region.width, region.height
        subprocess.run(["/usr/bin/convert", raw, "-crop", f"{w}x{h}+{x}+{y}", "+repage", path], check=True)
        os.remove(raw)
    else:
        os.replace(raw, path)


def tick():
    try:
        scene = bpy.context.scene
        p = scene.wondful_ai
        if _i[0] >= len(STATES):
            log("done")
            bpy.ops.wm.quit_blender()
            return None
        fn = STATES[_i[0]]
        if _phase[0] == 0:
            fn(p, scene)
            for area in bpy.context.window_manager.windows[0].screen.areas:
                area.tag_redraw()
            _phase[0] = 1
            return 8.0
        path = os.path.join(OUT, f"{fn.__name__}.png")
        grab(path)
        log(f"saved {path} (sidebar {_sidebar()[2].width}px)")
        _phase[0] = 0
        _i[0] += 1
        return 1.0
    except Exception:
        log(traceback.format_exc())
        bpy.ops.wm.quit_blender()
        return None


def start():
    try:
        setup()
    except Exception:
        log(traceback.format_exc())
    bpy.app.timers.register(lambda: (show_sidebar(), None)[1], first_interval=4.0)
    bpy.app.timers.register(widen_sidebar, first_interval=7.0)
    bpy.app.timers.register(tick, first_interval=22.0)
    return None


bpy.app.timers.register(start, first_interval=3.0)
