"""Real-Blender e2e for 更新检查 + 一键更新 (1.2.3).

    BLENDER_USER_SCRIPTS=<out>/scripts BLENDER_USER_CONFIG=<out>/config BLENDER_USER_EXTENSIONS=<out>/ext \\
    blender -b --factory-startup --python tools/e2e_update_bg.py -- <out>

Installs the add-on from the working tree, plants two duplicate copies (another add-on dir + the extensions
user repo), serves a fake 9.9.9 manifest + zip from a local HTTP server behind a dead first mirror, then:
startup check → banner; throttle (no second fetch within 24 h); 忽略此版本; manual check with the network
down (error shown) and an auto check (silent); update from a mirror serving a tampered zip (FAILED, install
untouched, duplicates kept); real update (files replaced, duplicates removed, no staging/backup left, the new
copy enables as 9.9.9). Prints the banner text per state. Exit 0 + 「E2E UPDATE OK」 on success.
"""
import hashlib
import http.server
import json
import os
import shutil
import sys
import threading
import zipfile
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = Path(argv[0] if argv else "/tmp/wondful_update_e2e").resolve()
OUT.mkdir(parents=True, exist_ok=True)
problems = []
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"   # the local server must not go via a proxy


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        problems.append(msg)


def zip_package(src: Path, out: Path):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            zf.write(path, Path("wondful_ai_renderer") / path.relative_to(src))


def tree_digest(path: Path):
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(path)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


class Recorder:
    """Minimal layout recorder: labels / operators / progress text in draw order."""
    def __init__(self, log):
        self.log = log

    def __getattr__(self, name):
        if name in ("row", "column", "box", "split", "grid_flow", "column_flow"):
            return lambda *a, **k: Recorder(self.log)
        if name == "label":
            return lambda **k: self.log.append(f"[{k.get('icon', '')}] {k.get('text', '')}")
        if name == "operator":
            def op(idname, **k):
                self.log.append(f"<按钮 {k.get('text', '')}> ({idname})")
                return type("P", (), {})()
            return op
        if name == "progress":
            return lambda **k: self.log.append(f"[进度条 {k.get('factor', 0):.0%}] {k.get('text', '')}")
        return lambda *a, **k: None

    def __setattr__(self, key, value):
        if key == "log":
            object.__setattr__(self, key, value)


def banner(label):
    from wondful_ai_renderer import ui
    log = []
    ui._draw_update_banner(Recorder(log), bpy.context)
    print(f"--- banner [{label}]")
    for line in log or ["(无横幅)"]:
        print("    " + line)
    return "\n".join(log)


class Handler(http.server.SimpleHTTPRequestHandler):
    hits = []

    def log_message(self, *a):
        pass

    def translate_path(self, path):
        # /good/<path> → OUT/server/good/<path>, /bad/<path> → OUT/server/bad/<path>
        Handler.hits.append(path)
        return str(OUT / "server" / path.lstrip("/").split("?")[0])


def main():
    # 1. install the working-tree add-on (version 1.2.3) into the sandboxed user scripts dir
    cur_zip = OUT / "current.zip"
    zip_package(ROOT / "wondful_ai_renderer", cur_zip)
    bpy.ops.preferences.addon_install(filepath=str(cur_zip), overwrite=True)
    bpy.ops.preferences.addon_enable(module="wondful_ai_renderer")
    import wondful_ai_renderer
    from wondful_ai_renderer import update_ops as UO, updater as U
    target = UO.ADDON_DIR
    print("loaded from", target, "version", UO.local_version())
    check(str(target).startswith(str(OUT)), f"add-on installed inside the sandbox dir ({target})")
    check(UO.local_version() == "1.2.4", "running version is 1.2.4")

    # duplicates: an older manually-unzipped copy beside it + one in the extensions user repo
    dup1 = target.parent / "Wondful-AI-Renderer-Blender-1.2.2"
    shutil.copytree(target, dup1, ignore=shutil.ignore_patterns("__pycache__"))
    (dup1 / "__init__.py").write_text((dup1 / "__init__.py").read_text(encoding="utf-8").replace(
        '"version": (1, 2, 3)', '"version": (1, 2, 2)'), encoding="utf-8")
    ext_dirs = [r.directory for r in bpy.context.preferences.extensions.repos if getattr(r, "directory", "")
                and str(r.directory).startswith(str(OUT))]
    dup2 = Path(ext_dirs[0]) / "wondful_ai_renderer" if ext_dirs else OUT / "ext_fallback" / "wondful_ai_renderer"
    shutil.copytree(target, dup2, ignore=shutil.ignore_patterns("__pycache__"))
    other = target.parent / "some_other_addon"
    other.mkdir(exist_ok=True)
    (other / "__init__.py").write_text("bl_info = {'name': 'Other', 'version': (1, 0, 0)}\n")
    roots = UO.addon_roots()
    print("addon roots", roots)
    dups = U.find_duplicates(target, roots)
    check(sorted(p.name for p in dups) == sorted([dup1.name, dup2.name]),
          f"duplicates detected: {[str(p) for p in dups]}")

    # 2. fake 9.9.9 release + local server (good + tampered roots), first mirror dead
    pkg = OUT / "pkg999" / "wondful_ai_renderer"
    shutil.copytree(ROOT / "wondful_ai_renderer", pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    init = (pkg / "__init__.py").read_text(encoding="utf-8")
    import re as _re
    ver_pat = _re.compile(r'"version": \(\d+, \d+, \d+\)')
    check(bool(ver_pat.search(init)), "source bl_info pattern found")   # was pinned to 1.2.3 (broke at 1.2.4)
    (pkg / "__init__.py").write_text(ver_pat.sub('"version": (9, 9, 9)', init, count=1), encoding="utf-8")
    (pkg / "UPDATED_MARKER.txt").write_text("9.9.9", encoding="utf-8")
    for root in ("good", "bad"):
        (OUT / "server" / root / "releases").mkdir(parents=True, exist_ok=True)
    zname = "Wondful-AI-Renderer-Blender-9.9.9.zip"
    zpath = OUT / "server" / "good" / "releases" / zname
    zip_package(pkg, zpath)
    manifest = U.build_manifest(zpath, "9.9.9", "测试版：一键更新")
    for root in ("good", "bad"):
        (OUT / "server" / root / "releases" / "latest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    data = zpath.read_bytes()
    (OUT / "server" / "bad" / "releases" / zname).write_bytes(data[:-10] + b"X" * 10)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    dead = "http://127.0.0.1:9/dead"
    good_chain = f"{dead},{base}/good"
    os.environ["WONDFUL_UPDATE_STATE_DIR"] = str(OUT / "state")
    os.environ["WONDFUL_UPDATE_BASE_URLS"] = good_chain

    # 3. startup check (what the deferred timer runs) → banner
    banner("启动前")
    UO._startup()
    UO.wait(30)
    state = U.load_state(OUT / "state")
    check(UO.STATE["phase"] == "AVAILABLE" and (UO.banner_manifest() or {}).get("version") == "9.9.9",
          f"startup check finds 9.9.9 via the 2nd mirror (phase {UO.STATE['phase']})")
    check(state.get("last_check", 0) > 0 and state.get("latest", {}).get("version") == "9.9.9", "throttle state saved")
    text = banner("发现新版本")
    check("发现新版本 9.9.9" in text and "<按钮 更新>" in text and "<按钮 忽略此版本>" in text, "banner: 发现新版本 + 更新 + 忽略此版本")

    # 4. throttle: a second startup within 24 h does not fetch, but still shows the cached update
    hits = len([h for h in Handler.hits if h.endswith("latest.json")])
    UO.STATE.update(phase="IDLE", manifest=None, message="")
    UO._startup()
    UO.wait(5)
    check(len([h for h in Handler.hits if h.endswith("latest.json")]) == hits, "no second fetch within 24 h")
    check((UO.banner_manifest() or {}).get("version") == "9.9.9", "cached update still shown after restart")

    # 5. 忽略此版本 → no banner, also after a restart; manual check still reports it in 诊断
    bpy.ops.wondful.update_ignore(version="9.9.9")
    check(UO.banner_manifest() is None, "ignored: banner hidden")
    UO.STATE.update(phase="IDLE", manifest=None, message="")
    UO._startup()
    UO.wait(5)
    check(UO.banner_manifest() is None, "ignored: still hidden after restart")
    bpy.ops.wondful.update_check()
    check(UO.STATE["phase"] == "AVAILABLE" and "9.9.9" in UO.STATE["message"], f"manual check reports 9.9.9: {UO.STATE['message']}")

    # 6. network down: manual check shows the error, auto check is silent
    os.environ["WONDFUL_UPDATE_BASE_URLS"] = dead
    bpy.ops.wondful.update_check()
    print("manual check offline:", UO.STATE["message"])
    check(UO.STATE["phase"] == "CHECK_FAILED" and "检查失败" in UO.STATE["message"], "manual offline check shows an error")
    UO.start_check(manual=False)
    UO.wait(30)
    check(UO.STATE["phase"] == "IDLE" and UO.STATE["message"] == "", "auto offline check is silent")
    st = U.load_state(OUT / "state")
    st.pop("ignored", None)
    st["last_check"] = 0
    U.save_state(OUT / "state", st)
    os.environ["WONDFUL_UPDATE_BASE_URLS"] = good_chain
    UO._startup()
    UO.wait(30)
    check((UO.banner_manifest() or {}).get("version") == "9.9.9", "un-ignored: banner back")

    # 7. tampered zip on every mirror → FAILED, install + duplicates untouched
    before = tree_digest(target)
    os.environ["WONDFUL_UPDATE_BASE_URLS"] = f"{dead},{base}/bad"
    bpy.ops.wondful.update_install()
    print("tampered:", UO.STATE["message"])
    check(UO.STATE["phase"] == "FAILED" and "sha256" in UO.STATE["message"] and "GitHub" in UO.STATE["message"],
          "tampered zip rejected with a sha256 error + GitHub hint")
    check(tree_digest(target) == before, "failed update left the install byte-identical")
    check(dup1.is_dir() and dup2.is_dir(), "failed update kept the duplicates")
    check(not [p for p in target.parent.iterdir() if p.name.startswith((U.STAGING_PREFIX, U.BACKUP_PREFIX))],
          "no staging/backup left after the failure")
    text = banner("校验失败")
    check("校验失败" in text and "<按钮 重试>" in text and "GitHub" in text, "banner: failure + 重试 + GitHub 下载")

    # 8. real update: progress phases, files replaced, duplicates removed
    os.environ["WONDFUL_UPDATE_BASE_URLS"] = good_chain
    phases = []
    orig_set = UO._set

    def spy(**kw):
        if "message" in kw:
            phases.append(kw["message"])
        orig_set(**kw)
    UO._set = spy
    UO.STATE["phase"] = "AVAILABLE"
    UO.start_install(UO.STATE["manifest"])
    mid = banner("下载中（刚开始）")
    UO.wait(60)
    UO._set = orig_set
    print("phases:", " → ".join(dict.fromkeys(phases)))
    check(any(p.startswith("下载中") for p in phases) and "校验" in phases and "安装" in phases, "phases 下载中 / 校验 / 安装 reported")
    check(UO.STATE["phase"] == "DONE" and UO.STATE["message"] == "已更新到 9.9.9，请重启 Blender",
          f"done message: {UO.STATE['message']} {UO.STATE.get('notes')}")
    new_info = U.read_bl_info(target / "__init__.py")
    check(tuple(new_info.get("version", ())) == (9, 9, 9), "installed __init__.py is 9.9.9")
    check((target / "UPDATED_MARKER.txt").is_file(), "new files present in the loaded add-on dir")
    check(not dup1.exists() and not dup2.exists(), "duplicates removed")
    check(other.is_dir(), "unrelated add-on untouched")
    check(not [p for p in target.parent.iterdir() if p.name.startswith((U.STAGING_PREFIX, U.BACKUP_PREFIX))],
          "no staging/backup left after success")
    text = banner("完成")
    check("已更新到 9.9.9，请重启 Blender" in text and "已删除旧版本副本 2 个" in text, "banner: done + duplicates note")

    # 9. the new copy really loads (what a Blender restart does)
    bpy.ops.preferences.addon_disable(module="wondful_ai_renderer")
    for name in [n for n in sys.modules if n == "wondful_ai_renderer" or n.startswith("wondful_ai_renderer.")]:
        del sys.modules[name]
    bpy.ops.preferences.addon_enable(module="wondful_ai_renderer")
    import wondful_ai_renderer as reloaded
    check(tuple(reloaded.bl_info["version"]) == (9, 9, 9), f"re-enabled add-on reports {reloaded.bl_info['version']}")
    from wondful_ai_renderer import update_ops as UO2
    UO2.STATE.update(phase="IDLE", manifest=None)
    UO2._startup()
    UO2.wait(30)
    check(UO2.banner_manifest() is None, "after the update no banner (9.9.9 is current)")
    bpy.ops.preferences.addon_disable(module="wondful_ai_renderer")
    srv.shutdown()
    print("PROBLEMS", len(problems))
    if problems:
        sys.exit(1)
    print("E2E UPDATE OK")
    sys.stdout.flush()


try:
    main()
except SystemExit:
    raise
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.exit(0)
