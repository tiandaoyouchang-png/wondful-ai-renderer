from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import bpy


def _safe_component(value: str, fallback: str) -> str:
    text = (value or "").strip() or fallback
    text = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", text)
    text = re.sub(r"\s+", "_", text).strip("._ ")
    return text[:80] or fallback


def default_output_directory() -> Path:
    pictures = Path.home() / "Pictures"
    base = pictures if pictures.exists() else Path.home()
    return base / "Wondful_AI_Renderer"


def resolve_output_directory(context, configured: str) -> Path:
    raw = (configured or "").strip()
    if raw:
        try:
            resolved = Path(bpy.path.abspath(raw)).expanduser()
        except Exception:
            resolved = Path(raw).expanduser()
    else:
        # For saved .blend files, keep project-local output by default; otherwise
        # fall back to the user's Pictures folder.
        blend_path = getattr(bpy.data, "filepath", "") or ""
        if blend_path:
            resolved = Path(blend_path).parent / "Wondful_Renders"
        else:
            resolved = default_output_directory()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


# 1.2.3: the output folder holds only the images; everything else (acceptance JSON, raw AI images, 多角度 pools,
# logs) lives in this hidden folder inside it (dot-folder; also marked hidden on Windows), so 打开文件夹 stays clean
# while 诊断 / 多角度 reuse still find their files next to the project.
INTERNAL_DIR_NAME = ".wondful"


def _hide_on_windows(path: Path) -> None:
    import os
    if os.name != "nt":
        return
    try:
        import ctypes
        FILE_ATTRIBUTE_HIDDEN = 0x02
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs != -1 and not attrs & FILE_ATTRIBUTE_HIDDEN:
            ctypes.windll.kernel32.SetFileAttributesW(str(path), attrs | FILE_ATTRIBUTE_HIDDEN)
    except Exception:
        pass


def internal_dir(output_dir, *parts) -> Path:
    """``<output>/.wondful/<parts…>`` (created; the .wondful folder hidden on Windows)."""
    root = Path(output_dir) / INTERNAL_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    _hide_on_windows(root)
    target = root.joinpath(*[str(p) for p in parts]) if parts else root
    target.mkdir(parents=True, exist_ok=True)
    return target


def sidecar_path(exported, suffix: str) -> Path:
    """Where a file that belongs to an exported image goes (``<stem><suffix>``), e.g. ``_iou.json`` / ``_raw.png``:
    ``<output>/.wondful/sidecars/``, never next to the image."""
    p = Path(exported)
    return internal_dir(p.parent, "sidecars") / (p.stem + suffix)   # detail_paste.raw_path mirrors this (no bpy)


def export_result(source_path: str, output_dir: Path, scene_name: str, camera_name: str, session_id: str) -> Path:
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"渲染结果不存在: {src}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = _safe_component(scene_name, "Scene")
    camera = _safe_component(camera_name, "Camera")
    session = _safe_component(session_id, "render")
    suffix = src.suffix.lower() if src.suffix else ".png"
    base = output_dir / f"Wondful_{scene}_{camera}_{session}{suffix}"
    target = base
    counter = 2
    while target.exists():
        target = output_dir / f"{base.stem}_{counter:02d}{base.suffix}"
        counter += 1
    shutil.copy2(src, target)
    return target


def preflight_output_directory(directory: Path) -> None:
    """Test write permission before spending a remote generation request."""
    with tempfile.TemporaryFile(prefix=".wondful-write-check-", dir=str(directory)) as stream:
        stream.write(b"wondful")
        stream.flush()
