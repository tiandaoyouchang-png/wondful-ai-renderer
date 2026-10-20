from __future__ import annotations

from array import array
import shutil
from pathlib import Path

import bpy



def image_filepath(image: bpy.types.Image | None) -> str | None:
    if not image:
        return None
    path = bpy.path.abspath(image.filepath_raw or image.filepath)
    if path and Path(path).exists():
        return path
    try:
        if image.packed_file:
            temp_dir = Path.home() / ".wondful_ai_renderer" / "packed_refs"
            temp_dir.mkdir(parents=True, exist_ok=True)
            ext = ".png"
            target = temp_dir / f"{image.name_full.replace('/', '_')}{ext}"
            image.save_render(str(target))
            return str(target)
    except Exception:
        pass
    return None


def load_image(filepath: str, name_prefix: str = "Wondful") -> bpy.types.Image:
    path = str(Path(filepath).resolve())
    existing = next((img for img in bpy.data.images if bpy.path.abspath(img.filepath_raw or img.filepath) == path), None)
    if existing:
        try:
            existing.reload()
        except Exception:
            pass
        return existing
    image = bpy.data.images.load(path, check_existing=True)
    image.name = f"{name_prefix}_{Path(path).stem}"
    return image


def compute_target_size(view_w: int, view_h: int, long_edge: int = 2048) -> tuple[int, int, str]:
    view_w = max(1, int(view_w))
    view_h = max(1, int(view_h))
    long_edge = max(1024, int(long_edge))
    ratio = view_w / view_h
    if ratio >= 1:
        width = long_edge
        height = int(round(long_edge / ratio))
    else:
        height = long_edge
        width = int(round(long_edge * ratio))
    width = max(8, int(round(width / 8.0) * 8))
    height = max(8, int(round(height / 8.0) * 8))
    return width, height, f"{width}x{height}"


def _resample_pixels(image, src_w: int, src_h: int, dst_w: int, dst_h: int):
    """Blender image pixels (RGBA float, bottom row first) resampled to dst size → flat float array."""
    import numpy as np
    from .layered_output import upscale
    buf = np.empty(src_w * src_h * 4, np.float32)
    image.pixels.foreach_get(buf)
    out = upscale(buf.reshape(src_h, src_w, 4), dst_w, dst_h)
    flat = array("f")
    flat.frombytes(np.ascontiguousarray(np.clip(out, 0.0, 1.0), dtype=np.float32).tobytes())
    return flat


def enforce_aspect_ratio(src: str, dst: str, target_w: int, target_h: int) -> tuple[str, bool]:
    """Fit a generated image into the target canvas without crop or distortion.

    The source is uniformly scaled until it fits inside ``target_w`` × ``target_h``;
    any remaining canvas is transparent padding. This also normalizes provider files
    that contain JPEG bytes but carry a ``.png`` name, so exported PNGs are real PNGs.
    """
    source_path = Path(src).expanduser().resolve()
    target_path = Path(dst).expanduser().resolve()
    target_w, target_h = max(1, int(target_w)), max(1, int(target_h))
    if not source_path.is_file():
        raise FileNotFoundError(f"渲染结果不存在: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    source_image = None
    canvas_image = None
    try:
        source_image = bpy.data.images.load(str(source_path), check_existing=False)
        source_w, source_h = map(int, source_image.size)
        if source_w < 1 or source_h < 1:
            raise ValueError(f"无法读取渲染结果尺寸: {source_path}")
        source_format = str(getattr(source_image, "file_format", "") or "").upper()
        scale = min(target_w / source_w, target_h / source_h)
        fit_w = max(1, min(target_w, int(round(source_w * scale))))
        fit_h = max(1, min(target_h, int(round(source_h * scale))))
        resized = (fit_w != source_w or fit_h != source_h)
        if resized:
            # 1.2.4: numpy Lanczos-3 when enlarging (was Blender's Image.scale), no sharpening
            source_pixels = _resample_pixels(source_image, source_w, source_h, fit_w, fit_h)
        else:
            source_pixels = array("f", [0.0]) * (fit_w * fit_h * 4)
            source_image.pixels.foreach_get(source_pixels)
        canvas_pixels = array("f", [0.0]) * (target_w * target_h * 4)
        offset_x = (target_w - fit_w) // 2
        offset_y = (target_h - fit_h) // 2
        source_stride = fit_w * 4
        canvas_stride = target_w * 4
        for row in range(fit_h):
            src_start = row * source_stride
            dst_start = (row + offset_y) * canvas_stride + offset_x * 4
            canvas_pixels[dst_start:dst_start + source_stride] = source_pixels[src_start:src_start + source_stride]

        canvas_image = bpy.data.images.new(
            f"Wondful_AspectFit_{target_path.stem}", width=target_w, height=target_h, alpha=True
        )
        try:
            canvas_image.colorspace_settings.name = source_image.colorspace_settings.name
        except Exception:
            pass
        canvas_image.pixels.foreach_set(canvas_pixels)
        canvas_image.filepath_raw = str(target_path)
        canvas_image.file_format = "PNG"
        canvas_image.save()
        return str(target_path), bool(resized or source_format != "PNG" or offset_x or offset_y)
    except Exception:
        # Keep the previous safe behavior if a local image backend cannot write a
        # fit canvas; the caller still receives the original bitmap for inspection.
        if source_path != target_path:
            shutil.copy2(source_path, target_path)
        return str(target_path), False
    finally:
        if canvas_image is not None:
            try:
                bpy.data.images.remove(canvas_image)
            except Exception:
                pass
        if source_image is not None:
            try:
                bpy.data.images.remove(source_image)
            except Exception:
                pass


def prepare_reference_for_upload(src: str, max_edge: int = 2560) -> str:
    """Downscale very large reference images inside the render session.

    Runs on Blender's main thread before network workers start. Small images are left untouched.
    """
    if not src or not Path(src).exists():
        return src
    try:
        import imbuf

        ib = imbuf.load(src)
        if not ib:
            return src
        width, height = int(ib.size[0]), int(ib.size[1])
        if max(width, height) <= max_edge:
            ib.free()
            return src
        scale = max_edge / float(max(width, height))
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        ib.resize((new_w, new_h), method="BILINEAR")
        try:
            ib.file_type = "PNG"
        except Exception:
            pass
        dst = str(Path(src).with_name(Path(src).stem + "_upload.png"))
        imbuf.write(ib, filepath=dst)
        ib.free()
        return dst if Path(dst).exists() else src
    except Exception:
        return src
