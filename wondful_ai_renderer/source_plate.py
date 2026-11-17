"""Clean plate for the single-source edit (2026-11 review).

The placed-product canvas composites the user's car over the photograph. Wherever the
photograph's own vehicle is larger than the new one (a GMC SUV behind a YU7: a taller hood, a
longer tail), its remnants stay visible around the new silhouette, and the image model fuses
them into the new car — the YU7 test grew a black "hood vent" exactly where the GMC's hood
stood above the YU7's hood line.

So before compositing, the old vehicle is removed with one masked image edit of the original
photograph. The mask is generous (the new product dilated by a fraction of its size plus the
ground band below it), because the old vehicle's extent is unknown; outside the mask every pixel
is restored from the photograph, so the environment is unchanged byte-for-byte except near the
car. The plate is cached per photograph + mask, so repeated generations pay for it once.

Worker-thread safe: numpy + the provider call only, no bpy.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

PLATE_VERSION = 1
GROW = 0.30            # editable margin around the new product, × its bbox height
GROUND = 0.30          # ground band below it, × bbox height
FEATHER = 6            # px seam between the edited plate and the untouched photograph

PLATE_PROMPT = (
    "【去车空镜】图 1 是一张照片。把透明遮罩区域内的汽车完全移除，包括车身、车轮、车窗、车底阴影、车身倒影，"
    "以及与这辆车相连的尘土和水花；按遮罩外的环境自然补全被遮挡的地面、路面、远景和天空，"
    "让它看起来像同一机位、同一光线、同一镜头效果下没有车时拍的空镜。"
    "保持原有曝光、色调、颗粒、运动模糊和景深。不要放入任何车辆、人物或新物体。只输出一张图。"
)


def cache_dir() -> Path:
    base = os.environ.get("WONDFUL_PLATE_CACHE") or str(Path.home() / ".wondful_ai_renderer" / "plates")
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def edit_region(product_alpha: np.ndarray, grow: float = GROW, ground: float = GROUND) -> np.ndarray:
    """Editable region: the new product dilated by ``grow`` × its height, plus a ground band."""
    from .layered_output import bbox, dilate
    hard = product_alpha > 0.5
    b = bbox(hard)
    if b is None:
        raise ValueError("产品遮罩为空，无法生成去车空镜。")
    h, w = hard.shape
    y0, y1, x0, x1 = b
    bh, bw = y1 - y0 + 1, x1 - x0 + 1
    region = dilate(hard, max(4, int(round(grow * bh))))
    gy0 = y0 + int(0.5 * bh)
    gy1 = min(h, y1 + 1 + int(round(ground * bh)))
    gx0 = max(0, x0 - int(round(0.2 * bw)))
    gx1 = min(w, x1 + 1 + int(round(0.2 * bw)))
    region[gy0:gy1, gx0:gx1] = True
    return region


def plate_key(source_sha: str, region: np.ndarray) -> str:
    rb = np.packbits(region.astype(np.uint8)).tobytes()
    return hashlib.sha256((f"v{PLATE_VERSION}:{source_sha}:".encode() + hashlib.sha256(rb).digest())).hexdigest()[:24]


def _resize_exact(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    from .layered_output import resize
    if rgb.shape[:2] == (h, w):
        return rgb
    return np.clip(resize(rgb.astype(np.float32), w, h), 0, 255)


def restore_outside(source_rgb: np.ndarray, edited_rgb: np.ndarray, region: np.ndarray,
                    feather: int = FEATHER) -> np.ndarray:
    """Edited pixels inside ``region`` (feathered inward), the photograph everywhere else."""
    from .layered_output import blur
    soft = np.clip(blur(region.astype(np.float32), feather), 0, 1) * region
    out = source_rgb.astype(np.float32) * (1 - soft[..., None]) + edited_rgb.astype(np.float32) * soft[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def make_plate(source_path, product_alpha, work_dir, edit_fn, *, cancel_check=None, log=None, extra_region=None) -> dict:
    """Return {"path", "cached", "key", ...}. ``edit_fn(base_png, mask_png, out_png)`` performs one
    masked image edit with ``PLATE_PROMPT`` (provider-specific; raises on failure)."""
    from .png_io import read_png, write_png
    from .reference_collage import read_rgb
    from .layered_output import edit_mask_rgba
    src = read_rgb(source_path)
    h, w = src.shape[:2]
    alpha = np.asarray(product_alpha, np.float32)
    if alpha.max() > 1.0:
        alpha = alpha / 255.0
    if alpha.shape != (h, w):
        from .layered_output import resize
        alpha = np.clip(resize(alpha[..., None], w, h)[..., 0], 0, 1)
    region = edit_region(alpha)
    if extra_region is not None:
        # 2026-11: the photograph's own vehicle box (vehicle_locate), so a larger old vehicle is
        # removed whole, not just where the new product overlaps it
        extra = np.asarray(extra_region, bool)
        if extra.shape == region.shape:
            region = region | extra
    sha = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    key = plate_key(sha, region)
    cached = cache_dir() / f"{key}.png"
    info = {"key": key, "version": PLATE_VERSION, "region_frac": round(float(region.mean()), 4)}
    if cached.is_file():
        try:
            if read_png(cached).shape[:2] == (h, w):
                return {**info, "path": str(cached), "cached": True}
        except Exception:
            pass
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    base = work / "plate_base.png"
    write_png(base, src)
    mask = work / "plate_editmask.png"
    write_png(mask, edit_mask_rgba(region))
    raw = work / "plate_raw.png"
    if cancel_check and cancel_check():
        raise RuntimeError("任务已取消")
    produced = edit_fn(str(base), str(mask), str(raw)) or str(raw)
    edited = read_rgb(produced)
    edited = _resize_exact(edited, w, h)
    plate = restore_outside(src, edited, region)
    write_png(cached, plate)
    (cache_dir() / f"{key}.json").write_text(json.dumps({**info, "source_sha256": sha}, indent=1), encoding="utf-8")
    return {**info, "path": str(cached), "cached": False, "raw": str(produced)}


def codex_edit_fn(*, model="", client_version="", long_edge=2048, timeout=600, log_dir="", cancel_check=None):
    """``edit_fn`` for ``make_plate`` on the Codex direct image-edit endpoint."""
    def fn(base, mask, out):
        from . import codex_direct as CD
        from .composition import image_dimensions
        w, h = image_dimensions(base)
        r = CD.edit_image(prompt=PLATE_PROMPT, edit_base_path=base, reference_paths=[], output_path=out,
                          size=CD.choose_edit_size(w, h, int(long_edge)), mask_path=mask, model=model,
                          client_version=client_version, timeout=timeout, log_dir=log_dir or str(Path(out).parent),
                          tag="source_plate", cancel_check=cancel_check, exact_prompt=True)
        return (r or {}).get("source") or out
    return fn
