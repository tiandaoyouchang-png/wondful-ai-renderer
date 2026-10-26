"""参考图超限自动拼图 (user 2026-10-21: 「参考图超限时拼图而不是丢图」).

The Mac test (精确模式, 7 image inputs) stopped the second image with 「本轮图像输入 7 张，超过当前提交上限 5 张」.
When an image request carries more inputs than the tool accepts, ``fit_inputs`` keeps every image it can:

- the camera base (the image being edited), an edit base and an edit mask always stay standalone, never merged;
- the other inputs (product three-view, structure control maps, building references, detail crops …) are merged
  by priority into labelled sheets: each cell keeps its original aspect (no crop), a letter (A, B, C …) sits in
  a label strip above the cell (never on the image itself), and the prompt note says what every letter is; a sheet's long edge never exceeds
  the request size;
- only when the sheets still cannot fit (more than ``MAX_CELLS`` per sheet would be needed) inputs fall back to a
  text line, by priority: detail crops first, then structure maps;
- environment references never go to the image model (user rule 2026-10-10: environment only as text); when a
  request is over the limit an environment image is taken out instead of merged.

Pure numpy + ``png_io`` (safe on worker threads); JPEG inputs are decoded with Pillow when present, else converted
to a temporary PNG by Blender's ``imbuf`` (only when it really writes PNG; 4.x keeps the loaded JPEG format), macOS
``sips`` or a background Blender process (no bpy data access on the calling thread).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

MAX_CELLS = 6               # cells per sheet (2 x 3): beyond this a cell gets too small to read
SHEET_LONG_EDGE = 2048      # default long edge when the request size is unknown
GAP = 12
BG = 28                     # sheet background (dark grey, distinct from the white-model canvas)

# keep priority (lower = kept as a standalone image first / merged last)
PRIORITY = {"camera": 0, "candidate": 0, "edit_base": 0, "mask": 0, "product": 1, "building": 2, "person": 3,
            "reference": 4, "appearance": 4, "control_bundle": 4, "sheet": 4, "structure": 5, "detail": 6,
            "environment": 9}
# text fallback order when even the sheets do not fit
TEXT_ORDER = ("detail", "structure")
ROLE_TEXT = {
    "product": "产品三视图（只看产品造型与身份细节，不取其打光和背景）",
    "building": "环境体块的建筑参考（只管对应体块的建筑风格）",
    "person": "人物参考（只看人物身份）",
    "structure": "Blender 结构控制图（只管几何，不是外观；伪彩色不得进入结果）",
    "detail": "产品细节图（只看局部细节）",
    "reference": "参考图",
    "appearance": "外观参考图集",
    "control_bundle": "参考图集",
    "sheet": "参考图集",
}
TEXT_FALLBACK = {
    "structure": "结构控制图未作为图像发送：几何、位置、尺度与透视一律以 Blender 白模底图为准。",
    "detail": "细节图未作为图像发送：产品局部细节以白模底图与产品三视图为准，Logo、字标、灯组形状不得改变。",
    "environment": "环境参考图不作为图像发送（环境只以文字描述进入生图）：环境、光线与氛围按上文文字要求。",
}

_GLYPHS = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10011", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}
LETTERS = tuple(_GLYPHS)


class CollageError(RuntimeError):
    pass


# ============================================================================== image helpers

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_png(path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(8) == PNG_MAGIC
    except OSError:
        return False


def _imbuf_to_png(src: str, dst: str) -> bool:
    """Blender's imbuf: decode anything Blender reads. In 4.x ``imbuf.write`` keeps the LOADED format (a JPEG stays
    a JPEG whatever the file name; ImBuf has no ``file_type`` there), so the result counts only when it is a PNG."""
    import imbuf
    ib = imbuf.load(src)
    try:
        try:
            ib.file_type = "PNG"         # newer Blender
        except Exception:
            pass
        imbuf.write(ib, filepath=dst)
    finally:
        try:
            ib.free()
        except Exception:
            pass
    return _is_png(dst)


def _sips_to_png(src: str, dst: str) -> bool:
    """macOS built-in converter (Mac Blender ships without Pillow)."""
    import shutil
    import subprocess
    import sys
    if sys.platform != "darwin":
        return False
    sips = shutil.which("sips") or "/usr/bin/sips"
    if not Path(sips).is_file():
        return False
    subprocess.run([sips, "-s", "format", "png", src, "--out", dst], capture_output=True, timeout=60)
    return _is_png(dst)


# a loaded JPEG keeps saving as JPEG even with file_format = 'PNG' (Blender 4.3), so the pixels go into a new
# image that is saved as PNG (byte sRGB → byte sRGB, exact)
_BLENDER_CONVERT = ("import bpy, sys, numpy as np\n"
                    "src, dst = sys.argv[sys.argv.index('--') + 1:][:2]\n"
                    "im = bpy.data.images.load(src)\n"
                    "w, h = im.size\n"
                    "px = np.empty(w * h * 4, np.float32)\n"
                    "im.pixels.foreach_get(px)\n"
                    "out = bpy.data.images.new('wondful_png', w, h, alpha=True)\n"
                    "out.pixels.foreach_set(px)\n"
                    "out.filepath_raw = dst\n"
                    "out.file_format = 'PNG'\n"
                    "out.save()\n")


def _blender_to_png(src: str, dst: str) -> bool:
    """Last resort on any OS: a background Blender (the running binary) loads the image and saves it as PNG.
    A separate process, so it is safe from a worker thread (no bpy data access here)."""
    import subprocess
    try:
        import bpy
        exe = bpy.app.binary_path
    except Exception:
        return False
    if not exe:
        return False
    subprocess.run([exe, "-b", "--factory-startup", "--python-expr", _BLENDER_CONVERT, "--", src, dst],
                   capture_output=True, timeout=120)
    return _is_png(dst)


# PNG converters tried in order when Pillow is missing (tests replace entries)
CONVERTERS = [_imbuf_to_png, _sips_to_png, _blender_to_png]


def read_rgb(path) -> np.ndarray:
    """(H, W, 3) uint8 of a PNG / JPEG / WebP, on any thread.

    PNG → own decoder; else Pillow; without Pillow (Mac Blender) the file is converted to a temporary PNG by
    imbuf (only when it really writes PNG), macOS ``sips`` or a background Blender, then decoded."""
    p = str(path)
    from .png_io import read_png
    try:
        a = read_png(p)
        return _flatten(a)
    except Exception:
        pass
    try:
        from PIL import Image
        with Image.open(p) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    except Exception:
        pass
    import shutil
    import tempfile
    tmpd = Path(tempfile.mkdtemp(prefix="wondful_collage_"))
    errors = []
    try:
        for i, conv in enumerate(CONVERTERS):
            dst = tmpd / f"x{i}.png"
            try:
                if conv(p, str(dst)):
                    return _flatten(read_png(str(dst)))
                errors.append(f"{getattr(conv, '__name__', conv)}: no PNG")
            except Exception as exc:
                errors.append(f"{getattr(conv, '__name__', conv)}: {str(exc)[:60]}")
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    raise CollageError(f"无法读取参考图 {Path(p).name}：" + "；".join(errors))


def _flatten(a) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    if a.shape[-1] == 4:
        al = a[..., 3:4].astype(np.float32) / 255.0
        rgb = a[..., :3].astype(np.float32) * al + 255.0 * (1 - al)     # transparent → white
        return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
    return a[..., :3].astype(np.uint8)


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Area-averaged downscale + bilinear (aspect is the caller's job)."""
    img = np.asarray(img, np.float32)
    H, W = img.shape[:2]
    w, h = max(1, int(w)), max(1, int(h))
    k = int(min(H / h, W / w))
    if k >= 2:      # box prefilter against aliasing
        Hc, Wc = (H // k) * k, (W // k) * k
        img = img[:Hc, :Wc].reshape(Hc // k, k, Wc // k, k, -1).mean((1, 3))
        H, W = img.shape[:2]
    ys = (np.arange(h) + 0.5) * H / h - 0.5
    xs = (np.arange(w) + 0.5) * W / w - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, H - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    fy = np.clip(ys - y0, 0, 1)[:, None, None]
    fx = np.clip(xs - x0, 0, 1)[None, :, None]
    top = img[y0][:, x0] * (1 - fx) + img[y0][:, x1] * fx
    bot = img[y1][:, x0] * (1 - fx) + img[y1][:, x1] * fx
    return np.clip(top * (1 - fy) + bot * fy + 0.5, 0, 255).astype(np.uint8)


def draw_letter(sheet: np.ndarray, x: int, y: int, letter: str, scale: int) -> None:
    """White letter on a black tab at (x, y) (in the label strip above a cell), in place."""
    glyph = _GLYPHS.get(letter.upper())
    if glyph is None:
        return
    pad = max(2, scale)
    tw, th = 5 * scale + 2 * pad, 7 * scale + 2 * pad
    H, W = sheet.shape[:2]
    sheet[y:min(H, y + th), x:min(W, x + tw)] = 0
    for gy, row in enumerate(glyph):
        for gx, bit in enumerate(row):
            if bit == "1":
                yy, xx = y + pad + gy * scale, x + pad + gx * scale
                sheet[yy:min(H, yy + scale), xx:min(W, xx + scale)] = 255


def letter_name(i: int) -> str:
    return LETTERS[i] if i < len(LETTERS) else LETTERS[i % len(LETTERS)] + str(i // len(LETTERS) + 1)


def label_scale(long_edge=SHEET_LONG_EDGE) -> int:
    """Glyph pixel size of the letter tabs (one size per sheet)."""
    return int(np.clip(int(long_edge) // 320, 3, 8))


def label_height(long_edge=SHEET_LONG_EDGE) -> int:
    """Height of the label strip above every row of cells (tab + 4 px air)."""
    sc = label_scale(long_edge)
    return 7 * sc + 2 * max(2, sc) + 4


def _layout(sizes, long_edge=SHEET_LONG_EDGE):
    """Full grid: (sheet_w, sheet_h, cells [(x, y, w, h)], tabs [(x, y)]). Every row has a label strip ABOVE its
    cells (user 2026-10-21: a nested structure sheet's own R numbers were hidden under the B tab), so a tab
    never covers any pixel of the image it names."""
    n = len(sizes)
    if n == 0:
        raise CollageError("empty sheet")
    LB = label_height(long_edge)
    ar = float(np.median([w / max(1, h) for w, h in sizes]))
    best = None
    for cols in range(1, n + 1):
        rows = int(math.ceil(n / cols))
        # cell aspect = median image aspect → square-ish sheets for landscape inputs
        avail_w = long_edge - GAP * (cols + 1)
        avail_h = long_edge - GAP * (rows + 1) - rows * LB
        if avail_w <= 0 or avail_h <= 0:
            continue
        scale = min(avail_w / (cols * ar), avail_h / rows)
        cw, ch = ar * scale, 1.0 * scale
        # area each image actually gets inside its cell
        used = sum(min(cw / w, ch / h) ** 2 * w * h for w, h in sizes)
        if best is None or used > best[0]:
            best = (used, cols, rows, cw, ch)
    if best is None:
        raise CollageError("sheet too small for its cells")
    _u, cols, rows, cw, ch = best
    cw, ch = max(1, int(cw)), max(1, int(ch))
    sheet_w = cols * cw + GAP * (cols + 1)
    sheet_h = rows * (ch + LB) + GAP * (rows + 1)
    cells, tabs = [], []
    for i, (w, h) in enumerate(sizes):
        r, c = divmod(i, cols)
        s = min(cw / w, ch / h)
        tw, th = max(1, int(w * s)), max(1, int(h * s))
        slot_x = GAP + c * (cw + GAP)
        slot_y = GAP + r * (ch + LB + GAP)
        x = slot_x + (cw - tw) // 2
        y = slot_y + LB + (ch - th) // 2
        cells.append((x, y, tw, th))
        tabs.append((x, max(slot_y, y - LB)))      # sits right above the image, inside the strip
    return sheet_w, sheet_h, cells, tabs


def layout(sizes, long_edge=SHEET_LONG_EDGE):
    """Grid for cells of ``sizes`` [(w, h)]: (sheet_w, sheet_h, [(x, y, w, h) per cell]); long edge ≤ long_edge,
    every image scaled to fit its cell with its own aspect (no crop); labels live in a strip above each row."""
    sw, sh, cells, _tabs = _layout(sizes, long_edge)
    return sw, sh, cells


def compose(images, letters, long_edge=SHEET_LONG_EDGE) -> np.ndarray:
    sizes = [(im.shape[1], im.shape[0]) for im in images]
    sw, sh, cells, tabs = _layout(sizes, long_edge)
    sheet = np.full((sh, sw, 3), BG, np.uint8)
    scale = label_scale(long_edge)
    for im, (x, y, w, h), (tx, ty), lt in zip(images, cells, tabs, letters):
        sheet[y:y + h, x:x + w] = resize(im, w, h)
        draw_letter(sheet, tx, ty, lt[0], scale)
    return sheet


# ============================================================================== roles

def guess_role(path: str) -> str:
    n = Path(str(path)).name.lower()
    if n.startswith("reference_sheet_"):
        if "structure" in n:
            return "structure"
        return "sheet"
    if n.startswith("building_ref") or "building_ref" in n:
        return "building"
    if any(k in n for k in ("detail_crop", "detail_ref", "_detail_", "crop_ref")):
        return "detail"
    if any(k in n for k in ("depth", "normal", "silhouette", "part_id", "partid", "structure", "product_mask")):
        return "structure"
    if "mask" in n:
        return "mask"
    return "reference"


def size_long_edge(size) -> int:
    """'1536x1024' → 1536; int passes through; unknown → SHEET_LONG_EDGE."""
    if isinstance(size, (int, float)) and size > 0:
        return int(size)
    try:
        w, h = str(size).lower().split("x")
        return max(int(w), int(h))
    except Exception:
        return SHEET_LONG_EDGE


# ============================================================================== the fit

def plan(items, limit, fixed_count):
    """items = [(path, role)] (non-fixed, in request order). → (standalone, sheets [[…]], text_only).

    Standalone = the highest-priority items; the rest go into the fewest sheets that hold them (≤ MAX_CELLS
    each); environment is text only; detail → structure fall back to text when nothing else fits."""
    free = int(limit) - int(fixed_count)
    env = [it for it in items if it[1] == "environment"]
    rest = [it for it in items if it[1] != "environment"]
    text = list(env)
    if free <= 0:
        return [], [], text + rest
    if len(rest) <= free:
        return rest, [], text
    order = sorted(range(len(rest)), key=lambda i: (PRIORITY.get(rest[i][1], 4), i))
    ranked = [rest[i] for i in order]
    while ranked and len(ranked) > free * MAX_CELLS:
        drop = None
        for role in TEXT_ORDER:
            cand = [it for it in ranked if it[1] == role]
            if cand:
                drop = cand[-1]
                break
        if drop is None:
            drop = ranked[-1]
        ranked.remove(drop)
        text.append(drop)
    if len(ranked) <= free:
        return ranked, [], text
    for k in range(1, free + 1):
        s = free - k
        if len(ranked) - s <= k * MAX_CELLS:
            break
    standalone = ranked[:s]
    merge = ranked[s:]
    per = int(math.ceil(len(merge) / k))
    sheets = [merge[i:i + per] for i in range(0, len(merge), per)]
    return standalone, sheets, text


def fit_inputs(paths, limit, *, out_dir, roles=None, labels=None, fixed=(), long_edge=SHEET_LONG_EDGE, tag="",
               reader=read_rgb):
    """Fit ``paths`` (request order, first = camera base) into ``limit`` image inputs.

    ``roles`` / ``labels``: path → role / human label (else guessed from the file name); ``fixed``: paths that
    stay standalone (camera base, edit base, mask). Returns ``(new_paths, note, info)``; under the limit the
    paths come back unchanged with an empty note (info["merged"] = 0)."""
    roles = {str(Path(k).resolve()): v for k, v in (roles or {}).items()}
    labels = {str(Path(k).resolve()): v for k, v in (labels or {}).items()}
    seen, ordered = set(), []
    for p in paths:
        if not p:
            continue
        rp = str(Path(p).resolve())
        if rp not in seen:
            seen.add(rp)
            ordered.append(rp)
    info = {"before": len(ordered), "after": len(ordered), "limit": int(limit), "merged": 0, "sheets": [],
            "text_only": [], "standalone": []}
    if len(ordered) <= int(limit):
        return ordered, "", info
    fixed_set = [str(Path(p).resolve()) for p in fixed if p]
    if ordered and ordered[0] not in fixed_set:
        fixed_set.insert(0, ordered[0])           # the camera base is never merged
    fixed_list = [p for p in ordered if p in fixed_set]
    if len(fixed_list) > int(limit):
        raise CollageError(f"必须单独发送的图像 {len(fixed_list)} 张已超过上限 {limit} 张")
    items = [(p, roles.get(p) or guess_role(p)) for p in ordered if p not in fixed_set]
    standalone, sheets, text = plan(items, limit, len(fixed_list))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheet_paths, lines, letter_i = {}, [], 0
    for si, group in enumerate(sheets, 1):
        imgs, letters, cells = [], [], []
        for p, role in group:
            try:
                im = reader(p)
            except Exception as exc:
                text.append((p, role))
                info.setdefault("unreadable", []).append(f"{Path(p).name}: {str(exc)[:80]}")
                continue
            lt = letter_name(letter_i)
            letter_i += 1
            imgs.append(im)
            letters.append(lt)
            cells.append({"letter": lt, "role": role, "file": Path(p).name,
                          "label": labels.get(p) or ROLE_TEXT.get(role, ROLE_TEXT["reference"]),
                          "size": [int(im.shape[1]), int(im.shape[0])]})
        if not imgs:
            continue
        if len(imgs) == 1:          # a lone survivor goes as itself
            only = [p for p, _r in group if Path(p).name == cells[0]["file"]][0]
            standalone.append((only, cells[0]["role"]))
            continue
        sheet = compose(imgs, letters, long_edge)
        name = f"reference_collage_{tag + '_' if tag else ''}{si:02d}.png"
        sp = out / name
        from .png_io import write_png
        write_png(str(sp), sheet)
        sheet_paths[si] = str(sp)
        info["sheets"].append({"path": str(sp), "size": [int(sheet.shape[1]), int(sheet.shape[0])], "cells": cells})
        info["merged"] += len(cells)
    keep = set(fixed_list) | {p for p, _r in standalone}
    new_paths = [p for p in ordered if p in keep]
    for si in sorted(sheet_paths):
        new_paths.append(sheet_paths[si])
    info["after"] = len(new_paths)
    info["standalone"] = [Path(p).name for p, _r in standalone]
    info["text_only"] = [{"file": Path(p).name, "role": r} for p, r in text]
    if len(new_paths) > int(limit):
        raise CollageError(f"拼图后仍有 {len(new_paths)} 张，超过上限 {limit} 张")
    # prompt note
    idx = {p: i for i, p in enumerate(new_paths, 1)}
    if info["merged"]:
        lines.append(f"【参考图拼图】本轮图像输入原有 {info['before']} 张，超过上限 {limit} 张；"
                     f"已把 {info['merged']} 张参考图拼合为 {len(info['sheets'])} 张拼图，相机底图单独发送、未拼合。")
        for sh in info["sheets"]:
            cells = "；".join(f"{c['letter']} = {c['label']}" for c in sh["cells"])
            lines.append(f"输入图 {idx.get(sh['path'], '?')}（拼图）：{cells}。")
        lines.append("拼图只供逐格读取：每格上方标签条里的字母只是编号（字母不在图内）；各格互相独立、比例未变。"
                     "不要在结果里画出拼版、边框、格子或字母，也不要把拼图当作构图。")
    seen_txt = set()
    for p, role in text:
        t = TEXT_FALLBACK.get(role)
        if t and t not in seen_txt:
            seen_txt.add(t)
            lines.append(t)
    note = "\n".join(lines)
    return new_paths, note, info


def panel_text(info) -> str:
    """'已拼合 N 张参考图' for the panel / console ('' when nothing was merged)."""
    if not info or not info.get("merged"):
        return ""
    s = f"已拼合 {info['merged']} 张参考图"
    if info.get("text_only"):
        s += f"（{len(info['text_only'])} 张改为文字描述）"
    return s
