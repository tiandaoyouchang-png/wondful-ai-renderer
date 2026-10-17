"""1.2.3 结果处理（后台线程）— everything numpy that happens to a finished AI image.

Runs on the render worker thread (never on Blender's main thread) and never imports or touches
``bpy``: fit to the camera canvas, the fixed-coordinate IoU acceptance (``contour_iou``), the
细节回贴 paste and the 产品还原度 scores. It only reads and writes files in the session
directory and returns plain dicts; the main thread copies the finished file to the output
folder and writes the panel notes (``apply`` in operators).

Cancel-safe: every step checks ``cancel_check`` first and raises ``Cancelled``; partially
written files stay inside the session directory only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

GATE_MAX = 3                 # user decision 2026-10-17: at most 3 generations per image
VERDICT_ORDER = {"通过": 2, "待复核": 1, "未通过": 0}


class Cancelled(RuntimeError):
    pass


def _check(cancel_check):
    if cancel_check is not None and cancel_check():
        raise Cancelled("任务已取消")


def fit_to_canvas(src, dst, width: int, height: int) -> bool:
    """Pure-numpy twin of ``image_manager.enforce_aspect_ratio``: uniform scale to fit inside
    width×height, centred, transparent padding, written as RGBA PNG. False if the source is
    not a PNG this reader understands (the main thread then uses Blender's loader)."""
    from .png_io import read_png, write_png
    from .layered_output import resize
    try:
        rgba = read_png(src)
    except Exception:
        return False
    sh, sw = rgba.shape[:2]
    width, height = max(1, int(width)), max(1, int(height))
    scale = min(width / sw, height / sh)
    fw = max(1, min(width, int(round(sw * scale))))
    fh = max(1, min(height, int(round(sh * scale))))
    img = rgba.astype(np.float32)
    if (fw, fh) != (sw, sh):
        img = resize(img, fw, fh)
    canvas = np.zeros((height, width, 4), np.float32)
    ox, oy = (width - fw) // 2, (height - fh) // 2
    canvas[oy:oy + fh, ox:ox + fw] = img
    write_png(dst, np.clip(np.rint(canvas), 0, 255).astype(np.uint8))
    return True


def gate_score(candidate, mask_path, width, height, work_dir, tag, cancel_check=None, detail_path="") -> dict:
    """Fixed-coordinate IoU of one raw candidate (resampled to the canvas, never fitted)."""
    from . import contour_iou
    _check(cancel_check)
    t0 = time.time()
    canvas = Path(work_dir) / f"gate_{tag}.png"
    if not fit_to_canvas(candidate, canvas, width, height):
        return {"error": "结果图格式无法读取（不是 PNG）", "verdict": contour_iou.VERDICT_REVIEW, "passed": False,
                "iou": None, "readable": False}
    _check(cancel_check)
    rep = contour_iou.score_file(str(canvas), mask_path,
                                 detail_path=detail_path if detail_path and Path(detail_path).is_file() else "")
    rep["readable"] = True
    rep["candidate"] = str(candidate)
    rep["seconds"] = round(time.time() - t0, 2)
    try:
        canvas.unlink()
    except OSError:
        pass
    return rep


def gate_rank(rep) -> tuple:
    """Pass > review > fail, then IoU (higher first)."""
    if not rep:
        return (-1, -1.0)
    iou = rep.get("iou")
    return (VERDICT_ORDER.get(rep.get("verdict"), -1), float(iou) if iou is not None else -1.0)


def gate_note(gate: dict) -> str:
    """Panel line for one variant's acceptance."""
    from .contour_iou import PASS_IOU
    if not gate or not gate.get("tries"):
        return ""
    best = gate.get("best") or {}
    iou = best.get("iou")
    n = len(gate["tries"])
    if iou is None:
        return f"轮廓 IoU：无法计算（待复核）· 生成 {n} 次"
    v = best.get("verdict", "")
    if v == "通过":
        return f"轮廓 IoU {iou:.3f} · 通过（≥{PASS_IOU:.2f}，固定坐标）· 第 {n} 次生成"
    if v == "待复核":
        return (f"轮廓 IoU {iou:.3f} · 待复核（轮廓不确定，范围 {best.get('iou_lo', 0):.3f}–{best.get('iou_hi', 0):.3f}）"
                f" · 生成 {n} 次" + ("，已用完" if gate.get("exhausted") else ""))
    return (f"轮廓 IoU {iou:.3f} · 未通过（{n} 次生成均未达 {PASS_IOU:.2f}，显示 IoU 最高的一张；"
            "其余候选已保存）" if gate.get("exhausted") else f"轮廓 IoU {iou:.3f} · 未通过")


# --------------------------------------------------------------------------- 细节回贴 + 产品还原度

def identity_scores(image, spec, matid, raw=None):
    """(report, panel note) — raw AI image (raw=None) or the pasted one (raw given)."""
    from .identity_score import score_file, summarize
    from .feature_lines import geometry_path_for
    geo = geometry_path_for(spec["detail"])
    rep = score_file(str(image), spec["detail"], spec["mask"], matid, placement_of=raw,
                     geometry_path=geo if Path(geo).is_file() else "")
    if raw is None:
        return rep, (summarize(rep) if not rep.get("error") else "")
    return rep, (summarize(raw, rep) if raw and not raw.get("error") else "")


def paste_and_score(variant_final, variant_index, spec, cancel_check=None) -> dict:
    """细节回贴 + 产品还原度 for one canvas-sized result. Returns dict(final, raw, detail_note,
    score_note, reports). Never raises except ``Cancelled``."""
    out = {"final": str(variant_final), "raw": None, "detail_note": "", "score_note": ""}
    if not spec or not Path(spec["detail"]).is_file() or not Path(spec["mask"]).is_file():
        return out
    try:
        from .detail_paste import mirrored_note, paste_file, summarize
        _check(cancel_check)
        target = Path(spec["work_dir"]) / f"output_variant_{variant_index:02d}_detail.png"
        matid = spec.get("matid", "") if spec.get("matid") and Path(spec["matid"]).is_file() else ""
        raw_rep, note = identity_scores(variant_final, spec, matid)
        out["score_note"] = note
        _check(cancel_check)
        rep = paste_file(str(variant_final), spec["detail"], spec["mask"], str(target),
                         keep_path=spec["keep"] if Path(spec["keep"]).is_file() else "",
                         strength=spec["strength"], matid_path=matid,
                         colour_strength=float(spec.get("colour_strength", 0.0) or 0.0))
        out["detail_note"] = summarize(rep) + mirrored_note(spec.get("mirrored"))
        out["paste_report"] = {k: v for k, v in rep.items() if k not in ("image", "weight", "keep_mask", "occluder")}
        if rep.get("skipped"):
            return out
        _check(cancel_check)
        final_rep, note = identity_scores(target, spec, matid, raw=raw_rep)
        out["score_note"] = note
        Path(spec["work_dir"], f"score_{variant_index:02d}.json").write_text(
            json.dumps({"raw": raw_rep, "final": final_rep}, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
        out["final"], out["raw"] = str(target), str(variant_final)
    except Cancelled:
        raise
    except Exception as exc:
        print(f"[Wondful] 细节回贴失败：{exc}")
        out["detail_note"] = f"细节回贴失败（已保留原图）：{str(exc)[:80]}"
    return out


def finish_variant(source, variant_index, spec, width, height, session_dir, cancel_check=None) -> dict | None:
    """Whole numpy result path of one variant: canvas fit + paste + scores. None when the file
    cannot be read here (the main thread then runs the Blender-based path)."""
    _check(cancel_check)
    variant_final = Path(session_dir) / f"output_variant_{variant_index:02d}.png"
    if not fit_to_canvas(source, variant_final, width, height):
        return None
    res = paste_and_score(variant_final, variant_index, spec, cancel_check)
    res["canvas"] = str(variant_final)
    return res
