"""1.2.3 从参考图生成场景粗模 — stage progress (no bpy, unit-tested).

The operator walks a fixed list of stages; the panel shows the list, the current stage with the elapsed
time (「分析参考图 · 已用 1:42」), a typical-time hint during the long AI stages and a bar.

The bar is deterministic: each stage owns a fixed slice of 0..1 (weights below).  Only the two AI stages
creep inside their slice with elapsed time (like the main 「生成」 flow's estimate), capped below the next
stage's start; every other stage sits at its slice start until it completes.
"""
from __future__ import annotations

READ, ANALYZE, BUILD, CAMERA = "READ", "ANALYZE", "BUILD", "CAMERA"
INSTALL, DEPTH = "INSTALL", "DEPTH"          # 1.2.5 精确模式: first-use download, local depth measurement
PREVIEW, SCORE, CORRECT, REBUILD, DONE = "PREVIEW", "SCORE", "CORRECT", "REBUILD", "DONE"

LABELS = {
    READ: "读取参考图",
    INSTALL: "安装精确模式",
    DEPTH: "测量深度",
    ANALYZE: "分析参考图",
    BUILD: "搭建粗模",
    CAMERA: "设置相机",
    PREVIEW: "预览渲染",
    SCORE: "评分",
    CORRECT: "校正分析",
    REBUILD: "重建",
    DONE: "完成",
}
# relative share of the bar per stage (DONE has none: it is the end point)
WEIGHTS = {READ: 2, INSTALL: 12, DEPTH: 4, ANALYZE: 60, BUILD: 8, CAMERA: 2, PREVIEW: 5, SCORE: 2, CORRECT: 30, REBUILD: 6, DONE: 0}
# AI stages: typical seconds (bar creep) and the hint shown while they run
AI_TYPICAL = {ANALYZE: 240.0, CORRECT: 120.0}
AI_HINT = {ANALYZE: "通常约 3–5 分钟", CORRECT: "通常约 1–3 分钟", INSTALL: "只在第一次：下载约 155 MB",
           DEPTH: "在本机测量，通常 5–20 秒"}
CREEP_CAP = 0.9  # an AI stage never fills more than 90 % of its own slice


def stages(correct: bool, set_camera: bool = True, precise: bool = False, install: bool = False) -> list:
    """Stage ids for one run, in order. 设置相机 only when 同时设置相机 is on; the 校正 stages only with 自动校正一轮;
    测量深度 (and, the first time, 安装精确模式) only with 精确模式."""
    out = [READ]
    if precise:
        if install:
            out.append(INSTALL)
        out.append(DEPTH)
    out += [ANALYZE, BUILD]
    if set_camera:
        out.append(CAMERA)
    if correct:
        out += [PREVIEW, SCORE, CORRECT, REBUILD]
    out.append(DONE)
    return out


def encode(ids) -> str:
    return ",".join(ids)


def decode(text: str) -> list:
    ids = [s for s in (text or "").split(",") if s in LABELS]
    return ids or stages(False)


def label(stage: str) -> str:
    return LABELS.get(stage, stage or "")


def index(ids, stage) -> int:
    try:
        return list(ids).index(stage)
    except ValueError:
        return -1


def fraction(ids, stage, stage_elapsed: float = 0.0, *, cached: bool = False, inner=None) -> float:
    """Bar factor 0..1 for ``stage`` of ``ids`` after ``stage_elapsed`` seconds in that stage
    (``inner`` = a measured 0..1 progress inside the stage, e.g. the 精确模式 download)."""
    ids = list(ids)
    if stage == DONE:
        return 1.0
    i = index(ids, stage)
    if i < 0:
        return 0.0
    total = float(sum(WEIGHTS[s] for s in ids)) or 1.0
    start = sum(WEIGHTS[s] for s in ids[:i]) / total
    span = WEIGHTS[stage] / total
    creep = 0.0
    if stage in AI_TYPICAL and not cached:
        creep = min(CREEP_CAP, max(0.0, float(stage_elapsed)) / AI_TYPICAL[stage])
    if inner is not None:
        creep = min(CREEP_CAP, max(0.0, float(inner)))
    return round(min(1.0, start + span * creep), 4)


def fmt_elapsed(seconds) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


def status_text(stage, elapsed) -> str:
    """「分析参考图 · 已用 1:42」"""
    return f"{label(stage)} · 已用 {fmt_elapsed(elapsed)}"


def hint(stage, *, cached: bool = False) -> str:
    if stage == ANALYZE and cached:
        return "使用已有分析结果"
    return AI_HINT.get(stage, "")


def failure_note(stage, elapsed, message) -> str:
    """Panel note when a stage fails: names the stage and the time spent."""
    return f"场景粗模失败（{label(stage)} · 已用 {fmt_elapsed(elapsed)}）：{message}"


def row_icons(ids, stage, failed_stage: str = "") -> list:
    """(label, icon, active) per stage for the panel list (running or failed)."""
    cur = index(ids, failed_stage or stage)
    rows = []
    for i, s in enumerate(ids):
        if failed_stage and i == cur:
            icon = "ERROR"
        elif i < cur or (stage == DONE and not failed_stage):
            icon = "CHECKMARK"
        elif i == cur:
            icon = "PLAY"
        else:
            icon = "RADIOBUT_OFF"
        rows.append((LABELS[s], icon, i <= cur))
    return rows
