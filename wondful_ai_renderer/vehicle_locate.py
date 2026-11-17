"""Where is the photograph's own vehicle? (2026-11, 参考图复刻 placement guard + camera alignment)

One text-only vision call per photograph (cached by file hash) returns the main vehicle's tight
bounding box, its wheel contact points and the horizon, in normalised image coordinates.

Used by
  * ``placement_check``: before 替换原图车辆 spends image edits, compare the old vehicle with the
    user's car as the current camera projects it; stop when they clearly do not overlap;
  * the clean plate: the removal region always covers the old vehicle's whole box, so a larger
    old vehicle (a Jeep behind a YU7) leaves no remnants;
  * camera alignment (「对齐到参考图」), which starts from these keypoints.

Worker-thread safe (no bpy).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

import numpy as np

LOCATE_VERSION = 2
_LOCK = threading.Lock()

LOCATE_PROMPT = """You are measuring ONE vehicle in a photo: the main (largest, most central) road vehicle.
Reply with JSON only (no prose, no code fence) matching the schema below. All coordinates are normalised:
x = 0 left edge, 1 right edge; y = 0 top edge, 1 bottom edge.

{"found": true,
 "bbox": [x0, y0, x1, y1],
 "wheels": [{"label": "front_left", "contact": [x, y], "center": [x, y], "visible": true}],
 "facing": "front-left",
 "heading_deg": 40,
 "horizon_y": 0.41,
 "cut_off": [],
 "dims": {"make_model": "...", "length": 4.79, "width": 1.88, "height": 1.87,
          "wheelbase": 3.01, "track": 1.60, "wheel_r": 0.40},
 "confidence": 0.8}

- bbox: tight box around the whole visible vehicle, including mirrors and tyres, excluding its shadow and dust.
- wheels: one entry per wheel you can actually see. label is front_left / front_right / rear_left / rear_right,
  where left/right are the CAR'S OWN left and right (as felt by a driver seated facing forward), NOT the
  image's left and right. contact = where the tyre touches the road (lowest point of the tyre).
  center = hub centre. Never guess a hidden wheel; omit it.
- facing: which faces of the car the camera sees: front, rear, left, right, or a pair such as
  "front-left" or "rear-right" (the car's own left/right).
- heading_deg: where the camera stands around the car. 0 = directly in front of the car,
  90 = at the car's left side, 180 or -180 = directly behind, -90 = at the car's right side.
- horizon_y: y of the horizon / eye-level line if you can see or infer it from the road vanishing, else null.
- cut_off: list of image edges ("left","right","top","bottom") that crop the vehicle; [] if none.
- dims: your best estimate of this vehicle's real dimensions in metres (identify make/model if you can and
  use its published figures; otherwise estimate from the body class): length, width (without mirrors),
  height, wheelbase, track, wheel_r (tyre outer radius).
- confidence: 0-1.
If there is no vehicle, reply {"found": false}."""


def cache_file() -> Path:
    base = os.environ.get("WONDFUL_LOCATE_CACHE") or str(Path.home() / ".wondful_ai_renderer" / "vehicle_locate.json")
    p = Path(base)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_reply(text: str) -> dict:
    """Extract and sanity-check the JSON object from the model's reply."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("定位结果不是 JSON")
    data = json.loads(m.group(0))
    if not data.get("found"):
        return {"found": False}
    box = [float(v) for v in data.get("bbox", [])][:4]
    if len(box) != 4:
        raise ValueError("定位结果缺少 bbox")
    x0, y0, x1, y1 = box
    x0, x1 = sorted((min(1.0, max(0.0, x0)), min(1.0, max(0.0, x1))))
    y0, y1 = sorted((min(1.0, max(0.0, y0)), min(1.0, max(0.0, y1))))
    if x1 - x0 < 0.01 or y1 - y0 < 0.01:
        raise ValueError("定位框过小")
    wheels = []
    for w in data.get("wheels") or []:
        try:
            c = [float(v) for v in w.get("contact", [])][:2]
            h = [float(v) for v in w.get("center", [])][:2]
            if len(c) == 2 and w.get("visible") is not False:
                wheels.append({"label": str(w.get("label", "unknown")), "contact": c,
                               "center": h if len(h) == 2 else None,
                               "visible": True})
        except (TypeError, ValueError):
            continue
    hz = data.get("horizon_y")
    return {"found": True, "bbox": [x0, y0, x1, y1], "wheels": wheels,
            "facing": str(data.get("facing", "")), "heading_deg": data.get("heading_deg"),
            "horizon_y": float(hz) if isinstance(hz, (int, float)) else None,
            "cut_off": [str(s) for s in data.get("cut_off") or []],
            "dims": data.get("dims") if isinstance(data.get("dims"), dict) else None,
            "confidence": float(data.get("confidence", 0.5) or 0.0)}


DIM_KEYS = ("length", "width", "height", "wheelbase", "track", "wheel_r")


def old_dims(located: dict, model_dims: dict):
    """(dims, trusted): the photographed vehicle's metric dimensions for the camera solve. The vision
    estimate is used when it passes sanity clamps; otherwise the user's model, marked untrusted
    (the new car then appears at the old car's apparent size)."""
    d = (located or {}).get("dims") or {}
    try:
        v = {k: float(d[k]) for k in DIM_KEYS}
    except (KeyError, TypeError, ValueError):
        return dict(model_dims), False
    ok = (3.2 <= v["length"] <= 6.0 and 1.5 <= v["width"] <= 2.2 and 1.1 <= v["height"] <= 2.2
          and 0.52 <= v["wheelbase"] / v["length"] <= 0.66 and 0.82 <= v["track"] / v["width"] <= 0.88
          and 0.28 <= v["wheel_r"] <= 0.45)
    return (v, True) if ok else (dict(model_dims), False)


def locate(path, ask_fn, *, use_cache=True) -> dict:
    """``ask_fn(prompt, image_path) -> text``; cached per photograph hash."""
    key = f"v{LOCATE_VERSION}:{_sha(path)}"
    with _LOCK:
        cache = {}
        f = cache_file()
        if f.is_file():
            try:
                cache = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cache = {}
        if use_cache and key in cache:
            return {**cache[key], "cached": True}
    result = parse_reply(ask_fn(LOCATE_PROMPT, str(path)))
    with _LOCK:
        cache = {}
        if f.is_file():
            try:
                cache = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cache = {}
        cache[key] = result
        f.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return {**result, "cached": False}


def codex_ask_fn(*, model="", client_version="", timeout=180, cancel_check=None):
    """Text-only vision call on the Codex responses endpoint."""
    def fn(prompt, image_path):
        from . import codex_direct as CD
        auth_path = CD.find_auth_file()
        if auth_path is None:
            raise RuntimeError("未找到 Codex 登录文件")
        auth = CD.load_auth(auth_path)
        if CD.token_expiring(CD.access_token(auth)):
            auth = CD.refresh_auth(auth, auth_path)
        base = os.environ.get(CD.BASE_URL_ENV, "").strip() or CD.DEFAULT_BASE_URL
        url_img, _mime, _n = CD.data_url(image_path)
        payload = {"model": (model or "").strip() or CD.DEFAULT_MODEL,
                   "instructions": "Annotate the image as asked. Do not use any tool.",
                   "input": [{"type": "message", "role": "user", "content": [
                       {"type": "input_text", "text": prompt},
                       {"type": "input_image", "image_url": url_img}]}],
                   "tools": [], "parallel_tool_calls": False, "store": False, "stream": True, "include": []}
        url = base.rstrip("/") + "/responses"
        try:
            return CD.post_responses(url, CD.auth_headers(auth, client_version), payload, timeout=timeout,
                                     cancel_check=cancel_check, want="text")
        except CD.DirectEditError as exc:
            if exc.kind == "auth" and exc.status == 401:
                auth = CD.refresh_auth(auth, auth_path)
                return CD.post_responses(url, CD.auth_headers(auth, client_version), payload, timeout=timeout,
                                         cancel_check=cancel_check, want="text")
            raise
    return fn


# ---------------------------------------------------------------- placement check

FAIL_IOU = 0.30         # box IoU below this: the camera does not look at the old vehicle's place
WARN_IOU = 0.55
FAIL_CENTRE = 0.45      # centre distance as a fraction of the old box diagonal
FAIL_SIZE = (0.55, 1.8)  # new box area / old box area


def box_of(alpha: np.ndarray):
    a = np.asarray(alpha)
    if a.dtype != np.bool_:
        a = a > (127 if a.max() > 1 else 0.5)
    ys, xs = np.nonzero(a)
    if not len(xs):
        return None
    h, w = a.shape
    return [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def placement_check(located: dict, product_alpha) -> dict:
    """Compare the user's car (as the current camera projects it) with the photograph's vehicle.

    verdict: "ok" | "warn" (run, but say so) | "fail" (stop before any image edit) | "unknown"
    (no vehicle found: nothing to compare, run as before)."""
    if not located or not located.get("found"):
        return {"verdict": "unknown", "reason": "参考图中未识别到车辆，跳过对齐检查"}
    new = box_of(product_alpha)
    if new is None:
        return {"verdict": "fail", "reason": "当前相机里看不到车模"}
    old = located["bbox"]
    iou = box_iou(new, old)
    ocx, ocy = (old[0] + old[2]) / 2, (old[1] + old[3]) / 2
    ncx, ncy = (new[0] + new[2]) / 2, (new[1] + new[3]) / 2
    diag = max(1e-6, float(np.hypot(old[2] - old[0], old[3] - old[1])))
    centre = float(np.hypot(ncx - ocx, ncy - ocy)) / diag
    area_old = (old[2] - old[0]) * (old[3] - old[1])
    area_new = (new[2] - new[0]) * (new[3] - new[1])
    size = area_new / max(1e-6, area_old)
    out = {"box_iou": round(iou, 3), "centre_offset": round(centre, 3), "size_ratio": round(size, 3),
           "old_bbox": [round(v, 4) for v in old], "new_bbox": [round(v, 4) for v in new]}
    problems = []
    if iou < FAIL_IOU:
        problems.append(f"位置重合仅 {iou:.0%}")
    if centre > FAIL_CENTRE:
        problems.append("车的位置偏离原车")
    if not FAIL_SIZE[0] <= size <= FAIL_SIZE[1]:
        problems.append("车在画面中" + ("偏小" if size < 1 else "偏大") + f"（面积 {size:.0%}）")
    if problems:
        return {**out, "verdict": "fail",
                "reason": "当前相机没有对准参考图里的车：" + "，".join(problems)
                          + "。请先点「对齐到参考图」，或手动调整相机让车模落在原车位置。"}
    if iou < WARN_IOU:
        return {**out, "verdict": "warn", "reason": f"相机与参考图大致对齐（重合 {iou:.0%}），边缘可能残留原车"}
    return {**out, "verdict": "ok", "reason": f"相机与参考图对齐（重合 {iou:.0%}）"}


def old_vehicle_region(located: dict, shape, grow: float = 0.06) -> np.ndarray | None:
    """Boolean region covering the photograph's vehicle box (+ ``grow`` × its size, plus a ground
    band for shadow and dust), for the clean plate."""
    if not located or not located.get("found"):
        return None
    h, w = shape
    x0, y0, x1, y1 = located["bbox"]
    bw, bh = x1 - x0, y1 - y0
    gx, gy = grow * bw, grow * bh
    X0 = int(max(0, (x0 - gx) * w)); X1 = int(min(w, np.ceil((x1 + gx) * w)))
    Y0 = int(max(0, (y0 - gy) * h)); Y1 = int(min(h, np.ceil((y1 + 0.18 * bh) * h)))
    region = np.zeros((h, w), bool)
    region[Y0:Y1, X0:X1] = True
    return region
