"""「对齐到参考图」 (2026-11): solve a Blender camera that sees the user's car where the reference
photograph's vehicle stands (Fable 5.2 design: ``camera_align``).

Main thread: inputs, model dimensions. Worker thread: the cached vision call (``vehicle_locate``) and
the numpy solve. Modal timer: apply the result to a dedicated camera 「Wondful 对齐相机」 (the user's
cameras are never moved), set it active, match the render aspect to the photograph.
A failed solve changes nothing.
"""
from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path

import bpy
from bpy.types import Operator

CAMERA_NAME = "Wondful 对齐相机"
WHEEL_WORDS = ("wheel", "tire", "tyre", "rim", "轮")


def model_frame(collection):
    """(dims, anchor_xy, wheel_centres) of the user's model in world space.

    dims for ``camera_align.car_template``; the anchor is the centre of the wheel rectangle on the
    ground (wheel objects when they can be found, else the body's bbox centre)."""
    import numpy as np
    from mathutils import Vector
    pts, wheels = [], []
    for ob in collection.all_objects:
        if ob.type != "MESH" or ob.hide_render:
            continue
        corners = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
        pts += corners
        if any(w in ob.name.lower() for w in WHEEL_WORDS):
            c = sum(corners, Vector()) / 8.0
            r = max((max(v.z for v in corners) - min(v.z for v in corners)) / 2.0, 0.05)
            wheels.append((c.x, c.y, c.z, r))
    if not pts:
        raise RuntimeError("产品集合里没有可渲染的网格")
    a = np.array([(p.x, p.y, p.z) for p in pts])
    lo, hi = a.min(0), a.max(0)
    width, length, height = float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2])
    w = np.array(wheels) if wheels else None
    centres = None
    if w is not None and len(w) >= 4:
        # four wheel groups by quadrant (several objects per wheel are common: tyre, rim, brake)
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        quads = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                sel = w[(np.sign(w[:, 0] - cx) == sx) & (np.sign(w[:, 1] - cy) == sy)]
                if len(sel):
                    quads.append(sel[np.argmax(sel[:, 3])])
        if len(quads) == 4:
            centres = np.array(quads)
    if centres is not None:
        track = float(np.ptp(centres[:, 0]))
        wheelbase = float(np.ptp(centres[:, 1]))
        wheel_r = float(np.median(centres[:, 3]))
        anchor = (float(centres[:, 0].mean()), float(centres[:, 1].mean()))
    else:
        track, wheelbase, wheel_r = 0.85 * width, 0.6 * length, 0.36
        anchor = (float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2))
    dims = {"length": length, "width": width, "height": height,
            "wheelbase": wheelbase, "track": track, "wheel_r": wheel_r}
    return dims, anchor, centres is not None


def apply_to_camera(cam_obj, scene, p, anchor_z, anchor_xy, img_w, img_h, heading_z=0.0, size_scale=1.0):
    """Write the solved camera (car frame → world via the model anchor) into ``cam_obj``; set the
    render aspect to the photograph's (required for the composite)."""
    from mathutils import Matrix, Vector
    from . import camera_align as ac
    C, R, f = ac.camera(p, anchor_z)
    anchor = Matrix.Translation((anchor_xy[0], anchor_xy[1], 0.0)) @ Matrix.Rotation(heading_z, 4, "Z")
    rot = Matrix(R.tolist()).to_quaternion()
    cam_obj.matrix_world = anchor @ Matrix.LocRotScale(Vector((C * size_scale).tolist()), rot, None)
    cam = cam_obj.data
    cam.type, cam.sensor_fit, cam.sensor_width = "PERSP", "HORIZONTAL", 36.0
    cam.lens, cam.shift_x, cam.shift_y = float(36.0 * f), 0.0, 0.0
    r = scene.render
    long_side = max(r.resolution_x, r.resolution_y)
    if img_w >= img_h:
        r.resolution_x, r.resolution_y = long_side, int(round(long_side * img_h / img_w))
    else:
        r.resolution_x, r.resolution_y = int(round(long_side * img_w / img_h)), long_side
    r.pixel_aspect_x = r.pixel_aspect_y = 1.0


STATUS_TEXT = {"aligned": "已对齐", "touch_up": "已对齐，建议微调", "failed": "无法自动对齐"}
REASON_TEXT = {
    "kp_rms": "车轮位置拟合误差偏大", "bbox_iou": "车身范围对不齐", "dbottom": "接地线有偏差",
    "dx": "左右位置有偏差", "sil_iou": "轮廓不一致", "lens_mm": "焦距估计不确定",
    "cam_height": "相机高度异常", "roll_deg": "画面倾斜", "too_few_wheels": "可见车轮不足两个",
    "mirror_ambiguous": "车头方向可能相反",
}


class WONDFUL_OT_align_to_reference(Operator):
    bl_idname = "wondful.align_to_reference"
    bl_label = "对齐到参考图"
    bl_description = ("识别参考图里原车的位置、朝向和尺寸，自动求一台相机让你的车模正好落在原车位置"
                      "（新建「Wondful 对齐相机」，不改动已有相机）")

    flip: bpy.props.BoolProperty(name="车头翻转 180°", default=False, options={"SKIP_SAVE"})

    def execute(self, context):
        from .operators import _analysis_provider
        from . import vehicle_locate as VL
        props = context.scene.wondful_ai
        prefs = context.preferences.addons[__package__].preferences
        provider = _analysis_provider(props, prefs)
        photo = bpy.path.abspath(props.sb_image_path or "")
        if not photo or not Path(photo).is_file():
            self.report({"ERROR"}, "请先选择复刻目标图")
            return {"CANCELLED"}
        if props.product_collection is None:
            self.report({"ERROR"}, "请先选择产品集合")
            return {"CANCELLED"}
        if provider["id"] != "codex" or props.codex_account_state != "LOGGED_IN":
            self.report({"ERROR"}, "对齐到参考图需要已登录的 Codex")
            return {"CANCELLED"}
        try:
            self._model_dims, self._anchor, self._wheels_found = model_frame(props.product_collection)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        from .composition import image_dimensions
        self._size = image_dimensions(photo)
        if not self._size:
            self.report({"ERROR"}, "无法读取参考图尺寸")
            return {"CANCELLED"}
        self._photo = photo
        self._result = {}
        ask = VL.codex_ask_fn(model=provider["model"], client_version=str(getattr(props, "codex_cli_version", "") or ""))
        flip = self.flip

        def work():
            try:
                from . import camera_align as ac
                located = VL.locate(photo, ask)
                if not located.get("found"):
                    self._result = {"error": "未能识别参考图中的车辆"}
                    return
                obs = dict(located)
                if flip and obs.get("heading_deg") is not None:
                    obs["heading_deg"] = float(obs["heading_deg"]) + 180.0
                    obs["facing"] = ""
                    obs["wheels"] = [dict(w, label=w["label"].replace("front", "§").replace("rear", "front")
                                          .replace("§", "rear")) for w in obs.get("wheels", [])]
                dims, trusted = VL.old_dims(located, self._model_dims)
                fit, report, tpl = ac.align(obs, dims, self._size[0], self._size[1], dims_trusted=trusted)
                self._result = {"located": located, "p": fit["p"].tolist(), "anchor_z": tpl["anchor_z"],
                                "report": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in report.items()},
                                "dims": dims, "trusted": trusted}
            except Exception as exc:
                self._result = {"error": str(exc)[:300], "trace": traceback.format_exc()[-1500:]}

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        props.task_phase = "识别参考图车辆并求解相机"
        props.alignment_message = "正在对齐到参考图…"
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.3, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER" or self._thread.is_alive():
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        props = context.scene.wondful_ai
        props.task_phase = ""
        res = self._result
        if res.get("error"):
            props.alignment_message = "对齐失败：" + res["error"]
            self.report({"ERROR"}, props.alignment_message)
            return {"CANCELLED"}
        rep = res["report"]
        status = rep.get("status", "failed")
        reasons = "、".join(REASON_TEXT.get(k, k) for k in rep.get("bad", []) + rep.get("warn", []))
        if status == "failed":
            props.alignment_message = "无法自动对齐（" + (reasons or "原因未知") + "），相机未改动；请手动摆放相机"
            self.report({"WARNING"}, props.alignment_message)
            self._save(res)
            return {"CANCELLED"}
        import numpy as np
        scene = context.scene
        cam = bpy.data.objects.get(CAMERA_NAME)
        if cam is None or cam.type != "CAMERA":
            cam = bpy.data.objects.new(CAMERA_NAME, bpy.data.cameras.new(CAMERA_NAME))
            scene.collection.objects.link(cam)
        elif cam.name not in scene.objects:
            scene.collection.objects.link(cam)
        apply_to_camera(cam, scene, np.array(res["p"]), res["anchor_z"], self._anchor, *self._size)
        scene.camera = cam
        cam["wondful_aligned_photo"] = self._photo
        cam["wondful_align_report"] = json.dumps({k: rep[k] for k in ("status", "bbox_iou", "kp_rms", "lens_mm")
                                                  if k in rep}, ensure_ascii=False)
        model = (res["located"].get("dims") or {}).get("make_model", "")
        size_note = "按识别车型尺寸" + (f"（{model}）" if model else "") if res["trusted"] else "按你的车模尺寸"
        props.alignment_message = (f"{STATUS_TEXT[status]}：焦距 {rep['lens_mm']:.0f}mm，{size_note}"
                                   + (f"；注意：{reasons}" if reasons else "")
                                   + "。渲染比例已设为参考图比例")
        self._save(res)
        self.report({"INFO"}, props.alignment_message)
        for area in context.screen.areas if context.screen else ():
            area.tag_redraw()
        return {"FINISHED"}

    def _save(self, res):
        try:
            d = Path.home() / ".wondful_ai_renderer" / "align"
            d.mkdir(parents=True, exist_ok=True)
            (d / "last_align.json").write_text(json.dumps({**res, "photo": self._photo, "model_dims": self._model_dims,
                                                           "anchor": self._anchor, "wheels_found": self._wheels_found},
                                                          ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        except Exception:
            pass


CLASSES = (WONDFUL_OT_align_to_reference,)
