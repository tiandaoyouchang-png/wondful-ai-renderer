"""Real headless 多角度场景一致 comparison: current carrier (LLM blocks) vs 细节粗模 carrier (1.2.5, V3 phase 1).

Reuses the e2e_scene_blockout_bg.py harness (YU7 product scene, modal pump, Cycles capture). Steps:
1. ④ with 精确模式 from a recorded analysis JSON + recorded depth (no AI call) → LLM blocks + hidden 细节粗模;
2. 「生成」 on the reference camera = the real AI view-1 (主视角), 1 image call (最多生成次数 1, 细节修复 crop regen off);
3. one orbit camera (yaw around the product); 多角度场景一致 once per arm (1 image call each, retries off):
   ``llm`` = 细节粗模投影 off, ``carrier`` = on (white model unchanged), ``clay`` = on + 白模也用细节粗模;
4. per arm: copies the pool, computes geometry metrics from the plugin's own passes (pos vs the measured
   surface), projection coverage / stretch / holes, and reads the operator's reproj ΔE + product gate.

    CODEX_HOME=… DMV_GEO=geo.npz DMV_YAW=-45 DMV_ARMS=llm,carrier[,clay] blender -b --factory-startup \
        --python tools/e2e_detail_mv_bg.py -- <addon.zip> yu7 <reference.jpg> <run_dir> 0 <analysis.json>
"""
import json
import math
import os
import shutil
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "e2e_scene_blockout_bg.py"), encoding="utf-8").read()
_src = _src[:_src.rindex("\nmain()")]
exec(compile(_src, "e2e_scene_blockout_bg.py", "exec"))   # noqa: S102 — harness: patch, wait, build_scene, log …

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

GEO = os.environ["DMV_GEO"]
YAW = float(os.environ.get("DMV_YAW", "-45"))
ARMS = [a for a in os.environ.get("DMV_ARMS", "llm,carrier").split(",") if a]
#   DMV_NOGEN=1: no image calls at all (canvas = projection only; needs DMV_MASTER)
MASTER_IMG = os.environ.get("DMV_MASTER", "")


def measured_bvh(geo, world, z_offset, st=2):
    """Open 2.5D relief of the measurement (the novel-view pseudo truth, like scene_blockout/eval_bl.py), world."""
    from mathutils.bvhtree import BVHTree
    P = np.asarray(geo["P"], np.float64)[::st, ::st].copy()
    P[..., 2] += z_offset
    m = np.asarray(geo["mask"])[::st, ::st] & (np.asarray(geo["cls"])[::st, ::st] != 0)
    r = np.hypot(P[..., 0], P[..., 1])
    m &= r < 5000
    h, w = m.shape
    idx = np.arange(h * w).reshape(h, w)
    F = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    rr = r.ravel()
    ok = m.ravel()[F].all(1)
    R4 = rr[F]
    ok &= (R4.max(1) / np.maximum(R4.min(1), 1e-3)) < 1.25
    F = F[ok]
    W = np.array(world)
    V = P.reshape(-1, 3) @ W[:3, :3].T + W[:3, 3]
    return BVHTree.FromPolygons([tuple(v) for v in V], [tuple(map(int, f)) for f in F], all_triangles=False)


def arm_metrics(MV, pool, cam_name, master_name, bvhA, cam_rec):
    """Geometry ±10 % vs the measured surface (< 150 m), holes, projection stretch / coverage at work size."""
    P = np.load(os.path.join(pool, MV.safe_name(cam_name) + "_pos.npy"))
    Pm = np.load(os.path.join(pool, MV.safe_name(master_name) + "_pos.npy"))
    H, W = P.shape[:2]
    D = MV.rays(cam_rec, W, H)
    o = Vector(np.array(cam_rec["matrix"])[:3, 3])
    hit = P[..., 3] > 0.5
    err, n = [], 0
    for j in range(0, H, 3):
        for i in range(0, W, 3):
            loc, _n, _f, dA = bvhA.ray_cast(o, Vector(D[j, i]), 20000.0)
            if loc is None or dA > 150:
                continue
            n += 1
            dv = float(np.linalg.norm(P[j, i, :3] - np.array(o))) if hit[j, i] else 1e9
            err.append(abs(dv - dA) / dA)
    err = np.array(err) if err else np.zeros(1)
    below = D[..., 2] < -0.02
    hole = float(((~hit) & below).sum()) / max(1, int(below.sum()))
    mk = np.load(os.path.join(pool, MV.safe_name(cam_name) + "_maskw.npy"))
    mm = np.load(os.path.join(pool, MV.safe_name(master_name) + "_maskw.npy"))
    vk = {"pos": P, "mask": mk, "cam": cam_rec}
    vj = {"pos": Pm, "mask": mm, "cam": json.load(open(os.path.join(pool, "pool.json")))["views"][0]["cam"]}
    U, V, vis, sky, hit2, sky_in = MV.correspond(vk, vj)
    a = np.zeros((H, W)); b = np.zeros((H, W)); c = np.zeros((H, W)); d = np.zeros((H, W))
    a[:, :-1] = U[:, 1:] - U[:, :-1]; c[:, :-1] = V[:, 1:] - V[:, :-1]
    b[:-1] = U[1:] - U[:-1]; d[:-1] = V[1:] - V[:-1]
    nb = np.zeros((H, W), bool)
    nb[:-1, :-1] = vis[:-1, :-1] & vis[1:, :-1] & vis[:-1, 1:]
    S = a * a + b * b + c * c + d * d
    det = np.abs(a * d - b * c)
    smin = np.sqrt(np.maximum(0, (S - np.sqrt(np.maximum(0, S * S - 4 * det * det))) / 2))
    env = (mk < 0.5) & hit
    stretched = nb & (smin < 1 / 3.0) & env
    return {"geom_within10": round(float((err < 0.10).mean()), 4), "geom_px": n,
            "hole_below_horizon": round(hole, 4),
            "proj_visible": round(float((vis & env).sum()) / max(1, int(env.sum())), 4),
            "proj_stretch3x": round(float(stretched.sum()) / max(1, int(env.sum())), 4)}


def main2():
    rep = {"ref": REF, "yaw": YAW, "arms": {}}
    try:
        bpy.ops.preferences.addon_install(filepath=ZIP, overwrite=True)
        bpy.ops.preferences.addon_enable(module="wondful_ai_renderer")
        from wondful_ai_renderer import operators as OPS
        from wondful_ai_renderer import multiview as MV, detail_refine as DR, precise_depth as PD
        from wondful_ai_renderer import detail_blockout as DB
        patch(OPS)
        DR.make_edit_fn = lambda *a, **k: None          # 细节修复 crop regeneration off (budget); Blender paste stays
        MV.MAX_CALLS = 1                                  # one image call per camera
        if os.environ.get("DMV_NOGEN") == "1":
            # no image model (quota): the "result" is the projected canvas itself, light estimate skipped
            from wondful_ai_renderer import codex_direct as CD

            def _nogen_edit(**kw):
                shutil.copy2(kw["edit_base_path"], kw["output_path"])
                return {"seconds": 0.0, "image_quality": "nogen", "image_model": "nogen"}
            CD.edit_image = _nogen_edit

            def _nolight(*a, **k):
                raise RuntimeError("nogen")
            MV.estimate_light = _nolight
        if os.environ.get("DMV_CALLS"):
            # strict image-call ledger: one line per edit_image call (shared across runs via the file)
            from wondful_ai_renderer import codex_direct as CD2
            _orig_edit = CD2.edit_image

            def _counted(**kw):
                cap = int(os.environ.get("DMV_MAX_CALLS", "12"))
                done = sum(1 for _ in open(os.environ["DMV_CALLS"], encoding="utf-8")) if os.path.isfile(os.environ["DMV_CALLS"]) else 0
                if done >= cap:
                    raise RuntimeError(f"image-call budget {cap} reached (ledger)")
                with open(os.environ["DMV_CALLS"], "a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%H:%M:%S')} {os.path.basename(RUN)} {kw.get('tag', '')}\n")
                return _orig_edit(**kw)
            CD2.edit_image = _counted
            _orig_gwf = CD2.generate_with_fallback

            def _no_fallback(kwargs, **kw):
                def _refuse(**_k):
                    raise RuntimeError("codex exec fallback disabled (image-call budget)")
                kw["fallback"] = _refuse
                return _orig_gwf(kwargs, **kw)
            CD2.generate_with_fallback = _no_fallback
        prefs = bpy.context.preferences.addons["wondful_ai_renderer"].preferences
        prefs.alignment_max_attempts = 1                  # 1.2.5: hard cap — gate / low-tier / mv retries stay ≤ 1
        coll = build_scene()
        sc = bpy.context.scene
        props = sc.wondful_ai
        props.product_collection = coll
        props.render_count = "1"                          # 张数 1 (default 4 → 4 variants = 4 calls)
        props.iou_gate_max = 1
        bpy.ops.wondful.load_reference(ref_kind="STYLE", filepath=REF)
        props.output_directory = os.path.join(RUN, "out")
        props.sb_set_camera = True
        props.sb_correct = False
        props.sb_precise = True
        bpy.ops.wondful.codex_check()
        wait(OPS, 90)
        t = time.time()
        r = bpy.ops.wondful.scene_blockout_build(analysis_json=CACHED, precise_geo=GEO)
        wait(OPS, 600)
        rep["blockout_note"] = props.last_sb_note
        rep["blockout_seconds"] = round(time.time() - t, 1)
        log("blockout", props.last_sb_note)
        from wondful_ai_renderer.render_session import CACHE_ROOT
        sess = sorted([p for p in CACHE_ROOT.iterdir() if (p / "scene_blockout_info.json").is_file()],
                      key=lambda p: p.stat().st_mtime)[-1]
        info = json.load(open(sess / "scene_blockout_info.json", encoding="utf-8"))
        rep["detail"] = info.get("detail")
        shutil.copy2(sess / "scene_blockout_info.json", os.path.join(RUN, "scene_blockout_info.json"))
        ref_cam = sc.camera
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(RUN, "scene.blend"), copy=True)
        # ---- 主视角 = real AI view 1
        if MASTER_IMG and os.path.isfile(MASTER_IMG):
            master_img = MASTER_IMG
            rep["master_reused"] = True
        else:
            t = time.time()
            bpy.ops.wondful.generate()
            wait(OPS, 1800)
            rep["generate_seconds"] = round(time.time() - t, 1)
            rep["generate_status"] = props.status
            rep["generate_error"] = props.last_error
            master_img = props.last_export_path
            if not master_img or not os.path.isfile(master_img):
                raise RuntimeError("主视角生成失败：" + props.last_error[:200])
        shutil.copy2(master_img, os.path.join(RUN, "master.png"))
        log("master", master_img)
        # ---- orbit camera around the product
        prod = coll
        pts = [o.matrix_world @ Vector(c) for o in prod.all_objects if o.type == "MESH" for c in o.bound_box]
        piv = Vector(((min(p.x for p in pts) + max(p.x for p in pts)) / 2, (min(p.y for p in pts) + max(p.y for p in pts)) / 2,
                      min(p.z for p in pts)))
        oc = bpy.data.objects.new("Orbit", bpy.data.cameras.new("Orbit"))
        sc.collection.objects.link(oc)
        rc = ref_cam.matrix_world.translation
        rel = rc - piv
        loc = piv + Matrix.Rotation(math.radians(YAW), 3, "Z") @ rel
        oc.location = loc
        fwd = ref_cam.matrix_world.to_3x3() @ Vector((0, 0, -1))
        tgt = piv + Vector((0, 0, 1.0))
        q = (tgt - loc).to_track_quat("-Z", "Y")
        oc.rotation_euler = q.to_euler()
        oc.data.sensor_fit = "HORIZONTAL"
        oc.data.sensor_width = 36.0
        oc.data.lens = ref_cam.data.lens
        oc.data.clip_start, oc.data.clip_end = 0.1, 100000.0
        props.mv_master_camera = ref_cam
        props.mv_master_image = master_img
        geo = PD.load_geo(GEO)
        z_off = float(info["cam_local_z"]) - float(geo["cam"]["height_m"])
        bvhA = measured_bvh(geo, Matrix(info["world"]), z_off)
        for arm in ARMS:
            props.sb_detail_carrier = arm in ("carrier", "clay", "sky")
            props.sb_detail_clay = arm == "clay"
            props.sb_detail_skyline = arm == "sky"          # 1.2.5 远山剪影
            for o in sc.objects:
                o.select_set(False)
            oc.select_set(True)
            bpy.context.view_layer.objects.active = oc
            t = time.time()
            r = bpy.ops.wondful.multiview_render()
            wait(OPS, 1800)
            pool = None
            for root, dirs, files in os.walk(os.path.join(RUN, "out")):
                if "pool.json" in files:
                    pool = root
            a = {"op": sorted(r), "seconds": round(time.time() - t, 1), "note": props.last_mv_note, "pool": pool}
            log("arm", arm, props.last_mv_note)
            if pool:
                dst = os.path.join(RUN, f"pool_{arm}")
                shutil.copytree(pool, dst, dirs_exist_ok=True)
                rec = MV.scene_camera_record(sc, oc)
                try:
                    a["metrics"] = arm_metrics(MV, pool, oc.name, ref_cam.name, bvhA, rec)
                except Exception:
                    a["metrics_error"] = traceback.format_exc()
                meta = os.path.join(pool, MV.safe_name(oc.name) + "_meta.json")
                if os.path.isfile(meta):
                    a["meta"] = json.load(open(meta, encoding="utf-8"))
                light = os.path.join(pool, MV.LIGHT_FILE)
                keep = open(light, "rb").read() if os.path.isfile(light) else None
                shutil.rmtree(pool, ignore_errors=True)        # next arm starts clean (no reuse) …
                if keep:                                       # … but with the same light estimate (no extra call)
                    os.makedirs(pool, exist_ok=True)
                    open(light, "wb").write(keep)
            rep["arms"][arm] = a
        rep["ok"] = True
    except Exception:
        rep["error"] = traceback.format_exc()
        log("driver error", rep["error"])
    with open(os.path.join(RUN, "report_mv.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=str)
    log("DONE")
    LOG.close()
    os._exit(0)


main2()
