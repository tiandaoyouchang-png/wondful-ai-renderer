"""Native offline smoke test, NOT executed in the delivery environment.

Usage in a separate Blender process (no user project):
  blender --background --factory-startup --python-exit-code 1 \
      --python tools/native_two_workflows_smoke.py -- /path/to/source /path/to/output

This script disables the startup updater and never calls AI or credentials APIs.
It tests registration, actual RNA callbacks, visibility restoration, preview and
comparison on disposable primitives. It does not validate YU7 or AI image quality.
"""
from pathlib import Path
import json
import sys
import traceback

import bpy
from mathutils import Vector


def main():
    if not bpy.app.background:
        raise RuntimeError("Run only in a dedicated background Blender process, not an open user project.")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(argv) != 2:
        raise RuntimeError("Pass source root and a dedicated output directory after --.")
    root, out = (Path(p).resolve() for p in argv)
    if not (root / "wondful_ai_renderer" / "__init__.py").is_file():
        raise RuntimeError("Source root does not contain the add-on package.")
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sys.path.insert(0, str(root))
    import wondful_ai_renderer as addon
    from wondful_ai_renderer import workflow as wf, operators as ops, comparison_view as cv
    addon.update_ops.schedule_startup_check = lambda: None
    registered = False
    result = {"blender": bpy.app.version_string, "build": wf.BUILD_ID,
              "network_calls": False, "checks": [], "visual_acceptance": "NOT_TESTED"}

    def check(name, condition):
        result["checks"].append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    def cube(name, collection, location, scale):
        mesh = bpy.data.meshes.new(name)
        points = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        mesh.from_pydata(points, [], [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                                      (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        collection.objects.link(obj)
        obj.location, obj.scale = location, scale
        return obj

    try:
        addon.register()
        registered = True
        scene = bpy.context.scene
        props = scene.wondful_ai
        check("legacy defaults to environment", props.workflow_mode == wf.ENVIRONMENT)
        check("preview operator registered", bpy.ops.wondful.replica_preview.get_rna_type() is not None)
        product = bpy.data.collections.new("Smoke product")
        blocks = bpy.data.collections.new("Smoke blockout")
        blocks["wondful_scene_blockout"] = 1
        scene.collection.children.link(product)
        scene.collection.children.link(blocks)
        obj = cube("Disposable test product", product, (0, 0, 0.7), (1, 2, 0.7))
        ground = cube("Disposable test ground", blocks, (0, 0, -0.15), (12, 12, 0.15))
        ground["wondful_sb_group"] = 1
        cam = bpy.data.objects.new("User camera", bpy.data.cameras.new("User camera"))
        scene.collection.objects.link(cam)
        cam.location = (7, -9, 4)
        cam.rotation_euler = (Vector((0, 0, 0.8)) - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.data.type = "PERSP"
        scene.camera = cam
        props.product_collection, props.massing_collection = product, blocks
        scene.render.resolution_x, scene.render.resolution_y = 160, 90
        scene.render.resolution_percentage = 100
        bpy.context.view_layer.update()
        before = [list(row) for row in cam.matrix_world]
        props.prompt, props.scene_brief = "Environment draft", "Environment brief"
        props.workflow_mode = wf.REFERENCE
        props.prompt, props.scene_brief = "Replica draft", "Replica brief"
        props.workflow_mode = wf.ENVIRONMENT
        check("environment draft restored by actual RNA callback", props.prompt == "Environment draft")
        check("mode switch preserves camera", scene.camera == cam and before == [list(row) for row in cam.matrix_world])
        with wf.isolate_environment_capture(bpy.context):
            check("generated block hidden for environment", ground.hide_render and ground.hide_get())
            check("product remains visible", not obj.hide_render and not obj.hide_get())
        check("capture visibility restored", not ground.hide_render and not ground.hide_get())
        try:
            with wf.isolate_environment_capture(bpy.context):
                raise RuntimeError("Intentional capture failure")
        except RuntimeError:
            pass
        check("capture exception restores state", not ground.hide_render and not ground.hide_get())
        props.workflow_mode = wf.REFERENCE
        check("replica draft restored by actual RNA callback", props.prompt == "Replica draft")
        image = bpy.data.images.new("Disposable reference", width=160, height=90)
        image.generated_color = (0.45, 0.45, 0.45, 1)
        image.filepath_raw, image.file_format = str(out / "target.png"), "PNG"
        image.save()
        props.sb_image_path = image.filepath_raw
        digest = wf.file_hash(image.filepath_raw)
        props.sb_built_target_hash = digest
        blocks["wondful_sb_reference_image"] = image.filepath_raw
        blocks["wondful_sb_reference_sha256"] = digest
        blocks["wondful_sb_original_sha256"] = digest
        check("render blocked before approval", bool(ops.workflow_preflight(bpy.context, props)))
        preview = ops._render_replica_preview(bpy.context, props, out)
        check("native preview created", Path(preview).is_file() and props.sb_preview_image is not None)
        check("preview requires fresh approval", props.replica_reviewed is False)
        check("target loaded via RNA callback", props.sb_reference_image is not None)
        comparison = cv.refresh_compare_image(bpy.context, force_cache=True)
        check("comparison works before final AI result", comparison is not None and props.result_image is None)
        props.replica_reviewed = True
        check("valid reviewed source passes local guard", ops.workflow_preflight(bpy.context, props) == "")
        check("preview preserves current camera", scene.camera == cam and before == [list(row) for row in cam.matrix_world])
        result["status"] = "PASSED"
    except Exception:
        result["status"] = "FAILED"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        if registered:
            try:
                addon.unregister()
                result["unregistered"] = not hasattr(bpy.types.Scene, "wondful_ai")
            except Exception:
                result["unregister_error"] = traceback.format_exc()
        (out / "native-smoke-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
