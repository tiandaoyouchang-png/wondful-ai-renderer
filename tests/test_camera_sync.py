"""Focused fake-RNA coverage for generated reference-camera activation."""
import importlib.util
from pathlib import Path
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("camera_sync", ROOT / "wondful_ai_renderer" / "camera_sync.py")
CAMERA_SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMERA_SYNC)
NS = types.SimpleNamespace


def view(*, quad=False):
    return NS(type="VIEW_3D", camera=None, use_local_camera=False,
              region_3d=NS(view_perspective="PERSP"),
              region_quadviews=[NS(view_perspective="ORTHO") for _ in range(4)] if quad else [])


class CameraSyncTests(unittest.TestCase):
    def test_matching_scene_views_use_generated_camera_and_camera_view(self):
        scene, other, camera = NS(camera=None), NS(camera=None), NS(name="Generated")
        main, quad, foreign = view(), view(quad=True), view()
        windows = [
            NS(scene=scene, screen=NS(areas=[NS(type="VIEW_3D", spaces=NS(active=main)),
                                               NS(type="PROPERTIES", spaces=NS(active=NS()))])),
            NS(scene=scene, screen=NS(areas=[NS(type="VIEW_3D", spaces=NS(active=quad))])),
            NS(scene=other, screen=NS(areas=[NS(type="VIEW_3D", spaces=NS(active=foreign))])),
        ]
        context = NS(scene=scene, window_manager=NS(windows=windows), space_data=None)

        self.assertTrue(CAMERA_SYNC.sync_generated_camera(context, camera))
        self.assertIs(scene.camera, camera)
        for space in (main, quad):
            self.assertIs(space.camera, camera)
            self.assertTrue(space.use_local_camera)
            self.assertEqual(space.region_3d.view_perspective, "CAMERA")
        self.assertTrue(all(region.view_perspective == "CAMERA" for region in quad.region_quadviews))
        self.assertIsNone(foreign.camera)
        self.assertEqual(foreign.region_3d.view_perspective, "PERSP")

    def test_originating_view_is_synced_without_window_manager(self):
        scene, camera, origin = NS(camera=None), NS(), view()
        context = NS(scene=scene, window_manager=None, space_data=origin)
        self.assertTrue(CAMERA_SYNC.sync_generated_camera(context, camera))
        self.assertIs(scene.camera, camera)
        self.assertIs(origin.camera, camera)
        self.assertEqual(origin.region_3d.view_perspective, "CAMERA")

    def test_origin_scene_override_does_not_change_the_new_active_scene(self):
        origin_scene, active_scene, camera = NS(camera=None), NS(camera=None), NS()
        old_view, active_view = view(), view()
        context = NS(scene=active_scene, window=NS(scene=active_scene),
                     window_manager=NS(windows=[NS(scene=origin_scene, screen=NS(areas=[
                         NS(type="VIEW_3D", spaces=NS(active=old_view))]))]),
                     space_data=active_view)
        self.assertTrue(CAMERA_SYNC.sync_generated_camera(context, camera, scene=origin_scene))
        self.assertIs(origin_scene.camera, camera)
        self.assertIsNone(active_scene.camera)
        self.assertIs(old_view.camera, camera)
        self.assertIsNone(active_view.camera)

    def test_headless_or_incomplete_context_is_safe(self):
        self.assertFalse(CAMERA_SYNC.sync_generated_camera(NS(scene=None), NS()))
        scene = NS(camera=None)
        self.assertFalse(CAMERA_SYNC.sync_generated_camera(NS(scene=scene), None))
        self.assertTrue(CAMERA_SYNC.sync_generated_camera(NS(scene=scene, window_manager=NS(windows=[]), space_data=None), NS()))

