"""Synchronize a successful reference blockout camera with visible 3D views."""


def _camera_view(region):
    """Put one RegionView3D in camera perspective without failing the build."""
    if region is None:
        return
    try:
        region.view_perspective = "CAMERA"
    except (AttributeError, RuntimeError, TypeError):
        pass


def _sync_space(space, camera):
    """Use ``camera`` in a single SpaceView3D, including quad subviews."""
    if space is None or getattr(space, "type", "") != "VIEW_3D":
        return
    try:
        space.camera = camera
        space.use_local_camera = True
    except (AttributeError, RuntimeError, TypeError):
        # Some headless or legacy SpaceView3D stand-ins do not expose local camera.
        pass
    _camera_view(getattr(space, "region_3d", None))
    for region in list(getattr(space, "region_quadviews", ()) or ()):
        _camera_view(region)


def sync_generated_camera(context, camera, scene=None):
    """Activate a finished generated camera for its scene and matching viewports.

    This is deliberately a UI/state helper: it never changes the camera's data or
    transform, and silently tolerates headless contexts.  Call it only after the
    reference blockout has fully succeeded so cancelled or failed work cannot move
    the user's active camera.
    """
    scene = scene or getattr(context, "_origin_scene", None) or getattr(context, "scene", None)
    if scene is None or camera is None:
        return False
    try:
        scene.camera = camera
    except (AttributeError, RuntimeError, TypeError):
        return False

    window_manager = getattr(context, "window_manager", None)
    for window in list(getattr(window_manager, "windows", ()) or ()):
        if getattr(window, "scene", None) is not scene:
            continue
        screen = getattr(window, "screen", None)
        for area in list(getattr(screen, "areas", ()) or ()):
            if getattr(area, "type", "") != "VIEW_3D":
                continue
            _sync_space(getattr(getattr(area, "spaces", None), "active", None), camera)

    # The originating context can be a temporary override without a WindowManager.
    # Do not update it if that window has since moved to another scene.
    origin_window = getattr(context, "window", None)
    if origin_window is None or getattr(origin_window, "scene", None) is scene:
        _sync_space(getattr(context, "space_data", None), camera)
    return True
