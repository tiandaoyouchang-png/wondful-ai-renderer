"""Bounded, deferred native image thumbnails; no decoding or stat work in Panel.draw.

Native Blender preview icons have a square footprint. Generate a square FILL
thumbnail (or a full-image FIT on an image-based background) so wide reference
photos do not occupy a tiny, letterboxed strip in the sidebar. Original images
are never cropped, rescaled, saved, or overwritten.
"""
import os
import time
from collections import OrderedDict

_previews = None
_records = OrderedDict()
_pending = set()
_registered = False
MAX_THUMBNAILS = 48
THUMB_SIDE = 256


def _redraw():
    import bpy
    for win in getattr(getattr(bpy.context, 'window_manager', None), 'windows', ()):
        for area in win.screen.areas:
            if area.type in {'VIEW_3D', 'PREFERENCES'}:
                area.tag_redraw()


def _pixels_thumbnail(image, thumb, fit='cover'):
    """Sample a fixed-size cover/contain tile without modifying the image.

    In 'contain' mode the entire picture stays legible above a dimmed cover
    backdrop, useful for wide front/side/rear identity reference sheets.
    """
    import numpy as np
    w, h = (int(v) for v in image.size)
    if w <= 0 or h <= 0 or len(image.pixels) != w * h * 4:
        raise ValueError('No RGBA pixels available for image preview')
    side = THUMB_SIDE
    # Avoid hundreds of slow per-row RNA accesses for normal-sized images.
    # A single foreach_get is much faster than slicing bpy_prop_array in a
    # Python loop. Very large 8K photos stay on the bounded-row path.
    full = None
    if w * h <= 8_000_000:
        rgba = np.empty(w * h * 4, dtype=np.float32)
        pixels = image.pixels
        if callable(getattr(pixels, 'foreach_get', None)):
            pixels.foreach_get(rgba)
        else:
            rgba[:] = np.asarray(pixels, dtype=np.float32)
        full = rgba.reshape(h, w, 4)

    def sample(x_coords, y_coords):
        xs = np.clip(x_coords.astype('int64'), 0, w - 1)
        ys = np.clip(y_coords.astype('int64'), 0, h - 1)
        if full is not None:
            return full[ys[:, None], xs[None, :], :].copy()
        sampled = np.empty((len(ys), len(xs), 4), dtype=np.float32)
        for i, y in enumerate(ys):
            row = np.asarray(image.pixels[int(y) * w * 4:(int(y) + 1) * w * 4], dtype=np.float32)
            sampled[i] = row.reshape(w, 4)[xs]
        return sampled

    # Enlarge and center-crop the image to a square rather than scaling it
    # down inside a 256x256 empty black preview frame.
    if fit.startswith('tile:'):
        try:
            _, index, count = fit.split(':')
            index, count = int(index), int(count)
            if not (0 <= index < count <= 3):
                raise ValueError('Invalid tile')
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid tile selection') from exc
        slice_width = w / count
        crop = min(slice_width, h)
        ox = (index + 0.5) * slice_width - crop * 0.5
        oy = (h - crop) * 0.5
    else:
        crop = min(w, h)
        ox, oy = (w - crop) * 0.5, (h - crop) * 0.5
    xx = ox + (np.arange(side) + 0.5) * crop / side
    yy = oy + (np.arange(side) + 0.5) * crop / side
    out = sample(xx, yy)
    if fit == 'contain' and w != h:
        out[:, :, :3] = np.clip(out[:, :, :3] * 0.25 + 0.035, 0, 1)
        if w > h:
            fg_w, fg_h = side, max(1, round(side * h / w))
        else:
            fg_w, fg_h = max(1, round(side * w / h)), side
        fg = sample((np.arange(fg_w) + 0.5) * w / fg_w,
                    (np.arange(fg_h) + 0.5) * h / fg_h)
        left, top = (side - fg_w) // 2, (side - fg_h) // 2
        out[top:top + fg_h, left:left + fg_w] = fg
    # Blender icon pixels expect straight RGBA, and the tile is opaque.
    out[:, :, 3] = 1.0
    thumb.image_size = (side, side)
    thumb.image_pixels_float = out.ravel().tolist()


def _tick():
    global _previews, _registered
    import bpy
    try:
        import bpy.utils.previews
        if _previews is None:
            _previews = bpy.utils.previews.new()
        for key in list(_pending):
            _pending.discard(key)
            rec = _records.get(key)
            if rec is None:
                continue
            temp_image = None
            try:
                image = rec.get('image')
                fit = rec.get('fit', 'cover')
                path = bpy.path.abspath(rec.get('path', ''))
                packed = image is not None and (getattr(image, 'packed_file', None) or getattr(image, 'source', '') == 'GENERATED')
                if packed:
                    stamp = (image.name_full, tuple(image.size), bool(image.is_dirty))
                else:
                    st = os.stat(path)
                    stamp = (st.st_mtime_ns, st.st_size, bool(getattr(image, 'is_dirty', False)))
                native_stamp = ('native', stamp[0], stamp[1]) if not packed else None
                if native_stamp is not None and rec.get('stamp') == native_stamp:
                    # A Blender-native preview fallback was already loaded for
                    # these exact file bytes. No repeated allocation per tick.
                    pass
                elif stamp != rec.get('stamp') or bool(getattr(image, 'is_dirty', False)):
                    if key in _previews:
                        del _previews[key]
                    thumb = _previews.new(key)
                    if image is None:
                        # Performed on a timer, NEVER inside a draw callback.
                        temp_image = bpy.data.images.load(path, check_existing=False)
                        image = temp_image
                    _pixels_thumbnail(image, thumb, fit=fit)
                    rec.update(stamp=stamp, icon=thumb.icon_id, size=tuple(image.size))
            except (OSError, RuntimeError, ValueError, ReferenceError, AttributeError):
                # Keep Blender's native photo preview as a graceful fallback.
                try:
                    if rec.get('path') and os.path.isfile(path):
                        info = os.stat(path)
                        fallback_stamp = ('native', info.st_mtime_ns, info.st_size)
                        if fallback_stamp != rec.get('stamp'):
                            if key in _previews:
                                del _previews[key]
                            thumb = _previews.load(key, path, 'IMAGE', force_reload=True)
                            rec.update(icon=thumb.icon_id, stamp=fallback_stamp)
                    else:
                        rec.update(icon=0, stamp=None)
                except (OSError, RuntimeError, ValueError):
                    rec.update(icon=0, stamp=None)
            finally:
                if temp_image is not None:
                    try:
                        bpy.data.images.remove(temp_image)
                    except (RuntimeError, ReferenceError):
                        pass
                rec['checked'] = time.monotonic()
        while len(_records) > MAX_THUMBNAILS:
            key, _ = _records.popitem(last=False)
            if key in _previews:
                del _previews[key]
        _redraw()
    finally:
        _registered = False
    return None


def _queue(key, path, image=None, fit='cover'):
    global _registered
    rec = _records.setdefault(key, {'icon': 0, 'checked': 0})
    rec.update(path=path, image=image, fit=fit)
    _records.move_to_end(key)
    if time.monotonic() - rec['checked'] > 2:
        _pending.add(key)
        if not _registered:
            try:
                import bpy
                bpy.app.timers.register(_tick, first_interval=0.1)
                _registered = True
            except (AttributeError, RuntimeError):
                pass
    return rec['icon']


def cached_dimensions_for_path(raw):
    """Read cached dimensions only; never touch disk or decode in UI.draw."""
    rec = _records.get('path:cover:%s' % raw)
    return rec.get('size') if rec is not None else None


def icon_for_path(raw, fit='cover'):
    return _queue('path:%s:%s' % (fit, raw), str(raw), fit=fit) if raw else 0


def icon_for_image(image, fit='cover'):
    if image is None:
        return 0
    raw = getattr(image, 'filepath_raw', '') or getattr(image, 'filepath', '')
    try:
        return _queue('image:%s:%s' % (fit, image.as_pointer() if callable(getattr(image, 'as_pointer', None)) else id(image)), raw, image, fit=fit)
    except ReferenceError:
        return 0


def reset():
    global _previews, _registered
    import bpy
    if _registered:
        try:
            if bpy.app.timers.is_registered(_tick):
                bpy.app.timers.unregister(_tick)
        except (AttributeError, RuntimeError):
            pass
    if _previews is not None:
        import bpy.utils.previews
        bpy.utils.previews.remove(_previews)
    _previews = None
    _registered = False
    _records.clear()
    _pending.clear()
