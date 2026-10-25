"""Bounded, deferred native thumbnails; no decoding or stat work in Panel.draw."""
import os
import time
from collections import OrderedDict

_previews = None
_records = OrderedDict()
_pending = set()
_registered = False
MAX_THUMBNAILS = 48


def _redraw():
    import bpy
    for win in getattr(getattr(bpy.context, 'window_manager', None), 'windows', ()):
        for area in win.screen.areas:
            if area.type in {'VIEW_3D', 'PREFERENCES'}:
                area.tag_redraw()


def _pixels_thumbnail(image, thumb):
    """Preview packed/generated images from their actual pixels, not a generic icon.

    Read only sampled rows, bounded to a 256px preview. Never scale, save or
    overwrite the source image; no full 4K float buffer is allocated here.
    """
    import numpy as np
    w, h = (int(v) for v in image.size)
    if w <= 0 or h <= 0 or len(image.pixels) != w * h * 4:
        raise ValueError('No RGBA pixels available for image preview')
    factor = min(1.0, 256.0 / max(w, h))
    tw, th = max(1, round(w * factor)), max(1, round(h * factor))
    xs = np.minimum(w - 1, ((np.arange(tw) + .5) * w / tw).astype(int))
    ys = np.minimum(h - 1, ((np.arange(th) + .5) * h / th).astype(int))
    out = np.empty((th, tw, 4), dtype=np.float32)
    for i, y in enumerate(ys):
        row = np.asarray(image.pixels[int(y) * w * 4:(int(y) + 1) * w * 4], dtype=np.float32)
        out[i] = row.reshape(w, 4)[xs]
    thumb.image_size = (tw, th)
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
            try:
                image = rec.get('image')
                path = bpy.path.abspath(rec.get('path', ''))
                if image is not None and (getattr(image, 'packed_file', None) or getattr(image, 'source', '') == 'GENERATED'):
                    # Packed data is authoritative even if a different disk file exists.
                    stamp = (image.name_full, tuple(image.size), bool(image.is_dirty))
                    if stamp != rec.get('stamp') or image.is_dirty:
                        if key in _previews:
                            del _previews[key]
                        thumb = _previews.new(key)
                        _pixels_thumbnail(image, thumb)
                        rec.update(stamp=stamp, icon=thumb.icon_id)
                else:
                    st = os.stat(path)
                    stamp = (st.st_mtime_ns, st.st_size)
                    if stamp != rec.get('stamp'):
                        if key in _previews:
                            del _previews[key]
                        thumb = _previews.load(key, path, 'IMAGE', force_reload=True)
                        rec.update(stamp=stamp, icon=thumb.icon_id)
            except (OSError, RuntimeError, ValueError, ReferenceError, AttributeError):
                rec['icon'] = 0
                rec['stamp'] = None
            rec['checked'] = time.monotonic()
        while len(_records) > MAX_THUMBNAILS:
            key, _ = _records.popitem(last=False)
            if key in _previews:
                del _previews[key]
        _redraw()
    finally:
        _registered = False
    return None


def _queue(key, path, image=None):
    global _registered
    rec = _records.setdefault(key, {'icon': 0, 'checked': 0})
    rec.update(path=path, image=image)
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


def icon_for_path(raw):
    return _queue(str(raw), str(raw)) if raw else 0


def icon_for_image(image):
    if image is None:
        return 0
    raw = getattr(image, 'filepath_raw', '') or getattr(image, 'filepath', '')
    if getattr(image, 'packed_file', None) or getattr(image, 'source', '') == 'GENERATED':
        try:
            return _queue('image:%s' % image.as_pointer(), raw, image)
        except ReferenceError:
            return 0
    return icon_for_path(raw)


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
