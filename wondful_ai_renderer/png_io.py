"""PNG read/write with numpy + zlib only (no bpy, Pillow or OpenCV): safe on worker threads.

The reader handles every non-interlaced PNG Blender and the image services write:
8/16-bit gray, gray+alpha, RGB, RGBA and 8-bit palette. Row filters are undone
along anti-diagonals (pixel x+y = const), so each step is one vectorised numpy
operation instead of a per-byte Python loop: Paeth/Average rows of a 1280×720
image decode in well under a second.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunks(data: bytes):
    pos = 8
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        yield kind, data[pos + 8:pos + 8 + length]
        if kind == b"IEND":
            return
        pos += 12 + length


def _unfilter(rows: np.ndarray, h: int, w: int, bpp: int) -> np.ndarray:
    """Undo PNG filters. ``rows`` is (h, 1 + w*bpp) uint8; returns (h, w, bpp) uint8."""
    ftypes = rows[:, 0].astype(np.int16)
    data = rows[:, 1:].reshape(h, w, bpp).astype(np.int16)
    if not np.any(ftypes):
        return data.astype(np.uint8)
    if np.all((ftypes == 0) | (ftypes == 1)):
        out = data.copy()
        sub = ftypes == 1
        out[sub] = np.cumsum(data[sub], axis=1) & 0xFF
        return out.astype(np.uint8)
    # General case: wavefront over anti-diagonals; padded so neighbours are always defined.
    out = np.zeros((h + 1, w + 1, bpp), dtype=np.int16)
    for d in range(h + w - 1):
        y0 = max(0, d - (w - 1))
        y1 = min(h - 1, d)
        ys = np.arange(y0, y1 + 1)
        xs = d - ys
        a = out[ys + 1, xs]        # left
        b = out[ys, xs + 1]        # up
        c = out[ys, xs]            # up-left
        f = ftypes[ys][:, None]
        p = a + b - c
        pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
        paeth = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
        pred = np.where(f == 1, a, np.where(f == 2, b, np.where(f == 3, (a + b) >> 1,
                                                                  np.where(f == 4, paeth, 0))))
        out[ys + 1, xs + 1] = (data[ys, xs] + pred) & 0xFF
    return out[1:, 1:].astype(np.uint8)


def read_png(path) -> np.ndarray:
    """Decode a PNG to an (H, W, 4) uint8 RGBA array."""
    data = Path(path).read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"not a PNG: {path}")
    ihdr, idat, plte, trns = None, [], None, None
    for kind, chunk in _chunks(data):
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            idat.append(chunk)
        elif kind == b"PLTE":
            plte = np.frombuffer(chunk, dtype=np.uint8).reshape(-1, 3)
        elif kind == b"tRNS":
            trns = chunk
    if ihdr is None:
        raise ValueError("PNG has no IHDR")
    w, h, depth, ctype, _comp, _filt, interlace = ihdr
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(ctype)
    if channels is None or interlace or depth not in (8, 16) or (ctype == 3 and depth != 8):
        raise ValueError(f"unsupported PNG layout (type {ctype}, depth {depth}, interlace {interlace})")
    bps = depth // 8
    bpp = channels * bps
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), dtype=np.uint8)
    rows = raw[: h * (1 + w * bpp)].reshape(h, 1 + w * bpp)
    px = _unfilter(rows, h, w, bpp)
    if bps == 2:
        px = px.reshape(h, w, channels, 2)[..., 0]  # big-endian: keep the high byte
    else:
        px = px.reshape(h, w, channels)
    rgba = np.full((h, w, 4), 255, dtype=np.uint8)
    if ctype == 3:
        if plte is None:
            raise ValueError("palette PNG without PLTE")
        idx = px[..., 0]
        rgba[..., :3] = plte[np.minimum(idx, len(plte) - 1)]
        if trns:
            alpha = np.full(256, 255, dtype=np.uint8)
            t = np.frombuffer(trns, dtype=np.uint8)
            alpha[: len(t)] = t
            rgba[..., 3] = alpha[idx]
    elif channels == 1:
        rgba[..., :3] = px
    elif channels == 2:
        rgba[..., :3] = px[..., :1]
        rgba[..., 3] = px[..., 1]
    elif channels == 3:
        rgba[..., :3] = px
    else:
        rgba[:] = px
    return rgba


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def write_png(path, array, level: int = 6) -> str:
    """Write an (H, W, 4) RGBA or (H, W, 3) RGB uint8 array as an 8-bit PNG (Up filter)."""
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.uint8))
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError("expected an HxWx3 or HxWx4 array")
    h, w, c = arr.shape
    flat = arr.reshape(h, w * c)
    up = flat.copy()
    up[1:] = flat[1:] - flat[:-1]  # uint8 wraps: PNG "Up" filter, compresses photos well
    rows = np.empty((h, 1 + w * c), dtype=np.uint8)
    rows[:, 0] = 2
    rows[0, 0] = 0
    rows[:, 1:] = up
    rows[0, 1:] = flat[0]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (PNG_SIGNATURE
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6 if c == 4 else 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(rows.tobytes(), level))
            + _chunk(b"IEND", b""))
    target.write_bytes(data)
    return str(target)
