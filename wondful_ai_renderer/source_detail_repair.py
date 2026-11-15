"""Validate frozen product-projection helpers for source-only native finishing."""
from __future__ import annotations

from . import material_color as MC
from .png_io import read_png


def read_helpers(projection_path, keep_path, matid_path):
    """Read exact-canvas RGBA helpers and fail closed on corrupt material metadata."""
    detail, keep, ids = (read_png(path) for path in (projection_path, keep_path, matid_path))
    if detail.shape != keep.shape or detail.shape != ids.shape or detail.shape[-1] != 4:
        raise ValueError("Projection/keep/material-ID helper dimensions do not match.")
    if not (detail[:, :, 3] > 127).any():
        raise ValueError("Product coverage is empty.")
    materials = MC.read_sidecar(MC.sidecar_path(matid_path))
    if not materials or [material.get("index") for material in materials] != list(range(len(materials))):
        raise ValueError("Material-ID metadata is missing, corrupt or not indexed uniquely.")
    labels = MC.decode_ids(ids, len(materials))
    if not (labels >= 0).any():
        raise ValueError("Material-ID pixels do not decode against the captured metadata.")
    return detail, keep, labels, materials
