# Wondful UI-r5 — seamless image previews

## Fix

- Removes the visible vertical panel-colour seam between adjacent square thumbnail crops.
- Adds one compensating separator between each pair of tiles, calibrated to Blender's built-in preview padding.
- Applies the same behavior to main image cards and advanced file-path previews.
- Keeps the original image pixels, crop order, zoom action, and all other UI-r4 behavior unchanged.

## Verification

- 152 offline tests pass, including two-tile, three-tile, and path-preview seam compensation checks.
- Python compile and ZIP integrity checks pass.
- Blender is not installed in this environment; confirm the appearance in the target Blender version before release.
