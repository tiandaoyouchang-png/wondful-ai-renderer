"""Frozen source-projection helper validation only; local image repair was retired."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_316_regressions import load_package_module


R = load_package_module("source_detail_repair")
MC = load_package_module("material_color")
PNG = load_package_module("png_io")


class HelperValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.detail = np.zeros((24, 48, 4), np.uint8)
        self.detail[3:-3, 5:-5] = (50, 90, 180, 255)
        self.keep = np.zeros_like(self.detail)
        self.ids = np.zeros_like(self.detail)
        self.materials = [{"index": 0, "name": "body paint", "kind": "reflective"}]
        self.ids[self.detail[:, :, 3] > 0] = (*MC.palette_colour(0), 255)
        self.paths = [self.root / name for name in ("detail.png", "keep.png", "matid.png")]
        for path, pixels in zip(self.paths, (self.detail, self.keep, self.ids)):
            PNG.write_png(path, pixels)
        Path(str(self.paths[2]) + ".json").write_text(json.dumps({"materials": self.materials}))

    def test_reads_exact_rgba_helpers_and_decodes_ids(self):
        detail, keep, labels, materials = R.read_helpers(*self.paths)
        np.testing.assert_array_equal(detail, self.detail)
        np.testing.assert_array_equal(keep, self.keep)
        self.assertEqual(materials, self.materials)
        self.assertTrue((labels[self.detail[:, :, 3] > 0] == 0).all())

    def test_rejects_corrupt_metadata_or_mismatched_dimensions(self):
        sidecar = Path(str(self.paths[2]) + ".json")
        sidecar.write_text("bad json")
        with self.assertRaises(ValueError):
            R.read_helpers(*self.paths)
        sidecar.write_text(json.dumps({"materials": self.materials}))
        PNG.write_png(self.paths[1], self.keep[:12])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            R.read_helpers(*self.paths)


if __name__ == "__main__":
    unittest.main()
