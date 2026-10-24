"""The proxy fallback must preserve the same source reference under image limits."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module

CC = load_package_module("codex_client")
PNG = load_package_module("png_io")


class ProxyReferenceContract(unittest.TestCase):
    def test_camera_and_recorded_scene_stay_standalone_when_identity_refs_are_packed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def picture(name, color):
                path = root / name
                pixels = np.full((48, 80, 3), color, dtype=np.uint8)
                PNG.write_png(str(path), pixels)
                return str(path.resolve())

            camera = picture("camera_base.png", 80)
            source = picture("scene_layout_reference.png", 160)
            products = [picture(f"identity_{i}.png", 20 + i * 10) for i in range(7)]
            roles = {source: ("scene_reference", "已记录的场景结构与光照来源")}
            roles.update({p: ("product", "同一产品身份参考") for p in products})
            seen = {}

            def execute(**kwargs):
                seen.update(kwargs)
                (root / "out.png").write_bytes(Path(camera).read_bytes())
                return "WONDFUL_IMAGEGEN_OK"

            with patch.object(CC, "_exec_turn", execute):
                result = CC.generate_image(
                    prompt="保持当前相机与产品投影。", reference_paths=[camera, source] + products,
                    reference_roles=roles, output_path=str(root / "out.png"), cwd=str(root),
                    edit_base_path=camera,
                )
            submitted = seen["image_paths"]
            self.assertLessEqual(len(submitted), 5)
            self.assertEqual(submitted[0], camera)
            self.assertIn(source, submitted)
            merged = result["collage"]
            cells = [cell for sheet in merged.get("sheets", []) for cell in sheet.get("cells", [])]
            retained = {Path(p).name for p in submitted} | {cell["file"] for cell in cells}
            self.assertTrue({Path(p).name for p in products}.issubset(retained))
            self.assertNotIn(Path(source).name, {cell["file"] for cell in cells})
            self.assertFalse(any(item.get("role") == "product" for item in merged.get("text_only", [])))


if __name__ == "__main__":
    unittest.main()
