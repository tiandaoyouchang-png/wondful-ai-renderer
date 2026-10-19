"""1.2.3 「最终的输出文件夹应该只有输出的图」: everything but the images goes to <output>/.wondful (hidden)."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_316_regressions import load_package_module  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "wondful_ai_renderer"
DP = load_package_module("detail_paste")


class Paths(unittest.TestCase):
    def test_raw_image_goes_to_hidden_sidecars(self):
        with tempfile.TemporaryDirectory() as d:
            p = DP.raw_path(os.path.join(d, "Wondful_S_C_1.png"))
            self.assertEqual(p, os.path.join(d, ".wondful", "sidecars", "Wondful_S_C_1_raw.png"))
            self.assertTrue(os.path.isdir(os.path.dirname(p)))

    def test_output_manager_internal_dir(self):
        src = (PKG / "output_manager.py").read_text(encoding="utf-8")
        self.assertIn('INTERNAL_DIR_NAME = ".wondful"', src)
        self.assertIn("SetFileAttributesW", src)            # hidden on Windows too
        self.assertIn('internal_dir(p.parent, "sidecars")', src)


class Writers(unittest.TestCase):
    """No writer puts a non-image next to an exported result or a work folder into the output folder."""

    def test_iou_json_and_pool_internal(self):
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        self.assertNotIn('exported.with_name(exported.stem + "_iou.json")', ops)
        self.assertIn('sidecar_path(exported, "_iou.json")', ops)
        self.assertNotIn('pool = out_dir / "多角度场景一致"', ops)
        self.assertIn('pool = internal_dir(out_dir, "多角度场景一致"', ops)
        self.assertNotIn("log_dir=str(out_dir)", ops)

    def test_only_images_beside_exports(self):
        ops = (PKG / "operators.py").read_text(encoding="utf-8")
        for line in ops.splitlines():
            if "exported.with_name(" in line:
                self.assertTrue("IoU未通过" in line or "_候选" in line, line)   # renamed image / candidate images only


@unittest.skipUnless(os.environ.get("WONDFUL_BLENDER") and os.environ.get("WONDFUL_ZIP"),
                     "set WONDFUL_BLENDER (blender binary) and WONDFUL_ZIP (release zip) for the real-Blender e2e")
class FakedEndToEnd(unittest.TestCase):
    def test_full_run_leaves_only_images(self):
        out = tempfile.mkdtemp(prefix="wondful_clean_out_")
        cmd = os.environ["WONDFUL_BLENDER"].split() + ["-b", "--factory-startup", "--python",
                                                       str(ROOT / "tools" / "e2e_render_views_bg.py"), "--",
                                                       os.environ["WONDFUL_ZIP"], out]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
        self.assertIn("[e2e] OK", r.stdout[-3000:], r.stdout[-3000:])


if __name__ == "__main__":
    unittest.main()
