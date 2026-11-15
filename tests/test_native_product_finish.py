"""Pure compositor contracts; Blender replay is kept as a task artifact."""
import unittest

import numpy as np

from test_316_regressions import load_package_module


NF = load_package_module("native_product_finish")


class NativeComposeTests(unittest.TestCase):
    def setUp(self):
        self.ai = np.zeros((24, 32, 4), np.uint8)
        self.ai[..., :3] = (17, 61, 113)
        self.ai[..., 3] = 137
        self.native = np.zeros_like(self.ai)
        self.native[7:17, 9:23, :3] = (220, 30, 10)
        self.native[7:17, 9:23, 3] = 255
        # An enclosed opening models wheels/spokes or transparent geometry.
        self.native[10:14, 13:19] = 0

    def test_replaces_native_footprint_and_preserves_outside_rgba(self):
        output, report = NF.compose_native(self.ai, self.native)
        self.assertTrue(np.array_equal(output[8, 10, :3], (220, 30, 10)))
        self.assertEqual(int(output[11, 15, 3]), 137)
        self.assertGreater(report["inpaint_pixels"], int((self.native[..., 3] == 0).sum() * .1))
        domain = NF.dilate(NF._filled(self.native[..., 3] > 0), 2)
        self.assertTrue(np.array_equal(output[~domain], self.ai[~domain]))
        self.assertTrue(report["outside_domain_bitexact"])

    def test_receiver_requires_both_passes_and_alpha_is_source_alpha(self):
        with self.assertRaises(ValueError):
            NF.compose_native(self.ai, self.native, shadowed=self.ai)
        receiver = np.full_like(self.ai, 255)
        shadowed = receiver.copy()
        shadowed[15:21, :, :3] = 80
        output, report = NF.compose_native(self.ai, self.native, shadowed, receiver)
        self.assertTrue(np.array_equal(output[..., 3], self.ai[..., 3]))
        self.assertGreater(report["shadow_ratio_pixels"], 0)

    def test_shape_and_empty_native_reject(self):
        with self.assertRaises(ValueError):
            NF.compose_native(self.ai, self.native[:20])
        with self.assertRaises(ValueError):
            NF.compose_native(self.ai, np.zeros_like(self.ai))

    def test_paint_allows_finish_but_never_base_colour(self):
        class Socket:
            def __init__(self): self.is_linked = False; self.default_value = 0
        class Inputs(dict):
            def get(self, key, default=None): return super().get(key, default)
        rough, colour = Socket(), Socket()
        material = type("Material", (), {"use_nodes": True, "node_tree": type("Tree", (), {
            "nodes": [type("Node", (), {"type": "BSDF_PRINCIPLED", "inputs": Inputs({"Roughness": rough, "Base Color": colour})})()]})()})()
        report = {}
        NF._apply_values(material, {"roughness": .2, "base_color": [.1, .2, .3]},
                         {"name": "body paint", "paint": True, "keep": False, "linked": True}, report)
        self.assertEqual(rough.default_value, .2)
        self.assertEqual(colour.default_value, 0)
        self.assertEqual(report["body paint"]["skipped"], ["base_color"])

    def test_alpha_agreement_rejects_projection_drift(self):
        projection = self.native.copy()
        evidence = NF._alpha_agreement(self.native, projection)
        self.assertEqual(evidence["iou"], 1.0)
        projection[..., 3] = 0
        with self.assertRaises(ValueError):
            NF._alpha_agreement(self.native, projection)


if __name__ == "__main__":
    unittest.main()
