"""Regressions for removing dark seams from Blender image reference cards.

This verifies Python thumbnail pixels and native UILayout invocation counts.
It does not pretend to capture an actual Blender screenshot.
"""
import unittest
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import test_two_workflows as T
import test_ui_revision as R

UIC = T.load('ui_components')
CACHE = T.load('preview_cache')


def setUpModule():
    T.setUpModule()


def tearDownModule():
    T.tearDownModule()


def gradient_image(w, h, axis):
    rgba = np.ones((h, w, 4), dtype=np.float32)
    ramp = np.linspace(0.12, 0.9, w if axis == 'x' else h)
    if axis == 'x':
        rgba[:, :, 0] = ramp[None, :]
    else:
        rgba[:, :, 0] = ramp[:, None]
    rgba[:, :, 1] = 0.5
    rgba[:, :, 2] = 0.3
    return NS(size=(w, h), pixels=rgba.ravel()), rgba


class PreviewR5Tests(unittest.TestCase):
    def test_ultrawide_contact_sheet_keeps_both_ends(self):
        image, source = gradient_image(360, 90, 'x')
        thumb = NS()
        CACHE._pixels_thumbnail(image, thumb, 'seamless')
        out = np.asarray(thumb.image_pixels_float).reshape(256, 256, 4)
        self.assertEqual(thumb.image_size, (256, 256))
        self.assertLess(out[128, 0, 0], 0.14)
        self.assertGreater(out[128, 255, 0], 0.88)
        self.assertTrue(np.all(out[:, :, 3] == 1))
        self.assertTrue(np.array_equal(image.pixels.reshape(source.shape), source))

    def test_portrait_keeps_top_and_bottom_and_soft_fills_sidebars(self):
        image, _ = gradient_image(90, 360, 'y')
        thumb = NS()
        CACHE._pixels_thumbnail(image, thumb, 'seamless')
        out = np.asarray(thumb.image_pixels_float).reshape(256, 256, 4)
        self.assertLess(out[0, 128, 0], .14)
        self.assertGreater(out[255, 128, 0], .88)
        self.assertGreater(out[128, 0, :3].sum(), .35)
        self.assertGreater(out[128, 255, :3].sum(), .35)

    def test_card_and_advanced_path_invoke_one_icon_instead_of_multiple(self):
        context = T.context()
        image = T.FakeImage('sample')
        p = T.props(sb_reference_image=image)
        for kind in ('PRODUCT', 'STYLE', 'PERSON', 'TARGET'):
            layout = R.RecordingLayout()
            with patch.object(sys.modules[UIC.__package__ + '.preview_cache'],
                              'icon_for_image', return_value=77) as picker:
                UIC.image_slot(layout, context, p, kind, item=NS(image=image, instruction=''), title=kind)
            self.assertEqual(sum(evt[0] == 'thumbnail' for evt in layout.log), 1)
            picker.assert_called_once_with(image, fit='seamless')
        path = R.RecordingLayout()
        with patch.object(sys.modules[UIC.__package__ + '.preview_cache'],
                          'icon_for_path', return_value=77) as picker:
            UIC.path_preview(path, context, '/tmp/look.png', 'Look')
        self.assertEqual(sum(evt[0] == 'thumbnail' for evt in path.log), 1)
        picker.assert_called_once_with('/tmp/look.png', fit='seamless')

    def test_legacy_tile_request_cannot_reintroduce_seams(self):
        img, _ = gradient_image(60, 30, 'x')
        with self.assertRaisesRegex(ValueError, 'Unsupported image preview mode'):
            CACHE._pixels_thumbnail(img, NS(), 'tile:1:2')


if __name__ == '__main__':
    unittest.main()
