from unittest import TestCase

from ..images import full_fit
import numpy as np
from . import __file__
import os
folder = os.path.dirname(__file__)


class TestImage(TestCase):
    def test_UV_bg(self):
        settingsfn = os.path.join(folder,
                                  'test_data/UVim300ulph_fitSettings.json')
        metadatafn = os.path.join(folder,
                                  'test_data/UVim300ulph_Metadata.json')
        radius, profiles, fits, pixel_size, images, image_type, overexposed = \
            full_fit(settingsfn, metadatafn)
        self.assertGreater(radius, 2.5e-9)
        self.assertLess(radius, 5e-9)

    def test_UV_bg_poly_2(self):
        settingsfn = os.path.join(
            folder, 'test_data/UVim300ulph_fitSettings_poly_2.json')
        metadatafn = os.path.join(
            folder, 'test_data/UVim300ulph_Metadata.json')
        radius, profiles, fits, pixel_size, images, image_type, overexposed = \
            full_fit(settingsfn, metadatafn)
        Rs, spectrum = radius
        self.assertGreater(Rs[np.argmax(spectrum)], 2.5e-9)
        self.assertLess(Rs[np.argmax(spectrum)], 5e-9)

    def test_UV_bg_poly_all(self):
        settingsfn = os.path.join(
            folder, 'test_data/UVim300ulph_fitSettings_poly_all.json')
        metadatafn = os.path.join(
            folder, 'test_data/UVim300ulph_Metadata.json')
        radius, profiles, fits, pixel_size, images, image_type, overexposed = \
            full_fit(settingsfn, metadatafn)
        Rs, spectrum = radius
        self.assertGreater(Rs[np.argmax(spectrum)], 2.5e-9)
        self.assertLess(Rs[np.argmax(spectrum)], 5e-9)

    def test_bright(self):
        settingsfn = os.path.join(
            folder, 'test_data/Brightim900ulph_fitSettings.json')
        metadatafn = os.path.join(
            folder, 'test_data/Brightim900ulph_Metadata.json')
        radius, profiles, fits, pixel_size, images, image_type, overexposed = \
            full_fit(settingsfn, metadatafn)
        self.assertGreater(radius, .2e-9)
        self.assertLess(radius, 1e-9)

    def test_film(self):
        settingsfn = os.path.join(
            folder, 'test_data/327.68ul-h-50um device_fitSettings.json')
        metadatafn = os.path.join(
            folder, 'test_data/327.68ul-h-50um device_Metadata.json')
        radius, profiles, fits, pixel_size, images, image_type, overexposed = \
            full_fit(settingsfn, metadatafn)
        self.assertGreater(np.sum(np.isfinite(radius)), 
                           np.sum(np.isnan(radius)))
        self.assertFalse(np.any(radius > 4e-9))

    def test_12pos(self):
        settingsfn = os.path.join(
            folder, 'test_data/350ulh_12pos/fitsettings.json')
        metadatafn = os.path.join(
            folder, 'test_data/350ulh_12pos/metadata.json')
        radius, profiles, fits, pixel_size, images, image_type, overexposed = \
            full_fit(settingsfn, metadatafn)
        self.assertGreater(radius, 2.5e-9)
        self.assertLess(radius, 4.5e-9)

    def test_bright_walls(self):
        settingsfn = os.path.join(
            folder, 'test_data/1tf_001tweentico75_100ulhr_Settings.json')
        metadatafn = os.path.join(
            folder, 'test_data/1tf_001tweentico75_100ulhr_Metadata.json')
        radius, profiles, fits, pixel_size, images, image_type, overexposed = \
            full_fit(settingsfn, metadatafn)
        self.assertGreater(radius, 2e-9)
        self.assertLess(radius, 4e-9)

    def test_Slice(self):
        settingsfn = os.path.join(
            folder, 'test_data/1tf_001tweentico75_100ulhr_Settings_Slice.json')
        metadatafn = os.path.join(
            folder, 'test_data/1tf_001tweentico75_100ulhr_Metadata.json')
        radius, profiles, fits, pixel_size, images, image_type, overexposed = \
            full_fit(settingsfn, metadatafn)
        self.assertGreater(radius, 2e-9)
        self.assertLess(radius, 4e-9)
