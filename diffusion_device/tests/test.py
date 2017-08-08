from unittest import TestCase

from diffusion_device.json import full_fit
import numpy as np
from . import __file__ 
import os
folder = os.path.dirname(__file__)


class TestImage(TestCase):
    def test_UV_bg(self):
        settingsfn=os.path.join(folder, 
                                'test_data/UVim300ulph_fitSettings.json')
        radius, profiles, fits, lse, pixel_size = full_fit(settingsfn)
        self.assertGreater(radius, 2.5e-9)
        self.assertLess(radius, 5e-9)
        
    def test_UV_bg_poly_2(self):
        settingsfn=os.path.join(folder, 
                            'test_data/UVim300ulph_fitSettings_poly_2.json')
        radius, profiles, fits, lse, pixel_size = full_fit(settingsfn)
        Rs, spectrum = radius
        self.assertGreater(Rs[np.argmax(spectrum)], 2.5e-9)
        self.assertLess(Rs[np.argmax(spectrum)], 5e-9)
        
    def test_UV_bg_poly_all(self):
        settingsfn=os.path.join(folder, 
                            'test_data/UVim300ulph_fitSettings_poly_all.json')
        radius, profiles, fits, lse, pixel_size = full_fit(settingsfn)
        Rs, spectrum = radius
        self.assertGreater(Rs[np.argmax(spectrum)], 2.5e-9)
        self.assertLess(Rs[np.argmax(spectrum)], 5e-9)
        
    def test_bright(self):
        settingsfn=os.path.join(folder, 
                                'test_data/Brightim900ulph_fitSettings.json')
        radius, profiles, fits, lse, pixel_size = full_fit(settingsfn)
        self.assertGreater(radius, .2e-9)
        self.assertLess(radius, 1e-9)
        
    def test_film(self):
        settingsfn = os.path.join(folder, 
                        'test_data/327.68ul-h-50um device_fitSettings.json')
        (radii, profiles_list, 
         fits_list, LSE, pixs, oe) = full_fit(settingsfn)
        self.assertGreater(np.sum(np.isfinite(radii)), np.sum(np.isnan(radii)))
        self.assertFalse(np.any(radii > 4e-9))
        
    def test_12pos(self):
        settingsfn = os.path.join(folder, 
                        'test_data/350ulh_12pos/fitsettings.json')
        radius, profiles, fits, lse, pixel_size = full_fit(settingsfn)
        self.assertGreater(radius, 2.5e-9)
        self.assertLess(radius, 4.5e-9)
        
    def test_bright_walls(self):
        settingsfn=os.path.join(folder, 
                        'test_data/1tf_001tweentico75_100ulhr_Settings.json')
        radius, profiles, fits, lse, pixel_size = full_fit(settingsfn)
        self.assertGreater(radius, 2e-9)
        self.assertLess(radius, 4e-9)
    

        
