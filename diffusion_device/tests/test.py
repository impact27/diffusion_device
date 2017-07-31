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
        
        self.assertTrue(2.5e-9<radius)
        self.assertTrue(5e-9>radius)
        
    def test_UV_bg_poly(self):
        settingsfn=os.path.join(folder, 
                                'test_data/UVim300ulph_fitSettings_poly.json')
        radius, profiles, fits, lse, pixel_size = full_fit(settingsfn)
        Rs, spectrum = radius
        self.assertTrue(2.5e-9<Rs[np.argmax(spectrum)])
        self.assertTrue(5e-9>Rs[np.argmax(spectrum)])
        
    def test_bright(self):
        settingsfn=os.path.join(folder, 
                                'test_data/Brightim900ulph_fitSettings.json')
        radius, profiles, fits, lse, pixel_size = full_fit(settingsfn)
        
        self.assertTrue(.2e-9<radius)
        self.assertTrue(1e-9>radius)
        
    def test_film(self):
        settingsfn = os.path.join(folder, 
                        'test_data/327.68ul-h-50um device_fitSettings.json')
        (radii, profiles_list, 
         fits_list, LSE, pixs, oe) = full_fit(settingsfn)
        
        self.assertTrue(np.sum(np.isnan(radii)) < np.sum(np.isfinite(radii)))
        self.assertFalse(np.any(radii > 4))
        
    def test_12pos(self):
        settingsfn = os.path.join(folder, 
                        'test_data/350ulh_12pos/fitsettings.json')
        radius, profiles, fits, lse, pixel_size = full_fit(settingsfn)
        
        self.assertTrue(2.5e-9<radius)
        self.assertTrue(4e-9>radius)
        
    

        
