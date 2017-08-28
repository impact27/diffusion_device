# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:27:04 2017

@author: quentinpeter
"""

from diffusion_device import plot


outpath = 'output'
plotpos = None

# Settings file location
#settingsfn = '../diffusion_device/tests/test_data/UVim300ulph_fitSettings.json'
#metadatafn = '../diffusion_device/tests/test_data/UVim300ulph_Metadata.json'
#settingsfn = '../diffusion_device/tests/test_data/Brightim900ulph_fitSettings.json'
#metadatafn = '../diffusion_device/tests/test_data/Brightim900ulph_Metadata.json'

settingsfn = '../diffusion_device/tests/test_data/1tf_001tweentico75_100ulhr_Settings.json'
settingsfn = '../diffusion_device/tests/test_data/1tf_001tweentico75_100ulhr_Settings_Slice.json'
metadatafn = '../diffusion_device/tests/test_data/1tf_001tweentico75_100ulhr_Metadata.json'


# Settings file name
#settingsfn='../diffusion_device/tests/test_data/350ulh_12pos/fitsettings_poly.json'
#metadatafn = '../diffusion_device/tests/test_data/350ulh_12pos/metadata.json'

# Settings file path
#settingsfn = '../diffusion_device/tests/test_data/327.68ul-h-50um device_fitSettings_poly_2.json'
#metadatafn = '../diffusion_device/tests/test_data/327.68ul-h-50um device_Metadata.json'
#plotpos = [0, 10, 15]


plot.plotpos(settingsfn, metadatafn, outpath, plotpos)
