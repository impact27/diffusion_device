# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:27:04 2017

@author: quentinpeter
"""

from diffusion_device import process_data


outpath = 'output'


# Settings file location
#settingsfn = '../diffusion_device/tests/test_data/settings_poly_all.json'
#
#metadatafn = '../diffusion_device/tests/test_data/UVim300ulph_Metadata.json'
#process_data.full_fit(settingsfn, metadatafn, outpath)


# Settings file location
settingsfn = '../diffusion_device/tests/test_data/settings.json'
#settingsfn = '../diffusion_device/tests/test_data/settings_poly_all.json'

metadatafn = '../diffusion_device/tests/test_data/UVim300ulph_Metadata.json'
process_data.full_fit(settingsfn, metadatafn, outpath)
#%%
metadatafn = '../diffusion_device/tests/test_data/Brightim900ulph_Metadata.json'
process_data.full_fit(settingsfn, metadatafn, outpath)
#%%
metadatafn = '../diffusion_device/tests/test_data/350ulh_12pos/metadata.json'
process_data.full_fit(settingsfn, metadatafn, outpath)
#%%
metadatafn = '../diffusion_device/tests/test_data/1tf_001tweentico75_100ulhr_Metadata.json'
process_data.full_fit(settingsfn, metadatafn, outpath)
#%%
metadatafn = '../diffusion_device/tests/test_data/327.68ul-h-50um device_Metadata.json'
process_data.full_fit(settingsfn, metadatafn, outpath)
#%%
metadatafn = '../diffusion_device/tests/test_data//M4-5_metadata.json'
process_data.full_fit(settingsfn, metadatafn, outpath)
#%%
metadatafn = '../diffusion_device/tests/test_data/stack_Metadata.json'
metadatafn = '../diffusion_device/tests/test_data/stack_Metadata_bright.json'

ret = process_data.full_fit(settingsfn, metadatafn, outpath)


#process_data.full_fit(settingsfn, metadatafn, outpath)

from matplotlib.pyplot import plot, imshow, colorbar, show
