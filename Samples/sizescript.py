# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:27:04 2017

@author: quentinpeter
"""

from diffusion_device import process_data
from glob import glob

# Where should the results be saved
outpath = 'output'

# json files location (use glob!)
settingsfn = 'SampleData/settings.json'
metadatafn = 'SampleData/UVim*ulph_metadata.json'

settingsfn = '/Users/quentinpeter/Desktop/200 ulh_2/settings.json'
metadatafn = '/Users/quentinpeter/Desktop/200 ulh_2/4 channels_200ulh_2_metadata.json'


settings = glob(settingsfn)
metadatas = glob(metadatafn)

for sfn in settings:
    for mfn in metadatas:
        # Call function
        process_data.full_fit(sfn, mfn, outpath)
