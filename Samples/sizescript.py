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
settingsfn = 'SampleData/fitSettings.json'
metadatafn = 'SampleData/UVim300ulph_Metadata.json'


settings = glob(settingsfn)
metadatas = glob(metadatafn)

for sfn in settings:
    for mfn in metadatas:
        # Call function
        process_data.full_fit(settingsfn, metadatafn, outpath)
    