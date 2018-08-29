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
settingsfn = 'martin_settings.json'
metadatafn = 'metadata.json'

settings = glob(settingsfn)
if len(settings) == 0:
    raise RuntimeError(f"Can't find {settingsfn}")
metadatas = glob(metadatafn)
if len(metadatas) == 0:
    raise RuntimeError(f"Can't find {metadatafn}")

for sfn in settings:
    for mfn in metadatas:
        # Call function
        process_data.full_fit(sfn, mfn, outpath)
