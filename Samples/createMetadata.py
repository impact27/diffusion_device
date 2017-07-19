# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:57:15 2017

@author: quentinpeter
"""
from diffusion_device.json import createMetadata, metadata_fn
import os

imfn = '../diffusion_device/tests/test_data/327.68ul-h-50um device.tif'
bgfn = None #relative to the image file
Wz = 5.3e-05 #m
Wy = 100e-6 #m
Q = 327.68 #ulph
readingpos = [
        0.0,
        0.003956,
        0.021953,
        0.0465
    ] #m
pixelsize = 1e-6 #m

########################

filename = os.path.basename(imfn)
metafn = metadata_fn(imfn)
createMetadata(metafn, filename, Wz, Wy, Q, readingpos,pixelsize, bgfn)

