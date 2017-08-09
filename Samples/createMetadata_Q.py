# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:57:15 2017

@author: quentinpeter
"""
from diffusion_device.json import createMetadata, metadata_fn
import os
from glob import glob
import re

imfn = '../diffusion_device/tests/test_data/*.tif'
for fn in glob(imfn):

    
    bgfn = None #relative to the image file
    Wz = 5.3e-05 #m
    Wy = 100e-6 #m
    Q = float(re.findall('([\d\.]+)ul?p?_?-?h', fn)[0])
    readingpos = [
            0.0,
            0.003956,
            0.021953,
            0.0465
        ] #m
    pixelsize = 1e-6 #m
    wallwidth = 100e-6
    nchannels = 4
    
    ########################
    
    filename = os.path.basename(fn)
    metafn = metadata_fn(fn)
    createMetadata(metafn, filename, Wz, Wy, Q, readingpos, pixelsize, 
                   bgfn, wallwidth, nchannels)
