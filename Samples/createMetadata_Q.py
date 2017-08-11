# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:57:15 2017

@author: quentinpeter
"""
from diffusion_device.json import createMetadata, metadata_fn
import os
from glob import glob
import re

imfn = '/Users/quentinpeter/Documents/PHD/Python/People project/Therese_sizing/Diffusion sample data/170713d26b5/cam_*r.tif'
for fn in glob(imfn):    
    bgfn = None #relative to the image file
    Wz = 2.663e-05 #m
    Wy = 0.0001 #m
    Q = float(re.findall('([\d\.]+)ul?p?_?-?h', fn)[0])
    readingpos = [
        .0013596,
        .0096404,
        .0203596,
        .0396404
    ] #m
    pixelsize = 1.6e-06 #m
    wallwidth = 100e-6
    nchannels = 4
    
    ########################
    
    filename = os.path.basename(fn)
    metafn = metadata_fn(fn)
    createMetadata(metafn, filename, Wz, Wy, Q, readingpos, pixelsize, 
                   bgfn, wallwidth, nchannels)
