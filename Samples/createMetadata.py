# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:57:15 2017

@author: quentinpeter
"""
from diffusion_device.json import createMetadata, metadata_fn
import os

# Image file
imfn = 'Data/10mgmL_Lyz_100mMphosphate_pH8_200ulh.tif'

# Background file, relative to the image file. None means no background
bgfn = 'bg_1500ms_m2d1t.tif'

# Height of the channel [m]
Wz = 50e-06  # m

# Width of the channel [m]
Wy = 300e-6  # m

# Width of the walls [m] (Only for multiple channel in an image)
wallwidth = 100e-6  # m

# Flow [ulph]
Q = 200

# Number of channels in the image
nchannels = 4

# Reading position at the middle of the image [m]
readingpos = [
    0.0015692,
    0.0110692,
    0.0205692,
    0.04106919999999999
]  # m

# Pixe size
pixelsize = 1e-6  # m

# Border of the region of interest [px]
border = [
    None,  # Top
    None,  # Bottom
    None,  # Left
    None  # Right
]

# Exposure time
exposure = 1.5  # s

# Date YYYMMDD
date = "20170906"

# Informations about the analyte
analyte = "Lysozyme 5mgmL sonicated diluted centrifuged"

# Informations about the buffer
buffer = "100mM phosphate pH8"

########################

filename = os.path.basename(imfn)
metafn = metadata_fn(imfn)
createMetadata(metafn, filename, Wz, Wy, Q, readingpos, pixelsize,
               exposure, date, analyte, buffer,
               bgfn, wallwidth, nchannels, border, )
