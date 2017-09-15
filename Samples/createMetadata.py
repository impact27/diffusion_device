# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:57:15 2017

@author: quentinpeter
"""
from diffusion_device.input_files import createMetadata, metadata_fn
import os
from glob import glob
import re

# Data Type
# One of "4pos", "4pos_stack", "12pos", "scans"
data_type = "4pos"

# data file
# Will be searched with glob. So can match several files.
image_filename = 'SampleData/UVim*ulph.tif'

# Flow [ulph]. If None, is taken from the image_filename. !!!Check!!!
Q = None

# Height of the channel [m]
Wz = 53e-06  # m

# Width of the channel [m]
Wy = 100e-6  # m

# Reading position at the middle of the image [m]
readingpos = [
    0.0,
    0.002556,
    0.021953,
    0.0451
]  # m

# Pixe size
pixelsize = .833e-6  # m

# Date YYYMMDD
date = "20170906"

# Informations about the analyte
analyte = "15uM BSA"

# Informations about the buffer
buffer = "100mM phosphate pH8"

# Informations about the device
device = "Kadi100x50"

# Where is the flow going for each reading position?
# For scans, only use 'u' up or 'd' down
flow_direction = ['u', 'd', 'u', 'd']  # 'u', 'd', 't', 'l'

# =============================================================================
# Image infos
# =============================================================================

# Border of the region of interest [px]
border = [
    None,  # Top
    None,  # Bottom
    None,  # Left
    None  # Right
]

# Exposure time of the image file
exposure = 1  # s

# Background file, relative to the image file. None means no background
background_filename = 'UVbg.tif'

# Exposure time of the background file (None if no background)
background_exposure = 1  # s

# image file to remove the background coming from the optics (Not chip related)
optics_background_filename = None

# Exposure  time of the optics background image. None if no file
optics_background_exposure = None  # s


# =============================================================================
# Multiple channels in a file
# =============================================================================

# Width of the walls [m] (Only for multiple channel in an image)
wallwidth = 100e-6  # m

# Number of channels in the image
nchannels = 4

# =============================================================================
# Scans
# =============================================================================

# profile position. None means the mean over Z is used.
prof_z = None  # m

########################

for fn in glob(image_filename):
    if Q is None:
        flowrate = float(re.findall('([\d\.]+)ul?L?O?p?_?-?h', fn)[0])
    else:
        flowrate = Q
    filename = os.path.basename(fn)
    metafn = metadata_fn(fn)
    createMetadata(metafn,
                   filename, exposure,
                   background_filename, background_exposure,
                   optics_background_filename, optics_background_exposure,
                   Wz, Wy, flowrate, readingpos, pixelsize,
                   date, analyte, buffer, device, wallwidth,
                   nchannels, border, data_type, flow_direction, prof_z)
