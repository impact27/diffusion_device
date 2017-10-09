# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:57:15 2017

@author: quentinpeter
"""
from diffusion_device.input_files import createMetadata
import os
from glob import glob
import numpy as np
import re

folder = 'Confocal diffusion/*/'

fns = 'Repeat*%d_0000.dat'
Pos = np.arange(13)

# Where is the flow going for each reading position?
# For scans, only use 'u' up or 'd' down
flow_direction = [
    'u' if p == 0 else 'd' for p in Pos // 2 %
    2]  # 'u', 'd', 't', 'l'

# Data Type
# One of "4pos", "4pos_stack", "12pos", "scans"
data_type = "scans"

# Flow [ulph]. If None, is taken from the image_filename. !!!Check!!!
Q = 160

# Height of the channel [m]
Wz = 25e-06  # m

# Width of the channel [m]
Wy = 100e-6  # m

# Reading position at the middle of the image [m]
readingpos = [0., 0.0035, 0.0053, 0.0086, 0.0103, 0.0186, 0.0204,
              0.0286, 0.0304, 0.0587, 0.0605, 0.0887, 0.0905]  # m

# Pixe size
pixelsize = 2e-6  # m

# Date YYYMMDD
date = "20170330"

# Informations about the analyte
analyte = "?"

# Informations about the buffer
buffer = "?"

# Informations about the device
device = "12pos"


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
exposure = None  # s

# Background file, relative to the image file. None means no background
background_filename = None

# Exposure time of the background file (None if no background)
background_exposure = None  # s

# image file to remove the background coming from the optics (Not chip related)
optics_background_filename = None

# Exposure  time of the optics background image. None if no file
optics_background_exposure = None  # s


# =============================================================================
# Multiple channels in a file
# =============================================================================

# Width of the walls [m] (Only for multiple channel in an image)
wallwidth = None  # m

# Number of channels in the image
nchannels = 1

# =============================================================================
# Scans
# =============================================================================

# profile position. None means the mean over Z is used.
prof_z = 12.5e-6  # m

########################
folder = glob(folder)
for fn in folder:
    if Q is None:
        flowrate = float(re.findall('([\d\.]+)ul?L?O?p?_?-?h', fn)[0])
    else:
        flowrate = Q
    filename = [os.path.basename(glob(
        os.path.join(fn, fns % (p,)))[0]) for p in Pos]
    metafn = os.path.join(fn, 'metadata_new.json')
    createMetadata(metafn,
                   filename, exposure,
                   background_filename, background_exposure,
                   optics_background_filename, optics_background_exposure,
                   Wz, Wy, flowrate, readingpos, pixelsize,
                   date, analyte, buffer, device, wallwidth,
                   nchannels, border, data_type, flow_direction, prof_z)
