# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:57:15 2017

@author: quentinpeter
"""
from diffusion_device.input_files import createMetadata, metadata_fn
import os

# Data Type
# One of "4pos", "4pos_stack", "12pos", "scans"
data_type = "4pos"

# data file
image_filename = 'Data/10mgmL_Lyz_100mMphosphate_pH8_200ulh.tif'

# Height of the channel [m]
Wz = 50e-06  # m

# Width of the channel [m]
Wy = 300e-6  # m

# Flow [ulph]
Q = 200

# Reading position at the middle of the image [m]
readingpos = [
    0.0015692,
    0.0110692,
    0.0205692,
    0.04106919999999999
]  # m

# Pixe size
pixelsize = 1e-6  # m

# Date YYYMMDD
date = "20170906"

# Informations about the analyte
analyte = "Lysozyme 5mgmL sonicated diluted centrifuged"

# Informations about the buffer
buffer = "100mM phosphate pH8"

# Informations about the device
device = "m2d1t4"

# Where is the flow going for each reading position?
# For scans, only use 'u' up or 'd' down
flow_direction = ['u', 'd', 'u', 'd'] # 'u', 'd', 't', 'l'

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
exposure = 1.5  # s

# Background file, relative to the image file. None means no background
background_filename = 'bg_1500ms_m2d1t.tif'

# Exposure time of the background file (None if no background)
background_exposure = 1.5  # s

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
prof_z = None # m

########################

filename = os.path.basename(image_filename)
metafn = metadata_fn(image_filename)
createMetadata(metafn,
               filename, exposure,
               background_filename, background_exposure,
               optics_background_filename, optics_background_exposure,
               Wz, Wy, Q, readingpos, pixelsize,
               date, analyte, buffer, device,
               wallwidth, nchannels, border, data_type, flow_direction, prof_z)
