# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:57:15 2017

@author: quentinpeter
"""

from diffusion_device.input_files import createFitSettings

settingsfn = ('fitSettings.json')

# Test radii
radii = [
    5e-10,  # Min [m]
    1e-08,  # Max [m]
    1e-10  # Step [m]
]  # m

# For movies, to analyse subset of frames
framesSlices = [
    None,  # First Frame
    None  # Last Frame
]

# Distance from wall ignored [m]
ignore = 0e-6

# Method to process first profile. Leave 'none' for now.
firstmethod = 'none'

# Positions to fit. None means all. Useful to ignore one of the channels
# In a multiple channel image
fitpos = None

# For bright Images, should the image be flattened?
flatten = True

# Number of spieces to fit
nspecies = 1

# Number of Z slices
Zgrid = 21

# Should the profiles be normlised?
normalise = True

# Option to only analyse a slice of the image
imslice = [
    None,  # Y Distance from center [m]
    None  # Y Width [m]
]

########################
rmin, rmax, rstep = radii
createFitSettings(settingsfn, rmin, rmax, rstep,
                  ignore, firstmethod,
                  fitpos, flatten,
                  framesSlices, nspecies,
                  Zgrid, normalise, imslice)
