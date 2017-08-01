# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:57:15 2017

@author: quentinpeter
"""

from diffusion_device.json import createFitSettings

settingsfn = '../diffusion_device/tests/test_data/327.68ul-h-50um device_fitSettings.json'
metafn = '327.68ul-h-50um device_Metadata.json' #relative to settings
radii = [
        5e-10,
        1e-08,
        1e-10
    ] #m
rmin, rmax, rstep = radii
border = [
        150,
        None,
        None,
        None
    ] #px
framesSlices =  [
        None,
        250
    ]

ignore=0e-6 #m
firstmethod='none'
fitpos=None # None means all
flatten=True #For bright Images
nspecies = 1

createFitSettings(settingsfn, metafn, rmin, rmax, rstep, 
                      ignore, firstmethod,
                      fitpos, flatten, border, 
                      framesSlices, nspecies)