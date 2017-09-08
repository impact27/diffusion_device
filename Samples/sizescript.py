# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:27:04 2017

@author: quentinpeter
"""

from diffusion_device import plot

# Where should the results be saved
outpath = 'output'

# What positions should be plotted if movie (None = All)
plotpos = None

# json files location
settingsfn = 'SampleData/UVim300ulph_fitSettings.json'
metadatafn = 'SampleData/UVim300ulph_Metadata.json'

# Call function
plot.plotpos(settingsfn, metadatafn, outpath, plotpos)
