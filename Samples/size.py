# -*- coding: utf-8 -*-

"""
Created on Wed Apr  5 16:58:39 2017

@author: quentinpeter
"""
from diffusion_device import plot

import argparse
parser = argparse.ArgumentParser(description='4 position single image sizing.')
parser.add_argument('settingsfn',
                    help="The settings json file")
parser.add_argument('metadatafn',
                    help="The metadata json file")
parser.add_argument('-o', '--output', dest='output', default=None,
                    help="The output folder")
parser.add_argument('--plot_pos', dest='plotpos', default=None, type=int,
                    nargs='+', help="Plot positions if stack")

args = parser.parse_args()
settingsfn = args.settingsfn
metadatafn = args.metadatafn
outpath = args.output
plotpos = args.plotpos

#outpath = 'output'
#plotpos = None

#Settings file location
#settingsfn='../diffusion_device/tests/test_data/UVim300ulph_fitSettings_poly_all.json'
#metadatafn = '../diffusion_device/tests/test_data/UVim300ulph_Metadata.json'
#settingsfn='../diffusion_device/tests/test_data/Brightim900ulph_fitSettings.json'
#metadatafn = '../diffusion_device/tests/test_data/Brightim900ulph_Metadata.json'


#Settings file name
#settingsfn='../diffusion_device/tests/test_data/350ulh_12pos/fitsettings_poly.json'
#metadatafn = '../diffusion_device/tests/test_data/350ulh_12pos/metadata.json'

#Settings file path
#settingsfn = '../diffusion_device/tests/test_data/327.68ul-h-50um device_fitSettings_poly_2.json'
#metadatafn = '../diffusion_device/tests/test_data/327.68ul-h-50um device_Metadata.json'
#plotpos = [0, 10, 15]



plot.plotpos(settingsfn, metadatafn, outpath, plotpos)