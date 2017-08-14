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

plot.plotpos(settingsfn, metadatafn, outpath, plotpos)