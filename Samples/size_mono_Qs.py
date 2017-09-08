# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:27:04 2017

@author: quentinpeter
"""

import diffusion_device.plot as ddplot
from glob import glob
import matplotlib.pyplot as plt
import re
import numpy as np
import os

outpath = 'output'
plotpos = None

metadatafn = '170815d301b3/*_Metadata.json'
settingsfn = '170815d301b3/50nm_h2o_fitSettings.json'

Rs = []
Qs = []
for fn in glob(metadatafn):
    radius = ddplot.plotpos(settingsfn, fn, outpath)
    Q = float(re.findall('([\d\.]+)ul?p?_?-?h', fn)[0])
    Rs.append(radius)
    Qs.append(Q)
#%%
Rs = np.asarray(Rs)
if len(np.shape(Rs)) == 1:
    plt.figure()
    plt.plot(Qs, Rs * 1e9, 'x')
    plt.xlabel('Flow Rate [ul-h]')
    plt.ylabel('Radius [nm]')
    plt.savefig(os.path.join(outpath, 'RvsQ.pdf'))
else:
    arg = np.argwhere(Rs[:, 1] > 0)
    plt.figure()
    for idx in arg:
        plt.plot(Qs[idx[0]], Rs[0, 0, idx[1]], 'bx')
    plt.xlabel('Flow Rate [ul-h]')
    plt.ylabel('Radius [nm]')
    plt.savefig(os.path.join(outpath, 'RvsQ.pdf'))
