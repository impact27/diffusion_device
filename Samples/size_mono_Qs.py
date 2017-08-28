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

metadatafn = '/Users/quentinpeter/Documents/PHD/Python/People project/Therese_sizing/Diffusion sample data/170713d26b5/cam_*_Metadata.json'
settingsfn = '/Users/quentinpeter/Documents/PHD/Python/People project/Therese_sizing/Diffusion sample data/170713d26b5/cam_003ddm01cacl210nacl10t75_80ulhr_fitSettings.json'

Rs = []
Qs = []
for fn in glob(metadatafn):
    radius = ddplot.plotpos(settingsfn, fn, outpath)
    Q = float(re.findall('([\d\.]+)ul?p?_?-?h', fn)[0])
    Rs.append(radius)
    Qs.append(Q)

plt.figure()
plt.plot(Qs, np.asarray(Rs) * 1e9, 'x')
plt.xlabel('Flow Rate [ul-h]')
plt.ylabel('Radius [nm]')
plt.savefig(os.path.join(outpath, 'RvsQ.pdf'))
