# -*- coding: utf-8 -*-
from diffusion_device.json import createFitSettings, createMetadata, metadata_fn
import os

imfn = '../diffusion_device/tests/test_data/327.68ul-h-50um device.tif'
settingsfn = '../diffusion_device/tests/test_data/327.68ul-h-50um device_fitSettings.json'
Wz = 5.3e-05
Wy = 100e-6
Q = 327.68
readingpos = [
        0.0,
        0.003956,
        0.021953,
        0.0465
    ]
radii = [
        5e-10,
        1e-08,
        1e-10
    ]
rmin, rmax, rstep = radii
border = [
        150,
        None,
        None,
        None
    ]
frameSlices =  [
        None,
        250
    ]


filename = os.path.basename(imfn)
metafn = metadata_fn(imfn)
createMetadata(metafn, filename, Wz, Wy, Q, readingpos,1)
createFitSettings(settingsfn, os.path.basename(metafn), rmin, rmax, rstep, 
                  framesSlices=frameSlices, border=border)
#%%
imfn='../diffusion_device/tests/test_data/UVim300ulph.tif'
settingsfn='../diffusion_device/tests/test_data/UVim300ulph_fitSettings.json'
Q = 300
readingpos = [
        0.0,
        0.002556,
        0.021953,
        0.0451
    ]
filename = os.path.basename(imfn)
metafn = metadata_fn(imfn)
createMetadata(metafn, filename, Wz, Wy, Q, readingpos,1, bgfn= "UVbg.tif")
createFitSettings(settingsfn, os.path.basename(metafn), rmin, rmax, rstep)

#%%
imfn='../diffusion_device/tests/test_data/Brightim900ulph.tif'
settingsfn='../diffusion_device/tests/test_data/Brightim900ulph_fitSettings.json'
Q = 900
filename = os.path.basename(imfn)
metafn = metadata_fn(imfn)
createMetadata(metafn, filename, Wz, Wy, Q, readingpos,1)
createFitSettings(settingsfn, os.path.basename(metafn), rmin, rmax, rstep)
#%%
#File Name. If using UV, [image filename, background filename]
mdfn='../diffusion_device/tests/test_data/350ulh_12pos/metadata.json'
settingsfn='../diffusion_device/tests/test_data/350ulh_12pos/fitsettings.json'


imfn = ['im_0.tif',
 'im_1.tif',
 'im_2.tif',
 'im_3.tif',
 'im_4.tif',
 'im_5.tif',
 'im_6.tif',
 'im_7.tif',
 'im_8.tif',
 'im_9.tif',
 'im_10.tif',
 'im_11.tif',
 'im_12.tif']

bgfn = ['bg_0.tif',
 'bg_1.tif',
 'bg_2.tif',
 'bg_3.tif',
 'bg_4.tif',
 'bg_5.tif',
 'bg_6.tif',
 'bg_7.tif',
 'bg_8.tif',
 'bg_9.tif',
 'bg_10.tif',
 'bg_11.tif',
 'bg_12.tif']

#Settings
Wz=53*1e-6 #m Height of the device
Wy=300*1e-6 #m Width of the device
rmin=.5e-9#m test radii
rmax=10e-9#m test radii
rstep=.5e-9#m test radii
Q=350/2 #ulph.
pixsize=.84e-6#pixel sixe in m

#Advanced settings
ignore=10e-6 #m Distance from the sides to ignore

readingpos = [ 0.    ,  0.0035,  0.0053,  0.0086,  0.0103,  0.0186,  0.0204,
        0.0286,  0.0304,  0.0587,  0.0605,  0.0887,  0.0905]

createMetadata(mdfn, imfn, Wz, Wy, Q, readingpos, pixelsize=pixsize, bgfn=bgfn)
createFitSettings(settingsfn, os.path.basename(mdfn), rmin, rmax, rstep, 
                  ignore=ignore)