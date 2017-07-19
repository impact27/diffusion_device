# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:23:24 2017

@author: quentinpeter
"""
import json
import numpy as np
import os
from tifffile import imread
import diffusion_device.four_channels_image as dd4
import diffusion_device.channel_image as ddx
from registrator.image import is_overexposed


KEY_FN = 'Image file name'
KEY_BGFN = 'Background file name'
KEY_MDFN = 'Metadata file name'
KEY_R = 'Radii[m] (min, max, step)'
KEY_WZ = 'Wz[m]'
KEY_WY = 'Wy[m]'
KEY_Q = 'Q[ulph]'
KEY_RPOS = 'Read Positions [m]'

def optional(dic, key, val):
    if val is not None:
        dic[key] = val
        
def createMetadata(metafn, fn, Wz, Wy, Q, readingpos, pixelsize, bgfn=None):
    Metadata = {}
    Metadata[KEY_FN] = fn
    optional(Metadata, KEY_BGFN, bgfn)
    Metadata[KEY_WZ] = Wz
    Metadata[KEY_WY] = Wy
    Metadata[KEY_Q] = Q
    Metadata[KEY_RPOS] = readingpos
    Metadata['Pixel Size [m]'] = pixelsize
    #Optional
    
    
    with open(metafn, 'w') as f:
        json.dump(Metadata, f, indent=4)

def createFitSettings(settingsfn, metafn, rmin, rmax, rstep, 
                      ignore=None, firstmethod=None,
                      fitpos=None, flatten=None, border=None, 
                      framesSlices=None):
    Settings = {}
    Settings[KEY_MDFN] = metafn
    Settings[KEY_R] = (rmin, rmax, rstep)
    #Optional
    optional(Settings,'Ignore Edge[m]', ignore)
    optional(Settings,'First Position Filter', firstmethod)
    optional(Settings,'Pos to fit', fitpos)
    optional(Settings,'Flatten bright field', flatten)
    optional(Settings,'Image border[px] (t, d, l, r)', border)
    #For multi frames
    optional(Settings,'Frames slice', framesSlices)
     
    with open(settingsfn, 'w') as f:
        json.dump(Settings, f, indent=4)


def default(dic, key, value):
    if key not in dic or dic[key] is None:
        dic[key] = value
        
def metadata_fn(filename):
    return os.path.splitext(filename)[0] + '_Metadata.json'

def makeabs(prefix, filename):
    if (filename is not None and 
        not os.path.isabs(filename)):
        filename = os.path.join(prefix, filename)
        filename = os.path.abspath(filename)
    return filename

def listmakeabs(prefix, filename):
    if isinstance(filename, (list, tuple)):
        return [makeabs(prefix, fn) for fn in filename]
    else:
        return makeabs(prefix, filename)

def loadJSON(settingsfn):    
    with open(settingsfn, 'r') as f:
        print()
        Settings = json.load(f)
        
    Settings[KEY_MDFN] = makeabs(os.path.dirname(settingsfn),
                                 Settings[KEY_MDFN])
        
    metadatafn = Settings[KEY_MDFN]
    
    with open(metadatafn, 'r') as f:
        Metadata = json.load(f)
        
    default(Metadata, KEY_BGFN, None)
    Metadata[KEY_BGFN] = listmakeabs(os.path.dirname(metadatafn),
                                 Metadata[KEY_BGFN])
    Metadata[KEY_FN] = listmakeabs(os.path.dirname(metadatafn),
                                 Metadata[KEY_FN])
        
    
    default(Settings, 'Ignore Edge[m]', 0)
    default(Settings, 'First Position Filter', 'none')
    default(Settings, 'Pos to fit', None)
    default(Settings, 'Flatten bright field', False)
    default(Settings, 'Image border[px] (t, d, l, r)', [None, None, None,None])
    default(Settings, 'Frames slice', [None, None])
        
    return Metadata, Settings

def full_fit(settingsfn):   
    
    Metadata, Settings = loadJSON(settingsfn)
    
    filename = Metadata[KEY_FN]
    bgfn = Metadata[KEY_BGFN]
    readingpos = Metadata[KEY_RPOS]
    Wz = Metadata[KEY_WZ]
    Wy = Metadata[KEY_WY]
    ActualFlowRate = Metadata[KEY_Q]
    pixsize = Metadata['Pixel Size [m]']
    rmin, rmax, rstep = Settings[KEY_R]
    ignore = Settings['Ignore Edge[m]']
    initmode = Settings['First Position Filter']
    fit_position_number = Settings['Pos to fit']
    flatten = Settings['Flatten bright field']
    imborder = Settings['Image border[px] (t, d, l, r)']
    framesSlice = Settings['Frames slice']
    
    test_radii=np.arange(rmin,rmax,rstep) 
    
    
    isfourpos = False
    if isinstance(filename, (list, tuple)):
        #This is a 12 pos
        ims = [imread(fn) for fn in filename]
        ims = [im[imborder[0]:imborder[1],
                  imborder[2]:imborder[3]] for im in ims]
        ims = [ims]
        
        
    else:
        # 4 pos
        isfourpos = True
        ims = imread(filename)
        if len(np.shape(ims)) == 3:
            #movie
            ims = ims[framesSlice[0]:framesSlice[1]]
        else:
            
            ims = [ims]
            
        ims = [im[imborder[0]:imborder[1], 
                  imborder[2]:imborder[3]] for im in ims]
            
    
    
    if bgfn is not None:
        if isfourpos:
            bg = imread(bgfn)
            ims = [[im, bg] for im in ims]
        else:
            bgs = [imread(fn) for fn in bgfn]
            ims[0] = [ims[0], bgs]
    
    
    
        
        
    radius_list = []
    profiles_list = []
    fits_list = []
    lse_list = []
    pixel_size_list = []
    
    for im in ims:
    
        # Infer variables
        data_dict={}
        
        # Get radius and LSE
        if isfourpos:
            radius=dd4.size_image(im, ActualFlowRate, Wz, Wy, readingpos,
                                  test_radii, data_dict=data_dict, 
                                  ignore=ignore, initmode=initmode, 
                                  fit_position_number=fit_position_number, 
                                  flatten=flatten)
        else:
            radius=ddx.size_images(im, ActualFlowRate, Wz, Wy, pixsize,
                                   readingpos, Rs=test_radii, 
                                   data_dict=data_dict, ignore=ignore,
                                   initmode=initmode)
        
        profiles, fits, lse, pixel_size = np.nan, np.nan, np.nan, np.nan
        
        if not np.isnan(radius):
            lse=np.sqrt(np.mean(np.square(data_dict['profiles'][1:]
                                        - data_dict['fits'])))
            
            #Get profiles and fit
            profiles=data_dict['profiles']
            fits=[data_dict['initprof'],*data_dict['fits']]
            pixel_size=data_dict['pixsize']
            
        radius_list.append(radius)
        profiles_list.append(profiles)
        fits_list.append(fits)
        lse_list.append(lse)
        pixel_size_list.append(pixel_size)

    if len(ims) == 1:
        return radius, profiles, fits, lse, pixel_size
    
    overexposed=[is_overexposed(im) for im in ims]
    return (np.asarray(radius_list), profiles_list, fits_list, 
            np.asarray(lse_list), np.asarray(pixel_size_list), overexposed)

