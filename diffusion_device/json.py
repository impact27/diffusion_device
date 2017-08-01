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
KEY_NSPECIES = 'Number of species to fit'
KEY_PIXSIZE = 'Pixel Size [m]'
KEY_IGNORE = 'Ignore Edge[m]'
KEY_NCHANNELS = 'Number of channels'
KEY_POS0FILTER = 'First Position Filter'
KEY_FITPOS = 'Pos to fit'
KEY_BFFLAT = 'Flatten bright field'
KEY_BORDER = 'Image border[px] (t, d, l, r)'
KEY_FRAMESSLICES = 'Frames slice'
KEY_WALLWIDTH = 'Wall Width [m]'

def optional(dic, key, val):
    if val is not None:
        dic[key] = val
        
def createMetadata(metafn, fn, Wz, Wy, Q, readingpos, pixelsize, 
                   bgfn=None, wallwidth=None, nchannels=None):
    Metadata = {}
    Metadata[KEY_FN] = fn
    optional(Metadata, KEY_BGFN, bgfn)
    Metadata[KEY_WZ] = Wz
    Metadata[KEY_WY] = Wy
    optional(Metadata, KEY_WALLWIDTH, wallwidth)
    optional(Metadata, KEY_NCHANNELS, nchannels)
    Metadata[KEY_Q] = Q
    Metadata[KEY_RPOS] = readingpos
    Metadata[KEY_PIXSIZE] = pixelsize
    #Optional
    
    
    with open(metafn, 'w') as f:
        json.dump(Metadata, f, indent=4)

def createFitSettings(settingsfn, metafn, rmin, rmax, rstep, 
                      ignore=None, firstmethod=None,
                      fitpos=None, flatten=None, border=None, 
                      framesSlices=None, nspecies=1):
    Settings = {}
    Settings[KEY_MDFN] = metafn
    Settings[KEY_R] = (rmin, rmax, rstep)
    #Optional
    optional(Settings, KEY_IGNORE, ignore)
    optional(Settings, KEY_POS0FILTER, firstmethod)
    optional(Settings, KEY_FITPOS, fitpos)
    optional(Settings, KEY_BFFLAT, flatten)
    optional(Settings, KEY_BORDER, border)
    #For multi frames
    optional(Settings, KEY_FRAMESSLICES, framesSlices)
    Settings[KEY_NSPECIES] = nspecies
     
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
    default(Metadata, KEY_WALLWIDTH, None)
    default(Metadata, KEY_NCHANNELS, 1)
    
    if Metadata[KEY_BGFN] is not None:
        Metadata[KEY_BGFN] = listmakeabs(os.path.dirname(metadatafn),
                                         Metadata[KEY_BGFN])
    Metadata[KEY_FN] = listmakeabs(os.path.dirname(metadatafn),
                                   Metadata[KEY_FN])
        
    
    default(Settings, KEY_IGNORE, 0)
    default(Settings, KEY_POS0FILTER, 'none')
    default(Settings, KEY_FITPOS, None)
    default(Settings, KEY_BFFLAT, False)
    default(Settings, KEY_BORDER, [None, None, None,None])
    default(Settings, KEY_FRAMESSLICES, [None, None])
    default(Settings, KEY_NSPECIES, 1)
        
    return Metadata, Settings

def full_fit(settingsfn):   
    
    Metadata, Settings = loadJSON(settingsfn)
    
    filename = Metadata[KEY_FN]
    bgfn = Metadata[KEY_BGFN]
    readingpos = Metadata[KEY_RPOS]
    Wz = Metadata[KEY_WZ]
    Wy = Metadata[KEY_WY]
    ActualFlowRate = Metadata[KEY_Q]
    pixsize = Metadata[KEY_PIXSIZE]
    rmin, rmax, rstep = Settings[KEY_R]
    ignore = Settings[KEY_IGNORE]
    initmode = Settings[KEY_POS0FILTER]
    fit_position_number = Settings[KEY_FITPOS]
    flatten = Settings[KEY_BFFLAT]
    imborder = Settings[KEY_BORDER]
    framesSlice = Settings[KEY_FRAMESSLICES]
    nspecies = Settings[KEY_NSPECIES]
    test_radii=np.arange(rmin,rmax,rstep) 
    
    nchannels = Metadata[KEY_NCHANNELS]
    wall_width = Metadata[KEY_WALLWIDTH]
    
    #load images
    if isinstance(filename, (list, tuple)):
        ims = np.asarray([imread(fn) for fn in filename])
    else:
        ims = imread(filename)
        if len(np.shape(ims)) == 3:
            #Check for full 0 
            ims = np.squeeze(ims[np.logical_not(np.all(ims == 0, (1,2)))])
            
        if len(np.shape(ims)) == 3:
            #movie
            ims = ims[framesSlice[0]:framesSlice[1]]            
      
    #Remove Border
    ims = ims[..., imborder[0]:imborder[1],
                   imborder[2]:imborder[3]]
    
    #load background
    bg = None
    if bgfn is not None:
        if isinstance(bgfn, (list, tuple)):
            bg = np.asarray([imread(fn) for fn in bgfn])
        else:
            bg = imread(bgfn)
    
    
    if nchannels == 1:
        if len(ims.shape) == 2:
            raise RuntimeError('Only 1 channel in 1 image. Please set "'
                               +KEY_NCHANNELS + '".')
        data_dict={}
        radius=ddx.size_images(ims, ActualFlowRate, Wz, Wy, pixsize,
                                   readingpos, Rs=test_radii, bgs=bg,
                                   data_dict=data_dict, ignore=ignore,
                                   initmode=initmode, 
                                   nspecies=nspecies)
        return (radius, *read_data_dict(data_dict))
        
    else:
        def process_im(im):
            data_dict={}
            radius=dd4.size_image(im, ActualFlowRate, Wz, Wy, readingpos,
                                  test_radii, bg=bg, data_dict=data_dict, 
                                  ignore=ignore, initmode=initmode, 
                                  fit_position_number=fit_position_number, 
                                  flatten=flatten, nspecies=nspecies,
                                  Nprofs=nchannels, wall_width=wall_width)
            return (radius, *read_data_dict(data_dict))
        
        if len(ims.shape) == 2:
            return process_im(ims)
        else:
            #movie
            radius_list = []
            profiles_list = []
            fits_list = []
            lse_list = []
            pixel_size_list = []
            for im in ims:
                radius, profiles, fits, lse, pixel_size =  process_im(im)
                radius_list.append(radius)
                profiles_list.append(profiles)
                fits_list.append(fits)
                lse_list.append(lse)
                pixel_size_list.append(pixel_size)
    
            overexposed=[is_overexposed(im) for im in ims]
            return (np.asarray(radius_list), profiles_list, fits_list, 
                    np.asarray(lse_list), np.asarray(pixel_size_list), 
                    overexposed)
    
    
    

def read_data_dict(data_dict):
    profiles, fits, lse, pixel_size = np.nan, np.nan, np.nan, np.nan
    
    if 'profiles' in data_dict and 'fits'in data_dict:
        lse=np.sqrt(np.mean(np.square(data_dict['profiles'][1:]
                                    - data_dict['fits'])))
        
        #Get profiles and fit
        profiles=data_dict['profiles']
        fits=[data_dict['initprof'],*data_dict['fits']]
        pixel_size=data_dict['pixsize']
    
    return profiles, fits, lse, pixel_size