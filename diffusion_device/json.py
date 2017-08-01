# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:23:24 2017

@author: quentinpeter
"""
import json
import numpy as np
import os
from tifffile import imread
import diffusion_device.multi_channels_image as dd4
import diffusion_device.channel_image as ddx
from registrator.image import is_overexposed


KEY_MD_FN = 'Image file name'
KEY_MD_BGFN = 'Background file name'
KEY_MD_WZ = 'Wz[m]'
KEY_MD_WY = 'Wy[m]'
KEY_MD_WALLWIDTH = 'Wall Width [m]'
KEY_MD_NCHANNELS = 'Number of channels'
KEY_MD_Q = 'Q[ulph]'
KEY_MD_RPOS = 'Read Positions [m]'
KEY_MD_PIXSIZE = 'Pixel Size [m]'



KEY_STG_MDFN = 'Metadata file name'
KEY_STG_R = 'Radii[m] (min, max, step)'
KEY_STG_NSPECIES = 'Number of species to fit'
KEY_STG_IGNORE = 'Ignore Edge[m]'
KEY_STG_POS0FILTER = 'First Position Filter'
KEY_STG_FITPOS = 'Pos to fit'
KEY_STG_BFFLAT = 'Flatten bright field'
KEY_STG_BORDER = 'Image border[px] (t, d, l, r)'
KEY_STG_FRAMESSLICES = 'Frames slice'


def optional(dic, key, val):
    """Set valur in dictionnary if not None
    
    Parameters
    ----------
    dic: dictionnary
        the dictionnary
    key: string
        the key
    val: object
        Value that might be None
    """
    if val is not None:
        dic[key] = val
        
def createMetadata(metafn, fn, Wz, Wy, Q, readingpos, pixelsize, 
                   bgfn=None, wallwidth=None, nchannels=None):
    """Creates the metadata for a file name
    
    Parameters
    ----------
    metafn: path
        Where to save the metadata
    fn: path
        The path to the image relative to the metadata file
    Wz: float
        Height of the channel in [m]
    Wy: float
        Width of the channel in [m]
    Q: float
        Flow rate in [ul/h]
    readingpos; list float
        The reading positions in [m]
    pixelsize: float
        The pixel size in [m] 
    bgfn: path, default None
        If there is a background, the path relative to the metadata file
    wallwidth: float, default None
        If this is a multichannel image, the width of the wall in [m]
    nchannels: int, default None
         If this is a multichannel image, The number of channels
    """
    Metadata = {}
    Metadata[KEY_MD_FN] = fn
    optional(Metadata, KEY_MD_BGFN, bgfn)
    Metadata[KEY_MD_WZ] = Wz
    Metadata[KEY_MD_WY] = Wy
    optional(Metadata, KEY_MD_WALLWIDTH, wallwidth)
    optional(Metadata, KEY_MD_NCHANNELS, nchannels)
    Metadata[KEY_MD_Q] = Q
    Metadata[KEY_MD_RPOS] = readingpos
    Metadata[KEY_MD_PIXSIZE] = pixelsize
    #Optional
    
    
    with open(metafn, 'w') as f:
        json.dump(Metadata, f, indent=4)

def createFitSettings(settingsfn, metafn, rmin, rmax, rstep, 
                      ignore=None, firstmethod=None,
                      fitpos=None, flatten=None, border=None, 
                      framesSlices=None, nspecies=1):
    """Creates the fit settings for the fitting
    
    Parameters
    ----------
    settingsfn: path
        The name of the settings file
    metafn: path 
        path to the metadata file relative to the settings
    rmin, rmax, rstep: 3 floats
        min, max, and step for the radius[m]
    ignore: float, default None
        The border to ignore. None is 0
    firstmethod: string, default None
        The method to process the first profile. See profiles.py
    fitpos: array of ints, default None
        It this is a multichannel images, the positions to fit. None=all.
    flatten: Bool, default None
        If there is no background, should the image be flattened
        Use if there is no background at all.
    border: 4 ints, default None
        The borber to apply on the image
    framesSlices: 2 ints, default None
        If this is a movie, the slice to apply
    nspecies: int
        The number of species to fit. 0=all.
    
    """
    Settings = {}
    Settings[KEY_STG_MDFN] = metafn
    Settings[KEY_STG_R] = (rmin, rmax, rstep)
    #Optional
    optional(Settings, KEY_STG_IGNORE, ignore)
    optional(Settings, KEY_STG_POS0FILTER, firstmethod)
    optional(Settings, KEY_STG_FITPOS, fitpos)
    optional(Settings, KEY_STG_BFFLAT, flatten)
    optional(Settings, KEY_STG_BORDER, border)
    #For multi frames
    optional(Settings, KEY_STG_FRAMESSLICES, framesSlices)
    Settings[KEY_STG_NSPECIES] = nspecies
     
    with open(settingsfn, 'w') as f:
        json.dump(Settings, f, indent=4)


def default(dic, key, value):
    """Set valur in dictionnary if None
    
    Parameters
    ----------
    dic: dictionnary
        the dictionnary
    key: string
        the key
    value: object
        Value that might be None
    """
    if key not in dic or dic[key] is None:
        dic[key] = value
        
def metadata_fn(filename):
    """Create a metadata file name from an image file name
    
    Parameters
    ----------
    filename: path
        the path to an image
        
    Returns
    -------
    metadatafn: path
        path to the metadata file
    """
    return os.path.splitext(filename)[0] + '_Metadata.json'

def makeabs(prefix, filename):
    """Combine a prefix and a filename to create an absolute path.
    
    Parameters
    ----------
    prefix: path
        Prefix to combine with the file name. Can be relative.
    filename: path
        Path to the file
        
    Returns
    -------
    filename: path
        absolute path
    """
    if (filename is not None and 
        not os.path.isabs(filename)):
        filename = os.path.join(prefix, filename)
        filename = os.path.abspath(filename)
    return filename

def listmakeabs(prefix, filename):
    """Combine a prefix and a list of filenames to create absolute paths.
    
    Parameters
    ----------
    prefix: path
        Prefix to combine with the file name. Can be relative.
    filename: path or list of paths
        Path to the file
        
    Returns
    -------
    filenames: path or path list
        (list of) absolute path
    """
    if isinstance(filename, (list, tuple)):
        return [makeabs(prefix, fn) for fn in filename]
    else:
        return makeabs(prefix, filename)

def loadJSON(settingsfn): 
    """Load fit settings and metadata from a settings file
    
    Parameters
    ----------
    settingsfn: path
        path to the fit settings file
        
    Returns
    -------
    Metadata: dict
        Dictionnary containing metadata
    Settings: dict
        Dictionnary containing fit settings
    
    """
    with open(settingsfn, 'r') as f:
        print()
        Settings = json.load(f)
        
    Settings[KEY_STG_MDFN] = makeabs(os.path.dirname(settingsfn),
                                 Settings[KEY_STG_MDFN])
        
    metadatafn = Settings[KEY_STG_MDFN]
    
    with open(metadatafn, 'r') as f:
        Metadata = json.load(f)
        
    default(Metadata, KEY_MD_BGFN, None)
    default(Metadata, KEY_MD_WALLWIDTH, None)
    default(Metadata, KEY_MD_NCHANNELS, 1)
    
    if Metadata[KEY_MD_BGFN] is not None:
        Metadata[KEY_MD_BGFN] = listmakeabs(os.path.dirname(metadatafn),
                                         Metadata[KEY_MD_BGFN])
    Metadata[KEY_MD_FN] = listmakeabs(os.path.dirname(metadatafn),
                                   Metadata[KEY_MD_FN])
        
    
    default(Settings, KEY_STG_IGNORE, 0)
    default(Settings, KEY_STG_POS0FILTER, 'none')
    default(Settings, KEY_STG_FITPOS, None)
    default(Settings, KEY_STG_BFFLAT, False)
    default(Settings, KEY_STG_BORDER, [None, None, None,None])
    default(Settings, KEY_STG_FRAMESSLICES, [None, None])
    default(Settings, KEY_STG_NSPECIES, 1)
        
    return Metadata, Settings

def full_fit(settingsfn):   
    """Perform a fit with the imformations found in the settings file
    
    Parameters
    ----------
    settingsfn: path
        path to the fit settings file
        
    Returns
    -------
    radius: float or list of floats or 2x list of floats
        If 1 species:
            The fitted radius
        If several species:
            radii, spectrum: The radii and corresponding coefficients 
        If movie:
            A list of the above
    profiles: 1d or 2d list of floats
        The extracted profiles. 2d for movie.
    fits: 1d or 2d list of floats
        The Fits. 2d for movie.
    lse: float or list of floats
        The least square error. List for movie.
    pixel_size: float or list of floats
        The detected pixel size. List for movie.
    """
    
    Metadata, Settings = loadJSON(settingsfn)
    
    filename = Metadata[KEY_MD_FN]
    bgfn = Metadata[KEY_MD_BGFN]
    readingpos = Metadata[KEY_MD_RPOS]
    Wz = Metadata[KEY_MD_WZ]
    Wy = Metadata[KEY_MD_WY]
    ActualFlowRate = Metadata[KEY_MD_Q]
    pixsize = Metadata[KEY_MD_PIXSIZE]
    rmin, rmax, rstep = Settings[KEY_STG_R]
    ignore = Settings[KEY_STG_IGNORE]
    initmode = Settings[KEY_STG_POS0FILTER]
    fit_position_number = Settings[KEY_STG_FITPOS]
    flatten = Settings[KEY_STG_BFFLAT]
    imborder = Settings[KEY_STG_BORDER]
    framesSlice = Settings[KEY_STG_FRAMESSLICES]
    nspecies = Settings[KEY_STG_NSPECIES]
    test_radii=np.arange(rmin,rmax,rstep) 
    
    nchannels = Metadata[KEY_MD_NCHANNELS]
    wall_width = Metadata[KEY_MD_WALLWIDTH]
    
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
            if len(np.shape(bg)) == 3:
                #Check for full 0 
                bg = np.squeeze(bg[np.logical_not(np.all(bg == 0, (1,2)))])
    
    
    if nchannels == 1:
        if len(ims.shape) == 2:
            raise RuntimeError('Only 1 channel in 1 image. Please set "'
                               +KEY_MD_NCHANNELS + '".')
        data_dict={}
        radius=ddx.size_images(ims, ActualFlowRate, Wz, Wy, pixsize,
                                   readingpos, Rs=test_radii, bgs=bg,
                                   data_dict=data_dict, ignore=ignore,
                                   initmode=initmode, 
                                   nspecies=nspecies)
        return (radius, *read_data_dict(data_dict))
        
    else:
        def process_im(im, ignore_error=False, plotim=False):
            data_dict={}
            radius=dd4.size_image(im, ActualFlowRate, Wz, Wy, readingpos,
                                  test_radii, bg=bg, data_dict=data_dict, 
                                  ignore=ignore, initmode=initmode, 
                                  fit_position_number=fit_position_number, 
                                  flatten=flatten, nspecies=nspecies,
                                  Nprofs=nchannels, wall_width=wall_width,
                                  ignore_error=ignore_error, plotim=plotim)
            return (radius, *read_data_dict(data_dict))
        
        if len(ims.shape) == 2:
            return process_im(ims, plotim=True)
        else:
            #movie
            radius_list = []
            profiles_list = []
            fits_list = []
            lse_list = []
            pixel_size_list = []
            for im in ims:
                radius, profiles, fits, lse, pixel_size = \
                                    process_im(im, ignore_error=True)
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
    """Extract interesting data from a data dict
    
    Parameters
    ----------
    data_dict: dict
        The data dictionnary
        
    Returns
    -------
    profiles: 1d list of floats
        The extracted profiles.
    fits: 1d list of floats
        The Fits. 
    lse: float 
        The least square error. 
    pixel_size: float
        The detected pixel size
    """
    profiles, fits, lse, pixel_size = np.nan, np.nan, np.nan, np.nan
    
    if 'profiles' in data_dict and 'fits'in data_dict:
        lse=np.sqrt(np.mean(np.square(data_dict['profiles'][1:]
                                    - data_dict['fits'])))
        
        #Get profiles and fit
        profiles=data_dict['profiles']
        fits=[data_dict['initprof'],*data_dict['fits']]
        pixel_size=data_dict['pixsize']
    
    return profiles, fits, lse, pixel_size