# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:32:20 2017

@author: quentinpeter
"""

from tifffile import imread
import numpy as np

from . import multi_channels_image as dd4
from . import channel_image as ddx
from . import keys, input_files
from . import profile as dp

def full_fit(settingsfn, metadatafn):
    """Perform a fit with the imformations found in the settings file

    Parameters
    ----------
    settingsfn: path
        path to the fit settings file
    metadatafn: path
        path to the metadata file

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

    # Get infos
    metadata = input_files.loadMetadata(metadatafn)
    settings = input_files.loadSettings(settingsfn)
    
    # Open images
    filename = metadata[keys.KEY_MD_FN]
    imborder = metadata[keys.KEY_MD_BORDER]
    images = load_images(filename, imborder)
    
    # Get type
    image_type = input_files.getType(metadata, images.shape)
    
    # Process images
    images, pixel_size, centers = process_images(
            images, image_type, metadata, settings)
    
    # Get profiles
    profiles = get_profiles(
            images, image_type, pixel_size, centers, metadata, settings)
    
    radius, fits = size_profiles(
            profiles, image_type, pixel_size, metadata, settings)
    
    return radius, profiles, fits, pixel_size, images, image_type

def size_profiles(profiles, image_type, pixel_size, metadata, settings):
    if image_type == '4pos_stack':
        radius = np.zeros(len(pixel_size))
        fits = []
        for i, (profs, pxs) in zip(profiles, pixel_size):
            radius[i], fit = size_profiles(
                    profs, '4pos', pxs, metadata, settings)
            fits.append(fit)
            
    else:
        fits = np.zeros_like(profiles)
        radius = dp.size_profiles(profiles, pixel_size, metadata, settings,
                                  fits=fits)
    return radius, fits
        
def get_profiles(images, image_type, pixel_size, centers, metadata, settings):
    
    channel_width = metadata[keys.KEY_MD_WY]
    imslice = settings[keys.KEY_STG_SLICE]
    ignore = settings[keys.KEY_STG_IGNORE]
    
    if image_type == '12pos':
        profiles = np.zeros((len(images), channel_width/pixel_size))
        for i, (im, center) in enumerate(zip(images, centers)):
            profiles[i] = ddx.extract_profile(
                    im, pixel_size, channel_width, center)
    
    elif image_type == '4pos':
        profiles = dd4.extract_profiles(
                images, centers, channel_width, ignore, pixel_size,
                imslice=imslice)
    
    elif image_type == '4pos_stack':
        profiles = [get_profiles(im, '4pos', pxs, cnt, metadata, settings)
                    for im, pxs, cnt in zip(images, pixel_size, centers)]
    
    return profiles

def process_images(images, image_type, metadata, settings):
    
    backgrounds = None
    background_fn = metadata[keys.KEY_MD_BGFN]
    if background_fn is not None:
        imborder = metadata[keys.KEY_MD_BORDER]
        backgrounds = load_images(background_fn, imborder)
    
    if image_type == '12pos':
        images, pixel_size, centers = ddx.process_images(
                images, backgrounds, metadata, settings, rebin=2)
        
    elif image_type == '4pos':
        images, centers, pixel_size = process_images(
                images, backgrounds, metadata, settings)
        
    elif image_type == '4pos_stack':
        for i in range(len(images)):
            centers = np.zeros((len(images), 4))
            pixel_size = np.zeros((len(images)))
            images[i], centers[i], pixel_size[i] = process_images(
                images[i], backgrounds[i], metadata, settings)
        
    else:
        raise RuntimeError("Unknown image type")
    
    return images, pixel_size, centers

def myimread(fn):
    ims = imread(fn)
    if len(ims.shape) == 3:
        ims = np.squeeze(ims[np.logical_not(np.all(ims == 0, (1, 2)))])
    return ims


    

def load_images(filename, imborder=None):
    """ Load the images files and do some preprocessing

    Parameters
    ----------
    metadata: dict
        The metadata informations

    Returns
    -------
    ims: array
        image or array of images
    bg: array
        background or array of backgrounds
    """

    # load images
    if isinstance(filename, (list, tuple)):
        images = np.asarray([myimread(fn) for fn in filename])
    else:
        images = myimread(filename)
        
    if imborder is not None:
        # Remove Border
        images = images[..., imborder[0]:imborder[1],
                        imborder[2]:imborder[3]]    
        
    return images
        
        
#    bgfn = metadata[keys.KEY_MD_BGFN]
#    optics_bgfn = metadata[keys.KEY_MD_OPBGFN]
#
#    # load background
#    bg = None
#    if bgfn is not None:
#        if isinstance(bgfn, (list, tuple)):
#            bg = np.asarray([myimread(fn) for fn in bgfn])
#        else:
#            bg = myimread(bgfn)
#
#    # Remove background from optics
#    if optics_bgfn is not None:
#        exposure = metadata[keys.KEY_MD_EXP]
#        background_exposure = metadata[keys.KEY_MD_BGEXP]
#        optics_background_exposure = metadata[keys.KEY_MD_OPBGEXP]
#        
#        optics = myimread(optics_bgfn) / optics_background_exposure
#        ims = ims / exposure - optics
#        if bg is not None:
#            bg = bg / background_exposure - optics
#
#    
#    if bg is not None:
#        bg = bg[..., imborder[0]:imborder[1],
#                imborder[2]:imborder[3]]
#
#    return ims, bg

#def read_data_dict(data_dict):
#    """Extract interesting data from a data dict
#
#    Parameters
#    ----------
#    data_dict: dict
#        The data dictionnary
#
#    Returns
#    -------
#    profiles: 1d list of floats
#        The extracted profiles.
#    fits: 1d list of floats
#        The Fits.
#    lse: float
#        The least square error.
#    pixel_size: float
#        The detected pixel size
#    """
#    profiles, fits, lse, pixel_size, im = np.nan * np.ones(5)
#
#    if 'profiles' in data_dict and 'fits'in data_dict:
#        profiles = data_dict['profiles']
#        fits = data_dict['fits']
#
#        if len(profiles) - 1 == len(fits):
#            fits = [data_dict['initprof'], *data_dict['fits']]
#
#        lse = np.sqrt(np.mean(np.square(profiles - fits)))
#        pixel_size = data_dict['pixel_size']
#
#    if 'image' in data_dict:
#        im = data_dict['image']
#
#    return profiles, fits, lse, pixel_size, im
