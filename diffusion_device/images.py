# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:32:20 2017

@author: quentinpeter
"""

from tifffile import imread
import numpy as np
from registrator.image import is_overexposed

from . import multi_channels_image as dd4
from . import channel_image as ddx
from . import keys, input_files
from . import profile as dp
from . import scans

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

    # Open data
    filename = metadata[keys.KEY_MD_FN]
    data = load_data(filename)

    # Get type
    image_type = input_files.getType(metadata, data.shape)
    
    #Check for overexposure
    overexposed = is_overexposed(data)

    # Process data
    data, pixel_size, centers = process_data(
        data, image_type, metadata, settings)

    # Get profiles
    profiles = get_profiles(
        data, image_type, pixel_size, centers, metadata, settings)

    radius, fits = size_profiles(
        profiles, image_type, pixel_size, metadata, settings)

    return radius, profiles, fits, pixel_size, data, image_type, overexposed


def size_profiles(profiles, image_type, pixel_size, metadata, settings):
    if image_type == '4pos_stack':
        radius = []
        fits = []
        for i, (profs, pxs) in enumerate(zip(profiles, pixel_size)):
            r, fit = size_profiles(
                profs, '4pos', pxs, metadata, settings)
            fits.append(fit)
            radius.append(r)
        radius = np.asarray(radius)
    else:
        fits = np.zeros_like(profiles)
        radius = dp.size_profiles(profiles, pixel_size, metadata, settings,
                                  fits=fits)
    return radius, fits


def get_profiles(data, image_type, pixel_size, centers, metadata, settings):

    channel_width = metadata[keys.KEY_MD_WY]
    if image_type == 'scans':
        profiles = scans.get_profiles(data, int(channel_width/pixel_size))

    elif image_type == '12pos':
        Npix = int(channel_width // pixel_size) + 1
        profiles = np.zeros((len(data), Npix))
        for i, (im, center) in enumerate(zip(data, centers)):
            profiles[i] = ddx.extract_profile(
                im, pixel_size, channel_width, center)

    elif image_type == '4pos':
        imslice = settings[keys.KEY_STG_SLICE]
        ignore = settings[keys.KEY_STG_IGNORE]
        profiles = dd4.extract_profiles(
            data, centers, channel_width, ignore, pixel_size,
            imslice=imslice)

    elif image_type == '4pos_stack':
        profiles = [get_profiles(im, '4pos', pxs, cnt, metadata, settings)
                    for im, pxs, cnt in zip(data, pixel_size, centers)]

    return profiles

def get_background(metadata):
    backgrounds = None
    background_fn = metadata[keys.KEY_MD_BGFN]
    if background_fn is not None:
        backgrounds = load_data(background_fn)
    return backgrounds

def remove_optics_background(images, backgrounds, metadata):
    # Remove background from optics    
    optics_bgfn = metadata[keys.KEY_MD_OPBGFN]
    if optics_bgfn is not None:
        exposure = metadata[keys.KEY_MD_EXP]
        background_exposure = metadata[keys.KEY_MD_BGEXP]
        optics_background_exposure = metadata[keys.KEY_MD_OPBGEXP]

        optics = dataread(optics_bgfn) / optics_background_exposure
        images = images / exposure - optics
        if backgrounds is not None:
            backgrounds = backgrounds / background_exposure - optics
    
    return images, backgrounds

def clip_border(images, backgrounds, metadata):
    imborder = metadata[keys.KEY_MD_BORDER]
    if imborder is not None:
        # Remove Border
        images = images[..., imborder[0]:imborder[1],
                             imborder[2]:imborder[3]]
        if backgrounds is not None: 
            # Remove Border
            backgrounds = backgrounds[..., imborder[0]:imborder[1],
                                           imborder[2]:imborder[3]]
    return images, backgrounds

def process_data(data, image_type, metadata, settings):
    
    if image_type == 'scans':
        centers = [None] * len(data)
        pixel_size = metadata[keys.KEY_MD_PIXSIZE]
        data= scans.process_scans(data, metadata)
        return data, pixel_size, centers

    
    backgrounds = get_background(metadata)    
    data, backgrounds = remove_optics_background(data, backgrounds, metadata)
    data, backgrounds = clip_border(data, backgrounds, metadata)

    if image_type == '12pos':
        centers = [None] * len(data)
        data, pixel_size = ddx.process_images(
            data, backgrounds, metadata, settings, rebin=2)

    elif image_type == '4pos':
        data, centers, pixel_size = dd4.process_images(
            data, backgrounds, metadata, settings)

    elif image_type == '4pos_stack':
        
        if backgrounds is None:
            backgrounds = [None] * len(data)
        data = np.asarray(data, dtype=float)
        centers = np.zeros((len(data), 4))
        pixel_size = np.zeros((len(data)))
        for i in range(len(data)):
            data[i], pixel_size[i], centers[i] = process_data(
                data[i], '4pos', metadata, settings)

    else:
        raise RuntimeError("Unknown image type")

    return data, pixel_size, centers


def dataread(fn):
    if fn[-4:] == ".dat":
        return np.loadtxt(fn,skiprows=22)
    
    ims = imread(fn)
    if len(ims.shape) == 3:
        ims = np.squeeze(ims[np.logical_not(np.all(ims == 0, (1, 2)))])
    return ims


def load_data(filename):
    """ Load the data files and do some preprocessing

    Parameters
    ----------
    metadata: dict
        The metadata informations

    Returns
    -------
    ims: array
        image or array of data
    bg: array
        background or array of backgrounds
    """

    # load data
    if isinstance(filename, (list, tuple)):
        data = np.asarray([dataread(fn) for fn in filename])
    else:
        data = dataread(filename)

    return data
