# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:21:01 2017

@author: quentinpeter
"""
import numpy as np
from tifffile import imread
from scipy import interpolate

from . import bright, uv
from .. import profile as dp
from .. import keys

def size_image(image, background, metadata, settings,
               data_dict=None, plotimage=False, ignore_error=False):
    """
    Get the hydrodynamic radius from the images

    Parameters
    ----------
    image: 2d image 
        image to analyse
    background: path or image
        The background to remove
    metadata: dict
        The metadata
    settings: dict
        The settings
    data_dict: dict, defaults None
        Output to get the profiles and fits
    ignore_error: Bool, default False
        Should the errors be ignored?
    plotimage: Bool, default False
        Plot how the image is flattened

    Returns
    -------
    r: float
        Radius in [m]

    """

    # Check images is numpy array
    image = np.asarray(image)

    # Read relevant values
    channel_width = metadata[keys.KEY_MD_WY]
    nchannels = metadata[keys.KEY_MD_NCHANNELS]
    wall_width = metadata[keys.KEY_MD_WALLWIDTH]
    ignore = settings[keys.KEY_STG_IGNORE]
    imslice = settings[keys.KEY_STG_SLICE]

    # load images if string
    if image.dtype.type == np.str_:
        image = imread(str(image))

    # Check shape
    if not len(np.shape(image)) == 2:
        raise RuntimeError("Incorrect image shape: " + str(np.shape(image)))

    if background is not None:
        background = np.asarray(background)
        # load background if string
        if background.dtype.type == np.str_:
            background = imread(str(background))

        # Check shape
        if not len(np.shape(background)) == 2:
            raise RuntimeError("Incorrect background shape: "
                               + str(np.shape(background)))

    try:
        # get profiles
        if background is None:
            flatten = settings[keys.KEY_STG_BFFLAT]
            # Single image
            image, centers, pixsize = bright.extract_profiles(
                image, nchannels, channel_width, wall_width, flatten,
                data_dict=data_dict, plotimage=plotimage)
        else:
            # images and background
            image, centers, pixsize = uv.extract_profiles(
                image, background, nchannels, channel_width, wall_width,
                data_dict=data_dict)

    except RuntimeError as error:
        print(error.args[0])
        if ignore_error:
            return np.nan
        else:
            raise error

    profiles = extract_profiles(image, centers, channel_width, ignore,
                                pixsize, imslice=imslice)

    if data_dict is not None:
        data_dict["image"] = image
        data_dict['pixsize'] = pixsize
        data_dict['profiles'] = profiles

    return dp.size_profiles(profiles, pixsize, metadata, settings,
                            data_dict=data_dict)


def extract_profiles(image, centers, chwidth, ignore, pixsize,
                     imslice=None):
    '''cut the image profile into profiles

    Parameters
    ----------
    image: 2d array
        The flat image
    centers: 1d array
        The position of the centers [px]
    chwidth: float
        Width of the channel [m]
    ignore: float
        Distance to sides to ignore [m]
    pixsize: float
        Size of the pixel [m]
    imslice: 2 floats, default None
        [Y distance from center, Y width] [m]

    Returns
    -------
    profiles: 2d array
        The profiles
    '''

    # convert ignore to px
    ignore = int(ignore / pixsize)

    if ignore == 0:
        pslice = slice(None)
    else:
        pslice = slice(ignore, -ignore)

    nchannels = len(centers)
    prof_npix = int(np.round(chwidth / pixsize))

    if imslice is None:
        image_profile = np.nanmean(image, 0)
    else:
        image_profile = imageProfileSlice(
            image, imslice[0], imslice[1], pixsize)

    profiles = np.empty((nchannels, prof_npix), dtype=float)

    # Extract profiles
    firstcenter = None
    for i, cent in enumerate(centers):

        X = np.arange(len(image_profile)) - cent
        Xc = np.arange(prof_npix) - (prof_npix - 1) / 2
        finterp = interpolate.interp1d(X, image_profile)
        p = finterp(Xc)

        # switch if uneven
        if i % 2 == 1:
            p = p[::-1]

        # If the profile is not too flat
        testflat = np.max(p[pslice]) > 1.2 * np.mean(p[pslice])
        if testflat:
            # Align by detecting center
            c = dp.center(p[pslice]) + ignore
            if firstcenter is not None:
                diff = c - firstcenter
                if i % 2 == 1:
                    diff *= -1
                X = np.arange(len(image_profile)) - cent - diff
                finterp = interpolate.interp1d(X, image_profile)
                p = finterp(Xc)
                if i % 2 == 1:
                    p = p[::-1]

            else:
                firstcenter = c

        profiles[i] = p

    # If image upside down, turn
    if profiles[-1].max() > profiles[0].max():
        profiles = profiles[::-1]


#    from matplotlib.pyplot import plot, figure, imshow
#    figure()
#    imshow(image)
#    figure()
#    plot(image_profile)

    return profiles


def imageProfileSlice(image, center, width, pixsize):
    '''Get the image profile corresponding to a center and width

    Parameters
    ----------
    image: 2d array
        The flat image
    center: float
        Y center of the slice [m]
    width: float
        Y width of the slice [m]
    pixsize: float
        Size of the pixel [m]
    Returns
    -------
    prof: array of float
        The profile corresponding to the slice

    '''
    center = len(image) // 2 + int(np.round(center / pixsize))
    width = int(np.round(width / pixsize))
    amin = (2 * center - width) // 2
    amax = (2 * center + width) // 2
    if amin < 0 or amax > len(image):
        raise RuntimeError("Poorly defined slice")
    return np.nanmean(image[amin:amax], 0)
