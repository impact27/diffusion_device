# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:21:01 2017

@author: quentinpeter
"""
import numpy as np
from tifffile import imread
from . import bright, background
from .. import profile as dp


def defaultReadingPos(startpos=400e-6, isFolded=True):
    '''
    Get the default reading positions for the 4 points diffusion device

    Parameters
    ----------
    startpos: float, default 400e-6 
        The center of the image, relative to the first turn [m]
    isFolded: Bool, default True
        If this is the folded or the straight device

    Returns
    -------
    readingPos: 1d array
        The reading positions
    '''
#    return np.array([  4183, 21446, 55879])*1e-6 #Kadi
#    return np.array([  3738, 21096, 55374])*1e-6 #Electrophoresis
    if isFolded:
        return np.array([0, 4556e-6 - 2 * startpos,
                         21953e-6,
                         47100e-6 - 2 * startpos])  # folded device
    else:
        return np.array([0, 4532e-6 - 2 * startpos,
                         21128e-6,
                         56214e-6 - 2 * startpos])  # folded device


def size_image(im, Q, Wz, Wy, readingpos, Rs, Nprofs, wall_width, *, bg=None,
               Zgrid=11, ignore=5e-6, normalise_profiles=True,
               initmode='none', data_dict=None, fit_position_number=None,
               flatten=False, nspecies=1, ignore_error=False, plotim=False):
    """
    Get the hydrodynamic radius from the images

    Parameters
    ----------
    im: 2d image or file name OR 2x 2d images
        If this is a string, it will be treated like a path
        If one image, treated like regular fluorescence image
        If two images, treated like image and background
    Q: float
        Flow rate in [ul/h]
    Wz: float
        Height of the channel in [m]
    Wy: float
        Width of the channel in [m]
    readingpos: 1d float array, defaults None
        Position at which the images are taken. If None, take the defaults
    Rs: 1d array, defaults None
        Hydrodimamic radii to simulate in [m].
        If None: between .5 and 10 nm
    Nprofs: integer
        the numbers of channels
    wall_width: float
        The wall width in [m]
    bg: path or image
        The background to remove
    Zgrid: int, defaults 11
        Number of Z slices
    ignore: float, defaults 5e-6
        Distance to sides to ignore
    normalise_profiles: Bool, defaults True
        Should the profiles be normalised?
    initmode: str, defaults 'none'
        The processing mode for the initial profile (See profiles.py)
    data_dict: dict, defaults None
        Output to get the profiles and fits
    fit_position_number: 1d list
        Positions to use in the fit
    flatten: Bool, defaut False
        (Bright field only) Should the image be flattened?
    nspecies: int, default 1
        Number of species to fit. 0=all.
    ignore_error: Bool, default False
        Should the errors be ignored?
    plotim: Bool, default False
        Plot how the image is flattened

    Returns
    -------
    r: float
        Radius in [m]

    """

    # Check images is numpy array
    im = np.asarray(im)

    # load images if string
    if im.dtype.type == np.str_:
        im = imread(str(im))

    # Check shape
    if not len(np.shape(im)) == 2:
        raise RuntimeError("Incorrect image shape: " + str(np.shape(im)))

    if bg is not None:
        bg = np.asarray(bg)
        # load bg if string
        if bg.dtype.type == np.str_:
            bg = imread(str(bg))

        # Check shape
        if not len(np.shape(bg)) == 2:
            raise RuntimeError("Incorrect background shape: "
                               + str(np.shape(bg)))

    try:
        # get profiles
        if bg is None:
            # Single image
            profiles = bright.extract_profiles(im, Nprofs,
                                               Wy, wall_width,
                                               flatten=flatten, plotim=plotim)

        else:
            #images and background
            profiles = background.extract_profiles(im, bg, Nprofs,
                                                   Wy, wall_width)
    except RuntimeError as error:
        print(error.args[0])
        if ignore_error:
            return np.nan
        else:
            raise error

    if fit_position_number is not None:
        profiles = profiles[np.sort(fit_position_number)]

    pixsize = Wy / np.shape(profiles)[1]
    if data_dict is not None:
        data_dict['pixsize'] = pixsize
        data_dict['profiles'] = profiles
    return dp.size_profiles(profiles, Q, Wz, pixsize, readingpos, Rs,
                            initmode=initmode, normalise_profiles=normalise_profiles,
                            Zgrid=Zgrid, ignore=ignore, data_dict=data_dict,
                            nspecies=nspecies)
