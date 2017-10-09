# -*- coding: utf-8 -*-
"""
Analyse 4 position channels

Created on Tue Apr  4 11:21:01 2017

@author: quentinpeter

Copyright (C) 2017  Quentin Peter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from tifffile import imread
from scipy import interpolate
from registrator.image import is_overexposed
import tifffile

from . import bright, uv, stack
from ... import profile as dp
from ... import keys, display_data
from .. import images_files


def load_data(metadata):
    """load data from metadata

    Parameters
    ----------
    metadata: dict
        The metadata information

    Returns
    -------
    data: array
        the image
    overexposed: bool
        An indicator to see if the data is overexposed
    """
    filename = metadata[keys.KEY_MD_FN]
    data = images_files.load_image(filename)
    overexposed = is_overexposed(data)
    return data, overexposed


def process_data(data, metadata, settings):
    """Do some data processing

    Parameters
    ----------
    data: array
        The data to process
    metadata: dict
        The metadata information
    settings: dict
        The settings

    Returns
    -------
    data: array
        The processed data
    pixel_size: float
        The pixel size
    centers: array
        The positions of the centers
    """
    data, background = images_files.process_background(data, metadata)
    data, centers, pixel_size = process_image(
        data, background, metadata, settings)
    return data, pixel_size, centers


def get_profiles(metadata, settings, data, pixel_size, centers):
    """Do some data processing

    Parameters
    ----------
    metadata: dict
        The metadata information
    settings: dict
        The settings
    data: array
        The data to process
    pixel_size: float
        The pixel size
    centers: array
        The positions of the centers

    Returns
    -------
    profiles: array
        The profiles
    """
    if np.isnan(pixel_size):
        return None
    channel_width = metadata[keys.KEY_MD_WY]
    imslice = settings[keys.KEY_STG_SLICE]
    ignore = settings[keys.KEY_STG_IGNORE]
    flowdir = metadata[keys.KEY_MD_FLOWDIR]
    profiles = extract_profiles(
        data, centers, flowdir, channel_width, ignore, pixel_size,
        imslice=imslice)
    
    norm_profiles = settings[keys.KEY_STG_NORMALISE]
    if norm_profiles:
        ignore_slice = dp.ignore_slice(ignore, pixel_size)
        profiles = dp.normalise_profiles(profiles, ignore_slice)
    
    # If image upside down, turn
    if profiles[-1].max() > profiles[0].max():
        profiles = profiles[::-1]
    return profiles


def size_profiles(profiles, pixel_size, metadata, settings):
    """Size the profiles

     Parameters
    ----------
    profiles: 2d array
        List of profiles to fit
    pixel_size:float
        The pixel size in [m]
    metadata: dict
        The metadata
    settings: dict
        The settings

    Returns
    -------
    radius:
        if nspecies==1:
            radii: float
                The best radius fit
        else:
            Rs, spectrum, the radii and corresponding spectrum
    fits: 2d array
        The fits
    """
    fits = np.zeros_like(profiles)
    radius = dp.size_profiles(profiles, pixel_size, metadata, settings,
                              fits=fits)
    return radius, fits

def savedata(data, outpath):
    """Save the data"""
    tifffile.imsave(outpath + '_im.tif', data)

def plot_and_save(radius, profiles, fits, pixel_size, state,
                  outpath, settings):
    """Plot the sizing data"""
    display_data.plot_and_save(
        radius, profiles, fits, pixel_size, outpath)


def process_image(image, background, metadata, settings):
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
    flowdir = metadata[keys.KEY_MD_FLOWDIR]

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
            
    image, background, metadata[keys.KEY_MD_FLOWDIR] = (
            rotate(image, background, flowdir))

    
    # get profiles
    if background is None:
        flatten = settings[keys.KEY_STG_BRIGHT_FLAT]
        # Single image
        image, centers, pixel_size = bright.extract_profiles(
            image, nchannels, channel_width, wall_width, flatten)
    else:
        # images and background
        image, centers, pixel_size = uv.extract_profiles(
            image, background, nchannels, channel_width, wall_width)

    return image, centers, pixel_size

def rotate(image, background, flowdir):
    flowdir = np.asarray(flowdir)
    if flowdir[0] == 'l' or flowdir[0] == 'r':
        image = np.rot90(image)
        if background is not None:
            background = np.rot90(background)
        flowdir[flowdir == 'l'] = 'u'
        flowdir[flowdir == 'r'] = 'd'
    return image, background, flowdir
    
    
def extract_profiles(image, centers, flowdir, chwidth, ignore, pixel_size,
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
    pixel_size: float
        Size of the pixel [m]
    imslice: 2 floats, default None
        [Y distance from center, Y width] [m]

    Returns
    -------
    profiles: 2d array
        The profiles (left to right)
    '''
    # convert ignore to px
    ignore = int(ignore / pixel_size)

    if ignore == 0:
        pslice = slice(None)
    else:
        pslice = slice(ignore, -ignore)

    nchannels = len(centers)
    prof_npix = int(np.round(chwidth / pixel_size))

    if imslice is None:
        image_profile = np.nanmean(image, 0)
    else:
        image_profile = imageProfileSlice(
            image, imslice[0], imslice[1], pixel_size)

    if (np.min(centers) - prof_npix/2 < 0 or
            np.max(centers) + prof_npix/2 > len(image_profile)):
        raise RuntimeError('Channel not fully contained in the image')
        
    profiles = np.empty((nchannels, prof_npix), dtype=float)

    # Extract profiles
    firstcenter = None
    for i, (cent, fd) in enumerate(zip(centers, flowdir)):

        X = np.arange(len(image_profile)) - cent
        Xc = np.arange(prof_npix) - (prof_npix - 1) / 2
        finterp = interpolate.interp1d(X, image_profile)
        p = finterp(Xc)
        
        if fd == 'u' or fd == 'up':
            switch = True
        elif fd == 'd' or fd == 'down':
            switch = False
        else:
            raise RuntimeError("unknown orientation: {}".format(fd))

        if switch:
            p = p[::-1]

        # If the profile is not too flat
        testflat = np.max(p[pslice]) > 1.2 * np.mean(p[pslice])
        if testflat:
            # Align by detecting center
            prof_center = dp.center(p[pslice]) + ignore
            if firstcenter is not None:
                diff = prof_center - firstcenter
                if switch:
                    diff *= -1
                X = np.arange(len(image_profile)) - cent - diff
                finterp = interpolate.interp1d(X, image_profile)
                p = finterp(Xc)
                if switch:
                    p = p[::-1]

            else:
                firstcenter = prof_center

        profiles[i] = p


#    from matplotlib.pyplot import plot, figure, imshow
#    figure()
#    imshow(image)
#    figure()
#    plot(image_profile)

    return profiles


def imageProfileSlice(image, center, width, pixel_size):
    '''Get the image profile corresponding to a center and width

    Parameters
    ----------
    image: 2d array
        The flat image
    center: float
        Y center of the slice [m]
    width: float
        Y width of the slice [m]
    pixel_size: float
        Size of the pixel [m]
    Returns
    -------
    prof: array of float
        The profile corresponding to the slice

    '''
    center = len(image) // 2 + int(np.round(center / pixel_size))
    width = int(np.round(width / pixel_size))
    amin = (2 * center - width) // 2
    amax = (2 * center + width) // 2
    if amin < 0 or amax > len(image):
        raise RuntimeError("Poorly defined slice")
    return np.nanmean(image[amin:amax], 0)
