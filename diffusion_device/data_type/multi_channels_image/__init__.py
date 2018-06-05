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
from registrator.image import is_overexposed
import tifffile

from . import bright, uv, stack
from ... import profile as dp
from ... import display_data
from .. import images_files


def load_data(metadata, infos):
    """load data from metadata

    Parameters
    ----------
    metadata: dict
        The metadata information
    infos: dict
        Dictionnary with other infos

    Returns
    -------
    data: array
        the image
    """
    filename = metadata["KEY_MD_FN"]
    data = images_files.load_image(filename)
    infos["Overexposed"] = is_overexposed(data)
    return data


def process_data(data, metadata, settings, infos):
    """Do some data processing

    Parameters
    ----------
    data: array
        The data to process
    metadata: dict
        The metadata information
    settings: dict
        The settings
    infos: dict
        Dictionnary with other infos

    Returns
    -------
    data: array
        The processed data
    """
    data, background = images_files.process_background(data, metadata)
    data, centers, pixel_size = process_image(
        data, background, metadata, settings)
    infos["Pixel size"] = pixel_size
    infos["Centers"] = centers
    return data


def get_profiles(data, metadata, settings, infos):
    """Do some data processing

    Parameters
    ----------
    data: array
        The data to process
    metadata: dict
        The metadata information
    settings: dict
        The settings
    infos: dict
        Dictionnary with other infos

    Returns
    -------
    profiles: array
        The profiles
    """
    pixel_size = infos["Pixel size"]
    centers = infos["Centers"]
    if np.isnan(pixel_size):
        return None
    channel_width = metadata["KEY_MD_WY"]
    imslice = settings["KEY_STG_SLICE"]
    ignore = settings["KEY_STG_IGNORE"]
    flowdir = metadata["KEY_MD_FLOWDIR"]
    profiles, noise = extract_profiles(
        data, centers, flowdir, channel_width, ignore, pixel_size,
        imslice=imslice)

    # If image upside down, turn
    if profiles[-1].max() > profiles[0].max():
        profiles = profiles[::-1]

    infos["Profiles noise std"] = noise

    return profiles


def process_profiles(profiles, settings, outpath, infos):
    profiles,infos["Pixel size"] = dp.process_profiles(
        profiles, settings, outpath, infos["Pixel size"])
    return profiles


def size_profiles(profiles, metadata, settings, infos):
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
    return dp.size_profiles(profiles, metadata, settings,
                            infos)


def savedata(data, outpath):
    """Save the data"""
    tifffile.imsave(outpath + '_im.tif', data)


def plot_and_save(radius, profiles, fits,
                  outpath, settings, infos):
    """Plot the sizing data"""
    display_data.plot_and_save(
        radius, profiles, fits, infos, outpath)


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

    Returns
    -------
    r: float
        Radius in [m]

    """

    # Check images is numpy array
    image = np.asarray(image)

    # Read relevant values
    channel_width = metadata["KEY_MD_WY"]
    nchannels = metadata["KEY_MD_NCHANNELS"]
    wall_width = metadata["KEY_MD_WALLWIDTH"]
    flowdir = metadata["KEY_MD_FLOWDIR"]

    # Check shape
    if len(np.shape(image)) not in [2, 3]:
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

    image, background, metadata["KEY_MD_FLOWDIR"] = (
        rotate(image, background, flowdir))

    # get profiles
    if background is None:
        flatten = settings["KEY_STG_BRIGHT_FLAT"]
        # Single image
        image, centers, pixel_size = bright.extract_data(
            image, nchannels, channel_width, wall_width, flatten)
    else:
        # images and background
        image, centers, pixel_size = uv.extract_data(
            image, background, nchannels, channel_width, wall_width,
            goodFeatures=settings["KEY_STG_GOODFEATURES"])

    return image, centers, pixel_size


def rotate(image, background, flowdir):
    """ Rotate 2d or 3d image
    """
    flowdir = np.asarray(flowdir)
    if flowdir[0] == 'l' or flowdir[0] == 'r':
        image = np.rot90(image, axes=(-2, -1))
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
    if imslice is None:
        profiles = np.nanmean(image, 0)
    else:
        profiles = imageProfileSlice(
            image, imslice[0], imslice[1], pixel_size)
    
    return dp.extract_profiles(
            profiles, centers, flowdir, chwidth, ignore, pixel_size)

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
