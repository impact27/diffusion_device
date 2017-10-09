# -*- coding: utf-8 -*-
"""
Module for movies of 4 pos channels

Created on Wed Sep 13 10:15:41 2017

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

from registrator.image import is_overexposed
import numpy as np
import tifffile

from .. import multi_channels_image as single

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
    data = images_files.load_images(filename)
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
    framesslices = slice(*settings[keys.KEY_STG_STACK_FRAMESSLICES])
    data = np.asarray(data, dtype=float)[framesslices]
    centers = np.zeros((len(data), 4))
    pixel_size = np.zeros((len(data)))
    dataout = []
    skip = []
    for i in range(len(data)):
        try:
            single_metadata = metadata.copy()
            if isinstance(metadata[keys.KEY_MD_EXP], list):
                single_metadata[keys.KEY_MD_EXP] = (
                    np.asarray(metadata[keys.KEY_MD_EXP])[framesslices][i])
            d, pixel_size[i], centers[i] = single.process_data(
                data[i], single_metadata, settings)
            dataout.append(d)
        except BaseException as error:
            if settings[keys.KEY_STG_IGNORE_ERROR]:
                print(error.args[0])
                pixel_size[i] = np.nan
                centers[i, :] = np.nan
                skip.append(i)
            else:
                raise

    # Fix metadata
    metadata[keys.KEY_MD_FLOWDIR] = single_metadata[keys.KEY_MD_FLOWDIR]
    dataout = np.asarray(dataout)
    if skip != []:
        for idx in skip:
            dataout = np.insert(dataout, idx,
                                np.ones(np.shape(dataout)[1:]) * np.nan, 0)

    return dataout, pixel_size, centers


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
    profiles = []
    for im, pxs, cnt in zip(data, pixel_size, centers):
        try:
            prof = single.get_profiles(metadata, settings, im, pxs, cnt, )
        except BaseException as error:
            if settings[keys.KEY_STG_IGNORE_ERROR]:
                print(error.args[0])
                prof = None
            else:
                print(settings[keys.KEY_STG_IGNORE_ERROR])
                raise
        profiles.append(prof)
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
    radius = []
    fits = []
    for i, (profs, pxs) in enumerate(zip(profiles, pixel_size)):
        if profs is None:
            fits.append(None)
        else:
            r, fit = single.size_profiles(profs, pxs, metadata, settings)
            fits.append(fit)
            radius.append(r)
    radius = np.asarray(radius)
    for idx, add in enumerate([p is None for p in profiles]):
        if add:
            radius = np.insert(radius, idx,
                               np.ones(np.shape(radius)[1:]) * np.nan, 0)
    return radius, fits


def savedata(data, outpath):
    """Save the data"""
    tifffile.imsave(outpath + '_ims.tif', data)


def plot_and_save(radius, profiles, fits, pixel_size, state,
                  outpath, settings):
    """Plot the sizing data"""
    plotpos = settings[keys.KEY_STG_STACK_POSPLOT]
    framesslices = slice(*settings[keys.KEY_STG_STACK_FRAMESSLICES])

    state = state[framesslices]
    display_data.plot_and_save_stack(
        radius, profiles, fits, pixel_size, state, outpath, plotpos)
