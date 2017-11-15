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
import sys

from .. import multi_channels_image as single
from ... import display_data
from .. import images_files
from ... import profile as dp


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
    filename = metadata["KEY_MD_FN"]
    data = images_files.load_images(filename)
    overexposed = [is_overexposed(d) for d in data]
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
    framesslices = slice(*settings["KEY_STG_STACK_FRAMESSLICES"])
    data = np.asarray(data, dtype=float)[framesslices]
    centers = np.zeros((len(data), 4))
    pixel_size = np.zeros((len(data)))
    dataout = []
    skip = []

    if settings["KEY_STG_STAT_STACK"]:
        # Check KEY_MD_EXP are all the same
        if isinstance(metadata["KEY_MD_EXP"], list):
            if np.all(np.equal(metadata["KEY_MD_EXP"],
                               metadata["KEY_MD_EXP"][0])):
                raise RuntimeError(
                    "Can not have different exposure times"
                    " when using stationnary option.")
            else:
                metadata["KEY_MD_EXP"] = metadata["KEY_MD_EXP"][0]

        return single.process_data(data, metadata, settings)

    for i in range(len(data)):
        try:
            # Get KEY_MD_EXP correct in the metadata
            single_metadata = metadata.copy()
            if isinstance(metadata["KEY_MD_EXP"], list):
                single_metadata["KEY_MD_EXP"] = (
                    np.asarray(metadata["KEY_MD_EXP"])[framesslices][i])
            d, pixel_size[i], centers[i] = single.process_data(
                data[i], single_metadata, settings)
            dataout.append(d)
        except BaseException:
            if settings["KEY_STG_IGNORE_ERROR"]:
                print(sys.exc_info()[1])
                pixel_size[i] = np.nan
                centers[i, :] = np.nan
                skip.append(i)
            else:
                raise

    # Fix metadata
    metadata["KEY_MD_FLOWDIR"] = single_metadata["KEY_MD_FLOWDIR"]
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
    for i, im in enumerate(data):
        try:
            if settings["KEY_STG_STAT_STACK"]:
                pxs, cnt = pixel_size, centers
            else:
                pxs, cnt = pixel_size[i], centers[i]
            prof = single.get_profiles(metadata, settings, im, pxs, cnt, )
        except BaseException:
            if settings["KEY_STG_IGNORE_ERROR"]:
                print(sys.exc_info()[1])
                prof = None
            else:
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
    errors = []
    for i, profs in enumerate(profiles):
        if profs is None:
            fits.append(None)
            errors.append(np.nan)
        else:
            try:
                if settings["KEY_STG_STAT_STACK"]:
                    pxs = pixel_size
                else:
                    pxs = pixel_size[i]
                r, fit, error = single.size_profiles(
                    profs, pxs, metadata, settings)
            except BaseException:
                if settings["KEY_STG_IGNORE_ERROR"]:
                    print(sys.exc_info()[1])
                    r = np.nan
                    fit = None
                    error = np.nan
                else:
                    raise
            fits.append(fit)
            radius.append(r)
            errors.append(error)
    radius = np.asarray(radius)
    for idx, add in enumerate([p is None for p in profiles]):
        if add:
            radius = np.insert(radius, idx,
                               np.ones(np.shape(radius)[1:]) * np.nan, 0)
    return radius, fits, errors


def savedata(data, outpath):
    """Save the data"""
    tifffile.imsave(outpath + '_ims.tif', data)


def plot_and_save(radius, profiles, fits, error, pixel_size, state,
                  outpath, settings):
    """Plot the sizing data"""
    plotpos = settings["KEY_STG_STACK_POSPLOT"]
    framesslices = slice(*settings["KEY_STG_STACK_FRAMESSLICES"])

    state = state[framesslices]
    display_data.plot_and_save_stack(
        radius, profiles, fits, error, pixel_size, state, outpath, plotpos)


def process_profiles(profiles, pixel_size, settings, outpath):
    ret = []
    for i, prof in enumerate(profiles):
        if prof is None:
            ret.append(None)
        else:
            if settings["KEY_STG_STAT_STACK"]:
                pxs = pixel_size
            else:
                pxs = pixel_size[i]
            ret.append(dp.process_profiles(prof, pxs, settings, outpath))

    return ret
