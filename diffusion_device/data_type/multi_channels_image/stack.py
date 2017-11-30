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
    data = images_files.load_images(filename)
    infos["Overexposed"] = [is_overexposed(d) for d in data]
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
    Nchannel = metadata['KEY_MD_NCHANNELS']
    framesslices = slice(*settings["KEY_STG_STACK_FRAMESSLICES"])
    data = np.asarray(data, dtype="float32")[framesslices]
    centers = np.zeros((len(data), Nchannel))
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
                    " when using stationary option.")
            else:
                metadata["KEY_MD_EXP"] = metadata["KEY_MD_EXP"][0]

        return single.process_data(data, metadata, settings, infos)

    for i in range(len(data)):
        try:
            # Get KEY_MD_EXP correct in the metadata
            single_metadata = metadata.copy()
            if isinstance(metadata["KEY_MD_EXP"], list):
                single_metadata["KEY_MD_EXP"] = (
                    np.asarray(metadata["KEY_MD_EXP"])[framesslices][i])
            infos_i = {}
            d = single.process_data(
                data[i], single_metadata, settings, infos_i)

            pixel_size[i] = infos_i["Pixel size"]
            centers[i] = infos_i["Centers"]
            dataout.append(d)
        except BaseException:
            if settings["KEY_STG_IGNORE_ERROR"]:
                print(sys.exc_info())
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

    infos["Pixel size"] = pixel_size
    infos["Centers"] = centers
    return dataout


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
    profiles = []
    noises = np.zeros(len(data)) * np.nan
    for i, im in enumerate(data):
        try:
            if settings["KEY_STG_STAT_STACK"]:
                pxs, cnt = pixel_size, centers
            else:
                pxs, cnt = pixel_size[i], centers[i]

            infos_i = {
                "Pixel size": pxs,
                "Centers": cnt}
            prof = single.get_profiles(im, metadata, settings, infos_i)
            noises[i] = infos_i["Profiles noise std"]
        except BaseException:
            if settings["KEY_STG_IGNORE_ERROR"]:
                print(sys.exc_info())
                prof = None
            else:
                raise
        profiles.append(prof)

    infos["Profiles noise std"] = noises
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
    radius = []
    fits = []
    errors = []
    r_errors = []
    shape_r = None
    signal_over_noise = np.zeros(len(profiles)) * np.nan
    for i, profs in enumerate(profiles):
        r = None
        fit = None
        error = np.nan
        r_error = np.nan

        if profs is not None:
            try:
                if settings["KEY_STG_STAT_STACK"]:
                    infos_i = {
                        "Pixel size": infos["Pixel size"],
                        "Profiles noise std": infos["Profiles noise std"][i]}
                else:
                    infos_i = {
                        "Pixel size": infos["Pixel size"][i],
                        "Profiles noise std": infos["Profiles noise std"][i]}
                r, fit = single.size_profiles(
                    profs, metadata, settings, infos_i)
                shape_r = np.shape(r)
                error = infos_i["Reduced least square"]
                r_error = infos_i["Radius error std"]
                signal_over_noise[i] = infos_i["Signal over noise"]
            except BaseException:
                if settings["KEY_STG_IGNORE_ERROR"]:
                    print(sys.exc_info())
                else:
                    raise
        fits.append(fit)
        radius.append(r)
        errors.append(error)
        r_errors.append(r_error)

    if shape_r is None:
        raise RuntimeError("Can't find a single good frame")

    for i, r in enumerate(radius):
        if r is None:
            radius[i] = np.nan * np.ones(shape_r)

    radius = np.asarray(radius, float)
    for idx, add in enumerate([r is None for r in radius]):
        if add:
            radius[idx] = np.ones(np.shape(radius)[1:]) * np.nan
    infos["Reduced least square"] = errors
    infos["Radius error std"] = r_errors
    infos["Signal over noise"] = signal_over_noise
    return radius, fits


def savedata(data, outpath):
    """Save the data"""
    tifffile.imsave(outpath + '_ims.tif', data)


def plot_and_save(radius, profiles, fits,
                  outpath, settings, infos):
    """Plot the sizing data"""
    framesslices = slice(*settings["KEY_STG_STACK_FRAMESSLICES"])

    infos["Overexposed"] = infos["Overexposed"][framesslices]

    display_data.plot_and_save_stack(
        radius, profiles, fits, infos, settings, outpath)


def process_profiles(profiles, settings, outpath, infos):
    ret = []
    for i, prof in enumerate(profiles):
        if prof is None:
            ret.append(None)
        else:
            if settings["KEY_STG_STAT_STACK"]:
                pxs = infos["Pixel size"]
            else:
                pxs = infos["Pixel size"][i]
            ret.append(dp.process_profiles(prof, settings, outpath, pxs))

    return ret
