# -*- coding: utf-8 -*-
"""
Main entry point for data processing. Will call modules in data_type.
The main purpose is to enforce uniform interface between modules
and load the correct one.

Created on Mon Sep 11 14:32:20 2017

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
from . import keys, display_data
from .data_type import scans, single_channel_image, multi_channels_image


def full_fit(settingsfn, metadatafn, outpath=None):
    """Perform a fit with the imformations found in the settings file

    Parameters
    ----------
    settingsfn: path
        Path to the fit settings file
    metadatafn: path
        Path to the metadata file
    outpath: path
        Path to the save folder

    Returns
    -------
    radius: float or list of floats or 2x list of floats
        If 1 species:
            The fitted radius
        If several species:
            radii, spectrum: The radii and corresponding coefficients
        If movie:
            A list of the above
    profiles: 2d or 3d list of floats
        The extracted profiles. 3d for movie.
    fits: 2d or 3d list of floats
        The Fits. 3d for movie.
    pixel_size: flat or list of floats
        The detected pixel size. List for movie.
    data: array of floats
        The data that was analysed
    overexposed: list of bool
        For each data file, is the file overexposed?

    """
    if outpath is not None:
        outpath = display_data.prepare_output(outpath, settingsfn, metadatafn)

    # Get infos
    metadata = keys.metadata.load_json(metadatafn)
    settings = keys.settings.load_json(settingsfn)

    # Get type
    data_type = metadata["KEY_MD_TYPE"]
    infos = {}
    mod = get_module(data_type)

    data = mod.load_data(metadata, infos=infos)
    data = mod.process_data(data, metadata, settings, infos=infos)

    if outpath is not None:
        mod.savedata(data, outpath)

    profiles = mod.get_profiles(data, metadata, settings, infos=infos)
    profiles = mod.process_profiles(profiles, settings, outpath, infos=infos)

    radius, fits = mod.size_profiles(profiles, metadata, settings, 
                                     infos=infos)

    if outpath is not None:
        mod.plot_and_save(
            radius, profiles, fits, outpath, settings, infos=infos)

    return radius, profiles, fits, data, infos


def get_module(data_type):
    """Returns the correct module corresponding to data_type

    Parameters
    ----------
    data_type: string
        A string corresponding to a data type

    Returns
    -------
    module: python module
        A module defining the following functions:

    data = load_data(metadata, infos)
    data = process_data(data, metadata, settings, infos)
    profiles = get_profiles(data, metadata, settings, infos)
    profiles = process_profiles(profiles, settings, outpath, infos)
    radius, fits = size_profiles(profiles, metadata, settings, infos)
    plot_and_save(radius, profiles, fits, outpath, settings, infos)
    savedata(data, outpath)

    """

    if data_type == 'scans':
        return scans

    elif data_type == '12pos':
        return single_channel_image

    elif data_type == '4pos':
        return multi_channels_image

    elif data_type == '4pos_stack':
        return multi_channels_image.stack
