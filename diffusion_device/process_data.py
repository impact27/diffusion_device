# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:32:20 2017

@author: quentinpeter
"""
from . import keys, input_files
from . import scans, single_channel_image, multi_channels_image, display_data


def full_fit(settingsfn, metadatafn, outpath=None):
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
    pixel_size: float or list of floats
        The detected pixel size. List for movie.
    data: array of floats
        The data that was analysed
    data_type: string
        The type of data
    overexposed: list of bool
        For each data file, is the file overexposed?

    """

    # Get infos
    metadata = input_files.loadMetadata(metadatafn)
    settings = input_files.loadSettings(settingsfn)

    # Get type
    data_type = metadata[keys.KEY_MD_TYPE]

    mod = get_module(data_type)

    data, state = mod.load_data(metadata)
    data, pixel_size, *infos = mod.process_data(data, metadata, settings)
    profiles = mod.get_profiles(metadata, settings, data, pixel_size, *infos)
    radius, fits = mod.size_profiles(profiles, pixel_size, metadata, settings)

    if outpath is not None:
        outpath = display_data.prepare_output(outpath, settingsfn, metadatafn)
        mod.plot_and_save(
            radius, profiles, fits, pixel_size, data, state, outpath, settings)

    return radius, profiles, fits, pixel_size, data, state


def get_module(data_type):

    if data_type == 'scans':
        return scans

    elif data_type == '12pos':
        return single_channel_image

    elif data_type == '4pos':
        return multi_channels_image

    elif data_type == '4pos_stack':
        return multi_channels_image.stack

# The modules define the following functions:
#data, state = load_data(metadata)
#data, pixel_size, *infos = process_data(data, metadata, settings)
#profiles = get_profiles(metadata, settings, data, pixel_size, *infos)
#radius, fits = size_profiles(profiles, pixel_size, metadata, settings)
#plot_and_save(radius, profiles, fits, pixel_size, data, outpath, settings)
