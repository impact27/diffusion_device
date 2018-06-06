# -*- coding: utf-8 -*-
"""
Analyse confocal scans

Created on Tue Sep 12 13:33:18 2017

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

from .. import profile as dp
from .. import display_data


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
    data = np.loadtxt(filename, delimiter=',')[:, 1]
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
    scan_slice = settings["KEY_SET_SCAN_SLICE"]
    channel_width = metadata["KEY_MD_WY"]
    nchannels = metadata["KEY_MD_NCHANNELS"]
    wall_width = metadata["KEY_MD_WALLWIDTH"]
    background_fn = metadata["KEY_MD_BGFN"]
    
    # Apply scan slice
    if scan_slice is not None:
        data = data[scan_slice[0]:scan_slice[1]]
        
    # Get background
    if background_fn is None:
        return data
    
    bg = np.loadtxt(background_fn, delimiter=',')[:, 1]
    if scan_slice is not None:
        bg = bg[scan_slice[0]:scan_slice[1]]
    
    # Find centers for processing
    centers, pixel_size = dp.get_scan_centers(
            data, nchannels, channel_width,  wall_width)

    # Determine what is in the channels and far from the channels
    X = np.arange(len(data))
    channel = np.any(np.abs(
            X[:, np.newaxis] -  centers[np.newaxis]
            ) * pixel_size < channel_width/2, axis=1)
    far = np.min(np.abs(
            X[:, np.newaxis] -  centers[np.newaxis]
            ), axis=1) * pixel_size > channel_width
    
    # Get dummy profiles and correlate
    p0 = data - np.mean(data[np.logical_not(channel)])
    p1 = bg - np.mean(bg)
    p0[channel] = 0
    p0[far] = 0
    corr = np.correlate(np.tile(p0, 2), p1, mode='valid')[:-1]
    
    # Get offset and apply
    offset = np.argmax(corr)
    if offset > len(p0)/2:
        offset -= len(p0)
    if offset < 0:
        bg = bg[-offset:]
        data = data[: offset]
    else:
        bg = bg[:-offset]
        data = data[offset:]
        centers -= offset
    
    # Get updated mask
    X = np.arange(len(data))
    channel = np.any(np.abs(
            X[:, np.newaxis] -  centers[np.newaxis]
            ) * pixel_size < channel_width/2, axis=1)
    out = np.logical_not(channel)
    
    # Scale background
    newbg = bg * np.sum(data[out]*bg[out])/np.sum(bg[out]*bg[out])
    
    # Subtract
    data = data - newbg
    return data


def get_profiles(data, metadata, settings, infos, outpath=None):
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
    channel_width = metadata["KEY_MD_WY"]
    nchannels = metadata["KEY_MD_NCHANNELS"]
    wall_width = metadata["KEY_MD_WALLWIDTH"]
    flowdir = metadata["KEY_MD_FLOWDIR"]
    ignore = settings["KEY_STG_IGNORE"]
    
    centers, pixel_size = dp.get_scan_centers(
            data, nchannels, channel_width,  wall_width)
    
    infos["Pixel size"] = pixel_size
    infos["Centers"] = centers
    infos["flow direction"] = flowdir
    
    profiles = dp.extract_profiles(data, channel_width, ignore, infos)
    
    profiles, infos["Pixel size"] = dp.process_profiles(
        profiles, metadata, settings, outpath, infos["Pixel size"])
    
    profiles = dp.align_profiles(profiles, data, metadata, settings, infos)
    
    profiles, infos["Pixel size"] = dp.process_profiles(
        profiles, metadata, settings, outpath, infos["Pixel size"])
    
    return profiles


def process_profiles(profiles, metadata, settings, outpath, infos):
    return profiles

def savedata(data, outpath):
    """Save the data"""
    # Todo: implement
    pass


def plot_and_save(radius, profiles, fits, outpath, settings, infos):
    """Plot the sizing data"""
    display_data.plot_and_save(
        radius, profiles, fits, infos, outpath)