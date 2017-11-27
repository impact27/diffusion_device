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
import warnings
from scipy.signal import savgol_filter

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
    data = np.asarray([np.loadtxt(fn, skiprows=22) for fn in filename])
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
    flow_dir = metadata["KEY_MD_FLOWDIR"]

    # put scans in the correct order
    if flow_dir is not None:
        for s, o in zip(data, flow_dir):
            if o == 'u':
                s[:] = s[::-1]
            elif o != 'd':
                raise RuntimeError(
                    'Flow direction must be up or down for scans.')

    infos["Pixel size"] = metadata["KEY_MD_PIXSIZE"]
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
    channel_width_px = int(metadata["KEY_MD_WY"] / infos["Pixel size"])
    profiles = scans_to_profiles(data, channel_width_px)
    # Guess measurment noise from savgol filter
    infos["Profiles noise std"] = np.std(profiles - savgol_filter(profiles, 31, 5))
    return profiles


def process_profiles(profiles, settings, outpath, infos):
    return dp.process_profiles(
        profiles, settings, outpath, infos["Pixel size"])


def size_profiles(profiles, metadata, settings, infos):
    """Size the profiles

     Parameters
    ----------
    profiles: 2d array
        List of profiles to fit
    metadata: dict
        The metadata
    settings: dict
        The settings
    infos: dict
        Dictionnary with other infos

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
    zpos = metadata["KEY_MD_SCANZ"]
    return dp.size_profiles(profiles, metadata, settings,
                            infos,
                            zpos=zpos)


def savedata(data, outpath):
    """Save the data"""
    # Todo: implement
    pass


def plot_and_save(radius, profiles, fits, outpath, settings, infos):
    """Plot the sizing data"""
    display_data.plot_and_save(
        radius, profiles, fits, infos, outpath)


def scans_to_profiles(scans, Npix, *,
                      offset_edge_idx=None):
    """Extract profiles from scans

    Parameters
    ----------
    scans:  2d array
        sacns to analyse
    Npix:   integer
        number of pixels in a profile
    offset_edge_idx: integer
        Index of a profile containing an edge and a maximum to detect offset
    offset: integer
        Manual offset

    Returns
    -------
    profiles: 1d array
        The profiles
    """
    offset = 0
    # Init return
    profiles = np.empty((scans.shape[0], Npix))
    scans = np.array(scans)
    if offset_edge_idx is not None and offset_edge_idx < 0:
        offset_edge_idx = len(scans) + offset_edge_idx

    # get the offset if needed
    if offset_edge_idx is not None:
        offset_scan = scans[offset_edge_idx]
        cent = dp.center(offset_scan)
        edge = get_edge(offset_scan)
        offset = np.abs(cent - edge) - Npix / 2
        edgeside = 1
        if edge > cent:
            edgeside = -1

    # For each scan
    for i, s in enumerate(scans):
        # Get the mid point
        if offset_edge_idx is None:
            mid = dp.center(s) - offset
        else:
            if i < offset_edge_idx:
                mid = dp.center(s) - edgeside * offset
            else:
                mid = get_edge(s) + edgeside * Npix / 2
        # First position
        amin = int(mid - Npix / 2)
        # If pixels missings:
        if amin < 0 or amin > len(s) - Npix:
            warnings.warn("Missing pixels, scan not large enough",
                          RuntimeWarning)
            while amin > len(s) - Npix:
                s = np.append(s, s[-1])
            while amin < 0:
                amin += 1
                s = np.append(s[0], s)
        # Get profile
        profiles[i] = s[amin:amin + Npix]

    return profiles


def get_edge(profile):
    """Get the largest edge in the profile

    Parameters
    ----------
    profile:  1d array
        profile to analyse

    Returns
    -------
    edgePos: float
        The edge position
    """
    diff = np.diff(profile)

    diffnorm = diff / (profile[:-1] + profile[1:])

    left_edge = getmaxaround(diff, np.argmax(diffnorm)) + .5
    right_edge = getmaxaround(diff, np.argmin(diffnorm)) + .5

    return left_edge  # , right_edge


def getmaxaround(profile, approxmax, window_r=3):
    valid = slice(approxmax - window_r, approxmax + window_r + 1)
    X = np.arange(len(profile))
    X = X[valid]
    Y = np.log(profile[valid])
    coeff = np.polyfit(X, Y, 2)
    edgePos = -coeff[1] / (2 * coeff[0])
    return edgePos

# def get_profiles(scans, Npix, *,
#                 offset_edge_idx=None):
#    """Extract profiles from scans
#
#    Parameters
#    ----------
#    scans:  2d array
#        sacns to analyse
#    Npix:   integer
#        number of pixels in a profile
#    offset_edge_idx: integer
#        Index of a profile containing an edge and a maximum to detect offset
#    offset: integer
#        Manual offset
#
#    Returns
#    -------
#    profiles: 1d array
#        The profiles
#    """
#    offset=0
#    # Init return
#    profiles = np.empty((scans.shape[0], Npix))
#    scans = np.array(scans)
#    if offset_edge_idx is not None and offset_edge_idx < 0:
#        offset_edge_idx = len(scans) + offset_edge_idx
#
#    # get the offset if needed
#    if offset_edge_idx is not None:
#        offset_scan = scans[offset_edge_idx]
#        cent = dp.center(offset_scan)
#        edge = get_edge(offset_scan)
#        offset = np.abs(cent - edge) - Npix / 2
#        edgeside = 1
#        if edge > cent:
#            edgeside = -1
#
#    # For each scan
#    for i, s in enumerate(scans):
#        # Get the mid point
#        if offset_edge_idx is None:
#            mid = dp.center(s) - offset
#        else:
#            if i < offset_edge_idx:
#                mid = dp.center(s) - edgeside * offset
#            else:
#                mid = get_edge(s) + edgeside * Npix / 2
#        # First position
#        amin = int(mid - Npix / 2)
#        # If pixels missings:
#        if amin < 0 or amin > len(s) - Npix:
#            warnings.warn("Missing pixels, scan not large enough",
#                          RuntimeWarning)
#            while amin > len(s) - Npix:
#                s = np.append(s, s[-1])
#            while amin < 0:
#                amin += 1
#                s = np.append(s[0], s)
#        # Get profile
#        profiles[i] = s[amin:amin + Npix]
#
#    return profiles
