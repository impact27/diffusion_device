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
from scipy.ndimage.measurements import label
from scipy import interpolate
from scipy.ndimage.filters import maximum_filter1d
from scipy.ndimage.filters import gaussian_filter1d as gfilter
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

    # Find centers for processing
    centers, pixel_size = get_scan_centers(
        data, nchannels, channel_width,  wall_width)
    infos["Pixel size"] = pixel_size
    infos["Centers"] = centers
    infos["flow direction"] = metadata["KEY_MD_FLOWDIR"]

    # Get background
    if background_fn is None:
        return data

    bg = np.loadtxt(background_fn, delimiter=',')[:, 1]
    if scan_slice is not None:
        bg = bg[scan_slice[0]:scan_slice[1]]

    # Determine what is in the channels and far from the channels
    X = np.arange(len(data))
    channel = np.any(np.abs(
        X[:, np.newaxis] - centers[np.newaxis]
    ) * pixel_size < channel_width/2, axis=1)
    far = np.min(np.abs(
        X[:, np.newaxis] - centers[np.newaxis]
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
        infos["Centers"] = centers

    # Get updated mask
    X = np.arange(len(data))
    channel = np.any(np.abs(
        X[:, np.newaxis] - centers[np.newaxis]
    ) * pixel_size < channel_width/2, axis=1)
    out = np.logical_not(channel)

    # Scale background
    newbg = bg * np.sum(data[out]*bg[out])/np.sum(bg[out]*bg[out])

    # Subtract
    data = data - newbg
    return data


def get_profiles(lin_profiles, metadata, settings, infos, outpath=None):
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
    ignore = settings["KEY_STG_IGNORE"]

    profiles = extract_profiles(
        lin_profiles, channel_width, ignore, infos)

    profiles, infos["Pixel size"] = dp.process_profiles(
        profiles, metadata, settings, outpath, infos["Pixel size"])

    profiles = align_profiles(profiles, lin_profiles,
                              metadata, settings, infos)

    profiles, infos["Pixel size"] = dp.process_profiles(
        profiles, metadata, settings, outpath, infos["Pixel size"])

    return profiles


def savedata(data, outpath):
    """Save the data"""
    # Todo: implement
    pass


def plot_and_save(radius, profiles, fits, outpath, settings, infos):
    """Plot the sizing data"""
    display_data.plot_and_save(
        radius, profiles, fits, infos, outpath)


def max_to_center(maxs, chwidth, wallwidth):
    number_profiles = len(maxs)
    # Get distances
    dist = np.abs(np.diff(maxs))
    dist_even = np.mean(dist[::2])
    dist_odd = np.mean(dist[1::2])
    meandist = 1/2*(dist_even + dist_odd)

    # Correct for any off balance
    centers = np.asarray(maxs, float)
    centers[::2] += (dist_even - meandist) / 2
    centers[1::2] += (dist_odd - meandist) / 2

    # Get evenly spaced centers
    start = np.mean(centers - np.arange(number_profiles) * meandist)
    centers = start + np.arange(number_profiles) * meandist

    pixel_size = np.abs((chwidth+wallwidth) / meandist)
    return centers, pixel_size


def get_scan_centers(profiles, number_profiles, chwidth,  wallwidth):
    """Get centers from a single scan"""
    profiles = profiles - np.nanmin(profiles)
    profiles[np.isnan(profiles)] = 0

    # Filter heavely and get position of the centers as a first approx.
    filter_width = len(profiles)/((number_profiles*2+1)*3)
    Hfiltered = gfilter(profiles, filter_width)
    maxs = np.where(maximum_filter1d(
        Hfiltered, int(filter_width)) == Hfiltered)[0]

    # Filter lightly and get 2nd derivative
    fprof = gfilter(profiles, 3)
    # If max negatuve, not a max
    maxs = maxs[profiles[maxs] > 0]
    # If filter reduces int by 50%, probably a wall
    maxs = maxs[(profiles[maxs] - fprof[maxs]) / profiles[maxs] < .5]
    # Remove sides
    maxs = maxs[np.logical_and(
        maxs > 3/2*filter_width, maxs < len(fprof) - 3/2*filter_width)]
    maxs = maxs[np.argsort(fprof[maxs])[- number_profiles:]][::-1]

    # Sort and check number
    maxs = sorted(maxs)
    if len(maxs) != number_profiles:
        raise RuntimeError("Can't find the center of the channels!")

    centers, pixel_size = max_to_center(maxs, chwidth, wallwidth)

#    from matplotlib.pyplot import figure, show, plot, imshow, title
#    plot(profiles); plot(centers, np.zeros(len(centers)), 'x'); show()

    return centers, pixel_size


def should_switch(flow_direction):
    if flow_direction == 'u' or flow_direction == 'up':
        switch = True
    elif flow_direction == 'd' or flow_direction == 'down':
        switch = False
    else:
        raise RuntimeError("unknown orientation: {}".format(flow_direction))
    return switch


def interpolate_profiles(lin_profiles, centers, flowdir, prof_npix,
                         prof_width, old_pixel_size):
    new_pixel_size = prof_width/prof_npix
    nchannels = len(centers)
    profiles = np.empty((nchannels, prof_npix), dtype=float)

    # Extract profiles
    for i, (cent, fd) in enumerate(zip(centers, flowdir)):

        X = np.arange(len(lin_profiles)) - cent
        finterp = interpolate.interp1d(X * old_pixel_size, lin_profiles)

        Xc = np.arange(prof_npix) - (prof_npix - 1) / 2
        p = finterp(Xc * new_pixel_size)

        if should_switch(fd):
            p = p[::-1]

        profiles[i] = p
    return profiles, new_pixel_size


def extract_profiles(lin_profiles, channel_width, ignore, infos):
    """Extract profiles from a single scan"""
    centers = infos["Centers"]
    flowdir = infos["flow direction"]
    pixel_size = infos["Pixel size"]

    prof_npix = int(np.round(channel_width / pixel_size))

    if (np.min(centers) - prof_npix / 2 < 0 or
            np.max(centers) + prof_npix / 2 > len(lin_profiles)):
        raise RuntimeError('Channel not fully contained in the image')

    profiles, pixel_size = interpolate_profiles(lin_profiles, centers, flowdir, prof_npix,
                                                prof_npix * pixel_size, pixel_size)

    outmask = np.all(np.abs(np.arange(len(lin_profiles))[:, np.newaxis]
                            - centers[np.newaxis]) > .55 * prof_npix, axis=1)
    outmask = np.logical_and(outmask, np.isfinite(lin_profiles))

    lbl, n = label(outmask)
    medians = np.zeros(n)
    stds = np.zeros(n)

    for i in np.arange(n):
        background = lin_profiles[lbl == i + 1]
        medians[i] = np.median(background)
        window = 31
        if len(background) < 3:
            stds[i] = np.sum(np.square(background
                                       - np.median(background)))
        else:
            if len(background) < window:
                window = 2*(len(background)//2) - 1
            stds[i] = np.sum(np.square(background
                                       - savgol_filter(background, window, window//6)))

    # Check the image profiles is not too bad
    if 2 * np.abs(np.median(medians)) > np.max(lin_profiles):
        raise RuntimeError("Large background. Probably incorrect.")

    noise_std = np.sqrt(np.sum(stds) / np.sum(outmask))

    # If image upside down, turn
    if profiles[-1].max() > profiles[0].max():
        profiles = profiles[::-1]
        centers = centers[::-1]
        flowdir = flowdir[::-1]

    infos["Profiles noise std"] = noise_std
    infos["Centers"] = centers
    infos["flow direction"] = flowdir
    infos["Pixel size"] = pixel_size

    return profiles


def align_profiles(profiles, lin_profiles, metadata, settings, infos):
    ignore = settings["KEY_STG_IGNORE"]
    rebin = settings["KEY_STG_REBIN"]
    nchannels = metadata["KEY_MD_NCHANNELS"]
    pixel_size = infos["Pixel size"]
    centers = infos["Centers"]
    flowdir = infos["flow direction"]
    channel_width = metadata["KEY_MD_WY"]
    wall_width = metadata["KEY_MD_WALLWIDTH"]

    prof_npix = np.shape(profiles)[1] * rebin

    __, fits = dp.size_profiles(profiles, metadata, settings, infos)

    profile_slice = dp.ignore_slice(ignore, pixel_size)

    new_centers = np.zeros(nchannels)
    for i, (cent, fd) in enumerate(zip(centers, flowdir)):

        # Get diff between profile and fit
        diff = dp.center(np.correlate(
            profiles[i, profile_slice],
            fits[i, profile_slice],
            mode='full')) - len(profiles[i, profile_slice]) + 1
        diff *= rebin
        switch = should_switch(fd)

        if switch:
            diff *= -1

        new_centers[i] = cent + diff

    meandist = np.mean(np.diff((new_centers[1:] + new_centers[:-1])/2))
    pixel_size = np.abs((channel_width + wall_width) / meandist)

    res = (new_centers - np.min(new_centers) + meandist/2) % meandist
    offset = (np.mean(res[::2]) - np.mean(res[1::2])) / 2

    new_centers -= offset * (-1)**np.arange(len(profiles))

    new_profiles, pixel_size = interpolate_profiles(
        lin_profiles, new_centers, flowdir,
        prof_npix, channel_width, pixel_size)

    infos["Centers"] = new_centers
    infos["Pixel size"] = pixel_size
    return new_profiles
