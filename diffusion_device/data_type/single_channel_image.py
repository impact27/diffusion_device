# -*- coding: utf-8 -*-
"""
Analyse 12 positions device

Created on Fri Mar 17 10:26:20 2017

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
import background_rm as rmbg
import registrator.image as ir
import warnings
import cv2
from scipy import interpolate
from registrator.image import is_overexposed
import tifffile

from .. import display_data
from . import images_files
from .. import profile as dp

warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)


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
    data, backgrounds = images_files.process_background(data, metadata)
    data, pixel_size = process_images(data, backgrounds, metadata, settings)
    infos["Pixel size"] = pixel_size
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
    channel_width = metadata["KEY_MD_WY"]
    Npix = int(channel_width // pixel_size) + 1
    profiles = np.zeros((len(data), Npix))
    flowdir = metadata["KEY_MD_FLOWDIR"]
    noise = np.zeros(len(data))
    for i, (im, fd) in enumerate(zip(data, flowdir)):
        if fd == 'u':
            pass
        elif fd == 'r':
            im = np.rot90(im, 1)
        elif fd == 'd':
            im = np.rot90(im, 2)
        elif fd == 'l':
            im = np.rot90(im, 3)
        else:
            raise RuntimeError('Unknown orientation')
        profiles[i], noise[i] = extract_profile(im, pixel_size, channel_width)

    noise = np.mean(noise)
    infos["Profiles noise std"] = noise
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
    tifffile.imsave(outpath + '_ims.tif', data)


def plot_and_save(radius, profiles, fits,
                  outpath, settings, infos):
    """Plot the sizing data"""
    display_data.plot_and_save(
        radius, profiles, fits, infos, outpath)


def process_images(images, backgrounds, metadata, settings, rebin=2):
    """
    Get the hydrodynamic radius from the images

    Parameters
    ----------
    images: 1d list of images
        images to treat
    backgrounds: 1d list of image, default None
        background images
        if None, no background image
    metadata: dict
        The metadata
    settings: dict
        The settings
    rebin: int, defaults 2
        Rebin factor to speed up code

    Returns
    -------
    r: float
        Radius in [m]

    """

    # Check images is numpy array
    images = np.asarray(images)
    pixel_size = metadata["KEY_MD_PIXSIZE"]
    Wy = metadata["KEY_MD_WY"]

    if backgrounds is not None:
        # Check bgs is numpy array
        backgrounds = np.asarray(backgrounds)

    if rebin > 1:
        size = tuple(np.array(np.shape(images)[1:][::-1]) // rebin)
        images = np.array(
            [cv2.resize(im, size, interpolation=cv2.INTER_AREA)
             for im in images])
        pixel_size *= rebin

        if backgrounds is not None:
            size = tuple(np.array(np.shape(backgrounds)[1:][::-1]) // rebin)
            backgrounds = np.array(
                [cv2.resize(im, size, interpolation=cv2.INTER_AREA)
                 for im in backgrounds])

    # Get flat images
    if backgrounds is None:
        # Single images
        flatimages = np.asarray(
            [flat_image(im, pixel_size, Wy)
             for im in images])
    else:
        # images and background
        flatimages = np.asarray(
            [remove_bg(im, bg, pixel_size, Wy)
             for im, bg in zip(images, backgrounds)])

    # Orientate
    for flatim in flatimages:
        flatim[:] = ir.rotate_scale(flatim, -dp.image_angle(flatim),
                                    1, borderValue=np.nan)

    return flatimages, pixel_size


def remove_bg(im, bg, pixel_size, chanWidth):
    """
    Remove background from image

    Parameters
    ----------
    im: 2d array
        image
    bg: 2d array
        background
    pixel_size: float
        pixel size in [m]
    chanWidth: float
        channel width  in [m]

    Returns
    -------
    im: 2d array
        The processed image

    """
    im = np.array(im, dtype=float)
    bg = np.array(bg, dtype=float)
    # remove dust peaks on images
    bg[rmbg.getPeaks(bg, maxsize=50 * 50)] = np.nan
    im[rmbg.getPeaks(im, maxsize=50 * 50)] = np.nan

    # Get the X positions (perpendicular to alignent axis) and check wide
    # enough
    X = np.arange(im.shape[1]) * pixel_size
    if not (1.2 * chanWidth < X[-1]):
        raise RuntimeError("image too small to get entire channel.")

    # Get the approximate expected channel position
    channel = np.absolute(X - X[-1] / 2) < .6 * chanWidth

    # Create mask to ignore channel when flattening image
    mask = np.ones(im.shape, dtype=bool)
    mask[:, channel] = False

    # Get data
    return rmbg.remove_curve_background(im, bg, maskim=mask)


def flat_image(im, pixel_size, chanWidth):
    """
    Flatten the image

    Parameters
    ----------
    im: 2d array
        image
    pixel_size: float
        pixel size in [m]
    chanWidth: float
        channel width  in [m]

    Returns
    -------
    im: 2d array
        The flattened image
    """

    im = np.asarray(im, dtype=float)
    # remove peaks
    im[rmbg.getPeaks(im, maxsize=20 * 20)] = np.nan
    # straighten
    angle = dp.image_angle(im - np.nanmedian(im))
    im = ir.rotate_scale(im, -angle, 1, borderValue=np.nan)

    # Get center
    prof = np.nanmean(im, 0)  # TODO: Maybe median?
    flatprof = prof - np.linspace(np.nanmean(prof[:len(prof) // 10]),
                                  np.nanmean(prof[-len(prof) // 10:]),
                                  len(prof))
    flatprof[np.isnan(flatprof)] = 0
    x = np.arange(len(prof)) - dp.center(flatprof)  # TODO: Fail ->np.argmax?
    x = x * pixel_size

    # Create mask
    channel = np.abs(x) < chanWidth / 2
    mask = np.ones(np.shape(im))
    mask[:, channel] = 0

    # Flatten
    im = im / rmbg.polyfit2d(im, mask=mask) - 1

    """
    from matplotlib.pyplot import figure, imshow, plot
    figure()
    imshow(im)
    imshow(mask, alpha=.5, cmap='Reds')
#    plot(x, flatprof)
#    plot(x, np.correlate(flatprof, flatprof[::-1], mode='same'))
    #"""

    return im


def extract_profile(flatim, pixel_size, chanWidth, center=None,
                    *, reflatten=True, ignore=10):
    """
    Get profile from a flat image

    Parameters
    ----------
    flatim: 2d array
        flat image
    pixel_size: float
        pixel size in [m]
    chanWidth: float
        channel width  in [m]
    center: float
        The position of the center of the profile
    reflatten: Bool, defaults True
        Should we reflatten the profile?
    ignore: int, defaults 10
        The number of pixels to ignore if reflattening

    Returns
    -------
    im: 2d array
        The flattened image
    """

    # get profile
    prof = np.nanmean(flatim, 0)

    # Center X
    X = np.arange(len(prof)) * pixel_size

    if center is None:
        center = dp.center(prof) * pixel_size
        inchannel = np.abs(X - center) < .45 * chanWidth
        center = dp.center(prof[inchannel]) + np.argmax(inchannel)

    X = X - center * pixel_size

    # get what is out
    out = np.logical_and(np.abs(X) > .55 * chanWidth, np.isfinite(prof))

    if reflatten:
        # fit ignoring extreme 10 pix
        fit = np.polyfit(X[out][ignore:-ignore], prof[out][ignore:-ignore], 2)
        bgfit = fit[0] * X**2 + fit[1] * X + fit[2]

        # Flatten the profile
        prof = (prof + 1) / (bgfit + 1) - 1

    # We restrict the profile to channel width - widthcut
    Npix = int(chanWidth // pixel_size) + 1

    Xc = np.arange(Npix) - (Npix - 1) / 2
    Xc *= pixel_size

    finterp = interpolate.interp1d(X, prof, bounds_error=False, fill_value=0)

    valid = np.logical_and(X > chanWidth / 2, np.isfinite(prof))
    noise = np.std(prof[valid])
    """
    from matplotlib.pyplot import figure, imshow, plot
    figure()
    plot(X, prof)
    #"""
    return finterp(Xc), noise

    """
    from matplotlib.pyplot import figure, imshow, plot
    figure()
    imshow(flatim)
    plot([c-Npix//2, c-Npix//2], [5, np.shape(flatim)[0]-5], 'r')
    plot([c+Npix//2, c+Npix//2], [5, np.shape(flatim)[0]-5], 'r')
    figure()
    pr=np.nanmean(flatim, 0)
    plot(pr)
    plot([c-Npix//2, c-Npix//2], [np.nanmin(pr), np.nanmax(pr)], 'r')
    plot([c+Npix//2, c+Npix//2], [np.nanmin(pr), np.nanmax(pr)], 'r')
    #"""


def process_profiles(profiles, settings, outpath, infos):
    return dp.process_profiles(
        profiles, settings, outpath, infos["Pixel size"])

#    return prof[channel]

#
# def outChannelMask(im, chAngle=0):
#    """Creates a mask that excludes the channel
#
#    Parameters
#    ----------
#    im: 2d array
#        The image
#    chAngle: number
#        The angle of the channel in radians
#
#    Returns
#    -------
#    mask: 2d array
#        the mask excluding the channel
#
#    Notes
#    -----
#    The channel should be clear(ish) on the image.
#    The angle should be aligned with the channel
#
#
#    """
#    im = np.array(im, dtype='float32')
#    # Remove clear dust
#    mask = rmbg.backgroundMask(im, nstd=6)
#    im[~mask] = np.nan
#
#    # get edge
#    scharr = cr.Scharr_edge(im)
#    # Orientate image along x if not done
#    if chAngle != 0:
#        scharr = ir.rotate_scale(scharr, -chAngle, 1, np.nan)
#
#    # get profile
#    prof = np.nanmean(scharr, 1)
#    # get threshold
#    threshold = np.nanmean(prof) + 3 * np.nanstd(prof)
#    mprof = prof > threshold
#    edgeargs = np.flatnonzero(mprof)
#
#    if edgeargs.size > 2:
#        mask = np.zeros(im.shape)
#        mask[edgeargs[0] - 5:edgeargs[-1] + 5, :] = 2
#        if chAngle != 0:
#            mask = ir.rotate_scale(mask, chAngle, 1, np.nan)
#        mask = np.logical_and(mask < 1, np.isfinite(im))
#    else:
#        mask = None
#    return mask
#
#
# def outGaussianBeamMask(data, chAngle=0):
#    """
#    get the outside of the channel from a gaussian fit
#
#    Parameters
#    ----------
#    data: 2d array
#        The image
#    chAngle: number
#        The angle of the channel in radians
#
#    Returns
#    -------
#    mask: 2d array
#        the mask excluding the channel
#
#    """
#    data = np.asarray(data)
#
#    # Filter to be used
#    gfilter = scipy.ndimage.filters.gaussian_filter1d
#
#    # get profile
#    if chAngle != 0:
#        data = ir.rotate_scale(data, -chAngle, 1, np.nan)
#    profile = np.nanmean(data, 1)
#
#    # guess position of max
#    amax = profile.size // 2
#
#    # get X and Y
#    X0 = np.arange(profile.size) - amax
#    Y0 = profile
#
#    # The cutting values are when the profiles goes below zero
#    rlim = np.flatnonzero(np.logical_and(Y0 < 0, X0 > 0))[0]
#    llim = np.flatnonzero(np.logical_and(Y0 < 0, X0 < 0))[-1]
#
#    # We can now detect the true center
#    fil = gfilter(profile, 21)
#    X0 = X0 - X0[np.nanargmax(fil[llim:rlim])] - llim
#
#    # restrict to the correct limits
#    X = X0[llim:rlim]
#    Y = Y0[llim:rlim] - np.nanmin(Y0)
#
#    # Fit the log, which should be a parabola
#    c = np.polyfit(X, np.log(Y), 2)
#
#    # Deduce the variance
#    var = -1 / (2 * c[0])
#
#    # compute the limits (3std, restricted to half the image)
#    mean = np.nanargmax(fil[llim:rlim]) + llim
#    dist = int(3 * np.sqrt(var))
#    if dist > profile.size // 4:
#        dist = profile.size // 4
#    llim = mean - dist
#    if llim < 0:
#        return None
#    rlim = mean + dist
#    if rlim > profile.size:
#        return None
#
#    # get mask
#    mask = np.ones(data.shape)
#
#    if chAngle != 0:
#        idx = np.indices(mask.shape)
#
#        idx[1] -= mask.shape[1] // 2
#        idx[0] -= mask.shape[0] // 2
#        X = np.cos(chAngle) * idx[1] + np.sin(chAngle) * idx[0]
#        Y = np.cos(chAngle) * idx[0] - np.sin(chAngle) * idx[1]
#
#        mask[np.abs(Y - mean + mask.shape[0] // 2) < dist] = 0
#
#    else:
#        mask[llim:rlim, :] = 0
#
#    # mask=np.logical_and(mask>.5, np.isfinite(data))
#    mask = mask > .5
#    return mask
#
#    """
#    import matplotlib.pyplot as plt
#    plt.figure()
#    #plot profile and fit
#    valmax=np.nanmax(Y)
#    plt.plot(X0, Y0)
#    plt.plot(X0, valmax*np.exp(-(X0**2)/(2*var))+np.nanmin(Y0))
#    plt.plot([llim-mean, llim-mean], [np.nanmin(Y0), np.nanmax(Y0)], 'r')
#    plt.plot([rlim-mean, rlim-mean], [np.nanmin(Y0), np.nanmax(Y0)], 'r')
#    #"""
