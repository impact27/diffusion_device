# -*- coding: utf-8 -*-
"""
Analyse bright 4 positions channels

Created on Tue Apr  4 11:18:33 2017

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
from scipy.ndimage.filters import maximum_filter1d
from scipy.ndimage.filters import gaussian_filter1d as gfilter
import background_rm as rmbg
import registrator.image as ir

from ..images_files import rotate_image
from ... import profile as dp


def image_infos(image, number_profiles):
    """
    Get the image angle, channel width, proteind offset, and origin

    Parameters
    ----------
    image: 2d array
        The image
    number_profiles: integer
        the numbers of channels

    Returns
    -------
    dict: dictionnary
        dictionnary containing infos

    """
    # Detect Angle
    angle = dp.image_angle(image)
    image = rotate_image(image, -angle)
    # Get channels infos
    w, a, origin = straight_image_infos(image, number_profiles)

    retdict = {
        'angle': angle,
        'origin': origin,
        'width': w,
        'offset': a}
    return retdict


def straight_image_infos(image, number_profiles):
    """
    Get the channel width, proteind offset, and origin from a straight image

    Parameters
    ----------
    image: 2d array
        The image
    number_profiles: integer
        the numbers of channels

    Returns
    -------
    w: float
        Channel width in pixels
    a: float
        offset of the proteins in the channel
    origin: float
        Position of the first channel center

    """
    assert number_profiles >= 4
    width_pixels = np.shape(image)[1] // 10

    profiles = np.nanmean(image - np.nanmedian(image), 0)

    # Find max positions
    fprof = gfilter(profiles, 3)
    maxs = np.where(maximum_filter1d(fprof, width_pixels) == fprof)[0]
    # If filter reduces int by 50%, probably a wall
    maxs = maxs[(profiles[maxs] - fprof[maxs]) / profiles[maxs] < .5]
    # Remove sides
    maxs = maxs[np.logical_and(maxs > 15, maxs < len(fprof) - 15)]
    maxs = maxs[np.argsort(fprof[maxs])[- number_profiles:]][::-1]
#    from matplotlib.pyplot import figure, show, plot, imshow
#    figure()
#    plot(fprof)
#    plot(maximum_filter1d(fprof, 100))
#    for m in maxs:
#        plot([m, m], [0, np.nanmax(fprof)])

    if len(maxs) < number_profiles:
        raise RuntimeError("Can't get image infos")

    profiles -= np.nanmin(profiles)

    maxs = np.asarray(maxs, dtype=float)
    for i, amax in enumerate(maxs):
        amax = int(amax)
        y = np.log(profiles[amax - 10:amax + 10])
        x = np.arange(len(y))
        coeff = np.polyfit(x[np.isfinite(y)], y[np.isfinite(y)], 2)
        maxs[i] = -coeff[1] / (2 * coeff[0]) - 10 + amax

    maxs = np.sort(maxs)

    if maxs[0] < 0 or maxs[-1] > len(profiles):
        raise RuntimeError("Can't get image infos")

    if fprof[int(maxs[0])] > fprof[int(maxs[-1])]:
        # Deduce relevant parameters
        w = (maxs[2] - maxs[0]) / 2
        a = w / 2 + (maxs[0] - maxs[1]) / 2
        origin = maxs[0] - a
        lastdist = maxs[3] - (origin + 3 * w - a)

    else:
        # Deduce relevant parameters
        w = (maxs[3] - maxs[1]) / 2
        a = w / 2 + (maxs[2] - maxs[3]) / 2
        origin = maxs[3] + a - 3 * w
        lastdist = maxs[0] - (origin + a)

    if not w > 0:
        raise RuntimeError('Something went wrong while analysing the images')
    # if position 4 is remotely correct, return infos
    if (np.abs(lastdist) > w / 2  # Last too far
            or np.any(np.isnan((a, w, origin, maxs[3])))  # got nans
            or origin - a + 3.2 * w > len(profiles)  # Right side not in
            or origin + a - .2 * w < 0):  # left side not in
        raise RuntimeError("Can't get image infos")
    return w, a, origin


def flat_image(image, chwidth, wallwidth, number_profiles, *,
               frac=.6, infosOut=None, subtract=False):
    """
    Flatten input images

    Parameters
    ----------
    image: 2d array
        The image
    chwidth: float
        The channel width in [m]
    wallwidth: float
        The wall width in [m]
    number_profiles: integer
        the numbers of channels
    frac: float
        fraction of the profile taken by fluorescence from channels
    infosOut: dict, defaults None
        dictionnary containing the return value of straight_image_infos
    subtract: Bool
        Should the shape be subtracted instead of divided

    Returns
    -------
    image: 2d array
        The flattened image

    """
    # Get a representative image of the stack (or the image itself)
    rep_image = best_image(image)
    # Detect Angle
    angle = dp.image_angle(rep_image - np.median(rep_image))
    rep_image = rotate_image(rep_image, -angle)
    # Get channels infos
    w, a, origin = straight_image_infos(
        rep_image, number_profiles=number_profiles)
    # get mask
    mask = np.ones(np.shape(rep_image)[-1])
    for i in range(number_profiles):
        amin = origin + i * w - frac * w * chwidth / (chwidth + wallwidth)
        amax = origin + i * w + frac * w * chwidth / (chwidth + wallwidth)

        if amin < 0:
            amin = origin + i * w - .5 * w * chwidth / (chwidth + wallwidth)
            if amin < 0:
                amin = 0

        if amax > len(mask):
            amax = origin + i * w + .5 * w * chwidth / (chwidth + wallwidth)
            if amax > len(mask):
                amax = len(mask)

        mask[int(amin):int(amax)] = 0
    mask = mask > 0
    mask = np.tile(mask[None, :], (np.shape(rep_image)[0], 1))

    fitted_image = rmbg.polyfit2d(image, mask=mask)
    # Flatten
    if not subtract:
        image = image / fitted_image - 1
    else:
        image = image - fitted_image

    if infosOut is not None:
        infosOut['infos'] = (w, a, origin)
    return image


def extract_data(image, number_profiles, chwidth,
                 wallwidth, flatten=False, subtract=False):
    '''
    Extract profiles from image

    Parameters
    ----------
    image: 2d or 3d array
        The flat image
    number_profiles: integer
        the numbers of channels
    chwidth: float
        The channel width in [m]
    wallwidth: float
        The wall width in [m]
    flatten: Bool, Defaults False
        Should the image be flatten

    Returns
    -------
    profiles: 2d array
        The four profiles
    '''
    image = np.asarray(image)
    infos = {}
    if flatten:
        image = flat_image(image, chwidth, wallwidth, number_profiles,
                           infosOut=infos, subtract=subtract)
    # Get a representative image of the stack (or the image itself)
    rep_image = best_image(image)
    angle = dp.image_angle(rep_image)
    image = rotate_image(image, -angle)
    rep_image = rotate_image(rep_image, -angle)

    if not flatten:
        infos['infos'] = straight_image_infos(rep_image, number_profiles)

    w, a, origin = infos['infos']
    centers = origin + np.arange(number_profiles) * w
    pixel_size = (chwidth + wallwidth) / w
    return image, centers, pixel_size


def best_image(images):
    if len(np.shape(images)) == 2:
        return images
    return images[np.argmax(np.nanmean(images, axis=(-2, -1)))]
