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
import background_rm as rmbg

from ..images_files import rotate_image
from ... import profile as dp
from .. import single_scan

def image_infos(image, number_profiles, chwidth,  wallwidth):
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
    centers, pixel_size = straight_image_infos(
        image, number_profiles, chwidth,  wallwidth)

    retdict = {
        'angle': angle,
        'centers': centers,
        'pixel_size': pixel_size}
    return retdict


def straight_image_infos(image, number_profiles, chwidth,  wallwidth):
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
    # Get profile
    profiles = np.nanmean(image - np.nanmedian(image), 0)
    
    return single_scan.get_scan_centers(profiles, number_profiles, chwidth,  wallwidth)

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
    centers, pixel_size = straight_image_infos(
        rep_image, number_profiles, chwidth,  wallwidth)
    # get mask
    mask = np.ones(np.shape(rep_image)[-1])
    for i in range(number_profiles):
        amin = centers[i] - frac * chwidth / pixel_size
        amax = centers[i] + frac * chwidth / pixel_size

        if amin < 0:
            amin = centers[i] - .5 * chwidth / pixel_size
            if amin < 0:
                amin = 0

        if amax > len(mask):
            amax = centers[i] + .5 * chwidth / pixel_size
            if amax > len(mask):
                amax = len(mask)

        mask[int(amin):int(amax)] = 0
    mask = mask > 0
    mask = np.tile(mask[None, :], (np.shape(rep_image)[0], 1))

    if np.min(image) < 0:
        image -= np.min(image)
    fitted_image = rmbg.polyfit2d(image, mask=mask)
    # Flatten
    if not subtract:
        image = image / fitted_image - 1
    else:
        image = image - fitted_image

    if infosOut is not None:
        infosOut['infos'] = (centers, pixel_size)
    return image


def extract_data(image, number_profiles, chwidth,
                 wallwidth, settings, subtract=False):
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
    flatten = settings["KEY_STG_BRIGHT_FLAT"]
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
        infos['infos'] = straight_image_infos(
            rep_image, number_profiles, chwidth,  wallwidth)

    centers, pixel_size = infos['infos']
    return image, centers, pixel_size


def best_image(images):
    if len(np.shape(images)) == 2:
        return images
    return images[np.argmax(np.nanpercentile(images, 99, axis=(-2, -1)))]
