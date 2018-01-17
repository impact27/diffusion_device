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
    #Get profile
    profiles = np.nanmean(image - np.nanmedian(image), 0)
    profiles -= np.nanmin(profiles)
    profiles[np.isnan(profiles)] = 0
    
    #Filter heavely and get position of the centers as a first approx.
    filter_width = np.shape(image)[1]/((number_profiles*2+1)*3)
    Hfiltered = gfilter(profiles, filter_width)
    maxs = np.where(maximum_filter1d(Hfiltered, int(filter_width)) == Hfiltered)[0]
    
    #Filter lightly and get 2nd derivative
    fprof = gfilter(profiles, 3)
    # If filter reduces int by 50%, probably a wall
    maxs = maxs[(profiles[maxs] - fprof[maxs]) / profiles[maxs] < .5]
    # Remove sides
    maxs = maxs[np.logical_and(maxs > 3/2*filter_width, maxs < len(fprof) - 3/2*filter_width)]
    maxs = maxs[np.argsort(fprof[maxs])[- number_profiles:]][::-1]
    
    #Sort and check number
    maxs = sorted(maxs)
    if len(maxs) != number_profiles:
        raise RuntimeError("Can't get image infos")
     
    #Get distances
    dist = np.abs(np.diff(maxs))
    dist_even = np.mean(dist[::2])
    dist_odd = np.mean(dist[1::2])
    meandist = 1/2*(dist_even + dist_odd)
    
    #Correct for any off balance
    centers = np.asarray(maxs, float)
    centers[::2] += (dist_even - meandist) /2
    centers[1::2] += (dist_odd - meandist) /2
    
    #Get evenly spaced centers
    start = np.mean(centers - np.arange(number_profiles)*meandist)
    centers = start + np.arange(number_profiles)*meandist
    
    pixel_size = np.abs((chwidth+wallwidth) / meandist)
    
#    from matplotlib.pyplot import figure, show, plot, imshow, title
#    plot(profiles); plot(centers, np.zeros(len(centers)), 'x'); show()
    
    return centers, pixel_size

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
        infos['infos'] = straight_image_infos(
                rep_image, number_profiles, chwidth,  wallwidth)

    centers, pixel_size = infos['infos']
    return image, centers, pixel_size


def best_image(images):
    if len(np.shape(images)) == 2:
        return images
    return images[np.argmax(np.nanpercentile(images, 99, axis=(-2, -1)))]
