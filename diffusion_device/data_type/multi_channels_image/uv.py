# -*- coding: utf-8 -*-
"""
Analyse uv 4 positions channels

Created on Fri Mar 17 10:26:48 2017

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
import registrator.image as ir
import numpy as np
import background_rm as rmbg
import scipy
from scipy.ndimage.morphology import binary_erosion
import warnings

from ... import profile as dp
from . import bright
from ..images_files import rotate_image

gfilter = scipy.ndimage.filters.gaussian_filter1d
warnings.filterwarnings('ignore', 'invalid value encountered in greater',
                        RuntimeWarning)


def channels_edges(bg, chwidth, wallwidth, Nprofs,
                   angle=None, std=10):
    """
    Get the position of the edges

    Parameters
    ----------
    bg:  2d array
        image containning the 4 channels
    chwidth: int
        The channel width in pixels
    wallwidth: int
        The wall width in pixels
    Nprofs: integer
        the numbers of channels
    angle: float, default None
        if given, angle at which the edges are
    std: integer, default 10
        Tolerence on wall position in pixels

    Returns
    -------
    edges: 1d integer array
        Position in the rotated image of the edges in pixels

    """

    bg = bg / rmbg.polyfit2d(bg)
    if angle is not None:
        bg = rotate_image(bg, -angle)

    prof = gfilter(np.nanmean(bg, 0), 3)
    edges = np.abs(np.diff(prof))
    edges[np.isnan(edges)] = 0
    # create approximate walls
    x = np.arange(len(edges))
    gwalls = np.zeros(len(edges), dtype=float)

    centers = np.arange(Nprofs * 2)
    centers[::2] = wallwidth / 2 + np.arange(Nprofs) * (chwidth + wallwidth)
    centers[1::2] = (wallwidth / 2 + np.arange(Nprofs) * (chwidth + wallwidth)
                     + chwidth)

    for center in centers:
        gwalls += edges.max() * np.exp(-(x - center)**2 / (2 * std**2))
    # Get best fit for approximate walls
    c = int(
        np.correlate(
            edges,
            gwalls,
            mode='same').argmax() -
        len(gwalls) /
        2)

    return centers + c
#    '''
#    from matplotlib.pyplot import plot, figure, imshow
#    figure()
#    imshow(bg)
#    figure()
#    plot(edges)
#    plot(gwalls)
#    figure()
#    plot(np.correlate(edges, gwalls, mode='same'))
#    #'''
#    # Roll
#    gwalls = np.roll(gwalls, c)
#    if c < 0:
#        gwalls[c:] = 0
#    else:
#        gwalls[:c] = 0
#    # label wall position
#    label, n = msr.label(gwalls > .1 * gwalls.max())
#
#    # Get the positions
#    edges = np.squeeze(msr.maximum_position(edges, label, range(1, n + 1)))
#    if not len(edges) == 2 * Nprofs:
#        raise RuntimeError('Did not detect edges')
#    return edges


def channels_mask(bg, chwidth, wallwidth, Nprofs, angle=None, centersOut=None):
    """
    Get the mask from the image

    Parameters
    ----------
    bg:  2d array
        image containning the 4 channels
    chwidth: int
        The channel width in pixels
    wallwidth: int
        The wall width in pixels
    Nprofs: integer
        the numbers of channels
    angle: float
        if given, angle at which the edges are
    edgesOut: 1d array
        output for the edges

    Returns
    -------
    mask: 2d array
        mask marking the channels

    """
    # Find edges position
    edges = channels_edges(bg, chwidth, wallwidth, Nprofs=Nprofs, angle=angle)
    if centersOut is not None:
        centersOut[:] = (edges[1::2] + edges[::2]) / 2
    # Fill mask
    mask = np.ones(bg.shape)
    for i in range(len(edges) // 2):
        mask[:, edges[2 * i]:edges[2 * i + 1]] = 0
    return mask


def remove_curve_background_alt(im, bg, maskim=None, maskbg=None,
                                infoDict=None, reflatten=False,
                                image_coord=False):
    """
    Try to flatten without good features :/
    """
    im = np.asarray(im, dtype='float32')
    bg = np.asarray(bg, dtype='float32')

    if maskim is None:
        if len(np.shape(im)) == 2:
            maskim = rmbg.backgroundMask(im)
        else:
            maskim = rmbg.backgroundMask(im[np.argmax(im)])
    if maskbg is None:
        maskbg = rmbg.backgroundMask(bg, nstd=6)

    # Flatten the image and background
    fim = rmbg.polyfit2d(im, 2, mask=maskim)
    fbg = rmbg.polyfit2d(bg, 2, mask=maskbg)

    if np.any(fim <= 0):
        raise RuntimeError("Image mask too small")

    if np.any(fbg <= 0):
        raise RuntimeError("Background mask too small")

    im = im / fim
    bg = bg / fbg

    bg_cpy = np.copy(bg)
    bg_cpy[rmbg.signalMask(bg)] = np.nan

    pbg = np.nanmean(bg_cpy, 0) - 1
    pbg[np.isnan(pbg)] = 0

    squeeze = False
    if len(np.shape(im)) == 2:
        squeeze = True
        im = im[np.newaxis]

    data = np.zeros_like(im)
    shifts = np.zeros(len(im))
    
    for i, image in enumerate(im):
        image_copy = np.copy(image)
        image_copy[rmbg.signalMask(image)] = np.nan
        pim = np.nanmean(image_copy, 0) - 1
        pim = np.diff(pim)
        pim[np.isnan(pim)] = 0
        cnv = np.correlate(np.abs(pim), np.abs(np.diff(pbg)), mode='full')
        shift = len(pim) - np.argmax(cnv) - 1
        if image_coord:
            data[i] = image - ir.shift_image(bg, (0, -shift), borderValue=np.nan)
        else:
            data[i] = ir.shift_image(image, (0, shift), borderValue=np.nan) - bg
        shifts[i] = shift

    if reflatten:
        data += 1
        data /= rmbg.polyfit2d(data, 2, mask=maskbg)
        data -= 1

    if squeeze:
        data = np.squeeze(data)
        shifts = np.squeeze(shifts)

    if infoDict is not None:
        infoDict['Shift'] = shifts
    return data

def remove_bg(im, bg, chwidth, wallwidth, Nprofs,
              settings, centersOut=None):
    """
    Flatten and background subtract images

    Parameters
    ----------
    im:  2d/3d array
        list of images containning the 4 channels
    bg: 2d array
        Background corresponding to the list
    chwidth: float
        The channel width in [m]
    wallwidth: float
        The wall width in [m]
    Nprofs: integer
        the numbers of channels
    edgesOut: 1d array
        output for the edges

    Returns
    -------
    flatIm: 2d array
        Flattened image

    """
    # Get settings
    goodFeatures = settings["KEY_STG_GOODFEATURES"]
    image_coord = settings["KEY_STG_IMAGE_COORD"]
    
    # Get brightest image if stack
    if len(np.shape(im)) == 3:
        im_tmp = im[np.argmax(np.nanmean(im, axis=(1, 2)))]
    else:
        im_tmp = im
    
    # Get first flattened image
    infoDict = {}
    if goodFeatures:
        data_tmp = rmbg.remove_curve_background(
            im_tmp, bg, infoDict=infoDict, bgCoord=not image_coord)
    else:
        data_tmp = remove_curve_background_alt(
                im_tmp, bg, infoDict=infoDict, image_coord=image_coord)

    # Get angle
    angle = dp.image_angle(data_tmp)
    
    # rotate
    bg = rotate_image(bg, -angle)
    im = rotate_image(im, -angle)
    data_tmp = rotate_image(data_tmp, -angle)

    # Get current centers 
    bright_infos = bright.image_infos(
        data_tmp, Nprofs, chwidth,  wallwidth)
    pixel_size = bright_infos['pixel_size']
    centers = bright_infos['centers']
    
    # Get data mask
    mask_data = np.ones(data_tmp.shape)
    X = np.arange(np.shape(data_tmp)[1])
    for c in centers:
        mask_data[:, np.abs(X - c) < chwidth / pixel_size/2] = 0
    
    if image_coord:
        mask_im = mask_data
        if goodFeatures:
            c, s = np.cos(angle), np.sin(angle)
            R = np.array(((c,-s), (s, c)))
            offset = -infoDict['offset']
            offset = R @ offset
            mask_bg = ir.rotate_scale_shift(
                    mask_im, -infoDict['diffAngle'],
                    1/infoDict['diffScale'], offset,
                    borderValue=np.nan) > .5
        else:
            offset = np.array([0, infoDict['Shift']])
            mask_bg = ir.shift_image(mask_im, offset,
                                borderValue=np.nan) > .5
    else:
        mask_bg = mask_data
        if goodFeatures:
            c, s = np.cos(angle), np.sin(angle)
            R = np.array(((c,-s), (s, c)))
            offset = infoDict['offset']
            offset = R @ offset
            mask_im = ir.rotate_scale_shift(
                    mask_bg, infoDict['diffAngle'],
                    infoDict['diffScale'], offset,
                    borderValue=np.nan) > .5
        else:
            offset = np.array(0, -infoDict['Shift'])
            mask_im = ir.shift_image(mask_bg, offset,
                                borderValue=np.nan) > .5

    if goodFeatures:
        # Get Intensity
        ret = rmbg.remove_curve_background(im, bg,
                                           maskbg=mask_bg, 
                                           maskim=mask_im,
                                           bgCoord=not image_coord,
                                           reflatten=True)
    else:
        ret = remove_curve_background_alt(
            im, bg, mask_im, mask_bg, reflatten=True, image_coord=image_coord)
    
    if centersOut is not None:
        im = ret
        if len(np.shape(im)) == 3:
            im = im[np.argmax(np.nanmean(im, axis=(1, 2)))]
        centersOut[:] = bright.image_infos(
                im, Nprofs, chwidth, wallwidth)['centers']
    return ret


def extract_data(im, bg, Nprofs, chwidth, wallwidth, settings):
    """
    Extract diffusion profiles


    Parameters
    ----------
    im:  2d array
        image containning the 4 channels
    bg: 2d array
        Background image
    Nprofs: integer
        the numbers of channels
    chwidth: float
        The channel width in [m]
    wallwidth: float
        The wall width in [m]

    Returns
    -------
    profiles: 2d array
        list of profiles
    """
    
    # get edges
    centers = np.empty(Nprofs, dtype=int)
    # Get flattened image
    flat_im = remove_bg(im, bg, chwidth, wallwidth, Nprofs, centersOut=centers,
                        settings=settings)
    # Get channel width
    pixel_size = (chwidth + wallwidth) / np.mean(np.diff(centers))

    return flat_im, centers, pixel_size
