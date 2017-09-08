# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:26:48 2017

@author: quentinpeter
"""
import registrator.image as ir
import numpy as np
import scipy.ndimage.measurements as msr
import background_rm as rmbg
from .. import profile as dp
import scipy
from . import bright, commun
gfilter = scipy.ndimage.filters.gaussian_filter1d
from scipy.ndimage.morphology import binary_erosion
import warnings
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
        bg = ir.rotate_scale(bg, -angle, 1, borderValue=np.nan)

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


def channels_mask(bg, chwidth, wallwidth, Nprofs, angle=None, edgesOut=None):
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
    maxpos = channels_edges(bg, chwidth, wallwidth, Nprofs=Nprofs, angle=angle)
    if edgesOut is not None:
        edgesOut[:] = maxpos
    # Fill mask
    mask = np.ones(bg.shape)
    for i in range(len(maxpos) // 2):
        mask[:, maxpos[2 * i]:maxpos[2 * i + 1]] = 0
    return mask


def bg_angle(im, bg, Nprofs, infoDict=None):
    """
    get the angle by remove_curve_background

    Parameters
    ----------
    im:  2d array
        image containning the 4 channels
    bg:  2d array
        background
    Nprofs: integer
        the numbers of channels
    infoDict: dict
        infos out

    Returns
    -------
    angle: float
        the image orientation angle
    """
    tmpout = rmbg.remove_curve_background(im, bg, infoDict=infoDict,
                                          bgCoord=True)
    if infoDict is not None:
        infoDict['BrightInfos'] = bright.image_infos(tmpout, Nprofs)
    return dp.image_angle(tmpout)


def remove_bg(im, bg, chwidth, wallwidth, Nprofs, edgesOut=None):
    """
    Flatten and background subtract images

    Parameters
    ----------
    im:  2d array
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
    # Get bg angle (the other images are the same)
    infoDict = {}
    angle = bg_angle(im, bg, Nprofs, infoDict=infoDict)
    approxwidth = infoDict['BrightInfos']['width']
    approxpixsize = (chwidth + wallwidth) / approxwidth
    # Get the mask
    maskbg = channels_mask(bg,
                           chwidth / approxpixsize,  # to pix
                           wallwidth / approxpixsize,  # to pix
                           Nprofs,
                           angle=angle,
                           edgesOut=edgesOut)
    # rotate and flatten the bg
    bg = ir.rotate_scale(bg, -angle, 1, borderValue=np.nan)
    im = ir.rotate_scale(im, -angle, 1, borderValue=np.nan)

    maskim = ir.rotate_scale_shift(maskbg, infoDict['diffAngle'],
                                   infoDict['diffScale'],
                                   infoDict['offset'],
                                   borderValue=np.nan) > .5

    maskim = binary_erosion(maskim, iterations=15)
    # Get Intensity
    ret = rmbg.remove_curve_background(im, bg, maskbg=maskbg, maskim=maskim,
                                       bgCoord=True, reflatten=True)
    return ret


def extract_profiles(im, bg, Nprofs, chwidth, wallwidth, ignore=0,
                     imslice=None, data_dict=None):
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
    data_dict: dict, defaults None
        Output to get the profiles and fits

    Returns
    -------
    profiles: 2d array
        list of profiles
    """
    # get edges
    edges = np.empty(Nprofs * 2, dtype=int)
    # Get flattened image
    flat_im = remove_bg(im, bg, chwidth, wallwidth, Nprofs, edgesOut=edges)
    # Get channel width
    widthpx = int(np.mean(np.diff(edges)[::2]))

    if (edges[1] + edges[0]) / 2 < widthpx:
        raise RuntimeError("Edges incorrectly detected.")

    centers = (edges[1::2] + edges[::2]) / 2

    profiles = commun.extract_profiles(flat_im, centers, chwidth, ignore,
                                       chwidth / widthpx, imslice=imslice)

    if data_dict is not None:
        data_dict["image"] = flat_im

    return profiles


#    # Profile
#    imProf = commun.image_profile(flat_im)
#    # Output profiles
#    profiles = np.empty((Nprofs, width), dtype=float)
#    # Extract profiles
#    firstcenter = None
#    for i, (e, prof) in enumerate(zip(2*centers, profiles)):
#        # e is 2*center of the channel
#        amin = (e - width) // 2
#        amax = (e + width) // 2
#        p = imProf[amin:amax]
#        # All even profiles are switched
#        if i % 2 == 1:
#            p = p[::-1]
#        # Align by detecting center
#        c = dp.center(p)
#        if firstcenter is not None:
#            diff = c - firstcenter
#            if i % 2 == 1:
#                diff *= -1
#            diff = int(diff)
#            p = imProf[amin + diff:amax + diff]
#            if i % 2 == 1:
#                p = p[::-1]
#        else:
#            firstcenter = c
#        prof[:] = p
#    # If image is inverted
#    if profiles[-1].max() > profiles[0].max():
#        profiles = profiles[::-1]
#
#    """
#    from matplotlib.pyplot import plot, figure, imshow
#    figure()
#    plot(np.nanmean(flat_im[:100], 0))
#    plot(np.nanmean(flat_im[-100:], 0))
#    #"""
#
#    return profiles
