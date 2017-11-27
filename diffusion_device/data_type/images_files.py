# -*- coding: utf-8 -*-
"""
Useful functions to work on images

Created on Wed Sep 13 07:30:56 2017

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
from tifffile import imread
import registrator.image as ir


def process_background(data, metadata):
    """Remove optical background from data and background, and clip images.

    Parameters
    ----------
    data: array of floats
        The data to process
    metadata: dict
        The metadata informations

    Returns
    -------
    data: array
        data
    backgrounds: array of float
        backgrounds if there is any

    """
    backgrounds = get_background(metadata)
    data, backgrounds = remove_optics_background(data, backgrounds, metadata)
    data, backgrounds = clip_border(data, backgrounds, metadata)
    return data, backgrounds


def load_images(filename):
    """ Load image or list of images

    Parameters
    ----------
    filename: dict
        The image filename

    Returns
    -------
    ims: array
        image
    """

    # load data
    if isinstance(filename, (list, tuple)):
        data = np.asarray([load_image(fn) for fn in filename])
    else:
        data = load_image(filename)

    return data


def load_image(fn):
    """ Load single image

    Parameters
    ----------
    filename: dict
        The image filename

    Returns
    -------
    im: array
        image
    """
    ims = imread(fn)
    if len(ims.shape) == 3:
        ims = np.squeeze(ims[np.logical_not(np.all(ims == 0, (1, 2)))])
    return ims


def get_background(metadata):
    """Load background

    Parameters
    ----------
    metadata: dict
        The metadata informations

    Returns
    -------
    backgrounds: array of float
        backgrounds if there is any

    """
    backgrounds = None
    background_fn = metadata["KEY_MD_BGFN"]
    if background_fn is not None:
        backgrounds = load_images(background_fn)
    return backgrounds


def remove_optics_background(images, backgrounds, metadata):
    """Remove optical background from data and background

    Parameters
    ----------
    images: array of floats
        The data to process
    backgrounds: array of floats
        The data to process
    metadata: dict
        The metadata informations

    Returns
    -------
    images: array
        images
    backgrounds: array of float
        backgrounds if there is any

    """
    # Remove background from optics
    optics_bgfn = metadata["KEY_MD_OPBGFN"]
    if optics_bgfn is not None:
        exposure = metadata["KEY_MD_EXP"]
        background_exposure = metadata["KEY_MD_BGEXP"]
        optics_background_exposure = metadata["KEY_MD_OPBGEXP"]

        optics = np.asarray(load_images(optics_bgfn), "float32")
        # images = images / exposure - optics + np.median(optics)
        # if backgrounds is not None:
        #     backgrounds = (backgrounds / background_exposure
        #                    - optics + np.median(optics))
        factor = np.sum(optics**2) / np.sum(optics * images, axis=(-2, -1))
        if len(np.shape(factor)) == 1:
            factor = factor[:, np.newaxis, np.newaxis]
        images = images * factor - optics + np.median(optics)

        if backgrounds is not None:
            backgrounds = (backgrounds * np.sum(optics**2) / np.sum(optics * backgrounds)
                           - optics + np.median(optics))

    return images, backgrounds


def clip_border(images, backgrounds, metadata):
    """Remove border from data and background

    Parameters
    ----------
    images: array of floats
        The data to process
    backgrounds: array of floats
        The data to process
    metadata: dict
        The metadata informations

    Returns
    -------
    images: array
        images
    backgrounds: array of float
        backgrounds if there is any

    """
    imborder = metadata["KEY_MD_BORDER"]
    if imborder is not None:
        # Remove Border
        images = images[..., imborder[0]:imborder[1],
                        imborder[2]:imborder[3]]
        if backgrounds is not None:
            # Remove Border
            backgrounds = backgrounds[..., imborder[0]:imborder[1],
                                      imborder[2]:imborder[3]]
    return images, backgrounds


def rotate_image(im, angle):
    """Rotate the image or stack of images
    """
    if len(np.shape(im)) == 3:
        for image in im:
            image[:] = ir.rotate_scale(image, angle, 1, borderValue=np.nan)
    else:
        im = ir.rotate_scale(im, angle, 1, borderValue=np.nan)
    return im
