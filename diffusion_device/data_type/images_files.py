# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 07:30:56 2017

@author: quentinpeter
"""
import numpy as np
from tifffile import imread

from .. import keys


def process_background(data, metadata):
    backgrounds = get_background(metadata)
    data, backgrounds = remove_optics_background(data, backgrounds, metadata)
    data, backgrounds = clip_border(data, backgrounds, metadata)
    return data, backgrounds


def load_images(filename):
    """ Load the data files and do some preprocessing

    Parameters
    ----------
    metadata: dict
        The metadata informations

    Returns
    -------
    ims: array
        image or array of data
    bg: array
        background or array of backgrounds
    """

    # load data
    if isinstance(filename, (list, tuple)):
        data = np.asarray([load_image(fn) for fn in filename])
    else:
        data = load_image(filename)

    return data


def load_image(fn):
    ims = imread(fn)
    if len(ims.shape) == 3:
        ims = np.squeeze(ims[np.logical_not(np.all(ims == 0, (1, 2)))])
    return ims


def get_background(metadata):
    backgrounds = None
    background_fn = metadata[keys.KEY_MD_BGFN]
    if background_fn is not None:
        backgrounds = load_images(background_fn)
    return backgrounds


def remove_optics_background(images, backgrounds, metadata):
    # Remove background from optics
    optics_bgfn = metadata[keys.KEY_MD_OPBGFN]
    if optics_bgfn is not None:
        exposure = metadata[keys.KEY_MD_EXP]
        background_exposure = metadata[keys.KEY_MD_BGEXP]
        optics_background_exposure = metadata[keys.KEY_MD_OPBGEXP]

        optics = load_images(optics_bgfn) / optics_background_exposure
        images = images / exposure - optics
        if backgrounds is not None:
            backgrounds = backgrounds / background_exposure - optics

    return images, backgrounds


def clip_border(images, backgrounds, metadata):
    imborder = metadata[keys.KEY_MD_BORDER]
    if imborder is not None:
        # Remove Border
        images = images[..., imborder[0]:imborder[1],
                        imborder[2]:imborder[3]]
        if backgrounds is not None:
            # Remove Border
            backgrounds = backgrounds[..., imborder[0]:imborder[1],
                                      imborder[2]:imborder[3]]
    return images, backgrounds
