# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:15:41 2017

@author: quentinpeter
"""

from registrator.image import is_overexposed
import numpy as np

from .. import multi_channels_image as single

from .. import keys, images_files, display_data



def load_data(metadata):
    filename = metadata[keys.KEY_MD_FN]
    data = images_files.load_images(filename)
    overexposed = is_overexposed(data)
    return data, overexposed

def process_data(data, metadata, settings):
    data = np.asarray(data, dtype=float)
    centers = np.zeros((len(data), 4))
    pixel_size = np.zeros((len(data)))
    for i in range(len(data)):
        data[i], pixel_size[i], centers[i] = single.process_data(
            data[i], metadata, settings)

    return data, pixel_size, centers

def get_profiles(metadata, settings, data, pixel_size, centers):
    profiles = [single.get_profiles(metadata, settings, im, pxs, cnt, )
                for im, pxs, cnt in zip(data, pixel_size, centers)]
    return profiles

def size_profiles(profiles, pixel_size, metadata, settings):
    radius = []
    fits = []
    for i, (profs, pxs) in enumerate(zip(profiles, pixel_size)):
        r, fit = single.size_profiles(profs, pxs, metadata, settings)
        fits.append(fit)
        radius.append(r)
    radius = np.asarray(radius)
    return radius, fits

def plot_and_save(radius, profiles, fits, pixel_size, data, state,
                  outpath, settings):
    plotpos = settings[keys.KEY_STG_STACKPLOT]
    display_data.plot_and_save_stack(
        radius, profiles, fits, pixel_size, data, state, outpath, plotpos)