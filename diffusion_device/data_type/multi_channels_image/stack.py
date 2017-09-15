# -*- coding: utf-8 -*-
"""
Module for movies of 4 pos channels

Created on Wed Sep 13 10:15:41 2017

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

from registrator.image import is_overexposed
import numpy as np

from .. import multi_channels_image as single

from ... import keys, display_data
from .. import images_files


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
    plotpos = settings[keys.KEY_STG_STACK_POSPLOT]
    display_data.plot_and_save_stack(
        radius, profiles, fits, pixel_size, data, state, outpath, plotpos)
