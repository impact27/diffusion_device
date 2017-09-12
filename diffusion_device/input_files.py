# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:23:24 2017

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
import json
import os

from . import keys


def optional(dic, key, val):
    """Set valur in dictionnary if not None

    Parameters
    ----------
    dic: dictionnary
        the dictionnary
    key: string
        the key
    val: object
        Value that might be None
    """
    if val is not None:
        dic[key] = val


def createMetadata(metadata_filename,
                   filename, exposure = None,
                   background_filename = None, background_exposure = None,
                   optics_background_filename = None, 
                   optics_background_exposure = None,
                   Wz = None, Wy = None, Q = None, readingpos = None, 
                   pixelsize = None, date = None, analyte = None, 
                   buffer = None, device = None,
                   wallwidth = None, nchannels = None, border = None):
    """Creates the metadata for a file name

    Parameters
    ----------
    metadata_filename: path
        Where to save the metadata
    filename: path
        The path to the image relative to the metadata file
    exposure: float
        Exposure time of the image [s]
    background_filename: path
        path to the background file
    background_exposure: float
        Exposure time of the background [s]
    optics_background_filename: path
        path to the optics background file
    optics_background_exposure: float
        Exposure time of the optics background [s]
    Wz: float
        Height of the channel in [m]
    Wy: float
        Width of the channel in [m]
    Q: float
        Flow rate in [ul/h]
    readingpos; list float
        The reading positions in [m]
    pixelsize: float
        The pixel size in [m]
    date: string
        YYYYMMDD
    analyte: string
        Infos about the analyte
    buffer: string
        infos about the buffer
    device: string
        infos about the device
    wallwidth: float, default None
        If this is a multichannel image, the width of the wall in [m]
    nchannels: int, default None
        If this is a multichannel image, The number of channels
    border: 4 ints, default None
        The borber to apply on the image (t, d, l, r)
    """
    metadata = {}

    metadata[keys.KEY_MD_FN] = filename
    metadata[keys.KEY_MD_EXP] = exposure

    optional(metadata, keys.KEY_MD_BGFN, background_filename)
    optional(metadata, keys.KEY_MD_BGEXP, background_exposure)

    optional(metadata, keys.KEY_MD_OPBGFN, optics_background_filename)
    optional(metadata, keys.KEY_MD_OPBGEXP, optics_background_exposure)

    metadata[keys.KEY_MD_WZ] = Wz
    metadata[keys.KEY_MD_WY] = Wy

    optional(metadata, keys.KEY_MD_WALLWIDTH, wallwidth)
    optional(metadata, keys.KEY_MD_NCHANNELS, nchannels)

    metadata[keys.KEY_MD_Q] = Q
    metadata[keys.KEY_MD_PIXSIZE] = pixelsize

    metadata[keys.KEY_MD_RPOS] = readingpos

    if border == [None, None, None, None]:
        border = None
    optional(metadata, keys.KEY_MD_BORDER, border)

    metadata[keys.KEY_MD_DATE] = date
    metadata[keys.KEY_MD_ANALYTE] = analyte
    metadata[keys.KEY_MD_BUFFER] = buffer
    metadata[keys.KEY_MD_DEVICE] = device
    # Optional

    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)


def createFitSettings(settingsfn, rmin, rmax, rstep,
                      ignore=None, firstmethod=None,
                      fitpos=None, flatten=None,
                      framesSlices=None, nspecies=1, Zgrid=None,
                      normalise=None, imslices=None):
    """Creates the fit settings for the fitting

    Parameters
    ----------
    settingsfn: path
        The name of the settings file
    metadata_filename: path
        path to the metadata file relative to the settings
    rmin, rmax, rstep: 3 floats
        min, max, and step for the radius[m]
    ignore: float, default None
        The border to ignore. None is 0
    firstmethod: string, default None
        The method to process the first profile. See profiles.py
    fitpos: array of ints, default None
        It this is a multichannel images, the positions to fit. None=all.
    flatten: Bool, default None
        If there is no background, should the image be flattened
        Use if there is no background at all.
    framesSlices: 2 ints, default None
        If this is a movie, the slice to apply
    nspecies: int
        The number of species to fit. 0=all.
    Zgrid: int, default None
        The number of Z slices in the simulation
    normalise: bool, default None
        Should the profiles be normalized?
    imslice: 2 floats, default None
        [Y center, Y width] [m]


    """
    settings = {}
    settings[keys.KEY_STG_R] = (rmin, rmax, rstep)
    # Optional
    optional(settings, keys.KEY_STG_IGNORE, ignore)
    optional(settings, keys.KEY_STG_POS0FILTER, firstmethod)
    optional(settings, keys.KEY_STG_FITPOS, fitpos)
    optional(settings, keys.KEY_STG_BFFLAT, flatten)
    # For multi frames
    if framesSlices == [None, None]:
        framesSlices = None
    optional(settings, keys.KEY_STG_FRAMESSLICES, framesSlices)
    optional(settings, keys.KEY_STG_ZGRID, Zgrid)
    optional(settings, keys.KEY_STG_NORMALISE, normalise)
    if imslices == [None, None]:
        imslices = None
    optional(settings, keys.KEY_STG_SLICE, imslices)
    settings[keys.KEY_STG_NSPECIES] = nspecies

    with open(settingsfn, 'w') as f:
        json.dump(settings, f, indent=4)


def default(dic, key, value):
    """Set valur in dictionnary if None

    Parameters
    ----------
    dic: dictionnary
        the dictionnary
    key: string
        the key
    value: object
        Value that might be None
    """
    if key not in dic or dic[key] is None:
        dic[key] = value


def metadata_fn(filename):
    """Create a metadata file name from an image file name

    Parameters
    ----------
    filename: path
        the path to an image

    Returns
    -------
    metadatafn: path
        path to the metadata file
    """
    return os.path.splitext(filename)[0] + '_metadata.json'


def makeabs(prefix, filename):
    """Combine a prefix and a filename to create an absolute path.

    Parameters
    ----------
    prefix: path
        Prefix to combine with the file name. Can be relative.
    filename: path
        Path to the file

    Returns
    -------
    filename: path
        absolute path
    """
    if (filename is not None and
            not os.path.isabs(filename)):
        filename = os.path.join(prefix, filename)
        filename = os.path.abspath(filename)
    return filename


def listmakeabs(prefix, filename):
    """Combine a prefix and a list of filenames to create absolute paths.

    Parameters
    ----------
    prefix: path
        Prefix to combine with the file name. Can be relative.
    filename: path or list of paths
        Path to the file

    Returns
    -------
    filenames: path or path list
        (list of) absolute path
    """
    if isinstance(filename, (list, tuple)):
        return [makeabs(prefix, fn) for fn in filename]
    else:
        return makeabs(prefix, filename)


def loadSettings(settingsfn):
    """Load fit settings from a json file

    Parameters
    ----------
    settingsfn: path
        path to the fit settings file

    Returns
    -------
    settings: dict
        Dictionnary containing fit settings

    """
    with open(settingsfn, 'r') as f:
        print()
        settings = json.load(f)

    default(settings, keys.KEY_STG_IGNORE, 0)
    default(settings, keys.KEY_STG_POS0FILTER, 'none')
    default(settings, keys.KEY_STG_FITPOS, None)
    default(settings, keys.KEY_STG_BFFLAT, False)
    default(settings, keys.KEY_STG_FRAMESSLICES, [None, None])
    default(settings, keys.KEY_STG_NSPECIES, 1)
    default(settings, keys.KEY_STG_ZGRID, 11)
    default(settings, keys.KEY_STG_NORMALISE, True)
    default(settings, keys.KEY_STG_SLICE, None)
    return settings


def loadMetadata(metadatafn):
    """Load metadata from a json file

    Parameters
    ----------
    metadatafn: path
        path to the metadata file

    Returns
    -------
    metadata: dict
        Dictionnary containing metadata

    """
    with open(metadatafn, 'r') as f:
        metadata = json.load(f)

    default(metadata, keys.KEY_MD_NCHANNELS, 1)
    default(metadata, keys.KEY_MD_BORDER, [None, None, None, None])

    for key in [keys.KEY_MD_FN, keys.KEY_MD_BGFN, keys.KEY_MD_OPBGFN]:
        default(metadata, key, None)
        if metadata[key] is not None:
            metadata[key] = listmakeabs(os.path.dirname(metadatafn),
                                        metadata[key])

    for key in [keys.KEY_MD_WALLWIDTH, keys.KEY_MD_EXP,
                keys.KEY_MD_BGEXP, keys.KEY_MD_OPBGEXP]:
        default(metadata, key, None)

    return metadata


def getType(metadata, images_shape):
    """Get the type of data this is

    Parameters
    ----------
    metadatafn: path
        path to the metadata file

    Returns
    -------
    dtype: str
        a string describing the type of data
    """
    if keys.KEY_MD_TYPE in metadata:
        return metadata[keys.KEY_MD_TYPE]
    
    nchannels = metadata[keys.KEY_MD_NCHANNELS]
    if nchannels == 1:
        if len(images_shape) == 2:
            raise RuntimeError('Only 1 channel in 1 image. Please set "'
                               + keys.KEY_MD_NCHANNELS + '".')
        return '12pos'
    elif nchannels != 4:
        return 'unknown'
    if len(images_shape) == 2:
        return '4pos'
    elif len(images_shape) == 3:
        return '4pos_stack'
    return 'unknown'
