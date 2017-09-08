# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:23:24 2017

@author: quentinpeter
"""
import json
import numpy as np
import os
from tifffile import imread
from . import multi_channels_image as dd4
from . import channel_image as ddx
from registrator.image import is_overexposed


KEY_MD_FN = 'Image file name'
KEY_MD_BGFN = 'Background file name'
KEY_MD_WZ = 'Wz[m]'
KEY_MD_WY = 'Wy[m]'
KEY_MD_WALLWIDTH = 'Wall Width [m]'
KEY_MD_NCHANNELS = 'Number of channels'
KEY_MD_Q = 'Q[ulph]'
KEY_MD_RPOS = 'Read Positions [m]'
KEY_MD_PIXSIZE = 'Pixel Size [m]'
KEY_MD_BORDER = 'Image border[px] (t, d, l, r)'
KEY_MD_EXPOSURE = "Exposition time [s]"
KEY_MD_DATE = "Date [YYYYMMDD]"
KEY_MD_ANALYTE = "Analyte informations"
KEY_MD_BUFFER = "Buffer informations"

KEY_STG_R = 'Radii[m] (min, max, step)'
KEY_STG_NSPECIES = 'Number of species to fit'
KEY_STG_IGNORE = 'Ignore Edge[m]'
KEY_STG_POS0FILTER = 'First Position Filter'
KEY_STG_FITPOS = 'Pos to fit'
KEY_STG_BFFLAT = 'Flatten bright field'
KEY_STG_FRAMESSLICES = 'Frames slice'
KEY_STG_ZGRID = "Number of z slices"
KEY_STG_NORMALISE = "Normalise the profiles?"
KEY_STG_SLICE = "Slice [m] (center(±), width(+))"


def myimread(fn):
    ims = imread(fn)
    if len(ims.shape) == 3:
        ims = np.squeeze(ims[np.logical_not(np.all(ims == 0, (1, 2)))])
    return ims


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


def createMetadata(metafn, fn, Wz, Wy, Q, readingpos, pixelsize,
                   exposure, date, analyte, buffer,
                   bgfn=None, wallwidth=None, nchannels=None, border=None):
    """Creates the metadata for a file name

    Parameters
    ----------
    metafn: path
        Where to save the metadata
    fn: path
        The path to the image relative to the metadata file
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
    bgfn: path, default None
        If there is a background, the path relative to the metadata file
    wallwidth: float, default None
        If this is a multichannel image, the width of the wall in [m]
    nchannels: int, default None
        If this is a multichannel image, The number of channels
    border: 4 ints, default None
        The borber to apply on the image (t, d, l, r)
    """
    Metadata = {}
    Metadata[KEY_MD_FN] = fn
    optional(Metadata, KEY_MD_BGFN, bgfn)
    Metadata[KEY_MD_WZ] = Wz
    Metadata[KEY_MD_WY] = Wy
    optional(Metadata, KEY_MD_WALLWIDTH, wallwidth)
    optional(Metadata, KEY_MD_NCHANNELS, nchannels)
    Metadata[KEY_MD_Q] = Q
    Metadata[KEY_MD_RPOS] = readingpos
    Metadata[KEY_MD_PIXSIZE] = pixelsize
    if border == [None, None, None, None]:
        border = None
    optional(Metadata, KEY_MD_BORDER, border)
    Metadata[KEY_MD_EXPOSURE] = exposure
    Metadata[KEY_MD_DATE] = date
    Metadata[KEY_MD_ANALYTE] = analyte
    Metadata[KEY_MD_BUFFER] = buffer
    # Optional

    with open(metafn, 'w') as f:
        json.dump(Metadata, f, indent=4)


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
    metafn: path
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
    Settings = {}
    Settings[KEY_STG_R] = (rmin, rmax, rstep)
    # Optional
    optional(Settings, KEY_STG_IGNORE, ignore)
    optional(Settings, KEY_STG_POS0FILTER, firstmethod)
    optional(Settings, KEY_STG_FITPOS, fitpos)
    optional(Settings, KEY_STG_BFFLAT, flatten)
    # For multi frames
    if framesSlices == [None, None]:
        framesSlices = None
    optional(Settings, KEY_STG_FRAMESSLICES, framesSlices)
    optional(Settings, KEY_STG_ZGRID, Zgrid)
    optional(Settings, KEY_STG_NORMALISE, normalise)
    if imslices == [None, None]:
        imslices = None
    optional(Settings, KEY_STG_SLICE, imslices)
    Settings[KEY_STG_NSPECIES] = nspecies

    with open(settingsfn, 'w') as f:
        json.dump(Settings, f, indent=4)


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
    return os.path.splitext(filename)[0] + '_Metadata.json'


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
    Settings: dict
        Dictionnary containing fit settings

    """
    with open(settingsfn, 'r') as f:
        print()
        Settings = json.load(f)

    default(Settings, KEY_STG_IGNORE, 0)
    default(Settings, KEY_STG_POS0FILTER, 'none')
    default(Settings, KEY_STG_FITPOS, None)
    default(Settings, KEY_STG_BFFLAT, False)
    default(Settings, KEY_STG_FRAMESSLICES, [None, None])
    default(Settings, KEY_STG_NSPECIES, 1)
    default(Settings, KEY_STG_ZGRID, 11)
    default(Settings, KEY_STG_NORMALISE, True)
    default(Settings, KEY_STG_SLICE, None)
    return Settings


def loadMetadata(metadatafn):
    """Load metadata from a json file

    Parameters
    ----------
    metadatafn: path
        path to the metadata file

    Returns
    -------
    Metadata: dict
        Dictionnary containing metadata

    """
    with open(metadatafn, 'r') as f:
        Metadata = json.load(f)

    default(Metadata, KEY_MD_BGFN, None)
    default(Metadata, KEY_MD_WALLWIDTH, None)
    default(Metadata, KEY_MD_NCHANNELS, 1)
    default(Metadata, KEY_MD_BORDER, [None, None, None, None])

    if Metadata[KEY_MD_BGFN] is not None:
        Metadata[KEY_MD_BGFN] = listmakeabs(os.path.dirname(metadatafn),
                                            Metadata[KEY_MD_BGFN])
    Metadata[KEY_MD_FN] = listmakeabs(os.path.dirname(metadatafn),
                                      Metadata[KEY_MD_FN])

    return Metadata


def getType(metadatafn):
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
    Metadata = loadMetadata(metadatafn)
    nchannels = Metadata[KEY_MD_NCHANNELS]
    if nchannels == 1:
        return '12pos'
    elif nchannels != 4:
        return 'unknown'
    filename = Metadata[KEY_MD_FN]
    if isinstance(filename, (list, tuple)):
        ims = np.asarray([myimread(fn) for fn in filename])
    else:
        ims = myimread(filename)
    if len(ims.shape) == 2:
        return '4pos'
    elif len(ims.shape) == 3:
        return '4pos_stack'
    return 'unknown'


def full_fit(settingsfn, metadatafn, plotim=False):
    """Perform a fit with the imformations found in the settings file

    Parameters
    ----------
    settingsfn: path
        path to the fit settings file
    metadatafn: path
        path to the metadata file

    Returns
    -------
    radius: float or list of floats or 2x list of floats
        If 1 species:
            The fitted radius
        If several species:
            radii, spectrum: The radii and corresponding coefficients
        If movie:
            A list of the above
    profiles: 1d or 2d list of floats
        The extracted profiles. 2d for movie.
    fits: 1d or 2d list of floats
        The Fits. 2d for movie.
    lse: float or list of floats
        The least square error. List for movie.
    pixel_size: float or list of floats
        The detected pixel size. List for movie.
    """

    Metadata = loadMetadata(metadatafn)
    Settings = loadSettings(settingsfn)

    filename = Metadata[KEY_MD_FN]
    bgfn = Metadata[KEY_MD_BGFN]
    readingpos = Metadata[KEY_MD_RPOS]
    Wz = Metadata[KEY_MD_WZ]
    Wy = Metadata[KEY_MD_WY]
    ActualFlowRate = Metadata[KEY_MD_Q]
    pixsize = Metadata[KEY_MD_PIXSIZE]
    nchannels = Metadata[KEY_MD_NCHANNELS]
    wall_width = Metadata[KEY_MD_WALLWIDTH]
    imborder = Metadata[KEY_MD_BORDER]

    rmin, rmax, rstep = Settings[KEY_STG_R]
    ignore = Settings[KEY_STG_IGNORE]
    initmode = Settings[KEY_STG_POS0FILTER]
    fit_position_number = Settings[KEY_STG_FITPOS]
    flatten = Settings[KEY_STG_BFFLAT]
    framesSlice = Settings[KEY_STG_FRAMESSLICES]
    nspecies = Settings[KEY_STG_NSPECIES]
    test_radii = np.arange(rmin, rmax, rstep)
    Zgrid = Settings[KEY_STG_ZGRID]
    normalise_profiles = Settings[KEY_STG_NORMALISE]
    imslice = Settings[KEY_STG_SLICE]

    # load images
    if isinstance(filename, (list, tuple)):
        ims = np.asarray([myimread(fn) for fn in filename])
    else:
        ims = myimread(filename)

        if len(np.shape(ims)) == 3:
            # movie
            ims = ims[framesSlice[0]:framesSlice[1]]

    # Remove Border
    ims = ims[..., imborder[0]:imborder[1],
              imborder[2]:imborder[3]]

    # load background
    bg = None
    if bgfn is not None:
        if isinstance(bgfn, (list, tuple)):
            bg = np.asarray([myimread(fn) for fn in bgfn])
        else:
            bg = myimread(bgfn)

        bg = bg[..., imborder[0]:imborder[1],
                imborder[2]:imborder[3]]

    if nchannels == 1:
        if len(ims.shape) == 2:
            raise RuntimeError('Only 1 channel in 1 image. Please set "'
                               + KEY_MD_NCHANNELS + '".')
        data_dict = {}
        radius = ddx.size_images(ims, ActualFlowRate, Wz, Wy, pixsize,
                                 readingpos, Rs=test_radii, bgs=bg,
                                 data_dict=data_dict, ignore=ignore,
                                 initmode=initmode,
                                 nspecies=nspecies,
                                 Zgrid=Zgrid,
                                 normalise_profiles=normalise_profiles)
        return (radius, *read_data_dict(data_dict))

    else:
        if imslice is not None:
            shift = np.resize([1, -1], nchannels) * imslice[0]
            readingpos = readingpos + shift

        def process_im(im, ignore_error=False, plotim=False):
            data_dict = {}
            radius = dd4.size_image(
                im, ActualFlowRate, Wz, Wy, readingpos,
                test_radii, bg=bg, data_dict=data_dict,
                ignore=ignore, initmode=initmode,
                fit_position_number=fit_position_number,
                flatten=flatten, nspecies=nspecies,
                Nprofs=nchannels, wall_width=wall_width,
                ignore_error=ignore_error, plotim=plotim,
                Zgrid=Zgrid,
                normalise_profiles=normalise_profiles,
                imslice=imslice)

            return (radius, *read_data_dict(data_dict))

        if len(ims.shape) == 2:
            return process_im(ims, plotim=plotim)
        else:
            # movie
            radius_list = []
            profiles_list = []
            fits_list = []
            lse_list = []
            pixel_size_list = []
            for im in ims:
                radius, profiles, fits, lse, pixel_size, __ = \
                    process_im(im, ignore_error=True)
                radius_list.append(radius)
                profiles_list.append(profiles)
                fits_list.append(fits)
                lse_list.append(lse)
                pixel_size_list.append(pixel_size)

            overexposed = [is_overexposed(im) for im in ims]
            return (np.asarray(radius_list), profiles_list, fits_list,
                    np.asarray(lse_list), np.asarray(pixel_size_list),
                    overexposed)


def read_data_dict(data_dict):
    """Extract interesting data from a data dict

    Parameters
    ----------
    data_dict: dict
        The data dictionnary

    Returns
    -------
    profiles: 1d list of floats
        The extracted profiles.
    fits: 1d list of floats
        The Fits.
    lse: float
        The least square error.
    pixel_size: float
        The detected pixel size
    """
    profiles, fits, lse, pixel_size, im = np.nan * np.ones(5)

    if 'profiles' in data_dict and 'fits'in data_dict:
        profiles = data_dict['profiles']
        fits = data_dict['fits']

        if len(profiles) - 1 == len(fits):
            fits = [data_dict['initprof'], *data_dict['fits']]

        lse = np.sqrt(np.mean(np.square(profiles - fits)))
        pixel_size = data_dict['pixsize']

    if 'image' in data_dict:
        im = data_dict['image']

    return profiles, fits, lse, pixel_size, im
