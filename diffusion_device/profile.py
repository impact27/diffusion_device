# -*- coding: utf-8 -*-
"""
Fitting and processing of profiles.

Created on Fri Mar 17 10:25:47 2017

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
from scipy.ndimage.filters import gaussian_filter1d as gfilter
from scipy.signal import savgol_filter

from .basis_generate import getprofiles
from . import display_data
from .profiles_fitting import fit_all, normalise_basis_factor


def size_profiles(infos, metadata, settings):
    """Size the profiles

     Parameters
    ----------
    profiles: 2d array
        List of profiles to fit
    pixel_size:float
        The pixel size in [m]
    metadata: dict
        The metadata
    settings: dict
        The settings
    zpos: float, default None
        Z position of the profile. None for mean

    Returns
    -------
    if nspecies==1:
        radii: float
            The best radius fit
    else:
        Rs, spectrum, the radii and corresponding spectrum
    """
    profiles = infos["Profiles"]
    # load variables
    nspecies = settings["KEY_STG_NSPECIES"]
    zpos = metadata["KEY_MD_SCANZ"]
    vary_offset = settings["KEY_STG_VARY_OFFSET"]
    test_radii = get_test_radii(settings)
    profile_slice = ignore_slice(
        settings["KEY_STG_IGNORE"], infos["Pixel size"])
    prof_noise = infos["Profiles noise std"]
    
    readingpos = get_reading_position(metadata, settings, len(profiles))
    profiles_arg_dir = get_profiles_arg_dir(metadata, settings)

    profiles = np.asarray(profiles)
    fits = np.zeros_like(profiles) * np.nan

    fit_init, fit_profiles, fit_readingpos, fit_index, fit_noise= get_fit_data(
        settings, profiles, readingpos, profile_slice, infos, fits, prof_noise)

    # Get basis function
    Basis = getprofiles(fit_init, Radii=test_radii,
                        readingpos=fit_readingpos,
                        zpos=zpos, infos=infos,
                        **profiles_arg_dir)
    fit_Basis = Basis[..., profile_slice]
    
    # Get best fit
    fit = fit_all(fit_profiles, fit_Basis, test_radii, nspecies=nspecies,
                  prof_noise=fit_noise, vary_offset=vary_offset,
                  fit_square=settings["KEY_STG_FIT_SQUARE"])

    infos["Radius error std"] = fit.dx
    infos["Radius range"] = fit.x_range

    if nspecies == 1:
        # Get resulting r
        r = fit.x
        if r < np.min(test_radii):
            raise RuntimeError(
                'The test radius are too big! ({} < {})'.format(
                    r, np.min(test_radii)))
        if r > np.max(test_radii):
            raise RuntimeError(
                'The test radius are too small! ({} > {})'.format(
                    r, np.max(test_radii)))
        # fill data if needed
        if not np.isnan(r):
            fits[fit_index] = getprofiles(
                fit_init, Radii=[r], readingpos=fit_readingpos,
                zpos=zpos, infos=infos, **profiles_arg_dir)[0]

            if np.any(infos['Fit error'] > 1e-2):
                raise RuntimeError("The relative error is too large "
                                   f"({100*infos['Fit error']:.2f}%), "
                                   "please adapt the step factor.")

        # One free parameter
        Mfreepar = 1

    else:

        # fill data if needed
        fits[fit_index] = np.sum(
            fit.basis_spectrum[:, np.newaxis, np.newaxis] * Basis, axis=0)

        r = (fit.x, fit.x_distribution)

        # 2n-1 free parameter
        Mfreepar = 2 * nspecies - 1
        if nspecies == 0:
            Mfreepar = 1  # TODO: fix that

    fact_a, fact_b = normalise_basis_factor(
            fits[fit_index][:, ..., profile_slice],
            fit_profiles, vary_offset)
    fits[fit_index] = fact_a[..., np.newaxis] * \
        fits[fit_index] + np.array(fact_b)[..., np.newaxis]

    get_fit_infos(profiles, fit_profiles, fits, profile_slice, Mfreepar,
                  infos, settings)
    infos["Radius"] = r
    infos["Fitted Profiles"] = fits
    return infos


def ignore_slice(ignore, pixel_size):
    """Get a slice to ignore sides
    """
    ignore = int(ignore / pixel_size)
    if ignore == 0:
        profile_slice = slice(None)
    else:
        profile_slice = slice(ignore, -ignore)
    return profile_slice


def get_test_radii(settings):
    """Get test radii"""
    if settings["KEY_STG_RLOG"]:
        rmin, rmax, Nr = settings["KEY_STG_R"]
        test_radii = np.exp(np.linspace(np.log(rmin), np.log(rmax), Nr))
    else:
        if settings["KEY_STG_R"] is not None:
            test_radii = np.linspace(*settings["KEY_STG_R"])
        else:
            test_radii = np.arange(*settings["KEY_STG_R_STEP"])

    if len(test_radii) == 0:
        raise RuntimeError("The test radius are incorrectly specified.")

    return test_radii


def get_reading_position(metadata, settings, nprofiles):
    """get_reading_position"""
    readingpos = np.asarray(metadata["KEY_MD_RPOS"])
    imslice = settings["KEY_STG_SLICE"]
    if len(readingpos) != nprofiles:
        raise RuntimeError(
            "Number of profiles and reading positions mismatching.")
    if imslice is not None:
        shift = np.resize([1, -1], len(readingpos)) * imslice[0]
        readingpos = readingpos + shift

    return readingpos


def get_profiles_arg_dir(metadata, settings):
    """get_profiles_arg_dir"""
    return {
        'Q': metadata["KEY_MD_Q"],
        'Wz': metadata["KEY_MD_WZ"],
        'Wy': metadata["KEY_MD_WY"],
        'temperature': metadata["KEY_MD_T"],
        'viscosity': metadata["KEY_MD_ETA"],
        'Zgrid': settings["KEY_STG_ZGRID"],
        'step_factor': settings["KEY_STG_DXFACTOR"]}


def get_fit_data(settings, profiles, readingpos, profile_slice, 
                 infos, fits, prof_noise):
    """get_fit_data"""
    
    initmode = settings["KEY_STG_POS0FILTER"]
    
    # Select fit index
    fit_index = settings["KEY_STG_FITPOS"]
    if fit_index is not None:
        fit_index = np.sort(fit_index)
    else:
        fit_index = np.arange(len(profiles))
    fit_profiles = profiles[fit_index]
    fit_readingpos = readingpos[fit_index]
        
    # If we have a different noise for each point
    if len(np.shape(prof_noise)) > 0:
        fit_noise = prof_noise[fit_index]
        if initmode != 'synthetic':
            fit_noise = fit_noise[1:]
        fit_noise = fit_noise[..., profile_slice]
    else:
        fit_noise = prof_noise

    # Process init mode
    if initmode == 'synthetic':
        fit_init = synthetic_init(fit_profiles[0], profile_slice)
    else:
        fit_init_index = fit_index[0]
        fit_index = fit_index[1:]
        # treat init profile
        fit_init = init_process(fit_profiles[0], initmode, profile_slice)

        fits[fit_init_index] = fit_init
        # First reading pos is initial profile
        fit_readingpos = fit_readingpos[1:] - fit_readingpos[0]
        fit_profiles = fit_profiles[1:]

    # Check if init is large enough
    threshold = 3 * np.median(infos["Profiles noise std"])
    if np.max(fit_init[profile_slice]) < threshold:
        raise RuntimeError("signal to noise too low")

    return (fit_init, fit_profiles[..., profile_slice], 
        fit_readingpos, fit_index, fit_noise)


def get_fit_infos(profiles, fit_profiles, fits, profile_slice, Mfreepar,
                  infos, settings):

    infos["Signal over noise"] = np.mean(
        (profiles / infos["Profiles noise std"])[..., profile_slice])
    slicesize = np.sum(np.ones_like(fit_profiles)[..., profile_slice])
    nu = slicesize - Mfreepar
    reduced_least_square = ((np.nansum(np.square(
        ((profiles - fits) / infos["Profiles noise std"])[..., profile_slice])
        )) / nu)
    infos["Reduced least square"] = np.sqrt(reduced_least_square)

    ratio = infos["Reduced least square"] / infos["Signal over noise"]
    if settings["KEY_STG_LSE_THRESHOLD"] and ratio > 1:
        raise RuntimeError("Least square error too large")


def synthetic_init(prof0, profile_slice):
    """Generates a synthetic profile that is 1/11 of the channel"""
    N = len(prof0)
    init = np.zeros_like(prof0)
    x = np.arange(N) - center(prof0)
    init[np.abs(x) < 1 / 22 * N] = 1
    init *= np.sum(prof0[profile_slice], -1) / np.sum(init[profile_slice], -1)
    return init


def center(prof, subtract_mean=False):
    """
    Uses correlation between Y and the mirror image of Y to get the center

    Parameters
    ----------
    prof:  1d array
        Profile

    Returns
    -------
    center: float
        The center position in pixel units

    """

    # We must now detect the position of the center. We use correlation
    # Correlation is equivalent to least squares (A-B)^2=-2AB+ some constants
    prof = np.array(prof)
    if subtract_mean:
        prof -= np.nanmean(prof)
    prof[np.isnan(prof)] = 0
    Yi = prof[::-1]
    corr = np.correlate(prof, Yi, mode='full')
    X = np.arange(len(corr))
    args = np.argsort(corr)
    x = X[args[-7:]]
    y = corr[args[-7:]]
    coeffs = np.polyfit(x, np.log(y), 2)
    center = -coeffs[1] / (2 * coeffs[0])
    center = (center - (len(corr) - 1) / 2) / 2 + (len(prof) - 1) / 2
    return center


def baseline(prof, frac=.05):
    """
    Get the apparent slope from the base of the profile

    Parameters
    ----------
    prof:  1d array
        Profile
    frac: float, defaults .05
        Fraction of the profile to use

    Returns
    -------
    baseline: 1d array
        The baseline

    """
    # we use 5% on left side to get the correct 0:
    # Get left and right zeros
    argvalid = np.argwhere(np.isfinite(prof))
    lims = np.squeeze([argvalid[0], argvalid[-1]])
    left = int(lims[0] + frac * np.diff(lims))
    right = int(lims[1] - frac * np.diff(lims))
    leftZero = np.nanmean(prof[lims[0]:left])
    rightZero = np.nanmean(prof[right:lims[1]])

    # Send profile to 0
    baseline = np.linspace(leftZero, rightZero, len(prof))
    return baseline


def flat_baseline(prof, frac=.05):
    """
    Get the apparent slope from the base of the profile

    Parameters
    ----------
    prof:  1d array
        Profile
    frac: float, defaults .05
        Fraction of the profile to use

    Returns
    -------
    baseline: 1d array
        The flat baseline

    """
    # we use 5% on left side to get the correct 0:
    # Get left and right zeros
    leftZero = np.nanmean(prof[:int(frac * len(prof))])
    rightZero = np.nanmean(prof[-int(frac * len(prof)):])

    # Send profile to 0
    ret = np.zeros(prof.shape) + np.mean([leftZero, rightZero])
    return ret


def image_angle(image, maxAngle=np.pi / 7):
    """
    Analyse an image with y invariance to extract a small angle.

    Parameters
    ----------
    image:  2d array
        image with y invariance
    maxAngle: float, defaults np.pi/7
        Maximal rotation angle

    Returns
    -------
    angle: float
        The rotation angle

    """
    # Difference left 50% with right 50%
    # We want to slice in two where we have data
    argvalid = np.argwhere(np.isfinite(np.nanmean(image, 1)))
    lims = np.squeeze([argvalid[0], argvalid[-1]])
    # should we flatten this?
    top = np.nanmean(image[lims[0]:np.mean(lims, dtype=int)], 0)
    bottom = np.nanmean(image[np.mean(lims, dtype=int):lims[1]], 0)
    # Remouve nans
    top[np.isnan(top)] = 0
    bottom[np.isnan(bottom)] = 0
    # correlate
    C = np.correlate(top - np.mean(top), bottom - np.mean(bottom), mode='full')

    pos = np.arange(len(C)) - (len(C) - 1) / 2
    disty = ((lims[1] - lims[0]) / 2)
    Angles = np.arctan(pos / disty)

    valid = np.abs(Angles) < maxAngle
    x = pos[valid]
    c = C[valid]

    argleft = c.argmax() - 5
    if argleft < 0:
        argleft = 0
    x = x[argleft:c.argmax() + 6]
    y = np.log(gfilter(c, 2)[argleft:c.argmax() + 6])

    if np.any(np.isnan(y)):
        raise RuntimeError('The signal is too noisy!')

    coeff = np.polyfit(x, y, 2)
    x = -coeff[1] / (2 * coeff[0])
    angle = np.arctan(x / disty)

    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arctan(pos/disty), C)
    plt.plot([maxAngle, maxAngle], [np.min(C), np.max(C)])
    plt.plot([-maxAngle, -maxAngle], [np.min(C), np.max(C)])
    plt.plot([angle, angle], [np.min(C), np.max(C)])
    #"""
    """
    import matplotlib.pyplot as plt
    x = np.arange(len(top))
    plt.figure()
    plt.plot(x, top)
    plt.plot(x+(C.argmax()-(len(C)-1)/2), bottom)
    plt.title('image angle')
    #"""
    return angle


def init_process(profile, mode, profile_slice):
    """
    Process the initial profile

    Parameters
    ----------
    profile:  1d array
        Profile to analyse
    mode: string
        'none':
            Nothing
        'gaussian':
            Return a gaussian fit
        'tails':
            Remove the tails
        'gfilter':
            Apply a gaussian filter of 2 px std
    profile_slice: slice
        The number of pixels to ignore on the edges

    Returns
    -------
    profile: 1d array
        the processed profile

    """
    Npix_ignore = profile_slice.start
    if Npix_ignore is not None:
        init = np.zeros_like(profile)
        init[profile_slice] = profile[profile_slice]
        init[:Npix_ignore] = profile[Npix_ignore]
        init[-Npix_ignore:] = profile[-Npix_ignore]
    else:
        init = np.array(profile)

    if mode == 'none':
        return init
    elif mode == 'gfilter':
        return gfilter(init, 2)
    elif mode == 'savgol':
        return savgol_filter(init, 31, 5)
    elif mode == 'gaussian' or mode == 'tails':
        Y = init
        X = np.arange(len(Y))
        valid = Y > .5 * Y.max()
        gauss = np.exp(np.poly1d(np.polyfit(X[valid], np.log(Y[valid]), 2))(X))
        if mode == 'gaussian':
            return gauss
        remove = gauss < .01 * gauss.max()
        init[remove] = 0
        return init


def get_fax(profiles):
    """
    returns a faxed verion of the profiles for easier plotting

    Parameters
    ----------
    profiles:  2d array
        List of profiles

    Returns
    -------
    profiles: 1d array
        The faxed profiles

    """
    return np.ravel(np.concatenate(
        (profiles, np.zeros((np.shape(profiles)[0], 1)) * np.nan), axis=1))

def rebin_profiles(profiles, rebin):
    """rebin profiles"""
    
    if rebin > 1:
        rebin_profiles = np.zeros(
            (np.shape(profiles)[0], np.shape(profiles)[1] // rebin))
        kern = np.ones(rebin)/rebin
        for i in range(len(profiles)):
            rebin_profiles[i] = np.convolve(
                profiles[i], kern, mode='valid')[::rebin]
        profiles = rebin_profiles
    return profiles
        
def process_profiles(infos, metadata, settings, outpath):
    """Process profiles according to settings

    Parameters
    ----------
    profiles: 2 dim floats array
        The profiles
    settings: dic
        The settings

    Returns
    -------
    profiles: 2 dim floats array
        The profiles
    """
    profiles = infos["Profiles"]
    pixel_size = infos["Pixel size"]
    
    
    rebin = settings["KEY_STG_REBIN"]
    profiles = rebin_profiles(profiles, rebin)
    pixel_size *= rebin

    profiles_filter = settings["KEY_STG_SGFILTER"]
    if profiles_filter is not None:
        filts = savgol_filter(
            profiles, profiles_filter[0], profiles_filter[1], axis=-1)
        display_data.save_plot_filt(profiles, filts, pixel_size,
                                    profiles_filter, outpath)
        profiles = filts

    infos["Pixel size"] = pixel_size
    infos["Profiles noise std"] = rebin_profiles(
            infos["Profiles noise std"], rebin)
    infos["Profiles"] = profiles
    return infos
