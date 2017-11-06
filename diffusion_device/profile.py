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
import warnings
from scipy.optimize import basinhopping, minimize
from itertools import combinations
from scipy.signal import savgol_filter

from .basis_generate import getprofiles
from . import display_data


def normalise_profiles(profiles, pslice):
    """Normalise a list of profiles
    """
    # if profile is mainly negative, error
    if np.any(np.sum((profiles * (profiles > 0))[..., pslice], 1) <
              5 * -np.sum((profiles * (profiles < 0))[..., pslice], 1)):
        warnings.warn("Negative profile", RuntimeWarning)
    profiles /= np.sum(profiles[..., pslice], -1)[..., np.newaxis]
    return profiles


def ignore_slice(ignore, pixel_size):
    """Get a slice to ignore sides
    """
    ignore = int(ignore / pixel_size)
    if ignore == 0:
        pslice = slice(None)
    else:
        pslice = slice(ignore, -ignore)
    return pslice


def size_profiles(profiles, pixel_size, metadata, settings,
                  fits=None, zpos=None):
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
    fits: 2d array
        if not None, same shape and type as profiles
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

    # load variables
    readingpos = np.asarray(metadata["KEY_MD_RPOS"])
    flow_rate = metadata["KEY_MD_Q"]
    channel_height = metadata["KEY_MD_WZ"]
    channel_width = metadata["KEY_MD_WY"]
    temperature = metadata["KEY_MD_T"]
    viscosity = metadata["KEY_MD_ETA"]

    fit_position_number = settings["KEY_STG_FITPOS"]
    ignore = settings["KEY_STG_IGNORE"]
    norm_profiles = settings["KEY_STG_NORMALISE"]
    initmode = settings["KEY_STG_POS0FILTER"]
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
    Zgrid = settings["KEY_STG_ZGRID"]
    nspecies = settings["KEY_STG_NSPECIES"]
    imslice = settings["KEY_STG_SLICE"]
    

    pslice = ignore_slice(ignore, pixel_size)

    profiles = np.asarray(profiles)
    if norm_profiles:
        profiles = normalise_profiles(profiles, pslice)

    readingpos = np.asarray(readingpos)
    if len(readingpos) != len(profiles):
        raise RuntimeError(
            "Number of profiles and reading positions mismatching.")
    if imslice is not None:
        shift = np.resize([1, -1], len(readingpos)) * imslice[0]
        readingpos = readingpos + shift

    if fit_position_number is not None:
        fit_position_number = np.sort(fit_position_number)
        profilesfit = profiles[fit_position_number]
        readingposfit = readingpos[fit_position_number]
    else:
        fit_position_number = np.arange(len(profiles))
        profilesfit = profiles
        readingposfit = readingpos

    if initmode == 'synthetic':
        init = synthetic_init(profiles[0], pslice)
    else:
        fit_position_number = fit_position_number[1:]
        # treat init profile
        init = init_process(profiles[0], initmode, pslice)
        # First reading pos is initial profile
        readingposfit = readingposfit[1:] - readingposfit[0]
        profilesfit = profilesfit[1:]

    # Get basis function
    Basis = getprofiles(init, flow_rate, test_radii,
                        Wy=channel_width, Wz=channel_height,
                        Zgrid=Zgrid, readingpos=readingposfit,
                        zpos=zpos, temperature=temperature, eta=viscosity)

    if norm_profiles:
        # Normalize basis in the same way as profiles
        Basis = normalise_profiles(Basis, pslice)

    if nspecies == 1:
        # Get best fit
        r = fit_radius(profilesfit, Basis, test_radii, pslice, nspecies=1)

        # fill data if needed
        if fits is not None and not np.isnan(r):

            fits[fit_position_number] = getprofiles(
                init, Q=flow_rate, Radii=[r], Wy=channel_width,
                Wz=channel_height, Zgrid=Zgrid,
                readingpos=readingposfit,
                zpos=zpos, temperature=temperature)[0]
            if initmode != 'synthetic':
                fits[0] = init

            if norm_profiles:
                # Normalize basis in the same way as profiles
                fits = normalise_profiles(fits, pslice)

        return r
    else:
        spectrum = fit_radius(profilesfit, Basis, test_radii, pslice,
                              nspecies=nspecies)

        # fill data if needed
        if fits is not None:
            fits[fit_position_number] = np.sum(
                spectrum[:, np.newaxis, np.newaxis] * Basis, axis=0)
            if initmode != 'synthetic':
                fits[0] = init

        return test_radii, spectrum


def synthetic_init(prof0, pslice):
    """Generates a synthetic profile that is 1/11 of the channel"""
    N = len(prof0)
    init = np.zeros_like(prof0)
    x = np.arange(N) - center(prof0)
    init[np.abs(x) < 1 / 22 * N] = 1
    init *= np.sum(prof0[pslice], -1) / np.sum(init[pslice], -1)
    return init


def get_matrices(profiles, Basis, profslice):
    """Return matrix representation of sums

     Parameters
    ----------
    profiles: 2d array
        List of profiles to fit
    Basis: 3d array
        List of basis to fit. The first dimention must correspond to Rs
    ignore: int, default 0
        Ignore on the sides [px]


    Returns
    -------
    M: 2d array
        The basis matrix. Mij = sum(basisi*basisj)
    b: 1d array
        bi = sum(profile*basisi)
    psquare: float
        psquare = sum(profiles*profile)
    """
    Nb = len(Basis)
    flatbasis = np.reshape(Basis[:, :, profslice], (Nb, -1))
    flatprofs = np.ravel(profiles[:, profslice])
    M = np.zeros((Nb, Nb))
    b = np.zeros((Nb))

    psquare = np.sum(flatprofs * flatprofs)
    for i in range(Nb):
        b[i] = np.sum(flatbasis[i] * flatprofs)
        for j in range(Nb):
            M[i, j] = np.sum(flatbasis[i] * flatbasis[j])

    return M, b, psquare


def fit_radius(profiles, Basis, Rs=None, profslice=slice(None), nspecies=1):
    """Find the best monodisperse radius

     Parameters
    ----------
    profiles: 2d array
        List of profiles to fit
    Basis: 3d array
        List of basis to fit. The first dimention must correspond to Rs
    Rs: 1d float
        The test radii [m]
    ignore: int, default 0
        Ignore on the sides [px]
    nspecies: int
        Number of species to fit. 0=all.


    Returns
    -------
    spectrum:
        The factors of Rs to get the best fit
    IF nspecies == 1:
        Radii: [m]
            The best radius fit
    """

    M, b, psquare = get_matrices(profiles, Basis, profslice)

    if nspecies == 1 and Rs is not None:
        return fit_monodisperse_radius(M, b, psquare, Rs)

    elif nspecies > 0:
        return fit_N_radius(M, b, psquare, nspecies)

    elif nspecies == 0:
        return fit_polydisperse_radius(M, b, psquare)

    else:
        raise RuntimeError('Number of species negative!')


def fit_monodisperse_radius(M, b, psquare, Rs):
    """Find the best monodisperse radius

    Parameters
    ----------
    M: 2d array
        The basis matrix. Mij = sum(basisi*basisj)
    b: 1d array
        bi = sum(profile*basisi)
    psquare: float
        psquare = sum(profiles*profile)
    Rs: 1d float
        The test radii [m]

    Returns
    -------
    radii: float
        The best radius fit
    """
    # get best residu
    res = psquare + np.diag(M) - 2 * b

    i, j = np.argsort(res)[:2]
    # np.sum((b1-b2)*(p0-b2))/np.sum((b1-b2)**2)
    c = (b[i] - b[j] - M[i, j] + M[j, j]) / \
        (M[i, i] + M[j, j] - M[i, j] - M[j, i])

    # Get resulting r
    r = c * (Rs[i] - Rs[j]) + Rs[j]

    '''
    from matplotlib.pyplot import figure, plot, title
    figure()
    plot(Rs, res)
    #'''

    if r < np.min(Rs):
        raise RuntimeError('The test radius are too big!')
    if r > np.max(Rs):
        raise RuntimeError('The test radius are too small!')

    return r


def fun(C, M, b, psquare):
    """Residus of the fitting

    Parameters
    ----------
    C: 1d array
        Coefficient for the basis function
    M: 2d array
        The basis matrix. Mij = sum(basisi*basisj)
    b: 1d array
        bi = sum(profile*basisi)
    psquare: float
        psquare = sum(profiles*profile)

    Returns
    -------
    Residus: float
        sum((d-p)^2)
    """
    return psquare + C@M@C - 2 * C@b


def jac(C, M, b, psquare):
    """Jacobian of the Residus function

    Parameters
    ----------
    C: 1d array
        Coefficient for the basis function
    M: 2d array
        The basis matrix. Mij = sum(basisi*basisj)
    b: 1d array
        bi = sum(profile*basisi)
    psquare: float
        psquare = sum(profiles*profile)

    Returns
    -------
    jacobian: 1d array
        The jacobian of fun
    """
    return 2 * C@M - 2 * b


def hess(C, M, b, psquare):
    """Hessian matrix of the Residus function

    Parameters
    ----------
    C: 1d array
        Coefficient for the basis function
    M: 2d array
        The basis matrix. Mij = sum(basisi*basisj)
    b: 1d array
        bi = sum(profile*basisi)
    psquare: float
        psquare = sum(profiles*profile)

    Returns
    -------
    hess: 2d array
        The hessian matrix
    """
    return 2 * M


def get_constraints(Nb):
    """Get constraints such as C>0

    Parameters
    ----------
    Nb: int
        number of coefficients

    Returns
    -------
    constr_dict: dict
        dictionnary containing constraints
    """
    constr = []

    # Need C[i]>0
    for i in range(Nb):
        def cfun(C, i=i):
            return C[i]

        def cjac(C, i=i):
            ret = np.zeros_like(C)
            ret[i] = 1
            return ret

        constr.append({

            "type": "ineq",
            "fun": cfun,
            "jac": cjac

        })
    return constr


def fit_N_radius(M, b, psquare, nspecies):
    """Find the best N-disperse radius

    Parameters
    ----------
    M: 2d array
        The basis matrix. Mij = sum(basisi*basisj)
    b: 1d array
        bi = sum(profile*basisi)
    psquare: float
        psquare = sum(profiles*profile)
    nspecies: int
        Number of species to fit.

    Returns
    -------
    spectrum: 1d array
        The best radius fit spectrum
    """
    NRs = len(b)
    indices = np.asarray([i for i in combinations(range(NRs), nspecies)])
    res = np.empty(len(indices))
    C = np.empty((len(indices), nspecies))
    C0 = np.ones(nspecies) / nspecies
    best = psquare
    for i, idx in enumerate(indices):
        bi = b[idx]
        Mi = M[idx][:, idx]
        min_res = minimize(fun, C0, args=(Mi, bi, psquare),
                           jac=jac, hess=hess,
                           constraints=get_constraints(nspecies))
        if min_res.fun < best:
            best = min_res.fun
            print('New best: ', best)
        res[i] = min_res.fun
        C[i] = min_res.x

    bestidx = np.argmin(res)
    idx = indices[bestidx]
    spectrum = np.zeros(NRs)
    spectrum[idx] = C[bestidx]

    return spectrum


def fit_polydisperse_radius(M, b, psquare):
    """Find the best N-disperse radius

    Parameters
    ----------
    M: 2d array
        The basis matrix. Mij = sum(basisi*basisj)
    b: 1d array
        bi = sum(profile*basisi)
    psquare: float
        psquare = sum(profiles*profile)

    Returns
    -------
    spectrum: 1d array
        The best fit spectrum
    """

    Nb = len(b)
    C0 = np.zeros(Nb)

    def fun2(C, M, b, psquare):
        return fun(np.abs(C), M, b, psquare)

    def jac2(C, M, b, psquare):
        return jac(np.abs(C), M, b, psquare) * np.sign(C)

    res = basinhopping(fun2, C0, 100, disp=True,
                       minimizer_kwargs={'args': (M, b, psquare),
                                         'jac': jac2,
                                         })
    spectrum = np.abs(res.x)
    return spectrum


def center(prof):
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
    prof = np.array(prof) - np.nanmean(prof)
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
    C = np.correlate(top, bottom, mode='full')

    pos = np.arange(len(C)) - (len(C) - 1) / 2
    disty = ((lims[1] - lims[0]) / 2)
    Angles = np.arctan(pos / disty)

    valid = np.abs(Angles) < maxAngle
    x = pos[valid]
    c = C[valid]

    x = x[c.argmax() - 5:c.argmax() + 6]
    y = np.log(gfilter(c, 2)[c.argmax() - 5:c.argmax() + 6])

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


def init_process(profile, mode, ignore_slice):
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
    ignore_slice: slice
        The number of pixels to ignore on the edges

    Returns
    -------
    profile: 1d array
        the processed profile

    """
    init = np.zeros_like(profile)
    init[ignore_slice] = profile[ignore_slice]

    if mode == 'none':
        return init
    elif mode == 'gfilter':
        return gfilter(init, 2)
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

def process_profiles(profiles, pixel_size, settings, outpath):
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
    profiles_filter = settings["KEY_STG_SGFILTER"]
    if profiles_filter is not None:
        filts = savgol_filter(
                profiles, profiles_filter[0], profiles_filter[1], axis=-1)
        display_data.save_plot_filt(profiles, filts, pixel_size, 
                                    profiles_filter, outpath)
        profiles = filts
        
    return profiles