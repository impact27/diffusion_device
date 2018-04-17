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


def scale_factor(profiles, pslice):
    """Normalise a list of profiles
    """
    # if profile is mainly negative, error
    if np.any(np.sum((profiles * (profiles > 0))[..., pslice], -1) <
              5 * -np.sum((profiles * (profiles < 0))[..., pslice], -1)):
        warnings.warn("Negative profile", RuntimeWarning)

    norm_factor = np.sum(profiles[..., pslice], -1)[..., np.newaxis]
#    if np.any(norm_factor <= 0):
#        raise RuntimeError("Can't normalise profiles")

    return norm_factor


def ignore_slice(ignore, pixel_size):
    """Get a slice to ignore sides
    """
    ignore = int(ignore / pixel_size)
    if ignore == 0:
        pslice = slice(None)
    else:
        pslice = slice(ignore, -ignore)
    return pslice


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


def get_fit_data(settings, profiles, readingpos, pslice, infos, fits):
    """get_fit_data"""
    fit_index = settings["KEY_STG_FITPOS"]
    if fit_index is not None:
        fit_index = np.sort(fit_index)
    else:
        fit_index = np.arange(len(profiles))

    fit_profiles = profiles[fit_index]
    fit_readingpos = readingpos[fit_index]

    initmode = settings["KEY_STG_POS0FILTER"]
    if initmode == 'synthetic':
        fit_init = synthetic_init(fit_profiles[0], pslice)
    else:
        fit_init_index = fit_index[0]
        fit_index = fit_index[1:]
        # treat init profile
        fit_init = init_process(fit_profiles[0], initmode, pslice)

        fits[fit_init_index] = fit_init
        # First reading pos is initial profile
        fit_readingpos = fit_readingpos[1:] - fit_readingpos[0]
        fit_profiles = fit_profiles[1:]

    # Check if init is large enough
    threshold = 3 * np.median(infos["Profiles noise std"])
    if np.mean(fit_init[pslice]) < threshold:
        raise RuntimeError("signal to noise too low")

    return fit_init, fit_profiles, fit_readingpos, fit_index


def normalise_basis(basis, profiles, pslice):
    """Normalise basis in the same way as profiles"""
    profiles_scales = scale_factor(profiles, pslice)
    basis_scales = scale_factor(basis, pslice)
    if len(np.shape(profiles_scales)) < len(np.shape(basis_scales)):
        profiles_scales = profiles_scales[np.newaxis, :]
    basis *= profiles_scales / basis_scales
    return basis


def get_fit_infos(profiles, fit_profiles, fits, pslice, Mfreepar,
                  infos, settings):

    infos["Signal over noise"] = np.mean(
        (profiles / infos["Profiles noise std"])[..., pslice])
    slicesize = np.sum(np.ones_like(fit_profiles)[..., pslice])
    nu = slicesize - Mfreepar
    reduced_least_square = ((np.nansum(np.square(
        (profiles[..., pslice] - fits[..., pslice])
        / infos["Profiles noise std"])))
        / nu)
    infos["Reduced least square"] = np.sqrt(reduced_least_square)

    ratio = infos["Reduced least square"] / infos["Signal over noise"]
    if settings["KEY_STG_LSE_THRESHOLD"] and ratio > 1:
        raise RuntimeError("Least square error too large")


def size_profiles(profiles, metadata, settings, infos, zpos=None):
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
    # load variables
    nspecies = settings["KEY_STG_NSPECIES"]

    test_radii = get_test_radii(settings)
    pslice = ignore_slice(settings["KEY_STG_IGNORE"], infos["Pixel size"])
    readingpos = get_reading_position(metadata, settings, len(profiles))
    profiles_arg_dir = get_profiles_arg_dir(metadata, settings)

    profiles = np.asarray(profiles)
    fits = np.zeros_like(profiles) * np.nan

    fit_init, fit_profiles, fit_readingpos, fit_index = get_fit_data(
        settings, profiles, readingpos, pslice, infos, fits)

    # Get basis function
    Basis = getprofiles(fit_init, Radii=test_radii,
                        readingpos=fit_readingpos,
                        zpos=zpos, infos=infos,
                        **profiles_arg_dir)

    # Get best fit
    fit = fit_all(fit_profiles, Basis, test_radii,
                  profslice=pslice, nspecies=nspecies,
                  prof_noise=infos["Profiles noise std"])

    infos["Radius error std"] = fit.dx

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
                raise RuntimeError("The relative error is larger than 1%")

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

    fits = normalise_basis(fits, profiles, pslice)

    get_fit_infos(profiles, fit_profiles, fits, pslice, Mfreepar,
                  infos, settings)

#    if settings["KEY_STG_GETOFFSET"]:
#        get_offset(readingpos, metadata, r, fit_init, profiles, pslice, nspecies)

    return r, fits


# def get_offset(readingpos, metadata, r, fit_init, profiles, pslice, nspecies):
#
#        from .basis_generate import get_unitless_parameters, get_unitless_profiles, get_D
#        readingpos -= readingpos[0]
#        D = get_D([r], metadata["KEY_MD_ETA"], metadata["KEY_MD_T"])
#        X, beta, mu_prime_E = get_unitless_parameters(
#                metadata["KEY_MD_Q"], D, readingpos,
#                metadata["KEY_MD_WY"], metadata["KEY_MD_WZ"])
#        X = np.squeeze(X)
#        Xtest = np.linspace(0, 2*np.max(X), 100)
#
#        ul_basis, Xtest, dx = get_unitless_profiles(fit_init, Xtest, beta, Zgrid=21)
#
#        best_X = np.zeros(len(readingpos))
#        sigma = np.zeros(len(readingpos))
#        Q = metadata["KEY_MD_Q"]/3600e9
#        for i, (prof, rp) in enumerate(zip(profiles, readingpos)):
#            try:
#                info_i = {"Profiles noise std":1}
#                best_X[i] = fit_all(prof, ul_basis, Xtest, pslice, nspecies=nspecies, infos = info_i)
#                sigma[i] = info_i["Radius error std"]
#            except:
#                sigma[i] = 1
#                print('nope')
#
#        x = np.arange(len(readingpos))
#        def fun(x, offset, D):
#            assert np.all(x == np.arange(len(readingpos)))
#            Rp = readingpos.copy()
#            Rp[::2] += offset
#            Rp[1::2] -= offset
#            Rp -= Rp[0]
#            return Rp*D/Q*beta
#
#        from scipy.optimize import curve_fit
#
#        res = curve_fit(fun, x, best_X, p0=[0, D], sigma=sigma)
#
#        print('offset [mm] = ', res[0][0]*1e3, res[0][1]/D)
#
#        if not hasattr(get_offset, 'idx'):
#            get_offset.idx = 0
#        else:
#            get_offset.idx += 1
#        from matplotlib.pyplot import figure, plot, show
#        figure(0)
#        plot([get_offset.idx], [res[0][0]*1e3], 'bx')
#        figure(1)
#        plot([get_offset.idx], [res[0][1]/D], 'bx')


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
    flatbasis = np.reshape(Basis[..., profslice], (Nb, -1))
    flatprofs = np.ravel(profiles[..., profslice])

    psquare = np.sum(flatprofs * flatprofs)
    b = np.sum(flatbasis * flatprofs[np.newaxis], -1)
    M = np.sum(flatbasis[:, np.newaxis] * flatbasis[np.newaxis, :], -1)
    return M, b, psquare


def fit_all(profiles, Basis, phi, *, profslice=slice(None), nspecies=1,
            prof_noise=1):
    """Find the best monodisperse radius

     Parameters
    ----------
    profiles: 2d array
        List of profiles to fit
    Basis: 3d array
        List of basis to fit. The first dimention must correspond to Rs
    phi: 1d float
        The test parameters
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
    if np.shape(np.unique(phi)) != np.shape(phi):
        raise RuntimeError('duplicated phi')

    # Normalize the basis to fit profiles
    Basis = normalise_basis(Basis, profiles, profslice)

    M, b, psquare = get_matrices(profiles, Basis, profslice)

    if nspecies == 1 and phi is not None:
        return fit_monodisperse(M, b, psquare, phi, prof_noise)

    elif nspecies == 2:
        return fit_2(M, b, psquare, phi, prof_noise)

    elif nspecies > 0:
        return fit_N(M, b, psquare, nspecies, phi, prof_noise)

    elif nspecies == 0:
        return fit_polydisperse(M, b, psquare, phi, prof_noise)

    else:
        raise RuntimeError('Number of species negative!')


class FitResult():
    def __init__(self, x, dx, *, x_distribution, basis_spectrum, residual,
                 success=True, status=0):
        self.x = x
        self.dx = dx
        self.x_distribution = x_distribution
        self.basis_spectrum = basis_spectrum
        self.residual = residual
        self.success = success
        self.status = status


def fit_monodisperse(M, b, psquare, phi, prof_noise=1):
    """Find the best monodisperse radius

    Parameters
    ----------
    M: 2d array
        The basis matrix. Mij = sum(basisi*basisj)
    b: 1d array
        bi = sum(profile*basisi)
    psquare: float
        psquare = sum(profiles*profile)
    phi:
        MUST BE SORTED
    Rs: 1d float
        The test radii [m]

    Returns
    -------
    radii: float
        The best radius fit
    """
    # get best residual
    res = psquare + np.diag(M) - 2 * b
    spectrum = np.zeros(len(b))

    arg_left, arg_right = np.argsort(res)[:2]

    if np.abs(arg_left - arg_right) != 1:
        raise RuntimeError("Non - consecutive minimum")

    Bl_minus_Br_square = (M[arg_left, arg_left] + M[arg_right, arg_right]
                          - M[arg_left, arg_right] - M[arg_right, arg_left])

    # If no diff
    if Bl_minus_Br_square == 0:
        raise RuntimeError("No Gradient in Basis")

    # np.sum((b1-b2)*(p0-b2))/np.sum((b1-b2)**2)
    coeff_basis = (
        (b[arg_left] - b[arg_right]
         + M[arg_right, arg_right] - M[arg_left, arg_right])
        / Bl_minus_Br_square)

    spectrum[arg_left] = 1 - coeff_basis
    spectrum[arg_right] = coeff_basis

    best_phi = np.exp(coeff_basis * np.log(phi[arg_left])
                      + (1 - coeff_basis) * np.log(phi[arg_right]))

#    best_phi = coeff_basis * phi[arg_left] + (1-coeff_basis) * phi[arg_right]

    if np.argmin(res) == 0:
        raise RuntimeError(f'Phi too large {best_phi}')
    if np.argmin(res) == len(b) - 1:
        raise RuntimeError(f'Phi too small {best_phi}')

    # sqrt(dR**2/np.sum((b1-b2)**2)*sigma
    phi_error = (prof_noise
                 * np.sqrt((phi[arg_left] - phi[arg_right])**2
                           / Bl_minus_Br_square))
    # Get residual
    # B = (1-c) B_0 + c B_1
    BB = ((1 - coeff_basis)**2 * M[arg_left, arg_left]
          + 2 * coeff_basis * (1 - coeff_basis) * M[arg_left, arg_right]
          + coeff_basis**2 * M[arg_right, arg_right])

    By = (1 - coeff_basis) * b[arg_left] + coeff_basis * b[arg_right]

    residual = BB - 2 * By + psquare
    result = FitResult(best_phi, phi_error, x_distribution=1,
                       basis_spectrum=spectrum, residual=residual)

    return result


def fun_interp_2(C, M, b, psquare, idx):
    nspecies = 2
    C_phi = [1 - C[0], C[0]]
    C_interp = C[1:]

    BB = np.zeros((nspecies, nspecies))
    By = np.zeros(nspecies)

    for i in range(nspecies):
        for j in range(nspecies):
            BB[i, j] = (
                (1 - C_interp[i]) * (1 - C_interp[j]) * M[idx[i, 0], idx[j, 0]]
                + C_interp[i] * (1 - C_interp[j]) * M[idx[i, 1], idx[j, 0]]
                + C_interp[j] * (1 - C_interp[i]) * M[idx[j, 1], idx[i, 0]]
                + C_interp[i] * C_interp[j] * M[idx[j, 1], idx[i, 1]])
        By[i] = (1 - C_interp[i]) * b[idx[i, 0]] + C_interp[i] * b[idx[i, 1]]

    FitFit = (C_phi[0]**2 * BB[0, 0]
              + 2 * C_phi[1] * C_phi[0] * BB[0, 1]
              + C_phi[1]**2 * BB[1, 1])
    Fity = C_phi[0] * By[0] + C_phi[1] * By[1]

    residual = FitFit - 2 * Fity + psquare
    return residual


def jac_interp_2(C, M, b, psquare, idx):
    nspecies = 2
    C_phi = [1 - C[0], C[0]]
    C_interp = C[1:]

    BB = np.zeros((nspecies, nspecies))
    By = np.zeros(nspecies)

    for i in range(nspecies):
        for j in range(nspecies):
            BB[i, j] = (
                (1 - C_interp[i]) * (1 - C_interp[j]) * M[idx[i, 0], idx[j, 0]]
                + C_interp[i] * (1 - C_interp[j]) * M[idx[i, 1], idx[j, 0]]
                + C_interp[j] * (1 - C_interp[i]) * M[idx[j, 1], idx[i, 0]]
                + C_interp[i] * C_interp[j] * M[idx[j, 1], idx[i, 1]])
        By[i] = (1 - C_interp[i]) * b[idx[i, 0]] + C_interp[i] * b[idx[i, 1]]

    FitB = np.zeros(nspecies)
    for i in range(nspecies):
        FitB[i] = C_phi[0] * BB[0, i] + C_phi[1] * BB[1, i]

    BBk = np.zeros((nspecies, nspecies, 2))
    for i in range(nspecies):
        for j in range(nspecies):
            for k in range(2):
                BBk[i, j, k] = ((1 - C_interp[i]) * M[idx[i, 0], idx[j, k]]
                                + C_interp[i] * M[idx[i, 1], idx[j, k]])
    FitBk = np.zeros((nspecies, 2))
    for i in range(nspecies):
        for k in range(2):
            FitBk[i, k] = C_phi[0] * BBk[0, i, k] + C_phi[1] * BBk[1, i, k]

    dinterp = np.zeros(nspecies)
    for i in range(nspecies):
        dinterp[i] = 2 * C_phi[i] * (FitBk[i, 1] - FitBk[i, 0]
                                     + b[idx[i, 0]] - b[idx[i, 1]])

    d0 = 2 * (FitB[1] - FitB[0] + By[0] - By[1])

    return np.array([d0, *dinterp])


def fit_2(M, b, psquare, phi, prof_noise=1):
    """Find the best monodisperse radius

    Parameters
    ----------
    M: 2d array
        The basis matrix. Mij = sum(basisi*basisj)
    b: 1d array
        bi = sum(profile*basisi)
    psquare: float
        psquare = sum(profiles*profile)
    phi:
        MUST BE SORTED
    Rs: 1d float
        The test radii [m]

    Returns
    -------
    radii: float
        The best radius fit
    """
    Nb = len(b)
    indices = np.asarray([i for i in combinations(range(Nb), 2)])
    fraction = np.zeros(len(indices))
    residual = np.zeros(len(indices))
    for i, idx in enumerate(indices):
        idx1, idx2 = idx
        # minimise sum((frac * B1 + (1-frac) * B2 - Y)^2)
        fact_1 = (M[idx1, idx2] - M[idx1, idx1] + b[idx1] - b[idx2])
        fact_2 = (M[idx2, idx2] - 2 * M[idx1, idx2] + M[idx1, idx1])
        fact_3 = (M[idx1, idx1] - 2 * b[idx1] + psquare)

        if fact_2 == 0:
            fraction[i] = 0
        else:
            fraction[i] = - fact_1 / fact_2
        if fraction[i] > 1:
            fraction[i] = 1
        elif fraction[i] < 0:
            fraction[i] = 0
        residual[i] = fraction[i]**2 * fact_2 + \
            2 * fraction[i] * fact_1 + fact_3

    bestidx = np.argmin(residual)
    idx = indices[bestidx]
    frac = fraction[bestidx]

    phi_error = np.zeros(2)
    spectrum = np.zeros(len(b))

    for rn, i in enumerate(idx):
        j = i + 1
        if j == Nb:
            j = Nb - 2
        Bl_minus_Br_square = (M[i, i] + M[j, j] - M[i, j] - M[j, i])
        if Bl_minus_Br_square == 0:
            raise RuntimeError("No Gradient in Basis")
        error = (prof_noise
                 * np.sqrt((phi[i] - phi[j])**2
                           / Bl_minus_Br_square))
        phi_error[rn] = error

    if not np.all(np.diff(idx) > 4):
        prop_phi = np.asarray([1 - frac, frac])
        warnings.warn(
            f"Can't fit polydisperse: Basis not fine enough {np.diff(idx)}")
        spectrum[idx] = prop_phi

        fit = FitResult(phi[idx], phi_error, x_distribution=prop_phi,
                        basis_spectrum=spectrum, residual=np.min(residual),
                        success=False, status=1)
        return fit

    C0 = [frac, 0, 0]
    idx_min = np.zeros((2, 2), int)
    idx_min[:, 0] = idx
    idx_min[:, 1] = idx_min[:, 0] + 1
    if idx_min[1, 1] >= Nb:
        idx_min[1, :] -= 1

    min_res = minimize(fun_interp_2, C0, args=(M, b, psquare, idx_min),
                       jac=jac_interp_2,
                       method='BFGS', options={'gtol': 1e-13, 'norm': 2})
# method = 'Nelder-Mead', options = {'xatol': 1e-16,  'fatol': 1e-16})
    frac = min_res.x[0]
    c = np.asarray(min_res.x[1:])

    phi_res = (1 - c) * phi[idx_min[:, 0]] + c * phi[idx_min[:, 1]]
    prop_phi = np.asarray([1 - frac, frac])

    spectrum[idx_min] = np.array([1 - c, c]) * prop_phi[np.newaxis]
    fit = FitResult(phi_res, phi_error, x_distribution=prop_phi,
                    basis_spectrum=spectrum, residual=min_res.fun,
                    success=True, status=0)
    return fit


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


def fit_N(M, b, psquare, nspecies, phi, prof_noise=1):
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
#            print('New best: ', best)
        res[i] = min_res.fun
        C[i] = min_res.x

    bestidx = np.argmin(res)
    idx = indices[bestidx]
    spectrum = np.zeros(NRs)
    spectrum[idx] = C[bestidx]

    radius_error = np.zeros(nspecies)

    for rn, i in enumerate(idx):
        j = i + 1
        if j == NRs:
            j = NRs - 2
        error = (prof_noise
                 * np.sqrt((phi[i] - phi[j])**2
                           / (M[i, i] + M[j, j] - M[i, j] - M[j, i])))
        radius_error[rn] = error

    fit = FitResult(phi[idx], radius_error, x_distribution=C[bestidx],
                    basis_spectrum=spectrum, residual=np.min(res))

    return fit


def fit_polydisperse(M, b, psquare, phi, prof_noise=1):
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

    radius_error = np.zeros(Nb)

    for i in range(1, Nb):
        j = i - 1
        error = (prof_noise
                 * np.sqrt((phi[i] - phi[j])**2
                           / (M[i, i] + M[j, j] - M[i, j] - M[j, i])))
        radius_error[i] = error
    radius_error[0] = radius_error[1]

    fit = FitResult(phi, radius_error, x_distribution=spectrum,
                    basis_spectrum=spectrum, residual=res.fun)
    return fit


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
    C = np.correlate(top, bottom, mode='full')

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


def process_profiles(profiles, settings, outpath, pixel_size):
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
