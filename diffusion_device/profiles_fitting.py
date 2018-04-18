# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 22:39:28 2018

@author: quentinpeter
"""

import numpy as np
from scipy.optimize import basinhopping, minimize
from itertools import combinations
import warnings

class FitResult():
    """
    Class to hold result of fittinmg in a consistant way
    
    Attributes:
    -----------
    x: float or 1d array
        The result radiuses or phi
    dx: float or 1d array
        The estimated error on x
    x_distribution: float or 1d array
        The relative concentration of x species
    basis_spectrum: 1d array
        The coefficient to use to get the fir on the basis
    residual: float
        The least square error residual
    success: bool
        Was the fit sucessful?
    status: int
        A code for addditional information on success/ failure
        
    """
    def __init__(self, x, dx, *, x_distribution, basis_spectrum, residual,
                 success=True, status=0):
        self.x = x
        self.dx = dx
        self.x_distribution = x_distribution
        self.basis_spectrum = basis_spectrum
        self.residual = residual
        self.success = success
        self.status = status

def fit_all(profiles, Basis, phi, *, profile_slice=slice(None), nspecies=1,
            prof_noise=1):
    """Find the best radius for monodisperse/polydisperse solutions

    Parameters
    ----------
    profiles: (N x L) / (L) 1/2d array of float
        List of profiles to fit
    Basis: (M x N x L) / (M x L) 2/3d array of float
        List of basis to fit
    phi: (M) 1d array of float
        The test parameters
    profile_slice: slice
        The slice to consider when fitting (applied on L)
    nspecies: int
        Number of species to fit. 0=all.
    prof_noise: float or 1d array
        The noise on the profiles


    Returns
    -------
    fit: FitResult object
        the fit results
        
    """
    if np.shape(np.unique(phi)) != np.shape(phi):
        raise RuntimeError('duplicated phi')

    # Normalize the basis to fit profiles
    Basis = normalise_basis(Basis, profiles, profile_slice)

    M, b, psquare = get_matrices(profiles, Basis, profile_slice)

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

def normalise_basis(basis, profiles, profile_slice):
    """Normalise basis so they corrrespond to profiles
    
    Parameters
    ----------
    Basis: (M x N x L) / (M x L) 2/3d array of float
        List of basis to normalise
    profiles: (N x L) / (L) 1/2d array of float
        List of reference profiles
    profile_slice: slice
        The slice to consider (applied on L)

    Returns
    -------
    Basis: (M x N x L) / (M x L) 2/3d array of float
        the normalised basis
    """
    profiles_scales = scale_factor(profiles, profile_slice)
    basis_scales = scale_factor(basis, profile_slice)
    if len(np.shape(profiles_scales)) < len(np.shape(basis_scales)):
        profiles_scales = profiles_scales[np.newaxis, :]
    basis *= profiles_scales / basis_scales
    return basis

def scale_factor(profiles, profile_slice):
    """Get the integral of a profile
    
    Parameters
    ----------
    profiles: (... x L) nd array of float
        List of profiles
    profile_slice: slice
        The slice to consider (applied on L)

    Returns
    -------
    norm_factor: (...) n-1d array of float
        The normalisation factor
    
    """
    # if profile is mainly negative, error
    if np.any(np.sum((profiles * (profiles > 0))[..., profile_slice], -1) <
              5 * -np.sum((profiles * (profiles < 0))[..., profile_slice], -1)):
        warnings.warn("Negative profile", RuntimeWarning)

    norm_factor = np.sum(profiles[..., profile_slice], -1)[..., np.newaxis]
#    if np.any(norm_factor <= 0):
#        raise RuntimeError("Can't normalise profiles")

    return norm_factor
        
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


def res_interp_2(C, M, b, psquare, idx):
    nspecies = 2
    C_phi = [1 - C[0], C[0]]
    C_interp = C[1:]
    
    idx = np.tile(idx, (2, 1)).T
    idx[:, 1] += 2*(C_interp > 0) -1
    C_interp = np.abs(C_interp)

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
    
    idx = np.tile(idx, (2, 1)).T
    sign_C_interp = 2*(C_interp > 0) -1
    idx[:, 1] += sign_C_interp
    C_interp = np.abs(C_interp)

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
    dinterp *= sign_C_interp

    d0 = 2 * (FitB[1] - FitB[0] + By[0] - By[1])

    return np.array([d0, *dinterp])

#@profile
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
    # Get 2 combination
    idx_mesh = np.asarray(np.meshgrid(np.arange(Nb), np.arange(Nb)))
    indices = idx_mesh[:, idx_mesh[0] > idx_mesh[1]].T
    indices = indices[:, ::-1]
    
    # Extract corresponding M and b 
    M11 = M[indices[:, 0], indices[:, 0]]
    M12 = M[indices[:, 0], indices[:, 1]]
    M22 = M[indices[:, 1], indices[:, 1]]
    b1 = b[indices[:, 0]]
    b2 = b[indices[:, 1]]
    
    # Intermediate values
    fact_1 = (M12 - M11 + b1 - b2)
    fact_2 = (M22 - 2 * M12 + M11)
    fact_3 = (M11 - 2 * b1 + psquare)
    
    # Get fractions (0 to 1!)
    fraction = np.zeros(len(indices))
    fraction[fact_2 != 0] = - fact_1[fact_2 != 0] / fact_2[fact_2 != 0]
    fraction[fraction > 1] = 1
    fraction[fraction < 0] = 0
    
    # Resibual for each combination
    residual = (fraction**2 * fact_2 
                        + 2 * fraction * fact_1 
                        + fact_3)
    
    # Get best
    bestidx = np.argmin(residual)
    idx = indices[bestidx]
    frac = fraction[bestidx]
    
    if np.min(idx) == 0 or np.max(idx) == Nb-1:
        raise RuntimeError("Fit out of range")

    # Get errors
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

    # If basis not fine enough, we still have a fit, but success is False
    if not np.all(np.diff(idx) > 3):
        prop_phi = np.asarray([1 - frac, frac])
        spectrum[idx] = prop_phi
        fit = FitResult(phi[idx], phi_error, x_distribution=prop_phi,
                        basis_spectrum=spectrum, residual=np.min(residual),
                        success=False, status=1)
        return fit

    # Find interpolation for each position
    C0 = [frac, 0, 0]
    min_res = minimize(res_interp_2, C0, args=(M, b, psquare, idx),
                       jac=jac_interp_2,
                       method='BFGS', options={'gtol': 1e-13, 'norm': 2})
    frac = min_res.x[0]
    C_interp = np.asarray(min_res.x[1:])
    
    # C < 0 mean interpolate to the left
    idx_min = np.tile(idx, (2, 1)).T
    idx_min[:, 1] += 2*(C_interp > 0) -1
    C_interp = np.abs(C_interp)
    
    if np.max(C_interp) > 3:
        raise RuntimeError("Interpolation failure, c out of bounds")
    
    # Result
    phi_res = np.exp((1 - C_interp) * np.log(phi[idx_min[:, 0]]) 
                     + C_interp * np.log(phi[idx_min[:, 1]]))
    # phi_res = (1 - c) * phi[idx_min[:, 0]] + c * phi[idx_min[:, 1]]
    prop_phi = np.asarray([1 - frac, frac])
    spectrum[idx_min] = np.array([1 - C_interp, C_interp]) * prop_phi[np.newaxis]
    fit = FitResult(phi_res, phi_error, x_distribution=prop_phi,
                    basis_spectrum=spectrum, residual=min_res.fun,
                    success=True, status=0)
    return fit


def res_polydisperse(C, M, b, psquare):
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


def jac_polydisperse(C, M, b, psquare):
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
        The jacobian of res_polydisperse
    """
    return 2 * C@M - 2 * b


def hess_polydisperse(C, M, b, psquare):
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
        min_res = minimize(res_polydisperse, C0, args=(Mi, bi, psquare),
                           jac=jac_polydisperse, hess=hess_polydisperse,
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
        return res_polydisperse(np.abs(C), M, b, psquare)

    def jac2(C, M, b, psquare):
        return jac_polydisperse(np.abs(C), M, b, psquare) * np.sign(C)

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