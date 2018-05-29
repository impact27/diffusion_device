# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 22:39:28 2018

@author: quentinpeter
"""

import numpy as np
from scipy.optimize import basinhopping, minimize, OptimizeResult
from itertools import combinations
import warnings

class FitResult(OptimizeResult):
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

    if nspecies == 1 and phi is not None:
        return fit_monodisperse(profiles, Basis, profile_slice, phi, prof_noise)

    elif nspecies == 2:
        return fit_2(profiles, Basis, profile_slice, phi, prof_noise)

    elif nspecies > 0:
        return fit_N(profiles, Basis, profile_slice, nspecies, phi, prof_noise)

    elif nspecies == 0:
        return fit_polydisperse(profiles, Basis, profile_slice, phi, prof_noise)

    else:
        raise RuntimeError('Number of species negative!')

def normalise_basis_factor(basis, profiles, profile_slice):
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
    return profiles_scales / basis_scales

def normalise_basis(basis, profiles, profile_slice=slice(None)):
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
    return basis * normalise_basis_factor(basis, profiles, profile_slice)

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
    
#@profile    
def get_matrices(profiles, Basis, profile_slice=slice(None), fullM=True):
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
    flatbasis = np.reshape(Basis[..., profile_slice], (Nb, -1))
    flatprofs = np.ravel(profiles[..., profile_slice])

    psquare = np.sum(flatprofs * flatprofs)
    b = np.sum(flatbasis * flatprofs[np.newaxis], -1)
    if fullM:
        M = np.tensordot(flatbasis, flatbasis, (1, 1))
        return M, b, psquare
    else:
        M_diag = np.sum(flatbasis**2, -1)
        M_udiag = np.sum(flatbasis[1:]*flatbasis[:-1], -1)
        return M_diag, M_udiag, b, psquare
    

def interpolate_1pos(arg_cent, arg_side, M_diag, M_udiag, b):
    #np.sum((b1-b2)**2
    Mij = M_udiag[np.min([arg_cent, arg_side])]
    Bl_minus_Br_square = (M_diag[arg_cent] + M_diag[arg_side]
                          - 2 * Mij)
     # If no diff
    if Bl_minus_Br_square == 0:
        raise RuntimeError("No Gradient in Basis")
    # np.sum((b1-b2)*(p0-b2))/np.sum((b1-b2)**2)
    coeff_basis = (
        (b[arg_side] - b[arg_cent]
         - Mij + M_diag[arg_cent])
        / Bl_minus_Br_square) 
    return coeff_basis, Bl_minus_Br_square

def rescale_basis(profiles, Basis):
    # return normalise_basis(Basis, profiles)
    mean_p = np.mean(profiles, -1)
    mean_Basis = np.mean(Basis, -1)
    mean_pBasis = np.mean(Basis * profiles, -1)
    mean_Bsquare = np.mean(Basis * Basis, -1)
    
    covBp = mean_pBasis - mean_p * mean_Basis
    varB = mean_Bsquare - mean_Basis * mean_Basis
    
    fact_a = covBp / varB
    fact_a[varB<1e-15] = 0
    fact_b = mean_p - fact_a * mean_Basis
    
    # return Basis * (mean_pBasis / mean_Bsquare)[..., np.newaxis]
    
    Basis = fact_a[..., np.newaxis] * Basis + fact_b[..., np.newaxis]
    return Basis
            
def fit_monodisperse(profiles, Basis, profile_slice, phi, prof_noise=1):
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
    # Normalize the basis to fit profiles
    # Basis = normalise_basis(Basis, profiles, profile_slice)
    profiles = profiles[..., profile_slice]
    Basis = Basis[..., profile_slice]

    Basis = rescale_basis(profiles, Basis)
    
    M_diag, M_udiag, b, psquare = get_matrices(
            profiles, Basis, fullM=False)
    
    
    # get best residual
    res = psquare + M_diag - 2 * b
    spectrum = np.zeros(len(b))
    
    arg_cent = np.argmin(res)
    
    if arg_cent == 0:
        raise RuntimeError(f'Phi too large')
    if arg_cent == len(b) - 1:
        raise RuntimeError(f'Phi too small')
    
    
    arg_side = arg_cent + 1
    coeff_basis, Bl_minus_Br_square = interpolate_1pos(
            arg_cent, arg_side, M_diag, M_udiag, b)        
    if coeff_basis < 0:
        arg_side = arg_cent - 1
        coeff_basis, Bl_minus_Br_square = interpolate_1pos(
                arg_cent, arg_side, M_diag, M_udiag, b)

    if np.abs(coeff_basis) > 3:
        raise RuntimeError("Interpolation failed: out of bounds")
    
    spectrum[arg_cent] = 1 - coeff_basis
    spectrum[arg_side] = coeff_basis

    best_phi = np.exp((1 - coeff_basis) * np.log(phi[arg_cent])
                      + coeff_basis * np.log(phi[arg_side]))

#    best_phi = coeff_basis * phi[arg_side] + (1-coeff_basis) * phi[arg_cent]

    # sqrt(dR**2/np.sum((b1-b2)**2)*sigma
    phi_error = (prof_noise
                 * np.sqrt(np.square(phi[arg_cent] - phi[arg_side])
                           / Bl_minus_Br_square))
    # Get residual
    # B = (1-c) B_0 + c B_1
    Mij = M_udiag[np.min([arg_cent, arg_side])]
    BB = ((1 - coeff_basis)**2 * M_diag[arg_cent]
          + 2 * coeff_basis * (1 - coeff_basis) * Mij
          + coeff_basis**2 * M_diag[arg_side])
    By = (1 - coeff_basis) * b[arg_cent] + coeff_basis * b[arg_side]
    residual = BB - 2 * By + psquare
    
    fit = FitResult(x=best_phi, dx=phi_error, x_distribution=1, 
                    interp_coeff=coeff_basis, basis_spectrum=spectrum, 
                    residual=residual)

    return fit

def get_idx(C_interp, idx):
    
    idx = np.tile(idx, (2, 1)).T
    idx[:, 0] += np.asarray(np.floor(C_interp), int)
    idx[:, 1] += np.asarray(np.ceil(C_interp), int)
    C_interp = C_interp - np.floor(C_interp)
    return C_interp, idx

def get_matrices_interp(index, M, b):
    
    interp_coeff = index - np.floor(index)
    index_floor = np.array(np.floor(index), int)
    index_ceil = np.array(np.ceil(index), int)
    
    C_i = interp_coeff[:, np.newaxis]
    C_j = interp_coeff[np.newaxis, :]
    i_idx_f = index_floor[:, np.newaxis]
    j_idx_f = index_floor[np.newaxis, :]
    i_idx_c = index_ceil[:, np.newaxis]
    j_idx_c = index_ceil[np.newaxis, :]
    
    
    BB = ((1 - C_i) * (1 - C_j) * M[i_idx_f, j_idx_f]
                    + C_i * (1 - C_j) * M[i_idx_c, j_idx_f]
                    + (1 - C_i) * C_j * M[i_idx_f, j_idx_c]
                    + C_i * C_j * M[i_idx_c, j_idx_c])
    By = (1 - interp_coeff) * b[index_floor] + interp_coeff * b[index_ceil]
    
    return By, BB

def residual_2_floating(index, M, b, psquare):

    By, BB = get_matrices_interp(index, M, b)
    
    # v = y - Bi
    # w = Bj - Bi
    
    # v * w
    VW = (BB[0, 0] - BB[0, 1] - By[0] + By[1])
    WW = (BB[1, 1] - 2 * BB[0, 1] + BB[0, 0])
    VV = (BB[0, 0] - 2 * By[0]) + psquare
    
    fraction = np.zeros(np.shape(BB[0, 1]))
    valid = WW != 0
    fraction[valid] = VW[valid]/WW[valid]
    fraction[fraction > 1] = 1
    fraction[fraction < 0] = 0
    
    # Resibual for each combination
    residual = (- fraction * VW + VV)
    
    return residual, fraction

def res_interp_2(index, M, b, psquare):
    try:
        return residual_2_floating(index, M, b, psquare)[0]
    except:
        return np.nan

def jac_interp_2(index, M, b, psquare):
    if np.min(index) < 0 or np.max(index) > len(b) -1:
        return index * np.nan
    nspecies = 2
    fraction = residual_2_floating(index, M, b, psquare)[1]
    C_phi = np.asarray([1-fraction, fraction])
    
    interp_coeff = index - np.floor(index)
    index_floor = np.array(np.floor(index), int)
    index_ceil = np.array(np.ceil(index), int)
    
    idx = np.asarray([index_floor,index_ceil ]).T

    BBk = np.zeros((nspecies, nspecies, 2))
    for i in range(nspecies):
        for j in range(nspecies):
            for k in range(2):
                BBk[i, j, k] = ((1 - interp_coeff[i]) * M[idx[i, 0], idx[j, k]]
                                + interp_coeff[i] * M[idx[i, 1], idx[j, k]])
    FitBk = np.zeros((nspecies, 2))
    for i in range(nspecies):
        for k in range(2):
            FitBk[i, k] = C_phi[0] * BBk[0, i, k] + C_phi[1] * BBk[1, i, k]

    dinterp = np.zeros(nspecies)
    for i in range(nspecies):
        dinterp[i] = 2 * C_phi[i] * (FitBk[i, 1] - FitBk[i, 0]
                                     + b[idx[i, 0]] - b[idx[i, 1]])
        
    return np.array(dinterp)

#def res_interp_2(C, M, b, psquare, idx):
#    C_phi = [1 - C[0], C[0]]
#    C_interp = C[1:]
#
#    By, BB = get_matrices_interp(idx + C_interp, M, b)
#    
#    FitFit = (C_phi[0]**2 * BB[0, 0]
#              + 2 * C_phi[1] * C_phi[0] * BB[0, 1]
#              + C_phi[1]**2 * BB[1, 1])
#    Fity = C_phi[0] * By[0] + C_phi[1] * By[1]
#
#    residual = FitFit - 2 * Fity + psquare
#    return residual
#
#
#def jac_interp_2(C, M, b, psquare, idx):
#    nspecies = 2
#    C_phi = [1 - C[0], C[0]]
#    C_interp = C[1:]
#    
#    By, BB = get_matrices_interp(idx + C_interp, M, b)
#    C_interp, idx = get_idx(C_interp, idx)
#
#    FitB = np.zeros(nspecies)
#    for i in range(nspecies):
#        FitB[i] = C_phi[0] * BB[0, i] + C_phi[1] * BB[1, i]
#
#    BBk = np.zeros((nspecies, nspecies, 2))
#    for i in range(nspecies):
#        for j in range(nspecies):
#            for k in range(2):
#                BBk[i, j, k] = ((1 - C_interp[i]) * M[idx[i, 0], idx[j, k]]
#                                + C_interp[i] * M[idx[i, 1], idx[j, k]])
#    FitBk = np.zeros((nspecies, 2))
#    for i in range(nspecies):
#        for k in range(2):
#            FitBk[i, k] = C_phi[0] * BBk[0, i, k] + C_phi[1] * BBk[1, i, k]
#
#    dinterp = np.zeros(nspecies)
#    for i in range(nspecies):
#        dinterp[i] = 2 * C_phi[i] * (FitBk[i, 1] - FitBk[i, 0]
#                                     + b[idx[i, 0]] - b[idx[i, 1]])
#
#    d0 = 2 * (FitB[1] - FitB[0] + By[0] - By[1])
#
#    return np.array([d0, *dinterp])

#@profile
def fit_2(profiles, Basis, profile_slice, phi, prof_noise=1):
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
    
    # Normalize the basis to fit profiles
    Basis_factor = normalise_basis_factor(Basis, profiles, profile_slice)
    Basis = Basis * Basis_factor
    Basis_factor = Basis_factor[..., 0]
    M, b, psquare = get_matrices(profiles, Basis, profile_slice)
    
    Nb = len(b)
    
    M_diag = np.diag(M)
    
    #Get 1d result. This is between the two!
    #Use that to limit search space
    res_1 = psquare + M_diag - 2 * b
    argmin_1 = np.argmin(res_1)
    coeff_basis, Bl_minus_Br_square = interpolate_1pos(
            argmin_1, argmin_1+1, np.diag(M), np.diag(M, 1), b)
        
    argmin_m = argmin_1 + coeff_basis
    
    N = np.min([argmin_m, len(b) - argmin_m])
        
    indices = np.array([np.linspace(argmin_m, argmin_m + N -1, int(N)),
                        np.linspace(argmin_m, argmin_m - N -1, int(N))])
    residual, fraction = residual_2_floating(indices, M, b, psquare)
    XY_arg = np.argmin(residual)
    if XY_arg == 0:
        raise RuntimeError("Monodisperse")
    XY = np.square(XY_arg)
        
    factor = np.square(XY_arg+1)/XY*1.1
    ratio = np.tan(np.linspace(
            np.arctan(XY/np.square(argmin_1-1)*factor),
            np.arctan(np.square(len(b)-argmin_1-1)/XY/factor),
            101))[:, np.newaxis]
    product = np.exp(np.linspace(np.log(XY/factor), 
                                 np.log(XY*factor), 
                                 101))[np.newaxis, :]
    x = np.sqrt(product*ratio)
    y = np.sqrt(product/ratio)
    
    
    indices = np.asarray([argmin_m - x, y + argmin_m])
    
    valid = np.logical_and(indices > 0, indices < len(b) -1)
    valid = np.logical_and(valid[0], valid[1])
    indices = indices[:, valid]
        
    residual, fraction = residual_2_floating(indices, M, b, psquare)
    
    
    
    
    
#    
#    sub_M = M[argmin_1:, :argmin_1+1]
#    
#    fact_1 = (sub_M 
#              - (M_diag - b)[argmin_1:, np.newaxis] 
#              - b[np.newaxis, :argmin_1+1])
#    fact_2 = (M_diag[np.newaxis, :argmin_1+1] 
#                - 2 * sub_M 
#                + M_diag[argmin_1:, np.newaxis])
#    fact_3 = (M_diag - 2*b)[argmin_1:, np.newaxis] + psquare 
#    
#    
#    
#    
#    fraction = np.zeros(np.shape(sub_M))
#    valid = fact_2 != 0
#    fraction[valid] = - fact_1[valid]/fact_2[valid]
#    fraction[fraction > 1] = 1
#    fraction[fraction < 0] = 0
#    
#    # Resibual for each combination
##    residual = (fraction**2 * fact_2 
##                    + 2 * fraction * fact_1 
##                    + fact_3)
#    
#    residual = (fraction * fact_1 + fact_3)
    
    # Get best
    idx = np.unravel_index(np.argmin(residual), np.shape(residual))
    frac = fraction[idx]
    index = indices[:, idx]
    
    if np.min(index) == 0 or np.max(index) == Nb-1:
        raise RuntimeError("Fit out of range")

    # Get errors
    phi_error = np.zeros(2)
    spectrum = np.zeros(len(b))

    for rn, i in enumerate(index):
        i = int(i)
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

#    # If basis not fine enough, we still have a fit, but success is False
#    if not np.all(np.diff(idx) > 3):
#        prop_phi = np.asarray([1 - frac, frac])
#        spectrum[idx] = prop_phi
#        fit = FitResult(x=phi[idx], dx=phi_error, x_distribution=prop_phi,
#                        basis_spectrum=spectrum, residual=np.min(residual),
#                        success=False, status=1)
#        return fit

    # Find interpolation for each position
#    C0 = [frac, 0, 0]
    C0 = index
    min_res = minimize(res_interp_2, C0, args=(M, b, psquare),
                       jac=jac_interp_2,
                       method='BFGS', options={'gtol': 1e-16, 'norm': 2})
    
    index = np.sort(min_res.x)
    
    __, frac = residual_2_floating(index, M, b, psquare)
    
#    frac = min_res.x[0]
#    C_interp = np.asarray(min_res.x[1:])
    
    
    
    # C < 0 mean interpolate to the left
    C_interp, idx_min = get_idx(index-np.floor(index), 
                                np.asarray(np.floor(index), int))
    
    
    # Result
    phi_res = np.exp((1 - C_interp) * np.log(phi[idx_min[:, 0]]) 
                     + C_interp * np.log(phi[idx_min[:, 1]]))
    if phi_res[1] < phi_res[0]:
        phi_res = np.sort(phi_res)
        frac = 1-frac
    # phi_res = (1 - c) * phi[idx_min[:, 0]] + c * phi[idx_min[:, 1]]
    prop_phi = np.asarray([1 - frac, frac])
    spectrum[idx_min] = (np.array([1 - C_interp, C_interp]).T 
                        * prop_phi[:, np.newaxis])
    
    spectrum[idx_min] *= Basis_factor[idx_min]
    prop_phi *= np.ravel((1 - C_interp) * Basis_factor[idx_min[:, 0]]
                          + C_interp * Basis_factor[idx_min[:, 1]])
    fit = FitResult(x=phi_res, dx=phi_error, x_distribution=prop_phi,
                    basis_spectrum=spectrum, residual=min_res.fun,
                    success=True, status=0, interp_coeff=C_interp)
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


def fit_N(profiles, Basis, profile_slice, nspecies, phi, prof_noise=1):
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
    
    M, b, psquare = get_matrices(profiles, Basis, profile_slice)
    
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

    fit = FitResult(x=phi[idx], dx=radius_error, x_distribution=C[bestidx],
                    basis_spectrum=spectrum, residual=np.min(res))

    return fit


def fit_polydisperse(profiles, Basis, profile_slice, phi, prof_noise=1):
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
    
    M, b, psquare = get_matrices(profiles, Basis, profile_slice)

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

    fit = FitResult(x=phi, dx=radius_error, x_distribution=spectrum,
                    basis_spectrum=spectrum, residual=res.fun)
    return fit