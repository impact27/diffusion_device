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
    Class to hold result of fitting in a consistant way

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


def fit_all(profiles, Basis, phi, *, nspecies=1,
            prof_noise, vary_offset=False):
    """Find the best radius for monodisperse/polydisperse solutions

    Parameters
    ----------
    profiles: (N x L) / (L) 1/2d array of float
        List of profiles to fit
    Basis: (M x N x L) / (M x L) 2/3d array of float
        List of basis to fit
    phi: (M) 1d array of float
        The test parameters
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

    profiles = profiles / prof_noise
    Basis = Basis / prof_noise

    if nspecies == 1 and phi is not None:
        return fit_monodisperse(profiles, Basis, phi, vary_offset)

    elif nspecies == 2:
        return fit_2_alt(profiles, Basis, phi, vary_offset)
        # return fit_2(profiles, Basis, phi)

    elif nspecies > 0:
        return fit_N(profiles, Basis, nspecies, phi)

    elif nspecies == 0:
        return fit_polydisperse(profiles, Basis, phi)

    else:
        raise RuntimeError('Number of species negative!')


def normalise_basis_factor(Basis, profiles, vary_offset):
    """Normalise basis so they correspond to profiles

    Parameters
    ----------
    Basis: (M x N x L) / (M x L) 2/3d array of float
        List of basis to normalise
    profiles: (N x L) / (L) 1/2d array of float
        List of reference profiles

    Returns
    -------
    Basis: (M x N x L) / (M x L) 2/3d array of float
        the normalised basis factors
    """
    # return np.mean(profiles, -1)/np.mean(Basis, -1), 0

    mean_p = np.mean(profiles, -1)
    mean_Basis = np.mean(Basis, -1)
    mean_pBasis = np.mean(Basis * profiles, -1)
    mean_Bsquare = np.mean(Basis * Basis, -1)

    if vary_offset:
        covBp = mean_pBasis - mean_p * mean_Basis
        varB = mean_Bsquare - mean_Basis * mean_Basis
        fact_a = covBp / varB
        fact_a[varB < 1e-15] = 0
        fact_b = mean_p - fact_a * mean_Basis
        return fact_a, fact_b
    else:
        return mean_pBasis / mean_Bsquare, 0


def normalise_basis(basis, profiles, vary_offset):
    """Normalise basis so they corrrespond to profiles

    Parameters
    ----------
    Basis: (M x N x L) / (M x L) 2/3d array of float
        List of basis to normalise
    profiles: (N x L) / (L) 1/2d array of float
        List of reference profiles

    Returns
    -------
    Basis: (M x N x L) / (M x L) 2/3d array of float
        the normalised basis
    """
    fact_a, fact_b = normalise_basis_factor(basis, profiles, vary_offset)
    return basis * fact_a[..., np.newaxis] + np.array(fact_b)[..., np.newaxis]

# @profile


def get_matrices(profiles, Basis, fullM=True):
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
    flatbasis = np.reshape(Basis, (Nb, -1))
    flatprofs = np.ravel(profiles)

    psquare = np.sum(flatprofs * flatprofs)
    b = np.sum(flatbasis * flatprofs[np.newaxis], -1)
    if fullM:
        M = np.tensordot(flatbasis, flatbasis, (1, 1))
        return M, b, psquare
    else:
        M_diag = np.sum(flatbasis**2, -1)
        M_udiag = np.sum(flatbasis[1:] * flatbasis[:-1], -1)
        return M_diag, M_udiag, b, psquare


def interpolate_1pos(arg_cent, arg_side, M_diag, M_udiag, b):
    """Interpolated 1 position to find the best fit"""
    # np.sum((b1-b2)**2
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


def fit_monodisperse(profiles, Basis, phi, vary_offset=False):
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
    if np.any(np.isnan(profiles)):
        raise RuntimeError(f'Profiles can not be nan')

    # Normalize the basis to fit profiles
    Basis = normalise_basis(Basis, profiles, vary_offset)

    # Get matrices to avoid recalculating
    M_diag, M_udiag, b, psquare = get_matrices(
        profiles, Basis, fullM=False)

    # get best residual
    res = psquare + M_diag - 2 * b
    arg_cent = np.argmin(res)
    if arg_cent == 0:
        raise RuntimeError(f'Phi too large')
    if arg_cent == len(b) - 1:
        raise RuntimeError(f'Phi too small')

    # Get Interpolated best result
    arg_side = arg_cent + 1
    coeff_basis, Bl_minus_Br_square = interpolate_1pos(
        arg_cent, arg_side, M_diag, M_udiag, b)
    arg_phi = arg_cent + coeff_basis

    if coeff_basis < 0:
        arg_side = arg_cent - 1
        coeff_basis, Bl_minus_Br_square = interpolate_1pos(
            arg_cent, arg_side, M_diag, M_udiag, b)
        arg_phi = arg_cent - coeff_basis

    if np.abs(coeff_basis) > 3:
        raise RuntimeError("Interpolation failed: out of bounds")

    best_phi = np.exp((1 - coeff_basis) * np.log(phi[arg_cent])
                      + coeff_basis * np.log(phi[arg_side]))

    # Save spectrum for consistent return
    spectrum = np.zeros(len(b))
    spectrum[arg_cent] = 1 - coeff_basis
    spectrum[arg_side] = coeff_basis

    # Get error
    # sqrt(dR**2/np.sum((b1-b2)**2)*sigma
    phi_error = (np.sqrt(np.square(phi[arg_cent] - phi[arg_side])
                         / Bl_minus_Br_square))
    # Get residual
    # B = (1-c) B_0 + c B_1
    Mij = M_udiag[np.min([arg_cent, arg_side])]
    BB = ((1 - coeff_basis)**2 * M_diag[arg_cent]
          + 2 * coeff_basis * (1 - coeff_basis) * Mij
          + coeff_basis**2 * M_diag[arg_side])
    By = (1 - coeff_basis) * b[arg_cent] + coeff_basis * b[arg_side]
    residual = BB - 2 * By + psquare

    # Get range (use delta xi^2)
    minres = np.min(res[res > 0])
    threshold = minres + 2 * np.sqrt(minres)
    possible = res <= threshold

    argmin = np.argwhere(possible)[0][0]
    argmax = np.argwhere(possible)[-1][0]

    if argmin > 0:
        roots = np.roots(np.polyfit(phi[argmin - 1:argmin + 2],
                                    res[argmin - 1:argmin + 2] - threshold,
                                    2))
        phi_min = np.min(roots)
    else:
        phi_min = phi[argmin]

    if argmax < len(res) - 1:
        roots = np.roots(np.polyfit(phi[argmax - 1:argmax + 2],
                                    res[argmax - 1:argmax + 2] - threshold,
                                    2))
        phi_max = np.max(roots)

    else:
        phi_max = phi[argmax]

    phiRange = [phi_min, phi_max]

    fit = FitResult(x=best_phi, dx=phi_error, x_distribution=1,
                    x_range=phiRange,
                    interp_coeff=coeff_basis, basis_spectrum=spectrum,
                    residual=residual, arg_x=arg_phi, success=True)

    return fit


def get_idx(C_interp, idx):
    """Separate index into two"""
    idx = np.tile(idx, (2, 1)).T
    idx[:, 0] += np.asarray(np.floor(C_interp), int)
    idx[:, 1] += np.asarray(np.ceil(C_interp), int)
    C_interp = C_interp - np.floor(C_interp)
    return C_interp, idx


def residual_2_floating(index, M, b, psquare):
    """Compute the residual and ratio for two spicies"""
    BB, By = get_matrices_interp_N(np.moveaxis(index, 0, -1), M, b)

    # v = y - Bi
    # w = Bj - Bi

    # v * w
    VW = (BB[..., 0, 0] - BB[..., 0, 1] - By[..., 0] + By[..., 1])
    WW = (BB[..., 1, 1] - 2 * BB[..., 0, 1] + BB[..., 0, 0])
    VV = (BB[..., 0, 0] - 2 * By[..., 0]) + psquare

    fraction = np.zeros(np.shape(BB[..., 0, 1]))
    valid = WW != 0
    fraction[valid] = VW[valid] / WW[valid]
    fraction[fraction > 1] = 1
    fraction[fraction < 0] = 0

    # Resibual for each combination
    residual = (- fraction * VW + VV)

    return residual, fraction


def get_matrices_interp_N(index, BB, Bp, B=None):
    """Get interpolated matrices and vector"""
    interp_coeff = index - np.floor(index)
    index_floor = np.array(np.floor(index), int)
    index_ceil = np.array(np.ceil(index), int)

    C_i = interp_coeff[..., np.newaxis]
    C_j = interp_coeff[..., np.newaxis, :]
    i_idx_f = index_floor[..., np.newaxis]
    j_idx_f = index_floor[..., np.newaxis, :]
    i_idx_c = index_ceil[..., np.newaxis]
    j_idx_c = index_ceil[..., np.newaxis, :]

    BiBi = ((BB[..., i_idx_f, j_idx_f] * (1 - C_j) +
             BB[..., i_idx_f, j_idx_c] * C_j) * (1 - C_i) +
            (BB[..., i_idx_c, j_idx_f] * (1 - C_j) +
             BB[..., i_idx_c, j_idx_c] * C_j) * C_i)

    Bip = (1 - interp_coeff) * Bp[..., index_floor] + \
        interp_coeff * Bp[..., index_ceil]
    if B is None:
        return BiBi, Bip

    Bi = (1 - interp_coeff) * B[..., index_floor] + \
        interp_coeff * B[..., index_ceil]

    return BiBi, Bip, Bi


def myinverse(M):
    Mm1 = np.ones_like(M) * np.nan
    mask = np.linalg.det(M) != 0
    Mm1[mask] = np.linalg.inv(M[mask])
    return Mm1


def residual_N_floating(index, BB, Bp, B, p, pp, Npix, vary_offset=False):
    """Compute the residual and ratio for two spicies"""
    BiBi, Bip, Bi = get_matrices_interp_N(index, BB, Bp, B)
    if len(np.shape(p)) > 0:
        scalar_shape = (len(pp), *np.ones(len(np.shape(index)) + 1, int))
        pp = np.reshape(pp, scalar_shape)
        p = np.reshape(p, scalar_shape)

    if not vary_offset:
        BiBim1 = myinverse(BiBi)
        # residual = (- Bip[..., np.newaxis, :] @ BiBim1 @ Bip[..., np.newaxis]
        #             + pp)[..., 0, 0]
        coeff_a = (BiBim1 @ Bip[..., np.newaxis])[..., 0]
        coeff_b = np.zeros_like(coeff_a[..., 0])[..., np.newaxis, np.newaxis]
    else:

        covBiBi = BiBi / Npix - (Bi[..., np.newaxis] /
                                 Npix) @ (Bi[..., np.newaxis, :] / Npix)
        covBip = Bip / Npix - Bi / Npix * p / Npix

        covBiBi = myinverse(covBiBi)
        coeff_a = (covBiBi @ covBip[..., np.newaxis])[..., 0]
        coeff_b = (p / Npix
                   - covBip[..., np.newaxis, :] @ coeff_a[..., np.newaxis]
                   )
    # Can not have negative values
    coeff_a[coeff_a < 0] = 0

    residual = (coeff_a[..., np.newaxis, :] @ BiBi @ coeff_a[..., np.newaxis]
                - 2 * coeff_a[..., np.newaxis, :] @ Bip[..., np.newaxis]
                + pp
                + coeff_b**2 * Npix
                + 2 * coeff_b * coeff_a[...,
                                        np.newaxis, :] @ Bi[..., np.newaxis]
                - 2 * coeff_b * p
                )[..., 0, 0]

    return residual, coeff_a, coeff_b[..., 0, 0]


def res_interp_N(index, BB, Bp, B, p, pp, Npix, vary_offset=False):
    """Compute the residual for two spicies"""
    try:
        return np.sum(residual_N_floating(
            index, BB, Bp, B, p, pp, vary_offset)[0], 0)
    except BaseException as e:
        return np.nan


def jac_interp_N(index, BB, Bp, B, p, pp, Npix, vary_offset=False):
    """Jacobian function of res_interp_2"""
    if np.min(index) < 0 or np.max(index) > len(Bp) - 1:
        return index * np.nan

    __, coeff_a, coeff_b = residual_N_floating(
        index, BB, Bp, B, p, pp, Npix, vary_offset)

    interp_coeff = index - np.floor(index)
    index_floor = np.array(np.floor(index), int)
    index_ceil = np.array(np.ceil(index), int)

    C_i = interp_coeff[..., np.newaxis]
    i_idx_f = index_floor[..., np.newaxis]
    j_idx_f = index_floor[..., np.newaxis, :]
    i_idx_c = index_ceil[..., np.newaxis]
    j_idx_c = index_ceil[..., np.newaxis, :]

    coeff_ai = coeff_a[..., np.newaxis]

    BidBi = ((1 - C_i) * (BB[i_idx_f, j_idx_c] - BB[i_idx_f, j_idx_f])
             + C_i * (BB[i_idx_c, j_idx_c] - BB[i_idx_c, j_idx_f]))

    dBi = B[index_ceil] - B[index_floor]

    dBip = Bp[index_ceil] - Bp[index_floor]

    dres = 2 * (coeff_ai * BidBi @ coeff_ai
                + coeff_b * coeff_ai * dBi[..., np.newaxis]
                - coeff_ai * dBip[..., np.newaxis])

    return dres[..., 0]


def res_interp_2(index, M, b, psquare):
    """Compute the residual for two spicies"""
    try:
        return residual_2_floating(index, M, b, psquare)[0]
    except RuntimeError as e:
        print(e)
        return np.nan


def jac_interp_2(index, M, b, psquare):
    """Jacobian function of res_interp_2"""
    if np.min(index) < 0 or np.max(index) > len(b) - 1:
        return index * np.nan
    nspecies = 2
    fraction = residual_2_floating(index, M, b, psquare)[1]
    C_phi = np.asarray([1 - fraction, fraction])

    interp_coeff = index - np.floor(index)
    index_floor = np.array(np.floor(index), int)
    index_ceil = np.array(np.ceil(index), int)

    idx = np.asarray([index_floor, index_ceil]).T

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


def get_zoom_indices(residual, indices, idx_min_mono, N, threshold):
    """Get the zoom indices"""
    zoom_mask = residual <= threshold

    zoom_x = (idx_min_mono - indices[..., 0])[zoom_mask]
    zoom_y = (indices[..., 1] - idx_min_mono)[zoom_mask]

    zoom_product = np.sqrt(zoom_x**2 * zoom_y**2)
    zoom_ratio = np.sqrt(zoom_x**2 / zoom_y**2)

    zoom_product = np.exp(np.linspace(
        np.log(np.min(zoom_product)),
        np.log(np.max(zoom_product)),
        101))[np.newaxis, :]
    zoom_ratio = np.tan(np.linspace(
        np.arctan(np.min(zoom_ratio)),
        np.arctan(np.max(zoom_ratio)),
        101))[:, np.newaxis]

    zoom_x = np.sqrt(zoom_product * zoom_ratio)
    zoom_y = np.sqrt(zoom_product / zoom_ratio)

    zoom_indices = np.asarray([idx_min_mono - zoom_x, zoom_y + idx_min_mono])
    zoom_indices = np.moveaxis(zoom_indices, 0, -1)

    zoom_valid = np.logical_and(zoom_indices > 0, zoom_indices < N - 1)
    zoom_valid = np.logical_and(zoom_valid[..., 0], zoom_valid[..., 1])
    zoom_indices = zoom_indices[zoom_valid]

    return zoom_indices, zoom_valid

# @profile


def fit_2_alt(profiles, Basis, phi, vary_offset=False):
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
    # Fit monodisperse to get mid point
    mono_fit = fit_monodisperse(profiles, Basis, phi, vary_offset)
    idx_min_mono = mono_fit.arg_x

    # Check shape Basis
    if len(Basis.shape) == 2:
        # add axis for pos
        Basis = Basis[:, np.newaxis]
        profiles = profiles[np.newaxis, :]
    # basis has phi. pos, pixel

    # Compute the matrices needed for res_interp_N
    # equivalent to BB = np.einsum('jik, lik -> ijl', Basis, Basis)?
    BB = np.empty((np.shape(Basis)[1], np.shape(Basis)[0], np.shape(Basis)[0]))
    for i in range(np.shape(Basis)[-2]):
        Bi = Basis[..., i, :]
        BB[i] = (np.tensordot(Bi, Bi, (-1, -1)))
    Bp = np.einsum('jik, ik -> ij', Basis, profiles)
    B = np.einsum('jik -> ij', Basis)
    pp = np.einsum('ik, ik -> i', profiles, profiles)
    p = np.einsum('ik -> i', profiles)
    Nb = np.shape(Basis)[0]
    Npix = np.shape(Basis)[-1]

    # Get the distance from min_mono to the wall
    N = np.min([idx_min_mono, Nb - idx_min_mono])

    # Get indices for a diagonal
    indices = np.array(
        [np.arange(1, N), - np.arange(1, N)]) + idx_min_mono
    indices = np.moveaxis(indices, 0, -1)
    # Compute diagonal
    res_diag = res_interp_N(indices, BB, Bp, B, p, pp, Npix, vary_offset)

    # If best position is mopnodisperse, stop fit
    if np.min(res_diag) > mono_fit.residual:
        warnings.warn("Monodisperse")
        fit = fit_monodisperse(profiles, Basis, phi, vary_offset)
        fit.dx = np.tile(fit.dx, 2)
        fit.x = np.tile(fit.x, 2)
        fit.interp_coeff = np.tile(fit.interp_coeff, 2)
        fit.x_distribution = np.array([1, 0])
        fit.x_range = [[x - dx, x + dx] for x, dx in zip(fit.x, fit.dx)]
        return fit

    # Get curve to look at (Cte XY)
    argmin_diag = np.argmin(res_diag) + 1
    XY = np.square(argmin_diag)
    factor = np.square(argmin_diag + 1) / XY * 2

    ratio = np.tan(
        np.linspace(0, np.pi / 2, 101)
    )[1:-1, np.newaxis]
    product = np.exp(np.linspace(np.log(XY / factor),
                                 np.log(XY * factor),
                                 101))[np.newaxis, :]
    x = np.sqrt(product * ratio)
    y = np.sqrt(product / ratio)

    indices = np.asarray([idx_min_mono - x, y + idx_min_mono])
    indices = np.moveaxis(indices, 0, -1)

    # only select valid values
    valid = np.logical_and(indices > 0, indices < Nb - 1)
    valid = np.logical_and(valid[..., 0], valid[..., 1])
    zoom_indices = indices[valid]

    # Get curve
    zoom_residual = res_interp_N(
        zoom_indices, BB, Bp, B, p, pp, Npix, vary_offset)

    # Compute threshold
    minres = np.min(zoom_residual[zoom_residual > 0])
    threshold = minres + 2 * np.sqrt(minres)

    indices_range = [np.min(zoom_indices[zoom_residual < threshold], axis=0),
                     np.max(zoom_indices[zoom_residual < threshold], axis=0)]
    phi_range = np.interp(indices_range, np.arange(len(phi)), phi)

    # Zoom twice
    for i in range(2):
        threshold = np.percentile(zoom_residual, 0.1)
        zoom_indices, zoom_valid = get_zoom_indices(
            zoom_residual, zoom_indices, idx_min_mono, Nb, threshold)
        zoom_residual = np.sum(residual_N_floating(
            zoom_indices, BB, Bp, B, p, pp, vary_offset)[0], 0)

    # Get best
    idx = np.unravel_index(np.argmin(zoom_residual), np.shape(zoom_residual))
    index = zoom_indices[idx]

    if np.min(index) == 0 or np.max(index) == Nb - 1:
        raise RuntimeError("Fit out of range")

    index = np.squeeze(index)

    # Reaffine result?
    # min_res = minimize(res_interp_N, index, args=(BB, Bp, B, p, pp, Npix, vary_offset),
    #                     jac=jac_interp_N,
    #                     method='BFGS', options={'gtol': 1e-16, 'norm': 2})
    # index = np.sort(min_res.x)

    # Finalise
    res_fit, coeff_a, coeff_b = residual_N_floating(
        index, BB, Bp, B, p, pp, Npix, vary_offset)

    # Get left and right index for interpolation of result
    C_interp, idx_min = get_idx(index - np.floor(index),
                                np.asarray(np.floor(index), int))
    phi_res = np.exp((1 - C_interp) * np.log(phi[idx_min[:, 0]])
                     + C_interp * np.log(phi[idx_min[:, 1]]))

    # Sort if needed
    if phi_res[1] < phi_res[0]:
        coeff_a = coeff_a[np.argsort(phi_res)]
        phi_res = np.sort(phi_res)

    # Get errors
    phi_error = np.zeros(2)
    spectrum = np.zeros(Nb)

    for rn, i in enumerate(index):
        # Left
        i = int(np.floor(i))
        # Right()
        j = i + 1
        # if right on the wall, then swap
        if j == Nb:
            j = Nb - 2
        # Get local gradient
        Bl_minus_Br_square = (BB[..., i, i] + BB[..., j, j]
                              - BB[..., i, j] - BB[..., j, i])
        Bl_minus_Br_square = np.sum(coeff_a[..., rn] * Bl_minus_Br_square)
        if Bl_minus_Br_square == 0:
            raise RuntimeError("No Gradient in Basis")
        # Compute error
        error = (np.sqrt((phi[i] - phi[j])**2
                         / Bl_minus_Br_square))
        phi_error[rn] = error

    # Dodgy for now
    coeff_a = coeff_a[np.argmin(res_fit)]
    # phi_res = (1 - c) * phi[idx_min[:, 0]] + c * phi[idx_min[:, 1]]
    spectrum[idx_min] = (np.array([1 - C_interp, C_interp]).T
                         * coeff_a[:, np.newaxis])

    distribution = coeff_a / np.sum(coeff_a)
    fit = FitResult(x=phi_res, dx=phi_error,
                    x_distribution=np.squeeze(distribution),
                    basis_spectrum=spectrum, residual=np.sum(res_fit, 0),
                    success=True, status=0, interp_coeff=C_interp)
    fit.x_range = phi_range.T
    return fit

def fit_2_fix_1(profiles, Basis, phi, phi_fix, vary_offset=False):
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
    # Fit monodisperse to get mid point
    mono_fit = fit_monodisperse(profiles, Basis, phi, vary_offset)
    idx_min_mono = mono_fit.arg_x

    # Check shape Basis
    if len(Basis.shape) == 2:
        # add axis for pos
        Basis = Basis[:, np.newaxis]
        profiles = profiles[np.newaxis, :]
    # basis has phi. pos, pixel

    # Compute the matrices needed for res_interp_N
    # equivalent to BB = np.einsum('jik, lik -> ijl', Basis, Basis)?
    BB = np.empty((np.shape(Basis)[1], np.shape(Basis)[0], np.shape(Basis)[0]))
    for i in range(np.shape(Basis)[-2]):
        Bi = Basis[..., i, :]
        BB[i] = (np.tensordot(Bi, Bi, (-1, -1)))
    Bp = np.einsum('jik, ik -> ij', Basis, profiles)
    B = np.einsum('jik -> ij', Basis)
    pp = np.einsum('ik, ik -> i', profiles, profiles)
    p = np.einsum('ik -> i', profiles)
    Nb = np.shape(Basis)[0]
    Npix = np.shape(Basis)[-1]

    # get indix for fixed phi
    idx_fix_minus = np.sum(phi - phi_fix < 0) - 1
    if idx_fix_minus == -1 or idx_fix_minus == len(phi) - 1:
        raise RuntimeError('Fixed phi out of range')
    idx_fix = ((np.log(phi_fix) - np.log(phi[idx_fix_minus]))
               / (np.log(phi[idx_fix_minus + 1]) - np.log(phi[idx_fix_minus]))
               + idx_fix_minus)
    index = np.arange(len(phi))

    if idx_min_mono > idx_fix:
        # Other larger
        idx_2 = index[phi > phi_fix]
    else:
        idx_2 = index[phi < phi_fix]

    idx_min = None
    for i in range(1):
        if idx_min is not None:
            if idx_min == 0:
                idx_min = 1
            elif idx_min == len(idx_2) - 1:
                idx_min = idx_min - 1
            idx_2 = np.lispace(idx_2[idx_min - 1], idx_2[idx_min + 1], 20)
        # Get indices for a diagonal
        indices = np.array(
            [[idx_fix] * len(idx_2), idx_2])
        indices = np.moveaxis(indices, 0, -1)

        # Get curve
        zoom_residual = res_interp_N(
            indices, BB, Bp, B, p, pp, Npix, vary_offset)
        idx_min = np.argmin(zoom_residual)


    idx_res = idx_2[idx_min]
    C_interp = index - np.floor(index)

    phi_res = np.exp((1 - C_interp) * np.log(phi[int(np.floor(idx_res))])
                     + C_interp * np.log(phi[int(np.ceil(idx_res))]))
    raise NotImplementedError()
    fit = FitResult(x=[phi_fix, phi_res],
                    dx=[0, phi_error],
                    x_distribution=np.squeeze(distribution),
                    basis_spectrum=spectrum,
                    residual=np.sum(res_fit, 0),
                    success=True, status=0, interp_coeff=C_interp)
    fit.x_range = phi_range.T
    return fit

def fit_2(profiles, Basis, phi):
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
    Basis_factor = np.mean(profiles, -1) / np.mean(Basis, -1)
    Basis = Basis * Basis_factor[..., np.newaxis]

    M, b, psquare = get_matrices(profiles, Basis)

    Nb = len(b)

    M_diag = np.diag(M)

    # Get 1d result. This is between the two!
    # Use that to limit search space
    res_1 = psquare + M_diag - 2 * b
    argmin_1 = np.argmin(res_1)

    coeff_basis, Bl_minus_Br_square = interpolate_1pos(
        argmin_1, argmin_1 + 1, np.diag(M), np.diag(M, 1), b)

    idx_min_mono = argmin_1 + coeff_basis

    N = np.min([idx_min_mono, len(b) - idx_min_mono])

    indices = np.array([np.linspace(idx_min_mono, idx_min_mono + N - 1, int(N)),
                        np.linspace(idx_min_mono, idx_min_mono - N - 1, int(N))])
    residual, fraction = residual_2_floating(indices, M, b, psquare)
    argmin_diag = np.argmin(residual)
    if argmin_diag == 0:
        raise RuntimeError("Monodisperse")
    XY = np.square(argmin_diag)

    factor = np.square(argmin_diag + 1) / XY * 1.1
    ratio = np.tan(
        np.linspace(
            np.arctan(XY / np.square(idx_min_mono - 1) * factor),
            np.arctan(np.square(len(b) - idx_min_mono - 1) / XY / factor),
            101)
    )[:, np.newaxis]
    product = np.exp(np.linspace(np.log(XY / factor),
                                 np.log(XY * factor),
                                 101))[np.newaxis, :]
    x = np.sqrt(product * ratio)
    y = np.sqrt(product / ratio)

    indices = np.asarray([idx_min_mono - x, y + idx_min_mono])

    valid = np.logical_and(indices > 0, indices < len(b) - 1)
    valid = np.logical_and(valid[0], valid[1])
    indices = indices[:, valid]

    residual, fraction = residual_2_floating(indices, M, b, psquare)

    # Get best
    idx = np.unravel_index(np.argmin(residual), np.shape(residual))
    frac = fraction[idx]
    index = indices[:, idx]

    if np.min(index) == 0 or np.max(index) == Nb - 1:
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
        error = (np.sqrt((phi[i] - phi[j])**2
                         / Bl_minus_Br_square))
        phi_error[rn] = error

    C0 = index
    min_res = minimize(res_interp_2, C0, args=(M, b, psquare),
                       jac=jac_interp_2,
                       method='BFGS', options={'gtol': 1e-16, 'norm': 2})

    index = np.sort(min_res.x)

    __, frac = residual_2_floating(index, M, b, psquare)

    # C < 0 mean interpolate to the left
    C_interp, idx_min = get_idx(index - np.floor(index),
                                np.asarray(np.floor(index), int))

    # Result
    phi_res = np.exp((1 - C_interp) * np.log(phi[idx_min[:, 0]])
                     + C_interp * np.log(phi[idx_min[:, 1]]))
    if phi_res[1] < phi_res[0]:
        phi_res = np.sort(phi_res)
        frac = 1 - frac
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
    fit.x_range = [[x - dx, x + dx] for x, dx in zip(fit.x, fit.dx)]
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


def fit_N(profiles, Basis, nspecies, phi):
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

    M, b, psquare = get_matrices(profiles, Basis)

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
        error = (np.sqrt((phi[i] - phi[j])**2
                         / (M[i, i] + M[j, j] - M[i, j] - M[j, i])))
        radius_error[rn] = error

    fit = FitResult(x=phi[idx], dx=radius_error, x_distribution=C[bestidx],
                    basis_spectrum=spectrum, residual=np.min(res))
    fit.x_range = [[x - dx, x + dx] for x, dx in zip(fit.x, fit.dx)]
    return fit


def fit_polydisperse(profiles, Basis, phi):
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

    M, b, psquare = get_matrices(profiles, Basis)

    Nb = len(b)
    C0 = np.zeros(Nb)
    C0[np.argmin(psquare + np.diag(M) - 2 * b)] = 1

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
        error = (np.sqrt((phi[i] - phi[j])**2
                         / (M[i, i] + M[j, j] - M[i, j] - M[j, i])))
        radius_error[i] = error
    radius_error[0] = radius_error[1]

    fit = FitResult(x=phi, dx=radius_error, x_distribution=spectrum,
                    basis_spectrum=spectrum, residual=res.fun)
    fit.x_range = [[x - dx, x + dx] for x, dx in zip(fit.x, fit.dx)]
    return fit

# %% Graveyard

# def res_interp_2(C, M, b, psquare, idx):
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
# def jac_interp_2(C, M, b, psquare, idx):
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

#    frac = min_res.x[0]
#    C_interp = np.asarray(min_res.x[1:])

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
# residual = (fraction**2 * fact_2
##                    + 2 * fraction * fact_1
# + fact_3)
#
#    residual = (fraction * fact_1 + fact_3)

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
