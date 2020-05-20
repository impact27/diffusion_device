# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 22:39:28 2018

@author: quentinpeter
"""
import numpy as np
from scipy.optimize import basinhopping, OptimizeResult
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
            prof_noise, vary_offset=False, fit_square=False):
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
        Number of species to fit.
    prof_noise: float or 1d array
        The noise on the profiles


    Returns
    -------
    fit: FitResult object
        the fit results

    """
    if phi is None:
        raise RuntimeError('Phi is None')

    if np.shape(np.unique(phi)) != np.shape(phi):
        raise RuntimeError('duplicated phi')

    profiles = profiles / prof_noise
    Basis = Basis / prof_noise

    if fit_square:
        profiles = np.square(profiles)
        Basis = np.square(Basis)

    if nspecies == 1:
        return fit_monodisperse(profiles, Basis, phi, vary_offset)

    if nspecies > 1:
        return fit_polydisperse(profiles, Basis, nspecies, phi, vary_offset)

    raise RuntimeError('Number of species negative!')


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
        warnings.warn("Interpolation failed: not smooth enough.")
        coeff_basis = 0
        arg_phi = arg_cent

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

    phi_background_error = error_on_fit_monodisperse(
        profiles, Basis, phi, spectrum, (arg_cent, arg_side))
    fit.phi_background_error = phi_background_error

    return fit


def fit_polydisperse(profiles, Basis, nspecies, phi, vary_offset):
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
    mono_fit = fit_monodisperse(profiles, Basis, phi, vary_offset)
    idx_min_mono = mono_fit.arg_x

    # Check shape Basis
    if len(Basis.shape) == 2:
        # add axis for pos
        Basis = Basis[:, np.newaxis]
        profiles = profiles[np.newaxis, :]

    sum_matrices = get_sum_matrices(profiles, Basis)
    C0 = np.arange(nspecies) * 3 + idx_min_mono - 3 * nspecies / 2

    Nbasis = np.shape(Basis)[0]
    constr = get_constraints(Nbasis, nspecies)

    min_res = basinhopping(
        residual_interpolated_polydisperse, C0, disp=False,
        minimizer_kwargs={'args': (sum_matrices, vary_offset),
                          'jac': jacobian_interpolated_polydisperse,
                          'constraints': constr,
                          })
    # min_res = minimize(residual_interpolated_polydisperse, C0,
    #                    args=(sum_matrices, vary_offset),
    #                    jac=jacobian_interpolated_polydisperse,
    #                    constraints=constr,
    #                    )
    return finalise(profiles, Basis, phi, min_res.x, sum_matrices, vary_offset)


def normalise_basis_factor(Basis, profiles, vary_offset=False):
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
        the normalised basis
    """
    fact_a, fact_b = normalise_basis_factor(basis, profiles, vary_offset)
    return basis * fact_a[..., np.newaxis] + np.array(fact_b)[..., np.newaxis]


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


def error_on_fit_monodisperse(profiles, basis, phi, spectrum, arg_fit):
    """Estimate the error on fit in the monodisperse case"""
    idx_a, idx_b = np.sort(arg_fit)
    dbasis = ((basis[idx_b] - basis[idx_a])
              / (phi[idx_b] - phi[idx_a]))
    fits = np.sum(basis[spectrum > 0]
                  * spectrum[spectrum > 0, np.newaxis, np.newaxis],
                  axis=0)
    background = (profiles - fits)
    error_phi = (background * dbasis
                 / np.mean(np.square(dbasis)))
    return error_phi


def error_on_fit(profiles, basis, phi, spectrum, arg_fits):
    """Estimate the error on fit in the monodisperse case"""
    dbasis = np.zeros((len(arg_fits), *np.shape(basis)[1:]))
    for i, arg_fit in enumerate(arg_fits):
        idx_a, idx_b = np.sort(arg_fit)
        dbasis[i] = (
            ((basis[idx_b] - basis[idx_a]) / (phi[idx_b] - phi[idx_a]))
            * np.sum(spectrum[[idx_a, idx_b]]))

    fits = np.sum(basis * spectrum[:, np.newaxis, np.newaxis],
                  axis=0)
    background = (profiles - fits)

    M = np.mean(dbasis[np.newaxis] * dbasis[:, np.newaxis], axis=(2, 3))

    try:
        error_phi = np.tensordot(np.linalg.inv(M), background * dbasis,
                                 axes=([0], [0]))
        return error_phi
    except np.linalg.LinAlgError:
        # The matrix is singular
        return [error_on_fit_monodisperse(
            profiles, basis, phi, spectrum, arg_fit) for arg_fit in arg_fits]


def get_idx(C_interp, idx):
    """Separate index into two"""
    idx = np.tile(idx, (2, 1)).T
    idx[:, 0] += np.asarray(np.floor(C_interp), int)
    idx[:, 1] += np.asarray(np.ceil(C_interp), int)
    C_interp = C_interp - np.floor(C_interp)
    return C_interp, idx


def interpolate_sum_matrices(index, sum_matrices, derivative=False):
    """
    Get interpolated matrices and vector

    Returned shape is:
        BB: Npart, Npart, (index.shape[:-1]), Npos
        Bp and B: Npart, (index.shape[:-1]), Npos
    """
    # Put the particle axis first as it is not present in all matrix
    index = np.rollaxis(index, -1)
    # Get the interpolation part
    # Add an axis for the position at the end
    interp_coeff = (index - np.floor(index))[..., np.newaxis]
    # i and j are symmetrical
    C_i = interp_coeff[:, np.newaxis]
    C_j = interp_coeff[np.newaxis]

    # Get floor and ceiling index to apply to the sum matrices
    index_floor = np.array(np.floor(index), int)
    index_ceil = np.array(np.ceil(index), int)
    i_idx_f = index_floor[:, np.newaxis]
    j_idx_f = index_floor[np.newaxis]
    i_idx_c = index_ceil[:, np.newaxis]
    j_idx_c = index_ceil[np.newaxis]

    BB, Bp, B = [sum_matrices[key] for key in ['BB', 'Bp', 'B']]

    if not derivative:
        # Simply interpolate BB matrix between neigboring index
        interp_BB = ((BB[i_idx_f, j_idx_f] * (1 - C_j) +
                      BB[i_idx_f, j_idx_c] * C_j) * (1 - C_i) +
                     (BB[i_idx_c, j_idx_f] * (1 - C_j) +
                      BB[i_idx_c, j_idx_c] * C_j) * C_i)

        interp_Bp = ((1 - interp_coeff) * Bp[index_floor]
                     + interp_coeff * Bp[index_ceil])

        interp_B = ((1 - interp_coeff) * B[index_floor]
                    + interp_coeff * B[index_ceil])

        return interp_BB, interp_Bp, interp_B

    # Derivative index is d. The index are packed in dij order
    # Create delta functions with the correct shape
    Ns = np.shape(index)[0]
    delta_di = np.eye(Ns)[:, :, np.newaxis]
    delta_dj = np.eye(Ns)[:, np.newaxis]
    delta_di.shape = (*np.shape(delta_di), *np.ones(len(np.shape(index)), int))
    delta_dj.shape = (*np.shape(delta_dj), *np.ones(len(np.shape(index)), int))

    # axis are dij...k
    dBB = (((BB[i_idx_c, j_idx_f] - BB[i_idx_f, j_idx_f]) * (1 - C_j) +
            (BB[i_idx_c, j_idx_c] - BB[i_idx_f, j_idx_c]) * C_j) * delta_di +
           ((BB[i_idx_f, j_idx_c] - BB[i_idx_f, j_idx_f]) * (1 - C_i) +
            (BB[i_idx_c, j_idx_c] - BB[i_idx_c, j_idx_f]) * C_i) * delta_dj)

    # axis are d...k
    dBp = Bp[index_ceil] - Bp[index_floor]
    dB = B[index_ceil] - B[index_floor]
    return dBB, dBp, dB


def myinverse(M):
    """Inverse or set to nan if det(M) = 0"""
    Mm1 = np.ones_like(M) * np.nan
    # Transpose as axis 0 and 1 must be at the end
    # The double precision is ~ 1e16. Remove unreasonable values (close to 0)
    mask = np.abs(np.linalg.det(M.T) / np.max(M, axis=(0, 1))**2) > 1e-12
    Mm1.T[mask] = np.linalg.inv(M.T[mask])
    return Mm1


def residual_N_floating(index, sum_matrices, vary_offset=False):
    """
    Compute the residual and ratio for two spicies.

    Fit = sum(ai * Bi + bi)
    """
    index_shape = np.shape(index)[:-1]
    BB_ijk, Bp_ik, B_ik = interpolate_sum_matrices(index, sum_matrices)
    pp, p, Npix = [sum_matrices[key] for key in ['pp', 'p', '1']]
    BB_ij = np.sum(BB_ijk, axis=-1)
    Bp_i = np.sum(Bp_ik, axis=-1)

    if not vary_offset:
        BB_ij_m1 = myinverse(BB_ij)
        coeff_a = np.einsum('ij..., j... -> i...', BB_ij_m1, Bp_i)
        coeff_b = np.zeros((*index_shape, len(p)))
    else:
        cov_BB_ij = np.sum(BB_ijk - 1 / Npix * B_ik[:, np.newaxis] * B_ik,
                           axis=-1)
        cov_Bp_i = np.sum(Bp_ik - 1 / Npix * (B_ik * p), axis=-1)

        cov_BB_ij_m1 = myinverse(cov_BB_ij)
        coeff_a = np.einsum('ij..., j... -> i...', cov_BB_ij_m1, cov_Bp_i)
        coeff_b = 1 / Npix * (
            p - np.einsum('i...k, i... -> ...k', B_ik, coeff_a))
    # Can not have negative values
    coeff_a[coeff_a < 0] = 0

    residual = (np.einsum('i..., ij..., j... -> ...', coeff_a, BB_ij, coeff_a)
                - 2 * np.einsum('i..., i... -> ...', coeff_a, Bp_i)
                + np.sum(pp))
    if vary_offset:
        residual += (
            Npix * np.sum(coeff_b**2, axis=-1)
            + 2 * np.einsum('i..., ...k, i...k -> ...', coeff_a, coeff_b, B_ik)
            - 2 * np.sum(coeff_b * p, axis=-1))

    return residual, coeff_a, coeff_b


def jacobian_interpolated_polydisperse(index, sum_matrices, vary_offset=False):
    """Jacobian function of res_interp_2"""
    # if np.min(index) < 0 or np.max(index) > len(Bp) - 1:
    #     return index * np.nan
    if np.any(index < 0) or np.any(index > len(sum_matrices['B']) - 1):
        return index * np.nan
    __, coeff_a, coeff_b = residual_N_floating(
        index, sum_matrices, vary_offset)

    dBB_dijk, dBp_dk, dB_dk = interpolate_sum_matrices(
        index, sum_matrices, derivative=True)

    dres = (
        np.einsum('i..., j..., dij...k -> d...', coeff_a, coeff_a, dBB_dijk)
        - 2 * np.einsum('d..., d...k -> d...', coeff_a, dBp_dk)
        + 2 * np.einsum('d..., ...k, d...k -> d...', coeff_a, coeff_b, dB_dk))

    return dres


def residual_interpolated_polydisperse(index, sum_matrices, vary_offset=False):
    """Compute the residual for two spicies"""
    if np.any(index < 0) or np.any(index > len(sum_matrices['B']) - 1):
        return np.nan
    try:
        residual, coeff_a, _ = residual_N_floating(
            index, sum_matrices, vary_offset)
        if np.any(coeff_a < 0):
            return np.nan
        return residual
    except Exception:
        return np.nan


def get_sum_matrices(profiles, Basis):
    """
    Return all the summations over the pixels
    """
    matrices = {}
    matrices['BB'] = np.einsum('ijk, ljk -> ilj', Basis, Basis)
    matrices['Bp'] = np.einsum('ijk, jk -> ij', Basis, profiles)
    matrices['B'] = np.einsum('ijk -> ij', Basis)
    matrices['pp'] = np.einsum('ik, ik -> i', profiles, profiles)
    matrices['p'] = np.einsum('ik -> i', profiles)
    matrices['1'] = np.shape(Basis)[-1]
    return matrices


def finalise(profiles, Basis, phi, index, sum_matrices, vary_offset):
    """Finalise fit and return results"""
    index = np.sort(index)
    # Finalise
    res_fit, coeff_a, _ = residual_N_floating(index, sum_matrices, vary_offset)

    # Get left and right index for interpolation of result
    C_interp, idx_min = get_idx(index - np.floor(index),
                                np.asarray(np.floor(index), int))
    phi_res = np.exp((1 - C_interp) * np.log(phi[idx_min[:, 0]])
                     + C_interp * np.log(phi[idx_min[:, 1]]))

    # Sort phis
    Nb = np.shape(Basis)[0]
    # Get errors
    phi_error = np.zeros(len(phi_res))
    spectrum = np.zeros(Nb)

    BB = sum_matrices['BB']
    for rn, i in enumerate(index):
        # Left
        i = int(np.floor(i))
        # Right()
        j = i + 1
        # if right on the wall, then swap
        if j == Nb:
            j = Nb - 2
        # Get local gradient

        Bl_minus_Br_square = (BB[i, i] + BB[j, j] - BB[i, j] - BB[j, i])
        Bl_minus_Br_square = np.sum(coeff_a[..., rn] * Bl_minus_Br_square)
        if Bl_minus_Br_square == 0:
            # raise RuntimeError("No Gradient in Basis")
            error = np.nan
        else:
            # Compute error
            error = (np.sqrt((phi[i] - phi[j])**2
                             / Bl_minus_Br_square))
        phi_error[rn] = error

    # phi_res = (1 - c) * phi[idx_min[:, 0]] + c * phi[idx_min[:, 1]]
    spectrum_values = (np.array([1 - C_interp, C_interp]).T
                       * coeff_a[:, np.newaxis])
    for idx, value in zip(idx_min, spectrum_values):
        spectrum[idx] += value

    distribution = coeff_a / np.sum(coeff_a)
    fit = FitResult(x=phi_res, dx=phi_error,
                    x_distribution=np.squeeze(distribution),
                    basis_spectrum=spectrum, residual=res_fit,
                    success=True, status=0, interp_coeff=C_interp)
    fit.x_range = [[x - dx, x + dx] for x, dx in zip(fit.x, fit.dx)]

    phi_background_error = error_on_fit(
        profiles, Basis, phi, spectrum, idx_min)
    fit.phi_background_error = phi_background_error

    return fit


def get_constraints(Nbasis, nspecies):
    """Get constraints such as C > 0 and C < 1

    Parameters
    ----------
    Nbasis: int
        number of coefficients
    nspecies: int
        the number of spicies

    Returns
    -------
    constr_dict: dict
        dictionnary containing constraints
    """
    constr = []
    # Need C[i]>0
    for i in range(nspecies):
        # > 0
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

        # < Nb - 1
        def cfun(C, i=i):
            return Nbasis - C[i] - 1

        def cjac(C, i=i):
            ret = np.zeros_like(C)
            ret[i] = -1
            return ret

        constr.append({
            "type": "ineq",
            "fun": cfun,
            "jac": cjac
        })
    return constr
