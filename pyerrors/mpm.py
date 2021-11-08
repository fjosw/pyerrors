#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.linalg
from .obs import Obs
from .linalg import svd, eig, pinv


def matrix_pencil_method(corrs, k=1, p=None, **kwargs):
    """ Matrix pencil method to extract k energy levels from data

    Implementation of the matrix pencil method based on
    eq. (2.17) of Y. Hua, T. K. Sarkar, IEEE Trans. Acoust. 38, 814-824 (1990)

    Parameters
    ----------
    data -- can be a list of Obs for the analysis of a single correlator, or a list of lists
            of Obs if several correlators are to analyzed at once.
    k -- Number of states to extract (default 1).
    p -- matrix pencil parameter which filters noise. The optimal value is expected between
         len(data)/3 and 2*len(data)/3. The computation is more expensive the closer p is
         to len(data)/2 but could possibly suppress more noise (default len(data)//2).
    """
    if isinstance(corrs[0], Obs):
        data = [corrs]
    else:
        data = corrs

    lengths = [len(d) for d in data]
    if lengths.count(lengths[0]) != len(lengths):
        raise Exception('All datasets have to have the same length.')

    data_sets = len(data)
    n_data = len(data[0])

    if p is None:
        p = max(n_data // 2, k)
    if n_data <= p:
        raise Exception('The pencil p has to be smaller than the number of data samples.')
    if p < k or n_data - p < k:
        raise Exception('Cannot extract', k, 'energy levels with p=', p, 'and N-p=', n_data - p)

    # Construct the hankel matrices
    matrix = []
    for n in range(data_sets):
        matrix.append(scipy.linalg.hankel(data[n][:n_data - p], data[n][n_data - p - 1:]))
    matrix = np.array(matrix)
    # Construct y1 and y2
    y1 = np.concatenate(matrix[:, :, :p])
    y2 = np.concatenate(matrix[:, :, 1:])
    # Apply SVD to y2
    u, s, vh = svd(y2, **kwargs)
    # Construct z from y1 and SVD of y2, setting all singular values beyond the kth to zero
    z = np.diag(1. / s[:k]) @ u[:, :k].T @ y1 @ vh.T[:, :k]
    # Return the sorted logarithms of the real eigenvalues as Obs
    energy_levels = np.log(np.abs(eig(z, **kwargs)))
    return sorted(energy_levels, key=lambda x: abs(x.value))


def matrix_pencil_method_old(data, p, noise_level=None, verbose=1, **kwargs):
    """ Older impleentation of the matrix pencil method with pencil p on given data to
        extract energy levels.

    Parameters
    ----------
    data -- lists of Obs, where the nth entry is considered to be the correlation function
            at x0=n+offset.
    p -- matrix pencil parameter which corresponds to the number of energy levels to extract.
         higher values for p can help decreasing noise.
    noise_level -- If this argument is not None an additional prefiltering via singular
                   value decomposition is performed in which all singular values below 10^(-noise_level)
                   times the largest singular value are discarded. This increases the computation time.
    verbose -- if larger than zero details about the noise filtering are printed to stdout
               (default 1)

    """
    n_data = len(data)
    if n_data <= p:
        raise Exception('The pencil p has to be smaller than the number of data samples.')

    matrix = scipy.linalg.hankel(data[:n_data - p], data[n_data - p - 1:]) @ np.identity(p + 1)

    if noise_level is not None:
        u, s, vh = svd(matrix)

        s_values = np.vectorize(lambda x: x.value)(s)
        if verbose > 0:
            print('Singular values: ', s_values)
        digit = np.argwhere(s_values / s_values[0] < 10.0**(-noise_level))
        if digit.size == 0:
            digit = len(s_values)
        else:
            digit = int(digit[0])
        if verbose > 0:
            print('Consider only', digit, 'out of', len(s), 'singular values')

        new_matrix = u[:, :digit] * s[:digit] @ vh[:digit, :]
        y1 = new_matrix[:, :-1]
        y2 = new_matrix[:, 1:]
    else:
        y1 = matrix[:, :-1]
        y2 = matrix[:, 1:]

    # Moore–Penrose pseudoinverse
    pinv_y1 = pinv(y1)

    e = eig((pinv_y1 @ y2), **kwargs)
    energy_levels = -np.log(np.abs(e))
    return sorted(energy_levels, key=lambda x: abs(x.value))
