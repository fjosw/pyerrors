#!/usr/bin/env python
# coding: utf-8

import numpy as np
import autograd.numpy as anp  # Thinly-wrapped numpy
from .pyerrors import derived_observable


# This code block is directly taken from the current master branch of autograd and remains
# only until the new version is released on PyPi
from functools import partial
from autograd.extend import defvjp

_dot = partial(anp.einsum, '...ij,...jk->...ik')
# batched diag
_diag = lambda a: anp.eye(a.shape[-1]) * a
# batched diagonal, similar to matrix_diag in tensorflow


def _matrix_diag(a):
    reps = anp.array(a.shape)
    reps[:-1] = 1
    reps[-1] = a.shape[-1]
    newshape = list(a.shape) + [a.shape[-1]]
    return _diag(anp.tile(a, reps).reshape(newshape))

# https://arxiv.org/pdf/1701.00392.pdf Eq(4.77)
# Note the formula from Sec3.1 in https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf is incomplete


def grad_eig(ans, x):
    """Gradient of a general square (complex valued) matrix"""
    e, u = ans  # eigenvalues as 1d array, eigenvectors in columns
    n = e.shape[-1]

    def vjp(g):
        ge, gu = g
        ge = _matrix_diag(ge)
        f = 1 / (e[..., anp.newaxis, :] - e[..., :, anp.newaxis] + 1.e-20)
        f -= _diag(f)
        ut = anp.swapaxes(u, -1, -2)
        r1 = f * _dot(ut, gu)
        r2 = -f * (_dot(_dot(ut, anp.conj(u)), anp.real(_dot(ut, gu)) * anp.eye(n)))
        r = _dot(_dot(anp.linalg.inv(ut), ge + r1 + r2), ut)
        if not anp.iscomplexobj(x):
            r = anp.real(r)
            # the derivative is still complex for real input (imaginary delta is allowed), real output
            # but the derivative should be real in real input case when imaginary delta is forbidden
        return r
    return vjp


defvjp(anp.linalg.eig, grad_eig)
# End of the code block from autograd.master


def scalar_mat_op(op, obs, **kwargs):
    """Computes the matrix to scalar operation op to a given matrix of Obs."""
    def _mat(x, **kwargs):
        dim = int(np.sqrt(len(x)))
        if np.sqrt(len(x)) != dim:
            raise Exception('Input has to have dim**2 entries')

        mat = []
        for i in range(dim):
            row = []
            for j in range(dim):
                row.append(x[j + dim * i])
            mat.append(row)

        return op(anp.array(mat))

    if isinstance(obs, np.ndarray):
        raveled_obs = (1 * (obs.ravel())).tolist()
    elif isinstance(obs, list):
        raveled_obs = obs
    else:
        raise TypeError('Unproper type of input.')
    return derived_observable(_mat, raveled_obs, **kwargs)


def mat_mat_op(op, obs, **kwargs):
    """Computes the matrix to matrix operation op to a given matrix of Obs."""
    if kwargs.get('num_grad') is True:
        return _num_diff_mat_mat_op(op, obs, **kwargs)
    return derived_observable(lambda x, **kwargs: op(x), obs)


def eigh(obs, **kwargs):
    """Computes the eigenvalues and eigenvectors of a given hermitian matrix of Obs according to np.linalg.eigh."""
    if kwargs.get('num_grad') is True:
        return _num_diff_eigh(obs, **kwargs)
    w = derived_observable(lambda x, **kwargs: anp.linalg.eigh(x)[0], obs)
    v = derived_observable(lambda x, **kwargs: anp.linalg.eigh(x)[1], obs)
    return w, v


def eig(obs, **kwargs):
    """Computes the eigenvalues of a given matrix of Obs according to np.linalg.eig."""
    if kwargs.get('num_grad') is True:
        return _num_diff_eig(obs, **kwargs)
        # Note: Automatic differentiation of eig is implemented in the git of autograd
        # but not yet released to PyPi (1.3)
    w = derived_observable(lambda x, **kwargs: anp.real(anp.linalg.eig(x)[0]), obs)
    return w


def pinv(obs, **kwargs):
    """Computes the Moore-Penrose pseudoinverse of a matrix of Obs."""
    if kwargs.get('num_grad') is True:
        return _num_diff_pinv(obs, **kwargs)
    return derived_observable(lambda x, **kwargs: anp.linalg.pinv(x), obs)


def svd(obs, **kwargs):
    """Computes the singular value decomposition of a matrix of Obs."""
    if kwargs.get('num_grad') is True:
        return _num_diff_svd(obs, **kwargs)
    u = derived_observable(lambda x, **kwargs: anp.linalg.svd(x, full_matrices=False)[0], obs)
    s = derived_observable(lambda x, **kwargs: anp.linalg.svd(x, full_matrices=False)[1], obs)
    vh = derived_observable(lambda x, **kwargs: anp.linalg.svd(x, full_matrices=False)[2], obs)
    return (u, s, vh)


def slog_det(obs, **kwargs):
    """Computes the determinant of a matrix of Obs via np.linalg.slogdet."""
    def _mat(x):
        dim = int(np.sqrt(len(x)))
        if np.sqrt(len(x)) != dim:
            raise Exception('Input has to have dim**2 entries')

        mat = []
        for i in range(dim):
            row = []
            for j in range(dim):
                row.append(x[j + dim * i])
            mat.append(row)

        (sign, logdet) = anp.linalg.slogdet(np.array(mat))
        return sign * anp.exp(logdet)

    if isinstance(obs, np.ndarray):
        return derived_observable(_mat, (1 * (obs.ravel())).tolist(), **kwargs)
    elif isinstance(obs, list):
        return derived_observable(_mat, obs, **kwargs)
    else:
        raise TypeError('Unproper type of input.')


# Variants for numerical differentiation

def _num_diff_mat_mat_op(op, obs, **kwargs):
    """Computes the matrix to matrix operation op to a given matrix of Obs elementwise
       which is suitable for numerical differentiation."""
    def _mat(x, **kwargs):
        dim = int(np.sqrt(len(x)))
        if np.sqrt(len(x)) != dim:
            raise Exception('Input has to have dim**2 entries')

        mat = []
        for i in range(dim):
            row = []
            for j in range(dim):
                row.append(x[j + dim * i])
            mat.append(row)

        return op(np.array(mat))[kwargs.get('i')][kwargs.get('j')]

    if isinstance(obs, np.ndarray):
        raveled_obs = (1 * (obs.ravel())).tolist()
    elif isinstance(obs, list):
        raveled_obs = obs
    else:
        raise TypeError('Unproper type of input.')

    dim = int(np.sqrt(len(raveled_obs)))

    res_mat = []
    for i in range(dim):
        row = []
        for j in range(dim):
            row.append(derived_observable(_mat, raveled_obs, i=i, j=j, **kwargs))
        res_mat.append(row)

    return np.array(res_mat) @ np.identity(dim)


def _num_diff_eigh(obs, **kwargs):
    """Computes the eigenvalues and eigenvectors of a given hermitian matrix of Obs according to np.linalg.eigh
       elementwise which is suitable for numerical differentiation."""
    def _mat(x, **kwargs):
        dim = int(np.sqrt(len(x)))
        if np.sqrt(len(x)) != dim:
            raise Exception('Input has to have dim**2 entries')

        mat = []
        for i in range(dim):
            row = []
            for j in range(dim):
                row.append(x[j + dim * i])
            mat.append(row)

        n = kwargs.get('n')
        res = np.linalg.eigh(np.array(mat))[n]

        if n == 0:
            return res[kwargs.get('i')]
        else:
            return res[kwargs.get('i')][kwargs.get('j')]

    if isinstance(obs, np.ndarray):
        raveled_obs = (1 * (obs.ravel())).tolist()
    elif isinstance(obs, list):
        raveled_obs = obs
    else:
        raise TypeError('Unproper type of input.')

    dim = int(np.sqrt(len(raveled_obs)))

    res_vec = []
    for i in range(dim):
        res_vec.append(derived_observable(_mat, raveled_obs, n=0, i=i, **kwargs))

    res_mat = []
    for i in range(dim):
        row = []
        for j in range(dim):
            row.append(derived_observable(_mat, raveled_obs, n=1, i=i, j=j, **kwargs))
        res_mat.append(row)

    return (np.array(res_vec) @ np.identity(dim), np.array(res_mat) @ np.identity(dim))


def _num_diff_eig(obs, **kwargs):
    """Computes the eigenvalues of a given matrix of Obs according to np.linalg.eig
       elementwise which is suitable for numerical differentiation."""
    def _mat(x, **kwargs):
        dim = int(np.sqrt(len(x)))
        if np.sqrt(len(x)) != dim:
            raise Exception('Input has to have dim**2 entries')

        mat = []
        for i in range(dim):
            row = []
            for j in range(dim):
                row.append(x[j + dim * i])
            mat.append(row)

        n = kwargs.get('n')
        res = np.linalg.eig(np.array(mat))[n]

        if n == 0:
            # Discard imaginary part of eigenvalue here
            return np.real(res[kwargs.get('i')])
        else:
            return res[kwargs.get('i')][kwargs.get('j')]

    if isinstance(obs, np.ndarray):
        raveled_obs = (1 * (obs.ravel())).tolist()
    elif isinstance(obs, list):
        raveled_obs = obs
    else:
        raise TypeError('Unproper type of input.')

    dim = int(np.sqrt(len(raveled_obs)))

    res_vec = []
    for i in range(dim):
        # Note: Automatic differentiation of eig is implemented in the git of autograd
        # but not yet released to PyPi (1.3)
        res_vec.append(derived_observable(_mat, raveled_obs, n=0, i=i, **kwargs))

    return np.array(res_vec) @ np.identity(dim)


def _num_diff_pinv(obs, **kwargs):
    """Computes the Moore-Penrose pseudoinverse of a matrix of Obs elementwise which is suitable
       for numerical differentiation."""
    def _mat(x, **kwargs):
        shape = kwargs.get('shape')

        mat = []
        for i in range(shape[0]):
            row = []
            for j in range(shape[1]):
                row.append(x[j + shape[1] * i])
            mat.append(row)

        return np.linalg.pinv(np.array(mat))[kwargs.get('i')][kwargs.get('j')]

    if isinstance(obs, np.ndarray):
        shape = obs.shape
        raveled_obs = (1 * (obs.ravel())).tolist()
    else:
        raise TypeError('Unproper type of input.')

    res_mat = []
    for i in range(shape[1]):
        row = []
        for j in range(shape[0]):
            row.append(derived_observable(_mat, raveled_obs, shape=shape, i=i, j=j, **kwargs))
        res_mat.append(row)

    return np.array(res_mat) @ np.identity(shape[0])


def _num_diff_svd(obs, **kwargs):
    """Computes the singular value decomposition of a matrix of Obs elementwise which
       is suitable for numerical differentiation."""
    def _mat(x, **kwargs):
        shape = kwargs.get('shape')

        mat = []
        for i in range(shape[0]):
            row = []
            for j in range(shape[1]):
                row.append(x[j + shape[1] * i])
            mat.append(row)

        res = np.linalg.svd(np.array(mat), full_matrices=False)

        if kwargs.get('n') == 1:
            return res[1][kwargs.get('i')]
        else:
            return res[kwargs.get('n')][kwargs.get('i')][kwargs.get('j')]

    if isinstance(obs, np.ndarray):
        shape = obs.shape
        raveled_obs = (1 * (obs.ravel())).tolist()
    else:
        raise TypeError('Unproper type of input.')

    mid_index = min(shape[0], shape[1])

    res_mat0 = []
    for i in range(shape[0]):
        row = []
        for j in range(mid_index):
            row.append(derived_observable(_mat, raveled_obs, shape=shape, n=0, i=i, j=j, **kwargs))
        res_mat0.append(row)

    res_mat1 = []
    for i in range(mid_index):
        res_mat1.append(derived_observable(_mat, raveled_obs, shape=shape, n=1, i=i, **kwargs))

    res_mat2 = []
    for i in range(mid_index):
        row = []
        for j in range(shape[1]):
            row.append(derived_observable(_mat, raveled_obs, shape=shape, n=2, i=i, j=j, **kwargs))
        res_mat2.append(row)

    return (np.array(res_mat0) @ np.identity(mid_index), np.array(res_mat1) @ np.identity(mid_index), np.array(res_mat2) @ np.identity(shape[1]))
