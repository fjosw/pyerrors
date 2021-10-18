import numpy as np
import jax.numpy as jnp
import math
import pyerrors as pe
import pytest
from jax.config import config
config.update("jax_enable_x64", True)

np.random.seed(0)


def test_matrix_inverse():
    content = []
    for t in range(9):
        exponent = np.random.normal(3, 5)
        content.append(pe.pseudo_Obs(2 + 10 ** exponent, 10 ** (exponent - 1), 't'))

    content.append(1.0) # Add 1.0 as a float
    matrix = np.diag(content)
    inverse_matrix = pe.linalg.mat_mat_op(jnp.linalg.inv, matrix)
    assert all([o.is_zero() for o in np.diag(matrix) * np.diag(inverse_matrix) - 1])


def test_complex_matrix_inverse():
    dimension = 6
    base_matrix = np.empty((dimension, dimension), dtype=object)
    matrix = np.empty((dimension, dimension), dtype=complex)
    for (n, m), entry in np.ndenumerate(base_matrix):
        exponent_real = np.random.normal(3, 5)
        exponent_imag = np.random.normal(3, 5)
        base_matrix[n, m] = pe.CObs(pe.pseudo_Obs(2 + 10 ** exponent_real, 10 ** (exponent_real - 1), 't'),
                                    pe.pseudo_Obs(2 + 10 ** exponent_imag, 10 ** (exponent_imag - 1), 't'))

    # Construct invertible matrix
    obs_matrix = np.identity(dimension) + base_matrix @ base_matrix.T

    for (n, m), entry in np.ndenumerate(obs_matrix):
        matrix[n, m] = entry.real.value + 1j * entry.imag.value

    inverse_matrix = np.linalg.inv(matrix)
    inverse_obs_matrix = pe.linalg.mat_mat_op(jnp.linalg.inv, obs_matrix)
    for (n, m), entry in np.ndenumerate(inverse_matrix):
        assert np.isclose(inverse_matrix[n, m].real,  inverse_obs_matrix[n, m].real.value)
        assert np.isclose(inverse_matrix[n, m].imag,  inverse_obs_matrix[n, m].imag.value)


def test_matrix_functions():
    dim = 3 + int(4 * np.random.rand())
    print(dim)
    matrix = []
    for i in range(dim):
        row = []
        for j in range(dim):
            row.append(pe.pseudo_Obs(np.random.rand(), 0.2 + 0.1 * np.random.rand(), 'e1'))
        matrix.append(row)
    matrix = np.array(matrix) @ np.identity(dim)

    # Check inverse of matrix
    inv = pe.linalg.mat_mat_op(jnp.linalg.inv, matrix)
    check_inv = matrix @ inv

    for (i, j), entry in np.ndenumerate(check_inv):
        entry.gamma_method()
        if(i == j):
            assert math.isclose(entry.value, 1.0, abs_tol=1e-9), 'value ' + str(i) + ',' + str(j) + ' ' + str(entry.value)
        else:
            assert math.isclose(entry.value, 0.0, abs_tol=1e-9), 'value ' + str(i) + ',' + str(j) + ' ' + str(entry.value)
        assert math.isclose(entry.dvalue, 0.0, abs_tol=1e-9), 'dvalue ' + str(i) + ',' + str(j) + ' ' + str(entry.dvalue)

    # Check Cholesky decomposition
    sym = np.dot(matrix, matrix.T)
    cholesky = pe.linalg.mat_mat_op(jnp.linalg.cholesky, sym)
    check = cholesky @ cholesky.T

    for (i, j), entry in np.ndenumerate(check):
        diff = entry - sym[i, j]
        diff.gamma_method()
        assert math.isclose(diff.value, 0.0, abs_tol=1e-9), 'value ' + str(i) + ',' + str(j)
        assert math.isclose(diff.dvalue, 0.0, abs_tol=1e-9), 'dvalue ' + str(i) + ',' + str(j)

    # Check eigh
    e, v = pe.linalg.eigh(sym)
    for i in range(dim):
        tmp = sym @ v[:, i] - v[:, i] * e[i]
        for j in range(dim):
            tmp[j].gamma_method()
            assert math.isclose(tmp[j].value, 0.0, abs_tol=1e-9), 'value ' + str(i) + ',' + str(j)
            assert math.isclose(tmp[j].dvalue, 0.0, abs_tol=1e-9), 'dvalue ' + str(i) + ',' + str(j)
