import numpy as np
import autograd.numpy as anp
import math
import pyerrors as pe
import pytest

np.random.seed(0)


def test_matmul():
    for dim in [4, 8]:
        my_list = []
        length = 1000 + np.random.randint(200)
        for i in range(dim ** 2):
            my_list.append(pe.Obs([np.random.rand(length), np.random.rand(length + 1)], ['t1', 't2']))
        my_array = np.array(my_list).reshape((dim, dim))
        tt = pe.linalg.matmul(my_array, my_array) - my_array @ my_array
        for t, e in np.ndenumerate(tt):
            assert e.is_zero(), t

        my_list = []
        length = 1000 + np.random.randint(200)
        for i in range(dim ** 2):
            my_list.append(pe.CObs(pe.Obs([np.random.rand(length), np.random.rand(length + 1)], ['t1', 't2']),
                                   pe.Obs([np.random.rand(length), np.random.rand(length + 1)], ['t1', 't2'])))
        my_array = np.array(my_list).reshape((dim, dim))
        tt = pe.linalg.matmul(my_array, my_array) - my_array @ my_array
        for t, e in np.ndenumerate(tt):
            assert e.is_zero(), t


def test_multi_dot():
    for dim in [4, 8]:
        my_list = []
        length = 1000 + np.random.randint(200)
        for i in range(dim ** 2):
            my_list.append(pe.Obs([np.random.rand(length), np.random.rand(length + 1)], ['t1', 't2']))
        my_array = np.array(my_list).reshape((dim, dim))
        tt = pe.linalg.matmul(my_array, my_array, my_array, my_array) - my_array @ my_array @ my_array @ my_array
        for t, e in np.ndenumerate(tt):
            assert e.is_zero(), t

        my_list = []
        length = 1000 + np.random.randint(200)
        for i in range(dim ** 2):
            my_list.append(pe.CObs(pe.Obs([np.random.rand(length), np.random.rand(length + 1)], ['t1', 't2']),
                                   pe.Obs([np.random.rand(length), np.random.rand(length + 1)], ['t1', 't2'])))
        my_array = np.array(my_list).reshape((dim, dim))
        tt = pe.linalg.matmul(my_array, my_array, my_array, my_array) - my_array @ my_array @ my_array @ my_array
        for t, e in np.ndenumerate(tt):
            assert e.is_zero(), t


def test_matmul_irregular_histories():
    dim = 2
    length = 500

    standard_array = []
    for i in range(dim ** 2):
        standard_array.append(pe.Obs([np.random.normal(1.1, 0.2, length)], ['ens1']))
    standard_matrix = np.array(standard_array).reshape((dim, dim))

    for idl in [range(1, 501, 2), range(250, 273), [2, 8, 19, 20, 78]]:
        irregular_array = []
        for i in range(dim ** 2):
            irregular_array.append(pe.Obs([np.random.normal(1.1, 0.2, len(idl))], ['ens1'], idl=[idl]))
        irregular_matrix = np.array(irregular_array).reshape((dim, dim))

        t1 = standard_matrix @ irregular_matrix
        t2 = pe.linalg.matmul(standard_matrix, irregular_matrix)

        assert np.all([o.is_zero() for o in (t1 - t2).ravel()])
        assert np.all([o.is_merged for o in t1.ravel()])
        assert np.all([o.is_merged for o in t2.ravel()])


def test_matrix_inverse():
    content = []
    for t in range(9):
        exponent = np.random.normal(3, 5)
        content.append(pe.pseudo_Obs(2 + 10 ** exponent, 10 ** (exponent - 1), 't'))

    content.append(1.0) # Add 1.0 as a float
    matrix = np.diag(content)
    inverse_matrix = pe.linalg.inv(matrix)
    assert all([o.is_zero() for o in np.diag(matrix) * np.diag(inverse_matrix) - 1])


def test_complex_matrix_inverse():
    dimension = 4
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
    inverse_obs_matrix = pe.linalg.inv(obs_matrix)
    for (n, m), entry in np.ndenumerate(inverse_matrix):
        assert np.isclose(inverse_matrix[n, m].real,  inverse_obs_matrix[n, m].real.value)
        assert np.isclose(inverse_matrix[n, m].imag,  inverse_obs_matrix[n, m].imag.value)


def test_matrix_functions():
    dim = 4
    matrix = []
    for i in range(dim):
        row = []
        for j in range(dim):
            row.append(pe.pseudo_Obs(np.random.rand(), 0.2 + 0.1 * np.random.rand(), 'e1'))
        matrix.append(row)
    matrix = np.array(matrix) @ np.identity(dim)

    # Check inverse of matrix
    inv = pe.linalg.inv(matrix)
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
    cholesky = pe.linalg.cholesky(sym)
    check = cholesky @ cholesky.T

    for (i, j), entry in np.ndenumerate(check):
        diff = entry - sym[i, j]
        assert diff.is_zero()

    # Check eigh
    e, v = pe.linalg.eigh(sym)
    for i in range(dim):
        tmp = sym @ v[:, i] - v[:, i] * e[i]
        for j in range(dim):
            assert tmp[j].is_zero()

    # Check svd
    u, v, vh = pe.linalg.svd(sym)
    diff = sym - u @ np.diag(v) @ vh

    for (i, j), entry in np.ndenumerate(diff):
        assert entry.is_zero()


def test_complex_matrix_operations():
    dimension = 4
    base_matrix = np.empty((dimension, dimension), dtype=object)
    for (n, m), entry in np.ndenumerate(base_matrix):
        exponent_real = np.random.normal(3, 5)
        exponent_imag = np.random.normal(3, 5)
        base_matrix[n, m] = pe.CObs(pe.pseudo_Obs(2 + 10 ** exponent_real, 10 ** (exponent_real - 1), 't'),
                                    pe.pseudo_Obs(2 + 10 ** exponent_imag, 10 ** (exponent_imag - 1), 't'))

    for other in [2, 2.3, (1 - 0.1j), (0 + 2.1j)]:
        ta = base_matrix * other
        tb = other * base_matrix
        diff = ta - tb
        for (i, j), entry in np.ndenumerate(diff):
            assert entry.is_zero()
        ta = base_matrix + other
        tb = other + base_matrix
        diff = ta - tb
        for (i, j), entry in np.ndenumerate(diff):
            assert entry.is_zero()
        ta = base_matrix - other
        tb = other - base_matrix
        diff = ta + tb
        for (i, j), entry in np.ndenumerate(diff):
            assert entry.is_zero()
        ta = base_matrix / other
        tb = other / base_matrix
        diff = ta * tb - 1
        for (i, j), entry in np.ndenumerate(diff):
            assert entry.is_zero()
