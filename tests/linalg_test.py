import numpy as np
import autograd.numpy as anp
import math
import pyerrors as pe
import pytest

np.random.seed(0)


def get_real_matrix(dimension):
    base_matrix = np.empty((dimension, dimension), dtype=object)
    for (n, m), entry in np.ndenumerate(base_matrix):
        exponent_real = np.random.normal(0, 1)
        exponent_imag = np.random.normal(0, 1)
        base_matrix[n, m] = pe.Obs([np.random.normal(1.0, 0.1, 100)], ['t'])


    return base_matrix

def get_complex_matrix(dimension):
    base_matrix = np.empty((dimension, dimension), dtype=object)
    for (n, m), entry in np.ndenumerate(base_matrix):
        exponent_real = np.random.normal(0, 1)
        exponent_imag = np.random.normal(0, 1)
        base_matrix[n, m] = pe.CObs(pe.Obs([np.random.normal(1.0, 0.1, 100)], ['t']),
                                    pe.Obs([np.random.normal(1.0, 0.1, 100)], ['t']))

    return base_matrix


def test_matmul():
    for dim in [4, 6]:
        for const in [1, pe.cov_Obs([1.0, 1.0], [[0.001,0.0001], [0.0001, 0.002]], 'norm')[1]]:
            my_list = []
            length = 100 + np.random.randint(200)
            for i in range(dim ** 2):
                my_list.append(pe.Obs([np.random.rand(length), np.random.rand(length + 1)], ['t1', 't2']))
            my_array = const * np.array(my_list).reshape((dim, dim))
            tt = pe.linalg.matmul(my_array, my_array) - my_array @ my_array
            for t, e in np.ndenumerate(tt):
                assert e.is_zero(), t

            my_list = []
            length = 100 + np.random.randint(200)
            for i in range(dim ** 2):
                my_list.append(pe.CObs(pe.Obs([np.random.rand(length), np.random.rand(length + 1)], ['t1', 't2']),
                                       pe.Obs([np.random.rand(length), np.random.rand(length + 1)], ['t1', 't2'])))
            my_array = np.array(my_list).reshape((dim, dim)) * const
            tt = pe.linalg.matmul(my_array, my_array) - my_array @ my_array
            for t, e in np.ndenumerate(tt):
                assert e.is_zero(), t


def test_jack_matmul():
    tt = get_real_matrix(8)
    check1 = pe.linalg.jack_matmul(tt, tt) - pe.linalg.matmul(tt, tt)
    [o.gamma_method() for o in check1.ravel()]
    assert np.all([o.is_zero_within_error(0.1) for o in check1.ravel()])
    assert np.all([o.dvalue < 0.001 for o in check1.ravel()])
    trace1 = np.trace(check1)
    trace1.gamma_method()
    assert trace1.dvalue < 0.001

    tr = np.random.rand(8, 8)
    check2 = pe.linalg.jack_matmul(tt, tr) - pe.linalg.matmul(tt, tr)
    [o.gamma_method() for o in check2.ravel()]
    assert np.all([o.is_zero_within_error(0.1) for o in check2.ravel()])
    assert np.all([o.dvalue < 0.001 for o in check2.ravel()])
    trace2 = np.trace(check2)
    trace2.gamma_method()
    assert trace2.dvalue < 0.001

    tt2 = get_complex_matrix(8)
    check3 = pe.linalg.jack_matmul(tt2, tt2) - pe.linalg.matmul(tt2, tt2)
    [o.gamma_method() for o in check3.ravel()]
    assert np.all([o.real.is_zero_within_error(0.1) for o in check3.ravel()])
    assert np.all([o.imag.is_zero_within_error(0.1) for o in check3.ravel()])
    assert np.all([o.real.dvalue < 0.001 for o in check3.ravel()])
    assert np.all([o.imag.dvalue < 0.001 for o in check3.ravel()])
    trace3 = np.trace(check3)
    trace3.gamma_method()
    assert trace3.real.dvalue < 0.001
    assert trace3.imag.dvalue < 0.001

    tr2 = np.random.rand(8, 8) + 1j * np.random.rand(8, 8)
    check4 = pe.linalg.jack_matmul(tt2, tr2) - pe.linalg.matmul(tt2, tr2)
    [o.gamma_method() for o in check4.ravel()]
    assert np.all([o.real.is_zero_within_error(0.1) for o in check4.ravel()])
    assert np.all([o.imag.is_zero_within_error(0.1) for o in check4.ravel()])
    assert np.all([o.real.dvalue < 0.001 for o in check4.ravel()])
    assert np.all([o.imag.dvalue < 0.001 for o in check4.ravel()])
    trace4 = np.trace(check4)
    trace4.gamma_method()
    assert trace4.real.dvalue < 0.001
    assert trace4.imag.dvalue < 0.001


def test_einsum():

    def _perform_real_check(arr):
        [o.gamma_method() for o in arr]
        assert np.all([o.is_zero_within_error(0.001) for o in arr])
        assert np.all([o.dvalue < 0.001 for o in arr])

    def _perform_complex_check(arr):
        [o.gamma_method() for o in arr]
        assert np.all([o.real.is_zero_within_error(0.001) for o in arr])
        assert np.all([o.real.dvalue < 0.001 for o in arr])
        assert np.all([o.imag.is_zero_within_error(0.001) for o in arr])
        assert np.all([o.imag.dvalue < 0.001 for o in arr])


    tt = [get_real_matrix(4), get_real_matrix(3)]
    q = np.tensordot(tt[0], tt[1], 0)
    c1 = tt[1] @ q
    c2 = pe.linalg.einsum('ij,abjd->abid', tt[1], q)
    check1 = c1 - c2
    _perform_real_check(check1.ravel())
    check2 = np.trace(tt[0]) - pe.linalg.einsum('ii', tt[0])
    _perform_real_check([check2])
    check3 = np.trace(tt[1]) - pe.linalg.einsum('ii', tt[1])
    _perform_real_check([check3])

    tt = [get_real_matrix(4), np.random.random((3, 3))]
    q = np.tensordot(tt[0], tt[1], 0)
    c1 = tt[1] @ q
    c2 = pe.linalg.einsum('ij,abjd->abid', tt[1], q)
    check1 = c1 - c2
    _perform_real_check(check1.ravel())

    tt = [get_complex_matrix(4), get_complex_matrix(3)]
    q = np.tensordot(tt[0], tt[1], 0)
    c1 = tt[1] @ q
    c2 = pe.linalg.einsum('ij,abjd->abid', tt[1], q)
    check1 = c1 - c2
    _perform_complex_check(check1.ravel())
    check2 = np.trace(tt[0]) - pe.linalg.einsum('ii', tt[0])
    _perform_complex_check([check2])
    check3 = np.trace(tt[1]) - pe.linalg.einsum('ii', tt[1])
    _perform_complex_check([check3])

    tt = [get_complex_matrix(4), np.random.random((3, 3))]
    q = np.tensordot(tt[0], tt[1], 0)
    c1 = tt[1] @ q
    c2 = pe.linalg.einsum('ij,abjd->abid', tt[1], q)
    check1 = c1 - c2
    _perform_complex_check(check1.ravel())


def test_multi_dot():
    for dim in [4, 6]:
        my_list = []
        length = 1000 + np.random.randint(200)
        for i in range(dim ** 2):
            my_list.append(pe.Obs([np.random.rand(length), np.random.rand(length + 1)], ['t1', 't2']))
        my_array = pe.cov_Obs(1.0, 0.002, 'cov') * np.array(my_list).reshape((dim, dim))
        tt = pe.linalg.matmul(my_array, my_array, my_array, my_array) - my_array @ my_array @ my_array @ my_array
        for t, e in np.ndenumerate(tt):
            assert e.is_zero(), t

        my_list = []
        length = 1000 + np.random.randint(200)
        for i in range(dim ** 2):
            my_list.append(pe.CObs(pe.Obs([np.random.rand(length), np.random.rand(length + 1)], ['t1', 't2']),
                                   pe.Obs([np.random.rand(length), np.random.rand(length + 1)], ['t1', 't2'])))
        my_array = np.array(my_list).reshape((dim, dim)) * pe.cov_Obs(1.0, 0.002, 'cov')
        tt = pe.linalg.matmul(my_array, my_array, my_array, my_array) - my_array @ my_array @ my_array @ my_array
        for t, e in np.ndenumerate(tt):
            assert e.is_zero(), t


def test_jack_multi_dot():
    for dim in [2, 4, 8]:
        my_array = get_real_matrix(dim)

        tt = pe.linalg.jack_matmul(my_array, my_array, my_array) - pe.linalg.matmul(my_array, my_array, my_array)

        for t, e in np.ndenumerate(tt):
            e.gamma_method()
            assert e.is_zero_within_error(0.01)
            assert e.is_zero(atol=1e-1), t
            assert np.isclose(e.value, 0.0)


def test_matmul_irregular_histories():
    dim = 2
    length = 500

    standard_array = []
    for i in range(dim ** 2):
        standard_array.append(pe.Obs([np.random.normal(1.1, 0.2, length)], ['ens1']))
    standard_matrix = np.array(standard_array).reshape((dim, dim)) * pe.cov_Obs(1.0, 0.002, 'cov') * pe.pseudo_Obs(0.1, 0.002, 'qr')

    for idl in [range(1, 501, 2), range(250, 273), [2, 8, 19, 20, 78]]:
        irregular_array = []
        for i in range(dim ** 2):
            irregular_array.append(pe.Obs([np.random.normal(1.1, 0.2, len(idl))], ['ens1'], idl=[idl]))
        irregular_matrix = np.array(irregular_array).reshape((dim, dim)) * pe.cov_Obs([1.0, 1.0], [[0.001,0.0001], [0.0001, 0.002]], 'norm')[0]

        t1 = standard_matrix @ irregular_matrix
        t2 = pe.linalg.matmul(standard_matrix, irregular_matrix)

        assert np.all([o.is_zero() for o in (t1 - t2).ravel()])
        assert np.all([o.is_merged for o in t1.ravel()])
        assert np.all([o.is_merged for o in t2.ravel()])


def test_irregular_matrix_inverse():
    dim = 3
    length = 500

    for idl in [range(8, 508, 10), range(250, 273), [2, 8, 19, 20, 78, 99, 828, 10548979]]:
        irregular_array = []
        for i in range(dim ** 2):
            irregular_array.append(pe.Obs([np.random.normal(1.1, 0.2, len(idl)), np.random.normal(0.25, 0.1, 10)], ['ens1', 'ens2'], idl=[idl, range(1, 11)]))
        irregular_matrix = np.array(irregular_array).reshape((dim, dim)) * pe.cov_Obs(1.0, 0.002, 'cov') * pe.pseudo_Obs(1.0, 0.002, 'ens2|r23')

        invertible_irregular_matrix = np.identity(dim) + irregular_matrix @ irregular_matrix.T

        inverse = pe.linalg.inv(invertible_irregular_matrix)

        assert np.allclose(np.linalg.inv(np.vectorize(lambda x: x.value)(invertible_irregular_matrix)) - np.vectorize(lambda x: x.value)(inverse), 0.0)

        check1 = pe.linalg.matmul(invertible_irregular_matrix, inverse)
        assert np.all([o.is_zero() for o in (check1 - np.identity(dim)).ravel()])
        check2 = invertible_irregular_matrix @ inverse
        assert np.all([o.is_zero() for o in (check2 - np.identity(dim)).ravel()])


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
        exponent_real = np.random.normal(2, 3)
        exponent_imag = np.random.normal(2, 3)
        base_matrix[n, m] = pe.CObs(pe.pseudo_Obs(2 + 10 ** exponent_real, 10 ** (exponent_real - 1), 't'),
                                    pe.pseudo_Obs(2 + 10 ** exponent_imag, 10 ** (exponent_imag - 1), 't'))

    # Construct invertible matrix
    obs_matrix = np.identity(dimension) + base_matrix @ base_matrix.T

    for (n, m), entry in np.ndenumerate(obs_matrix):
        matrix[n, m] = entry.real.value + 1j * entry.imag.value

    inverse_matrix = np.linalg.inv(matrix)
    inverse_obs_matrix = pe.linalg.inv(obs_matrix)
    for (n, m), entry in np.ndenumerate(inverse_matrix):
        assert np.isclose(inverse_matrix[n, m].real, inverse_obs_matrix[n, m].real.value)
        assert np.isclose(inverse_matrix[n, m].imag, inverse_obs_matrix[n, m].imag.value)


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

    # Check eig function
    e2 = pe.linalg.eig(sym)
    assert np.all(np.sort(e) == np.sort(e2))

    # Check svd
    u, v, vh = pe.linalg.svd(sym)
    diff = sym - u @ np.diag(v) @ vh

    for (i, j), entry in np.ndenumerate(diff):
        assert entry.is_zero()

    # Check determinant
    assert pe.linalg.det(np.diag(np.diag(matrix))) == np.prod(np.diag(matrix))

    with pytest.raises(Exception):
        pe.linalg.det(5)

    pe.linalg.pinv(matrix[:,:3])


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


def test_complex_matrix_real_entries():
    my_mat = get_complex_matrix(4)
    my_mat[0, 1] = 4
    my_mat[2, 0] = pe.Obs([np.random.normal(1.0, 0.1, 100)], ['t'])
    assert np.all((my_mat @ pe.linalg.inv(my_mat) - np.identity(4)) == 0)
