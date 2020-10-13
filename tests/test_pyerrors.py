import sys
sys.path.append('..')
import autograd.numpy as np
import os
import random
import math
import string
import copy
import scipy.optimize
from scipy.odr import ODR, Model, Data, RealData
import pyerrors as pe
import pytest

test_iterations = 100

def test_dump():
    value = np.random.normal(5, 10)
    dvalue = np.abs(np.random.normal(0, 1))
    test_obs = pe.pseudo_Obs(value, dvalue, 't')
    test_obs.dump('test_dump')
    new_obs = pe.load_object('test_dump.p')
    os.remove('test_dump.p')
    assert test_obs.deltas['t'].all() == new_obs.deltas['t'].all()


def test_comparison():
    value1 = np.random.normal(0, 100)
    test_obs1 = pe.pseudo_Obs(value1, 0.1, 't')
    value2 = np.random.normal(0, 100)
    test_obs2 = pe.pseudo_Obs(value2, 0.1, 't')
    assert (value1 > value2) == (test_obs1 > test_obs2)
    assert (value1 < value2) == (test_obs1 < test_obs2)


def test_man_grad():
    a = pe.pseudo_Obs(17,2.9,'e1')
    b = pe.pseudo_Obs(4,0.8,'e1')

    fs = [lambda x: x[0] + x[1], lambda x: x[1] + x[0], lambda x: x[0] - x[1], lambda x: x[1] - x[0],
          lambda x: x[0] * x[1], lambda x: x[1] * x[0], lambda x: x[0] / x[1], lambda x: x[1] / x[0],
          lambda x: np.exp(x[0]), lambda x: np.sin(x[0]), lambda x: np.cos(x[0]), lambda x: np.tan(x[0]),
          lambda x: np.log(x[0]), lambda x: np.sqrt(x[0]),
          lambda x: np.sinh(x[0]), lambda x: np.cosh(x[0]), lambda x: np.tanh(x[0])]

    for i, f in enumerate(fs):
        t1 = f([a,b])
        t2 = pe.derived_observable(f, [a,b])
        c = t2 - t1
        assert c.value == 0.0, str(i)
        assert np.all(np.abs(c.deltas['e1']) < 1e-14), str(i)


def test_overloading_vectorization():
    a = np.array([5, 4, 8])
    b = pe.pseudo_Obs(4,0.8,'e1')

    assert [o.value for o in a * b] == [o.value for o in b * a]
    assert [o.value for o in a + b] == [o.value for o in b + a]
    assert [o.value for o in a - b] == [-1 * o.value for o in b - a]
    assert [o.value for o in a / b] == [o.value for o in [p / b for p in a]]
    assert [o.value for o in b / a] == [o.value for o in [b / p for p in a]]


@pytest.mark.parametrize("n", np.arange(test_iterations // 10))
def test_covariance_is_variance(n):
    value = np.random.normal(5, 10)
    dvalue = np.abs(np.random.normal(0, 1))
    test_obs = pe.pseudo_Obs(value, dvalue, 't')
    test_obs.gamma_method()
    assert np.abs(test_obs.dvalue ** 2 - pe.covariance(test_obs, test_obs)) <= 10 * np.finfo(np.float).eps
    test_obs = test_obs + pe.pseudo_Obs(value, dvalue, 'q', 200)
    test_obs.gamma_method(e_tag=0)
    assert np.abs(test_obs.dvalue ** 2 - pe.covariance(test_obs, test_obs)) <= 10 * np.finfo(np.float).eps


@pytest.mark.parametrize("n", np.arange(test_iterations // 10))
def test_fft(n):
    value = np.random.normal(5, 100)
    dvalue = np.abs(np.random.normal(0, 5))
    test_obs1 = pe.pseudo_Obs(value, dvalue, 't', int(500 + 1000 * np.random.rand()))
    test_obs2 = copy.deepcopy(test_obs1)
    test_obs1.gamma_method()
    test_obs2.gamma_method(fft=False)
    assert max(np.abs(test_obs1.e_rho[''] - test_obs2.e_rho[''])) <= 10 * np.finfo(np.float).eps
    assert np.abs(test_obs1.dvalue - test_obs2.dvalue) <= 10 * max(test_obs1.dvalue, test_obs2.dvalue) * np.finfo(np.float).eps


@pytest.mark.parametrize('n', np.arange(test_iterations // 10))
def test_standard_fit(n):
    dim = 10 + int(30 * np.random.rand())
    x = np.arange(dim)
    y = 2 * np.exp(-0.06 * x) + np.random.normal(0.0, 0.15, dim)
    yerr = 0.1 + 0.1 * np.random.rand(dim)

    oy = []
    for i, item in enumerate(x):
        oy.append(pe.pseudo_Obs(y[i], yerr[i], str(i)))

    def f(x, a, b):
        return a * np.exp(-b * x)

    popt, pcov = scipy.optimize.curve_fit(f, x, y, sigma=[o.dvalue for o in oy], absolute_sigma=True)

    def func(a, x):
        y = a[0] * np.exp(-a[1] * x)
        return y

    beta = pe.fits.standard_fit(x, oy, func)

    pe.Obs.e_tag_global = 5
    for i in range(2):
        beta[i].gamma_method(e_tag=5, S=1.0)
        assert math.isclose(beta[i].value, popt[i], abs_tol=1e-5)
        assert math.isclose(pcov[i, i], beta[i].dvalue ** 2, abs_tol=1e-3)
    assert math.isclose(pe.covariance(beta[0], beta[1]), pcov[0, 1], abs_tol=1e-3)
    pe.Obs.e_tag_global = 0

    chi2_pyerrors = np.sum(((f(x, *[o.value for o in beta]) - y) / yerr) ** 2) / (len(x) - 2)
    chi2_scipy = np.sum(((f(x, *popt) - y) / yerr) ** 2) / (len(x) - 2)
    assert math.isclose(chi2_pyerrors, chi2_scipy, abs_tol=1e-10)


@pytest.mark.parametrize('n', np.arange(test_iterations // 10))
def test_odr_fit(n):
    dim = 10 + int(30 * np.random.rand())
    x = np.arange(dim) + np.random.normal(0.0, 0.15, dim)
    xerr = 0.1 + 0.1 * np.random.rand(dim)
    y = 2 * np.exp(-0.06 * x) + np.random.normal(0.0, 0.15, dim)
    yerr = 0.1 + 0.1 * np.random.rand(dim)

    ox = []
    for i, item in enumerate(x):
        ox.append(pe.pseudo_Obs(x[i], xerr[i], str(i)))

    oy = []
    for i, item in enumerate(x):
        oy.append(pe.pseudo_Obs(y[i], yerr[i], str(i)))

    def f(x, a, b):
        return a * np.exp(-b * x)

    def func(a, x):
        y = a[0] * np.exp(-a[1] * x)
        return y

    data = RealData([o.value for o in ox], [o.value for o in oy], sx=[o.dvalue for o in ox], sy=[o.dvalue for o in oy])
    model = Model(func)
    odr = ODR(data, model, [0,0], partol=np.finfo(np.float).eps)
    odr.set_job(fit_type=0, deriv=1)
    output = odr.run()

    beta = pe.fits.odr_fit(ox, oy, func)

    pe.Obs.e_tag_global = 5
    for i in range(2):
        beta[i].gamma_method(e_tag=5, S=1.0)
        assert math.isclose(beta[i].value, output.beta[i], rel_tol=1e-5)
        assert math.isclose(output.cov_beta[i,i], beta[i].dvalue**2, rel_tol=2.5e-1), str(output.cov_beta[i,i]) + ' ' + str(beta[i].dvalue**2)
    assert math.isclose(pe.covariance(beta[0], beta[1]), output.cov_beta[0,1], rel_tol=2.5e-1)
    pe.Obs.e_tag_global = 0


@pytest.mark.parametrize('n', np.arange(test_iterations // 10))
def test_odr_derivatives(n):
    x = []
    y = []
    x_err = 0.01
    y_err = 0.01

    for n in np.arange(1, 9, 2):
        loc_xvalue = n + np.random.normal(0.0, x_err)
        x.append(pe.pseudo_Obs(loc_xvalue, x_err, str(n)))
        y.append(pe.pseudo_Obs((lambda x: x ** 2 - 1)(loc_xvalue) +
        np.random.normal(0.0, y_err), y_err, str(n)))

        def func(a, x):
            return a[0] + a[1] * x ** 2

    fit1 = pe.fits.odr_fit(x, y, func)

    tfit = pe.fits.fit_general(x, y, func, base_step=0.1, step_ratio=1.1, num_steps=20)
    assert np.abs(np.max(np.array(list(fit1[1].deltas.values()))
                       - np.array(list(tfit[1].deltas.values())))) < 10e-8


@pytest.mark.parametrize('n', np.arange(test_iterations))
def test_covariance_symmetry(n):
    value1 = np.random.normal(5, 10)
    dvalue1 = np.abs(np.random.normal(0, 1))
    test_obs1 = pe.pseudo_Obs(value1, dvalue1, 't')
    test_obs1.gamma_method()
    value2 = np.random.normal(5, 10)
    dvalue2 = np.abs(np.random.normal(0, 1))
    test_obs2 = pe.pseudo_Obs(value2, dvalue2, 't')
    test_obs2.gamma_method()
    cov_ab = pe.covariance(test_obs1, test_obs2)
    cov_ba = pe.covariance(test_obs2, test_obs1)
    assert np.abs(cov_ab - cov_ba) <= 10 * np.finfo(np.float).eps
    assert np.abs(cov_ab) < test_obs1.dvalue * test_obs2.dvalue * (1 + 10 * np.finfo(np.float).eps)


@pytest.mark.parametrize('n', np.arange(test_iterations))
def test_gamma_method(n):
    # Construct pseudo Obs with random shape
    value = np.random.normal(5, 10)
    dvalue = np.abs(np.random.normal(0, 1))

    test_obs = pe.pseudo_Obs(value, dvalue, 't', int(1000 * (1 + np.random.rand())))

    # Test if the error is processed correctly
    test_obs.gamma_method(e_tag=1)
    assert np.abs(test_obs.value - value) < 1e-12
    assert abs(test_obs.dvalue - dvalue) < 1e-10 * dvalue


@pytest.mark.parametrize('n', np.arange(test_iterations))
def test_overloading(n):
    # Construct pseudo Obs with random shape
    obs_list = []
    for i in range(5):
        value = np.abs(np.random.normal(5, 2)) + 2.0
        dvalue = np.abs(np.random.normal(0, 0.1)) + 1e-5
        obs_list.append(pe.pseudo_Obs(value, dvalue, 't', 2000))

    # Test if the error is processed correctly
    def f(x):
        return x[0] * x[1] + np.sin(x[2]) * np.exp(x[3] / x[1] / x[0]) - np.sqrt(2) / np.cosh(x[4] / x[0])

    o_obs = f(obs_list)
    d_obs = pe.derived_observable(f, obs_list)

    assert np.max(np.abs((o_obs.deltas['t'] - d_obs.deltas['t']) / o_obs.deltas['t'])) < 1e-7, str(obs_list)
    assert np.abs((o_obs.value - d_obs.value) / o_obs.value) < 1e-10


@pytest.mark.parametrize('n', np.arange(test_iterations))
def test_derived_observables(n):
    # Construct pseudo Obs with random shape
    test_obs = pe.pseudo_Obs(2, 0.1 * (1 + np.random.rand()), 't', int(1000 * (1 + np.random.rand())))

    # Check if autograd and numgrad give the same result
    d_Obs_ad = pe.derived_observable(lambda x, **kwargs: x[0] * x[1] * np.sin(x[0] * x[1]), [test_obs, test_obs])
    d_Obs_ad.gamma_method()
    d_Obs_fd = pe.derived_observable(lambda x, **kwargs: x[0] * x[1] * np.sin(x[0] * x[1]), [test_obs, test_obs], num_grad=True)
    d_Obs_fd.gamma_method()

    assert d_Obs_ad.value == d_Obs_fd.value
    assert np.abs(4.0 * np.sin(4.0) - d_Obs_ad.value) < 1000 * np.finfo(np.float).eps * np.abs(d_Obs_ad.value)
    assert np.abs(d_Obs_ad.dvalue-d_Obs_fd.dvalue) < 1000 * np.finfo(np.float).eps * d_Obs_ad.dvalue

    i_am_one = pe.derived_observable(lambda x, **kwargs: x[0] / x[1], [d_Obs_ad, d_Obs_ad])
    i_am_one.gamma_method(e_tag=1)

    assert i_am_one.value == 1.0
    assert i_am_one.dvalue < 2 * np.finfo(np.float).eps
    assert i_am_one.e_dvalue['t'] <= 2 * np.finfo(np.float).eps
    assert i_am_one.e_ddvalue['t'] <= 2 * np.finfo(np.float).eps


@pytest.mark.parametrize('n', np.arange(test_iterations // 10))
def test_multi_ens_system(n):
    names = []
    for i in range(100 + int(np.random.rand() * 50)):
        tmp_string = ''
        for _ in range(int(2 + np.random.rand() * 4)):
            tmp_string += random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
        names.append(tmp_string)
    names = list(set(names))
    samples = [np.random.rand(5)] * len(names)
    new_obs = pe.Obs(samples, names)

    for e_tag_length in range(1, 6):
        new_obs.gamma_method(e_tag=e_tag_length)
        e_names = sorted(set([n[:e_tag_length] for n in names]))
        assert e_names == new_obs.e_names
        assert sorted(x for y in sorted(new_obs.e_content.values()) for x in y) == sorted(new_obs.names)


@pytest.mark.parametrize('n', np.arange(test_iterations))
def test_overloaded_functions(n):
    funcs = [np.exp, np.log, np.sin, np.cos, np.tan, np.sinh, np.cosh, np.arcsinh, np.arccosh]
    deriv = [np.exp, lambda x: 1 / x, np.cos, lambda x: -np.sin(x), lambda x: 1 / np.cos(x) ** 2, np.cosh, np.sinh, lambda x: 1 / np.sqrt(x ** 2 + 1), lambda x: 1 / np.sqrt(x ** 2 - 1)]
    val = 3 + 0.5 * np.random.rand()
    dval = 0.3 + 0.4 * np.random.rand()
    test_obs = pe.pseudo_Obs(val, dval, 't', int(1000 * (1 + np.random.rand())))

    for i, item in enumerate(funcs):
        ad_obs = item(test_obs)
        fd_obs = pe.derived_observable(lambda x, **kwargs: item(x[0]), [test_obs], num_grad=True)
        ad_obs.gamma_method(S=0.01, e_tag=1)
        assert np.max((ad_obs.deltas['t'] - fd_obs.deltas['t']) / ad_obs.deltas['t']) < 1e-8, item.__name__
        assert np.abs((ad_obs.value - item(val)) / ad_obs.value) < 1e-10, item.__name__
        assert np.abs(ad_obs.dvalue - dval * np.abs(deriv[i](val))) < 1e-6, item.__name__


@pytest.mark.parametrize('n', np.arange(test_iterations // 10))
def test_matrix_functions(n):
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
    inv = pe.linalg.mat_mat_op(np.linalg.inv, matrix)
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
    cholesky = pe.linalg.mat_mat_op(np.linalg.cholesky, sym)
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

