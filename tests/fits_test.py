import autograd.numpy as np
import math
import scipy.optimize
from scipy.odr import ODR, Model, RealData
from scipy.linalg import cholesky
from scipy.stats import norm
import pyerrors as pe
import pytest

np.random.seed(0)


def test_least_squares():
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

    out = pe.least_squares(x, oy, func, expected_chisquare=True, resplot=True, qqplot=True)
    beta = out.fit_parameters

    for i in range(2):
        beta[i].gamma_method(S=1.0)
        assert math.isclose(beta[i].value, popt[i], abs_tol=1e-5)
        assert math.isclose(pcov[i, i], beta[i].dvalue ** 2, abs_tol=1e-3)
    assert math.isclose(pe.covariance(beta[0], beta[1]), pcov[0, 1], abs_tol=1e-3)

    chi2_pyerrors = np.sum(((f(x, *[o.value for o in beta]) - y) / yerr) ** 2) / (len(x) - 2)
    chi2_scipy = np.sum(((f(x, *popt) - y) / yerr) ** 2) / (len(x) - 2)
    assert math.isclose(chi2_pyerrors, chi2_scipy, abs_tol=1e-10)

    out = pe.least_squares(x, oy, func, const_par=[beta[1]])
    assert((out.fit_parameters[0] - beta[0]).is_zero())
    assert((out.fit_parameters[1] - beta[1]).is_zero())

    oyc = []
    for i, item in enumerate(x):
        oyc.append(pe.cov_Obs(y[i], yerr[i]**2, 'cov' + str(i)))

    outc = pe.least_squares(x, oyc, func)
    betac = outc.fit_parameters

    for i in range(2):
        betac[i].gamma_method(S=1.0)
        assert math.isclose(betac[i].value, popt[i], abs_tol=1e-5)
        assert math.isclose(pcov[i, i], betac[i].dvalue ** 2, abs_tol=1e-3)
    assert math.isclose(pe.covariance(betac[0], betac[1]), pcov[0, 1], abs_tol=1e-3)

    num_samples = 400
    N = 10

    x = norm.rvs(size=(N, num_samples))

    r = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            r[i, j] = np.exp(-0.1 * np.fabs(i - j))

    errl = np.sqrt([3.4, 2.5, 3.6, 2.8, 4.2, 4.7, 4.9, 5.1, 3.2, 4.2])
    errl *= 4
    for i in range(N):
        for j in range(N):
            r[i, j] *= errl[i] * errl[j]

    c = cholesky(r, lower=True)
    y = np.dot(c, x)

    x = np.arange(N)
    for linear in [True, False]:
        data = []
        for i in range(N):
            if linear:
                data.append(pe.Obs([[i + 1 + o for o in y[i]]], ['ens']))
            else:
                data.append(pe.Obs([[np.exp(-(i + 1)) + np.exp(-(i + 1)) * o for o in y[i]]], ['ens']))

        [o.gamma_method() for o in data]

        if linear:
            def fitf(p, x):
                return p[1] + p[0] * x
        else:
            def fitf(p, x):
                return p[1] * np.exp(-p[0] * x)

        fitp = pe.least_squares(x, data, fitf, expected_chisquare=True)

        fitpc = pe.least_squares(x, data, fitf, correlated_fit=True)
        for i in range(2):
            diff = fitp[i] - fitpc[i]
            diff.gamma_method()
            assert(diff.is_zero_within_error(sigma=1.5))


def test_total_least_squares():
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
    odr = ODR(data, model, [0, 0], partol=np.finfo(np.float64).eps)
    odr.set_job(fit_type=0, deriv=1)
    output = odr.run()

    out = pe.total_least_squares(ox, oy, func, expected_chisquare=True)
    beta = out.fit_parameters

    for i in range(2):
        beta[i].gamma_method(S=1.0)
        assert math.isclose(beta[i].value, output.beta[i], rel_tol=1e-5)
        assert math.isclose(output.cov_beta[i, i], beta[i].dvalue ** 2, rel_tol=2.5e-1), str(output.cov_beta[i, i]) + ' ' + str(beta[i].dvalue ** 2)
    assert math.isclose(pe.covariance(beta[0], beta[1]), output.cov_beta[0, 1], rel_tol=2.5e-1)

    out = pe.total_least_squares(ox, oy, func, const_par=[beta[1]])

    diff = out.fit_parameters[0] - beta[0]
    assert(diff / beta[0] < 1e-3 * beta[0].dvalue)
    assert((out.fit_parameters[1] - beta[1]).is_zero())

    oxc = []
    for i, item in enumerate(x):
        oxc.append(pe.cov_Obs(x[i], xerr[i]**2, 'covx' + str(i)))

    oyc = []
    for i, item in enumerate(x):
        oyc.append(pe.cov_Obs(y[i], yerr[i]**2, 'covy' + str(i)))

    outc = pe.total_least_squares(oxc, oyc, func)
    betac = outc.fit_parameters

    for i in range(2):
        betac[i].gamma_method(S=1.0)
        assert math.isclose(betac[i].value, output.beta[i], rel_tol=1e-3)
        assert math.isclose(output.cov_beta[i, i], betac[i].dvalue ** 2, rel_tol=2.5e-1), str(output.cov_beta[i, i]) + ' ' + str(betac[i].dvalue ** 2)
    assert math.isclose(pe.covariance(betac[0], betac[1]), output.cov_beta[0, 1], rel_tol=2.5e-1)

    outc = pe.total_least_squares(oxc, oyc, func, const_par=[betac[1]])

    diffc = outc.fit_parameters[0] - betac[0]
    assert(diffc / betac[0] < 1e-3 * betac[0].dvalue)
    assert((outc.fit_parameters[1] - betac[1]).is_zero())

    outc = pe.total_least_squares(oxc, oy, func)
    betac = outc.fit_parameters

    for i in range(2):
        betac[i].gamma_method(S=1.0)
        assert math.isclose(betac[i].value, output.beta[i], rel_tol=1e-3)
        assert math.isclose(output.cov_beta[i, i], betac[i].dvalue ** 2, rel_tol=2.5e-1), str(output.cov_beta[i, i]) + ' ' + str(betac[i].dvalue ** 2)
    assert math.isclose(pe.covariance(betac[0], betac[1]), output.cov_beta[0, 1], rel_tol=2.5e-1)

    outc = pe.total_least_squares(oxc, oy, func, const_par=[betac[1]])

    diffc = outc.fit_parameters[0] - betac[0]
    assert(diffc / betac[0] < 1e-3 * betac[0].dvalue)
    assert((outc.fit_parameters[1] - betac[1]).is_zero())


def test_odr_derivatives():
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
    out = pe.total_least_squares(x, y, func)
    fit1 = out.fit_parameters

    with pytest.warns(DeprecationWarning):
        tfit = pe.fits.fit_general(x, y, func, base_step=0.1, step_ratio=1.1, num_steps=20)
    assert np.abs(np.max(np.array(list(fit1[1].deltas.values()))
                  - np.array(list(tfit[1].deltas.values())))) < 10e-8


def test_r_value_persistence():
    def f(a, x):
        return a[0] + a[1] * x

    a = pe.pseudo_Obs(1.1, .1, 'a')
    assert np.isclose(a.value, a.r_values['a'])

    a_2 = a ** 2
    assert np.isclose(a_2.value, a_2.r_values['a'])

    b = pe.pseudo_Obs(2.1, .2, 'b')

    y = [a, b]
    [o.gamma_method() for o in y]

    fitp = pe.fits.least_squares([1, 2], y, f)

    assert np.isclose(fitp[0].value, fitp[0].r_values['a'])
    assert np.isclose(fitp[0].value, fitp[0].r_values['b'])
    assert np.isclose(fitp[1].value, fitp[1].r_values['a'])
    assert np.isclose(fitp[1].value, fitp[1].r_values['b'])

    fitp = pe.fits.total_least_squares(y, y, f)

    assert np.isclose(fitp[0].value, fitp[0].r_values['a'])
    assert np.isclose(fitp[0].value, fitp[0].r_values['b'])
    assert np.isclose(fitp[1].value, fitp[1].r_values['a'])
    assert np.isclose(fitp[1].value, fitp[1].r_values['b'])

    fitp = pe.fits.least_squares([1, 2], y, f, priors=y)

    assert np.isclose(fitp[0].value, fitp[0].r_values['a'])
    assert np.isclose(fitp[0].value, fitp[0].r_values['b'])
    assert np.isclose(fitp[1].value, fitp[1].r_values['a'])
    assert np.isclose(fitp[1].value, fitp[1].r_values['b'])

