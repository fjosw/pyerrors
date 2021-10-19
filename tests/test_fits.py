import autograd.numpy as np
import math
import scipy.optimize
from scipy.odr import ODR, Model, RealData
import pyerrors as pe
import pytest

np.random.seed(0)


def test_standard_fit():
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


def test_odr_fit():
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

    beta = pe.fits.odr_fit(ox, oy, func)

    pe.Obs.e_tag_global = 5
    for i in range(2):
        beta[i].gamma_method(e_tag=5, S=1.0)
        assert math.isclose(beta[i].value, output.beta[i], rel_tol=1e-5)
        assert math.isclose(output.cov_beta[i, i], beta[i].dvalue ** 2, rel_tol=2.5e-1), str(output.cov_beta[i, i]) + ' ' + str(beta[i].dvalue ** 2)
    assert math.isclose(pe.covariance(beta[0], beta[1]), output.cov_beta[0, 1], rel_tol=2.5e-1)
    pe.Obs.e_tag_global = 0


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

    fit1 = pe.fits.odr_fit(x, y, func)

    tfit = pe.fits.fit_general(x, y, func, base_step=0.1, step_ratio=1.1, num_steps=20)
    assert np.abs(np.max(np.array(list(fit1[1].deltas.values()))
                  - np.array(list(tfit[1].deltas.values())))) < 10e-8
