import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt
import math
import scipy.optimize
from scipy.odr import ODR, Model, RealData
from scipy.linalg import cholesky
from scipy.stats import norm
import iminuit
from autograd import jacobian
from autograd import hessian
import pyerrors as pe
import pytest

np.random.seed(0)


def test_fit_lin():
    x = [0, 2]
    y = [pe.pseudo_Obs(0, 0.1, 'ensemble'),
         pe.pseudo_Obs(2, 0.1, 'ensemble')]

    res = pe.fits.fit_lin(x, y)

    assert res[0] == y[0]
    assert res[1] == (y[1] - y[0]) / (x[1] - x[0])

    x = y = [pe.pseudo_Obs(0, 0.1, 'ensemble'),
         pe.pseudo_Obs(2, 0.1, 'ensemble')]

    res = pe.fits.fit_lin(x, y)

    m = (y[1] - y[0]) / (x[1] - x[0])
    assert res[0] == y[1] - x[1] * m
    assert res[1] == m


def test_least_squares():
    dim = 10 + int(30 * np.random.rand())
    x = np.arange(dim)
    y = 2 * np.exp(-0.06 * x) + np.random.normal(0.0, 0.15, dim)
    yerr = 0.1 + 0.1 * np.random.rand(dim)

    oy = []
    for i, item in enumerate(x):
        oy.append(pe.pseudo_Obs(y[i], yerr[i], str(i)))

    def f(x, a, b):
        return a * anp.exp(-b * x)

    popt, pcov = scipy.optimize.curve_fit(f, x, y, sigma=[o.dvalue for o in oy], absolute_sigma=True)

    def func(a, x):
        y = a[0] * anp.exp(-a[1] * x)
        return y

    out = pe.least_squares(x, oy, func, expected_chisquare=True, resplot=True, qqplot=True)
    beta = out.fit_parameters

    str(out)
    repr(out)
    len(out)

    for i in range(2):
        beta[i].gamma_method(S=1.0)
        assert math.isclose(beta[i].value, popt[i], abs_tol=1e-5)
        assert math.isclose(pcov[i, i], beta[i].dvalue ** 2, abs_tol=1e-3)

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


def test_least_squares_num_grad():
    x = []
    y = []
    for i in range(2, 5):
        x.append(i * 0.01)
        y.append(pe.pseudo_Obs(i * 0.01, 0.0001, "ens"))

    num = pe.fits.least_squares(x, y, lambda a, x: np.exp(a[0] * x) + a[1], num_grad=True)
    auto = pe.fits.least_squares(x, y, lambda a, x: anp.exp(a[0] * x) + a[1], num_grad=False)

    assert(num[0] == auto[0])
    assert(num[1] == auto[1])


def test_prior_fit_num_grad():
    x = []
    y = []
    for i in range(2, 5):
        x.append(i * 0.01)
        y.append(pe.pseudo_Obs(i * 0.01, 0.0001, "ens"))

    num = pe.fits.least_squares(x, y, lambda a, x: np.exp(a[0] * x) + a[1], num_grad=True, priors=y[:2])
    auto = pe.fits.least_squares(x, y, lambda a, x: anp.exp(a[0] * x) + a[1], num_grad=False, piors=y[:2])


def test_total_least_squares_num_grad():
    x = []
    y = []
    for i in range(2, 5):
        x.append(pe.pseudo_Obs(i * 0.01, 0.0001, "ens"))
        y.append(pe.pseudo_Obs(i * 0.01, 0.0001, "ens"))

    num = pe.fits.total_least_squares(x, y, lambda a, x: np.exp(a[0] * x) + a[1], num_grad=True)
    auto = pe.fits.total_least_squares(x, y, lambda a, x: anp.exp(a[0] * x) + a[1], num_grad=False)

    assert(num[0] == auto[0])
    assert(num[1] == auto[1])


def test_alternative_solvers():
    dim = 92
    x = np.arange(dim)
    y = 2 * np.exp(-0.06 * x) + np.random.normal(0.0, 0.15, dim)
    yerr = 0.1 + 0.1 * np.random.rand(dim)

    oy = []
    for i, item in enumerate(x):
        oy.append(pe.pseudo_Obs(y[i], yerr[i], 'test'))

    def func(a, x):
        y = a[0] * anp.exp(-a[1] * x)
        return y

    chisquare_values = []
    out = pe.least_squares(x, oy, func, method='migrad')
    chisquare_values.append(out.chisquare)
    out = pe.least_squares(x, oy, func, method='Powell')
    chisquare_values.append(out.chisquare)
    out = pe.least_squares(x, oy, func, method='Nelder-Mead')
    chisquare_values.append(out.chisquare)
    out = pe.least_squares(x, oy, func, method='Levenberg-Marquardt')
    chisquare_values.append(out.chisquare)
    chisquare_values = np.array(chisquare_values)
    assert np.all(np.isclose(chisquare_values, chisquare_values[0]))


def test_correlated_fit():
    num_samples = 400
    N = 10

    x = norm.rvs(size=(N, num_samples))

    r = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            r[i, j] = np.exp(-0.8 * np.fabs(i - j))

    errl = np.sqrt([3.4, 2.5, 3.6, 2.8, 4.2, 4.7, 4.9, 5.1, 3.2, 4.2])
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
                return p[1] * anp.exp(-p[0] * x)

        fitp = pe.least_squares(x, data, fitf, expected_chisquare=True)
        assert np.isclose(fitp.chisquare / fitp.dof, fitp.chisquare_by_dof, atol=1e-14)

        fitpc = pe.least_squares(x, data, fitf, correlated_fit=True)
        assert np.isclose(fitpc.chisquare / fitpc.dof, fitpc.chisquare_by_dof, atol=1e-14)
        for i in range(2):
            diff = fitp[i] - fitpc[i]
            diff.gamma_method()
            assert(diff.is_zero_within_error(sigma=5))


def test_hotelling_t():
    tt1 = pe.Obs([np.random.rand(50)], ["ens"])
    tt1.gamma_method()
    tt2 = pe.Obs([np.random.rand(50)], ["ens"])
    tt2.gamma_method()
    ft = pe.fits.least_squares([1, 2], [tt1, tt2], lambda a, x: a[0], correlated_fit=True)
    assert ft.t2_p_value >= ft.p_value


def test_fit_corr_independent():
    dim = 30
    x = np.arange(dim)
    y = 0.84 * np.exp(-0.12 * x) + np.random.normal(0.0, 0.1, dim)
    yerr = [0.1] * dim

    oy = []
    for i, item in enumerate(x):
        oy.append(pe.pseudo_Obs(y[i], yerr[i], str(i)))

    def func(a, x):
        y = a[0] * anp.exp(-a[1] * x)
        return y

    for method in ["Levenberg-Marquardt", "migrad", "Nelder-Mead"]:
        out = pe.least_squares(x, oy, func, method=method)
        out_corr = pe.least_squares(x, oy, func, correlated_fit=True, method=method)

        assert np.isclose(out.chisquare, out_corr.chisquare)
        assert np.isclose(out.dof, out_corr.dof)
        assert np.isclose(out.chisquare_by_dof, out_corr.chisquare_by_dof)
        assert (out[0] - out_corr[0]).is_zero(atol=1e-4)
        assert (out[1] - out_corr[1]).is_zero(atol=1e-4)


def test_linear_fit_guesses():
    for err in [10, 0.1, 0.001]:
        xvals = []
        yvals = []
        for x in range(1, 8, 2):
            xvals.append(x)
            yvals.append(pe.pseudo_Obs(x + np.random.normal(0.0, err), err, 'test1') + pe.pseudo_Obs(0, err / 100, 'test2', samples=87))
        lin_func = lambda a, x: a[0] + a[1] * x
        with pytest.raises(Exception):
            pe.least_squares(xvals, yvals, lin_func)
        [o.gamma_method() for o in yvals]
        with pytest.raises(Exception):
            pe.least_squares(xvals, yvals, lin_func, initial_guess=[5])

        bad_guess = pe.least_squares(xvals, yvals, lin_func, initial_guess=[999, 999])
        good_guess = pe.least_squares(xvals, yvals, lin_func, initial_guess=[0, 1])
        assert np.isclose(bad_guess.chisquare, good_guess.chisquare, atol=1e-8)
        assert np.all([(go - ba).is_zero(atol=1e-6) for (go, ba) in zip(good_guess, bad_guess)])


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
        return a * anp.exp(-b * x)

    def func(a, x):
        y = a[0] * anp.exp(-a[1] * x)
        return y

    data = RealData([o.value for o in ox], [o.value for o in oy], sx=[o.dvalue for o in ox], sy=[o.dvalue for o in oy])
    model = Model(func)
    odr = ODR(data, model, [0, 0], partol=np.finfo(np.float64).eps)
    odr.set_job(fit_type=0, deriv=1)
    output = odr.run()

    out = pe.total_least_squares(ox, oy, func, expected_chisquare=True)
    beta = out.fit_parameters

    str(out)
    repr(out)
    len(out)

    for i in range(2):
        beta[i].gamma_method(S=1.0)
        assert math.isclose(beta[i].value, output.beta[i], rel_tol=1e-5)
        assert math.isclose(output.cov_beta[i, i], beta[i].dvalue ** 2, rel_tol=2.5e-1), str(output.cov_beta[i, i]) + ' ' + str(beta[i].dvalue ** 2)

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

    tfit = fit_general(x, y, func, base_step=0.1, step_ratio=1.1, num_steps=20)
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


def test_prior_fit():
    def f(a, x):
        return a[0] + a[1] * x

    a = pe.pseudo_Obs(0.0, 0.1, 'a')
    b = pe.pseudo_Obs(1.0, 0.2, 'a')

    y = [a, b]
    with pytest.raises(Exception):
        fitp = pe.fits.least_squares([0, 1], 1 * np.array(y), f, priors=['0.0(8)', '1.0(8)'])

    [o.gamma_method() for o in y]

    fitp = pe.fits.least_squares([0, 1], y, f, priors=['0.0(8)', '1.0(8)'])
    fitp = pe.fits.least_squares([0, 1], y, f, priors=y, resplot=True, qqplot=True)


def test_vs_old_prior_implementation():
    x = np.arange(1, 5)
    y = [pe.pseudo_Obs(2 * i + 1.1 + np.random.normal(0.0, 0.1), .1, 't') for i in x]
    [o.gm() for o in y];
    def fitf(a, x):
        return a[0] * x + a[1]
    priors = [pe.cov_Obs(1.10, 0.01 ** 2, "p0"), pe.cov_Obs(1.1, 0.3 ** 2, "p1")]
    pr = pe.fits.least_squares(x, y, fitf, priors=priors, method="migrad")
    fr = old_prior_fit(x, y, fitf, priors=priors)
    assert pr[0] == fr[0]
    assert pr[1] == fr[1]


def test_correlated_fit_covobs():
    x1 = pe.cov_Obs(1.01, 0.01 ** 2, 'test1')
    x2 = pe.cov_Obs(2.01, 0.01 ** 2, 'test2')
    x3 = pe.cov_Obs(2.99, 0.01 ** 2, 'test3')

    [o.gamma_method() for o in [x1, x2, x3]]

    def func(a, x):
        return a[0] * x + a[1]

    fit_res = pe.fits.least_squares(np.arange(1, 4), [x1, x2, x3], func, expected_chisquare=True)
    assert np.isclose(fit_res.chisquare_by_dof, fit_res.chisquare_by_expected_chisquare)

    fit_res_corr = pe.fits.least_squares(np.arange(1, 4), [x1, x2, x3], func, correlated_fit=True)
    assert np.isclose(fit_res.chisquare_by_dof, fit_res_corr.chisquare_by_dof)


def test_error_band():
    def f(a, x):
        return a[0] + a[1] * x

    a = pe.pseudo_Obs(0.0, 0.1, 'a')
    b = pe.pseudo_Obs(1.0, 0.2, 'a')

    x = [0, 1]
    y = [a, b]

    fitp = pe.fits.least_squares(x, y, f)

    with pytest.raises(Exception):
        pe.fits.error_band(x, f, fitp.fit_parameters)
    fitp.gamma_method()
    pe.fits.error_band(x, f, fitp.fit_parameters)


def test_fit_vs_jackknife():
    od = 0.9999999999
    cov1 = np.array([[1, od, od], [od, 1.0, od], [od, od, 1.0]])
    cov1 *= 0.05
    nod = -0.4
    cov2 = np.array([[1, nod, nod], [nod, 1.0, nod], [nod, nod, 1.0]])
    cov2 *= 0.05
    cov3 = np.identity(3)
    cov3 *= 0.05
    samples = 500

    for i, cov in enumerate([cov1, cov2, cov3]):
        dat = pe.misc.gen_correlated_data(np.arange(1, 4), cov, 'test', 0.5, samples=samples)
        [o.gamma_method(S=0) for o in dat]
        func = lambda a, x: a[0] + a[1] * x
        fr = pe.least_squares(np.arange(1, 4), dat, func)
        fr.gamma_method(S=0)

        jd = np.array([o.export_jackknife() for o in dat]).T
        jfr = []
        for jacks in jd:

            def chisqfunc_residuals(p):
                model = func(p, np.arange(1, 4))
                chisq = ((jacks - model) / [o.dvalue for o in dat])
                return chisq

            tf = scipy.optimize.least_squares(chisqfunc_residuals, [0.0, 0.0], method='lm', ftol=1e-15, gtol=1e-15, xtol=1e-15)
            jfr.append(tf.x)
        ajfr = np.array(jfr).T
        err = np.array([np.sqrt(np.var(ajfr[j][1:], ddof=0) * (samples - 1)) for j in range(2)])
        assert np.allclose(err, [o.dvalue for o in fr], atol=1e-8)

def test_correlated_fit_vs_jackknife():
    od = 0.999999
    cov1 = np.array([[1, od, od], [od, 1.0, od], [od, od, 1.0]])
    cov1 *= 0.1
    nod = -0.44
    cov2 = np.array([[1, nod, nod], [nod, 1.0, nod], [nod, nod, 1.0]])
    cov2 *= 0.1
    cov3 = np.identity(3)
    cov3 *= 0.01

    samples = 250
    x_val = np.arange(1, 6, 2)
    for i, cov in enumerate([cov1, cov2, cov3]):
        dat = pe.misc.gen_correlated_data(x_val + x_val ** 2 + np.random.normal(0.0, 0.1, 3), cov, 'test', 0.5, samples=samples)
        [o.gamma_method(S=0) for o in dat]
        func = lambda a, x: a[0] * x + a[1] * x ** 2
        fr = pe.least_squares(x_val, dat, func, correlated_fit=True, silent=True)
        [o.gamma_method(S=0) for o in fr]

        cov = pe.covariance(dat)
        chol = np.linalg.cholesky(cov)
        chol_inv = np.linalg.inv(chol)

        jd = np.array([o.export_jackknife() for o in dat]).T
        jfr = []
        for jacks in jd:

            def chisqfunc_residuals(p):
                model = func(p, x_val)
                chisq = np.dot(chol_inv, (jacks - model))
                return chisq

            tf = scipy.optimize.least_squares(chisqfunc_residuals, [0.0, 0.0], method='lm', ftol=1e-15, gtol=1e-15, xtol=1e-15)
            jfr.append(tf.x)
        ajfr = np.array(jfr).T
        err = np.array([np.sqrt(np.var(ajfr[j][1:], ddof=0) * (samples - 1)) for j in range(2)])
        assert np.allclose(err, [o.dvalue for o in fr], atol=1e-7)
        assert np.allclose(ajfr.T[0], [o.value for o in fr], atol=1e-8)


def test_fit_no_autograd():
    dim = 3
    x = np.arange(dim)
    y = 2 * np.exp(-0.08 * x) + np.random.normal(0.0, 0.15, dim)
    yerr = 0.1 + 0.1 * np.random.rand(dim)

    oy = []
    for i, item in enumerate(x):
        oy.append(pe.pseudo_Obs(y[i], yerr[i], str(i)))

    def func(a, x):
        y = a[0] * np.exp(-a[1] * x)
        return y

    with pytest.raises(Exception):
        pe.least_squares(x, oy, func)

    pe.least_squares(x, oy, func, num_grad=True)

    with pytest.raises(Exception):
        pe.total_least_squares(oy, oy, func)


def test_invalid_fit_function():
    def func1(a, x):
        return a[0] + a[1] * x + a[2] * anp.sinh(x) + a[199]

    def func2(a, x, y):
        return a[0] + a[1] * x

    def func3(x):
        return x

    xvals =[]
    yvals =[]
    err = 0.1

    for x in range(1, 8, 2):
        xvals.append(x)
        yvals.append(pe.pseudo_Obs(x + np.random.normal(0.0, err), err, 'test1') + pe.pseudo_Obs(0, err / 100, 'test2', samples=87))
    [o.gamma_method() for o in yvals]
    for func in [func1, func2, func3]:
        with pytest.raises(Exception):
            pe.least_squares(xvals, yvals, func)
        with pytest.raises(Exception):
            pe.total_least_squares(yvals, yvals, func)


def test_singular_correlated_fit():
    obs1 = pe.pseudo_Obs(1.0, 0.1, 'test')
    with pytest.raises(Exception):
        pe.fits.fit_lin([0, 1], [obs1, obs1], correlated_fit=True)


def test_ks_test():
    def f(a, x):
        y = a[0] + a[1] * x
        return y

    fit_res = []

    for i in range(20):
        data = []
        for j in range(10):
            data.append(pe.pseudo_Obs(j + np.random.normal(0.0, 0.25), 0.25, 'test'))
        my_corr = pe.Corr(data)

        fit_res.append(my_corr.fit(f, silent=True))

    pe.fits.ks_test()
    pe.fits.ks_test(fit_res)


def test_combined_fit_list_v_array():
    res = []
    for y_test in [{'a': [pe.Obs([np.random.normal(i, 0.5, 1000)], ['ensemble1']) for i in range(1, 7)]},
               {'a': np.array([pe.Obs([np.random.normal(i, 0.5, 1000)], ['ensemble1']) for i in range(1, 7)])}]:
        for x_test in [{'a': [0, 1, 2, 3, 4, 5]}, {'a': np.arange(6)}]:
            for key in y_test.keys():
                [item.gamma_method() for item in y_test[key]]
            def func_a(a, x):
                return a[1] * x + a[0]

            funcs_test = {"a": func_a}
            res.append(pe.fits.least_squares(x_test, y_test, funcs_test))

        assert (res[0][0] - res[1][0]).is_zero(atol=1e-8)
        assert (res[0][1] - res[1][1]).is_zero(atol=1e-8)


def test_combined_fit_vs_standard_fit():

    x_const = {'a':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'b':np.arange(10, 20)}
    y_const = {'a':[pe.Obs([np.random.normal(1, val, 1000)], ['ensemble1']) 
                for val in [0.25, 0.3, 0.01, 0.2, 0.5, 1.3, 0.26, 0.4, 0.1, 1.0]],
            'b':[pe.Obs([np.random.normal(1, val, 1000)], ['ensemble1'])
                for val in [0.5, 1.12, 0.26, 0.25, 0.3, 0.01, 0.2, 1.0, 0.38, 0.1]]}
    for key in y_const.keys():
        [item.gamma_method() for item in y_const[key]]
    y_const_ls = np.concatenate([np.array(o) for o in y_const.values()])
    x_const_ls = np.arange(0, 20)

    def func_const(a,x):
        return  0 * x + a[0]

    funcs_const = {"a": func_const,"b": func_const}
    for method_kw in ['Levenberg-Marquardt', 'migrad', 'Powell', 'Nelder-Mead']:
        res = []
        res.append(pe.fits.least_squares(x_const, y_const, funcs_const, method = method_kw, expected_chisquare=True))
        res.append(pe.fits.least_squares(x_const_ls, y_const_ls, func_const, method = method_kw, expected_chisquare=True))
        [item.gamma_method for item in res]
        assert np.isclose(0.0, (res[0].chisquare_by_dof - res[1].chisquare_by_dof), 1e-14, 1e-8)
        assert np.isclose(0.0, (res[0].chisquare_by_expected_chisquare - res[1].chisquare_by_expected_chisquare), 1e-14, 1e-8)
        assert np.isclose(0.0, (res[0].p_value - res[1].p_value), 1e-14, 1e-8)
        assert (res[0][0] - res[1][0]).is_zero(atol=1e-8)


def test_combined_fit_no_autograd():

    def func_exp1(x):
        return 0.3*np.exp(0.5*x)

    def func_exp2(x):
        return 0.3*np.exp(0.8*x)

    xvals_b = np.arange(0,6)
    xvals_a = np.arange(0,8)

    def func_a(a,x):
        return a[0]*np.exp(a[1]*x)

    def func_b(a,x):
        return a[0]*np.exp(a[2]*x)

    funcs = {'a':func_a, 'b':func_b}
    xs = {'a':xvals_a, 'b':xvals_b}
    ys = {'a':[pe.Obs([np.random.normal(item, item*1.5, 1000)],['ensemble1']) for item in func_exp1(xvals_a)],
        'b':[pe.Obs([np.random.normal(item, item*1.4, 1000)],['ensemble1']) for item in func_exp2(xvals_b)]}

    for key in funcs.keys():
        [item.gamma_method() for item in ys[key]]

    with pytest.raises(Exception):
        pe.least_squares(xs, ys, funcs)

    pe.least_squares(xs, ys, funcs, num_grad=True)

def test_plot_combined_fit_function():

    def func_exp1(x):
        return 0.3*anp.exp(0.5*x)

    def func_exp2(x):
        return 0.3*anp.exp(0.8*x)

    xvals_b = np.arange(0,6)
    xvals_a = np.arange(0,8)

    def func_a(a,x):
        return a[0]*anp.exp(a[1]*x)

    def func_b(a,x):
        return a[0]*anp.exp(a[2]*x)

    corr_a = pe.Corr([pe.Obs([np.random.normal(item, item*1.5, 1000)],['ensemble1']) for item in func_exp1(xvals_a)])
    corr_b = pe.Corr([pe.Obs([np.random.normal(item, item*1.4, 1000)],['ensemble1']) for item in func_exp2(xvals_b)])

    funcs = {'a':func_a, 'b':func_b}
    xs = {'a':xvals_a, 'b':xvals_b}
    ys = {'a': [o[0] for o in corr_a.content],
          'b': [o[0] for o in corr_b.content]}

    corr_a.gm()
    corr_b.gm()

    comb_fit = pe.least_squares(xs, ys, funcs)

    with pytest.raises(ValueError):
        corr_a.show(x_range=[xs["a"][0], xs["a"][-1]], fit_res=comb_fit)

    corr_a.show(x_range=[xs["a"][0], xs["a"][-1]], fit_res=comb_fit, fit_key="a")
    corr_b.show(x_range=[xs["b"][0], xs["b"][-1]], fit_res=comb_fit, fit_key="b")


def test_combined_fit_invalid_fit_functions():
    def func1(a, x):
        return a[0] + a[1] * x + a[2] * anp.sinh(x) + a[199]

    def func2(a, x, y):
        return a[0] + a[1] * x

    def func3(x):
        return x

    def func_valid(a,x):
        return a[0] + a[1] * x

    xvals =[]
    yvals =[]
    err = 0.1

    for x in range(1, 8, 2):
        xvals.append(x)
        yvals.append(pe.pseudo_Obs(x + np.random.normal(0.0, err), err, 'test1') + pe.pseudo_Obs(0, err / 100, 'test2', samples=87))
    [o.gamma_method() for o in yvals]
    for func in [func1, func2, func3]:
        with pytest.raises(Exception):
            pe.least_squares({'a':xvals}, {'a':yvals}, {'a':func})
        with pytest.raises(Exception):
            pe.least_squares({'a':xvals, 'b':xvals}, {'a':yvals, 'b':yvals}, {'a':func, 'b':func_valid})
        with pytest.raises(Exception):
            pe.least_squares({'a':xvals, 'b':xvals}, {'a':yvals, 'b':yvals}, {'a':func_valid, 'b':func})


def test_combined_fit_invalid_input():
    xvals = []
    yvals = []
    err = 0.1
    def func_valid(a,x):
        return a[0] + a[1] * x
    for x in range(1, 8, 2):
        xvals.append(x)
        yvals.append(pe.pseudo_Obs(x + np.random.normal(0.0, err), err, 'test1') + pe.pseudo_Obs(0, err / 100, 'test2', samples=87))
    with pytest.raises(ValueError):
        pe.least_squares({'a':xvals}, {'b':yvals}, {'a':func_valid})
    with pytest.raises(Exception):
        pe.least_squares({'a':xvals}, {'a':yvals}, {'a':func_valid})


def test_combined_fit_no_autograd():

    def func_exp1(x):
        return 0.3*np.exp(0.5*x)

    def func_exp2(x):
        return 0.3*np.exp(0.8*x)

    xvals_b = np.arange(0,6)
    xvals_a = np.arange(0,8)

    def func_a(a,x):
        return a[0]*np.exp(a[1]*x)

    def func_b(a,x):
        return a[0]*np.exp(a[2]*x)

    funcs = {'a':func_a, 'b':func_b}
    xs = {'a':xvals_a, 'b':xvals_b}
    ys = {'a':[pe.Obs([np.random.normal(item, item*1.5, 1000)],['ensemble1']) for item in func_exp1(xvals_a)],
        'b':[pe.Obs([np.random.normal(item, item*1.4, 1000)],['ensemble1']) for item in func_exp2(xvals_b)]}

    for key in funcs.keys():
        [item.gamma_method() for item in ys[key]]

    with pytest.raises(Exception):
        pe.least_squares(xs, ys, funcs)

    pe.least_squares(xs, ys, funcs, num_grad=True)


def test_combined_fit_num_grad():
    def func_exp1(x):
        return 0.3*np.exp(0.5*x)

    def func_exp2(x):
        return 0.3*np.exp(0.8*x)

    xvals_b = np.arange(0,6)
    xvals_a = np.arange(0,8)

    def func_num_a(a,x):
        return a[0]*np.exp(a[1]*x)

    def func_num_b(a,x):
        return a[0]*np.exp(a[2]*x)

    def func_auto_a(a,x):
        return a[0]*anp.exp(a[1]*x)

    def func_auto_b(a,x):
        return a[0]*anp.exp(a[2]*x)

    funcs_num = {'a':func_num_a, 'b':func_num_b}
    funcs_auto = {'a':func_auto_a, 'b':func_auto_b}
    xs = {'a':xvals_a, 'b':xvals_b}
    ys = {'a':[pe.Obs([np.random.normal(item, item*1.5, 1000)],['ensemble1']) for item in func_exp1(xvals_a)],
        'b':[pe.Obs([np.random.normal(item, item*1.4, 1000)],['ensemble1']) for item in func_exp2(xvals_b)]}

    for key in funcs_num.keys():
        [item.gamma_method() for item in ys[key]]

    num = pe.fits.least_squares(xs, ys, funcs_num, num_grad=True)
    auto = pe.fits.least_squares(xs, ys, funcs_auto, num_grad=False)

    assert(num[0] == auto[0])
    assert(num[1] == auto[1])


def test_combined_fit_dictkeys_no_order():
    def func_exp1(x):
        return 0.3*np.exp(0.5*x)

    def func_exp2(x):
        return 0.3*np.exp(0.8*x)

    xvals_b = np.arange(0,6)
    xvals_a = np.arange(0,8)

    def func_num_a(a,x):
        return a[0]*np.exp(a[1]*x)

    def func_num_b(a,x):
        return a[0]*np.exp(a[2]*x)

    def func_auto_a(a,x):
        return a[0]*anp.exp(a[1]*x)

    def func_auto_b(a,x):
        return a[0]*anp.exp(a[2]*x)

    funcs = {'a':func_auto_a, 'b':func_auto_b}
    funcs_no_order = {'b':func_auto_b, 'a':func_auto_a}
    xs = {'a':xvals_a, 'b':xvals_b}
    xs_no_order = {'b':xvals_b, 'a':xvals_a}
    yobs_a = [pe.Obs([np.random.normal(item, item*1.5, 1000)],['ensemble1']) for item in func_exp1(xvals_a)]
    yobs_b = [pe.Obs([np.random.normal(item, item*1.4, 1000)],['ensemble1']) for item in func_exp2(xvals_b)]
    ys = {'a': yobs_a, 'b': yobs_b}
    ys_no_order = {'b': yobs_b, 'a': yobs_a}

    for key in funcs.keys():
        [item.gamma_method() for item in ys[key]]
        [item.gamma_method() for item in ys_no_order[key]]
    for method_kw in ['Levenberg-Marquardt', 'migrad', 'Powell', 'Nelder-Mead']:
        order = pe.fits.least_squares(xs, ys, funcs,method = method_kw)
        no_order_func = pe.fits.least_squares(xs, ys, funcs_no_order,method = method_kw)
        no_order_x = pe.fits.least_squares(xs_no_order, ys, funcs,method = method_kw)
        no_order_y = pe.fits.least_squares(xs, ys_no_order, funcs,method = method_kw)
        no_order_func_x = pe.fits.least_squares(xs_no_order, ys, funcs_no_order,method = method_kw)
        no_order_func_y = pe.fits.least_squares(xs, ys_no_order, funcs_no_order,method = method_kw)
        no_order_x_y = pe.fits.least_squares(xs_no_order, ys_no_order, funcs,method = method_kw)

        assert(no_order_func[0] == order[0])
        assert(no_order_func[1] == order[1])

        assert(no_order_x[0] == order[0])
        assert(no_order_x[1] == order[1])

        assert(no_order_y[0] == order[0])
        assert(no_order_y[1] == order[1])

        assert(no_order_func_x[0] == order[0])
        assert(no_order_func_x[1] == order[1])

        assert(no_order_func_y[0] == order[0])
        assert(no_order_func_y[1] == order[1])

        assert(no_order_x_y[0] == order[0])
        assert(no_order_x_y[1] == order[1])


def test_correlated_combined_fit_vs_correlated_standard_fit():

    x_const = {'a':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'b':np.arange(10, 20)}
    y_const = {'a':[pe.Obs([np.random.normal(1, val, 1000)], ['ensemble1']) 
                for val in [0.25, 0.3, 0.01, 0.2, 0.5, 1.3, 0.26, 0.4, 0.1, 1.0]],
            'b':[pe.Obs([np.random.normal(1, val, 1000)], ['ensemble1'])
                for val in [0.5, 1.12, 0.26, 0.25, 0.3, 0.01, 0.2, 1.0, 0.38, 0.1]]}
    for key in y_const.keys():
        [item.gamma_method() for item in y_const[key]]
    y_const_ls = np.concatenate([np.array(o) for o in y_const.values()])
    x_const_ls = np.arange(0, 20)

    def func_const(a,x):
        return  0 * x + a[0]

    funcs_const = {"a": func_const,"b": func_const}
    for method_kw in ['Levenberg-Marquardt', 'migrad', 'Powell', 'Nelder-Mead']:
        res = []
        res.append(pe.fits.least_squares(x_const, y_const, funcs_const, method = method_kw, correlated_fit=True))
        res.append(pe.fits.least_squares(x_const_ls, y_const_ls, func_const, method = method_kw, correlated_fit=True))
        [item.gamma_method for item in res]
        assert np.isclose(0.0, (res[0].chisquare_by_dof - res[1].chisquare_by_dof), 1e-14, 1e-8)
        assert np.isclose(0.0, (res[0].p_value - res[1].p_value), 1e-14, 1e-8)
        assert np.isclose(0.0, (res[0].t2_p_value - res[1].t2_p_value), 1e-14, 1e-8)
        assert (res[0][0] - res[1][0]).is_zero(atol=1e-8)


def test_combined_fit_hotelling_t():
    xvals_b = np.arange(0,6)
    xvals_a = np.arange(0,8)

    def func_exp1(x):
        return 0.3*np.exp(0.5*x)

    def func_exp2(x):
        return 0.3*np.exp(0.8*x)

    def func_a(a,x):
        return a[0]*anp.exp(a[1]*x)

    def func_b(a,x):
        return a[0]*anp.exp(a[2]*x)

    funcs = {'a':func_a, 'b':func_b}
    xs = {'a':xvals_a, 'b':xvals_b}
    yobs_a = [pe.Obs([np.random.normal(item, item*1.5, 1000)],['ensemble1']) for item in func_exp1(xvals_a)]
    yobs_b = [pe.Obs([np.random.normal(item, item*1.4, 1000)],['ensemble1']) for item in func_exp2(xvals_b)]
    ys = {'a': yobs_a, 'b': yobs_b}

    for key in funcs.keys():
        [item.gamma_method() for item in ys[key]]
    ft = pe.fits.least_squares(xs, ys, funcs, correlated_fit=True)
    assert ft.t2_p_value >= ft.p_value


def test_combined_resplot_qqplot():
    x = np.arange(3)
    y1 = [pe.pseudo_Obs(2 * o + np.random.normal(0, 0.1), 0.1, "test") for o in x]
    y2 = [pe.pseudo_Obs(3 * o ** 2 + np.random.normal(0, 0.1), 0.1, "test") for o in x]
    fr = pe.least_squares(x, y1, lambda a, x: a[0] + a[1] * x, resplot=True, qqplot=True)

    xd = {"1": x,
          "2": x}
    yd = {"1": y1,
          "2": y2}
    fd = {"1": lambda a, x: a[0] + a[1] * x,
          "2": lambda a, x: a[0] + a[2] * x ** 2}
    fr = pe.least_squares(xd, yd, fd, resplot=True, qqplot=True)
    plt.close('all')


def test_x_multidim_fit():
    x1 = np.arange(1, 10)
    x = np.array([[xi, xi] for xi in x1]).T
    y = [pe.pseudo_Obs(i + 2 / i, .1 * i, 't') for i in x[0]]
    [o.gm() for o in y]
    def fitf(a, x):
        return a[0] * x[0] + a[1] / x[1]

    pe.fits.least_squares(x, y, fitf)


def test_priors_dict_vs_list():
    x = np.arange(1, 5)
    y = [pe.pseudo_Obs(2 * i + 1.1 + np.random.normal(0.0, 0.01), .01, 't') for i in x]
    [o.gm() for o in y];
    def fitf(a, x):
        return a[0] * x + a[1]
    priors = [pe.cov_Obs(1.0, 0.0001 ** 2, "p0"), pe.cov_Obs(1.1, 0.8 ** 2, "p1")]
    pr1 = pe.fits.least_squares(x, y, fitf, priors=priors)
    prd = {0: priors[0],
           1: priors[1]}

    pr2 = pe.fits.least_squares(x, y, fitf, priors=prd)

    prd = {1: priors[1],
           0: priors[0]}

    pr3 = pe.fits.least_squares(x, y, fitf, priors=prd)
    assert (pr1[0] - pr2[0]).is_zero(1e-6)
    assert (pr1[1] - pr2[1]).is_zero(1e-6)
    assert (pr1[0] - pr3[0]).is_zero(1e-6)
    assert (pr1[1] - pr3[1]).is_zero(1e-6)


def test_not_all_priors_set():
    x = np.arange(1, 5)
    y = [pe.pseudo_Obs(2 * i + 1.1 + np.random.normal(0.0, 0.01), .01, 't') for i in x]
    [o.gm() for o in y];
    def fitf(a, x):
        return a[0] * x + a[1] + a[2] * x ** 2
    priors = [pe.cov_Obs(2.0, 0.1 ** 2, "p0"), pe.cov_Obs(2, 0.8 ** 2, "p2")]
    prd = {0: priors[0],
           2: priors[1]}

    pr1 = pe.fits.least_squares(x, y, fitf, priors=prd)

    prd = {2: priors[1],
           0: priors[0]}

    pr2 = pe.fits.least_squares(x, y, fitf, priors=prd)
    assert (pr1[0] - pr2[0]).is_zero(1e-6)
    assert (pr1[1] - pr2[1]).is_zero(1e-6)
    assert (pr1[2] - pr2[2]).is_zero(1e-6)


def test_force_fit_on_prior():
    x = np.arange(1, 10)
    y = [pe.pseudo_Obs(2 + np.random.normal(0.0, 0.1), .1, 't') for i in x]
    [o.gm() for o in y];
    def fitf(a, x):
        return a[0]

    prd = {0: pe.cov_Obs(0.0, 0.0000001 ** 2, "prior0")}

    pr = pe.fits.least_squares(x, y, fitf, priors=prd)
    pr.gm()

    diff = prd[0] - pr[0]
    diff.gm()
    assert diff.is_zero_within_error(5)


def test_constrained_and_prior_fit():

    for my_constant in [pe.pseudo_Obs(5.0, 0.00000001, "test"),
                        pe.pseudo_Obs(6.5, 0.00000001, "test"),
                        pe.pseudo_Obs(6.5, 0.00000001, "another_ensmble")]:
        dim = 10
        x = np.arange(dim)
        y = -0.06 * x + 5 + np.random.normal(0.0, 0.3, dim)
        yerr = [0.3] * dim

        oy = []
        for i, item in enumerate(x):
            oy.append(pe.pseudo_Obs(y[i], yerr[i], 'test'))

        def func(a, x):
            y = a[0] * x + a[1]
            return y

        # Fit with constrained parameter
        out = pe.least_squares(x, oy, func, priors={1: my_constant}, silent=True)
        out.gm()

        def alt_func(a, x):
            y = a[0] * x
            return y

        alt_y = np.array(oy) - my_constant
        [o.gm() for o in alt_y]

        # Fit with the constant subtracted from the data
        alt_out = pe.least_squares(x, alt_y, alt_func, silent=True)
        alt_out.gm()

        assert np.isclose(out[0].value, alt_out[0].value, atol=1e-5, rtol=1e-6)
        assert np.isclose(out[0].dvalue, alt_out[0].dvalue, atol=1e-5, rtol=1e-6)
        assert np.isclose(out.chisquare_by_dof, alt_out.chisquare_by_dof, atol=1e-5, rtol=1e-6)


def test_prior_fit_different_methods():
    dim = 5
    x = np.arange(dim)
    y = 2 * x + 0.5 + np.random.normal(0.0, 0.3, dim) + 0.02 * x ** 5
    yerr = [0.3] * dim

    oy = []
    for i, item in enumerate(x):
        oy.append(pe.pseudo_Obs(y[i], yerr[i], 'test'))

    def func(a, x):
        return a[0] * x + a[1] + a[2] * x ** 5

    for priors in [None, {1: "0.5(4)"}, ["2(1)", "0.6(3)", "0(5)"]]:
        chisquare_list = []
        for method in ["Levenberg-Marquardt", "migrad", "Powell"]:
            fr = pe.least_squares(x, oy, func, silent=True, priors=priors, method=method)
            print(fr.iterations)
            chisquare_list.append(fr.chisquare)

        assert np.allclose(chisquare_list[0], chisquare_list[1:])


def test_resplot_lists_in_dict():
    xd = {
        'a': [1, 2, 3],
        'b': [1, 2, 3],
    }
    yd = {
        'a': [pe.pseudo_Obs(i, .1*i, 't') for i in range(1, 4)],
        'b': [pe.pseudo_Obs(2*i**2, .1*i**2, 't') for i in range(1, 4)]
    }
    [[o.gamma_method() for o in y] for y in yd.values()]
    fd = {
        'a': lambda a, x: a[0] + a[1] * x,
        'b': lambda a, x: a[0] + a[2] * x**2,
    }

    fitp = pe.fits.least_squares(xd, yd, fd, resplot=True)


def test_fit_dof():

    def func(a, x):
        return a[1] * anp.exp(-x * a[0])

    dof = []
    cd = []
    for dim in [2, 3]:

        x = np.arange(dim)
        y = 2 * np.exp(-0.3 * x) + np.random.normal(0.0, 0.3, dim)
        yerr = [0.3] * dim

        oy = []
        for i, item in enumerate(x):
            oy.append(pe.pseudo_Obs(y[i], yerr[i], 'test'))

        for priors in [None, {0: "0(2)"}]:
            fr = pe.least_squares(x, oy, func, silent=True, priors=priors)
            dof.append(fr.dof)
            cd.append(fr.chisquare_by_dof)

    assert np.allclose(dof, [0, 1, 1, 2])
    assert cd[0] != cd[0]  # Check for nan
    assert np.all(np.array(cd[1:]) > 0)


def fit_general(x, y, func, silent=False, **kwargs):
    """Performs a non-linear fit to y = func(x) and returns a list of Obs corresponding to the fit parameters.

    Plausibility of the results should be checked. To control the numerical differentiation
    the kwargs of numdifftools.step_generators.MaxStepGenerator can be used.

    func has to be of the form

    def func(a, x):
        y = a[0] + a[1] * x + a[2] * np.sinh(x)
        return y

    y has to be a list of Obs, the dvalues of the Obs are used as yerror for the fit.
    x can either be a list of floats in which case no xerror is assumed, or
    a list of Obs, where the dvalues of the Obs are used as xerror for the fit.

    Keyword arguments
    -----------------
    silent -- If true all output to the console is omitted (default False).
    initial_guess -- can provide an initial guess for the input parameters. Relevant for non-linear fits
                     with many parameters.
    """

    if not callable(func):
        raise TypeError('func has to be a function.')

    for i in range(10):
        try:
            func(np.arange(i), 0)
        except:
            pass
        else:
            break
    n_parms = i
    if not silent:
        print('Fit with', n_parms, 'parameters')

    global print_output, beta0
    print_output = 1
    if 'initial_guess' in kwargs:
        beta0 = kwargs.get('initial_guess')
        if len(beta0) != n_parms:
            raise Exception('Initial guess does not have the correct length.')
    else:
        beta0 = np.arange(n_parms)

    if len(x) != len(y):
        raise Exception('x and y have to have the same length')

    if all(isinstance(n, pe.Obs) for n in x):
        obs = x + y
        x_constants = None
        xerr = [o.dvalue for o in x]
        yerr = [o.dvalue for o in y]
    elif all(isinstance(n, float) or isinstance(n, int) for n in x) or isinstance(x, np.ndarray):
        obs = y
        x_constants = x
        xerr = None
        yerr = [o.dvalue for o in y]
    else:
        raise Exception('Unsupported types for x')

    def do_the_fit(obs, **kwargs):

        global print_output, beta0

        func = kwargs.get('function')
        yerr = kwargs.get('yerr')
        length = len(yerr)

        xerr = kwargs.get('xerr')

        if length == len(obs):
            assert 'x_constants' in kwargs
            data = RealData(kwargs.get('x_constants'), obs, sy=yerr)
            fit_type = 2
        elif length == len(obs) // 2:
            data = RealData(obs[:length], obs[length:], sx=xerr, sy=yerr)
            fit_type = 0
        else:
            raise Exception('x and y do not fit together.')

        model = Model(func)

        odr = ODR(data, model, beta0, partol=np.finfo(np.float64).eps)
        odr.set_job(fit_type=fit_type, deriv=1)
        output = odr.run()
        if print_output and not silent:
            print(*output.stopreason)
            print('chisquare/d.o.f.:', output.res_var)
            print_output = 0
        beta0 = output.beta
        return output.beta[kwargs.get('n')]
    res = []
    for n in range(n_parms):
        res.append(pe.derived_observable(do_the_fit, obs, function=func, xerr=xerr, yerr=yerr, x_constants=x_constants, num_grad=True, n=n, **kwargs))
    return res


def old_prior_fit(x, y, func, priors, silent=False, **kwargs):
    output = pe.fits.Fit_result()

    output.fit_function = func

    x = np.asarray(x)

    if not callable(func):
        raise TypeError('func has to be a function.')

    for i in range(100):
        try:
            func(np.arange(i), 0)
        except TypeError:
            continue
        except IndexError:
            continue
        else:
            break
    else:
        raise RuntimeError("Fit function is not valid.")

    n_parms = i

    if n_parms != len(priors):
        raise Exception('Priors does not have the correct length.')

    def extract_val_and_dval(string):
        split_string = string.split('(')
        if '.' in split_string[0] and '.' not in split_string[1][:-1]:
            factor = 10 ** -len(split_string[0].partition('.')[2])
        else:
            factor = 1
        return float(split_string[0]), float(split_string[1][:-1]) * factor

    loc_priors = []
    for i_n, i_prior in enumerate(priors):
        if isinstance(i_prior, pe.Obs):
            loc_priors.append(i_prior)
        else:
            loc_val, loc_dval = extract_val_and_dval(i_prior)
            loc_priors.append(pe.cov_Obs(loc_val, loc_dval ** 2, '#prior' + str(i_n) + f"_{np.random.randint(2147483647):010d}"))

    output.priors = loc_priors

    if not silent:
        print('Fit with', n_parms, 'parameter' + 's' * (n_parms > 1))

    y_f = [o.value for o in y]
    dy_f = [o.dvalue for o in y]

    if np.any(np.asarray(dy_f) <= 0.0):
        raise Exception('No y errors available, run the gamma method first.')

    p_f = [o.value for o in loc_priors]
    dp_f = [o.dvalue for o in loc_priors]

    if np.any(np.asarray(dp_f) <= 0.0):
        raise Exception('No prior errors available, run the gamma method first.')

    if 'initial_guess' in kwargs:
        x0 = kwargs.get('initial_guess')
        if len(x0) != n_parms:
            raise Exception('Initial guess does not have the correct length.')
    else:
        x0 = p_f

    def chisqfunc(p):
        model = func(p, x)
        chisq = anp.sum(((y_f - model) / dy_f) ** 2) + anp.sum(((p_f - p) / dp_f) ** 2)
        return chisq

    if not silent:
        print('Method: migrad')

    m = iminuit.Minuit(chisqfunc, x0)
    m.errordef = 1
    m.print_level = 0
    if 'tol' in kwargs:
        m.tol = kwargs.get('tol')
    else:
        m.tol = 1e-4
    m.migrad()
    params = np.asarray(m.values)

    output.chisquare_by_dof = m.fval / len(x)

    output.method = 'migrad'

    if not silent:
        print('chisquare/d.o.f.:', output.chisquare_by_dof)

    if not m.fmin.is_valid:
        raise Exception('The minimization procedure did not converge.')

    hess = hessian(chisqfunc)(params)
    hess_inv = np.linalg.pinv(hess)

    def chisqfunc_compact(d):
        model = func(d[:n_parms], x)
        chisq = anp.sum(((d[n_parms: n_parms + len(x)] - model) / dy_f) ** 2) + anp.sum(((d[n_parms + len(x):] - d[:n_parms]) / dp_f) ** 2)
        return chisq

    jac_jac = hessian(chisqfunc_compact)(np.concatenate((params, y_f, p_f)))

    deriv = -hess_inv @ jac_jac[:n_parms, n_parms:]

    result = []
    for i in range(n_parms):
        result.append(pe.derived_observable(lambda x, **kwargs: (x[0] + np.finfo(np.float64).eps) / (y[0].value + np.finfo(np.float64).eps) * params[i], list(y) + list(loc_priors), man_grad=list(deriv[i])))

    output.fit_parameters = result
    output.chisquare = chisqfunc(np.asarray(params))

    if kwargs.get('resplot') is True:
        residual_plot(x, y, func, result)

    if kwargs.get('qqplot') is True:
        qqplot(x, y, func, result)

    return output
