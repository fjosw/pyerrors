#!/usr/bin/env python
# coding: utf-8

import warnings
import numpy as np
import jax.numpy as jnp
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.odr import ODR, Model, RealData
import iminuit
from jax import jacobian
#from jax import elementwise_grad as egrad
from .pyerrors import Obs, derived_observable, covariance, pseudo_Obs


def standard_fit(x, y, func, silent=False, **kwargs):
    """Performs a non-linear fit to y = func(x) and returns a list of Obs corresponding to the fit parameters.

    x has to be a list of floats.
    y has to be a list of Obs, the dvalues of the Obs are used as yerror for the fit.

    func has to be of the form

    def func(a, x):
        return a[0] + a[1] * x + a[2] * jnp.sinh(x)

    For multiple x values func can be of the form

    def func(a, x):
    (x1, x2) = x
    return a[0] * x1 ** 2 + a[1] * x2

    It is important that all numpy functions refer to autograd.numpy, otherwise the differentiation
    will not work

    Keyword arguments
    -----------------
    dict_output -- If true, the output is a dictionary containing all relevant
                   data instead of just a list of the fit parameters.
    silent -- If true all output to the console is omitted (default False).
    initial_guess -- can provide an initial guess for the input parameters. Relevant for
                     non-linear fits with many parameters.
    method -- can be used to choose an alternative method for the minimization of chisquare.
              The possible methods are the ones which can be used for scipy.optimize.minimize and
              migrad of iminuit. If no method is specified, Levenberg-Marquard is used.
              Reliable alternatives are migrad, Powell and Nelder-Mead.
    resplot -- If true, a plot which displays fit, data and residuals is generated (default False).
    qqplot -- If true, a quantile-quantile plot of the fit result is generated (default False).
    expected_chisquare -- If true prints the expected chisquare which is
                          corrected by effects caused by correlated input data.
                          This can take a while as the full correlation matrix
                          has to be calculated (default False).
    """

    result_dict = {}

    result_dict['fit_function'] = func

    x = np.asarray(x)

    if x.shape[-1] != len(y):
        raise Exception('x and y input have to have the same length')

    if len(x.shape) > 2:
        raise Exception('Unkown format for x values')

    if not callable(func):
        raise TypeError('func has to be a function.')

    for i in range(25):
        try:
            func(np.arange(i), x.T[0])
        except:
            pass
        else:
            break

    n_parms = i

    if not silent:
        print('Fit with', n_parms, 'parameters')

    y_f = np.array([o.value for o in y])
    dy_f = np.array([o.dvalue for o in y])

    if np.any(dy_f <= 0.0):
        raise Exception('No y errors available, run the gamma method first.')

    if 'initial_guess' in kwargs:
        x0 = kwargs.get('initial_guess')
        if len(x0) != n_parms:
            raise Exception('Initial guess does not have the correct length.')
    else:
        x0 = [0.1] * n_parms

    def chisqfunc(p):
        model = func(p, x)
        chisq = jnp.sum(((y_f - model) / dy_f) ** 2)
        return chisq

    if 'method' in kwargs:
        result_dict['method'] = kwargs.get('method')
        if not silent:
            print('Method:', kwargs.get('method'))
        if kwargs.get('method') == 'migrad':
            fit_result = iminuit.minimize(chisqfunc, x0)
            fit_result = iminuit.minimize(chisqfunc, fit_result.x)
        else:
            fit_result = scipy.optimize.minimize(chisqfunc, x0, method=kwargs.get('method'))
            fit_result = scipy.optimize.minimize(chisqfunc, fit_result.x, method=kwargs.get('method'), tol=1e-12)

        chisquare = fit_result.fun
    else:
        result_dict['method'] = 'Levenberg-Marquardt'
        if not silent:
            print('Method: Levenberg-Marquardt')

        def chisqfunc_residuals(p):
            model = func(p, x)
            chisq = ((y_f - model) / dy_f)
            return chisq

        fit_result = scipy.optimize.least_squares(chisqfunc_residuals, x0, method='lm', ftol=1e-15, gtol=1e-15, xtol=1e-15)

        chisquare = np.sum(fit_result.fun ** 2)

    if not fit_result.success:
        raise Exception('The minimization procedure did not converge.')

    if x.shape[-1] - n_parms > 0:
        result_dict['chisquare/d.o.f.'] = chisquare / (x.shape[-1] - n_parms)
    else:
        result_dict['chisquare/d.o.f.'] = float('nan')

    if not silent:
        print(fit_result.message)
        print('chisquare/d.o.f.:', result_dict['chisquare/d.o.f.'])

    if kwargs.get('expected_chisquare') is True:
        W = np.diag(1 / np.asarray(dy_f))
        cov = covariance_matrix(y)
        A = W @ jacobian(func)(fit_result.x, x)
        P_phi = A @ np.linalg.inv(A.T @ A) @ A.T
        expected_chisquare = np.trace((np.identity(x.shape[-1]) - P_phi) @ W @ cov @ W)
        result_dict['chisquare/expected_chisquare'] = chisquare / expected_chisquare
        if not silent:
            print('chisquare/expected_chisquare:',
                  result_dict['chisquare/expected_chisquare'])

    hess_inv = np.linalg.pinv(jacobian(jacobian(chisqfunc))(fit_result.x))

    def chisqfunc_compact(d):
        model = func(d[:n_parms], x)
        chisq = jnp.sum(((d[n_parms:] - model) / dy_f) ** 2)
        return chisq

    jac_jac = jacobian(jacobian(chisqfunc_compact))(np.concatenate((fit_result.x, y_f)))

    deriv = -hess_inv @ jac_jac[:n_parms, n_parms:]

    result = []
    for i in range(n_parms):
        result.append(derived_observable(lambda x, **kwargs: x[0], [pseudo_Obs(fit_result.x[i], 0.0, y[0].names[0], y[0].shape[y[0].names[0]])] + list(y), man_grad=[0] + list(deriv[i])))

    result_dict['fit_parameters'] = result

    result_dict['chisquare'] = chisqfunc(fit_result.x)
    result_dict['d.o.f.'] = x.shape[-1] - n_parms

    if kwargs.get('resplot') is True:
        residual_plot(x, y, func, result)

    if kwargs.get('qqplot') is True:
        qqplot(x, y, func, result)

    return result_dict if kwargs.get('dict_output') else result


def odr_fit(x, y, func, silent=False, **kwargs):
    """Performs a non-linear fit to y = func(x) and returns a list of Obs corresponding to the fit parameters.

    x has to be a list of Obs, or a tuple of lists of Obs
    y has to be a list of Obs
    the dvalues of the Obs are used as x- and yerror for the fit.

    func has to be of the form

    def func(a, x):
        y = a[0] + a[1] * x + a[2] * jnp.sinh(x)
        return y

    For multiple x values func can be of the form

    def func(a, x):
    (x1, x2) = x
    return a[0] * x1 ** 2 + a[1] * x2

    It is important that all numpy functions refer to autograd.numpy, otherwise the differentiation
    will not work.
    Based on the orthogonal distance regression module of scipy

    Keyword arguments
    -----------------
    dict_output -- If true, the output is a dictionary containing all relevant
                   data instead of just a list of the fit parameters.
    silent -- If true all output to the console is omitted (default False).
    initial_guess -- can provide an initial guess for the input parameters. Relevant for non-linear
                     fits with many parameters.
    expected_chisquare -- If true prints the expected chisquare which is
                          corrected by effects caused by correlated input data.
                          This can take a while as the full correlation matrix
                          has to be calculated (default False).
    """

    result_dict = {}

    result_dict['fit_function'] = func

    x = np.array(x)

    x_shape = x.shape

    if not callable(func):
        raise TypeError('func has to be a function.')

    for i in range(25):
        try:
            func(np.arange(i), x.T[0])
        except:
            pass
        else:
            break

    n_parms = i
    if not silent:
        print('Fit with', n_parms, 'parameters')

    x_f = np.vectorize(lambda o: o.value)(x)
    dx_f = np.vectorize(lambda o: o.dvalue)(x)
    y_f = np.array([o.value for o in y])
    dy_f = np.array([o.dvalue for o in y])

    if np.any(np.asarray(dx_f) <= 0.0):
        raise Exception('No x errors available, run the gamma method first.')

    if np.any(np.asarray(dy_f) <= 0.0):
        raise Exception('No y errors available, run the gamma method first.')

    if 'initial_guess' in kwargs:
        x0 = kwargs.get('initial_guess')
        if len(x0) != n_parms:
            raise Exception('Initial guess does not have the correct length.')
    else:
        x0 = [1] * n_parms

    data = RealData(x_f, y_f, sx=dx_f, sy=dy_f)
    model = Model(func)
    odr = ODR(data, model, x0, partol=np.finfo(np.float64).eps)
    odr.set_job(fit_type=0, deriv=1)
    output = odr.run()

    result_dict['residual_variance'] = output.res_var

    result_dict['method'] = 'ODR'

    result_dict['xplus'] = output.xplus

    if not silent:
        print('Method: ODR')
        print(*output.stopreason)
        print('Residual variance:', result_dict['residual_variance'])

    if output.info > 3:
        raise Exception('The minimization procedure did not converge.')

    m = x_f.size

    def odr_chisquare(p):
        model = func(p[:n_parms], p[n_parms:].reshape(x_shape))
        chisq = jnp.sum(((y_f - model) / dy_f) ** 2) + jnp.sum(((x_f - p[n_parms:].reshape(x_shape)) / dx_f) ** 2)
        return chisq

    if kwargs.get('expected_chisquare') is True:
        W = np.diag(1 / np.asarray(np.concatenate((dy_f.ravel(), dx_f.ravel()))))

        if kwargs.get('covariance') is not None:
            cov = kwargs.get('covariance')
        else:
            cov = covariance_matrix(np.concatenate((y, x.ravel())))

        number_of_x_parameters = int(m / x_f.shape[-1])

        old_jac = jacobian(func)(output.beta, output.xplus)
        fused_row1 = np.concatenate((old_jac, np.concatenate((number_of_x_parameters * [np.zeros(old_jac.shape)]), axis=0)))
        fused_row2 = np.concatenate((jacobian(lambda x, y: func(y, x))(output.xplus, output.beta).reshape(x_f.shape[-1], x_f.shape[-1] * number_of_x_parameters), np.identity(number_of_x_parameters * old_jac.shape[0])))
        new_jac = np.concatenate((fused_row1, fused_row2), axis=1)

        A = W @ new_jac
        P_phi = A @ np.linalg.inv(A.T @ A) @ A.T
        expected_chisquare = np.trace((np.identity(P_phi.shape[0]) - P_phi) @ W @ cov @ W)
        if expected_chisquare <= 0.0:
            warnings.warn("Negative expected_chisquare.", RuntimeWarning)
            expected_chisquare = np.abs(expected_chisquare)
        result_dict['chisquare/expected_chisquare'] = odr_chisquare(np.concatenate((output.beta, output.xplus.ravel()))) / expected_chisquare
        if not silent:
            print('chisquare/expected_chisquare:',
                  result_dict['chisquare/expected_chisquare'])

    hess_inv = np.linalg.pinv(jacobian(jacobian(odr_chisquare))(np.concatenate((output.beta, output.xplus.ravel()))))

    def odr_chisquare_compact_x(d):
        model = func(d[:n_parms], d[n_parms:n_parms + m].reshape(x_shape))
        chisq = jnp.sum(((y_f - model) / dy_f) ** 2) + jnp.sum(((d[n_parms + m:].reshape(x_shape) - d[n_parms:n_parms + m].reshape(x_shape)) / dx_f) ** 2)
        return chisq

    jac_jac_x = jacobian(jacobian(odr_chisquare_compact_x))(np.concatenate((output.beta, output.xplus.ravel(), x_f.ravel())))

    deriv_x = -hess_inv @ jac_jac_x[:n_parms + m, n_parms + m:]

    def odr_chisquare_compact_y(d):
        model = func(d[:n_parms], d[n_parms:n_parms + m].reshape(x_shape))
        chisq = jnp.sum(((d[n_parms + m:] - model) / dy_f) ** 2) + jnp.sum(((x_f - d[n_parms:n_parms + m].reshape(x_shape)) / dx_f) ** 2)
        return chisq

    jac_jac_y = jacobian(jacobian(odr_chisquare_compact_y))(np.concatenate((output.beta, output.xplus.ravel(), y_f)))

    deriv_y = -hess_inv @ jac_jac_y[:n_parms + m, n_parms + m:]

    result = []
    for i in range(n_parms):
        result.append(derived_observable(lambda x, **kwargs: x[0], [pseudo_Obs(output.beta[i], 0.0, y[0].names[0], y[0].shape[y[0].names[0]])] + list(x.ravel()) + list(y), man_grad=[0] + list(deriv_x[i]) + list(deriv_y[i])))

    result_dict['fit_parameters'] = result

    result_dict['odr_chisquare'] = odr_chisquare(np.concatenate((output.beta, output.xplus.ravel())))
    result_dict['d.o.f.'] = x.shape[-1] - n_parms

    return result_dict if kwargs.get('dict_output') else result


def prior_fit(x, y, func, priors, silent=False, **kwargs):
    """Performs a non-linear fit to y = func(x) with given priors and returns a list of Obs corresponding to the fit parameters.

    x has to be a list of floats.
    y has to be a list of Obs, the dvalues of the Obs are used as yerror for the fit.

    func has to be of the form

    def func(a, x):
        y = a[0] + a[1] * x + a[2] * jnp.sinh(x)
        return y

    It is important that all numpy functions refer to autograd.numpy, otherwise the differentiation
    will not work

    priors has to be a list with an entry for every parameter in the fit. The entries can either be
    Obs (e.g. results from a previous fit) or strings containing a value and an error formatted like
    0.548(23), 500(40) or 0.5(0.4)

    It is important for the subsequent error estimation that the e_tag for the gamma method is large
    enough.

    Keyword arguments
    -----------------
    dict_output -- If true, the output is a dictionary containing all relevant
                   data instead of just a list of the fit parameters.
    silent -- If true all output to the console is omitted (default False).
    initial_guess -- can provide an initial guess for the input parameters.
                     If no guess is provided, the prior values are used.
    resplot -- if true, a plot which displays fit, data and residuals is generated (default False)
    qqplot -- if true, a quantile-quantile plot of the fit result is generated (default False)
    tol -- Specify the tolerance of the migrad solver (default 1e-4)
    """

    result_dict = {}

    result_dict['fit_function'] = func

    if Obs.e_tag_global < 4:
        warnings.warn("e_tag_global is smaller than 4, this can cause problems when calculating errors from fits with priors", RuntimeWarning)

    x = np.asarray(x)

    if not callable(func):
        raise TypeError('func has to be a function.')

    for i in range(100):
        try:
            func(np.arange(i), 0)
        except:
            pass
        else:
            break

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
        if isinstance(i_prior, Obs):
            loc_priors.append(i_prior)
        else:
            loc_val, loc_dval = extract_val_and_dval(i_prior)
            loc_priors.append(pseudo_Obs(loc_val, loc_dval, 'p' + str(i_n)))

    result_dict['priors'] = loc_priors

    if not silent:
        print('Fit with', n_parms, 'parameters')

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
        chisq = jnp.sum(((y_f - model) / dy_f) ** 2) + jnp.sum(((p_f - p) / dp_f) ** 2)
        return chisq

    if not silent:
        print('Method: migrad')

    m = iminuit.Minuit.from_array_func(chisqfunc, x0, error=np.asarray(x0) * 0.01, errordef=1, print_level=0)
    if 'tol' in kwargs:
        m.tol = kwargs.get('tol')
    else:
        m.tol = 1e-4
    m.migrad()
    params = np.asarray(m.values.values())

    result_dict['chisquare/d.o.f.'] = m.fval / len(x)

    result_dict['method'] = 'migrad'

    if not silent:
        print('chisquare/d.o.f.:', result_dict['chisquare/d.o.f.'])

    if not m.get_fmin().is_valid:
        raise Exception('The minimization procedure did not converge.')

    hess_inv = np.linalg.pinv(jacobian(jacobian(chisqfunc))(params))

    def chisqfunc_compact(d):
        model = func(d[:n_parms], x)
        chisq = jnp.sum(((d[n_parms: n_parms + len(x)] - model) / dy_f) ** 2) + jnp.sum(((d[n_parms + len(x):] - d[:n_parms]) / dp_f) ** 2)
        return chisq

    jac_jac = jacobian(jacobian(chisqfunc_compact))(np.concatenate((params, y_f, p_f)))

    deriv = -hess_inv @ jac_jac[:n_parms, n_parms:]

    result = []
    for i in range(n_parms):
        result.append(derived_observable(lambda x, **kwargs: x[0], [pseudo_Obs(params[i], 0.0, y[0].names[0], y[0].shape[y[0].names[0]])] + list(y) + list(loc_priors), man_grad=[0] + list(deriv[i])))

    result_dict['fit_parameters'] = result
    result_dict['chisquare'] = chisqfunc(np.asarray(params))

    if kwargs.get('resplot') is True:
        residual_plot(x, y, func, result)

    if kwargs.get('qqplot') is True:
        qqplot(x, y, func, result)

    return result_dict if kwargs.get('dict_output') else result


def fit_lin(x, y, **kwargs):
    """Performs a linear fit to y = n + m * x and returns two Obs n, m.

    y has to be a list of Obs, the dvalues of the Obs are used as yerror for the fit.
    x can either be a list of floats in which case no xerror is assumed, or
    a list of Obs, where the dvalues of the Obs are used as xerror for the fit.
    """

    def f(a, x):
        y = a[0] + a[1] * x
        return y

    if all(isinstance(n, Obs) for n in x):
        return odr_fit(x, y, f, **kwargs)
    elif all(isinstance(n, float) or isinstance(n, int) for n in x) or isinstance(x, np.ndarray):
        return standard_fit(x, y, f, **kwargs)
    else:
        raise Exception('Unsupported types for x')


def fit_exp(data, **kwargs):
    """Fit a single exponential to a discrete time series of Obs without errors.

    Keyword arguments
    -----------------
    shift -- specifies the absolute timeslice value of the first entry of data (default 0.0)
             only important if one is interested in the matrix element, for the mass this is irrelevant.
    """
    if 'shift' in kwargs:
        shift = kwargs.get("shift")
    else:
        shift = 0
    length = len(data)
    xsum = 0
    xsum2 = 0
    ysum = 0
    xysum = 0
    for i in range(shift, length + shift):
        xsum += i
        xsum2 += i ** 2
        tmp_log = np.log(np.abs(data[i - shift]))
        ysum += tmp_log
        xysum += i * tmp_log
    res0 = -(length * xysum - xsum * ysum) / (length * xsum2 - xsum * xsum)  # mass
    res1 = np.exp((xsum2 * ysum - xsum * xysum) / (length * xsum2 - xsum * xsum))  # matrix element
    return [res0, res1]


def qqplot(x, o_y, func, p):
    """ Generates a quantile-quantile plot of the fit result which can be used to
        check if the residuals of the fit are gaussian distributed.
    """

    residuals = []
    for i_x, i_y in zip(x, o_y):
        residuals.append((i_y - func(p, i_x)) / i_y.dvalue)
    residuals = sorted(residuals)
    my_y = [o.value for o in residuals]
    probplot = scipy.stats.probplot(my_y)
    my_x = probplot[0][0]
    plt.figure(figsize=(8, 8 / 1.618))
    plt.errorbar(my_x, my_y, fmt='o')
    fit_start = my_x[0]
    fit_stop = my_x[-1]
    samples = np.arange(fit_start, fit_stop, 0.01)
    plt.plot(samples, samples, 'k--', zorder=11, label='Standard normal distribution')
    plt.plot(samples, probplot[1][0] * samples + probplot[1][1], zorder=10, label='Least squares fit, r=' + str(np.around(probplot[1][2], 3)))

    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Ordered Values')
    plt.legend()
    plt.show()


def residual_plot(x, y, func, fit_res):
    """ Generates a plot which compares the fit to the data and displays the corresponding residuals"""
    xstart = x[0] - 0.5
    xstop = x[-1] + 0.5
    x_samples = np.arange(xstart, xstop, 0.01)

    plt.figure(figsize=(8, 8 / 1.618))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], wspace=0.0, hspace=0.0)
    ax0 = plt.subplot(gs[0])
    ax0.errorbar(x, [o.value for o in y], yerr=[o.dvalue for o in y], ls='none', fmt='o', capsize=3, markersize=5, label='Data')
    ax0.plot(x_samples, func([o.value for o in fit_res], x_samples), label='Fit', zorder=10, ls='-', ms=0)
    ax0.set_xticklabels([])
    ax0.set_xlim([xstart, xstop])
    ax0.set_xticklabels([])
    ax0.legend()

    residuals = (np.asarray([o.value for o in y]) - func([o.value for o in fit_res], x)) / np.asarray([o.dvalue for o in y])
    ax1 = plt.subplot(gs[1])
    ax1.plot(x, residuals, 'ko', ls='none', markersize=5)
    ax1.tick_params(direction='out')
    ax1.tick_params(axis="x", bottom=True, top=True, labelbottom=True)
    ax1.axhline(y=0.0, ls='--', color='k')
    ax1.fill_between(x_samples, -1.0, 1.0, alpha=0.1, facecolor='k')
    ax1.set_xlim([xstart, xstop])
    ax1.set_ylabel('Residuals')
    plt.subplots_adjust(wspace=None, hspace=None)
    plt.show()


def covariance_matrix(y):
    """Returns the covariance matrix of y."""
    length = len(y)
    cov = np.zeros((length, length))
    for i, item in enumerate(y):
        for j, jtem in enumerate(y[:i + 1]):
            if i == j:
                cov[i, j] = item.dvalue ** 2
            else:
                cov[i, j] = covariance(item, jtem)
    return cov + cov.T - np.diag(np.diag(cov))


def error_band(x, func, beta):
    """Returns the error band for an array of sample values x, for given fit function func with optimized parameters beta."""
    cov = covariance_matrix(beta)
    if np.any(np.abs(cov - cov.T) > 1000 * np.finfo(np.float64).eps):
        warnings.warn("Covariance matrix is not symmetric within floating point precision", RuntimeWarning)

    deriv = []
    for i, item in enumerate(x):
        deriv.append(np.array(egrad(func)([o.value for o in beta], item)))

    err = []
    for i, item in enumerate(x):
        err.append(np.sqrt(deriv[i] @ cov @ deriv[i]))
    err = np.array(err)

    return err


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

    warnings.warn("New fit functions with exact error propagation are now available as alternative.", DeprecationWarning)

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

    if all(isinstance(n, Obs) for n in x):
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
        res.append(derived_observable(do_the_fit, obs, function=func, xerr=xerr, yerr=yerr, x_constants=x_constants, num_grad=True, n=n, **kwargs))
    return res
