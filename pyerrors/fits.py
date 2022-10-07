import gc
from collections.abc import Sequence
import warnings
import numpy as np
import autograd.numpy as anp
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.odr import ODR, Model, RealData
import iminuit
from autograd import jacobian as auto_jacobian
from autograd import hessian as auto_hessian
from autograd import elementwise_grad as egrad
from numdifftools import Jacobian as num_jacobian
from numdifftools import Hessian as num_hessian
from .obs import Obs, derived_observable, covariance, cov_Obs


class Fit_result(Sequence):
    """Represents fit results.

    Attributes
    ----------
    fit_parameters : list
        results for the individual fit parameters,
        also accessible via indices.
    """

    def __init__(self):
        self.fit_parameters = None

    def __getitem__(self, idx):
        return self.fit_parameters[idx]

    def __len__(self):
        return len(self.fit_parameters)

    def gamma_method(self, **kwargs):
        """Apply the gamma method to all fit parameters"""
        [o.gamma_method(**kwargs) for o in self.fit_parameters]

    def __str__(self):
        my_str = 'Goodness of fit:\n'
        if hasattr(self, 'chisquare_by_dof'):
            my_str += '\u03C7\u00b2/d.o.f. = ' + f'{self.chisquare_by_dof:2.6f}' + '\n'
        elif hasattr(self, 'residual_variance'):
            my_str += 'residual variance = ' + f'{self.residual_variance:2.6f}' + '\n'
        if hasattr(self, 'chisquare_by_expected_chisquare'):
            my_str += '\u03C7\u00b2/\u03C7\u00b2exp  = ' + f'{self.chisquare_by_expected_chisquare:2.6f}' + '\n'
        if hasattr(self, 'p_value'):
            my_str += 'p-value   = ' + f'{self.p_value:2.4f}' + '\n'
        my_str += 'Fit parameters:\n'
        for i_par, par in enumerate(self.fit_parameters):
            my_str += str(i_par) + '\t' + ' ' * int(par >= 0) + str(par).rjust(int(par < 0.0)) + '\n'
        return my_str

    def __repr__(self):
        m = max(map(len, list(self.__dict__.keys()))) + 1
        return '\n'.join([key.rjust(m) + ': ' + repr(value) for key, value in sorted(self.__dict__.items())])


def least_squares(x, y, func, priors=None, silent=False, **kwargs):
    r'''Performs a non-linear fit to y = func(x).

    Parameters
    ----------
    x : list
        list of floats.
    y : list
        list of Obs.
    func : object
        fit function, has to be of the form

        ```python
        import autograd.numpy as anp

        def func(a, x):
            return a[0] + a[1] * x + a[2] * anp.sinh(x)
        ```

        For multiple x values func can be of the form

        ```python
        def func(a, x):
            (x1, x2) = x
            return a[0] * x1 ** 2 + a[1] * x2
        ```

        It is important that all numpy functions refer to autograd.numpy, otherwise the differentiation
        will not work.
    priors : list, optional
        priors has to be a list with an entry for every parameter in the fit. The entries can either be
        Obs (e.g. results from a previous fit) or strings containing a value and an error formatted like
        0.548(23), 500(40) or 0.5(0.4)
    silent : bool, optional
        If true all output to the console is omitted (default False).
    initial_guess : list
        can provide an initial guess for the input parameters. Relevant for
        non-linear fits with many parameters. In case of correlated fits the guess is used to perform
        an uncorrelated fit which then serves as guess for the correlated fit.
    method : str, optional
        can be used to choose an alternative method for the minimization of chisquare.
        The possible methods are the ones which can be used for scipy.optimize.minimize and
        migrad of iminuit. If no method is specified, Levenberg-Marquard is used.
        Reliable alternatives are migrad, Powell and Nelder-Mead.
    correlated_fit : bool
        If True, use the full inverse covariance matrix in the definition of the chisquare cost function.
        For details about how the covariance matrix is estimated see `pyerrors.obs.covariance`.
        In practice the correlation matrix is Cholesky decomposed and inverted (instead of the covariance matrix).
        This procedure should be numerically more stable as the correlation matrix is typically better conditioned (Jacobi preconditioning).
        At the moment this option only works for `prior==None` and when no `method` is given.
    expected_chisquare : bool
        If True estimates the expected chisquare which is
        corrected by effects caused by correlated input data (default False).
    resplot : bool
        If True, a plot which displays fit, data and residuals is generated (default False).
    qqplot : bool
        If True, a quantile-quantile plot of the fit result is generated (default False).
    num_grad : bool
        Use numerical differentation instead of automatic differentiation to perform the error propagation (default False).
    '''
    if priors is not None:
        return _prior_fit(x, y, func, priors, silent=silent, **kwargs)
    else:
        return _standard_fit(x, y, func, silent=silent, **kwargs)


def total_least_squares(x, y, func, silent=False, **kwargs):
    r'''Performs a non-linear fit to y = func(x) and returns a list of Obs corresponding to the fit parameters.

    Parameters
    ----------
    x : list
        list of Obs, or a tuple of lists of Obs
    y : list
        list of Obs. The dvalues of the Obs are used as x- and yerror for the fit.
    func : object
        func has to be of the form

        ```python
        import autograd.numpy as anp

        def func(a, x):
            return a[0] + a[1] * x + a[2] * anp.sinh(x)
        ```

        For multiple x values func can be of the form

        ```python
        def func(a, x):
            (x1, x2) = x
            return a[0] * x1 ** 2 + a[1] * x2
        ```

        It is important that all numpy functions refer to autograd.numpy, otherwise the differentiation
        will not work.
    silent : bool, optional
        If true all output to the console is omitted (default False).
    initial_guess : list
        can provide an initial guess for the input parameters. Relevant for non-linear
        fits with many parameters.
    expected_chisquare : bool
        If true prints the expected chisquare which is
        corrected by effects caused by correlated input data.
        This can take a while as the full correlation matrix
        has to be calculated (default False).
    num_grad : bool
        Use numerical differentation instead of automatic differentiation to perform the error propagation (default False).

    Notes
    -----
    Based on the orthogonal distance regression module of scipy
    '''

    output = Fit_result()

    output.fit_function = func

    x = np.array(x)

    x_shape = x.shape

    if kwargs.get('num_grad') is True:
        jacobian = num_jacobian
        hessian = num_hessian
    else:
        jacobian = auto_jacobian
        hessian = auto_hessian

    if not callable(func):
        raise TypeError('func has to be a function.')

    for i in range(42):
        try:
            func(np.arange(i), x.T[0])
        except TypeError:
            continue
        except IndexError:
            continue
        else:
            break
    else:
        raise RuntimeError("Fit function is not valid.")

    n_parms = i
    if not silent:
        print('Fit with', n_parms, 'parameter' + 's' * (n_parms > 1))

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
            raise Exception('Initial guess does not have the correct length: %d vs. %d' % (len(x0), n_parms))
    else:
        x0 = [1] * n_parms

    data = RealData(x_f, y_f, sx=dx_f, sy=dy_f)
    model = Model(func)
    odr = ODR(data, model, x0, partol=np.finfo(np.float64).eps)
    odr.set_job(fit_type=0, deriv=1)
    out = odr.run()

    output.residual_variance = out.res_var

    output.method = 'ODR'

    output.message = out.stopreason

    output.xplus = out.xplus

    if not silent:
        print('Method: ODR')
        print(*out.stopreason)
        print('Residual variance:', output.residual_variance)

    if out.info > 3:
        raise Exception('The minimization procedure did not converge.')

    m = x_f.size

    def odr_chisquare(p):
        model = func(p[:n_parms], p[n_parms:].reshape(x_shape))
        chisq = anp.sum(((y_f - model) / dy_f) ** 2) + anp.sum(((x_f - p[n_parms:].reshape(x_shape)) / dx_f) ** 2)
        return chisq

    if kwargs.get('expected_chisquare') is True:
        W = np.diag(1 / np.asarray(np.concatenate((dy_f.ravel(), dx_f.ravel()))))

        if kwargs.get('covariance') is not None:
            cov = kwargs.get('covariance')
        else:
            cov = covariance(np.concatenate((y, x.ravel())))

        number_of_x_parameters = int(m / x_f.shape[-1])

        old_jac = jacobian(func)(out.beta, out.xplus)
        fused_row1 = np.concatenate((old_jac, np.concatenate((number_of_x_parameters * [np.zeros(old_jac.shape)]), axis=0)))
        fused_row2 = np.concatenate((jacobian(lambda x, y: func(y, x))(out.xplus, out.beta).reshape(x_f.shape[-1], x_f.shape[-1] * number_of_x_parameters), np.identity(number_of_x_parameters * old_jac.shape[0])))
        new_jac = np.concatenate((fused_row1, fused_row2), axis=1)

        A = W @ new_jac
        P_phi = A @ np.linalg.pinv(A.T @ A) @ A.T
        expected_chisquare = np.trace((np.identity(P_phi.shape[0]) - P_phi) @ W @ cov @ W)
        if expected_chisquare <= 0.0:
            warnings.warn("Negative expected_chisquare.", RuntimeWarning)
            expected_chisquare = np.abs(expected_chisquare)
        output.chisquare_by_expected_chisquare = odr_chisquare(np.concatenate((out.beta, out.xplus.ravel()))) / expected_chisquare
        if not silent:
            print('chisquare/expected_chisquare:',
                  output.chisquare_by_expected_chisquare)

    fitp = out.beta
    try:
        hess = hessian(odr_chisquare)(np.concatenate((fitp, out.xplus.ravel())))
    except TypeError:
        raise Exception("It is required to use autograd.numpy instead of numpy within fit functions, see the documentation for details.") from None

    def odr_chisquare_compact_x(d):
        model = func(d[:n_parms], d[n_parms:n_parms + m].reshape(x_shape))
        chisq = anp.sum(((y_f - model) / dy_f) ** 2) + anp.sum(((d[n_parms + m:].reshape(x_shape) - d[n_parms:n_parms + m].reshape(x_shape)) / dx_f) ** 2)
        return chisq

    jac_jac_x = hessian(odr_chisquare_compact_x)(np.concatenate((fitp, out.xplus.ravel(), x_f.ravel())))

    # Compute hess^{-1} @ jac_jac_x[:n_parms + m, n_parms + m:] using LAPACK dgesv
    try:
        deriv_x = -scipy.linalg.solve(hess, jac_jac_x[:n_parms + m, n_parms + m:])
    except np.linalg.LinAlgError:
        raise Exception("Cannot invert hessian matrix.")

    def odr_chisquare_compact_y(d):
        model = func(d[:n_parms], d[n_parms:n_parms + m].reshape(x_shape))
        chisq = anp.sum(((d[n_parms + m:] - model) / dy_f) ** 2) + anp.sum(((x_f - d[n_parms:n_parms + m].reshape(x_shape)) / dx_f) ** 2)
        return chisq

    jac_jac_y = hessian(odr_chisquare_compact_y)(np.concatenate((fitp, out.xplus.ravel(), y_f)))

    # Compute hess^{-1} @ jac_jac_y[:n_parms + m, n_parms + m:] using LAPACK dgesv
    try:
        deriv_y = -scipy.linalg.solve(hess, jac_jac_y[:n_parms + m, n_parms + m:])
    except np.linalg.LinAlgError:
        raise Exception("Cannot invert hessian matrix.")

    result = []
    for i in range(n_parms):
        result.append(derived_observable(lambda my_var, **kwargs: (my_var[0] + np.finfo(np.float64).eps) / (x.ravel()[0].value + np.finfo(np.float64).eps) * out.beta[i], list(x.ravel()) + list(y), man_grad=list(deriv_x[i]) + list(deriv_y[i])))

    output.fit_parameters = result

    output.odr_chisquare = odr_chisquare(np.concatenate((out.beta, out.xplus.ravel())))
    output.dof = x.shape[-1] - n_parms
    output.p_value = 1 - scipy.stats.chi2.cdf(output.odr_chisquare, output.dof)

    return output


def _prior_fit(x, y, func, priors, silent=False, **kwargs):
    output = Fit_result()

    output.fit_function = func

    x = np.asarray(x)

    if kwargs.get('num_grad') is True:
        hessian = num_hessian
    else:
        hessian = auto_hessian

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
        if isinstance(i_prior, Obs):
            loc_priors.append(i_prior)
        else:
            loc_val, loc_dval = extract_val_and_dval(i_prior)
            loc_priors.append(cov_Obs(loc_val, loc_dval ** 2, '#prior' + str(i_n) + f"_{np.random.randint(2147483647):010d}"))

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
        result.append(derived_observable(lambda x, **kwargs: (x[0] + np.finfo(np.float64).eps) / (y[0].value + np.finfo(np.float64).eps) * params[i], list(y) + list(loc_priors), man_grad=list(deriv[i])))

    output.fit_parameters = result
    output.chisquare = chisqfunc(np.asarray(params))

    if kwargs.get('resplot') is True:
        residual_plot(x, y, func, result)

    if kwargs.get('qqplot') is True:
        qqplot(x, y, func, result)

    return output


def _standard_fit(x, y, func, silent=False, **kwargs):

    output = Fit_result()

    output.fit_function = func

    x = np.asarray(x)

    if kwargs.get('num_grad') is True:
        jacobian = num_jacobian
        hessian = num_hessian
    else:
        jacobian = auto_jacobian
        hessian = auto_hessian

    if x.shape[-1] != len(y):
        raise Exception('x and y input have to have the same length')

    if len(x.shape) > 2:
        raise Exception('Unknown format for x values')

    if not callable(func):
        raise TypeError('func has to be a function.')

    for i in range(42):
        try:
            func(np.arange(i), x.T[0])
        except TypeError:
            continue
        except IndexError:
            continue
        else:
            break
    else:
        raise RuntimeError("Fit function is not valid.")

    n_parms = i

    if not silent:
        print('Fit with', n_parms, 'parameter' + 's' * (n_parms > 1))

    y_f = [o.value for o in y]
    dy_f = [o.dvalue for o in y]

    if np.any(np.asarray(dy_f) <= 0.0):
        raise Exception('No y errors available, run the gamma method first.')

    if 'initial_guess' in kwargs:
        x0 = kwargs.get('initial_guess')
        if len(x0) != n_parms:
            raise Exception('Initial guess does not have the correct length: %d vs. %d' % (len(x0), n_parms))
    else:
        x0 = [0.1] * n_parms

    if kwargs.get('correlated_fit') is True:
        corr = covariance(y, correlation=True, **kwargs)
        covdiag = np.diag(1 / np.asarray(dy_f))
        condn = np.linalg.cond(corr)
        if condn > 0.1 / np.finfo(float).eps:
            raise Exception(f"Cannot invert correlation matrix as its condition number exceeds machine precision ({condn:1.2e})")
        if condn > 1e13:
            warnings.warn("Correlation matrix may be ill-conditioned, condition number: {%1.2e}" % (condn), RuntimeWarning)
        chol = np.linalg.cholesky(corr)
        chol_inv = scipy.linalg.solve_triangular(chol, covdiag, lower=True)

        def chisqfunc_corr(p):
            model = func(p, x)
            chisq = anp.sum(anp.dot(chol_inv, (y_f - model)) ** 2)
            return chisq

    def chisqfunc(p):
        model = func(p, x)
        chisq = anp.sum(((y_f - model) / dy_f) ** 2)
        return chisq

    output.method = kwargs.get('method', 'Levenberg-Marquardt')
    if not silent:
        print('Method:', output.method)

    if output.method != 'Levenberg-Marquardt':
        if output.method == 'migrad':
            fit_result = iminuit.minimize(chisqfunc, x0, tol=1e-4)  # Stopping criterion 0.002 * tol * errordef
            if kwargs.get('correlated_fit') is True:
                fit_result = iminuit.minimize(chisqfunc_corr, fit_result.x, tol=1e-4)  # Stopping criterion 0.002 * tol * errordef
            output.iterations = fit_result.nfev
        else:
            fit_result = scipy.optimize.minimize(chisqfunc, x0, method=kwargs.get('method'), tol=1e-12)
            if kwargs.get('correlated_fit') is True:
                fit_result = scipy.optimize.minimize(chisqfunc_corr, fit_result.x, method=kwargs.get('method'), tol=1e-12)
            output.iterations = fit_result.nit

        chisquare = fit_result.fun

    else:
        if kwargs.get('correlated_fit') is True:
            def chisqfunc_residuals_corr(p):
                model = func(p, x)
                chisq = anp.dot(chol_inv, (y_f - model))
                return chisq

        def chisqfunc_residuals(p):
            model = func(p, x)
            chisq = ((y_f - model) / dy_f)
            return chisq

        fit_result = scipy.optimize.least_squares(chisqfunc_residuals, x0, method='lm', ftol=1e-15, gtol=1e-15, xtol=1e-15)
        if kwargs.get('correlated_fit') is True:
            fit_result = scipy.optimize.least_squares(chisqfunc_residuals_corr, fit_result.x, method='lm', ftol=1e-15, gtol=1e-15, xtol=1e-15)

        chisquare = np.sum(fit_result.fun ** 2)
        if kwargs.get('correlated_fit') is True:
            assert np.isclose(chisquare, chisqfunc_corr(fit_result.x), atol=1e-14)
        else:
            assert np.isclose(chisquare, chisqfunc(fit_result.x), atol=1e-14)

        output.iterations = fit_result.nfev

    if not fit_result.success:
        raise Exception('The minimization procedure did not converge.')

    if x.shape[-1] - n_parms > 0:
        output.chisquare_by_dof = chisquare / (x.shape[-1] - n_parms)
    else:
        output.chisquare_by_dof = float('nan')

    output.message = fit_result.message
    if not silent:
        print(fit_result.message)
        print('chisquare/d.o.f.:', output.chisquare_by_dof)

    if kwargs.get('expected_chisquare') is True:
        if kwargs.get('correlated_fit') is not True:
            W = np.diag(1 / np.asarray(dy_f))
            cov = covariance(y)
            A = W @ jacobian(func)(fit_result.x, x)
            P_phi = A @ np.linalg.pinv(A.T @ A) @ A.T
            expected_chisquare = np.trace((np.identity(x.shape[-1]) - P_phi) @ W @ cov @ W)
            output.chisquare_by_expected_chisquare = chisquare / expected_chisquare
            if not silent:
                print('chisquare/expected_chisquare:',
                      output.chisquare_by_expected_chisquare)

    fitp = fit_result.x
    try:
        if kwargs.get('correlated_fit') is True:
            hess = hessian(chisqfunc_corr)(fitp)
        else:
            hess = hessian(chisqfunc)(fitp)
    except TypeError:
        raise Exception("It is required to use autograd.numpy instead of numpy within fit functions, see the documentation for details.") from None

    if kwargs.get('correlated_fit') is True:
        def chisqfunc_compact(d):
            model = func(d[:n_parms], x)
            chisq = anp.sum(anp.dot(chol_inv, (d[n_parms:] - model)) ** 2)
            return chisq

    else:
        def chisqfunc_compact(d):
            model = func(d[:n_parms], x)
            chisq = anp.sum(((d[n_parms:] - model) / dy_f) ** 2)
            return chisq

    jac_jac = hessian(chisqfunc_compact)(np.concatenate((fitp, y_f)))

    # Compute hess^{-1} @ jac_jac[:n_parms, n_parms:] using LAPACK dgesv
    try:
        deriv = -scipy.linalg.solve(hess, jac_jac[:n_parms, n_parms:])
    except np.linalg.LinAlgError:
        raise Exception("Cannot invert hessian matrix.")

    result = []
    for i in range(n_parms):
        result.append(derived_observable(lambda x, **kwargs: (x[0] + np.finfo(np.float64).eps) / (y[0].value + np.finfo(np.float64).eps) * fit_result.x[i], list(y), man_grad=list(deriv[i])))

    output.fit_parameters = result

    output.chisquare = chisquare
    output.dof = x.shape[-1] - n_parms
    output.p_value = 1 - scipy.stats.chi2.cdf(output.chisquare, output.dof)

    if kwargs.get('resplot') is True:
        residual_plot(x, y, func, result)

    if kwargs.get('qqplot') is True:
        qqplot(x, y, func, result)

    return output


def fit_lin(x, y, **kwargs):
    """Performs a linear fit to y = n + m * x and returns two Obs n, m.

    Parameters
    ----------
    x : list
        Can either be a list of floats in which case no xerror is assumed, or
        a list of Obs, where the dvalues of the Obs are used as xerror for the fit.
    y : list
        List of Obs, the dvalues of the Obs are used as yerror for the fit.
    """

    def f(a, x):
        y = a[0] + a[1] * x
        return y

    if all(isinstance(n, Obs) for n in x):
        out = total_least_squares(x, y, f, **kwargs)
        return out.fit_parameters
    elif all(isinstance(n, float) or isinstance(n, int) for n in x) or isinstance(x, np.ndarray):
        out = least_squares(x, y, f, **kwargs)
        return out.fit_parameters
    else:
        raise Exception('Unsupported types for x')


def qqplot(x, o_y, func, p):
    """Generates a quantile-quantile plot of the fit result which can be used to
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
    plt.plot(samples, probplot[1][0] * samples + probplot[1][1], zorder=10, label='Least squares fit, r=' + str(np.around(probplot[1][2], 3)), marker='', ls='-')

    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Ordered Values')
    plt.legend()
    plt.draw()


def residual_plot(x, y, func, fit_res):
    """ Generates a plot which compares the fit to the data and displays the corresponding residuals"""
    sorted_x = sorted(x)
    xstart = sorted_x[0] - 0.5 * (sorted_x[1] - sorted_x[0])
    xstop = sorted_x[-1] + 0.5 * (sorted_x[-1] - sorted_x[-2])
    x_samples = np.arange(xstart, xstop + 0.01, 0.01)

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
    ax1.axhline(y=0.0, ls='--', color='k', marker=" ")
    ax1.fill_between(x_samples, -1.0, 1.0, alpha=0.1, facecolor='k')
    ax1.set_xlim([xstart, xstop])
    ax1.set_ylabel('Residuals')
    plt.subplots_adjust(wspace=None, hspace=None)
    plt.draw()


def error_band(x, func, beta):
    """Returns the error band for an array of sample values x, for given fit function func with optimized parameters beta."""
    cov = covariance(beta)
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


def ks_test(objects=None):
    """Performs a Kolmogorov–Smirnov test for the p-values of all fit object.

    Parameters
    ----------
    objects : list
        List of fit results to include in the analysis (optional).
    """

    if objects is None:
        obs_list = []
        for obj in gc.get_objects():
            if isinstance(obj, Fit_result):
                obs_list.append(obj)
    else:
        obs_list = objects

    p_values = [o.p_value for o in obs_list]

    bins = len(p_values)
    x = np.arange(0, 1.001, 0.001)
    plt.plot(x, x, 'k', zorder=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('p-value')
    plt.ylabel('Cumulative probability')
    plt.title(str(bins) + ' p-values')

    n = np.arange(1, bins + 1) / np.float64(bins)
    Xs = np.sort(p_values)
    plt.step(Xs, n)
    diffs = n - Xs
    loc_max_diff = np.argmax(np.abs(diffs))
    loc = Xs[loc_max_diff]
    plt.annotate('', xy=(loc, loc), xytext=(loc, loc + diffs[loc_max_diff]), arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))
    plt.draw()

    print(scipy.stats.kstest(p_values, 'uniform'))
