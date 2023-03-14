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
    chisquare_by_dof : float
        reduced chisquare.
    p_value : float
        p-value of the fit
    t2_p_value : float
        Hotelling t-squared p-value for correlated fits.
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

    gm = gamma_method

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
        if hasattr(self, 't2_p_value'):
            my_str += 't\u00B2p-value = ' + f'{self.t2_p_value:2.4f}' + '\n'
        my_str += 'Fit parameters:\n'
        for i_par, par in enumerate(self.fit_parameters):
            my_str += str(i_par) + '\t' + ' ' * int(par >= 0) + str(par).rjust(int(par < 0.0)) + '\n'
        return my_str

    def __repr__(self):
        m = max(map(len, list(self.__dict__.keys()))) + 1
        return '\n'.join([key.rjust(m) + ': ' + repr(value) for key, value in sorted(self.__dict__.items())])


def least_squares(x, y, func, priors=None, silent=False, **kwargs):
    r'''Performs a non-linear fit to y = func(x).
        ```

    Parameters
    ----------
    For an uncombined fit:

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

    OR For a combined fit:

    x : dict
        dict of lists.
    y : dict
        dict of lists of Obs.
    funcs : dict
        dict of objects
        fit functions have to be of the form (here a[0] is the common fit parameter)
        ```python
        import autograd.numpy as anp
        funcs = {"a": func_a,
                "b": func_b}

        def func_a(a, x):
            return a[1] * anp.exp(-a[0] * x)

        def func_b(a, x):
            return a[2] * anp.exp(-a[0] * x)

        It is important that all numpy functions refer to autograd.numpy, otherwise the differentiation
        will not work.

    priors : dict or list, optional
        priors can either be a dictionary with integer keys and the corresponding priors as values or
        a list with an entry for every parameter in the fit. The entries can either be
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
    tol: float, optional
        can be used (only for combined fits and methods other than Levenberg-Marquard) to set the tolerance for convergence
        to a different value to either speed up convergence at the cost of a larger error on the fitted parameters (and possibly
        invalid estimates for parameter uncertainties) or smaller values to get more accurate parameter values
        The stopping criterion depends on the method, e.g. migrad: edm_max = 0.002 * tol * errordef (EDM criterion: edm < edm_max)
    correlated_fit : bool
        If True, use the full inverse covariance matrix in the definition of the chisquare cost function.
        For details about how the covariance matrix is estimated see `pyerrors.obs.covariance`.
        In practice the correlation matrix is Cholesky decomposed and inverted (instead of the covariance matrix).
        This procedure should be numerically more stable as the correlation matrix is typically better conditioned (Jacobi preconditioning).
    expected_chisquare : bool
        If True estimates the expected chisquare which is
        corrected by effects caused by correlated input data (default False).
    resplot : bool
        If True, a plot which displays fit, data and residuals is generated (default False).
    qqplot : bool
        If True, a quantile-quantile plot of the fit result is generated (default False).
    num_grad : bool
        Use numerical differentation instead of automatic differentiation to perform the error propagation (default False).

    Returns
    -------
    output : Fit_result
        Parameters and information on the fitted result.
    '''
    output = Fit_result()

    if (type(x) == dict and type(y) == dict and type(func) == dict):
        xd = {key: anp.asarray(x[key]) for key in x}
        yd = y
        funcd = func
        output.fit_function = func
    elif (type(x) == dict or type(y) == dict or type(func) == dict):
        raise TypeError("All arguments have to be dictionaries in order to perform a combined fit.")
    else:
        x = np.asarray(x)
        xd = {"": x}
        yd = {"": y}
        funcd = {"": func}
        output.fit_function = func

    if kwargs.get('num_grad') is True:
        jacobian = num_jacobian
        hessian = num_hessian
    else:
        jacobian = auto_jacobian
        hessian = auto_hessian

    key_ls = sorted(list(xd.keys()))

    if sorted(list(yd.keys())) != key_ls:
        raise ValueError('x and y dictionaries do not contain the same keys.')

    if sorted(list(funcd.keys())) != key_ls:
        raise ValueError('x and func dictionaries do not contain the same keys.')

    x_all = np.concatenate([np.array(xd[key]) for key in key_ls])
    y_all = np.concatenate([np.array(yd[key]) for key in key_ls])

    y_f = [o.value for o in y_all]
    dy_f = [o.dvalue for o in y_all]

    if len(x_all.shape) > 2:
        raise ValueError("Unknown format for x values")

    if np.any(np.asarray(dy_f) <= 0.0):
        raise Exception("No y errors available, run the gamma method first.")

    # number of fit parameters
    n_parms_ls = []
    for key in key_ls:
        if not callable(funcd[key]):
            raise TypeError('func (key=' + key + ') is not a function.')
        if np.asarray(xd[key]).shape[-1] != len(yd[key]):
            raise ValueError('x and y input (key=' + key + ') do not have the same length')
        for n_loc in range(100):
            try:
                funcd[key](np.arange(n_loc), x_all.T[0])
            except TypeError:
                continue
            except IndexError:
                continue
            else:
                break
        else:
            raise RuntimeError("Fit function (key=" + key + ") is not valid.")
        n_parms_ls.append(n_loc)

    n_parms = max(n_parms_ls)
    if not silent:
        print('Fit with', n_parms, 'parameter' + 's' * (n_parms > 1))

    if priors is not None:
        if isinstance(priors, (list, np.ndarray)):
            if n_parms != len(priors):
                raise ValueError("'priors' does not have the correct length.")

            loc_priors = []
            for i_n, i_prior in enumerate(priors):
                loc_priors.append(_construct_prior_obs(i_prior, i_n))

            prior_mask = np.arange(len(priors))
            output.priors = loc_priors

        elif isinstance(priors, dict):
            loc_priors = []
            prior_mask = []
            output.priors = {}
            for pos, prior in priors.items():
                if isinstance(pos, int):
                    prior_mask.append(pos)
                else:
                    raise TypeError("Prior position needs to be an integer.")
                loc_priors.append(_construct_prior_obs(prior, pos))

                output.priors[pos] = loc_priors[-1]
            if max(prior_mask) >= n_parms:
                raise ValueError("Prior position out of range.")
        else:
            raise TypeError("Unkown type for `priors`.")

        p_f = [o.value for o in loc_priors]
        dp_f = [o.dvalue for o in loc_priors]
        if np.any(np.asarray(dp_f) <= 0.0):
            raise Exception("No prior errors available, run the gamma method first.")
    else:
        p_f = dp_f = np.array([])
        prior_mask = []
        loc_priors = []

    if 'initial_guess' in kwargs:
        x0 = kwargs.get('initial_guess')
        if len(x0) != n_parms:
            raise ValueError('Initial guess does not have the correct length: %d vs. %d' % (len(x0), n_parms))
    else:
        x0 = [0.1] * n_parms

    if priors is None:
        def general_chisqfunc_uncorr(p, ivars, pr):
            model = anp.concatenate([anp.array(funcd[key](p, xd[key])).reshape(-1) for key in key_ls])
            return (ivars - model) / dy_f
    else:
        def general_chisqfunc_uncorr(p, ivars, pr):
            model = anp.concatenate([anp.array(funcd[key](p, xd[key])).reshape(-1) for key in key_ls])
            return anp.concatenate(((ivars - model) / dy_f, (p[prior_mask] - pr) / dp_f))

    def chisqfunc_uncorr(p):
        return anp.sum(general_chisqfunc_uncorr(p, y_f, p_f) ** 2)

    if kwargs.get('correlated_fit') is True:
        corr = covariance(y_all, correlation=True, **kwargs)
        covdiag = np.diag(1 / np.asarray(dy_f))
        condn = np.linalg.cond(corr)
        if condn > 0.1 / np.finfo(float).eps:
            raise Exception(f"Cannot invert correlation matrix as its condition number exceeds machine precision ({condn:1.2e})")
        if condn > 1e13:
            warnings.warn("Correlation matrix may be ill-conditioned, condition number: {%1.2e}" % (condn), RuntimeWarning)
        chol = np.linalg.cholesky(corr)
        chol_inv = scipy.linalg.solve_triangular(chol, covdiag, lower=True)

        def general_chisqfunc(p, ivars, pr):
            model = anp.concatenate([anp.array(funcd[key](p, xd[key])).reshape(-1) for key in key_ls])
            return anp.concatenate((anp.dot(chol_inv, (ivars - model)), (p[prior_mask] - pr) / dp_f))

        def chisqfunc(p):
            return anp.sum(general_chisqfunc(p, y_f, p_f) ** 2)
    else:
        general_chisqfunc = general_chisqfunc_uncorr
        chisqfunc = chisqfunc_uncorr

    output.method = kwargs.get('method', 'Levenberg-Marquardt')
    if not silent:
        print('Method:', output.method)

    if output.method != 'Levenberg-Marquardt':
        if output.method == 'migrad':
            tolerance = 1e-4  # default value of 1e-1 set by iminuit can be problematic
            if 'tol' in kwargs:
                tolerance = kwargs.get('tol')
            fit_result = iminuit.minimize(chisqfunc_uncorr, x0, tol=tolerance)  # Stopping criterion 0.002 * tol * errordef
            if kwargs.get('correlated_fit') is True:
                fit_result = iminuit.minimize(chisqfunc, fit_result.x, tol=tolerance)
            output.iterations = fit_result.nfev
        else:
            tolerance = 1e-12
            if 'tol' in kwargs:
                tolerance = kwargs.get('tol')
            fit_result = scipy.optimize.minimize(chisqfunc_uncorr, x0, method=kwargs.get('method'), tol=tolerance)
            if kwargs.get('correlated_fit') is True:
                fit_result = scipy.optimize.minimize(chisqfunc, fit_result.x, method=kwargs.get('method'), tol=tolerance)
            output.iterations = fit_result.nit

        chisquare = fit_result.fun

    else:
        if 'tol' in kwargs:
            print('tol cannot be set for Levenberg-Marquardt')

        def chisqfunc_residuals_uncorr(p):
            return general_chisqfunc_uncorr(p, y_f, p_f)

        fit_result = scipy.optimize.least_squares(chisqfunc_residuals_uncorr, x0, method='lm', ftol=1e-15, gtol=1e-15, xtol=1e-15)
        if kwargs.get('correlated_fit') is True:

            def chisqfunc_residuals(p):
                return general_chisqfunc(p, y_f, p_f)

            fit_result = scipy.optimize.least_squares(chisqfunc_residuals, fit_result.x, method='lm', ftol=1e-15, gtol=1e-15, xtol=1e-15)

        chisquare = np.sum(fit_result.fun ** 2)
        assert np.isclose(chisquare, chisqfunc(fit_result.x), atol=1e-14)

        output.iterations = fit_result.nfev

    if not fit_result.success:
        raise Exception('The minimization procedure did not converge.')

    output.chisquare = chisquare
    output.dof = x_all.shape[-1] - n_parms + len(loc_priors)
    output.p_value = 1 - scipy.stats.chi2.cdf(output.chisquare, output.dof)
    if output.dof > 0:
        output.chisquare_by_dof = output.chisquare / output.dof
    else:
        output.chisquare_by_dof = float('nan')

    output.message = fit_result.message
    if not silent:
        print(fit_result.message)
        print('chisquare/d.o.f.:', output.chisquare_by_dof)
        print('fit parameters', fit_result.x)

    def prepare_hat_matrix():
        hat_vector = []
        for key in key_ls:
            if (len(xd[key]) != 0):
                hat_vector.append(jacobian(funcd[key])(fit_result.x, xd[key]))
        hat_vector = [item for sublist in hat_vector for item in sublist]
        return hat_vector

    if kwargs.get('expected_chisquare') is True:
        if kwargs.get('correlated_fit') is not True:
            W = np.diag(1 / np.asarray(dy_f))
            cov = covariance(y_all)
            hat_vector = prepare_hat_matrix()
            A = W @ hat_vector
            P_phi = A @ np.linalg.pinv(A.T @ A) @ A.T
            expected_chisquare = np.trace((np.identity(x_all.shape[-1]) - P_phi) @ W @ cov @ W)
            output.chisquare_by_expected_chisquare = output.chisquare / expected_chisquare
            if not silent:
                print('chisquare/expected_chisquare:', output.chisquare_by_expected_chisquare)

    fitp = fit_result.x

    try:
        hess = hessian(chisqfunc)(fitp)
    except TypeError:
        raise Exception("It is required to use autograd.numpy instead of numpy within fit functions, see the documentation for details.") from None

    len_y = len(y_f)

    def chisqfunc_compact(d):
        return anp.sum(general_chisqfunc(d[:n_parms], d[n_parms: n_parms + len_y], d[n_parms + len_y:]) ** 2)

    jac_jac_y = hessian(chisqfunc_compact)(np.concatenate((fitp, y_f, p_f)))

    # Compute hess^{-1} @ jac_jac_y[:n_parms + m, n_parms + m:] using LAPACK dgesv
    try:
        deriv_y = -scipy.linalg.solve(hess, jac_jac_y[:n_parms, n_parms:])
    except np.linalg.LinAlgError:
        raise Exception("Cannot invert hessian matrix.")

    result = []
    for i in range(n_parms):
        result.append(derived_observable(lambda x_all, **kwargs: (x_all[0] + np.finfo(np.float64).eps) / (y_all[0].value + np.finfo(np.float64).eps) * fitp[i], list(y_all) + loc_priors, man_grad=list(deriv_y[i])))

    output.fit_parameters = result

    # Hotelling t-squared p-value for correlated fits.
    if kwargs.get('correlated_fit') is True:
        n_cov = np.min(np.vectorize(lambda x_all: x_all.N)(y_all))
        output.t2_p_value = 1 - scipy.stats.f.cdf((n_cov - output.dof) / (output.dof * (n_cov - 1)) * output.chisquare,
                                                  output.dof, n_cov - output.dof)

    if kwargs.get('resplot') is True:
        for key in key_ls:
            residual_plot(xd[key], yd[key], funcd[key], result, title=key)

    if kwargs.get('qqplot') is True:
        for key in key_ls:
            qqplot(xd[key], yd[key], funcd[key], result, title=key)

    return output


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
    Based on the orthogonal distance regression module of scipy.

    Returns
    -------
    output : Fit_result
        Parameters and information on the fitted result.
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


def fit_lin(x, y, **kwargs):
    """Performs a linear fit to y = n + m * x and returns two Obs n, m.

    Parameters
    ----------
    x : list
        Can either be a list of floats in which case no xerror is assumed, or
        a list of Obs, where the dvalues of the Obs are used as xerror for the fit.
    y : list
        List of Obs, the dvalues of the Obs are used as yerror for the fit.

    Returns
    -------
    fit_parameters : list[Obs]
        LIist of fitted observables.
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
        raise TypeError('Unsupported types for x')


def qqplot(x, o_y, func, p, title=""):
    """Generates a quantile-quantile plot of the fit result which can be used to
       check if the residuals of the fit are gaussian distributed.

    Returns
    -------
    None
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
    plt.legend(title=title)
    plt.draw()


def residual_plot(x, y, func, fit_res, title=""):
    """Generates a plot which compares the fit to the data and displays the corresponding residuals

    For uncorrelated data the residuals are expected to be distributed ~N(0,1).

    Returns
    -------
    None
    """
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
    ax0.legend(title=title)

    residuals = (np.asarray([o.value for o in y]) - func([o.value for o in fit_res], np.asarray(x))) / np.asarray([o.dvalue for o in y])
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
    """Calculate the error band for an array of sample values x, for given fit function func with optimized parameters beta.

    Returns
    -------
    err : np.array(Obs)
        Error band for an array of sample values x
    """
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

    Returns
    -------
    None
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


def _extract_val_and_dval(string):
    split_string = string.split('(')
    if '.' in split_string[0] and '.' not in split_string[1][:-1]:
        factor = 10 ** -len(split_string[0].partition('.')[2])
    else:
        factor = 1
    return float(split_string[0]), float(split_string[1][:-1]) * factor


def _construct_prior_obs(i_prior, i_n):
    if isinstance(i_prior, Obs):
        return i_prior
    elif isinstance(i_prior, str):
        loc_val, loc_dval = _extract_val_and_dval(i_prior)
        return cov_Obs(loc_val, loc_dval ** 2, '#prior' + str(i_n) + f"_{np.random.randint(2147483647):010d}")
    else:
        raise TypeError("Prior entries need to be 'Obs' or 'str'.")
