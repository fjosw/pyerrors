import iminuit
import autograd.numpy as anp
from autograd import jacobian
from pyerrors.fits import Fit_result
import numpy as np
import pyerrors as pe
from autograd import jacobian as auto_jacobian
from autograd import hessian as auto_hessian
from autograd import elementwise_grad as egrad
from numdifftools import Jacobian as num_jacobian
from numdifftools import Hessian as num_hessian
import scipy.optimize
import scipy.stats

def combined_fit(x,y,funcs,silent=False,**kwargs):
    r'''Performs a combined non-linear fit.
    Parameters
    ----------
    x : ordered dict
        dict of lists.
    y : ordered dict
        dict of lists of Obs.
    funcs : ordered dict
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
        ```
        It is important that all numpy functions refer to autograd.numpy, otherwise the differentiation
        will not work.
    silent : bool, optional
        If true all output to the console is omitted (default False).
    initial_guess : list
        can provide an initial guess for the input parameters. Relevant for
        non-linear fits with many parameters.
    num_grad : bool
        Use numerical differentation instead of automatic differentiation to perform the error propagation (default False).
    '''
  
    output = Fit_result()
    output.fit_function = funcs
    
    if kwargs.get('num_grad') is True:
        jacobian = num_jacobian
        hessian = num_hessian
    else:
        jacobian = auto_jacobian
        hessian = auto_hessian
  
    x_all = []
    y_all = []
    for key in x.keys():
        x_all+=x[key]
        y_all+=y[key]
    
    x_all = np.asarray(x_all)
    
    if len(x_all.shape) > 2:
        raise Exception('Unknown format for x values')
    
    # number of fit parameters
    n_parms_ls = []
    for key in funcs.keys():
        if not callable(funcs[key]):
            raise TypeError('func (key='+ key + ') is not a function.')
        if len(x[key]) != len(y[key]):
            raise Exception('x and y input (key='+ key + ') do not have the same length')
        for i in range(42):
            try:
                funcs[key](np.arange(i), x_all.T[0])
            except TypeError:
                continue
            except IndexError:
                continue
            else:
                break
        else:
            raise RuntimeError("Fit function (key="+ key + ") is not valid.")
        n_parms = i
        n_parms_ls.append(n_parms)
    n_parms = max(n_parms_ls)
    if not silent:
        print('Fit with', n_parms, 'parameter' + 's' * (n_parms > 1))
        
    if 'initial_guess' in kwargs:
        x0 = kwargs.get('initial_guess')
        if len(x0) != n_parms:
            raise Exception('Initial guess does not have the correct length: %d vs. %d' % (len(x0), n_parms))
    else:
        x0 = [0.1] * n_parms
        
    def chisqfunc(p):
        chisq = 0.0
        for key in funcs.keys():
            x_array = np.asarray(x[key])
            model = anp.array(funcs[key](p,x_array))
            y_obs = y[key]
            y_f = [o.value for o in y_obs]
            dy_f = [o.dvalue for o in y_obs]
            C_inv =  np.diag(np.diag(np.ones((len(x_array),len(x_array)))))/dy_f/dy_f
            chisq += anp.sum((y_f - model)@ C_inv @(y_f - model))
        return chisq
    
    output.method = kwargs.get('method', 'Levenberg-Marquardt')
    if not silent:
        print('Method:', output.method)
        
    if output.method == 'migrad':
        tolerance = 1e-4
        if 'tol' in kwargs:
            tolerance = kwargs.get('tol')
        fit_result = iminuit.minimize(chisqfunc, x0, tol=tolerance) # Stopping criterion 0.002 * tol * errordef
        output.iterations = fit_result.nfev
    else:
        tolerance = 1e-12
        if 'tol' in kwargs:
            tolerance = kwargs.get('tol')
        fit_result = scipy.optimize.minimize(chisqfunc, x0, method=kwargs.get('method'), tol=tolerance)
        output.iterations = fit_result.nit

    chisquare = fit_result.fun
    output.message = fit_result.message
    
    if not fit_result.success:
        raise Exception('The minimization procedure did not converge.')

    if x_all.shape[-1] - n_parms > 0:
        output.chisquare = chisqfunc(fit_result.x)
        output.dof = x_all.shape[-1] - n_parms
        output.chisquare_by_dof = output.chisquare/output.dof
        output.p_value = 1 - scipy.stats.chi2.cdf(output.chisquare, output.dof)
    else:
        output.chisquare_by_dof = float('nan')
    
    if not silent:
        print(fit_result.message)
        print('chisquare/d.o.f.:', output.chisquare_by_dof )
        print('fit parameters',fit_result.x)
        
    # use ordered dicts so the data and fit parameters can be mapped correctly 
    def chisqfunc_compact(d):
        chisq = 0.0
        list_tmp = []
        c1 = 0
        c2 = 0
        for key in funcs.keys():
            x_array = np.asarray(x[key])
            c2+=len(x_array)
            model = anp.array(funcs[key](d[:n_parms],x_array))
            y_obs = y[key]
            y_f = [o.value for o in y_obs]
            dy_f = [o.dvalue for o in y_obs]
            C_inv =  np.diag(np.diag(np.ones((len(x_array),len(x_array)))))/dy_f/dy_f
            list_tmp.append(anp.sum((d[n_parms+c1:n_parms+c2]- model)@ C_inv @(d[n_parms+c1:n_parms+c2]- model)))
            c1+=len(x_array)
        chisq = anp.sum(list_tmp)
        return chisq
    
    def prepare_hat_matrix(): # should be cross-checked again
        hat_vector = []
        for key in funcs.keys():
            x_array = np.asarray(x[key])
            if (len(x_array)!= 0):
                hat_vector.append(anp.array(jacobian(funcs[key])(fit_result.x, x_array)))
        hat_vector = [item for sublist in hat_vector for item in sublist]
        return hat_vector
    
    fitp = fit_result.x
    y_f = [o.value for o in y_all] # y_f is constructed based on the ordered dictionary if the order is changed then the y values are not allocated to the the correct x and func values in the hessian
    dy_f = [o.dvalue for o in y_all] # the same goes for dy_f
    
    if np.any(np.asarray(dy_f) <= 0.0):
        raise Exception('No y errors available, run the gamma method first.')
        
    try:
        hess = hessian(chisqfunc)(fitp)
    except TypeError:
        raise Exception("It is required to use autograd.numpy instead of numpy within fit functions, see the documentation for details.") from None
        
    jac_jac_y = hessian(chisqfunc_compact)(np.concatenate((fitp, y_f)))

    # Compute hess^{-1} @ jac_jac_y[:n_parms + m, n_parms + m:] using LAPACK dgesv
    try:
        deriv_y = -scipy.linalg.solve(hess, jac_jac_y[:n_parms, n_parms:])
    except np.linalg.LinAlgError:
        raise Exception("Cannot invert hessian matrix.")
        
        
    if kwargs.get('expected_chisquare') is True:
        if kwargs.get('correlated_fit') is not True:
            W = np.diag(1 / np.asarray(dy_f))
            cov = covariance(y_all)
            hat_vector = prepare_hat_matrix()
            A = W @ hat_vector #hat_vector = 'jacobian(func)(fit_result.x, x)'
            P_phi = A @ np.linalg.pinv(A.T @ A) @ A.T
            expected_chisquare = np.trace((np.identity(x.shape[-1]) - P_phi) @ W @ cov @ W)
            output.chisquare_by_expected_chisquare = chisquare / expected_chisquare
            if not silent:
                print('chisquare/expected_chisquare:', output.chisquare_by_expected_chisquare)


    result = []
    for i in range(n_parms):
        result.append(pe.derived_observable(lambda x_all, **kwargs: (x_all[0] + np.finfo(np.float64).eps) / (y_all[0].value + np.finfo(np.float64).eps) * fitp[i], list(y_all), man_grad=list(deriv_y[i])))
    
    output.fit_parameters = result
    
    return output