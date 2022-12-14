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

def combined_total_least_squares(x,y,funcs,silent=False,**kwargs):
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

    # number of fit parameters
    n_parms_ls = []
    for key in funcs.keys():
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
            raise RuntimeError("Fit function is not valid.")
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
    
    if 'tol' in kwargs:
        fit_result = iminuit.minimize(chisqfunc, x0,tol=kwargs.get('tol'))
        fit_result = iminuit.minimize(chisqfunc, fit_result.x,tol=kwargs.get('tol'))
    else:
        fit_result = iminuit.minimize(chisqfunc, x0,tol=1e-4)
        fit_result = iminuit.minimize(chisqfunc, fit_result.x,tol=1e-4)
        
    chisquare = fit_result.fun
    
    output.method = 'migrad'
    output.message = fit_result.message

    if x_all.shape[-1] - n_parms > 0:
        output.chisquare = chisqfunc(fit_result.x)
        output.dof = x_all.shape[-1] - n_parms
        output.chisquare_by_dof = output.chisquare/output.dof
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
    
    fitp = fit_result.x
    y_f = [o.value for o in y_all] # y_f is constructed based on the ordered dictionary if the order is changed then the y values are not allocated to the the correct x and func values in the hessian
    dy_f = [o.dvalue for o in y_all] # the same goes for dy_f
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

    result = []
    for i in range(n_parms):
        result.append(pe.derived_observable(lambda x_all, **kwargs: (x_all[0] + np.finfo(np.float64).eps) / (y_all[0].value + np.finfo(np.float64).eps) * fitp[i], list(y_all), man_grad=list(deriv_y[i])))
    
    output.fit_parameters = result
    
    return output