import numpy as np
from .obs import derived_observable, Obs
from autograd import jacobian
from scipy.integrate import quad as squad


def quad(func, p, a, b, **kwargs):
    '''Performs a (one-dimensional) numeric integration of f(p, x) from a to b.

    The integration is performed using scipy.integrate.quad().
    All parameters that can be passed to scipy.integrate.quad may also be passed to this function.
    The output is the same as for scipy.integrate.quad, the first element being an Obs.

    Parameters
    ----------
    func : object
        function to integrate, has to be of the form

        ```python
        import autograd.numpy as anp

        def func(p, x):
            return p[0] + p[1] * x + p[2] * anp.sinh(x)
        ```
        where x is the integration variable.
    p : list of floats or Obs
        parameters of the function func.
    a: float or Obs
        Lower limit of integration (use -numpy.inf for -infinity).
    b: float or Obs
        Upper limit of integration (use -numpy.inf for -infinity).
    All parameters of scipy.integrate.quad

    Returns
    -------
    y : Obs
        The integral of func from `a` to `b`.
    abserr : float
        An estimate of the absolute error in the result.
    infodict : dict
        A dictionary containing additional information.
        Run scipy.integrate.quad_explain() for more information.
    message
        A convergence message.
    explain
        Appended only with 'cos' or 'sin' weighting and infinite
        integration limits, it contains an explanation of the codes in
        infodict['ierlst']
    '''

    Np = len(p)
    isobs = [True if isinstance(pi, Obs) else False for pi in p]
    pval = np.array([p[i].value if isobs[i] else p[i] for i in range(Np)],)
    pobs = [p[i] for i in range(Np) if isobs[i]]

    bounds = [a, b]
    isobs_b = [True if isinstance(bi, Obs) else False for bi in bounds]
    bval = np.array([bounds[i].value if isobs_b[i] else bounds[i] for i in range(2)])
    bobs = [bounds[i] for i in range(2) if isobs_b[i]]
    bsign = [-1, 1]

    ifunc = np.vectorize(lambda x: func(pval, x))

    intpars = squad.__code__.co_varnames[3:3 + len(squad.__defaults__)]
    ikwargs = {k: kwargs[k] for k in intpars if k in kwargs}

    integration_result = squad(ifunc, bval[0], bval[1], **ikwargs)
    val = integration_result[0]

    jac = jacobian(func)

    derivint = []
    for i in range(Np):
        if isobs[i]:
            ifunc = np.vectorize(lambda x: jac(pval, x)[i])
            derivint.append(squad(ifunc, bounds[0], bounds[1], **ikwargs)[0])

    for i in range(2):
        if isobs_b[i]:
            derivint.append(bsign[i] * func(pval, bval[i]))

    if len(derivint) == 0:
        return integration_result

    res = derived_observable(lambda x, **kwargs: 0 * (x[0] + np.finfo(np.float64).eps) * (pval[0] + np.finfo(np.float64).eps) + val, pobs + bobs, man_grad=derivint)

    return (res, *integration_result[1:])
