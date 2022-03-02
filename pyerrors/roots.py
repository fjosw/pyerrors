import numpy as np
import scipy.optimize
from autograd import jacobian
from .obs import derived_observable


def find_root(d, func, guess=1.0, **kwargs):
    r'''Finds the root of the function func(x, d) where d is an `Obs`.

    Parameters
    -----------------
    d : Obs
        Obs passed to the function.
    func : object
        Function to be minimized. Any numpy functions have to use the autograd.numpy wrapper.
        Example:
        ```python
        import autograd.numpy as anp
        def root_func(x, d):
            return anp.exp(-x ** 2) - d
        ```
    guess : float
        Initial guess for the minimization.

    Returns
    -------
    Obs
        `Obs` valued root of the function.
    '''
    root = scipy.optimize.fsolve(func, guess, d.value)

    # Error propagation as detailed in arXiv:1809.01289
    dx = jacobian(func)(root[0], d.value)
    try:
        da = jacobian(lambda u, v: func(v, u))(d.value, root[0])
    except TypeError:
        raise Exception("It is required to use autograd.numpy instead of numpy within root functions, see the documentation for details.") from None
    deriv = - da / dx

    res = derived_observable(lambda x, **kwargs: (x[0] + np.finfo(np.float64).eps) / (d.value + np.finfo(np.float64).eps) * root[0], [d], man_grad=[deriv])
    return res
