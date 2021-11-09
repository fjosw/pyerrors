import scipy.optimize
from autograd import jacobian
from .obs import derived_observable, pseudo_Obs


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
    da = jacobian(lambda u, v: func(v, u))(d.value, root[0])
    deriv = - da / dx

    return derived_observable(lambda x, **kwargs: x[0], [pseudo_Obs(root, 0.0, d.names[0], d.shape[d.names[0]]), d], man_grad=[0, deriv])
