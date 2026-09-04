import pickle
import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from .obs import Obs
from .version import __version__


def _extract_plot_values(data):
    """Extract central values from a sequence of Obs."""
    if any(isinstance(o, Obs) for o in data):
        return [float(o) if isinstance(o, Obs) else o for o in data]
    return data


def _extract_plot_data(data):
    """Extract central values and errors from a sequence of Obs."""
    obs = [o for o in data if isinstance(o, Obs)]
    if obs:
        if not all(hasattr(o, 'e_dvalue') for o in obs):
            [o.gamma_method() for o in obs]
        return _extract_plot_values(data), [o.dvalue if isinstance(o, Obs) else 0 for o in data]
    return data, None


def print_config():
    """Print information about version of python, pyerrors and dependencies."""
    config = {"system": platform.system(),
              "python": platform.python_version(),
              "pyerrors": __version__,
              "numpy": np.__version__,
              "scipy": scipy.__version__,
              "matplotlib": matplotlib.__version__,
              "pandas": pd.__version__}

    for key, value in config.items():
        print(f"{key: <10}\t {value}")


def errorbar(x, y, axes=plt, **kwargs):
    """pyerrors wrapper for the errorbars method of matplotlib

    Parameters
    ----------
    x : list
        A list of x-values which can be Obs.
    y : list
        A list of y-values which can be Obs.
    axes : (matplotlib.pyplot.axes)
        The axes to plot on. default is plt.
    """
    val = {}
    err = {}
    for name, comp in zip(["x", "y"], [x, y], strict=True):
        val[name], err[name] = _extract_plot_data(comp)

        if f"{name}err" in kwargs:
            err[name] = kwargs.get(f"{name}err")
            kwargs.pop(f"{name}err", None)

    axes.errorbar(val["x"], val["y"], xerr=err["x"], yerr=err["y"], **kwargs)


def fill_between(x, y, axes=plt, **kwargs):
    """Plot the uncertainty band of a sequence of Obs.

    Parameters
    ----------
    x : list
        A list of x-values which can be Obs.
    y : list
        A list of y-values which can be Obs. Numeric values without ``yerr``
        produce a zero-width band.
    axes : matplotlib.pyplot or matplotlib.axes.Axes
        The axes to plot on. Default is ``matplotlib.pyplot``.
    yerr : array-like, optional
        Errors used instead of the uncertainties of ``y``.

    Returns
    -------
    matplotlib.collections.FillBetweenPolyCollection
        The plotted uncertainty band.
    """
    x_val = _extract_plot_values(x)
    y_val, y_err = _extract_plot_data(y)
    if "yerr" in kwargs:
        y_err = kwargs.pop("yerr")
    if y_err is None:
        y_err = 0

    y_val = np.asarray(y_val)
    y_err = np.asarray(y_err)
    return axes.fill_between(x_val, y_val - y_err, y_val + y_err, **kwargs)


def fill_betweenx(y, x, axes=plt, **kwargs):
    """Plot the horizontal uncertainty band of a sequence of Obs.

    Parameters
    ----------
    y : list
        A list of y-values which can be Obs.
    x : list
        A list of x-values which can be Obs. Numeric values without ``xerr``
        produce a zero-width band.
    axes : matplotlib.pyplot or matplotlib.axes.Axes
        The axes to plot on. Default is ``matplotlib.pyplot``.
    xerr : array-like, optional
        Errors used instead of the uncertainties of ``x``.

    Returns
    -------
    matplotlib.collections.FillBetweenPolyCollection
        The plotted uncertainty band.
    """
    y_val = _extract_plot_values(y)
    x_val, x_err = _extract_plot_data(x)
    if "xerr" in kwargs:
        x_err = kwargs.pop("xerr")
    if x_err is None:
        x_err = 0

    x_val = np.asarray(x_val)
    x_err = np.asarray(x_err)
    return axes.fill_betweenx(y_val, x_val - x_err, x_val + x_err, **kwargs)


def dump_object(obj, name, **kwargs):
    """Dump object into pickle file.

    Parameters
    ----------
    obj : object
        object to be saved in the pickle file
    name : str
        name of the file
    path : str
        specifies a custom path for the file (default '.')

    Returns
    -------
    None
    """
    if 'path' in kwargs:
        file_name = kwargs.get('path') + '/' + name + '.p'
    else:
        file_name = name + '.p'
    with open(file_name, 'wb') as fb:
        pickle.dump(obj, fb)


def load_object(path):
    """Load object from pickle file.

    Parameters
    ----------
    path : str
        path to the file

    Returns
    -------
    object : Obs
        Loaded Object
    """
    with open(path, 'rb') as file:
        return pickle.load(file)


def pseudo_Obs(value, dvalue, name, samples=1000):
    """Generate an Obs object with given value, dvalue and name for test purposes

    Parameters
    ----------
    value : float
        central value of the Obs to be generated.
    dvalue : float
        error of the Obs to be generated.
    name : str
        name of the ensemble for which the Obs is to be generated.
    samples: int
        number of samples for the Obs (default 1000).

    Returns
    -------
    res : Obs
        Generated Observable
    """
    if dvalue <= 0.0:
        return Obs([np.zeros(samples) + value], [name])
    else:
        for _ in range(100):
            deltas = [np.random.normal(0.0, dvalue * np.sqrt(samples), samples)]  # noqa: NPY002
            deltas -= np.mean(deltas)
            deltas *= dvalue / np.sqrt(np.var(deltas) / samples) / np.sqrt(1 + 3 / samples)
            deltas += value
            res = Obs(deltas, [name])
            res.gamma_method(S=2, tau_exp=0)
            if abs(res.dvalue - dvalue) < 1e-10 * dvalue:
                break

        res._value = float(value)

        return res


def gen_correlated_data(means, cov, name, tau=0.5, samples=1000):
    """ Generate observables with given covariance and autocorrelation times.

    Parameters
    ----------
    means : list
        list containing the mean value of each observable.
    cov : numpy.ndarray
        covariance matrix for the data to be generated.
    name : str
        ensemble name for the data to be geneated.
    tau : float or list
        can either be a real number or a list with an entry for
        every dataset.
    samples : int
        number of samples to be generated for each observable.

    Returns
    -------
    corr_obs : list[Obs]
        Generated observable list
    """

    assert len(means) == cov.shape[-1]
    tau = np.asarray(tau)
    if np.min(tau) < 0.5:
        raise ValueError('All integrated autocorrelations have to be >= 0.5.')

    a = (2 * tau - 1) / (2 * tau + 1)
    rand = np.random.multivariate_normal(np.zeros_like(means), cov * samples, samples)  # noqa: NPY002

    # Normalize samples such that sample variance matches input
    norm = np.array([np.var(o, ddof=1) / samples for o in rand.T])
    rand = rand @ np.diag(np.sqrt(np.diag(cov))) @ np.diag(1 / np.sqrt(norm))

    data = [rand[0]]
    for i in range(1, samples):
        data.append(np.sqrt(1 - a ** 2) * rand[i] + a * data[-1])
    corr_data = np.array(data) - np.mean(data, axis=0) + means
    return [Obs([dat], [name]) for dat in corr_data.T]


def _assert_equal_properties(ol, otype=Obs):
    otype = type(ol[0])
    for o in ol[1:]:
        if not isinstance(o, otype):
            raise TypeError("Wrong data type in list.")
        for attr in ["reweighted", "e_content", "idl"]:
            if hasattr(ol[0], attr):
                if not getattr(ol[0], attr) == getattr(o, attr):
                    raise ValueError(f"All Obs in list have to have the same state '{attr}'.")
