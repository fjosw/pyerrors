import numpy as np
import matplotlib.pyplot as plt
import pyerrors as pe
import pytest


def test_obs_errorbar():
    x_float = np.arange(5)
    x_obs = []
    y_obs = []
    for x in x_float:
        x_obs.append(pe.pseudo_Obs(x, 0.1, "test"))
        y_obs.append(pe.pseudo_Obs(x ** 2, 0.1, "test"))

    for xerr in [2, None]:
        for yerr in [0.1, None]:
            pe.errorbar(x_float, y_obs, marker="x", ms=2, xerr=xerr, yerr=yerr)
            pe.errorbar(x_obs, y_obs, marker="x", ms=2, xerr=xerr, yerr=yerr)

    plt.close('all')


def test_obsval():
    o1 = pe.pseudo_Obs(1, .1, 'test')
    o2 = pe.cov_Obs(1., .1**2, 'test2')
    assert(pe.misc.obsval(o1) == o1.value)
    assert(pe.misc.obsval(o2) == o2.value)

    for o in [None, 1, [1, 2], 'st']:
        assert(pe.misc.obsval(o) == o)
