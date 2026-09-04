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


def test_obs_fill_between():
    x = np.arange(5)
    y_obs = [pe.pseudo_Obs(value, 0.1, "test") for value in x]

    fig, ax = plt.subplots()
    band = pe.fill_between(x, y_obs, axes=ax, alpha=0.3)
    assert band.axes is ax

    pe.fill_between(x, x, yerr=0.2)
    pe.fill_between(x, x)

    plt.close('all')


def test_print_config():
    pe.print_config()
