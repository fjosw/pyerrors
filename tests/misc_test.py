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


def test_print_config():
    pe.print_config()


def test_pseudo_Obs_seed_independence():
    # pseudo_Obs now uses a module-local np.random.default_rng() generator,
    # so np.random.seed() no longer controls its output. The per-sample
    # deltas therefore differ between successive calls even with a re-seed,
    # though the normalized value / dvalue still match the requested inputs.
    np.random.seed(0)
    a = pe.pseudo_Obs(1.0, 0.1, "e")
    np.random.seed(0)
    b = pe.pseudo_Obs(1.0, 0.1, "e")

    assert not np.allclose(a.deltas["e"], b.deltas["e"])
    assert np.isclose(a.value, b.value)
    assert np.isclose(a.dvalue, b.dvalue)
