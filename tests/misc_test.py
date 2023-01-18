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

    pe.errorbar(x_float, y_obs, marker="x", ms=2)
    pe.errorbar(x_obs, y_obs, marker="x", ms=2)

