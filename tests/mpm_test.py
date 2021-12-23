import numpy as np
import pyerrors as pe
import pytest

np.random.seed(0)


def test_mpm():
    corr_content = []
    for t in range(8):
        f = 0.8 * np.exp(-0.4 * t)
        corr_content.append(pe.pseudo_Obs(np.random.normal(f, 1e-2 * f), 1e-2 * f, 't'))

    res = pe.mpm.matrix_pencil_method(corr_content)
