import numpy as np
import pyerrors as pe
import pytest

np.random.seed(0)


def test_gamma_matrices():
    for matrix in pe.dirac.gamma:
        assert np.allclose(matrix @ matrix, np.identity(4))
        assert np.allclose(matrix, matrix.T.conj())
    assert np.allclose(pe.dirac.gamma5, pe.dirac.gamma[0] @ pe.dirac.gamma[1] @ pe.dirac.gamma[2] @ pe.dirac.gamma[3])
