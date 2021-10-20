import numpy as np
import pyerrors as pe
import pytest

np.random.seed(0)


def test_gamma_matrices():
    for matrix in pe.npr.gamma:
        assert np.allclose(matrix @ matrix, np.identity(4))
        assert np.allclose(matrix, matrix.T.conj())
    assert np.allclose(pe.npr.gamma5, pe.npr.gamma[0] @ pe.npr.gamma[1] @ pe.npr.gamma[2] @ pe.npr.gamma[3])
