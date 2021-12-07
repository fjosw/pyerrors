import numpy as np
import pyerrors as pe
import pytest

np.random.seed(0)


def test_gamma_matrices():
    for matrix in pe.dirac.gamma:
        assert np.allclose(matrix @ matrix, np.identity(4))
        assert np.allclose(matrix, matrix.T.conj())
    assert np.allclose(pe.dirac.gamma5, pe.dirac.gamma[0] @ pe.dirac.gamma[1] @ pe.dirac.gamma[2] @ pe.dirac.gamma[3])


def test_grid_dirac():
    for gamma in ['Identity',
                  'Gamma5',
                  'GammaX',
                  'GammaY',
                  'GammaZ',
                  'GammaT',
                  'GammaXGamma5',
                  'GammaYGamma5',
                  'GammaZGamma5',
                  'GammaTGamma5',
                  'SigmaXT',
                  'SigmaXY',
                  'SigmaXZ',
                  'SigmaYT',
                  'SigmaYZ',
                  'SigmaZT']:
        pe.dirac.Grid_gamma(gamma)
    with pytest.raises(Exception):
        pe.dirac.Grid_gamma('Not a gamma matrix')
