import numpy as np
import scipy
import pyerrors as pe
import pytest

from autograd import jacobian
from numdifftools import Jacobian as num_jacobian

def test_kn():
    for n in np.arange(0, 10):
        for val in np.linspace(0.1, 7.3, 10):
            assert np.isclose(num_jacobian(lambda x: scipy.special.kn(n, x))(val), jacobian(lambda x: pe.special.kn(n, x))(val), rtol=1e-10, atol=1e-10)
