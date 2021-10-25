import numpy as np
import pyerrors as pe
import pytest

np.random.seed(0)


def test_root_linear():

    def root_function(x, d):
        return x - d

    value = np.random.normal(0, 100)
    my_obs = pe.pseudo_Obs(value, 0.1, 't')
    my_root = pe.roots.find_root(my_obs, root_function)

    assert np.isclose(my_root.value, value)
    difference = my_obs - my_root
    assert difference.is_zero()
