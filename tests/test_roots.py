import numpy as np
import pyerrors as pe
import pytest

def test_find_root():

    def root_function(x, d):
        return x - d

    value = np.random.normal(0, 100)
    my_obs = pe.pseudo_Obs(value, 0.1, 't')
    my_root = pe.roots.find_root(my_obs, root_function)

    assert np.isclose(my_root.value, value)
