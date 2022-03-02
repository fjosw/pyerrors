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
    assert np.isclose(my_root.value, my_root.r_values['t'])
    difference = my_obs - my_root
    assert difference.is_zero()


def test_root_linear_idl():

    def root_function(x, d):
        return x - d

    my_obs = pe.Obs([np.random.rand(50)], ['t'], idl=[range(20, 120, 2)])
    my_root = pe.roots.find_root(my_obs, root_function)

    difference = my_obs - my_root
    assert difference.is_zero()


def test_root_no_autograd():

    def root_function(x, d):
        return x - np.log(np.exp(d))

    value = np.random.normal(0, 100)
    my_obs = pe.pseudo_Obs(value, 0.1, 't')

    with pytest.raises(Exception):
        my_root = pe.roots.find_root(my_obs, root_function)
