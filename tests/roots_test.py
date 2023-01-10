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


def test_root_multi_parameter():
    o1 = pe.pseudo_Obs(1.1, 0.1, "test")
    o2 = pe.pseudo_Obs(1.3, 0.12, "test")

    f2 = lambda x, d: d[0] + d[1] * x

    assert f2(-o1 / o2, [o1, o2]) == 0
    assert pe.find_root([o1, o2], f2) == -o1 / o2
