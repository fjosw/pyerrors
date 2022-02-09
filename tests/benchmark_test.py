import numpy as np
import pyerrors as pe
import pytest

np.random.seed(0)

length = 1000

def mul(x, y):
    return x * y


def test_b_mul(benchmark):
    my_obs = pe.Obs([np.random.rand(length)], ['t1'])

    benchmark(mul, my_obs, my_obs)


def test_b_cmul(benchmark):
    my_obs = pe.CObs(pe.Obs([np.random.rand(length)], ['t1']),
                     pe.Obs([np.random.rand(length)], ['t1']))

    benchmark(mul, my_obs, my_obs)


def test_b_matmul(benchmark):
    dim = 4
    my_list = []
    for i in range(dim ** 2):
        my_list.append(pe.Obs([np.random.rand(length)], ['t1']))
    my_array = np.array(my_list).reshape((dim, dim))

    benchmark(pe.linalg.matmul, my_array, my_array)

def test_b_cmatmul(benchmark):
    dim = 4
    my_list = []
    for i in range(dim ** 2):
        my_list.append(pe.CObs(pe.Obs([np.random.rand(length)], ['t1']),
                               pe.Obs([np.random.rand(length)], ['t1'])))
    my_array = np.array(my_list).reshape((dim, dim))

    benchmark(pe.linalg.matmul, my_array, my_array)


def test_b_gamma(benchmark):
    my_obs = pe.Obs([np.random.rand(length)], ['t1'])
    benchmark(my_obs.gamma_method)
