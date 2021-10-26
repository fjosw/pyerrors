import numpy as np
import pyerrors as pe
import pytest
import time

np.random.seed(0)

def test_bench_mul(benchmark):
    length = 1000
    my_obs = pe.Obs([np.random.rand(length)], ['t1'])

    def mul(x, y):
        return x * y

    benchmark(mul, my_obs, my_obs)


def test_bench_cmul(benchmark):
    length = 1000
    my_obs = pe.CObs(pe.Obs([np.random.rand(length)], ['t1']),
                     pe.Obs([np.random.rand(length)], ['t1']))

    def mul(x, y):
        return x * y

    benchmark(mul, my_obs, my_obs)


def test_bench_matmul(benchmark):
    dim = 4
    my_list = []
    length = 1000
    for i in range(dim ** 2):
        my_list.append(pe.Obs([np.random.rand(length)], ['t1']))
    my_array = np.array(my_list).reshape((dim, dim))

    benchmark(pe.linalg.matmul, my_array, my_array)


def test_bench_cmatmul(benchmark):
    dim = 4
    my_list = []
    length = 1000
    for i in range(dim ** 2):
        my_list.append(pe.CObs(pe.Obs([np.random.rand(length)], ['t1']),
                               pe.Obs([np.random.rand(length)], ['t1'])))
    my_array = np.array(my_list).reshape((dim, dim))

    benchmark(pe.linalg.matmul, my_array, my_array)


def test_bench_gamma_method(benchmark):
    length = 1000
    my_obs = pe.Obs([np.random.rand(length)], ['t1'])
    benchmark(my_obs.gamma_method)
