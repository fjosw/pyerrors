import autograd.numpy as np
import os
import copy
import matplotlib.pyplot as plt
import pyerrors as pe
import pytest

np.random.seed(0)


def test_Obs_exceptions():
    with pytest.raises(Exception):
        pe.Obs([np.random.rand(10)], ['1', '2'])
    with pytest.raises(Exception):
        pe.Obs([np.random.rand(10)], ['1'], idl=[])
    with pytest.raises(Exception):
        pe.Obs([np.random.rand(10), np.random.rand(10)], ['1', '1'])
    with pytest.raises(Exception):
        pe.Obs([np.random.rand(10), np.random.rand(10)], ['1', 1])
    with pytest.raises(Exception):
        pe.Obs([np.random.rand(10)], [1])
    with pytest.raises(Exception):
        pe.Obs([np.random.rand(4)], ['name'])
    with pytest.raises(Exception):
        pe.Obs([np.random.rand(5)], ['1'], idl=[[5, 3, 2 ,4 ,1]])
    with pytest.raises(Exception):
        pe.Obs([np.random.rand(5)], ['1'], idl=['t'])
    with pytest.raises(Exception):
        pe.Obs([np.random.rand(5)], ['1'], idl=[range(1, 8)])

    my_obs = pe.Obs([np.random.rand(6)], ['name'])
    my_obs._value = 0.0
    my_obs.details()
    with pytest.raises(Exception):
        my_obs.plot_tauint()
    with pytest.raises(Exception):
        my_obs.plot_rho()
    with pytest.raises(Exception):
        my_obs.plot_rep_dist()
    with pytest.raises(Exception):
        my_obs.plot_piechart()
    with pytest.raises(Exception):
        my_obs.gamma_method(S='2.3')
    with pytest.raises(Exception):
        my_obs.gamma_method(tau_exp=2.3)
    my_obs.gamma_method()
    my_obs.details()
    my_obs.plot_rep_dist()

    my_obs += pe.Obs([np.random.rand(6)], ['name2|r1'], idl=[[1, 3, 4, 5, 6, 7]])
    my_obs += pe.Obs([np.random.rand(6)], ['name2|r2'])
    my_obs.gamma_method()
    my_obs.details()

    obs = pe.Obs([np.random.normal(1.0, 0.1, 100)], ['t'])
    one = obs / obs
    one.gamma_method()
    with pytest.raises(Exception):
        one.plot_piechart()
    plt.close('all')

def test_dump():
    value = np.random.normal(5, 10)
    dvalue = np.abs(np.random.normal(0, 1))
    test_obs = pe.pseudo_Obs(value, dvalue, 't')
    test_obs.dump('test_dump', datatype="pickle", path=".")
    test_obs.dump('test_dump', datatype="pickle")
    new_obs = pe.load_object('test_dump.p')
    os.remove('test_dump.p')
    assert test_obs == new_obs
    test_obs.dump('test_dump', dataype="json.gz", path=".")
    test_obs.dump('test_dump', dataype="json.gz")
    new_obs = pe.input.json.load_json("test_dump")
    os.remove('test_dump.json.gz')
    assert test_obs == new_obs


def test_comparison():
    value1 = np.random.normal(0, 100)
    test_obs1 = pe.pseudo_Obs(value1, 0.1, 't')
    value2 = np.random.normal(0, 100)
    test_obs2 = pe.pseudo_Obs(value2, 0.1, 't')
    assert (value1 > value2) == (test_obs1 > test_obs2)
    assert (value1 < value2) == (test_obs1 < test_obs2)
    assert (value1 >= value2) == (test_obs1 >= test_obs2)
    assert (value1 <= value2) == (test_obs1 <= test_obs2)
    assert test_obs1 >= test_obs1
    assert test_obs2 <= test_obs2
    assert test_obs1 == test_obs1
    assert test_obs2 == test_obs2
    assert test_obs1 - test_obs1 == 0.0
    assert test_obs1 / test_obs1 == 1.0
    assert test_obs1 != value1
    assert test_obs2 != value2
    assert test_obs1 != test_obs2
    assert test_obs2 != test_obs1
    assert +test_obs1 == test_obs1
    assert -test_obs1 == 0 - test_obs1


def test_function_overloading():
    a = pe.pseudo_Obs(17, 2.9, 'e1')
    b = pe.pseudo_Obs(4, 0.8, 'e1')

    fs = [lambda x: x[0] + x[1], lambda x: x[1] + x[0], lambda x: x[0] - x[1], lambda x: x[1] - x[0],
          lambda x: x[0] * x[1], lambda x: x[1] * x[0], lambda x: x[0] / x[1], lambda x: x[1] / x[0],
          lambda x: np.exp(x[0]), lambda x: np.sin(x[0]), lambda x: np.cos(x[0]), lambda x: np.tan(x[0]),
          lambda x: np.log(x[0]), lambda x: np.sqrt(np.abs(x[0])),
          lambda x: np.sinh(x[0]), lambda x: np.cosh(x[0]), lambda x: np.tanh(x[0])]

    for i, f in enumerate(fs):
        t1 = f([a, b])
        t2 = pe.derived_observable(f, [a, b])
        c = t2 - t1
        assert c.is_zero()

    assert np.log(np.exp(b)) == b
    assert np.exp(np.log(b)) == b
    assert np.sqrt(b ** 2) == b
    assert np.sqrt(b) ** 2 == b

    np.arcsin(1 / b)
    np.arccos(1 / b)
    np.arctan(1 / b)
    np.arctanh(1 / b)
    np.sinc(1 / b)

    b ** b
    0.5 ** b
    b ** 0.5


def test_overloading_vectorization():
    a = np.random.randint(1, 100, 10)
    b = pe.pseudo_Obs(4, 0.8, 't')

    assert [o.value for o in a * b] == [o.value for o in b * a]
    assert [o.value for o in a + b] == [o.value for o in b + a]
    assert [o.value for o in a - b] == [-1 * o.value for o in b - a]
    assert [o.value for o in a / b] == [o.value for o in [p / b for p in a]]
    assert [o.value for o in b / a] == [o.value for o in [b / p for p in a]]

    a = np.random.normal(0.0, 1e10, 10)
    b = pe.pseudo_Obs(4, 0.8, 't')

    assert [o.value for o in a * b] == [o.value for o in b * a]
    assert [o.value for o in a + b] == [o.value for o in b + a]
    assert [o.value for o in a - b] == [-1 * o.value for o in b - a]
    assert [o.value for o in a / b] == [o.value for o in [p / b for p in a]]
    assert [o.value for o in b / a] == [o.value for o in [b / p for p in a]]


def test_gamma_method_standard_data():
    for data in [np.tile([1, -1], 1000),
                 np.zeros(1195),
                 np.sin(np.sqrt(2) * np.pi * np.arange(1812))]:
        test_obs = pe.Obs([data], ['t'])
        test_obs.gamma_method()
        assert test_obs.dvalue - test_obs.ddvalue <= np.std(data, ddof=1) / np.sqrt(len(data))
        assert test_obs.e_tauint['t'] - 0.5 <= test_obs.e_dtauint['t']
        test_obs.gamma_method(tau_exp=10)
        assert test_obs.e_tauint['t'] - 10.5 <= test_obs.e_dtauint['t']


def test_gamma_method_no_windowing():
    for iteration in range(50):
        obs = pe.Obs([np.random.normal(1.02, 0.02, 733 + np.random.randint(1000))], ['ens'])
        obs.gamma_method(S=0)
        assert obs.e_tauint['ens'] == 0.5
        assert np.isclose(np.sqrt(np.var(obs.deltas['ens'], ddof=1) / obs.shape['ens']), obs.dvalue)
        obs.gamma_method(S=1.1)
        assert obs.e_tauint['ens'] > 0.5
    with pytest.raises(Exception):
        obs.gamma_method(S=-0.2)


def test_gamma_method_persistance():
    my_obs = pe.Obs([np.random.rand(730)], ['t'])
    my_obs.gamma_method()
    value = my_obs.value
    dvalue = my_obs.dvalue
    ddvalue = my_obs.ddvalue
    my_obs = 1.0 * my_obs
    my_obs.gamma_method()
    assert value == my_obs.value
    assert dvalue == my_obs.dvalue
    assert ddvalue == my_obs.ddvalue
    my_obs.gamma_method()
    assert value == my_obs.value
    assert dvalue == my_obs.dvalue
    assert ddvalue == my_obs.ddvalue
    my_obs.gamma_method(S=3.7)
    my_obs.gamma_method()
    assert value == my_obs.value
    assert dvalue == my_obs.dvalue
    assert ddvalue == my_obs.ddvalue


def test_gamma_method_kwargs():

    my_obs = pe.Obs([np.random.normal(1, 0.8, 5)], ['ens'], idl=[[1, 2, 3, 6, 17]])

    pe.Obs.S_dict['ens13.7'] = 3

    my_obs.gamma_method()
    assert my_obs.S['ens'] == pe.Obs.S_global
    assert my_obs.tau_exp['ens'] == pe.Obs.tau_exp_global
    assert my_obs.N_sigma['ens'] == pe.Obs.N_sigma_global

    my_obs.gamma_method(S=3.71)
    assert my_obs.S['ens'] == 3.71
    assert my_obs.tau_exp['ens'] == pe.Obs.tau_exp_global
    assert my_obs.N_sigma['ens'] == pe.Obs.N_sigma_global

    my_obs.gamma_method(tau_exp=17)
    assert my_obs.S['ens'] == pe.Obs.S_global
    assert my_obs.tau_exp['ens'] == 17
    assert my_obs.N_sigma['ens'] == pe.Obs.N_sigma_global

    my_obs.gamma_method(tau_exp=1.7, N_sigma=2.123)
    assert my_obs.S['ens'] == pe.Obs.S_global
    assert my_obs.tau_exp['ens'] == 1.7
    assert my_obs.N_sigma['ens'] == 2.123

    pe.Obs.S_dict['ens'] = 3
    pe.Obs.S_dict['ens|23'] = 7

    my_obs.gamma_method()
    assert my_obs.S['ens'] == pe.Obs.S_dict['ens'] == 3
    assert my_obs.tau_exp['ens'] == pe.Obs.tau_exp_global
    assert my_obs.N_sigma['ens'] == pe.Obs.N_sigma_global

    pe.Obs.tau_exp_dict['ens'] = 4
    pe.Obs.N_sigma_dict['ens'] = 4

    my_obs.gamma_method()
    assert my_obs.S['ens'] == pe.Obs.S_dict['ens'] == 3
    assert my_obs.tau_exp['ens'] == pe.Obs.tau_exp_dict['ens'] == 4
    assert my_obs.N_sigma['ens'] == pe.Obs.N_sigma_dict['ens'] == 4

    my_obs.gamma_method(S=1.1, tau_exp=1.2, N_sigma=1.3)
    assert my_obs.S['ens'] == 1.1
    assert my_obs.tau_exp['ens'] == 1.2
    assert my_obs.N_sigma['ens'] == 1.3

    pe.Obs.S_dict = {}
    pe.Obs.tau_exp_dict = {}
    pe.Obs.N_sigma_dict = {}

    my_obs = pe.Obs([np.random.normal(1, 0.8, 5)], ['ens'])

    my_obs.gamma_method()
    assert my_obs.S['ens'] == pe.Obs.S_global
    assert my_obs.tau_exp['ens'] == pe.Obs.tau_exp_global
    assert my_obs.N_sigma['ens'] == pe.Obs.N_sigma_global


def test_fft():
    value = np.random.normal(5, 100)
    dvalue = np.abs(np.random.normal(0, 5))
    test_obs1 = pe.pseudo_Obs(value, dvalue, 't', int(500 + 1000 * np.random.rand()))
    test_obs2 = copy.deepcopy(test_obs1)
    test_obs1.gamma_method()
    test_obs2.gamma_method(fft=False)
    assert max(np.abs(test_obs1.e_rho['t'] - test_obs2.e_rho['t'])) <= 10 * np.finfo(np.float64).eps
    assert np.abs(test_obs1.dvalue - test_obs2.dvalue) <= 10 * max(test_obs1.dvalue, test_obs2.dvalue) * np.finfo(np.float64).eps


def test_gamma_method_uncorrelated():
    # Construct pseudo Obs with random shape
    value = np.random.normal(5, 10)
    dvalue = np.abs(np.random.normal(0, 1))

    test_obs = pe.pseudo_Obs(value, dvalue, 't', int(1000 * (1 + np.random.rand())))

    # Test if the error is processed correctly
    test_obs.gamma_method()
    assert np.abs(test_obs.value - value) < 1e-12
    assert abs(test_obs.dvalue - dvalue) < 1e-10 * dvalue


def test_derived_observables():
    # Construct pseudo Obs with random shape
    test_obs = pe.pseudo_Obs(2, 0.1 * (1 + np.random.rand()), 't', int(1000 * (1 + np.random.rand())))

    # Check if autograd and numgrad give the same result
    d_Obs_ad = pe.derived_observable(lambda x, **kwargs: x[0] * x[1] * np.sin(x[0] * x[1]), [test_obs, test_obs])
    d_Obs_ad.gamma_method()
    d_Obs_fd = pe.derived_observable(lambda x, **kwargs: x[0] * x[1] * np.sin(x[0] * x[1]), [test_obs, test_obs], num_grad=True)
    d_Obs_fd.gamma_method()

    assert d_Obs_ad == d_Obs_fd
    assert np.abs(4.0 * np.sin(4.0) - d_Obs_ad.value) < 1000 * np.finfo(np.float64).eps * np.abs(d_Obs_ad.value)
    assert np.abs(d_Obs_ad.dvalue-d_Obs_fd.dvalue) < 1000 * np.finfo(np.float64).eps * d_Obs_ad.dvalue

    i_am_one = pe.derived_observable(lambda x, **kwargs: x[0] / x[1], [d_Obs_ad, d_Obs_ad])
    i_am_one.gamma_method()

    assert i_am_one == 1.0
    assert i_am_one.dvalue < 2 * np.finfo(np.float64).eps
    assert i_am_one.e_dvalue['t'] <= 2 * np.finfo(np.float64).eps
    assert i_am_one.e_ddvalue['t'] <= 2 * np.finfo(np.float64).eps


def test_multi_ens():
    names = ['A0', 'A1|r001', 'A1|r002']
    test_obs = pe.Obs([np.random.rand(50), np.random.rand(50), np.random.rand(50)], names)
    assert test_obs.e_names == ['A0', 'A1']
    assert test_obs.e_content['A0'] == ['A0']
    assert test_obs.e_content['A1'] == ['A1|r001', 'A1|r002']

    my_sum = 0
    ensembles = []
    for i in range(100):
        my_sum += pe.Obs([np.random.rand(50)], [str(i)])
        ensembles.append(str(i))
    assert my_sum.e_names == sorted(ensembles)


def test_multi_ens2():
    names = ['ens', 'e', 'en', 'e|r010', 'E|er', 'ens|', 'Ens|34', 'ens|r548984654ez4e3t34terh']

    my_sum = 0
    for name in names:
        my_sum += pe.pseudo_Obs(1, 0.1, name)

    assert my_sum.e_names == ['E', 'Ens', 'e', 'en', 'ens']
    assert my_sum.e_content == {'E': ['E|er'],
                                'Ens': ['Ens|34'],
                                'e': ['e|r010', 'e'],
                                'en': ['en'],
                                'ens': ['ens|', 'ens|r548984654ez4e3t34terh', 'ens']}



def test_overloaded_functions():
    funcs = [np.exp, np.log, np.sin, np.cos, np.tan, np.sinh, np.cosh, np.arcsinh, np.arccosh]
    deriv = [np.exp, lambda x: 1 / x, np.cos, lambda x: -np.sin(x), lambda x: 1 / np.cos(x) ** 2, np.cosh, np.sinh, lambda x: 1 / np.sqrt(x ** 2 + 1), lambda x: 1 / np.sqrt(x ** 2 - 1)]
    val = 3 + 0.5 * np.random.rand()
    dval = 0.3 + 0.4 * np.random.rand()
    test_obs = pe.pseudo_Obs(val, dval, 't', int(1000 * (1 + np.random.rand())))

    for i, item in enumerate(funcs):
        ad_obs = item(test_obs)
        fd_obs = pe.derived_observable(lambda x, **kwargs: item(x[0]), [test_obs], num_grad=True)
        ad_obs.gamma_method(S=0.01)
        assert np.max((ad_obs.deltas['t'] - fd_obs.deltas['t']) / ad_obs.deltas['t']) < 1e-8, item.__name__
        assert np.abs((ad_obs.value - item(val)) / ad_obs.value) < 1e-10, item.__name__
        assert np.abs(ad_obs.dvalue - dval * np.abs(deriv[i](val))) < 1e-6, item.__name__


def test_utils():
    zero_pseudo_obs = pe.pseudo_Obs(1.0, 0.0, 'null')
    my_obs = pe.pseudo_Obs(1.0, 0.5, 't|r01')
    my_obs += pe.pseudo_Obs(1.0, 0.5, 't|r02')
    str(my_obs)
    for tau_exp in [0, 5]:
        my_obs.gamma_method(tau_exp=tau_exp)
        my_obs.tag = "Test description"
        my_obs.details(False)
        my_obs.details(True)
        assert not my_obs.is_zero_within_error()
        my_obs.plot_tauint()
        my_obs.plot_rho()
        my_obs.plot_rep_dist()
        my_obs.plot_history(True)
        my_obs.plot_history(False)
        my_obs.plot_piechart()
        assert my_obs > (my_obs - 1)
        assert my_obs < (my_obs + 1)
        float(my_obs)
        str(my_obs)
        plt.close('all')


def test_cobs():
    obs1 = pe.pseudo_Obs(1.0, 0.1, 't')
    obs2 = pe.pseudo_Obs(-0.2, 0.03, 't')

    my_cobs = pe.CObs(obs1, obs2)
    assert +my_cobs == my_cobs
    assert -my_cobs == 0 - my_cobs
    my_cobs == my_cobs
    str(my_cobs)
    repr(my_cobs)
    assert not (my_cobs + my_cobs.conjugate()).real.is_zero()
    assert (my_cobs + my_cobs.conjugate()).imag.is_zero()
    assert (my_cobs - my_cobs.conjugate()).real.is_zero()
    assert not (my_cobs - my_cobs.conjugate()).imag.is_zero()
    np.abs(my_cobs)

    assert (my_cobs * my_cobs / my_cobs - my_cobs).is_zero()
    assert (my_cobs + my_cobs - 2 * my_cobs).is_zero()

    fs = [[lambda x: x[0] + x[1], lambda x: x[1] + x[0]],
          [lambda x: x[0] * x[1], lambda x: x[1] * x[0]]]
    for other in [3, 1.1, (1.1 - 0.2j), (2.3 + 0j), (0.0 + 7.7j), pe.CObs(obs1), pe.CObs(obs1, obs2)]:
        for funcs in fs:
            ta = funcs[0]([my_cobs, other])
            tb = funcs[1]([my_cobs, other])
            diff = ta - tb
            assert diff.is_zero()

        ta = my_cobs - other
        tb = other - my_cobs
        diff = ta + tb
        assert diff.is_zero()

        ta = my_cobs / other
        tb = other / my_cobs
        diff = ta * tb - 1
        assert diff.is_zero()

        assert (my_cobs / other * other - my_cobs).is_zero()
        assert (other / my_cobs * my_cobs - other).is_zero()


def test_cobs_overloading():
    obs = pe.pseudo_Obs(1.1, 0.1, 't')
    cobs = pe.CObs(obs, obs)

    cobs + obs
    obs + cobs

    cobs - obs
    obs - cobs

    cobs * obs
    obs * cobs

    cobs / obs
    obs / cobs


def test_reweighting():
    my_obs = pe.Obs([np.random.rand(1000)], ['t'])
    assert not my_obs.reweighted
    r_obs = pe.reweight(my_obs, [my_obs])
    assert r_obs[0].reweighted
    r_obs2 = r_obs[0] * my_obs
    assert r_obs2.reweighted

    my_irregular_obs = pe.Obs([np.random.rand(500)], ['t'], idl=[range(1, 1001, 2)])
    assert not my_irregular_obs.reweighted
    r_obs = pe.reweight(my_obs, [my_irregular_obs], all_configs=True)
    r_obs = pe.reweight(my_obs, [my_irregular_obs], all_configs=False)
    r_obs = pe.reweight(my_obs, [my_obs])
    assert r_obs[0].reweighted
    r_obs2 = r_obs[0] * my_obs
    assert r_obs2.reweighted
    my_covobs = pe.cov_Obs(1.0, 0.003, 'cov')
    with pytest.raises(Exception):
        pe.reweight(my_obs, [my_covobs])
    my_obs2 = pe.Obs([np.random.rand(1000)], ['t2'])
    with pytest.raises(Exception):
        pe.reweight(my_obs, [my_obs + my_obs2])
    with pytest.raises(Exception):
        pe.reweight(my_irregular_obs, [my_obs])


def test_merge_obs():
    my_obs1 = pe.Obs([np.random.rand(100)], ['t'])
    my_obs2 = pe.Obs([np.random.rand(100)], ['q'], idl=[range(1, 200, 2)])
    merged = pe.merge_obs([my_obs1, my_obs2])
    diff = merged - my_obs2 - my_obs1
    assert diff == -(my_obs1.value + my_obs2.value) / 2
    with pytest.raises(Exception):
        pe.merge_obs([my_obs1, my_obs1])
    my_covobs = pe.cov_Obs(1.0, 0.003, 'cov')
    with pytest.raises(Exception):
        pe.merge_obs([my_obs1, my_covobs])



def test_merge_obs_r_values():
    a1 = pe.pseudo_Obs(1.1, .1, 'a|1')
    a2 = pe.pseudo_Obs(1.2, .1, 'a|2')
    a = pe.merge_obs([a1, a2])

    assert np.isclose(a.r_values['a|1'], a1.value)
    assert np.isclose(a.r_values['a|2'], a2.value)
    assert np.isclose(a.value, np.mean([a1.value, a2.value]))


def test_correlate():
    my_obs1 = pe.Obs([np.random.rand(100)], ['t'])
    my_obs2 = pe.Obs([np.random.rand(100)], ['t'])
    corr1 = pe.correlate(my_obs1, my_obs2)
    corr2 = pe.correlate(my_obs2, my_obs1)
    assert corr1 == corr2

    my_obs3 = pe.Obs([np.random.rand(100)], ['t'], idl=[range(2, 102)])
    with pytest.raises(Exception):
        pe.correlate(my_obs1, my_obs3)

    my_obs4 = pe.Obs([np.random.rand(99)], ['t'])
    with pytest.raises(Exception):
        pe.correlate(my_obs1, my_obs4)

    my_obs5 = pe.Obs([np.random.rand(100)], ['t'], idl=[range(5, 505, 5)])
    my_obs6 = pe.Obs([np.random.rand(100)], ['t'], idl=[range(5, 505, 5)])
    corr3 = pe.correlate(my_obs5, my_obs6)
    assert my_obs5.idl == corr3.idl

    my_new_obs = pe.Obs([np.random.rand(100)], ['q3'])
    with pytest.raises(Exception):
        pe.correlate(my_obs1, my_new_obs)
    my_covobs = pe.cov_Obs(1.0, 0.003, 'cov')
    with pytest.raises(Exception):
        pe.correlate(my_covobs, my_covobs)
    r_obs = pe.reweight(my_obs1, [my_obs1])[0]
    with pytest.warns(RuntimeWarning):
        pe.correlate(r_obs, r_obs)


def test_merge_idx():
    assert pe.obs._merge_idx([range(10, 1010, 10), range(10, 1010, 50)]) == range(10, 1010, 10)
    assert pe.obs._merge_idx([range(500, 6050, 50), range(500, 6250, 250)]) == range(500, 6250, 50)


def test_intersection_idx():
    assert pe.obs._intersection_idx([range(1, 100), range(1, 100), range(1, 100)]) == range(1, 100)
    assert pe.obs._intersection_idx([range(1, 100, 10), range(1, 100, 2)]) == range(1, 100, 10)
    assert pe.obs._intersection_idx([range(10, 1010, 10), range(10, 1010, 50)]) == range(10, 1010, 50)
    assert pe.obs._intersection_idx([range(500, 6050, 50), range(500, 6250, 250)]) == range(500, 6050, 250)

    for ids in [[list(range(1, 80, 3)), list(range(1, 100, 2))], [range(1, 80, 3), range(1, 100, 2), range(1, 100, 7)]]:
        assert list(pe.obs._intersection_idx(ids)) == pe.obs._intersection_idx([list(o) for o in ids])

def test_merge_intersection():
    for idl_list in [[range(1, 100), range(1, 100), range(1, 100)],
                     [range(4, 80, 6), range(4, 80, 6)],
                     [[0, 2, 8, 19, 205], [0, 2, 8, 19, 205]]]:
        assert pe.obs._merge_idx(idl_list) == pe.obs._intersection_idx(idl_list)


def test_intersection_collapse():
    range1 = range(1, 2000, 2)
    range2 = range(2, 2001, 8)

    obs1 = pe.Obs([np.random.normal(1.0, 0.1, len(range1))], ["ens"], idl=[range1])
    obs_merge = obs1 + pe.Obs([np.random.normal(1.0, 0.1, len(range2))], ["ens"], idl=[range2])

    intersection = pe.obs._intersection_idx([o.idl["ens"] for o in [obs1, obs_merge]])
    coll = pe.obs._collapse_deltas_for_merge(obs_merge.deltas["ens"], obs_merge.idl["ens"], len(obs_merge.idl["ens"]), range1)

    assert np.all(coll == obs1.deltas["ens"])


def test_irregular_error_propagation():
    obs_list = [pe.Obs([np.random.rand(100)], ['t']),
                pe.Obs([np.random.rand(50)], ['t'], idl=[range(1, 100, 2)]),
                pe.Obs([np.random.rand(50)], ['t'], idl=[np.arange(1, 100, 2)]),
                pe.Obs([np.random.rand(6)], ['t'], idl=[[4, 18, 27, 29, 57, 80]]),
                pe.Obs([np.random.rand(50)], ['t'], idl=[list(range(1, 26)) + list(range(50, 100, 2))])]
    for obs1 in obs_list:
        obs1.details()
        for obs2 in obs_list:
            assert obs1 == (obs1 / obs2) * obs2
            assert obs1 == (obs1 * obs2) / obs2
            assert obs1 == obs1 * (obs2 / obs2)
            assert obs1 == (obs1 + obs2) - obs2
            assert obs1 == obs1 + (obs2 - obs2)


def test_gamma_method_irregular():
    N = 20000
    arr = np.random.normal(1, .2, size=N)
    afull = pe.Obs([arr], ['a'])

    configs = np.ones_like(arr)
    for i in np.random.uniform(0, len(arr), size=int(.8 * N)):
        configs[int(i)] = 0
    zero_arr = [arr[i] for i in range(len(arr)) if not configs[i] == 0]
    idx = [i + 1 for i in range(len(configs)) if configs[i] == 1]
    a = pe.Obs([zero_arr], ['a'], idl=[idx])

    afull.gamma_method()
    a.gamma_method()
    ad = a.dvalue

    expe = (afull.dvalue * np.sqrt(N / np.sum(configs)))
    assert (a.dvalue - 5 * a.ddvalue < expe and expe < a.dvalue + 5 * a.ddvalue)

    afull.gamma_method(fft=False)
    a.gamma_method(fft=False)

    expe = (afull.dvalue * np.sqrt(N / np.sum(configs)))
    assert (a.dvalue - 5 * a.ddvalue < expe and expe < a.dvalue + 5 * a.ddvalue)
    assert np.abs(a.dvalue - ad) <= 10 * max(a.dvalue, ad) * np.finfo(np.float64).eps

    afull.gamma_method(tau_exp=.00001)
    a.gamma_method(tau_exp=.00001)

    expe = (afull.dvalue * np.sqrt(N / np.sum(configs)))
    assert (a.dvalue - 5 * a.ddvalue < expe and expe < a.dvalue + 5 * a.ddvalue)

    arr2 = np.random.normal(1, .2, size=N)
    afull = pe.Obs([arr, arr2], ['a1', 'a2'])

    configs = np.ones_like(arr2)
    for i in np.random.uniform(0, len(arr2), size=int(.8*N)):
        configs[int(i)] = 0
    zero_arr2 = [arr2[i] for i in range(len(arr2)) if not configs[i] == 0]
    idx2 = [i + 1 for i in range(len(configs)) if configs[i] == 1]
    a = pe.Obs([zero_arr, zero_arr2], ['a1', 'a2'], idl=[idx, idx2])

    afull.gamma_method()
    a.gamma_method()

    expe = (afull.dvalue * np.sqrt(N / np.sum(configs)))
    assert (a.dvalue - 5 * a.ddvalue < expe and expe < a.dvalue + 5 * a.ddvalue)

    def gen_autocorrelated_array(inarr, rho):
        outarr = np.copy(inarr)
        for i in range(1, len(outarr)):
            outarr[i] = rho * outarr[i - 1] + np.sqrt(1 - rho**2) * outarr[i]
        return outarr

    arr = np.random.normal(1, .2, size=N)
    carr = gen_autocorrelated_array(arr, .346)
    a = pe.Obs([carr], ['a'])
    a.gamma_method()

    ae = pe.Obs([[carr[i] for i in range(len(carr)) if i % 2 == 0]], ['a'], idl=[[i for i in range(len(carr)) if i % 2 == 0]])
    ae.gamma_method()

    ao = pe.Obs([[carr[i] for i in range(len(carr)) if i % 2 == 1]], ['a'], idl=[[i for i in range(len(carr)) if i % 2 == 1]])
    ao.gamma_method()

    assert(ae.e_tauint['a'] < a.e_tauint['a'])
    assert((ae.e_tauint['a'] - 4 * ae.e_dtauint['a'] < ao.e_tauint['a']))
    assert((ae.e_tauint['a'] + 4 * ae.e_dtauint['a'] > ao.e_tauint['a']))

    a = pe.pseudo_Obs(1, .1, 'a', samples=10)
    a.idl['a'] = range(4, 15)
    b = pe.pseudo_Obs(1, .1, 'a', samples=151)
    b.idl['a'] = range(4, 608, 4)
    ol = [a, b]
    o = (ol[0] - ol[1]) / (ol[1])


def test_covariance_is_variance():
    value = np.random.normal(5, 10)
    dvalue = np.abs(np.random.normal(0, 1))
    test_obs = pe.pseudo_Obs(value, dvalue, 't')
    test_obs.gamma_method()
    assert np.isclose(test_obs.dvalue ** 2, pe.covariance([test_obs, test_obs])[0, 1])
    test_obs = test_obs + pe.pseudo_Obs(value, dvalue, 'q', 200)
    test_obs.gamma_method()
    assert np.isclose(test_obs.dvalue ** 2, pe.covariance([test_obs, test_obs])[0, 1])


def test_covariance_vs_numpy():
    N = 1078
    data1 = np.random.normal(2.5, 0.2, N)
    data2 = np.random.normal(0.5, 0.08, N)
    data3 = np.random.normal(-178, 5, N)
    uncorr = np.row_stack([data1, data2, data3])
    corr = np.random.multivariate_normal([0.0, 17, -0.0487], [[1.0, 0.6, -0.22], [0.6, 0.8, 0.01], [-0.22, 0.01, 1.9]], N).T

    for X in [uncorr, corr]:
        obs1 = pe.Obs([X[0]], ["ens1"])
        obs2 = pe.Obs([X[1]], ["ens1"])
        obs3 = pe.Obs([X[2]], ["ens1"])
        obs1.gamma_method(S=0.0)
        obs2.gamma_method(S=0.0)
        obs3.gamma_method(S=0.0)
        pe_cov = pe.covariance([obs1, obs2, obs3])
        np_cov = np.cov(X) / N
        assert np.allclose(pe_cov, np_cov, atol=1e-14)


def test_covariance_symmetry():
    value1 = np.random.normal(5, 10)
    dvalue1 = np.abs(np.random.normal(0, 1))
    test_obs1 = pe.pseudo_Obs(value1, dvalue1, 't')
    test_obs1.gamma_method()
    value2 = np.random.normal(5, 10)
    dvalue2 = np.abs(np.random.normal(0, 1))
    test_obs2 = pe.pseudo_Obs(value2, dvalue2, 't')
    test_obs2.gamma_method()
    cov_ab = pe.covariance([test_obs1, test_obs2])[0, 1]
    cov_ba = pe.covariance([test_obs2, test_obs1])[0, 1]
    assert np.isclose(cov_ab, cov_ba)
    assert np.abs(cov_ab) < test_obs1.dvalue * test_obs2.dvalue * (1 + 10 * np.finfo(np.float64).eps)

    N = 100
    arr = np.random.normal(1, .2, size=N)
    configs = np.ones_like(arr)
    for i in np.random.uniform(0, len(arr), size=int(.8 * N)):
        configs[int(i)] = 0
    zero_arr = [arr[i] for i in range(len(arr)) if not configs[i] == 0]
    idx = [i + 1 for i in range(len(configs)) if configs[i] == 1]
    a = pe.Obs([zero_arr], ['t'], idl=[idx])
    a.gamma_method()
    assert np.isclose(a.dvalue ** 2, pe.covariance([a, a])[0, 1], atol=100, rtol=1e-4)

    cov_ab = pe.covariance([test_obs1, a])[0, 1]
    cov_ba = pe.covariance([a, test_obs1])[0, 1]
    assert np.abs(cov_ab - cov_ba) <= 10 * np.finfo(np.float64).eps
    assert np.abs(cov_ab) < test_obs1.dvalue * a.dvalue * (1 + 10 * np.finfo(np.float64).eps)


def test_covariance_sum():
    length = 2
    t_fac = 0.4
    tt = pe.misc.gen_correlated_data(np.zeros(length), 0.99 * np.ones((length, length)) + 0.01 * np.diag(np.ones(length)), 'test', tau=0.5 + t_fac * np.random.rand(length), samples=1000)
    [o.gamma_method(S=0) for o in tt]

    t_cov = pe.covariance(tt)

    my_sum = tt[0] + tt[1]
    my_sum.gamma_method(S=0)
    e_cov = (my_sum.dvalue ** 2 - tt[0].dvalue ** 2 - tt[1].dvalue ** 2) / 2

    assert np.isclose(e_cov, t_cov[0, 1])


def test_covariance_positive_semidefinite():
    length = 64
    t_fac = 1.5
    tt = pe.misc.gen_correlated_data(np.zeros(length), 0.99999 * np.ones((length, length)) + 0.00001 * np.diag(np.ones(length)), 'test', tau=0.5 + t_fac * np.random.rand(length), samples=1000)
    [o.gamma_method() for o in tt]
    cov = pe.covariance(tt)
    assert np.all(np.linalg.eigh(cov)[0] >= -1e-15)


def test_covariance_factorizing():
    length = 2
    t_fac = 1.5

    tt = pe.misc.gen_correlated_data(np.zeros(length), 0.75 * np.ones((length, length)) + 0.8 * np.diag(np.ones(length)), 'test', tau=0.5 + t_fac * np.random.rand(length), samples=1000)
    [o.gamma_method() for o in tt]

    mt0 = -tt[0]
    mt0.gamma_method()

    assert np.isclose(pe.covariance([mt0, tt[1]])[0, 1], -pe.covariance(tt)[0, 1])


def test_covariance_smooth_eigenvalues():
    for c_coeff in range(0, 14, 2):
        length = 14
        sm = 5
        t_fac = 1.5
        tt = pe.misc.gen_correlated_data(np.zeros(length), 1 - 0.1 ** c_coeff * np.ones((length, length)) + 0.1 ** c_coeff * np.diag(np.ones(length)), 'test', tau=0.5 + t_fac * np.random.rand(length), samples=200)
        [o.gamma_method() for o in tt]
        full_corr = pe.covariance(tt, correlation=True)
        cov = pe.covariance(tt, smooth=sm, correlation=True)

        full_evals = np.linalg.eigh(full_corr)[0]
        sm_length = np.where(full_evals < np.mean(full_evals[:-sm]))[0][-1]

        evals = np.linalg.eigh(cov)[0]
        assert np.all(np.isclose(evals[:sm_length], evals[0], atol=1e-8))


def test_covariance_alternation():
    length = 12
    t_fac = 2.5

    tt1 = pe.misc.gen_correlated_data(np.zeros(length), -0.00001 * np.ones((length, length)) + 0.002 * np.diag(np.ones(length)), 'test', tau=0.5 + t_fac * np.random.rand(length), samples=88)
    tt2 = pe.misc.gen_correlated_data(np.zeros(length), 0.9999 * np.ones((length, length)) + 0.0001 * np.diag(np.ones(length)), 'another_test|r0', tau=0.7 + t_fac * np.random.rand(length), samples=73)
    tt3 = pe.misc.gen_correlated_data(np.zeros(length), 0.9999 * np.ones((length, length)) + 0.0001 * np.diag(np.ones(length)), 'another_test|r1', tau=0.7 + t_fac * np.random.rand(length), samples=91)

    tt = np.array(tt1) + (np.array(tt2) + np.array(tt3))
    tt *= np.resize([1, -1], length)

    [o.gamma_method() for o in tt]
    cov = pe.covariance(tt, True)

    assert np.all(np.linalg.eigh(cov)[0] > -1e-15)


def test_covariance_correlation():
    test_obs = pe.pseudo_Obs(-4, 0.8, 'test', samples=784)
    assert np.allclose(pe.covariance([test_obs, test_obs, test_obs], correlation=True), np.ones((3, 3)))


def test_covariance_rank_deficient():
    obs = []
    for i in range(5):
        obs.append(pe.pseudo_Obs(1.0, 0.1, 'test', 5))

    with pytest.warns(RuntimeWarning):
        pe.covariance(obs)

def test_covariance_idl():
    range1 = range(10, 1010, 10)
    range2 = range(10, 1010, 50)

    obs1 = pe.Obs([np.random.normal(1.0, 0.1, len(range1))], ["ens"], idl=[range1])
    obs2 = pe.Obs([np.random.normal(1.0, 0.1, len(range2))], ["ens"], idl=[range2])
    obs1.gamma_method()
    obs2.gamma_method()

    pe.covariance([obs1, obs2])


def test_correlation_intersection_of_idls():
    range1 = range(1, 2000, 2)
    range2 = range(2, 2001, 2)

    obs1 = pe.Obs([np.random.normal(1.0, 0.1, len(range1))], ["ens"], idl=[range1])
    obs2_a = 0.4 * pe.Obs([np.random.normal(1.0, 0.1, len(range1))], ["ens"], idl=[range1]) + 0.6 * obs1
    obs1.gamma_method()
    obs2_a.gamma_method()

    cov1 = pe.covariance([obs1, obs2_a])
    corr1 = pe.covariance([obs1, obs2_a], correlation=True)

    obs2_b = obs2_a + pe.Obs([np.random.normal(1.0, 0.1, len(range2))], ["ens"], idl=[range2])
    obs2_b.gamma_method()

    cov2 = pe.covariance([obs1, obs2_b])
    corr2 = pe.covariance([obs1, obs2_b], correlation=True)

    assert np.isclose(corr1[0, 1], corr2[0, 1], atol=1e-14)
    assert cov1[0, 1] > cov2[0, 1]

    obs2_c = pe.Obs([np.random.normal(1.0, 0.1, len(range2))], ["ens"], idl=[range2])
    obs2_c.gamma_method()
    assert np.isclose(0, pe.covariance([obs1, obs2_c])[0, 1], atol=1e-14)


def test_covariance_non_identical_objects():
    obs1 = pe.Obs([np.random.normal(1.0, 0.1, 1000), np.random.normal(1.0, 0.1, 1000), np.random.normal(1.0, 0.1, 732)], ["ens|r1", "ens|r2", "ens2"])
    obs1.gamma_method()
    obs2 = obs1 + 1e-18
    obs2.gamma_method()
    assert obs1 == obs2
    assert obs1 is not obs2
    assert np.allclose(np.ones((2, 2)), pe.covariance([obs1, obs2], correlation=True), atol=1e-14)


def test_covariance_additional_non_overlapping_data():
    range1 = range(1, 20, 2)

    data2 = np.random.normal(0.0, 0.1, len(range1))

    obs1 = pe.Obs([np.random.normal(1.0, 0.1, len(range1))], ["ens"], idl=[range1])
    obs2_a = pe.Obs([data2], ["ens"], idl=[range1])
    obs1.gamma_method()
    obs2_a.gamma_method()

    corr1 = pe.covariance([obs1, obs2_a], correlation=True)

    added_data = np.random.normal(0.0, 0.1, len(range1))
    added_data -= np.mean(added_data) - np.mean(data2)
    data2_extended = np.ravel([data2, added_data], 'F')

    obs2_b = pe.Obs([data2_extended], ["ens"])
    obs2_b.gamma_method()

    corr2 = pe.covariance([obs1, obs2_b], correlation=True)

    assert np.isclose(corr1[0, 1], corr2[0, 1], atol=1e-14)


def test_covariance_reorder_non_overlapping_data():
    range1 = range(1, 20, 2)
    range2 = range(1, 41, 2)

    obs1 = pe.Obs([np.random.normal(1.0, 0.1, len(range1))], ["ens"], idl=[range1])
    obs2_b = pe.Obs([np.random.normal(1.0, 0.1, len(range2))], ["ens"], idl=[range2])
    obs1.gamma_method()
    obs2_b.gamma_method()

    corr1 = pe.covariance([obs1, obs2_b], correlation=True)

    deltas = list(obs2_b.deltas['ens'][:len(range1)]) + sorted(obs2_b.deltas['ens'][len(range1):])
    obs2_a = pe.Obs([obs2_b.value + np.array(deltas)], ["ens"], idl=[range2])
    obs2_a.gamma_method()

    corr2 = pe.covariance([obs1, obs2_a], correlation=True)

    assert np.isclose(corr1[0, 1], corr2[0, 1], atol=1e-14)


def test_empty_obs():
    o = pe.Obs([np.random.rand(100)], ['test'])
    q = o + pe.Obs([], [], means=[])
    assert q == o


def test_reweight_method():
    obs1 = pe.pseudo_Obs(0.2, 0.01, 'test')
    rw = pe.pseudo_Obs(0.999, 0.001, 'test')
    assert obs1.reweight(rw) == pe.reweight(rw, [obs1])[0]


def test_jackknife():
    full_data = np.random.normal(1.1, 0.87, 5487)

    my_obs = pe.Obs([full_data], ['test'])

    n = full_data.size
    mean = np.mean(full_data)
    tmp_jacks = np.zeros(n + 1)
    tmp_jacks[0] = mean
    for i in range(n):
        tmp_jacks[i + 1] = (n * mean - full_data[i]) / (n - 1)

    assert np.allclose(tmp_jacks, my_obs.export_jackknife())
    my_new_obs = my_obs + pe.Obs([full_data], ['test2'])
    with pytest.raises(Exception):
        my_new_obs.export_jackknife()


def test_import_jackknife():
    full_data = np.random.normal(1.105, 0.021, 754)
    my_obs = pe.Obs([full_data], ['test'])
    my_jacks = my_obs.export_jackknife()
    reconstructed_obs = pe.import_jackknife(my_jacks, 'test')
    assert my_obs == reconstructed_obs


def test_reduce_deltas():
    idx_old = range(1, 101)
    deltas = [float(i) for i in idx_old]
    idl = [
        range(2, 26, 2),
        range(1, 101),
        np.arange(1, 101),
        [1, 2, 3, 5, 6, 7, 9, 12],
        [7],
    ]
    for idx_new in idl:
        new = pe.obs._reduce_deltas(deltas, idx_old, idx_new)
        print(new)
        assert(np.alltrue([float(i) for i in idx_new] == new))


def test_merge_idx():
    idl = [list(np.arange(1, 14)) + list(range(16, 100, 4)), range(4, 604, 4), [2, 4, 5, 6, 8, 9, 12, 24], range(1, 20, 1), range(50, 789, 7)]
    new_idx = pe.obs._merge_idx(idl)
    assert(new_idx[-1] > new_idx[0])
    for i in range(1, len(new_idx)):
        assert(new_idx[i - 1] < new_idx[i])


def test_cobs_array():
    cobs = pe.Obs([np.random.normal(1.0, 0.1, 100)], ['t']) * (1 + 2j)
    np.identity(4) + cobs
    cobs + np.identity(4)
    np.identity(4) - cobs
    cobs - np.identity(4)
    np.identity(4) * cobs
    cobs * np.identity(4)
    np.identity(4) / cobs
    cobs / np.ones((4, 4))


def test_hash():
    obs = pe.pseudo_Obs(0.3, 0.1, "test") + pe.Obs([np.random.normal(2.3, 0.2, 200)], ["test2"], [range(1, 400, 2)])
    o1 = obs + pe.cov_Obs(0.0, 0.1, "co") + pe.cov_Obs(0.0, 0.8, "co2")
    o2 = obs + pe.cov_Obs(0.0, 0.2, "co") + pe.cov_Obs(0.0, 0.8, "co2")

    for i_obs in [obs, o1, o2]:
        assert hash(i_obs) == hash(i_obs ** 2 / i_obs) == hash(1 * i_obs)
        assert hash(i_obs) == hash((1 + 1e-16) * i_obs)
        assert hash(i_obs) != hash((1 + 1e-7) * i_obs)
    assert hash(obs) != hash(o1)
    assert hash(o1) != hash(o2)
