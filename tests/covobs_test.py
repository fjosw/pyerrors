import autograd.numpy as np
import pyerrors as pe
import pytest

np.random.seed(0)


def test_covobs():
    val = 1.123124
    cov = .243423
    name = 'Covariance'
    co = pe.cov_Obs(val, cov, name)
    co.gamma_method()
    assert (co.dvalue == np.sqrt(cov))
    assert (co.value == val)

    do = 2 * co
    assert (do.covobs[name].grad[0] == 2)

    do = co * co
    assert (do.covobs[name].grad[0] == 2 * val)
    assert np.array_equal(do.covobs[name].cov, co.covobs[name].cov)

    pi = [16.7457, -19.0475]
    cov = [[3.49591, -6.07560], [-6.07560, 10.5834]]

    cl = pe.cov_Obs(pi, cov, 'rAP')
    pl = pe.misc.gen_correlated_data(pi, np.asarray(cov), 'rAPpseudo')

    def rAP(p, g0sq):
        return -0.0010666 * g0sq * (1 + np.exp(p[0] + p[1] / g0sq))

    for g0sq in [1, 1.5, 1.8]:
        oc = rAP(cl, g0sq)
        oc.gamma_method()
        op = rAP(pl, g0sq)
        op.gamma_method()
        assert(np.isclose(oc.value, op.value, rtol=1e-14, atol=1e-14))

    assert(pe.covariance(cl[0], cl[1]) == cov[0][1])
    assert(pe.covariance2(cl[0], cl[1]) == cov[1][0])

    do = cl[0] * cl[1]
    assert(np.array_equal(do.covobs['rAP'].grad, np.transpose([pi[1], pi[0]]).reshape(2, 1)))


def test_covobs_overloading():
    covobs = pe.cov_Obs([0.5, 0.5], np.array([[0.02, 0.02], [0.02, 0.02]]), 'test')
    assert (covobs[0] / covobs[1]) == 1
    assert (covobs[0] - covobs[1]) == 0

    my_obs = pe.pseudo_Obs(2.3, 0.2, 'obs')

    assert (my_obs * covobs[0] / covobs[1]) == my_obs

    covobs = pe.cov_Obs(0.0, 0.3, 'test')
    assert not covobs.is_zero()


def test_covobs_name_collision():
    covobs = pe.cov_Obs(0.5, 0.002, 'test')
    my_obs = pe.pseudo_Obs(2.3, 0.2, 'test')
    with pytest.raises(Exception):
        summed_obs = my_obs + covobs


def test_covobs_replica_separator():
    with pytest.raises(Exception):
        covobs = pe.cov_Obs(0.5, 0.002, 'test|r2')
