import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pyerrors as pe
import pytest

np.random.seed(0)


def test_function_overloading():
    corr_content_a = []
    corr_content_b = []
    for t in range(24):
        corr_content_a.append(pe.pseudo_Obs(np.random.normal(1e-10, 1e-8), 1e-4, 't'))
        corr_content_b.append(pe.pseudo_Obs(np.random.normal(1e8, 1e10), 1e7, 't'))

    corr_a = pe.correlators.Corr(corr_content_a)
    corr_b = pe.correlators.Corr(corr_content_b)

    fs = [lambda x: x[0] + x[1], lambda x: x[1] + x[0], lambda x: x[0] - x[1], lambda x: x[1] - x[0],
          lambda x: x[0] * x[1], lambda x: x[1] * x[0], lambda x: x[0] / x[1], lambda x: x[1] / x[0],
          lambda x: np.exp(x[0]), lambda x: np.sin(x[0]), lambda x: np.cos(x[0]), lambda x: np.tan(x[0]),
          lambda x: np.log(x[0] + 0.1), lambda x: np.sqrt(np.abs(x[0])),
          lambda x: np.sinh(x[0]), lambda x: np.cosh(x[0]), lambda x: np.tanh(x[0])]

    for i, f in enumerate(fs):
        t1 = f([corr_a, corr_b])
        t1.gamma_method()
        for o_a, o_b, con in zip(corr_content_a, corr_content_b, t1.content):
            t2 = f([o_a, o_b])
            t2.gamma_method()
            assert np.isclose(con[0].value, t2.value)
            assert np.isclose(con[0].dvalue, t2.dvalue)
            assert np.allclose(con[0].deltas['t'], t2.deltas['t'])

    np.arcsin(corr_a)
    np.arccos(corr_a)
    np.arctan(corr_a)
    np.arcsinh(corr_a)
    np.arccosh(corr_a + 1.1)
    np.arctanh(corr_a)


def test_modify_correlator():
    corr_content = []
    for t in range(24):
        exponent = np.random.normal(3, 5)
        corr_content.append(pe.pseudo_Obs(2 + 10 ** exponent, 10 ** (exponent - 1), 't'))

    corr = pe.Corr(corr_content)

    with pytest.warns(RuntimeWarning):
        corr.symmetric()
    with pytest.warns(RuntimeWarning):
        corr.anti_symmetric()

    for pad in [0, 2]:
        corr = pe.Corr(corr_content, padding=[pad, pad])
        corr.roll(np.random.randint(100))
        corr.deriv(variant="symmetric")
        corr.deriv(variant="forward")
        corr.deriv(variant="backward")
        corr.deriv(variant="improved")
        corr.deriv(variant="log")
        corr.deriv().deriv()
        corr.second_deriv(variant="symmetric")
        corr.second_deriv(variant="big_symmetric")
        corr.second_deriv(variant="improved")
        corr.second_deriv(variant="log")
        corr.second_deriv().second_deriv()

    for i, e in enumerate(corr.content):
        corr.content[i] = None

    for func in [pe.Corr.deriv, pe.Corr.second_deriv]:
        for variant in ["symmetric", "improved", "forward", "gibberish", None]:
            with pytest.raises(Exception):
                func(corr, variant=variant)


def test_deriv():
    corr_content = []
    for t in range(24):
        exponent = 1.2
        corr_content.append(pe.pseudo_Obs(2 + t ** exponent, 0.2, 't'))

    corr = pe.Corr(corr_content)

    forward = corr.deriv(variant="forward")
    backward = corr.deriv(variant="backward")
    sym = corr.deriv(variant="symmetric")
    assert np.all([o == 0 for o in (0.5 * (forward + backward) - sym)[1:-1]])
    assert np.all([o == 0 for o in (corr.deriv('forward').deriv('backward') - corr.second_deriv())[1:-1]])
    assert np.all([o == 0 for o in (corr.deriv('backward').deriv('forward') - corr.second_deriv())[1:-1]])

    corr_content = []
    exponent = -0.05
    for t in range(24):
        corr_content.append(pe.pseudo_Obs(np.exp(t * exponent), np.exp(t * exponent) * 0.02, 't'))

    corr = pe.Corr(corr_content)

    for o in [(corr.deriv('log') / corr / exponent - 1)[10], (corr.second_deriv('log') / corr / exponent**2 - 1)[12]]:
        o.gamma_method()
        assert (o.is_zero_within_error() and np.isclose(0.0, o.value, 1e-12, 1e-12))


def test_m_eff():
    for padding in [0, 4]:
        my_corr = pe.correlators.Corr([pe.pseudo_Obs(10, 0.1, 't'), pe.pseudo_Obs(9, 0.05, 't'), pe.pseudo_Obs(9, 0.1, 't'), pe.pseudo_Obs(10, 0.05, 't')], padding=[padding, padding])
        my_corr.m_eff('log')
        my_corr.m_eff('logsym')
        my_corr.m_eff('cosh')
        my_corr.m_eff('arccosh')

    corr_content = []
    exponent = -2.2
    for t in range(24):
        corr_content.append(pe.pseudo_Obs(np.exp(t * exponent), np.exp(t * exponent) * 0.02, 't'))

    corr = pe.Corr(corr_content)

    for variant in ['log', 'logsym']:
        o = (corr.m_eff(variant) / exponent + 1)[7]
        o.gamma_method()
        assert (o.is_zero_within_error() and np.isclose(0.0, o.value, 1e-12, 1e-12))

    with pytest.warns(RuntimeWarning):
        my_corr.m_eff('sinh')

    with pytest.raises(ValueError):
        my_corr.m_eff('unkown_variant')


def test_m_eff_negative_values():
    for padding in [0, 4]:
        my_corr = pe.correlators.Corr([1.0 * pe.pseudo_Obs(10, 0.1, 't'), 1.0 * pe.pseudo_Obs(9, 0.05, 't'), -pe.pseudo_Obs(9, 0.1, 't')], padding=[padding, padding])
        m_eff_log = my_corr.m_eff('log')
        assert m_eff_log[padding + 1] is None
        m_eff_cosh = my_corr.m_eff('cosh')
        assert m_eff_cosh[padding + 1] is None
        with pytest.raises(ValueError):
            my_corr.m_eff('logsym')


def test_reweighting():
    my_corr = pe.correlators.Corr([pe.pseudo_Obs(10, 0.1, 't'), pe.pseudo_Obs(0, 0.05, 't')])
    assert my_corr.reweighted is False
    r_my_corr = my_corr.reweight(pe.pseudo_Obs(1, 0.1, 't'))
    assert r_my_corr.reweighted is True


def test_correlate():
    my_corr = pe.correlators.Corr([pe.pseudo_Obs(10, 0.1, 't'), pe.pseudo_Obs(0, 0.05, 't')])
    corr1 = my_corr.correlate(my_corr)
    corr2 = my_corr.correlate(my_corr[0])
    with pytest.raises(TypeError):
        corr3 = my_corr.correlate(7.3)


def test_T_symmetry():
    my_corr = pe.correlators.Corr([pe.pseudo_Obs(10, 0.1, 't'), pe.pseudo_Obs(0, 0.05, 't')])
    with pytest.warns(RuntimeWarning):
        T_symmetric = my_corr.T_symmetry(my_corr)


def test_fit_correlator():
    my_corr = pe.correlators.Corr([pe.pseudo_Obs(1.01324, 0.05, 't'), pe.pseudo_Obs(2.042345, 0.0004, 't')])

    def f(a, x):
        y = a[0] + a[1] * x
        return y

    fit_res = my_corr.fit(f)
    assert fit_res[0] == my_corr[0]
    assert fit_res[1] == my_corr[1] - my_corr[0]

    with pytest.raises(TypeError):
        my_corr.fit(f, "from 0 to 3")
    with pytest.raises(ValueError):
        my_corr.fit(f, [0, 2, 3])


def test_plateau():
    my_corr = pe.correlators.Corr([pe.pseudo_Obs(1.01324, 0.05, 't'), pe.pseudo_Obs(1.042345, 0.008, 't')])

    my_corr.plateau([0, 1], method="fit")
    my_corr.plateau([0, 1], method="mean")
    with pytest.raises(Exception):
        my_corr.plateau()


def test_padded_correlator():
    my_list = [pe.Obs([np.random.normal(1.0, 0.1, 100)], ['ens1']) for o in range(8)]
    my_corr = pe.Corr(my_list, padding=[7, 3])
    my_corr.reweighted
    [o for o in my_corr]


def test_corr_exceptions():
    obs_a = pe.Obs([np.random.normal(0.1, 0.1, 100)], ['test'])
    obs_b = pe.Obs([np.random.normal(0.1, 0.1, 99)], ['test'])
    with pytest.raises(Exception):
        pe.Corr([obs_a, obs_b])

    obs_a = pe.Obs([np.random.normal(0.1, 0.1, 100)], ['test'])
    obs_b = pe.Obs([np.random.normal(0.1, 0.1, 100)], ['test'], idl=[range(1, 200, 2)])
    with pytest.raises(Exception):
        pe.Corr([obs_a, obs_b])

    obs_a = pe.Obs([np.random.normal(0.1, 0.1, 100)], ['test'])
    obs_b = pe.Obs([np.random.normal(0.1, 0.1, 100)], ['test2'])
    with pytest.raises(Exception):
        pe.Corr([obs_a, obs_b])


def test_utility():
    corr_content = []
    for t in range(8):
        exponent = np.random.normal(3, 5)
        corr_content.append(pe.pseudo_Obs(2 + 10 ** exponent, 10 ** (exponent - 1), 't'))

    corr = pe.correlators.Corr(corr_content)
    corr.gamma_method()
    corr.print()
    corr.print([2, 4])
    corr.show()
    corr.show(comp=corr)

    corr.dump('test_dump', datatype="pickle", path='.')
    corr.dump('test_dump', datatype="pickle")
    new_corr = pe.load_object('test_dump.p')
    new_corr.gamma_method()
    os.remove('test_dump.p')
    for o_a, o_b in zip(corr.content, new_corr.content):
        assert np.isclose(o_a[0].value, o_b[0].value)
        assert np.isclose(o_a[0].dvalue, o_b[0].dvalue)
        assert np.allclose(o_a[0].deltas['t'], o_b[0].deltas['t'])

    corr.dump('test_dump', datatype="json.gz", path='.')
    corr.dump('test_dump', datatype="json.gz")
    new_corr = pe.input.json.load_json('test_dump')
    new_corr.gamma_method()
    os.remove('test_dump.json.gz')
    for o_a, o_b in zip(corr.content, new_corr.content):
        assert np.isclose(o_a[0].value, o_b[0].value)
        assert np.isclose(o_a[0].dvalue, o_b[0].dvalue)
        assert np.allclose(o_a[0].deltas['t'], o_b[0].deltas['t'])


def test_prange():
    corr_content = []
    for t in range(8):
        corr_content.append(pe.pseudo_Obs(2 + 10 ** (1.1 * t), 0.2, 't'))
    corr = pe.correlators.Corr(corr_content)

    corr.set_prange([2, 4])
    with pytest.raises(ValueError):
        corr.set_prange([2])
    with pytest.raises(TypeError):
        corr.set_prange([2, 2.3])
    with pytest.raises(ValueError):
        corr.set_prange([4, 1])


def test_matrix_corr():
    corr_aa = _gen_corr(1)
    corr_ab = 0.5 * corr_aa

    corr_mat = pe.Corr(np.array([[corr_aa, corr_ab], [corr_ab, corr_aa]]))
    corr_mat.gamma_method()
    corr_mat.item(0, 0)

    for (ts, sort) in zip([None, 1, 1], ["Eigenvalue", "Eigenvector", None]):
        vecs = corr_mat.GEVP(0, ts=ts, sort=sort)

        corr_0 = corr_mat.projected(vecs[0])
        corr_1 = corr_mat.projected(vecs[1])

    assert np.all([o == 0 for o in corr_0 - corr_aa])
    assert np.all([o == 0 for o in corr_1 - corr_aa])

    corr_mat.matrix_symmetric()
    corr_mat.GEVP(0, state=0)
    corr_mat.Eigenvalue(2, state=0)


def test_corr_none_entries():
    a = pe.pseudo_Obs(1.0, 0.1, 'a')
    la = np.asarray([[a, a], [a, a]])
    n = np.asarray([[None, None], [None, None]])
    x = [la, n]
    matr = pe.Corr(x)
    matr.projected(np.asarray([1.0, 0.0]))

    matr * 2 - 2 * matr
    matr * matr + matr ** 2 / matr

    for func in [np.sqrt, np.log, np.exp, np.sin, np.cos, np.tan, np.sinh, np.cosh, np.tanh]:
        func(matr)


def test_GEVP_warnings():
    corr_aa = _gen_corr(1)
    corr_ab = 0.5 * corr_aa

    corr_mat = pe.Corr(np.array([[corr_aa, corr_ab], [corr_ab, corr_aa]]))
    corr_mat.item(0, 0)

    with pytest.warns(RuntimeWarning):
        corr_mat.GEVP(0, 1, sort="Eigenvalue")

    with pytest.warns(DeprecationWarning):
        corr_mat.GEVP(0, sorted_list="Eigenvalue")


def test_GEVP_exceptions():
    corr_aa = _gen_corr(1)
    corr_ab = 0.5 * corr_aa

    corr_mat = pe.Corr(np.array([[corr_aa, corr_ab], [corr_ab, corr_aa]]))
    corr_mat.item(0, 0)

    with pytest.raises(Exception):
        corr_mat.item(0, 0).projected()

    with pytest.raises(Exception):
        corr_mat.item(0, 0).GEVP(2)

    with pytest.raises(Exception):
        corr_mat.item(0, 0).matrix_symmetric()

    with pytest.raises(Exception):
        corr_mat.GEVP(0, 0, sort=None)

    with pytest.raises(Exception):
        corr_mat.GEVP(0, sort=None)

    with pytest.raises(Exception):
        corr_mat.GEVP(1, 0, sort="Eigenvector")

    with pytest.raises(Exception):
        corr_mat.GEVP(0, 1, sort="This sorting method does not exist.")

    with pytest.raises(Exception):
        corr_mat.plottable()

    with pytest.raises(Exception):
        corr_mat.spaghetti_plot()

    with pytest.raises(Exception):
        corr_mat.show()

    with pytest.raises(Exception):
        corr_mat.m_eff()

    with pytest.raises(Exception):
        corr_mat.Hankel(2)

    with pytest.raises(Exception):
        corr_mat.plateau()

    with pytest.raises(Exception):
        corr_mat.plateau([2, 4])

    with pytest.raises(Exception):
        corr_mat.hankel(3)

    with pytest.raises(Exception):
        corr_mat.fit(lambda x: x[0])

    with pytest.raises(Exception):
        corr_0.item(0, 0)

    with pytest.raises(Exception):
        corr_0.matrix_symmetric()

    n_corr_mat = -corr_mat
    with pytest.raises(np.linalg.LinAlgError):
        n_corr_mat.GEVP(2)


def test_matrix_symmetric():
    corr_aa = _gen_corr(1)
    corr_ab = _gen_corr(0.3)
    corr_ba = _gen_corr(0.2)
    corr_bb = _gen_corr(0.8)
    corr_mat = pe.Corr(np.array([[corr_aa, corr_ab], [corr_ba, corr_bb]]))

    sym_corr_mat = corr_mat.matrix_symmetric()

    assert np.all([np.all(o == o.T) for o in sym_corr_mat])

    t_obs = pe.pseudo_Obs(1.0, 0.1, 'test')
    o_mat = np.array([[t_obs, t_obs], [t_obs, t_obs]])
    corr1 = pe.Corr([o_mat, None, o_mat])
    corr2 = pe.Corr([o_mat, np.array([[None, None], [None, None]]), o_mat])
    corr3 = pe.Corr([o_mat, np.array([[t_obs, None], [None, t_obs]], dtype=object), o_mat])
    corr1.matrix_symmetric()
    corr2.matrix_symmetric()
    corr3.matrix_symmetric()


def test_is_matrix_symmetric():
    corr_data = []
    for t in range(4):
        mat = np.zeros((4, 4), dtype=object)
        for i in range(4):
            for j in range(i, 4):
                obs = pe.pseudo_Obs(0.1, 0.047, "rgetrasrewe53455b153v13v5/*/*sdfgb")
                mat[i, j] = obs
                if i != j:
                    mat[j, i] = obs
        corr_data.append(mat)
    corr = pe.Corr(corr_data, padding=[0, 2])

    assert corr.is_matrix_symmetric()
    corr[0][0, 1] = 1.0 * corr[0][0, 1]
    assert corr.is_matrix_symmetric()
    corr[3][2, 1] = (1 + 1e-14) * corr[3][2, 1]
    assert corr.is_matrix_symmetric()
    corr[0][0, 1] = 1.1 * corr[0][0, 1]
    assert not corr.is_matrix_symmetric()


def test_GEVP_solver():

    mat1 = np.random.rand(15, 15)
    mat2 = np.random.rand(15, 15)
    mat1 = mat1 @ mat1.T
    mat2 = mat2 @ mat2.T

    sp_val, sp_vecs = scipy.linalg.eigh(mat1, mat2)
    sp_vecs = [sp_vecs[:, np.argsort(sp_val)[-i]] for i in range(1, sp_vecs.shape[0] + 1)]
    sp_vecs = [v / np.sqrt((v.T @ mat2 @ v)) for v in sp_vecs]

    assert np.allclose(sp_vecs, pe.correlators._GEVP_solver(mat1, mat2), atol=1e-14)
    assert np.allclose(np.abs(sp_vecs), np.abs(pe.correlators._GEVP_solver(mat1, mat2, method='cholesky')))


def test_GEVP_none_entries():
    t_obs = pe.pseudo_Obs(1.0, 0.1, 'test')
    t_obs2 = pe.pseudo_Obs(0.1, 0.1, 'test')

    o_mat = np.array([[t_obs, t_obs2], [t_obs2, t_obs2]])
    n_arr = np.array([[None, None], [None, None]])

    corr = pe.Corr([o_mat, o_mat, o_mat, o_mat, o_mat, o_mat, None, o_mat, n_arr, None, o_mat])
    corr.GEVP(t0=2)


def test_hankel():
    corr_content = []
    for t in range(8):
        exponent = 1.2
        corr_content.append(pe.pseudo_Obs(2 + t ** exponent, 0.2, 't'))

    corr = pe.Corr(corr_content)
    corr.Hankel(2)
    corr.Hankel(6, periodic=True)


def test_thin():
    c = pe.Corr([pe.pseudo_Obs(i, .1, 'test') for i in range(10)])
    c *= pe.cov_Obs(1., .1, '#ren')
    thin = c.thin()
    thin.gamma_method()
    thin.fit(lambda a, x: a[0] * x)
    c.thin(offset=1)
    c.thin(3, offset=1)


def test_corr_matrix_none_entries():
    dim = 8
    x = np.arange(dim)
    y = 2 * np.exp(-0.06 * x) + np.random.normal(0.0, 0.15, dim)
    yerr = [0.1] * dim

    oy = []
    for i, item in enumerate(x):
        oy.append(pe.pseudo_Obs(y[i], yerr[i], 'test'))

    corr = pe.Corr(oy)
    corr = corr.deriv()
    pe.Corr(np.array([[corr, corr], [corr, corr]]))


def test_corr_vector_operations():
    my_corr = _gen_corr(1.0)
    my_vec = np.arange(1, 17)

    my_corr + my_vec
    my_corr - my_vec
    my_corr * my_vec
    my_corr / my_vec

    assert np.all([o == 0 for o in ((my_corr + my_vec) - my_vec) - my_corr])
    assert np.all([o == 0 for o in ((my_corr - my_vec) + my_vec) - my_corr])
    assert np.all([o == 0 for o in ((my_corr * my_vec) / my_vec) - my_corr])
    assert np.all([o == 0 for o in ((my_corr / my_vec) * my_vec) - my_corr])


def test_spaghetti_plot():
    corr = _gen_corr(12, 50)
    corr += pe.pseudo_Obs(0.0, 0.1, 'another_ensemble|r0')
    corr += pe.pseudo_Obs(0.0, 0.1, 'another_ensemble|r1')
    corr += pe.cov_Obs(0.0, 0.01 ** 2, 'covobs')

    corr.spaghetti_plot(True)
    corr.spaghetti_plot(False)
    plt.close('all')


def _gen_corr(val, samples=2000):
    corr_content = []
    for t in range(16):
        corr_content.append(pe.pseudo_Obs(val, 0.1, 't', samples))

    return pe.correlators.Corr(corr_content)


def test_prune():

    corr_aa = _gen_corr(1)
    corr_ab = 0.5 * corr_aa
    corr_ac = 0.25 * corr_aa

    corr_mat = pe.Corr(np.array([[corr_aa, corr_ab, corr_ac], [corr_ab, corr_aa, corr_ab], [corr_ac, corr_ab, corr_aa]]))

    p = corr_mat.prune(2)
    assert(all([o.is_zero() for o in p.item(0, 1)]))
    a = [(o - 1) for o in p.item(1, 1)]
    [o.gamma_method() for o in a]
    assert(all([o.is_zero_within_error() for o in a]))

    with pytest.raises(Exception):
        corr_mat.prune(3)
        corr_mat.prune(4)


def test_complex_Corr():
    o1 = pe.pseudo_Obs(1.0, 0.1, "test")
    cobs = pe.CObs(o1, -o1)
    ccorr = pe.Corr([cobs, cobs, cobs])
    assert np.all([ccorr.imag[i] == -ccorr.real[i] for i in range(ccorr.T)])
    print(ccorr)
    mcorr = pe.Corr(np.array([[ccorr, ccorr], [ccorr, ccorr]]))
    assert np.all([mcorr.imag[i] == -mcorr.real[i] for i in range(mcorr.T)])


def test_corr_no_filtering():
    li = [-pe.pseudo_Obs(.2, .1, 'a', samples=10) for i in range(96)]
    for i in range(len(li)):
        li[i].idl['a'] = range(1, 21, 2)
    c = pe.Corr(li)
    b = pe.pseudo_Obs(1, 1e-11, 'a', samples=30)
    c *= b
    assert np.all([c[0].idl == o.idl for o in c])


def test_corr_symmetric():
    obs = []
    for _ in range(4):
        obs.append(pe.pseudo_Obs(np.random.rand(), 0.1, "test"))

    for corr in [pe.Corr([obs[0] + 8, obs[1], obs[2], obs[3]]),
                 pe.Corr([obs[0] + 8, obs[1], obs[2], None]),
                 pe.Corr([None, obs[1], obs[2], obs[3]])]:
        scorr = corr.symmetric()
        assert scorr[1] == scorr[3]
        assert scorr[2] == corr[2]
        assert scorr[0] == corr[0]


def test_error_GEVP():
    corr = pe.input.json.load_json("tests/data/test_matrix_corr.json.gz")
    t0, ts, state = 3, 6, 0
    vec_regular = corr.GEVP(t0=t0)[state][ts]
    vec_errors = corr.GEVP(t0=t0, vector_obs=True, auto_gamma=True)[state][ts]
    vec_regular_chol = corr.GEVP(t0=t0, method='cholesky')[state][ts]
    print(vec_errors)
    print(type(vec_errors[0]))
    assert(np.isclose(np.asarray([e.value for e in vec_errors]), vec_regular).all())
    assert(all([e.dvalue > 0. for e in vec_errors]))

    projected_regular = corr.projected(vec_regular).content[ts + 1][0]
    projected_errors = corr.projected(vec_errors).content[ts + 1][0]
    projected_regular.gamma_method()
    projected_errors.gamma_method()
    assert(projected_errors.dvalue > projected_regular.dvalue)
    assert(corr.GEVP(t0, vector_obs=True)[state][42] is None)

    assert(np.isclose(vec_regular_chol, vec_regular).all())
    assert(np.isclose(corr.GEVP(t0=t0, state=state)[ts], vec_regular).all())


def test_corr_array_ndim1_init():
    y = [pe.pseudo_Obs(2 + np.random.normal(0.0, 0.1), .1, 't') for i in np.arange(5)]
    cc1 = pe.Corr(y)
    cc2 = pe.Corr(np.array(y))
    assert np.all([o1 == o2 for o1, o2 in zip(cc1, cc2)])


def test_corr_array_ndim3_init():
    y = np.array([pe.pseudo_Obs(np.random.normal(2.0, 0.1), .1, 't') for i in np.arange(12)]).reshape(3, 2, 2)
    tt1 = pe.Corr(list(y))
    tt2 = pe.Corr(y)
    tt3 = pe.Corr(np.array([pe.Corr(o) for o in y.reshape(3, 4).T]).reshape(2, 2))
    assert np.all([o1 == o2 for o1, o2 in zip(tt1, tt2)])
    assert np.all([o1 == o2 for o1, o2 in zip(tt1, tt3)])
    assert tt1.T == y.shape[0]
    assert tt1.N == y.shape[1] == y.shape[2]

    with pytest.raises(ValueError):
        pe.Corr(y.reshape(6, 2, 1))


def test_two_matrix_corr_inits():
    T = 4
    rn = lambda : np.random.normal(0.5, 0.1)

    # Generate T random CObs in a list
    list_of_timeslices =[]
    for i in range(T):
        re = pe.pseudo_Obs(rn(), rn(), "test")
        im = pe.pseudo_Obs(rn(), rn(), "test")
        list_of_timeslices.append(pe.CObs(re, im))

    # First option: Correlator of matrix of correlators
    corr = pe.Corr(list_of_timeslices)
    mat_corr1 = pe.Corr(np.array([[corr, corr], [corr, corr]]))

    # Second option: Correlator of list of arrays per timeslice
    list_of_arrays = [np.array([[elem, elem], [elem, elem]]) for elem in list_of_timeslices]
    mat_corr2 = pe.Corr(list_of_arrays)

    for el in mat_corr1 - mat_corr2:
        assert np.all(el == 0)


def test_matmul_overloading():
    N = 4
    rn = lambda : np.random.normal(0.5, 0.1)

    # Generate N^2 random CObs and assemble them in an array
    ll =[]
    for i in range(N ** 2):
        re = pe.pseudo_Obs(rn(), rn(), "test")
        im = pe.pseudo_Obs(rn(), rn(), "test")
        ll.append(pe.CObs(re, im))
    mat = np.array(ll).reshape(N, N)

    # Multiply with gamma matrix
    corr = pe.Corr([mat] * 4, padding=[0, 1])

    # __matmul__
    mcorr = corr @ pe.dirac.gammaX
    comp = mat @ pe.dirac.gammaX
    for i in range(4):
        assert np.all(mcorr[i] == comp)

    # __rmatmul__
    mcorr = pe.dirac.gammaX @ corr
    comp = pe.dirac.gammaX @ mat
    for i in range(4):
        assert np.all(mcorr[i] == comp)

    test_mat = pe.dirac.gamma5 + pe.dirac.gammaX
    icorr = corr @ test_mat @ np.linalg.inv(test_mat)
    tt = corr - icorr
    for i in range(4):
        assert np.all(tt[i] == 0)

    # associative property
    tt = (corr.real @ pe.dirac.gammaX + corr.imag @ (pe.dirac.gammaX * 1j)) - corr @ pe.dirac.gammaX
    for el in tt:
        if el is not None:
            assert np.all(el == 0)

    corr2 = corr @ corr
    for i in range(4):
        np.all(corr2[i] == corr[i] @ corr[i])


def test_matrix_trace():
    N = 4
    rn = lambda : np.random.normal(0.5, 0.1)

    # Generate N^2 random CObs and assemble them in an array
    ll =[]
    for i in range(N ** 2):
        re = pe.pseudo_Obs(rn(), rn(), "test")
        im = pe.pseudo_Obs(rn(), rn(), "test")
        ll.append(pe.CObs(re, im))
    mat = np.array(ll).reshape(N, N)

    corr = pe.Corr([mat] * 4)

    # Explicitly check trace
    for el in corr.trace():
        el == np.sum(np.diag(mat))

    # Trace is cyclic
    for one, two in zip((pe.dirac.gammaX @ corr).trace(), (corr @ pe.dirac.gammaX).trace()):
        assert np.all(one == two)

    # Antisymmetric matrices are traceless.
    mat = (mat - mat.T) / 2
    corr = pe.Corr([mat] * 4)
    for el in corr.trace():
        assert el == 0

    with pytest.raises(ValueError):
        corr.item(0, 0).trace()


def test_corr_roll():
    T = 4
    rn = lambda : np.random.normal(0.5, 0.1)

    ll = []
    for i in range(T):
        re = pe.pseudo_Obs(rn(), rn(), "test")
        im = pe.pseudo_Obs(rn(), rn(), "test")
        ll.append(pe.CObs(re, im))

    # Rolling by T should produce the same correlator
    corr = pe.Corr(ll)
    tt = corr - corr.roll(T)
    for el in tt:
        assert np.all(el == 0)

    mcorr = pe.Corr(np.array([[corr, corr + 0.1], [corr - 0.1, 2 * corr]]))
    tt = mcorr.roll(T) - mcorr
    for el in tt:
        assert np.all(el == 0)


def test_correlator_comparison():
    scorr = pe.Corr([pe.pseudo_Obs(0.3, 0.1, "test") for o in range(4)])
    mcorr = pe.Corr(np.array([[scorr, scorr], [scorr, scorr]]))
    for corr in [scorr, mcorr]:
        assert (corr == corr).all()
        assert np.all(corr == 1 * corr)
        assert np.all(corr == (1 + 1e-16) * corr)
        assert not np.all(corr == (1 + 1e-5) * corr)
        assert np.all(corr == 1 / (1 / corr))
        assert np.all(corr - corr == 0)
        assert np.all(corr * 0 == 0)
        assert np.all(0 * corr == 0)
        assert np.all(0 * corr + scorr[2] == scorr[2])
        assert np.all(-corr == 0 - corr)
        assert np.all(corr ** 2 == corr * corr)
    acorr = pe.Corr([scorr[0]] * 6)
    assert np.all(acorr == scorr[0])
    assert not np.all(acorr == scorr[1])

    mcorr[1][0, 1] = None
    assert not np.all(mcorr == pe.Corr(np.array([[scorr, scorr], [scorr, scorr]])))

    pcorr = pe.Corr([pe.pseudo_Obs(0.25, 0.1, "test") for o in range(2)], padding=[1, 1])
    assert np.all(pcorr == pcorr)
    assert np.all(1 * pcorr == pcorr)


def test_corr_item():
    corr_aa = _gen_corr(1)
    corr_ab = 0.5 * corr_aa

    corr_mat = pe.Corr(np.array([[corr_aa, corr_ab], [corr_ab, corr_aa]]))
    corr_mat.item(0, 0)
    assert corr_mat[0].item(0, 1) == corr_mat.item(0, 1)[0]


def test_complex_add_and_mul():
    o = pe.pseudo_Obs(1.0, 0.3, "my_r345sfg16£$%&$%^%$^$", samples=47)
    co = pe.CObs(o, 0.341 * o)
    for obs in [o, co]:
        cc = pe.Corr([obs for _ in range(4)])
        cc += 2j
        cc = cc * 4j
        cc.real + cc.imag


def test_prune_with_Nones():
    N = 3
    T = 10

    front_padding = 1
    back_padding = T // 2

    Ntrunc = N - 1
    t0proj = 2
    tproj = 3

    corr_content = np.array([[[pe.pseudo_Obs((i+j+1)**(-t), .01, "None_prune_test") for i in range(N)] for j in range(N)] for t in range(T // 2 - front_padding)])
    unpadded_corr = pe.Corr(corr_content)
    padded_corr = pe.Corr(corr_content, padding=[front_padding, back_padding])

    tmp_corr = unpadded_corr.prune(Ntrunc, t0proj=t0proj-front_padding, tproj=tproj-front_padding)
    pruned_then_padded = pe.Corr(tmp_corr.content, padding=[front_padding, back_padding])
    padded_then_pruned = padded_corr.prune(Ntrunc, t0proj=t0proj, tproj=tproj)

    for t in range(T):
        assert np.all(pruned_then_padded.content[t] == padded_then_pruned.content[t])


def test_Corr_padding_default_not_shared():
    data = [pe.pseudo_Obs(i + 1, 0.1, "e") for i in range(5)]
    c1 = pe.Corr(data)
    c2 = pe.Corr(data)
    assert c1 is not c2
    assert len(c1.content) == len(c2.content) == 5
    assert all(a[0] == b[0] for a, b in zip(c1.content, c2.content))


def test_Corr_unhashable():
    c = pe.Corr([pe.pseudo_Obs(i + 1, 0.1, "e") for i in range(5)])
    with pytest.raises(TypeError):
        hash(c)


def _value(x):
    if isinstance(x, pe.CObs):
        return complex(_value(x.real), _value(x.imag))
    return x.value if isinstance(x, pe.Obs) else x



def _values(matrix):
    matrix = np.asarray(matrix)
    return np.array([_value(x) for x in matrix.flat], dtype=complex).reshape(matrix.shape)



def _aligned_vectors(a, b, reference, reference_b):
    _, v = scipy.linalg.eigh(a, b)
    overlap = np.diag(reference.conj().T @ reference_b @ v)
    return v * np.exp(-1j * np.angle(overlap))[None, :]



@pytest.fixture
def complex_corr():
    rng = np.random.default_rng(255)
    n, samples = 3, 80
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)) + 2 * np.eye(n)
    energies = np.array([0.2, 0.5, 0.9])
    noise = rng.normal(size=(samples, 6))
    noise -= noise.mean(axis=0)
    patterns = rng.normal(size=(6, n, n)) + 1j * rng.normal(size=(6, n, n))
    patterns = (patterns + patterns.conj().transpose(0, 2, 1)) / 2
    content = []
    for t in range(8):
        if t in (0, 5):
            content.append(None)
            continue
        mean = z @ np.diag(np.exp(-energies * t)) @ z.conj().T
        raw = mean + 1e-3 * np.exp(-0.3 * t) * np.einsum('sk,kij->sij', noise, patterns)
        matrix = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(i, n):
                real = pe.Obs([raw[:, i, j].real], ['complex|r1'], idl=[range(10, 170, 2)])
                if i == j:
                    matrix[i, j] = real  # Deliberately mix real Obs and CObs.
                else:
                    imag = pe.Obs([raw[:, i, j].imag], ['complex|r1'], idl=[range(10, 170, 2)])
                    matrix[i, j] = pe.CObs(real, imag)
                    matrix[j, i] = matrix[i, j].conjugate()
        content.append(matrix)
    return pe.Corr(content), energies



@pytest.mark.parametrize('method', ['eigh', 'cholesky'])
@pytest.mark.parametrize('sort', ['Eigenvalue', 'Eigenvector', None])
@pytest.mark.parametrize('vector_obs', [False, True])
def test_complex_gevp_modes(complex_corr, method, sort, vector_obs):
    corr, energies = complex_corr
    kwargs = dict(t0=1, sort=sort, method=method, vector_obs=vector_obs)
    if sort != 'Eigenvalue':
        kwargs['ts'] = 3
    vectors = corr.GEVP(**kwargs)
    b = _values(corr[1])
    for state in range(3):
        vec = vectors[state] if sort is None else vectors[state][3]
        numeric = _values(vec)
        np.testing.assert_allclose(numeric.conj() @ b @ numeric, 1, atol=2e-13)
        lam = np.exp(-energies[state] * 2)
        np.testing.assert_allclose(_values(corr[3]) @ numeric, lam * b @ numeric, atol=2e-13)
        eigenvalue = corr.Eigenvalue(state=state, **kwargs)
        assert eigenvalue[0] is None and eigenvalue[5] is None
        assert isinstance(eigenvalue[3], pe.Obs)
        np.testing.assert_allclose(eigenvalue[3].value, lam, atol=2e-13)
        normalized = corr.projected(vectors[state], normalize=True)
        assert normalized[0] is None and normalized[5] is None
        if vector_obs:
            assert all(isinstance(x, pe.CObs) for x in vec)
            norm = vec.conjugate() @ corr[1] @ vec
            assert (norm.real - 1).is_zero() and norm.imag.is_zero()
    if sort is not None:
        assert vectors[0][0] is None and vectors[0][5] is None
    selected = corr.GEVP(state=1, **kwargs)
    assert len(selected) == (3 if sort is None else corr.T)



@pytest.mark.parametrize('method', ['eigh', 'cholesky'])
@pytest.mark.parametrize('vector_obs', [False, True])
@pytest.mark.parametrize('complex_input', [False, True], ids=['real', 'complex'])
def test_fixed_gevp_solver_failure(complex_corr, monkeypatch, method, vector_obs, complex_input):
    """A failed fixed-time solve must not return C_00 via projected(None)."""
    corr, _ = complex_corr
    if not complex_input:
        corr = corr.real
    kwargs = dict(t0=1, ts=3, sort=None, method=method, vector_obs=vector_obs)
    assert corr.Eigenvalue(**kwargs)[3] is not None  # The unmodified input is valid.

    def failed_solver(*args, **kwargs):
        raise np.linalg.LinAlgError('injected convergence failure')

    if vector_obs and not complex_input:
        monkeypatch.setattr(pe.linalg, 'eigv', failed_solver)
    elif vector_obs or method == 'eigh':
        monkeypatch.setattr(scipy.linalg, 'eigh', failed_solver)
    else:
        monkeypatch.setattr(np.linalg, 'eigh', failed_solver)
    for operation in (corr.GEVP, corr.Eigenvalue):
        with pytest.raises(np.linalg.LinAlgError, match='injected convergence failure'):
            operation(**kwargs)



def test_eigenvalue_rejects_failed_vector(complex_corr, monkeypatch):
    """Keep explicit default projections valid, but reject missing GEVP vectors."""
    corr, _ = complex_corr
    np.testing.assert_allclose(_value(corr.projected()[3]), _value(corr[3][0, 0]))
    monkeypatch.setattr(pe.Corr, 'GEVP', lambda self, *args, **kwargs: [None] * self.N)
    with pytest.raises(ValueError, match='did not return an eigenvector'):
        corr.Eigenvalue(t0=1, ts=3, sort=None)



@pytest.mark.parametrize('method', ['eigh', 'cholesky'])
@pytest.mark.parametrize('sort', ['Eigenvalue', 'Eigenvector'])
@pytest.mark.parametrize('vector_obs', [False, True])
@pytest.mark.parametrize('complex_input', [False, True], ids=['real', 'complex'])
def test_time_dependent_gevp_solver_failure(complex_corr, monkeypatch, method, sort, vector_obs, complex_input):
    """An unusable separation must not discard the other separations."""
    corr, _ = complex_corr
    if not complex_input:
        corr = corr.real
    kwargs = dict(t0=1, sort=sort, method=method, vector_obs=vector_obs)
    if sort == 'Eigenvector':
        kwargs['ts'] = 3
    baseline = corr.Eigenvalue(**kwargs)
    failed_mean = _values(corr[6])
    original_solver = pe.correlators._GEVP_solver

    def fail_one_separation(gt, *args, **kwargs):
        if np.allclose(_values(gt), failed_mean):
            raise np.linalg.LinAlgError('injected convergence failure')
        return original_solver(gt, *args, **kwargs)

    monkeypatch.setattr(pe.correlators, '_GEVP_solver', fail_one_separation)
    vectors = corr.GEVP(**kwargs)
    assert all(state[6] is None for state in vectors)
    projected = corr.Eigenvalue(**kwargs)
    assert projected[6] is None and projected[5] is None
    # Check both earlier and later successful solves, including their errors.
    _assert_same_observables([projected[2], projected[7]], [baseline[2], baseline[7]])



@pytest.mark.parametrize('error', [TypeError, ValueError, AttributeError])
@pytest.mark.parametrize('complex_input', [False, True], ids=['real', 'complex'])
def test_time_dependent_gevp_unexpected_error(complex_corr, monkeypatch, error, complex_input):
    """Do not hide invalid inputs or programming errors as failed solves."""
    corr, _ = complex_corr
    if not complex_input:
        corr = corr.real

    def broken_solver(*args, **kwargs):
        raise error('unexpected solver error')

    monkeypatch.setattr(pe.correlators, '_GEVP_solver', broken_solver)
    with pytest.raises(error, match='unexpected solver error'):
        corr.GEVP(t0=1)



def test_complex_frozen_projection_and_distinct_vectors(complex_corr):
    corr, _ = complex_corr
    v = corr.GEVP(t0=1, ts=3, sort=None)[0]
    projected = corr.Eigenvalue(t0=1, ts=3, sort=None)
    # Independent contraction of the input fluctuations: no d v terms.
    delta = np.zeros(80)
    for i in range(3):
        delta += abs(v[i])**2 * corr[4][i, i].deltas['complex|r1']
        for j in range(i + 1, 3):
            weight = v[i].conjugate() * v[j]
            entry = corr[4][i, j]
            delta += 2 * (weight.real * entry.real.deltas['complex|r1']
                          - weight.imag * entry.imag.deltas['complex|r1'])
    np.testing.assert_allclose(projected[4].deltas['complex|r1'], delta, atol=1e-14)
    assert projected[4].idl['complex|r1'] == range(10, 170, 2)
    right = np.array([1 + 2j, 0.3j, -0.2])
    left = np.array([0.4, 1j, 0.8 - 0.5j])
    for vl, vr in [(left, right), ([left] * corr.T, right),
                   (left, [right] * corr.T), ([left] * corr.T, [right] * corr.T)]:
        result = corr.projected(vl, vr)
        np.testing.assert_allclose(_value(result[4]), left.conj() @ _values(corr[4]) @ right)
        assert abs(_value(result[4]).imag) > 0.01



def test_complex_hermitian_helpers_and_components(complex_corr):
    corr, _ = complex_corr
    assert corr.is_matrix_hermitian()
    assert all(isinstance(x, pe.Obs) for x in corr.real[3].flat)
    assert all(isinstance(x, pe.Obs) for x in corr.imag[3].flat)
    modified = pe.Corr([None if m is None else m.copy() for m in corr.content])
    modified[3][0, 1] += 0.2j
    assert not modified.is_matrix_hermitian()
    averaged = modified.matrix_hermitian()
    assert averaged.is_matrix_hermitian()
    assert averaged[0] is None and averaged[5] is None
    # The imaginary diagonal must also be checked and removed when averaging.
    modified[3][0, 0] = pe.CObs(modified[3][0, 0], 0.1)
    assert not modified.is_matrix_hermitian()
    assert modified.matrix_hermitian().is_matrix_hermitian()
    modified.GEVP(t0=1, ts=3, sort=None)
    with pytest.raises(TypeError):
        corr.item(0, 0).is_matrix_hermitian()
    with pytest.raises(ValueError):
        corr.item(0, 0).matrix_hermitian()



def test_complex_prune(complex_corr):
    corr, energies = complex_corr
    reference = pe.Corr(corr.content)
    reduced = corr.prune(2, tproj=3, t0proj=1, basematrix=reference)
    assert reduced.N == 2 and reduced[0] is None and reduced[5] is None
    assert reduced.is_matrix_hermitian()
    for state in range(2):
        result = reduced.Eigenvalue(t0=1, state=state)
        np.testing.assert_allclose(result[4].value, np.exp(-energies[state] * 3), atol=2e-13)



def test_complex_sorting_phase_invariance(complex_corr):
    corr, _ = complex_corr
    # Use a complete sequence with undefined t <= t0.
    vectors = [None, None] + [corr.GEVP(t0=1, ts=t, sort=None) if corr[t] is not None else None
                              for t in range(2, corr.T)]
    rng = np.random.default_rng(22)
    phased = [None if v is None else v * np.exp(1j * rng.uniform(-3, 3, (3, 1)))
              for v in vectors]
    original = pe.correlators._sort_vectors(vectors, 3)
    changed = pe.correlators._sort_vectors(phased, 3)
    for a, b in zip(original, changed, strict=True):
        if a is not None:
            for u, v in zip(a, b, strict=True):
                np.testing.assert_allclose(np.outer(u, u.conj()), np.outer(v, v.conj()), atol=2e-13)



def test_complex_vector_mc_derivative_and_auto_gamma(complex_corr):
    corr, _ = complex_corr
    t, t0 = 3, 1
    a, b = _values(corr[t]), _values(corr[t0])
    _, reference = scipy.linalg.eigh(a, b)
    vectors = corr.GEVP(t0=t0, ts=t, sort=None, vector_obs=True, auto_gamma=True)
    source = 'complex|r1'

    def fluctuations(matrix):
        delta = np.empty((3, 3, 80), dtype=complex)
        for ij, entry in np.ndenumerate(matrix):
            if isinstance(entry, pe.CObs):
                delta[ij] = entry.real.deltas[source] + 1j * entry.imag.deltas[source]
            else:
                delta[ij] = entry.deltas[source]
        return delta

    da, db = fluctuations(corr[t]), fluctuations(corr[t0])
    h = 1e-3
    for sample in range(80):
        plus = _aligned_vectors(a + h * da[:, :, sample], b + h * db[:, :, sample], reference, b)
        minus = _aligned_vectors(a - h * da[:, :, sample], b - h * db[:, :, sample], reference, b)
        expected = (plus - minus) / (2 * h)
        for state, vec in enumerate(vectors):
            actual = np.array([x.real.deltas[source][sample] + 1j * x.imag.deltas[source][sample] for x in vec])
            np.testing.assert_allclose(actual, expected[:, -state - 1], atol=2e-11)
    assert all(hasattr(x.imag, 'dvalue') for vec in vectors for x in vec)
    assert all(hasattr(x.real, 'dvalue') for vec in vectors for x in vec)
    # The list-valued auto_gamma path and determinant sorting must handle CObs.
    corr.GEVP(t0=1, ts=3, sort='Eigenvector', vector_obs=True, auto_gamma=True)



@pytest.mark.filterwarnings('ignore:Covariance matrix is not positive semi-definite:RuntimeWarning')
def test_complex_generalized_covariance():
    rng = np.random.default_rng(91)
    z = rng.normal(size=(8, 8))
    covariance = z @ z.T * 1e-6
    mean = np.array([2., 3., 0.2, 0.4, 1.2, 1.7, 0.1, -0.15])
    p = pe.cov_Obs(mean, covariance, 'both-times')

    def matrices(x):
        return tuple(np.array([[x[k], x[k + 2] + 1j * x[k + 3]],
                               [x[k + 2] - 1j * x[k + 3], x[k + 1]]]) for k in (0, 4))

    obs = tuple(np.array([[p[k], pe.CObs(p[k + 2], p[k + 3])],
                          [pe.CObs(p[k + 2], -p[k + 3]), p[k + 1]]]) for k in (0, 4))
    corr = pe.Corr([obs[1], obs[0]])
    vectors = corr.GEVP(t0=0, ts=1, sort=None, vector_obs=True)
    eigenvalues = [corr.Eigenvalue(t0=0, ts=1, sort=None, state=s, vector_obs=True)[1] for s in (1, 0)]
    columns = np.array(vectors).T[:, ::-1]
    outputs = eigenvalues + [x.real for x in columns.flat] + [x.imag for x in columns.flat]
    pe.gm(outputs)
    a, b = matrices(mean)
    _, reference = scipy.linalg.eigh(a, b)
    h = 1e-5
    jac = np.empty((len(outputs), 8))
    for k in range(8):
        delta = np.zeros(8)
        delta[k] = h
        packed = []
        for sign in (1, -1):
            aa, bb = matrices(mean + sign * delta)
            ww = scipy.linalg.eigh(aa, bb)[0]
            vv = _aligned_vectors(aa, bb, reference, b)
            packed.append(np.concatenate((ww, vv.real.ravel(), vv.imag.ravel())))
        jac[:, k] = (packed[0] - packed[1]) / (2 * h)
    np.testing.assert_allclose(pe.covariance(outputs), jac @ covariance @ jac.T, atol=1e-13, rtol=2e-7)
    for state in range(2):
        v = vectors[state]
        lam = eigenvalues[1 - state]
        residual = obs[0] @ v - (obs[1] @ v) * lam
        assert all(entry.is_zero() for entry in residual)



@pytest.mark.parametrize('sort', ['Eigenvalue', 'Eigenvector', None])
def test_complex_degeneracy_and_missing_reference(sort):
    unit = pe.cov_Obs(1., 0., 'constant')
    identity = np.array([[pe.CObs(unit), pe.CObs(0 * unit)], [pe.CObs(0 * unit), pe.CObs(unit)]])
    corr = pe.Corr([identity, identity * 2, identity * 3])
    kwargs = dict(t0=0, sort=sort)
    if sort != 'Eigenvalue':
        kwargs['ts'] = 1
    # Zero determinant candidates are expected for this exactly degenerate
    # sorting problem; NumPy 1.26 can issue spurious divide/invalid warnings.
    with np.errstate(divide='ignore', invalid='ignore'):
        corr.GEVP(**kwargs)  # A numerical choice of basis is still valid.
    if sort is None:
        with pytest.raises(np.linalg.LinAlgError, match='degenerate'):
            corr.GEVP(vector_obs=True, **kwargs)
    elif sort == 'Eigenvector':
        # State tracking needs a usable sorting reference even if other
        # time separations can individually be marked as failed.
        with pytest.raises(np.linalg.LinAlgError, match='sorting reference'):
            corr.GEVP(vector_obs=True, **kwargs)
    else:
        assert all(v is None for state in corr.GEVP(vector_obs=True, **kwargs) for v in state)
    corr.content[0] = None
    for propagate in (False, True):
        with pytest.raises(ValueError, match='not defined'):
            corr.GEVP(vector_obs=propagate, **kwargs)



def test_complex_numeric_zeros_are_defined():
    a = pe.cov_Obs(1., 1e-5, 'diagonal')
    b = pe.cov_Obs(0.2, 2e-6, 'off-diagonal')
    metric = np.array([[a, pe.CObs(0, b), 0],
                       [pe.CObs(0, -b), 2 * a, 0j], [0, 0, 3 * a]])
    target = metric + np.diag([0.1, 0.4, 0.9])
    corr = pe.Corr([metric, target])
    for propagate in (False, True):
        v = corr.GEVP(t0=0, ts=1, sort=None, vector_obs=propagate)[0]
        assert v is not None
        assert corr.Eigenvalue(t0=0, ts=1, sort=None, vector_obs=propagate)[1] is not None
    assert corr.matrix_hermitian()[1] is not None



def test_complex_and_real_timeslices():
    a = pe.cov_Obs(1., 1e-5, 'shared')
    reference = np.array([[a, 0], [0, 2 * a]])
    complex_time = np.array([[2 * a, pe.CObs(0, 0.1 * a)],
                             [pe.CObs(0, -0.1 * a), 3 * a]])
    real_time = np.array([[1.5 * a, 0.2 * a], [0.2 * a, 2 * a]])
    corr = pe.Corr([reference, complex_time, real_time])
    vectors = corr.GEVP(t0=0, vector_obs=True, auto_gamma=True)
    for state in range(2):
        vec = vectors[state][2]
        assert all(isinstance(x, pe.CObs) and x.imag.is_zero() for x in vec)
        norm = vec.conjugate() @ reference @ vec
        assert (norm.real - 1).is_zero() and norm.imag.is_zero()



def _align_observable_vector(vector, reference, metric):
    """Align to prescribed input vectors, without calling an eigensolver."""
    vector = [x if isinstance(x, pe.CObs) else pe.CObs(x) for x in vector]
    overlap = sum(weight * x for weight, x in zip(reference.conj() @ metric, vector, strict=True))
    phase = overlap.conjugate() / abs(overlap)
    return np.array([x * phase for x in vector])



def _assert_same_observables(actual, expected):
    """Check means, errors and source fluctuations/gradients, not just overlap."""
    pe.gm(actual)
    pe.gm(expected)
    np.testing.assert_allclose([x.value for x in actual], [x.value for x in expected], atol=2e-12)
    np.testing.assert_allclose([x.dvalue for x in actual], [x.dvalue for x in expected],
                               rtol=2e-9, atol=2e-13)
    for result, original in zip(actual, expected, strict=True):
        for name in set(result.deltas) | set(original.deltas):
            np.testing.assert_allclose(result.deltas.get(name, 0), original.deltas.get(name, 0),
                                       rtol=2e-9, atol=2e-12)
        for name in set(result.cov_names) | set(original.cov_names):
            # derived_observable may introduce a zero-variance dummy source for
            # constants. Only the physical source gradients carry information.
            if name == '###dummy_covobs###':
                continue
            np.testing.assert_allclose(result.covobs[name].grad, original.covobs[name].grad,
                                       rtol=2e-9, atol=2e-12)
    # covariance() normalizes by component errors internally. Exact constants
    # (e.g. real-vector imaginary parts) were checked above; omit their zero
    # rows/columns from this additional covariance comparison.
    active = [i for i, x in enumerate(expected) if x.dvalue > 1e-14]
    if active:
        np.testing.assert_allclose(pe.covariance([actual[i] for i in active]),
                                   pe.covariance([expected[i] for i in active]),
                                   rtol=2e-8, atol=2e-13)



@pytest.mark.parametrize('generalized', [False, True], ids=['EVP', 'GEVP'])
@pytest.mark.parametrize('complex_input', [False, True], ids=['real', 'complex'])
@pytest.mark.parametrize('source', ['MC', 'covariance'])
@pytest.mark.filterwarnings('ignore:Covariance matrix is not positive semi-definite:RuntimeWarning')
def test_prescribed_uncertain_eigensystem(generalized, complex_input, source):
    """Construct A=D U Lambda U^H D, B=D^2 from known uncertain inputs.

    The exact eigenvalues are Lambda and the B-normalized vector columns are
    V=D^-1 U. U is an elementary rotation with an optional complex phase.
    Neither the expected eigensystem nor its errors uses a numerical eigensolver
    or the new eigenvector Jacobian. The errors come from elementary Obs algebra.
    """
    rng = np.random.default_rng(1021)
    means = np.array([0.7, 2.2, 0.41, 0.63, 1.1, 1.4])
    factor = rng.normal(size=(6, 6))
    covariance = factor @ factor.T * 1e-6
    if source == 'covariance':
        parameters = pe.cov_Obs(means, covariance, 'prescribed')
    else:
        noise = rng.multivariate_normal(np.zeros(6), covariance * 120, size=120)
        noise -= noise.mean(axis=0)
        parameters = [pe.Obs([mean + noise[:, k]], ['prescribed|r1'], idl=[range(5, 245, 2)])
                      for k, mean in enumerate(means)]
    eigenvalues = parameters[:2]
    theta, phi = parameters[2:4]
    cosine, sine = np.cos(theta), np.sin(theta)
    if complex_input:
        phase = pe.CObs(np.cos(phi), np.sin(phi))
        unitary = np.array([[pe.CObs(cosine), -sine * phase.conjugate()],
                            [sine * phase, pe.CObs(cosine)]])
        unitary_h = unitary.conjugate().T
    else:
        unitary = np.array([[cosine, -sine], [sine, cosine]])
        unitary_h = unitary.T
    scales = parameters[4:6] if generalized else [1., 1.]
    diagonal = np.diag(scales)
    metric = diagonal @ diagonal
    matrix = diagonal @ unitary @ np.diag(eigenvalues) @ unitary_h @ diagonal
    prescribed_vectors = unitary / np.asarray(scales)[:, None]
    reference = _values(prescribed_vectors)
    metric_mean = _values(metric)

    if generalized:
        corr = pe.Corr([metric, matrix])
        rows = corr.GEVP(t0=0, ts=1, sort=None, vector_obs=True)
        recovered_vectors = np.asarray(rows)[::-1].T
        recovered_values = [corr.Eigenvalue(t0=0, ts=1, sort=None, state=state,
                                            vector_obs=True)[1] for state in (1, 0)]
        # Also check the numerical routes against the prescribed vectors. For
        # frozen coefficients, compare errors to that fixed projection, rather
        # than to the fluctuating generalized eigenvalue.
        for method in ('eigh', 'cholesky'):
            numeric = corr.GEVP(t0=0, ts=1, sort=None, method=method)[::-1].T
            for k in range(2):
                np.testing.assert_allclose(np.outer(numeric[:, k], numeric[:, k].conj()),
                                           np.outer(reference[:, k], reference[:, k].conj()), atol=2e-12)
                expected_fixed = reference[:, k].conj() @ matrix @ reference[:, k]
                if isinstance(expected_fixed, pe.CObs):
                    expected_fixed = expected_fixed.real
                actual_fixed = corr.Eigenvalue(t0=0, ts=1, sort=None, state=1 - k, method=method)[1]
                _assert_same_observables([actual_fixed], [expected_fixed])
    else:
        recovered_values, recovered_vectors = pe.linalg.eigh(matrix)

    # Compare in the same fixed-reference phase convention. This removes only
    # eigenvector phase freedom, not normalization or component fluctuations.
    actual_columns, expected_columns = [], []
    for k in range(2):
        actual_columns.append(_align_observable_vector(recovered_vectors[:, k], reference[:, k], metric_mean))
        expected_columns.append(_align_observable_vector(prescribed_vectors[:, k], reference[:, k], metric_mean))
    actual_columns = np.array(actual_columns).T
    expected_columns = np.array(expected_columns).T
    actual = list(recovered_values) + [x.real for x in actual_columns.flat] + [x.imag for x in actual_columns.flat]
    expected = list(eigenvalues) + [x.real for x in expected_columns.flat] + [x.imag for x in expected_columns.flat]
    _assert_same_observables(actual, expected)

    # Independently reconstruct the original uncertain matrix. This checks that
    # the recovered values and vectors together retain the input correlations.
    reconstruction = metric @ actual_columns @ np.diag(recovered_values) @ actual_columns.conjugate().T @ metric
    original_parts, reconstructed_parts = [], []
    for original, recovered in zip(matrix.flat, reconstruction.flat, strict=True):
        original = original if isinstance(original, pe.CObs) else pe.CObs(original)
        original_parts.extend([original.real, original.imag])
        reconstructed_parts.extend([recovered.real, recovered.imag])
    # Imaginary components of purely real input can be numeric constants.
    for i, entry in enumerate(original_parts):
        if not isinstance(entry, pe.Obs):
            original_parts[i] = 0 * eigenvalues[0] + entry
    _assert_same_observables(reconstructed_parts, original_parts)
