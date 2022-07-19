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
        corr.deriv(variant="forward")
        corr.deriv(variant="symmetric")
        corr.deriv(variant="improved")
        corr.deriv().deriv()
        corr.second_deriv(variant="symmetric")
        corr.second_deriv(variant="improved")
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


def test_m_eff():
    for padding in [0, 4]:
        my_corr = pe.correlators.Corr([pe.pseudo_Obs(10, 0.1, 't'), pe.pseudo_Obs(9, 0.05, 't'), pe.pseudo_Obs(9, 0.1, 't'), pe.pseudo_Obs(10, 0.05, 't')], padding=[padding, padding])
        my_corr.m_eff('log')
        my_corr.m_eff('cosh')
        my_corr.m_eff('arccosh')

    with pytest.warns(RuntimeWarning):
        my_corr.m_eff('sinh')

    with pytest.raises(Exception):
        my_corr.m_eff('unkown_variant')


def test_m_eff_negative_values():
    for padding in [0, 4]:
        my_corr = pe.correlators.Corr([1.0 * pe.pseudo_Obs(10, 0.1, 't'), 1.0 * pe.pseudo_Obs(9, 0.05, 't'), -pe.pseudo_Obs(9, 0.1, 't')], padding=[padding, padding])
        m_eff_log = my_corr.m_eff('log')
        assert m_eff_log[padding + 1] is None
        m_eff_cosh = my_corr.m_eff('cosh')
        assert m_eff_cosh[padding + 1] is None


def test_reweighting():
    my_corr = pe.correlators.Corr([pe.pseudo_Obs(10, 0.1, 't'), pe.pseudo_Obs(0, 0.05, 't')])
    assert my_corr.reweighted is False
    r_my_corr = my_corr.reweight(pe.pseudo_Obs(1, 0.1, 't'))
    assert r_my_corr.reweighted is True


def test_correlate():
    my_corr = pe.correlators.Corr([pe.pseudo_Obs(10, 0.1, 't'), pe.pseudo_Obs(0, 0.05, 't')])
    corr1 = my_corr.correlate(my_corr)
    corr2 = my_corr.correlate(my_corr[0])
    with pytest.raises(Exception):
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

    with pytest.raises(Exception):
        my_corr.fit(f, "from 0 to 3")
    with pytest.raises(Exception):
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
    obs_b= pe.Obs([np.random.normal(0.1, 0.1, 99)], ['test'])
    with pytest.raises(Exception):
        pe.Corr([obs_a, obs_b])

    obs_a = pe.Obs([np.random.normal(0.1, 0.1, 100)], ['test'])
    obs_b= pe.Obs([np.random.normal(0.1, 0.1, 100)], ['test'], idl=[range(1, 200, 2)])
    with pytest.raises(Exception):
        pe.Corr([obs_a, obs_b])

    obs_a = pe.Obs([np.random.normal(0.1, 0.1, 100)], ['test'])
    obs_b= pe.Obs([np.random.normal(0.1, 0.1, 100)], ['test2'])
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
    with pytest.raises(Exception):
        corr.set_prange([2])
    with pytest.raises(Exception):
        corr.set_prange([2, 2.3])
    with pytest.raises(Exception):
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
    corr += pe.pseudo_Obs(0.0, 0.1, 'another_ensemble')
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
