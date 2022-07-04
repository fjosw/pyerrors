import os
import gzip
import rapidjson
import numpy as np
import pyerrors as pe
import pyerrors.input.json as jsonio
import pyerrors.input.dobs as dobsio
import pytest


def test_jsonio():
    o = pe.pseudo_Obs(1.0, .2, 'one')
    o2 = pe.pseudo_Obs(0.5, .1, 'two|r1')
    o3 = pe.pseudo_Obs(0.5, .1, 'two|r2')
    o4 = pe.merge_obs([o2, o3])
    otag = 'This has been merged!'
    o4.tag = otag
    do = o - .2 * o4
    co1 = pe.cov_Obs(1., .123, 'cov1')
    co3 = pe.cov_Obs(4., .1 ** 2, 'cov3')
    do *= co1 / co3
    do.tag = {'A': 2}

    o5 = pe.pseudo_Obs(0.8, .1, 'two|r2')
    co2 = pe.cov_Obs([1, 2], [[.12, .004], [.004, .02]], 'cov2')
    o5 /= co2[0]
    o3 /= co2[1]
    o5.tag = 2 * otag
    testl = [o3, o5]

    arr = np.array([o3, o5])
    mat = np.array([[pe.pseudo_Obs(1.0, .1, 'mat'), pe.pseudo_Obs(0.3, .1, 'mat')], [pe.pseudo_Obs(0.2, .1, 'mat'), pe.pseudo_Obs(2.0, .4, 'mat')]])
    mat[0][1].tag = ['This', 'is', 2, None]
    mat[1][0].tag = '{testt}'
    mat[1][1].tag = '[tag]'

    tt1 = pe.Obs([np.random.rand(100), np.random.rand(100)], ['t|r1', 't|r2'], idl=[range(2, 202, 2), range(22, 222, 2)])
    tt3 = pe.Obs([np.random.rand(102)], ['qe'])

    tt = tt1 + tt3

    tt.tag = 'Test Obs: Ä'

    ol = [o4, do, testl, mat, arr, np.array([o]), np.array([tt, tt]), [tt, tt], co1, co2, np.array(co2), co1 / co2[0]]
    fname = 'test_rw'

    jsonio.dump_to_json(ol, fname, indent=1, description='[I am a tricky description]')

    rl = jsonio.load_json(fname)

    os.remove(fname + '.json.gz')

    for o, r in zip(ol, rl):
        assert np.all(o == r)

    for i in range(len(ol)):
        if isinstance(ol[i], pe.Obs):
            o = ol[i] - rl[i]
            assert(o.is_zero())
            assert(ol[i].tag == rl[i].tag)
        or1 = np.ravel(ol[i])
        or2 = np.ravel(rl[i])
        for j in range(len(or1)):
            o = or1[j] - or2[j]
            assert(o.is_zero())
        if isinstance(ol[i], pe.Obs):
            for name in ol[i].r_values:
                assert(np.isclose(ol[i].r_values[name], rl[i].r_values[name]))
        elif isinstance(ol[i], list):
            for j in range(len(ol[i])):
                for name in ol[i][j].r_values:
                    assert(np.isclose(ol[i][j].r_values[name], rl[i][j].r_values[name]))

    description = {'I': {'Am': {'a': 'nested dictionary!'}}}
    jsonio.dump_to_json(ol, fname, indent=0, gz=False, description=description)

    rl = jsonio.load_json(fname, gz=False, full_output=True)

    os.remove(fname + '.json')

    for o, r in zip(ol, rl['obsdata']):
        assert np.all(o == r)

    assert(description == rl['description'])


def test_json_string_reconstruction():
    my_obs = pe.Obs([np.random.rand(100)], ['name'])

    json_string = pe.input.json.create_json_string(my_obs)
    reconstructed_obs1 = pe.input.json.import_json_string(json_string)
    assert my_obs == reconstructed_obs1

    compressed_string = gzip.compress(json_string.encode('utf-8'))

    reconstructed_string = gzip.decompress(compressed_string).decode('utf-8')
    reconstructed_obs2 = pe.input.json.import_json_string(reconstructed_string)

    assert reconstructed_string == json_string
    assert my_obs == reconstructed_obs2


def test_json_corr_io():
    my_list = [pe.Obs([np.random.normal(1.0, 0.1, 100)], ['ens1']) for o in range(8)]
    rw_list = pe.reweight(pe.Obs([np.random.normal(1.0, 0.1, 100)], ['ens1']), my_list)

    for obs_list in [my_list, rw_list]:
        for tag in [None, "test"]:
            obs_list[3].tag = tag
            for pad in [0, 2]:
                for corr_tag in [None, 'my_Corr_tag']:
                    for prange in [None, [3, 6]]:
                        for gap in [False, True]:
                            my_corr = pe.Corr(obs_list, padding=[pad, pad], prange=prange)
                            my_corr.tag = corr_tag
                            if gap:
                                my_corr.content[4] = None
                            pe.input.json.dump_to_json(my_corr, 'corr')
                            recover = pe.input.json.load_json('corr')
                            os.remove('corr.json.gz')
                            assert np.all([o.is_zero() for o in [x for x in (my_corr - recover) if x is not None]])
                            for index, entry in enumerate(my_corr):
                                if entry is None:
                                    assert recover[index] is None
                            assert my_corr.tag == recover.tag
                            assert my_corr.prange == recover.prange
                            assert my_corr.reweighted == recover.reweighted


def test_json_corr_2d_io():
    obs_list = [np.array([[pe.pseudo_Obs(1.0 + i, 0.1 * i, 'test'), pe.pseudo_Obs(0.0, 0.1 * i, 'test')], [pe.pseudo_Obs(0.0, 0.1 * i, 'test'), pe.pseudo_Obs(1.0 + i, 0.1 * i, 'test')]]) for i in range(4)]

    for tag in [None, "test"]:
        obs_list[3][0, 1].tag = tag
        for padding in [0, 1]:
            for prange in [None, [3, 6]]:
                my_corr = pe.Corr(obs_list, padding=[padding, padding], prange=prange)
                my_corr.tag = tag
                pe.input.json.dump_to_json(my_corr, 'corr')
                recover = pe.input.json.load_json('corr')
                os.remove('corr.json.gz')
                assert np.all([np.all([o.is_zero() for o in q]) for q in [x.ravel() for x in (my_corr - recover) if x is not None]])
                for index, entry in enumerate(my_corr):
                    if entry is None:
                        assert recover[index] is None
                assert my_corr.tag == recover.tag
                assert my_corr.prange == recover.prange


def test_json_dict_io():
    def check_dict_equality(d1, d2):
        def dict_check_obs(d1, d2):
            for k, v in d1.items():
                if isinstance(v, dict):
                    v = dict_check_obs(v, d2[k])
                elif isinstance(v, list) and all([isinstance(o, pe.Obs) for o in v]):
                    for i in range(len(v)):
                        assert((v[i] - d2[k][i]).is_zero())
                elif isinstance(v, list):
                    v = list_check_obs(v, d2[k])
                elif isinstance(v, pe.Obs):
                    assert((v - d2[k]).is_zero())
                elif isinstance(v, pe.Corr):
                    for i in range(v.T):
                        assert((v[i] - d2[k][i]).is_zero())
                elif isinstance(v, np.ndarray):
                    a1 = np.ravel(v)
                    a2 = np.ravel(d2[k])
                    for i in range(len(a1)):
                        assert((a1[i] - a2[i]).is_zero())

        def list_check_obs(l1, l2):
            for ei in range(len(l1)):
                e = l1[ei]
                if isinstance(e, list):
                    e = list_check_obs(e, l2[ei])
                elif isinstance(e, list) and all([isinstance(o, pe.Obs) for o in e]):
                    for i in range(len(e)):
                        assert((e[i] - l2[ei][i]).is_zero())
                elif isinstance(e, dict):
                    e = dict_check_obs(e, l2[ei])
                elif isinstance(e, pe.Obs):
                    assert((e - l2[ei]).is_zero())
                elif isinstance(e, pe.Corr):
                    for i in range(e.T):
                        assert((e[i] - l2[ei][i]).is_zero())
                elif isinstance(e, np.ndarray):
                    a1 = np.ravel(e)
                    a2 = np.ravel(l2[ei])
                    for i in range(len(a1)):
                        assert((a1[i] - a2[i]).is_zero())
        dict_check_obs(d1, d2)
        return True

    od = {
        'l':
        {
            'a': pe.pseudo_Obs(1, .2, 'testa', samples=10),
            'b': [pe.pseudo_Obs(1.1, .1, 'test', samples=10), pe.pseudo_Obs(1.2, .1, 'test', samples=10), pe.pseudo_Obs(1.3, .1, 'test', samples=10)],
            'c': {
                'd': 1,
                'e': pe.pseudo_Obs(.2, .01, 'teste', samples=10),
                'f': pe.Corr([pe.pseudo_Obs(.1, .01, 'a', samples=10), pe.pseudo_Obs(.1, .01, 'a', samples=10), pe.pseudo_Obs(.1, .01, 'a', samples=10), pe.pseudo_Obs(.1, .01, 'a', samples=10)]),
                'g': np.reshape(np.asarray([pe.pseudo_Obs(.1, .01, 'a', samples=10), pe.pseudo_Obs(.1, .01, 'a', samples=10), pe.pseudo_Obs(.1, .01, 'a', samples=10), pe.pseudo_Obs(.1, .01, 'a', samples=10)]), (2, 2)),
            }
        },
        's':
        {
            'a': 'Infor123',
            'b': ['Some', 'list'],
            'd': pe.pseudo_Obs(.01, .001, 'testd', samples=10) * pe.cov_Obs(1, .01, 'cov1'),
            'se': None,
            'sf': 1.2,
        }
    }

    fname = 'test_rw'

    desc = 'This is a random description'

    with pytest.raises(Exception):
        jsonio.dump_dict_to_json(od, fname, description=desc, reps='|Test')

    jsonio.dump_dict_to_json(od, fname, description=desc, reps='TEST')
    nd = jsonio.load_json_dict(fname, full_output=True, reps='TEST')

    with pytest.raises(Exception):
        nd = jsonio.load_json_dict(fname, full_output=True)

    jsonio.dump_dict_to_json(od, fname, description=desc)
    nd = jsonio.load_json_dict(fname, full_output=True)
    assert (desc == nd['description'])

    assert(check_dict_equality(od, nd['obsdata']))
    nd = jsonio.load_json_dict(fname, full_output=False)
    assert(check_dict_equality(od, nd))

    nl = jsonio.load_json(fname, full_output=True)
    nl = jsonio.load_json(fname, full_output=False)

    with pytest.raises(Exception):
        jsonio.dump_dict_to_json(nl, fname, description=desc)

    od['k'] = 'DICTOBS2'
    with pytest.raises(Exception):
        jsonio.dump_dict_to_json(od, fname, description=desc)

    od['k'] = ['DICTOBS2']
    with pytest.raises(Exception):
        jsonio.dump_dict_to_json(od, fname, description=desc)

    os.remove(fname + '.json.gz')


def test_renorm_deriv_of_corr(tmp_path):
    c = pe.Corr([pe.pseudo_Obs(i, .1, 'test') for i in range(10)])
    c *= pe.cov_Obs(1., .1, '#ren')
    c = c.deriv()
    pe.input.json.dump_to_json(c, (tmp_path / 'test').as_posix())
    recover = pe.input.json.load_json((tmp_path / 'test').as_posix())
    assert np.all([o == 0 for o in (c - recover)[1:-1]])


def test_dobsio():
    o = pe.pseudo_Obs(1.0, .2, 'one|r1')
    o1 = pe.pseudo_Obs(1.5, .2, 'one|r1')

    ol = [o, o1]
    fname = 'test_rw'
    dobsio.write_pobs(ol, fname, 'Testobs')
    rl = dobsio.read_pobs(fname)

    for i in range(len(ol)):
        assert (ol[i] - rl[i].is_zero())

    od = {
        'obsdata': ol,
        'name': 'testn',
        'spec': 'tests',
        'origin': 'testo',
        'symbol': ['A', 'B']
    }
    dobsio.write_pobs(ol, fname, od['name'], od['spec'], od['origin'], od['symbol'])
    rd = dobsio.read_pobs(fname, full_output=True)

    for i in range(len(od['obsdata'])):
        assert (od['obsdata'][i] - rd['obsdata'][i].is_zero())

    assert(od['spec'] == rd['description']['spec'])
    assert(od['origin'] == rd['description']['origin'])
    assert(od['name'] == rd['description']['name'])
    assert(rd['description']['enstag'] == ol[0].e_names[0])

    dobsio.write_dobs(ol, fname, 'Testobs')
    rl = dobsio.read_dobs(fname)

    for i in range(len(ol)):
        assert (ol[i] - rl[i].is_zero())

    dobsio.write_dobs(ol, fname, od['name'], od['spec'], od['origin'], od['symbol'])
    rd = dobsio.read_dobs(fname, full_output=True)
    os.remove(fname + '.xml.gz')

    for i in range(len(od['obsdata'])):
        assert (od['obsdata'][i] - rd['obsdata'][i].is_zero())

    assert(od['spec'] == rd['description']['spec'])
    assert(od['origin'] == rd['description']['origin'])
    assert(od['name'] == rd['description']['name'])

    o2 = pe.pseudo_Obs(0.5, .1, 'two|r1')
    o3 = pe.pseudo_Obs(0.5, .1, 'two|r2')
    o4 = pe.merge_obs([o2, o3])
    otag = 'This has been merged!'
    o4.tag = otag
    do = o - .2 * o4
    co1 = pe.cov_Obs(1., .123, 'cov1')
    co3 = pe.cov_Obs(4., .1 ** 2, 'cov3')
    do *= co1 / co3
    do.tag = {'A': 2}

    o5 = pe.pseudo_Obs(0.8, .1, 'two|r2')
    co2 = pe.cov_Obs([1, 2], [[.12, .004], [.004, .02]], 'cov2')
    o5 /= co2[0]
    o5.tag = 2 * otag

    tt1 = pe.Obs([np.random.rand(100), np.random.rand(100)], ['t|r1', 't|r2'], idl=[range(2, 202, 2), range(22, 222, 2)])
    tt3 = pe.Obs([np.random.rand(102)], ['qe|r1'])

    tt = tt1 + tt3

    tt.tag = 'Test Obs: Ä'

    tt4 = pe.Obs([np.random.rand(100), np.random.rand(100)], ['t|r1', 't|r2'], idl=[range(1, 101, 1), range(2, 202, 2)])

    ol = [o2, o3, o4, do, o5, tt, tt4, np.log(tt4 / o5**2), np.exp(o5 + np.log(co3 / tt3 + o4) / tt)]
    print(ol)
    fname = 'test_rw'

    dobsio.write_dobs(ol, fname, 'TEST')

    rl = dobsio.read_dobs(fname, noempty=True)
    os.remove(fname + '.xml.gz')
    [o.gamma_method() for o in rl]

    for i in range(len(ol)):
        if isinstance(ol[i], pe.Obs):
            o = ol[i] - rl[i]
            assert(o.is_zero())
        or1 = np.ravel(ol[i])
        or2 = np.ravel(rl[i])
        for j in range(len(or1)):
            o = or1[j] - or2[j]
            assert(o.is_zero())
        if isinstance(ol[i], pe.Obs):
            for name in ol[i].r_values:
                assert(np.isclose(ol[i].r_values[name], rl[i].r_values[name]))


def test_reconstruct_non_linear_r_obs(tmp_path):
    to = pe.Obs([np.random.rand(500), np.random.rand(500), np.random.rand(111)],
                ["e|r1", "e|r2", "my_new_ensemble_54^£$|8'[@124435%6^7&()~#"],
                idl=[range(1, 501), range(0, 500), range(1, 999, 9)])
    to = np.log(to ** 2) / to
    to.dump((tmp_path / "test_equality").as_posix())
    ro = pe.input.json.load_json((tmp_path / "test_equality").as_posix())
    assert assert_equal_Obs(to, ro)


def test_reconstruct_non_linear_r_obs_list(tmp_path):
    to = pe.Obs([np.random.rand(500), np.random.rand(500), np.random.rand(111)],
                ["e|r1", "e|r2", "my_new_ensemble_54^£$|8'[@124435%6^7&()~#"],
                idl=[range(1, 501), range(0, 500), range(1, 999, 9)])
    to = np.log(to ** 2) / to
    for to_list in [[to, to, to], np.array([to, to, to])]:
        pe.input.json.dump_to_json(to_list, (tmp_path / "test_equality_list").as_posix())
        ro_list = pe.input.json.load_json((tmp_path / "test_equality_list").as_posix())
        for oa, ob in zip(to_list, ro_list):
            assert assert_equal_Obs(oa, ob)


def test_import_non_json_string():
    with pytest.raises(rapidjson.JSONDecodeError):
        pe.input.json.import_json_string("this is garbage")


def assert_equal_Obs(to, ro):
    for kw in ["N", "cov_names", "covobs", "ddvalue", "dvalue", "e_content",
               "e_names", "idl", "mc_names", "names",
               "reweighted", "shape", "tag"]:
        if not getattr(to, kw) == getattr(ro, kw):
            print(kw, "does not match.")
            return False

    for kw in ["value"]:
        if not np.isclose(getattr(to, kw), getattr(ro, kw), atol=1e-14):
            print(kw, "does not match.")
            return False


    for kw in ["r_values", "deltas"]:
        for (k, v), (k2, v2) in zip(getattr(to, kw).items(), getattr(ro, kw).items()):
            assert k == k2
            if not np.allclose(v, v2, atol=1e-14):
                print(kw, "does not match.")
                return False

    m_to = getattr(to, "is_merged")
    m_ro = getattr(ro, "is_merged")
    if not m_to == m_ro:
        if not (all(value is False for value in m_ro.values()) and all(value is False for value in m_to.values())):
            print("is_merged", "does not match.")
            return False
    return True
