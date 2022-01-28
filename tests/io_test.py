import os
import gzip
import numpy as np
import pyerrors as pe
import pyerrors.input.json as jsonio


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

    tt1 = pe.Obs([np.random.rand(100)], ['t|r1'], idl=[range(2, 202, 2)])
    tt2 = pe.Obs([np.random.rand(100)], ['t|r2'], idl=[range(2, 202, 2)])
    tt3 = pe.Obs([np.random.rand(102)], ['qe'])

    tt = tt1 + tt2 + tt3

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
