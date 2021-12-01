import pyerrors.obs as pe
import pyerrors.input.json as jsonio
import numpy as np
import os


def test_jsonio():
    o = pe.pseudo_Obs(1.0, .2, 'one')
    o2 = pe.pseudo_Obs(0.5, .1, 'two|r1')
    o3 = pe.pseudo_Obs(0.5, .1, 'two|r2')
    o4 = pe.merge_obs([o2, o3])
    otag = 'This has been merged!'
    o4.tag = otag
    do = o - .2 * o4

    o5 = pe.pseudo_Obs(0.8, .1, 'two|r2')
    testl = [o3, o5]

    arr = np.array([o3, o5])
    mat = np.array([[pe.pseudo_Obs(1.0, .1, 'mat'), pe.pseudo_Obs(0.3, .1, 'mat')], [pe.pseudo_Obs(0.2, .1, 'mat'), pe.pseudo_Obs(2.0, .4, 'mat')]])

    tt1 = pe.Obs([np.random.rand(100)], ['t|r1'], idl=[range(2, 202, 2)])
    tt2 = pe.Obs([np.random.rand(100)], ['t|r2'], idl=[range(2, 202, 2)])
    tt3 = pe.Obs([np.random.rand(102)], ['qe'])

    tt = tt1 + tt2 + tt3

    tt.tag = 'Test Obs'

    ol = [o4, do, testl, mat, arr, np.array([o]), np.array([tt, tt]), [tt, tt]]
    fname = 'test_rw'

    jsonio.dump_to_json(ol, fname, indent=1)

    rl = jsonio.load_json(fname)

    os.remove(fname + '.json.gz')

    for i in range(len(rl)):
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
    jsonio.dump_to_json(ol, fname, indent=1, gz=False, description=description)

    rl = jsonio.load_json(fname, gz=False, full_output=True)

    assert(description == rl['description'])

    os.remove(fname + '.json')
