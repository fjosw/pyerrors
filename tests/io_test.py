import pyerrors.obs as pe
import pyerrors.input.json as jsonio
import numpy as np
import os


def test_jsonio():
    o = pe.pseudo_Obs(1.0, .2, 'one')
    o2 = pe.pseudo_Obs(0.5, .1, 'two|r1')
    o3 = pe.pseudo_Obs(0.5, .1, 'two|r2')
    o4 = pe.merge_obs([o2, o3])
    do = o - .2 * o4

    o5 = pe.pseudo_Obs(0.8, .1, 'two|r2')
    testl = [o3, o5]

    mat = np.array([[pe.pseudo_Obs(1.0, .1, 'mat'), pe.pseudo_Obs(0.3, .1, 'mat')], [pe.pseudo_Obs(0.2, .1, 'mat'), pe.pseudo_Obs(2.0, .4, 'mat')]])

    ol = [do, testl, mat]
    fname = 'test_rw'

    jsonio.dump_to_json(ol, fname, indent=1)

    rl = jsonio.load_json(fname)

    os.remove(fname + '.json.gz')

    for i in range(len(rl)):
        if isinstance(ol[i], pe.Obs):
            o = ol[i] - rl[i]
            assert(o.is_zero())
        or1 = np.ravel(ol[i])
        or2 = np.ravel(rl[i])
        for j in range(len(or1)):
            o = or1[i] - or2[i]
            assert(o.is_zero())
