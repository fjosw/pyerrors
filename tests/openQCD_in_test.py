import os
import numpy as np
import pyerrors as pe
import pytest


def test_openqcd():
    path = './tests//data/openqcd_test/'
    prefix = 'sfqcd'
    postfix = '.rwms'

    # sfqcd-1.6: Trajectories instead of confignumbers are printed to file.
    rwfo = pe.input.openQCD.read_rwms(path, prefix, version='1.6', postfix=postfix)
    repname = list(rwfo[0].idl.keys())[0]
    assert(rwfo[0].idl[repname] == range(1, 13))
    rwfo = pe.input.openQCD.read_rwms(path, prefix, version='1.6', postfix=postfix, r_start=[1], r_stop=[12])
    assert(rwfo[0].idl[repname] == range(1, 13))
    rwfo = pe.input.openQCD.read_rwms(path, prefix, version='1.6', postfix=postfix, r_start=[3], r_stop=[8])
    assert(rwfo[0].idl[repname] == range(3, 9))
    rwfo = pe.input.openQCD.read_rwms(path, prefix, version='1.6', postfix=postfix, r_start=[2], r_stop=[6])
    assert(rwfo[0].idl[repname] == range(2, 7))
    rwfs = pe.input.openQCD.read_rwms(path, prefix, version='1.6', postfix=postfix, r_start=[1], r_stop=[12], r_step=2)
    assert(rwfs[0].idl[repname] == range(1, 12, 2))
    rwfs = pe.input.openQCD.read_rwms(path, prefix, version='1.6', postfix=postfix, r_start=[2], r_stop=[12], r_step=2)
    assert(rwfs[0].idl[repname] == range(2, 13, 2))
    rwfo = pe.input.openQCD.read_rwms(path, prefix, version='1.6', postfix=postfix)
    assert((rwfo[0].r_values[repname] + rwfo[0].deltas[repname][1]) == (rwfs[0].r_values[repname] + rwfs[0].deltas[repname][0]))

    o = pe.pseudo_Obs(1., .01, repname, samples=12)
    pe.reweight(rwfo[0], [o])

    o = pe.pseudo_Obs(1., .01, repname, samples=6)
    pe.reweight(rwfo[0], [o])
    o.idl[repname] = range(2, 13, 2)
    pe.reweight(rwfo[0], [o])
    pe.reweight(rwfs[0], [o])

    files = ['openqcd2r1.ms1.dat']
    names = ['openqcd2|r1']

    # TM with 2 Hasenbusch factors and 2 sources each + RHMC with one source, openQCD 2.0
    rwfo = pe.input.openQCD.read_rwms(path, prefix, version='2.0', files=files, names=names)
    assert(len(rwfo) == 2)
    assert(rwfo[0].value == 0.9999974970236312)
    assert(rwfo[1].value == 1.184681251089919)
    repname = list(rwfo[0].idl.keys())[0]
    assert(rwfo[0].idl[repname] == range(1, 10))
    rwfo = pe.input.openQCD.read_rwms(path, prefix, version='2.0', files=files, names=names, r_start=[1], r_stop=[8])
    assert(rwfo[0].idl[repname] == range(1, 9))
