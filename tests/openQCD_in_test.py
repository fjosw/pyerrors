import os
import numpy as np
import pyerrors as pe
import pytest


def test_rwms():
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
    rwfo = pe.input.openQCD.read_rwms(path, prefix, version='2.0', files=files, names=names, r_start=[1], r_stop=[8], print_err=True)
    assert(rwfo[0].idl[repname] == range(1, 9))

    # t0
    prefix = 'openqcd'

    t0 = pe.input.openQCD.extract_t0(path, prefix, dtr_read=3, xmin=0, spatial_extent=4)
    files = ['openqcd2r1.ms.dat']
    names = ['openqcd2|r1']
    t0 = pe.input.openQCD.extract_t0(path, '', dtr_read=3, xmin=0, spatial_extent=4, files=files, names=names, fit_range=2)
    t0 = pe.input.openQCD.extract_t0(path, prefix, dtr_read=3, xmin=0, spatial_extent=4, r_start=[1])
    repname = list(rwfo[0].idl.keys())[0]
    assert(t0.idl[repname] == range(1, 10))
    t0 = pe.input.openQCD.extract_t0(path, prefix, dtr_read=3, xmin=0, spatial_extent=4, r_start=[2], r_stop=[8])
    repname = list(rwfo[0].idl.keys())[0]
    assert(t0.idl[repname] == range(2, 9))
    t0 = pe.input.openQCD.extract_t0(path, prefix, dtr_read=3, xmin=0, spatial_extent=4, fit_range=2, plaquette=True, assume_thermalization=True)

    pe.input.openQCD.extract_t0(path, '', dtr_read=3, xmin=0, spatial_extent=4, files=files, names=names, fit_range=2, plot_fit=True)


def test_Qtop():
    path = './tests//data/openqcd_test/'
    prefix = 'sfqcd'

    qtop = pe.input.openQCD.read_qtop(path, prefix, c=0.3, version='sfqcd')
    repname = list(qtop.idl.keys())[0]
    qtop = pe.input.openQCD.read_qtop(path, prefix, c=0.3, version='sfqcd', Zeuthen_flow=True, L=4)
    qi = pe.input.openQCD.read_qtop(path, prefix, c=0.3, version='sfqcd', integer_charge=True)
    for conf in range(len(qi.idl[repname])):
        assert(0 == qi.r_values[repname] + qi.deltas[repname][conf])

    qtop = pe.input.openQCD.read_qtop(path, prefix, c=0.0, version='sfqcd')
    assert (np.isclose(-4.572999e-02, qtop.r_values[repname] + qtop.deltas[repname][0]))
    qtop = pe.input.openQCD.read_qtop(path, prefix, c=0.28, version='sfqcd')
    assert (np.isclose(3.786893e-02, qtop.r_values[repname] + qtop.deltas[repname][0]))
    qtop = pe.input.openQCD.read_qtop(path, prefix, c=0.08, version='sfqcd', Zeuthen_flow=True)
    assert (np.isclose(3.653140e-02, qtop.r_values[repname] + qtop.deltas[repname][1]))
    qtop = pe.input.openQCD.read_qtop(path, prefix, c=0.40, version='sfqcd', Zeuthen_flow=True)
    assert (np.isclose(2.745865e-01, qtop.r_values[repname] + qtop.deltas[repname][1]))

    qtop = pe.input.openQCD.read_qtop(path, prefix, c=0.40, version='sfqcd', Zeuthen_flow=True, r_start=[2])
    assert(qtop.idl[repname] == range(2, 7))
    assert (np.isclose(2.745865e-01, qtop.r_values[repname] + qtop.deltas[repname][0]))

    qtop = pe.input.openQCD.read_qtop(path, prefix, c=0.40, version='sfqcd', Zeuthen_flow=True, r_stop=[5])
    assert(qtop.idl[repname] == range(1, 6))

    names = ['sfqcd|r1']
    files = ['sfqcdr1.gfms.dat']
    qs = pe.input.openQCD.read_qtop_sector(path, '', 0.3, target=0, Zeuthen_flow=True, version='sfqcd')

    assert((pe.input.openQCD.qtop_projection(qi, target=0) - qs).is_zero())


def test_gf_coupling():
    path = './tests//data/openqcd_test/'
    prefix = 'sfqcd'
    gf = pe.input.openQCD.read_gf_coupling(path, prefix, c=0.3)
    with pytest.raises(Exception):
        pe.input.openQCD.read_gf_coupling(path, prefix, c=0.35)
    with pytest.raises(Exception):
        pe.input.openQCD.read_gf_coupling(path, prefix, c=0.3, Zeuthen_flow=False)
