#!/usr/bin/env python
# coding: utf-8

import os
import fnmatch
import re
import struct
import numpy as np  # Thinly-wrapped numpy
from ..pyerrors import Obs
from ..fits import fit_lin


def read_rwms(path, prefix, version='2.0', names=None, **kwargs):
    """Read rwms format from given folder structure. Returns a list of length nrw

    Attributes
    -----------------
    version -- version of openQCD, default 2.0

    Keyword arguments
    -----------------
    r_start -- list which contains the first config to be read for each replicum
    r_stop -- list which contains the last config to be read for each replicum
    postfix -- postfix of the file to read, e.g. '.ms1' for openQCD-files
    """
    #oqcd versions implemented in this method
    known_oqcd_versions = ['1.4','1.6','2.0']
    if not (version in known_oqcd_versions):
        raise Exception('Unknown openQCD version defined!')
    print("Working with openQCD version " + version)
    if 'postfix' in kwargs:
        postfix = kwargs.get('postfix')
    else:
        postfix = ''
    ls = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        ls.extend(filenames)
        break

    if not ls:
        raise Exception('Error, directory not found')

    # Exclude files with different names
    for exc in ls:
        if not fnmatch.fnmatch(exc, prefix + '*'+postfix+'.dat'):
            ls = list(set(ls) - set([exc]))
    if len(ls) > 1:
        ls.sort(key=lambda x: int(re.findall(r'\d+', x[len(prefix):])[0]))
    #ls = fnames
    #print(ls)
    replica = len(ls)

    if 'r_start' in kwargs:
        r_start = kwargs.get('r_start')
        if len(r_start) != replica:
            raise Exception('r_start does not match number of replicas')
        # Adjust Configuration numbering to python index
        r_start = [o - 1 if o else None for o in r_start]
    else:
        r_start = [None] * replica

    if 'r_stop' in kwargs:
        r_stop = kwargs.get('r_stop')
        if len(r_stop) != replica:
            raise Exception('r_stop does not match number of replicas')
    else:
        r_stop = [None] * replica

    print('Read reweighting factors from', prefix[:-1], ',', replica, 'replica', end='')

    print_err = 0
    if 'print_err' in kwargs:
        print_err = 1
        print()

    deltas = []

    for rep in range(replica):
        tmp_array = []
        with open(path+ '/' + ls[rep], 'rb') as fp:

            #header
            t = fp.read(4) # number of reweighting factors
            if rep == 0:
                nrw = struct.unpack('i', t)[0]
                if version == '2.0':
                    nrw = int(nrw/2)
                for k in range(nrw):
                    deltas.append([])
            else:
                if ((nrw != struct.unpack('i', t)[0] and (not verion == '2.0')) or (nrw != struct.unpack('i', t)[0]/2 and version == '2.0')):# little weird if-clause due to the /2 operation needed.
                    raise Exception('Error: different number of reweighting factors for replicum', rep)

            for k in range(nrw):
                tmp_array.append([])

            # This block is necessary for openQCD1.6 and openQCD2.0 ms1 files
            nfct = []
            if version in ['1.6','2.0']:
                for i in range(nrw):
                    t = fp.read(4)
                    nfct.append(struct.unpack('i', t)[0])
                #print('nfct: ', nfct) # Hasenbusch factor, 1 for rat reweighting
            else:
                for i in range(nrw):
                    nfct.append(1)

            nsrc = []
            for i in range(nrw):
                t = fp.read(4)
                nsrc.append(struct.unpack('i', t)[0])
            if version is '2.0':
                if not struct.unpack('i', fp.read(4))[0] == 0:
                    print('something is wrong!')

            #body
            while 0 < 1:
                t = fp.read(4)
                if len(t) < 4:
                    break
                if print_err:
                    config_no = struct.unpack('i', t)
                for i in range(nrw):
                    if(version == '2.0'):
                        tmpd = _read_array_openQCD2(fp)
                        tmpd = _read_array_openQCD2(fp)
                        tmp_rw = tmpd['arr']
                        tmp_nfct = 1.0
                        for j in range(tmpd['n'][0]):
                            tmp_nfct *= np.mean(np.exp(-np.asarray(tmp_rw[j])))
                            if print_err:
                                print(config_no, i, j, np.mean(np.exp(-np.asarray(tmp_rw[j]))), np.std(np.exp(-np.asarray(tmp_rw[j]))))
                                print('Sources:', np.exp(-np.asarray(tmp_rw[j])))
                                print('Partial factor:', tmp_nfct)
                    elif version is '1.6' or version is '1.4':
                        tmp_nfct = 1.0
                        for j in range(nfct[i]):
                            t = fp.read(8 * nsrc[i])
                            t = fp.read(8 * nsrc[i])
                            tmp_rw = struct.unpack('d' * nsrc[i], t)
                            tmp_nfct *= np.mean(np.exp(-np.asarray(tmp_rw)))
                            if print_err:
                                print(config_no, i, j, np.mean(np.exp(-np.asarray(tmp_rw))), np.std(np.exp(-np.asarray(tmp_rw))))
                                print('Sources:', np.exp(-np.asarray(tmp_rw)))
                                print('Partial factor:', tmp_nfct)
                    tmp_array[i].append(tmp_nfct)

            for k in range(nrw):
                deltas[k].append(tmp_array[k][r_start[rep]:r_stop[rep]])

    print(',', nrw, 'reweighting factors with', nsrc, 'sources')
    result = []
    for t in range(nrw):
        if names == None:
            result.append(Obs(deltas[t], [w.split(".")[0] for w in ls]))
        else:
            print(names)
            result.append(Obs(deltas[t], names))
    return result




def extract_t0(path, prefix, dtr_read, xmin, spatial_extent, fit_range=5, **kwargs):
    """Extract t0 from given .ms.dat files. Returns t0 as Obs.

    It is assumed that all boundary effects have sufficiently decayed at x0=xmin.
    The data around the zero crossing of t^2<E> - 0.3 is fitted with a linear function
    from which the exact root is extracted.
    Only works with openQCD v 1.2.

    Parameters
    ----------
    path -- Path to .ms.dat files
    prefix -- Ensemble prefix
    dtr_read -- Determines how many trajectories should be skipped when reading the ms.dat files.
                Corresponds to dtr_cnfg / dtr_ms in the openQCD input file.
    xmin -- First timeslice where the boundary effects have sufficiently decayed.
    spatial_extent -- spatial extent of the lattice, required for normalization.
    fit_range -- Number of data points left and right of the zero crossing to be included in the linear fit. (Default: 5)

    Keyword arguments
    -----------------
    r_start -- list which contains the first config to be read for each replicum.
    r_stop -- list which contains the last config to be read for each replicum.
    plaquette -- If true extract the plaquette estimate of t0 instead.
    """

    ls = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        ls.extend(filenames)
        break

    if not ls:
        raise Exception('Error, directory not found')

    # Exclude files with different names
    for exc in ls:
        if not fnmatch.fnmatch(exc, prefix + '*.ms.dat'):
            ls = list(set(ls) - set([exc]))
    if len(ls) > 1:
        ls.sort(key=lambda x: int(re.findall(r'\d+', x[len(prefix):])[0]))
    replica = len(ls)

    if 'r_start' in kwargs:
        r_start = kwargs.get('r_start')
        if len(r_start) != replica:
            raise Exception('r_start does not match number of replicas')
        # Adjust Configuration numbering to python index
        r_start = [o - 1 if o else None for o in r_start]
    else:
        r_start = [None] * replica

    if 'r_stop' in kwargs:
        r_stop = kwargs.get('r_stop')
        if len(r_stop) != replica:
            raise Exception('r_stop does not match number of replicas')
    else:
        r_stop = [None] * replica

    print('Extract t0 from', prefix, ',', replica, 'replica')

    Ysum = []

    for rep in range(replica):

        with open(path + '/' + ls[rep], 'rb') as fp:
            # Read header
            t = fp.read(12)
            header = struct.unpack('iii', t)
            if rep == 0:
                dn = header[0]
                nn = header[1]
                tmax = header[2]
            elif dn != header[0] or nn != header[1] or tmax != header[2]:
                raise Exception('Replica parameters do not match.')

            t = fp.read(8)
            if rep == 0:
                eps = struct.unpack('d', t)[0]
                print('Step size:', eps, ', Maximal t value:', dn * (nn) * eps)
            elif eps != struct.unpack('d', t)[0]:
                raise Exception('Values for eps do not match among replica.')

            Ysl = []

            # Read body
            while 0 < 1:
                t = fp.read(4)
                if(len(t) < 4):
                    break
                nc = struct.unpack('i', t)[0]

                t = fp.read(8 * tmax * (nn + 1))
                if kwargs.get('plaquette'):
                    if nc % dtr_read == 0:
                        Ysl.append(struct.unpack('d' * tmax * (nn + 1), t))
                t = fp.read(8 * tmax * (nn + 1))
                if not kwargs.get('plaquette'):
                    if nc % dtr_read == 0:
                        Ysl.append(struct.unpack('d' * tmax * (nn + 1), t))
                t = fp.read(8 * tmax * (nn + 1))

        Ysum.append([])
        for i, item in enumerate(Ysl):
            Ysum[-1].append([np.mean(item[current + xmin:current + tmax - xmin]) for current in range(0, len(item), tmax)])

    t2E_dict = {}
    for n in range(nn + 1):
        samples = []
        for nrep, rep in enumerate(Ysum):
            samples.append([])
            for cnfg in rep:
                samples[-1].append(cnfg[n])
            samples[-1] = samples[-1][r_start[nrep]:r_stop[nrep]]
        new_obs = Obs(samples, [(w.split('.'))[0] for w in ls])
        t2E_dict[n * dn * eps] = (n * dn * eps) ** 2 * new_obs / (spatial_extent ** 3) - 0.3

    zero_crossing = np.argmax(np.array([o.value for o in t2E_dict.values()]) > 0.0)

    x = list(t2E_dict.keys())[zero_crossing - fit_range: zero_crossing + fit_range]
    y = list(t2E_dict.values())[zero_crossing - fit_range: zero_crossing + fit_range]
    [o.gamma_method() for o in y]

    fit_result = fit_lin(x, y)
    return -fit_result[0] / fit_result[1]


def _parse_array_openQCD2(d, n, size, wa, quadrupel=False):
    arr = []
    if d == 2:
        tot = 0
        for i in range(n[d-1]-1):
            if quadrupel:
                tmp = wa[tot:n[d-1]]
                tmp2 = []
                for i in range(len(tmp)):
                    if i % 2 == 0:
                        tmp2.append(tmp[i])
                arr.append(tmp2)
            else:
                arr.append(np.asarray(wa[tot:n[d-1]]))
    return arr


# mimic the read_array routine of openQCD-2.0.
# fp is the opened file handle
# returns the dict array
# at this point we only parse a 2d array
# d = 2
# n = [nfct[irw], 2*nsrc[irw]]
def _read_array_openQCD2(fp):
    t = fp.read(4)
    d = struct.unpack('i', t)[0]
    t = fp.read(4*d)
    n = struct.unpack('%di' % (d), t)
    t = fp.read(4)
    size = struct.unpack('i', t)[0]
    if size == 4:
        types = 'i'
    elif size == 8:
        types = 'd'
    elif size == 16:
        types = 'dd'
    else:
        print('Type not known!')
    m = n[0]
    for i in range(1,d):
        m *= n[i]

    t = fp.read(m*size)
    tmp = struct.unpack('%d%s' % (m, types), t)

    arr = _parse_array_openQCD2(d, n, size, tmp, quadrupel=True)
    return {'d': d, 'n': n, 'size': size, 'arr': arr}
