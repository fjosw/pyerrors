#!/usr/bin/env python
# coding: utf-8

import sys
import os
import fnmatch
import re
import struct
import autograd.numpy as np  # Thinly-wrapped numpy
from ..pyerrors import Obs
from ..fits import fit_lin


def read_sfcf(path, prefix, name, **kwargs):
    """Read sfcf C format from given folder structure.

    Keyword arguments
    -----------------
    im -- if True, read imaginary instead of real part of the correlation function.
    single -- if True, read a boundary-to-boundary correlation function with a single value
    b2b -- if True, read a time-dependent boundary-to-boundary correlation function
    names -- Alternative labeling for replicas/ensembles. Has to have the appropriate length
    """
    if kwargs.get('im'):
        im = 1
        part = 'imaginary'
    else:
        im = 0
        part = 'real'

    if kwargs.get('single'):
        b2b = 1
        single = 1
    else:
        b2b = 0
        single = 0

    if kwargs.get('b2b'):
        b2b = 1

    read = 0
    T = 0
    start = 0
    ls = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        ls.extend(dirnames)
        break
    if not ls:
        print('Error, directory not found')
        sys.exit()
    for exc in ls:
        if fnmatch.fnmatch(exc, prefix + '*'):
            ls = list(set(ls) - set(exc))
    if len(ls) > 1:
        ls.sort(key=lambda x: int(re.findall(r'\d+', x[len(prefix):])[0]))
    replica = len(ls)
    print('Read', part, 'part of', name, 'from', prefix, ',', replica, 'replica')
    if 'names' in kwargs:
        new_names = kwargs.get('names')
        if len(new_names) != replica:
            raise Exception('Names does not have the required length', replica)
    else:
        new_names = ls
    print(replica, 'replica')
    for i, item in enumerate(ls):
        print(item)
        sub_ls = []
        for (dirpath, dirnames, filenames) in os.walk(path+'/'+item):
            sub_ls.extend(dirnames)
            break
        for exc in sub_ls:
            if fnmatch.fnmatch(exc, 'cfg*'):
                sub_ls = list(set(sub_ls) - set(exc))
        sub_ls.sort(key=lambda x: int(x[3:]))
        no_cfg = len(sub_ls)
        print(no_cfg, 'configurations')

        if i == 0:
            with open(path + '/' + item + '/' + sub_ls[0] + '/' + name) as fp:
                for k, line in enumerate(fp):
                    if read == 1 and not line.strip() and k > start + 1:
                        break
                    if read == 1 and k >= start:
                        T += 1
                    if '[correlator]' in line:
                        read = 1
                        start = k + 7 + b2b
                        T -= b2b

            deltas = []
            for j in range(T):
                deltas.append([])

        sublength = len(sub_ls)
        for j in range(T):
            deltas[j].append(np.zeros(sublength))

        for cnfg, subitem in enumerate(sub_ls):
            with open(path + '/' + item + '/' + subitem + '/'+name) as fp:
                for k, line in enumerate(fp):
                    if(k >= start and k < start + T):
                        floats = list(map(float, line.split()))
                        deltas[k-start][i][cnfg] = floats[1 + im - single]

    result = []
    for t in range(T):
        result.append(Obs(deltas[t], new_names))

    return result


def read_sfcf_c(path, prefix, name, quarks='.*', noffset=0, wf=0, wf2=0, **kwargs):
    """Read sfcf c format from given folder structure.

    Arguments
    -----------------
    quarks -- Label of the quarks used in the sfcf input file
    noffset -- Offset of the source (only relevant when wavefunctions are used)
    wf -- ID of wave function
    wf2 -- ID of the second wavefunction (only relevant for boundary-to-boundary correlation functions)

    Keyword arguments
    -----------------
    im -- if True, read imaginary instead of real part of the correlation function.
    b2b -- if True, read a time-dependent boundary-to-boundary correlation function
    names -- Alternative labeling for replicas/ensembles. Has to have the appropriate length
    """
    
    if kwargs.get('im'):
        im = 1
        part = 'imaginary'
    else:
        im = 0
        part = 'real'

    if kwargs.get('b2b'):
        b2b = 1
    else:
        b2b = 0

    read = 0
    T = 0
    start = 0
    ls = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        ls.extend(dirnames)
        break
    if not ls:
        print('Error, directory not found')
        sys.exit()
    # Exclude folders with different names
    for exc in ls:
        if not fnmatch.fnmatch(exc, prefix+'*'):
            ls = list(set(ls) - set([exc]))
    if len(ls) > 1:
        ls.sort(key=lambda x: int(re.findall(r'\d+', x[len(prefix):])[0]))  # New version, to cope with ids, etc.
    replica = len(ls)
    if 'names' in kwargs:
        new_names = kwargs.get('names')
        if len(new_names) != replica:
            raise Exception('Names does not have the required length', replica)
    else:
        new_names = ls
    print('Read', part, 'part of', name, 'from', prefix[:-1], ',', replica, 'replica')
    for i, item in enumerate(ls):
        sub_ls = []
        for (dirpath, dirnames, filenames) in os.walk(path+'/'+item):
            sub_ls.extend(filenames)
            break
        for exc in sub_ls:
            if not fnmatch.fnmatch(exc, prefix+'*'):
                sub_ls = list(set(sub_ls) - set([exc]))
        sub_ls.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))

        first_cfg = int(re.findall(r'\d+', sub_ls[0])[-1])

        last_cfg = len(sub_ls) + first_cfg - 1

        for cfg in range(1, len(sub_ls)):
            if int(re.findall(r'\d+', sub_ls[cfg])[-1]) != first_cfg + cfg:
                last_cfg = cfg + first_cfg - 1
                break

        no_cfg = last_cfg - first_cfg + 1
        print(item, ':', no_cfg, 'evenly spaced configurations (', first_cfg, '-', last_cfg, ') ,', len(sub_ls) - no_cfg, 'configs omitted\n')

        if i == 0:
            pattern = 'name      ' + name + '\nquarks    ' + quarks + '\noffset    ' + str(noffset) + '\nwf        ' + str(wf)
            if b2b:
                pattern += '\nwf_2      ' + str(wf2)

            with open(path+'/'+item+'/'+sub_ls[0], 'r') as file:
                content = file.read()
                match = re.search(pattern, content)
                if match:
                    start_read = content.count('\n', 0, match.start()) + 5 + b2b
                    end_match = re.search('\n\s*\n', content[match.start():])
                    T = content[match.start():].count('\n', 0, end_match.start()) - 4 - b2b
                    assert T > 0
                    print(T, 'entries, starting to read in line', start_read)
                else:
                    raise Exception('Correlator with pattern\n' + pattern + '\nnot found.')

            deltas = []
            for j in range(T):
                deltas.append([])

        sublength = no_cfg
        for j in range(T):
            deltas[j].append(np.zeros(sublength))

        for cfg in range(no_cfg):
            with open(path+'/'+item+'/'+sub_ls[cfg]) as fp:
                for k, line in enumerate(fp):
                    if k == start_read - 5 - b2b:
                        if line.strip() != 'name      ' + name:
                            raise Exception('Wrong format', sub_ls[cfg])
                    if(k >= start_read and k < start_read + T):
                        floats = list(map(float, line.split()))
                        deltas[k-start_read][i][cfg] = floats[-2:][im]

    result = []
    for t in range(T):
        result.append(Obs(deltas[t], new_names))

    return result


def read_qtop(path, prefix, **kwargs):
    """Read qtop format from given folder structure.

    Keyword arguments
    -----------------
    target -- specifies the topological sector to be reweighted to (default 0)
    full -- if true read the charge instead of the reweighting factor.
    """

    if 'target' in kwargs:
        target = kwargs.get('target')
    else:
        target = 0

    if kwargs.get('full'):
        full = 1
    else:
        full = 0

    ls = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        ls.extend(filenames)
        break

    if not ls:
        print('Error, directory not found')
        sys.exit()

    # Exclude files with different names
    for exc in ls:
        if not fnmatch.fnmatch(exc, prefix+'*'):
            ls = list(set(ls) - set([exc]))
    if len(ls) > 1:
        ls.sort(key=lambda x: int(re.findall(r'\d+', x[len(prefix):])[0]))  # New version, to cope with ids, etc.
    replica = len(ls)
    print('Read Q_top from', prefix[:-1], ',', replica, 'replica')

    deltas = []

    for rep in range(replica):
        tmp = []
        with open(path+'/'+ls[rep]) as fp:
            for k, line in enumerate(fp):
                floats = list(map(float, line.split()))
                if full == 1:
                    tmp.append(floats[1])
                else:
                    if int(floats[1]) == target:
                        tmp.append(1.0)
                    else:
                        tmp.append(0.0)

        deltas.append(np.array(tmp))

    result = Obs(deltas, [(w.split('.'))[0] for w in ls])

    return result


def read_rwms(path, prefix, **kwargs):
    """Read rwms format from given folder structure. Returns a list of length nrw

    Keyword arguments
    -----------------
    new_format -- if True, the array of the associated numbers of Hasenbusch factors is extracted (v>=openQCD1.6)
    r_start -- list which contains the first config to be read for each replicum
    r_stop -- list which contains the last config to be read for each replicum

    """

    if kwargs.get('new_format'):
        extract_nfct = 1
    else:
        extract_nfct = 0

    ls = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        ls.extend(filenames)
        break

    if not ls:
        print('Error, directory not found')
        sys.exit()

    # Exclude files with different names
    for exc in ls:
        if not fnmatch.fnmatch(exc, prefix + '*.dat'):
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
                for k in range(nrw):
                    deltas.append([])
            else:
                if nrw != struct.unpack('i', t)[0]:
                    print('Error: different number of reweighting factors for replicum', rep)
                    sys.exit()

            for k in range(nrw):
                tmp_array.append([])

            # This block is necessary for openQCD1.6 ms1 files
            nfct = []
            if extract_nfct == 1:
                for i in range(nrw):
                    t = fp.read(4)
                    nfct.append(struct.unpack('i', t)[0])
                print('nfct: ', nfct) # Hasenbusch factor, 1 for rat reweighting
            else:
                for i in range(nrw):
                    nfct.append(1)

            nsrc = []
            for i in range(nrw):
                t = fp.read(4)
                nsrc.append(struct.unpack('i', t)[0])

            #body
            while 0 < 1:
                t = fp.read(4)
                if len(t) < 4:
                    break
                if print_err:
                    config_no = struct.unpack('i', t)
                for i in range(nrw):
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
        result.append(Obs(deltas[t], [(w.split('.'))[0] for w in ls]))

    return result


def read_pbp(path, prefix, **kwargs):
    """Read pbp format from given folder structure. Returns a list of length nrw

    Keyword arguments
    -----------------
    r_start -- list which contains the first config to be read for each replicum
    r_stop -- list which contains the last config to be read for each replicum

    """

    extract_nfct = 1

    ls = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        ls.extend(filenames)
        break

    if not ls:
        print('Error, directory not found')
        sys.exit()

    # Exclude files with different names
    for exc in ls:
        if not fnmatch.fnmatch(exc, prefix + '*.dat'):
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

    print('Read <bar{psi}\psi> from', prefix[:-1], ',', replica, 'replica', end='')

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
                for k in range(nrw):
                    deltas.append([])
            else:
                if nrw != struct.unpack('i', t)[0]:
                    print('Error: different number of reweighting factors for replicum', rep)
                    sys.exit()

            for k in range(nrw):
                tmp_array.append([])

            # This block is necessary for openQCD1.6 ms1 files
            nfct = []
            if extract_nfct == 1:
                for i in range(nrw):
                    t = fp.read(4)
                    nfct.append(struct.unpack('i', t)[0])
                print('nfct: ', nfct) # Hasenbusch factor, 1 for rat reweighting
            else:
                for i in range(nrw):
                    nfct.append(1)

            nsrc = []
            for i in range(nrw):
                t = fp.read(4)
                nsrc.append(struct.unpack('i', t)[0])

            #body
            while 0 < 1:
                t = fp.read(4)
                if len(t) < 4:
                    break
                if print_err:
                    config_no = struct.unpack('i', t)
                for i in range(nrw):
                    tmp_nfct = 1.0
                    for j in range(nfct[i]):
                        t = fp.read(8 * nsrc[i])
                        t = fp.read(8 * nsrc[i])
                        tmp_rw = struct.unpack('d' * nsrc[i], t)
                        tmp_nfct *= np.mean(np.asarray(tmp_rw))
                        if print_err:
                            print(config_no, i, j, np.mean(np.asarray(tmp_rw)), np.std(np.asarray(tmp_rw)))
                            print('Sources:', np.asarray(tmp_rw))
                            print('Partial factor:', tmp_nfct)
                    tmp_array[i].append(tmp_nfct)

            for k in range(nrw):
                deltas[k].append(tmp_array[k][r_start[rep]:r_stop[rep]])

    print(',', nrw, '<bar{psi}\psi> with', nsrc, 'sources')
    result = []
    for t in range(nrw):
        result.append(Obs(deltas[t], [(w.split('.'))[0] for w in ls]))

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
        print('Error, directory not found')
        sys.exit()

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
