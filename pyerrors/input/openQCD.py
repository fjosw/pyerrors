#!/usr/bin/env python
# coding: utf-8

import os
import fnmatch
import re
import struct
import numpy as np  # Thinly-wrapped numpy
from ..obs import Obs
from ..fits import fit_lin


def read_rwms(path, prefix, version='2.0', names=None, **kwargs):
    """Read rwms format from given folder structure. Returns a list of length nrw

    Parameters
    ----------
    version : str
        version of openQCD, default 2.0
    r_start : list
        list which contains the first config to be read for each replicum
    r_stop : list
        list which contains the last config to be read for each replicum
    postfix : str
        postfix of the file to read, e.g. '.ms1' for openQCD-files
    idl_offsets : list
        offsets to the idl range of obs. Useful for the case that the measurements of rwms are only starting at cfg. 20
    """
    known_oqcd_versions = ['1.4', '1.6', '2.0']
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
    if 'files' in kwargs:
        ls = kwargs.get('files')
    else:
        # Exclude files with different names
        for exc in ls:
            if not fnmatch.fnmatch(exc, prefix + '*' + postfix + '.dat'):
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

    print('Read reweighting factors from', prefix[:-1], ',',
          replica, 'replica', end='')

    # Adjust replica names to new bookmarking system
    if names is None:
        rep_names = []
        for entry in ls:
            truncated_entry = entry.split('.')[0]
            idx = truncated_entry.index('r')
            rep_names.append(truncated_entry[:idx] + '|' + truncated_entry[idx:])

    print_err = 0
    if 'print_err' in kwargs:
        print_err = 1
        print()

    deltas = []

    for rep in range(replica):
        tmp_array = []
        with open(path + '/' + ls[rep], 'rb') as fp:

            # header
            t = fp.read(4)  # number of reweighting factors
            if rep == 0:
                nrw = struct.unpack('i', t)[0]
                if version == '2.0':
                    nrw = int(nrw / 2)
                for k in range(nrw):
                    deltas.append([])
            else:
                # little weird if-clause due to the /2 operation needed.
                if ((nrw != struct.unpack('i', t)[0] and (not version == '2.0')) or (nrw != struct.unpack('i', t)[0] / 2 and version == '2.0')):
                    raise Exception('Error: different number of reweighting factors for replicum', rep)

            for k in range(nrw):
                tmp_array.append([])

            # This block is necessary for openQCD1.6 and openQCD2.0 ms1 files
            nfct = []
            if version in ['1.6', '2.0']:
                for i in range(nrw):
                    t = fp.read(4)
                    nfct.append(struct.unpack('i', t)[0])
                # print('nfct: ', nfct) # Hasenbusch factor,
                # 1 for rat reweighting
            else:
                for i in range(nrw):
                    nfct.append(1)

            nsrc = []
            for i in range(nrw):
                t = fp.read(4)
                nsrc.append(struct.unpack('i', t)[0])
            if version == '2.0':
                if not struct.unpack('i', fp.read(4))[0] == 0:
                    print('something is wrong!')

            # body
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
                                print(config_no, i, j,
                                      np.mean(np.exp(-np.asarray(tmp_rw[j]))),
                                      np.std(np.exp(-np.asarray(tmp_rw[j]))))
                                print('Sources:',
                                      np.exp(-np.asarray(tmp_rw[j])))
                                print('Partial factor:', tmp_nfct)
                    elif version == '1.6' or version == '1.4':
                        tmp_nfct = 1.0
                        for j in range(nfct[i]):
                            t = fp.read(8 * nsrc[i])
                            t = fp.read(8 * nsrc[i])
                            tmp_rw = struct.unpack('d' * nsrc[i], t)
                            tmp_nfct *= np.mean(np.exp(-np.asarray(tmp_rw)))
                            if print_err:
                                print(config_no, i, j,
                                      np.mean(np.exp(-np.asarray(tmp_rw))),
                                      np.std(np.exp(-np.asarray(tmp_rw))))
                                print('Sources:', np.exp(-np.asarray(tmp_rw)))
                                print('Partial factor:', tmp_nfct)
                    tmp_array[i].append(tmp_nfct)

            for k in range(nrw):
                deltas[k].append(tmp_array[k][r_start[rep]:r_stop[rep]])

    print(',', nrw, 'reweighting factors with', nsrc, 'sources')
    if "idl_offsets" in kwargs:
        idl_offsets = kwargs.get("idl_offsets")
    else:
        idl_offsets = np.ones(nrw, dtype = int)
    result = []
    for t in range(nrw):
        idl = []
        for rep in range(replica):
            idl.append(range(idl_offsets[rep],len(deltas[t][rep]+idl_offsets[rep])))
        if names is None:
            result.append(Obs(deltas[t], rep_names, idl = idl))
        else:
            print(names)
            result.append(Obs(deltas[t], names, idl = idl))
    return result


def extract_t0(path, prefix, dtr_read, xmin,
               spatial_extent, fit_range=5, **kwargs):
    """Extract t0 from given .ms.dat files. Returns t0 as Obs.

    It is assumed that all boundary effects have
    sufficiently decayed at x0=xmin.
    The data around the zero crossing of t^2<E> - 0.3
    is fitted with a linear function
    from which the exact root is extracted.
    Only works with openQCD v 1.2.

    Parameters
    ----------
    path : str
        Path to .ms.dat files
    prefix : str
        Ensemble prefix
    dtr_read : int
        Determines how many trajectories should be skipped
        when reading the ms.dat files.
        Corresponds to dtr_cnfg / dtr_ms in the openQCD input file.
    xmin : int
        First timeslice where the boundary
        effects have sufficiently decayed.
    spatial_extent : int
        spatial extent of the lattice, required for normalization.
    fit_range : int
        Number of data points left and right of the zero
        crossing to be included in the linear fit. (Default: 5)
    r_start : list
        list which contains the first config to be read for each replicum.
    r_stop: list
        list which contains the last config to be read for each replicum.
    plaquette : bool
        If true extract the plaquette estimate of t0 instead.
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
            Ysum[-1].append([np.mean(item[current + xmin:
                             current + tmax - xmin])
                            for current in range(0, len(item), tmax)])

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

    zero_crossing = np.argmax(np.array(
        [o.value for o in t2E_dict.values()]) > 0.0)

    x = list(t2E_dict.keys())[zero_crossing - fit_range:
                              zero_crossing + fit_range]
    y = list(t2E_dict.values())[zero_crossing - fit_range:
                                zero_crossing + fit_range]
    [o.gamma_method() for o in y]

    fit_result = fit_lin(x, y)
    return -fit_result[0] / fit_result[1]


def _parse_array_openQCD2(d, n, size, wa, quadrupel=False):
    arr = []
    if d == 2:
        tot = 0
        for i in range(n[d - 1] - 1):
            if quadrupel:
                tmp = wa[tot:n[d - 1]]
                tmp2 = []
                for i in range(len(tmp)):
                    if i % 2 == 0:
                        tmp2.append(tmp[i])
                arr.append(tmp2)
            else:
                arr.append(np.asarray(wa[tot:n[d - 1]]))
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
    t = fp.read(4 * d)
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
    for i in range(1, d):
        m *= n[i]

    t = fp.read(m * size)
    tmp = struct.unpack('%d%s' % (m, types), t)

    arr = _parse_array_openQCD2(d, n, size, tmp, quadrupel=True)
    return {'d': d, 'n': n, 'size': size, 'arr': arr}


def read_qtop(path, prefix, c, dtr_cnfg=1, version="1.2", **kwargs):
    """Read qtop format from given folder structure.

    Parameters
    ----------
    path:
        path of the measurement files
    prefix:
        prefix of the measurement files, e.g. <prefix>_id0_r0.ms.dat
    c: double
        Smearing radius in units of the lattice extent, c = sqrt(8 t0) / L
    dtr_cnfg: int
        (optional) parameter that specifies the number of trajectories
        between two configs.
        if it is not set, the distance between two measurements
        in the file is assumed to be
        the distance between two configurations.
    steps: int
        (optional) (maybe only necessary for openQCD2.0)
        nt step size, guessed if not given
    version: str
        version string of the openQCD (sfqcd) version used to create
        the ensemble
    L: int
        spatial length of the lattice in L/a.
        HAS to be set if version != sfqcd, since openQCD does not provide
        this in the header
    r_start: list
        offset of the first ensemble, making it easier to match
        later on with other Obs
    r_stop: list
        last configurations that need to be read (per replicum)
    files: list
        specify the exact files that need to be read
        from path, pratical if e.g. only one replicum is needed
    names: list
        Alternative labeling for replicas/ensembles.
        Has to have the appropriate length
    """
    # one could read L from the header in case of sfQCD
    # c = 0.35
    known_versions = ["1.0", "1.2", "1.4", "1.6", "2.0", "sfqcd"]

    if version not in known_versions:
        raise Exception("Unknown openQCD version.")
    if "steps" in kwargs:
        steps = kwargs.get("steps")
    if version == "sfqcd":
        if "L" in kwargs:
            supposed_L = kwargs.get("L")
    else:
        if "L" not in kwargs:
            raise Exception("This version of openQCD needs you to provide the spatial length of the lattice as parameter 'L'.")
        else:
            L = kwargs.get("L")
    r_start = 1
    if "r_start" in kwargs:
        r_start = kwargs.get("r_start")
    if "r_stop" in kwargs:
        r_stop = kwargs.get("r_stop")
    # if one wants to read specific files with this method...
    if "files" in kwargs:
        files = kwargs.get("files")
    else:
        # find files in path
        found = []
        files = []
        for (dirpath, dirnames, filenames) in os.walk(path + "/"):
            # print(filenames)
            found.extend(filenames)
            break
        for f in found:
            if fnmatch.fnmatch(f, prefix + "*" + ".ms.dat"):
                files.append(f)
        print(files)
    # now that we found our files, we dechiffer them...
    rep_names = []

    deltas = []
    idl = []
    for rep, file in enumerate(files):
        with open(path + "/" + file, "rb") as fp:
            # header
            t = fp.read(12)
            header = struct.unpack('<iii', t)
            # step size in integration steps "dnms"
            dn = header[0]
            # number of measurements, so "ntot"/dn
            nn = header[1]
            # lattice T/a
            tmax = header[2]
            if version == "sfqcd":
                t = fp.read(12)
                Ls = struct.unpack('<iii', t)
                if(Ls[0] == Ls[1] and Ls[1] == Ls[2]):
                    L = Ls[0]
                    if not (supposed_L == L):
                        raise Exception("It seems the length given in the header and by you contradict each other")
                else:
                    raise Exception("Found more than one spatial length in header!")

            print('dnms:', dn)
            print('nn:', nn)
            print('tmax:', tmax)
            t = fp.read(8)
            eps = struct.unpack('d', t)[0]
            print('eps:', eps)

            Q = []
            ncs = []
            while 0 < 1:
                # int nt
                t = fp.read(4)
                if(len(t) < 4):
                    break
                ncs.append(struct.unpack('i', t)[0])
                # Wsl
                t = fp.read(8 * tmax * (nn + 1))
                # Ysl
                t = fp.read(8 * tmax * (nn + 1))
                # Qsl, which is asked for in this method
                t = fp.read(8 * tmax * (nn + 1))
                # unpack the array of Qtops,
                # on each timeslice t=0,...,tmax-1 and the
                # measurement number in = 0...nn (see README.qcd1)
                tmpd = struct.unpack('d' * tmax * (nn + 1), t)
                Q.append(tmpd)

        if not len(set([ncs[i] - ncs[i - 1] for i in range(1, len(ncs))])):
            raise Exception("Irregularities in stepsize found")
        else:
            if 'steps' in kwargs:
                if steps != ncs[1] - ncs[0]:
                    raise Exception("steps and the found stepsize are not the same")
            else:
                steps = ncs[1] - ncs[0]

        print(len(Q))
        print('max_t:', dn * (nn) * eps)

        t_aim = (c * L) ** 2 / 8

        print('t_aim:', t_aim)
        index_aim = round(t_aim / eps / dn)
        print('index_aim:', index_aim)

        Q_sum = []
        for i, item in enumerate(Q):
            Q_sum.append([sum(item[current:current + tmax])
                         for current in range(0, len(item), tmax)])
        print(len(Q_sum))
        print(len(Q_sum[0]))
        Q_round = []
        for i in range(len(Q) // dtr_cnfg):
            Q_round.append(round(Q_sum[dtr_cnfg * i][index_aim]))
        if len(Q_round) != len(ncs) // dtr_cnfg:
            raise Exception("qtops and ncs dont have the same length")

        # replica = len(files)

        truncated_file = file[:-7]
        print(truncated_file)
        idl_start = 1

        if "r_start" in kwargs:
            Q_round = Q_round[r_start[rep]:]
            idl_start = r_start[rep]
        if "r_stop" in kwargs:
            Q_round = Q_round[:r_stop[rep]]
        idl_stop = idl_start + len(Q_round)
        # keyword "names" prevails over "ens_name"
        if "names" not in kwargs:
            try:
                idx = truncated_file.index('r')
            except Exception:
                if "names" not in kwargs:
                    raise Exception("Automatic recognition of replicum failed, please enter the key word 'names'.")
            if "ens_name" in kwargs:
                ens_name = kwargs.get("ens_name")
            else:
                ens_name = truncated_file[:idx]
            rep_names.append(ens_name + '|' + truncated_file[idx:])
        else:
            names = kwargs.get("names")
            rep_names = names
        deltas.append(np.array(Q_round))
        idl.append(range(idl_start, idl_stop))
    # print(idl)
    result = Obs(deltas, rep_names, idl=idl)
    return result


def read_qtop_sector(target=0, **kwargs):
    """target: int
            specifies the topological sector to be reweighted to (default 0)
        q_top: Obs
        alternatively takes args of read_qtop method as kwargs
    """
    if "q_top" in kwargs:
        qtop = kwargs.get("q_top")
    else:
        if "path" in kwargs:
            path = kwargs.get("path")
            del kwargs["path"]
        else:
            raise Exception("If you are not providing q_top, please provide path")
        if "prefix" in kwargs:
            prefix = kwargs.get("prefix")
            del kwargs["prefix"]
        else:
            raise Exception("If you are not providing q_top, please provide prefix")
        if "c" in kwargs:
            c = kwargs.get("c")
            del kwargs["c"]
        else:
            raise Exception("If you are not providing q_top, please provide c")
        if "version" in kwargs:
            version = kwargs.get("version")
            del kwargs["version"]
        else:
            version = "1.2"
        if "dtr_cnfg" in kwargs:
            dtr_cnfg = kwargs.get("dtr_cnfg")
            del kwargs["dtr_cnfg"]
        else:
            dtr_cnfg = 1
        qtop = read_qtop(path, prefix, c, dtr_cnfg=dtr_cnfg,
                         version=version, **kwargs)
    # unpack to original values, project onto target sector
    names = qtop.names
    print(names)
    print(qtop.deltas.keys())
    proj_qtop = []
    for n in qtop.deltas:
        proj_qtop.append(np.array([1 if int(qtop.value + q) == target else 0 for q in qtop.deltas[n]]))

    result = Obs(proj_qtop, qtop.names)
    return result
