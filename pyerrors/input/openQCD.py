import os
import fnmatch
import re
import struct
import numpy as np  # Thinly-wrapped numpy
import warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
from ..obs import Obs
from ..fits import fit_lin


def read_rwms(path, prefix, version='2.0', names=None, **kwargs):
    """Read rwms format from given folder structure. Returns a list of length nrw

    Parameters
    ----------
    path : str
        path that contains the data files
    prefix : str
        all files in path that start with prefix are considered as input files.
        May be used together postfix to consider only special file endings.
        Prefix is ignored, if the keyword 'files' is used.
    version : str
        version of openQCD, default 2.0
    names : list
        list of names that is assigned to the data according according
        to the order in the file list. Use careful, if you do not provide file names!
    r_start : list
        list which contains the first config to be read for each replicum
    r_stop : list
        list which contains the last config to be read for each replicum
    r_step : int
        integer that defines a fixed step size between two measurements (in units of configs)
        If not given, r_step=1 is assumed.
    postfix : str
        postfix of the file to read, e.g. '.ms1' for openQCD-files
    files : list
        list which contains the filenames to be read. No automatic detection of
        files performed if given.
    print_err : bool
        Print additional information that is useful for debugging.
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
        r_start = [o if o else None for o in r_start]
    else:
        r_start = [None] * replica

    if 'r_stop' in kwargs:
        r_stop = kwargs.get('r_stop')
        if len(r_stop) != replica:
            raise Exception('r_stop does not match number of replicas')
    else:
        r_stop = [None] * replica

    if 'r_step' in kwargs:
        r_step = kwargs.get('r_step')
    else:
        r_step = 1

    print('Read reweighting factors from', prefix[:-1], ',',
          replica, 'replica', end='')

    if names is None:
        rep_names = []
        for entry in ls:
            truncated_entry = entry.split('.')[0]
            idx = truncated_entry.index('r')
            rep_names.append(truncated_entry[:idx] + '|' + truncated_entry[idx:])
    else:
        rep_names = names

    print_err = 0
    if 'print_err' in kwargs:
        print_err = 1
        print()

    deltas = []

    configlist = []
    r_start_index = []
    r_stop_index = []

    for rep in range(replica):
        tmp_array = []
        with open(path + '/' + ls[rep], 'rb') as fp:

            t = fp.read(4)  # number of reweighting factors
            if rep == 0:
                nrw = struct.unpack('i', t)[0]
                if version == '2.0':
                    nrw = int(nrw / 2)
                for k in range(nrw):
                    deltas.append([])
            else:
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

            configlist.append([])
            while 0 < 1:
                t = fp.read(4)
                if len(t) < 4:
                    break
                config_no = struct.unpack('i', t)[0]
                configlist[-1].append(config_no)
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

            if r_start[rep] is None:
                r_start_index.append(0)
            else:
                try:
                    r_start_index.append(configlist[-1].index(r_start[rep]))
                except ValueError:
                    raise Exception('Config %d not in file with range [%d, %d]' % (
                        r_start[rep], configlist[-1][0], configlist[-1][-1])) from None

            if r_stop[rep] is None:
                r_stop_index.append(len(configlist[-1]) - 1)
            else:
                try:
                    r_stop_index.append(configlist[-1].index(r_stop[rep]))
                except ValueError:
                    raise Exception('Config %d not in file with range [%d, %d]' % (
                        r_stop[rep], configlist[-1][0], configlist[-1][-1])) from None

            for k in range(nrw):
                deltas[k].append(tmp_array[k][r_start_index[rep]:r_stop_index[rep]][::r_step])

    if np.any([len(np.unique(np.diff(cl))) != 1 for cl in configlist]):
        raise Exception('Irregular spaced data in input file!', [len(np.unique(np.diff(cl))) for cl in configlist])
    stepsizes = [list(np.unique(np.diff(cl)))[0] for cl in configlist]
    if np.any([step != 1 for step in stepsizes]):
        warnings.warn('Stepsize between configurations is greater than one!' + str(stepsizes), RuntimeWarning)

    print(',', nrw, 'reweighting factors with', nsrc, 'sources')
    result = []
    idl = [range(configlist[rep][r_start_index[rep]], configlist[rep][r_stop_index[rep]], r_step) for rep in range(replica)]
    for t in range(nrw):
        result.append(Obs(deltas[t], rep_names, idl=idl))
    return result


def extract_t0(path, prefix, dtr_read, xmin,
               spatial_extent, fit_range=5, **kwargs):
    """Extract t0 from given .ms.dat files. Returns t0 as Obs.

    It is assumed that all boundary effects have
    sufficiently decayed at x0=xmin.
    The data around the zero crossing of t^2<E> - 0.3
    is fitted with a linear function
    from which the exact root is extracted.
    Only works with openQCD

    It is assumed that one measurement is performed for each config.
    If this is not the case, the resulting idl, as well as the handling
    of r_start, r_stop and r_step is wrong and the user has to correct
    this in the resulting observable.

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
    r_stop : list
        list which contains the last config to be read for each replicum.
    r_step : int
        integer that defines a fixed step size between two measurements (in units of configs)
        If not given, r_step=1 is assumed.
    plaquette : bool
        If true extract the plaquette estimate of t0 instead.
    names : list
        list of names that is assigned to the data according according
        to the order in the file list. Use careful, if you do not provide file names!
    files : list
        list which contains the filenames to be read. No automatic detection of
        files performed if given.
    plot_fit : bool
        If true, the fit for the extraction of t0 is shown together with the data.
    assume_thermalization : bool
        If True: If the first record divided by the distance between two measurements is larger than
        1, it is assumed that this is due to thermalization and the first measurement belongs
        to the first config (default).
        If False: The config numbers are assumed to be traj_number // difference
    """

    ls = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        ls.extend(filenames)
        break

    if not ls:
        raise Exception('Error, directory not found')

    if 'files' in kwargs:
        ls = kwargs.get('files')
    else:
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
        r_start = [o if o else None for o in r_start]
    else:
        r_start = [None] * replica

    if 'r_stop' in kwargs:
        r_stop = kwargs.get('r_stop')
        if len(r_stop) != replica:
            raise Exception('r_stop does not match number of replicas')
    else:
        r_stop = [None] * replica

    if 'r_step' in kwargs:
        r_step = kwargs.get('r_step')
    else:
        r_step = 1

    print('Extract t0 from', prefix, ',', replica, 'replica')

    if 'names' in kwargs:
        rep_names = kwargs.get('names')
    else:
        rep_names = []
        for entry in ls:
            truncated_entry = entry.split('.')[0]
            idx = truncated_entry.index('r')
            rep_names.append(truncated_entry[:idx] + '|' + truncated_entry[idx:])

    Ysum = []

    configlist = []
    r_start_index = []
    r_stop_index = []

    for rep in range(replica):

        with open(path + '/' + ls[rep], 'rb') as fp:
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

            configlist.append([])
            while 0 < 1:
                t = fp.read(4)
                if(len(t) < 4):
                    break
                nc = struct.unpack('i', t)[0]
                configlist[-1].append(nc)

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

        diffmeas = configlist[-1][-1] - configlist[-1][-2]
        configlist[-1] = [item // diffmeas for item in configlist[-1]]
        if kwargs.get('assume_thermalization', True) and configlist[-1][0] > 1:
            warnings.warn('Assume thermalization and that the first measurement belongs to the first config.')
            offset = configlist[-1][0] - 1
            configlist[-1] = [item - offset for item in configlist[-1]]

        if r_start[rep] is None:
            r_start_index.append(0)
        else:
            try:
                r_start_index.append(configlist[-1].index(r_start[rep]))
            except ValueError:
                raise Exception('Config %d not in file with range [%d, %d]' % (
                    r_start[rep], configlist[-1][0], configlist[-1][-1])) from None

        if r_stop[rep] is None:
            r_stop_index.append(len(configlist[-1]) - 1)
        else:
            try:
                r_stop_index.append(configlist[-1].index(r_stop[rep]))
            except ValueError:
                raise Exception('Config %d not in file with range [%d, %d]' % (
                    r_stop[rep], configlist[-1][0], configlist[-1][-1])) from None

    if np.any([len(np.unique(np.diff(cl))) != 1 for cl in configlist]):
        raise Exception('Irregular spaced data in input file!', [len(np.unique(np.diff(cl))) for cl in configlist])
    stepsizes = [list(np.unique(np.diff(cl)))[0] for cl in configlist]
    if np.any([step != 1 for step in stepsizes]):
        warnings.warn('Stepsize between configurations is greater than one!' + str(stepsizes), RuntimeWarning)

    idl = [range(configlist[rep][r_start_index[rep]], configlist[rep][r_stop_index[rep]], r_step) for rep in range(replica)]
    t2E_dict = {}
    for n in range(nn + 1):
        samples = []
        for nrep, rep in enumerate(Ysum):
            samples.append([])
            for cnfg in rep:
                samples[-1].append(cnfg[n])
            samples[-1] = samples[-1][r_start_index[nrep]:r_stop_index[nrep]][::r_step]
        new_obs = Obs(samples, rep_names, idl=idl)
        t2E_dict[n * dn * eps] = (n * dn * eps) ** 2 * new_obs / (spatial_extent ** 3) - 0.3

    zero_crossing = np.argmax(np.array(
        [o.value for o in t2E_dict.values()]) > 0.0)

    x = list(t2E_dict.keys())[zero_crossing - fit_range:
                              zero_crossing + fit_range]
    y = list(t2E_dict.values())[zero_crossing - fit_range:
                                zero_crossing + fit_range]
    [o.gamma_method() for o in y]

    fit_result = fit_lin(x, y)

    if kwargs.get('plot_fit'):
        plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], wspace=0.0, hspace=0.0)
        ax0 = plt.subplot(gs[0])
        xmore = list(t2E_dict.keys())[zero_crossing - fit_range - 2: zero_crossing + fit_range + 2]
        ymore = list(t2E_dict.values())[zero_crossing - fit_range - 2: zero_crossing + fit_range + 2]
        [o.gamma_method() for o in ymore]
        ax0.errorbar(xmore, [yi.value for yi in ymore], yerr=[yi.dvalue for yi in ymore], fmt='x')
        xplot = np.linspace(np.min(x), np.max(x))
        yplot = [fit_result[0] + fit_result[1] * xi for xi in xplot]
        [yi.gamma_method() for yi in yplot]
        ax0.fill_between(xplot, y1=[yi.value - yi.dvalue for yi in yplot], y2=[yi.value + yi.dvalue for yi in yplot])
        retval = (-fit_result[0] / fit_result[1])
        retval.gamma_method()
        ylim = ax0.get_ylim()
        ax0.fill_betweenx(ylim, x1=retval.value - retval.dvalue, x2=retval.value + retval.dvalue, color='gray', alpha=0.4)
        ax0.set_ylim(ylim)
        ax0.set_ylabel(r'$t^2 \langle E(t) \rangle - 0.3 $')
        xlim = ax0.get_xlim()

        fit_res = [fit_result[0] + fit_result[1] * xi for xi in x]
        residuals = (np.asarray([o.value for o in y]) - [o.value for o in fit_res]) / np.asarray([o.dvalue for o in y])
        ax1 = plt.subplot(gs[1])
        ax1.plot(x, residuals, 'ko', ls='none', markersize=5)
        ax1.tick_params(direction='out')
        ax1.tick_params(axis="x", bottom=True, top=True, labelbottom=True)
        ax1.axhline(y=0.0, ls='--', color='k')
        ax1.fill_between(xlim, -1.0, 1.0, alpha=0.1, facecolor='k')
        ax1.set_xlim(xlim)
        ax1.set_ylabel('Residuals')
        ax1.set_xlabel(r'$t/a^2$')

        plt.show()
    return -fit_result[0] / fit_result[1]


def _parse_array_openQCD2(d, n, size, wa, quadrupel=False):
    arr = []
    if d == 2:
        for i in range(n[0]):
            tmp = wa[i * n[1]:(i + 1) * n[1]]
            if quadrupel:
                tmp2 = []
                for j in range(0, len(tmp), 2):
                    tmp2.append(tmp[j])
                arr.append(tmp2)
            else:
                arr.append(np.asarray(tmp))

    else:
        raise Exception('Only two-dimensional arrays supported!')

    return arr


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
    path : str
        path of the measurement files
    prefix : str
        prefix of the measurement files, e.g. <prefix>_id0_r0.ms.dat
    c : double
        Smearing radius in units of the lattice extent, c = sqrt(8 t0) / L
    dtr_cnfg : int
        (optional) parameter that specifies the number of trajectories
        between two configs.
        if it is not set, the distance between two measurements
        in the file is assumed to be
        the distance between two configurations.
    steps : int
        (optional) (maybe only necessary for openQCD2.0)
        nt step size, guessed if not given
    version : str
        version string of the openQCD (sfqcd) version used to create
        the ensemble
    L : int
        spatial length of the lattice in L/a.
        HAS to be set if version != sfqcd, since openQCD does not provide
        this in the header
    r_start : list
        offset of the first ensemble, making it easier to match
        later on with other Obs
    r_stop : list
        last configurations that need to be read (per replicum)
    files : list
        specify the exact files that need to be read
        from path, practical if e.g. only one replicum is needed
    names : list
        Alternative labeling for replicas/ensembles.
        Has to have the appropriate length
    """
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
    if "files" in kwargs:
        files = kwargs.get("files")
    else:
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
    rep_names = []

    deltas = []
    idl = []
    for rep, file in enumerate(files):
        with open(path + "/" + file, "rb") as fp:
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
    result = Obs(deltas, rep_names, idl=idl)
    return result


def read_qtop_sector(target=0, **kwargs):
    """Constructs reweighting factors to a specified topological sector.

    Parameters
    ----------
    target : int
        Specifies the topological sector to be reweighted to (default 0)
    q_top : Obs
        Alternatively takes args of read_qtop method as kwargs
    """

    if not isinstance(target, int):
        raise Exception("'target' has to be an integer.")

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
    names = qtop.names
    print(names)
    print(qtop.deltas.keys())
    proj_qtop = []
    for n in qtop.deltas:
        proj_qtop.append(np.array([1 if int(qtop.value + q) == target else 0 for q in qtop.deltas[n]]))

    result = Obs(proj_qtop, qtop.names)
    return result
