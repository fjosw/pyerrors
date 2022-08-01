import os
import fnmatch
import re
import struct
import warnings
import numpy as np  # Thinly-wrapped numpy
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
            truncated_entry = entry
            suffixes = [".dat", ".rwms", ".ms1"]
            for suffix in suffixes:
                if truncated_entry.endswith(suffix):
                    truncated_entry = truncated_entry[0:-len(suffix)]
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
            while True:
                t = fp.read(4)
                if len(t) < 4:
                    break
                config_no = struct.unpack('i', t)[0]
                configlist[-1].append(config_no)
                for i in range(nrw):
                    if (version == '2.0'):
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

            diffmeas = configlist[-1][-1] - configlist[-1][-2]
            configlist[-1] = [item // diffmeas for item in configlist[-1]]
            if configlist[-1][0] > 1 and diffmeas > 1:
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

            for k in range(nrw):
                deltas[k].append(tmp_array[k][r_start_index[rep]:r_stop_index[rep] + 1][::r_step])

    if np.any([len(np.unique(np.diff(cl))) != 1 for cl in configlist]):
        raise Exception('Irregular spaced data in input file!', [len(np.unique(np.diff(cl))) for cl in configlist])
    stepsizes = [list(np.unique(np.diff(cl)))[0] for cl in configlist]
    if np.any([step != 1 for step in stepsizes]):
        warnings.warn('Stepsize between configurations is greater than one!' + str(stepsizes), RuntimeWarning)

    print(',', nrw, 'reweighting factors with', nsrc, 'sources')
    result = []
    idl = [range(configlist[rep][r_start_index[rep]], configlist[rep][r_stop_index[rep]] + 1, r_step) for rep in range(replica)]

    for t in range(nrw):
        result.append(Obs(deltas[t], rep_names, idl=idl))
    return result


def extract_t0(path, prefix, dtr_read, xmin, spatial_extent, fit_range=5, **kwargs):
    """Extract t0 from given .ms.dat files. Returns t0 as Obs.

    It is assumed that all boundary effects have
    sufficiently decayed at x0=xmin.
    The data around the zero crossing of t^2<E> - 0.3
    is fitted with a linear function
    from which the exact root is extracted.

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
            while True:
                t = fp.read(4)
                if (len(t) < 4):
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

    idl = [range(configlist[rep][r_start_index[rep]], configlist[rep][r_stop_index[rep]] + 1, r_step) for rep in range(replica)]
    t2E_dict = {}
    for n in range(nn + 1):
        samples = []
        for nrep, rep in enumerate(Ysum):
            samples.append([])
            for cnfg in rep:
                samples[-1].append(cnfg[n])
            samples[-1] = samples[-1][r_start_index[nrep]:r_stop_index[nrep] + 1][::r_step]
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

        plt.draw()
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
        raise Exception("Type for size '" + str(size) + "' not known.")
    m = n[0]
    for i in range(1, d):
        m *= n[i]

    t = fp.read(m * size)
    tmp = struct.unpack('%d%s' % (m, types), t)

    arr = _parse_array_openQCD2(d, n, size, tmp, quadrupel=True)
    return {'d': d, 'n': n, 'size': size, 'arr': arr}


def read_qtop(path, prefix, c, dtr_cnfg=1, version="openQCD", **kwargs):
    """Read the topologial charge based on openQCD gradient flow measurements.

    Parameters
    ----------
    path : str
        path of the measurement files
    prefix : str
        prefix of the measurement files, e.g. <prefix>_id0_r0.ms.dat.
        Ignored if file names are passed explicitly via keyword files.
    c : double
        Smearing radius in units of the lattice extent, c = sqrt(8 t0) / L.
    dtr_cnfg : int
        (optional) parameter that specifies the number of measurements
        between two configs.
        If it is not set, the distance between two measurements
        in the file is assumed to be the distance between two configurations.
    steps : int
        (optional) Distance between two configurations in units of trajectories /
         cycles. Assumed to be the distance between two measurements * dtr_cnfg if not given
    version : str
        Either openQCD or sfqcd, depending on the data.
    L : int
        spatial length of the lattice in L/a.
        HAS to be set if version != sfqcd, since openQCD does not provide
        this in the header
    r_start : list
        list which contains the first config to be read for each replicum.
    r_stop : list
        list which contains the last config to be read for each replicum.
    files : list
        specify the exact files that need to be read
        from path, practical if e.g. only one replicum is needed
    names : list
        Alternative labeling for replicas/ensembles.
        Has to have the appropriate length.
    Zeuthen_flow : bool
        (optional) If True, the Zeuthen flow is used for Qtop. Only possible
        for version=='sfqcd' If False, the Wilson flow is used.
    integer_charge : bool
        If True, the charge is rounded towards the nearest integer on each config.
    """

    return _read_flow_obs(path, prefix, c, dtr_cnfg=dtr_cnfg, version=version, obspos=0, **kwargs)


def read_gf_coupling(path, prefix, c, dtr_cnfg=1, Zeuthen_flow=True, **kwargs):
    """Read the gradient flow coupling based on sfqcd gradient flow measurements. See 1607.06423 for details.

    Note: The current implementation only works for c=0.3 and T=L. The definition of the coupling in 1607.06423 requires projection to topological charge zero which is not done within this function but has to be performed in a separate step.

    Parameters
    ----------
    path : str
        path of the measurement files
    prefix : str
        prefix of the measurement files, e.g. <prefix>_id0_r0.ms.dat.
        Ignored if file names are passed explicitly via keyword files.
    c : double
        Smearing radius in units of the lattice extent, c = sqrt(8 t0) / L.
    dtr_cnfg : int
        (optional) parameter that specifies the number of measurements
        between two configs.
        If it is not set, the distance between two measurements
        in the file is assumed to be the distance between two configurations.
    steps : int
        (optional) Distance between two configurations in units of trajectories /
         cycles. Assumed to be the distance between two measurements * dtr_cnfg if not given
    r_start : list
        list which contains the first config to be read for each replicum.
    r_stop : list
        list which contains the last config to be read for each replicum.
    files : list
        specify the exact files that need to be read
        from path, practical if e.g. only one replicum is needed
    names : list
        Alternative labeling for replicas/ensembles.
        Has to have the appropriate length.
    Zeuthen_flow : bool
        (optional) If True, the Zeuthen flow is used for the coupling. If False, the Wilson flow is used.
    """

    if c != 0.3:
        raise Exception("The required lattice norm is only implemented for c=0.3 at the moment.")

    plaq = _read_flow_obs(path, prefix, c, dtr_cnfg=dtr_cnfg, version="sfqcd", obspos=6, sum_t=False, Zeuthen_flow=Zeuthen_flow, integer_charge=False, **kwargs)
    C2x1 = _read_flow_obs(path, prefix, c, dtr_cnfg=dtr_cnfg, version="sfqcd", obspos=7, sum_t=False, Zeuthen_flow=Zeuthen_flow, integer_charge=False, **kwargs)
    L = plaq.tag["L"]
    T = plaq.tag["T"]

    if T != L:
        raise Exception("The required lattice norm is only implemented for T=L at the moment.")

    if Zeuthen_flow is not True:
        raise Exception("The required lattice norm is only implemented for the Zeuthen flow at the moment.")

    t = (c * L) ** 2 / 8

    normdict = {4: 0.012341170468270,
                6: 0.010162691462430,
                8: 0.009031614807931,
                10: 0.008744966371393,
                12: 0.008650917856809,
                14: 8.611154391267955E-03,
                16: 0.008591758449508,
                20: 0.008575359627103,
                24: 0.008569387847540,
                28: 8.566803713382559E-03,
                32: 0.008565541650006,
                40: 8.564480684962046E-03,
                48: 8.564098025073460E-03,
                64: 8.563853943383087E-03}

    return t * t * (5 / 3 * plaq - 1 / 12 * C2x1) / normdict[L]


def _read_flow_obs(path, prefix, c, dtr_cnfg=1, version="openQCD", obspos=0, sum_t=True, **kwargs):
    """Read a flow observable based on openQCD gradient flow measurements.

    Parameters
    ----------
    path : str
        path of the measurement files
    prefix : str
        prefix of the measurement files, e.g. <prefix>_id0_r0.ms.dat.
        Ignored if file names are passed explicitly via keyword files.
    c : double
        Smearing radius in units of the lattice extent, c = sqrt(8 t0) / L.
    dtr_cnfg : int
        (optional) parameter that specifies the number of measurements
        between two configs.
        If it is not set, the distance between two measurements
        in the file is assumed to be the distance between two configurations.
    steps : int
        (optional) Distance between two configurations in units of trajectories /
         cycles. Assumed to be the distance between two measurements * dtr_cnfg if not given
    version : str
        Either openQCD or sfqcd, depending on the data.
    obspos : int
        position of the obeservable in the measurement file. Only relevant for sfqcd files.
    sum_t : bool
        If true sum over all timeslices, if false only take the value at T/2.
    L : int
        spatial length of the lattice in L/a.
        HAS to be set if version != sfqcd, since openQCD does not provide
        this in the header
    r_start : list
        list which contains the first config to be read for each replicum.
    r_stop : list
        list which contains the last config to be read for each replicum.
    files : list
        specify the exact files that need to be read
        from path, practical if e.g. only one replicum is needed
    names : list
        Alternative labeling for replicas/ensembles.
        Has to have the appropriate length.
    Zeuthen_flow : bool
        (optional) If True, the Zeuthen flow is used for Qtop. Only possible
        for version=='sfqcd' If False, the Wilson flow is used.
    integer_charge : bool
        If True, the charge is rounded towards the nearest integer on each config.
    """
    known_versions = ["openQCD", "sfqcd"]

    if version not in known_versions:
        raise Exception("Unknown openQCD version.")
    if "steps" in kwargs:
        steps = kwargs.get("steps")
    if version == "sfqcd":
        if "L" in kwargs:
            supposed_L = kwargs.get("L")
        else:
            supposed_L = None
        postfix = ".gfms.dat"
    else:
        if "L" not in kwargs:
            raise Exception("This version of openQCD needs you to provide the spatial length of the lattice as parameter 'L'.")
        else:
            L = kwargs.get("L")
        postfix = ".ms.dat"

    if "files" in kwargs:
        files = kwargs.get("files")
        postfix = ''
    else:
        found = []
        files = []
        for (dirpath, dirnames, filenames) in os.walk(path + "/"):
            found.extend(filenames)
            break
        for f in found:
            if fnmatch.fnmatch(f, prefix + "*" + postfix):
                files.append(f)

    if 'r_start' in kwargs:
        r_start = kwargs.get('r_start')
        if len(r_start) != len(files):
            raise Exception('r_start does not match number of replicas')
        r_start = [o if o else None for o in r_start]
    else:
        r_start = [None] * len(files)

    if 'r_stop' in kwargs:
        r_stop = kwargs.get('r_stop')
        if len(r_stop) != len(files):
            raise Exception('r_stop does not match number of replicas')
    else:
        r_stop = [None] * len(files)
    rep_names = []

    zeuthen = kwargs.get('Zeuthen_flow', False)
    if zeuthen and version not in ['sfqcd']:
        raise Exception('Zeuthen flow can only be used for version==sfqcd')

    r_start_index = []
    r_stop_index = []
    deltas = []
    configlist = []
    if not zeuthen:
        obspos += 8
    for rep, file in enumerate(files):
        with open(path + "/" + file, "rb") as fp:

            Q = []
            traj_list = []
            if version in ['sfqcd']:
                t = fp.read(12)
                header = struct.unpack('<iii', t)
                zthfl = header[0]  # Zeuthen flow -> if it's equal to 2 it means that the Zeuthen flow is also 'measured' (apart from the Wilson flow)
                ncs = header[1]  # number of different values for c in t_flow=1/8 c² L² -> measurements done for ncs c's
                tmax = header[2]  # lattice T/a

                t = fp.read(12)
                Ls = struct.unpack('<iii', t)
                if (Ls[0] == Ls[1] and Ls[1] == Ls[2]):
                    L = Ls[0]
                    if not (supposed_L == L) and supposed_L:
                        raise Exception("It seems the length given in the header and by you contradict each other")
                else:
                    raise Exception("Found more than one spatial length in header!")

                t = fp.read(16)
                header2 = struct.unpack('<dd', t)
                tol = header2[0]
                cmax = header2[1]  # highest value of c used

                if c > cmax:
                    raise Exception('Flow has been determined between c=0 and c=%lf with tolerance %lf' % (cmax, tol))

                if (zthfl == 2):
                    nfl = 2  # number of flows
                else:
                    nfl = 1
                iobs = 8 * nfl  # number of flow observables calculated

                while True:
                    t = fp.read(4)
                    if (len(t) < 4):
                        break
                    traj_list.append(struct.unpack('i', t)[0])   # trajectory number when measurement was done

                    for j in range(ncs + 1):
                        for i in range(iobs):
                            t = fp.read(8 * tmax)
                            if (i == obspos):  # determines the flow observable -> i=0 <-> Zeuthen flow
                                Q.append(struct.unpack('d' * tmax, t))

            else:
                t = fp.read(12)
                header = struct.unpack('<iii', t)
                # step size in integration steps "dnms"
                dn = header[0]
                # number of measurements, so "ntot"/dn
                nn = header[1]
                # lattice T/a
                tmax = header[2]

                t = fp.read(8)
                eps = struct.unpack('d', t)[0]

                while True:
                    t = fp.read(4)
                    if (len(t) < 4):
                        break
                    traj_list.append(struct.unpack('i', t)[0])
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

        if len(np.unique(np.diff(traj_list))) != 1:
            raise Exception("Irregularities in stepsize found")
        else:
            if 'steps' in kwargs:
                if steps != traj_list[1] - traj_list[0]:
                    raise Exception("steps and the found stepsize are not the same")
            else:
                steps = traj_list[1] - traj_list[0]

        configlist.append([tr // steps // dtr_cnfg for tr in traj_list])
        if configlist[-1][0] > 1:
            offset = configlist[-1][0] - 1
            warnings.warn('Assume thermalization and that the first measurement belongs to the first config. Offset = %d configs (%d trajectories / cycles)' % (
                offset, offset * steps))
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

        if version in ['sfqcd']:
            cstepsize = cmax / ncs
            index_aim = round(c / cstepsize)
        else:
            t_aim = (c * L) ** 2 / 8
            index_aim = round(t_aim / eps / dn)

        Q_sum = []
        for i, item in enumerate(Q):
            if sum_t is True:
                Q_sum.append([sum(item[current:current + tmax])
                             for current in range(0, len(item), tmax)])
            else:
                Q_sum.append([item[int(tmax / 2)]])
        Q_top = []
        if version in ['sfqcd']:
            for i in range(len(Q_sum) // (ncs + 1)):
                Q_top.append(Q_sum[i * (ncs + 1) + index_aim][0])
        else:
            for i in range(len(Q) // dtr_cnfg):
                Q_top.append(Q_sum[dtr_cnfg * i][index_aim])
        if len(Q_top) != len(traj_list) // dtr_cnfg:
            raise Exception("qtops and traj_list dont have the same length")

        if kwargs.get('integer_charge', False):
            Q_top = [round(q) for q in Q_top]

        truncated_file = file[:-len(postfix)]

        if "names" not in kwargs:
            try:
                idx = truncated_file.index('r')
            except Exception:
                if "names" not in kwargs:
                    raise Exception("Automatic recognition of replicum failed, please enter the key word 'names'.")
            ens_name = truncated_file[:idx]
            rep_names.append(ens_name + '|' + truncated_file[idx:])
        else:
            names = kwargs.get("names")
            rep_names = names
        deltas.append(Q_top)

    idl = [range(int(configlist[rep][r_start_index[rep]]), int(configlist[rep][r_stop_index[rep]]) + 1, 1) for rep in range(len(deltas))]
    deltas = [deltas[nrep][r_start_index[nrep]:r_stop_index[nrep] + 1] for nrep in range(len(deltas))]
    result = Obs(deltas, rep_names, idl=idl)
    result.tag = {"T": tmax - 1,
                  "L": L}
    return result


def qtop_projection(qtop, target=0):
    """Returns the projection to the topological charge sector defined by target.

    Parameters
    ----------
    path : Obs
        Topological charge.
    target : int
        Specifies the topological sector to be reweighted to (default 0)
    """
    if qtop.reweighted:
        raise Exception('You can not use a reweighted observable for reweighting!')

    proj_qtop = []
    for n in qtop.deltas:
        proj_qtop.append(np.array([1 if round(qtop.value + q) == target else 0 for q in qtop.deltas[n]]))

    reto = Obs(proj_qtop, qtop.names, idl=[qtop.idl[name] for name in qtop.names])
    reto.is_merged = qtop.is_merged
    return reto


def read_qtop_sector(path, prefix, c, target=0, **kwargs):
    """Constructs reweighting factors to a specified topological sector.

    Parameters
    ----------
    path : str
        path of the measurement files
    prefix : str
        prefix of the measurement files, e.g. <prefix>_id0_r0.ms.dat
    c : double
        Smearing radius in units of the lattice extent, c = sqrt(8 t0) / L
    target : int
        Specifies the topological sector to be reweighted to (default 0)
    dtr_cnfg : int
        (optional) parameter that specifies the number of trajectories
        between two configs.
        if it is not set, the distance between two measurements
        in the file is assumed to be the distance between two configurations.
    steps : int
        (optional) Distance between two configurations in units of trajectories /
         cycles. Assumed to be the distance between two measurements * dtr_cnfg if not given
    version : str
        version string of the openQCD (sfqcd) version used to create
        the ensemble. Default is 2.0. May also be set to sfqcd.
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
    Zeuthen_flow : bool
        (optional) If True, the Zeuthen flow is used for Qtop. Only possible
        for version=='sfqcd' If False, the Wilson flow is used.
    """

    if not isinstance(target, int):
        raise Exception("'target' has to be an integer.")

    kwargs['integer_charge'] = True
    qtop = read_qtop(path, prefix, c, **kwargs)

    return qtop_projection(qtop, target=target)
