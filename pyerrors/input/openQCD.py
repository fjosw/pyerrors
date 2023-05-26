import os
import fnmatch
import struct
import warnings
import numpy as np  # Thinly-wrapped numpy
from ..obs import Obs
from ..obs import CObs
from ..correlators import Corr
from .misc import fit_t0
from .utils import sort_names


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

    Returns
    -------
    rwms : Obs
        Reweighting factors read
    """
    known_oqcd_versions = ['1.4', '1.6', '2.0']
    if not (version in known_oqcd_versions):
        raise Exception('Unknown openQCD version defined!')
    print("Working with openQCD version " + version)
    if 'postfix' in kwargs:
        postfix = kwargs.get('postfix')
    else:
        postfix = ''

    if 'files' in kwargs:
        known_files = kwargs.get('files')
    else:
        known_files = []

    ls = _find_files(path, prefix, postfix, 'dat', known_files=known_files)

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

    rep_names = sort_names(rep_names)

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
                    raise Exception("You are using the input for openQCD version 2.0, this is not correct.")

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


def _extract_flowed_energy_density(path, prefix, dtr_read, xmin, spatial_extent, postfix='ms', **kwargs):
    """Extract a dictionary with the flowed Yang-Mills action density from given .ms.dat files.
    Returns a dictionary with Obs as values and flow times as keys.

    It is assumed that all boundary effects have
    sufficiently decayed at x0=xmin.

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
    postfix : str
        Postfix of measurement file (Default: ms)
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
    assume_thermalization : bool
        If True: If the first record divided by the distance between two measurements is larger than
        1, it is assumed that this is due to thermalization and the first measurement belongs
        to the first config (default).
        If False: The config numbers are assumed to be traj_number // difference

    Returns
    -------
    E_dict : dictionary
        Dictionary with the flowed action density at flow times t
    """

    if 'files' in kwargs:
        known_files = kwargs.get('files')
    else:
        known_files = []

    ls = _find_files(path, prefix, postfix, 'dat', known_files=known_files)

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

    print('Extract flowed Yang-Mills action density from', prefix, ',', replica, 'replica')

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
    E_dict = {}
    for n in range(nn + 1):
        samples = []
        for nrep, rep in enumerate(Ysum):
            samples.append([])
            for cnfg in rep:
                samples[-1].append(cnfg[n])
            samples[-1] = samples[-1][r_start_index[nrep]:r_stop_index[nrep] + 1][::r_step]
        new_obs = Obs(samples, rep_names, idl=idl)
        E_dict[n * dn * eps] = new_obs / (spatial_extent ** 3)

    return E_dict


def extract_t0(path, prefix, dtr_read, xmin, spatial_extent, fit_range=5, postfix='ms', c=0.3, **kwargs):
    """Extract t0/a^2 from given .ms.dat files. Returns t0 as Obs.

    It is assumed that all boundary effects have
    sufficiently decayed at x0=xmin.
    The data around the zero crossing of t^2<E> - c (where c=0.3 by default)
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
    postfix : str
        Postfix of measurement file (Default: ms)
    c: float
        Constant that defines the flow scale. Default 0.3 for t_0, choose 2./3 for t_1.
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

    Returns
    -------
    t0 : Obs
        Extracted t0
    """

    E_dict = _extract_flowed_energy_density(path, prefix, dtr_read, xmin, spatial_extent, postfix, **kwargs)
    t2E_dict = {}
    for t in sorted(E_dict.keys()):
        t2E_dict[t] = t ** 2 * E_dict[t] - c

    return fit_t0(t2E_dict, fit_range, plot_fit=kwargs.get('plot_fit'))


def extract_w0(path, prefix, dtr_read, xmin, spatial_extent, fit_range=5, postfix='ms', c=0.3, **kwargs):
    """Extract w0/a from given .ms.dat files. Returns w0 as Obs.

    It is assumed that all boundary effects have
    sufficiently decayed at x0=xmin.
    The data around the zero crossing of t d(t^2<E>)/dt -  (where c=0.3 by default)
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
    postfix : str
        Postfix of measurement file (Default: ms)
    c: float
        Constant that defines the flow scale. Default 0.3 for w_0, choose 2./3 for w_1.
    r_start : list
        list which contains the first config to be read for each replicum.
    r_stop : list
        list which contains the last config to be read for each replicum.
    r_step : int
        integer that defines a fixed step size between two measurements (in units of configs)
        If not given, r_step=1 is assumed.
    plaquette : bool
        If true extract the plaquette estimate of w0 instead.
    names : list
        list of names that is assigned to the data according according
        to the order in the file list. Use careful, if you do not provide file names!
    files : list
        list which contains the filenames to be read. No automatic detection of
        files performed if given.
    plot_fit : bool
        If true, the fit for the extraction of w0 is shown together with the data.
    assume_thermalization : bool
        If True: If the first record divided by the distance between two measurements is larger than
        1, it is assumed that this is due to thermalization and the first measurement belongs
        to the first config (default).
        If False: The config numbers are assumed to be traj_number // difference

    Returns
    -------
    w0 : Obs
        Extracted w0
    """

    E_dict = _extract_flowed_energy_density(path, prefix, dtr_read, xmin, spatial_extent, postfix, **kwargs)

    ftimes = sorted(E_dict.keys())

    t2E_dict = {}
    for t in ftimes:
        t2E_dict[t] = t ** 2 * E_dict[t]

    tdtt2E_dict = {}
    tdtt2E_dict[ftimes[0]] = ftimes[0] * (t2E_dict[ftimes[1]] - t2E_dict[ftimes[0]]) / (ftimes[1] - ftimes[0]) - c
    for i in range(1, len(ftimes) - 1):
        tdtt2E_dict[ftimes[i]] = ftimes[i] * (t2E_dict[ftimes[i + 1]] - t2E_dict[ftimes[i - 1]]) / (ftimes[i + 1] - ftimes[i - 1]) - c
    tdtt2E_dict[ftimes[-1]] = ftimes[-1] * (t2E_dict[ftimes[-1]] - t2E_dict[ftimes[-2]]) / (ftimes[-1] - ftimes[-2]) - c

    return np.sqrt(fit_t0(tdtt2E_dict, fit_range, plot_fit=kwargs.get('plot_fit'), observable='w0'))


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


def _find_files(path, prefix, postfix, ext, known_files=[]):
    found = []
    files = []

    if postfix != "":
        if postfix[-1] != ".":
            postfix = postfix + "."
        if postfix[0] != ".":
            postfix = "." + postfix

    if ext[0] == ".":
        ext = ext[1:]

    pattern = prefix + "*" + postfix + ext

    for (dirpath, dirnames, filenames) in os.walk(path + "/"):
        found.extend(filenames)
        break

    if known_files != []:
        for kf in known_files:
            if kf not in found:
                raise FileNotFoundError("Given file " + kf + " does not exist!")

        return known_files

    if not found:
        raise FileNotFoundError(f"Error, directory '{path}' not found")

    for f in found:
        if fnmatch.fnmatch(f, pattern):
            files.append(f)

    if files == []:
        raise Exception("No files found after pattern filter!")

    files = sort_names(files)
    return files


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
    postfix : str
        postfix of the file to read, e.g. '.gfms.dat' for openQCD-files
    names : list
        Alternative labeling for replicas/ensembles.
        Has to have the appropriate length.
    Zeuthen_flow : bool
        (optional) If True, the Zeuthen flow is used for Qtop. Only possible
        for version=='sfqcd' If False, the Wilson flow is used.
    integer_charge : bool
        If True, the charge is rounded towards the nearest integer on each config.

    Returns
    -------
    result : Obs
        Read topological charge
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
    postfix : str
        postfix of the file to read, e.g. '.gfms.dat' for openQCD-files
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
    postfix : str
        postfix of the file to read, e.g. '.gfms.dat' for openQCD-files
    Zeuthen_flow : bool
        (optional) If True, the Zeuthen flow is used for Qtop. Only possible
        for version=='sfqcd' If False, the Wilson flow is used.
    integer_charge : bool
        If True, the charge is rounded towards the nearest integer on each config.

    Returns
    -------
    result : Obs
        flow observable specified
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
        postfix = "gfms"
    else:
        if "L" not in kwargs:
            raise Exception("This version of openQCD needs you to provide the spatial length of the lattice as parameter 'L'.")
        else:
            L = kwargs.get("L")
        postfix = "ms"

    if "postfix" in kwargs:
        postfix = kwargs.get("postfix")

    if "files" in kwargs:
        known_files = kwargs.get("files")
    else:
        known_files = []

    files = _find_files(path, prefix, postfix, "dat", known_files=known_files)

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
            rep_names.append(ens_name + '|' + truncated_file[idx:].split(".")[0])
        else:
            names = kwargs.get("names")
            rep_names = names

        deltas.append(Q_top)

    rep_names = sort_names(rep_names)

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

    Returns
    -------
    reto : Obs
        projection to the topological charge sector defined by target
    """
    if qtop.reweighted:
        raise Exception('You can not use a reweighted observable for reweighting!')

    proj_qtop = []
    for n in qtop.deltas:
        proj_qtop.append(np.array([1 if round(qtop.r_values[n] + q) == target else 0 for q in qtop.deltas[n]]))

    reto = Obs(proj_qtop, qtop.names, idl=[qtop.idl[name] for name in qtop.names])
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

    Returns
    -------
    reto : Obs
        projection to the topological charge sector defined by target
    """

    if not isinstance(target, int):
        raise Exception("'target' has to be an integer.")

    kwargs['integer_charge'] = True
    qtop = read_qtop(path, prefix, c, **kwargs)

    return qtop_projection(qtop, target=target)


def read_ms5_xsf(path, prefix, qc, corr, sep="r", **kwargs):
    """
    Read data from files in the specified directory with the specified prefix and quark combination extension, and return a `Corr` object containing the data.

    Parameters
    ----------
    path : str
        The directory to search for the files in.
    prefix : str
        The prefix to match the files against.
    qc : str
        The quark combination extension to match the files against.
    corr : str
        The correlator to extract data for.
    sep : str, optional
        The separator to use when parsing the replika names.
    **kwargs
        Additional keyword arguments. The following keyword arguments are recognized:

        - names (List[str]): A list of names to use for the replicas.
        - files (List[str]): A list of files to read data from.
        - idl (List[List[int]]): A list of idls per replicum, resticting data to the idls given.

    Returns
    -------
    Corr
        A complex valued `Corr` object containing the data read from the files. In case of boudary to bulk correlators.
    or
    CObs
        A complex valued `CObs` object containing the data read from the files. In case of boudary to boundary correlators.


    Raises
    ------
    FileNotFoundError
        If no files matching the specified prefix and quark combination extension are found in the specified directory.
    IOError
        If there is an error reading a file.
    struct.error
        If there is an error unpacking binary data.
    """

    # found = []
    files = []
    names = []

    # test if the input is correct
    if qc not in ['dd', 'ud', 'du', 'uu']:
        raise Exception("Unknown quark conbination!")

    if corr not in ["gS", "gP", "gA", "gV", "gVt", "lA", "lV", "lVt", "lT", "lTt", "g1", "l1"]:
        raise Exception("Unknown correlator!")

    if "files" in kwargs:
        known_files = kwargs.get("files")
    else:
        known_files = []
    files = _find_files(path, prefix, "ms5_xsf_" + qc, "dat", known_files=known_files)

    if "names" in kwargs:
        names = kwargs.get("names")
    else:
        for f in files:
            if not sep == "":
                se = f.split(".")[0]
                for s in f.split(".")[1:-2]:
                    se += "." + s
                names.append(se.split(sep)[0] + "|r" + se.split(sep)[1])
            else:
                names.append(prefix)
    if 'idl' in kwargs:
        expected_idl = kwargs.get('idl')
    names = sorted(names)
    files = sorted(files)

    cnfgs = []
    realsamples = []
    imagsamples = []
    repnum = 0
    for file in files:
        with open(path + "/" + file, "rb") as fp:

            t = fp.read(8)
            kappa = struct.unpack('d', t)[0]
            t = fp.read(8)
            csw = struct.unpack('d', t)[0]
            t = fp.read(8)
            dF = struct.unpack('d', t)[0]
            t = fp.read(8)
            zF = struct.unpack('d', t)[0]

            t = fp.read(4)
            tmax = struct.unpack('i', t)[0]
            t = fp.read(4)
            bnd = struct.unpack('i', t)[0]

            placesBI = ["gS", "gP",
                        "gA", "gV",
                        "gVt", "lA",
                        "lV", "lVt",
                        "lT", "lTt"]
            placesBB = ["g1", "l1"]

            # the chunks have the following structure:
            # confignumber, 10x timedependent complex correlators as doubles, 2x timeindependent complex correlators as doubles

            chunksize = 4 + (8 * 2 * tmax * 10) + (8 * 2 * 2)
            packstr = '=i' + ('d' * 2 * tmax * 10) + ('d' * 2 * 2)
            cnfgs.append([])
            realsamples.append([])
            imagsamples.append([])
            for t in range(tmax):
                realsamples[repnum].append([])
                imagsamples[repnum].append([])
            if 'idl' in kwargs:
                left_idl = set(expected_idl[repnum])
            while True:
                cnfgt = fp.read(chunksize)
                if not cnfgt:
                    break
                asascii = struct.unpack(packstr, cnfgt)
                cnfg = asascii[0]
                idl_wanted = True
                if 'idl' in kwargs:
                    idl_wanted = (cnfg in expected_idl[repnum])
                    left_idl = left_idl - set([cnfg])
                if idl_wanted:
                    cnfgs[repnum].append(cnfg)

                    if corr not in placesBB:
                        tmpcorr = asascii[1 + 2 * tmax * placesBI.index(corr):1 + 2 * tmax * placesBI.index(corr) + 2 * tmax]
                    else:
                        tmpcorr = asascii[1 + 2 * tmax * len(placesBI) + 2 * placesBB.index(corr):1 + 2 * tmax * len(placesBI) + 2 * placesBB.index(corr) + 2]

                    corrres = [[], []]
                    for i in range(len(tmpcorr)):
                        corrres[i % 2].append(tmpcorr[i])
                    for t in range(int(len(tmpcorr) / 2)):
                        realsamples[repnum][t].append(corrres[0][t])
                    for t in range(int(len(tmpcorr) / 2)):
                        imagsamples[repnum][t].append(corrres[1][t])
            if 'idl' in kwargs:
                left_idl = list(left_idl)
                if len(left_idl) > 0:
                    warnings.warn('Could not find idls ' + str(left_idl) + ' in replikum of file ' + file, UserWarning)
        repnum += 1
    s = "Read correlator " + corr + " from " + str(repnum) + " replika with idls" + str(realsamples[0][t])
    for rep in range(1, repnum):
        s += ", " + str(realsamples[rep][t])
    print(s)
    print("Asserted run parameters:\n T:", tmax, "kappa:", kappa, "csw:", csw, "dF:", dF, "zF:", zF, "bnd:", bnd)

    # we have the data now... but we need to re format the whole thing and put it into Corr objects.

    compObs = []

    for t in range(int(len(tmpcorr) / 2)):
        compObs.append(CObs(Obs([realsamples[rep][t] for rep in range(repnum)], names=names, idl=cnfgs),
                            Obs([imagsamples[rep][t] for rep in range(repnum)], names=names, idl=cnfgs)))

    if len(compObs) == 1:
        return compObs[0]
    else:
        return Corr(compObs)
