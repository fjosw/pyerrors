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


def fit_t0(t2E_dict, fit_range, plot_fit=False, observable='t0'):
    """Compute the root of (flow-based) data based on a dictionary that contains
    the necessary information in key-value pairs a la (flow time: observable at flow time).

    It is assumed that the data is monotonically increasing and passes zero from below.
    No exception is thrown if this is not the case (several roots, no monotonic increase).
    An exception is thrown if no root can be found in the data.

    A linear fit in the vicinity of the root is performed to exctract the root from the
    two fit parameters.

    Parameters
    ----------
    t2E_dict : dict
        Dictionary with pairs of (flow time: observable at flow time) where the flow times
        are of type float and the observables of type Obs.
    fit_range : int
        Number of data points left and right of the zero
        crossing to be included in the linear fit.
    plot_fit : bool
        If true, the fit for the extraction of t0 is shown together with the data. (Default: False)
    observable: str
        Keyword to identify the observable to print the correct ylabel (if plot_fit is True)
        for the observables 't0' and 'w0'. No y label is printed otherwise. (Default: 't0')

    Returns
    -------
    root : Obs
        The root of the data series.
    """

    zero_crossing = np.argmax(np.array(
        [o.value for o in t2E_dict.values()]) > 0.0)

    if zero_crossing == 0:
        raise Exception('Desired flow time not in data')

    x = list(t2E_dict.keys())[zero_crossing - fit_range:
                              zero_crossing + fit_range]
    y = list(t2E_dict.values())[zero_crossing - fit_range:
                                zero_crossing + fit_range]
    [o.gamma_method() for o in y]

    if len(x) < 2 * fit_range:
        warnings.warn('Fit range smaller than expected! Fitting from %1.2e to %1.2e' % (x[0], x[-1]))

    fit_result = fit_lin(x, y)

    if plot_fit is True:
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
        if observable == 't0':
            ax0.set_ylabel(r'$t^2 \langle E(t) \rangle - 0.3 $')
        elif observable == 'w0':
            ax0.set_ylabel(r'$t d(t^2 \langle E(t) \rangle)/dt - 0.3 $')
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


def read_pbp(path, prefix, **kwargs):
    """Read pbp format from given folder structure.

    Parameters
    ----------
    r_start : list
        list which contains the first config to be read for each replicum
    r_stop : list
        list which contains the last config to be read for each replicum

    Returns
    -------
    result : list[Obs]
        list of observables read
    """

    ls = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        ls.extend(filenames)
        break

    if not ls:
        raise Exception('Error, directory not found')

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

    print(r'Read <bar{psi}\psi> from', prefix[:-1], ',', replica, 'replica', end='')

    print_err = 0
    if 'print_err' in kwargs:
        print_err = 1
        print()

    deltas = []

    for rep in range(replica):
        tmp_array = []
        with open(path + '/' + ls[rep], 'rb') as fp:

            t = fp.read(4)  # number of reweighting factors
            if rep == 0:
                nrw = struct.unpack('i', t)[0]
                for k in range(nrw):
                    deltas.append([])
            else:
                if nrw != struct.unpack('i', t)[0]:
                    raise Exception('Error: different number of factors for replicum', rep)

            for k in range(nrw):
                tmp_array.append([])

            # This block is necessary for openQCD1.6 ms1 files
            nfct = []
            for i in range(nrw):
                t = fp.read(4)
                nfct.append(struct.unpack('i', t)[0])
            print('nfct: ', nfct)  # Hasenbusch factor, 1 for rat reweighting

            nsrc = []
            for i in range(nrw):
                t = fp.read(4)
                nsrc.append(struct.unpack('i', t)[0])

            # body
            while True:
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

    rep_names = []
    for entry in ls:
        truncated_entry = entry.split('.')[0]
        idx = truncated_entry.index('r')
        rep_names.append(truncated_entry[:idx] + '|' + truncated_entry[idx:])
    print(',', nrw, r'<bar{psi}\psi> with', nsrc, 'sources')
    result = []
    for t in range(nrw):
        result.append(Obs(deltas[t], rep_names))

    return result
