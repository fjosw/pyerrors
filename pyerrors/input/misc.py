import os
import fnmatch
import re
import struct
import numpy as np  # Thinly-wrapped numpy
from ..obs import Obs


def read_pbp(path, prefix, **kwargs):
    """Read pbp format from given folder structure. Returns a list of length nrw

    Parameters
    ----------
    r_start : list
        list which contains the first config to be read for each replicum
    r_stop : list
        list which contains the last config to be read for each replicum
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
