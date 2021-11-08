#!/usr/bin/env python
# coding: utf-8

import os
import fnmatch
import re
import numpy as np  # Thinly-wrapped numpy
from ..obs import Obs


def read_sfcf(path, prefix, name, **kwargs):
    """Read sfcf C format from given folder structure.

    Parameters
    ----------
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
        raise Exception('Error, directory not found')
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
        # Adjust replica names to new bookmarking system
        new_names = []
        for entry in ls:
            idx = entry.index('r')
            new_names.append(entry[:idx] + '|' + entry[idx:])

    print(replica, 'replica')
    for i, item in enumerate(ls):
        print(item)
        sub_ls = []
        for (dirpath, dirnames, filenames) in os.walk(path + '/' + item):
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
            with open(path + '/' + item + '/' + subitem + '/' + name) as fp:
                for k, line in enumerate(fp):
                    if(k >= start and k < start + T):
                        floats = list(map(float, line.split()))
                        deltas[k - start][i][cnfg] = floats[1 + im - single]

    result = []
    for t in range(T):
        result.append(Obs(deltas[t], new_names))

    return result


def read_sfcf_c(path, prefix, name, quarks='.*', noffset=0, wf=0, wf2=0, **kwargs):
    """Read sfcf c format from given folder structure.

    Parameters
    ----------
    quarks -- Label of the quarks used in the sfcf input file
    noffset -- Offset of the source (only relevant when wavefunctions are used)
    wf -- ID of wave function
    wf2 -- ID of the second wavefunction (only relevant for boundary-to-boundary correlation functions)
    im -- if True, read imaginary instead of real part of the correlation function.
    b2b -- if True, read a time-dependent boundary-to-boundary correlation function
    names -- Alternative labeling for replicas/ensembles. Has to have the appropriate length
    ens_name : str
        replaces the name of the ensemble
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

    T = 0
    ls = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        ls.extend(dirnames)
        break
    if not ls:
        raise Exception('Error, directory not found')
    # Exclude folders with different names
    for exc in ls:
        if not fnmatch.fnmatch(exc, prefix + '*'):
            ls = list(set(ls) - set([exc]))
    if len(ls) > 1:
        ls.sort(key=lambda x: int(re.findall(r'\d+', x[len(prefix):])[0]))  # New version, to cope with ids, etc.
    replica = len(ls)
    if 'names' in kwargs:
        new_names = kwargs.get('names')
        if len(new_names) != replica:
            raise Exception('Names does not have the required length', replica)
    else:
        # Adjust replica names to new bookmarking system
        new_names = []
        for entry in ls:
            idx = entry.index('r')
            if 'ens_name' in kwargs:
                new_names.append(kwargs.get('ens_name') + '|' + entry[idx:])
            else:
                new_names.append(entry[:idx] + '|' + entry[idx:])

    print('Read', part, 'part of', name, 'from', prefix[:-1], ',', replica, 'replica')
    for i, item in enumerate(ls):
        sub_ls = []
        for (dirpath, dirnames, filenames) in os.walk(path + '/' + item):
            sub_ls.extend(filenames)
            break
        for exc in sub_ls:
            if not fnmatch.fnmatch(exc, prefix + '*'):
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

            with open(path + '/' + item + '/' + sub_ls[0], 'r') as file:
                content = file.read()
                match = re.search(pattern, content)
                if match:
                    start_read = content.count('\n', 0, match.start()) + 5 + b2b
                    end_match = re.search(r'\n\s*\n', content[match.start():])
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
            with open(path + '/' + item + '/' + sub_ls[cfg]) as fp:
                for k, line in enumerate(fp):
                    if k == start_read - 5 - b2b:
                        if line.strip() != 'name      ' + name:
                            raise Exception('Wrong format', sub_ls[cfg])
                    if(k >= start_read and k < start_read + T):
                        floats = list(map(float, line.split()))
                        deltas[k - start_read][i][cfg] = floats[-2:][im]

    result = []
    for t in range(T):
        result.append(Obs(deltas[t], new_names))
    return result


def read_qtop(path, prefix, **kwargs):
    """Read qtop format from given folder structure.

    Parameters
    ----------
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
        raise Exception('Error, directory not found')

    # Exclude files with different names
    for exc in ls:
        if not fnmatch.fnmatch(exc, prefix + '*'):
            ls = list(set(ls) - set([exc]))
    if len(ls) > 1:
        ls.sort(key=lambda x: int(re.findall(r'\d+', x[len(prefix):])[0]))  # New version, to cope with ids, etc.
    replica = len(ls)
    print('Read Q_top from', prefix[:-1], ',', replica, 'replica')

    deltas = []

    for rep in range(replica):
        tmp = []
        with open(path + '/' + ls[rep]) as fp:
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

    rep_names = []
    for entry in ls:
        truncated_entry = entry.split('.')[0]
        idx = truncated_entry.index('r')
        rep_names.append(truncated_entry[:idx] + '|' + truncated_entry[idx:])

    result = Obs(deltas, rep_names)

    return result
