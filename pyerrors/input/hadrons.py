#!/usr/bin/env python
# coding: utf-8

import os
import h5py
import numpy as np
from ..pyerrors import Obs
from ..correlators import Corr


def read_meson_hd5(path, filestem, ens_id, meson='meson_0', tree='meson'):
    """Read hadrons meson hdf5 file and extract the meson labeled 'meson'

    Parameters
    -----------------
    path -- path to the files to read
    filestem -- namestem of the files to read
    ens_id -- name of the ensemble, required for internal bookkeeping
    meson -- label of the meson to be extracted, standard value meson_0 which corresponds to the pseudoscalar pseudoscalar two-point function.
    """
    ls = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        ls.extend(filenames)
        break

    # Clean up file list
    files = []
    for line in ls:
        if line.startswith(filestem):
            files.append(line)

    if not files:
        raise Exception('No files starting with', filestem, 'in folder', path)

    def get_cnfg_number(n):
        return int(n[len(filestem) + 1:-3])

    # Sort according to configuration number
    files.sort(key=get_cnfg_number)

    # Check that configurations are evenly spaced
    cnfg_numbers = []
    for line in files:
        cnfg_numbers.append(get_cnfg_number(line))

    if not all(np.diff(cnfg_numbers) == np.diff(cnfg_numbers)[0]):
        raise Exception('Configurations are not evenly spaced.')

    corr_data = []
    for hd5_file in files:
        file = h5py.File(path + '/' + hd5_file, "r")

        raw_data = list(file[tree + '/' + meson + '/corr'])
        real_data = [o[0] for o in raw_data]
        corr_data.append(real_data)
    corr_data = np.array(corr_data)

    l_obs = []
    for c in corr_data.T:
        l_obs.append(Obs([c], [ens_id]))

    return Corr(l_obs)
