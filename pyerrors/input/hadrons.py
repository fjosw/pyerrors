#!/usr/bin/env python
# coding: utf-8

import os
import h5py
import numpy as np
from ..obs import Obs, CObs
from ..correlators import Corr
from ..npr import Npr_matrix


def _get_files(path, filestem):
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

    return files, cnfg_numbers


def read_meson_hd5(path, filestem, ens_id, meson='meson_0', tree='meson'):
    """Read hadrons meson hdf5 file and extract the meson labeled 'meson'

    Parameters
    -----------------
    path : str
        path to the files to read
    filestem : str
        namestem of the files to read
    ens_id : str
        name of the ensemble, required for internal bookkeeping
    meson : str
        label of the meson to be extracted, standard value meson_0 which
        corresponds to the pseudoscalar pseudoscalar two-point function.
    tree : str
        Label of the upmost directory in the hdf5 file, default 'meson'
        for outputs of the Meson module. Can be altered to read input
        from other modules with similar structures.
    """

    files, cnfg_numbers = _get_files(path, filestem)

    corr_data = []
    infos = []
    for hd5_file in files:
        file = h5py.File(path + '/' + hd5_file, "r")
        raw_data = list(file[tree + '/' + meson + '/corr'])
        real_data = [o[0] for o in raw_data]
        corr_data.append(real_data)
        if not infos:
            for k, i in file[tree + '/' + meson].attrs.items():
                infos.append(k + ': ' + i[0].decode())
        file.close()
    corr_data = np.array(corr_data)

    l_obs = []
    for c in corr_data.T:
        l_obs.append(Obs([c], [ens_id], idl=[cnfg_numbers]))

    corr = Corr(l_obs)
    corr.tag = r", ".join(infos)
    return corr


def read_ExternalLeg_hd5(path, filestem, ens_id, order='F'):
    """Read hadrons ExternalLeg hdf5 file and output an array of CObs

    Parameters
    -----------------
    path -- path to the files to read
    filestem -- namestem of the files to read
    ens_id -- name of the ensemble, required for internal bookkeeping
    order -- order in which the array is to be reshaped,
             'F' for the first index changing fastest (9 4x4 matrices) default.
             'C' for the last index changing fastest (16 3x3 matrices),
    """

    files, cnfg_numbers = _get_files(path, filestem)

    mom = None

    corr_data = []
    for hd5_file in files:
        file = h5py.File(path + '/' + hd5_file, "r")
        raw_data = file['ExternalLeg/corr'][0][0].view('complex')
        corr_data.append(raw_data)
        if mom is None:
            mom = np.array(str(file['ExternalLeg/info'].attrs['pIn'])[3:-2].strip().split(' '), dtype=int)
        file.close()
    corr_data = np.array(corr_data)

    rolled_array = np.rollaxis(corr_data, 0, 5)

    matrix = np.empty((rolled_array.shape[:-1]), dtype=object)
    for si, sj, ci, cj in np.ndindex(rolled_array.shape[:-1]):
        real = Obs([rolled_array[si, sj, ci, cj].real], [ens_id], idl=[cnfg_numbers])
        imag = Obs([rolled_array[si, sj, ci, cj].imag], [ens_id], idl=[cnfg_numbers])
        matrix[si, sj, ci, cj] = CObs(real, imag)

    return Npr_matrix(matrix.swapaxes(1, 2).reshape((12, 12), order=order), mom_in=mom)


def read_Bilinear_hd5(path, filestem, ens_id, order='F'):
    """Read hadrons Bilinear hdf5 file and output an array of CObs

    Parameters
    -----------------
    path -- path to the files to read
    filestem -- namestem of the files to read
    ens_id -- name of the ensemble, required for internal bookkeeping
    order -- order in which the array is to be reshaped,
             'F' for the first index changing fastest (9 4x4 matrices) default.
             'C' for the last index changing fastest (16 3x3 matrices),
    """

    files, cnfg_numbers = _get_files(path, filestem)

    mom_in = None
    mom_out = None

    corr_data = {}
    for hd5_file in files:
        file = h5py.File(path + '/' + hd5_file, "r")
        for i in range(16):
            name = file['Bilinear/Bilinear_' + str(i) + '/info'].attrs['gamma'][0].decode('UTF-8')
            if name not in corr_data:
                corr_data[name] = []
            raw_data = file['Bilinear/Bilinear_' + str(i) + '/corr'][0][0].view('complex')
            corr_data[name].append(raw_data)
            if mom_in is None:
                mom_in = np.array(str(file['Bilinear/Bilinear_' + str(i) + '/info'].attrs['pIn'])[3:-2].strip().split(' '), dtype=int)
            if mom_out is None:
                mom_out = np.array(str(file['Bilinear/Bilinear_' + str(i) + '/info'].attrs['pOut'])[3:-2].strip().split(' '), dtype=int)

        file.close()

    result_dict = {}

    for key, data in corr_data.items():
        local_data = np.array(data)

        rolled_array = np.rollaxis(local_data, 0, 5)

        matrix = np.empty((rolled_array.shape[:-1]), dtype=object)
        for si, sj, ci, cj in np.ndindex(rolled_array.shape[:-1]):
            real = Obs([rolled_array[si, sj, ci, cj].real], [ens_id], idl=[cnfg_numbers])
            imag = Obs([rolled_array[si, sj, ci, cj].imag], [ens_id], idl=[cnfg_numbers])
            matrix[si, sj, ci, cj] = CObs(real, imag)

        result_dict[key] = Npr_matrix(matrix.swapaxes(1, 2).reshape((12, 12), order=order), mom_in=mom_in, mom_out=mom_out)

    return result_dict
