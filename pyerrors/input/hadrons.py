import os
import warnings
from collections import Counter
import h5py
import numpy as np
from ..obs import Obs, CObs
from ..correlators import Corr
from ..dirac import epsilon_tensor_rank4


def _get_files(path, filestem, idl):
    ls = os.listdir(path)

    # Clean up file list
    files = list(filter(lambda x: x.startswith(filestem), ls))

    if not files:
        raise Exception('No files starting with', filestem, 'in folder', path)

    def get_cnfg_number(n):
        return int(n[len(filestem) + 1:-3])

    # Sort according to configuration number
    files.sort(key=get_cnfg_number)

    cnfg_numbers = []
    filtered_files = []
    for line in files:
        no = get_cnfg_number(line)
        if idl:
            if no in list(idl):
                filtered_files.append(line)
                cnfg_numbers.append(no)
        else:
            filtered_files.append(line)
            cnfg_numbers.append(no)

    if idl:
        if Counter(list(idl)) != Counter(cnfg_numbers):
            raise Exception("Not all configurations specified in idl found, configurations " + str(list(Counter(list(idl)) - Counter(cnfg_numbers))) + " are missing.")

    # Check that configurations are evenly spaced
    dc = np.unique(np.diff(cnfg_numbers))
    if np.any(dc < 0):
        raise Exception("Unsorted files")
    if len(dc) == 1:
        idx = range(cnfg_numbers[0], cnfg_numbers[-1] + dc[0], dc[0])
    elif idl:
        idx = idl
        warnings.warn("Configurations are not evenly spaced.", RuntimeWarning)
    else:
        raise Exception("Configurations are not evenly spaced.")

    return filtered_files, idx


def read_meson_hd5(path, filestem, ens_id, meson='meson_0', idl=None):
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
    idl : range
        If specified only configurations in the given range are read in.
    """

    files, idx = _get_files(path, filestem, idl)

    tree = meson.rsplit('_')[0]
    corr_data = []
    infos = []
    for hd5_file in files:
        h5file = h5py.File(path + '/' + hd5_file, "r")
        if not tree + '/' + meson in h5file:
            raise Exception("Entry '" + meson + "' not contained in the files.")
        raw_data = list(h5file[tree + '/' + meson + '/corr'])
        real_data = [o[0] for o in raw_data]
        corr_data.append(real_data)
        if not infos:
            for k, i in h5file[tree + '/' + meson].attrs.items():
                infos.append(k + ': ' + i[0].decode())
        h5file.close()
    corr_data = np.array(corr_data)

    l_obs = []
    for c in corr_data.T:
        l_obs.append(Obs([c], [ens_id], idl=[idx]))

    corr = Corr(l_obs)
    corr.tag = r", ".join(infos)
    return corr


class Npr_matrix(np.ndarray):

    def __new__(cls, input_array, mom_in=None, mom_out=None):
        obj = np.asarray(input_array).view(cls)
        obj.mom_in = mom_in
        obj.mom_out = mom_out
        return obj

    @property
    def g5H(self):
        """Gamma_5 hermitean conjugate

        Uses the fact that the propagator is gamma5 hermitean, so just the
        in and out momenta of the propagator are exchanged.
        """
        return Npr_matrix(self,
                          mom_in=self.mom_out,
                          mom_out=self.mom_in)

    def _propagate_mom(self, other, name):
        s_mom = getattr(self, name, None)
        o_mom = getattr(other, name, None)
        if s_mom is not None and o_mom is not None:
            if not np.allclose(s_mom, o_mom):
                raise Exception(name + ' does not match.')
        return o_mom if o_mom is not None else s_mom

    def __matmul__(self, other):
        return self.__new__(Npr_matrix,
                            super().__matmul__(other),
                            self._propagate_mom(other, 'mom_in'),
                            self._propagate_mom(other, 'mom_out'))

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.mom_in = getattr(obj, 'mom_in', None)
        self.mom_out = getattr(obj, 'mom_out', None)


def read_ExternalLeg_hd5(path, filestem, ens_id, idl=None):
    """Read hadrons ExternalLeg hdf5 file and output an array of CObs

    Parameters
    ----------
    path : str
        path to the files to read
    filestem : str
        namestem of the files to read
    ens_id : str
        name of the ensemble, required for internal bookkeeping
    idl : range
        If specified only configurations in the given range are read in.
    """

    files, idx = _get_files(path, filestem, idl)

    mom = None

    corr_data = []
    for hd5_file in files:
        file = h5py.File(path + '/' + hd5_file, "r")
        raw_data = file['ExternalLeg/corr'][0][0].view('complex')
        corr_data.append(raw_data)
        if mom is None:
            mom = np.array(str(file['ExternalLeg/info'].attrs['pIn'])[3:-2].strip().split(), dtype=float)
        file.close()
    corr_data = np.array(corr_data)

    rolled_array = np.rollaxis(corr_data, 0, 5)

    matrix = np.empty((rolled_array.shape[:-1]), dtype=object)
    for si, sj, ci, cj in np.ndindex(rolled_array.shape[:-1]):
        real = Obs([rolled_array[si, sj, ci, cj].real], [ens_id], idl=[idx])
        imag = Obs([rolled_array[si, sj, ci, cj].imag], [ens_id], idl=[idx])
        matrix[si, sj, ci, cj] = CObs(real, imag)

    return Npr_matrix(matrix, mom_in=mom)


def read_Bilinear_hd5(path, filestem, ens_id, idl=None):
    """Read hadrons Bilinear hdf5 file and output an array of CObs

    Parameters
    ----------
    path : str
        path to the files to read
    filestem : str
        namestem of the files to read
    ens_id : str
        name of the ensemble, required for internal bookkeeping
    idl : range
        If specified only configurations in the given range are read in.
    """

    files, idx = _get_files(path, filestem, idl)

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
                mom_in = np.array(str(file['Bilinear/Bilinear_' + str(i) + '/info'].attrs['pIn'])[3:-2].strip().split(), dtype=float)
            if mom_out is None:
                mom_out = np.array(str(file['Bilinear/Bilinear_' + str(i) + '/info'].attrs['pOut'])[3:-2].strip().split(), dtype=float)

        file.close()

    result_dict = {}

    for key, data in corr_data.items():
        local_data = np.array(data)

        rolled_array = np.rollaxis(local_data, 0, 5)

        matrix = np.empty((rolled_array.shape[:-1]), dtype=object)
        for si, sj, ci, cj in np.ndindex(rolled_array.shape[:-1]):
            real = Obs([rolled_array[si, sj, ci, cj].real], [ens_id], idl=[idx])
            imag = Obs([rolled_array[si, sj, ci, cj].imag], [ens_id], idl=[idx])
            matrix[si, sj, ci, cj] = CObs(real, imag)

        result_dict[key] = Npr_matrix(matrix, mom_in=mom_in, mom_out=mom_out)

    return result_dict


def read_Fourquark_hd5(path, filestem, ens_id, idl=None, vertices=["VA", "AV"]):
    """Read hadrons FourquarkFullyConnected hdf5 file and output an array of CObs

    Parameters
    ----------
    path : str
        path to the files to read
    filestem : str
        namestem of the files to read
    ens_id : str
        name of the ensemble, required for internal bookkeeping
    idl : range
        If specified only configurations in the given range are read in.
    vertices : list
        Vertex functions to be extracted.
    """

    files, idx = _get_files(path, filestem, idl)

    mom_in = None
    mom_out = None

    vertex_names = []
    for vertex in vertices:
        vertex_names += _get_lorentz_names(vertex)

    corr_data = {}

    tree = 'FourQuarkFullyConnected/FourQuarkFullyConnected_'

    for hd5_file in files:
        file = h5py.File(path + '/' + hd5_file, "r")

        for i in range(32):
            name = (file[tree + str(i) + '/info'].attrs['gammaA'][0].decode('UTF-8'), file[tree + str(i) + '/info'].attrs['gammaB'][0].decode('UTF-8'))
            if name in vertex_names:
                if name not in corr_data:
                    corr_data[name] = []
                raw_data = file[tree + str(i) + '/corr'][0][0].view('complex')
                corr_data[name].append(raw_data)
                if mom_in is None:
                    mom_in = np.array(str(file[tree + str(i) + '/info'].attrs['pIn'])[3:-2].strip().split(), dtype=float)
                if mom_out is None:
                    mom_out = np.array(str(file[tree + str(i) + '/info'].attrs['pOut'])[3:-2].strip().split(), dtype=float)

        file.close()

    intermediate_dict = {}

    for vertex in vertices:
        lorentz_names = _get_lorentz_names(vertex)
        for v_name in lorentz_names:
            if v_name in [('SigmaXY', 'SigmaZT'),
                          ('SigmaXT', 'SigmaYZ'),
                          ('SigmaYZ', 'SigmaXT'),
                          ('SigmaZT', 'SigmaXY')]:
                sign = -1
            else:
                sign = 1
            if vertex not in intermediate_dict:
                intermediate_dict[vertex] = sign * np.array(corr_data[v_name])
            else:
                intermediate_dict[vertex] += sign * np.array(corr_data[v_name])

    result_dict = {}

    for key, data in intermediate_dict.items():

        rolled_array = np.moveaxis(data, 0, 8)

        matrix = np.empty((rolled_array.shape[:-1]), dtype=object)
        for index in np.ndindex(rolled_array.shape[:-1]):
            real = Obs([rolled_array[index].real], [ens_id], idl=[idx])
            imag = Obs([rolled_array[index].imag], [ens_id], idl=[idx])
            matrix[index] = CObs(real, imag)

        result_dict[key] = Npr_matrix(matrix, mom_in=mom_in, mom_out=mom_out)

    return result_dict


def _get_lorentz_names(name):
    lorentz_index = ['X', 'Y', 'Z', 'T']

    res = []

    if name == "TT":
        for i in range(4):
            for j in range(i + 1, 4):
                res.append(("Sigma" + lorentz_index[i] + lorentz_index[j], "Sigma" + lorentz_index[i] + lorentz_index[j]))
        return res

    if name == "TTtilde":
        for i in range(4):
            for j in range(i + 1, 4):
                for k in range(4):
                    for o in range(k + 1, 4):
                        fac = epsilon_tensor_rank4(i, j, k, o)
                        if not np.isclose(fac, 0.0):
                            res.append(("Sigma" + lorentz_index[i] + lorentz_index[j], "Sigma" + lorentz_index[k] + lorentz_index[o]))
        return res

    assert len(name) == 2

    if 'S' in name or 'P' in name:
        if not set(name) <= set(['S', 'P']):
            raise Exception("'" + name + "' is not a Lorentz scalar")

        g_names = {'S': 'Identity',
                   'P': 'Gamma5'}

        res.append((g_names[name[0]], g_names[name[1]]))

    else:
        if not set(name) <= set(['V', 'A']):
            raise Exception("'" + name + "' is not a Lorentz scalar")

        for ind in lorentz_index:
            res.append(('Gamma' + ind + (name[0] == 'A') * 'Gamma5',
                        'Gamma' + ind + (name[1] == 'A') * 'Gamma5'))

    return res
