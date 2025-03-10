import os
from collections import Counter
import h5py
from pathlib import Path
import numpy as np
from ..obs import Obs, CObs
from ..correlators import Corr
from ..dirac import epsilon_tensor_rank4
from .misc import fit_t0


def _get_files(path, filestem, idl):
    ls = os.listdir(path)

    # Clean up file list
    files = list(filter(lambda x: x.startswith(filestem + "."), ls))

    if not files:
        raise Exception('No files starting with', filestem, 'in folder', path)

    def get_cnfg_number(n):
        return int(n.replace(".h5", "")[len(filestem) + 1:])  # From python 3.9 onward the safer 'removesuffix' method can be used.

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
    else:
        raise Exception("Configurations are not evenly spaced. Provide an idl if you want to proceed with this set of configurations.")

    return filtered_files, idx


def read_hd5(filestem, ens_id, group, attrs=None, idl=None, part="real"):
    r'''Read hadrons hdf5 file and extract entry based on attributes.

    Parameters
    -----------------
    filestem : str
        Full namestem of the files to read, including the full path.
    ens_id : str
        name of the ensemble, required for internal bookkeeping
    group : str
        label of the group to be extracted.
    attrs : dict or int
        Dictionary containing the attributes. For example
        ```python
        attrs = {"gamma_snk": "Gamma5",
                 "gamma_src": "Gamma5"}
         ```
        Alternatively an integer can be specified to identify the sub group.
        This is discouraged as the order in the file is not guaranteed.
    idl : range
        If specified only configurations in the given range are read in.
    part: str
        string specifying whether to extract the real part ('real'),
        the imaginary part ('imag') or a complex correlator ('complex').
        Default 'real'.

    Returns
    -------
    corr : Corr
        Correlator of the source sink combination in question.
    '''

    path_obj = Path(filestem)
    path = path_obj.parent.as_posix()
    filestem = path_obj.name

    files, idx = _get_files(path, filestem, idl)

    if isinstance(attrs, dict):
        h5file = h5py.File(path + '/' + files[0], "r")
        entry = None
        for key in h5file[group].keys():
            if attrs.items() <= {k: v[0].decode() for k, v in h5file[group][key].attrs.items()}.items():
                if entry is None:
                    entry = key
                else:
                    raise ValueError("More than one fitting entry found. More constraint on attributes needed.")
        h5file.close()
        if entry is None:
            raise ValueError(f"Entry with attributes {attrs} not found.")
    elif isinstance(attrs, int):
        entry = group + f"_{attrs}"
    else:
        raise TypeError("Invalid type for 'attrs'. Needs to be dict or int.")

    corr_data = []
    infos = []
    for hd5_file in files:
        h5file = h5py.File(path + '/' + hd5_file, "r")
        if group + '/' + entry not in h5file:
            raise Exception("Entry '" + entry + "' not contained in the files.")
        raw_data = h5file[group + '/' + entry + '/corr']
        real_data = raw_data[:].view("complex")
        corr_data.append(real_data)
        if not infos:
            for k, i in h5file[group + '/' + entry].attrs.items():
                infos.append(k + ': ' + i[0].decode())
        h5file.close()
    corr_data = np.array(corr_data)

    if part == "complex":
        l_obs = []
        for c in corr_data.T:
            l_obs.append(CObs(Obs([c.real], [ens_id], idl=[idx]),
                              Obs([c.imag], [ens_id], idl=[idx])))
    else:
        corr_data = getattr(corr_data, part)
        l_obs = []
        for c in corr_data.T:
            l_obs.append(Obs([c], [ens_id], idl=[idx]))

    corr = Corr(l_obs)
    corr.tag = r", ".join(infos)
    return corr


def read_meson_hd5(path, filestem, ens_id, meson='meson_0', idl=None, gammas=None):
    r'''Read hadrons meson hdf5 file and extract the meson labeled 'meson'

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
    gammas : tuple of strings
        Instrad of a meson label one can also provide a tuple of two strings
        indicating the gamma matrices at sink and source (gamma_snk, gamma_src).
        ("Gamma5", "Gamma5") corresponds to the pseudoscalar pseudoscalar
        two-point function. The gammas argument dominateds over meson.
    idl : range
        If specified only configurations in the given range are read in.

    Returns
    -------
    corr : Corr
        Correlator of the source sink combination in question.
    '''
    if gammas is None:
        attrs = int(meson.rsplit('_', 1)[-1])
    else:
        if len(gammas) != 2:
            raise ValueError("'gammas' needs to have exactly two entries")
        attrs = {"gamma_snk": gammas[0],
                 "gamma_src": gammas[1]}
    return read_hd5(filestem=path + "/" + filestem, ens_id=ens_id,
                    group=meson.rsplit('_', 1)[0], attrs=attrs, idl=idl,
                    part="real")


def _extract_real_arrays(path, files, tree, keys):
    corr_data = {}
    for key in keys:
        corr_data[key] = []
    for hd5_file in files:
        h5file = h5py.File(path + '/' + hd5_file, "r")
        for key in keys:
            if tree + '/' + key not in h5file:
                raise Exception("Entry '" + key + "' not contained in the files.")
            raw_data = h5file[tree + '/' + key + '/data']
            real_data = raw_data[:].astype(np.double)
            corr_data[key].append(real_data)
        h5file.close()
    for key in keys:
        corr_data[key] = np.array(corr_data[key])
    return corr_data


def extract_t0_hd5(path, filestem, ens_id, obs='Clover energy density', fit_range=5, idl=None, **kwargs):
    r'''Read hadrons FlowObservables hdf5 file and extract t0

    Parameters
    -----------------
    path : str
        path to the files to read
    filestem : str
        namestem of the files to read
    ens_id : str
        name of the ensemble, required for internal bookkeeping
    obs : str
        label of the observable from which t0 should be extracted.
        Options: 'Clover energy density' and 'Plaquette energy density'
    fit_range : int
        Number of data points left and right of the zero
        crossing to be included in the linear fit. (Default: 5)
    idl : range
        If specified only configurations in the given range are read in.
    plot_fit : bool
        If true, the fit for the extraction of t0 is shown together with the data.
    '''

    files, idx = _get_files(path, filestem, idl)
    tree = "FlowObservables"

    h5file = h5py.File(path + '/' + files[0], "r")
    obs_key = None
    for key in h5file[tree].keys():
        if obs == h5file[tree][key].attrs["description"][0].decode():
            obs_key = key
            break
    h5file.close()
    if obs_key is None:
        raise Exception(f"Observable {obs} not found.")

    corr_data = _extract_real_arrays(path, files, tree, ["FlowObservables_0", obs_key])

    if not np.allclose(corr_data["FlowObservables_0"][0], corr_data["FlowObservables_0"][:]):
        raise Exception("Not all flow times were equal.")

    t2E_dict = {}
    for t2, dat in zip(corr_data["FlowObservables_0"][0], corr_data[obs_key].T):
        t2E_dict[t2] = Obs([dat], [ens_id], idl=[idx]) - 0.3

    return fit_t0(t2E_dict, fit_range, plot_fit=kwargs.get('plot_fit'))


def read_DistillationContraction_hd5(path, ens_id, diagrams=["direct"], idl=None):
    """Read hadrons DistillationContraction hdf5 files in given directory structure

    Parameters
    -----------------
    path : str
        path to the directories to read
    ens_id : str
        name of the ensemble, required for internal bookkeeping
    diagrams : list
        List of strings of the diagrams to extract, e.g. ["direct", "box", "cross"].
    idl : range
        If specified only configurations in the given range are read in.

    Returns
    -------
    result : dict
        extracted DistillationContration data
    """

    res_dict = {}

    directories, idx = _get_files(path, "data", idl)

    explore_path = Path(path + "/" + directories[0])

    for explore_file in explore_path.iterdir():
        if explore_file.is_file():
            stem = explore_file.with_suffix("").with_suffix("").as_posix().split("/")[-1]
        else:
            continue

        file_list = []
        for dir in directories:
            tmp_path = Path(path + "/" + dir)
            file_list.append((tmp_path / stem).as_posix() + tmp_path.suffix + ".h5")

        corr_data = {}

        for diagram in diagrams:
            corr_data[diagram] = []

        try:
            for n_file, (hd5_file, n_traj) in enumerate(zip(file_list, list(idx))):
                h5file = h5py.File(hd5_file)

                if n_file == 0:
                    if h5file["DistillationContraction/Metadata"].attrs.get("TimeSources")[0].decode() != "0...":
                        raise Exception("Routine is only implemented for files containing inversions on all timeslices.")

                    Nt = h5file["DistillationContraction/Metadata"].attrs.get("Nt")[0]

                    identifier = []
                    for in_file in range(len(h5file["DistillationContraction/Metadata/DmfInputFiles"].attrs.keys()) - 1):
                        encoded_info = h5file["DistillationContraction/Metadata/DmfInputFiles"].attrs.get("DmfInputFiles_" + str(in_file))
                        full_info = encoded_info[0].decode().split("/")[-1].replace(".h5", "").split("_")
                        my_tuple = (full_info[0], full_info[1][1:], full_info[2], full_info[3])
                        identifier.append(my_tuple)
                    identifier = tuple(identifier)
                    # "DistillationContraction/Metadata/DmfSuffix" contains info about different quarks, irrelevant in the SU(3) case.

                for diagram in diagrams:

                    if diagram == "triangle" and "Identity" not in str(identifier):
                        part = "im"
                    else:
                        part = "re"

                    real_data = np.zeros(Nt)
                    for x0 in range(Nt):
                        raw_data = h5file["DistillationContraction/Correlators/" + diagram + "/" + str(x0)][:][part].astype(np.double)
                        real_data += np.roll(raw_data, -x0)
                    real_data /= Nt

                    corr_data[diagram].append(real_data)
                h5file.close()

            res_dict[str(identifier)] = {}

            for diagram in diagrams:

                tmp_data = np.array(corr_data[diagram])

                l_obs = []
                for c in tmp_data.T:
                    l_obs.append(Obs([c], [ens_id], idl=[idx]))

                corr = Corr(l_obs)
                corr.tag = str(identifier)

                res_dict[str(identifier)][diagram] = corr
        except FileNotFoundError:
            print("Skip", stem)

    return res_dict


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

    Returns
    -------
    result : Npr_matrix
        read Cobs-matrix
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

    Returns
    -------
    result_dict: dict[Npr_matrix]
        extracted Bilinears
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

    Returns
    -------
    result_dict : dict
        extracted fourquark matrizes
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
