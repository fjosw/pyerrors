import os
import fnmatch
import re
import numpy as np  # Thinly-wrapped numpy
from ..obs import Obs
from .utils import sort_names, check_idl
import itertools


sep = "/"


def read_sfcf(path, prefix, name, quarks='.*', corr_type="bi", noffset=0, wf=0, wf2=0, version="1.0c", cfg_separator="n", silent=False, **kwargs):
    """Read sfcf files from given folder structure.

    Parameters
    ----------
    path : str
        Path to the sfcf files.
    prefix : str
        Prefix of the sfcf files.
    name : str
        Name of the correlation function to read.
    quarks : str
        Label of the quarks used in the sfcf input file. e.g. "quark quark"
        for version 0.0 this does NOT need to be given with the typical " - "
        that is present in the output file,
        this is done automatically for this version
    corr_type : str
        Type of correlation function to read. Can be
        - 'bi' for boundary-inner
        - 'bb' for boundary-boundary
        - 'bib' for boundary-inner-boundary
    noffset : int
        Offset of the source (only relevant when wavefunctions are used)
    wf : int
        ID of wave function
    wf2 : int
        ID of the second wavefunction
        (only relevant for boundary-to-boundary correlation functions)
    im : bool
        if True, read imaginary instead of real part
        of the correlation function.
    names : list
        Alternative labeling for replicas/ensembles.
        Has to have the appropriate length
    ens_name : str
        replaces the name of the ensemble
    version: str
        version of SFCF, with which the measurement was done.
        if the compact output option (-c) was specified,
        append a "c" to the version (e.g. "1.0c")
        if the append output option (-a) was specified,
        append an "a" to the version
    cfg_separator : str
        String that separates the ensemble identifier from the configuration number (default 'n').
    replica: list
        list of replica to be read, default is all
    files: list
        list of files to be read per replica, default is all.
        for non-compact output format, hand the folders to be read here.
    check_configs: list[list[int]]
        list of list of supposed configs, eg. [range(1,1000)]
        for one replicum with 1000 configs

    Returns
    -------
    result: list[Obs]
        list of Observables with length T, observable per timeslice.
        bb-type correlators have length 1.
    """
    ret = read_sfcf_multi(path, prefix, [name], quarks_list=[quarks], corr_type_list=[corr_type],
                          noffset_list=[noffset], wf_list=[wf], wf2_list=[wf2], version=version,
                          cfg_separator=cfg_separator, silent=silent, **kwargs)
    return ret[name][quarks][str(noffset)][str(wf)][str(wf2)]


def read_sfcf_multi(path, prefix, name_list, quarks_list=['.*'], corr_type_list=['bi'], noffset_list=[0], wf_list=[0], wf2_list=[0], version="1.0c", cfg_separator="n", silent=False, keyed_out=False, **kwargs):
    """Read sfcf files from given folder structure.

    Parameters
    ----------
    path : str
        Path to the sfcf files.
    prefix : str
        Prefix of the sfcf files.
    name : str
        Name of the correlation function to read.
    quarks_list : list[str]
        Label of the quarks used in the sfcf input file. e.g. "quark quark"
        for version 0.0 this does NOT need to be given with the typical " - "
        that is present in the output file,
        this is done automatically for this version
    corr_type_list : list[str]
        Type of correlation function to read. Can be
        - 'bi' for boundary-inner
        - 'bb' for boundary-boundary
        - 'bib' for boundary-inner-boundary
    noffset_list : list[int]
        Offset of the source (only relevant when wavefunctions are used)
    wf_list : int
        ID of wave function
    wf2_list : list[int]
        ID of the second wavefunction
        (only relevant for boundary-to-boundary correlation functions)
    im : bool
        if True, read imaginary instead of real part
        of the correlation function.
    names : list
        Alternative labeling for replicas/ensembles.
        Has to have the appropriate length
    ens_name : str
        replaces the name of the ensemble
    version: str
        version of SFCF, with which the measurement was done.
        if the compact output option (-c) was specified,
        append a "c" to the version (e.g. "1.0c")
        if the append output option (-a) was specified,
        append an "a" to the version
    cfg_separator : str
        String that separates the ensemble identifier from the configuration number (default 'n').
    replica: list
        list of replica to be read, default is all
    files: list[list[int]]
        list of files to be read per replica, default is all.
        for non-compact output format, hand the folders to be read here.
    check_configs: list[list[int]]
        list of list of supposed configs, eg. [range(1,1000)]
        for one replicum with 1000 configs

    Returns
    -------
    result: dict[list[Obs]]
        dict with one of the following properties:
        if keyed_out:
            dict[key] = list[Obs]
            where key has the form name/quarks/offset/wf/wf2
        if not keyed_out:
            dict[name][quarks][offset][wf][wf2] = list[Obs]
    """

    if kwargs.get('im'):
        im = 1
        part = 'imaginary'
    else:
        im = 0
        part = 'real'

    known_versions = ["0.0", "1.0", "2.0", "1.0c", "2.0c", "1.0a", "2.0a"]

    if version not in known_versions:
        raise Exception("This version is not known!")
    if (version[-1] == "c"):
        appended = False
        compact = True
        version = version[:-1]
    elif (version[-1] == "a"):
        appended = True
        compact = False
        version = version[:-1]
    else:
        compact = False
        appended = False
    ls = []
    if "replica" in kwargs:
        ls = kwargs.get("replica")
    else:
        for (dirpath, dirnames, filenames) in os.walk(path):
            if not appended:
                ls.extend(dirnames)
            else:
                ls.extend(filenames)
            break
        if not ls:
            raise Exception('Error, directory not found')
        # Exclude folders with different names
        for exc in ls:
            if not fnmatch.fnmatch(exc, prefix + '*'):
                ls = list(set(ls) - set([exc]))

    if not appended:
        ls = sort_names(ls)
        replica = len(ls)

    else:
        replica = len([file.split(".")[-1] for file in ls]) // len(set([file.split(".")[-1] for file in ls]))
    if replica == 0:
        raise Exception('No replica found in directory')
    if not silent:
        print('Read', part, 'part of', name_list, 'from', prefix[:-1], ',', replica, 'replica')

    if 'names' in kwargs:
        new_names = kwargs.get('names')
        if len(new_names) != len(set(new_names)):
            raise Exception("names are not unique!")
        if len(new_names) != replica:
            raise Exception('names should have the length', replica)

    else:
        ens_name = kwargs.get("ens_name")
        if not appended:
            new_names = _get_rep_names(ls, ens_name)
        else:
            new_names = _get_appended_rep_names(ls, prefix, name_list[0], ens_name)
        new_names = sort_names(new_names)

    idl = []

    noffset_list = [str(x) for x in noffset_list]
    wf_list = [str(x) for x in wf_list]
    wf2_list = [str(x) for x in wf2_list]

    # setup dict structures
    intern = {}
    for name, corr_type in zip(name_list, corr_type_list):
        intern[name] = {}
        b2b, single = _extract_corr_type(corr_type)
        intern[name]["b2b"] = b2b
        intern[name]["single"] = single
        intern[name]["spec"] = {}
        for quarks in quarks_list:
            intern[name]["spec"][quarks] = {}
            for off in noffset_list:
                intern[name]["spec"][quarks][off] = {}
                for w in wf_list:
                    intern[name]["spec"][quarks][off][w] = {}
                    for w2 in wf2_list:
                        intern[name]["spec"][quarks][off][w][w2] = {}
                        intern[name]["spec"][quarks][off][w][w2]["pattern"] = _make_pattern(version, name, off, w, w2, intern[name]['b2b'], quarks)

    internal_ret_dict = {}
    needed_keys = _lists2key(name_list, quarks_list, noffset_list, wf_list, wf2_list)
    for key in needed_keys:
        internal_ret_dict[key] = []

    if not appended:
        for i, item in enumerate(ls):
            rep_path = path + '/' + item
            if "files" in kwargs:
                files = kwargs.get("files")
                if isinstance(files, list):
                    if all(isinstance(f, list) for f in files):
                        files = files[i]
                    elif all(isinstance(f, str) for f in files):
                        files = files
                    else:
                        raise TypeError("files has to be of type list[list[str]] or list[str]!")
                else:
                    raise TypeError("files has to be of type list[list[str]] or list[str]!")

            else:
                files = []
            sub_ls = _find_files(rep_path, prefix, compact, files)
            rep_idl = []
            no_cfg = len(sub_ls)
            for cfg in sub_ls:
                try:
                    if compact:
                        rep_idl.append(int(cfg.split(cfg_separator)[-1]))
                    else:
                        rep_idl.append(int(cfg[3:]))
                except Exception:
                    raise Exception("Couldn't parse idl from directory, problem with file " + cfg)
            rep_idl.sort()
            # maybe there is a better way to print the idls
            if not silent:
                print(item, ':', no_cfg, ' configurations')
            idl.append(rep_idl)
            # here we have found all the files we need to look into.
            if i == 0:
                if version != "0.0" and compact:
                    file = path + '/' + item + '/' + sub_ls[0]
                for name in name_list:
                    if version == "0.0" or not compact:
                        file = path + '/' + item + '/' + sub_ls[0] + '/' + name
                    for key in _lists2key(quarks_list, noffset_list, wf_list, wf2_list):
                        specs = _key2specs(key)
                        quarks = specs[0]
                        off = specs[1]
                        w = specs[2]
                        w2 = specs[3]
                        # here, we want to find the place within the file,
                        # where the correlator we need is stored.
                        # to do so, the pattern needed is put together
                        # from the input values
                        start_read, T = _find_correlator(file, version, intern[name]["spec"][quarks][str(off)][str(w)][str(w2)]["pattern"], intern[name]['b2b'], silent=silent)
                        intern[name]["spec"][quarks][str(off)][str(w)][str(w2)]["start"] = start_read
                        intern[name]["T"] = T
                        # preparing the datastructure
                        # the correlators get parsed into...
                        deltas = []
                        for j in range(intern[name]["T"]):
                            deltas.append([])
                        internal_ret_dict[sep.join([name, key])] = deltas

            if compact:
                rep_deltas = _read_compact_rep(path, item, sub_ls, intern, needed_keys, im)
                for key in needed_keys:
                    name = _key2specs(key)[0]
                    for t in range(intern[name]["T"]):
                        internal_ret_dict[key][t].append(rep_deltas[key][t])
            else:
                for key in needed_keys:
                    rep_data = []
                    name = _key2specs(key)[0]
                    for subitem in sub_ls:
                        cfg_path = path + '/' + item + '/' + subitem
                        file_data = _read_o_file(cfg_path, name, needed_keys, intern, version, im)
                        rep_data.append(file_data)
                    for t in range(intern[name]["T"]):
                        internal_ret_dict[key][t].append([])
                        for cfg in range(no_cfg):
                            internal_ret_dict[key][t][i].append(rep_data[cfg][key][t])
    else:
        for key in needed_keys:
            specs = _key2specs(key)
            name = specs[0]
            quarks = specs[1]
            off = specs[2]
            w = specs[3]
            w2 = specs[4]
            if "files" in kwargs:
                if isinstance(kwargs.get("files"), list) and all(isinstance(f, str) for f in kwargs.get("files")):
                    name_ls = kwargs.get("files")
                else:
                    raise TypeError("In append mode, files has to be of type list[str]!")
            else:
                name_ls = ls
                for exc in name_ls:
                    if not fnmatch.fnmatch(exc, prefix + '*.' + name):
                        name_ls = list(set(name_ls) - set([exc]))
            name_ls = sort_names(name_ls)
            pattern = intern[name]['spec'][quarks][off][w][w2]['pattern']
            deltas = []
            for rep, file in enumerate(name_ls):
                rep_idl = []
                filename = path + '/' + file
                T, rep_idl, rep_data = _read_append_rep(filename, pattern, intern[name]['b2b'], cfg_separator, im, intern[name]['single'])
                if rep == 0:
                    intern[name]['T'] = T
                    for t in range(intern[name]['T']):
                        deltas.append([])
                for t in range(intern[name]['T']):
                    deltas[t].append(rep_data[t])
                internal_ret_dict[key] = deltas
                if name == name_list[0]:
                    idl.append(rep_idl)

    if kwargs.get("check_configs") is True:
        if not silent:
            print("Checking for missing configs...")
        che = kwargs.get("check_configs")
        if not (len(che) == len(idl)):
            raise Exception("check_configs has to be the same length as replica!")
        for r in range(len(idl)):
            if not silent:
                print("checking " + new_names[r])
            check_idl(idl[r], che[r])
        if not silent:
            print("Done")

    result_dict = {}
    if keyed_out:
        for key in needed_keys:
            name = _key2specs(key)[0]
            result = []
            for t in range(intern[name]["T"]):
                result.append(Obs(internal_ret_dict[key][t], new_names, idl=idl))
            result_dict[key] = result
    else:
        for name in name_list:
            result_dict[name] = {}
            for quarks in quarks_list:
                result_dict[name][quarks] = {}
                for off in noffset_list:
                    result_dict[name][quarks][off] = {}
                    for w in wf_list:
                        result_dict[name][quarks][off][w] = {}
                        for w2 in wf2_list:
                            key = _specs2key(name, quarks, off, w, w2)
                            result = []
                            for t in range(intern[name]["T"]):
                                result.append(Obs(internal_ret_dict[key][t], new_names, idl=idl))
                            result_dict[name][quarks][str(off)][str(w)][str(w2)] = result
    return result_dict


def _lists2key(*lists):
    keys = []
    for tup in itertools.product(*lists):
        keys.append(sep.join(tup))
    return keys


def _key2specs(key):
    return key.split(sep)


def _specs2key(*specs):
    return sep.join(specs)


def _read_o_file(cfg_path, name, needed_keys, intern, version, im):
    return_vals = {}
    for key in needed_keys:
        file = cfg_path + '/' + name
        specs = _key2specs(key)
        if specs[0] == name:
            with open(file) as fp:
                lines = fp.readlines()
                quarks = specs[1]
                off = specs[2]
                w = specs[3]
                w2 = specs[4]
                T = intern[name]["T"]
                start_read = intern[name]["spec"][quarks][off][w][w2]["start"]
                deltas = []
                for line in lines[start_read:start_read + T]:
                    floats = list(map(float, line.split()))
                    if version == "0.0":
                        deltas.append(floats[im - intern[name]["single"]])
                    else:
                        deltas.append(floats[1 + im - intern[name]["single"]])
                return_vals[key] = deltas
    return return_vals


def _extract_corr_type(corr_type):
    if corr_type == 'bb':
        b2b = True
        single = True
    elif corr_type == 'bib':
        b2b = True
        single = False
    else:
        b2b = False
        single = False
    return b2b, single


def _find_files(rep_path, prefix, compact, files=[]):
    sub_ls = []
    if not files == []:
        files.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
    else:
        for (dirpath, dirnames, filenames) in os.walk(rep_path):
            if compact:
                sub_ls.extend(filenames)
            else:
                sub_ls.extend(dirnames)
            break
        if compact:
            for exc in sub_ls:
                if not fnmatch.fnmatch(exc, prefix + '*'):
                    sub_ls = list(set(sub_ls) - set([exc]))
            sub_ls.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
        else:
            for exc in sub_ls:
                if not fnmatch.fnmatch(exc, 'cfg*'):
                    sub_ls = list(set(sub_ls) - set([exc]))
            sub_ls.sort(key=lambda x: int(x[3:]))
        files = sub_ls
    if len(files) == 0:
        raise FileNotFoundError("Did not find files in", rep_path, "with prefix", prefix, "and the given structure.")
    return files


def _make_pattern(version, name, noffset, wf, wf2, b2b, quarks):
    if version == "0.0":
        pattern = "# " + name + " : offset " + str(noffset) + ", wf " + str(wf)
        if b2b:
            pattern += ", wf_2 " + str(wf2)
        qs = quarks.split(" ")
        pattern += " : " + qs[0] + " - " + qs[1]
    else:
        pattern = 'name      ' + name + '\nquarks    ' + quarks + '\noffset    ' + str(noffset) + '\nwf        ' + str(wf)
        if b2b:
            pattern += '\nwf_2      ' + str(wf2)
    return pattern


def _find_correlator(file_name, version, pattern, b2b, silent=False):
    T = 0

    with open(file_name, "r") as my_file:

        content = my_file.read()
        match = re.search(pattern, content)
        if match:
            if version == "0.0":
                start_read = content.count('\n', 0, match.start()) + 1
                T = content.count('\n', start_read)
            else:
                start_read = content.count('\n', 0, match.start()) + 5 + b2b
                end_match = re.search(r'\n\s*\n', content[match.start():])
                T = content[match.start():].count('\n', 0, end_match.start()) - 4 - b2b
            if not T > 0:
                raise ValueError("Correlator with pattern\n" + pattern + "\nis empty!")
            if not silent:
                print(T, 'entries, starting to read in line', start_read)

        else:
            raise ValueError('Correlator with pattern\n' + pattern + '\nnot found.')

    return start_read, T


def _read_compact_file(rep_path, cfg_file, intern, needed_keys, im):
    return_vals = {}
    with open(rep_path + cfg_file) as fp:
        lines = fp.readlines()
        for key in needed_keys:
            keys = _key2specs(key)
            name = keys[0]
            quarks = keys[1]
            off = keys[2]
            w = keys[3]
            w2 = keys[4]

            T = intern[name]["T"]
            start_read = intern[name]["spec"][quarks][off][w][w2]["start"]
            # check, if the correlator is in fact
            # printed completely
            if (start_read + T + 1 > len(lines)):
                raise Exception("EOF before end of correlator data! Maybe " + rep_path + cfg_file + " is corrupted?")
            corr_lines = lines[start_read - 6: start_read + T]
            t_vals = []

            if corr_lines[1 - intern[name]["b2b"]].strip() != 'name      ' + name:
                raise Exception('Wrong format in file', cfg_file)

            for k in range(6, T + 6):
                floats = list(map(float, corr_lines[k].split()))
                t_vals.append(floats[-2:][im])
            return_vals[key] = t_vals
    return return_vals


def _read_compact_rep(path, rep, sub_ls, intern, needed_keys, im):
    rep_path = path + '/' + rep + '/'
    no_cfg = len(sub_ls)

    return_vals = {}
    for key in needed_keys:
        name = _key2specs(key)[0]
        deltas = []
        for t in range(intern[name]["T"]):
            deltas.append(np.zeros(no_cfg))
        return_vals[key] = deltas

    for cfg in range(no_cfg):
        cfg_file = sub_ls[cfg]
        cfg_data = _read_compact_file(rep_path, cfg_file, intern, needed_keys, im)
        for key in needed_keys:
            name = _key2specs(key)[0]
            for t in range(intern[name]["T"]):
                return_vals[key][t][cfg] = cfg_data[key][t]
    return return_vals


def _read_chunk(chunk, gauge_line, cfg_sep, start_read, T, corr_line, b2b, pattern, im, single):
    try:
        idl = int(chunk[gauge_line].split(cfg_sep)[-1])
    except Exception:
        raise Exception("Couldn't parse idl from directory, problem with chunk around line ", gauge_line)

    found_pat = ""
    data = []
    for li in chunk[corr_line + 1:corr_line + 6 + b2b]:
        found_pat += li
    if re.search(pattern, found_pat):
        for t, line in enumerate(chunk[start_read:start_read + T]):
            floats = list(map(float, line.split()))
            data.append(floats[im + 1 - single])
    return idl, data


def _read_append_rep(filename, pattern, b2b, cfg_separator, im, single):
    with open(filename, 'r') as fp:
        content = fp.readlines()
        data_starts = []
        for linenumber, line in enumerate(content):
            if "[run]" in line:
                data_starts.append(linenumber)
        if len(set([data_starts[i] - data_starts[i - 1] for i in range(1, len(data_starts))])) > 1:
            raise Exception("Irregularities in file structure found, not all runs have the same output length")
        chunk = content[:data_starts[1]]
        for linenumber, line in enumerate(chunk):
            if line.startswith("gauge_name"):
                gauge_line = linenumber
            elif line.startswith("[correlator]"):
                corr_line = linenumber
                found_pat = ""
                for li in chunk[corr_line + 1: corr_line + 6 + b2b]:
                    found_pat += li
                if re.search(pattern, found_pat):
                    start_read = corr_line + 7 + b2b
                    break
                else:
                    raise ValueError("Did not find pattern\n", pattern, "\nin\n", filename)
        endline = corr_line + 6 + b2b
        while not chunk[endline] == "\n":
            endline += 1
        T = endline - start_read

        # all other chunks should follow the same structure
        rep_idl = []
        rep_data = []

        for cnfg in range(len(data_starts)):
            start = data_starts[cnfg]
            stop = start + data_starts[1]
            chunk = content[start:stop]
            idl, data = _read_chunk(chunk, gauge_line, cfg_separator, start_read, T, corr_line, b2b, pattern, im, single)
            rep_idl.append(idl)
            rep_data.append(data)

        data = []

        for t in range(T):
            data.append([])
            for c in range(len(rep_data)):
                data[t].append(rep_data[c][t])
        return T, rep_idl, data


def _get_rep_names(ls, ens_name=None):
    new_names = []
    for entry in ls:
        try:
            idx = entry.index('r')
        except Exception:
            raise Exception("Automatic recognition of replicum failed, please enter the key word 'names'.")

        if ens_name:
            new_names.append('ens_name' + '|' + entry[idx:])
        else:
            new_names.append(entry[:idx] + '|' + entry[idx:])
    return new_names


def _get_appended_rep_names(ls, prefix, name, ens_name=None):
    new_names = []
    for exc in ls:
        if not fnmatch.fnmatch(exc, prefix + '*.' + name):
            ls = list(set(ls) - set([exc]))
    ls.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
    for entry in ls:
        myentry = entry[:-len(name) - 1]
        try:
            idx = myentry.index('r')
        except Exception:
            raise Exception("Automatic recognition of replicum failed, please enter the key word 'names'.")

        if ens_name:
            new_names.append('ens_name' + '|' + entry[idx:])
        else:
            new_names.append(myentry[:idx] + '|' + myentry[idx:])
    return new_names
