"""Utilities for the input"""

import re
import fnmatch
import os


def sort_names(ll):
    """Sorts a list of names of replika with searches for `r` and `id` in the replikum string.
    If this search fails, a fallback method is used,
    where the strings are simply compared and the first diffeing numeral is used for differentiation.

    Parameters
    ----------
    ll: list
        list to sort

    Returns
    -------
    ll: list
        sorted list
    """

    if len(ll) > 1:
        sorted = False
        r_pattern = r'r(\d+)'
        id_pattern = r'id(\d+)'

        # sort list by id first
        if all([re.search(id_pattern, entry) for entry in ll]):
            ll.sort(key=lambda x: int(re.findall(id_pattern, x)[0]))
            sorted = True
        # then by replikum
        if all([re.search(r_pattern, entry) for entry in ll]):
            ll.sort(key=lambda x: int(re.findall(r_pattern, x)[0]))
            sorted = True
        # as the rearrangements by one key let the other key untouched, the list is sorted now

        if not sorted:
            # fallback
            sames = ''
            for i in range(len(ll[0])):
                checking = ll[0][i]
                for rn in ll[1:]:
                    is_same = (rn[i] == checking)
                if is_same:
                    sames += checking
                else:
                    break
            print("Using prefix:", sames)
            ll.sort(key=lambda x: int(re.findall(r'\d+', x[len(sames):])[0]))
    return ll


def check_idl(idl, che):
    """Checks if list of configurations is contained in an idl

    Parameters
    ----------
    idl : range or list
        idl of the current replicum
    che : list
        list of configurations to be checked against

    Returns
    -------
    miss_str : str
        string with integers of which idls are missing
    """

    missing = []
    for c in che:
        if c not in idl:
            missing.append(c)
    # print missing configurations such that it can directly be parsed to slurm terminal
    if not (len(missing) == 0):
        print(len(missing), "configs missing")
        miss_str = str(missing[0])
        for i in missing[1:]:
            miss_str += "," + str(i)
        print(miss_str)
    return miss_str


def check_params(path, param_hash, prefix, param_prefix="parameters_"):
    """
    Check if, for sfcf, the parameter hashes at the end of the parameter files are in fact the expected one.

    Parameters
    ----------
    path: str
        measurement path, same as for sfcf read method
    param_hash: str
        expected parameter hash
    prefix: str
        data prefix to find the appropriate replicum folders in path
    param_prefix: str
        prefix of the parameter file. Defaults to 'parameters_'

    Returns
    -------
    nums: dict
        dictionary of faulty parameter files sorted by the replica paths
    """

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

    ls = sort_names(ls)
    nums = {}
    for rep in ls:
        rep_path = path + '/' + rep
        # files of replicum
        sub_ls = []
        for (dirpath, dirnames, filenames) in os.walk(rep_path):
            sub_ls.extend(filenames)

        # filter
        param_files = []
        for file in sub_ls:
            if fnmatch.fnmatch(file, param_prefix + '*'):
                param_files.append(file)

        rep_nums = ''
        for file in param_files:
            with open(rep_path + '/' + file) as fp:
                for line in fp:
                    pass
                last_line = line
                if last_line.split()[2] != param_hash:
                    rep_nums += file.split("_")[1] + ','
        nums[rep_path] = rep_nums

        if not len(rep_nums) == 0:
            raise Warning("found differing parameter hash in the param files in " + rep_path)
    return nums
