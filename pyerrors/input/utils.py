import os
import fnmatch
import re
"""Utilities for the input"""


def sort_names(ll):
    r_pattern = r'r(\d+)'
    id_pattern = r'id(\d+)'

    # sort list by id first
    if all([re.search(id_pattern, entry) for entry in ll]):
        ll.sort(key=lambda x: int(re.findall(id_pattern, x)[0]))
    # then by replikum
    if all([re.search(r_pattern, entry) for entry in ll]):
        ll.sort(key=lambda x: int(re.findall(r_pattern, x)[0]))
    # as the rearrangements by one key let the other key untouched, the list is sorted now

    else:
        # fallback
        sames = ''
        if len(ll) > 1:
            for i in range(len(ll[0])):
                checking = ll[0][i]
                for rn in ll[1:]:
                    is_same = (rn[i] == checking)
                if is_same:
                    sames += checking
                else:
                    break
            print(ll[0][len(sames):])
        ll.sort(key=lambda x: int(re.findall(r'\d+', x[len(sames):])[0]))
    return ll


def find_files(path, prefix, postfix, ext, known_files=[]):
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
