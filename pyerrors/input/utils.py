import re
"""Utilities for the input"""


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
