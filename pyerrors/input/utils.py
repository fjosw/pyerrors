"""Utilities for the input"""


def check_idl(idl, che):
    """Checks if list of configurations is contained in an idl

    Parameters
    ----------
    idl : range or list
        idl of the current replicum
    che : list
        list of configurations to be checked against
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
