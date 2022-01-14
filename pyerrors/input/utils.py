"""Utilities for the input"""


def check_idl(idl, che):
    missing = []
    for c in che:
        if c not in idl:
            missing.append(c)
    # print missing such that it can directly be parsed to slurm terminal
    if not (len(missing) == 0):
        print(len(missing), "configs missing")
        miss_str = str(missing[0])
        for i in missing[1:]:
            miss_str += "," + str(i)
        print(miss_str)
