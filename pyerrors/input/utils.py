import fnmatch

def check_missing(idl,che):
    missing = []
    for ind in che:
            if not ind in idl:
                missing.append(ind)
    if(len(missing) == 0):
        print("There are no measurements missing.")
    else:
        print(len(missing),"measurements missing")
        miss_str = str(missing[0])
        for i in missing[1:]:
            miss_str += ","+str(i)
        print(miss_str)
