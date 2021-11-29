#!/usr/bin/env python
# coding: utf-8

import os
import fnmatch
import re
import numpy as np  # Thinly-wrapped numpy
from ..obs import Obs


def read_sfcf_old(path, prefix, name, quarks, noffset = 0, wf=0, wf2=0, **kwargs):
    """Read sfcf format (from around 2012) from given folder structure.

    Keyword arguments
    -----------------
    im -- if True, read imaginary instead of real part of the correlation function.
    single -- if True, read a boundary-to-boundary correlation function with a single value
    b2b -- if True, read a time-dependent boundary-to-boundary correlation function
    names -- Alternative labeling for replicas/ensembles. Has to have the appropriate length
    """
    if kwargs.get('im'):
        im = 1
        part = 'imaginary'
    else:
        im = 0
        part = 'real'
        
    b2b = 0

    if kwargs.get('b2b'):
        b2b = 1
    
    quarks = quarks.split(" ")
    read = 0
    T = 0
    start = 0
    ls = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        ls.extend(dirnames)
        break
    if not ls:
        print('Error, directory not found')
        #sys.exit()
    for exc in ls:
        if fnmatch.fnmatch(exc, prefix + '*'):
            ls = list(set(ls) - set(exc))
    if len(ls) > 1:
        ls.sort(key=lambda x: int(re.findall(r'\d+', x[len(prefix):])[0]))
    replica = len(ls)
    print('Read', part, 'part of', name, 'from', prefix, ',', replica, 'replica')
    if 'names' in kwargs:
        new_names = kwargs.get('names')
        if len(new_names) != replica:
            raise Exception('Names does not have the required length', replica)
    else:
        new_names = ls
    print(replica, 'replica')
    for i, item in enumerate(ls):
        print(item)
        sub_ls = []
        for (dirpath, dirnames, filenames) in os.walk(path+'/'+item):
            sub_ls.extend(dirnames)
            break
        for exc in sub_ls:
            if fnmatch.fnmatch(exc, 'cfg*'):
                sub_ls = list(set(sub_ls) - set(exc))
        sub_ls.sort(key=lambda x: int(x[3:]))
        no_cfg = len(sub_ls)
        print(no_cfg, 'configurations')
        if i == 0:
            with open(path + '/' + item + '/' + sub_ls[0] + '/' + name) as fp:
                for k, line in enumerate(fp):
                    #check if this is really the right file
                    pattern = "# "+name+" : offset "+str(noffset)+", wf "+"0"
                    #if b2b, a second wf is needed
                    if b2b:
                        pattern+=", wf_2 "+"0"
                    pattern+=" : "+quarks[0]+" - "+quarks[1]

                    if read == 1 and not line.strip() and k > start + 1:
                        break
                    if read == 1 and k >= start:
                        T += 1
                    if pattern in line:
                        #print(line)
                        read = 1
                        start = k+1
                print(str(T)+" entries found.")

            deltas = []
            for j in range(T):
                deltas.append([])

        sublength = len(sub_ls)
        for j in range(T):
            deltas[j].append(np.zeros(sublength))

        for cnfg, subitem in enumerate(sub_ls):
            with open(path + '/' + item + '/' + subitem + '/'+name) as fp:
                for k, line in enumerate(fp):
                    if(k >= start and k < start + T):
                        floats = list(map(float, line.split()))
                        deltas[k-start][i][cnfg] = floats[im]
                        

    result = []
    for t in range(T):
        result.append(Obs(deltas[t], new_names))

    return result


def read_sfcf(path, prefix, name, quarks='.*', noffset=0, wf=0, wf2=0, **kwargs):
    """Read sfcf c format from given folder structure.

    Parameters
    ----------
    quarks -- Label of the quarks used in the sfcf input file
    noffset -- Offset of the source (only relevant when wavefunctions are used)
    wf -- ID of wave function
    wf2 -- ID of the second wavefunction (only relevant for boundary-to-boundary correlation functions)
    im -- if True, read imaginary instead of real part of the correlation function.
    b2b -- if True, read a time-dependent boundary-to-boundary correlation function
    single -- if True, read time independent boundary to boundary correlation function
    names -- Alternative labeling for replicas/ensembles. Has to have the appropriate length
    ens_name : str
        replaces the name of the ensemble
    """
    if kwargs.get('im'):
        im = 1
        part = 'imaginary'
    else:
        im = 0
        part = 'real'

    if kwargs.get('single'):
        b2b = 1
        single = 1
    else:
        if kwargs.get('b2b'):
            b2b = 1
        else:
            b2b = 0
        single = 0

    files = []
    if "files" in kwargs:
        files = kwargs.get("files")

    #due to higher usage in current projects, compact file format is default
    compact = True
    #get version string
    version = "1.0"
    known_versions = ["0.0","1.0","2.0","1.0c","2.0c"]
    if "version" in kwargs:
        version = kwargs.get("version")
        if not version in known_versions:
            raise Exception("This version is not known!")
        #if the letter c is appended to the version, the compact fileformat is used (former read_sfcf_c)
        if(version[-1] == "c"):
            compact = True
            version = version[:-1]
        else:
            compact = False
    read = 0
    T = 0
    start = 0
    ls = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        ls.extend(dirnames)
        break
    if not ls:
        raise Exception('Error, directory not found')
    # Exclude folders with different names
    if len(files) != 0:
        ls = files
    else:
        for exc in ls:
            if not fnmatch.fnmatch(exc, prefix + '*'):
                ls = list(set(ls) - set([exc]))
    if len(ls) > 1:
        ls.sort(key=lambda x: int(re.findall(r'\d+', x[len(prefix):])[0]))  # New version, to cope with ids, etc.
    replica = len(ls)
    print('Read', part, 'part of', name, 'from', prefix[:-1], ',', replica, 'replica')

    if 'names' in kwargs:
        new_names = kwargs.get('names')
        if len(new_names) != replica:
            raise Exception('Names does not have the required length', replica)
    else:
        # Adjust replica names to new bookmarking system
        new_names = []
        for entry in ls:
            try:
                idx = entry.index('r')
            except:
                idx = len(entry)-2
            if 'ens_name' in kwargs:
                new_names.append(kwargs.get('ens_name') + '|' + entry[idx:])
            else:
                new_names.append(entry[:idx] + '|' + entry[idx:])
    for i, item in enumerate(ls):
        sub_ls = []
        for (dirpath, dirnames, filenames) in os.walk(path + '/' + item):
            if compact:
                sub_ls.extend(filenames)
            else:
                sub_ls.extend(dirnames)
            break
        
        #print(sub_ls)
        for exc in sub_ls:    
            if compact:
                if not fnmatch.fnmatch(exc, prefix + '*'):
                    sub_ls = list(set(sub_ls) - set([exc]))
                sub_ls.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
            else:
                if not fnmatch.fnmatch(exc, 'cfg*'):
                    sub_ls = list(set(sub_ls) - set([exc]))
                sub_ls.sort(key=lambda x: int(x[3:]))
        
        if compact:
            first_cfg = int(re.findall(r'\d+', sub_ls[0])[-1])

            last_cfg = len(sub_ls) + first_cfg - 1

            for cfg in range(1, len(sub_ls)):
                if int(re.findall(r'\d+', sub_ls[cfg])[-1]) != first_cfg + cfg:
                    last_cfg = cfg + first_cfg - 1
                    break

            no_cfg = last_cfg - first_cfg + 1
            print(item, ':', no_cfg, 'evenly spaced configurations (', first_cfg, '-', last_cfg, ') ,', len(sub_ls) - no_cfg, 'configs omitted\n')
        else:
            no_cfg = len(sub_ls)
            print(no_cfg, 'configurations')

        #here we have found all the files we need to look into.
        if i == 0:
            if compact:
    
                pattern = 'name      ' + name + '\nquarks    ' + quarks + '\noffset    ' + str(noffset) + '\nwf        ' + str(wf)
                if b2b:
                    pattern += '\nwf_2      ' + str(wf2)

                with open(path + '/' + item + '/' + sub_ls[0], 'r') as file:
                    content = file.read()
                    match = re.search(pattern, content)
                    if match:
                        start_read = content.count('\n', 0, match.start()) + 5 + b2b
                        end_match = re.search(r'\n\s*\n', content[match.start():])
                        T = content[match.start():].count('\n', 0, end_match.start()) - 4 - b2b
                        assert T > 0
                        print(T, 'entries, starting to read in line', start_read)
                    else:
                        raise Exception('Correlator with pattern\n' + pattern + '\nnot found.')
            else:
                #print(path + '/' + item + '/')# + sub_ls[0] + '/' + name)
                with open(path + '/' + item + '/' + sub_ls[0] + '/' + name) as fp:
                    for k, line in enumerate(fp):
                        if version == "0.0":
                            #check if this is really the right file
                            pattern = "# "+name+" : offset "+str(noffset)+", wf "+str(wf)
                            #if b2b, a second wf is needed
                            if b2b:
                                pattern+=", wf_2 "+str(wf2)
                            qs = quarks.split(" ")
                            pattern+=" : "+qs[0]+" - "+qs[1]
                            #print(pattern)
                        if read == 1 and not line.strip() and k > start + 1:
                            break
                        if read == 1 and k >= start:
                            T += 1

                        if version == "0.0":
                            if pattern in line:
                                #print(line)
                                read = 1
                                start = k+1
                        else:
                            if '[correlator]' in line:
                                read = 1
                                start = k + 7 + b2b
                                T -= b2b
                    print(str(T)+" entries found.")
            #we found where the correlator that is to be read is in the files
            deltas = []
            for j in range(T):
                deltas.append([])

        sublength = no_cfg
        for j in range(T):
            deltas[j].append(np.zeros(sublength))
        if compact:
            for cfg in range(no_cfg):
                with open(path + '/' + item + '/' + sub_ls[cfg]) as fp:
                    lines = fp.readlines()
                    if(start_read + T>len(lines)):
                        raise Exception("EOF before end of correlator data! Maybe "+path + '/' + item + '/' + sub_ls[cfg]+" is corrupted?")
                    for k in range(start_read - 6,start_read + T):
                        if k == start_read - 5 - b2b:
                            if lines[k].strip() != 'name      ' + name:
                                raise Exception('Wrong format', sub_ls[cfg])
                        if(k >= start_read and k < start_read + T):
                            floats = list(map(float, lines[k].split()))
                            deltas[k - start_read][i][cfg] = floats[-2:][im]
        else:
            for cnfg, subitem in enumerate(sub_ls):
                with open(path + '/' + item + '/' + subitem + '/' + name) as fp:
                    for k, line in enumerate(fp):
                        if(k >= start and k < start + T):
                            floats = list(map(float, line.split()))
                            if version == "0.0":
                                deltas[k-start][i][cnfg] = floats[im]
                            else:
                                deltas[k - start][i][cnfg] = floats[1 + im - single]


    result = []
    for t in range(T):
        result.append(Obs(deltas[t], new_names))
    return result

