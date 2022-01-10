#!/usr/bin/env python
# coding: utf-8

import os
import fnmatch
import re
import numpy as np  # Thinly-wrapped numpy
from ..obs import Obs
from . import utils

def read_sfcf(path, prefix, name, quarks='.*', noffset=0, wf=0, wf2=0, version = "1.0c", **kwargs):
    """Read sfcf c format from given folder structure.

    Parameters
    ----------
    quarks: str
        Label of the quarks used in the sfcf input file. e.g. "quark quark"
        for version 0.0 this does NOT need to be given with the typical " - " that is present in the output file,
        this is done automatically for this version
    noffset: int
        Offset of the source (only relevant when wavefunctions are used)
    wf: int
        ID of wave function
    wf2: int
        ID of the second wavefunction (only relevant for boundary-to-boundary correlation functions)
    im: bool
        if True, read imaginary instead of real part of the correlation function.
    b2b: bool
        if True, read a time-dependent boundary-to-boundary correlation function
    single: bool
        if True, read time independent boundary to boundary correlation function
    names: list
        Alternative labeling for replicas/ensembles. Has to have the appropriate length
    ens_name : str
        replaces the name of the ensemble
    version: str
        version of SFCF, with which the measurement was done. if the compact output option (-c) was spectified, append a c to the version (e.g. "1.0c")
    replica: list
        list of replica to be read, default is all
    files: list
        list of files to be read per replica, default is all. for non-conpact ouztput format, hand the folders to be read here.
    check_configs:
        list of list of supposed configs, eg. [range(1,1000)] for one replicum with 1000 configs
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
    if "replica" in kwargs:
        reps = kwargs.get("replica")
    if "files" in kwargs:
        files = kwargs.get("files")

    #due to higher usage in current projects, compact file format is default
    compact = True
    appended = False
    #get version string
    known_versions = ["0.0","1.0","2.0","1.0c","2.0c","1.0a","2.0a"]

    if not version in known_versions:
        raise Exception("This version is not known!")
    #if the letter c is appended to the version, the compact fileformat is used (former read_sfcf_c)
    if(version[-1] == "c"):
        appended = False
        compact = True
        version = version[:-1]
    elif(version[-1] == "a"):
        appended = True
        compact = False
        version = version[:-1]
    else:
        compact = False
        appended = False
    read = 0
    T = 0
    start = 0
    ls = []
    if "replica" in kwargs:
        ls = reps
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
    if len(ls) > 1:
        ls.sort(key=lambda x: int(re.findall(r'\d+', x[len(prefix):])[0]))  # New version, to cope with ids, etc.
    if not appended:
        replica = len(ls)
    else:
        replica = len([l.split(".")[-1] for l in ls])//len(set([l.split(".")[-1] for l in ls]))
    print('Read', part, 'part of', name, 'from', prefix[:-1], ',', replica, 'replica')
    if 'names' in kwargs:
        new_names = kwargs.get('names')
        if len(new_names)!=len(set(new_names)):
            raise Exception("names are not unique!")
        if len(new_names) != replica:
            raise Exception('Names does not have the required length', replica)
    else:
        # Adjust replica names to new bookmarking system

        new_names = []
        if not appended:
            for entry in ls:
                try:
                    idx = entry.index('r')
                except:
                    raise Exception("Automatic recognition of replicum failed, please enter the key word 'names'.")
                    
                if 'ens_name' in kwargs:
                    new_names.append(kwargs.get('ens_name') + '|' + entry[idx:])
                else:
                    new_names.append(entry[:idx] + '|' + entry[idx:])
        else:
            
            for exc in ls:
                if not fnmatch.fnmatch(exc, prefix + '*.'+name):
                    ls = list(set(ls) - set([exc]))
            ls.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
            for entry in ls:
                myentry = entry[:-len(name)-1]
                print(myentry)
                try:
                    idx = myentry.index('r')
                except:
                    raise Exception("Automatic recognition of replicum failed, please enter the key word 'names'.")
                
                if 'ens_name' in kwargs:
                    new_names.append(kwargs.get('ens_name') + '|' + myentry[idx:])
                else:
                    new_names.append(myentry[:idx] + '|' + myentry[idx:])
            #print(new_names)
    idl = []
    if not appended:
        for i, item in enumerate(ls):
            sub_ls = []
            if "files" in kwargs:
                sub_ls = kwargs.get("files")
                sub_ls.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
            else:
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
            #print(sub_ls)
            rep_idl = []
            no_cfg = len(sub_ls)
            for cfg in sub_ls:
                try:
                    if compact:
                        rep_idl.append(int(cfg.split("n")[-1]))
                    else:
                        rep_idl.append(int(cfg[3:]))
                except:
                    raise Exception("Couldn't parse idl from directroy, problem with file "+cfg)
            rep_idl.sort()
            #maybe there is a better way to print the idls
            print(item, ':', no_cfg, ' configurations')
            idl.append(rep_idl)
        #here we have found all the files we need to look into.
            if i == 0:
                #here, we want to find the place within the file, where the correlator we need is stored.
                if compact:
                    #to do so, the pattern needed is put together from the input values
                    pattern = 'name      ' + name + '\nquarks    ' + quarks + '\noffset    ' + str(noffset) + '\nwf        ' + str(wf)
                    if b2b:
                        pattern += '\nwf_2      ' + str(wf2)
                    #and the file is parsed through to find the pattern
                    with open(path + '/' + item + '/' + sub_ls[0], 'r') as file:
                        content = file.read()
                        match = re.search(pattern, content)
                        if match:
                            #the start and end point of the correlator in quaetion is extracted for later use in the other files
                            start_read = content.count('\n', 0, match.start()) + 5 + b2b
                            end_match = re.search(r'\n\s*\n', content[match.start():])
                            T = content[match.start():].count('\n', 0, end_match.start()) - 4 - b2b
                            assert T > 0
                            print(T, 'entries, starting to read in line', start_read)
                        else:
                            raise Exception('Correlator with pattern\n' + pattern + '\nnot found.')
                else:
                    #this part does the same as above, but for non-compactified versions of the files
                    with open(path + '/' + item + '/' + sub_ls[0] + '/' + name) as fp:
                        for k, line in enumerate(fp):
                            if version == "0.0":
                                #check if this is really the right file by matchin pattern similar to above
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
                #after preparing the datastructure the correlators get parsed into...
                deltas = []
                for j in range(T):
                    deltas.append([])
            
            
            for t in range(T):
                deltas[t].append(np.zeros(no_cfg))
            #... the actual parsing can start. we iterate through all measurement files in the path given...
            if compact:
                for cfg in range(no_cfg):
                    with open(path + '/' + item + '/' + sub_ls[cfg]) as fp:
                        lines = fp.readlines()
                        #check, if the correlator is in fact printed completely
                        if(start_read + T>len(lines)):
                            raise Exception("EOF before end of correlator data! Maybe "+path + '/' + item + '/' + sub_ls[cfg]+" is corrupted?")
                        #and start to read the correlator.
                        #the range here is chosen like this, since this allows for implementing a security check for every read correlator later...
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
                        #since the non-compatified files are typically not so long, we can iterate over the whole file.
                        #here one can also implement the chekc from above.
                        for k, line in enumerate(fp):
                            if(k >= start and k < start + T):
                                floats = list(map(float, line.split()))
                                if version == "0.0":
                                    deltas[k-start][i][cnfg] = floats[im]
                                else:
                                    deltas[k - start][i][cnfg] = floats[1 + im - single]
                                        
    else:
        for exc in ls:
            if not fnmatch.fnmatch(exc, prefix + '*.'+name):
                ls = list(set(ls) - set([exc]))
            ls.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
        #print(ls)
        pattern = 'name      ' + name + '\nquarks    ' + quarks + '\noffset    ' + str(noffset) + '\nwf        ' + str(wf)
        if b2b:
            pattern += '\nwf_2      ' + str(wf2)
        for rep,file in enumerate(ls):
            rep_idl = []
            with open(path + '/' + file, 'r') as fp:
                content = fp.readlines()
                data_starts = []
                for l,line in enumerate(content):
                    if "[run]" in line:
                        data_starts.append(l)
                if len(set([data_starts[i]-data_starts[i-1] for i in range(1,len(data_starts))])) > 1:
                    raise Exception ("Irregularities in file structure found, not all runs have the same output length")
                #print(data_starts)
                #first chunk of data
                chunk = content[:data_starts[1]]
                for l,line in enumerate(chunk):
                    if line.startswith("gauge_name"):
                        gauge_line = l
                        #meta_data["gauge_name"] = (line.strip()).split("/")[-1]
                    elif line.startswith("[correlator]"):
                        corr_line = l
                        found_pat = ""
                        for li in chunk[corr_line+1:corr_line+6+b2b]:
                            found_pat += li
                        if re.search(pattern,found_pat):
                            start_read = corr_line+7+b2b
                            T=len(chunk)-1-start_read
                if rep == 0:
                    deltas = []
                    for t in range(T):
                        deltas.append([])
                for t in range(T):
                    deltas[t].append(np.zeros(len(data_starts)))
                #all other chunks should follow the same structure
                for cnfg in range(len(data_starts)):
                    start = data_starts[cnfg]
                    stop = start+data_starts[1]
                    chunk = content[start:stop]
                    #meta_data = {}
                    
                    try:
                        rep_idl.append(int(chunk[gauge_line].split("n")[-1]))
                    except:
                        raise Exception("Couldn't parse idl from directroy, problem with chunk around line "+gauge_line)
                    
                    found_pat = ""
                    for li in chunk[corr_line+1:corr_line+6+b2b]:
                        found_pat += li
                    if re.search(pattern,found_pat):
                        #print("found pattern")
                        for t,line in enumerate(chunk[start_read:start_read+T]):
                            floats = list(map(float, line.split()))
                            deltas[t][rep][cnfg] = floats[-2:][im]
            idl.append(rep_idl)

    #print(new_names)
    #print(deltas)    
    #print(idl)
    if "check_configs" in kwargs:
        print("Checking for missing configs...")
        che = kwargs.get("check_configs")
        if not (len(che) == len(idl)):
            raise Exception("check_configs has to be the same length as replica!")
        for r in range(len(idl)):
            print("checking "+new_names[r])
            utils.check_idl(idl[r], che[r])
        print("Done")
    result = []
    for t in range(T):
        result.append(Obs(deltas[t], new_names, idl = idl))
    return result

