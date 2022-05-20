import rapidjson as json
import gzip
import getpass
import socket
import datetime
import platform
import warnings
import re
import numpy as np
from ..obs import Obs
from ..covobs import Covobs
from ..correlators import Corr
from ..misc import _assert_equal_properties
from .. import version as pyerrorsversion


def create_json_string(ol, description='', indent=1):
    """Generate the string for the export of a list of Obs or structures containing Obs
    to a .json(.gz) file

    Parameters
    ----------
    ol : list
        List of objects that will be exported. At the moment, these objects can be
        either of: Obs, list, numpy.ndarray, Corr.
        All Obs inside a structure have to be defined on the same set of configurations.
    description : str
        Optional string that describes the contents of the json file.
    indent : int
        Specify the indentation level of the json file. None or 0 is permissible and
        saves disk space.
    """

    def _gen_data_d_from_list(ol):
        dl = []
        No = len(ol)
        for name in ol[0].mc_names:
            ed = {}
            ed['id'] = name
            ed['replica'] = []
            for r_name in ol[0].e_content[name]:
                rd = {}
                rd['name'] = r_name
                if ol[0].is_merged.get(r_name, False):
                    rd['is_merged'] = True
                rd['deltas'] = []
                offsets = [o.r_values[r_name] - o.value for o in ol]
                deltas = np.column_stack([ol[oi].deltas[r_name] + offsets[oi] for oi in range(No)])
                for i in range(len(ol[0].idl[r_name])):
                    rd['deltas'].append([ol[0].idl[r_name][i]])
                    rd['deltas'][-1] += deltas[i].tolist()
                ed['replica'].append(rd)
            dl.append(ed)
        return dl

    def _gen_cdata_d_from_list(ol):
        dl = []
        for name in ol[0].cov_names:
            ed = {}
            ed['id'] = name
            ed['layout'] = str(ol[0].covobs[name].cov.shape).lstrip('(').rstrip(')').rstrip(',')
            ed['cov'] = list(np.ravel(ol[0].covobs[name].cov))
            ncov = ol[0].covobs[name].cov.shape[0]
            ed['grad'] = []
            for i in range(ncov):
                ed['grad'].append([])
                for o in ol:
                    ed['grad'][-1].append(o.covobs[name].grad[i][0])
            dl.append(ed)
        return dl

    def write_Obs_to_dict(o):
        d = {}
        d['type'] = 'Obs'
        d['layout'] = '1'
        if o.tag:
            d['tag'] = [o.tag]
        if o.reweighted:
            d['reweighted'] = o.reweighted
        d['value'] = [o.value]
        data = _gen_data_d_from_list([o])
        if len(data) > 0:
            d['data'] = data
        cdata = _gen_cdata_d_from_list([o])
        if len(cdata) > 0:
            d['cdata'] = cdata
        return d

    def write_List_to_dict(ol):
        _assert_equal_properties(ol)
        d = {}
        d['type'] = 'List'
        d['layout'] = '%d' % len(ol)
        taglist = [o.tag for o in ol]
        if np.any([tag is not None for tag in taglist]):
            d['tag'] = taglist
        if ol[0].reweighted:
            d['reweighted'] = ol[0].reweighted
        d['value'] = [o.value for o in ol]
        data = _gen_data_d_from_list(ol)
        if len(data) > 0:
            d['data'] = data
        cdata = _gen_cdata_d_from_list(ol)
        if len(cdata) > 0:
            d['cdata'] = cdata
        return d

    def write_Array_to_dict(oa):
        ol = np.ravel(oa)
        _assert_equal_properties(ol)
        d = {}
        d['type'] = 'Array'
        d['layout'] = str(oa.shape).lstrip('(').rstrip(')').rstrip(',')
        taglist = [o.tag for o in ol]
        if np.any([tag is not None for tag in taglist]):
            d['tag'] = taglist
        if ol[0].reweighted:
            d['reweighted'] = ol[0].reweighted
        d['value'] = [o.value for o in ol]
        data = _gen_data_d_from_list(ol)
        if len(data) > 0:
            d['data'] = data
        cdata = _gen_cdata_d_from_list(ol)
        if len(cdata) > 0:
            d['cdata'] = cdata
        return d

    def _nan_Obs_like(obs):
        samples = []
        names = []
        idl = []
        for key, value in obs.idl.items():
            samples.append([np.nan] * len(value))
            names.append(key)
            idl.append(value)
        my_obs = Obs(samples, names, idl)
        my_obs._covobs = obs._covobs
        for name in obs._covobs:
            my_obs.names.append(name)
        my_obs.reweighted = obs.reweighted
        my_obs.is_merged = obs.is_merged
        return my_obs

    def write_Corr_to_dict(my_corr):
        first_not_none = next(i for i, j in enumerate(my_corr.content) if np.all(j))
        dummy_array = np.empty((my_corr.N, my_corr.N), dtype=object)
        dummy_array[:] = _nan_Obs_like(my_corr.content[first_not_none].ravel()[0])
        content = [o if o is not None else dummy_array for o in my_corr.content]
        dat = write_Array_to_dict(np.array(content, dtype=object))
        dat['type'] = 'Corr'
        corr_meta_data = str(my_corr.tag)
        if 'tag' in dat.keys():
            dat['tag'].append(corr_meta_data)
        else:
            dat['tag'] = [corr_meta_data]
        taglist = dat['tag']
        dat['tag'] = {}  # tag is now a dictionary, that contains the previous taglist in the key "tag"
        dat['tag']['tag'] = taglist
        if my_corr.prange is not None:
            dat['tag']['prange'] = my_corr.prange
        return dat

    if not isinstance(ol, list):
        ol = [ol]

    d = {}
    d['program'] = 'pyerrors %s' % (pyerrorsversion.__version__)
    d['version'] = '1.1'
    d['who'] = getpass.getuser()
    d['date'] = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')
    d['host'] = socket.gethostname() + ', ' + platform.platform()

    if description:
        d['description'] = description

    d['obsdata'] = []
    for io in ol:
        if isinstance(io, Obs):
            d['obsdata'].append(write_Obs_to_dict(io))
        elif isinstance(io, list):
            d['obsdata'].append(write_List_to_dict(io))
        elif isinstance(io, np.ndarray):
            d['obsdata'].append(write_Array_to_dict(io))
        elif isinstance(io, Corr):
            d['obsdata'].append(write_Corr_to_dict(io))
        else:
            raise Exception("Unkown datatype.")

    if indent:
        return json.dumps(d, indent=indent, ensure_ascii=False, write_mode=json.WM_SINGLE_LINE_ARRAY)
    else:
        return json.dumps(d, indent=indent, ensure_ascii=False, write_mode=json.WM_COMPACT)


def dump_to_json(ol, fname, description='', indent=1, gz=True):
    """Export a list of Obs or structures containing Obs to a .json(.gz) file

    Parameters
    ----------
    ol : list
        List of objects that will be exported. At the moment, these objects can be
        either of: Obs, list, numpy.ndarray, Corr.
        All Obs inside a structure have to be defined on the same set of configurations.
    fname : str
        Filename of the output file.
    description : str
        Optional string that describes the contents of the json file.
    indent : int
        Specify the indentation level of the json file. None or 0 is permissible and
        saves disk space.
    gz : bool
        If True, the output is a gzipped json. If False, the output is a json file.
    """

    jsonstring = create_json_string(ol, description, indent)

    if not fname.endswith('.json') and not fname.endswith('.gz'):
        fname += '.json'

    if gz:
        if not fname.endswith('.gz'):
            fname += '.gz'

        fp = gzip.open(fname, 'wb')
        fp.write(jsonstring.encode('utf-8'))
    else:
        fp = open(fname, 'w', encoding='utf-8')
        fp.write(jsonstring)
    fp.close()


def _parse_json_dict(json_dict, verbose=True, full_output=False):
    """Reconstruct a list of Obs or structures containing Obs from a dict that
    was built out of a json string.

    The following structures are supported: Obs, list, numpy.ndarray, Corr
    If the list contains only one element, it is unpacked from the list.

    Parameters
    ----------
    json_string : str
        json string containing the data.
    verbose : bool
        Print additional information that was written to the file.
    full_output : bool
        If True, a dict containing auxiliary information and the data is returned.
        If False, only the data is returned.
    """

    def _gen_obsd_from_datad(d):
        retd = {}
        if d:
            retd['names'] = []
            retd['idl'] = []
            retd['deltas'] = []
            retd['is_merged'] = {}
            for ens in d:
                for rep in ens['replica']:
                    rep_name = rep['name']
                    if len(rep_name) > len(ens["id"]):
                        if rep_name[len(ens["id"])] != "|":
                            tmp_list = list(rep_name)
                            tmp_list = tmp_list[:len(ens["id"])] + ["|"] + tmp_list[len(ens["id"]):]
                            rep_name = ''.join(tmp_list)
                    retd['names'].append(rep_name)
                    retd['idl'].append([di[0] for di in rep['deltas']])
                    retd['deltas'].append(np.array([di[1:] for di in rep['deltas']]))
                    retd['is_merged'][rep_name] = rep.get('is_merged', False)
        return retd

    def _gen_covobsd_from_cdatad(d):
        retd = {}
        for ens in d:
            retl = []
            name = ens['id']
            layouts = ens.get('layout', '1').strip()
            layout = [int(ls.strip()) for ls in layouts.split(',') if len(ls) > 0]
            cov = np.reshape(ens['cov'], layout)
            grad = ens['grad']
            nobs = len(grad[0])
            for i in range(nobs):
                retl.append({'name': name, 'cov': cov, 'grad': [g[i] for g in grad]})
            retd[name] = retl
        return retd

    def get_Obs_from_dict(o):
        layouts = o.get('layout', '1').strip()
        if layouts != '1':
            raise Exception("layout is %s has to be 1 for type Obs." % (layouts), RuntimeWarning)

        values = o['value']
        od = _gen_obsd_from_datad(o.get('data', {}))
        cd = _gen_covobsd_from_cdatad(o.get('cdata', {}))

        if od:
            ret = Obs([[ddi[0] + values[0] for ddi in di] for di in od['deltas']], od['names'], idl=od['idl'])
            ret._value = values[0]
            ret.is_merged = od['is_merged']
        else:
            ret = Obs([], [], means=[])
            ret._value = values[0]
        for name in cd:
            co = cd[name][0]
            ret._covobs[name] = Covobs(None, co['cov'], co['name'], grad=co['grad'])
            ret.names.append(co['name'])

        ret.reweighted = o.get('reweighted', False)
        ret.tag = o.get('tag', [None])[0]
        return ret

    def get_List_from_dict(o):
        layouts = o.get('layout', '1').strip()
        layout = int(layouts)
        values = o['value']
        od = _gen_obsd_from_datad(o.get('data', {}))
        cd = _gen_covobsd_from_cdatad(o.get('cdata', {}))

        ret = []
        taglist = o.get('tag', layout * [None])
        for i in range(layout):
            if od:
                ret.append(Obs([list(di[:, i] + values[i]) for di in od['deltas']], od['names'], idl=od['idl']))
                ret[-1]._value = values[i]
                ret[-1].is_merged = od['is_merged']
            else:
                ret.append(Obs([], [], means=[]))
                ret[-1]._value = values[i]
                print('Created Obs with means= ', values[i])
            for name in cd:
                co = cd[name][i]
                ret[-1]._covobs[name] = Covobs(None, co['cov'], co['name'], grad=co['grad'])
                ret[-1].names.append(co['name'])

            ret[-1].reweighted = o.get('reweighted', False)
            ret[-1].tag = taglist[i]
        return ret

    def get_Array_from_dict(o):
        layouts = o.get('layout', '1').strip()
        layout = [int(ls.strip()) for ls in layouts.split(',') if len(ls) > 0]
        N = np.prod(layout)
        values = o['value']
        od = _gen_obsd_from_datad(o.get('data', {}))
        cd = _gen_covobsd_from_cdatad(o.get('cdata', {}))

        ret = []
        taglist = o.get('tag', N * [None])
        for i in range(N):
            if od:
                ret.append(Obs([di[:, i] + values[i] for di in od['deltas']], od['names'], idl=od['idl']))
                ret[-1]._value = values[i]
                ret[-1].is_merged = od['is_merged']
            else:
                ret.append(Obs([], [], means=[]))
                ret[-1]._value = values[i]
            for name in cd:
                co = cd[name][i]
                ret[-1]._covobs[name] = Covobs(None, co['cov'], co['name'], grad=co['grad'])
                ret[-1].names.append(co['name'])
            ret[-1].reweighted = o.get('reweighted', False)
            ret[-1].tag = taglist[i]
        return np.reshape(ret, layout)

    def get_Corr_from_dict(o):
        if isinstance(o.get('tag'), list):  # supports the old way
            taglist = o.get('tag')  # This had to be modified to get the taglist from the dictionary
            temp_prange = None
        elif isinstance(o.get('tag'), dict):
            tagdic = o.get('tag')
            taglist = tagdic['tag']
            if 'prange' in tagdic:
                temp_prange = tagdic['prange']
            else:
                temp_prange = None
        else:
            raise Exception("The tag is not a list or dict")

        corr_tag = taglist[-1]
        tmp_o = o
        tmp_o['tag'] = taglist[:-1]
        if len(tmp_o['tag']) == 0:
            del tmp_o['tag']
        dat = get_Array_from_dict(tmp_o)
        my_corr = Corr([None if np.isnan(o.ravel()[0].value) else o for o in list(dat)])
        if corr_tag != 'None':
            my_corr.tag = corr_tag

        my_corr.prange = temp_prange
        return my_corr

    prog = json_dict.get('program', '')
    version = json_dict.get('version', '')
    who = json_dict.get('who', '')
    date = json_dict.get('date', '')
    host = json_dict.get('host', '')
    if prog and verbose:
        print('Data has been written using %s.' % (prog))
    if version and verbose:
        print('Format version %s' % (version))
    if np.any([who, date, host] and verbose):
        print('Written by %s on %s on host %s' % (who, date, host))
    description = json_dict.get('description', '')
    if description and verbose:
        print()
        print('Description: ', description)
    obsdata = json_dict['obsdata']
    ol = []
    for io in obsdata:
        if io['type'] == 'Obs':
            ol.append(get_Obs_from_dict(io))
        elif io['type'] == 'List':
            ol.append(get_List_from_dict(io))
        elif io['type'] == 'Array':
            ol.append(get_Array_from_dict(io))
        elif io['type'] == 'Corr':
            ol.append(get_Corr_from_dict(io))
        else:
            raise Exception("Unkown datatype.")

    if full_output:
        retd = {}
        retd['program'] = prog
        retd['version'] = version
        retd['who'] = who
        retd['date'] = date
        retd['host'] = host
        retd['description'] = description
        retd['obsdata'] = ol

        return retd
    else:
        if len(obsdata) == 1:
            ol = ol[0]

        return ol


def import_json_string(json_string, verbose=True, full_output=False):
    """Reconstruct a list of Obs or structures containing Obs from a json string.

    The following structures are supported: Obs, list, numpy.ndarray, Corr
    If the list contains only one element, it is unpacked from the list.

    Parameters
    ----------
    json_string : str
        json string containing the data.
    verbose : bool
        Print additional information that was written to the file.
    full_output : bool
        If True, a dict containing auxiliary information and the data is returned.
        If False, only the data is returned.
    """

    return _parse_json_dict(json.loads(json_string), verbose, full_output)


def load_json(fname, verbose=True, gz=True, full_output=False):
    """Import a list of Obs or structures containing Obs from a .json(.gz) file.

    The following structures are supported: Obs, list, numpy.ndarray, Corr
    If the list contains only one element, it is unpacked from the list.

    Parameters
    ----------
    fname : str
        Filename of the input file.
    verbose : bool
        Print additional information that was written to the file.
    gz : bool
        If True, assumes that data is gzipped. If False, assumes JSON file.
    full_output : bool
        If True, a dict containing auxiliary information and the data is returned.
        If False, only the data is returned.
    """
    if not fname.endswith('.json') and not fname.endswith('.gz'):
        fname += '.json'
    if gz:
        if not fname.endswith('.gz'):
            fname += '.gz'
        with gzip.open(fname, 'r') as fin:
            d = json.load(fin)
    else:
        if fname.endswith('.gz'):
            warnings.warn("Trying to read from %s without unzipping!" % fname, UserWarning)
        with open(fname, 'r', encoding='utf-8') as fin:
            d = json.loads(fin.read())

    return _parse_json_dict(d, verbose, full_output)


def _ol_from_dict(ind, reps='DICTOBS'):
    """Convert a dictionary of Obs objects to a list and a dictionary that contains
    placeholders instead of the Obs objects.

    Parameters
    ----------
    ind : dict
        Dict of JSON valid structures and objects that will be exported.
        At the moment, these object can be either of: Obs, list, numpy.ndarray, Corr.
        All Obs inside a structure have to be defined on the same set of configurations.
    reps : str
        Specify the structure of the placeholder in exported dict to be reps[0-9]+.
    """

    obstypes = (Obs, Corr, np.ndarray)

    if not reps.isalnum():
        raise Exception('Placeholder string has to be alphanumeric!')
    ol = []
    counter = 0

    def dict_replace_obs(d):
        nonlocal ol
        nonlocal counter
        x = {}
        for k, v in d.items():
            if isinstance(v, dict):
                v = dict_replace_obs(v)
            elif isinstance(v, list) and all([isinstance(o, Obs) for o in v]):
                v = obslist_replace_obs(v)
            elif isinstance(v, list):
                v = list_replace_obs(v)
            elif isinstance(v, obstypes):
                ol.append(v)
                v = reps + '%d' % (counter)
                counter += 1
            elif isinstance(v, str):
                if bool(re.match(r'%s[0-9]+' % (reps), v)):
                    raise Exception('Dict contains string %s that matches the placeholder! %s Cannot be savely exported.' % (v, reps))
            x[k] = v
        return x

    def list_replace_obs(li):
        nonlocal ol
        nonlocal counter
        x = []
        for e in li:
            if isinstance(e, list):
                e = list_replace_obs(e)
            elif isinstance(e, list) and all([isinstance(o, Obs) for o in e]):
                e = obslist_replace_obs(e)
            elif isinstance(e, dict):
                e = dict_replace_obs(e)
            elif isinstance(e, obstypes):
                ol.append(e)
                e = reps + '%d' % (counter)
                counter += 1
            elif isinstance(e, str):
                if bool(re.match(r'%s[0-9]+' % (reps), e)):
                    raise Exception('Dict contains string %s that matches the placeholder! %s Cannot be savely exported.' % (e, reps))
            x.append(e)
        return x

    def obslist_replace_obs(li):
        nonlocal ol
        nonlocal counter
        il = []
        for e in li:
            il.append(e)

        ol.append(il)
        x = reps + '%d' % (counter)
        counter += 1
        return x

    nd = dict_replace_obs(ind)

    return ol, nd


def dump_dict_to_json(od, fname, description='', indent=1, reps='DICTOBS', gz=True):
    """Export a dict of Obs or structures containing Obs to a .json(.gz) file

    Parameters
    ----------
    od : dict
        Dict of JSON valid structures and objects that will be exported.
        At the moment, these objects can be either of: Obs, list, numpy.ndarray, Corr.
        All Obs inside a structure have to be defined on the same set of configurations.
    fname : str
        Filename of the output file.
    description : str
        Optional string that describes the contents of the json file.
    indent : int
        Specify the indentation level of the json file. None or 0 is permissible and
        saves disk space.
    reps : str
        Specify the structure of the placeholder in exported dict to be reps[0-9]+.
    gz : bool
        If True, the output is a gzipped json. If False, the output is a json file.
    """

    if not isinstance(od, dict):
        raise Exception('od has to be a dictionary. Did you want to use dump_to_json?')

    infostring = ('This JSON file contains a python dictionary that has been parsed to a list of structures. '
                  'OBSDICT contains the dictionary, where Obs or other structures have been replaced by '
                  '' + reps + '[0-9]+. The field description contains the additional description of this JSON file. '
                  'This file may be parsed to a dict with the pyerrors routine load_json_dict.')

    desc_dict = {'INFO': infostring, 'OBSDICT': {}, 'description': description}
    ol, desc_dict['OBSDICT'] = _ol_from_dict(od, reps=reps)

    dump_to_json(ol, fname, description=desc_dict, indent=indent, gz=gz)


def _od_from_list_and_dict(ol, ind, reps='DICTOBS'):
    """Parse a list of Obs or structures containing Obs and an accompanying
    dict, where the structures have been replaced by placeholders to a
    dict that contains the structures.

    The following structures are supported: Obs, list, numpy.ndarray, Corr

    Parameters
    ----------
    ol : list
        List of objects -
        At the moment, these objects can be either of: Obs, list, numpy.ndarray, Corr.
        All Obs inside a structure have to be defined on the same set of configurations.
    ind : dict
        Dict that defines the structure of the resulting dict and contains placeholders
    reps : str
        Specify the structure of the placeholder in imported dict to be reps[0-9]+.
    """
    if not reps.isalnum():
        raise Exception('Placeholder string has to be alphanumeric!')

    counter = 0

    def dict_replace_string(d):
        nonlocal counter
        nonlocal ol
        x = {}
        for k, v in d.items():
            if isinstance(v, dict):
                v = dict_replace_string(v)
            elif isinstance(v, list):
                v = list_replace_string(v)
            elif isinstance(v, str) and bool(re.match(r'%s[0-9]+' % (reps), v)):
                index = int(v[len(reps):])
                v = ol[index]
                counter += 1
            x[k] = v
        return x

    def list_replace_string(li):
        nonlocal counter
        nonlocal ol
        x = []
        for e in li:
            if isinstance(e, list):
                e = list_replace_string(e)
            elif isinstance(e, dict):
                e = dict_replace_string(e)
            elif isinstance(e, str) and bool(re.match(r'%s[0-9]+' % (reps), e)):
                index = int(e[len(reps):])
                e = ol[index]
                counter += 1
            x.append(e)
        return x

    nd = dict_replace_string(ind)

    if counter == 0:
        raise Exception('No placeholder has been replaced! Check if reps is set correctly.')

    return nd


def load_json_dict(fname, verbose=True, gz=True, full_output=False, reps='DICTOBS'):
    """Import a dict of Obs or structures containing Obs from a .json(.gz) file.

    The following structures are supported: Obs, list, numpy.ndarray, Corr

    Parameters
    ----------
    fname : str
        Filename of the input file.
    verbose : bool
        Print additional information that was written to the file.
    gz : bool
        If True, assumes that data is gzipped. If False, assumes JSON file.
    full_output : bool
        If True, a dict containing auxiliary information and the data is returned.
        If False, only the data is returned.
    reps : str
        Specify the structure of the placeholder in imported dict to be reps[0-9]+.
    """
    indata = load_json(fname, verbose=verbose, gz=gz, full_output=True)
    description = indata['description']['description']
    indict = indata['description']['OBSDICT']
    ol = indata['obsdata']
    od = _od_from_list_and_dict(ol, indict, reps=reps)

    if full_output:
        indata['description'] = description
        indata['obsdata'] = od
        return indata
    else:
        return od
