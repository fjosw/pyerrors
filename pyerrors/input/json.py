import json
import gzip
from ..obs import Obs
import getpass
import socket
import datetime
from .. import version as pyerrorsversion
import platform
import numpy as np


def dump_to_json(ol, fname, description='', indent=4):
    """Export a list of Obs or structures containing Obs to a .json.gz file

    Parameters
    -----------------
    ol : list
        List of objects that will be exported. At the moments, these objects can be
        either of: Obs, list, np.ndarray
        All Obs inside a structure have to be defined on the same set of configurations.
    fname : str
        Filename of the output file
    description : str
        Optional string that describes the contents of the json file
    indent : int
        Specify the indentation level of the json file. None or 0 is permissible and
        saves disk space.
    """

    def _default(self, obj):
        return str(obj)
    my_encoder = json.JSONEncoder
    _default.default = json.JSONEncoder().default
    my_encoder.default = _default

    class deltalist:
        def __init__(self, li):
            self.cnfg = li[0]
            self.deltas = li[1:]

        def __repr__(self):
            s = '[%d' % (self.cnfg)
            for d in self.deltas:
                s += ', %1.15e' % (d)
            s += ']'
            return s

        def __str__(self):
            return self.__repr__()

    def _gen_data_d_from_list(ol):
        dl = []
        for name in ol[0].e_names:
            ed = {}
            ed['id'] = name
            ed['replica'] = []
            for r_name in ol[0].e_content[name]:
                rd = {}
                rd['name'] = r_name
                if ol[0].is_merged.get(r_name, False):
                    rd['is_merged'] = True
                rd['deltas'] = []
                for i in range(len(ol[0].idl[r_name])):
                    rd['deltas'].append([ol[0].idl[r_name][i]])
                    for o in ol:
                        rd['deltas'][-1].append(o.deltas[r_name][i])
                    rd['deltas'][-1] = deltalist(rd['deltas'][-1])
                ed['replica'].append(rd)
            dl.append(ed)
        return dl

    def _assert_equal_properties(ol, otype=Obs):
        for o in ol:
            if not isinstance(o, otype):
                raise Exception('Wrong data type in list!')
        for o in ol[1:]:
            if not ol[0].is_merged == o.is_merged:
                raise Exception('All Obs in list have to be defined on the same set of configs!')
            if not ol[0].reweighted == o.reweighted:
                raise Exception('All Obs in list have to have the same property .reweighted!')
            if not ol[0].e_content == o.e_content:
                raise Exception('All Obs in list have to be defined on the same set of configs!')
            # more stringend tests --> compare idl?

    def write_Obs_to_dict(o):
        d = {}
        d['type'] = 'Obs'
        d['layout'] = '1'
        d['tag'] = o.tag
        if o.reweighted:
            d['reweighted'] = o.reweighted
        d['value'] = [o.value]
        d['data'] = _gen_data_d_from_list([o])
        return d

    def write_List_to_dict(ol):
        _assert_equal_properties(ol)
        d = {}
        d['type'] = 'List'
        d['layout'] = '%d' % len(ol)
        if len(set([o.tag for o in ol])) > 1:
            d['tag'] = ''
            for o in ol:
                d['tag'] += '%s\n' % (o.tag)
        else:
            d['tag'] = ol[0].tag
        if ol[0].reweighted:
            d['reweighted'] = ol[0].reweighted
        d['value'] = [o.value for o in ol]
        d['data'] = _gen_data_d_from_list(ol)

        return d

    def write_Array_to_dict(oa):
        ol = np.ravel(oa)
        _assert_equal_properties(ol)
        d = {}
        d['type'] = 'Array'
        d['layout'] = str(oa.shape).lstrip('(').rstrip(')')
        if len(set([o.tag for o in ol])) > 1:
            d['tag'] = ''
            for o in ol:
                d['tag'] += '%s\n' % (o.tag)
        else:
            d['tag'] = ol[0].tag
        if ol[0].reweighted:
            d['reweighted'] = ol[0].reweighted
        d['value'] = [o.value for o in ol]
        d['data'] = _gen_data_d_from_list(ol)
        return d
    if not isinstance(ol, list):
        ol = [ol]
    d = {}
    d['program'] = 'pyerrors %s' % (pyerrorsversion.__version__)
    d['version'] = '0.1'
    d['who'] = getpass.getuser()
    d['date'] = str(datetime.datetime.now())[:-7]
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
    if not fname.endswith('.json') and not fname.endswith('.gz'):
        fname += '.json'
    if not fname.endswith('.gz'):
        fname += '.gz'
    jsonstring = json.dumps(d, indent=indent, cls=my_encoder)
    # workaround for un-indentation of delta lists
    jsonstring = jsonstring.replace('"[', '[').replace(']"', ']')
    fp = gzip.open(fname, 'wb')
    fp.write(jsonstring.encode('utf-8'))
    fp.close()

    # this would be nicer, since it does not need a string
    # with gzip.open(fname, 'wt', encoding='UTF-8') as zipfile:
    #    json.dump(d, zipfile, indent=indent)


def load_json(fname, verbose=True):
    """Import a list of Obs or structures containing Obs to a .json.gz file.
    The following structures are supported: Obs, list, np.ndarray

    Parameters
    -----------------
    fname : str
        Filename of the input file
    verbose : bool
        Print additional information that was written to the file.
    """

    def _gen_obsd_from_datad(d):
        retd = {}
        retd['names'] = []
        retd['idl'] = []
        retd['deltas'] = []
        retd['is_merged'] = {}
        for ens in d:
            for rep in ens['replica']:
                retd['names'].append(rep['name'])
                retd['idl'].append([di[0] for di in rep['deltas']])
                retd['deltas'].append([di[1:] for di in rep['deltas']])
                retd['is_merged'][rep['name']] = rep.get('is_merged', False)
        retd['deltas'] = np.array(retd['deltas'])
        return retd

    def get_Obs_from_dict(o):
        layouts = o.get('layout', '1').strip()
        if layouts != '1':
            raise Exception("layout is %s has to be 1 for type Obs." % (layouts), RuntimeWarning)

        values = o['value']
        od = _gen_obsd_from_datad(o['data'])

        ret = Obs([[ddi[0] + values[0] for ddi in di] for di in od['deltas']], od['names'], idl=od['idl'])
        ret.reweighted = o.get('reweighted', False)
        ret.is_merged = od['is_merged']
        ret.tag = o.get('tag', '')
        return ret

    def get_List_from_dict(o):
        layouts = o.get('layout', '1').strip()
        layout = int(layouts)
        values = o['value']
        od = _gen_obsd_from_datad(o['data'])

        ret = []
        for i in range(layout):
            ret.append(Obs([list(di[:, i] + values[i]) for di in od['deltas']], od['names'], idl=od['idl']))
            ret[-1].reweighted = o.get('reweighted', False)
            ret[-1].is_merged = od['is_merged']
            ret[-1].tag = o.get('tag', '')
        return ret

    def get_Array_from_dict(o):
        layouts = o.get('layout', '1').strip()
        layout = [int(ls.strip()) for ls in layouts.split(',')]
        values = o['value']
        od = _gen_obsd_from_datad(o['data'])

        ret = []
        for i in range(np.prod(layout)):
            ret.append(Obs([di[:, i] + values[i] for di in od['deltas']], od['names'], idl=od['idl']))
            ret[-1].reweighted = o.get('reweighted', False)
            ret[-1].is_merged = od['is_merged']
            ret[-1].tag = o.get('tag', '')
        return np.reshape(ret, layout)

    if not fname.endswith('.json') and not fname.endswith('.gz'):
        fname += '.json'
    if not fname.endswith('.gz'):
        fname += '.gz'
    with gzip.open(fname, 'r') as fin:
        d = json.loads(fin.read().decode('utf-8'))
    prog = d.get('program', '')
    version = d.get('version', '')
    who = d.get('who', '')
    date = d.get('date', '')
    host = d.get('host', '')
    if prog and verbose:
        print('Data has been written using %s.' % (prog))
    if version and verbose:
        print('Format version %s' % (version))
    if np.any([who, date, host] and verbose):
        print('Written by %s on %s on host %s' % (who, date, host))
    description = d.get('description', '')
    if description and verbose:
        print()
        print(description)
    obsdata = d['obsdata']
    ol = []
    for io in obsdata:
        if io['type'] == 'Obs':
            ol.append(get_Obs_from_dict(io))
        elif io['type'] == 'List':
            ol.append(get_List_from_dict(io))
        elif io['type'] == 'Array':
            ol.append(get_Array_from_dict(io))
    return ol
