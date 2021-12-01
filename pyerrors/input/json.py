import json
import gzip
from ..obs import Obs
import getpass
import socket
import datetime
from .. import version as pyerrorsversion
import platform
import numpy as np
import warnings


def create_json_string(ol, fname, description='', indent=1):
    """Generate the string for the export of a list of Obs or structures containing Obs
    to a .json(.gz) file

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
        if o.tag:
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
        if ol[0].tag:
            d['tag'] = ol[0].tag
            if isinstance(ol[0].tag, str):
                if len(set([o.tag for o in ol])) > 1:
                    d['tag'] = ''
                    for o in ol:
                        d['tag'] += '%s\n' % (o.tag)
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
        d['layout'] = str(oa.shape).lstrip('(').rstrip(')').rstrip(',')
        if ol[0].tag:
            d['tag'] = ol[0].tag
            if isinstance(ol[0].tag, str):
                if len(set([o.tag for o in ol])) > 1:
                    d['tag'] = ''
                    for o in ol:
                        d['tag'] += '%s\n' % (o.tag)
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

    jsonstring = json.dumps(d, indent=indent, cls=my_encoder, ensure_ascii=False)
    # workaround for un-indentation of delta lists
    jsonstring = jsonstring.replace('    "[', '    [').replace(']",', '],').replace(']"\n', ']\n')

    return jsonstring


def dump_to_json(ol, fname, description='', indent=1, gz=True):
    """Export a list of Obs or structures containing Obs to a .json(.gz) file

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
    gz : bool
        If True, the output is a gzipped json. If False, the output is a json file.
    """

    jsonstring = create_json_string(ol, fname, description, indent)

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


def load_json(fname, verbose=True, gz=True, full_output=False):
    """Import a list of Obs or structures containing Obs from a .json.gz file.
    The following structures are supported: Obs, list, np.ndarray
    If the list contains only one element, it is unpacked from the list.

    Parameters
    -----------------
    fname : str
        Filename of the input file
    verbose : bool
        Print additional information that was written to the file.
    gz : bool
        If True, assumes that data is gzipped. If False, assumes JSON file.
    full_output : bool
        If True, a dict containing auxiliary information and the data is returned.
        If False, only the data is returned.
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
                retd['deltas'].append(np.array([di[1:] for di in rep['deltas']]))
                retd['is_merged'][rep['name']] = rep.get('is_merged', False)
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
        ret.tag = o.get('tag', None)
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
            ret[-1].tag = o.get('tag', None)
        return ret

    def get_Array_from_dict(o):
        layouts = o.get('layout', '1').strip()
        layout = [int(ls.strip()) for ls in layouts.split(',') if len(ls) > 0]
        values = o['value']
        od = _gen_obsd_from_datad(o['data'])

        ret = []
        for i in range(np.prod(layout)):
            ret.append(Obs([di[:, i] + values[i] for di in od['deltas']], od['names'], idl=od['idl']))
            ret[-1].reweighted = o.get('reweighted', False)
            ret[-1].is_merged = od['is_merged']
            ret[-1].tag = o.get('tag', None)
        return np.reshape(ret, layout)

    if not fname.endswith('.json') and not fname.endswith('.gz'):
        fname += '.json'
    if gz:
        if not fname.endswith('.gz'):
            fname += '.gz'
        with gzip.open(fname, 'r') as fin:
            d = json.loads(fin.read().decode('utf-8'))
    else:
        if fname.endswith('.gz'):
            warnings.warn("Trying to read from %s without unzipping!" % fname, UserWarning)
        with open(fname, 'r', encoding='utf-8') as fin:
            d = json.loads(fin.read())

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
        print('Description: ', description)
    obsdata = d['obsdata']
    ol = []
    for io in obsdata:
        if io['type'] == 'Obs':
            ol.append(get_Obs_from_dict(io))
        elif io['type'] == 'List':
            ol.append(get_List_from_dict(io))
        elif io['type'] == 'Array':
            ol.append(get_Array_from_dict(io))

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
