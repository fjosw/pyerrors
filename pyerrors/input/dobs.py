from collections import defaultdict
import gzip
import lxml.etree as et
import getpass
import socket
import datetime
import json
import warnings
import numpy as np
from ..obs import Obs
from ..obs import _merge_idx
from ..covobs import Covobs
from .. import version as pyerrorsversion


# standard pobs
# https://stackoverflow.com/a/10076823
def _etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(_etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#data'] = [text]
        else:
            d[t.tag] = text
    return d


def _get_data_from_root(d, root):
    tree = et.ElementTree(root)
    nl = []
    pl = []
    for node in root.findall('.//array'):
        nl.append(node)
        pl.append(tree.getpath(node))

    for nd in range(len(nl)):
        s = pl[nd].split('/')
        ad = d
        for i in range(len(s) - 1):
            if len(s[i]) > 0:
                ad = ad[s[i]]
        nr = int(ad['nr'])
        ad = ad['array']
        if '[' in s[-1]:
            ai = int(s[-1].lstrip('array[').rstrip(']')) - 1
            ad = ad[ai]
        if nr == 1:
            dats = [t for t in nl[nd].itertext()][0]
        else:
            dats = [t for t in nl[nd].itertext()][-1]
        dats = [o.strip() for o in dats.split('\n')]

        layout = ad['layout']
        sl = layout.split()
        datl = []
        idx = []
        for di in dats:
            if len(di) > 0:
                ds = di.split()
                if sl[1] == 'i':
                    i = int(ds[0])
                    idx.append(i)
                    df = np.asarray(ds[1:], dtype=np.float64)
                else:
                    df = np.asarray(ds, dtype=np.float64)
                nfloat = int(sl[-1].lstrip('f'))
                assert len(df) == nfloat
                datl.append(df)
        ad['idx'] = idx
        ad['#data'] = np.transpose(np.array(datl))
    return d


# https://stackoverflow.com/a/9808122
def _find(key, value):
    for k, v in (value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else []):
        if k == key:
            yield v
        elif isinstance(v, (dict, list)):
            for result in _find(key, v):
                yield result


# generate obs from all 'array' entries inside a dict
def _gen_obs_from_dict(d):
    pobs_list = [a for a in _find('array', d)]
    ol = []
    for po in pobs_list:
        if isinstance(po, list):
            names = []
            idl = [poi['idx'] for poi in po]
            names = [poi['id'] for poi in po]
            for i in range(len(po[0])):
                tag = po[0]['symbol'].split()[-8 + i]
                data = [poi['#data'][i] for poi in po]
                ol.append(Obs(data, names, idl=idl))
                ol[-1].tag = tag
                print(tag)
    return ol


# generate obs from entries inside a 'pobs'-dict
def _gen_obs_from_pobs_dict(d):
    po = d['array']
    if isinstance(po, dict):
        po = [po]
    enstag = d['enstag']
    nr = int(d['nr'])
    assert len(po) == nr, 'Lenght of array %d does not match nr=%d' % (len(po), nr)
    ol = []
    names = []
    idl = [poi['idx'] for poi in po]
    names = [poi['id'] for poi in po]
    names = [name.replace(enstag, enstag + '|') for name in names]
    if len(po[0]['symbol'].split()) >= len(po[0]):
        gettag = True
    else:
        gettag = False
    nobs = np.shape(po[0]['#data'])[0]
    print(nobs, 'observables')
    for i in range(nobs):
        if gettag:
            tag = po[0]['symbol'].split()[i + 1]
        else:
            tag = po[0]['symbol']
        data = [poi['#data'][i] for poi in po]
        ol.append(Obs(data, names, idl=idl))
        ol[-1].tag = tag
    return ol


def _read_pobs_traditional(fname, full_output=False):
    """Import a list of Obs from an xml.gz file in the Zeuthen pobs format,
    this does not include pobs that are written using the dobs package.

    Parameters
    ----------
    fname : str
        Filename of the input file.
    full_output : bool
        If True, a dict containing auxiliary information and the data is returned.
        If False, only the data is returned as list.
    """
    # open and read gzipped xml file
    infile = gzip.open(fname)
    content = infile.read()

    # parse xml file content
    root = et.fromstring(content)

    # parse to dict
    d = _etree_to_dict(root)

    # get the actual data from the xml file
    d = _get_data_from_root(d, root)

    # pobs dict:
    pod = d[list(d.keys())[0]]['pobs']

    # convert data to Obs
    ol = _gen_obs_from_pobs_dict(pod)

    if full_output:
        retd = {}
        retd['obsdata'] = ol
        for k in pod.keys():
            if k == 'spec' and isinstance(pod[k], dict):
                retd[k] = ''
                if '#data' in pod[k]:
                    if isinstance(pod[k]['#data'], list):
                        for el in pod[k]['#data']:
                            retd[k] += el
                    else:
                        retd[k] = pod[k]['#data']
                for li in pod[k]:
                    if not li == '#data':
                        retd[k] += '\n<%s>\n  ' % (li) + pod[k][li] + '\n<%s%s>' % ('/', li)
            elif not k == 'array':
                retd[k] = pod[k]

        return retd
    else:
        return ol


def _dict_to_xmlstring(d):
    if isinstance(d, dict):
        iters = ''
        for k in d:
            if k.startswith('#'):
                for la in d[k]:
                    iters += la
                iters = '<array>\n' + iters + '<%sarray>\n' % ('/')
                return iters
            if isinstance(d[k], dict):
                iters += '<%s>\n' % (k) + _dict_to_xmlstring(d[k]) + '<%s%s>\n' % ('/', k)
            elif isinstance(d[k], str):
                if len(d[k]) > 100:
                    iters += '<%s>\n ' % (k) + d[k] + ' \n<%s%s>\n' % ('/', k)
                else:
                    iters += '<%s> ' % (k) + d[k] + ' <%s%s>\n' % ('/', k)
            elif isinstance(d[k], list):
                for i in range(len(d[k])):
                    iters += _dict_to_xmlstring(d[k][i])
            elif not d[k]:
                return '\n'
            else:
                raise Exception('Type', type(d[k]), 'not supported in export!')
    else:
        raise Exception('Type', type(d), 'not supported in export!')
    return iters


def _dict_to_xmlstring_spaces(d, space='  '):
    s = _dict_to_xmlstring(d)
    o = ''
    c = 0
    cm = False
    for li in s.split('\n'):
        if li.startswith('<%s' % ('/')):
            c -= 1
            cm = True
        for i in range(c):
            o += space
        o += li + '\n'
        if li.startswith('<') and not cm:
            if not '<%s' % ('/') in li:
                c += 1
        cm = False
    return o


def create_pobs_string(obsl, name, spec='', origin='', symbol=[]):
    """Export a list of Obs or structures containing Obs to an xml string
    according to the Zeuthen pobs format.

    Tags are not written or recovered automatically.

    Parameters
    ----------
    obsl : list
        List of Obs that will be exported.
        The Obs inside a structure have to be defined on the same ensemble.
    name : str
        The name of the observable.
    spec : str
        Optional string that describes the contents of the file.
    origin : str
        Specify where the data has its origin.
    symbol : list
        A list of symbols that describe the observables to be written. May be empty.
    """

    od = {}
    ename = obsl[0].e_names[0]
    names = list(obsl[0].deltas.keys())
    nr = len(names)
    onames = [name.replace('|', '') for name in names]
    for o in obsl:
        if len(o.e_names) != 1:
            raise Exception('You try to export dobs to obs!')
        if o.e_names[0] != ename:
            raise Exception('You try to export dobs to obs!')
        if len(o.deltas.keys()) != nr:
            raise Exception('Incompatible obses in list')
    od['observables'] = {}
    od['observables']['schema'] = {'name': 'lattobs', 'version': '1.0'}
    od['observables']['origin'] = {
        'who': getpass.getuser(),
        'date': str(datetime.datetime.now())[:-7],
        'host': socket.gethostname(),
        'tool': {'name': 'pyerrors', 'version': pyerrorsversion.__version__}}
    od['observables']['pobs'] = {}
    pd = od['observables']['pobs']
    pd['spec'] = spec
    pd['origin'] = origin
    pd['name'] = name
    pd['enstag'] = ename
    pd['nr'] = '%d' % (nr)
    pd['array'] = []
    osymbol = 'cfg'
    if not isinstance(symbol, list):
        raise Exception('Symbol has to be a list!')
    if not (len(symbol) == 0 or len(symbol) == len(obsl)):
        raise Exception('Symbol has to be a list of lenght 0 or %d!' % (len(obsl)))
    for s in symbol:
        osymbol += ' %s' % s
    for r in range(nr):
        ad = {}
        ad['id'] = onames[r]
        Nconf = len(obsl[0].deltas[names[r]])
        layout = '%d i f%d' % (Nconf, len(obsl))
        ad['layout'] = layout
        ad['symbol'] = osymbol
        data = []
        for c in range(Nconf):
            data += '%d ' % obsl[0].idl[names[r]][c]
            for o in obsl:
                num = o.deltas[names[r]][c] + o.r_values[names[r]]
                if num == 0:
                    data += '0 '
                else:
                    data += '%1.16e ' % (num)
            data += '\n'
        ad['#data'] = data
        pd['array'].append(ad)

    rs = '<?xml version="1.0" encoding="utf-8"?>\n' + _dict_to_xmlstring_spaces(od)
    return rs


def write_pobs(obsl, fname, name, spec='', origin='', symbol=[], gz=True):
    """Export a list of Obs or structures containing Obs to a .xml.gz file
    according to the Zeuthen pobs format.

    Tags are not written or recovered automatically.

    Parameters
    ----------
    obsl : list
        List of Obs that will be exported.
        The Obs inside a structure have to be defined on the same ensemble.
    fname : str
        Filename of the output file.
    name : str
        The name of the observable.
    spec : str
        Optional string that describes the contents of the file.
    origin : str
        Specify where the data has its origin.
    symbol : list
        A list of symbols that describe the observables to be written. May be empty.
    gz : bool
        If True, the output is a gzipped json. If False, the output is a json file.
    """
    pobsstring = create_pobs_string(obsl, name, spec, origin, symbol)

    if not fname.endswith('.xml') and not fname.endswith('.gz'):
        fname += '.xml'

    if gz:
        if not fname.endswith('.gz'):
            fname += '.gz'

        fp = gzip.open(fname, 'wb')
        fp.write(pobsstring.encode('utf-8'))
    else:
        fp = open(fname, 'w', encoding='utf-8')
        fp.write(pobsstring)
    fp.close()


def _import_data(string):
    return json.loads("[" + ",".join(string.replace(' +', ' ').split()) + "]")


def _check(condition):
    if not condition:
        raise Exception("XML file format not supported")


def _find_tag(dat, tag):
    for i in range(len(dat)):
        if dat[i].tag == tag:
            return i
    raise Exception('Tag %s not in data!' % (tag))


def _import_array(arr):
    name = arr[_find_tag(arr, 'id')].text.strip()
    index = _find_tag(arr, 'layout')
    try:
        sindex = _find_tag(arr, 'symbol')
    except:
        sindex = 0
    if sindex > index:
        tmp = _import_data(arr[sindex].tail)
    else:
        tmp = _import_data(arr[index].tail)

    li = arr[index].text.strip()
    m = li.split()
    if m[1] == "i" and m[2][0] == "f":
        nc = int(m[0])
        na = int(m[2].lstrip('f'))
        _dat = []
        mask = []
        for a in range(na):
            mask += [a]
            _dat += [np.array(tmp[1 + a:: na + 1])]
        _check(len(tmp[0:: na + 1]) == nc)
        return [name, tmp[0:: na + 1], mask, _dat]
    elif m[1][0] == 'f' and len(m) < 3:
        sh = (int(m[0]), int(m[1].lstrip('f')))
        return np.reshape(tmp, sh)
    elif any(['f' in s for s in m]):
        for si in range(len(m)):
            if m[si] == 'f':
                break
        sh = [int(m[i]) for i in range(si)]
        return np.reshape(tmp, sh)
    else:
        print(name, m)
        _check(False)


def _import_rdata(rd):
    name, idx, mask, deltas = _import_array(rd)
    return deltas, name, idx


def _import_cdata(cd):
    _check(cd[0].tag == "id")
    _check(cd[1][0].text.strip() == "cov")
    cov = _import_array(cd[1])
    grad = _import_array(cd[2])
    return cd[0].text.strip(), cov, grad


def read_pobs(fname, full_output=False, gz=True):
    """Import a list of Obs from an xml.gz file in the Zeuthen pobs format.

    Tags are not written or recovered automatically.

    Parameters
    ----------
    fname : str
        Filename of the input file.
    full_output : bool
        If True, a dict containing auxiliary information and the data is returned.
        If False, only the data is returned as list.
    """

    if not fname.endswith('.xml') and not fname.endswith('.gz'):
        fname += '.xml'
    if gz:
        if not fname.endswith('.gz'):
            fname += '.gz'
        with gzip.open(fname, 'r') as fin:
            content = fin.read()
    else:
        if fname.endswith('.gz'):
            warnings.warn("Trying to read from %s without unzipping!" % fname, UserWarning)
        with open(fname, 'r') as fin:
            content = fin.read()

    # parse xml file content
    root = et.fromstring(content)

    _check(root[2].tag == 'pobs')
    pobs = root[2]

    if root.tag == 'observables' and False:
        _check(root[0].tag in ['schema'])
        version = root[0][1].text.strip()

        _check(root[1].tag == 'origin')
        file_origin = _etree_to_dict(root[1])['origin']

        res = _read_pobs_traditional(fname)

    else:
        # _check(root.tag in ['OBSERVABLES'])
        # _check(root[0].tag in ['SCHEMA'])
        version = root[0][1].text.strip()

        _check(root[1].tag == 'origin')
        file_origin = _etree_to_dict(root[1])['origin']

        deltas = []
        names = []
        idl = []
        for i in range(5, len(pobs)):
            delta, name, idx = _import_rdata(pobs[i])
            deltas.append(delta)
            names.append(name)
            idl.append(idx)
        res = [Obs([d[i] for d in deltas], names, idl=idl) for i in range(len(deltas[0]))]

    descriptiond = {}
    for i in range(4):
        descriptiond[pobs[i].tag] = pobs[i].text.strip()

    _check(pobs[4].tag == "nr")

    _check(pobs[5].tag == 'array')
    if pobs[5][1].tag == 'symbol':
        symbol = pobs[5][1].text.strip()
        descriptiond['symbol'] = symbol

    if full_output:
        retd = {}
        tool = file_origin.get('tool', None)
        if tool:
            program = tool['name'] + ' ' + tool['version']
        else:
            program = ''
        retd['program'] = program
        retd['version'] = version
        retd['who'] = file_origin['who']
        retd['date'] = file_origin['date']
        retd['host'] = file_origin['host']
        retd['description'] = descriptiond
        retd['obsdata'] = res
        return retd
    else:
        return res


# this is based on Mattia Bruno's implementation at https://github.com/mbruno46/pyobs/blob/master/pyobs/IO/xml.py
def import_dobs_string(content, noempty=False, full_output=False):
    """Import a list of Obs from an xml.gz file in the Zeuthen dobs format.

    Tags are not written or recovered automatically.

    Parameters
    ----------
    content : str
        XML string containing the data
    noemtpy : bool
        If True, ensembles with no contribution to the Obs are not included.
        If False, ensembles are included as written in the file.
    full_output : bool
        If True, a dict containing auxiliary information and the data is returned.
        If False, only the data is returned as list.
    """

    root = et.fromstring(content)

    _check(root.tag == 'OBSERVABLES')
    _check(root[0].tag == 'SCHEMA')
    version = root[0][1].text.strip()

    _check(root[1].tag == 'origin')
    file_origin = _etree_to_dict(root[1])['origin']

    _check(root[2].tag == 'dobs')

    dobs = root[2]

    descriptiond = {}
    for i in range(3):
        descriptiond[dobs[i].tag] = dobs[i].text.strip()

    _check(dobs[3].tag == 'array')

    symbol = []
    if dobs[3][1].tag == 'symbol':
        symbol = dobs[3][1].text.strip()
        descriptiond['symbol'] = symbol
    mean = _import_array(dobs[3])[0]

    _check(dobs[4].tag == "ne")
    ne = int(dobs[4].text.strip())
    _check(dobs[5].tag == "nc")
    nc = int(dobs[5].text.strip())

    idld = {}
    deltad = {}
    covd = {}
    gradd = {}
    names = []
    e_names = []
    for k in range(6, len(list(dobs))):
        if dobs[k].tag == "edata":
            _check(dobs[k][0].tag == "enstag")
            ename = dobs[k][0].text.strip()
            e_names.append(ename)
            _check(dobs[k][1].tag == "nr")
            R = int(dobs[k][1].text.strip())
            for i in range(2, 2 + R):
                deltas, rname, idx = _import_rdata(dobs[k][i])
                if rname != ename:
                    rname = rname[:len(ename)] + '|' + rname[len(ename):]
                idld[rname] = idx
                deltad[rname] = deltas
                names.append(rname)
        elif dobs[k].tag == "cdata":
            cname, cov, grad = _import_cdata(dobs[k])
            covd[cname] = cov
            if grad.shape[1] == 1:
                gradd[cname] = [grad for i in range(len(mean))]
            else:
                gradd[cname] = grad.T
        else:
            _check(False)
    names = list(set(names))

    for name in names:
        for i in range(len(deltad[name])):
            deltad[name][i] = np.array(deltad[name][i]) + mean[i]
    res = []
    for i in range(len(mean)):
        deltas = []
        idl = []
        obs_names = []
        for name in names:
            h = np.unique(deltad[name][i])
            if len(h) == 1 and np.all(h == mean[i]) and noempty:
                continue
            deltas.append(deltad[name][i])
            obs_names.append(name)
            idl.append(idld[name])
        res.append(Obs(deltas, obs_names, idl=idl))

    _check(len(e_names) == ne)

    cnames = list(covd.keys())
    for i in range(len(res)):
        new_covobs = {name: Covobs(0, covd[name], name, grad=gradd[name][i]) for name in cnames}
        if noempty:
            for name in cnames:
                if np.all(new_covobs[name].grad == 0):
                    del new_covobs[name]
            cnames_loc = list(new_covobs.keys())
        else:
            cnames_loc = cnames
        for name in cnames_loc:
            res[i].names.append(name)
            res[i].shape[name] = 1
            res[i].idl[name] = []
        res[i]._covobs = new_covobs

    if symbol:
        for i in range(len(res)):
            res[i].tag = symbol[i]
            if res[i].tag == 'None':
                res[i].tag = None
    if not noempty:
        _check(len(res[0].covobs.keys()) == nc)
    if full_output:
        retd = {}
        tool = file_origin.get('tool', None)
        if tool:
            program = tool['name'] + ' ' + tool['version']
        else:
            program = ''
        retd['program'] = program
        retd['version'] = version
        retd['who'] = file_origin['who']
        retd['date'] = file_origin['date']
        retd['host'] = file_origin['host']
        retd['description'] = descriptiond
        retd['obsdata'] = res
        return retd
    else:
        return res


def read_dobs(fname, noempty=False, full_output=False, gz=True):
    """Import a list of Obs from an xml.gz file in the Zeuthen dobs format.

    Tags are not written or recovered automatically.

    Parameters
    ----------
    fname : str
        Filename of the input file.
    noemtpy : bool
        If True, ensembles with no contribution to the Obs are not included.
        If False, ensembles are included as written in the file.
    full_output : bool
        If True, a dict containing auxiliary information and the data is returned.
        If False, only the data is returned as list.
    gz : bool
        If True, assumes that data is gzipped. If False, assumes XML file.
    """

    if not fname.endswith('.xml') and not fname.endswith('.gz'):
        fname += '.xml'
    if gz:
        if not fname.endswith('.gz'):
            fname += '.gz'
        with gzip.open(fname, 'r') as fin:
            content = fin.read().decode('utf-8')
    else:
        if fname.endswith('.gz'):
            warnings.warn("Trying to read from %s without unzipping!" % fname, UserWarning)
        with open(fname, 'r', encoding='utf-8') as fin:
            content = fin.read()

    # open and read gzipped xml file
    infile = gzip.open(fname)
    content = infile.read()

    return import_dobs_string(content, noempty, full_output)


def _dobsdict_to_xmlstring(d):
    if isinstance(d, dict):
        iters = ''
        for k in d:
            if k.startswith('#value'):
                for li in d[k]:
                    iters += li
                return iters + '\n'
            elif k.startswith('#'):
                for li in d[k]:
                    iters += li
                iters = '<array>\n' + iters + '<%sarray>\n' % ('/')
                return iters
            if isinstance(d[k], dict):
                iters += '<%s>\n' % (k) + _dobsdict_to_xmlstring(d[k]) + '<%s%s>\n' % ('/', k)
            elif isinstance(d[k], str):
                if len(d[k]) > 100:
                    iters += '<%s>\n ' % (k) + d[k] + ' \n<%s%s>\n' % ('/', k)
                else:
                    iters += '<%s> ' % (k) + d[k] + ' <%s%s>\n' % ('/', k)
            elif isinstance(d[k], list):
                tmps = ''
                if k in ['edata', 'cdata']:
                    for i in range(len(d[k])):
                        tmps += '<%s>\n' % (k) + _dobsdict_to_xmlstring(d[k][i]) + '</%s>\n' % (k)
                else:
                    for i in range(len(d[k])):
                        tmps += _dobsdict_to_xmlstring(d[k][i])
                iters += tmps
            elif isinstance(d[k], (int, float)):
                iters += '<%s> ' % (k) + str(d[k]) + ' <%s%s>\n' % ('/', k)
            elif not d[k]:
                return '\n'
            else:
                raise Exception('Type', type(d[k]), 'not supported in export!')
    else:
        raise Exception('Type', type(d), 'not supported in export!')
    return iters


def _dobsdict_to_xmlstring_spaces(d, space='  '):
    s = _dobsdict_to_xmlstring(d)
    o = ''
    c = 0
    cm = False
    for li in s.split('\n'):
        if li.startswith('<%s' % ('/')):
            c -= 1
            cm = True
        for i in range(c):
            o += space
        o += li + '\n'
        if li.startswith('<') and not cm:
            if not '<%s' % ('/') in li:
                c += 1
        cm = False
    return o


def create_dobs_string(obsl, name, spec='dobs v1.0', origin='', symbol=[], who=None):
    """Generate the string for the export of a list of Obs or structures containing Obs
    to a .xml.gz file according to the Zeuthen dobs format.

    Tags are not written or recovered automatically.

    Parameters
    ----------
    obsl : list
        List of Obs that will be exported.
        The Obs inside a structure do not have to be defined on the same set of configurations,
        but the storage requirement is increased, if this is not the case.
    name : str
        The name of the observable.
    spec : str
        Optional string that describes the contents of the file.
    origin : str
        Specify where the data has its origin.
    symbol : list
        A list of symbols that describe the observables to be written. May be empty.
    who : str
        Provide the name of the person that exports the data.
    """
    od = {}
    r_names = []
    for o in obsl:
        r_names += [name for name in o.names if name.split('|')[0] in o.mc_names]
    r_names = sorted(set(r_names))
    mc_names = sorted(set([n.split('|')[0] for n in r_names]))
    ne = len(set(mc_names))
    cov_names = []
    for o in obsl:
        cov_names += list(o.cov_names)
    cov_names = sorted(set(cov_names))
    nc = len(set(cov_names))
    od['OBSERVABLES'] = {}
    od['OBSERVABLES']['SCHEMA'] = {'NAME': 'lattobs', 'VERSION': '1.0'}
    if who is None:
        who = getpass.getuser()
    od['OBSERVABLES']['origin'] = {
        'who': who,
        'date': str(datetime.datetime.now())[:-7],
        'host': socket.gethostname(),
        'tool': {'name': 'pyerrors', 'version': pyerrorsversion.__version__}}
    od['OBSERVABLES']['dobs'] = {}
    pd = od['OBSERVABLES']['dobs']
    pd['spec'] = spec
    pd['origin'] = origin
    pd['name'] = name
    pd['array'] = {}
    pd['array']['id'] = 'val'
    pd['array']['layout'] = '1 f%d' % (len(obsl))
    osymbol = ''
    if symbol:
        if not isinstance(symbol, list):
            raise Exception('Symbol has to be a list!')
        if not (len(symbol) == 0 or len(symbol) == len(obsl)):
            raise Exception('Symbol has to be a list of lenght 0 or %d!' % (len(obsl)))
        osymbol = symbol[0]
        for s in symbol[1:]:
            osymbol += ' %s' % s
        pd['array']['symbol'] = osymbol

    pd['array']['#values'] = ['  '.join(['%1.16e' % o.value for o in obsl])]
    pd['ne'] = '%d' % (ne)
    pd['nc'] = '%d' % (nc)
    pd['edata'] = []
    for name in mc_names:
        ed = {}
        ed['enstag'] = name
        onames = sorted([n for n in r_names if (n.startswith(name + '|') or n == name)])
        nr = len(onames)
        ed['nr'] = nr
        ed[''] = []

        for r in range(nr):
            ad = {}
            repname = onames[r]
            ad['id'] = repname.replace('|', '')
            idx = _merge_idx([o.idl.get(repname, []) for o in obsl])
            Nconf = len(idx)
            layout = '%d i f%d' % (Nconf, len(obsl))
            ad['layout'] = layout
            data = ''
            counters = [0 for o in obsl]
            for ci in idx:
                data += '%d ' % ci
                for oi in range(len(obsl)):
                    o = obsl[oi]
                    if repname in o.idl:
                        if counters[oi] < 0:
                            data += '0 '
                            continue
                        if o.idl[repname][counters[oi]] == ci:
                            # c = o.idl[repname][counters[oi]]
                            num = o.deltas[repname][counters[oi]]
                            if num == 0:
                                data += '0 '
                            else:
                                data += '%1.16e ' % (num)
                            counters[oi] += 1
                            if counters[oi] >= len(o.idl[repname]):
                                counters[oi] = -1
                        else:
                            data += '0 '
                    else:
                        data += '0 '
                data += '\n'
            ad['#data'] = data
            ed[''].append(ad)
        pd['edata'].append(ed)

        allcov = {}
        for o in obsl:
            for name in o.cov_names:
                if name in allcov:
                    if not np.array_equal(allcov[name], o.covobs[name].cov):
                        raise Exception('Inconsistent covariance matrices for %s!' % (name))
                else:
                    allcov[name] = o.covobs[name].cov
        pd['cdata'] = []
        for name in cov_names:
            cd = {}
            cd['id'] = name

            covd = {'id': 'cov'}
            if allcov[name].shape == ():
                ncov = 1
                covd['layout'] = '1 1 f'
                covd['#data'] = '%1.14e' % (allcov[name])
            else:
                shape = allcov[name].shape
                assert (shape[0] == shape[1])
                ncov = shape[0]
                covd['layout'] = '%d %d f' % (ncov, ncov)
                ds = ''
                for i in range(ncov):
                    for j in range(ncov):
                        val = allcov[name][i][j]
                        if val == 0:
                            ds += '0 '
                        else:
                            ds += '%1.14e ' % (val)
                    ds += '\n'
                covd['#data'] = ds

            gradd = {'id': 'grad'}
            gradd['layout'] = '%d f%d' % (ncov, len(obsl))
            ds = ''
            for i in range(ncov):
                for o in obsl:
                    if name in o.covobs:
                        val = o.covobs[name].grad[i]
                        if val != 0:
                            ds += '%1.14e ' % (val)
                        else:
                            ds += '0 '
                    else:
                        ds += '0 '
            gradd['#data'] = ds
            cd['array'] = [covd, gradd]
            pd['cdata'].append(cd)

    rs = '<?xml version="1.0" encoding="utf-8"?>\n' + _dobsdict_to_xmlstring_spaces(od)

    return rs


def write_dobs(obsl, fname, name, spec='dobs v1.0', origin='', symbol=[], who=None, gz=True):
    """Export a list of Obs or structures containing Obs to a .xml.gz file
    according to the Zeuthen dobs format.

    Tags are not written or recovered automatically.

    Parameters
    ----------
    obsl : list
        List of Obs that will be exported.
        The Obs inside a structure do not have to be defined on the same set of configurations,
        but the storage requirement is increased, if this is not the case.
    fname : str
        Filename of the output file.
    name : str
        The name of the observable.
    spec : str
        Optional string that describes the contents of the file.
    origin : str
        Specify where the data has its origin.
    symbol : list
        A list of symbols that describe the observables to be written. May be empty.
    who : str
        Provide the name of the person that exports the data.
    gz : bool
        If True, the output is a gzipped json. If False, the output is a json file.
    """

    dobsstring = create_dobs_string(obsl, name, spec, origin, symbol, who)

    if not fname.endswith('.xml') and not fname.endswith('.gz'):
        fname += '.xml'

    if gz:
        if not fname.endswith('.gz'):
            fname += '.gz'

        fp = gzip.open(fname, 'wb')
        fp.write(dobsstring.encode('utf-8'))
    else:
        fp = open(fname, 'w', encoding='utf-8')
        fp.write(dobsstring)
    fp.close()
