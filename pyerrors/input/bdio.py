import ctypes
import hashlib
import autograd.numpy as np  # Thinly-wrapped numpy
from ..obs import Obs


def read_ADerrors(file_path, bdio_path='./libbdio.so', **kwargs):
    """ Extract generic MCMC data from a bdio file

    read_ADerrors requires bdio to be compiled into a shared library. This can be achieved by
    adding the flag -fPIC to CC and changing the all target to

    all:		bdio.o $(LIBDIR)
                gcc -shared -Wl,-soname,libbdio.so -o $(BUILDDIR)/libbdio.so $(BUILDDIR)/bdio.o
                cp $(BUILDDIR)/libbdio.so $(LIBDIR)/

    Parameters
    ----------
    file_path -- path to the bdio file
    bdio_path -- path to the shared bdio library libbdio.so (default ./libbdio.so)
    """
    bdio = ctypes.cdll.LoadLibrary(bdio_path)

    bdio_open = bdio.bdio_open
    bdio_open.restype = ctypes.c_void_p

    bdio_close = bdio.bdio_close
    bdio_close.restype = ctypes.c_int
    bdio_close.argtypes = [ctypes.c_void_p]

    bdio_seek_record = bdio.bdio_seek_record
    bdio_seek_record.restype = ctypes.c_int
    bdio_seek_record.argtypes = [ctypes.c_void_p]

    bdio_get_rlen = bdio.bdio_get_rlen
    bdio_get_rlen.restype = ctypes.c_int
    bdio_get_rlen.argtypes = [ctypes.c_void_p]

    bdio_get_ruinfo = bdio.bdio_get_ruinfo
    bdio_get_ruinfo.restype = ctypes.c_int
    bdio_get_ruinfo.argtypes = [ctypes.c_void_p]

    bdio_read = bdio.bdio_read
    bdio_read.restype = ctypes.c_size_t
    bdio_read.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p]

    bdio_read_f64 = bdio.bdio_read_f64
    bdio_read_f64.restype = ctypes.c_size_t
    bdio_read_f64.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

    bdio_read_int32 = bdio.bdio_read_int32
    bdio_read_int32.restype = ctypes.c_size_t
    bdio_read_int32.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

    b_path = file_path.encode('utf-8')
    read = 'r'
    b_read = read.encode('utf-8')

    fbdio = bdio_open(ctypes.c_char_p(b_path), ctypes.c_char_p(b_read), None)

    return_list = []

    print('Reading of bdio file started')
    while True:
        bdio_seek_record(fbdio)
        ruinfo = bdio_get_ruinfo(fbdio)

        if ruinfo == 7:
            print('MD5sum found')  # For now we just ignore these entries and do not perform any checks on them
            continue

        if ruinfo < 0:
            # EOF reached
            break
        bdio_get_rlen(fbdio)

        def read_c_double():
            d_buf = ctypes.c_double
            pd_buf = d_buf()
            ppd_buf = ctypes.c_void_p(ctypes.addressof(pd_buf))
            bdio_read_f64(ppd_buf, ctypes.c_size_t(8), ctypes.c_void_p(fbdio))
            return pd_buf.value

        mean = read_c_double()
        print('mean', mean)

        def read_c_size_t():
            d_buf = ctypes.c_size_t
            pd_buf = d_buf()
            ppd_buf = ctypes.c_void_p(ctypes.addressof(pd_buf))
            bdio_read_int32(ppd_buf, ctypes.c_size_t(4), ctypes.c_void_p(fbdio))
            return pd_buf.value

        neid = read_c_size_t()
        print('neid', neid)

        ndata = []
        for index in range(neid):
            ndata.append(read_c_size_t())
        print('ndata', ndata)

        nrep = []
        for index in range(neid):
            nrep.append(read_c_size_t())
        print('nrep', nrep)

        vrep = []
        for index in range(neid):
            vrep.append([])
            for jndex in range(nrep[index]):
                vrep[-1].append(read_c_size_t())
        print('vrep', vrep)

        ids = []
        for index in range(neid):
            ids.append(read_c_size_t())
        print('ids', ids)

        nt = []
        for index in range(neid):
            nt.append(read_c_size_t())
        print('nt', nt)

        zero = []
        for index in range(neid):
            zero.append(read_c_double())
        print('zero', zero)

        four = []
        for index in range(neid):
            four.append(read_c_double())
        print('four', four)

        d_buf = ctypes.c_double * np.sum(ndata)
        pd_buf = d_buf()
        ppd_buf = ctypes.c_void_p(ctypes.addressof(pd_buf))
        bdio_read_f64(ppd_buf, ctypes.c_size_t(8 * np.sum(ndata)), ctypes.c_void_p(fbdio))
        delta = pd_buf[:]

        samples = np.split(np.asarray(delta) + mean, np.cumsum([a for su in vrep for a in su])[:-1])
        no_reps = [len(o) for o in vrep]
        assert len(ids) == len(no_reps)
        tmp_names = []
        ens_length = max([len(str(o)) for o in ids])
        for loc_id, reps in zip(ids, no_reps):
            for index in range(reps):
                missing_chars = ens_length - len(str(loc_id))
                tmp_names.append(str(loc_id) + ' ' * missing_chars + '|r' + '{0:03d}'.format(index))

        return_list.append(Obs(samples, tmp_names))

    bdio_close(fbdio)
    print()
    print(len(return_list), 'observable(s) extracted.')
    return return_list


def write_ADerrors(obs_list, file_path, bdio_path='./libbdio.so', **kwargs):
    """ Write Obs to a bdio file according to ADerrors conventions

    read_mesons requires bdio to be compiled into a shared library. This can be achieved by
    adding the flag -fPIC to CC and changing the all target to

    all:		bdio.o $(LIBDIR)
                gcc -shared -Wl,-soname,libbdio.so -o $(BUILDDIR)/libbdio.so $(BUILDDIR)/bdio.o
                cp $(BUILDDIR)/libbdio.so $(LIBDIR)/

    Parameters
    ----------
    file_path -- path to the bdio file
    bdio_path -- path to the shared bdio library libbdio.so (default ./libbdio.so)
    """

    for obs in obs_list:
        if not hasattr(obs, 'e_names'):
            raise Exception('Run the gamma method first for all obs.')

    bdio = ctypes.cdll.LoadLibrary(bdio_path)

    bdio_open = bdio.bdio_open
    bdio_open.restype = ctypes.c_void_p

    bdio_close = bdio.bdio_close
    bdio_close.restype = ctypes.c_int
    bdio_close.argtypes = [ctypes.c_void_p]

    bdio_start_record = bdio.bdio_start_record
    bdio_start_record.restype = ctypes.c_int
    bdio_start_record.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]

    bdio_flush_record = bdio.bdio_flush_record
    bdio_flush_record.restype = ctypes.c_int
    bdio_flush_record.argytpes = [ctypes.c_void_p]

    bdio_write_f64 = bdio.bdio_write_f64
    bdio_write_f64.restype = ctypes.c_size_t
    bdio_write_f64.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

    bdio_write_int32 = bdio.bdio_write_int32
    bdio_write_int32.restype = ctypes.c_size_t
    bdio_write_int32.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

    b_path = file_path.encode('utf-8')
    write = 'w'
    b_write = write.encode('utf-8')
    form = 'pyerrors ADerror export'
    b_form = form.encode('utf-8')

    fbdio = bdio_open(ctypes.c_char_p(b_path), ctypes.c_char_p(b_write), b_form)

    for obs in obs_list:
        # mean = obs.value
        neid = len(obs.e_names)
        vrep = [[obs.shape[o] for o in sl] for sl in list(obs.e_content.values())]
        vrep_write = [item for sublist in vrep for item in sublist]
        ndata = [np.sum(o) for o in vrep]
        nrep = [len(o) for o in vrep]
        print('ndata', ndata)
        print('nrep', nrep)
        print('vrep', vrep)
        keys = list(obs.e_content.keys())
        ids = []
        for key in keys:
            try:  # Try to convert key to integer
                ids.append(int(key))
            except Exception:  # If not possible construct a hash
                ids.append(int(hashlib.sha256(key.encode('utf-8')).hexdigest(), 16) % 10 ** 8)
        print('ids', ids)
        nt = []
        for e, e_name in enumerate(obs.e_names):

            r_length = []
            for r_name in obs.e_content[e_name]:
                r_length.append(len(obs.deltas[r_name]))

            # e_N = np.sum(r_length)
            nt.append(max(r_length) // 2)
        print('nt', nt)
        zero = neid * [0.0]
        four = neid * [4.0]
        print('zero', zero)
        print('four', four)
        delta = np.concatenate([item for sublist in [[obs.deltas[o] for o in sl] for sl in list(obs.e_content.values())] for item in sublist])

        bdio_start_record(0x00, 8, fbdio)

        def write_c_double(double):
            pd_buf = ctypes.c_double(double)
            ppd_buf = ctypes.c_void_p(ctypes.addressof(pd_buf))
            bdio_write_f64(ppd_buf, ctypes.c_size_t(8), ctypes.c_void_p(fbdio))

        def write_c_size_t(int32):
            pd_buf = ctypes.c_size_t(int32)
            ppd_buf = ctypes.c_void_p(ctypes.addressof(pd_buf))
            bdio_write_int32(ppd_buf, ctypes.c_size_t(4), ctypes.c_void_p(fbdio))

        write_c_double(obs.value)
        write_c_size_t(neid)

        for element in ndata:
            write_c_size_t(element)
        for element in nrep:
            write_c_size_t(element)
        for element in vrep_write:
            write_c_size_t(element)
        for element in ids:
            write_c_size_t(element)
        for element in nt:
            write_c_size_t(element)

        for element in zero:
            write_c_double(element)
        for element in four:
            write_c_double(element)

        for element in delta:
            write_c_double(element)

    bdio_close(fbdio)
    return 0


def _get_kwd(string, key):
    return (string.split(key, 1)[1]).split(" ", 1)[0]


def _get_corr_name(string, key):
    return (string.split(key, 1)[1]).split(' NDIM=', 1)[0]


def read_mesons(file_path, bdio_path='./libbdio.so', **kwargs):
    """ Extract mesons data from a bdio file and return it as a dictionary

    The dictionary can be accessed with a tuple consisting of (type, source_position, kappa1, kappa2)

    read_mesons requires bdio to be compiled into a shared library. This can be achieved by
    adding the flag -fPIC to CC and changing the all target to

    all:		bdio.o $(LIBDIR)
                gcc -shared -Wl,-soname,libbdio.so -o $(BUILDDIR)/libbdio.so $(BUILDDIR)/bdio.o
                cp $(BUILDDIR)/libbdio.so $(LIBDIR)/

    Parameters
    ----------
    file_path : str
        path to the bdio file
    bdio_path : str
        path to the shared bdio library libbdio.so (default ./libbdio.so)
    start : int
        The first configuration to be read (default 1)
    stop : int
        The last configuration to be read (default None)
    step : int
        Fixed step size between two measurements (default 1)
    alternative_ensemble_name : str
        Manually overwrite ensemble name
    """

    start = kwargs.get('start', 1)
    stop = kwargs.get('stop', None)
    step = kwargs.get('step', 1)

    bdio = ctypes.cdll.LoadLibrary(bdio_path)

    bdio_open = bdio.bdio_open
    bdio_open.restype = ctypes.c_void_p

    bdio_close = bdio.bdio_close
    bdio_close.restype = ctypes.c_int
    bdio_close.argtypes = [ctypes.c_void_p]

    bdio_seek_record = bdio.bdio_seek_record
    bdio_seek_record.restype = ctypes.c_int
    bdio_seek_record.argtypes = [ctypes.c_void_p]

    bdio_get_rlen = bdio.bdio_get_rlen
    bdio_get_rlen.restype = ctypes.c_int
    bdio_get_rlen.argtypes = [ctypes.c_void_p]

    bdio_get_ruinfo = bdio.bdio_get_ruinfo
    bdio_get_ruinfo.restype = ctypes.c_int
    bdio_get_ruinfo.argtypes = [ctypes.c_void_p]

    bdio_read = bdio.bdio_read
    bdio_read.restype = ctypes.c_size_t
    bdio_read.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p]

    bdio_read_f64 = bdio.bdio_read_f64
    bdio_read_f64.restype = ctypes.c_size_t
    bdio_read_f64.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

    b_path = file_path.encode('utf-8')
    read = 'r'
    b_read = read.encode('utf-8')
    form = 'Generic Correlator Format 1.0'
    b_form = form.encode('utf-8')

    ensemble_name = ''
    volume = []  # lattice volume
    boundary_conditions = []
    corr_name = []  # Contains correlator names
    corr_type = []  # Contains correlator data type (important for reading out numerical data)
    corr_props = []  # Contanis propagator types (Component of corr_kappa)
    d0 = 0  # tvals
    d1 = 0  # nnoise
    prop_kappa = []  # Contains propagator kappas (Component of corr_kappa)
    prop_source = []  # Contains propagator source positions
    # Check noise type for multiple replica?
    corr_no = -1
    data = []
    idl = []

    fbdio = bdio_open(ctypes.c_char_p(b_path), ctypes.c_char_p(b_read), ctypes.c_char_p(b_form))

    print('Reading of bdio file started')
    while True:
        bdio_seek_record(fbdio)
        ruinfo = bdio_get_ruinfo(fbdio)
        if ruinfo < 0:
            # EOF reached
            break
        rlen = bdio_get_rlen(fbdio)
        if ruinfo == 5:
            d_buf = ctypes.c_double * (2 + d0 * d1 * 2)
            pd_buf = d_buf()
            ppd_buf = ctypes.c_void_p(ctypes.addressof(pd_buf))
            bdio_read_f64(ppd_buf, ctypes.c_size_t(rlen), ctypes.c_void_p(fbdio))
            if corr_type[corr_no] == 'complex':
                tmp_mean = np.mean(np.asarray(np.split(np.asarray(pd_buf[2 + 2 * d1:-2 * d1:2]), d0 - 2)), axis=1)
            else:
                tmp_mean = np.mean(np.asarray(np.split(np.asarray(pd_buf[2 + d1:-d0 * d1 - d1]), d0 - 2)), axis=1)

            data[corr_no].append(tmp_mean)
            corr_no += 1
        else:
            alt_buf = ctypes.create_string_buffer(1024)
            palt_buf = ctypes.c_char_p(ctypes.addressof(alt_buf))
            iread = bdio_read(palt_buf, ctypes.c_size_t(rlen), ctypes.c_void_p(fbdio))
            if rlen != iread:
                print('Error')
            for i, item in enumerate(alt_buf):
                if item == b'\x00':
                    alt_buf[i] = b' '
            tmp_string = (alt_buf[:].decode("utf-8")).rstrip()
            if ruinfo == 0:
                ensemble_name = _get_kwd(tmp_string, 'ENSEMBLE=')
                volume.append(int(_get_kwd(tmp_string, 'L0=')))
                volume.append(int(_get_kwd(tmp_string, 'L1=')))
                volume.append(int(_get_kwd(tmp_string, 'L2=')))
                volume.append(int(_get_kwd(tmp_string, 'L3=')))
                boundary_conditions.append(_get_kwd(tmp_string, 'BC0='))
                boundary_conditions.append(_get_kwd(tmp_string, 'BC1='))
                boundary_conditions.append(_get_kwd(tmp_string, 'BC2='))
                boundary_conditions.append(_get_kwd(tmp_string, 'BC3='))

            if ruinfo == 1:
                corr_name.append(_get_corr_name(tmp_string, 'CORR_NAME='))
                corr_type.append(_get_kwd(tmp_string, 'DATATYPE='))
                corr_props.append([_get_kwd(tmp_string, 'PROP0='), _get_kwd(tmp_string, 'PROP1=')])
                if d0 == 0:
                    d0 = int(_get_kwd(tmp_string, 'D0='))
                else:
                    if d0 != int(_get_kwd(tmp_string, 'D0=')):
                        print('Error: Varying number of time values')
                if d1 == 0:
                    d1 = int(_get_kwd(tmp_string, 'D1='))
                else:
                    if d1 != int(_get_kwd(tmp_string, 'D1=')):
                        print('Error: Varying number of random sources')
            if ruinfo == 2:
                prop_kappa.append(_get_kwd(tmp_string, 'KAPPA='))
                prop_source.append(_get_kwd(tmp_string, 'x0='))
            if ruinfo == 4:
                cnfg_no = int(_get_kwd(tmp_string, 'CNFG_ID='))
                if stop:
                    if cnfg_no > kwargs.get('stop'):
                        break
                idl.append(cnfg_no)
                print('\r%s %i' % ('Reading configuration', cnfg_no), end='\r')
                if len(idl) == 1:
                    no_corrs = len(corr_name)
                    data = []
                    for c in range(no_corrs):
                        data.append([])

                corr_no = 0

    bdio_close(fbdio)

    print('\nEnsemble: ', ensemble_name)
    if 'alternative_ensemble_name' in kwargs:
        ensemble_name = kwargs.get('alternative_ensemble_name')
        print('Ensemble name overwritten to', ensemble_name)
    print('Lattice volume: ', volume)
    print('Boundary conditions: ', boundary_conditions)
    print('Number of time values: ', d0)
    print('Number of random sources: ', d1)
    print('Number of corrs: ', len(corr_name))
    print('Number of configurations: ', len(idl))

    corr_kappa = []  # Contains kappa values for both propagators of given correlation function
    corr_source = []
    for item in corr_props:
        corr_kappa.append([float(prop_kappa[int(item[0])]), float(prop_kappa[int(item[1])])])
        if prop_source[int(item[0])] != prop_source[int(item[1])]:
            raise Exception('Source position do not match for correlator' + str(item))
        else:
            corr_source.append(int(prop_source[int(item[0])]))

    if stop is None:
        stop = idl[-1]
    idl_target = range(start, stop + 1, step)

    if set(idl) != set(idl_target):
        try:
            indices = [idl.index(i) for i in idl_target]
        except ValueError as err:
            raise Exception('Configurations in file do no match target list!', err)
    else:
        indices = None

    result = {}
    for c in range(no_corrs):
        tmp_corr = []
        tmp_data = np.asarray(data[c])
        for t in range(d0 - 2):
            if indices:
                deltas = [tmp_data[:, t][index] for index in indices]
            else:
                deltas = tmp_data[:, t]
            tmp_corr.append(Obs([deltas], [ensemble_name], idl=[idl_target]))
        result[(corr_name[c], corr_source[c]) + tuple(corr_kappa[c])] = tmp_corr

    # Check that all data entries have the same number of configurations
    if len(set([o[0].N for o in list(result.values())])) != 1:
        raise Exception('Error: Not all correlators have the same number of configurations. bdio file is possibly corrupted.')

    return result


def read_dSdm(file_path, bdio_path='./libbdio.so', **kwargs):
    """ Extract dSdm data from a bdio file and return it as a dictionary

    The dictionary can be accessed with a tuple consisting of (type, kappa)

    read_dSdm requires bdio to be compiled into a shared library. This can be achieved by
    adding the flag -fPIC to CC and changing the all target to

    all:		bdio.o $(LIBDIR)
                gcc -shared -Wl,-soname,libbdio.so -o $(BUILDDIR)/libbdio.so $(BUILDDIR)/bdio.o
                cp $(BUILDDIR)/libbdio.so $(LIBDIR)/

    Parameters
    ----------
    file_path : str
        path to the bdio file
    bdio_path : str
        path to the shared bdio library libbdio.so (default ./libbdio.so)
    start : int
        The first configuration to be read (default 1)
    stop : int
        The last configuration to be read (default None)
    step : int
        Fixed step size between two measurements (default 1)
    alternative_ensemble_name : str
        Manually overwrite ensemble name
    """

    start = kwargs.get('start', 1)
    stop = kwargs.get('stop', None)
    step = kwargs.get('step', 1)

    bdio = ctypes.cdll.LoadLibrary(bdio_path)

    bdio_open = bdio.bdio_open
    bdio_open.restype = ctypes.c_void_p

    bdio_close = bdio.bdio_close
    bdio_close.restype = ctypes.c_int
    bdio_close.argtypes = [ctypes.c_void_p]

    bdio_seek_record = bdio.bdio_seek_record
    bdio_seek_record.restype = ctypes.c_int
    bdio_seek_record.argtypes = [ctypes.c_void_p]

    bdio_get_rlen = bdio.bdio_get_rlen
    bdio_get_rlen.restype = ctypes.c_int
    bdio_get_rlen.argtypes = [ctypes.c_void_p]

    bdio_get_ruinfo = bdio.bdio_get_ruinfo
    bdio_get_ruinfo.restype = ctypes.c_int
    bdio_get_ruinfo.argtypes = [ctypes.c_void_p]

    bdio_read = bdio.bdio_read
    bdio_read.restype = ctypes.c_size_t
    bdio_read.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p]

    bdio_read_f64 = bdio.bdio_read_f64
    bdio_read_f64.restype = ctypes.c_size_t
    bdio_read_f64.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]

    b_path = file_path.encode('utf-8')
    read = 'r'
    b_read = read.encode('utf-8')
    form = 'Generic Correlator Format 1.0'
    b_form = form.encode('utf-8')

    ensemble_name = ''
    volume = []  # lattice volume
    boundary_conditions = []
    corr_name = []  # Contains correlator names
    corr_type = []  # Contains correlator data type (important for reading out numerical data)
    corr_props = []  # Contains propagator types (Component of corr_kappa)
    d0 = 0  # tvals
    # d1 = 0  # nnoise
    prop_kappa = []  # Contains propagator kappas (Component of corr_kappa)
    # Check noise type for multiple replica?
    corr_no = -1
    data = []
    idl = []

    fbdio = bdio_open(ctypes.c_char_p(b_path), ctypes.c_char_p(b_read), ctypes.c_char_p(b_form))

    print('Reading of bdio file started')
    while True:
        bdio_seek_record(fbdio)
        ruinfo = bdio_get_ruinfo(fbdio)
        if ruinfo < 0:
            # EOF reached
            break
        rlen = bdio_get_rlen(fbdio)
        if ruinfo == 5:
            d_buf = ctypes.c_double * (2 + d0)
            pd_buf = d_buf()
            ppd_buf = ctypes.c_void_p(ctypes.addressof(pd_buf))
            bdio_read_f64(ppd_buf, ctypes.c_size_t(rlen), ctypes.c_void_p(fbdio))
            tmp_mean = np.mean(np.asarray(pd_buf[2:]))

            data[corr_no].append(tmp_mean)
            corr_no += 1
        else:
            alt_buf = ctypes.create_string_buffer(1024)
            palt_buf = ctypes.c_char_p(ctypes.addressof(alt_buf))
            iread = bdio_read(palt_buf, ctypes.c_size_t(rlen), ctypes.c_void_p(fbdio))
            if rlen != iread:
                print('Error')
            for i, item in enumerate(alt_buf):
                if item == b'\x00':
                    alt_buf[i] = b' '
            tmp_string = (alt_buf[:].decode("utf-8")).rstrip()
            if ruinfo == 0:
                creator = _get_kwd(tmp_string, 'CREATOR=')
                ensemble_name = _get_kwd(tmp_string, 'ENSEMBLE=')
                volume.append(int(_get_kwd(tmp_string, 'L0=')))
                volume.append(int(_get_kwd(tmp_string, 'L1=')))
                volume.append(int(_get_kwd(tmp_string, 'L2=')))
                volume.append(int(_get_kwd(tmp_string, 'L3=')))
                boundary_conditions.append(_get_kwd(tmp_string, 'BC0='))
                boundary_conditions.append(_get_kwd(tmp_string, 'BC1='))
                boundary_conditions.append(_get_kwd(tmp_string, 'BC2='))
                boundary_conditions.append(_get_kwd(tmp_string, 'BC3='))

            if ruinfo == 1:
                corr_name.append(_get_corr_name(tmp_string, 'CORR_NAME='))
                corr_type.append(_get_kwd(tmp_string, 'DATATYPE='))
                corr_props.append(_get_kwd(tmp_string, 'PROP0='))
                if d0 == 0:
                    d0 = int(_get_kwd(tmp_string, 'D0='))
                else:
                    if d0 != int(_get_kwd(tmp_string, 'D0=')):
                        print('Error: Varying number of time values')
            if ruinfo == 2:
                prop_kappa.append(_get_kwd(tmp_string, 'KAPPA='))
            if ruinfo == 4:
                cnfg_no = int(_get_kwd(tmp_string, 'CNFG_ID='))
                if stop:
                    if cnfg_no > kwargs.get('stop'):
                        break
                idl.append(cnfg_no)
                print('\r%s %i' % ('Reading configuration', cnfg_no), end='\r')
                if len(idl) == 1:
                    no_corrs = len(corr_name)
                    data = []
                    for c in range(no_corrs):
                        data.append([])

                corr_no = 0
    bdio_close(fbdio)

    print('\nCreator: ', creator)
    print('Ensemble: ', ensemble_name)
    print('Lattice volume: ', volume)
    print('Boundary conditions: ', boundary_conditions)
    print('Number of random sources: ', d0)
    print('Number of corrs: ', len(corr_name))
    print('Number of configurations: ', cnfg_no + 1)

    corr_kappa = []  # Contains kappa values for both propagators of given correlation function
    for item in corr_props:
        corr_kappa.append(float(prop_kappa[int(item)]))

    if stop is None:
        stop = idl[-1]
    idl_target = range(start, stop + 1, step)
    try:
        indices = [idl.index(i) for i in idl_target]
    except ValueError as err:
        raise Exception('Configurations in file do no match target list!', err)

    result = {}
    for c in range(no_corrs):
        deltas = [np.asarray(data[c])[index] for index in indices]
        result[(corr_name[c], str(corr_kappa[c]))] = Obs([deltas], [ensemble_name], idl=[idl_target])

    # Check that all data entries have the same number of configurations
    if len(set([o.N for o in list(result.values())])) != 1:
        raise Exception('Error: Not all correlators have the same number of configurations. bdio file is possibly corrupted.')

    return result
