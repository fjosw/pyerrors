import numpy as np
from autograd import jacobian
import autograd.numpy as anp  # Thinly-wrapped numpy
from .obs import derived_observable, CObs, Obs, _merge_idx, _expand_deltas_for_merge, _filter_zeroes, import_jackknife

from functools import partial
from autograd.extend import defvjp


def derived_array(func, data, **kwargs):
    """Construct a derived Obs for a matrix valued function according to func(data, **kwargs) using automatic differentiation.

    Parameters
    ----------
    func : object
        arbitrary function of the form func(data, **kwargs). For the
        automatic differentiation to work, all numpy functions have to have
        the autograd wrapper (use 'import autograd.numpy as anp').
    data : list
        list of Obs, e.g. [obs1, obs2, obs3].
    man_grad : list
        manually supply a list or an array which contains the jacobian
        of func. Use cautiously, supplying the wrong derivative will
        not be intercepted.
    """

    data = np.asarray(data)
    raveled_data = data.ravel()

    # Workaround for matrix operations containing non Obs data
    for i_data in raveled_data:
        if isinstance(i_data, Obs):
            first_name = i_data.names[0]
            first_shape = i_data.shape[first_name]
            first_idl = i_data.idl[first_name]
            break

    for i in range(len(raveled_data)):
        if isinstance(raveled_data[i], (int, float)):
            raveled_data[i] = Obs([raveled_data[i] + np.zeros(first_shape)], [first_name], idl=[first_idl])

    n_obs = len(raveled_data)
    new_names = sorted(set([y for x in [o.names for o in raveled_data] for y in x]))

    is_merged = {name: (len(list(filter(lambda o: o.is_merged.get(name, False) is True, raveled_data))) > 0) for name in new_names}
    reweighted = len(list(filter(lambda o: o.reweighted is True, raveled_data))) > 0
    new_idl_d = {}
    for name in new_names:
        idl = []
        for i_data in raveled_data:
            tmp = i_data.idl.get(name)
            if tmp is not None:
                idl.append(tmp)
        new_idl_d[name] = _merge_idx(idl)
        if not is_merged[name]:
            is_merged[name] = (1 != len(set([len(idx) for idx in [*idl, new_idl_d[name]]])))

    if data.ndim == 1:
        values = np.array([o.value for o in data])
    else:
        values = np.vectorize(lambda x: x.value)(data)

    new_values = func(values, **kwargs)

    new_r_values = {}
    for name in new_names:
        tmp_values = np.zeros(n_obs)
        for i, item in enumerate(raveled_data):
            tmp = item.r_values.get(name)
            if tmp is None:
                tmp = item.value
            tmp_values[i] = tmp
        tmp_values = np.array(tmp_values).reshape(data.shape)
        new_r_values[name] = func(tmp_values, **kwargs)

    if 'man_grad' in kwargs:
        deriv = np.asarray(kwargs.get('man_grad'))
        if new_values.shape + data.shape != deriv.shape:
            raise Exception('Manual derivative does not have correct shape.')
    else:
        deriv = jacobian(func)(values, **kwargs)

    final_result = np.zeros(new_values.shape, dtype=object)

    d_extracted = {}
    for name in new_names:
        d_extracted[name] = []
        for i_dat, dat in enumerate(data):
            ens_length = len(new_idl_d[name])
            d_extracted[name].append(np.array([_expand_deltas_for_merge(o.deltas[name], o.idl[name], o.shape[name], new_idl_d[name]) for o in dat.reshape(np.prod(dat.shape))]).reshape(dat.shape + (ens_length, )))

    for i_val, new_val in np.ndenumerate(new_values):
        new_deltas = {}
        for name in new_names:
            ens_length = d_extracted[name][0].shape[-1]
            new_deltas[name] = np.zeros(ens_length)
            for i_dat, dat in enumerate(d_extracted[name]):
                new_deltas[name] += np.tensordot(deriv[i_val + (i_dat, )], dat)

        new_samples = []
        new_means = []
        new_idl = []
        for name in new_names:
            if is_merged[name]:
                filtered_deltas, filtered_idl_d = _filter_zeroes(new_deltas[name], new_idl_d[name])
            else:
                filtered_deltas = new_deltas[name]
                filtered_idl_d = new_idl_d[name]

            new_samples.append(filtered_deltas)
            new_idl.append(filtered_idl_d)
            new_means.append(new_r_values[name][i_val])
        final_result[i_val] = Obs(new_samples, new_names, means=new_means, idl=new_idl)
        final_result[i_val]._value = new_val
        final_result[i_val].is_merged = is_merged
        final_result[i_val].reweighted = reweighted

    return final_result


def matmul(*operands):
    """Matrix multiply all operands.

    Parameters
    ----------
    operands : numpy.ndarray
        Arbitrary number of 2d-numpy arrays which can be real or complex
        Obs valued.

    This implementation is faster compared to standard multiplication via the @ operator.
    """
    if any(isinstance(o[0, 0], CObs) for o in operands):
        extended_operands = []
        for op in operands:
            tmp = np.vectorize(lambda x: (np.real(x), np.imag(x)))(op)
            extended_operands.append(tmp[0])
            extended_operands.append(tmp[1])

        def multi_dot(operands, part):
            stack_r = operands[0]
            stack_i = operands[1]
            for op_r, op_i in zip(operands[2::2], operands[3::2]):
                tmp_r = stack_r @ op_r - stack_i @ op_i
                tmp_i = stack_r @ op_i + stack_i @ op_r

                stack_r = tmp_r
                stack_i = tmp_i

            if part == 'Real':
                return stack_r
            else:
                return stack_i

        def multi_dot_r(operands):
            return multi_dot(operands, 'Real')

        def multi_dot_i(operands):
            return multi_dot(operands, 'Imag')

        Nr = derived_array(multi_dot_r, extended_operands)
        Ni = derived_array(multi_dot_i, extended_operands)

        res = np.empty_like(Nr)
        for (n, m), entry in np.ndenumerate(Nr):
            res[n, m] = CObs(Nr[n, m], Ni[n, m])

        return res
    else:
        def multi_dot(operands):
            stack = operands[0]
            for op in operands[1:]:
                stack = stack @ op
            return stack
        return derived_array(multi_dot, operands)


def jack_matmul(a, b):
    """Matrix multiply both operands making use of the jackknife approximation.

    Parameters
    ----------
    a : numpy.ndarray
        First matrix, can be real or complex Obs valued
    b : numpy.ndarray
        Second matrix, can be real or complex Obs valued

    For large matrices this is considerably faster compared to matmul.
    """

    if any(isinstance(o[0, 0], CObs) for o in [a, b]):
        def _exp_to_jack(matrix):
            base_matrix = np.empty_like(matrix)
            for (n, m), entry in np.ndenumerate(matrix):
                base_matrix[n, m] = entry.real.export_jackknife() + 1j * entry.imag.export_jackknife()
            return base_matrix

        def _imp_from_jack(matrix, name):
            base_matrix = np.empty_like(matrix)
            for (n, m), entry in np.ndenumerate(matrix):
                base_matrix[n, m] = CObs(import_jackknife(entry.real, name),
                                         import_jackknife(entry.imag, name))
            return base_matrix

        j_a = _exp_to_jack(a)
        j_b = _exp_to_jack(b)
        r = j_a @ j_b
        return _imp_from_jack(r, a.ravel()[0].real.names[0])
    else:
        def _exp_to_jack(matrix):
            base_matrix = np.empty_like(matrix)
            for (n, m), entry in np.ndenumerate(matrix):
                base_matrix[n, m] = entry.export_jackknife()
            return base_matrix

        def _imp_from_jack(matrix, name):
            base_matrix = np.empty_like(matrix)
            for (n, m), entry in np.ndenumerate(matrix):
                base_matrix[n, m] = import_jackknife(entry, name)
            return base_matrix

        j_a = _exp_to_jack(a)
        j_b = _exp_to_jack(b)
        r = j_a @ j_b
        return _imp_from_jack(r, a.ravel()[0].names[0])


def boot_matmul(a, b):
    """Matrix multiply both operands making use of the bootstrap approximation.

    Parameters
    ----------
    a : numpy.ndarray
        First matrix, can be real or complex Obs valued
    b : numpy.ndarray
        Second matrix, can be real or complex Obs valued

    For large matrices this is considerably faster compared to matmul.
    """

    def export_boot(obs):
        ret = np.zeros(obs.N + 1)
        ret[0] = obs.value
        ret[1:] = proj @ (obs.deltas[name] + obs.r_values[name])
        return ret

    def import_boot(boots):
        samples = inv_proj @ boots[1:]
        ret = Obs([samples], [name])
        ret._value = boots[0]
        return ret

    if any(isinstance(o[0, 0], CObs) for o in [a, b]):
        assert len(a[0, 0].real.names) == 1

        name = a[0, 0].real.names[0]

        length = a[0, 0].real.N

        random_numbers = np.random.randint(0, length, (length, length))

        proj = np.vstack([np.bincount(o, minlength=length) for o in random_numbers]).T / length

        inv_proj = np.linalg.inv(proj)

        def _exp_to_boot(matrix):
            base_matrix = np.empty_like(matrix)
            for (n, m), entry in np.ndenumerate(matrix):
                base_matrix[n, m] = export_boot(entry.real) + 1j * export_boot(entry.imag)
            return base_matrix

        def _imp_from_boot(matrix, name):
            base_matrix = np.empty_like(matrix)
            for (n, m), entry in np.ndenumerate(matrix):
                base_matrix[n, m] = CObs(import_boot(entry.real),
                                         import_boot(entry.imag))
            return base_matrix

        j_a = _exp_to_boot(a)
        j_b = _exp_to_boot(b)
        r = j_a @ j_b
        return _imp_from_boot(r, a.ravel()[0].real.names[0])
    else:
        assert len(a[0, 0].names) == 1

        name = a[0, 0].names[0]

        length = a[0, 0].N

        random_numbers = np.random.randint(0, length, (length, length))

        proj = np.vstack([np.bincount(o, minlength=length) for o in random_numbers]).T / length

        inv_proj = np.linalg.inv(proj)

        def _exp_to_boot(matrix):
            base_matrix = np.empty_like(matrix)
            for (n, m), entry in np.ndenumerate(matrix):
                base_matrix[n, m] = export_boot(entry)
            return base_matrix

        def _imp_from_boot(matrix, name):
            base_matrix = np.empty_like(matrix)
            for (n, m), entry in np.ndenumerate(matrix):
                base_matrix[n, m] = import_boot(entry)
            return base_matrix

        j_a = _exp_to_boot(a)
        j_b = _exp_to_boot(b)
        r = j_a @ j_b
        return _imp_from_boot(r, a.ravel()[0].names[0])


def inv(x):
    """Inverse of Obs or CObs valued matrices."""
    return _mat_mat_op(anp.linalg.inv, x)


def cholesky(x):
    """Cholesky decomposition of Obs or CObs valued matrices."""
    return _mat_mat_op(anp.linalg.cholesky, x)


def scalar_mat_op(op, obs, **kwargs):
    """Computes the matrix to scalar operation op to a given matrix of Obs."""
    def _mat(x, **kwargs):
        dim = int(np.sqrt(len(x)))
        if np.sqrt(len(x)) != dim:
            raise Exception('Input has to have dim**2 entries')

        mat = []
        for i in range(dim):
            row = []
            for j in range(dim):
                row.append(x[j + dim * i])
            mat.append(row)

        return op(anp.array(mat))

    if isinstance(obs, np.ndarray):
        raveled_obs = (1 * (obs.ravel())).tolist()
    elif isinstance(obs, list):
        raveled_obs = obs
    else:
        raise TypeError('Unproper type of input.')
    return derived_observable(_mat, raveled_obs, **kwargs)


def _mat_mat_op(op, obs, **kwargs):
    """Computes the matrix to matrix operation op to a given matrix of Obs."""
    # Use real representation to calculate matrix operations for complex matrices
    if isinstance(obs.ravel()[0], CObs):
        A = np.empty_like(obs)
        B = np.empty_like(obs)
        for (n, m), entry in np.ndenumerate(obs):
            if hasattr(entry, 'real') and hasattr(entry, 'imag'):
                A[n, m] = entry.real
                B[n, m] = entry.imag
            else:
                A[n, m] = entry
                B[n, m] = 0.0
        big_matrix = np.block([[A, -B], [B, A]])
        if kwargs.get('num_grad') is True:
            op_big_matrix = _num_diff_mat_mat_op(op, big_matrix, **kwargs)
        else:
            op_big_matrix = derived_array(lambda x, **kwargs: op(x), [big_matrix])[0]
        dim = op_big_matrix.shape[0]
        op_A = op_big_matrix[0: dim // 2, 0: dim // 2]
        op_B = op_big_matrix[dim // 2:, 0: dim // 2]
        res = np.empty_like(op_A)
        for (n, m), entry in np.ndenumerate(op_A):
            res[n, m] = CObs(op_A[n, m], op_B[n, m])
        return res
    else:
        if kwargs.get('num_grad') is True:
            return _num_diff_mat_mat_op(op, obs, **kwargs)
        return derived_array(lambda x, **kwargs: op(x), [obs])[0]


def eigh(obs, **kwargs):
    """Computes the eigenvalues and eigenvectors of a given hermitian matrix of Obs according to np.linalg.eigh."""
    if kwargs.get('num_grad') is True:
        return _num_diff_eigh(obs, **kwargs)
    w = derived_observable(lambda x, **kwargs: anp.linalg.eigh(x)[0], obs)
    v = derived_observable(lambda x, **kwargs: anp.linalg.eigh(x)[1], obs)
    return w, v


def eig(obs, **kwargs):
    """Computes the eigenvalues of a given matrix of Obs according to np.linalg.eig."""
    if kwargs.get('num_grad') is True:
        return _num_diff_eig(obs, **kwargs)
        # Note: Automatic differentiation of eig is implemented in the git of autograd
        # but not yet released to PyPi (1.3)
    w = derived_observable(lambda x, **kwargs: anp.real(anp.linalg.eig(x)[0]), obs)
    return w


def pinv(obs, **kwargs):
    """Computes the Moore-Penrose pseudoinverse of a matrix of Obs."""
    if kwargs.get('num_grad') is True:
        return _num_diff_pinv(obs, **kwargs)
    return derived_observable(lambda x, **kwargs: anp.linalg.pinv(x), obs)


def svd(obs, **kwargs):
    """Computes the singular value decomposition of a matrix of Obs."""
    if kwargs.get('num_grad') is True:
        return _num_diff_svd(obs, **kwargs)
    u = derived_observable(lambda x, **kwargs: anp.linalg.svd(x, full_matrices=False)[0], obs)
    s = derived_observable(lambda x, **kwargs: anp.linalg.svd(x, full_matrices=False)[1], obs)
    vh = derived_observable(lambda x, **kwargs: anp.linalg.svd(x, full_matrices=False)[2], obs)
    return (u, s, vh)


def slogdet(obs, **kwargs):
    """Computes the determinant of a matrix of Obs via np.linalg.slogdet."""
    def _mat(x):
        dim = int(np.sqrt(len(x)))
        if np.sqrt(len(x)) != dim:
            raise Exception('Input has to have dim**2 entries')

        mat = []
        for i in range(dim):
            row = []
            for j in range(dim):
                row.append(x[j + dim * i])
            mat.append(row)

        (sign, logdet) = anp.linalg.slogdet(np.array(mat))
        return sign * anp.exp(logdet)

    if isinstance(obs, np.ndarray):
        return derived_observable(_mat, (1 * (obs.ravel())).tolist(), **kwargs)
    elif isinstance(obs, list):
        return derived_observable(_mat, obs, **kwargs)
    else:
        raise TypeError('Unproper type of input.')


# Variants for numerical differentiation

def _num_diff_mat_mat_op(op, obs, **kwargs):
    """Computes the matrix to matrix operation op to a given matrix of Obs elementwise
       which is suitable for numerical differentiation."""
    def _mat(x, **kwargs):
        dim = int(np.sqrt(len(x)))
        if np.sqrt(len(x)) != dim:
            raise Exception('Input has to have dim**2 entries')

        mat = []
        for i in range(dim):
            row = []
            for j in range(dim):
                row.append(x[j + dim * i])
            mat.append(row)

        return op(np.array(mat))[kwargs.get('i')][kwargs.get('j')]

    if isinstance(obs, np.ndarray):
        raveled_obs = (1 * (obs.ravel())).tolist()
    elif isinstance(obs, list):
        raveled_obs = obs
    else:
        raise TypeError('Unproper type of input.')

    dim = int(np.sqrt(len(raveled_obs)))

    res_mat = []
    for i in range(dim):
        row = []
        for j in range(dim):
            row.append(derived_observable(_mat, raveled_obs, i=i, j=j, **kwargs))
        res_mat.append(row)

    return np.array(res_mat) @ np.identity(dim)


def _num_diff_eigh(obs, **kwargs):
    """Computes the eigenvalues and eigenvectors of a given hermitian matrix of Obs according to np.linalg.eigh
       elementwise which is suitable for numerical differentiation."""
    def _mat(x, **kwargs):
        dim = int(np.sqrt(len(x)))
        if np.sqrt(len(x)) != dim:
            raise Exception('Input has to have dim**2 entries')

        mat = []
        for i in range(dim):
            row = []
            for j in range(dim):
                row.append(x[j + dim * i])
            mat.append(row)

        n = kwargs.get('n')
        res = np.linalg.eigh(np.array(mat))[n]

        if n == 0:
            return res[kwargs.get('i')]
        else:
            return res[kwargs.get('i')][kwargs.get('j')]

    if isinstance(obs, np.ndarray):
        raveled_obs = (1 * (obs.ravel())).tolist()
    elif isinstance(obs, list):
        raveled_obs = obs
    else:
        raise TypeError('Unproper type of input.')

    dim = int(np.sqrt(len(raveled_obs)))

    res_vec = []
    for i in range(dim):
        res_vec.append(derived_observable(_mat, raveled_obs, n=0, i=i, **kwargs))

    res_mat = []
    for i in range(dim):
        row = []
        for j in range(dim):
            row.append(derived_observable(_mat, raveled_obs, n=1, i=i, j=j, **kwargs))
        res_mat.append(row)

    return (np.array(res_vec) @ np.identity(dim), np.array(res_mat) @ np.identity(dim))


def _num_diff_eig(obs, **kwargs):
    """Computes the eigenvalues of a given matrix of Obs according to np.linalg.eig
       elementwise which is suitable for numerical differentiation."""
    def _mat(x, **kwargs):
        dim = int(np.sqrt(len(x)))
        if np.sqrt(len(x)) != dim:
            raise Exception('Input has to have dim**2 entries')

        mat = []
        for i in range(dim):
            row = []
            for j in range(dim):
                row.append(x[j + dim * i])
            mat.append(row)

        n = kwargs.get('n')
        res = np.linalg.eig(np.array(mat))[n]

        if n == 0:
            # Discard imaginary part of eigenvalue here
            return np.real(res[kwargs.get('i')])
        else:
            return res[kwargs.get('i')][kwargs.get('j')]

    if isinstance(obs, np.ndarray):
        raveled_obs = (1 * (obs.ravel())).tolist()
    elif isinstance(obs, list):
        raveled_obs = obs
    else:
        raise TypeError('Unproper type of input.')

    dim = int(np.sqrt(len(raveled_obs)))

    res_vec = []
    for i in range(dim):
        # Note: Automatic differentiation of eig is implemented in the git of autograd
        # but not yet released to PyPi (1.3)
        res_vec.append(derived_observable(_mat, raveled_obs, n=0, i=i, **kwargs))

    return np.array(res_vec) @ np.identity(dim)


def _num_diff_pinv(obs, **kwargs):
    """Computes the Moore-Penrose pseudoinverse of a matrix of Obs elementwise which is suitable
       for numerical differentiation."""
    def _mat(x, **kwargs):
        shape = kwargs.get('shape')

        mat = []
        for i in range(shape[0]):
            row = []
            for j in range(shape[1]):
                row.append(x[j + shape[1] * i])
            mat.append(row)

        return np.linalg.pinv(np.array(mat))[kwargs.get('i')][kwargs.get('j')]

    if isinstance(obs, np.ndarray):
        shape = obs.shape
        raveled_obs = (1 * (obs.ravel())).tolist()
    else:
        raise TypeError('Unproper type of input.')

    res_mat = []
    for i in range(shape[1]):
        row = []
        for j in range(shape[0]):
            row.append(derived_observable(_mat, raveled_obs, shape=shape, i=i, j=j, **kwargs))
        res_mat.append(row)

    return np.array(res_mat) @ np.identity(shape[0])


def _num_diff_svd(obs, **kwargs):
    """Computes the singular value decomposition of a matrix of Obs elementwise which
       is suitable for numerical differentiation."""
    def _mat(x, **kwargs):
        shape = kwargs.get('shape')

        mat = []
        for i in range(shape[0]):
            row = []
            for j in range(shape[1]):
                row.append(x[j + shape[1] * i])
            mat.append(row)

        res = np.linalg.svd(np.array(mat), full_matrices=False)

        if kwargs.get('n') == 1:
            return res[1][kwargs.get('i')]
        else:
            return res[kwargs.get('n')][kwargs.get('i')][kwargs.get('j')]

    if isinstance(obs, np.ndarray):
        shape = obs.shape
        raveled_obs = (1 * (obs.ravel())).tolist()
    else:
        raise TypeError('Unproper type of input.')

    mid_index = min(shape[0], shape[1])

    res_mat0 = []
    for i in range(shape[0]):
        row = []
        for j in range(mid_index):
            row.append(derived_observable(_mat, raveled_obs, shape=shape, n=0, i=i, j=j, **kwargs))
        res_mat0.append(row)

    res_mat1 = []
    for i in range(mid_index):
        res_mat1.append(derived_observable(_mat, raveled_obs, shape=shape, n=1, i=i, **kwargs))

    res_mat2 = []
    for i in range(mid_index):
        row = []
        for j in range(shape[1]):
            row.append(derived_observable(_mat, raveled_obs, shape=shape, n=2, i=i, j=j, **kwargs))
        res_mat2.append(row)

    return (np.array(res_mat0) @ np.identity(mid_index), np.array(res_mat1) @ np.identity(mid_index), np.array(res_mat2) @ np.identity(shape[1]))


# This code block is directly taken from the current master branch of autograd and remains
# only until the new version is released on PyPi
_dot = partial(anp.einsum, '...ij,...jk->...ik')


# batched diag
def _diag(a):
    return anp.eye(a.shape[-1]) * a


# batched diagonal, similar to matrix_diag in tensorflow
def _matrix_diag(a):
    reps = anp.array(a.shape)
    reps[:-1] = 1
    reps[-1] = a.shape[-1]
    newshape = list(a.shape) + [a.shape[-1]]
    return _diag(anp.tile(a, reps).reshape(newshape))

# https://arxiv.org/pdf/1701.00392.pdf Eq(4.77)
# Note the formula from Sec3.1 in https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf is incomplete


def grad_eig(ans, x):
    """Gradient of a general square (complex valued) matrix"""
    e, u = ans  # eigenvalues as 1d array, eigenvectors in columns
    n = e.shape[-1]

    def vjp(g):
        ge, gu = g
        ge = _matrix_diag(ge)
        f = 1 / (e[..., anp.newaxis, :] - e[..., :, anp.newaxis] + 1.e-20)
        f -= _diag(f)
        ut = anp.swapaxes(u, -1, -2)
        r1 = f * _dot(ut, gu)
        r2 = -f * (_dot(_dot(ut, anp.conj(u)), anp.real(_dot(ut, gu)) * anp.eye(n)))
        r = _dot(_dot(anp.linalg.inv(ut), ge + r1 + r2), ut)
        if not anp.iscomplexobj(x):
            r = anp.real(r)
            # the derivative is still complex for real input (imaginary delta is allowed), real output
            # but the derivative should be real in real input case when imaginary delta is forbidden
        return r
    return vjp


defvjp(anp.linalg.eig, grad_eig)
# End of the code block from autograd.master
