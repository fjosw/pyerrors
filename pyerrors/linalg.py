import numpy as np
import autograd.numpy as anp  # Thinly-wrapped numpy
from .obs import derived_observable, CObs, Obs, import_jackknife


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

        Nr = derived_observable(multi_dot_r, extended_operands, array_mode=True)
        Ni = derived_observable(multi_dot_i, extended_operands, array_mode=True)

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
        return derived_observable(multi_dot, operands, array_mode=True)


def jack_matmul(*operands):
    """Matrix multiply both operands making use of the jackknife approximation.

    Parameters
    ----------
    operands : numpy.ndarray
        Arbitrary number of 2d-numpy arrays which can be real or complex
        Obs valued.

    For large matrices this is considerably faster compared to matmul.
    """

    def _exp_to_jack(matrix):
        base_matrix = np.empty_like(matrix)
        for index, entry in np.ndenumerate(matrix):
            base_matrix[index] = entry.export_jackknife()
        return base_matrix

    def _imp_from_jack(matrix, name, idl):
        base_matrix = np.empty_like(matrix)
        for index, entry in np.ndenumerate(matrix):
            base_matrix[index] = import_jackknife(entry, name, [idl])
        return base_matrix

    def _exp_to_jack_c(matrix):
        base_matrix = np.empty_like(matrix)
        for index, entry in np.ndenumerate(matrix):
            base_matrix[index] = entry.real.export_jackknife() + 1j * entry.imag.export_jackknife()
        return base_matrix

    def _imp_from_jack_c(matrix, name, idl):
        base_matrix = np.empty_like(matrix)
        for index, entry in np.ndenumerate(matrix):
            base_matrix[index] = CObs(import_jackknife(entry.real, name, [idl]),
                                      import_jackknife(entry.imag, name, [idl]))
        return base_matrix

    if any(isinstance(o.flat[0], CObs) for o in operands):
        name = operands[0].flat[0].real.names[0]
        idl = operands[0].flat[0].real.idl[name]

        r = _exp_to_jack_c(operands[0])
        for op in operands[1:]:
            if isinstance(op.flat[0], CObs):
                r = r @ _exp_to_jack_c(op)
            else:
                r = r @ op
        return _imp_from_jack_c(r, name, idl)
    else:
        name = operands[0].flat[0].names[0]
        idl = operands[0].flat[0].idl[name]

        r = _exp_to_jack(operands[0])
        for op in operands[1:]:
            if isinstance(op.flat[0], Obs):
                r = r @ _exp_to_jack(op)
            else:
                r = r @ op
        return _imp_from_jack(r, name, idl)


def einsum(subscripts, *operands):
    """Wrapper for numpy.einsum

    Parameters
    ----------
    subscripts : str
        Subscripts for summation (see numpy documentation for details)
    operands : numpy.ndarray
        Arbitrary number of 2d-numpy arrays which can be real or complex
        Obs valued.
    """

    def _exp_to_jack(matrix):
        base_matrix = []
        for index, entry in np.ndenumerate(matrix):
            base_matrix.append(entry.export_jackknife())
        return np.asarray(base_matrix).reshape(matrix.shape + base_matrix[0].shape)

    def _exp_to_jack_c(matrix):
        base_matrix = []
        for index, entry in np.ndenumerate(matrix):
            base_matrix.append(entry.real.export_jackknife() + 1j * entry.imag.export_jackknife())
        return np.asarray(base_matrix).reshape(matrix.shape + base_matrix[0].shape)

    def _imp_from_jack(matrix, name, idl):
        base_matrix = np.empty(shape=matrix.shape[:-1], dtype=object)
        for index in np.ndindex(matrix.shape[:-1]):
            base_matrix[index] = import_jackknife(matrix[index], name, [idl])
        return base_matrix

    def _imp_from_jack_c(matrix, name, idl):
        base_matrix = np.empty(shape=matrix.shape[:-1], dtype=object)
        for index in np.ndindex(matrix.shape[:-1]):
            base_matrix[index] = CObs(import_jackknife(matrix[index].real, name, [idl]),
                                      import_jackknife(matrix[index].imag, name, [idl]))
        return base_matrix

    for op in operands:
        if isinstance(op.flat[0], CObs):
            name = op.flat[0].real.names[0]
            idl = op.flat[0].real.idl[name]
            break
        elif isinstance(op.flat[0], Obs):
            name = op.flat[0].names[0]
            idl = op.flat[0].idl[name]
            break

    conv_operands = []
    for op in operands:
        if isinstance(op.flat[0], CObs):
            conv_operands.append(_exp_to_jack_c(op))
        elif isinstance(op.flat[0], Obs):
            conv_operands.append(_exp_to_jack(op))
        else:
            conv_operands.append(op)

    tmp_subscripts = ','.join([o + '...' for o in subscripts.split(',')])
    extended_subscripts = '->'.join([o + '...' for o in tmp_subscripts.split('->')[:-1]] + [tmp_subscripts.split('->')[-1]])
    jack_einsum = np.einsum(extended_subscripts, *conv_operands)

    if jack_einsum.dtype == complex:
        result = _imp_from_jack_c(jack_einsum, name, idl)
    elif jack_einsum.dtype == float:
        result = _imp_from_jack(jack_einsum, name, idl)
    else:
        raise Exception("Result has unexpected datatype")

    if result.shape == ():
        return result.flat[0]
    else:
        return result


def inv(x):
    """Inverse of Obs or CObs valued matrices."""
    return _mat_mat_op(anp.linalg.inv, x)


def cholesky(x):
    """Cholesky decomposition of Obs or CObs valued matrices."""
    return _mat_mat_op(anp.linalg.cholesky, x)


def det(x):
    """Determinant of Obs valued matrices."""
    return _scalar_mat_op(anp.linalg.det, x)


def _scalar_mat_op(op, obs, **kwargs):
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
            op_big_matrix = derived_observable(lambda x, **kwargs: op(x), [big_matrix], array_mode=True)[0]
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
        return derived_observable(lambda x, **kwargs: op(x), [obs], array_mode=True)[0]


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
