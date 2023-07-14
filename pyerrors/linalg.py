import numpy as np
import autograd.numpy as anp  # Thinly-wrapped numpy
from .obs import derived_observable, CObs, Obs, import_jackknife
from autograd import jacobian
from scipy.integrate import quad


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
    einsum_path = np.einsum_path(extended_subscripts, *conv_operands, optimize='optimal')[0]
    jack_einsum = np.einsum(extended_subscripts, *conv_operands, optimize=einsum_path)

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
    """Cholesky decomposition of Obs valued matrices."""
    if any(isinstance(o, CObs) for o in x.ravel()):
        raise Exception("Cholesky decomposition is not implemented for CObs.")
    return _mat_mat_op(anp.linalg.cholesky, x)


def det(x):
    """Determinant of Obs valued matrices."""
    return _scalar_mat_op(anp.linalg.det, x)


def _scalar_mat_op(op, obs, **kwargs):
    """Computes the matrix to scalar operation op to a given matrix of Obs."""
    def _mat(x, **kwargs):
        dim = int(np.sqrt(len(x)))

        mat = []
        for i in range(dim):
            row = []
            for j in range(dim):
                row.append(x[j + dim * i])
            mat.append(row)

        return op(anp.array(mat))

    if isinstance(obs, np.ndarray):
        raveled_obs = (1 * (obs.ravel())).tolist()
    else:
        raise TypeError('Unproper type of input.')
    return derived_observable(_mat, raveled_obs, **kwargs)


def _mat_mat_op(op, obs, **kwargs):
    """Computes the matrix to matrix operation op to a given matrix of Obs."""
    # Use real representation to calculate matrix operations for complex matrices
    if any(isinstance(o, CObs) for o in obs.ravel()):
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
        op_big_matrix = derived_observable(lambda x, **kwargs: op(x), [big_matrix], array_mode=True)[0]
        dim = op_big_matrix.shape[0]
        op_A = op_big_matrix[0: dim // 2, 0: dim // 2]
        op_B = op_big_matrix[dim // 2:, 0: dim // 2]
        res = np.empty_like(op_A)
        for (n, m), entry in np.ndenumerate(op_A):
            res[n, m] = CObs(op_A[n, m], op_B[n, m])
        return res
    else:
        return derived_observable(lambda x, **kwargs: op(x), [obs], array_mode=True)[0]


def eigh(obs, **kwargs):
    """Computes the eigenvalues and eigenvectors of a given hermitian matrix of Obs according to np.linalg.eigh."""
    w = derived_observable(lambda x, **kwargs: anp.linalg.eigh(x)[0], obs)
    v = derived_observable(lambda x, **kwargs: anp.linalg.eigh(x)[1], obs)
    return w, v


def eig(obs, **kwargs):
    """Computes the eigenvalues of a given matrix of Obs according to np.linalg.eig."""
    w = derived_observable(lambda x, **kwargs: anp.real(anp.linalg.eig(x)[0]), obs)
    return w


def pinv(obs, **kwargs):
    """Computes the Moore-Penrose pseudoinverse of a matrix of Obs."""
    return derived_observable(lambda x, **kwargs: anp.linalg.pinv(x), obs)


def svd(obs, **kwargs):
    """Computes the singular value decomposition of a matrix of Obs."""
    u = derived_observable(lambda x, **kwargs: anp.linalg.svd(x, full_matrices=False)[0], obs)
    s = derived_observable(lambda x, **kwargs: anp.linalg.svd(x, full_matrices=False)[1], obs)
    vh = derived_observable(lambda x, **kwargs: anp.linalg.svd(x, full_matrices=False)[2], obs)
    return (u, s, vh)


def integrate(func, p, a, b, **kwargs):
    '''Performs a (one-dimensional) numeric integration of f(p, x) from bounds[0] to bounds[1]

    The integration is performed using quad().
    All parameters that can be passed to quad may also be passed to integrate().
    The output is the same as for quad, the first element being an Obs.

    Parameters
    ----------
    func : object
        function to integrate, has to be of the form

        ```python
        import autograd.numpy as anp

        def func(p, x):
            return p[0] + p[1] * x + p[2] * anp.sinh(x)
        ```
        where x is the integration variable.
    p : list of floats or Obs
        parameters of the function func.
    a: float or Obs
        Lower limit of integration (use -numpy.inf for -infinity).
    b: float or Obs
        Upper limit of integration (use -numpy.inf for -infinity).
    All parameters of quad

    Returns
    -------
    y : Obs
        The integral of func from `a` to `b`.
    abserr : float
        An estimate of the absolute error in the result.
    infodict : dict
        A dictionary containing additional information.
        Run quad_explain() for more information.
    message
        A convergence message.
    explain
        Appended only with 'cos' or 'sin' weighting and infinite
        integration limits, it contains an explanation of the codes in
        infodict['ierlst']
    '''

    Np = len(p)
    isobs = [True if isinstance(pi, Obs) else False for pi in p]
    pval = np.array([p[i].value if isobs[i] else p[i] for i in range(Np)],)
    pobs = [p[i] for i in range(Np) if isobs[i]]

    bounds = [a, b]
    isobs_b = [True if isinstance(bi, Obs) else False for bi in bounds]
    bval = np.array([bounds[i].value if isobs_b[i] else bounds[i] for i in range(2)])
    bobs = [bounds[i] for i in range(2) if isobs_b[i]]
    bsign = [-1, 1]

    ifunc = np.vectorize(lambda x: func(pval, x))

    intpars = quad.__code__.co_varnames[3:3 + len(quad.__defaults__)]
    ikwargs = {k: kwargs[k] for k in intpars if k in kwargs}

    integration_result = quad(ifunc, bval[0], bval[1], **ikwargs)
    val = integration_result[0]

    jac = jacobian(func)

    derivint = []
    for i in range(Np):
        if isobs[i]:
            ifunc = np.vectorize(lambda x: jac(pval, x)[i])
            derivint.append(quad(ifunc, bounds[0], bounds[1], **ikwargs)[0])

    for i in range(2):
        if isobs_b[i]:
            derivint.append(bsign[i] * func(pval, bval[i]))

    res = derived_observable(lambda x, **kwargs: 0 * (x[0] + np.finfo(np.float64).eps) * (pval[0] + np.finfo(np.float64).eps) + val, pobs + bobs, man_grad=derivint)

    return (res, *integration_result[1:])
