import autograd.numpy as anp  # Thinly-wrapped numpy
import numpy as np
import scipy.linalg
from autograd.extend import Box, notrace_primitive

from .obs import CObs, Obs, derived_observable, import_jackknife


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
            for op_r, op_i in zip(operands[2::2], operands[3::2], strict=True):
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
        for (n, m), _entry in np.ndenumerate(Nr):
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
        for _index, entry in np.ndenumerate(matrix):
            base_matrix.append(entry.export_jackknife())
        return np.asarray(base_matrix).reshape(matrix.shape + base_matrix[0].shape)

    def _exp_to_jack_c(matrix):
        base_matrix = []
        for _index, entry in np.ndenumerate(matrix):
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
        raise NotImplementedError("Cholesky decomposition is not implemented for CObs.")
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
        for (n, m), _entry in np.ndenumerate(op_A):
            res[n, m] = CObs(op_A[n, m], op_B[n, m])
        return res
    else:
        return derived_observable(lambda x, **kwargs: op(x), [obs], array_mode=True)[0]


def eigh(obs, **kwargs):
    """Eigenvalues and eigenvector columns of a Hermitian Obs or CObs matrix.

    For complex input, eigenvector errors use a local phase convention in which
    the overlap with the central vector is real and positive. Individual vector
    derivatives are undefined at degenerate eigenvalues.
    """
    if _is_complex(obs):
        return _eigh_complex(obs)
    w = derived_observable(lambda x, **kwargs: anp.linalg.eigh(x)[0], obs)
    v = eigv(obs)
    return w, v


def eig(obs, **kwargs):
    """Computes the eigenvalues of a given matrix of Obs according to np.linalg.eig."""
    w = derived_observable(lambda x, **kwargs: anp.real(anp.linalg.eig(x)[0]), obs)
    return w


def eigv(obs, **kwargs):
    """Eigenvectors of a Hermitian matrix, with propagated uncertainties.

    Numerically unresolved eigenvalue gaps at the central matrix raise
    LinAlgError because individual eigenvector derivatives are undefined at
    degeneracy.
    """
    if _is_complex(obs):
        return _eigh_complex(obs)[1]

    def eigenvectors(x, **kwargs):
        w, v = anp.linalg.eigh(x)
        if isinstance(x, Box):  # Only the central Jacobian needs this check.
            _check_eigenvalue_gaps(w)
        return v

    return derived_observable(eigenvectors, obs)


def _is_complex(matrix):
    """Whether any entry requires complex observable handling."""
    return any(isinstance(entry, (CObs, complex, np.complexfloating)) for entry in np.asarray(matrix).flat)


def _central_value(entry):
    """Central value of a numeric scalar, Obs or CObs."""
    if isinstance(entry, CObs):
        return complex(_central_value(entry.real), _central_value(entry.imag))
    return entry.value if isinstance(entry, Obs) else entry


def _matrix_values(matrix):
    """Extract means without dropping imaginary parts in mixed Obs/CObs arrays."""
    matrix = np.asarray(matrix)
    return np.array([_central_value(entry) for entry in matrix.flat],
                    dtype=complex if _is_complex(matrix) else float).reshape(matrix.shape)


def _conjugate(array):
    """Conjugate numeric/Obs/CObs arrays, including real Obs without conjugate()."""
    array = np.asarray(array)
    return np.array([entry.conjugate() if hasattr(entry, 'conjugate') else entry
                     for entry in array.flat]).reshape(array.shape)


@notrace_primitive
def _check_eigenvalue_gaps(eigenvalues):
    """Reject unresolved gaps before differentiating individual eigenvectors."""
    n = len(eigenvalues)
    if n > 1 and np.any(np.diff(eigenvalues) <= 100 * n * np.finfo(float).eps * np.max(np.abs(eigenvalues))):
        raise np.linalg.LinAlgError('Eigenvector derivatives are undefined for degenerate eigenvalues.')


def _eigh_complex(obs, metric=None):
    r"""Propagate a Hermitian definite eigensystem using real-coordinate Jacobians.

    Parameters
    ----------
    obs : array_like, shape (N, N)
        Hermitian matrix A, with Obs, CObs or numeric entries.
    metric : array_like, shape (N, N), optional
        Hermitian reference matrix B in A v = lambda B v. It must have the
        same shape as obs and a positive-definite central value. In a correlator
        GEVP this is C(t0). If None, B is the identity and an ordinary Hermitian
        eigenproblem is solved. Uncertainties in B and their correlations with
        A are included in the error propagation.

    Returns
    -------
    eigenvalues : numpy.ndarray of Obs, shape (N,)
        Real eigenvalues in ascending order.
    eigenvectors : numpy.ndarray of CObs, shape (N, N)
        Eigenvectors as columns, normalized with respect to B: V^H B V = I.
        Thus B defines the inner product used for orthonormality.

    Notes
    -----
    Let A and B denote the central matrices, with a nondegenerate eigensystem

    $$
    A v_k = \lambda_k B v_k, \qquad v_m^\dagger B v_k = \delta_{mk}.
    $$

    All derivatives below are evaluated at these central values, for Hermitian
    perturbations dA and dB. Differentiating the eigenvalue equation and
    multiplying by the corresponding left eigenvector cancels the dv terms:

    $$
    d\lambda_k = v_k^\dagger(dA - \lambda_k dB)v_k.
    $$

    Expand the vector variation as $dv_k = \sum_m v_m c_m$. Projection of the
    differentiated equation onto v_m gives, for m != k,

    $$
    c_m = \frac{v_m^\dagger(dA - \lambda_k dB)v_k}{\lambda_k - \lambda_m}.
    $$

    The coefficient along v_k is fixed by differentiating its normalization,
    together with a choice of local phase:

    $$
    2\operatorname{Re}(v_k^\dagger B\,dv_k) + v_k^\dagger dB v_k = 0,
    \qquad \operatorname{Im}(v_k^\dagger B\,dv_k) = 0.
    $$

    Hence the complete phase-fixed vector derivative is

    $$
    dv_k = \sum_{m\ne k} v_m
    \frac{v_m^\dagger(dA - \lambda_k dB)v_k}{\lambda_k - \lambda_m}
    - \frac12 v_k(v_k^\dagger dB v_k).
    $$

    The numerical vectors are aligned to the fixed central vectors by making
    their overlaps, evaluated with the central B, real and positive.

    Degeneracies prevent unique individual vector derivatives.
    Numerically unresolved eigenvalue gaps raise LinAlgError; larger gaps can
    still require care if they are small compared with the input fluctuations.
    """
    obs = np.asarray(obs, dtype=object)
    if obs.ndim != 2 or obs.shape[0] != obs.shape[1]:
        raise ValueError('Expected a square Hermitian matrix.')
    if metric is None:
        metric = np.eye(obs.shape[0])
    metric = np.asarray(metric, dtype=object)
    if metric.shape != obs.shape:
        raise ValueError('Matrix and metric must have the same shape.')

    data = np.empty((4, *obs.shape), dtype=object)
    for k, matrix in enumerate((obs, metric)):
        for ij, entry in np.ndenumerate(matrix):
            data[(2 * k, *ij)] = entry if isinstance(entry, Obs) else entry.real
            data[(2 * k + 1, *ij)] = 0.0 if isinstance(entry, Obs) else entry.imag

    def unpack(x):
        """Hermitian matrices from unrestricted real/imaginary coordinates."""
        a, b = x[0] + 1j * x[1], x[2] + 1j * x[3]
        # Extend to unconstrained real coordinates for the manual Jacobian.
        return (a + a.conj().T) / 2, (b + b.conj().T) / 2

    a = np.asarray(_matrix_values(obs), dtype=complex)
    b = np.asarray(_matrix_values(metric), dtype=complex)
    for matrix in (a, b):
        if not np.allclose(matrix, matrix.conj().T):
            raise ValueError('Expected Hermitian matrix and metric.')
    a, b = (a + a.conj().T) / 2, (b + b.conj().T) / 2
    w, v = scipy.linalg.eigh(a, b)
    n = len(w)
    _check_eigenvalue_gaps(w)

    # d lambda_n = v_n^H (dA - lambda_n dB) v_n.
    q = np.einsum('in,jn->nij', v.conj(), v)
    jac = np.empty((n + 2 * n * n, *data.shape))
    jac[:n] = np.stack((q.real, -q.imag, -w[:, None, None] * q.real,
                       w[:, None, None] * q.imag), axis=1)
    for index in np.ndindex(data.shape):
        dx = np.zeros(data.shape)
        dx[index] = 1
        da, db = unpack(dx)
        dv = np.empty_like(v)
        for k in range(n):
            coeff = v.conj().T @ (da - w[k] * db) @ v[:, k]
            for m in range(n):
                if m != k:
                    coeff[m] /= w[k] - w[m]
                else:
                    coeff[m] = -0.5 * (v[:, k].conj() @ db @ v[:, k]).real
            dv[:, k] = v @ coeff
        jac[(slice(n, n + n * n), *index)] = dv.real.ravel()
        jac[(slice(n + n * n, None), *index)] = dv.imag.ravel()

    def eigensystem(x, **kwargs):
        """Pack real outputs after alignment to the fixed central phase."""
        aa, bb = unpack(x)
        ww, vv = scipy.linalg.eigh(aa, bb)
        # Align to a fixed central reference, also for per-replica means.
        overlap = np.diag(v.conj().T @ b @ vv)
        vv = vv * np.exp(-1j * np.angle(overlap))[None, :]
        return np.concatenate((ww, vv.real.ravel(), vv.imag.ravel()))

    result = derived_observable(eigensystem, data, man_grad=jac)
    vr = result[n:n + n * n].reshape(obs.shape)
    vi = result[n + n * n:].reshape(obs.shape)
    vectors = np.empty(obs.shape, dtype=object)
    for ij in np.ndindex(obs.shape):
        vectors[ij] = CObs(vr[ij], vi[ij])
    return result[:n], vectors


def pinv(obs, **kwargs):
    """Computes the Moore-Penrose pseudoinverse of a matrix of Obs."""
    return derived_observable(lambda x, **kwargs: anp.linalg.pinv(x), obs)


def svd(obs, **kwargs):
    """Computes the singular value decomposition of a matrix of Obs."""
    u = derived_observable(lambda x, **kwargs: anp.linalg.svd(x, full_matrices=False)[0], obs)
    s = derived_observable(lambda x, **kwargs: anp.linalg.svd(x, full_matrices=False)[1], obs)
    vh = derived_observable(lambda x, **kwargs: anp.linalg.svd(x, full_matrices=False)[2], obs)
    return (u, s, vh)
