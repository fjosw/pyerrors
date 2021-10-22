import warnings
import numpy as np
from .linalg import inv, matmul
from .dirac import gamma, gamma5


L = None
T = None


class Npr_matrix(np.ndarray):

    def __new__(cls, input_array, mom_in=None, mom_out=None):
        obj = np.asarray(input_array).view(cls)
        obj.mom_in = mom_in
        obj.mom_out = mom_out
        return obj

    @property
    def g5H(self):
        """Gamma_5 hermitean conjugate

        Returns gamma_5 @ M.T.conj() @ gamma_5 and exchanges in and out going
        momenta. Works only for 12x12 matrices.
        """
        if self.shape != (12, 12):
            raise Exception('g5H only works for 12x12 matrices.')
        extended_g5 = np.kron(np.eye(3, dtype=int), gamma5)
        new_matrix = extended_g5 @ self.conj().T @ extended_g5
        new_matrix.mom_in = self.mom_out
        new_matrix.mom_out = self.mom_in
        return new_matrix

    def _propagate_mom(self, other, name):
        s_mom = getattr(self, name, None)
        o_mom = getattr(other, name, None)
        if s_mom is not None and o_mom is not None:
            if not np.allclose(s_mom, o_mom):
                raise Exception(name + ' does not match.')
        return o_mom if o_mom is not None else s_mom

    def __matmul__(self, other):
        return self.__new__(Npr_matrix,
                            super().__matmul__(other),
                            self._propagate_mom(other, 'mom_in'),
                            self._propagate_mom(other, 'mom_out'))

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.mom_in = getattr(obj, 'mom_in', None)
        self.mom_out = getattr(obj, 'mom_out', None)


def _check_geometry():
    if L is None:
        raise Exception("Spatial extent 'L' not set.")
    else:
        if not isinstance(L, int):
            raise Exception("Spatial extent 'L' must be an integer.")
    if T is None:
        raise Exception("Temporal extent 'T' not set.")
        if not isinstance(T, int):
            raise Exception("Temporal extent 'T' must be an integer.")


def inv_propagator(prop):
    """ Inverts a 12x12 quark propagator"""
    if prop.shape != (12, 12):
        raise Exception("Only 12x12 propagators can be inverted.")
    return Npr_matrix(inv(prop), prop.mom_in)


def Zq(inv_prop, fermion='Wilson'):
    """ Calculates the quark field renormalization constant Zq

        Attributes:
        inv_prop -- Inverted 12x12 quark propagator
        fermion -- Fermion type for which the tree-level propagator is used
                   in the calculation of Zq. Default Wilson.
    """
    _check_geometry()
    mom = np.copy(inv_prop.mom_in)
    mom[3] /= T / L
    sin_mom = np.sin(2 * np.pi / L * mom)

    if fermion == 'Wilson':
        p_slash = -1j * (sin_mom[0] * gamma[0] + sin_mom[1] * gamma[1] + sin_mom[2] * gamma[2] + sin_mom[3] * gamma[3]) / np.sum(sin_mom ** 2)
    else:
        raise Exception("Fermion type '" + fermion + "' not implemented")

    res = 1 / 12. * np.trace(matmul(inv_prop, np.kron(np.eye(3, dtype=int), p_slash)))
    res.gamma_method()

    if not res.imag.is_zero_within_error(5):
        warnings.warn("Imaginary part of Zq is not zero within 5 sigma")
        return res
    return res.real
