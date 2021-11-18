import warnings
import numpy as np
from .linalg import inv, matmul
from .dirac import gamma


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

        Uses the fact that the propagator is gamma5 hermitean, so just the
        in and out momenta of the propagator are exchanged.
        """
        return Npr_matrix(self,
                          mom_in=self.mom_out,
                          mom_out=self.mom_in)

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

        Parameters
        ----------
        inv_prop : array
            Inverted 12x12 quark propagator
        fermion : str
            Fermion type for which the tree-level propagator is used
                   in the calculation of Zq. Default Wilson.
    """
    _check_geometry()
    mom = np.copy(inv_prop.mom_in)
    mom[3] /= T / L
    sin_mom = np.sin(2 * np.pi / L * mom)

    if fermion == 'Wilson':
        p_slash = -1j * (sin_mom[0] * gamma[0] + sin_mom[1] * gamma[1] + sin_mom[2] * gamma[2] + sin_mom[3] * gamma[3]) / np.sum(sin_mom ** 2)
    elif fermion == 'Continuum':
        p_mom = 2 * np.pi / L * mom
        p_slash = -1j * (p_mom[0] * gamma[0] + p_mom[1] * gamma[1] + p_mom[2] * gamma[2] + p_mom[3] * gamma[3]) / np.sum(p_mom ** 2)
    elif fermion == 'DWF':
        W = np.sum(1 - np.cos(2 * np.pi / L * mom))
        s2 = np.sum(sin_mom ** 2)
        p_slash = -1j * (sin_mom[0] * gamma[0] + sin_mom[1] * gamma[1] + sin_mom[2] * gamma[2] + sin_mom[3] * gamma[3])
        p_slash /= 2 * (W - 1 + np.sqrt((1 - W) ** 2 + s2))
    else:
        raise Exception("Fermion type '" + fermion + "' not implemented")

    res = 1 / 12. * np.trace(matmul(inv_prop, np.kron(np.eye(3, dtype=int), p_slash)))
    res.gamma_method()

    if not res.imag.is_zero_within_error(5):
        warnings.warn("Imaginary part of Zq is not zero within 5 sigma")
        return res

    res.real.tag = "Zq '" + fermion + "', p=" + str(inv_prop.mom_in)

    return res.real
