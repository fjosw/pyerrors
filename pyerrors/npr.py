import numpy as np


_gamma = ['gammas', 0, 0, 0, 0, 0]
_gamma[1] = np.array(
    [[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]],
    dtype=complex)
_gamma[2] = np.array(
    [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
    dtype=complex)
_gamma[3] = np.array(
    [[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]],
    dtype=complex)
_gamma[4] = np.array(
    [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
    dtype=complex)
_gamma[5] = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
    dtype=complex)
_imat = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=complex)


def gamma_matrix(gamma_tag):
    """Returns gamma matrix in Grid labeling."""
    if gamma_tag == 'Identity':
        g = _imat
    elif gamma_tag == 'Gamma5':
        g = _gamma[5]
    elif gamma_tag == 'GammaX':
        g = _gamma[1]
    elif gamma_tag == 'GammaY':
        g = _gamma[2]
    elif gamma_tag == 'GammaZ':
        g = _gamma[3]
    elif gamma_tag == 'GammaT':
        g = _gamma[4]
    elif gamma_tag == 'GammaXGamma5':
        g = _gamma[1] @ _gamma[5]
    elif gamma_tag == 'GammaYGamma5':
        g = _gamma[2] @ _gamma[5]
    elif gamma_tag == 'GammaZGamma5':
        g = _gamma[3] @ _gamma[5]
    elif gamma_tag == 'GammaTGamma5':
        g = _gamma[4] @ _gamma[5]
    elif gamma_tag == 'SigmaXT':
        g = 0.5 * (_gamma[1] @ _gamma[4] - _gamma[4] @ _gamma[1])
    elif gamma_tag == 'SigmaXY':
        g = 0.5 * (_gamma[1] @ _gamma[2] - _gamma[2] @ _gamma[1])
    elif gamma_tag == 'SigmaXZ':
        g = 0.5 * (_gamma[1] @ _gamma[3] - _gamma[3] @ _gamma[1])
    elif gamma_tag == 'SigmaYT':
        g = 0.5 * (_gamma[2] @ _gamma[4] - _gamma[4] @ _gamma[2])
    elif gamma_tag == 'SigmaYZ':
        g = 0.5 * (_gamma[2] @ _gamma[3] - _gamma[3] @ _gamma[2])
    elif gamma_tag == 'SigmaZT':
        g = 0.5 * (_gamma[3] @ _gamma[4] - _gamma[4] @ _gamma[3])
    else:
        raise Exception('Unkown gamma structure', gamma_tag)
    return g

class Npr_matrix(np.ndarray):

    g5 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
                  dtype=complex)

    def __new__(cls, input_array, mom_in=None, mom_out=None):
        obj = np.asarray(input_array).view(cls)
        obj.mom_in = mom_in
        obj.mom_out = mom_out
        return obj

    @property
    def g5H(self):
        new_matrix = Npr_matrix.g5 @ self.conj().T @ Npr_matrix.g5
        new_matrix.mom_in = self.mom_out
        new_matrix.mom_out = self.mom_in
        return new_matrix

    def __matmul__(self, other):
        if hasattr(other, 'mom_in'):
            if self.mom_in != other.mom_in and self.mom_in and other.mom_in:
                    raise Exception('mom_in does not match.')
            mom_in = self.mom_in if self.mom_in else other.mom_in
        else:
            mom_in = self.mom_in

        if hasattr(other, 'mom_out'):
            if self.mom_out != other.mom_out and self.mom_out and other.mom_out:
                    raise Exception('mom_out does not match.')
            mom_out = self.mom_out if self.mom_out else other.mom_out
        else:
            mom_out = self.mom_out

        return self.__new__(Npr_matrix, super().__matmul__(other), mom_in, mom_out)

    def __array_finalize__(self, obj):
        if obj is None: return
        self.mom_in = getattr(obj, 'mom_in', None)
        self.mom_out = getattr(obj, 'mom_out', None)
