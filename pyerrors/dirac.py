import numpy as np


gammaX = np.array(
    [[0, 0, 0, 1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [-1j, 0, 0, 0]],
    dtype=complex)
gammaY = np.array(
    [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
    dtype=complex)
gammaZ = np.array(
    [[0, 0, 1j, 0], [0, 0, 0, -1j], [-1j, 0, 0, 0], [0, 1j, 0, 0]],
    dtype=complex)
gammaT = np.array(
    [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
    dtype=complex)
gamma = np.array([gammaX, gammaY, gammaZ, gammaT])
gamma5 = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
    dtype=complex)
identity = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=complex)


def Grid_gamma(gamma_tag):
    """Returns gamma matrix in Grid labeling."""
    if gamma_tag == 'Identity':
        g = identity
    elif gamma_tag == 'Gamma5':
        g = gamma5
    elif gamma_tag == 'GammaX':
        g = gamma[0]
    elif gamma_tag == 'GammaY':
        g = gamma[1]
    elif gamma_tag == 'GammaZ':
        g = gamma[2]
    elif gamma_tag == 'GammaT':
        g = gamma[3]
    elif gamma_tag == 'GammaXGamma5':
        g = gamma[0] @ gamma5
    elif gamma_tag == 'GammaYGamma5':
        g = gamma[1] @ gamma5
    elif gamma_tag == 'GammaZGamma5':
        g = gamma[2] @ gamma5
    elif gamma_tag == 'GammaTGamma5':
        g = gamma[3] @ gamma5
    elif gamma_tag == 'SigmaXT':
        g = 0.5 * (gamma[0] @ gamma[3] - gamma[3] @ gamma[0])
    elif gamma_tag == 'SigmaXY':
        g = 0.5 * (gamma[0] @ gamma[1] - gamma[1] @ gamma[0])
    elif gamma_tag == 'SigmaXZ':
        g = 0.5 * (gamma[0] @ gamma[2] - gamma[2] @ gamma[0])
    elif gamma_tag == 'SigmaYT':
        g = 0.5 * (gamma[1] @ gamma[3] - gamma[3] @ gamma[1])
    elif gamma_tag == 'SigmaYZ':
        g = 0.5 * (gamma[1] @ gamma[2] - gamma[2] @ gamma[1])
    elif gamma_tag == 'SigmaZT':
        g = 0.5 * (gamma[2] @ gamma[3] - gamma[3] @ gamma[2])
    else:
        raise Exception('Unkown gamma structure', gamma_tag)
    return g
