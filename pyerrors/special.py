import scipy
import numpy as np
from autograd.extend import primitive, defvjp
from autograd.scipy.special import j0, y0, j1, y1, jn, yn, i0, i1, iv, ive, beta, betainc, betaln
from autograd.scipy.special import polygamma, psi, digamma, gamma, gammaln, gammainc, gammaincc, gammasgn, rgamma, multigammaln
from autograd.scipy.special import erf, erfc, erfinv, erfcinv, logit, expit, logsumexp


__all__ = ["beta", "betainc", "betaln",
           "polygamma", "psi", "digamma", "gamma", "gammaln", "gammainc", "gammaincc", "gammasgn", "rgamma", "multigammaln",
           "kn", "j0", "y0", "j1", "y1", "jn", "yn", "i0", "i1", "iv", "ive",
           "erf", "erfc", "erfinv", "erfcinv", "logit", "expit", "logsumexp"]


@primitive
def kn(n, x):
    """Modified Bessel function of the second kind of integer order n"""
    if int(n) != n:
        raise TypeError("The order 'n' needs to be an integer.")
    return scipy.special.kn(n, x)


defvjp(kn, None, lambda ans, n, x: lambda g: - g * 0.5 * (kn(np.abs(n - 1), x) + kn(n + 1, x)))
