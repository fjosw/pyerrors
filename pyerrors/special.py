import numpy as np
import scipy
from autograd.extend import defvjp, primitive
from autograd.scipy.special import (
    beta,
    betainc,
    betaln,
    digamma,
    erf,
    erfc,
    erfcinv,
    erfinv,
    expit,
    gamma,
    gammainc,
    gammaincc,
    gammaln,
    gammasgn,
    i0,
    i1,
    iv,
    ive,
    j0,
    j1,
    jn,
    logit,
    logsumexp,
    multigammaln,
    polygamma,
    psi,
    rgamma,
    y0,
    y1,
    yn,
)

__all__ = [
    "beta",
    "betainc",
    "betaln",
    "digamma",
    "erf",
    "erfc",
    "erfcinv",
    "erfinv",
    "expit",
    "gamma",
    "gammainc",
    "gammaincc",
    "gammaln",
    "gammasgn",
    "i0",
    "i1",
    "iv",
    "ive",
    "j0",
    "j1",
    "jn",
    "kn",
    "logit",
    "logsumexp",
    "multigammaln",
    "polygamma",
    "psi",
    "rgamma",
    "y0",
    "y1",
    "yn",
]


@primitive
def kn(n, x):
    """Modified Bessel function of the second kind of integer order n"""
    if int(n) != n:
        raise TypeError("The order 'n' needs to be an integer.")
    return scipy.special.kn(n, x)


defvjp(kn, None, lambda ans, n, x: lambda g: - g * 0.5 * (kn(np.abs(n - 1), x) + kn(n + 1, x)))
