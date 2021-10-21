#!/usr/bin/env python
# coding: utf-8

import numpy as np
from .pyerrors import Obs


def gen_correlated_data(means, cov, name, tau=0.5, samples=1000):
    """ Generate observables with given covariance and autocorrelation times.

    Arguments
    -----------------
    means -- list containing the mean value of each observable.
    cov -- covariance matrix for the data to be geneated.
    name -- ensemble name for the data to be geneated.
    tau -- can either be a real number or a list with an entry for
           every dataset.
    samples -- number of samples to be generated for each observable.
    """

    assert len(means) == cov.shape[-1]
    tau = np.asarray(tau)
    if np.min(tau) < 0.5:
        raise Exception('All integrated autocorrelations have to be >= 0.5.')

    a = (2 * tau - 1) / (2 * tau + 1)
    rand = np.random.multivariate_normal(np.zeros_like(means), cov * samples, samples)

    # Normalize samples such that sample variance matches input
    norm = np.array([np.var(o, ddof=1) / samples for o in rand.T])
    rand = rand @ np.diag(np.sqrt(np.diag(cov))) @ np.diag(1 / np.sqrt(norm))

    data = [rand[0]]
    for i in range(1, samples):
        data.append(np.sqrt(1 - a ** 2) * rand[i] + a * data[-1])
    corr_data = np.array(data) - np.mean(data, axis=0) + means
    return [Obs([dat], [name]) for dat in corr_data.T]
