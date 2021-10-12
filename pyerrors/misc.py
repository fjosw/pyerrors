#!/usr/bin/env python
# coding: utf-8

import gc
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
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


def ks_test(obs=None):
    """Performs a Kolmogorov–Smirnov test for the Q-values of a list of Obs.

    If no list is given all Obs in memory are used.

    Disclaimer: The determination of the individual Q-values as well as this function have not been tested yet.
    """

    if obs is None:
        obs_list = []
        for obj in gc.get_objects():
            if isinstance(obj, Obs):
                obs_list.append(obj)
    else:
        obs_list = obs

    Qs = []
    for obs_i in obs_list:
        for ens in obs_i.e_names:
            if obs_i.e_Q[ens] is not None:
                Qs.append(obs_i.e_Q[ens])

    bins = len(Qs)
    x = np.arange(0, 1.001, 0.001)
    plt.plot(x, x, 'k', zorder=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Q value')
    plt.ylabel('Cumulative probability')
    plt.title(str(bins) + ' Q values')

    n = np.arange(1, bins + 1) / np.float64(bins)
    Xs = np.sort(Qs)
    plt.step(Xs, n)
    diffs = n - Xs
    loc_max_diff = np.argmax(np.abs(diffs))
    loc = Xs[loc_max_diff]
    plt.annotate(s='', xy=(loc, loc), xytext=(loc, loc + diffs[loc_max_diff]), arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))
    plt.show()

    print(scipy.stats.kstest(Qs, 'uniform'))
