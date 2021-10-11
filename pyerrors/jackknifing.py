#!/usr/bin/env python
# coding: utf-8

import pickle
import matplotlib.pyplot as plt
import numpy as np


def _jack_error(jack):
    n = jack.size
    mean = np.mean(jack)
    error = 0
    for i in range(n):
        error += (jack[i] - mean) ** 2

    return np.sqrt((n - 1) / n * error)


class Jack:

    def __init__(self, value, jacks):
        self.jacks = jacks
        self.N = list(map(np.size, self.jacks))
        self.max_binsize = len(self.N)
        self.value = value  # list(map(np.mean, self.jacks))
        self.dvalue = list(map(_jack_error, self.jacks))

    def print(self, **kwargs):
        """Print basic properties of the Jack."""

        if 'binsize' in kwargs:
            b = kwargs.get('binsize') - 1
            if b == -1:
                b = 0
            if not isinstance(b, int):
                raise TypeError('binsize has to be integer')
            if b + 1 > self.max_binsize:
                raise Exception('Chosen binsize not calculated')
        else:
            b = 0

        print('Result:\t %3.8e +/- %3.8e +/- %3.8e (%3.3f%%)' % (self.value, self.dvalue[b], self.dvalue[b] * np.sqrt(2 * b / self.N[0]), np.abs(self.dvalue[b] / self.value * 100)))

    def plot_tauint(self):
        plt.xlabel('binsize')
        plt.ylabel('tauint')
        length = self.max_binsize
        x = np.arange(length) + 1
        plt.errorbar(x[:], (self.dvalue[:] / self.dvalue[0]) ** 2 / 2, yerr=np.sqrt(((2 * (self.dvalue[:] / self.dvalue[0]) ** 2 * np.sqrt(2 * x[:] / self.N[0])) / 2) ** 2 + ((2 * (self.dvalue[:] / self.dvalue[0]) ** 2 * np.sqrt(2 / self.N[0])) / 2) ** 2), linewidth=1, capsize=2)
        plt.xlim(0.5, length + 0.5)
        plt.title('Tauint')
        plt.show()

    def plot_history(self):
        N = self.N
        x = np.arange(N)
        tmp = []
        for i in range(self.replicas):
            tmp.append(self.deltas[i] + self.r_values[i])
        y = np.concatenate(tmp, axis=0)  # Think about including kwarg to look only at some replica
        plt.errorbar(x, y, fmt='.', markersize=3)
        plt.xlim(-0.5, N - 0.5)
        plt.show()

    def dump(self, name, **kwargs):
        """Dump the Jack to a pickle file 'name'.

        Keyword arguments:
        path -- specifies a custom path for the file (default '.')
        """
        if 'path' in kwargs:
            file_name = kwargs.get('path') + '/' + name + '.p'
        else:
            file_name = name + '.p'
        with open(file_name, 'wb') as fb:
            pickle.dump(self, fb)


def generate_jack(obs, **kwargs):
    full_data = []
    for r, name in enumerate(obs.names):
        if r == 0:
            full_data = obs.deltas[name] + obs.r_values[name]
        else:
            full_data = np.append(full_data, obs.deltas[name] + obs.r_values[name])

    jacks = []
    if 'max_binsize' in kwargs:
        max_b = kwargs.get('max_binsize')
        if not isinstance(max_b, int):
            raise TypeError('max_binsize has to be integer')
    else:
        max_b = 1

    for b in range(max_b):
        # binning if necessary
        if b > 0:
            n = full_data.size // (b + 1)
            binned_data = np.zeros(n)
            for i in range(n):
                for j in range(b + 1):
                    binned_data[i] += full_data[i * (b + 1) + j]
                binned_data[i] /= (b + 1)
        else:
            binned_data = full_data
            n = binned_data.size
        # generate jacks from data
        mean = np.mean(binned_data)
        tmp_jacks = np.zeros(n)
        for i in range(n):
            tmp_jacks[i] = (n * mean - binned_data[i]) / (n - 1)
        jacks.append(tmp_jacks)

        # Value is not correctly reproduced here
    return Jack(obs.value, jacks)


def derived_jack(func, data, **kwargs):
    """Construct a derived Jack according to func(data, **kwargs).

    Parameters
    ----------
    func -- arbitrary function of the form func(data, **kwargs). For the automatic differentiation to work,
            all numpy functions have to have the autograd wrapper (use 'import autograd.numpy as np').
    data -- list of Jacks, e.g. [jack1, jack2, jack3].

    Notes
    -----
    For simple mathematical operations it can be practical to use anonymous functions.
    For the ratio of two jacks one can e.g. use

    new_jack = derived_jack(lambda x : x[0] / x[1], [jack1, jack2])

    """

    # Check shapes of data
    if not all(x.N == data[0].N for x in data):
        raise Exception('Error: Shape of data does not fit')

    values = np.zeros(len(data))
    for j, item in enumerate(data):
        values[j] = item.value
    new_value = func(values, **kwargs)

    jacks = []
    for b in range(data[0].max_binsize):
        tmp_jacks = np.zeros(data[0].N[b])
        for i in range(data[0].N[b]):
            values = np.zeros(len(data))
            for j, item in enumerate(data):
                values[j] = item.jacks[b][i]
            tmp_jacks[i] = func(values, **kwargs)
        jacks.append(tmp_jacks)

    return Jack(new_value, jacks)
