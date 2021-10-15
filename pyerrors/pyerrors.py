#!/usr/bin/env python
# coding: utf-8

import warnings
import pickle
import numpy as np
import autograd.numpy as anp  # Thinly-wrapped numpy
from autograd import jacobian
import matplotlib.pyplot as plt
import numdifftools as nd
import scipy.special


class Obs:
    """Class for a general observable.

    Instances of Obs are the basic objects of a pyerrors error analysis.
    They are initialized with a list which contains arrays of samples for
    different ensembles/replica and another list of same length which contains
    the names of the ensembles/replica. Mathematical operations can be
    performed on instances. The result is another instance of Obs. The error of
    an instance can be computed with the gamma_method. Also contains additional
    methods for output and visualization of the error calculation.

    Attributes
    ----------
    e_tag_global -- Integer which determines which part of the name belongs
                    to the ensemble and which to the replicum.
    S_global -- Standard value for S (default 2.0)
    S_dict -- Dictionary for S values. If an entry for a given ensemble
              exists this overwrites the standard value for that ensemble.
    tau_exp_global -- Standard value for tau_exp (default 0.0)
    tau_exp_dict -- Dictionary for tau_exp values. If an entry for a given
                    ensemble exists this overwrites the standard value for that
                    ensemble.
    N_sigma_global -- Standard value for N_sigma (default 1.0)
    """

    e_tag_global = 0
    S_global = 2.0
    S_dict = {}
    tau_exp_global = 0.0
    tau_exp_dict = {}
    N_sigma_global = 1.0

    def __init__(self, samples, names):

        if len(samples) != len(names):
            raise Exception('Length of samples and names incompatible.')
        if len(names) != len(set(names)):
            raise Exception('Names are not unique.')
        if not all(isinstance(x, str) for x in names):
            raise TypeError('All names have to be strings.')
        if min(len(x) for x in samples) <= 4:
            raise Exception('Samples have to have at least 4 entries.')

        self.names = sorted(names)
        self.shape = {}
        self.r_values = {}
        self.deltas = {}
        for name, sample in sorted(zip(names, samples)):
            self.shape[name] = np.size(sample)
            self.r_values[name] = np.mean(sample)
            self.deltas[name] = sample - self.r_values[name]

        self.N = sum(map(np.size, list(self.deltas.values())))

        self.value = 0
        for name in self.names:
            self.value += self.shape[name] * self.r_values[name]
        self.value /= self.N

        self.dvalue = 0.0
        self.ddvalue = 0.0
        self.reweighted = 0

        self.S = {}
        self.tau_exp = {}
        self.N_sigma = 0

        self.e_names = {}
        self.e_content = {}

        self.e_dvalue = {}
        self.e_ddvalue = {}
        self.e_tauint = {}
        self.e_dtauint = {}
        self.e_windowsize = {}
        self.e_Q = {}
        self.e_rho = {}
        self.e_drho = {}
        self.e_n_tauint = {}
        self.e_n_dtauint = {}

    def gamma_method(self, **kwargs):
        """Calculate the error and related properties of the Obs.

        Keyword arguments
        -----------------
        S -- specifies a custom value for the parameter S (default 2.0), can be
             a float or an array of floats for different ensembles
        tau_exp -- positive value triggers the critical slowing down analysis
                   (default 0.0), can be a float or an array of floats for
                   different ensembles
        N_sigma -- number of standard deviations from zero until the tail is
                   attached to the autocorrelation function (default 1)
        e_tag -- number of characters which label the ensemble. The remaining
                 ones label replica (default 0)
        fft -- boolean, which determines whether the fft algorithm is used for
               the computation of the autocorrelation function (default True)
        """

        if 'e_tag' in kwargs:
            e_tag_local = kwargs.get('e_tag')
            if not isinstance(e_tag_local, int):
                raise TypeError('Error: e_tag is not integer')
        else:
            e_tag_local = Obs.e_tag_global

        self.e_names = sorted(set([o[:e_tag_local] for o in self.names]))
        self.e_content = {}
        self.e_dvalue = {}
        self.e_ddvalue = {}
        self.e_tauint = {}
        self.e_dtauint = {}
        self.e_windowsize = {}
        self.e_n_tauint = {}
        self.e_n_dtauint = {}
        e_gamma = {}
        self.e_rho = {}
        self.e_drho = {}
        self.dvalue = 0
        self.ddvalue = 0

        self.S = {}
        self.tau_exp = {}

        if kwargs.get('fft') is False:
            fft = False
        else:
            fft = True

        if 'S' in kwargs:
            tmp = kwargs.get('S')
            if isinstance(tmp, list):
                if len(tmp) != len(self.e_names):
                    raise Exception('Length of S array does not match ensembles.')
                for e, e_name in enumerate(self.e_names):
                    if tmp[e] <= 0:
                        raise Exception('S has to be larger than 0.')
                    self.S[e_name] = tmp[e]
            else:
                if isinstance(tmp, (int, float)):
                    if tmp <= 0:
                        raise Exception('S has to be larger than 0.')
                    for e, e_name in enumerate(self.e_names):
                        self.S[e_name] = tmp
                else:
                    raise TypeError('S is not in proper format.')
        else:
            for e, e_name in enumerate(self.e_names):
                if e_name in Obs.S_dict:
                    self.S[e_name] = Obs.S_dict[e_name]
                else:
                    self.S[e_name] = Obs.S_global

        if 'tau_exp' in kwargs:
            tmp = kwargs.get('tau_exp')
            if isinstance(tmp, list):
                if len(tmp) != len(self.e_names):
                    raise Exception('Length of tau_exp array does not match ensembles.')
                for e, e_name in enumerate(self.e_names):
                    if tmp[e] < 0:
                        raise Exception('tau_exp smaller than 0.')
                    self.tau_exp[e_name] = tmp[e]
            else:
                if isinstance(tmp, (int, float)):
                    if tmp < 0:
                        raise Exception('tau_exp smaller than 0.')
                    for e, e_name in enumerate(self.e_names):
                        self.tau_exp[e_name] = tmp
                else:
                    raise TypeError('tau_exp is not in proper format.')
        else:
            for e, e_name in enumerate(self.e_names):
                if e_name in Obs.tau_exp_dict:
                    self.tau_exp[e_name] = Obs.tau_exp_dict[e_name]
                else:
                    self.tau_exp[e_name] = Obs.tau_exp_global

        if 'N_sigma' in kwargs:
            self.N_sigma = kwargs.get('N_sigma')
            if not isinstance(self.N_sigma, (int, float)):
                raise TypeError('N_sigma is not a number.')
        else:
            self.N_sigma = Obs.N_sigma_global

        if max([len(x) for x in self.names]) <= e_tag_local:
            for e, e_name in enumerate(self.e_names):
                self.e_content[e_name] = [e_name]
        else:
            for e, e_name in enumerate(self.e_names):
                if len(e_name) < e_tag_local:
                    self.e_content[e_name] = [e_name]
                else:
                    self.e_content[e_name] = sorted(filter(lambda x: x.startswith(e_name), self.names))

        for e, e_name in enumerate(self.e_names):

            r_length = []
            for r_name in self.e_content[e_name]:
                r_length.append(len(self.deltas[r_name]))

            e_N = np.sum(r_length)
            w_max = max(r_length) // 2
            e_gamma[e_name] = np.zeros(w_max)
            self.e_rho[e_name] = np.zeros(w_max)
            self.e_drho[e_name] = np.zeros(w_max)

            if fft:
                for r_name in self.e_content[e_name]:
                    max_gamma = min(self.shape[r_name], w_max)
                    # The padding for the fft has to be even
                    padding = self.shape[r_name] + max_gamma + (self.shape[r_name] + max_gamma) % 2
                    e_gamma[e_name][:max_gamma] += np.fft.irfft(np.abs(np.fft.rfft(self.deltas[r_name], padding)) ** 2)[:max_gamma]
            else:
                for n in range(w_max):
                    for r_name in self.e_content[e_name]:
                        if self.shape[r_name] - n >= 0:
                            e_gamma[e_name][n] += self.deltas[r_name][0:self.shape[r_name] - n].dot(self.deltas[r_name][n:self.shape[r_name]])

            e_shapes = []
            for r_name in self.e_content[e_name]:
                e_shapes.append(self.shape[r_name])

            div = np.array([])
            mul = np.array([])
            sorted_shapes = sorted(e_shapes)
            for i, item in enumerate(sorted_shapes):
                if len(div) > w_max:
                    break
                if i == 0:
                    samples = item
                else:
                    samples = item - sorted_shapes[i - 1]
                div = np.append(div, np.repeat(np.sum(sorted_shapes[i:]), samples))
                mul = np.append(mul, np.repeat(len(sorted_shapes) - i, samples))
            div = div - np.arange(len(div)) * mul

            e_gamma[e_name] /= div[:w_max]

            if np.abs(e_gamma[e_name][0]) < 10 * np.finfo(float).tiny:  # Prevent division by zero
                self.e_tauint[e_name] = 0.5
                self.e_dtauint[e_name] = 0.0
                self.e_dvalue[e_name] = 0.0
                self.e_ddvalue[e_name] = 0.0
                self.e_windowsize[e_name] = 0
                continue

            self.e_rho[e_name] = e_gamma[e_name][:w_max] / e_gamma[e_name][0]
            self.e_n_tauint[e_name] = np.cumsum(np.concatenate(([0.5], self.e_rho[e_name][1:])))
            # Make sure no entry of tauint is smaller than 0.5
            self.e_n_tauint[e_name][self.e_n_tauint[e_name] < 0.5] = 0.500000000001
            # hep-lat/0306017 eq. (42)
            self.e_n_dtauint[e_name] = self.e_n_tauint[e_name] * 2 * np.sqrt(np.abs(np.arange(w_max) + 0.5 - self.e_n_tauint[e_name]) / e_N)
            self.e_n_dtauint[e_name][0] = 0.0

            def _compute_drho(i):
                tmp = self.e_rho[e_name][i + 1:w_max] + np.concatenate([self.e_rho[e_name][i - 1::-1], self.e_rho[e_name][1:w_max - 2 * i]]) - 2 * self.e_rho[e_name][i] * self.e_rho[e_name][1:w_max - i]
                self.e_drho[e_name][i] = np.sqrt(np.sum(tmp ** 2) / e_N)

            _compute_drho(1)
            if self.tau_exp[e_name] > 0:
                # Critical slowing down analysis
                for n in range(1, w_max // 2):
                    _compute_drho(n + 1)
                    if (self.e_rho[e_name][n] - self.N_sigma * self.e_drho[e_name][n]) < 0 or n >= w_max // 2 - 2:
                        # Bias correction hep-lat/0306017 eq. (49) included
                        self.e_tauint[e_name] = self.e_n_tauint[e_name][n] * (1 + (2 * n + 1) / e_N) / (1 + 1 / e_N) + self.tau_exp[e_name] * np.abs(self.e_rho[e_name][n + 1])  # The absolute makes sure, that the tail contribution is always positive
                        self.e_dtauint[e_name] = np.sqrt(self.e_n_dtauint[e_name][n] ** 2 + self.tau_exp[e_name] ** 2 * self.e_drho[e_name][n + 1] ** 2)
                        # Error of tau_exp neglected so far, missing term: self.e_rho[e_name][n + 1] ** 2 * d_tau_exp ** 2
                        self.e_dvalue[e_name] = np.sqrt(2 * self.e_tauint[e_name] * e_gamma[e_name][0] * (1 + 1 / e_N) / e_N)
                        self.e_ddvalue[e_name] = self.e_dvalue[e_name] * np.sqrt((n + 0.5) / e_N)
                        self.e_windowsize[e_name] = n
                        break
            else:
                # Standard automatic windowing procedure
                g_w = self.S[e_name] / np.log((2 * self.e_n_tauint[e_name][1:] + 1) / (2 * self.e_n_tauint[e_name][1:] - 1))
                g_w = np.exp(- np.arange(1, w_max) / g_w) - g_w / np.sqrt(np.arange(1, w_max) * e_N)
                for n in range(1, w_max):
                    if n < w_max // 2 - 2:
                        _compute_drho(n + 1)
                    if g_w[n - 1] < 0 or n >= w_max - 1:
                        self.e_tauint[e_name] = self.e_n_tauint[e_name][n] * (1 + (2 * n + 1) / e_N) / (1 + 1 / e_N)  # Bias correction hep-lat/0306017 eq. (49)
                        self.e_dtauint[e_name] = self.e_n_dtauint[e_name][n]
                        self.e_dvalue[e_name] = np.sqrt(2 * self.e_tauint[e_name] * e_gamma[e_name][0] * (1 + 1 / e_N) / e_N)
                        self.e_ddvalue[e_name] = self.e_dvalue[e_name] * np.sqrt((n + 0.5) / e_N)
                        self.e_windowsize[e_name] = n
                        break

            if len(self.e_content[e_name]) > 1 and self.e_dvalue[e_name] > np.finfo(np.float64).eps:
                e_mean = 0
                for r_name in self.e_content[e_name]:
                    e_mean += self.shape[r_name] * self.r_values[r_name]
                e_mean /= e_N
                xi2 = 0
                for r_name in self.e_content[e_name]:
                    xi2 += self.shape[r_name] * (self.r_values[r_name] - e_mean) ** 2
                xi2 /= self.e_dvalue[e_name] ** 2 * e_N
                self.e_Q[e_name] = 1 - scipy.special.gammainc((len(self.e_content[e_name]) - 1.0) / 2.0, xi2 / 2.0)
            else:
                self.e_Q[e_name] = None

            self.dvalue += self.e_dvalue[e_name] ** 2
            self.ddvalue += (self.e_dvalue[e_name] * self.e_ddvalue[e_name]) ** 2

        self.dvalue = np.sqrt(self.dvalue)
        if self.dvalue == 0.0:
            self.ddvalue = 0.0
        else:
            self.ddvalue = np.sqrt(self.ddvalue) / self.dvalue
        return 0

    def print(self, level=1):
        """Print basic properties of the Obs."""
        if level == 0:
            print(self)
        else:
            print('Result\t %3.8e +/- %3.8e +/- %3.8e (%3.3f%%)' % (self.value, self.dvalue, self.ddvalue, np.abs(self.dvalue / self.value) * 100))
            if len(self.e_names) > 1:
                print(' Ensemble errors:')
            for e_name in self.e_names:
                if len(self.e_names) > 1:
                    print('', e_name, '\t %3.8e +/- %3.8e' % (self.e_dvalue[e_name], self.e_ddvalue[e_name]))
                if self.tau_exp[e_name] > 0:
                    print('  t_int\t %3.8e +/- %3.8e tau_exp = %3.2f,  N_sigma = %1.0i' % (self.e_tauint[e_name], self.e_dtauint[e_name], self.tau_exp[e_name], self.N_sigma))
                else:
                    print('  t_int\t %3.8e +/- %3.8e S = %3.2f' % (self.e_tauint[e_name], self.e_dtauint[e_name], self.S[e_name]))
            if level > 1:
                print(self.N, 'samples in', len(self.e_names), 'ensembles:')
                for e_name in self.e_names:
                    print(e_name, ':', self.e_content[e_name])

    def zero_within_error(self):
        return np.abs(self.value) <= self.dvalue

    def plot_tauint(self, save=None):
        """Plot integrated autocorrelation time for each ensemble."""
        if not self.e_names:
            raise Exception('Run the gamma method first.')

        fig = plt.figure()
        for e, e_name in enumerate(self.e_names):
            plt.xlabel(r'$W$')
            plt.ylabel(r'$\tau_\mathrm{int}$')
            length = int(len(self.e_n_tauint[e_name]))
            if self.tau_exp[e_name] > 0:
                base = self.e_n_tauint[e_name][self.e_windowsize[e_name]]
                x_help = np.arange(2 * self.tau_exp[e_name])
                y_help = (x_help + 1) * np.abs(self.e_rho[e_name][self.e_windowsize[e_name] + 1]) * (1 - x_help / (2 * (2 * self.tau_exp[e_name] - 1))) + base
                x_arr = np.arange(self.e_windowsize[e_name] + 1, self.e_windowsize[e_name] + 1 + 2 * self.tau_exp[e_name])
                plt.plot(x_arr, y_help, 'C' + str(e), linewidth=1, ls='--', marker=',')
                plt.errorbar([self.e_windowsize[e_name] + 2 * self.tau_exp[e_name]], [self.e_tauint[e_name]],
                             yerr=[self.e_dtauint[e_name]], fmt='C' + str(e), linewidth=1, capsize=2, marker='o', mfc=plt.rcParams['axes.facecolor'])
                xmax = self.e_windowsize[e_name] + 2 * self.tau_exp[e_name] + 1.5
                label = e_name + r', $\tau_\mathrm{exp}$=' + str(np.around(self.tau_exp[e_name], decimals=2))
            else:
                label = e_name + ', S=' + str(np.around(self.S[e_name], decimals=2))
                xmax = max(10.5, 2 * self.e_windowsize[e_name] - 0.5)

            plt.errorbar(np.arange(length), self.e_n_tauint[e_name][:], yerr=self.e_n_dtauint[e_name][:], linewidth=1, capsize=2, label=label)
            plt.axvline(x=self.e_windowsize[e_name], color='C' + str(e), alpha=0.5, marker=',', ls='--')
            plt.legend()
            plt.xlim(-0.5, xmax)
            plt.ylim(bottom=0.0)
            plt.draw()
            if save:
                fig.savefig(save)

    def plot_rho(self):
        """Plot normalized autocorrelation function time for each ensemble."""
        if not self.e_names:
            raise Exception('Run the gamma method first.')
        for e, e_name in enumerate(self.e_names):
            plt.xlabel('W')
            plt.ylabel('rho')
            length = int(len(self.e_drho[e_name]))
            plt.errorbar(np.arange(length), self.e_rho[e_name][:length], yerr=self.e_drho[e_name][:], linewidth=1, capsize=2)
            plt.axvline(x=self.e_windowsize[e_name], color='r', alpha=0.25, ls='--', marker=',')
            if self.tau_exp[e_name] > 0:
                plt.plot([self.e_windowsize[e_name] + 1, self.e_windowsize[e_name] + 1 + 2 * self.tau_exp[e_name]],
                         [self.e_rho[e_name][self.e_windowsize[e_name] + 1], 0], 'k-', lw=1)
                xmax = self.e_windowsize[e_name] + 2 * self.tau_exp[e_name] + 1.5
                plt.title('Rho ' + e_name + r', tau\_exp=' + str(np.around(self.tau_exp[e_name], decimals=2)))
            else:
                xmax = max(10.5, 2 * self.e_windowsize[e_name] - 0.5)
                plt.title('Rho ' + e_name + ', S=' + str(np.around(self.S[e_name], decimals=2)))
            plt.plot([-0.5, xmax], [0, 0], 'k--', lw=1)
            plt.xlim(-0.5, xmax)
            plt.draw()

    def plot_rep_dist(self):
        """Plot replica distribution for each ensemble with more than one replicum."""
        if not self.e_names:
            raise Exception('Run the gamma method first.')
        for e, e_name in enumerate(self.e_names):
            if len(self.e_content[e_name]) == 1:
                print('No replica distribution for a single replicum (', e_name, ')')
                continue
            r_length = []
            sub_r_mean = 0
            for r, r_name in enumerate(self.e_content[e_name]):
                r_length.append(len(self.deltas[r_name]))
                sub_r_mean += self.shape[r_name] * self.r_values[r_name]
            e_N = np.sum(r_length)
            sub_r_mean /= e_N
            arr = np.zeros(len(self.e_content[e_name]))
            for r, r_name in enumerate(self.e_content[e_name]):
                arr[r] = (self.r_values[r_name] - sub_r_mean) / (self.e_dvalue[e_name] * np.sqrt(e_N / self.shape[r_name] - 1))
            plt.hist(arr, rwidth=0.8, bins=len(self.e_content[e_name]))
            plt.title('Replica distribution' + e_name + ' (mean=0, var=1), Q=' + str(np.around(self.e_Q[e_name], decimals=2)))
            plt.draw()

    def plot_history(self):
        """Plot derived Monte Carlo history for each ensemble."""
        if not self.e_names:
            raise Exception('Run the gamma method first.')

        for e, e_name in enumerate(self.e_names):
            plt.figure()
            r_length = []
            for r, r_name in enumerate(self.e_content[e_name]):
                r_length.append(len(self.deltas[r_name]))
            e_N = np.sum(r_length)
            x = np.arange(e_N)
            tmp = []
            for r, r_name in enumerate(self.e_content[e_name]):
                tmp.append(self.deltas[r_name] + self.r_values[r_name])
            y = np.concatenate(tmp, axis=0)
            plt.errorbar(x, y, fmt='.', markersize=3)
            plt.xlim(-0.5, e_N - 0.5)
            plt.title(e_name)
            plt.draw()

    def plot_piechart(self):
        """Plot piechart which shows the fractional contribution of each
        ensemble to the error and returns a dictionary containing the fractions."""
        if not self.e_names:
            raise Exception('Run the gamma method first.')
        if self.dvalue == 0.0:
            raise Exception('Error is 0.0')
        labels = self.e_names
        sizes = [i ** 2 for i in list(self.e_dvalue.values())] / self.dvalue ** 2
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, startangle=90, normalize=True)
        ax1.axis('equal')
        plt.draw()

        return dict(zip(self.e_names, sizes))

    def dump(self, name, **kwargs):
        """Dump the Obs to a pickle file 'name'.

        Keyword arguments
        -----------------
        path -- specifies a custom path for the file (default '.')
        """
        if 'path' in kwargs:
            file_name = kwargs.get('path') + '/' + name + '.p'
        else:
            file_name = name + '.p'
        with open(file_name, 'wb') as fb:
            pickle.dump(self, fb)

    def __repr__(self):
        if self.dvalue == 0.0:
            return 'Obs[' + str(self.value) + ']'
        fexp = np.floor(np.log10(self.dvalue))
        if fexp < 0.0:
            return 'Obs[{:{form}}({:2.0f})]'.format(self.value, self.dvalue * 10 ** (-fexp + 1), form='.' + str(-int(fexp) + 1) + 'f')
        elif fexp == 0.0:
            return 'Obs[{:.1f}({:1.1f})]'.format(self.value, self.dvalue)
        else:
            return 'Obs[{:.0f}({:2.0f})]'.format(self.value, self.dvalue)

    # Overload comparisons
    def __lt__(self, other):
        return self.value < other

    def __gt__(self, other):
        return self.value > other

    # Overload math operations
    def __add__(self, y):
        if isinstance(y, Obs):
            return derived_observable(lambda x, **kwargs: x[0] + x[1], [self, y], man_grad=[1, 1])
        else:
            if isinstance(y, np.ndarray):
                return np.array([self + o for o in y])
            elif y.__class__.__name__ == 'Corr':
                return NotImplemented
            else:
                return derived_observable(lambda x, **kwargs: x[0] + y, [self], man_grad=[1])

    def __radd__(self, y):
        return self + y

    def __mul__(self, y):
        if isinstance(y, Obs):
            return derived_observable(lambda x, **kwargs: x[0] * x[1], [self, y], man_grad=[y.value, self.value])
        else:
            if isinstance(y, np.ndarray):
                return np.array([self * o for o in y])
            elif y.__class__.__name__ == 'Corr':
                return NotImplemented

            else:
                return derived_observable(lambda x, **kwargs: x[0] * y, [self], man_grad=[y])

    def __rmul__(self, y):
        return self * y

    def __sub__(self, y):
        if isinstance(y, Obs):
            return derived_observable(lambda x, **kwargs: x[0] - x[1], [self, y], man_grad=[1, -1])
        else:
            if isinstance(y, np.ndarray):
                return np.array([self - o for o in y])

            elif y.__class__.__name__ == 'Corr':
                return NotImplemented

            else:
                return derived_observable(lambda x, **kwargs: x[0] - y, [self], man_grad=[1])

    def __rsub__(self, y):
        return -1 * (self - y)

    def __neg__(self):
        return -1 * self

    def __truediv__(self, y):
        if isinstance(y, Obs):
            return derived_observable(lambda x, **kwargs: x[0] / x[1], [self, y], man_grad=[1 / y.value, - self.value / y.value ** 2])
        else:
            if isinstance(y, np.ndarray):
                return np.array([self / o for o in y])

            elif y.__class__.__name__ == 'Corr':
                return NotImplemented

            else:
                return derived_observable(lambda x, **kwargs: x[0] / y, [self], man_grad=[1 / y])

    def __rtruediv__(self, y):
        if isinstance(y, Obs):
            return derived_observable(lambda x, **kwargs: x[0] / x[1], [y, self], man_grad=[1 / self.value, - y.value / self.value ** 2])
        else:
            if isinstance(y, np.ndarray):
                return np.array([o / self for o in y])
            else:
                return derived_observable(lambda x, **kwargs: y / x[0], [self], man_grad=[-y / self.value ** 2])

    def __pow__(self, y):
        if isinstance(y, Obs):
            return derived_observable(lambda x: x[0] ** x[1], [self, y])
        else:
            return derived_observable(lambda x: x[0] ** y, [self])

    def __rpow__(self, y):
        if isinstance(y, Obs):
            return derived_observable(lambda x: x[0] ** x[1], [y, self])
        else:
            return derived_observable(lambda x: y ** x[0], [self])

    def __abs__(self):
        return derived_observable(lambda x: anp.abs(x[0]), [self])

    # Overload numpy functions
    def sqrt(self):
        return derived_observable(lambda x, **kwargs: np.sqrt(x[0]), [self], man_grad=[1 / 2 / np.sqrt(self.value)])

    def log(self):
        return derived_observable(lambda x, **kwargs: np.log(x[0]), [self], man_grad=[1 / self.value])

    def exp(self):
        return derived_observable(lambda x, **kwargs: np.exp(x[0]), [self], man_grad=[np.exp(self.value)])

    def sin(self):
        return derived_observable(lambda x, **kwargs: np.sin(x[0]), [self], man_grad=[np.cos(self.value)])

    def cos(self):
        return derived_observable(lambda x, **kwargs: np.cos(x[0]), [self], man_grad=[-np.sin(self.value)])

    def tan(self):
        return derived_observable(lambda x, **kwargs: np.tan(x[0]), [self], man_grad=[1 / np.cos(self.value) ** 2])

    def arcsin(self):
        return derived_observable(lambda x: anp.arcsin(x[0]), [self])

    def arccos(self):
        return derived_observable(lambda x: anp.arccos(x[0]), [self])

    def arctan(self):
        return derived_observable(lambda x: anp.arctan(x[0]), [self])

    def sinh(self):
        return derived_observable(lambda x, **kwargs: np.sinh(x[0]), [self], man_grad=[np.cosh(self.value)])

    def cosh(self):
        return derived_observable(lambda x, **kwargs: np.cosh(x[0]), [self], man_grad=[np.sinh(self.value)])

    def tanh(self):
        return derived_observable(lambda x, **kwargs: np.tanh(x[0]), [self], man_grad=[1 / np.cosh(self.value) ** 2])

    def arcsinh(self):
        return derived_observable(lambda x: anp.arcsinh(x[0]), [self])

    def arccosh(self):
        return derived_observable(lambda x: anp.arccosh(x[0]), [self])

    def arctanh(self):
        return derived_observable(lambda x: anp.arctanh(x[0]), [self])

    def sinc(self):
        return derived_observable(lambda x: anp.sinc(x[0]), [self])


def derived_observable(func, data, **kwargs):
    """Construct a derived Obs according to func(data, **kwargs) using automatic differentiation.

    Parameters
    ----------
    func -- arbitrary function of the form func(data, **kwargs). For the
            automatic differentiation to work, all numpy functions have to have
            the autograd wrapper (use 'import autograd.numpy as anp').
    data -- list of Obs, e.g. [obs1, obs2, obs3].

    Keyword arguments
    -----------------
    num_grad -- if True, numerical derivatives are used instead of autograd
                (default False). To control the numerical differentiation the
                kwargs of numdifftools.step_generators.MaxStepGenerator
                can be used.
    man_grad -- manually supply a list or an array which contains the jacobian
                of func. Use cautiously, supplying the wrong derivative will
                not be intercepted.
    bias_correction -- if True, the bias correction specified in
                       hep-lat/0306017 eq. (19) is performed, not recommended.
                       (Only applicable for more than 1 replicum)

    Notes
    -----
    For simple mathematical operations it can be practical to use anonymous
    functions. For the ratio of two observables one can e.g. use

    new_obs = derived_observable(lambda x: x[0] / x[1], [obs1, obs2])
    """

    data = np.asarray(data)
    raveled_data = data.ravel()

    n_obs = len(raveled_data)
    new_names = sorted(set([y for x in [o.names for o in raveled_data] for y in x]))
    replicas = len(new_names)

    new_shape = {}
    for i_data in raveled_data:
        for name in new_names:
            tmp = i_data.shape.get(name)
            if tmp is not None:
                if new_shape.get(name) is None:
                    new_shape[name] = tmp
                else:
                    if new_shape[name] != tmp:
                        raise Exception('Shapes of ensemble', name, 'do not match.')

    values = np.vectorize(lambda x: x.value)(data)

    new_values = func(values, **kwargs)

    multi = 0
    if isinstance(new_values, np.ndarray):
        multi = 1

    new_r_values = {}
    for name in new_names:
        tmp_values = np.zeros(n_obs)
        for i, item in enumerate(raveled_data):
            tmp = item.r_values.get(name)
            if tmp is None:
                tmp = item.value
            tmp_values[i] = tmp
        if multi > 0:
            tmp_values = np.array(tmp_values).reshape(data.shape)
        new_r_values[name] = func(tmp_values, **kwargs)

    if 'man_grad' in kwargs:
        deriv = np.asarray(kwargs.get('man_grad'))
        if new_values.shape + data.shape != deriv.shape:
            raise Exception('Manual derivative does not have correct shape.')
    elif kwargs.get('num_grad') is True:
        if multi > 0:
            raise Exception('Multi mode currently not supported for numerical derivative')
        options = {
            'base_step': 0.1,
            'step_ratio': 2.5,
            'num_steps': None,
            'step_nom': None,
            'offset': None,
            'num_extrap': None,
            'use_exact_steps': None,
            'check_num_steps': None,
            'scale': None}
        for key in options.keys():
            kwarg = kwargs.get(key)
            if kwarg is not None:
                options[key] = kwarg
        tmp_df = nd.Gradient(func, order=4, **{k: v for k, v in options.items() if v is not None})(values, **kwargs)
        if tmp_df.size == 1:
            deriv = np.array([tmp_df.real])
        else:
            deriv = tmp_df.real
    else:
        deriv = jacobian(func)(values, **kwargs)

    final_result = np.zeros(new_values.shape, dtype=object)

    for i_val, new_val in np.ndenumerate(new_values):
        new_deltas = {}
        for j_obs, obs in np.ndenumerate(data):
            for name in obs.names:
                new_deltas[name] = new_deltas.get(name, 0) + deriv[i_val + j_obs] * obs.deltas[name]

        new_samples = []
        for name in new_names:
            new_samples.append(new_deltas[name] + new_r_values[name][i_val])

        final_result[i_val] = Obs(new_samples, new_names)

        # Bias correction
        if replicas > 1 and kwargs.get('bias_correction'):
            final_result[i_val].value = (replicas * new_val - final_result[i_val].value) / (replicas - 1)
        else:
            final_result[i_val].value = new_val

    if multi == 0:
        final_result = final_result.item()

    return final_result


def reweight(weight, obs, **kwargs):
    """Reweight a list of observables."""
    result = []
    for i in range(len(obs)):
        if sorted(weight.names) != sorted(obs[i].names):
            raise Exception('Error: Ensembles do not fit')
        for name in weight.names:
            if weight.shape[name] != obs[i].shape[name]:
                raise Exception('Error: Shapes of ensemble', name, 'do not fit')
        new_samples = []
        for name in sorted(weight.names):
            new_samples.append((weight.deltas[name] + weight.r_values[name]) * (obs[i].deltas[name] + obs[i].r_values[name]))
        tmp_obs = Obs(new_samples, sorted(weight.names))

        result.append(derived_observable(lambda x, **kwargs: x[0] / x[1], [tmp_obs, weight], **kwargs))
        result[-1].reweighted = 1

    return result


def correlate(obs_a, obs_b):
    """Correlate two observables.

    Keep in mind to only correlate primary observables which have not been reweighted
    yet. The reweighting has to be applied after correlating the observables.
    Currently only works if ensembles are identical. This is not really necessary.
    """

    if sorted(obs_a.names) != sorted(obs_b.names):
        raise Exception('Ensembles do not fit')
    for name in obs_a.names:
        if obs_a.shape[name] != obs_b.shape[name]:
            raise Exception('Shapes of ensemble', name, 'do not fit')

    if obs_a.reweighted == 1:
        warnings.warn("The first observable is already reweighted.", RuntimeWarning)
    if obs_b.reweighted == 1:
        warnings.warn("The second observable is already reweighted.", RuntimeWarning)

    new_samples = []
    for name in sorted(obs_a.names):
        new_samples.append((obs_a.deltas[name] + obs_a.r_values[name]) * (obs_b.deltas[name] + obs_b.r_values[name]))

    return Obs(new_samples, sorted(obs_a.names))


def covariance(obs1, obs2, correlation=False, **kwargs):
    """Calculates the covariance of two observables.

    covariance(obs, obs) is equal to obs.dvalue ** 2
    The gamma method has to be applied first to both observables.

    If abs(covariance(obs1, obs2)) > obs1.dvalue * obs2.dvalue, the covariance
    is constrained to the maximum value in order to make sure that covariance
    matrices are positive semidefinite.

    Keyword arguments
    -----------------
    correlation -- if true the correlation instead of the covariance is
                   returned (default False)
    """

    for name in sorted(set(obs1.names + obs2.names)):
        if (obs1.shape.get(name) != obs2.shape.get(name)) and (obs1.shape.get(name) is not None) and (obs2.shape.get(name) is not None):
            raise Exception('Shapes of ensemble', name, 'do not fit')

    if obs1.e_names == {} or obs2.e_names == {}:
        raise Exception('The gamma method has to be applied to both Obs first.')

    dvalue = 0

    for e_name in obs1.e_names:

        if e_name not in obs2.e_names:
            continue

        gamma = 0
        r_length = []
        for r_name in obs1.e_content[e_name]:
            if r_name not in obs2.e_content[e_name]:
                continue

            r_length.append(len(obs1.deltas[r_name]))

            gamma += np.sum(obs1.deltas[r_name] * obs2.deltas[r_name])

        e_N = np.sum(r_length)

        tau_combined = (obs1.e_tauint[e_name] + obs2.e_tauint[e_name]) / 2
        dvalue += gamma / e_N * (1 + 1 / e_N) / e_N * 2 * tau_combined

    if np.abs(dvalue / obs1.dvalue / obs2.dvalue) > 1.0:
        dvalue = np.sign(dvalue) * obs1.dvalue * obs2.dvalue

    if correlation:
        dvalue = dvalue / obs1.dvalue / obs2.dvalue

    return dvalue


def covariance2(obs1, obs2, correlation=False, **kwargs):
    """Alternative implementation of the covariance of two observables.

    covariance(obs, obs) is equal to obs.dvalue ** 2
    The gamma method has to be applied first to both observables.

    If abs(covariance(obs1, obs2)) > obs1.dvalue * obs2.dvalue, the covariance
    is constrained to the maximum value in order to make sure that covariance
    matrices are positive semidefinite.

    Keyword arguments
    -----------------
    correlation -- if true the correlation instead of the covariance is
                   returned (default False)
    """

    for name in sorted(set(obs1.names + obs2.names)):
        if (obs1.shape.get(name) != obs2.shape.get(name)) and (obs1.shape.get(name) is not None) and (obs2.shape.get(name) is not None):
            raise Exception('Shapes of ensemble', name, 'do not fit')

    if obs1.e_names == {} or obs2.e_names == {}:
        raise Exception('The gamma method has to be applied to both Obs first.')

    dvalue = 0
    e_gamma = {}
    e_dvalue = {}
    e_n_tauint = {}
    e_rho = {}

    for e_name in obs1.e_names:

        if e_name not in obs2.e_names:
            continue

        r_length = []
        for r_name in obs1.e_content[e_name]:
            r_length.append(len(obs1.deltas[r_name]))

        e_N = np.sum(r_length)
        w_max = max(r_length) // 2
        e_gamma[e_name] = np.zeros(w_max)

        for r_name in obs1.e_content[e_name]:
            if r_name not in obs2.e_content[e_name]:
                continue
            max_gamma = min(obs1.shape[r_name], w_max)
            # The padding for the fft has to be even
            padding = obs1.shape[r_name] + max_gamma + (obs1.shape[r_name] + max_gamma) % 2
            e_gamma[e_name][:max_gamma] += (np.fft.irfft(np.fft.rfft(obs1.deltas[r_name], padding) * np.conjugate(np.fft.rfft(obs2.deltas[r_name], padding)))[:max_gamma] + np.fft.irfft(np.fft.rfft(obs2.deltas[r_name], padding) * np.conjugate(np.fft.rfft(obs1.deltas[r_name], padding)))[:max_gamma]) / 2.0

        if np.all(e_gamma[e_name]) == 0.0:
            continue

        e_shapes = []
        for r_name in obs1.e_content[e_name]:
            e_shapes.append(obs1.shape[r_name])

        div = np.array([])
        mul = np.array([])
        sorted_shapes = sorted(e_shapes)
        for i, item in enumerate(sorted_shapes):
            if len(div) > w_max:
                break
            if i == 0:
                samples = item
            else:
                samples = item - sorted_shapes[i - 1]
            div = np.append(div, np.repeat(np.sum(sorted_shapes[i:]), samples))
            mul = np.append(mul, np.repeat(len(sorted_shapes) - i, samples))
        div = div - np.arange(len(div)) * mul

        e_gamma[e_name] /= div[:w_max]

        e_rho[e_name] = e_gamma[e_name][:w_max] / e_gamma[e_name][0]
        e_n_tauint[e_name] = np.cumsum(np.concatenate(([0.5], e_rho[e_name][1:])))
        # Make sure no entry of tauint is smaller than 0.5
        e_n_tauint[e_name][e_n_tauint[e_name] < 0.5] = 0.500000000001

        window = max(obs1.e_windowsize[e_name], obs2.e_windowsize[e_name])
        # Bias correction hep-lat/0306017 eq. (49)
        e_dvalue[e_name] = 2 * (e_n_tauint[e_name][window] + obs1.tau_exp[e_name] * np.abs(e_rho[e_name][window + 1])) * (1 + (2 * window + 1) / e_N) * e_gamma[e_name][0] / e_N

        dvalue += e_dvalue[e_name]

    if np.abs(dvalue / obs1.dvalue / obs2.dvalue) > 1.0:
        dvalue = np.sign(dvalue) * obs1.dvalue * obs2.dvalue

    if correlation:
        dvalue = dvalue / obs1.dvalue / obs2.dvalue

    return dvalue


def covariance3(obs1, obs2, correlation=False, **kwargs):
    """Another alternative implementation of the covariance of two observables.

    covariance2(obs, obs) is equal to obs.dvalue ** 2
    Currently only works if ensembles are identical.
    The gamma method has to be applied first to both observables.

    If abs(covariance2(obs1, obs2)) > obs1.dvalue * obs2.dvalue, the covariance
    is constrained to the maximum value in order to make sure that covariance
    matrices are positive semidefinite.

    Keyword arguments
    -----------------
    correlation -- if true the correlation instead of the covariance is
                   returned (default False)
    plot -- if true, the integrated autocorrelation time for each ensemble is
            plotted.
    """

    for name in sorted(set(obs1.names + obs2.names)):
        if (obs1.shape.get(name) != obs2.shape.get(name)) and (obs1.shape.get(name) is not None) and (obs2.shape.get(name) is not None):
            raise Exception('Shapes of ensemble', name, 'do not fit')

    if obs1.e_names == {} or obs2.e_names == {}:
        raise Exception('The gamma method has to be applied to both Obs first.')

    tau_exp = []
    S = []
    for e_name in sorted(set(obs1.e_names + obs2.e_names)):
        t_1 = obs1.tau_exp.get(e_name)
        t_2 = obs2.tau_exp.get(e_name)
        if t_1 is None:
            t_1 = 0
        if t_2 is None:
            t_2 = 0
        tau_exp.append(max(t_1, t_2))
        S_1 = obs1.S.get(e_name)
        S_2 = obs2.S.get(e_name)
        if S_1 is None:
            S_1 = Obs.S_global
        if S_2 is None:
            S_2 = Obs.S_global
        S.append(max(S_1, S_2))

    check_obs = obs1 + obs2
    check_obs.gamma_method(tau_exp=tau_exp, S=S)

    if kwargs.get('plot'):
        check_obs.plot_tauint()
        check_obs.plot_rho()

    cov = (check_obs.dvalue ** 2 - obs1.dvalue ** 2 - obs2.dvalue ** 2) / 2

    if np.abs(cov / obs1.dvalue / obs2.dvalue) > 1.0:
        cov = np.sign(cov) * obs1.dvalue * obs2.dvalue

    if correlation:
        cov = cov / obs1.dvalue / obs2.dvalue

    return cov


def use_time_reversal_symmetry(data1, data2, **kwargs):
    """Combine two correlation functions (lists of Obs) according to time reversal symmetry

    Keyword arguments
    -----------------
    minus -- if True, multiply the second correlation function by a minus sign.
    """
    if kwargs.get('minus'):
        sign = -1
    else:
        sign = 1

    result = []
    T = int(len(data1))
    for i in range(T):
        result.append(derived_observable(lambda x, **kwargs: (x[0] + sign * x[1]) / 2, [data1[i], data2[T - i - 1]], **kwargs))

    return result


def pseudo_Obs(value, dvalue, name, samples=1000):
    """Generate a pseudo Obs with given value, dvalue and name

    The standard number of samples is a 1000. This can be adjusted.
    """
    if dvalue <= 0.0:
        return Obs([np.zeros(samples) + value], [name])
    else:
        for _ in range(100):
            deltas = [np.random.normal(0.0, dvalue * np.sqrt(samples), samples)]
            deltas -= np.mean(deltas)
            deltas *= dvalue / np.sqrt((np.var(deltas) / samples)) / np.sqrt(1 + 3 / samples)
            deltas += value
            res = Obs(deltas, [name])
            res.gamma_method(S=2, tau_exp=0)
            if abs(res.dvalue - dvalue) < 1e-10 * dvalue:
                break

        res.value = float(value)

        return res


def dump_object(obj, name, **kwargs):
    """Dump object into pickle file.

    Keyword arguments
    -----------------
    path -- specifies a custom path for the file (default '.')
    """
    if 'path' in kwargs:
        file_name = kwargs.get('path') + '/' + name + '.p'
    else:
        file_name = name + '.p'
    with open(file_name, 'wb') as fb:
        pickle.dump(obj, fb)


def load_object(path):
    """Load object from pickle file. """
    with open(path, 'rb') as file:
        return pickle.load(file)


def merge_obs(list_of_obs):
    """Combine all observables in list_of_obs into one new observable

    It is not possible to combine obs which are based on the same replicum
    """
    replist = [item for obs in list_of_obs for item in obs.names]
    if (len(replist) == len(set(replist))) is False:
        raise Exception('list_of_obs contains duplicate replica: %s' % (str(replist)))
    new_dict = {}
    for o in list_of_obs:
        new_dict.update({key: o.deltas.get(key, 0) + o.r_values.get(key, 0)
                        for key in set(o.deltas) | set(o.r_values)})

    return Obs(list(new_dict.values()), list(new_dict.keys()))
