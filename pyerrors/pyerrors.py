#!/usr/bin/env python
# coding: utf-8

import warnings
import pickle
import numpy as np
import autograd.numpy as anp  # Thinly-wrapped numpy
from autograd import jacobian
import matplotlib.pyplot as plt
import numdifftools as nd
from itertools import groupby


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
    S_global : float
        Standard value for S (default 2.0)
    S_dict : dict
        Dictionary for S values. If an entry for a given ensemble
        exists this overwrites the standard value for that ensemble.
    tau_exp_global : float
        Standard value for tau_exp (default 0.0)
    tau_exp_dict :dict
        Dictionary for tau_exp values. If an entry for a given ensemble exists
        this overwrites the standard value for that ensemble.
    N_sigma_global : float
        Standard value for N_sigma (default 1.0)
    """
    __slots__ = ['names', 'shape', 'r_values', 'deltas', 'N', '_value', '_dvalue',
                 'ddvalue', 'reweighted', 'S', 'tau_exp', 'N_sigma',
                 'e_dvalue', 'e_ddvalue', 'e_tauint', 'e_dtauint',
                 'e_windowsize', 'e_rho', 'e_drho', 'e_n_tauint', 'e_n_dtauint',
                 'idl', 'is_merged', 'tag', '__dict__']

    S_global = 2.0
    S_dict = {}
    tau_exp_global = 0.0
    tau_exp_dict = {}
    N_sigma_global = 1.0
    filter_eps = 1e-10

    def __init__(self, samples, names, idl=None, means=None, **kwargs):
        """ Initialize Obs object.

        Attributes
        ----------
        samples : list
            list of numpy arrays containing the Monte Carlo samples
        names : list
            list of strings labeling the indivdual samples
        idl : list, optional
            list of ranges or lists on which the samples are defined
        means : list, optional
            list of mean values for the case that the mean values were
            already subtracted from the samples
        """

        if means is None:
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

        self.idl = {}
        if idl is not None:
            for name, idx in zip(names, idl):
                if isinstance(idx, range):
                    self.idl[name] = idx
                elif isinstance(idx, (list, np.ndarray)):
                    dc = np.unique(np.diff(idx))
                    if np.any(dc < 0):
                        raise Exception("Unsorted idx for idl[%s]" % (name))
                    if len(dc) == 1:
                        self.idl[name] = range(idx[0], idx[-1] + dc[0], dc[0])
                    else:
                        self.idl[name] = list(idx)
                else:
                    raise Exception('incompatible type for idl[%s].' % (name))
        else:
            for name, sample in zip(names, samples):
                self.idl[name] = range(1, len(sample) + 1)

        if means is not None:
            for name, sample, mean in zip(names, samples, means):
                self.shape[name] = len(self.idl[name])
                if len(sample) != self.shape[name]:
                    raise Exception('Incompatible samples and idx for %s: %d vs. %d' % (name, len(sample), self.shape[name]))
                self.r_values[name] = mean
                self.deltas[name] = sample
        else:
            for name, sample in zip(names, samples):
                self.shape[name] = len(self.idl[name])
                if len(sample) != self.shape[name]:
                    raise Exception('Incompatible samples and idx for %s: %d vs. %d' % (name, len(sample), self.shape[name]))
                self.r_values[name] = np.mean(sample)
                self.deltas[name] = sample - self.r_values[name]
        self.is_merged = False
        self.N = sum(list(self.shape.values()))

        self._value = 0
        if means is None:
            for name in self.names:
                self._value += self.shape[name] * self.r_values[name]
            self._value /= self.N

        self._dvalue = 0.0
        self.ddvalue = 0.0
        self.reweighted = False

        self.tag = None

    @property
    def value(self):
        return self._value

    @property
    def dvalue(self):
        return self._dvalue

    @property
    def e_names(self):
        return sorted(set([o.split('|')[0] for o in self.names]))

    @property
    def e_content(self):
        res = {}
        for e, e_name in enumerate(self.e_names):
            res[e_name] = sorted(filter(lambda x: x.startswith(e_name + '|'), self.names))
            if e_name in self.names:
                res[e_name].append(e_name)
        return res

    def expand_deltas(self, deltas, idx, shape):
        """Expand deltas defined on idx to a regular, contiguous range, where holes are filled by 0.
           If idx is of type range, the deltas are not changed

        Parameters
        ----------
        deltas  -- List of fluctuations
        idx     -- List or range of configs on which the deltas are defined.
        shape   -- Number of configs in idx.
        """
        if type(idx) is range:
            return deltas
        else:
            ret = np.zeros(idx[-1] - idx[0] + 1)
            for i in range(shape):
                ret[idx[i] - idx[0]] = deltas[i]
            return ret

    def calc_gamma(self, deltas, idx, shape, w_max, fft):
        """Calculate Gamma_{AA} from the deltas, which are defined on idx.
           idx is assumed to be a contiguous range (possibly with a stepsize != 1)

        Parameters
        ----------
        deltas  -- List of fluctuations
        idx     -- List or range of configs on which the deltas are defined.
        shape   -- Number of configs in idx.
        w_max   -- Upper bound for the summation window
        fft     -- boolean, which determines whether the fft algorithm is used for
                   the computation of the autocorrelation function
        """
        gamma = np.zeros(w_max)
        deltas = self.expand_deltas(deltas, idx, shape)
        new_shape = len(deltas)
        if fft:
            max_gamma = min(new_shape, w_max)
            # The padding for the fft has to be even
            padding = new_shape + max_gamma + (new_shape + max_gamma) % 2
            gamma[:max_gamma] += np.fft.irfft(np.abs(np.fft.rfft(deltas, padding)) ** 2)[:max_gamma]
        else:
            for n in range(w_max):
                if new_shape - n >= 0:
                    gamma[n] += deltas[0:new_shape - n].dot(deltas[n:new_shape])

        return gamma

    def gamma_method(self, **kwargs):
        """Calculate the error and related properties of the Obs.

        Keyword arguments
        -----------------
        S : float
            specifies a custom value for the parameter S (default 2.0), can be
            a float or an array of floats for different ensembles
        tau_exp : float
            positive value triggers the critical slowing down analysis
            (default 0.0), can be a float or an array of floats for different
            ensembles
        N_sigma : float
            number of standard deviations from zero until the tail is
            attached to the autocorrelation function (default 1)
        fft : bool
            determines whether the fft algorithm is used for the computation
            of the autocorrelation function (default True)
        """

        e_content = self.e_content
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
        self._dvalue = 0
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

        for e, e_name in enumerate(self.e_names):

            r_length = []
            for r_name in e_content[e_name]:
                if self.idl[r_name] is range:
                    r_length.append(len(self.idl[r_name]))
                else:
                    r_length.append((self.idl[r_name][-1] - self.idl[r_name][0] + 1))

            e_N = np.sum([self.shape[r_name] for r_name in e_content[e_name]])
            w_max = max(r_length) // 2
            e_gamma[e_name] = np.zeros(w_max)
            self.e_rho[e_name] = np.zeros(w_max)
            self.e_drho[e_name] = np.zeros(w_max)

            for r_name in e_content[e_name]:
                e_gamma[e_name] += self.calc_gamma(self.deltas[r_name], self.idl[r_name], self.shape[r_name], w_max, fft)

            gamma_div = np.zeros(w_max)
            for r_name in e_content[e_name]:
                gamma_div += self.calc_gamma(np.ones((self.shape[r_name])), self.idl[r_name], self.shape[r_name], w_max, fft)
            e_gamma[e_name] /= gamma_div[:w_max]

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
            self.e_n_tauint[e_name][self.e_n_tauint[e_name] <= 0.5] = 0.5 + np.finfo(np.float64).eps
            # hep-lat/0306017 eq. (42)
            self.e_n_dtauint[e_name] = self.e_n_tauint[e_name] * 2 * np.sqrt(np.abs(np.arange(w_max) + 0.5 - self.e_n_tauint[e_name]) / e_N)
            self.e_n_dtauint[e_name][0] = 0.0

            def _compute_drho(i):
                tmp = self.e_rho[e_name][i + 1:w_max] + np.concatenate([self.e_rho[e_name][i - 1::-1], self.e_rho[e_name][1:w_max - 2 * i]]) - 2 * self.e_rho[e_name][i] * self.e_rho[e_name][1:w_max - i]
                self.e_drho[e_name][i] = np.sqrt(np.sum(tmp ** 2) / e_N)

            _compute_drho(1)
            if self.tau_exp[e_name] > 0:
                texp = self.tau_exp[e_name]
                # if type(self.idl[e_name]) is range: # scale tau_exp according to step size
                #    texp /= self.idl[e_name].step
                # Critical slowing down analysis
                for n in range(1, w_max // 2):
                    _compute_drho(n + 1)
                    if (self.e_rho[e_name][n] - self.N_sigma * self.e_drho[e_name][n]) < 0 or n >= w_max // 2 - 2:
                        # Bias correction hep-lat/0306017 eq. (49) included
                        self.e_tauint[e_name] = self.e_n_tauint[e_name][n] * (1 + (2 * n + 1) / e_N) / (1 + 1 / e_N) + texp * np.abs(self.e_rho[e_name][n + 1])  # The absolute makes sure, that the tail contribution is always positive
                        self.e_dtauint[e_name] = np.sqrt(self.e_n_dtauint[e_name][n] ** 2 + texp ** 2 * self.e_drho[e_name][n + 1] ** 2)
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

            self._dvalue += self.e_dvalue[e_name] ** 2
            self.ddvalue += (self.e_dvalue[e_name] * self.e_ddvalue[e_name]) ** 2

        self._dvalue = np.sqrt(self.dvalue)
        if self._dvalue == 0.0:
            self.ddvalue = 0.0
        else:
            self.ddvalue = np.sqrt(self.ddvalue) / self.dvalue
        return

    def print(self, level=1):
        warnings.warn("Method 'print' renamed to 'details'", DeprecationWarning)
        self.details(level > 1)

    def details(self, ens_content=False):
        """Output detailed properties of the Obs."""
        if self.value == 0.0:
            percentage = np.nan
        else:
            percentage = np.abs(self.dvalue / self.value) * 100
        print('Result\t %3.8e +/- %3.8e +/- %3.8e (%3.3f%%)' % (self.value, self.dvalue, self.ddvalue, percentage))
        if hasattr(self, 'e_dvalue'):
            if len(self.e_names) > 1:
                print(' Ensemble errors:')
            for e_name in self.e_names:
                if len(self.e_names) > 1:
                    print('', e_name, '\t %3.8e +/- %3.8e' % (self.e_dvalue[e_name], self.e_ddvalue[e_name]))
                if self.tau_exp[e_name] > 0:
                    print('  t_int\t %3.8e +/- %3.8e tau_exp = %3.2f,  N_sigma = %1.0i' % (self.e_tauint[e_name], self.e_dtauint[e_name], self.tau_exp[e_name], self.N_sigma))
                else:
                    print('  t_int\t %3.8e +/- %3.8e S = %3.2f' % (self.e_tauint[e_name], self.e_dtauint[e_name], self.S[e_name]))
        if ens_content is True:
            print(self.N, 'samples in', len(self.e_names), 'ensembles:')
            for e_name in self.e_names:
                print(e_name, ':', self.e_content[e_name])

    def is_zero_within_error(self, sigma=1):
        """Checks whether the observable is zero within 'sigma' standard errors.

        Works only properly when the gamma method was run.
        """
        return self.is_zero() or np.abs(self.value) <= sigma * self.dvalue

    def is_zero(self):
        """Checks whether the observable is zero within machine precision."""
        return np.isclose(0.0, self.value) and all(np.allclose(0.0, delta) for delta in self.deltas.values())

    def plot_tauint(self, save=None):
        """Plot integrated autocorrelation time for each ensemble."""
        if not hasattr(self, 'e_names'):
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
        if not hasattr(self, 'e_names'):
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
        if not hasattr(self, 'e_names'):
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
            plt.title('Replica distribution' + e_name + ' (mean=0, var=1)')
            plt.draw()

    def plot_history(self, expand=True):
        """Plot derived Monte Carlo history for each ensemble."""
        if not hasattr(self, 'e_names'):
            raise Exception('Run the gamma method first.')

        for e, e_name in enumerate(self.e_names):
            plt.figure()
            r_length = []
            tmp = []
            for r, r_name in enumerate(self.e_content[e_name]):
                if expand:
                    tmp.append(self.expand_deltas(self.deltas[r_name], self.idl[r_name], self.shape[r_name]) + self.r_values[r_name])
                else:
                    tmp.append(self.deltas[r_name] + self.r_values[r_name])
                r_length.append(len(tmp[-1]))
            e_N = np.sum(r_length)
            x = np.arange(e_N)
            y = np.concatenate(tmp, axis=0)
            plt.errorbar(x, y, fmt='.', markersize=3)
            plt.xlim(-0.5, e_N - 0.5)
            plt.title(e_name)
            plt.draw()

    def plot_piechart(self):
        """Plot piechart which shows the fractional contribution of each
        ensemble to the error and returns a dictionary containing the fractions."""
        if not hasattr(self, 'e_names'):
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

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return 'Obs[' + str(self) + ']'

    def __str__(self):
        if self.dvalue == 0.0:
            return str(self.value)
        fexp = np.floor(np.log10(self.dvalue))
        if fexp < 0.0:
            return '{:{form}}({:2.0f})'.format(self.value, self.dvalue * 10 ** (-fexp + 1), form='.' + str(-int(fexp) + 1) + 'f')
        elif fexp == 0.0:
            return '{:.1f}({:1.1f})'.format(self.value, self.dvalue)
        else:
            return '{:.0f}({:2.0f})'.format(self.value, self.dvalue)

    # Overload comparisons
    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other

    def __eq__(self, other):
        return (self - other).is_zero()

    def __ne__(self, other):
        return not (self - other).is_zero()

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
            elif isinstance(y, complex):
                return CObs(self * y.real, self * y.imag)
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


class CObs:
    """Class for a complex valued observable."""
    __slots__ = ['_real', '_imag', 'tag']

    def __init__(self, real, imag=0.0):
        self._real = real
        self._imag = imag
        self.tag = None

    @property
    def real(self):
        return self._real

    @property
    def imag(self):
        return self._imag

    def gamma_method(self, **kwargs):
        """Executes the gamma_method for the real and the imaginary part."""
        if isinstance(self.real, Obs):
            self.real.gamma_method(**kwargs)
        if isinstance(self.imag, Obs):
            self.imag.gamma_method(**kwargs)

    def is_zero(self):
        """Checks whether both real and imaginary part are zero within machine precision."""
        return self.real == 0.0 and self.imag == 0.0

    def conjugate(self):
        return CObs(self.real, -self.imag)

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            return other + self
        elif hasattr(other, 'real') and hasattr(other, 'imag'):
            return CObs(self.real + other.real,
                        self.imag + other.imag)
        else:
            return CObs(self.real + other, self.imag)

    def __radd__(self, y):
        return self + y

    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            return -1 * (other - self)
        elif hasattr(other, 'real') and hasattr(other, 'imag'):
            return CObs(self.real - other.real, self.imag - other.imag)
        else:
            return CObs(self.real - other, self.imag)

    def __rsub__(self, other):
        return -1 * (self - other)

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return other * self
        elif hasattr(other, 'real') and hasattr(other, 'imag'):
            if all(isinstance(i, Obs) for i in [self.real, self.imag, other.real, other.imag]):
                return CObs(derived_observable(lambda x, **kwargs: x[0] * x[1] - x[2] * x[3],
                                               [self.real, other.real, self.imag, other.imag],
                                               man_grad=[other.real.value, self.real.value, -other.imag.value, -self.imag.value]),
                            derived_observable(lambda x, **kwargs: x[2] * x[1] + x[0] * x[3],
                                               [self.real, other.real, self.imag, other.imag],
                                               man_grad=[other.imag.value, self.imag.value, other.real.value, self.real.value]))
            elif getattr(other, 'imag', 0) != 0:
                return CObs(self.real * other.real - self.imag * other.imag,
                            self.imag * other.real + self.real * other.imag)
            else:
                return CObs(self.real * other.real, self.imag * other.real)
        else:
            return CObs(self.real * other, self.imag * other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, np.ndarray):
            return 1 / (other / self)
        elif hasattr(other, 'real') and hasattr(other, 'imag'):
            r = other.real ** 2 + other.imag ** 2
            return CObs((self.real * other.real + self.imag * other.imag) / r, (self.imag * other.real - self.real * other.imag) / r)
        else:
            return CObs(self.real / other, self.imag / other)

    def __rtruediv__(self, other):
        r = self.real ** 2 + self.imag ** 2
        if hasattr(other, 'real') and hasattr(other, 'imag'):
            return CObs((self.real * other.real + self.imag * other.imag) / r, (self.real * other.imag - self.imag * other.real) / r)
        else:
            return CObs(self.real * other / r, -self.imag * other / r)

    def __abs__(self):
        return np.sqrt(self.real**2 + self.imag**2)

    def __neg__(other):
        return -1 * other

    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag

    def __str__(self):
        return '(' + str(self.real) + int(self.imag >= 0.0) * '+' + str(self.imag) + 'j)'

    def __repr__(self):
        return 'CObs[' + str(self) + ']'


def merge_idx(idl):
    """Returns the union of all lists in idl

    Parameters
    ----------
    idl  -- List of lists or ranges.
    """

    # Use groupby to efficiently check whether all elements of idl are identical
    try:
        g = groupby(idl)
        if next(g, True) and not next(g, False):
            return idl[0]
    except:
        pass

    if np.all([type(idx) is range for idx in idl]):
        if len(set([idx[0] for idx in idl])) == 1:
            idstart = min([idx.start for idx in idl])
            idstop = max([idx.stop for idx in idl])
            idstep = min([idx.step for idx in idl])
            return range(idstart, idstop, idstep)

    return list(set().union(*idl))


def expand_deltas_for_merge(deltas, idx, shape, new_idx):
    """Expand deltas defined on idx to the list of configs that is defined by new_idx.
       New, empy entries are filled by 0. If idx and new_idx are of type range, the smallest
       common divisor of the step sizes is used as new step size.

    Parameters
    ----------
    deltas : list
        List of fluctuations
    idx : list
        List or range of configs on which the deltas are defined.
        Has to be a subset of new_idx.
    shape : list
        Number of configs in idx.
    new_idx : list
        List of configs that defines the new range.
    """

    if type(idx) is range and type(new_idx) is range:
        if idx == new_idx:
            return deltas
    ret = np.zeros(new_idx[-1] - new_idx[0] + 1)
    for i in range(shape):
        ret[idx[i] - new_idx[0]] = deltas[i]
    return np.array([ret[new_idx[i] - new_idx[0]] for i in range(len(new_idx))])


def filter_zeroes(names, deltas, idl, eps=Obs.filter_eps):
    """Filter out all configurations with vanishing fluctuation such that they do not
       contribute to the error estimate anymore. Returns the new names, deltas and
       idl according to the filtering.
       A fluctuation is considered to be vanishing, if it is smaller than eps times
       the mean of the absolute values of all deltas in one list.

    Parameters
    ----------
    names  -- List of names
    deltas -- Dict lists of fluctuations
    idx    -- Dict of lists or ranges of configs on which the deltas are defined.
               Has to be a subset of new_idx.

    Optional parameters
    ----------
    eps    -- Prefactor that enters the filter criterion.
    """
    new_names = []
    new_deltas = {}
    new_idl = {}
    for name in names:
        nd = []
        ni = []
        maxd = np.mean(np.fabs(deltas[name]))
        for i in range(len(deltas[name])):
            if not np.isclose(0.0, deltas[name][i], atol=eps * maxd):
                nd.append(deltas[name][i])
                ni.append(idl[name][i])
        if nd:
            new_names.append(name)
            new_deltas[name] = np.array(nd)
            new_idl[name] = ni
    return (new_names, new_deltas, new_idl)


def derived_observable(func, data, **kwargs):
    """Construct a derived Obs according to func(data, **kwargs) using automatic differentiation.

    Parameters
    ----------
    func : object
        arbitrary function of the form func(data, **kwargs). For the
        automatic differentiation to work, all numpy functions have to have
        the autograd wrapper (use 'import autograd.numpy as anp').
    data : list
        list of Obs, e.g. [obs1, obs2, obs3].

    Keyword arguments
    -----------------
    num_grad : bool
        if True, numerical derivatives are used instead of autograd
        (default False). To control the numerical differentiation the
        kwargs of numdifftools.step_generators.MaxStepGenerator
        can be used.
    man_grad : list
        manually supply a list or an array which contains the jacobian
        of func. Use cautiously, supplying the wrong derivative will
        not be intercepted.

    Notes
    -----
    For simple mathematical operations it can be practical to use anonymous
    functions. For the ratio of two observables one can e.g. use

    new_obs = derived_observable(lambda x: x[0] / x[1], [obs1, obs2])
    """

    data = np.asarray(data)
    raveled_data = data.ravel()

    # Workaround for matrix operations containing non Obs data
    for i_data in raveled_data:
        if isinstance(i_data, Obs):
            first_name = i_data.names[0]
            first_shape = i_data.shape[first_name]
            break

    for i in range(len(raveled_data)):
        if isinstance(raveled_data[i], (int, float)):
            raveled_data[i] = Obs([raveled_data[i] + np.zeros(first_shape)], [first_name])

    n_obs = len(raveled_data)
    new_names = sorted(set([y for x in [o.names for o in raveled_data] for y in x]))

    is_merged = len(list(filter(lambda o: o.is_merged is True, raveled_data))) > 0
    reweighted = len(list(filter(lambda o: o.reweighted is True, raveled_data))) > 0
    new_idl_d = {}
    for name in new_names:
        idl = []
        for i_data in raveled_data:
            tmp = i_data.idl.get(name)
            if tmp is not None:
                idl.append(tmp)
        new_idl_d[name] = merge_idx(idl)
        if not is_merged:
            is_merged = (1 != len(set([len(idx) for idx in [*idl, new_idl_d[name]]])))

    if data.ndim == 1:
        values = np.array([o.value for o in data])
    else:
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
                new_deltas[name] = new_deltas.get(name, 0) + deriv[i_val + j_obs] * expand_deltas_for_merge(obs.deltas[name], obs.idl[name], obs.shape[name], new_idl_d[name])

        new_samples = []
        new_means = []
        new_idl = []
        if is_merged:
            filtered_names, filtered_deltas, filtered_idl_d = filter_zeroes(new_names, new_deltas, new_idl_d)
        else:
            filtered_names = new_names
            filtered_deltas = new_deltas
            filtered_idl_d = new_idl_d
        for name in filtered_names:
            new_samples.append(filtered_deltas[name])
            new_means.append(new_r_values[name][i_val])
            new_idl.append(filtered_idl_d[name])
        final_result[i_val] = Obs(new_samples, filtered_names, means=new_means, idl=new_idl)
        final_result[i_val]._value = new_val
        final_result[i_val].is_merged = is_merged
        final_result[i_val].reweighted = reweighted

    if multi == 0:
        final_result = final_result.item()

    return final_result


def reduce_deltas(deltas, idx_old, idx_new):
    """Extract deltas defined on idx_old on all configs of idx_new.

    Parameters
    ----------
    deltas  -- List of fluctuations
    idx_old -- List or range of configs on which the deltas are defined
    idx_new -- List of configs for which we want to extract the deltas.
               Has to be a subset of idx_old.
    """
    if not len(deltas) == len(idx_old):
        raise Exception('Lenght of deltas and idx_old have to be the same: %d != %d' % (len(deltas), len(idx_old)))
    if type(idx_old) is range and type(idx_new) is range:
        if idx_old == idx_new:
            return deltas
    shape = len(idx_new)
    ret = np.zeros(shape)
    oldpos = 0
    for i in range(shape):
        if oldpos == idx_old[i]:
            raise Exception('idx_old and idx_new do not match!')
        pos = -1
        for j in range(oldpos, len(idx_old)):
            if idx_old[j] == idx_new[i]:
                pos = j
                break
        if pos < 0:
            raise Exception('Error in reduce_deltas: Config %d not in idx_old' % (idx_new[i]))
        ret[i] = deltas[j]
    return np.array(ret)


def reweight(weight, obs, **kwargs):
    """Reweight a list of observables.

    Parameters
    ----------
    weight : Obs
        Reweighting factor. An Observable that has to be defined on a superset of the
        configurations in obs[i].idl for all i.
    obs : list
        list of Obs, e.g. [obs1, obs2, obs3].

    Keyword arguments
    -----------------
    all_configs : bool
        if True, the reweighted observables are normalized by the average of
        the reweighting factor on all configurations in weight.idl and not
        on the configurations in obs[i].idl.
    """
    result = []
    for i in range(len(obs)):
        if sorted(weight.names) != sorted(obs[i].names):
            raise Exception('Error: Ensembles do not fit')
        for name in weight.names:
            if not set(obs[i].idl[name]).issubset(weight.idl[name]):
                raise Exception('obs[%d] has to be defined on a subset of the configs in weight.idl[%s]!' % (i, name))
        new_samples = []
        w_deltas = {}
        for name in sorted(weight.names):
            w_deltas[name] = reduce_deltas(weight.deltas[name], weight.idl[name], obs[i].idl[name])
            new_samples.append((w_deltas[name] + weight.r_values[name]) * (obs[i].deltas[name] + obs[i].r_values[name]))
        tmp_obs = Obs(new_samples, sorted(weight.names), idl=[obs[i].idl[name] for name in sorted(weight.names)])

        if kwargs.get('all_configs'):
            new_weight = weight
        else:
            new_weight = Obs([w_deltas[name] + weight.r_values[name] for name in sorted(weight.names)], sorted(weight.names), idl=[obs[i].idl[name] for name in sorted(weight.names)])

        result.append(derived_observable(lambda x, **kwargs: x[0] / x[1], [tmp_obs, new_weight], **kwargs))
        result[-1].reweighted = True
        result[-1].is_merged = obs[i].is_merged

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

    if obs_a.reweighted is True:
        warnings.warn("The first observable is already reweighted.", RuntimeWarning)
    if obs_b.reweighted is True:
        warnings.warn("The second observable is already reweighted.", RuntimeWarning)

    new_samples = []
    for name in sorted(obs_a.names):
        new_samples.append((obs_a.deltas[name] + obs_a.r_values[name]) * (obs_b.deltas[name] + obs_b.r_values[name]))

    o = Obs(new_samples, sorted(obs_a.names))
    o.is_merged = obs_a.is_merged or obs_b.is_merged
    o.reweighted = obs_a.reweighted or obs_b.reweighted
    return o


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
        if (1 != len(set([len(idx) for idx in [obs1.idl[name], obs2.idl[name], merge_idx([obs1.idl[name], obs2.idl[name]])]]))):
            raise Exception('Shapes of ensemble', name, 'do not fit')

    if not hasattr(obs1, 'e_names') or not hasattr(obs2, 'e_names'):
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

    def expand_deltas(deltas, idx, shape, new_idx):
        """Expand deltas defined on idx to a contiguous range [new_idx[0], new_idx[-1]].
           New, empy entries are filled by 0. If idx and new_idx are of type range, the smallest
           common divisor of the step sizes is used as new step size.

        Parameters
        ----------
        deltas  -- List of fluctuations
        idx     -- List or range of configs on which the deltas are defined.
                   Has to be a subset of new_idx.
        shape   -- Number of configs in idx.
        new_idx -- List of configs that defines the new range.
        """

        if type(idx) is range and type(new_idx) is range:
            if idx == new_idx:
                return deltas
        ret = np.zeros(new_idx[-1] - new_idx[0] + 1)
        for i in range(shape):
            ret[idx[i] - new_idx[0]] = deltas[i]
        return ret

    def calc_gamma(deltas1, deltas2, idx1, idx2, new_idx, w_max):
        gamma = np.zeros(w_max)
        deltas1 = expand_deltas(deltas1, idx1, len(idx1), new_idx)
        deltas2 = expand_deltas(deltas2, idx2, len(idx2), new_idx)
        new_shape = len(deltas1)
        max_gamma = min(new_shape, w_max)
        # The padding for the fft has to be even
        padding = new_shape + max_gamma + (new_shape + max_gamma) % 2
        gamma[:max_gamma] += (np.fft.irfft(np.fft.rfft(deltas1, padding) * np.conjugate(np.fft.rfft(deltas2, padding)))[:max_gamma] + np.fft.irfft(np.fft.rfft(deltas2, padding) * np.conjugate(np.fft.rfft(deltas1, padding)))[:max_gamma]) / 2.0

        return gamma

    if not hasattr(obs1, 'e_names') or not hasattr(obs2, 'e_names'):
        raise Exception('The gamma method has to be applied to both Obs first.')

    dvalue = 0
    e_gamma = {}
    e_dvalue = {}
    e_n_tauint = {}
    e_rho = {}

    for e_name in obs1.e_names:

        if e_name not in obs2.e_names:
            continue

        idl_d = {}
        r_length = []
        for r_name in obs1.e_content[e_name]:
            if r_name not in obs2.e_content[e_name]:
                continue
            idl_d[r_name] = merge_idx([obs1.idl[r_name], obs2.idl[r_name]])
            if idl_d[r_name] is range:
                r_length.append(len(idl_d[r_name]))
            else:
                r_length.append((idl_d[r_name][-1] - idl_d[r_name][0] + 1))

        if not r_length:
            return 0.

        w_max = max(r_length) // 2
        e_gamma[e_name] = np.zeros(w_max)

        for r_name in obs1.e_content[e_name]:
            if r_name not in obs2.e_content[e_name]:
                continue
            e_gamma[e_name] += calc_gamma(obs1.deltas[r_name], obs2.deltas[r_name], obs1.idl[r_name], obs2.idl[r_name], idl_d[r_name], w_max)

        if np.all(e_gamma[e_name] == 0.0):
            continue

        e_shapes = []
        for r_name in obs1.e_content[e_name]:
            e_shapes.append(obs1.shape[r_name])
        gamma_div = np.zeros(w_max)
        e_N = 0
        for r_name in obs1.e_content[e_name]:
            if r_name not in obs2.e_content[e_name]:
                continue
            gamma_div += calc_gamma(np.ones(obs1.shape[r_name]), np.ones(obs2.shape[r_name]), obs1.idl[r_name], obs2.idl[r_name], idl_d[r_name], w_max)
            e_N += np.sum(np.ones_like(idl_d[r_name]))
        e_gamma[e_name] /= gamma_div[:w_max]

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
        if (1 != len(set([len(idx) for idx in [obs1.idl[name], obs2.idl[name], merge_idx([obs1.idl[name], obs2.idl[name]])]]))):
            raise Exception('Shapes of ensemble', name, 'do not fit')

    if not hasattr(obs1, 'e_names') or not hasattr(obs2, 'e_names'):
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

        res._value = float(value)

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
    idl_dict = {}
    for o in list_of_obs:
        new_dict.update({key: o.deltas.get(key, 0) + o.r_values.get(key, 0)
                        for key in set(o.deltas) | set(o.r_values)})
        idl_dict.update({key: o.idl.get(key, 0) for key in set(o.deltas)})

    names = sorted(new_dict.keys())
    o = Obs([new_dict[name] for name in names], names, idl=[idl_dict[name] for name in names])
    o.is_merged = np.any([oi.is_merged for oi in list_of_obs])
    o.reweighted = np.max([oi.reweighted for oi in list_of_obs])
    return o
