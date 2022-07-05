import warnings
import hashlib
import pickle
from math import gcd
from functools import reduce
import numpy as np
import autograd.numpy as anp  # Thinly-wrapped numpy
from autograd import jacobian
import matplotlib.pyplot as plt
from scipy.stats import skew, skewtest, kurtosis, kurtosistest
import numdifftools as nd
from itertools import groupby
from .covobs import Covobs


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
    tau_exp_dict : dict
        Dictionary for tau_exp values. If an entry for a given ensemble exists
        this overwrites the standard value for that ensemble.
    N_sigma_global : float
        Standard value for N_sigma (default 1.0)
    N_sigma_dict : dict
        Dictionary for N_sigma values. If an entry for a given ensemble exists
        this overwrites the standard value for that ensemble.
    """
    __slots__ = ['names', 'shape', 'r_values', 'deltas', 'N', '_value', '_dvalue',
                 'ddvalue', 'reweighted', 'S', 'tau_exp', 'N_sigma',
                 'e_dvalue', 'e_ddvalue', 'e_tauint', 'e_dtauint',
                 'e_windowsize', 'e_rho', 'e_drho', 'e_n_tauint', 'e_n_dtauint',
                 'idl', 'is_merged', 'tag', '_covobs', '__dict__']

    S_global = 2.0
    S_dict = {}
    tau_exp_global = 0.0
    tau_exp_dict = {}
    N_sigma_global = 1.0
    N_sigma_dict = {}
    filter_eps = 1e-10

    def __init__(self, samples, names, idl=None, **kwargs):
        """ Initialize Obs object.

        Parameters
        ----------
        samples : list
            list of numpy arrays containing the Monte Carlo samples
        names : list
            list of strings labeling the individual samples
        idl : list, optional
            list of ranges or lists on which the samples are defined
        """

        if kwargs.get("means") is None and len(samples):
            if len(samples) != len(names):
                raise Exception('Length of samples and names incompatible.')
            if idl is not None:
                if len(idl) != len(names):
                    raise Exception('Length of idl incompatible with samples and names.')
            name_length = len(names)
            if name_length > 1:
                if name_length != len(set(names)):
                    raise Exception('names are not unique.')
                if not all(isinstance(x, str) for x in names):
                    raise TypeError('All names have to be strings.')
            else:
                if not isinstance(names[0], str):
                    raise TypeError('All names have to be strings.')
            if min(len(x) for x in samples) <= 4:
                raise Exception('Samples have to have at least 5 entries.')

        self.names = sorted(names)
        self.shape = {}
        self.r_values = {}
        self.deltas = {}
        self._covobs = {}

        self._value = 0
        self.N = 0
        self.is_merged = {}
        self.idl = {}
        if idl is not None:
            for name, idx in sorted(zip(names, idl)):
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
            for name, sample in sorted(zip(names, samples)):
                self.idl[name] = range(1, len(sample) + 1)

        if kwargs.get("means") is not None:
            for name, sample, mean in sorted(zip(names, samples, kwargs.get("means"))):
                self.shape[name] = len(self.idl[name])
                self.N += self.shape[name]
                self.r_values[name] = mean
                self.deltas[name] = sample
        else:
            for name, sample in sorted(zip(names, samples)):
                self.shape[name] = len(self.idl[name])
                self.N += self.shape[name]
                if len(sample) != self.shape[name]:
                    raise Exception('Incompatible samples and idx for %s: %d vs. %d' % (name, len(sample), self.shape[name]))
                self.r_values[name] = np.mean(sample)
                self.deltas[name] = sample - self.r_values[name]
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
    def cov_names(self):
        return sorted(set([o for o in self.covobs.keys()]))

    @property
    def mc_names(self):
        return sorted(set([o.split('|')[0] for o in self.names if o not in self.cov_names]))

    @property
    def e_content(self):
        res = {}
        for e, e_name in enumerate(self.e_names):
            res[e_name] = sorted(filter(lambda x: x.startswith(e_name + '|'), self.names))
            if e_name in self.names:
                res[e_name].append(e_name)
        return res

    @property
    def covobs(self):
        return self._covobs

    def gamma_method(self, **kwargs):
        """Estimate the error and related properties of the Obs.

        Parameters
        ----------
        S : float
            specifies a custom value for the parameter S (default 2.0).
            If set to 0 it is assumed that the data exhibits no
            autocorrelation. In this case the error estimates coincides
            with the sample standard error.
        tau_exp : float
            positive value triggers the critical slowing down analysis
            (default 0.0).
        N_sigma : float
            number of standard deviations from zero until the tail is
            attached to the autocorrelation function (default 1).
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
        self.N_sigma = {}

        if kwargs.get('fft') is False:
            fft = False
        else:
            fft = True

        def _parse_kwarg(kwarg_name):
            if kwarg_name in kwargs:
                tmp = kwargs.get(kwarg_name)
                if isinstance(tmp, (int, float)):
                    if tmp < 0:
                        raise Exception(kwarg_name + ' has to be larger or equal to 0.')
                    for e, e_name in enumerate(self.e_names):
                        getattr(self, kwarg_name)[e_name] = tmp
                else:
                    raise TypeError(kwarg_name + ' is not in proper format.')
            else:
                for e, e_name in enumerate(self.e_names):
                    if e_name in getattr(Obs, kwarg_name + '_dict'):
                        getattr(self, kwarg_name)[e_name] = getattr(Obs, kwarg_name + '_dict')[e_name]
                    else:
                        getattr(self, kwarg_name)[e_name] = getattr(Obs, kwarg_name + '_global')

        _parse_kwarg('S')
        _parse_kwarg('tau_exp')
        _parse_kwarg('N_sigma')

        for e, e_name in enumerate(self.mc_names):
            r_length = []
            for r_name in e_content[e_name]:
                if isinstance(self.idl[r_name], range):
                    r_length.append(len(self.idl[r_name]))
                else:
                    r_length.append((self.idl[r_name][-1] - self.idl[r_name][0] + 1))

            e_N = np.sum([self.shape[r_name] for r_name in e_content[e_name]])
            w_max = max(r_length) // 2
            e_gamma[e_name] = np.zeros(w_max)
            self.e_rho[e_name] = np.zeros(w_max)
            self.e_drho[e_name] = np.zeros(w_max)

            for r_name in e_content[e_name]:
                e_gamma[e_name] += self._calc_gamma(self.deltas[r_name], self.idl[r_name], self.shape[r_name], w_max, fft)

            gamma_div = np.zeros(w_max)
            for r_name in e_content[e_name]:
                gamma_div += self._calc_gamma(np.ones((self.shape[r_name])), self.idl[r_name], self.shape[r_name], w_max, fft)
            gamma_div[gamma_div < 1] = 1.0
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
                # Critical slowing down analysis
                if w_max // 2 <= 1:
                    raise Exception("Need at least 8 samples for tau_exp error analysis")
                for n in range(1, w_max // 2):
                    _compute_drho(n + 1)
                    if (self.e_rho[e_name][n] - self.N_sigma[e_name] * self.e_drho[e_name][n]) < 0 or n >= w_max // 2 - 2:
                        # Bias correction hep-lat/0306017 eq. (49) included
                        self.e_tauint[e_name] = self.e_n_tauint[e_name][n] * (1 + (2 * n + 1) / e_N) / (1 + 1 / e_N) + texp * np.abs(self.e_rho[e_name][n + 1])  # The absolute makes sure, that the tail contribution is always positive
                        self.e_dtauint[e_name] = np.sqrt(self.e_n_dtauint[e_name][n] ** 2 + texp ** 2 * self.e_drho[e_name][n + 1] ** 2)
                        # Error of tau_exp neglected so far, missing term: self.e_rho[e_name][n + 1] ** 2 * d_tau_exp ** 2
                        self.e_dvalue[e_name] = np.sqrt(2 * self.e_tauint[e_name] * e_gamma[e_name][0] * (1 + 1 / e_N) / e_N)
                        self.e_ddvalue[e_name] = self.e_dvalue[e_name] * np.sqrt((n + 0.5) / e_N)
                        self.e_windowsize[e_name] = n
                        break
            else:
                if self.S[e_name] == 0.0:
                    self.e_tauint[e_name] = 0.5
                    self.e_dtauint[e_name] = 0.0
                    self.e_dvalue[e_name] = np.sqrt(e_gamma[e_name][0] / (e_N - 1))
                    self.e_ddvalue[e_name] = self.e_dvalue[e_name] * np.sqrt(0.5 / e_N)
                    self.e_windowsize[e_name] = 0
                else:
                    # Standard automatic windowing procedure
                    tau = self.S[e_name] / np.log((2 * self.e_n_tauint[e_name][1:] + 1) / (2 * self.e_n_tauint[e_name][1:] - 1))
                    g_w = np.exp(- np.arange(1, w_max) / tau) - tau / np.sqrt(np.arange(1, w_max) * e_N)
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

        for e_name in self.cov_names:
            self.e_dvalue[e_name] = np.sqrt(self.covobs[e_name].errsq())
            self.e_ddvalue[e_name] = 0
            self._dvalue += self.e_dvalue[e_name]**2

        self._dvalue = np.sqrt(self._dvalue)
        if self._dvalue == 0.0:
            self.ddvalue = 0.0
        else:
            self.ddvalue = np.sqrt(self.ddvalue) / self._dvalue
        return

    def _calc_gamma(self, deltas, idx, shape, w_max, fft):
        """Calculate Gamma_{AA} from the deltas, which are defined on idx.
           idx is assumed to be a contiguous range (possibly with a stepsize != 1)

        Parameters
        ----------
        deltas : list
            List of fluctuations
        idx : list
            List or range of configurations on which the deltas are defined.
        shape : int
            Number of configurations in idx.
        w_max : int
            Upper bound for the summation window.
        fft : bool
            determines whether the fft algorithm is used for the computation
            of the autocorrelation function.
        """
        gamma = np.zeros(w_max)
        deltas = _expand_deltas(deltas, idx, shape)
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

    def details(self, ens_content=True):
        """Output detailed properties of the Obs.

        Parameters
        ----------
        ens_content : bool
            print details about the ensembles and replica if true.
        """
        if self.tag is not None:
            print("Description:", self.tag)
        if not hasattr(self, 'e_dvalue'):
            print('Result\t %3.8e' % (self.value))
        else:
            if self.value == 0.0:
                percentage = np.nan
            else:
                percentage = np.abs(self._dvalue / self.value) * 100
            print('Result\t %3.8e +/- %3.8e +/- %3.8e (%3.3f%%)' % (self.value, self._dvalue, self.ddvalue, percentage))
            if len(self.e_names) > 1:
                print(' Ensemble errors:')
            for e_name in self.mc_names:
                if len(self.e_names) > 1:
                    print('', e_name, '\t %3.8e +/- %3.8e' % (self.e_dvalue[e_name], self.e_ddvalue[e_name]))
                if self.tau_exp[e_name] > 0:
                    print(' t_int\t %3.8e +/- %3.8e tau_exp = %3.2f,  N_sigma = %1.0i' % (self.e_tauint[e_name], self.e_dtauint[e_name], self.tau_exp[e_name], self.N_sigma[e_name]))
                else:
                    print(' t_int\t %3.8e +/- %3.8e S = %3.2f' % (self.e_tauint[e_name], self.e_dtauint[e_name], self.S[e_name]))
            for e_name in self.cov_names:
                print('', e_name, '\t %3.8e' % (self.e_dvalue[e_name]))
        if ens_content is True:
            if len(self.e_names) == 1:
                print(self.N, 'samples in', len(self.e_names), 'ensemble:')
            else:
                print(self.N, 'samples in', len(self.e_names), 'ensembles:')
            my_string_list = []
            for key, value in sorted(self.e_content.items()):
                if key not in self.covobs:
                    my_string = '  ' + "\u00B7 Ensemble '" + key + "' "
                    if len(value) == 1:
                        my_string += f': {self.shape[value[0]]} configurations'
                        if isinstance(self.idl[value[0]], range):
                            my_string += f' (from {self.idl[value[0]].start} to {self.idl[value[0]][-1]}' + int(self.idl[value[0]].step != 1) * f' in steps of {self.idl[value[0]].step}' + ')'
                        else:
                            my_string += ' (irregular range)'
                    else:
                        sublist = []
                        for v in value:
                            my_substring = '    ' + "\u00B7 Replicum '" + v[len(key) + 1:] + "' "
                            my_substring += f': {self.shape[v]} configurations'
                            if isinstance(self.idl[v], range):
                                my_substring += f' (from {self.idl[v].start} to {self.idl[v][-1]}' + int(self.idl[v].step != 1) * f' in steps of {self.idl[v].step}' + ')'
                            else:
                                my_substring += ' (irregular range)'
                            sublist.append(my_substring)

                        my_string += '\n' + '\n'.join(sublist)
                else:
                    my_string = '  ' + "\u00B7 Covobs   '" + key + "' "
                my_string_list.append(my_string)
            print('\n'.join(my_string_list))

    def reweight(self, weight):
        """Reweight the obs with given rewighting factors.

        Parameters
        ----------
        weight : Obs
            Reweighting factor. An Observable that has to be defined on a superset of the
            configurations in obs[i].idl for all i.
        all_configs : bool
            if True, the reweighted observables are normalized by the average of
            the reweighting factor on all configurations in weight.idl and not
            on the configurations in obs[i].idl. Default False.
        """
        return reweight(weight, [self])[0]

    def is_zero_within_error(self, sigma=1):
        """Checks whether the observable is zero within 'sigma' standard errors.

        Parameters
        ----------
        sigma : int
            Number of standard errors used for the check.

        Works only properly when the gamma method was run.
        """
        return self.is_zero() or np.abs(self.value) <= sigma * self._dvalue

    def is_zero(self, atol=1e-10):
        """Checks whether the observable is zero within a given tolerance.

        Parameters
        ----------
        atol : float
            Absolute tolerance (for details see numpy documentation).
        """
        return np.isclose(0.0, self.value, 1e-14, atol) and all(np.allclose(0.0, delta, 1e-14, atol) for delta in self.deltas.values()) and all(np.allclose(0.0, delta.errsq(), 1e-14, atol) for delta in self.covobs.values())

    def plot_tauint(self, save=None):
        """Plot integrated autocorrelation time for each ensemble.

        Parameters
        ----------
        save : str
            saves the figure to a file named 'save' if.
        """
        if not hasattr(self, 'e_dvalue'):
            raise Exception('Run the gamma method first.')

        for e, e_name in enumerate(self.mc_names):
            fig = plt.figure()
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

            plt.errorbar(np.arange(length)[:int(xmax) + 1], self.e_n_tauint[e_name][:int(xmax) + 1], yerr=self.e_n_dtauint[e_name][:int(xmax) + 1], linewidth=1, capsize=2, label=label)
            plt.axvline(x=self.e_windowsize[e_name], color='C' + str(e), alpha=0.5, marker=',', ls='--')
            plt.legend()
            plt.xlim(-0.5, xmax)
            ylim = plt.ylim()
            plt.ylim(bottom=0.0, top=max(1.0, ylim[1]))
            plt.draw()
            if save:
                fig.savefig(save + "_" + str(e))

    def plot_rho(self, save=None):
        """Plot normalized autocorrelation function time for each ensemble.

        Parameters
        ----------
        save : str
            saves the figure to a file named 'save' if.
        """
        if not hasattr(self, 'e_dvalue'):
            raise Exception('Run the gamma method first.')
        for e, e_name in enumerate(self.mc_names):
            fig = plt.figure()
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
            if save:
                fig.savefig(save + "_" + str(e))

    def plot_rep_dist(self):
        """Plot replica distribution for each ensemble with more than one replicum."""
        if not hasattr(self, 'e_dvalue'):
            raise Exception('Run the gamma method first.')
        for e, e_name in enumerate(self.mc_names):
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
        """Plot derived Monte Carlo history for each ensemble

        Parameters
        ----------
        expand : bool
            show expanded history for irregular Monte Carlo chains (default: True).
        """
        for e, e_name in enumerate(self.mc_names):
            plt.figure()
            r_length = []
            tmp = []
            tmp_expanded = []
            for r, r_name in enumerate(self.e_content[e_name]):
                tmp.append(self.deltas[r_name] + self.r_values[r_name])
                if expand:
                    tmp_expanded.append(_expand_deltas(self.deltas[r_name], list(self.idl[r_name]), self.shape[r_name]) + self.r_values[r_name])
                    r_length.append(len(tmp_expanded[-1]))
                else:
                    r_length.append(len(tmp[-1]))
            e_N = np.sum(r_length)
            x = np.arange(e_N)
            y_test = np.concatenate(tmp, axis=0)
            if expand:
                y = np.concatenate(tmp_expanded, axis=0)
            else:
                y = y_test
            plt.errorbar(x, y, fmt='.', markersize=3)
            plt.xlim(-0.5, e_N - 0.5)
            plt.title(e_name + f'\nskew: {skew(y_test):.3f} (p={skewtest(y_test).pvalue:.3f}), kurtosis: {kurtosis(y_test):.3f} (p={kurtosistest(y_test).pvalue:.3f})')
            plt.draw()

    def plot_piechart(self, save=None):
        """Plot piechart which shows the fractional contribution of each
        ensemble to the error and returns a dictionary containing the fractions.

        Parameters
        ----------
        save : str
            saves the figure to a file named 'save' if.
        """
        if not hasattr(self, 'e_dvalue'):
            raise Exception('Run the gamma method first.')
        if np.isclose(0.0, self._dvalue, atol=1e-15):
            raise Exception('Error is 0.0')
        labels = self.e_names
        sizes = [self.e_dvalue[name] ** 2 for name in labels] / self._dvalue ** 2
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, startangle=90, normalize=True)
        ax1.axis('equal')
        plt.draw()
        if save:
            fig1.savefig(save)

        return dict(zip(self.e_names, sizes))

    def dump(self, filename, datatype="json.gz", description="", **kwargs):
        """Dump the Obs to a file 'name' of chosen format.

        Parameters
        ----------
        filename : str
            name of the file to be saved.
        datatype : str
            Format of the exported file. Supported formats include
            "json.gz" and "pickle"
        description : str
            Description for output file, only relevant for json.gz format.
        path : str
            specifies a custom path for the file (default '.')
        """
        if 'path' in kwargs:
            file_name = kwargs.get('path') + '/' + filename
        else:
            file_name = filename

        if datatype == "json.gz":
            from .input.json import dump_to_json
            dump_to_json([self], file_name, description=description)
        elif datatype == "pickle":
            with open(file_name + '.p', 'wb') as fb:
                pickle.dump(self, fb)
        else:
            raise Exception("Unknown datatype " + str(datatype))

    def export_jackknife(self):
        """Export jackknife samples from the Obs

        Returns
        -------
        numpy.ndarray
            Returns a numpy array of length N + 1 where N is the number of samples
            for the given ensemble and replicum. The zeroth entry of the array contains
            the mean value of the Obs, entries 1 to N contain the N jackknife samples
            derived from the Obs. The current implementation only works for observables
            defined on exactly one ensemble and replicum. The derived jackknife samples
            should agree with samples from a full jackknife analysis up to O(1/N).
        """

        if len(self.names) != 1:
            raise Exception("'export_jackknife' is only implemented for Obs defined on one ensemble and replicum.")

        name = self.names[0]
        full_data = self.deltas[name] + self.r_values[name]
        n = full_data.size
        mean = self.value
        tmp_jacks = np.zeros(n + 1)
        tmp_jacks[0] = mean
        tmp_jacks[1:] = (n * mean - full_data) / (n - 1)
        return tmp_jacks

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return 'Obs[' + str(self) + ']'

    def __str__(self):
        if self._dvalue == 0.0:
            return str(self.value)
        fexp = np.floor(np.log10(self._dvalue))
        if fexp < 0.0:
            return '{:{form}}({:2.0f})'.format(self.value, self._dvalue * 10 ** (-fexp + 1), form='.' + str(-int(fexp) + 1) + 'f')
        elif fexp == 0.0:
            return '{:.1f}({:1.1f})'.format(self.value, self._dvalue)
        else:
            return '{:.0f}({:2.0f})'.format(self.value, self._dvalue)

    def __hash__(self):
        hash_tuple = (np.array([self.value]).astype(np.float32).data.tobytes(),)
        hash_tuple += tuple([o.astype(np.float32).data.tobytes() for o in self.deltas.values()])
        hash_tuple += tuple([np.array([o.errsq()]).astype(np.float32).data.tobytes() for o in self.covobs.values()])
        hash_tuple += tuple([o.encode() for o in self.names])
        m = hashlib.md5()
        [m.update(o) for o in hash_tuple]
        return int(m.hexdigest(), 16) & 0xFFFFFFFF

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
            elif y.__class__.__name__ in ['Corr', 'CObs']:
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
            elif y.__class__.__name__ in ['Corr', 'CObs']:
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
            elif y.__class__.__name__ in ['Corr', 'CObs']:
                return NotImplemented
            else:
                return derived_observable(lambda x, **kwargs: x[0] - y, [self], man_grad=[1])

    def __rsub__(self, y):
        return -1 * (self - y)

    def __pos__(self):
        return self

    def __neg__(self):
        return -1 * self

    def __truediv__(self, y):
        if isinstance(y, Obs):
            return derived_observable(lambda x, **kwargs: x[0] / x[1], [self, y], man_grad=[1 / y.value, - self.value / y.value ** 2])
        else:
            if isinstance(y, np.ndarray):
                return np.array([self / o for o in y])
            elif y.__class__.__name__ in ['Corr', 'CObs']:
                return NotImplemented
            else:
                return derived_observable(lambda x, **kwargs: x[0] / y, [self], man_grad=[1 / y])

    def __rtruediv__(self, y):
        if isinstance(y, Obs):
            return derived_observable(lambda x, **kwargs: x[0] / x[1], [y, self], man_grad=[1 / self.value, - y.value / self.value ** 2])
        else:
            if isinstance(y, np.ndarray):
                return np.array([o / self for o in y])
            elif y.__class__.__name__ in ['Corr', 'CObs']:
                return NotImplemented
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

    def __pos__(self):
        return self

    def __neg__(self):
        return -1 * self

    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag

    def __str__(self):
        return '(' + str(self.real) + int(self.imag >= 0.0) * '+' + str(self.imag) + 'j)'

    def __repr__(self):
        return 'CObs[' + str(self) + ']'


def _expand_deltas(deltas, idx, shape):
    """Expand deltas defined on idx to a regular, contiguous range, where holes are filled by 0.
       If idx is of type range, the deltas are not changed

    Parameters
    ----------
    deltas : list
        List of fluctuations
    idx : list
        List or range of configs on which the deltas are defined, has to be sorted in ascending order.
    shape : int
        Number of configs in idx.
    """
    if isinstance(idx, range):
        return deltas
    else:
        ret = np.zeros(idx[-1] - idx[0] + 1)
        for i in range(shape):
            ret[idx[i] - idx[0]] = deltas[i]
        return ret


def _merge_idx(idl):
    """Returns the union of all lists in idl as sorted list

    Parameters
    ----------
    idl : list
        List of lists or ranges.
    """

    # Use groupby to efficiently check whether all elements of idl are identical
    try:
        g = groupby(idl)
        if next(g, True) and not next(g, False):
            return idl[0]
    except Exception:
        pass

    if np.all([type(idx) is range for idx in idl]):
        if len(set([idx[0] for idx in idl])) == 1:
            idstart = min([idx.start for idx in idl])
            idstop = max([idx.stop for idx in idl])
            idstep = min([idx.step for idx in idl])
            return range(idstart, idstop, idstep)

    return sorted(set().union(*idl))


def _intersection_idx(idl):
    """Returns the intersection of all lists in idl as sorted list

    Parameters
    ----------
    idl : list
        List of lists or ranges.
    """

    def _lcm(*args):
        """Returns the lowest common multiple of args.

        From python 3.9 onwards the math library contains an lcm function."""
        return reduce(lambda a, b: a * b // gcd(a, b), args)

    # Use groupby to efficiently check whether all elements of idl are identical
    try:
        g = groupby(idl)
        if next(g, True) and not next(g, False):
            return idl[0]
    except Exception:
        pass

    if np.all([type(idx) is range for idx in idl]):
        if len(set([idx[0] for idx in idl])) == 1:
            idstart = max([idx.start for idx in idl])
            idstop = min([idx.stop for idx in idl])
            idstep = _lcm(*[idx.step for idx in idl])
            return range(idstart, idstop, idstep)

    return sorted(set.intersection(*[set(o) for o in idl]))


def _expand_deltas_for_merge(deltas, idx, shape, new_idx):
    """Expand deltas defined on idx to the list of configs that is defined by new_idx.
       New, empty entries are filled by 0. If idx and new_idx are of type range, the smallest
       common divisor of the step sizes is used as new step size.

    Parameters
    ----------
    deltas : list
        List of fluctuations
    idx : list
        List or range of configs on which the deltas are defined.
        Has to be a subset of new_idx and has to be sorted in ascending order.
    shape : list
        Number of configs in idx.
    new_idx : list
        List of configs that defines the new range, has to be sorted in ascending order.
    """

    if type(idx) is range and type(new_idx) is range:
        if idx == new_idx:
            return deltas
    ret = np.zeros(new_idx[-1] - new_idx[0] + 1)
    for i in range(shape):
        ret[idx[i] - new_idx[0]] = deltas[i]
    return np.array([ret[new_idx[i] - new_idx[0]] for i in range(len(new_idx))])


def _collapse_deltas_for_merge(deltas, idx, shape, new_idx):
    """Collapse deltas defined on idx to the list of configs that is defined by new_idx.
       If idx and new_idx are of type range, the smallest
       common divisor of the step sizes is used as new step size.

    Parameters
    ----------
    deltas : list
        List of fluctuations
    idx : list
        List or range of configs on which the deltas are defined.
        Has to be a subset of new_idx and has to be sorted in ascending order.
    shape : list
        Number of configs in idx.
    new_idx : list
        List of configs that defines the new range, has to be sorted in ascending order.
    """

    if type(idx) is range and type(new_idx) is range:
        if idx == new_idx:
            return deltas
    ret = np.zeros(new_idx[-1] - new_idx[0] + 1)
    for i in range(shape):
        if idx[i] in new_idx:
            ret[idx[i] - new_idx[0]] = deltas[i]
    return np.array([ret[new_idx[i] - new_idx[0]] for i in range(len(new_idx))])


def _filter_zeroes(deltas, idx, eps=Obs.filter_eps):
    """Filter out all configurations with vanishing fluctuation such that they do not
       contribute to the error estimate anymore. Returns the new deltas and
       idx according to the filtering.
       A fluctuation is considered to be vanishing, if it is smaller than eps times
       the mean of the absolute values of all deltas in one list.

    Parameters
    ----------
    deltas : list
        List of fluctuations
    idx : list
        List or ranges of configs on which the deltas are defined.
    eps : float
        Prefactor that enters the filter criterion.
    """
    new_deltas = []
    new_idx = []
    maxd = np.mean(np.fabs(deltas))
    for i in range(len(deltas)):
        if abs(deltas[i]) > eps * maxd:
            new_deltas.append(deltas[i])
            new_idx.append(idx[i])
    if new_idx:
        return np.array(new_deltas), new_idx
    else:
        return deltas, idx


def derived_observable(func, data, array_mode=False, **kwargs):
    """Construct a derived Obs according to func(data, **kwargs) using automatic differentiation.

    Parameters
    ----------
    func : object
        arbitrary function of the form func(data, **kwargs). For the
        automatic differentiation to work, all numpy functions have to have
        the autograd wrapper (use 'import autograd.numpy as anp').
    data : list
        list of Obs, e.g. [obs1, obs2, obs3].
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
    if not all(isinstance(x, Obs) for x in raveled_data):
        for i in range(len(raveled_data)):
            if isinstance(raveled_data[i], (int, float)):
                raveled_data[i] = cov_Obs(raveled_data[i], 0.0, "###dummy_covobs###")

    allcov = {}
    for o in raveled_data:
        for name in o.cov_names:
            if name in allcov:
                if not np.allclose(allcov[name], o.covobs[name].cov):
                    raise Exception('Inconsistent covariance matrices for %s!' % (name))
            else:
                allcov[name] = o.covobs[name].cov

    n_obs = len(raveled_data)
    new_names = sorted(set([y for x in [o.names for o in raveled_data] for y in x]))
    new_cov_names = sorted(set([y for x in [o.cov_names for o in raveled_data] for y in x]))
    new_sample_names = sorted(set(new_names) - set(new_cov_names))

    is_merged = {name: (len(list(filter(lambda o: o.is_merged.get(name, False) is True, raveled_data))) > 0) for name in new_sample_names}
    reweighted = len(list(filter(lambda o: o.reweighted is True, raveled_data))) > 0

    if data.ndim == 1:
        values = np.array([o.value for o in data])
    else:
        values = np.vectorize(lambda x: x.value)(data)

    new_values = func(values, **kwargs)

    multi = int(isinstance(new_values, np.ndarray))

    new_r_values = {}
    new_idl_d = {}
    for name in new_sample_names:
        idl = []
        tmp_values = np.zeros(n_obs)
        for i, item in enumerate(raveled_data):
            tmp_values[i] = item.r_values.get(name, item.value)
            tmp_idl = item.idl.get(name)
            if tmp_idl is not None:
                idl.append(tmp_idl)
        if multi > 0:
            tmp_values = np.array(tmp_values).reshape(data.shape)
        new_r_values[name] = func(tmp_values, **kwargs)
        new_idl_d[name] = _merge_idx(idl)
        if not is_merged[name]:
            is_merged[name] = (1 != len(set([len(idx) for idx in [*idl, new_idl_d[name]]])))

    if 'man_grad' in kwargs:
        deriv = np.asarray(kwargs.get('man_grad'))
        if new_values.shape + data.shape != deriv.shape:
            raise Exception('Manual derivative does not have correct shape.')
    elif kwargs.get('num_grad') is True:
        if multi > 0:
            raise Exception('Multi mode currently not supported for numerical derivative')
        options = {
            'base_step': 0.1,
            'step_ratio': 2.5}
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

    if array_mode is True:

        class _Zero_grad():
            def __init__(self, N):
                self.grad = np.zeros((N, 1))

        new_covobs_lengths = dict(set([y for x in [[(n, o.covobs[n].N) for n in o.cov_names] for o in raveled_data] for y in x]))
        d_extracted = {}
        g_extracted = {}
        for name in new_sample_names:
            d_extracted[name] = []
            ens_length = len(new_idl_d[name])
            for i_dat, dat in enumerate(data):
                d_extracted[name].append(np.array([_expand_deltas_for_merge(o.deltas.get(name, np.zeros(ens_length)), o.idl.get(name, new_idl_d[name]), o.shape.get(name, ens_length), new_idl_d[name]) for o in dat.reshape(np.prod(dat.shape))]).reshape(dat.shape + (ens_length, )))
        for name in new_cov_names:
            g_extracted[name] = []
            zero_grad = _Zero_grad(new_covobs_lengths[name])
            for i_dat, dat in enumerate(data):
                g_extracted[name].append(np.array([o.covobs.get(name, zero_grad).grad for o in dat.reshape(np.prod(dat.shape))]).reshape(dat.shape + (new_covobs_lengths[name], 1)))

    for i_val, new_val in np.ndenumerate(new_values):
        new_deltas = {}
        new_grad = {}
        if array_mode is True:
            for name in new_sample_names:
                ens_length = d_extracted[name][0].shape[-1]
                new_deltas[name] = np.zeros(ens_length)
                for i_dat, dat in enumerate(d_extracted[name]):
                    new_deltas[name] += np.tensordot(deriv[i_val + (i_dat, )], dat)
            for name in new_cov_names:
                new_grad[name] = 0
                for i_dat, dat in enumerate(g_extracted[name]):
                    new_grad[name] += np.tensordot(deriv[i_val + (i_dat, )], dat)
        else:
            for j_obs, obs in np.ndenumerate(data):
                for name in obs.names:
                    if name in obs.cov_names:
                        new_grad[name] = new_grad.get(name, 0) + deriv[i_val + j_obs] * obs.covobs[name].grad
                    else:
                        new_deltas[name] = new_deltas.get(name, 0) + deriv[i_val + j_obs] * _expand_deltas_for_merge(obs.deltas[name], obs.idl[name], obs.shape[name], new_idl_d[name])

        new_covobs = {name: Covobs(0, allcov[name], name, grad=new_grad[name]) for name in new_grad}

        if not set(new_covobs.keys()).isdisjoint(new_deltas.keys()):
            raise Exception('The same name has been used for deltas and covobs!')
        new_samples = []
        new_means = []
        new_idl = []
        new_names_obs = []
        for name in new_names:
            if name not in new_covobs:
                if is_merged[name]:
                    filtered_deltas, filtered_idl_d = _filter_zeroes(new_deltas[name], new_idl_d[name])
                else:
                    filtered_deltas = new_deltas[name]
                    filtered_idl_d = new_idl_d[name]

                new_samples.append(filtered_deltas)
                new_idl.append(filtered_idl_d)
                new_means.append(new_r_values[name][i_val])
                new_names_obs.append(name)
        final_result[i_val] = Obs(new_samples, new_names_obs, means=new_means, idl=new_idl)
        for name in new_covobs:
            final_result[i_val].names.append(name)
        final_result[i_val]._covobs = new_covobs
        final_result[i_val]._value = new_val
        final_result[i_val].is_merged = is_merged
        final_result[i_val].reweighted = reweighted

    if multi == 0:
        final_result = final_result.item()

    return final_result


def _reduce_deltas(deltas, idx_old, idx_new):
    """Extract deltas defined on idx_old on all configs of idx_new.

    Assumes, that idx_old and idx_new are correctly defined idl, i.e., they
    are ordered in an ascending order.

    Parameters
    ----------
    deltas : list
        List of fluctuations
    idx_old : list
        List or range of configs on which the deltas are defined
    idx_new : list
        List of configs for which we want to extract the deltas.
        Has to be a subset of idx_old.
    """
    if not len(deltas) == len(idx_old):
        raise Exception('Length of deltas and idx_old have to be the same: %d != %d' % (len(deltas), len(idx_old)))
    if type(idx_old) is range and type(idx_new) is range:
        if idx_old == idx_new:
            return deltas
    shape = len(idx_new)
    ret = np.zeros(shape)
    oldpos = 0
    for i in range(shape):
        pos = -1
        for j in range(oldpos, len(idx_old)):
            if idx_old[j] == idx_new[i]:
                pos = j
                break
        if pos < 0:
            raise Exception('Error in _reduce_deltas: Config %d not in idx_old' % (idx_new[i]))
        ret[i] = deltas[pos]
        oldpos = pos
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
    all_configs : bool
        if True, the reweighted observables are normalized by the average of
        the reweighting factor on all configurations in weight.idl and not
        on the configurations in obs[i].idl. Default False.
    """
    result = []
    for i in range(len(obs)):
        if len(obs[i].cov_names):
            raise Exception('Error: Not possible to reweight an Obs that contains covobs!')
        if not set(obs[i].names).issubset(weight.names):
            raise Exception('Error: Ensembles do not fit')
        for name in obs[i].names:
            if not set(obs[i].idl[name]).issubset(weight.idl[name]):
                raise Exception('obs[%d] has to be defined on a subset of the configs in weight.idl[%s]!' % (i, name))
        new_samples = []
        w_deltas = {}
        for name in sorted(obs[i].names):
            w_deltas[name] = _reduce_deltas(weight.deltas[name], weight.idl[name], obs[i].idl[name])
            new_samples.append((w_deltas[name] + weight.r_values[name]) * (obs[i].deltas[name] + obs[i].r_values[name]))
        tmp_obs = Obs(new_samples, sorted(obs[i].names), idl=[obs[i].idl[name] for name in sorted(obs[i].names)])

        if kwargs.get('all_configs'):
            new_weight = weight
        else:
            new_weight = Obs([w_deltas[name] + weight.r_values[name] for name in sorted(obs[i].names)], sorted(obs[i].names), idl=[obs[i].idl[name] for name in sorted(obs[i].names)])

        result.append(tmp_obs / new_weight)
        result[-1].reweighted = True
        result[-1].is_merged = obs[i].is_merged

    return result


def correlate(obs_a, obs_b):
    """Correlate two observables.

    Parameters
    ----------
    obs_a : Obs
        First observable
    obs_b : Obs
        Second observable

    Notes
    -----
    Keep in mind to only correlate primary observables which have not been reweighted
    yet. The reweighting has to be applied after correlating the observables.
    Currently only works if ensembles are identical (this is not strictly necessary).
    """

    if sorted(obs_a.names) != sorted(obs_b.names):
        raise Exception('Ensembles do not fit')
    if len(obs_a.cov_names) or len(obs_b.cov_names):
        raise Exception('Error: Not possible to correlate Obs that contain covobs!')
    for name in obs_a.names:
        if obs_a.shape[name] != obs_b.shape[name]:
            raise Exception('Shapes of ensemble', name, 'do not fit')
        if obs_a.idl[name] != obs_b.idl[name]:
            raise Exception('idl of ensemble', name, 'do not fit')

    if obs_a.reweighted is True:
        warnings.warn("The first observable is already reweighted.", RuntimeWarning)
    if obs_b.reweighted is True:
        warnings.warn("The second observable is already reweighted.", RuntimeWarning)

    new_samples = []
    new_idl = []
    for name in sorted(obs_a.names):
        new_samples.append((obs_a.deltas[name] + obs_a.r_values[name]) * (obs_b.deltas[name] + obs_b.r_values[name]))
        new_idl.append(obs_a.idl[name])

    o = Obs(new_samples, sorted(obs_a.names), idl=new_idl)
    o.is_merged = {name: (obs_a.is_merged.get(name, False) or obs_b.is_merged.get(name, False)) for name in o.names}
    o.reweighted = obs_a.reweighted or obs_b.reweighted
    return o


def covariance(obs, visualize=False, correlation=False, smooth=None, **kwargs):
    r'''Calculates the error covariance matrix of a set of observables.

    The gamma method has to be applied first to all observables.

    Parameters
    ----------
    obs : list or numpy.ndarray
        List or one dimensional array of Obs
    visualize : bool
        If True plots the corresponding normalized correlation matrix (default False).
    correlation : bool
        If True the correlation matrix instead of the error covariance matrix is returned (default False).
    smooth : None or int
        If smooth is an integer 'E' between 2 and the dimension of the matrix minus 1 the eigenvalue
        smoothing procedure of hep-lat/9412087 is applied to the correlation matrix which leaves the
        largest E eigenvalues essentially unchanged and smoothes the smaller eigenvalues to avoid extremely
        small ones.

    Notes
    -----
    The error covariance is defined such that it agrees with the squared standard error for two identical observables
    $$\operatorname{cov}(a,a)=\sum_{s=1}^N\delta_a^s\delta_a^s/N^2=\Gamma_{aa}(0)/N=\operatorname{var}(a)/N=\sigma_a^2$$
    in the absence of autocorrelation.
    The error covariance is estimated by calculating the correlation matrix assuming no autocorrelation and then rescaling the correlation matrix by the full errors including the previous gamma method estimate for the autocorrelation of the observables. The covariance at windowsize 0 is guaranteed to be positive semi-definite
    $$\sum_{i,j}v_i\Gamma_{ij}(0)v_j=\frac{1}{N}\sum_{s=1}^N\sum_{i,j}v_i\delta_i^s\delta_j^s v_j=\frac{1}{N}\sum_{s=1}^N\sum_{i}|v_i\delta_i^s|^2\geq 0\,,$$ for every $v\in\mathbb{R}^M$, while such an identity does not hold for larger windows/lags.
    For observables defined on a single ensemble our approximation is equivalent to assuming that the integrated autocorrelation time of an off-diagonal element is equal to the geometric mean of the integrated autocorrelation times of the corresponding diagonal elements.
    $$\tau_{\mathrm{int}, ij}=\sqrt{\tau_{\mathrm{int}, i}\times \tau_{\mathrm{int}, j}}$$
    This construction ensures that the estimated covariance matrix is positive semi-definite (up to numerical rounding errors).
    '''

    length = len(obs)

    max_samples = np.max([o.N for o in obs])
    if max_samples <= length and not [item for sublist in [o.cov_names for o in obs] for item in sublist]:
        warnings.warn(f"The dimension of the covariance matrix ({length}) is larger or equal to the number of samples ({max_samples}). This will result in a rank deficient matrix.", RuntimeWarning)

    cov = np.zeros((length, length))
    for i in range(length):
        for j in range(i, length):
            cov[i, j] = _covariance_element(obs[i], obs[j])
    cov = cov + cov.T - np.diag(np.diag(cov))

    corr = np.diag(1 / np.sqrt(np.diag(cov))) @ cov @ np.diag(1 / np.sqrt(np.diag(cov)))

    if isinstance(smooth, int):
        corr = _smooth_eigenvalues(corr, smooth)

    if visualize:
        plt.matshow(corr, vmin=-1, vmax=1)
        plt.set_cmap('RdBu')
        plt.colorbar()
        plt.draw()

    if correlation is True:
        return corr

    errors = [o.dvalue for o in obs]
    cov = np.diag(errors) @ corr @ np.diag(errors)

    eigenvalues = np.linalg.eigh(cov)[0]
    if not np.all(eigenvalues >= 0):
        warnings.warn("Covariance matrix is not positive semi-definite (Eigenvalues: " + str(eigenvalues) + ")", RuntimeWarning)

    return cov


def _smooth_eigenvalues(corr, E):
    """Eigenvalue smoothing as described in hep-lat/9412087

    corr : np.ndarray
        correlation matrix
    E : integer
        Number of eigenvalues to be left substantially unchanged
    """
    if not (2 < E < corr.shape[0] - 1):
        raise Exception(f"'E' has to be between 2 and the dimension of the correlation matrix minus 1 ({corr.shape[0] - 1}).")
    vals, vec = np.linalg.eigh(corr)
    lambda_min = np.mean(vals[:-E])
    vals[vals < lambda_min] = lambda_min
    vals /= np.mean(vals)
    return vec @ np.diag(vals) @ vec.T


def _covariance_element(obs1, obs2):
    """Estimates the covariance of two Obs objects, neglecting autocorrelations."""

    def calc_gamma(deltas1, deltas2, idx1, idx2, new_idx):
        deltas1 = _collapse_deltas_for_merge(deltas1, idx1, len(idx1), new_idx)
        deltas2 = _collapse_deltas_for_merge(deltas2, idx2, len(idx2), new_idx)
        return np.sum(deltas1 * deltas2)

    if set(obs1.names).isdisjoint(set(obs2.names)):
        return 0.0

    if not hasattr(obs1, 'e_dvalue') or not hasattr(obs2, 'e_dvalue'):
        raise Exception('The gamma method has to be applied to both Obs first.')

    dvalue = 0.0

    for e_name in obs1.mc_names:

        if e_name not in obs2.mc_names:
            continue

        idl_d = {}
        for r_name in obs1.e_content[e_name]:
            if r_name not in obs2.e_content[e_name]:
                continue
            idl_d[r_name] = _intersection_idx([obs1.idl[r_name], obs2.idl[r_name]])

        gamma = 0.0

        for r_name in obs1.e_content[e_name]:
            if r_name not in obs2.e_content[e_name]:
                continue
            if len(idl_d[r_name]) == 0:
                continue
            gamma += calc_gamma(obs1.deltas[r_name], obs2.deltas[r_name], obs1.idl[r_name], obs2.idl[r_name], idl_d[r_name])

        if gamma == 0.0:
            continue

        gamma_div = 0.0
        for r_name in obs1.e_content[e_name]:
            if r_name not in obs2.e_content[e_name]:
                continue
            if len(idl_d[r_name]) == 0:
                continue
            gamma_div += np.sqrt(calc_gamma(obs1.deltas[r_name], obs1.deltas[r_name], obs1.idl[r_name], obs1.idl[r_name], idl_d[r_name]) * calc_gamma(obs2.deltas[r_name], obs2.deltas[r_name], obs2.idl[r_name], obs2.idl[r_name], idl_d[r_name]))
        gamma /= gamma_div

        dvalue += gamma

    for e_name in obs1.cov_names:

        if e_name not in obs2.cov_names:
            continue

        dvalue += float(np.dot(np.transpose(obs1.covobs[e_name].grad), np.dot(obs1.covobs[e_name].cov, obs2.covobs[e_name].grad)))

    return dvalue


def import_jackknife(jacks, name, idl=None):
    """Imports jackknife samples and returns an Obs

    Parameters
    ----------
    jacks : numpy.ndarray
        numpy array containing the mean value as zeroth entry and
        the N jackknife samples as first to Nth entry.
    name : str
        name of the ensemble the samples are defined on.
    """
    length = len(jacks) - 1
    prj = (np.ones((length, length)) - (length - 1) * np.identity(length))
    samples = jacks[1:] @ prj
    mean = np.mean(samples)
    new_obs = Obs([samples - mean], [name], idl=idl, means=[mean])
    new_obs._value = jacks[0]
    return new_obs


def merge_obs(list_of_obs):
    """Combine all observables in list_of_obs into one new observable

    Parameters
    ----------
    list_of_obs : list
        list of the Obs object to be combined

    Notes
    -----
    It is not possible to combine obs which are based on the same replicum
    """
    replist = [item for obs in list_of_obs for item in obs.names]
    if (len(replist) == len(set(replist))) is False:
        raise Exception('list_of_obs contains duplicate replica: %s' % (str(replist)))
    if any([len(o.cov_names) for o in list_of_obs]):
        raise Exception('Not possible to merge data that contains covobs!')
    new_dict = {}
    idl_dict = {}
    for o in list_of_obs:
        new_dict.update({key: o.deltas.get(key, 0) + o.r_values.get(key, 0)
                        for key in set(o.deltas) | set(o.r_values)})
        idl_dict.update({key: o.idl.get(key, 0) for key in set(o.deltas)})

    names = sorted(new_dict.keys())
    o = Obs([new_dict[name] for name in names], names, idl=[idl_dict[name] for name in names])
    o.is_merged = {name: np.any([oi.is_merged.get(name, False) for oi in list_of_obs]) for name in o.names}
    o.reweighted = np.max([oi.reweighted for oi in list_of_obs])
    return o


def cov_Obs(means, cov, name, grad=None):
    """Create an Obs based on mean(s) and a covariance matrix

    Parameters
    ----------
    mean : list of floats or float
        N mean value(s) of the new Obs
    cov : list or array
        2d (NxN) Covariance matrix, 1d diagonal entries or 0d covariance
    name : str
        identifier for the covariance matrix
    grad : list or array
        Gradient of the Covobs wrt. the means belonging to cov.
    """

    def covobs_to_obs(co):
        """Make an Obs out of a Covobs

        Parameters
        ----------
        co : Covobs
            Covobs to be embedded into the Obs
        """
        o = Obs([], [], means=[])
        o._value = co.value
        o.names.append(co.name)
        o._covobs[co.name] = co
        o._dvalue = np.sqrt(co.errsq())
        return o

    ol = []
    if isinstance(means, (float, int)):
        means = [means]

    for i in range(len(means)):
        ol.append(covobs_to_obs(Covobs(means[i], cov, name, pos=i, grad=grad)))
    if ol[0].covobs[name].N != len(means):
        raise Exception('You have to provide %d mean values!' % (ol[0].N))
    if len(ol) == 1:
        return ol[0]
    return ol
