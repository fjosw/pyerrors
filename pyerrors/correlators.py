import warnings
import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt
import scipy.linalg
from .pyerrors import Obs, dump_object, reweight, correlate
from .fits import least_squares
from .linalg import eigh, inv, cholesky
from .roots import find_root


class Corr:
    """The class for a correlator (time dependent sequence of pe.Obs).

    Everything, this class does, can be achieved using lists or arrays of Obs.
    But it is simply more convenient to have a dedicated object for correlators.
    One often wants to add or multiply correlators of the same length at every timeslice and it is inconvinient
    to iterate over all timeslices for every operation. This is especially true, when dealing with smearing matrices.

    The correlator can have two types of content: An Obs at every timeslice OR a GEVP
    smearing matrix at every timeslice. Other dependency (eg. spacial) are not supported.

    """

    def __init__(self, data_input, padding_front=0, padding_back=0, prange=None):
        # All data_input should be a list of things at different timeslices. This needs to be verified

        if not isinstance(data_input, list):
            raise TypeError('Corr__init__ expects a list of timeslices.')
        # data_input can have multiple shapes. The simplest one is a list of Obs.
        # We check, if this is the case
        if all([isinstance(item, Obs) for item in data_input]):
            self.content = [np.asarray([item]) for item in data_input]
            # Wrapping the Obs in an array ensures that the data structure is consistent with smearing matrices.
            self.N = 1  # number of smearings

        # data_input in the form [np.array(Obs,NxN)]
        elif all([isinstance(item, np.ndarray) or item is None for item in data_input]) and any([isinstance(item, np.ndarray) for item in data_input]):
            self.content = data_input

            noNull = [a for a in self.content if not (a is None)]  # To check if the matrices are correct for all undefined elements
            self.N = noNull[0].shape[0]
            # The checks are now identical to the case above
            if self.N > 1 and noNull[0].shape[0] != noNull[0].shape[1]:
                raise Exception("Smearing matrices are not NxN")
            if (not all([item.shape == noNull[0].shape for item in noNull])):
                raise Exception("Items in data_input are not of identical shape." + str(noNull))
        else:  # In case its a list of something else.
            raise Exception("data_input contains item of wrong type")

        self.tag = None

        # We now apply some padding to our list. In case that our list represents a correlator of length T but is not defined at every value.
        # An undefined timeslice is represented by the None object
        self.content = [None] * padding_front + self.content + [None] * padding_back
        self.T = len(self.content)  # for convenience: will be used a lot

        # The attribute "range" [start,end] marks a range of two timeslices.
        # This is useful for keeping track of plateaus and fitranges.
        # The range can be inherited from other Corrs, if the operation should not alter a chosen range eg. multiplication with a constant.
        self.prange = prange

        self.gamma_method()

    def __getitem__(self, idx):
        """Return the content of timeslice idx"""
        if len(self.content[idx]) == 1:
            return self.content[idx][0]
        else:
            return self.content[idx]

    @property
    def reweighted(self):
        bool_array = np.array([list(map(lambda x: x.reweighted, o)) for o in self.content])
        if np.all(bool_array == 1):
            return True
        elif np.all(bool_array == 0):
            return False
        else:
            raise Exception("Reweighting status of correlator corrupted.")

    def gamma_method(self):
        """Apply the gamma method to the content of the Corr."""
        for item in self.content:
            if not(item is None):
                if self.N == 1:
                    item[0].gamma_method()
                else:
                    for i in range(self.N):
                        for j in range(self.N):
                            item[i, j].gamma_method()

    # We need to project the Correlator with a Vector to get a single value at each timeslice.
    # The method can use one or two vectors.
    # If two are specified it returns v1@G@v2 (the order might be very important.)
    # By default it will return the lowest source, which usually means unsmeared-unsmeared (0,0), but it does not have to
    def projected(self, vector_l=None, vector_r=None):
        if self.N == 1:
            raise Exception("Trying to project a Corr, that already has N=1.")
            # This Exception is in no way necessary. One could just return self
            # But there is no scenario, where a user would want that to happen and the error message might be more informative.

        self.gamma_method()

        if vector_l is None:
            vector_l, vector_r = np.asarray([1.] + (self.N - 1) * [0.]), np.asarray([1.] + (self.N - 1) * [0.])
        elif(vector_r is None):
            vector_r = vector_l

        if not vector_l.shape == vector_r.shape == (self.N,):
            raise Exception("Vectors are of wrong shape!")

        # We always normalize before projecting! But we only raise a warning, when it is clear, they where not meant to be normalized.
        if (not (0.95 < vector_r @ vector_r < 1.05)) or (not (0.95 < vector_l @ vector_l < 1.05)):
            print("Vectors are normalized before projection!")

        vector_l, vector_r = vector_l / np.sqrt((vector_l @ vector_l)), vector_r / np.sqrt(vector_r @ vector_r)

        newcontent = [None if (item is None) else np.asarray([vector_l.T @ item @ vector_r]) for item in self.content]
        return Corr(newcontent)

    def sum(self):
        return np.sqrt(self.N) * self.projected(np.ones(self.N))

    # For purposes of debugging and verification, one might want to see a single smearing level. smearing will return a Corr at the specified i,j. where both are integers 0<=i,j<N.
    def smearing(self, i, j):
        if self.N == 1:
            raise Exception("Trying to pick smearing from projected Corr")
        newcontent = [None if(item is None) else item[i, j] for item in self.content]
        return Corr(newcontent)

    # Obs and Matplotlib do not play nicely
    # We often want to retrieve x,y,y_err as lists to pass them to something like pyplot.errorbar
    def plottable(self):
        """Outputs the correlator in a plotable format.

        Outputs three lists containing the timeslice index, the value on each
        timeslice and the error on each timeslice.
        """
        if self.N != 1:
            raise Exception("Can only make Corr[N=1] plottable")  # We could also autoproject to the groundstate or expect vectors, but this is supposed to be a super simple function.
        x_list = [x for x in range(self.T) if not self.content[x] is None]
        y_list = [y[0].value for y in self.content if y is not None]
        y_err_list = [y[0].dvalue for y in self.content if y is not None]

        return x_list, y_list, y_err_list

    # symmetric returns a Corr, that has been symmetrized.
    # A symmetry checker is still to be implemented
    # The method will not delete any redundant timeslices (Bad for memory, Great for convenience)
    def symmetric(self):
        """ Symmetrize the correlator around x0=0."""
        if self.T % 2 != 0:
            raise Exception("Can not symmetrize odd T")

        if np.argmax(np.abs(self.content)) != 0:
            warnings.warn("Correlator does not seem to be symmetric around x0=0.", RuntimeWarning)

        newcontent = [self.content[0]]
        for t in range(1, self.T):
            if (self.content[t] is None) or (self.content[self.T - t] is None):
                newcontent.append(None)
            else:
                newcontent.append(0.5 * (self.content[t] + self.content[self.T - t]))
        if(all([x is None for x in newcontent])):
            raise Exception("Corr could not be symmetrized: No redundant values")
        return Corr(newcontent, prange=self.prange)

    def anti_symmetric(self):
        """Anti-symmetrize the correlator around x0=0."""
        if self.T % 2 != 0:
            raise Exception("Can not symmetrize odd T")

        if not all([o.is_zero_within_error() for o in self.content[0]]):
            warnings.warn("Correlator does not seem to be anti-symmetric around x0=0.", RuntimeWarning)

        newcontent = [self.content[0]]
        for t in range(1, self.T):
            if (self.content[t] is None) or (self.content[self.T - t] is None):
                newcontent.append(None)
            else:
                newcontent.append(0.5 * (self.content[t] - self.content[self.T - t]))
        if(all([x is None for x in newcontent])):
            raise Exception("Corr could not be symmetrized: No redundant values")
        return Corr(newcontent, prange=self.prange)

    # This method will symmetrice the matrices and therefore make them positive definit.
    def smearing_symmetric(self):
        if self.N > 1:
            transposed = [None if (G is None) else G.T for G in self.content]
            return 0.5 * (Corr(transposed) + self)
        if self.N == 1:
            raise Exception("Trying to symmetrize a smearing matrix, that already has N=1.")

    # We also include a simple GEVP method based on Scipy.linalg
    def GEVP(self, t0, ts, state=1):
        if (self.content[t0] is None) or (self.content[ts] is None):
            raise Exception("Corr not defined at t0/ts")
        G0, Gt = np.empty([self.N, self.N], dtype="double"), np.empty([self.N, self.N], dtype="double")
        for i in range(self.N):
            for j in range(self.N):
                G0[i, j] = self.content[t0][i, j].value
                Gt[i, j] = self.content[ts][i, j].value

        sp_val, sp_vec = scipy.linalg.eig(Gt, G0)
        sp_vec = sp_vec[:, np.argsort(sp_val)[-state]]  # We only want the eigenvector belonging to the selected state
        sp_vec = sp_vec / np.sqrt(sp_vec @ sp_vec)
        return sp_vec

    def Eigenvalue(self, t0, state=1):
        G = self.smearing_symmetric()
        G0 = G.content[t0]
        L = cholesky(G0)
        Li = inv(L)
        LT = L.T
        LTi = inv(LT)
        newcontent = []
        for t in range(self.T):
            Gt = G.content[t]
            M = Li @ Gt @ LTi
            eigenvalues = eigh(M)[0]
            eigenvalue = eigenvalues[-state]
            newcontent.append(eigenvalue)
        return Corr(newcontent)

    def roll(self, dt):
        """Periodically shift the correlator by dt timeslices

        Attributes:
        -----------
        dt : int
            number of timeslices
        """
        return Corr(list(np.roll(np.array(self.content, dtype=object), dt)))

    def reverse(self):
        """Reverse the time ordering of the Corr"""
        return Corr(self.content[::-1])

    def correlate(self, partner):
        """Correlate the correlator with another correlator or Obs"""
        new_content = []
        for x0, t_slice in enumerate(self.content):
            if t_slice is None:
                new_content.append(None)
            else:
                if isinstance(partner, Corr):
                    if partner.content[x0] is None:
                        new_content.append(None)
                    else:
                        new_content.append(np.array([correlate(o, partner.content[x0][0]) for o in t_slice]))
                elif isinstance(partner, Obs):
                    new_content.append(np.array([correlate(o, partner) for o in t_slice]))
                else:
                    raise Exception("Can only correlate with an Obs or a Corr.")

        return Corr(new_content)

    def reweight(self, weight, **kwargs):
        """Reweight the correlator.

        Parameters
        ----------
        weight : Obs
            Reweighting factor. An Observable that has to be defined on a superset of the
            configurations in obs[i].idl for all i.

        Keyword arguments
        -----------------
        all_configs : bool
            if True, the reweighted observables are normalized by the average of
            the reweighting factor on all configurations in weight.idl and not
            on the configurations in obs[i].idl.
        """
        new_content = []
        for t_slice in self.content:
            if t_slice is None:
                new_content.append(None)
            else:
                new_content.append(np.array(reweight(weight, t_slice, **kwargs)))
        return Corr(new_content)

    def T_symmetry(self, partner, parity=+1):
        """Return the time symmetry average of the correlator and its partner

        Attributes:
        -----------
        partner : Corr
            Time symmetry partner of the Corr
        partity : int
            Parity quantum number of the correlator, can be +1 or -1
        """
        if not isinstance(partner, Corr):
            raise Exception("T partner has to be a Corr object.")
        if parity not in [+1, -1]:
            raise Exception("Parity has to be +1 or -1.")
        T_partner = parity * partner.reverse()

        t_slices = []
        for x0, t_slice in enumerate((self - T_partner).content):
            if t_slice is not None:
                if not t_slice[0].is_zero_within_error(5):
                    t_slices.append(x0)
        if t_slices:
            warnings.warn("T symmetry partners do not agree within 5 sigma on time slices " + str(t_slices) + ".", RuntimeWarning)

        return (self + T_partner) / 2

    def deriv(self, symmetric=True):
        """Return the first derivative of the correlator with respect to x0.

        Attributes:
        -----------
        symmetric : bool
            decides whether symmertic of simple finite differences are used. Default: True
        """
        if not symmetric:
            newcontent = []
            for t in range(self.T - 1):
                if (self.content[t] is None) or (self.content[t + 1] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t + 1] - self.content[t])
            if(all([x is None for x in newcontent])):
                raise Exception("Derivative is undefined at all timeslices")
            return Corr(newcontent, padding_back=1)
        if symmetric:
            newcontent = []
            for t in range(1, self.T - 1):
                if (self.content[t - 1] is None) or (self.content[t + 1] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(0.5 * (self.content[t + 1] - self.content[t - 1]))
            if(all([x is None for x in newcontent])):
                raise Exception('Derivative is undefined at all timeslices')
            return Corr(newcontent, padding_back=1, padding_front=1)

    def second_deriv(self):
        """Return the second derivative of the correlator with respect to x0."""
        newcontent = []
        for t in range(1, self.T - 1):
            if (self.content[t - 1] is None) or (self.content[t + 1] is None):
                newcontent.append(None)
            else:
                newcontent.append((self.content[t + 1] - 2 * self.content[t] + self.content[t - 1]))
        if(all([x is None for x in newcontent])):
            raise Exception("Derivative is undefined at all timeslices")
        return Corr(newcontent, padding_back=1, padding_front=1)

    def m_eff(self, variant='log', guess=1.0):
        """Returns the effective mass of the correlator as correlator object

        Parameters
        ----------
        variant : str
            log: uses the standard effective mass log(C(t) / C(t+1))
            cosh : Use periodicitiy of the correlator by solving C(t) / C(t+1) = cosh(m * (t - T/2)) / cosh(m * (t + 1 - T/2)) for m.
            sinh : Use anti-periodicitiy of the correlator by solving C(t) / C(t+1) = sinh(m * (t - T/2)) / sinh(m * (t + 1 - T/2)) for m.
            See, e.g., arXiv:1205.5380
        guess : float
            guess for the root finder, only relevant for the root variant
        """
        if self.N != 1:
            raise Exception('Correlator must be projected before getting m_eff')
        if variant == 'log':
            newcontent = []
            for t in range(self.T - 1):
                if (self.content[t] is None) or (self.content[t + 1] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] / self.content[t + 1])
            if(all([x is None for x in newcontent])):
                raise Exception('m_eff is undefined at all timeslices')

            return np.log(Corr(newcontent, padding_back=1))

        elif variant in ['periodic', 'cosh', 'sinh']:
            if variant in ['periodic', 'cosh']:
                func = anp.cosh
            else:
                func = anp.sinh

            def root_function(x, d):
                return func(x * (t - self.T / 2)) / func(x * (t + 1 - self.T / 2)) - d

            newcontent = []
            for t in range(self.T - 1):
                if (self.content[t] is None) or (self.content[t + 1] is None):
                    newcontent.append(None)
                # Fill the two timeslices in the middle of the lattice with their predecessors
                elif variant == 'sinh' and t in [self.T / 2, self.T / 2 - 1]:
                    newcontent.append(newcontent[-1])
                else:
                    newcontent.append(np.abs(find_root(self.content[t][0] / self.content[t + 1][0], root_function, guess=guess)))
            if(all([x is None for x in newcontent])):
                raise Exception('m_eff is undefined at all timeslices')

            return Corr(newcontent, padding_back=1)

        elif variant == 'arccosh':
            newcontent = []
            for t in range(1, self.T - 1):
                if (self.content[t] is None) or (self.content[t + 1] is None) or (self.content[t - 1] is None):
                    newcontent.append(None)
                else:
                    newcontent.append((self.content[t + 1] + self.content[t - 1]) / (2 * self.content[t]))
            if(all([x is None for x in newcontent])):
                raise Exception("m_eff is undefined at all timeslices")
            return np.arccosh(Corr(newcontent, padding_back=1, padding_front=1))

        else:
            raise Exception('Unkown variant.')

    def fit(self, function, fitrange=None, silent=False, **kwargs):
        """Fits function to the data

        Attributes:
        -----------
        function : obj
            function to fit to the data. See fits.least_squares for details.
        fitrange : list
            Range in which the function is to be fitted to the data.
            If not specified, self.prange or all timeslices are used.
        silent : bool
            Decides whether output is printed to the standard output.
        """
        if self.N != 1:
            raise Exception("Correlator must be projected before fitting")

        # The default behaviour is:
        # 1 use explicit fitrange
        # if none is provided, use the range of the corr
        # if this is also not set, use the whole length of the corr (This could come with a warning!)

        if fitrange is None:
            if self.prange:
                fitrange = self.prange
            else:
                fitrange = [0, self.T]

        xs = [x for x in range(fitrange[0], fitrange[1] + 1) if not self.content[x] is None]
        ys = [self.content[x][0] for x in range(fitrange[0], fitrange[1] + 1) if not self.content[x] is None]
        result = least_squares(xs, ys, function, silent=silent, **kwargs)
        result.gamma_method()
        return result

    def plateau(self, plateau_range=None, method="fit"):
        """ Extract a plateu value from a Corr object

        Attributes:
        -----------
        plateau_range : list
            list with two entries, indicating the first and the last timeslice
            of the plateau region.
        method : str
            method to extract the plateau.
                'fit' fits a constant to the plateau region
                'avg', 'average' or 'mean' just average over the given timeslices.
        """
        if not plateau_range:
            if self.prange:
                plateau_range = self.prange
            else:
                raise Exception("no plateau range provided")
        if self.N != 1:
            raise Exception("Correlator must be projected before getting a plateau.")
        if(all([self.content[t] is None for t in range(plateau_range[0], plateau_range[1] + 1)])):
            raise Exception("plateau is undefined at all timeslices in plateaurange.")
        if method == "fit":
            def const_func(a, t):
                return a[0]
            return self.fit(const_func, plateau_range)[0]
        elif method in ["avg", "average", "mean"]:
            returnvalue = np.mean([item[0] for item in self.content[plateau_range[0]:plateau_range[1] + 1] if item is not None])
            returnvalue.gamma_method()
            return returnvalue

        else:
            raise Exception("Unsupported plateau method: " + method)

    def set_prange(self, prange):
        """Sets the attribute prange of the Corr object."""
        if not len(prange) == 2:
            raise Exception("prange must be a list or array with two values")
        if not ((isinstance(prange[0], int)) and (isinstance(prange[1], int))):
            raise Exception("Start and end point must be integers")
        if not (0 <= prange[0] <= self.T and 0 <= prange[1] <= self.T and prange[0] < prange[1]):
            raise Exception("Start and end point must define a range in the interval 0,T")

        self.prange = prange
        return

    def show(self, x_range=None, comp=None, y_range=None, logscale=False, plateau=None, fit_res=None, save=None, ylabel=None):
        """Plots the correlator, uses tag as label if available.

        Parameters
        ----------
        x_range -- list of two values, determining the range of the x-axis e.g. [4, 8]
        comp -- Correlator or list of correlators which are plotted for comparison.
        logscale -- Sets y-axis to logscale
        save -- path to file in which the figure should be saved
        """
        if self.N != 1:
            raise Exception("Correlator must be projected before plotting")
        if x_range is None:
            x_range = [0, self.T]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        x, y, y_err = self.plottable()
        ax1.errorbar(x, y, y_err, label=self.tag)
        if logscale:
            ax1.set_yscale('log')
        else:
            # we generate ylim instead of using autoscaling.
            if y_range is None:
                try:
                    y_min = min([(x[0].value - x[0].dvalue) for x in self.content[x_range[0]: x_range[1] + 1] if (x is not None) and x[0].dvalue < 2 * np.abs(x[0].value)])
                    y_max = max([(x[0].value + x[0].dvalue) for x in self.content[x_range[0]: x_range[1] + 1] if (x is not None) and x[0].dvalue < 2 * np.abs(x[0].value)])
                    ax1.set_ylim([y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min)])
                except:
                    pass
            else:
                ax1.set_ylim(y_range)
        if comp:
            if isinstance(comp, Corr) or isinstance(comp, list):
                for corr in comp if isinstance(comp, list) else [comp]:
                    x, y, y_err = corr.plottable()
                    plt.errorbar(x, y, y_err, label=corr.tag, mfc=plt.rcParams['axes.facecolor'])
            else:
                raise Exception('comp must be a correlator or a list of correlators.')

        if plateau:
            if isinstance(plateau, Obs):
                ax1.axhline(y=plateau.value, linewidth=2, color=plt.rcParams['text.color'], alpha=0.6, marker=',', ls='--', label=str(plateau))
                ax1.axhspan(plateau.value - plateau.dvalue, plateau.value + plateau.dvalue, alpha=0.25, color=plt.rcParams['text.color'], ls='-')
            else:
                raise Exception('plateau must be an Obs')
        if self.prange:
            ax1.axvline(self.prange[0], 0, 1, ls='-', marker=',')
            ax1.axvline(self.prange[1], 0, 1, ls='-', marker=',')

        if fit_res:
            x_samples = np.arange(x_range[0], x_range[1] + 1, 0.05)
            ax1.plot(x_samples,
                     fit_res['fit_function']([o.value for o in fit_res['fit_parameters']], x_samples),
                     ls='-', marker=',', lw=2)

        ax1.set_xlabel(r'$x_0 / a$')
        if ylabel:
            ax1.set_ylabel(ylabel)
        ax1.set_xlim([x_range[0] - 0.5, x_range[1] + 0.5])

        handles, labels = ax1.get_legend_handles_labels()
        if labels:
            ax1.legend()
        plt.draw()

        if save:
            if isinstance(save, str):
                fig.savefig(save)
            else:
                raise Exception("Safe has to be a string.")

        return

    def dump(self, filename):
        """Dumps the Corr into a pickel file

        Attributes:
        -----------
        filename : str
            Name of the file
        """
        dump_object(self, filename)
        return

    def print(self, range=[0, None]):
        print(self.__repr__(range))

    def __repr__(self, range=[0, None]):
        if range[1]:
            range[1] += 1
        content_string = 'x0/a\tCorr(x0/a)\n------------------\n'
        for i, sub_corr in enumerate(self.content[range[0]:range[1]]):
            if sub_corr is None:
                content_string += str(i + range[0]) + '\n'
            else:
                content_string += str(i + range[0])
                for element in sub_corr:
                    content_string += '\t' + ' ' * int(element >= 0) + str(element)
                content_string += '\n'
        return content_string

    def __str__(self):
        return self.__repr__()

    # We define the basic operations, that can be performed with correlators.
    # While */+- get defined here, they only work for Corr*Obs and not Obs*Corr.
    # This is because Obs*Corr checks Obs.__mul__ first and does not catch an exception.
    # One could try and tell Obs to check if the y in __mul__ is a Corr and

    def __add__(self, y):
        if isinstance(y, Corr):
            if ((self.N != y.N) or (self.T != y.T)):
                raise Exception("Addition of Corrs with different shape")
            newcontent = []
            for t in range(self.T):
                if (self.content[t] is None) or (y.content[t] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] + y.content[t])
            return Corr(newcontent)

        elif isinstance(y, Obs) or isinstance(y, int) or isinstance(y, float):
            newcontent = []
            for t in range(self.T):
                if (self.content[t] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] + y)
            return Corr(newcontent, prange=self.prange)
        else:
            raise TypeError("Corr + wrong type")

    def __mul__(self, y):
        if isinstance(y, Corr):
            if not((self.N == 1 or y.N == 1 or self.N == y.N) and self.T == y.T):
                raise Exception("Multiplication of Corr object requires N=N or N=1 and T=T")
            newcontent = []
            for t in range(self.T):
                if (self.content[t] is None) or (y.content[t] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] * y.content[t])
            return Corr(newcontent)

        elif isinstance(y, Obs) or isinstance(y, int) or isinstance(y, float):
            newcontent = []
            for t in range(self.T):
                if (self.content[t] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] * y)
            return Corr(newcontent, prange=self.prange)
        else:
            raise TypeError("Corr * wrong type")

    def __truediv__(self, y):
        if isinstance(y, Corr):
            if not((self.N == 1 or y.N == 1 or self.N == y.N) and self.T == y.T):
                raise Exception("Multiplication of Corr object requires N=N or N=1 and T=T")
            newcontent = []
            for t in range(self.T):
                if (self.content[t] is None) or (y.content[t] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] / y.content[t])
            # Here we set the entire timeslice to undefined, if one of the smearings has encountered an division by zero.
            # While this might throw away perfectly good values in other smearings, we will never have to check, if all values in our matrix are defined
            for t in range(self.T):
                if newcontent[t] is None:
                    continue
                if np.isnan(np.sum(newcontent[t]).value):
                    newcontent[t] = None

            if all([item is None for item in newcontent]):
                raise Exception("Division returns completely undefined correlator")
            return Corr(newcontent)

        elif isinstance(y, Obs):
            if y.value == 0:
                raise Exception('Division by zero will return undefined correlator')
            newcontent = []
            for t in range(self.T):
                if (self.content[t] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] / y)
            return Corr(newcontent, prange=self.prange)

        elif isinstance(y, int) or isinstance(y, float):
            if y == 0:
                raise Exception('Division by zero will return undefined correlator')
            newcontent = []
            for t in range(self.T):
                if (self.content[t] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] / y)
            return Corr(newcontent, prange=self.prange)
        else:
            raise TypeError('Corr / wrong type')

    def __neg__(self):
        newcontent = [None if (item is None) else -1. * item for item in self.content]
        return Corr(newcontent, prange=self.prange)

    def __sub__(self, y):
        return self + (-y)

    def __pow__(self, y):
        if isinstance(y, Obs) or isinstance(y, int) or isinstance(y, float):
            newcontent = [None if (item is None) else item**y for item in self.content]
            return Corr(newcontent, prange=self.prange)
        else:
            raise TypeError('Type of exponent not supported')

    def __abs__(self):
        newcontent = [None if (item is None) else np.abs(item) for item in self.content]
        return Corr(newcontent, prange=self.prange)

    # The numpy functions:
    def sqrt(self):
        return self**0.5

    def log(self):
        newcontent = [None if (item is None) else np.log(item) for item in self.content]
        return Corr(newcontent, prange=self.prange)

    def exp(self):
        newcontent = [None if (item is None) else np.exp(item) for item in self.content]
        return Corr(newcontent, prange=self.prange)

    def _apply_func_to_corr(self, func):
        newcontent = [None if (item is None) else func(item) for item in self.content]
        for t in range(self.T):
            if newcontent[t] is None:
                continue
            if np.isnan(np.sum(newcontent[t]).value):
                newcontent[t] = None
        if all([item is None for item in newcontent]):
            raise Exception('Operation returns undefined correlator')
        return Corr(newcontent)

    def sin(self):
        return self._apply_func_to_corr(np.sin)

    def cos(self):
        return self._apply_func_to_corr(np.cos)

    def tan(self):
        return self._apply_func_to_corr(np.tan)

    def sinh(self):
        return self._apply_func_to_corr(np.sinh)

    def cosh(self):
        return self._apply_func_to_corr(np.cosh)

    def tanh(self):
        return self._apply_func_to_corr(np.tanh)

    def arcsin(self):
        return self._apply_func_to_corr(np.arcsin)

    def arccos(self):
        return self._apply_func_to_corr(np.arccos)

    def arctan(self):
        return self._apply_func_to_corr(np.arctan)

    def arcsinh(self):
        return self._apply_func_to_corr(np.arcsinh)

    def arccosh(self):
        return self._apply_func_to_corr(np.arccosh)

    def arctanh(self):
        return self._apply_func_to_corr(np.arctanh)

    # Right hand side operations (require tweak in main module to work)
    def __radd__(self, y):
        return self + y

    def __rsub__(self, y):
        return -self + y

    def __rmul__(self, y):
        return self * y

    def __rtruediv__(self, y):
        return (self / y) ** (-1)
