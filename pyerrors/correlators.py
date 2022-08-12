import warnings
from itertools import permutations
import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt
import scipy.linalg
from .obs import Obs, reweight, correlate, CObs
from .misc import dump_object, _assert_equal_properties
from .fits import least_squares
from .roots import find_root


class Corr:
    """The class for a correlator (time dependent sequence of pe.Obs).

    Everything, this class does, can be achieved using lists or arrays of Obs.
    But it is simply more convenient to have a dedicated object for correlators.
    One often wants to add or multiply correlators of the same length at every timeslice and it is inconvenient
    to iterate over all timeslices for every operation. This is especially true, when dealing with matrices.

    The correlator can have two types of content: An Obs at every timeslice OR a GEVP
    matrix at every timeslice. Other dependency (eg. spatial) are not supported.

    """

    def __init__(self, data_input, padding=[0, 0], prange=None):
        """ Initialize a Corr object.

        Parameters
        ----------
        data_input : list or array
            list of Obs or list of arrays of Obs or array of Corrs
        padding : list, optional
            List with two entries where the first labels the padding
            at the front of the correlator and the second the padding
            at the back.
        prange : list, optional
            List containing the first and last timeslice of the plateau
            region indentified for this correlator.
        """

        if isinstance(data_input, np.ndarray):

            # This only works, if the array fulfills the conditions below
            if not len(data_input.shape) == 2 and data_input.shape[0] == data_input.shape[1]:
                raise Exception("Incompatible array shape")
            if not all([isinstance(item, Corr) for item in data_input.flatten()]):
                raise Exception("If the input is an array, its elements must be of type pe.Corr")
            if not all([item.N == 1 for item in data_input.flatten()]):
                raise Exception("Can only construct matrix correlator from single valued correlators")
            if not len(set([item.T for item in data_input.flatten()])) == 1:
                raise Exception("All input Correlators must be defined over the same timeslices.")

            T = data_input[0, 0].T
            N = data_input.shape[0]
            input_as_list = []
            for t in range(T):
                if any([(item.content[t] is None) for item in data_input.flatten()]):
                    if not all([(item.content[t] is None) for item in data_input.flatten()]):
                        warnings.warn("Input ill-defined at different timeslices. Conversion leads to data loss!", RuntimeWarning)
                    input_as_list.append(None)
                else:
                    array_at_timeslace = np.empty([N, N], dtype="object")
                    for i in range(N):
                        for j in range(N):
                            array_at_timeslace[i, j] = data_input[i, j][t]
                    input_as_list.append(array_at_timeslace)
            data_input = input_as_list

        if isinstance(data_input, list):

            if all([isinstance(item, (Obs, CObs)) for item in data_input]):
                _assert_equal_properties(data_input)
                self.content = [np.asarray([item]) for item in data_input]
            if all([isinstance(item, (Obs, CObs)) or item is None for item in data_input]):
                _assert_equal_properties([o for o in data_input if o is not None])
                self.content = [np.asarray([item]) if item is not None else None for item in data_input]
                self.N = 1

            elif all([isinstance(item, np.ndarray) or item is None for item in data_input]) and any([isinstance(item, np.ndarray) for item in data_input]):
                self.content = data_input
                noNull = [a for a in self.content if not (a is None)]  # To check if the matrices are correct for all undefined elements
                self.N = noNull[0].shape[0]
                if self.N > 1 and noNull[0].shape[0] != noNull[0].shape[1]:
                    raise Exception("Smearing matrices are not NxN")
                if (not all([item.shape == noNull[0].shape for item in noNull])):
                    raise Exception("Items in data_input are not of identical shape." + str(noNull))
            else:
                raise Exception("data_input contains item of wrong type")
        else:
            raise Exception("Data input was not given as list or correct array")

        self.tag = None

        # An undefined timeslice is represented by the None object
        self.content = [None] * padding[0] + self.content + [None] * padding[1]
        self.T = len(self.content)
        self.prange = prange

    def __getitem__(self, idx):
        """Return the content of timeslice idx"""
        if self.content[idx] is None:
            return None
        elif len(self.content[idx]) == 1:
            return self.content[idx][0]
        else:
            return self.content[idx]

    @property
    def reweighted(self):
        bool_array = np.array([list(map(lambda x: x.reweighted, o)) for o in [x for x in self.content if x is not None]])
        if np.all(bool_array == 1):
            return True
        elif np.all(bool_array == 0):
            return False
        else:
            raise Exception("Reweighting status of correlator corrupted.")

    def gamma_method(self, **kwargs):
        """Apply the gamma method to the content of the Corr."""
        for item in self.content:
            if not (item is None):
                if self.N == 1:
                    item[0].gamma_method(**kwargs)
                else:
                    for i in range(self.N):
                        for j in range(self.N):
                            item[i, j].gamma_method(**kwargs)

    def projected(self, vector_l=None, vector_r=None, normalize=False):
        """We need to project the Correlator with a Vector to get a single value at each timeslice.

        The method can use one or two vectors.
        If two are specified it returns v1@G@v2 (the order might be very important.)
        By default it will return the lowest source, which usually means unsmeared-unsmeared (0,0), but it does not have to
        """
        if self.N == 1:
            raise Exception("Trying to project a Corr, that already has N=1.")

        if vector_l is None:
            vector_l, vector_r = np.asarray([1.] + (self.N - 1) * [0.]), np.asarray([1.] + (self.N - 1) * [0.])
        elif (vector_r is None):
            vector_r = vector_l
        if isinstance(vector_l, list) and not isinstance(vector_r, list):
            if len(vector_l) != self.T:
                raise Exception("Length of vector list must be equal to T")
            vector_r = [vector_r] * self.T
        if isinstance(vector_r, list) and not isinstance(vector_l, list):
            if len(vector_r) != self.T:
                raise Exception("Length of vector list must be equal to T")
            vector_l = [vector_l] * self.T

        if not isinstance(vector_l, list):
            if not vector_l.shape == vector_r.shape == (self.N,):
                raise Exception("Vectors are of wrong shape!")
            if normalize:
                vector_l, vector_r = vector_l / np.sqrt((vector_l @ vector_l)), vector_r / np.sqrt(vector_r @ vector_r)
            newcontent = [None if _check_for_none(self, item) else np.asarray([vector_l.T @ item @ vector_r]) for item in self.content]

        else:
            # There are no checks here yet. There are so many possible scenarios, where this can go wrong.
            if normalize:
                for t in range(self.T):
                    vector_l[t], vector_r[t] = vector_l[t] / np.sqrt((vector_l[t] @ vector_l[t])), vector_r[t] / np.sqrt(vector_r[t] @ vector_r[t])

            newcontent = [None if (_check_for_none(self, self.content[t]) or vector_l[t] is None or vector_r[t] is None) else np.asarray([vector_l[t].T @ self.content[t] @ vector_r[t]]) for t in range(self.T)]
        return Corr(newcontent)

    def item(self, i, j):
        """Picks the element [i,j] from every matrix and returns a correlator containing one Obs per timeslice.

        Parameters
        ----------
        i : int
            First index to be picked.
        j : int
            Second index to be picked.
        """
        if self.N == 1:
            raise Exception("Trying to pick item from projected Corr")
        newcontent = [None if (item is None) else item[i, j] for item in self.content]
        return Corr(newcontent)

    def plottable(self):
        """Outputs the correlator in a plotable format.

        Outputs three lists containing the timeslice index, the value on each
        timeslice and the error on each timeslice.
        """
        if self.N != 1:
            raise Exception("Can only make Corr[N=1] plottable")
        x_list = [x for x in range(self.T) if not self.content[x] is None]
        y_list = [y[0].value for y in self.content if y is not None]
        y_err_list = [y[0].dvalue for y in self.content if y is not None]

        return x_list, y_list, y_err_list

    def symmetric(self):
        """ Symmetrize the correlator around x0=0."""
        if self.N != 1:
            raise Exception('symmetric cannot be safely applied to multi-dimensional correlators.')
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
        if (all([x is None for x in newcontent])):
            raise Exception("Corr could not be symmetrized: No redundant values")
        return Corr(newcontent, prange=self.prange)

    def anti_symmetric(self):
        """Anti-symmetrize the correlator around x0=0."""
        if self.N != 1:
            raise Exception('anti_symmetric cannot be safely applied to multi-dimensional correlators.')
        if self.T % 2 != 0:
            raise Exception("Can not symmetrize odd T")

        test = 1 * self
        test.gamma_method()
        if not all([o.is_zero_within_error(3) for o in test.content[0]]):
            warnings.warn("Correlator does not seem to be anti-symmetric around x0=0.", RuntimeWarning)

        newcontent = [self.content[0]]
        for t in range(1, self.T):
            if (self.content[t] is None) or (self.content[self.T - t] is None):
                newcontent.append(None)
            else:
                newcontent.append(0.5 * (self.content[t] - self.content[self.T - t]))
        if (all([x is None for x in newcontent])):
            raise Exception("Corr could not be symmetrized: No redundant values")
        return Corr(newcontent, prange=self.prange)

    def is_matrix_symmetric(self):
        """Checks whether a correlator matrices is symmetric on every timeslice."""
        if self.N == 1:
            raise Exception("Only works for correlator matrices.")
        for t in range(self.T):
            if self[t] is None:
                continue
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    if self[t][i, j] is self[t][j, i]:
                        continue
                    if hash(self[t][i, j]) != hash(self[t][j, i]):
                        return False
        return True

    def matrix_symmetric(self):
        """Symmetrizes the correlator matrices on every timeslice."""
        if self.N == 1:
            raise Exception("Trying to symmetrize a correlator matrix, that already has N=1.")
        if self.is_matrix_symmetric():
            return 1.0 * self
        else:
            transposed = [None if _check_for_none(self, G) else G.T for G in self.content]
            return 0.5 * (Corr(transposed) + self)

    def GEVP(self, t0, ts=None, sort="Eigenvalue", **kwargs):
        r'''Solve the generalized eigenvalue problem on the correlator matrix and returns the corresponding eigenvectors.

        The eigenvectors are sorted according to the descending eigenvalues, the zeroth eigenvector(s) correspond to the
        largest eigenvalue(s). The eigenvector(s) for the individual states can be accessed via slicing
        ```python
        C.GEVP(t0=2)[0]  # Ground state vector(s)
        C.GEVP(t0=2)[:3]  # Vectors for the lowest three states
        ```

        Parameters
        ----------
        t0 : int
            The time t0 for the right hand side of the GEVP according to $G(t)v_i=\lambda_i G(t_0)v_i$
        ts : int
            fixed time $G(t_s)v_i=\lambda_i G(t_0)v_i$ if sort=None.
            If sort="Eigenvector" it gives a reference point for the sorting method.
        sort : string
            If this argument is set, a list of self.T vectors per state is returned. If it is set to None, only one vector is returned.
            - "Eigenvalue": The eigenvector is chosen according to which eigenvalue it belongs individually on every timeslice.
            - "Eigenvector": Use the method described in arXiv:2004.10472 to find the set of v(t) belonging to the state.
              The reference state is identified by its eigenvalue at $t=t_s$.

        Other Parameters
        ----------------
        state : int
           Returns only the vector(s) for a specified state. The lowest state is zero.
        '''

        if self.N == 1:
            raise Exception("GEVP methods only works on correlator matrices and not single correlators.")
        if ts is not None:
            if (ts <= t0):
                raise Exception("ts has to be larger than t0.")

        if "sorted_list" in kwargs:
            warnings.warn("Argument 'sorted_list' is deprecated, use 'sort' instead.", DeprecationWarning)
            sort = kwargs.get("sorted_list")

        if self.is_matrix_symmetric():
            symmetric_corr = self
        else:
            symmetric_corr = self.matrix_symmetric()

        if sort is None:
            if (ts is None):
                raise Exception("ts is required if sort=None.")
            if (self.content[t0] is None) or (self.content[ts] is None):
                raise Exception("Corr not defined at t0/ts.")
            G0, Gt = np.empty([self.N, self.N], dtype="double"), np.empty([self.N, self.N], dtype="double")
            for i in range(self.N):
                for j in range(self.N):
                    G0[i, j] = symmetric_corr[t0][i, j].value
                    Gt[i, j] = symmetric_corr[ts][i, j].value

            reordered_vecs = _GEVP_solver(Gt, G0)

        elif sort in ["Eigenvalue", "Eigenvector"]:
            if sort == "Eigenvalue" and ts is not None:
                warnings.warn("ts has no effect when sorting by eigenvalue is chosen.", RuntimeWarning)
            all_vecs = [None] * (t0 + 1)
            for t in range(t0 + 1, self.T):
                try:
                    G0, Gt = np.empty([self.N, self.N], dtype="double"), np.empty([self.N, self.N], dtype="double")
                    for i in range(self.N):
                        for j in range(self.N):
                            G0[i, j] = symmetric_corr[t0][i, j].value
                            Gt[i, j] = symmetric_corr[t][i, j].value

                    all_vecs.append(_GEVP_solver(Gt, G0))
                except Exception:
                    all_vecs.append(None)
            if sort == "Eigenvector":
                if (ts is None):
                    raise Exception("ts is required for the Eigenvector sorting method.")
                all_vecs = _sort_vectors(all_vecs, ts)

            reordered_vecs = [[v[s] if v is not None else None for v in all_vecs] for s in range(self.N)]
        else:
            raise Exception("Unkown value for 'sort'.")

        if "state" in kwargs:
            return reordered_vecs[kwargs.get("state")]
        else:
            return reordered_vecs

    def Eigenvalue(self, t0, ts=None, state=0, sort="Eigenvalue"):
        """Determines the eigenvalue of the GEVP by solving and projecting the correlator

        Parameters
        ----------
        state : int
            The state one is interested in ordered by energy. The lowest state is zero.

        All other parameters are identical to the ones of Corr.GEVP.
        """
        vec = self.GEVP(t0, ts=ts, sort=sort)[state]
        return self.projected(vec)

    def Hankel(self, N, periodic=False):
        """Constructs an NxN Hankel matrix

        C(t) c(t+1) ... c(t+n-1)
        C(t+1) c(t+2) ... c(t+n)
        .................
        C(t+(n-1)) c(t+n) ... c(t+2(n-1))

        Parameters
        ----------
        N : int
            Dimension of the Hankel matrix
        periodic : bool, optional
            determines whether the matrix is extended periodically
        """

        if self.N != 1:
            raise Exception("Multi-operator Prony not implemented!")

        array = np.empty([N, N], dtype="object")
        new_content = []
        for t in range(self.T):
            new_content.append(array.copy())

        def wrap(i):
            while i >= self.T:
                i -= self.T
            return i

        for t in range(self.T):
            for i in range(N):
                for j in range(N):
                    if periodic:
                        new_content[t][i, j] = self.content[wrap(t + i + j)][0]
                    elif (t + i + j) >= self.T:
                        new_content[t] = None
                    else:
                        new_content[t][i, j] = self.content[t + i + j][0]

        return Corr(new_content)

    def roll(self, dt):
        """Periodically shift the correlator by dt timeslices

        Parameters
        ----------
        dt : int
            number of timeslices
        """
        return Corr(list(np.roll(np.array(self.content, dtype=object), dt)))

    def reverse(self):
        """Reverse the time ordering of the Corr"""
        return Corr(self.content[:: -1])

    def thin(self, spacing=2, offset=0):
        """Thin out a correlator to suppress correlations

        Parameters
        ----------
        spacing : int
            Keep only every 'spacing'th entry of the correlator
        offset : int
            Offset the equal spacing
        """
        new_content = []
        for t in range(self.T):
            if (offset + t) % spacing != 0:
                new_content.append(None)
            else:
                new_content.append(self.content[t])
        return Corr(new_content)

    def correlate(self, partner):
        """Correlate the correlator with another correlator or Obs

        Parameters
        ----------
        partner : Obs or Corr
            partner to correlate the correlator with.
            Can either be an Obs which is correlated with all entries of the
            correlator or a Corr of same length.
        """
        if self.N != 1:
            raise Exception("Only one-dimensional correlators can be safely correlated.")
        new_content = []
        for x0, t_slice in enumerate(self.content):
            if _check_for_none(self, t_slice):
                new_content.append(None)
            else:
                if isinstance(partner, Corr):
                    if _check_for_none(partner, partner.content[x0]):
                        new_content.append(None)
                    else:
                        new_content.append(np.array([correlate(o, partner.content[x0][0]) for o in t_slice]))
                elif isinstance(partner, Obs):  # Should this include CObs?
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
        all_configs : bool
            if True, the reweighted observables are normalized by the average of
            the reweighting factor on all configurations in weight.idl and not
            on the configurations in obs[i].idl.
        """
        if self.N != 1:
            raise Exception("Reweighting only implemented for one-dimensional correlators.")
        new_content = []
        for t_slice in self.content:
            if _check_for_none(self, t_slice):
                new_content.append(None)
            else:
                new_content.append(np.array(reweight(weight, t_slice, **kwargs)))
        return Corr(new_content)

    def T_symmetry(self, partner, parity=+1):
        """Return the time symmetry average of the correlator and its partner

        Parameters
        ----------
        partner : Corr
            Time symmetry partner of the Corr
        partity : int
            Parity quantum number of the correlator, can be +1 or -1
        """
        if self.N != 1:
            raise Exception("T_symmetry only implemented for one-dimensional correlators.")
        if not isinstance(partner, Corr):
            raise Exception("T partner has to be a Corr object.")
        if parity not in [+1, -1]:
            raise Exception("Parity has to be +1 or -1.")
        T_partner = parity * partner.reverse()

        t_slices = []
        test = (self - T_partner)
        test.gamma_method()
        for x0, t_slice in enumerate(test.content):
            if t_slice is not None:
                if not t_slice[0].is_zero_within_error(5):
                    t_slices.append(x0)
        if t_slices:
            warnings.warn("T symmetry partners do not agree within 5 sigma on time slices " + str(t_slices) + ".", RuntimeWarning)

        return (self + T_partner) / 2

    def deriv(self, variant="symmetric"):
        """Return the first derivative of the correlator with respect to x0.

        Parameters
        ----------
        variant : str
            decides which definition of the finite differences derivative is used.
            Available choice: symmetric, forward, backward, improved, default: symmetric
        """
        if self.N != 1:
            raise Exception("deriv only implemented for one-dimensional correlators.")
        if variant == "symmetric":
            newcontent = []
            for t in range(1, self.T - 1):
                if (self.content[t - 1] is None) or (self.content[t + 1] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(0.5 * (self.content[t + 1] - self.content[t - 1]))
            if (all([x is None for x in newcontent])):
                raise Exception('Derivative is undefined at all timeslices')
            return Corr(newcontent, padding=[1, 1])
        elif variant == "forward":
            newcontent = []
            for t in range(self.T - 1):
                if (self.content[t] is None) or (self.content[t + 1] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t + 1] - self.content[t])
            if (all([x is None for x in newcontent])):
                raise Exception("Derivative is undefined at all timeslices")
            return Corr(newcontent, padding=[0, 1])
        elif variant == "backward":
            newcontent = []
            for t in range(1, self.T):
                if (self.content[t - 1] is None) or (self.content[t] is None):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] - self.content[t - 1])
            if (all([x is None for x in newcontent])):
                raise Exception("Derivative is undefined at all timeslices")
            return Corr(newcontent, padding=[1, 0])
        elif variant == "improved":
            newcontent = []
            for t in range(2, self.T - 2):
                if (self.content[t - 2] is None) or (self.content[t - 1] is None) or (self.content[t + 1] is None) or (self.content[t + 2] is None):
                    newcontent.append(None)
                else:
                    newcontent.append((1 / 12) * (self.content[t - 2] - 8 * self.content[t - 1] + 8 * self.content[t + 1] - self.content[t + 2]))
            if (all([x is None for x in newcontent])):
                raise Exception('Derivative is undefined at all timeslices')
            return Corr(newcontent, padding=[2, 2])
        else:
            raise Exception("Unknown variant.")

    def second_deriv(self, variant="symmetric"):
        """Return the second derivative of the correlator with respect to x0.

        Parameters
        ----------
        variant : str
            decides which definition of the finite differences derivative is used.
            Available choice: symmetric, improved, default: symmetric
        """
        if self.N != 1:
            raise Exception("second_deriv only implemented for one-dimensional correlators.")
        if variant == "symmetric":
            newcontent = []
            for t in range(1, self.T - 1):
                if (self.content[t - 1] is None) or (self.content[t + 1] is None):
                    newcontent.append(None)
                else:
                    newcontent.append((self.content[t + 1] - 2 * self.content[t] + self.content[t - 1]))
            if (all([x is None for x in newcontent])):
                raise Exception("Derivative is undefined at all timeslices")
            return Corr(newcontent, padding=[1, 1])
        elif variant == "improved":
            newcontent = []
            for t in range(2, self.T - 2):
                if (self.content[t - 2] is None) or (self.content[t - 1] is None) or (self.content[t] is None) or (self.content[t + 1] is None) or (self.content[t + 2] is None):
                    newcontent.append(None)
                else:
                    newcontent.append((1 / 12) * (-self.content[t + 2] + 16 * self.content[t + 1] - 30 * self.content[t] + 16 * self.content[t - 1] - self.content[t - 2]))
            if (all([x is None for x in newcontent])):
                raise Exception("Derivative is undefined at all timeslices")
            return Corr(newcontent, padding=[2, 2])
        else:
            raise Exception("Unknown variant.")

    def m_eff(self, variant='log', guess=1.0):
        """Returns the effective mass of the correlator as correlator object

        Parameters
        ----------
        variant : str
            log : uses the standard effective mass log(C(t) / C(t+1))
            cosh, periodic : Use periodicitiy of the correlator by solving C(t) / C(t+1) = cosh(m * (t - T/2)) / cosh(m * (t + 1 - T/2)) for m.
            sinh : Use anti-periodicitiy of the correlator by solving C(t) / C(t+1) = sinh(m * (t - T/2)) / sinh(m * (t + 1 - T/2)) for m.
            See, e.g., arXiv:1205.5380
            arccosh : Uses the explicit form of the symmetrized correlator (not recommended)
        guess : float
            guess for the root finder, only relevant for the root variant
        """
        if self.N != 1:
            raise Exception('Correlator must be projected before getting m_eff')
        if variant == 'log':
            newcontent = []
            for t in range(self.T - 1):
                if ((self.content[t] is None) or (self.content[t + 1] is None)) or (self.content[t + 1][0].value == 0):
                    newcontent.append(None)
                elif self.content[t][0].value / self.content[t + 1][0].value < 0:
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] / self.content[t + 1])
            if (all([x is None for x in newcontent])):
                raise Exception('m_eff is undefined at all timeslices')

            return np.log(Corr(newcontent, padding=[0, 1]))

        elif variant in ['periodic', 'cosh', 'sinh']:
            if variant in ['periodic', 'cosh']:
                func = anp.cosh
            else:
                func = anp.sinh

            def root_function(x, d):
                return func(x * (t - self.T / 2)) / func(x * (t + 1 - self.T / 2)) - d

            newcontent = []
            for t in range(self.T - 1):
                if (self.content[t] is None) or (self.content[t + 1] is None) or (self.content[t + 1][0].value == 0):
                    newcontent.append(None)
                # Fill the two timeslices in the middle of the lattice with their predecessors
                elif variant == 'sinh' and t in [self.T / 2, self.T / 2 - 1]:
                    newcontent.append(newcontent[-1])
                elif self.content[t][0].value / self.content[t + 1][0].value < 0:
                    newcontent.append(None)
                else:
                    newcontent.append(np.abs(find_root(self.content[t][0] / self.content[t + 1][0], root_function, guess=guess)))
            if (all([x is None for x in newcontent])):
                raise Exception('m_eff is undefined at all timeslices')

            return Corr(newcontent, padding=[0, 1])

        elif variant == 'arccosh':
            newcontent = []
            for t in range(1, self.T - 1):
                if (self.content[t] is None) or (self.content[t + 1] is None) or (self.content[t - 1] is None) or (self.content[t][0].value == 0):
                    newcontent.append(None)
                else:
                    newcontent.append((self.content[t + 1] + self.content[t - 1]) / (2 * self.content[t]))
            if (all([x is None for x in newcontent])):
                raise Exception("m_eff is undefined at all timeslices")
            return np.arccosh(Corr(newcontent, padding=[1, 1]))

        else:
            raise Exception('Unknown variant.')

    def fit(self, function, fitrange=None, silent=False, **kwargs):
        r'''Fits function to the data

        Parameters
        ----------
        function : obj
            function to fit to the data. See fits.least_squares for details.
        fitrange : list
            Two element list containing the timeslices on which the fit is supposed to start and stop.
            Caution: This range is inclusive as opposed to standard python indexing.
            `fitrange=[4, 6]` corresponds to the three entries 4, 5 and 6.
            If not specified, self.prange or all timeslices are used.
        silent : bool
            Decides whether output is printed to the standard output.
        '''
        if self.N != 1:
            raise Exception("Correlator must be projected before fitting")

        if fitrange is None:
            if self.prange:
                fitrange = self.prange
            else:
                fitrange = [0, self.T - 1]
        else:
            if not isinstance(fitrange, list):
                raise Exception("fitrange has to be a list with two elements")
            if len(fitrange) != 2:
                raise Exception("fitrange has to have exactly two elements [fit_start, fit_stop]")

        xs = [x for x in range(fitrange[0], fitrange[1] + 1) if not self.content[x] is None]
        ys = [self.content[x][0] for x in range(fitrange[0], fitrange[1] + 1) if not self.content[x] is None]
        result = least_squares(xs, ys, function, silent=silent, **kwargs)
        return result

    def plateau(self, plateau_range=None, method="fit", auto_gamma=False):
        """ Extract a plateau value from a Corr object

        Parameters
        ----------
        plateau_range : list
            list with two entries, indicating the first and the last timeslice
            of the plateau region.
        method : str
            method to extract the plateau.
                'fit' fits a constant to the plateau region
                'avg', 'average' or 'mean' just average over the given timeslices.
        auto_gamma : bool
            apply gamma_method with default parameters to the Corr. Defaults to None
        """
        if not plateau_range:
            if self.prange:
                plateau_range = self.prange
            else:
                raise Exception("no plateau range provided")
        if self.N != 1:
            raise Exception("Correlator must be projected before getting a plateau.")
        if (all([self.content[t] is None for t in range(plateau_range[0], plateau_range[1] + 1)])):
            raise Exception("plateau is undefined at all timeslices in plateaurange.")
        if auto_gamma:
            self.gamma_method()
        if method == "fit":
            def const_func(a, t):
                return a[0]
            return self.fit(const_func, plateau_range)[0]
        elif method in ["avg", "average", "mean"]:
            returnvalue = np.mean([item[0] for item in self.content[plateau_range[0]:plateau_range[1] + 1] if item is not None])
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

    def show(self, x_range=None, comp=None, y_range=None, logscale=False, plateau=None, fit_res=None, ylabel=None, save=None, auto_gamma=False, hide_sigma=None, references=None, title=None):
        """Plots the correlator using the tag of the correlator as label if available.

        Parameters
        ----------
        x_range : list
            list of two values, determining the range of the x-axis e.g. [4, 8].
        comp : Corr or list of Corr
            Correlator or list of correlators which are plotted for comparison.
            The tags of these correlators are used as labels if available.
        logscale : bool
            Sets y-axis to logscale.
        plateau : Obs
            Plateau value to be visualized in the figure.
        fit_res : Fit_result
            Fit_result object to be visualized.
        ylabel : str
            Label for the y-axis.
        save : str
            path to file in which the figure should be saved.
        auto_gamma : bool
            Apply the gamma method with standard parameters to all correlators and plateau values before plotting.
        hide_sigma : float
            Hides data points from the first value on which is consistent with zero within 'hide_sigma' standard errors.
        references : list
            List of floating point values that are displayed as horizontal lines for reference.
        title : string
            Optional title of the figure.
        """
        if self.N != 1:
            raise Exception("Correlator must be projected before plotting")

        if auto_gamma:
            self.gamma_method()

        if x_range is None:
            x_range = [0, self.T - 1]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        x, y, y_err = self.plottable()
        if hide_sigma:
            hide_from = np.argmax((hide_sigma * np.array(y_err[1:])) > np.abs(y[1:])) - 1
        else:
            hide_from = None
        ax1.errorbar(x[:hide_from], y[:hide_from], y_err[:hide_from], label=self.tag)
        if logscale:
            ax1.set_yscale('log')
        else:
            if y_range is None:
                try:
                    y_min = min([(x[0].value - x[0].dvalue) for x in self.content[x_range[0]: x_range[1] + 1] if (x is not None) and x[0].dvalue < 2 * np.abs(x[0].value)])
                    y_max = max([(x[0].value + x[0].dvalue) for x in self.content[x_range[0]: x_range[1] + 1] if (x is not None) and x[0].dvalue < 2 * np.abs(x[0].value)])
                    ax1.set_ylim([y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min)])
                except Exception:
                    pass
            else:
                ax1.set_ylim(y_range)
        if comp:
            if isinstance(comp, (Corr, list)):
                for corr in comp if isinstance(comp, list) else [comp]:
                    if auto_gamma:
                        corr.gamma_method()
                    x, y, y_err = corr.plottable()
                    if hide_sigma:
                        hide_from = np.argmax((hide_sigma * np.array(y_err[1:])) > np.abs(y[1:])) - 1
                    else:
                        hide_from = None
                    ax1.errorbar(x[:hide_from], y[:hide_from], y_err[:hide_from], label=corr.tag, mfc=plt.rcParams['axes.facecolor'])
            else:
                raise Exception("'comp' must be a correlator or a list of correlators.")

        if plateau:
            if isinstance(plateau, Obs):
                if auto_gamma:
                    plateau.gamma_method()
                ax1.axhline(y=plateau.value, linewidth=2, color=plt.rcParams['text.color'], alpha=0.6, marker=',', ls='--', label=str(plateau))
                ax1.axhspan(plateau.value - plateau.dvalue, plateau.value + plateau.dvalue, alpha=0.25, color=plt.rcParams['text.color'], ls='-')
            else:
                raise Exception("'plateau' must be an Obs")

        if references:
            if isinstance(references, list):
                for ref in references:
                    ax1.axhline(y=ref, linewidth=1, color=plt.rcParams['text.color'], alpha=0.6, marker=',', ls='--')
            else:
                raise Exception("'references' must be a list of floating pint values.")

        if self.prange:
            ax1.axvline(self.prange[0], 0, 1, ls='-', marker=',')
            ax1.axvline(self.prange[1], 0, 1, ls='-', marker=',')

        if fit_res:
            x_samples = np.arange(x_range[0], x_range[1] + 1, 0.05)
            ax1.plot(x_samples,
                     fit_res.fit_function([o.value for o in fit_res.fit_parameters], x_samples),
                     ls='-', marker=',', lw=2)

        ax1.set_xlabel(r'$x_0 / a$')
        if ylabel:
            ax1.set_ylabel(ylabel)
        ax1.set_xlim([x_range[0] - 0.5, x_range[1] + 0.5])

        handles, labels = ax1.get_legend_handles_labels()
        if labels:
            ax1.legend()

        if title:
            plt.title(title)

        plt.draw()

        if save:
            if isinstance(save, str):
                fig.savefig(save, bbox_inches='tight')
            else:
                raise Exception("'save' has to be a string.")

    def spaghetti_plot(self, logscale=True):
        """Produces a spaghetti plot of the correlator suited to monitor exceptional configurations.

        Parameters
        ----------
        logscale : bool
            Determines whether the scale of the y-axis is logarithmic or standard.
        """
        if self.N != 1:
            raise Exception("Correlator needs to be projected first.")

        mc_names = list(set([item for sublist in [o[0].mc_names for o in self.content if o is not None] for item in sublist]))
        x0_vals = [n for (n, o) in zip(np.arange(self.T), self.content) if o is not None]

        for name in mc_names:
            data = np.array([o[0].deltas[name] + o[0].r_values[name] for o in self.content if o is not None]).T

            fig = plt.figure()
            ax = fig.add_subplot(111)
            for dat in data:
                ax.plot(x0_vals, dat, ls='-', marker='')

            if logscale is True:
                ax.set_yscale('log')

            ax.set_xlabel(r'$x_0 / a$')
            plt.title(name)
            plt.draw()

    def dump(self, filename, datatype="json.gz", **kwargs):
        """Dumps the Corr into a file of chosen type
        Parameters
        ----------
        filename : str
            Name of the file to be saved.
        datatype : str
            Format of the exported file. Supported formats include
            "json.gz" and "pickle"
        path : str
            specifies a custom path for the file (default '.')
        """
        if datatype == "json.gz":
            from .input.json import dump_to_json
            if 'path' in kwargs:
                file_name = kwargs.get('path') + '/' + filename
            else:
                file_name = filename
            dump_to_json(self, file_name)
        elif datatype == "pickle":
            dump_object(self, filename, **kwargs)
        else:
            raise Exception("Unknown datatype " + str(datatype))

    def print(self, print_range=None):
        print(self.__repr__(print_range))

    def __repr__(self, print_range=None):
        if print_range is None:
            print_range = [0, None]

        content_string = ""
        content_string += "Corr T=" + str(self.T) + " N=" + str(self.N) + "\n"  # +" filled with"+ str(type(self.content[0][0])) there should be a good solution here

        if self.tag is not None:
            content_string += "Description: " + self.tag + "\n"
        if self.N != 1:
            return content_string

        if print_range[1]:
            print_range[1] += 1
        content_string += 'x0/a\tCorr(x0/a)\n------------------\n'
        for i, sub_corr in enumerate(self.content[print_range[0]:print_range[1]]):
            if sub_corr is None:
                content_string += str(i + print_range[0]) + '\n'
            else:
                content_string += str(i + print_range[0])
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
                if _check_for_none(self, self.content[t]) or _check_for_none(y, y.content[t]):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] + y.content[t])
            return Corr(newcontent)

        elif isinstance(y, (Obs, int, float, CObs)):
            newcontent = []
            for t in range(self.T):
                if _check_for_none(self, self.content[t]):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] + y)
            return Corr(newcontent, prange=self.prange)
        elif isinstance(y, np.ndarray):
            if y.shape == (self.T,):
                return Corr(list((np.array(self.content).T + y).T))
            else:
                raise ValueError("operands could not be broadcast together")
        else:
            raise TypeError("Corr + wrong type")

    def __mul__(self, y):
        if isinstance(y, Corr):
            if not ((self.N == 1 or y.N == 1 or self.N == y.N) and self.T == y.T):
                raise Exception("Multiplication of Corr object requires N=N or N=1 and T=T")
            newcontent = []
            for t in range(self.T):
                if _check_for_none(self, self.content[t]) or _check_for_none(y, y.content[t]):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] * y.content[t])
            return Corr(newcontent)

        elif isinstance(y, (Obs, int, float, CObs)):
            newcontent = []
            for t in range(self.T):
                if _check_for_none(self, self.content[t]):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] * y)
            return Corr(newcontent, prange=self.prange)
        elif isinstance(y, np.ndarray):
            if y.shape == (self.T,):
                return Corr(list((np.array(self.content).T * y).T))
            else:
                raise ValueError("operands could not be broadcast together")
        else:
            raise TypeError("Corr * wrong type")

    def __truediv__(self, y):
        if isinstance(y, Corr):
            if not ((self.N == 1 or y.N == 1 or self.N == y.N) and self.T == y.T):
                raise Exception("Multiplication of Corr object requires N=N or N=1 and T=T")
            newcontent = []
            for t in range(self.T):
                if _check_for_none(self, self.content[t]) or _check_for_none(y, y.content[t]):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] / y.content[t])
            for t in range(self.T):
                if _check_for_none(self, newcontent[t]):
                    continue
                if np.isnan(np.sum(newcontent[t]).value):
                    newcontent[t] = None

            if all([item is None for item in newcontent]):
                raise Exception("Division returns completely undefined correlator")
            return Corr(newcontent)

        elif isinstance(y, (Obs, CObs)):
            if isinstance(y, Obs):
                if y.value == 0:
                    raise Exception('Division by zero will return undefined correlator')
            if isinstance(y, CObs):
                if y.is_zero():
                    raise Exception('Division by zero will return undefined correlator')

            newcontent = []
            for t in range(self.T):
                if _check_for_none(self, self.content[t]):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] / y)
            return Corr(newcontent, prange=self.prange)

        elif isinstance(y, (int, float)):
            if y == 0:
                raise Exception('Division by zero will return undefined correlator')
            newcontent = []
            for t in range(self.T):
                if _check_for_none(self, self.content[t]):
                    newcontent.append(None)
                else:
                    newcontent.append(self.content[t] / y)
            return Corr(newcontent, prange=self.prange)
        elif isinstance(y, np.ndarray):
            if y.shape == (self.T,):
                return Corr(list((np.array(self.content).T / y).T))
            else:
                raise ValueError("operands could not be broadcast together")
        else:
            raise TypeError('Corr / wrong type')

    def __neg__(self):
        newcontent = [None if _check_for_none(self, item) else -1. * item for item in self.content]
        return Corr(newcontent, prange=self.prange)

    def __sub__(self, y):
        return self + (-y)

    def __pow__(self, y):
        if isinstance(y, (Obs, int, float, CObs)):
            newcontent = [None if _check_for_none(self, item) else item**y for item in self.content]
            return Corr(newcontent, prange=self.prange)
        else:
            raise TypeError('Type of exponent not supported')

    def __abs__(self):
        newcontent = [None if _check_for_none(self, item) else np.abs(item) for item in self.content]
        return Corr(newcontent, prange=self.prange)

    # The numpy functions:
    def sqrt(self):
        return self ** 0.5

    def log(self):
        newcontent = [None if _check_for_none(self, item) else np.log(item) for item in self.content]
        return Corr(newcontent, prange=self.prange)

    def exp(self):
        newcontent = [None if _check_for_none(self, item) else np.exp(item) for item in self.content]
        return Corr(newcontent, prange=self.prange)

    def _apply_func_to_corr(self, func):
        newcontent = [None if _check_for_none(self, item) else func(item) for item in self.content]
        for t in range(self.T):
            if _check_for_none(self, newcontent[t]):
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

    @property
    def real(self):
        def return_real(obs_OR_cobs):
            if isinstance(obs_OR_cobs, CObs):
                return obs_OR_cobs.real
            else:
                return obs_OR_cobs

        return self._apply_func_to_corr(return_real)

    @property
    def imag(self):
        def return_imag(obs_OR_cobs):
            if isinstance(obs_OR_cobs, CObs):
                return obs_OR_cobs.imag
            else:
                return obs_OR_cobs * 0  # So it stays the right type

        return self._apply_func_to_corr(return_imag)

    def prune(self, Ntrunc, tproj=3, t0proj=2, basematrix=None):
        r''' Project large correlation matrix to lowest states

        This method can be used to reduce the size of an (N x N) correlation matrix
        to (Ntrunc x Ntrunc) by solving a GEVP at very early times where the noise
        is still small.

        Parameters
        ----------
        Ntrunc: int
            Rank of the target matrix.
        tproj: int
            Time where the eigenvectors are evaluated, corresponds to ts in the GEVP method.
            The default value is 3.
        t0proj: int
            Time where the correlation matrix is inverted. Choosing t0proj=1 is strongly
            discouraged for O(a) improved theories, since the correctness of the procedure
            cannot be granted in this case. The default value is 2.
        basematrix : Corr
            Correlation matrix that is used to determine the eigenvectors of the
            lowest states based on a GEVP. basematrix is taken to be the Corr itself if
            is is not specified.

        Notes
        -----
        We have the basematrix $C(t)$ and the target matrix $G(t)$. We start by solving
        the GEVP $$C(t) v_n(t, t_0) = \lambda_n(t, t_0) C(t_0) v_n(t, t_0)$$ where $t \equiv t_\mathrm{proj}$
        and $t_0 \equiv t_{0, \mathrm{proj}}$. The target matrix is projected onto the subspace of the
        resulting eigenvectors $v_n, n=1,\dots,N_\mathrm{trunc}$ via
        $$G^\prime_{i, j}(t) = (v_i, G(t) v_j)$$. This allows to reduce the size of a large
        correlation matrix and to remove some noise that is added by irrelevant operators.
        This may allow to use the GEVP on $G(t)$ at late times such that the theoretically motivated
        bound $t_0 \leq t/2$ holds, since the condition number of $G(t)$ is decreased, compared to $C(t)$.
        '''

        if self.N == 1:
            raise Exception('Method cannot be applied to one-dimensional correlators.')
        if basematrix is None:
            basematrix = self
        if Ntrunc >= basematrix.N:
            raise Exception('Cannot truncate using Ntrunc <= %d' % (basematrix.N))
        if basematrix.N != self.N:
            raise Exception('basematrix and targetmatrix have to be of the same size.')

        evecs = basematrix.GEVP(t0proj, tproj, sort=None)[:Ntrunc]

        tmpmat = np.empty((Ntrunc, Ntrunc), dtype=object)
        rmat = []
        for t in range(basematrix.T):
            for i in range(Ntrunc):
                for j in range(Ntrunc):
                    tmpmat[i][j] = evecs[i].T @ self[t] @ evecs[j]
            rmat.append(np.copy(tmpmat))

        newcontent = [None if (self.content[t] is None) else rmat[t] for t in range(self.T)]
        return Corr(newcontent)


def _sort_vectors(vec_set, ts):
    """Helper function used to find a set of Eigenvectors consistent over all timeslices"""
    reference_sorting = np.array(vec_set[ts])
    N = reference_sorting.shape[0]
    sorted_vec_set = []
    for t in range(len(vec_set)):
        if vec_set[t] is None:
            sorted_vec_set.append(None)
        elif not t == ts:
            perms = [list(o) for o in permutations([i for i in range(N)], N)]
            best_score = 0
            for perm in perms:
                current_score = 1
                for k in range(N):
                    new_sorting = reference_sorting.copy()
                    new_sorting[perm[k], :] = vec_set[t][k]
                    current_score *= abs(np.linalg.det(new_sorting))
                if current_score > best_score:
                    best_score = current_score
                    best_perm = perm
            sorted_vec_set.append([vec_set[t][k] for k in best_perm])
        else:
            sorted_vec_set.append(vec_set[t])

    return sorted_vec_set


def _check_for_none(corr, entry):
    """Checks if entry for correlator corr is None"""
    return len(list(filter(None, np.asarray(entry).flatten()))) < corr.N ** 2


def _GEVP_solver(Gt, G0):
    """Helper function for solving the GEVP and sorting the eigenvectors.

    The helper function assumes that both provided matrices are symmetric and
    only processes the lower triangular part of both matrices. In case the matrices
    are not symmetric the upper triangular parts are effectively discarded."""
    return scipy.linalg.eigh(Gt, G0, lower=True)[1].T[::-1]
