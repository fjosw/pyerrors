import numpy as np


class Covobs:

    def __init__(self, mean, cov, name, pos=None, grad=None):
        """ Initialize Covobs object.

        Parameters
        ----------
        mean : float
            Mean value of the new Obs
        cov : list or array
            2d Covariance matrix or 1d diagonal entries
        name : str
            identifier for the covariance matrix
        pos : int
            Position of the variance belonging to mean in cov.
            Is taken to be 1 if cov is 0-dimensional
        grad : list or array
            Gradient of the Covobs wrt. the means belonging to cov.
        """
        self._set_cov(cov)
        if '|' in name:
            raise Exception("Covobs name must not contain replica separator '|'.")
        self.name = name
        if grad is None:
            if pos is None:
                if self.N == 1:
                    pos = 0
                else:
                    raise Exception('Have to specify position of cov-element belonging to mean!')
            else:
                if pos > self.N:
                    raise Exception('pos %d too large for covariance matrix with dimension %dx%d!' % (pos, self.N, self.N))
            self._grad = np.zeros((self.N, 1))
            self._grad[pos] = 1.
        else:
            self._set_grad(grad)
        self.value = mean

    def errsq(self):
        """ Return the variance (= square of the error) of the Covobs
        """
        return float(np.dot(np.transpose(self.grad), np.dot(self.cov, self.grad)))

    def _set_cov(self, cov):
        """ Set the covariance matrix of the covobs

        Parameters
        ----------
        cov : list or array
            Has to be either of:
            0 dimensional number: variance of a single covobs,
            1 dimensional list or array of lenght N: variances of multiple covobs
            2 dimensional list or array (N x N): Symmetric, positive-semidefinite covariance matrix
        """
        self._cov = np.array(cov)
        if self._cov.ndim == 0:
            self.N = 1
            self._cov = np.diag([self._cov])
        elif self._cov.ndim == 1:
            self.N = len(self._cov)
            self._cov = np.diag(self._cov)
        elif self._cov.ndim == 2:
            self.N = self._cov.shape[0]
            if self._cov.shape[1] != self.N:
                raise Exception('Covariance matrix has to be a square matrix!')
        else:
            raise Exception('Covariance matrix has to be a 2 dimensional square matrix!')

        for i in range(self.N):
            for j in range(i):
                if not self._cov[i][j] == self._cov[j][i]:
                    raise Exception('Covariance matrix is non-symmetric for (%d, %d' % (i, j))

        evals = np.linalg.eigvalsh(self._cov)
        for ev in evals:
            if ev < 0:
                raise Exception('Covariance matrix is not positive-semidefinite!')

    def _set_grad(self, grad):
        """ Set the gradient of the covobs

        Parameters
        ----------
        grad : list or array
            Has to be either of:
            0 dimensional number: gradient w.r.t. a single covobs,
            1 dimensional list or array of lenght N: gradient w.r.t. multiple covobs
        """
        self._grad = np.array(grad)
        if self._grad.ndim in [0, 1]:
            self._grad = np.reshape(self._grad, (self.N, 1))
        elif self._grad.ndim != 2:
            raise Exception('Invalid dimension of grad!')

    @property
    def cov(self):
        return self._cov

    @property
    def grad(self):
        return self._grad
