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

    def _set_grad(self, grad):
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
