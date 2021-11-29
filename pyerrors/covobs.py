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
        self.cov = np.array(cov)
        if self.cov.ndim == 0:
            self.N = 1
        elif self.cov.ndim == 1:
            self.N = len(self.cov)
            self.cov = np.diag(self.cov)
        elif self.cov.ndim == 2:
            self.N = self.cov.shape[0]
            if self.cov.shape[1] != self.N:
                raise Exception('Covariance matrix has to be a square matrix!')
        else:
            raise Exception('Covariance matrix has to be a 2 dimensional square matrix!')
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
            self.grad = np.zeros((self.N, 1))
            self.grad[pos] = 1.
        else:
            self.grad = np.array(grad)
        self.value = mean

    def errsq(self):
        """ Return the variance (= square of the error) of the Covobs
        """
        return float(np.dot(np.transpose(self.grad), np.dot(self.cov, self.grad)))
