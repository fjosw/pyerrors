import pickle
import numpy as np
from .obs import Obs


def dump_object(obj, name, **kwargs):
    """Dump object into pickle file.

    Parameters
    ----------
    obj : object
        object to be saved in the pickle file
    name : str
        name of the file
    path : str
        specifies a custom path for the file (default '.')
    """
    if 'path' in kwargs:
        file_name = kwargs.get('path') + '/' + name + '.p'
    else:
        file_name = name + '.p'
    with open(file_name, 'wb') as fb:
        pickle.dump(obj, fb)


def load_object(path):
    """Load object from pickle file.

    Parameters
    ----------
    path : str
        path to the file
    """
    with open(path, 'rb') as file:
        return pickle.load(file)


def pseudo_Obs(value, dvalue, name, samples=1000):
    """Generate an Obs object with given value, dvalue and name for test purposes

    Parameters
    ----------
    value : float
        central value of the Obs to be generated.
    dvalue : float
        error of the Obs to be generated.
    name : str
        name of the ensemble for which the Obs is to be generated.
    samples: int
        number of samples for the Obs (default 1000).
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


def gen_correlated_data(means, cov, name, tau=0.5, samples=1000):
    """ Generate observables with given covariance and autocorrelation times.

    Parameters
    ----------
    means : list
        list containing the mean value of each observable.
    cov : numpy.ndarray
        covariance matrix for the data to be generated.
    name : str
        ensemble name for the data to be geneated.
    tau : float or list
        can either be a real number or a list with an entry for
        every dataset.
    samples : int
        number of samples to be generated for each observable.
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


def _assert_equal_properties(ol, otype=Obs):
    if not isinstance(ol[0], otype):
        raise Exception("Wrong data type in list.")
    for o in ol[1:]:
        if not isinstance(o, otype):
            raise Exception("Wrong data type in list.")
        if not ol[0].is_merged == o.is_merged:
            raise Exception("All Obs in list have to have the same state 'is_merged'.")
        if not ol[0].reweighted == o.reweighted:
            raise Exception("All Obs in list have to have the same property 'reweighted'.")
        if not ol[0].e_content == o.e_content:
            raise Exception("All Obs in list have to be defined on the same set of configs.")
        if not ol[0].idl == o.idl:
            raise Exception("All Obs in list have to be defined on the same set of configurations.")
