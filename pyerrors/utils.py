import numpy as np
from .obs import Obs, _reduce_deltas


def remove_corrupted(old_obs, threshold=5.0, verbose=True, exception_at=None):
    """Identifies and removes configurations, that might be corrupted. For this we assume the median to be the expected center of the distribution.

    Args:
        old_obs (Obs): The Obs that might contain faulty data
        threshold (float, optional): The threshold for a confoguration to count as faulty, given in standard deviations. Defaults to 5.0.
        verbose (bool, optional): raises warnings, if a configuration was removed. Defaults to True.
        exception_at (_type_, optional): If not None: Number of configurations, which might be removed without an Exception. Defaults to None.


    Returns:
        Obs: Defined on equal or fewer configurations.
    """

    new_idl, new_values = [], []
    for name in old_obs.names:
        median = np.median(old_obs.deltas[name])
        N = len(old_obs.deltas[name])
        sigma = np.sqrt((1. / N) * np.sum([(delta - median)**2 for delta in old_obs.deltas[name]]))
        reduced_idx = list(np.array(old_obs.idl[name])[abs(old_obs.deltas[name] - median) < threshold * sigma])
        if verbose and len(reduced_idx) < len(old_obs.idl[name]):
            print("WARNING excluded configs:", set(old_obs.idl[name]) - set(reduced_idx))
        new_values.append([d + old_obs.r_values[name] for d in _reduce_deltas(old_obs.deltas[name], old_obs.idl[name], reduced_idx)])
        new_idl.append(reduced_idx)
    new_obs = Obs(new_values, old_obs.names, idl=new_idl)
    if exception_at is not None:
        if np.sum([len(old_obs.deltas[n]) for n in old_obs.names]) - np.sum([len(new_obs.deltas[n]) for n in new_obs.names]) >= exception_at:
            raise Exception("Too many outliers removed")
    return new_obs
