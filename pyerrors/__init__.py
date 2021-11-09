r'''
# What is pyerrors?
`pyerrors` is a python package for error computation and propagation of Markov chain Monte Carlo data.
It is based on the **gamma method** [arXiv:hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017). Some of its features are:
- **automatic differentiation** as suggested in [arXiv:1809.01289](https://arxiv.org/abs/1809.01289) (partly based on the [autograd](https://github.com/HIPS/autograd) package)
- **treatment of slow modes** in the simulation as suggested in [arXiv:1009.5228](https://arxiv.org/abs/1009.5228)
- coherent **error propagation** for data from **different Markov chains**
- **non-linear fits with x- and y-errors** and exact linear error propagation based on automatic differentiation as introduced in [arXiv:1809.01289]
- **real and complex matrix operations** and their error propagation based on automatic differentiation (cholesky decomposition, calculation of eigenvalues and eigenvectors, singular value decomposition...)

## Getting started

```python
import numpy as np
import pyerrors as pe

my_obs = pe.Obs([samples], ['ensemble_name'])
my_new_obs = 2 * np.log(my_obs) / my_obs
my_new_obs.gamma_method()
my_new_obs.details()
print(my_new_obs)
```
# The `Obs` class

`pyerrors` introduces a new datatype, `Obs`, which simplifies error propagation and estimation for auto- and cross-correlated data.
An `Obs` object can be initialized with two arguments, the first is a list containining the samples for an Observable from a Monte Carlo chain.
The samples can either be provided as python list or as numpy array.
The second argument is a list containing the names of the respective Monte Carlo chains as strings. These strings uniquely identify a Monte Carlo chain/ensemble.

Example:
```python
import pyerrors as pe

my_obs = pe.Obs([samples], ['ensemble_name'])
```

## Error propagation

When performing mathematical operations on `Obs` objects the correct error propagation is intrinsically taken care using a first order Taylor expansion
$$\delta_f^i=\sum_\alpha \bar{f}_\alpha \delta_\alpha^i\,,\quad \delta_\alpha^i=a_\alpha^i-\bar{a}_\alpha$$
as introduced in [arXiv:hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017).

The required derivatives $\bar{f}_\alpha$ are evaluated up to machine precision via automatic differentiation as suggested in [arXiv:1809.01289](https://arxiv.org/abs/1809.01289).

The `Obs` class is designed such that mathematical numpy functions can be used on `Obs` just as for regular floats.

Example:
```python
import numpy as np
import pyerrors as pe

my_obs1 = pe.Obs([samples1], ['ensemble_name'])
my_obs2 = pe.Obs([samples2], ['ensemble_name'])

my_sum = my_obs1 + my_obs2

my_m_eff = np.log(my_obs1 / my_obs2)
```

## Error estimation

The error propagation is based on the gamma method introduced in [arXiv:hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017).


For the full API see `pyerrors.obs.Obs.gamma_method`
### Exponential tails

## Multiple ensembles/replica

Error propagation for multiple ensembles (Markov chains with different simulation parameters) is handeled automatically. Ensembles are uniquely identified by their `name`.

Example:
```python
obs1 = pe.Obs([samples1], ['ensemble1'])
obs2 = pe.Obs([samples2], ['ensemble2'])

my_sum = obs1 + obs2
my_sum.details()
> Result	 2.00596631e+00 +/- 0.00000000e+00 +/- 0.00000000e+00 (0.000%)
> 1500 samples in 2 ensembles:
>    ensemble1: ['ensemble1']
>    ensemble2: ['ensemble2']

```

`pyerrors` identifies multiple replica (independent Markov chains with identical simulation parameters) by the vertical bar `|` in the name of the dataset.

Example:
```python
obs1 = pe.Obs([samples1], ['ensemble1|r01'])
obs2 = pe.Obs([samples2], ['ensemble1|r02'])

my_sum = obs1 + obs2
my_sum.details()
> Result	 2.00596631e+00 +/- 0.00000000e+00 +/- 0.00000000e+00 (0.000%)
> 1500 samples in 1 ensemble:
>    ensemble1: ['ensemble1|r01', 'ensemble1|r02']
```
## Irregular Monte Carlo chains

Irregular Monte Carlo chains can be initilized with the parameter `idl`.

Example:
```python
# Observable defined on configurations 20 to 519
obs1 = pe.Obs([samples1], ['ensemble1'], idl=[range(20, 520)])

# Observable defined on every second configuration between 5 and 1003
obs2 = pe.Obs([samples2], ['ensemble1'], idl=[range(5, 1005, 2)])

# Observable defined on configurations 2, 9, 28, 29 and 501
obs3 = pe.Obs([samples3], ['ensemble1'], idl=[[2, 9, 28, 29, 501]])
```

**Warning:** Irregular Monte Carlo chains can result in odd patterns in the autocorrelation functions.
Make sure to check the with e.g. `pyerrors.obs.Obs.plot_rho` or `pyerrors.obs.Obs.plot_tauint`.

For the full API see `pyerrors.obs.Obs`

# Correlators
For the full API see `pyerrors.correlators.Corr`

# Complex observables
`pyerrors.obs.CObs`

# Optimization / fits / roots
`pyerrors.fits`
`pyerrors.roots`

# Matrix operations
`pyerrors.linalg`

# Input
`pyerrors.input`
'''
from .obs import *
from .correlators import *
from .fits import *
from . import dirac
from . import linalg
from . import misc
from . import mpm
from . import npr
from . import roots

from .version import __version__
