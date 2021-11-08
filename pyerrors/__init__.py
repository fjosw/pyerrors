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
`pyerrors.obs.Obs`
```python
import pyerrors as pe

my_obs = pe.Obs([samples], ['ensemble_name'])
```

## Multiple ensembles/replica

Error propagation for multiple ensembles (Markov chains with different simulation parameters) is handeled automatically. Ensembles are uniquely identified by their `name`.

Example:
```python
obs1 = pe.Obs([samples1], ['ensemble1'])
obs2 = pe.Obs([samples1], ['ensemble2'])

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
obs2 = pe.Obs([samples1], ['ensemble1|r02'])

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

# Error propagation
Automatic differentiation, cite Alberto,

numpy overloaded
```python
import numpy as np
import pyerrors as pe

my_obs = pe.Obs([samples], ['ensemble_name'])
my_new_obs = 2 * np.log(my_obs) / my_obs
my_new_obs.gamma_method()
my_new_obs.details()
```

# Error estimation
`pyerrors.obs.Obs.gamma_method`

$\delta_i\delta_j$

## Exponential tails

## Covariance

# Correlators
`pyerrors.correlators.Corr`

# Optimization / fits / roots
`pyerrors.fits`
`pyerrors.roots`


# Complex observables
`pyerrors.obs.CObs`

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
