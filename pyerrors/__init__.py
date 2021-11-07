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

## Irregular Monte Carlo chains

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

# Optimization / fits / roots

# Complex observables

# Matrix operations

# Input
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
