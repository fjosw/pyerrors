r'''
# What is pyerrors?
`pyerrors` is a python package for error computation and propagation of Markov chain Monte Carlo data.

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
