r'''
# What is pyerrors?
`pyerrors` is a python package for error computation and propagation of Markov chain Monte Carlo data.
It is based on the **gamma method** [arXiv:hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017). Some of its features are:
- **automatic differentiation** as suggested in [arXiv:1809.01289](https://arxiv.org/abs/1809.01289) (partly based on the [autograd](https://github.com/HIPS/autograd) package)
- **treatment of slow modes** in the simulation as suggested in [arXiv:1009.5228](https://arxiv.org/abs/1009.5228)
- coherent **error propagation** for data from **different Markov chains**
- **non-linear fits with x- and y-errors** and exact linear error propagation based on automatic differentiation as introduced in [arXiv:1809.01289](https://arxiv.org/abs/1809.01289)
- **real and complex matrix operations** and their error propagation based on automatic differentiation (Cholesky decomposition, calculation of eigenvalues and eigenvectors, singular value decomposition...)

## Getting started

```python
import numpy as np
import pyerrors as pe

my_obs = pe.Obs([samples], ['ensemble_name'])
my_new_obs = 2 * np.log(my_obs) / my_obs ** 2
my_new_obs.gamma_method()
print(my_new_obs)
> 0.31498(72)

iamzero = my_new_obs - my_new_obs
iamzero.gamma_method()
print(iamzero == 0.0)
> True
```

# The `Obs` class

`pyerrors` introduces a new datatype, `Obs`, which simplifies error propagation and estimation for auto- and cross-correlated data.
An `Obs` object can be initialized with two arguments, the first is a list containing the samples for an Observable from a Monte Carlo chain.
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

The error estimation within `pyerrors` is based on the gamma method introduced in [arXiv:hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017).
After having arrived at the derived quantity of interest the `gamma_method` can be called as detailed in the following example.

Example:
```python
my_sum.gamma_method()
print(my_sum)
> 1.70(57)
my_sum.details()
> Result	 1.70000000e+00 +/- 5.72046658e-01 +/- 7.56746598e-02 (33.650%)
>  t_int	 2.71422900e+00 +/- 6.40320983e-01 S = 2.00
> 1000 samples in 1 ensemble:
>   · Ensemble 'ensemble_name' : 1000 configurations (from 1 to 1000)

```

We use the following definition of the integrated autocorrelation time established in [Madras & Sokal 1988](https://link.springer.com/article/10.1007/BF01022990)
$$\tau_\mathrm{int}=\frac{1}{2}+\sum_{t=1}^{W}\rho(t)\geq \frac{1}{2}$$
The window $W$ is determined via the automatic windowing procedure described in [arXiv:hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017)
The standard value for the parameter $S$ of this automatic windowing procedure is $S=2$. Other values for $S$ can be passed to the `gamma_method` as parameter.

Example:
```python
my_sum.gamma_method(S=3.0)
my_sum.details()
> Result	 1.70000000e+00 +/- 6.30675201e-01 +/- 1.04585650e-01 (37.099%)
>  t_int	 3.29909703e+00 +/- 9.77310102e-01 S = 3.00
> 1000 samples in 1 ensemble:
>   · Ensemble 'ensemble_name' : 1000 configurations (from 1 to 1000)

```

The integrated autocorrelation time $\tau_\mathrm{int}$ and the autocorrelation function $\rho(W)$ can be monitored via the methods `pyerrors.obs.Obs.plot_tauint` and `pyerrors.obs.Obs.plot_tauint`.

Example:
```python
my_sum.plot_tauint()
my_sum.plot_rho()
```

### Exponential tails

Slow modes in the Monte Carlo history can be accounted for by attaching an exponential tail to the autocorrelation function $\rho$ as suggested in [arXiv:1009.5228](https://arxiv.org/abs/1009.5228). The longest autocorrelation time in the history, $\tau_\mathrm{exp}$, can be passed to the `gamma_method` as parameter. In this case the automatic windowing procedure is vacated and the parameter $S$ does not affect the error estimate.

Example:
```python
my_sum.gamma_method(tau_exp=7.2)
my_sum.details()
> Result	 1.70000000e+00 +/- 6.28097762e-01 +/- 5.79077524e-02 (36.947%)
>  t_int	 3.27218667e+00 +/- 7.99583654e-01 tau_exp = 7.20,  N_sigma = 1
> 1000 samples in 1 ensemble:
>   · Ensemble 'ensemble_name' : 1000 configurations (from 1 to 1000)
```

For the full API see `pyerrors.obs.Obs.gamma_method`

## Multiple ensembles/replica

Error propagation for multiple ensembles (Markov chains with different simulation parameters) is handled automatically. Ensembles are uniquely identified by their `name`.

Example:
```python
obs1 = pe.Obs([samples1], ['ensemble1'])
obs2 = pe.Obs([samples2], ['ensemble2'])

my_sum = obs1 + obs2
my_sum.details()
> Result   2.00697958e+00 +/- 0.00000000e+00 +/- 0.00000000e+00 (0.000%)
> 1500 samples in 2 ensembles:
>   · Ensemble 'ensemble1' : 1000 configurations (from 1 to 1000)
>   · Ensemble 'ensemble2' : 500 configurations (from 1 to 500)
```

`pyerrors` identifies multiple replica (independent Markov chains with identical simulation parameters) by the vertical bar `|` in the name of the data set.

Example:
```python
obs1 = pe.Obs([samples1], ['ensemble1|r01'])
obs2 = pe.Obs([samples2], ['ensemble1|r02'])

> my_sum = obs1 + obs2
> my_sum.details()
> Result   2.00697958e+00 +/- 0.00000000e+00 +/- 0.00000000e+00 (0.000%)
> 1500 samples in 1 ensemble:
>   · Ensemble 'ensemble1'
>     · Replicum 'r01' : 1000 configurations (from 1 to 1000)
>     · Replicum 'r02' : 500 configurations (from 1 to 500)
```

### Error estimation for multiple ensembles

In order to keep track of different error analysis parameters for different ensembles one can make use of global dictionaries as detailed in the following example.

Example:
```python
pe.Obs.S_dict['ensemble1'] = 2.5
pe.Obs.tau_exp_dict['ensemble2'] = 8.0
pe.Obs.tau_exp_dict['ensemble3'] = 2.0
```

In case the `gamma_method` is called without any parameters it will use the values specified in the dictionaries for the respective ensembles.
Passing arguments to the `gamma_method` still dominates over the dictionaries.


## Irregular Monte Carlo chains

Irregular Monte Carlo chains can be initialized with the parameter `idl`.

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
Make sure to check the autocorrelation time with e.g. `pyerrors.obs.Obs.plot_rho` or `pyerrors.obs.Obs.plot_tauint`.

For the full API see `pyerrors.obs.Obs`

# Correlators
For the full API see `pyerrors.correlators.Corr`

# Complex observables

`pyerrors` can handle complex valued observables via the class `pyerrors.obs.CObs`.
`CObs` are initialized with a real and an imaginary part which both can be `Obs` valued.

Example:
```python
my_real_part = pe.Obs([samples1], ['ensemble1'])
my_imag_part = pe.Obs([samples2], ['ensemble1'])

my_cobs = pe.CObs(my_real_part, my_imag_part)
my_cobs.gamma_method()
print(my_cobs)
> (0.9959(91)+0.659(28)j)
```

Elementary mathematical operations are overloaded and samples are properly propagated as for the `Obs` class.
```python
my_derived_cobs = (my_cobs + my_cobs.conjugate()) / np.abs(my_cobs)
my_derived_cobs.gamma_method()
print(my_derived_cobs)
> (1.668(23)+0.0j)
```

# Optimization / fits / roots
`pyerrors.fits`
`pyerrors.roots`

# Matrix operations
`pyerrors.linalg`

# Export data
The preferred exported file format within `pyerrors` is

## Jackknife samples
For comparison with other analysis workflows `pyerrors` can generate jackknife samples from an `Obs` object.
See `pyerrors.obs.Obs.export_jackknife` for details.

# Input
`pyerrors.input`
'''
from .obs import *
from .correlators import *
from .fits import *
from . import dirac
from . import input
from . import linalg
from . import misc
from . import mpm
from . import npr
from . import roots

from .version import __version__
