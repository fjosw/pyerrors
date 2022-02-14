r'''
# What is pyerrors?
`pyerrors` is a python package for error computation and propagation of Markov chain Monte Carlo data.
It is based on the gamma method [arXiv:hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017). Some of its features are:
- automatic differentiation for exact liner error propagation as suggested in [arXiv:1809.01289](https://arxiv.org/abs/1809.01289) (partly based on the [autograd](https://github.com/HIPS/autograd) package).
- treatment of slow modes in the simulation as suggested in [arXiv:1009.5228](https://arxiv.org/abs/1009.5228).
- coherent error propagation for data from different Markov chains.
- non-linear fits with x- and y-errors and exact linear error propagation based on automatic differentiation as introduced in [arXiv:1809.01289](https://arxiv.org/abs/1809.01289).
- real and complex matrix operations and their error propagation based on automatic differentiation (Matrix inverse, Cholesky decomposition, calculation of eigenvalues and eigenvectors, singular value decomposition...).

There exist similar publicly available implementations of gamma method error analysis suites in [Fortran](https://gitlab.ift.uam-csic.es/alberto/aderrors), [Julia](https://gitlab.ift.uam-csic.es/alberto/aderrors.jl) and [Python](https://github.com/mbruno46/pyobs).

## Basic example

```python
import numpy as np
import pyerrors as pe

my_obs = pe.Obs([samples], ['ensemble_name']) # Initialize an Obs object
my_new_obs = 2 * np.log(my_obs) / my_obs ** 2 # Construct derived Obs object
my_new_obs.gamma_method()                     # Estimate the statistical error
print(my_new_obs)                             # Print the result to stdout
> 0.31498(72)
```

# The `Obs` class

`pyerrors` introduces a new datatype, `Obs`, which simplifies error propagation and estimation for auto- and cross-correlated data.
An `Obs` object can be initialized with two arguments, the first is a list containing the samples for an Observable from a Monte Carlo chain.
The samples can either be provided as python list or as numpy array.
The second argument is a list containing the names of the respective Monte Carlo chains as strings. These strings uniquely identify a Monte Carlo chain/ensemble.

```python
import pyerrors as pe

my_obs = pe.Obs([samples], ['ensemble_name'])
```

## Error propagation

When performing mathematical operations on `Obs` objects the correct error propagation is intrinsically taken care using a first order Taylor expansion
$$\delta_f^i=\sum_\alpha \bar{f}_\alpha \delta_\alpha^i\,,\quad \delta_\alpha^i=a_\alpha^i-\bar{a}_\alpha\,,$$
as introduced in [arXiv:hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017).
The required derivatives $\bar{f}_\alpha$ are evaluated up to machine precision via automatic differentiation as suggested in [arXiv:1809.01289](https://arxiv.org/abs/1809.01289).

The `Obs` class is designed such that mathematical numpy functions can be used on `Obs` just as for regular floats.

```python
import numpy as np
import pyerrors as pe

my_obs1 = pe.Obs([samples1], ['ensemble_name'])
my_obs2 = pe.Obs([samples2], ['ensemble_name'])

my_sum = my_obs1 + my_obs2

my_m_eff = np.log(my_obs1 / my_obs2)

iamzero = my_m_eff - my_m_eff
# Check that value and fluctuations are zero within machine precision
print(iamzero == 0.0)
> True
```

## Error estimation

The error estimation within `pyerrors` is based on the gamma method introduced in [arXiv:hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017).
After having arrived at the derived quantity of interest the `gamma_method` can be called as detailed in the following example.

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
$$\tau_\mathrm{int}=\frac{1}{2}+\sum_{t=1}^{W}\rho(t)\geq \frac{1}{2}\,.$$
The window $W$ is determined via the automatic windowing procedure described in [arXiv:hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017).
The standard value for the parameter $S$ of this automatic windowing procedure is $S=2$. Other values for $S$ can be passed to the `gamma_method` as parameter.

```python
my_sum.gamma_method(S=3.0)
my_sum.details()
> Result	 1.70000000e+00 +/- 6.30675201e-01 +/- 1.04585650e-01 (37.099%)
>  t_int	 3.29909703e+00 +/- 9.77310102e-01 S = 3.00
> 1000 samples in 1 ensemble:
>   · Ensemble 'ensemble_name' : 1000 configurations (from 1 to 1000)

```

The integrated autocorrelation time $\tau_\mathrm{int}$ and the autocorrelation function $\rho(W)$ can be monitored via the methods `pyerrors.obs.Obs.plot_tauint` and `pyerrors.obs.Obs.plot_tauint`.

If the parameter $S$ is set to zero it is assumed that dataset does not exhibit any autocorrelation and the windowsize is chosen to be zero.
In this case the error estimate is identical to the sample standard error.

### Exponential tails

Slow modes in the Monte Carlo history can be accounted for by attaching an exponential tail to the autocorrelation function $\rho$ as suggested in [arXiv:1009.5228](https://arxiv.org/abs/1009.5228). The longest autocorrelation time in the history, $\tau_\mathrm{exp}$, can be passed to the `gamma_method` as parameter. In this case the automatic windowing procedure is vacated and the parameter $S$ does not affect the error estimate.

```python
my_sum.gamma_method(tau_exp=7.2)
my_sum.details()
> Result	 1.70000000e+00 +/- 6.28097762e-01 +/- 5.79077524e-02 (36.947%)
>  t_int	 3.27218667e+00 +/- 7.99583654e-01 tau_exp = 7.20,  N_sigma = 1
> 1000 samples in 1 ensemble:
>   · Ensemble 'ensemble_name' : 1000 configurations (from 1 to 1000)
```

For the full API see `pyerrors.obs.Obs.gamma_method`.

## Multiple ensembles/replica

Error propagation for multiple ensembles (Markov chains with different simulation parameters) is handled automatically. Ensembles are uniquely identified by their `name`.

```python
obs1 = pe.Obs([samples1], ['ensemble1'])
obs2 = pe.Obs([samples2], ['ensemble2'])

my_sum = obs1 + obs2
my_sum.details()
> Result   2.00697958e+00
> 1500 samples in 2 ensembles:
>   · Ensemble 'ensemble1' : 1000 configurations (from 1 to 1000)
>   · Ensemble 'ensemble2' : 500 configurations (from 1 to 500)
```

`pyerrors` identifies multiple replica (independent Markov chains with identical simulation parameters) by the vertical bar `|` in the name of the data set.

```python
obs1 = pe.Obs([samples1], ['ensemble1|r01'])
obs2 = pe.Obs([samples2], ['ensemble1|r02'])

> my_sum = obs1 + obs2
> my_sum.details()
> Result   2.00697958e+00
> 1500 samples in 1 ensemble:
>   · Ensemble 'ensemble1'
>     · Replicum 'r01' : 1000 configurations (from 1 to 1000)
>     · Replicum 'r02' : 500 configurations (from 1 to 500)
```

### Error estimation for multiple ensembles

In order to keep track of different error analysis parameters for different ensembles one can make use of global dictionaries as detailed in the following example.

```python
pe.Obs.S_dict['ensemble1'] = 2.5
pe.Obs.tau_exp_dict['ensemble2'] = 8.0
pe.Obs.tau_exp_dict['ensemble3'] = 2.0
```

In case the `gamma_method` is called without any parameters it will use the values specified in the dictionaries for the respective ensembles.
Passing arguments to the `gamma_method` still dominates over the dictionaries.


## Irregular Monte Carlo chains

`Obs` objects defined on irregular Monte Carlo chains can be initialized with the parameter `idl`.

```python
# Observable defined on configurations 20 to 519
obs1 = pe.Obs([samples1], ['ensemble1'], idl=[range(20, 520)])
obs1.details()
> Result	 9.98319881e-01
> 500 samples in 1 ensemble:
>   · Ensemble 'ensemble1' : 500 configurations (from 20 to 519)

# Observable defined on every second configuration between 5 and 1003
obs2 = pe.Obs([samples2], ['ensemble1'], idl=[range(5, 1005, 2)])
obs2.details()
> Result	 9.99100712e-01
> 500 samples in 1 ensemble:
>   · Ensemble 'ensemble1' : 500 configurations (from 5 to 1003 in steps of 2)

# Observable defined on configurations 2, 9, 28, 29 and 501
obs3 = pe.Obs([samples3], ['ensemble1'], idl=[[2, 9, 28, 29, 501]])
obs3.details()
> Result	 1.01718064e+00
> 5 samples in 1 ensemble:
>   · Ensemble 'ensemble1' : 5 configurations (irregular range)

```

`Obs` objects defined on regular and irregular histories of the same ensemble can be computed with each other and the correct error propagation and estimation is automatically taken care of.

**Warning:** Irregular Monte Carlo chains can result in odd patterns in the autocorrelation functions.
Make sure to check the autocorrelation time with e.g. `pyerrors.obs.Obs.plot_rho` or `pyerrors.obs.Obs.plot_tauint`.

For the full API see `pyerrors.obs.Obs`.

# Correlators
When one is not interested in single observables but correlation functions, `pyerrors` offers the `Corr` class which simplifies the corresponding error propagation and provides the user with a set of standard methods. In order to initialize a `Corr` objects one needs to arrange the data as a list of `Obs´
```python
my_corr = pe.Corr([obs_0, obs_1, obs_2, obs_3])
print(my_corr)
> x0/a	Corr(x0/a)
> ------------------
> 0	 0.7957(80)
> 1	 0.5156(51)
> 2	 0.3227(33)
> 3	 0.2041(21)
```
In case the correlation functions are not defined on the outermost timeslices, for example because of fixed boundary conditions, a padding can be introduced.
```python
my_corr = pe.Corr([obs_0, obs_1, obs_2, obs_3], padding=[1, 1])
print(my_corr)
> x0/a	Corr(x0/a)
> ------------------
> 0
> 1	 0.7957(80)
> 2	 0.5156(51)
> 3	 0.3227(33)
> 4	 0.2041(21)
> 5
```
The individual entries of a correlator can be accessed via slicing
```python
print(my_corr[3])
> 0.3227(33)
```
Error propagation with the `Corr` class works very similar to `Obs` objects. Mathematical operations are overloaded and `Corr` objects can be computed together with other `Corr` objects, `Obs` objects or real numbers and integers.
```python
my_new_corr = 0.3 * my_corr[2] * my_corr * my_corr + 12 / my_corr
```

`pyerrors` provides the user with a set of regularly used methods for the manipulation of correlator objects:
- `Corr.gamma_method` applies the gamma method to all entries of the correlator.
- `Corr.m_eff` to construct effective masses. Various variants for periodic and fixed temporal boundary conditions are available.
- `Corr.deriv` returns the first derivative of the correlator as `Corr`. Different discretizations of the numerical derivative are available.
- `Corr.second_deriv` returns the second derivative of the correlator as `Corr`. Different discretizations of the numerical derivative are available.
- `Corr.symmetric` symmetrizes parity even correlations functions, assuming periodic boundary conditions.
- `Corr.anti_symmetric` anti-symmetrizes parity odd correlations functions, assuming periodic boundary conditions.
- `Corr.T_symmetry` averages a correlator with its time symmetry partner, assuming fixed boundary conditions.
- `Corr.plateau` extracts a plateau value from the correlator in a given range.
- `Corr.roll` periodically shifts the correlator.
- `Corr.reverse` reverses the time ordering of the correlator.
- `Corr.correlate` constructs a disconnected correlation function from the correlator and another `Corr` or `Obs` object.
- `Corr.reweight` reweights the correlator.

`pyerrors` can also handle matrices of correlation functions and extract energy states from these matrices via a generalized eigenvalue problem (see `pyerrors.correlators.Corr.GEVP`).

For the full API see `pyerrors.correlators.Corr`.

# Complex valued observables

`pyerrors` can handle complex valued observables via the class `pyerrors.obs.CObs`.
`CObs` are initialized with a real and an imaginary part which both can be `Obs` valued.

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

# Error propagation in iterative algorithms

`pyerrors` supports exact linear error propagation for iterative algorithms like various variants of non-linear least sqaures fits or root finding. The derivatives required for the error propagation are calculated as described in [arXiv:1809.01289](https://arxiv.org/abs/1809.01289).

## Least squares fits

Standard non-linear least square fits with errors on the dependent but not the independent variables can be performed with `pyerrors.fits.least_squares`. As default solver the Levenberg-Marquardt algorithm implemented in [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) is used.

Fit functions have to be of the following form
```python
import autograd.numpy as anp

def func(a, x):
    return a[1] * anp.exp(-a[0] * x)
```
**It is important that numerical functions refer to `autograd.numpy` instead of `numpy` for the automatic differentiation to work properly.**

Fits can then be performed via
```python
fit_result = pe.fits.least_squares(x, y, func)
print("\n", fit_result)
> Fit with 2 parameters
> Method: Levenberg-Marquardt
> `ftol` termination condition is satisfied.
> chisquare/d.o.f.: 0.9593035785160936

>  Goodness of fit:
> χ²/d.o.f. = 0.959304
> p-value   = 0.5673
> Fit parameters:
> 0	 0.0548(28)
> 1	 1.933(64)
```
where x is a `list` or `numpy.array` of `floats` and y is a `list` or `numpy.array` of `Obs`.

Data stored in `Corr` objects can be fitted directly using the `Corr.fit` method.
```python
my_corr = pe.Corr(y)
fit_result = my_corr.fit(func, fitrange=[12, 25])
```
this can simplify working with absolute fit ranges and takes care of gaps in the data automatically.

For fit functions with multiple independent variables the fit function can be of the form

```python
def func(a, x):
    (x1, x2) = x
    return a[0] * x1 ** 2 + a[1] * x2
```

## Total least squares fits
`pyerrors` can also fit data with errors on both the dependent and independent variables using the total least squares method also referred to orthogonal distance regression as implemented in [scipy](https://docs.scipy.org/doc/scipy/reference/odr.html), see `pyerrors.fits.least_squares`. The syntax is identical to the standard least squares case, the only diffrence being that `x` also has to be a `list` or `numpy.array` of `Obs`.

For the full API see `pyerrors.fits` for fits and `pyerrors.roots` for finding roots of functions.

# Matrix operations
`pyerrors` provides wrappers for `Obs`-valued matrix operations based on `numpy.linalg`. The supported functions include:
- `inv` for the matrix inverse.
- `cholseky` for the Cholesky decomposition.
- `det` for the matrix determinant.
- `eigh` for eigenvalues and eigenvectors of hermitean matrices.
- `eig` for eigenvalues of general matrices.
- `pinv` for the Moore-Penrose pseudoinverse.
- `svd` for the singular-value-decomposition.

For the full API see `pyerrors.linalg`.

# Export data
The preferred exported file format within `pyerrors` is json.gz

## Jackknife samples
For comparison with other analysis workflows `pyerrors` can generate jackknife samples from an `Obs` object or import jackknife samples into an `Obs` object.
See `pyerrors.obs.Obs.export_jackknife` and `pyerrors.obs.import_jackknife` for details.

# Input
`pyerrors` includes an `input` submodule in which input routines and parsers for the output of various numerical programs are contained. For details see `pyerrors.input`.

# Citing
If you use `pyerrors` for research that leads to a publication please consider citing:
- Ulli Wolff, "Monte Carlo errors with less errors". Comput.Phys.Commun. 156 (2004) 143-153, Comput.Phys.Commun. 176 (2007) 383 (erratum).
- Stefan Schaefer, Rainer Sommer, Francesco Virotta, "Critical slowing down and error analysis in lattice QCD simulations". Nucl.Phys.B 845 (2011) 93-119.
- Alberto Ramos, "Automatic differentiation for error analysis of Monte Carlo data". Comput.Phys.Commun. 238 (2019) 19-35.
'''
from .obs import *
from .correlators import *
from .fits import *
from .misc import *
from . import dirac
from . import input
from . import linalg
from . import mpm
from . import roots

from .version import __version__
