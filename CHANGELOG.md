# Changelog

All notable changes to this project will be documented in this file.

## [2.3.0] - 2022-10-13
### Added
- `least_squares` and `total_least_squares` fits now have an optional keyword argument `num_grad`. If this argument is set to `True` the error propagation of the fit is performed via numerical instead of automatic differentiation. This options allows for fits functions which contain special functions or which are not analytically known.

### Fixed
- Bug in `Corr.show` `comp` option fixed.

## [2.2.0] - 2022-08-01
### Added
- New submodule `input.pandas` added which adds the possibility to read and write pandas DataFrames containing `Obs` or `Corr` objects to csv files or SQLite databases.
- `hash` method for `Obs` objects added.
- `Obs.reweight` method added in analogy to `Corr.reweight` which allows for a more convenient reweighting of individual observables.
- `Corr.show` now has the additional argument `title` which allows to add a title to the figure. Figures are now saved with `bbox_inches='tight'`.
- Function for the extraction of the gradient flow coupling added (see 1607.06423 for details).
- `Corr.is_matrix_symmetric` added which efficiently checks whether a correlator matrix is symmetric. This is used to speed up the GEVP method.

### Fixed
- `Corr.m_eff` can now deal with correlator entries which are exactly zero.
- Minor bugs in `input.dobs` fixed.

## [2.1.3] - 2022-06-13
### Fixed
- Further bugs in connection with correlator objects which have arrays with None entries as content fixed.

## [2.1.2] - 2022-06-10
### Fixed
- Bug in `Corr.matrix_symmetric` fixed which appeared when a time slice contained an array with `None` entries.

## [2.1.1] - 2022-06-06
### Fixed
- Bug in error propagation of correlated least square fits fixed.
- `Fit_result.gamma_method` can now be called with kwargs.

## [2.1.0] - 2022-05-31
### Added
- `obs.covariance` now has the option to smooth small eigenvalues of the matrix with the method described in hep-lat/9412087.
- `Corr.prune` was added which can reduce the size of a correlator matrix before solving the GEVP.
- `Corr.show` has two additional optional arguments. `hide_sigma` to hide data points with large errors and `references` to display reference values as horizontal lines.
- I/O routines for ALPHA dobs format added.
- `input.hadrons` functionality extended.

### Changed
- The standard behavior of the `Corr.GEVP` method has changed. It now returns all eigenvectors of the system instead of only the specified ones as default. The standard way of sorting the eigenvectors was changed to `Eigenvalue`. The argument `sorted_list` was deprecated in favor of `sort`.
- Before performing a correlated fit the routine first runs an uncorrelated one to obtain a better guess for the initial parameters.

### Fixed
- `obs.covariance` now also gives correct estimators if data defined on non-identical configurations is passed to the function.
- Rounding errors in estimating derivatives of fit parameters with respect to input data from the inverse hessian reduced. This should only make a difference when the magnitude of the errors of different fit parameters vary vastly.
- Bug in json.gz format fixed which did not properly store the replica mean values. Format version bumped to 1.1.
- The GEVP matrix is now symmetrized before solving the system for all sorting options not only the one with fixed `ts`.
- Automatic range estimation improved in `fits.residual_plot`.
- Bugs in `input.bdio` fixed.

## [2.0.0] - 2022-03-31
### Added
- The possibility to work with Monte Carlo histories which are evenly or unevenly spaced was added.
- `cov_Obs` added as a possibility to propagate the error of non Monte Carlo data together with Monte Carlo data.
- `CObs` class added which can handle complex valued Markov chain Monte Carlo data and the corresponding error propagation.
- Matrix to matrix operations like the matrix inverse now also work for complex matrices and matrices containing entries that are not `Obs` but `float` or `int`.
- Support for a new `json.gz` file format was added.
- The Corr class now has additional methods like `reverse`, `T_symmetry`, `correlate` and `reweight`.
- `Corr.m_eff` can now cope with periodic and anti-periodic correlation functions.
- Forward, backward and improved variants of the first and second derivative were added to the `Corr` class.
- `GEVP` functionality of the `Corr` class was reworked and improved.
- The `linalg` module now has explicit functions `inv`, `cholesky` and `det`.
- `Obs` objects now have methods `is_zero` and `is_zero_within_error` as well as overloaded comparison operations.
- Functions to convert `Obs` data to or from jackknife was added.
- Alternative matrix multiplication routines `einsum` and `jack_matmul` were added to `linalg` module which make use of the jackknife approximation and are much faster for large matrices.
- Additional input routines for npr data added to `input.hadrons`.
- The `sfcf` and `openQCD` input modules can now handle all recent file type versions.
- `extract_t0` can now visualize the extraction on the fly.
- Module added which provides the Dirac gamma matrices in the Grid convention.
- Version number added.

### Changed
- The internal bookkeeping system for ensembles/replica was changed. The separator for replica is now `|`.
- The fit functions were renamed to `least_squares` and `total_least_squares`.
- The output of the fit functions is now a dedicated results class which keeps track of all relevant information.
- The fit functions can now deal with provided covariance matrices.
- `covariance` can now operate on a list or array of `Obs` and returns a matrix. The covariance estimate by pyerrors is now always positive semi-definite (within machine precision. Various warnings and exceptions were added for cases in which estimated covariances are close to singular.
- The convention for the fit range in the Corr class has been changed.
- Various method of the `Corr` class were renamed.
- `Obs.print` was renamed to `Obs.details` and the output was improved.
- The default value for `Corr.prange` is now `None`.
- The `input` module was restructured to contain one submodule per data source.
- Performance of Obs.__init__ improved.

### Removed
- The function `plot_corrs` was deprecated as all its functionality is now contained within `Corr.show`.
- `fits.covariance_matrix` was removed as it is now redundant with the functionality of `covariance`.
- The kwarg `bias_correction` in `derived_observable` was removed.
- Obs no longer have an attribute `e_Q`.
- Removed `fits.fit_exp`.
- Removed jackknife module.

## [1.1.0] - 2021-10-11
### Added
- `Corr` class added
- `roots` module added which can find the roots of a function that depends on Monte Carlo data via pyerrors `Obs`
- `input/hadrons` module added which can read hdf5 files written by [Hadrons](https://github.com/aportelli/Hadrons)
- `read_rwms` can now read reweighting factors in the format used by openQCD-2.0.

## [1.0.1] - 2020-11-03
### Fixed
- Bug in `pyerrors.covariance` fixed that appeared when working with several
  replica of different length.

## [1.0.0] - 2020-10-13
### Added
- Compatibility with the BDIO Native format outlined [here](https://ific.uv.es/~alramos/docs/ADerrors/tutorial/). Read and write function added to input.bdio
- new function `input.bdio.read_dSdm` which can read the bdio output of the
  program `dSdm` by Tomasz Korzec
- Expected chisquare implemented for fits with xerrors 
- New implementation of the covariance of two observables which employs the
  arithmetic mean of the integrated autocorrelation times of the two
  observables. This new procedure has proven to be less biased in simulated
  data and is also much faster to compute as the computation time is of O(N)
  whereas the evaluation of the full correlation function is of O(Nlog(N)).
- Added function `gen_correlated_data` to `misc` which generates a set of
  observables with given covariance and autocorrelation.

### Fixed
- Bias correction hep-lat/0306017 eq. (49) is no longer applied to the
  exponential tail in the critical slowing down analysis, but only to the part
  which is directly estimated from rho. This can lead to slightly smaller
  errors when using the critical slowing down analysis. The values for the
  integrated autocorrelation time tauint now include this bias correction (up
  to now the bias correction was applied after estimating tauint). The errors
  resulting from the automatic windowing procedure are unchanged.

## [0.8.1] - 2020-06-09
### Fixed
- Bug in `fits.standard_fit` fixed which occurred when attempting a fit with zero
  degrees of freedom.

## [0.8.0] - 2020-06-05
### Added
- `merge_obs` function added which allows to merge Obs which describe different replica of the same observable and have been read in separately. Use with care as there is no safeguard implemented which prevent you from merging unrelated Obs.
- `standard fit` and `odr_fit` can now treat fits with several x-values via tuples.
- Fit functions have a new kwarg `dict_output` which allows to change the
  output to a dictionary containing additional information.
- `S_dict` and `tau_exp_dict` added to Obs in which global values for individual ensembles can be stored.
- new function `read_pbp` added which reads dS/dm_q from pbp.dat files.
- new function `extract_t0` added which can extract the value of t0 from .ms.dat files of openQCD v 1.2

### Changed
- When creating an Obs object defined for multiple replica/ensembles, the given names are now sorted alphabetically before assigning the internal dictionaries. This makes sure that `my_Obs` has the same dictionaries as `my_Obs * 1` (`derived_observable` always sorted the names). WARNING: `Obs` created with previous versions of pyerrors may not be completely identical to new ones (The internal dictionaries may have different ordering). However, this should not affect the inner workings of the error analysis.

### Fixed
- Bug in `covariance` fixed which appeared when different ensemble contents were used.

## [0.7.0] - 2020-03-10
### Added
- New fit functions for fitting with and without x-errors added which use automatic differentiation and should be more reliable than the old ones.
- Fitting with Bayesian priors added.
- New functions for visualization of fits which can be activated via the kwargs resplot and qqplot.
- chisquare/expected_chisquared which takes into account correlations in the data and non-linearities in the fit function can now be activated with the kwarg expected_chisquare.
- Silent mode added to fit functions.
- Examples reworked.
- Changed default function to compute covariances.
- output of input.bdio.read_mesons is now a dictionary instead of a list.

### Deprecated
- The function `fit_general` which is based on numerical differentiation will be removed in future versions as new fit functions based on automatic differentiation are now available.

## [0.6.1] - 2020-01-14
### Added
- mesons bdio functionality improved and accelerated, progress report added.
- added the possibility to manually supply a jacobian to derived_observable via the kwarg `man_grad`. This feature was not implemented for the user, but for internal optimization of most basic arithmetic operations which now do not require a call to the autograd package anymore. This results in a speed up of 2 to 3, especially relevant for the multiplication of large matrices.

### Changed
- input.py and bdio.py moved into submodule input. This should not affect the user API.
- autograd.numpy was replaced by pure numpy wherever it was possible. This should result in a slight speed up.

### Fixed
- fixed bias_correction which broke as a result of the vectorized derived_observable.
- linalg.eig does not give an error anymore if the eigenvalues are complex by just truncating the imaginary part.

## [0.6.0] - 2020-01-06
### Added
- Matrix pencil method for algebraic extraction of energy levels implemented according to [Y. Hua, T. K. Sarkar, IEEE Trans. Acoust. 38, 814-824 (1990)](https://ieeexplore.ieee.org/document/56027) in module `mpm.py`.
- Import API simplified. After `import pyerrors as pe`, some submodules can be accessed via `pe.fits` etc.
- `derived_observable` now supports functions which have single- or multi-dimensional numpy arrays as input and/or output (Works only with automatic differentiation).
- Matrix functions accelerated by using the new version of `derived_observable`.
- New matrix functions: Moore-Penrose Pseudoinverse, Singular Value Decomposition, eigenvalue determination of a general matrix (automatic differentiation included from autograd master).
- Obs can now be compared with < or >, a list of Obs can now be sorted.
- Numerical differentiation can now be controlled via the kwargs of numdifftools.step_generators.MaxStepGenerator.
- Tuned standard parameters for numerical derivative to `base_step=0.1` and `step_ratio=2.5`.

### Changed
- Matrix functions moved to new module `linalg.py`.
- Kolmogorov-Smirnov test moved to new module `misc.py`.

## [0.5.0] - 2019-12-19
### Added
- Numerical differentiation is now based on the package numdifftools which should be more reliable.

### Changed
- kwarg `h_num_grad` changed to `num_grad` which takes boolean values (default False).
- Speed up of rfft calculation of the autocorrelation by reducing the zero padding.
