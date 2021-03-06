# pyerrors
pyerrors is a python package for error computation and propagation of Markov Chain Monte Carlo data.
It is based on the gamma method [arXiv:hep-lat/0306017](https://arxiv.org/abs/hep-lat/0306017). Some of its features are:
* automatic differentiation as suggested in [arXiv:1809.01289](https://arxiv.org/abs/1809.01289) (partly based on the [autograd](https://github.com/HIPS/autograd) package)
* the treatment of slow modes in the simulation as suggested in [arXiv:1009.5228](https://arxiv.org/abs/1009.5228)
* multi ensemble analyses
* non-linear fits with y-errors and exact linear error propagation based on automatic differentiation as introduced in [arXiv:1809.01289]
* non-linear fits with x- and y-errors and exact linear error propagation based on automatic differentiation
* matrix valued operations and their error propagation based on automatic differentiation (cholesky decomposition, calculation of eigenvalues and eigenvectors, singular value decomposition...)
* implementation of the matrix-pencil-method [IEEE Trans. Acoust. 38, 814-824 (1990)](https://ieeexplore.ieee.org/document/56027) for the extraction of energy levels, especially suited for noisy data and excited states

There exist similar implementations of gamma method error analysis suites in
- [Fortran](https://gitlab.ift.uam-csic.es/alberto/aderrors).
- [Julia](https://gitlab.ift.uam-csic.es/alberto/aderrors.jl)
- [Python 3](https://github.com/mbruno46/pyobs)

## Installation
pyerrors requires python versions >= 3.5.0

Install the package for the local user:
```bash
pip install . --user
```

Run tests to verify the installation:
```bash
pytest .
```

## Usage
The basic objects of a pyerrors analysis are instances of the class `Obs`. They can be initialized with an array of Monte Carlo data (e.g. `samples1`) and a name for the given ensemble (e.g. `'ensemble1'`). The `gamma_method` can then be used to compute the statistical error, taking into account autocorrelations. The `print` method  outputs a human readable result.
```python
import numpy as np
import pyerrors as pe

observable1 = pe.Obs([samples1], ['ensemble1'])
observable1.gamma_method()
observable1.print()
```
Often one is interested in secondary observables which can be arbitrary functions of primary observables. `pyerrors` overloads most basic math operations and numpy functions such that the user can work with `Obs` objects as if they were floats
```python
observable3 = 12.0 / observable1 ** 2 - np.exp(-1.0 / observable2)
observable3.gamma_method()
observable3.print()
```

More detailed examples can be found in  the `/examples` folder:

* [01_basic_example](examples/01_basic_example.ipynb)
* [02_pcac_example](examples/02_pcac_example.ipynb)
* [03_fit_example](examples/03_fit_example.ipynb)
* [04_matrix_operations](examples/04_matrix_operations.ipynb)


## License
[MIT](https://choosealicense.com/licenses/mit/)
