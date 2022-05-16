# Development
### Setup
If you want to contribute to `pyerrors` please [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) `pyerrors` on Github, clone the current `develop` branch
```
git clone http://github.com/my_username/pyerrors.git --branch develop
```
and create your own branch
```
cd pyerrors
git checkout -b feature/my_feature
```
It can be convenient to install the package in editable mode in the local python environment when developing new features
```
pip install -e .
```
Please send any pull requests to the `develop` branch.

### Documentation
Please add docstrings to any new function, class or method you implement. The documentation is automatically generated from these docstrings. The startpage of the documentation is generated from the docstring of `pyerrors/__init__.py`.

### Tests
When implementing a new feature or fixing a bug please add meaningful tests to the files in the `tests` directory which cover the new code.
For all pull requests tests are executed for the most recent python releases via
```
pytest -vv --cov=pyerrors
pytest -vv --nbmake examples/*.ipynb
```
requiring `pytest`, `pytest-cov`, `pytest-benchmark` and `nbmake`. To get a coverage report in html run
```
pytest --cov=pyerrors --cov-report html
```
The linter `flake8` is executed with the command
```
flake8 --ignore=E501 --exclude=__init__.py pyerrors
```
Please make sure that all tests are passed for a new pull requests.
