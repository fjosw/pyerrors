# Development
### Setup
If you want to contribute to `pyerrors` please fork `pyerrors` on Github, clone the current `develop` branch
```
git clone http://github.com/my_username/pyerrors.git --branch develop
```
and create your own branch
```
cd pyerrors
git checkout -b feature/my_feature
```
I find it convenient to install the package in editable mode in the local python environment
```
pip install -e .
```
Please send any pull requests to the `develop` branch.
### Documentation
Please add docstrings to any new function, class or method you implement.

### Tests
When implementing a new feature or fixing a bug please add meaningful tests to the files in the `tests` directory which cover the new code.

### Continous integration
For all pull requests tests are executed for the most recent python releases via
```
pytest -v
```
and the linter `flake8` is executed with the command
```
flake8 --ignore=E501,E722 --exclude=__init__.py pyerrors
```
Please make sure that all tests are passed for a new pull requests.
