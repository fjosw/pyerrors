# Development
### Setup
If you want to contribute to `pyerrors` please clone the current `develop` branch
```
git clone http://github.com/fjosw/pyerrors.git --branch develop
```
and create your own branch
```
cd pyerrors
git checkout -b feature/my_feature
```
I find it convenient to install the package in editable mode in your local python environment
```
pip install -e .
```
### Documentation
Please add meaningful docstrings to any new function, class or method you implement.

### Tests
When implementing a new feature or fixing a bug please add meaningful tests to the files in the `tests` directory which cover the new code.

### Continous integration
For all pull requests to the `develop` branch tests are executed for the most recent python releases via
```
pytest -v
```
and `flake8` is executed with the command
```
flake8 --ignore=E501,E722 --exclude=__init__.py pyerrors
```
Please make sure that all tests are passed for a new pull request.
