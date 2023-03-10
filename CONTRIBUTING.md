# Contributing

If you are new to contributing to open source software [this guide](https://opensource.guide/how-to-contribute) can help get you started.

### Setup
If you want to contribute to `pyerrors` please [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) `pyerrors` on Github, clone the repository
```
git clone http://github.com/my_username/pyerrors.git
```
and create your own branch for the feature or bug fix
```
cd pyerrors
git checkout -b feature/my_feature
```
After committing your changes please send a pull requests to the `develop` branch. A guide on how to create a pull request can be found [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

### Documentation
Please add docstrings to any new function, class or method you implement. The documentation is automatically generated from these docstrings. We follow the [numpydoc style](https://numpydoc.readthedocs.io/en/latest/format.html) for docstrings. The startpage of the documentation is generated from the docstring of `pyerrors/__init__.py`.

### Tests
When implementing a new feature or fixing a bug please add meaningful tests to the files in the `tests` directory which cover the new code.
For all pull requests tests are executed for the most recent python releases via
```
pytest
pytest --nbmake examples/*.ipynb
```
requiring `pytest`, `pytest-cov`, `pytest-benchmark`, `hypothesis` and `nbmake`. To install the test dependencies one can run `pip install pyerrors[test]`

To get a coverage report in html run
```
pytest --cov=pyerrors --cov-report html
```
The linter `flake8` is executed with the command
```
flake8 --ignore=E501,W503 --exclude=__init__.py pyerrors
```
Please make sure that all tests pass for a new pull requests.
