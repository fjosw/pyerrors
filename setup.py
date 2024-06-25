from setuptools import setup, find_packages
from pathlib import Path
from distutils.util import convert_path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

version = {}
with open(convert_path('pyerrors/version.py')) as ver_file:
    exec(ver_file.read(), version)

setup(name='pyerrors',
      version=version['__version__'],
      description='Error propagation and statistical analysis for Monte Carlo simulations',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/fjosw/pyerrors",
      project_urls= {
        'Documentation': 'https://fjosw.github.io/pyerrors/pyerrors.html',
        'Bug Tracker':   'https://github.com/fjosw/pyerrors/issues',
        'Changelog' :    'https://github.com/fjosw/pyerrors/blob/master/CHANGELOG.md'
      },
      author='Fabian Joswig',
      author_email='fabian.joswig@ed.ac.uk',
      license="MIT",
      packages=find_packages(),
      python_requires='>=3.8.0',
      install_requires=['numpy>=1.24,<2', 'autograd>=1.6.2', 'numdifftools>=0.9.41', 'matplotlib>=3.7', 'scipy>=1.10', 'iminuit>=2.21', 'h5py>=3.8', 'lxml>=4.9', 'python-rapidjson>=1.10', 'pandas>=2.0'],
      extras_require={'test': ['pytest', 'pytest-cov', 'pytest-benchmark', 'hypothesis', 'nbmake', 'flake8']},
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Topic :: Scientific/Engineering :: Physics'
      ],
     )
