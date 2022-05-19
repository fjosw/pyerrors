#!/usr/bin/env python

from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='pyerrors',
      version='2.0.0',
      description='Error analysis for lattice QCD',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Fabian Joswig',
      author_email='fabian.joswig@ed.ac.uk',
      packages=find_packages(),
      python_requires='>=3.6.0',
      install_requires=['numpy>=1.16', 'autograd>=1.4', 'numdifftools', 'matplotlib>=3.3', 'scipy', 'iminuit>=2', 'h5py']
     )
