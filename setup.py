#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='pyerrors',
      version='2.1.0+dev',
      description='Error analysis for lattice QCD',
      author='Fabian Joswig',
      author_email='fabian.joswig@ed.ac.uk',
      packages=find_packages(),
      python_requires='>=3.6.0',
      install_requires=['numpy>=1.16', 'autograd>=1.4', 'numdifftools', 'matplotlib>=3.3', 'scipy', 'iminuit>=2', 'h5py', 'lxml', 'python-rapidjson']
     )
