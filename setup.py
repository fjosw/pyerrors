#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='pyerrors',
      version='1.1.0',
      description='Error analysis for lattice QCD',
      author='Fabian Joswig',
      author_email='fabian.joswig@wwu.de',
      packages=find_packages(),
      python_requires='>=3.5.0',
      install_requires=['numpy>=1.16', 'autograd>=1.2', 'numdifftools', 'matplotlib', 'scipy', 'iminuit']
     )
