from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='pyerrors',
      version='2.1.3',
      description='Error analysis for lattice QCD',
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
      python_requires='>=3.6.0',
      install_requires=['numpy>=1.16', 'autograd>=1.4', 'numdifftools', 'matplotlib>=3.3', 'scipy>=1', 'iminuit>=2', 'h5py>=3', 'lxml>=4', 'python-rapidjson>=1'],
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Topic :: Scientific/Engineering :: Physics'
      ],
     )
