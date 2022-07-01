r'''
`pyerrors` includes an `input` submodule in which input routines and parsers for the output of various numerical programs are contained.

# Jackknife samples
For comparison with other analysis workflows `pyerrors` can also generate jackknife samples from an `Obs` object or import jackknife samples into an `Obs` object.
See `pyerrors.obs.Obs.export_jackknife` and `pyerrors.obs.import_jackknife` for details.
'''
from . import bdio
from . import hadrons
from . import json
from . import misc
from . import openQCD
from . import pandas
from . import sfcf
