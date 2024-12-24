r'''
`pyerrors` includes an `input` submodule in which input routines and parsers for the output of various numerical programs are contained.

# Jackknife samples
For comparison with other analysis workflows `pyerrors` can also generate jackknife samples from an `Obs` object or import jackknife samples into an `Obs` object.
See `pyerrors.obs.Obs.export_jackknife` and `pyerrors.obs.import_jackknife` for details.
'''
from . import bdio as bdio
from . import dobs as dobs
from . import hadrons as hadrons
from . import json as json
from . import misc as misc
from . import openQCD as openQCD
from . import pandas as pandas
from . import sfcf as sfcf
