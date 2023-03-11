import platform
import numpy as np
import scipy
import matplotlib
import pandas as pd


__version__ = "2.7.0-dev"


def print_config():
    """Print information about version of python, pyerrors and dependencies."""
    config = {"python": platform.python_version(),
              "pyerrors": __version__,
              "numpy": np.__version__,
              "scipy": scipy.__version__,
              "matplotlib": matplotlib.__version__,
              "pandas": pd.__version__}

    for key, value in config.items():
        print(f"{key : <10}\t {value}")
