import warnings
import gzip
import pandas as pd
from ..obs import Obs
from ..correlators import Corr
from .json import create_json_string, import_json_string


def dump_df(df, fname, gz=True):
    """Exports a pandas DataFrame containing Obs valued columns to a (gzipped) csv file.

    Before making use of pandas to_csv functionality Obs objects are serialized via the standardized
    json format of pyerrors.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to be dumped to a file.
    fname : str
        Filename of the output file.
    gz : bool
        If True, the output is a gzipped csv file. If False, the output is a csv file.
    """

    out = serialize_df(df)

    if not fname.endswith('.csv'):
        fname += '.csv'

    if gz is True:
        if not fname.endswith('.gz'):
            fname += '.gz'
        out.to_csv(fname, index=False, compression='gzip')
    else:
        out.to_csv(fname, index=False)


def load_df(fname, auto_gamma=False, gz=True):
    """Imports a pandas DataFrame from a csv.(gz) file in which Obs objects are serialized as json strings.

    Parameters
    ----------
    fname : str
        Filename of the input file.
    auto_gamma : bool
        If True applies the gamma_method to all imported Obs objects with the default parameters for
        the error analysis. Default False.
    gz : bool
        If True, assumes that data is gzipped. If False, assumes JSON file.
    """

    if not fname.endswith('.csv') and not fname.endswith('.gz'):
        fname += '.csv'

    if gz is True:
        if not fname.endswith('.gz'):
            fname += '.gz'
        with gzip.open(fname) as f:
            re_import = pd.read_csv(f)
    else:
        if fname.endswith('.gz'):
            warnings.warn("Trying to read from %s without unzipping!" % fname, UserWarning)
        re_import = pd.read_csv(fname)

    return deserialize_df(re_import, auto_gamma)


def serialize_df(df):
    """Serializes all Obs or Corr valued columns into json strings according to the pyerrors json specification.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be serilized.
    """
    out = df.copy()
    for column in out:
        if isinstance(out[column][0], (Obs, Corr)):
            out[column] = out[column].transform(lambda x: create_json_string(x, indent=0))
    return out


def deserialize_df(df, auto_gamma=False):
    """Deserializes all pyerrors json strings into Obs or Corr objects according to the pyerrors json specification.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be deserilized.
    auto_gamma : bool
        If True applies the gamma_method to all imported Obs objects with the default parameters for
        the error analysis. Default False.
    """
    for column in df.select_dtypes(include="object"):
        if isinstance(df[column][0], str):
            if df[column][0][:20] == '{"program":"pyerrors':
                df[column] = df[column].transform(lambda x: import_json_string(x, verbose=False))
                if auto_gamma is True:
                    df[column].apply(lambda x: x.gamma_method())
    return df
