import warnings
import gzip
import pandas as pd
from ..obs import Obs
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

    out = df.copy()
    for column in out:
        if isinstance(out[column][0], Obs):
            out[column] = out[column].transform(lambda x: create_json_string(x, indent=0))

    if not fname.endswith('.csv'):
        fname += '.csv'

    out.to_csv(fname, index=False)
    if gz is True:
        with open(fname, 'rb') as f_in, gzip.open(fname + ".gz", 'wb') as f_out:
            f_out.writelines(f_in)


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

    for column in re_import.select_dtypes(include="object"):
        if isinstance(re_import[column][0], str):
            if re_import[column][0][:20] == '{"program":"pyerrors':
                re_import[column] = re_import[column].transform(lambda x: import_json_string(x, verbose=False))
                if auto_gamma is True:
                    re_import[column].apply(Obs.gamma_method)

    return re_import
