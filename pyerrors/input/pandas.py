import warnings
import gzip
import sqlite3
import pandas as pd
from ..obs import Obs
from ..correlators import Corr
from .json import create_json_string, import_json_string


def to_sql(df, table_name, db, if_exists='fail', gz=True, **kwargs):
    """Write DataFrame including Obs or Corr valued columns to sqlite database.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to be written to the database.
    table_name : str
        Name of the table in the database.
    db : str
        Path to the sqlite database.
    if exists : str
        How to behave if table already exists. Options 'fail', 'replace', 'append'.
    gz : bool
        If True the json strings are gzipped.
    """
    se_df = _serialize_df(df, gz=gz)
    con = sqlite3.connect(db)
    se_df.to_sql(table_name, con, if_exists=if_exists, index=False, **kwargs)
    con.close()


def read_sql(sql, db, auto_gamma=False, **kwargs):
    """Execute SQL query on sqlite database and obtain DataFrame including Obs or Corr valued columns.

    Parameters
    ----------
    sql : str
        SQL query to be executed.
    db : str
        Path to the sqlite database.
    auto_gamma : bool
        If True applies the gamma_method to all imported Obs objects with the default parameters for
        the error analysis. Default False.
    """
    con = sqlite3.connect(db)
    extract_df = pd.read_sql(sql, con, **kwargs)
    con.close()
    return _deserialize_df(extract_df, auto_gamma=auto_gamma)


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
    out = _serialize_df(df, gz=False)

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

    return _deserialize_df(re_import, auto_gamma=auto_gamma)


def _serialize_df(df, gz=False):
    """Serializes all Obs or Corr valued columns into json strings according to the pyerrors json specification.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be serilized.
    gz: bool
        gzip the json string representation. Default False.
    """
    out = df.copy()
    for column in out:
        if isinstance(out[column][0], (Obs, Corr)):
            out[column] = out[column].transform(lambda x: create_json_string(x, indent=0))
            if gz is True:
                out[column] = out[column].transform(lambda x: gzip.compress(x.encode('utf-8')))
    return out


def _deserialize_df(df, auto_gamma=False):
    """Deserializes all pyerrors json strings into Obs or Corr objects according to the pyerrors json specification.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be deserilized.
    auto_gamma : bool
        If True applies the gamma_method to all imported Obs objects with the default parameters for
        the error analysis. Default False.

    Notes:
    ------
    In case any column of the DataFrame is gzipped it is gunzipped in the process.
    """
    for column in df.select_dtypes(include="object"):
        if isinstance(df[column][0], bytes):
            if df[column][0].startswith(b"\x1f\x8b\x08\x00"):
                df[column] = df[column].transform(lambda x: gzip.decompress(x).decode('utf-8'))
        if isinstance(df[column][0], str):
            if '"program":' in df[column][0][:20]:
                df[column] = df[column].transform(lambda x: import_json_string(x, verbose=False))
                if auto_gamma is True:
                    df[column].apply(lambda x: x.gamma_method())
    return df
