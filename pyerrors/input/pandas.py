import warnings
import gzip
import sqlite3
import pandas as pd
from ..obs import Obs
from ..correlators import Corr
from .json import create_json_string, import_json_string
import numpy as np


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

    Returns
    -------
    None
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

    Returns
    -------
    data : pandas.DataFrame
        Dataframe with the content of the sqlite database.
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

    Returns
    -------
    None
    """
    for column in df:
        serialize = _need_to_serialize(df[column])
        if not serialize:
            if all(isinstance(entry, (int, np.integer, float, np.floating)) for entry in df[column]):
                if any([np.isnan(entry) for entry in df[column]]):
                    warnings.warn("nan value in column " + column + " will be replaced by None", UserWarning)

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

    Returns
    -------
    data : pandas.DataFrame
        Dataframe with the content of the sqlite database.
    """
    if not fname.endswith('.csv') and not fname.endswith('.gz'):
        fname += '.csv'

    if gz is True:
        if not fname.endswith('.gz'):
            fname += '.gz'
        with gzip.open(fname) as f:
            re_import = pd.read_csv(f, keep_default_na=False)
    else:
        if fname.endswith('.gz'):
            warnings.warn("Trying to read from %s without unzipping!" % fname, UserWarning)
        re_import = pd.read_csv(fname, keep_default_na=False)

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
        serialize = _need_to_serialize(out[column])

        if serialize is True:
            out[column] = out[column].transform(lambda x: create_json_string(x, indent=0) if x is not None else None)
            if gz is True:
                out[column] = out[column].transform(lambda x: gzip.compress((x if x is not None else '').encode('utf-8')))
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

        if not all([e is None for e in df[column]]):
            df[column] = df[column].replace({r'^$': None}, regex=True)
            i = 0
            while df[column][i] is None:
                i += 1
            if isinstance(df[column][i], str):
                if '"program":' in df[column][i][:20]:
                    df[column] = df[column].transform(lambda x: import_json_string(x, verbose=False) if x is not None else None)
                    if auto_gamma is True:
                        if isinstance(df[column][i], list):
                            df[column].apply(lambda x: [o.gm() if o is not None else x for o in x])
                        else:
                            df[column].apply(lambda x: x.gm() if x is not None else x)
    return df


def _need_to_serialize(col):
    serialize = False
    i = 0
    while i < len(col) and col[i] is None:
        i += 1
    if i == len(col):
        return serialize
    if isinstance(col[i], (Obs, Corr)):
        serialize = True
    elif isinstance(col[i], list):
        if all(isinstance(o, Obs) for o in col[i]):
            serialize = True
    return serialize
