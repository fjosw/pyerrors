import numpy as np
import pandas as pd
import pyerrors as pe

def test_df_export_import(tmp_path):
    my_dict = {"int": 1,
           "float": -0.01,
           "Obs1": pe.pseudo_Obs(87, 21, "test_ensemble"),
           "Obs2": pe.pseudo_Obs(-87, 21, "test_ensemble2")}
    for gz in [True, False]:
        my_df = pd.DataFrame([my_dict] * 10)

        pe.input.pandas.dump_df(my_df, (tmp_path / 'df_output').as_posix(), gz=gz)
        reconstructed_df = pe.input.pandas.load_df((tmp_path / 'df_output').as_posix(), auto_gamma=True, gz=gz)
        assert np.all(my_df == reconstructed_df)

        pe.input.pandas.load_df((tmp_path / 'df_output.csv').as_posix(), gz=gz)


def test_df_Corr(tmp_path):

    my_corr = pe.Corr([pe.pseudo_Obs(-0.48, 0.04, "test"), pe.pseudo_Obs(-0.154, 0.03, "test")])

    my_dict = {"int": 1,
           "float": -0.01,
           "Corr": my_corr}
    my_df = pd.DataFrame([my_dict] * 5)

    pe.input.pandas.dump_df(my_df, (tmp_path / 'df_output').as_posix())
    reconstructed_df = pe.input.pandas.load_df((tmp_path / 'df_output').as_posix(), auto_gamma=True)


def test_default_export_pe_import(tmp_path):
    df = pd.DataFrame([{"Column1": 1.1, "Column2": 2, "Column3": "my string£"}])
    df.to_csv((tmp_path / 'plain_df.csv').as_posix(), index=False)
    re_df = pe.input.pandas.load_df((tmp_path / 'plain_df').as_posix(), gz=False)
    assert np.all(df == re_df)


def test_pe_export_default_import(tmp_path):
    df = pd.DataFrame([{"Column1": 1.1, "Column2": 2, "Column3": "my string£"}])
    pe.input.pandas.dump_df(df, (tmp_path / 'pe_df').as_posix(), gz=False)
    re_df = pd.read_csv((tmp_path / 'pe_df.csv').as_posix())
    assert np.all(df == re_df)
