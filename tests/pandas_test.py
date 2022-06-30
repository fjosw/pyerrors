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


