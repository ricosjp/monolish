import numpy as np
import pandas as pd

def dict_list_to_ndarray(dict_list):
    """
        dict_list : [{}, {}, ...] -> array : [], ndarray : [[], [], [], ...]
    """
    df = pd.DataFrame(dict_list)
    column_array = np.array(df.columns)
    values_ndarray = np.array([np.array(row) for index, row in df.iterrows()])
    return column_array, values_ndarray


def dict_list_to_dataframe(dict_list):
    df = pd.DataFrame(dict_list)
    return df