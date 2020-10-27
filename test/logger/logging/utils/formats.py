"""type convert."""

import numpy as np
import pandas as pd

def dict_list_to_ndarray(dict_list):
    """
        dict_list : [{}, {}, ...] -> array : [], ndarray : [[], [], [], ...]
    """
    dataframe = pd.DataFrame(dict_list)
    column_array = np.array(dataframe.columns)
    values_ndarray = np.array([np.array(row) for index, row in dataframe.iterrows()])
    return column_array, values_ndarray

def dict_list_to_dataframe(dict_list):
    """
        dict_list : [{}, {}, ...] -> DataFrame
    """
    dataframe = pd.DataFrame(dict_list)
    return dataframe
