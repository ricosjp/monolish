"""type convert."""

import numpy
import pandas

def dict_list_to_ndarray(dict_list):
    """
        dict_list : [{}, {}, ...] -> array : [], ndarray : [[], [], [], ...]
    """
    dataframe = pandas.DataFrame(dict_list)
    column_array = numpy.array(dataframe.columns)
    values_ndarray = numpy.array([numpy.array(row) for index, row in dataframe.iterrows()])
    return column_array, values_ndarray

def dict_list_to_dataframe(dict_list):
    """
        dict_list : [{}, {}, ...] -> DataFrame
    """
    dataframe = pandas.DataFrame(dict_list)
    return dataframe
