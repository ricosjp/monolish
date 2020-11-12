import pytest
import os
import sys
import pandas as pd

from monolish_log_viewer.libs import aggregate
from monolish_log_viewer.utils import read

data_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../test_data/cg_iter.yml")
print(data_dir)

def test_layer_1_aggregated():
    with open(data_dir, "r") as file:
        dict_list = read.reader(file, "yaml")

    aggregate_dataframe = aggregate.AggregateDataFrame()
    layer1_aggr_df = aggregate_dataframe.layer_1_aggregated(dict_list)

    # colum check
    assert list(layer1_aggr_df.columns) == ["name", "layer", "time", "cnt"]

    # layer check
    assert layer1_aggr_df["layer"].sample(n=1, random_state=1).values[0] == 1