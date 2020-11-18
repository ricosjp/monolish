""" test aggregate """
import pytest
import os
import sys
import pandas

from monolish_log_viewer import aggregate, read

data_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../test_data/cg_iter.yml")

def test_layer_1_aggregated():
    with open(data_dir, "r") as file:
        dict_list = read.reader(file, "yaml")

    aggregate_dataframe = aggregate.AggregateDataFrame()
    layer1_aggr_df = aggregate_dataframe.layer_1_aggregated(dict_list)

    # colum check
    assert list(layer1_aggr_df.columns) == ["name", "layer", "time", "cnt"]

    # layer check
    assert layer1_aggr_df["layer"].sum() == len(layer1_aggr_df["layer"])

def test_aggregated_continuous_values():
    with open(data_dir, "r") as file:
        dict_list = read.reader(file, "yaml")

    test_dataframe = pandas.DataFrame(dict_list)
    test_dataframe = test_dataframe[["type", "name", "time", "stat"]]

    aggregate_dataframe = aggregate.AggregateDataFrame()
    aggr_cont_df = aggregate_dataframe.aggregated_continuous_values(test_dataframe)

    # colum check
    assert list(aggr_cont_df.columns) == ["cont_cnt", "group", "name", "stat", "time", "type"]

def test_aggregated():
    with open(data_dir, "r") as file:
        dict_list = read.reader(file, "yaml")

    aggregate_dataframe = aggregate.AggregateDataFrame()
    solve_df = aggregate_dataframe.aggregated(dict_list)

    # colum check
    assert list(solve_df.columns) == ["name", "layer", "group_0", "group_1", "group_2", "group_3", "time", "cont_cnt", "breakdown_0 layer1/layer0[%]", "breakdown_1 layer2/layer1[%]", "breakdown_2 layer3/layer2[%]", "breakdown_0[%] / count", "breakdown_1[%] / count", "breakdown_2[%] / count"]
