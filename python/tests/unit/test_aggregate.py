""" test_aggregate
    集約処理のテスト
"""
import os
import pandas

from monolish_log_viewer import aggregate, read

def test_layer_1_aggregated():
    """test_layer_1_aggregated
        1層目の集計テーブルのカラムとデータが抽出出来ているかのテスト
    """
    data_list = [
        "only_solver",
        "normal_data",
        "normal_data_order",
        "only_other",
        "cg_iter"
        ]

    for file_name in data_list:
        data_path = f"/../test_data/{file_name}.yml"
        data_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + data_path)
        # read test data
        with open(data_dir, "r") as file:
            dict_list = read.reader(file, "yaml")

        # test method
        aggregate_dataframe = aggregate.AggregateDataFrame()
        layer1_aggr_df = aggregate_dataframe.layer_1_aggregated(dict_list)

        # column check
        assert list(layer1_aggr_df.columns) == ["name", "layer", "time", "cnt"]

        # layer check
        assert layer1_aggr_df["layer"].sum() == len(layer1_aggr_df["layer"])

def test_all_block_solver():
    """test_all_block_solver
        すべてのsolverの処理を表示できているか確認
    """
    data_list = [
        "only_solver",
        "normal_data",
        "normal_data_order",
        "cg_iter"
        ]

    for file_name in data_list:
        data_path = f"/../test_data/{file_name}.yml"
        data_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + data_path)


        # read test data
        with open(data_dir, "r") as file:
            dict_list = read.reader(file, "yaml")

        # test data
        dataframe = pandas.DataFrame(dict_list)

        # test method
        aggregate_dataframe = aggregate.AggregateDataFrame()
        aggregated_df = aggregate_dataframe.aggregated(dict_list)

        # check max block num
        assert max(aggregated_df["group"]) == len(dataframe[dataframe["name"] == "solve/"])/2


def test_create_preprocessing_table():
    """test_create_preprocessing_table
        集計できる状態か確認する(timeにnanが入っていないか確認)
    """
    data_list = [
        "only_solver",
        "normal_data",
        "normal_data_order",
        "cg_iter"
        ]

    for file_name in data_list:
        data_path = f"/../test_data/{file_name}.yml"
        data_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + data_path)


        # read test data
        with open(data_dir, "r") as file:
            dict_list = read.reader(file, "yaml")

        # test data
        dataframe = pandas.DataFrame(dict_list)
        aggr_col_list = ["type", "name", "time", "stat"]
        dataframe = pandas.DataFrame(dict_list, columns=aggr_col_list)
        dataframe = dataframe[aggr_col_list]
        dataframe["layer"] = dataframe["name"].str.count("/")

        # test method
        aggregate_dataframe = aggregate.AggregateDataFrame()
        aggregated_df = aggregate_dataframe.create_preprocessing_table(dataframe)

        # check max block num
        assert aggregated_df["time"].isnull().count() == aggregated_df["time"].count()

def test_aggregated_column_sum():
    """test_aggregated_column_sum
        誤差がないデータで合計を確認する
    """
    data_path = "/../test_data/no_error.yml"
    data_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + data_path)

    # read test data
    with open(data_dir, "r") as file:
        dict_list = read.reader(file, "yaml")

    # test method
    aggregate_dataframe = aggregate.AggregateDataFrame()
    solve_df = aggregate_dataframe.aggregated(dict_list)

    for i in range(1, max(solve_df["layer"])):
        left_side = sum(solve_df[solve_df["layer"] == i]["time"])
        right_side = sum(solve_df[solve_df["layer"] == i+1]["time"])
        error_val = left_side - right_side
        assert abs(error_val) < 0.000000001
