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
        assert max(aggregated_df["group_0"]) == len(dataframe[dataframe["name"] == "solve/"])/2

def test_aggregated_continuous_values():
    """test_aggregated_continuous_values
        連続で呼び出される関数の集計テスト
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

        test_dataframe = pandas.DataFrame(dict_list)
        test_dataframe = test_dataframe[["type", "name", "time", "stat"]]

        aggregate_dataframe = aggregate.AggregateDataFrame()
        aggr_cont_df = aggregate_dataframe.aggregated_continuous_values(test_dataframe)

        # colum check
        assert list(aggr_cont_df.columns) == ["cont_cnt", "group", "name", "stat", "time", "type"]

def test_aggregated():
    """test aggregated
        集約値のカラムを確認する
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

        with open(data_dir, "r") as file:
            dict_list = read.reader(file, "yaml")

        # test data
        dataframe = pandas.DataFrame(dict_list)
        max_group_num = max(dataframe["name"].apply(lambda name:name.count("/")))
        check_cols = ["name", "layer"]
        check_cols = check_cols + [f"group_{i}" for i in range(max_group_num)]
        check_cols = check_cols + ["time", "cont_cnt"]
        loop_range = range(max_group_num-1)
        check_cols = check_cols + [f"breakdown_{i} layer{i+1}/layer{i}[%]" for i in loop_range]
        check_cols = check_cols + [f"breakdown_{i}[%] / count" for i in loop_range]

        # test method
        aggregate_dataframe = aggregate.AggregateDataFrame()
        solve_df = aggregate_dataframe.aggregated(dict_list)

        # colum check
        assert list(solve_df.columns) == check_cols

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
