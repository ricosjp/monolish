#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import pandas as pd
from utils import debug, html as html_module, read
from libs import aggregate, drop_information, grouping

def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("out_path")

    args = parser.parse_args()
    log_path = args.log_path
    out_path = args.out_path

    # io data
    with open(log_path, "r") as f:
        io_data = read.IOData()
        yaml_dict_list = io_data.reader(f, "yaml")
        # print(yaml_dict_list)
        debug.log_success(f"read {format(log_path)}")

        # # drop information
        # drop_dir = "solve/monolish_cg/monolish_jacobi/"
        # target_dict_list = drop_information.drop_dict(drop_dir, yaml_dict_list)
        # debug.log_success("drop information")

        # # drop dir info
        # target_dict_list = drop_information.drop_dir_info(target_dict_list)
        # debug.log_success("drop dir info")

        # # 1st layer type
        # title_list, block_dict_lists = grouping.grouping_1st_layer(target_dict_list)
        # debug.log_success("1st layer type")

        # # aggregate
        # aggregate_numpy = aggregate.AggregateNumpy()
        # aggr_column_lists, aggr_ndarrays, index = aggregate_numpy.aggregated_by_floor(block_dict_lists)
        # debug.log_success("aggregate")

        # # create html
        # html_table_list = []
        # for i in range(index):
        #     html_table = html_module.create_table(title_list[i], aggr_column_lists[i], aggr_ndarrays[i])
        #     html_table_list.append(html_table)
        # html = html_module.create_html(html_table_list)
        # debug.log_success("create html")

        # Aggregate
        aggregate_pandas = aggregate.AggregatePandas()
        df = aggregate_pandas.aggregated(yaml_dict_list)

        # create html
        table_html = df.to_html()
        html = f"""
            <!DOCTYPE html>
                <html lang="ja">
                <head>
                    <meta charset="utf-8">
                </head>
                <body>
                    {table_html}
                </body>
            </html>
        """

        # write html
        with open(out_path, 'wb') as file:
            file.write(html.encode("utf-8"))
            debug.log_success(f"write {format(out_path)}")

if __name__ == "__main__":
    main()