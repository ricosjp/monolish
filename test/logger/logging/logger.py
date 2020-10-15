#!/usr/bin/env python3
# coding: utf-8

import sys
import os
from utils import debug, html, read
from libs import aggregate, drop_information, grouping

# io data
log_path = sys.argv[1]
out_path = sys.argv[2]

# data dir
try:
    # read data
    with open(log_path, "r") as f:
        io_data = read.IOData()
        yaml_dict_list = io_data.reader(f, "yaml")
        debug.log_success(f"read {format(log_path)}")

        # drop information
        drop_dir = "solve/monolish_cg/monolish_jacobi/"
        target_dict_list = drop_information.drop_dict(drop_dir, yaml_dict_list)
        debug.log_success("drop information")

        # drop dir info
        target_dict_list = drop_information.drop_dir_info(target_dict_list)
        debug.log_success("drop dir info")

        # 1st layer type
        grouping = grouping.Grouping()
        title_list, block_dict_lists = grouping.grouping_1st_layer(target_dict_list)
        debug.log_success("1st layer type")

        # aggregate
        # aggregate = aggregate.Aggregate()
        aggr_column_lists, aggr_ndarrays, index = aggregate.aggregated_by_floor(block_dict_lists)
        debug.log_success("aggregate")

        # create html
        html_table_list = []
        for i in range(index):
            html_table = html.create_table(title_list[i], aggr_column_lists[i], aggr_ndarrays[i])
            html_table_list.append(html_table)
        html = html.create_html(html_table_list)
        debug.log_success("create html")

        # write html
        try:
            with open(out_path, 'wb') as file:
                file.write(html.encode("utf-8"))
                debug.log_success(f"write {format(out_path)}")
        except FileNotFoundError as e:
            debug.log_error("write: The specified file was not found.")
        except Exception as e:
            debug.log_general(f"{e}")

except FileNotFoundError as e:
    debug.log_error("load: The specified file was not found.")

except Exception as e:
    debug.log_general(f"{e}")