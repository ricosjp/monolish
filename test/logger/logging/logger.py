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
        debug.log_success(f"read {format(log_path)}")

        # Aggregate
        aggregate_pandas = aggregate.AggregatePandas()
        df = aggregate_pandas.aggregated(yaml_dict_list)
        debug.log_success("aggregated")

        # create html
        table_html = html_module.df_to_html_table(df)
        html = html_module.table_in_html(table_html)
        debug.log_success("html")

        # write html
        with open(out_path, 'wb') as file:
            file.write(html.encode("utf-8"))
            debug.log_success(f"write {format(out_path)}")

if __name__ == "__main__":
    main()