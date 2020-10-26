#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
import pandas as pd
from utils import debug, html as html, read
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

        # split block
        split_dict_list = grouping.split_1st_layer(yaml_dict_list)

        # Aggregate
        aggregate_pandas = aggregate.AggregatePandas()
        other_df, solve_df = aggregate_pandas.aggregated(split_dict_list)
        debug.log_success("aggregated")

        # create html
        other_table_html = html.df_to_html_table(other_df)
        other_table_html = html.to_caption_on_html("other", other_table_html)

        solve_table_html = html.df_to_html_table(solve_df)
        solve_table_html = html.to_caption_on_html("solver", solve_table_html)

        all_table_html = other_table_html + solve_table_html

        # decoration
        all_table_html = html.to_bold_on_html(all_table_html)
        text_html = html.table_in_html(all_table_html)
        debug.log_success("html")

        # write html
        with open(out_path, 'wb') as file:
            file.write(text_html.encode("utf-8"))
            debug.log_success(f"write {format(out_path)}")

if __name__ == "__main__":
    main()