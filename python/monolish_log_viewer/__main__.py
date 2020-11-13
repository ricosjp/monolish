"""Create log program."""
# coding: utf-8
import argparse

from .utils import debug, html, read
from .libs import aggregate, grouping

def controll_argument():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("out_path")

    args = parser.parse_args()
    log_path = args.log_path
    out_path = args.out_path

    return log_path, out_path

def execute_create_log(log_path, out_path):
    """executive function"""
    # io data
    with open(log_path, "r") as file:
        yaml_dict_list = read.reader(file, "yaml")
        debug.log_success(f"read {format(log_path)}")

        # split block
        split_dict_list = grouping.split_1st_layer(yaml_dict_list)

        # Aggregate
        aggregate_dataframe = aggregate.AggregateDataFrame()
        solve_df = aggregate_dataframe.aggregated(split_dict_list)
        debug.log_success("aggregated")

        # layer 1
        layer_1_aggr_df = aggregate_dataframe.layer_1_aggregated(yaml_dict_list)
        debug.log_success("layer_1_aggregated")

        # create html
        layer_1_aggr_table_html = html.df_to_html_table(layer_1_aggr_df)
        layer_1_aggr_table_html = html.to_caption_on_html("layer1", layer_1_aggr_table_html)

        solve_table_html = html.df_to_html_table(solve_df)
        solve_table_html = html.to_caption_on_html("solver", solve_table_html)

        all_table_html = layer_1_aggr_table_html + solve_table_html

        # decoration
        all_table_html = html.to_bold_on_html(all_table_html)
        text_html = html.table_in_html(all_table_html)
        debug.log_success("html")

        # write html
        with open(out_path, 'wb') as file:
            file.write(text_html.encode("utf-8"))
            debug.log_success(f"write {format(out_path)}")

def main():
    log_path, out_path = controll_argument()
    execute_create_log(log_path, out_path)

if __name__ == "__main__":
    main()
