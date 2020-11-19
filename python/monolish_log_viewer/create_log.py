"""Create log program."""

from . import debug, html, read, aggregate, grouping

def execute_create_log(log_path, out_path):
    """executive function"""
    # io data
    with open(log_path, "r") as file:
        yaml_dict_list = read.reader(file, "yaml")
    debug.log_success(f"read {format(log_path)}")

    # layer 1
    aggregate_dataframe = aggregate.AggregateDataFrame()
    layer_1_aggr_df = aggregate_dataframe.layer_1_aggregated(yaml_dict_list)
    debug.log_success("layer_1_aggregated")

    # split block
    split_dict_list = grouping.split_1st_layer(yaml_dict_list)

    # Aggregate
    solve_df = aggregate_dataframe.aggregated(split_dict_list)
    debug.log_success("aggregated")

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
        write_number_of_character = file.write(text_html.encode("utf-8"))
    debug.log_success(f"write {format(out_path)}")
    debug.log_success(f"number of character {write_number_of_character}")

    return write_number_of_character
