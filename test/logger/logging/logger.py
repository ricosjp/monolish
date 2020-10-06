#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import datetime

# logger
class Logger:
    def log_general(self, message) -> str:
        dt_now = datetime.datetime.now()
        print(f"[{dt_now}] {message}")

    def log_success(self, message) -> str:
        dt_now = datetime.datetime.now()
        print(f"[{dt_now}] success {message}")

    def log_error(self, message) -> str:
        dt_now = datetime.datetime.now()
        print(f"[{dt_now}] error {message}")

    def dump(self, dict_list) -> str:
        dt_now = datetime.datetime.now()
        print(f"[{dt_now}] dump dict")
        for one_dict in dict_list:
            print(f"{one_dict}")

# HTML Class
class CreateHTML:
    def __init__(self, html_tables):
        self.html_tables = []

    def create_table(self, title, columns_list, aggr_ndarray) -> str:
        # caption
        caption = f"<caption>{title}</caption>"
        # table header
        column_html = "".join(list(map(lambda column:f"<th>{column}</th>", columns_list)))
        table_header = f"<tr>{column_html}</tr>"
        # table data
        table_data = "".join(list(map(lambda aggr_narray:"<tr>" + "".join(list(map(lambda elem:f'<td>{str(elem)}</td>', aggr_narray))) + "</tr>", aggr_ndarray)))
        # table
        table = caption + table_header + table_data
        table = f"<table border='1'>{table}</table>"
        return table

    def create_html(self, html_tables) -> str:
        # join html tables
        html_tables = ','.join(html_table_list)
        html_tables = html_tables.replace(",", "")
        # create html
        html = f"""
            <!DOCTYPE html>
                <html lang="ja">
                <head>
                    <meta charset="utf-8">
                </head>
                <body>
                    {html_tables}
                </body>
            </html>
        """
        return html

# Drop Information Class
class DropInformation:
    def drop_dict(self, directory, dict_list):
        target_dict_list = list(filter(lambda x:(directory not in x["name"]) or ("stat" in x), dict_list))
        target_dict_list = list(filter(lambda x:(directory not in x["name"]) or ("time" in x), target_dict_list))
        target_dict_list = list(filter(lambda x: x.pop("stat") if x["name"] == directory else x, target_dict_list))
        return target_dict_list

    def drop_dir_info(self, target_dict_list):
        min_layer = min(map(lambda any_dict:any_dict["name"].count("/"), target_dict_list))
        min_dict_list = list(filter(lambda any_dict:any_dict["name"].count("/") == min_layer, target_dict_list))
        min_dict = min_dict_list[0]
        min_dir = min_dict["name"]
        drop_list = min_dir.split("/")
        drop_dir_text = ",".join(drop_list[:-2])
        drop_dir_text = drop_dir_text.replace(",", "/")
        drop_dir_text = drop_dir_text + "/"

        temp_dict_list = []
        for any_dict in target_dict_list:
            any_dict["name"] = any_dict["name"].replace(drop_dir_text, "")
            temp_dict_list.append(any_dict)
        target_dict_list = temp_dict_list
        return target_dict_list

# class Split1stLayer:
class Grouping:

    def grouping_1st_layer(self, target_dict_list):
        solver_dict_list = list(filter(lambda x:"solve/" in x["name"], target_dict_list))
        other_dict_list = list(filter(lambda x:"solve/" not in x["name"], target_dict_list))

        filter_list = list(map(lambda x:(("stat" in x) and x["stat"] == "IN" and x["name"] == "solve/"), solver_dict_list))
        split_index_list = [i for i, x in enumerate(filter_list) if x == True] + [len(filter_list)]
        solver_dict_block_list = [solver_dict_list[split_index_list[i]: split_index_list[i+1]] for i in range(len(split_index_list)-1)]

        block_dict_lists = [other_dict_list] + solver_dict_block_list
        title_list = ["other"] + [f"solver {str(i)}" for i in range(len(solver_dict_block_list))]
        return title_list, block_dict_lists

# Aggregate Class
"""
Classes to represent the definitions of aggregate functions.
"""
class Aggregate:
    def __init__(self):
        self.aggr_column_lists = []
        self.aggr_ndarrays = []
        self.index = 0

    def aggregated_by_floor(self, block_dict_lists):
        import numpy as np

        aggr_column_lists, aggr_ndarrays = [], []
        block_dict_lists = filter(lambda x: x != [], block_dict_lists)
        for index, block_dict_list in enumerate(block_dict_lists):
            print(f"loop {index}")
            block_dict_list = list(map(lambda block_dict: dict(list(block_dict.items())+[("stat", "")]) if ("stat" not in block_dict) else block_dict, block_dict_list))
            block_dict_list = list(map(lambda block_dict: dict(list(block_dict.items())+[("time", "")]) if ("time" not in block_dict) else block_dict, block_dict_list))
            # sorted
            block_dict_list = list(map(lambda block_dict: dict(type=block_dict["type"], name=block_dict["name"], stat=block_dict["stat"], time=block_dict["time"]), block_dict_list))

            # columns : type, name, stat, time
            block_ndarray = np.array([list(block_dict.values()) for block_dict in block_dict_list])
            max_layer = max(map(lambda x:x.count("/"), block_ndarray[:, 1]))
            print(max_layer)

            aggr_ndarray = np.empty((0, 4))
            for layer in range(1, max_layer+1):
                layer_ndarray = block_ndarray[np.array(list(map(lambda x: (x[1]!="IN") and (x[0].count("/")==layer), block_ndarray[:, 1:3])))][:, [1,3]]
                for col in np.unique(layer_ndarray[:,0]):
                    temp_ndarray = layer_ndarray[np.array(list(map(lambda x: x==col, layer_ndarray[:, 0])))]
                    count = np.count_nonzero(temp_ndarray[:,0])
                    total_time = np.sum(np.array(temp_ndarray[:,1], dtype="float32"))
                    rst_narray = np.array([layer, col, count, total_time])
                    aggr_ndarray = np.append(aggr_ndarray, [rst_narray], axis=0)
            
            aggr_column_list = ["layer", "name", "count", "total_time [s]"]
            for layer in range(1, max_layer):
                denominator = (float)(aggr_ndarray[np.array(list(map(lambda x: int(x[0])==layer, aggr_ndarray)))][:, 3][0])
                percent = np.array(aggr_ndarray[:, 3], dtype="float32") / denominator * 100.0
                percent = np.round(percent, decimals=3)
                percent = np.where(percent <= 100.0, percent, "")
                aggr_ndarray = np.insert(aggr_ndarray, aggr_ndarray.shape[1], percent, axis=1)
                aggr_column_list.append(f"breakdown_layer {str(layer)} [%]")
            aggr_ndarray[:, 3] = np.round(np.array(aggr_ndarray[:, 3], dtype="float32"), decimals=3)
            # print(aggr_ndarray)

            # aggregate list
            aggr_column_lists.append(aggr_column_list)
            aggr_ndarrays.append(aggr_ndarray)

        else:
            index = index + 1

        return aggr_column_lists, aggr_ndarrays, index

# i/o data Class
class IOData:
    def __init__(self):
        self.aggr_column_lists = []
        self.aggr_ndarrays = []
        self.index = 0

    def reader(self, file_object, file_extension):
        if file_extension == "yaml":
            import yaml
            dict_list = yaml.safe_load(file_object)
        elif file_extension == "json":
            import json
            dict_list = json.load(file_object)
        else:
            dict_list = []
        
        return dict_list

# io data
log_path = sys.argv[1]
out_path = sys.argv[2]

logger = Logger()
# data dir
try:
    # read data
    with open(log_path, "r") as f:
        io_data = IOData()
        yaml_dict_list = io_data.reader(f, "yaml")
        logger.log_success(f"read {format(log_path)}")

        # drop information
        drop_dir = "solve/monolish_cg/monolish_jacobi/"
        drop_information = DropInformation()
        target_dict_list = drop_information.drop_dict(drop_dir, yaml_dict_list)
        logger.log_success("drop information")

        # drop dir info
        target_dict_list = drop_information.drop_dir_info(target_dict_list)
        logger.log_success("drop dir infon")

        # 1st layer type
        grouping = Grouping()
        title_list, block_dict_lists = grouping.grouping_1st_layer(target_dict_list)
        logger.log_success("1st layer type")
        # logger.dump(block_dict_lists)

        # aggregate
        aggregate = Aggregate()
        aggr_column_lists, aggr_ndarrays, index = aggregate.aggregated_by_floor(block_dict_lists)
        logger.log_success("aggregate")

        # create html
        html_table_list = []
        create_html = CreateHTML(html_table_list)
        for i in range(index):
            html_table = create_html.create_table(title_list[i], aggr_column_lists[i], aggr_ndarrays[i])
            html_table_list.append(html_table)
        html = create_html.create_html(html_table_list)
        logger.log_success("create html")

        # write html
        try:
            with open(out_path, 'wb') as file:
                file.write(html.encode("utf-8"))
                logger.log_success(f"write {format(out_path)}")
        except FileNotFoundError as e:
            logger.log_error("write: The specified file was not found.")
        except Exception as e:
            logger.log_general(f"{e}")

except FileNotFoundError as e:
    logger.log_error("load: The specified file was not found.")

except Exception as e:
    logger.log_general(f"{e}")