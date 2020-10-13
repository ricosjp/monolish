#!/usr/bin/env python3
# coding: utf-8
import numpy as np

class Aggregate:
    """
    Classes to represent the definitions of aggregate functions.
    """
    def __init__(self):
        self.aggr_column_lists = []
        self.aggr_ndarrays = []
        self.index = 0

    def aggregated_by_floor(self, block_dict_lists):
        aggr_column_lists, aggr_ndarrays = [], []
        block_dict_lists = filter(lambda x: x != [], block_dict_lists)
        for index, block_dict_list in enumerate(block_dict_lists):
            block_dict_list = list(map(lambda block_dict: dict(list(block_dict.items())+[("stat", "")]) if ("stat" not in block_dict) else block_dict, block_dict_list))
            block_dict_list = list(map(lambda block_dict: dict(list(block_dict.items())+[("time", "")]) if ("time" not in block_dict) else block_dict, block_dict_list))
            # sorted
            block_dict_list = list(map(lambda block_dict: dict(type=block_dict["type"], name=block_dict["name"], stat=block_dict["stat"], time=block_dict["time"]), block_dict_list))

            # columns : type, name, stat, time
            block_ndarray = np.array([list(block_dict.values()) for block_dict in block_dict_list])
            max_layer = max(map(lambda x:x.count("/"), block_ndarray[:, 1]))

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

            # aggregate list
            aggr_column_lists.append(aggr_column_list)
            aggr_ndarrays.append(aggr_ndarray)

        else:
            index = index + 1

        return aggr_column_lists, aggr_ndarrays, index
