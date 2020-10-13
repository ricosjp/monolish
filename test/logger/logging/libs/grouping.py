#!/usr/bin/env python3
# coding: utf-8

class Grouping:
    def grouping_1st_layer(self, target_dict_list):
        # solve
        layer_1st = "solve/"
        solver_dict_list = list(filter(lambda any_dict:layer_1st in any_dict["name"], target_dict_list))
        filter_list = list(map(lambda any_dict:(("stat" in any_dict) and any_dict["stat"] == "IN" and any_dict["name"] == layer_1st), solver_dict_list))
        split_index_list = [index for index, value in enumerate(filter_list) if value == True] + [len(filter_list)]
        solver_dict_block_list = [solver_dict_list[split_index_list[index]: split_index_list[index+1]] for index in range(len(split_index_list)-1)]

        # other
        other_dict_list = list(filter(lambda any_dict:layer_1st not in any_dict["name"], target_dict_list))
        if other_dict_list:
            layer_list = list(map(lambda any_dict:any_dict["name"].split("/")[0], other_dict_list))
            layer_list = list(set(layer_list))
            layer_list = list(map(lambda x:x+"/", layer_list))
            for other_layer_1st in layer_list:
                filter_list = list(map(lambda any_dict:any_dict["name"] == other_layer_1st, other_dict_list))
                split_index_list = [index for index, value in enumerate(filter_list) if value == True] + [len(filter_list)]
                other_dict_block_list = [other_dict_list[split_index_list[index]: split_index_list[index+1]] for index in range(len(split_index_list)-1)]
        else:
            other_dict_block_list = []

        block_dict_lists = other_dict_block_list + solver_dict_block_list
        title_list = [f"other {str(no)}" for no in range(len(other_dict_block_list))] + [f"solver {str(no)}" for no in range(len(solver_dict_block_list))]
        return title_list, block_dict_lists