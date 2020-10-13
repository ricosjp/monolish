#!/usr/bin/env python3
# coding: utf-8

import yaml
import json

class IOData:
    def __init__(self):
        self.aggr_column_lists = []
        self.aggr_ndarrays = []
        self.index = 0

    def reader(self, file_object, file_extension):
        if file_extension == "yaml":
            dict_list = yaml.safe_load(file_object)
        elif file_extension == "json":
            dict_list = json.load(file_object)
        else:
            dict_list = []
        
        return dict_list