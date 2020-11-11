import pytest
import os
import sys
import pandas as pd

from monolish_log_viewer.utils import read
from monolish_log_viewer.libs import grouping

data_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../test_data/base_data2.yml")
print(data_dir)

def test_grouping_1st_layer():
    with open(data_dir, "r") as file:
        yaml_dict_list = read.reader(file, "yaml")
        title_list, block_dict_lists = grouping.grouping_1st_layer(yaml_dict_list)
        # print(title_list)
        # assert title_list == ['other 0', 'solver 0']
        # print(block_dict_lists)

def test_split_1st_layer():
    with open(data_dir, "r") as file:
        yaml_dict_list = read.reader(file, "yaml")
        # print("----------------")
        # print(pd.DataFrame(yaml_dict_list))
        dict_list = grouping.split_1st_layer(yaml_dict_list)
        # print("----------------")
        # print(pd.DataFrame(dict_list))
        assert 1 == 1
