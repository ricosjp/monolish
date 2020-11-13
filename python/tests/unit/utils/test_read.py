import pytest
import os
import sys
import json
import yaml

from monolish_log_viewer.utils import read

data_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../test_data/base_data1.yml")

def test_read_unown_data():
    with open(data_dir, "r") as file:
        dict_list = read.reader(file, "xxx")
        assert dict_list == []

def test_read_normal_data():
    with open(data_dir, "r") as file:
        yaml_dict_list = read.reader(file, "yaml")
        assert isinstance(yaml_dict_list, list) == True
