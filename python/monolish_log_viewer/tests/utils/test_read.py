import os
import json
import yaml
from .monolish_log_viewer.utils import read

data_dir = os.getcwd().replace("/utils", "/test_data/base_data1.yml")
# print(data_dir)

with open(data_dir, "r") as file:
    yaml_dict_list = read.reader(file, "xxx")
    assert yaml_dict_list == []