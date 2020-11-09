import pytest
import os
import sys
import json
import yaml

# from ...utils import read
# from ....monolish_log_viewer.utils import read
from monolish_log_viewer.utils import read

# from unittest.mock import Mock
# from .utils import read
# from utils import read
# from .monolish_log_viewer.utils import read
# from monolish_log_viewer.utils import read
# from .python.monolish_log_viewer.utils import read
# from python.monolish_log_viewer.utils import read

data_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../test_data/base_data1.yml")

with open(data_dir, "r") as file:
    yaml_dict_list = read.reader(file, "xxx")
    assert yaml_dict_list == []