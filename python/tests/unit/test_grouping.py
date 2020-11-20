""" test grouping """
import os
from monolish_log_viewer import grouping, read

data_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../test_data/cg_iter.yml")
print(data_dir)

def test_grouping_1st_layer():
    """ test grouping_1st_layer """
    with open(data_dir, "r") as file:
        yaml_dict_list = read.reader(file, "yaml")
    title_list, block_dict_lists = grouping.grouping_1st_layer(yaml_dict_list)

    assert title_list == [
        "other 0", "other 1", "other 2", "other 3", "other 4",
        "other 5", "other 6", "other 7", "other 8", "other 9",
        "other 10", "other 11", "other 12", "other 13",
        "solver 0", "solver 1", "solver 2", "solver 3", "solver 4",
        "solver 5", "solver 6", "solver 7", "solver 8", "solver 9",
        "solver 10", "solver 11", "solver 12", "solver 13"]

    assert block_dict_lists != []

def test_split_1st_layer():
    """ test split_1st_layer """
    with open(data_dir, "r") as file:
        yaml_dict_list = read.reader(file, "yaml")
    dict_list = grouping.split_1st_layer(yaml_dict_list)

    assert dict_list != []
