""" read 機能のテスト """
import os

from monolish_log_viewer import read

data_dir = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + "/../test_data/only_solver.yml")

def test_read_unknown_data():
    """ 指定拡張子以外のデータ """
    with open(data_dir, "r") as file:
        dict_list = read.reader(file, "xxx")
    assert dict_list == []

def test_read_normal_data():
    """ 指定拡張子のデータ """
    with open(data_dir, "r") as file:
        yaml_dict_list = read.reader(file, "yaml")
    assert isinstance(yaml_dict_list, list) is True
