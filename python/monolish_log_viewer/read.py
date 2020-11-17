""" read data """
import json
import yaml

def reader(file_object, file_extension:str) -> list:
    """read data"""
    if file_extension == "yaml":
        dict_list = yaml.safe_load(file_object)
    elif file_extension == "json":
        dict_list = json.load(file_object)
    else:
        dict_list = []

    return dict_list
