""" Drop unnecessary information """

def drop_dict(directory, dict_list):
    """drop record"""
    target_dict_list = dict_list
    target_dict_list = list(filter(
        lambda any_dict:(directory not in any_dict["name"]) or ("stat" in any_dict),
        target_dict_list))
    target_dict_list = list(filter(
        lambda any_dict:(directory not in any_dict["name"]) or ("time" in any_dict),
        target_dict_list))
    target_dict_list = list(filter(
        lambda any_dict:any_dict.pop("stat") if any_dict["name"] == directory else any_dict,
        target_dict_list))

    return target_dict_list

def drop_dir_info(target_dict_list):
    """drop process"""
    min_layer = min(map(
        lambda any_dict:any_dict["name"].count("/"),
        target_dict_list))
    min_dict_list = list(filter(
        lambda any_dict:any_dict["name"].count("/") == min_layer,
        target_dict_list))
    min_dict = min_dict_list[0]
    min_dir = min_dict["name"]
    drop_list = min_dir.split("/")
    drop_dir_list = drop_list[:-2]
    drop_dir_list = [] if drop_dir_list == [] else drop_dir_list + [""]
    drop_dir_text = ",".join(drop_dir_list)
    drop_dir_text = drop_dir_text.replace(",", "/")

    temp_dict_list = []
    for any_dict in target_dict_list:
        any_dict["name"] = any_dict["name"].replace(drop_dir_text, "")
        temp_dict_list.append(any_dict)
    target_dict_list = temp_dict_list

    return target_dict_list
