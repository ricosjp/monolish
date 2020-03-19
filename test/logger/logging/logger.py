#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import yaml
import numpy as np

def create_html(title, columns_list, aggr_ndarray):
	html_table = "<table border='1'>"
	# ----
	html_table += "<caption>"+title+"</caption>"
	# ----
	html_table += "<tr>"
	for column in columns_list:
		html_table += "<th>" + column + "</th>"
	html_table += "</tr>"
	# ----
	for aggr_narray in aggr_ndarray:
		html_table += "<tr>"
		for elem in aggr_narray:
			html_table += "<th>" + str(elem) + "</th>"
		html_table += "</tr>"
	# ----
	html_table += "</table>"
	return html_table

# io data
log_path = sys.argv[1]
out_path = sys.argv[2]

# read data
if(os.path.isfile(os.getcwd() +"/"+ str(log_path))):
	f = open(log_path, "r")
	yaml_dict_list = yaml.safe_load(f)
	print("read ok")

	# drop information
	## "solve/monolish_cg/monolish_jacobi/"
	target_dict_list = list(filter(lambda x:("solve/monolish_cg/monolish_jacobi/" not in x["name"]) or ("stat" in x), yaml_dict_list))
	target_dict_list = list(filter(lambda x:("solve/monolish_cg/monolish_jacobi/" not in x["name"]) or ("time" in x), target_dict_list))
	target_dict_list = list(filter(lambda x: x.pop("stat") if x["name"] == "solve/monolish_cg/monolish_jacobi/" else x, target_dict_list))

	# split type
	## other list
	other_dict_list = list(filter(lambda x:"solve" not in x["name"], target_dict_list))
	## solver list
	solver_dict_list = list(filter(lambda x:"solve" in x["name"], target_dict_list))
	is_list = list(map(lambda x:(("stat" in x) and x["stat"] == "IN" and x["name"] == "solve/"), solver_dict_list))
	split_index_list = [i for i, x in enumerate(is_list) if x == True] + [len(is_list)]
	solver_dict_block_list = [solver_dict_list[split_index_list[i]: split_index_list[i+1]] for i in range(len(split_index_list)-1)]
	## Per process list
	block_dict_lists = [other_dict_list] + solver_dict_block_list
	title_list = ["other"] + ["solver"+ str(i) for i in range(len(solver_dict_block_list))]

	# Aggregation
	html_table_list = []
	for index, block_dict_list in enumerate(block_dict_lists):
		block_dict_list = list(map(lambda block_dict: dict(list(block_dict.items())+[("stat", "")]) if ("stat" not in block_dict) else block_dict, block_dict_list))
		block_dict_list = list(map(lambda block_dict: dict(list(block_dict.items())+[("time", "")]) if ("time" not in block_dict) else block_dict, block_dict_list))
		# ----
		new_block_dict_list = []
		for block_dict in block_dict_list:
			af_temp_dict = {}
			af_temp_dict["type"] = block_dict["type"]
			af_temp_dict["name"] = block_dict["name"]
			af_temp_dict["stat"] = block_dict["stat"]
			af_temp_dict["time"] = block_dict["time"]
			new_block_dict_list.append(af_temp_dict)
		block_dict_list = new_block_dict_list

		# columns : type, name, stat, time
		block_ndarray = np.array([list(block_dict.values()) for block_dict in block_dict_list])
		layer_list = range(1, max(map(lambda x:x.count("/"), block_ndarray[:, 1]))+1)
		aggr_ndarray = np.empty((0, 4))
		for i in layer_list:
			layer_ndarray = block_ndarray[np.array(list(map(lambda x: (x[1]!="IN") and (x[0].count("/")==i), block_ndarray[:, 1:3])))][:, [1,3]]
			for col in np.unique(layer_ndarray[:,0]):
				temp_ndarray = layer_ndarray[np.array(list(map(lambda x: x==col, layer_ndarray[:, 0])))]
				count = np.count_nonzero(temp_ndarray[:,0])
				total_time = np.sum(np.array(temp_ndarray[:,1], dtype="float32"))
				rst_narray = np.array([i, col, count, total_time])
				aggr_ndarray=np.append(aggr_ndarray, [rst_narray], axis=0)
		
		aggr_column_list = ["layer", "name", "count", "total_time [s]"]
		for i in range(1, max(map(lambda x:x.count("/"), block_ndarray[:, 1]))):
			percent = np.array(aggr_ndarray[:, 3], dtype="float32") / (float)(aggr_ndarray[np.array(list(map(lambda x: int(x[0])==i, aggr_ndarray)))][:, 3][0]) * 100.0
			percent = np.round(percent, decimals=3)
			percent = np.where(percent <= 100.0 , percent, "")
			aggr_ndarray = np.insert(aggr_ndarray, aggr_ndarray.shape[1], percent, axis=1)
			aggr_column_list.append("breakdown_layer"+str(i)+" [%]")
		aggr_ndarray[:, 3] = np.round(np.array(aggr_ndarray[:, 3], dtype="float32"), decimals=3)

		# create html table
		html_table = create_html(title_list[index], aggr_column_list, aggr_ndarray)
		html_table_list.append(html_table)

	# create html
	html_tables = ','.join(html_table_list)
	html_tables = html_tables.replace(",", "")

	# htmlへのcontent埋め込みと、html作成
	html = f"""
		<!DOCTYPE html>
			<html lang="ja">
			<head>
				<meta charset="utf-8">
			</head>
			<body>
				{html_tables}
			</body>
		</html>
	"""

	# write html
	with open(out_path, 'wb') as file:
		file.write(html.encode("utf-8"))
	print("pass")

else:
	print("load error: The specified file was not found.")

