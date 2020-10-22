import math
import numpy as np
import pandas as pd

class AggregateNumpy:
    def aggregated_by_floor(self, block_dict_2multiple_list):
        aggr_column_lists, aggr_ndarrays = [], []
        block_dict_2multiple_list = filter(lambda any_dict_list: any_dict_list != [], block_dict_2multiple_list)
        for index, block_dict_list in enumerate(block_dict_2multiple_list):
            block_dict_list = list(map(lambda block_dict: dict(list(block_dict.items())+[("stat", "")]) if ("stat" not in block_dict) else block_dict, block_dict_list))
            block_dict_list = list(map(lambda block_dict: dict(list(block_dict.items())+[("time", "")]) if ("time" not in block_dict) else block_dict, block_dict_list))
            # sorted
            block_dict_list = list(map(lambda block_dict: dict(type=block_dict["type"], name=block_dict["name"], stat=block_dict["stat"], time=block_dict["time"]), block_dict_list))

            # columns : type, name, stat, time
            block_ndarray = np.array([list(block_dict.values()) for block_dict in block_dict_list])
            max_layer = max(map(lambda x:x.count("/"), block_ndarray[:, 1]))

            # group by
            aggr_ndarray = groupby_with_name(block_ndarray, max_layer)

            # culc_percent
            aggr_column_list = ["layer", "name", "count", "total_time [s]"]
            aggr_ndarray, aggr_column_list = culc_percent(aggr_ndarray, aggr_column_list, max_layer)

            # erase information
            aggr_ndarray = erase_information(aggr_ndarray, max_layer)

            # aggregate list
            aggr_column_lists.append(aggr_column_list)
            aggr_ndarrays.append(aggr_ndarray)

        else:
            index = index + 1

        return aggr_column_lists, aggr_ndarrays, index

    def culc_percent(self, aggr_ndarray, aggr_column_list, max_layer):
        for layer in range(1, max_layer):
            aggr_column_list.append(f"breakdown_layer {str(layer)} [%] (breakdown[%] / count)")

            denominator = (float)(aggr_ndarray[np.array(list(map(lambda x: int(x[0])==layer, aggr_ndarray)))][:, 3][0])
            percent = np.array(aggr_ndarray[:, 3], dtype="float32") / denominator * 100.0
            percent = np.round(percent, decimals=3)
            percent = np.where(percent <= 100.0, percent, "")

            # breakdown[%] / count
            # percent_zero_fill = np.where(percent != "", percent, 0.0)
            # percent_zero_fill = np.array(percent_zero_fill, dtype='float32')
            # count = np.array(aggr_ndarray[:, 2], dtype="float32")
            # percent_divided_by_count = percent_zero_fill / count

            aggr_ndarray = np.insert(aggr_ndarray, aggr_ndarray.shape[1], percent, axis=1)
        aggr_ndarray[:, 3] = np.round(np.array(aggr_ndarray[:, 3], dtype="float32"), decimals=3)

        return aggr_ndarray, aggr_column_list

    def erase_information(self, aggr_ndarray, max_layer):
        temp_aggr_ndarray = np.empty((0, aggr_ndarray.shape[1]))
        for layer in range(1, max_layer-1):
            for any_ndarray in aggr_ndarray:
                if any_ndarray[0] == str(layer) or any_ndarray[0] == str(layer+1):
                    temp_aggr_ndarray = np.append(temp_aggr_ndarray, [any_ndarray], axis=0)
                else:
                    temp_ndarray = np.copy(any_ndarray)
                    temp_ndarray[3+layer] = ""
                    temp_aggr_ndarray = np.append(temp_aggr_ndarray, [temp_ndarray], axis=0)
            aggr_ndarray = temp_aggr_ndarray
        return aggr_ndarray

    def groupby_with_name(self, block_ndarray, max_layer):
        aggr_ndarray = np.empty((0, 4))
        for layer in range(1, max_layer+1):
            layer_ndarray = block_ndarray[np.array(list(map(lambda x: (x[1]!="IN") and (x[0].count("/")==layer), block_ndarray[:, 1:3])))][:, [1,3]]
            for col in np.unique(layer_ndarray[:,0]):
                temp_ndarray = layer_ndarray[np.array(list(map(lambda x: x==col, layer_ndarray[:, 0])))]
                count = np.count_nonzero(temp_ndarray[:,0])
                total_time = np.sum(np.array(temp_ndarray[:,1], dtype="float32"))
                rst_narray = np.array([layer, col, count, total_time])
                aggr_ndarray = np.append(aggr_ndarray, [rst_narray], axis=0)
        return aggr_ndarray

class AggregatePandas:
    def aggregated(self, dict_list):
        # dict_list to list
        df = pd.DataFrame(dict_list)

        # add flg
        any_group_df = df
        any_group_df = any_group_df[["type", "name", "time", "stat"]]
        any_group_df["layer"] = any_group_df.name.apply(lambda x:x.count("/")-1)

        max_layer = max(any_group_df.layer+1)
        for any_layer in range(max_layer):
            for index, row in any_group_df.iterrows():
                row[f"layer_{any_layer}_flg"] = 1 if row.layer == any_layer else np.nan
                any_group_df.loc[index, f"layer_{any_layer}_flg"] = row[f"layer_{any_layer}_flg"]

        # group lable
        for any_layer in range(max_layer):
            number_of_groups = any_group_df[f"layer_{any_layer}_flg"].sum()
            temp_df1 = any_group_df[(any_group_df.layer ==any_layer) & (np.isnan(any_group_df.time) == False)]
            temp_df1[f"group_{any_layer}"] = [i for i in range(len(temp_df1))]
            any_group_df = any_group_df.merge(temp_df1[[f"group_{any_layer}"]], how="left", left_index=True, right_index=True)
            any_group_df.stat = any_group_df.stat.fillna("-")
            any_group_df[any_group_df["layer"]==any_layer] = any_group_df[any_group_df["layer"]==any_layer].fillna(method="bfill")
            any_group_df[any_group_df["layer"]>=any_layer] = any_group_df[any_group_df["layer"]>=any_layer].fillna(method="bfill")

        # drop "IN"
        any_group_df = any_group_df[any_group_df.stat != "IN"]
        for any_layer in range(max_layer):
            any_group_df[f"layer_{any_layer}_flg"] = any_group_df[f"layer_{any_layer}_flg"].fillna(0.0)
        any_group_df = any_group_df.fillna("-")

        # aggregate base
        temp_list = [f"group_{target_layer}" for target_layer in range(max_layer)]
        aggr_df = any_group_df.groupby(["name", "layer"] + temp_list).sum()
        aggr_df = aggr_df.reset_index()

        # split
        other_df = aggr_df[aggr_df["name"].str.contains("solve/") == False]
        solve_df = aggr_df[aggr_df["name"].str.contains("solve/")]

        max_layer = max(solve_df.layer)
        for target_layer in range(max_layer):
            denominator_group_list = solve_df[solve_df.layer == target_layer+1][f"group_{target_layer}"].values
            denominator_df = solve_df[(solve_df[f"group_{target_layer}"].isin(denominator_group_list)) & (solve_df.layer == target_layer)]
            denominator_df2 = denominator_df[[f"group_{target_layer}", "time"]]
            denominator_df2.columns = [f"group_{target_layer}", f"denominator_{target_layer+1}_time"]

            solve_df = solve_df.merge(denominator_df2, how="left", on=[f"group_{target_layer}"])
            solve_df[f"layer{target_layer+1}/layer{target_layer}[%]"] = solve_df.time / solve_df[f"denominator_{target_layer+1}_time"] * 100

            solve_df = solve_df.drop(columns=[f"denominator_{target_layer+1}_time"])

            solve_df[f"group_{target_layer+1}"] = solve_df[f"group_{target_layer+1}"].apply(lambda x:str(x).replace("-", "-1"))

        temp_list = [f"group_{target_layer+1}" for target_layer in range(max_layer)]
        solve_df = solve_df.sort_values(temp_list)
        solve_df = solve_df.fillna("")

        for target_layer in range(max_layer):
            solve_df[f"group_{target_layer+1}"] = solve_df[f"group_{target_layer+1}"].apply(lambda x:str(x).replace("-1", ""))

        temp_list = [f"layer_{target_layer}_flg" for target_layer in range(max_layer+1)]
        solve_df = solve_df.drop(columns=temp_list)

        return solve_df