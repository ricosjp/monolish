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

            # calc_percent
            aggr_column_list = ["layer", "name", "count", "total_time [s]"]
            aggr_ndarray, aggr_column_list = calc_percent(aggr_ndarray, aggr_column_list, max_layer)

            # erase information
            aggr_ndarray = erase_information(aggr_ndarray, max_layer)

            # aggregate list
            aggr_column_lists.append(aggr_column_list)
            aggr_ndarrays.append(aggr_ndarray)

        else:
            index = index + 1

        return aggr_column_lists, aggr_ndarrays, index

    def calc_percent(self, aggr_ndarray, aggr_column_list, max_layer):
        for layer in range(1, max_layer):
            aggr_column_list.append(f"breakdown_layer {str(layer)} [%] (breakdown[%] / count)")

            denominator = (float)(aggr_ndarray[np.array(list(map(lambda x: int(x[0])==layer, aggr_ndarray)))][:, 3][0])
            percent = np.array(aggr_ndarray[:, 3], dtype="float32") / denominator * 100.0
            percent = np.round(percent, decimals=3)
            percent = np.where(percent <= 100.0, percent, "")

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

        # aggregate column list
        aggr_col_list = ["type", "name", "time", "stat"]

        # drop useless information
        df = df[aggr_col_list]
        # aggregate continuous values
        df = self.aggregated_continuous_values(df)
        aggr_col_list = aggr_col_list + ["group", "cont_cnt"]
        df = df[aggr_col_list]

        # add column layer
        df["layer"] = df.name.apply(lambda x:x.count("/")-1)

        # max layer
        global_max_layer = max(df.layer) + 1

        # global aggeregate
        for any_layer in range(global_max_layer):
            for index, row in df.iterrows():
                row[f"layer_{any_layer}_flg"] = 1 if row.layer == any_layer else np.nan
                df.loc[index, f"layer_{any_layer}_flg"] = row[f"layer_{any_layer}_flg"]

        # group lable
        for any_layer in range(global_max_layer):
            number_of_groups = df[f"layer_{any_layer}_flg"].sum()
            temp_df1 = df[(df.layer ==any_layer) & (np.isnan(df.time) == False)].copy()
            temp_df1[f"group_{any_layer}"] = [i for i in range(len(temp_df1))]
            df = df.merge(temp_df1[[f"group_{any_layer}"]], how="left", left_index=True, right_index=True)
            df.stat = df.stat.fillna("-")
            df[df["layer"]==any_layer] = df[df["layer"]==any_layer].fillna(method="bfill")
            df[df["layer"]>=any_layer] = df[df["layer"]>=any_layer].fillna(method="bfill")

        # drop "IN"
        df = df[df.stat != "IN"]

        # add column layer
        for any_layer in range(global_max_layer):
            df[f"layer_{any_layer}_flg"] = df[f"layer_{any_layer}_flg"].fillna(0.0)
        df = df.fillna("-")

        # aggregate base
        group_column_list = [f"group_{target_layer}" for target_layer in range(global_max_layer)]
        aggr_df = df.groupby(["name", "layer"] + group_column_list).sum()
        aggr_df = aggr_df.reset_index()

        # split
        other_df = aggr_df[aggr_df["name"].str.contains("solve/") == False]
        solve_df = aggr_df[aggr_df["name"].str.contains("solve/")]
        
        # local max layer
        solve_max_layer = max(solve_df.layer)
        for target_layer in range(solve_max_layer):
            denominator_group_list = solve_df[solve_df.layer == target_layer+1][f"group_{target_layer}"].values
            denominator_df = solve_df[(solve_df[f"group_{target_layer}"].isin(denominator_group_list)) & (solve_df.layer == target_layer)]
            denominator_df = denominator_df[[f"group_{target_layer}", "time"]]
            denominator_df.columns = [f"group_{target_layer}", f"denominator_{target_layer+1}_time"]

            solve_df = solve_df.merge(denominator_df, how="left", on=[f"group_{target_layer}"])
            solve_df[f"breakdown_{target_layer} layer{target_layer+1}/layer{target_layer}[%]"] = solve_df.time / solve_df[f"denominator_{target_layer+1}_time"] * 100

            # drop column denominator
            solve_df = solve_df.drop(columns=[f"denominator_{target_layer+1}_time"])

            solve_df[f"group_{target_layer+1}"] = solve_df[f"group_{target_layer+1}"].apply(lambda x:str(x).replace("-", "-1"))

        # sort
        for target_layer in range(solve_max_layer):
            solve_df[f"group_{target_layer}"] = solve_df[f"group_{target_layer}"].apply(lambda x:int(float(x)))
        group_column_list = [f"group_{target_layer+1}" for target_layer in range(solve_max_layer)]
        solve_df = solve_df.sort_values(group_column_list)
        solve_df = solve_df.fillna("")

        solve_df[f"group_{solve_max_layer-1}"] = solve_df[f"group_{solve_max_layer-1}"].apply(lambda x:int(float(x)))

        # breakdown[%] / count
        for target_layer in range(solve_max_layer):
            solve_df[f"breakdown_{target_layer}[%] / count"] = solve_df[f"breakdown_{target_layer} layer{target_layer+1}/layer{target_layer}[%]"].replace("", 0.0) / solve_df["cont_cnt"]
            solve_df[f"breakdown_{target_layer}[%] / count"] = solve_df[f"breakdown_{target_layer}[%] / count"].replace(0.000000, "")

        # -1 to black
        for target_layer in range(solve_max_layer):
            solve_df[f"group_{target_layer+1}"] = solve_df[f"group_{target_layer+1}"].apply(lambda x:str(x).replace("-1", ""))

        # drop layer info
        for target_layer in range(solve_max_layer):
            solve_df[f"breakdown_{target_layer} layer{target_layer+1}/layer{target_layer}[%]"] = solve_df[["layer", f"breakdown_{target_layer} layer{target_layer+1}/layer{target_layer}[%]"]].T.apply(lambda x: x[f"breakdown_{target_layer} layer{target_layer+1}/layer{target_layer}[%]"] if x["layer"] in [target_layer, target_layer+1] else 0.0)
            solve_df[f"breakdown_{target_layer} layer{target_layer+1}/layer{target_layer}[%]"] = solve_df[f"breakdown_{target_layer} layer{target_layer+1}/layer{target_layer}[%]"].replace(0.000000, "")
            solve_df[f"breakdown_{target_layer}[%] / count"] = solve_df[["layer", f"breakdown_{target_layer}[%] / count"]].T.apply(lambda x: x[f"breakdown_{target_layer}[%] / count"] if x["layer"] in [target_layer, target_layer+1] else 0.0)
            solve_df[f"breakdown_{target_layer}[%] / count"] = solve_df[f"breakdown_{target_layer}[%] / count"].replace(0.000000, "")
            solve_df[f"breakdown_{target_layer}[%] / count"] = solve_df[f"breakdown_{target_layer}[%] / count"].replace(50, "")

        # drop column layer flag
        layer_column_list = [f"layer_{target_layer}_flg" for target_layer in range(solve_max_layer+1)]
        solve_df = solve_df.drop(columns=layer_column_list)
        solve_df = solve_df.reset_index()
        solve_df = solve_df.drop(columns=["index"])

        # final sort
        group_0_min = min(solve_df["group_0"])
        group_0_max = max(solve_df["group_0"])

        if group_0_min == group_0_max:
            group_0_max = group_0_max + 1

        final_sort_solve_df = pd.DataFrame(columns=solve_df.columns)
        for group_0_index in range(group_0_min, group_0_max):
            final_sort_solve_df = pd.concat([final_sort_solve_df, solve_df[solve_df["group_0"] == group_0_index]])
        solve_df = final_sort_solve_df

        other_df = other_df.drop(columns=layer_column_list)
        layer_column_list2 = [f"group_{target_layer}" for target_layer in range(1, solve_max_layer+1)]
        other_df = other_df.drop(columns=layer_column_list2)

        return other_df, solve_df

    def aggregated_continuous_values(self, df):
        base_df = df

        center_temp_df = base_df.name
        center_temp_df = center_temp_df.reset_index()
        center_temp_df = center_temp_df.rename(columns={"index":"back_idx"})
        center_temp_df = center_temp_df.reset_index()

        plus_temp_df = center_temp_df.copy()
        plus_temp_df["index"] = plus_temp_df["index"].apply(lambda x:x+1)
        plus_temp_df = plus_temp_df.rename(columns={"name":"name_plus"})

        temp_df2 = center_temp_df.merge(plus_temp_df, how="left", on="index")
        temp_df3 = temp_df2[["index", "name", "name_plus"]]
        temp_df4 = temp_df3[temp_df3["name"] == temp_df3["name_plus"]]
        temp_df4["count_flg_plus"] = 1
        plus_temp_df2 = center_temp_df.merge(temp_df4[["index", "count_flg_plus"]], how="left", on="index")

        minus_temp_df = center_temp_df.copy()
        minus_temp_df["index"] = minus_temp_df["index"].apply(lambda x:x-1)
        minus_temp_df = minus_temp_df.rename(columns={"name":"name_minus"})

        temp_df2 = center_temp_df.merge(minus_temp_df, how="left", on="index")
        temp_df3 = temp_df2[["index", "name", "name_minus"]]
        temp_df4 = temp_df3[temp_df3["name"] == temp_df3["name_minus"]]
        temp_df4["count_flg_minus"] = 1
        minus_temp_df2 = center_temp_df.merge(temp_df4[["index", "count_flg_minus"]], how="left", on="index")

        center_temp_df2 = plus_temp_df2.merge(minus_temp_df2[["index", "count_flg_minus"]], how="left", on="index")

        center_temp_df2 = center_temp_df2.fillna(0)
        center_temp_df2["count_flg_plus"] = center_temp_df2["count_flg_plus"].apply(lambda x:int(x))
        center_temp_df2["count_flg_minus"] = center_temp_df2["count_flg_minus"].apply(lambda x:int(x))

        center_temp_df2["flg"] = center_temp_df2["count_flg_plus"] - center_temp_df2["count_flg_minus"]

        center_temp_df3 = center_temp_df2[center_temp_df2["flg"] == -1]
        center_temp_df3["m_group"] = [i+1 for i in range(len(center_temp_df3))]
        center_temp_df3 = center_temp_df3[["back_idx", "m_group"]]

        center_temp_df4 = center_temp_df2[center_temp_df2["flg"] == 1]
        center_temp_df4["p_group"] = [i+1 for i in range(len(center_temp_df4))]
        center_temp_df4 = center_temp_df4[["back_idx", "p_group"]]

        center_temp_df2 = center_temp_df2.merge(center_temp_df3, how="left", on="back_idx")
        center_temp_df2 = center_temp_df2.merge(center_temp_df4, how="left", on="back_idx")
        center_temp_df2["m_group"] = center_temp_df2["m_group"].fillna(method="ffill")
        center_temp_df2["p_group"] = center_temp_df2["p_group"].fillna(method="bfill")
        center_temp_df3 = center_temp_df2[center_temp_df2["m_group"] == center_temp_df2["p_group"]]
        center_temp_df3 = center_temp_df3.rename(columns={"p_group":"group"})
        center_temp_df4 = center_temp_df2.merge(center_temp_df3[["back_idx", "group"]], how="left", on="back_idx")
        center_temp_df4 = center_temp_df4[["group"]]

        base_df = base_df.merge(center_temp_df4, how="left", left_index=True, right_index=True)
        base_df = base_df.reset_index()
        base_df["cont_flg"] = 1

        temp_any_df = base_df[(np.isnan(base_df["group"]) == False)]
        temp_any_df1 = temp_any_df.groupby(["group"]).max()
        temp_any_df1 = temp_any_df1.reset_index()
        temp_any_df1 = temp_any_df1[["index", "group"]]
        temp_any_df2 = temp_any_df.groupby(["type", "name", "group"]).sum()
        temp_any_df2 = temp_any_df2.reset_index()
        temp_any_df2 = temp_any_df2.drop(columns=["index"])

        any_df1 = temp_any_df2.merge(temp_any_df1, how="left", on = "group")
        any_df2 = base_df[np.isnan(base_df["group"])]

        any_df3 = pd.concat([any_df1, any_df2])
        any_df3 = any_df3.sort_values("index")
        any_df3 = any_df3.drop(columns=["index"])
        any_df3 = any_df3.reset_index()
        any_df3 = any_df3.drop(columns=["index"])
        any_df3 = any_df3.rename(columns={"cont_flg":"cont_cnt"})

        return any_df3