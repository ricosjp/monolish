""" aggregate """
import numpy
import pandas

class AggregatePandas:
    def layer_1_aggregated(self, dict_list:list) -> pandas.DataFrame:
        # dict_list -> dataframe
        row_df =  pandas.DataFrame(dict_list)

        # add column "layer"
        row_df["layer"] = row_df.name.apply(lambda any_series:any_series.count("/"))

        # where layer == "layer"
        layer1_df = row_df[(row_df["layer"]==1) & (row_df.time != "IN")]

        # groupby "name", "layer"
        df_groupby_obj = layer1_df.groupby(["name", "layer"])
        layer1_aggr_df_sum = df_groupby_obj.sum().reset_index()
        layer1_aggr_df_cnt = df_groupby_obj.count().reset_index().rename(columns={"time":"cnt"})

        # join
        layer1_aggr_df = pandas.merge(layer1_aggr_df_sum[["name", "layer", "time"]], layer1_aggr_df_cnt[["name", "cnt"]], how = "left", on="name")

        return layer1_aggr_df

    def aggregated(self, dict_list:list) -> pandas.DataFrame:
        # dict_list -> dataframe
        dataframe = pandas.DataFrame(dict_list)

        # aggregate column list
        aggr_col_list = ["type", "name", "time", "stat"]

        # drop useless information
        dataframe = dataframe[aggr_col_list]

        # aggregate continuous values
        dataframe = self.aggregated_continuous_values(dataframe)
        aggr_col_list = aggr_col_list + ["group", "cont_cnt"]
        dataframe = dataframe[aggr_col_list]

        # add column layer
        dataframe["layer"] = dataframe["name"].apply(lambda any_series:any_series.count("/")-1)

        # max layer
        global_max_layer = max(dataframe["layer"]) + 1

        # global aggeregate
        for any_layer in range(global_max_layer):
            for index, row in dataframe.iterrows():
                row[f"layer_{any_layer}_flg"] = 1 if row["layer"] == any_layer else numpy.nan
                dataframe.loc[index, f"layer_{any_layer}_flg"] = row[f"layer_{any_layer}_flg"]

        # group lable
        for any_layer in range(global_max_layer):
            number_of_groups = dataframe[f"layer_{any_layer}_flg"].sum()
            temp_df1 = dataframe[(dataframe["layer"] ==any_layer) & (numpy.isnan(dataframe.time) == False)].copy()
            temp_df1[f"group_{any_layer}"] = [i for i in range(len(temp_df1))]
            dataframe = dataframe.merge(temp_df1[[f"group_{any_layer}"]], how="left", left_index=True, right_index=True)
            dataframe["stat"] = dataframe["stat"].fillna("-")
            dataframe[dataframe["layer"]==any_layer] = dataframe[dataframe["layer"]==any_layer].fillna(method="bfill")
            dataframe[dataframe["layer"]>=any_layer] = dataframe[dataframe["layer"]>=any_layer].fillna(method="bfill")

        # drop "IN"
        dataframe = dataframe[dataframe["stat"] != "IN"]

        # add column layer
        for any_layer in range(global_max_layer):
            dataframe[f"layer_{any_layer}_flg"] = dataframe[f"layer_{any_layer}_flg"].fillna(0.0)
        dataframe = dataframe.fillna("-")

        # aggregate base
        group_column_list = [f"group_{target_layer}" for target_layer in range(global_max_layer)]
        aggr_df = dataframe.groupby(["name", "layer"] + group_column_list).sum()
        aggr_df = aggr_df.reset_index()

        # split
        solve_df = aggr_df[aggr_df["name"].str.contains("solve/")]
        
        # local max layer
        solve_max_layer = max(solve_df["layer"])
        for target_layer in range(solve_max_layer):
            denominator_group_list = solve_df[solve_df["layer"] == target_layer+1][f"group_{target_layer}"].values
            denominator_df = solve_df[(solve_df[f"group_{target_layer}"].isin(denominator_group_list)) & (solve_df["layer"] == target_layer)]
            denominator_df = denominator_df[[f"group_{target_layer}", "time"]]
            denominator_df.columns = [f"group_{target_layer}", f"denominator_{target_layer+1}_time"]

            solve_df = solve_df.merge(denominator_df, how="left", on=[f"group_{target_layer}"])
            solve_df[f"breakdown_{target_layer} layer{target_layer+1}/layer{target_layer}[%]"] = solve_df["time"] / solve_df[f"denominator_{target_layer+1}_time"] * 100

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

        final_sort_solve_df = pandas.DataFrame(columns=solve_df.columns)
        for group_0_index in range(group_0_min, group_0_max):
            final_sort_solve_df = pandas.concat([final_sort_solve_df, solve_df[solve_df["group_0"] == group_0_index]])
        solve_df = final_sort_solve_df
        solve_df = solve_df.reset_index().drop("index", axis=1)

        return solve_df

    def aggregated_continuous_values(self, dataframe:pandas.DataFrame) -> pandas.DataFrame:
        base_df = dataframe

        center_temp_df = base_df.name
        center_temp_df = center_temp_df.reset_index()
        center_temp_df = center_temp_df.rename(columns={"index":"back_idx"})
        center_temp_df = center_temp_df.reset_index()

        plus_temp_df = center_temp_df.copy()
        plus_temp_df["index"] = plus_temp_df["index"].apply(lambda x:x+1)
        plus_temp_df = plus_temp_df.rename(columns={"name":"name_plus"})

        center_p_temp_df = center_temp_df.merge(plus_temp_df, how="left", on="index")
        center_p_temp_df = center_p_temp_df[center_p_temp_df["name"] == center_p_temp_df["name_plus"]].copy()
        center_p_temp_df["count_flg_plus"] = 1
        plus_temp_df2 = center_temp_df.merge(center_p_temp_df[["index", "count_flg_plus"]], how="left", on="index")

        minus_temp_df = center_temp_df.copy()
        minus_temp_df["index"] = minus_temp_df["index"].apply(lambda x:x-1)
        minus_temp_df = minus_temp_df.rename(columns={"name":"name_minus"})

        center_m_temp_df = center_temp_df.merge(minus_temp_df, how="left", on="index")
        center_m_temp_df = center_m_temp_df[center_m_temp_df["name"] == center_m_temp_df["name_minus"]].copy()
        center_m_temp_df["count_flg_minus"] = 1
        minus_temp_df2 = center_temp_df.merge(center_m_temp_df[["index", "count_flg_minus"]], how="left", on="index")

        center_temp_df2 = plus_temp_df2.merge(minus_temp_df2[["index", "count_flg_minus"]], how="left", on="index")

        center_temp_df2 = center_temp_df2.fillna(0)
        center_temp_df2["count_flg_plus"] = center_temp_df2["count_flg_plus"].apply(lambda x:int(x))
        center_temp_df2["count_flg_minus"] = center_temp_df2["count_flg_minus"].apply(lambda x:int(x))

        center_temp_df2["flg"] = center_temp_df2["count_flg_plus"] - center_temp_df2["count_flg_minus"]

        center_temp_df3 = center_temp_df2[center_temp_df2["flg"] == -1].copy()
        center_temp_df3["m_group"] = [i+1 for i in range(len(center_temp_df3))]
        center_temp_df3 = center_temp_df3[["back_idx", "m_group"]]

        center_temp_df4 = center_temp_df2[center_temp_df2["flg"] == 1].copy()
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

        temp_any_df = base_df[(numpy.isnan(base_df["group"]) == False)]
        temp_any_df1 = temp_any_df.groupby(["group"]).max()
        temp_any_df1 = temp_any_df1.reset_index()
        temp_any_df1 = temp_any_df1[["index", "group"]]
        temp_any_df2 = temp_any_df.groupby(["type", "name", "group"]).sum()
        temp_any_df2 = temp_any_df2.reset_index()
        temp_any_df2 = temp_any_df2.drop(columns=["index"])

        any_df1 = temp_any_df2.merge(temp_any_df1, how="left", on = "group")
        any_df2 = base_df[numpy.isnan(base_df["group"])]

        aggr_cont_df = pandas.concat([any_df1, any_df2], sort=True)
        aggr_cont_df = aggr_cont_df.sort_values("index")
        aggr_cont_df = aggr_cont_df.drop(columns=["index"])
        aggr_cont_df = aggr_cont_df.reset_index()
        aggr_cont_df = aggr_cont_df.drop(columns=["index"])
        aggr_cont_df = aggr_cont_df.rename(columns={"cont_flg":"cont_cnt"})

        return aggr_cont_df