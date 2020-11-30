""" aggregate """
import numpy
import pandas

class AggregateDataFrame:
    """AggregateDataFrame
        Aggregation using DataFrame
        pandas.DataFrameを用いた集約処理
    """
    def __init__(self) -> None:
        self.layer1_aggr_df = pandas.DataFrame()
        self.solve_df = pandas.DataFrame()
        self.aggregate_df = pandas.DataFrame()
        self.group_df = pandas.DataFrame()
        self.solve_aggr_df = pandas.DataFrame()
        self.group_0_sort_df = pandas.DataFrame()

    def layer_1_aggregated(self, dict_list:list) -> pandas.DataFrame:
        """layer_1_aggregated
            一層目の集約処理
            Args:
                dict_list(list): logger data
            Returns:
                pandas.DataFrame:1層の集約データ
        """
        # dict_list -> dataframe
        row_df = pandas.DataFrame(dict_list)

        # add column "layer"
        row_df["layer"] = row_df.name.apply(lambda any_seri:any_seri.count("/"))

        # where layer == "layer"
        layer1_df = row_df[(row_df["layer"]==1) & (row_df["time"] != "IN")]

        # groupby "name", "layer"
        df_groupby_obj = layer1_df.groupby(["name", "layer"])
        layer1_aggr_df_sum = df_groupby_obj.sum().reset_index()
        layer1_aggr_df_sum = layer1_aggr_df_sum[["name", "layer", "time"]]

        layer1_aggr_df_cnt = df_groupby_obj.count().reset_index().rename(columns={"time":"cnt"})
        layer1_aggr_df_cnt = layer1_aggr_df_cnt[["name", "cnt"]]

        # join
        self.layer1_aggr_df = pandas.merge(
            layer1_aggr_df_sum, layer1_aggr_df_cnt, how="left", on="name")

        return self.layer1_aggr_df

    def aggregated(self, dict_list:list) -> pandas.DataFrame:
        """aggregated
            全体の集約処理
            Args:
                dict_list(list): logger data
            Returns:
                pandas.DataFrame:全層の集約処理
        """
        # aggregate column list
        aggr_col_list = ["type", "name", "time", "stat"]

        # dict_list -> dataframe
        dataframe = pandas.DataFrame(dict_list, columns=aggr_col_list)

        # drop useless information
        dataframe = dataframe[aggr_col_list]

        # aggregate continuous values
        dataframe = self.aggregated_continuous_values(dataframe)
        aggr_col_list = aggr_col_list + ["group", "cont_cnt"]
        dataframe = dataframe[aggr_col_list]

        # add column layer
        dataframe["layer"] = dataframe["name"].apply(lambda any_seri:any_seri.count("/")-1)

        # max layer
        global_max_layer = max(dataframe["layer"]) + 1

        # global aggeregate
        for any_layer in range(global_max_layer):
            for index, row in dataframe.iterrows():
                row[f"layer_{any_layer}_flg"] = 1 if row["layer"] == any_layer else numpy.nan
                dataframe.loc[index, f"layer_{any_layer}_flg"] = row[f"layer_{any_layer}_flg"]

        # group lable
        base_df = dataframe
        for any_layer in range(global_max_layer):
            temp_df = base_df[
                (base_df["layer"] == any_layer) & (numpy.isnan(base_df["time"]) == bool(False))
            ].copy()

            add_col = f"group_{any_layer}"
            temp_df[add_col] = [i for i, _ in temp_df.iterrows()]
            base_df = base_df.merge(
                temp_df[[add_col]], how="left", left_index=True, right_index=True)

            base_df["stat"] = base_df["stat"].fillna("-")
            base_df[base_df["layer"]==any_layer] = base_df[
                base_df["layer"]==any_layer].fillna(method="bfill")
            base_df[base_df["layer"]>=any_layer] = base_df[
                base_df["layer"]>=any_layer].fillna(method="bfill")

        # drop "IN"
        base_df = base_df[base_df["stat"] != "IN"]

        # add column layer
        for any_layer in range(global_max_layer):
            base_df[f"layer_{any_layer}_flg"] = base_df[f"layer_{any_layer}_flg"].fillna(0.0)
        base_df = base_df.fillna("-")

        # aggregate base
        group_column_list = [f"group_{any_layer}" for any_layer in range(global_max_layer)]
        aggr_df = base_df.groupby(["name", "layer"] + group_column_list).sum()
        aggr_df = aggr_df.reset_index()

        # aggregate solve
        solve_df = self.aggregated_solve(aggr_df)

        return solve_df

    def aggregated_continuous_values(self, dataframe:pandas.DataFrame) -> pandas.DataFrame:
        """aggregate continuous values
            同系で連続している処理の集約
            Args:
                dataframe(pandas.DataFrame): 元データ
            Returns:
                pandas.DataFrame:同系を集約した後の処理
        """
        base_df = dataframe

        # create grouping flg
        group_df = self.create_grouping_flg(base_df)

        # add grouping values
        base_df = base_df.merge(group_df, how="left", left_index=True, right_index=True)
        base_df = base_df.reset_index()
        base_df["cont_flg"] = 1

        temp_any_df = base_df[(numpy.isnan(base_df["group"])  == bool(False))]
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

        self.aggregate_df = aggr_cont_df

        return self.aggregate_df

    def create_grouping_flg(self, dataframe:pandas.DataFrame) -> pandas.DataFrame:
        """create_grouping_flg
            連続した処理のgrouping
            Args:
                dataframe(pandas.DataFrame): logger data
            Returns:
                pandas.DataFrame:全層のgrouping df
        """
        center_df = dataframe["name"].reset_index()
        center_df = center_df.rename(columns={"index":"back_idx"})
        center_df = center_df.reset_index()

        plus_df = center_df.copy()
        plus_df["index"] = plus_df["index"].apply(lambda any_seri:any_seri+1)
        plus_df = plus_df.rename(columns={"name":"name_plus"})
        center_p_df = center_df.merge(plus_df, how="left", on="index")
        center_p_df = center_p_df[center_p_df["name"] == center_p_df["name_plus"]].copy()
        center_p_df["cnt_flg_p"] = 1
        plus_flg_df = center_df.merge(center_p_df[["index", "cnt_flg_p"]], how="left", on="index")

        minus_df = center_df.copy()
        minus_df["index"] = minus_df["index"].apply(lambda any_seri:any_seri-1)
        minus_df = minus_df.rename(columns={"name":"name_minus"})
        center_m_df = center_df.merge(minus_df, how="left", on="index")
        center_m_df = center_m_df[center_m_df["name"] == center_m_df["name_minus"]].copy()
        center_m_df["cnt_flg_m"] = 1
        minus_flg_df = center_df.merge(center_m_df[["index", "cnt_flg_m"]], how="left", on="index")
        minus_flg_df = minus_flg_df[["index", "cnt_flg_m"]]

        center_pm_df = plus_flg_df.merge(minus_flg_df, how="left", on="index")
        center_pm_df = center_pm_df.fillna(0)
        center_pm_df["flg"] = center_pm_df["cnt_flg_p"] - center_pm_df["cnt_flg_m"]

        create_m_group_df = center_pm_df[center_pm_df["flg"] == -1].copy()
        create_m_group_df["m_group"] = [i+1 for i in range(len(create_m_group_df))]
        create_m_group_df = create_m_group_df[["back_idx", "m_group"]]

        create_p_group_df = center_pm_df[center_pm_df["flg"] == 1].copy()
        create_p_group_df["p_group"] = [i+1 for i in range(len(create_p_group_df))]
        create_p_group_df = create_p_group_df[["back_idx", "p_group"]]

        center_flg_df = center_pm_df
        center_flg_df = center_flg_df.merge(create_m_group_df, how="left", on="back_idx")
        center_flg_df = center_flg_df.merge(create_p_group_df, how="left", on="back_idx")
        center_flg_df["m_group"] = center_flg_df["m_group"].fillna(method="ffill")
        center_flg_df["p_group"] = center_flg_df["p_group"].fillna(method="bfill")
        center_flg_where_df = center_flg_df[center_flg_df["m_group"] == center_flg_df["p_group"]]
        center_flg_where_df = center_flg_where_df.rename(columns={"p_group":"group"})
        center_flg_where_df = center_flg_where_df[["back_idx", "group"]]
        center_group_df = center_flg_df.merge(center_flg_where_df, how="left", on="back_idx")

        self.group_df = center_group_df[["group"]]

        return self.group_df

    def aggregated_solve(self, dataframe:pandas.DataFrame) -> pandas.DataFrame:
        """aggregated_solve
            solveの処理の集約
            Args:
                dataframe(pandas.DataFrame): logger data
            Returns:
                pandas.DataFrame:solveの集約処理
        """
        solve_df = dataframe[dataframe["name"].str.contains("solve/")]

        if not solve_df.empty:
            # local max layer
            solve_max_layer = max(solve_df["layer"])

            for any_layer in range(solve_max_layer):
                col_group = f"group_{any_layer}"
                col_denominator = f"denominator_{any_layer+1}_time"

                denominator_group_seri = solve_df[solve_df["layer"] == any_layer+1][col_group]
                denominator_group_list = denominator_group_seri.values

                denominator_df = solve_df[
                    (solve_df[col_group].isin(denominator_group_list)) &
                    (solve_df["layer"] == any_layer)
                ]
                denominator_df = denominator_df[[col_group, "time"]]
                denominator_df.columns = [col_group, col_denominator]

                col_per = f"breakdown_{any_layer} layer{any_layer+1}/layer{any_layer}[%]"
                solve_df = solve_df.merge(denominator_df, how="left", on=[col_group])
                solve_df[col_per] = solve_df["time"] / solve_df[col_denominator] * 100

                # drop column denominator
                solve_df = solve_df.drop(columns=[col_denominator])

                col_group_p1 = f"group_{any_layer}"
                solve_df[col_group_p1] = solve_df[col_group_p1].apply(
                    lambda any_seri:str(any_seri).replace("-", "-1"))

            # sort
            for any_layer in range(solve_max_layer):
                col_group = f"group_{any_layer}"
                solve_df[col_group] = solve_df[col_group].apply(
                    lambda any_seri:int(float(any_seri)))

            narrow_col_list = [f"group_{any_layer+1}" for any_layer in range(solve_max_layer)]
            solve_df = solve_df.sort_values(narrow_col_list)
            solve_df = solve_df.fillna("")

            col_group_m1 = f"group_{solve_max_layer-1}"
            solve_df[col_group_m1] = solve_df[col_group_m1].apply(
                lambda any_seri:int(float(any_seri)))

            # breakdown[%] / count
            for any_layer in range(solve_max_layer):
                col_per_cnt = f"breakdown_{any_layer}[%] / count"
                col_per = f"breakdown_{any_layer} layer{any_layer+1}/layer{any_layer}[%]"
                solve_df[col_per_cnt] = solve_df[col_per].replace("", 0.0) / solve_df["cont_cnt"]
                solve_df[col_per_cnt] = solve_df[col_per_cnt].replace(0.000000, "")

            # -1 to black
            for any_layer in range(solve_max_layer):
                col_group = f"group_{any_layer+1}"
                solve_df[col_group] = solve_df[col_group].apply(
                    lambda any_seri:str(any_seri).replace("-1", ""))

            # drop layer info
            for any_layer in range(solve_max_layer):
                col_per = f"breakdown_{any_layer} layer{any_layer+1}/layer{any_layer}[%]"
                solve_df[col_per] = solve_df[["layer", col_per]].T.apply(
                    lambda x: x[col_per] if x["layer"] in [any_layer, any_layer+1] else 0.0)
                solve_df[col_per] = solve_df[col_per].replace(0.000000, "")

                col_per_cnt = f"breakdown_{any_layer}[%] / count"
                solve_df[col_per_cnt] = solve_df[["layer", col_per_cnt]].T.apply(
                    lambda x: x[col_per_cnt] if x["layer"] in [any_layer, any_layer+1] else 0.0)
                solve_df[col_per_cnt] = solve_df[col_per_cnt].replace(0.000000, "")
                solve_df[col_per_cnt] = solve_df[col_per_cnt].replace(50, "")

            # drop column layer flag
            narrow_col_list = [f"layer_{any_layer}_flg" for any_layer in range(solve_max_layer+1)]
            solve_df = solve_df.drop(columns=narrow_col_list)
            solve_df = solve_df.reset_index()
            solve_df = solve_df.drop(columns=["index"])

            # group 0 sort
            solve_df = self.group_0_sort(solve_df)

        else:
            solve_df = pandas.DataFrame()

        self.solve_aggr_df = solve_df

        return self.solve_aggr_df

    def group_0_sort(self, dataframe:pandas.DataFrame) -> pandas.DataFrame:
        """group_0_sort
            group 0 に置けるsort
            Args:
                dataframe(pandas.DataFrame): logger data
            Returns:
                pandas.DataFrame:sort結果
        """
        group_0_min = min(dataframe["group_0"])
        group_0_max = max(dataframe["group_0"])

        if group_0_min == group_0_max:
            group_0_max = group_0_max + 1

        final_sort_solve_df = pandas.DataFrame(columns=dataframe.columns)
        for index in range(group_0_min, group_0_max):
            final_sort_solve_df = pandas.concat(
                [final_sort_solve_df, dataframe[dataframe["group_0"] == index]])
        solve_df = final_sort_solve_df
        solve_df = solve_df.reset_index().drop("index", axis=1)
        self.group_0_sort_df = solve_df

        return self.group_0_sort_df
