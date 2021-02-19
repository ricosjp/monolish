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
        self.solve_df       = pandas.DataFrame()
        self.calculated_df  = pandas.DataFrame()
        self.shaping_df     = pandas.DataFrame()
        self.base_df        = pandas.DataFrame()

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
                Hash Table
            Returns:
                pandas.DataFrame:全層の集約処理
        """
        # aggregate column list
        aggr_col_list = ["type", "name", "time", "stat"]

        # dict_list -> dataframe
        dataframe = pandas.DataFrame(dict_list, columns=aggr_col_list)

        # drop useless information
        dataframe = dataframe[aggr_col_list]
        dataframe["layer"] = dataframe["name"].str.count("/")

        # create preprocessing table
        base_df = self.create_preprocessing_table(dataframe)

        # calculate_table
        calculated_df = self.calculate_table(base_df)

        # shaping
        shaping_df = self.shaping(calculated_df)

        self.solve_df = shaping_df

        return self.solve_df

    def create_preprocessing_table(self, dataframe:pandas.DataFrame) -> pandas.DataFrame:
        """create_preprocessing_table
            計算するtableを作る
            Args:
                dataframe(pandas.DataFrame): 元データ
            Returns:
                pandas.DataFrame:sort後のdf
        """
        # apply group lable
        grouping_df = dataframe[(dataframe["layer"] == 1) & (dataframe["stat"] != "IN")].copy()
        grouping_df.loc[:, "group"] = range(1, len(grouping_df)+1)
        dataframe = dataframe.merge(
            grouping_df[["group"]], how="left", left_index=True, right_index=True)
        dataframe["group"] = dataframe["group"].fillna(method="bfill")

        # drop IN record
        dataframe = dataframe[dataframe["stat"] != "IN"]

        # group by
        dataframe.time = dataframe.time.astype(float)
        groupby_group_dfgb = dataframe.groupby(["group", "name"])

        sum_by_aggr_df = groupby_group_dfgb.sum()
        sum_by_aggr_df = sum_by_aggr_df.reset_index()
        sum_by_aggr_df = sum_by_aggr_df[["group", "name", "time"]]

        count_by_aggr_df = groupby_group_dfgb.count()
        count_by_aggr_df = count_by_aggr_df.reset_index()
        count_by_aggr_df = count_by_aggr_df[["group", "name", "time"]]
        count_by_aggr_df = count_by_aggr_df.rename(columns={"time":"cnt"})

        aggr_df = sum_by_aggr_df.merge(count_by_aggr_df, how="left", on=["group", "name"])
        aggr_df["group"] = aggr_df["group"].astype("int")
        aggr_df["layer"] = aggr_df["name"].str.count("/")
        aggr_df = aggr_df.sort_values(by=["group", "name"])
        aggr_df = aggr_df.reset_index(drop=True)

        # apply parent flg_child lable
        aggr_df["parent"] = aggr_df["name"].apply(lambda x:"/".join(x.split("/")[:-2]+[""]))
        temp_df = aggr_df[["group", "parent"]]
        temp_df = temp_df[~temp_df.duplicated()]
        temp_df = temp_df.rename(columns={"parent":"flg_child"})
        aggr_df = aggr_df.merge(
            temp_df, how="left", left_on=["group", "name"], right_on=["group", "flg_child"])
        aggr_df["flg_child"] = aggr_df["flg_child"].apply(
            lambda x: str(1) if x is not numpy.nan else numpy.nan)

        self.base_df = aggr_df[["group", "layer", "name", "parent", "flg_child", "time", "cnt"]]

        return self.base_df

    def calculate_table(self, dataframe:pandas.DataFrame) -> pandas.DataFrame:
        """calculate_table
            per, cnt_perを出す
            Args:
                dataframe(pandas.DataFrame): 計算前のtable
            Returns:
                pandas.DataFrame:sort後のdf
        """
        max_layer = max(dataframe["layer"])
        for layer in range(1, max_layer):
            target_df = dataframe[(dataframe["layer"] == layer) | (dataframe["layer"]  == layer+1)]
            denom_by_df = target_df[["group", "name", "time", "cnt"]]
            denom_by_df.columns = ["group", "parent_name", "parent_time", "parent_cnt"]
            target_df = target_df.merge(denom_by_df,
                how="inner", left_on=["group", "parent"], right_on=["group", "parent_name"])

            # percent
            per_col_name = f"l{layer+1}/l{layer}_per"
            per_cnt_col_name = f"l{layer+1}/l{layer}_per/count"

            target_df[per_col_name] = target_df["time"] / target_df["parent_time"] * 100
            target_df[per_cnt_col_name] = target_df[per_col_name] / target_df["cnt"]
            target_df = target_df[["group", "layer", "name", per_col_name, per_cnt_col_name]]
            dataframe = dataframe.merge(target_df, how="left", on=["group", "layer", "name"])
            # add 100%
            dataframe[per_col_name] = dataframe[per_col_name].mask(
                (dataframe["layer"] == layer) & (dataframe["flg_child"]=="1"), 100.0)
            dataframe[per_cnt_col_name] = dataframe[per_cnt_col_name].mask(
                (dataframe["layer"] == layer) & (dataframe["flg_child"]=="1"), 100.0)

        self.calculated_df = dataframe

        return self.calculated_df

    def shaping(self, dataframe:pandas.DataFrame) -> pandas.DataFrame:
        """shaping
            出力の形を整える
            Args:
                dataframe(pandas.DataFrame): 整形前のデータ
            Returns:
                pandas.DataFrame:sort後のdf
        """
        dataframe = dataframe[dataframe["name"].str.contains("solve/")]
        dataframe = dataframe.reset_index(drop=True)
        group_list = dataframe["group"].unique()
        replace_dict = {value:index+1 for index, value in enumerate(group_list)}
        dataframe["group"] = dataframe["group"].map(replace_dict)

        if not dataframe.empty:
            max_layer = max(dataframe["layer"])
            per_list = []
            for layer in range(1, max_layer):
                per_list =  per_list + [f"l{layer+1}/l{layer}_per"]
                per_list =  per_list + [f"l{layer+1}/l{layer}_per/count"]

            colum_list = ["name", "layer" , "group", "time", "cnt"] + per_list
            dataframe = dataframe[colum_list]
            dataframe = dataframe.fillna("")
        else:
            dataframe = pandas.DataFrame()

        self.shaping_df = dataframe

        return self.shaping_df
