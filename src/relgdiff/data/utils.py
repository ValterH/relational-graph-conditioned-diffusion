from copy import deepcopy
import pandas as pd


def encode_datetime(df, column):
    datetime_columns = []
    nulls = df[column].isnull()
    df[column] = pd.to_datetime(df[column], errors="coerce")
    df[f"{column}_Year"] = df[column].dt.year
    df[f"{column}_Month"] = df[column].dt.month
    df[f"{column}_Day"] = df[column].dt.day
    df.loc[nulls, f"{column}_Year"] = 0
    df.loc[nulls, f"{column}_Month"] = 0
    df.loc[nulls, f"{column}_Day"] = 0
    datetime_columns.extend([f"{column}_Year", f"{column}_Month", f"{column}_Day"])
    # check if hours, minutes, seconds are needed
    if df[column].dt.hour.sum() > 0:
        df[f"{column}_Hour"] = df[column].dt.hour
        df[f"{column}_Minute"] = df[column].dt.minute
        df[f"{column}_Second"] = df[column].dt.second
        df.loc[nulls, f"{column}_Hour"] = 0
        df.loc[nulls, f"{column}_Minute"] = 0
        df.loc[nulls, f"{column}_Second"] = 0
        datetime_columns.extend(
            [f"{column}_Hour", f"{column}_Minute", f"{column}_Second"]
        )
    return df.drop(columns=[column]), datetime_columns


def add_aggregations(data, metadata, update_metadata=True):
    aggregated_data = deepcopy(data)
    for relationship in metadata.relationships:
        parent_table_name = relationship["parent_table_name"]
        child_table_name = relationship["child_table_name"]
        parent_column = relationship["parent_primary_key"]
        child_fk = relationship["child_foreign_key"]

        # add child counts
        child_df = pd.DataFrame(
            {
                f"{child_table_name}_{child_fk}_counts": data[child_table_name][
                    child_fk
                ].value_counts()
            }
        )
        cardinality_df = (
            pd.DataFrame({"parent": data[parent_table_name][parent_column]})
            .join(child_df, on="parent")
            .fillna(0)
        )
        aggregated_data[parent_table_name] = (
            aggregated_data[parent_table_name]
            .merge(cardinality_df, how="left", left_on=parent_column, right_on="parent")
            .drop(columns="parent")
        )

        if update_metadata:
            metadata.add_column(
                parent_table_name,
                f"{child_table_name}_{child_fk}_counts",
                sdtype="numerical",
                computer_representation="Int64",
            )

        # add categorical counts
        categorical_columns = []
        for column_name, column_info in (
            metadata.tables[child_table_name].to_dict()["columns"].items()
        ):
            if column_info["sdtype"] == "categorical":
                categorical_columns.append(column_name)

        if len(categorical_columns) > 0:
            categorical_df = data[child_table_name][categorical_columns + [child_fk]]
            categorical_column_names = [
                f"{child_table_name}_{child_fk}_{column}_nunique"
                for column in categorical_columns
            ]
            categorical_df.columns = categorical_column_names + [child_fk]

            aggregated_data[parent_table_name] = aggregated_data[
                parent_table_name
            ].merge(
                categorical_df.groupby(child_fk).nunique(),
                how="left",
                left_on=parent_column,
                right_index=True,
                suffixes=("", "_nunique"),
            )
            aggregated_data[parent_table_name][categorical_column_names] = (
                aggregated_data[parent_table_name][categorical_column_names].fillna(0)
            )

            if update_metadata:
                for column in categorical_column_names:
                    metadata.add_column(
                        parent_table_name,
                        column,
                        sdtype="numerical",
                        computer_representation="Int64",
                    )

        # add numerical means
        numerical_columns = []
        for column_name, column_info in (
            metadata.tables[child_table_name].to_dict()["columns"].items()
        ):
            if column_info["sdtype"] == "numerical":
                numerical_columns.append(column_name)

        if len(numerical_columns) > 0:
            numerical_df = data[child_table_name][numerical_columns + [child_fk]]
            numerical_column_names = [
                f"{child_table_name}_{child_fk}_{column}_mean"
                for column in numerical_columns
            ]
            numerical_df.columns = numerical_column_names + [child_fk]

            aggregated_data[parent_table_name] = aggregated_data[
                parent_table_name
            ].merge(
                numerical_df.groupby(child_fk).mean(),
                how="left",
                left_on=parent_column,
                right_index=True,
                suffixes=("", "_mean"),
            )

            if update_metadata:
                for column in numerical_column_names:
                    metadata.add_column(
                        parent_table_name,
                        column,
                        sdtype="numerical",
                        computer_representation="Float",
                    )

    return aggregated_data, metadata


def get_table_order(metadata):
    parents = sorted(metadata.get_root_tables())
    all_tables = metadata.get_tables()
    table_order = parents

    while len(table_order) < len(all_tables):
        for relationship in metadata.relationships:
            if (
                relationship["parent_table_name"] in table_order
                and relationship["child_table_name"] not in table_order
            ):
                table_order.append(relationship["child_table_name"])
    return table_order


def get_positional_encoding(dataset_name):
    if dataset_name == "rossmann_subsampled":
        return {"historical": "Date"}, True
    elif dataset_name == "Biodegradability_v1":
        return None, False
    elif dataset_name == "walmart_subsampled":
        return {"features": "Date", "depts": "Date"}, True
    elif dataset_name == "CORA_v1":
        return None, False
    elif dataset_name == "imdb_MovieLens_v1":
        return None, False
    elif dataset_name == "airbnb-simplified_subsampled":
        return {"users": "date_account_created"}, True
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
