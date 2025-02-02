from typing import Union
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from syntherela.metadata import Metadata
from torch_geometric.data import HeteroData

from relgdiff.data.utils import encode_datetime


def preprocess_data(
    table: pd.DataFrame,
    table_name: str,
    metadata: Metadata,
    categories: dict = {},
    fillna: Union[str, float] = 0.0,
) -> tuple:
    # drop ids
    id_cols = metadata.get_column_names(table_name, sdtype="id")
    temp_table = table.drop(columns=id_cols)

    categorical_columns = metadata.get_column_names(table_name, sdtype="categorical")

    for column in categorical_columns:
        if temp_table[column].isnull().sum() > 0:
            temp_table[column] = (
                temp_table[column].astype("category").cat.add_categories(["?"])
            )
            temp_table[column] = temp_table[column].fillna("?")
        if column not in categories:
            categories[column] = temp_table[column].cat.categories.tolist()
        temp_table[column] = (
            temp_table[column].astype("category").cat.set_categories(categories[column])
        )
    binary_columns = metadata.get_column_names(table_name, sdtype="boolean")
    categorical_columns += binary_columns

    numerical_columns = metadata.get_column_names(table_name, sdtype="numerical")
    datetime_columns = metadata.get_column_names(table_name, sdtype="datetime")

    temp_table = temp_table[categorical_columns + numerical_columns + datetime_columns]
    for column in datetime_columns:
        date_format = (
            metadata.tables[table_name]
            .columns[column]
            .get("datetime_format", "%Y-%m-%d %H:%M:%S")
        )
        temp_table, new_columns = encode_datetime(temp_table, column, date_format)
        numerical_columns.extend(new_columns)

    # one-hot encode the categorical columns
    if len(categorical_columns) > 0:
        temp_table = pd.get_dummies(temp_table, columns=categorical_columns)
    cat_mask = temp_table.dtypes == bool
    cat_columns = temp_table.columns[cat_mask]
    num_columns = temp_table.columns[~cat_mask]
    # convert boolean columns to floats
    temp_table[cat_columns] = temp_table[cat_columns].astype("float64")

    # fill missing values and standardize the numerical columns
    if fillna == "mean":
        temp_table[num_columns] = temp_table[num_columns].fillna(
            temp_table[num_columns].mean()
        )
    elif type(fillna) == float:
        temp_table[num_columns] = temp_table[num_columns].fillna(fillna)
    df_numerical = temp_table[num_columns].astype("float64")
    numerical_std = df_numerical.std()
    non_constant_columns = df_numerical.columns[numerical_std > 0]
    return (
        temp_table,
        categories,
        df_numerical.mean(),
        df_numerical[non_constant_columns].std(),
    )


def tables_to_heterodata(
    tables,
    metadata,
    masked_tables=[],
    embedding_table=None,
    latents={},
    dim_empty=16,
    pos_enc={},
):
    if embedding_table is not None:
        assert embedding_table in latents, (
            f"Missing latent reconstruction targets for target table({embedding_table})"
        )
    data = HeteroData()

    # Transform the ids to 0, 1, 2, ...
    id_map = {}

    for parent_table_name in metadata.get_tables():
        primary_key = metadata.get_primary_key(parent_table_name)
        if primary_key is None:
            tables[parent_table_name].reset_index(inplace=True)
            primary_key = "index"

        if parent_table_name not in id_map:
            id_map[parent_table_name] = {}

        if primary_key not in id_map[parent_table_name]:
            id_map[parent_table_name][primary_key] = {}
            idx = 0
            for primary_key_val in tables[parent_table_name][primary_key].unique():
                id_map[parent_table_name][primary_key][primary_key_val] = idx
                idx += 1

        for relationship in metadata.relationships:
            if relationship["parent_table_name"] != parent_table_name:
                continue
            if relationship["child_table_name"] not in id_map:
                id_map[relationship["child_table_name"]] = {}

            id_map[relationship["child_table_name"]][
                relationship["child_foreign_key"]
            ] = id_map[parent_table_name][relationship["parent_primary_key"]]

    # remap the ids
    for table_name in id_map.keys():
        for column_name in id_map[table_name].keys():
            if column_name not in tables[table_name].columns:
                raise ValueError(
                    f"Column {column_name} not found in table {table_name}"
                )
            tables[table_name][column_name] = tables[table_name][column_name].map(
                id_map[table_name][column_name]
            )

    # Set edges based on relationships.
    for relationship in metadata.relationships:
        parent_table = relationship["parent_table_name"]
        child_table = relationship["child_table_name"]
        foreign_key = relationship["child_foreign_key"]
        if tables[child_table].empty:
            continue

        child_primary_key = metadata.get_primary_key(child_table)
        if child_primary_key is None:
            child_primary_key = "index"
        tables[child_table] = tables[child_table].dropna(subset=[child_primary_key])

        # some relationships can have missing foreign keys
        fks = tables[child_table][[foreign_key, child_primary_key]]
        fks = fks.dropna().astype("int64")
        if fks.empty:
            continue
        data[parent_table, foreign_key, child_table].edge_index = torch.tensor(
            fks.values.T
        )
        data[child_table, foreign_key, parent_table].edge_index = torch.tensor(
            fks.loc[:, [child_primary_key, foreign_key]].values.T
        )

    # set the features for each node to the HeteroData object
    for key in metadata.get_tables():
        # tranform the data to all numerical values
        temp_table, categories, means, stds = preprocess_data(
            tables[key], table_name=key, metadata=metadata
        )
        # standardize the numerical columns
        temp_table[means.index] = temp_table[means.index] - means
        temp_table[stds.index] = temp_table[stds.index] / stds

        # store the categories, means, and stds
        data[key]["categories"] = categories
        data[key]["mean"] = means
        data[key]["std"] = stds

        if key in masked_tables:
            data[key].x = torch.ones(
                (temp_table.shape[0], dim_empty), dtype=torch.float32
            )
        else:
            table_values = temp_table.values.astype("float32")
            if table_values.size == 0:
                data[key].x = torch.ones(
                    (temp_table.shape[0], dim_empty), dtype=torch.float32
                )
            else:
                data[key].x = torch.tensor(table_values, dtype=torch.float32)

        if key in latents:
            data[key].y = torch.tensor(latents[key][:, 1:, :]).reshape(
                data[key].x.shape[0], -1
            )

        if pos_enc is not None and key in pos_enc:
            enc_col = pos_enc[key]
            column_type = metadata.tables[key].columns[enc_col]["sdtype"]
            if column_type == "numerical":
                raise NotImplementedError(
                    "Positional encoding only supported for datetime columns"
                )
            values = tables[key][enc_col].values
            values = values - values.min()
            order = torch.tensor(values / np.timedelta64(1, "D")).long()
            data[key].pe = order

    return data


if __name__ == "__main__":
    from syntherela.data import load_tables, remove_sdv_columns

    DATA_DIR = "./data"
    database_name = "rossmann_subsampled"
    metadata = Metadata().load_from_json(f"{DATA_DIR}/{database_name}/metadata.json")

    tables = load_tables(Path(f"{DATA_DIR}/{database_name}"), metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)
    tables_to_heterodata(tables, metadata)
