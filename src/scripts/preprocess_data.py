import os
import json
import argparse
import warnings

import numpy as np
from syntherela.metadata import Metadata
from syntherela.data import load_tables, remove_sdv_columns

from relgdiff.data.utils import encode_datetime

TYPE_TRANSFORM = {"float", np.float32, "str", str, "int", int}


def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, column_names=None):
    if not column_names:
        column_names = np.array(data_df.columns.tolist())

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):
        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1

    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k

    idx_name_mapping = {}

    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(
    data_df, cat_columns, num_train=0, num_test=0, max_retries=100
):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)

    seed = 1234
    retries = 0
    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]

        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]

        flag = 0
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break

        if flag == 0:
            break
        else:
            seed += 1
            retries += 1
            if retries > max_retries:
                warnings.warn(
                    f"Unable to split the train and test data for table:\n {data_df.head()}"
                )
                return data_df, data_df, None, np.zeros(0)

    return train_df, test_df, seed, idx


def process_data(
    data,
    name,
    metadata,
    factor_missing=True,
    data_path="data",
    dataset_name="",
):
    # Preprocessing
    # replace nans in categorical columns with '?'
    categorical_columns = metadata.get_column_names(
        sdtype="categorical"
    ) + metadata.get_column_names(sdtype="boolean")
    datetime_columns = metadata.get_column_names(sdtype="datetime")
    numerical_columns = metadata.get_column_names(sdtype="numerical")
    id_columns = metadata.get_column_names(sdtype="id")

    for col in categorical_columns:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].cat.add_categories("?")
            data[col] = data[col].fillna("?")

    # convert datetime column to int
    for col in datetime_columns:
        data, new_columns = encode_datetime(data, col)
        numerical_columns.extend(new_columns)

    for col in numerical_columns:
        if factor_missing and data[col].isnull().sum() > 0:
            data[f"{col}_missing"] = data[col].isnull().astype(int)
            categorical_columns.append(f"{col}_missing")
        data[col] = data[col].fillna(0)

    # drop id columns
    data = data.drop(columns=id_columns)

    cat_col_idx = sorted([data.columns.get_loc(c) for c in categorical_columns])
    num_col_idx = sorted([data.columns.get_loc(c) for c in numerical_columns])
    num_data = data.shape[0]

    column_names = data.columns.tolist()

    save_dir = f"{data_path}/processed/{dataset_name}/{name}{'_factor' if factor_missing else ''}"

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(
        data, num_col_idx, cat_col_idx, column_names
    )

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]

    # Train/Test Split, 90% Training, 10% Testing (Validation set will be selected from Training set)
    num_train = int(num_data * 0.9)
    num_test = num_data - num_train

    train_df, test_df, seed, idx = train_val_test_split(
        data, cat_columns, num_train, num_test
    )

    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))

    print(name, train_df.shape, test_df.shape, data.shape)

    col_info = {}

    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info[col_idx]["type"] = "numerical"
        col_name = idx_name_mapping[col_idx]
        if col_name not in metadata.columns:
            split_name = col_name.split("_")
            reconstructed_name = "_".join(split_name[:-1])
            if reconstructed_name in metadata.columns:
                subtype = "float"
            else:
                raise ValueError(f"Column {col_name} not found in metadata")
        else:
            col_meta = metadata.columns[col_name]
            if col_meta["sdtype"] == "numerical":
                if col_meta["computer_representation"] == "Int64":
                    subtype = "int"
                elif col_meta["computer_representation"] == "Float":
                    subtype = "float"
                else:
                    raise ValueError(
                        f"Unknown computer representation {col_meta['computer_representatin']}"
                    )
        col_info[col_idx]["subtype"] = subtype
        col_info[col_idx]["max"] = float(train_df[col_idx].max())
        col_info[col_idx]["min"] = float(train_df[col_idx].min())

    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info[col_idx]["type"] = "categorical"
        col_info[col_idx]["subtype"] = "str"
        col_info[col_idx]["categorizes"] = train_df[col_idx].unique().tolist()

    info = {
        "name": name,
        "column_info": col_info,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
    }

    train_df.rename(columns=idx_name_mapping, inplace=True)
    test_df.rename(columns=idx_name_mapping, inplace=True)

    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()

    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(f"{save_dir}/X_num_train.npy", X_num_train)
    np.save(f"{save_dir}/X_cat_train.npy", X_cat_train)

    np.save(f"{save_dir}/X_num_test.npy", X_num_test)
    np.save(f"{save_dir}/X_cat_test.npy", X_cat_test)

    # save the index
    np.save(f"{save_dir}/idx.npy", idx)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)

    train_df.to_csv(f"{save_dir}/train.csv", index=False)
    test_df.to_csv(f"{save_dir}/test.csv", index=False)

    print("Numerical", X_num_train.shape)
    print("Categorical", X_cat_train.shape)

    info["column_names"] = column_names
    info["train_num"] = train_df.shape[0]
    info["test_num"] = test_df.shape[0]

    info["idx_mapping"] = idx_mapping
    info["inverse_idx_mapping"] = inverse_idx_mapping
    info["idx_name_mapping"] = idx_name_mapping

    metadata = {"columns": {}}

    for i in num_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "numerical"
        metadata["columns"][i]["computer_representation"] = "Float"

    for i in cat_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "categorical"

    info["metadata"] = metadata

    with open(f"{save_dir}/info.json", "w") as file:
        json.dump(info, file, indent=4)

    print(f"Processing and Saving {name} Successfully!")

    print(name)
    print("Total", info["train_num"] + info["test_num"])
    print("Train", info["train_num"])
    print("Test", info["test_num"])

    print("Num", info["num_col_idx"])
    print("Cat", info["cat_col_idx"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="rossmann_subsampled", type=str)
    parser.add_argument("--factor-missing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    DATA_PATH = "data"

    args = parse_args()
    dataset_name = args.dataset_name
    factor_missing = args.factor_missing

    metadata = Metadata().load_from_json(
        f"{DATA_PATH}/original/{dataset_name}/metadata.json"
    )
    tables = load_tables(f"{DATA_PATH}/original/{dataset_name}/", metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)

    has_nan = False
    for table_name, table in tables.items():
        if factor_missing:
            has_nan = has_nan or table.isnull().values.any()
    factor_missing = factor_missing and has_nan

    for table_name, table in tables.items():
        process_data(
            table,
            name=table_name,
            metadata=metadata.get_table_meta(table_name, to_dict=False),
            factor_missing=factor_missing,
            data_path=DATA_PATH,
            dataset_name=dataset_name,
        )
