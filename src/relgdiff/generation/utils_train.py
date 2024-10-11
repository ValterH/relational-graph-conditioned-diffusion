import os

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
import relgdiff.generation.tabsyn_utils as tabsyn_utils


def get_dummy_numerical_features(X_cat):
    x_train = X_cat["train"][:, -1:]
    x_test = X_cat["test"][:, -1:]
    # check if conversion to float is possible
    if x_train[0][0].replace(".", "").replace("-", "").isnumeric():
        t_train = x_train.astype(float)
        t_test = x_test.astype(float)
    else:
        t_train = np.ones_like(x_train).astype(float)
        t_test = np.ones_like(x_test).astype(float)
    return t_train, t_test


def get_dummy_categorical_features(X_num):
    x_train = X_num["train"][:, -1:]
    x_test = X_num["test"][:, -1:]
    if len(np.unique(x_train)) < 10:
        t_train = x_train.astype(int).astype(str)
        t_test = x_test.astype(int).astype(str)
    else:
        # discretize
        bins = np.linspace(np.min(x_train), np.max(x_train), 10)
        t_train = np.digitize(x_train, bins).astype(str)
        t_test = np.digitize(x_test, bins).astype(str)
    return t_train, t_test


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0]


def preprocess(
    dataset_path,
    task_type="binclass",
    inverse=False,
    cat_encoding=None,
    normalization="quantile",
):
    T_dict = {}

    T_dict["normalization"] = normalization
    T_dict["num_nan_policy"] = None  # handled in preprocessing
    T_dict["cat_nan_policy"] = None
    T_dict["cat_min_frequency"] = None
    T_dict["cat_encoding"] = cat_encoding
    T_dict["y_policy"] = "default"

    T = tabsyn_utils.Transformations(**T_dict)

    dataset = make_dataset(
        data_path=dataset_path,
        T=T,
        task_type=task_type,
        change_val=False,
    )

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        X_train_num, X_test_num = X_num["train"], X_num["test"]
        X_train_cat, X_test_cat = X_cat["train"], X_cat["test"]

        categories = tabsyn_utils.get_categories(X_train_cat)
        d_numerical = X_train_num.shape[1]

        X_num = (X_train_num, X_test_num)
        X_cat = (X_train_cat, X_test_cat)

        idx = np.load(os.path.join(dataset_path, "idx.npy"))

        if inverse:
            num_inverse = dataset.num_transform.inverse_transform
            cat_inverse = dataset.cat_transform.inverse_transform

            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
        else:
            return X_num, X_cat, idx, categories, d_numerical
    else:
        return dataset


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)


def make_dataset(
    data_path: str, T: tabsyn_utils.Transformations, task_type, change_val: bool
):
    # classification
    if task_type == "binclass" or task_type == "multiclass":
        X_cat = (
            {} if os.path.exists(os.path.join(data_path, "X_cat_train.npy")) else None
        )
        X_num = (
            {} if os.path.exists(os.path.join(data_path, "X_num_train.npy")) else None
        )

        for split in ["train", "test"]:
            X_num_t, X_cat_t = tabsyn_utils.read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
    else:
        # regression
        X_cat = (
            {} if os.path.exists(os.path.join(data_path, "X_cat_train.npy")) else None
        )
        X_num = (
            {} if os.path.exists(os.path.join(data_path, "X_num_train.npy")) else None
        )

        for split in ["train", "test"]:
            X_num_t, X_cat_t, y_t = tabsyn_utils.read_pure_data(data_path, split)

            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t

    info = tabsyn_utils.load_json(os.path.join(data_path, "info.json"))

    # TODO: SUPPORT NO NUMERICAL FEATURES
    # when there are no numerical features
    # add a dummy numerical feature from
    # the categorical features
    if X_num["train"].shape[1] == 0:
        x_num_train, x_num_test = get_dummy_numerical_features(X_cat)
        X_num["train"] = x_num_train.reshape(-1, 1)
        X_num["test"] = x_num_test.reshape(-1, 1)
        T.__dict__["normalization"] = "minmax"
    if X_cat["train"].shape[1] == 0:
        x_cat_train, x_cat_test = get_dummy_categorical_features(X_num)
        X_cat["train"] = x_cat_train
        X_cat["test"] = x_cat_test
    D = tabsyn_utils.Dataset(
        X_num,
        X_cat,
        np.zeros(
            (X_num["train"].shape[0], 1)
        ),  # dummy y TODO: find a way to circumvent this
        y_info={},
        task_type=tabsyn_utils.TaskType(task_type),
        n_classes=info.get("n_classes"),
    )

    if change_val:
        D = tabsyn_utils.change_val(D)

    # def categorical_to_idx(feature):
    #     unique_categories = np.unique(feature)
    #     idx_mapping = {category: index for index, category in enumerate(unique_categories)}
    #     idx_feature = np.array([idx_mapping[category] for category in feature])
    #     return idx_feature

    # for split in ['train', 'val', 'test']:
    # D.y[split] = categorical_to_idx(D.y[split].squeeze(1))

    return tabsyn_utils.transform_dataset(D, T, None)
