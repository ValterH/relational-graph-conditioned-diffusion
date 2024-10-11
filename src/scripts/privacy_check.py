import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

from syntherela.metadata import Metadata
from syntherela.data import load_tables, remove_sdv_columns


def load_data(dataset_name, run=1, data_path="data"):
    metadata = Metadata().load_from_json(
        f"{data_path}/original/{dataset_name}/metadata.json"
    )

    tables = load_tables(f"{data_path}/original/{dataset_name}/", metadata)

    tables, metadata = remove_sdv_columns(tables, metadata)
    tables_synthetic = load_tables(
        f"{data_path}/synthetic/{dataset_name}/ours/{run}/sample1", metadata
    )

    return tables, tables_synthetic, metadata


def prepare_data(table_real, table_syn, metadata, table_name):
    id_columns = metadata.get_column_names(table_name, sdtype="id")
    numerical_columns = metadata.get_column_names(table_name, sdtype="numerical")
    categorical_columns = metadata.get_column_names(table_name, sdtype="categorical")
    datetime_columns = metadata.get_column_names(table_name, sdtype="datetime")

    for column in categorical_columns:
        table_real[column] = pd.Categorical(table_real[column])
        table_syn[column] = pd.Categorical(
            table_syn[column], categories=table_real[column].cat.categories
        )

    table_real.drop(columns=id_columns, inplace=True)
    table_syn.drop(columns=id_columns, inplace=True)

    for column in datetime_columns:
        table_real[column] = (
            pd.to_datetime(table_real[column]).astype(np.int64) // 10**9
        )
        table_syn[column] = pd.to_datetime(table_syn[column]).astype(np.int64) // 10**9

    numerical_columns += datetime_columns

    table_real[numerical_columns] = table_real[numerical_columns].fillna(0)
    table_syn[numerical_columns] = table_syn[numerical_columns].fillna(0)

    table_real = pd.get_dummies(table_real, columns=categorical_columns).astype(
        np.float32
    )
    table_syn = pd.get_dummies(table_syn, columns=categorical_columns).astype(
        np.float32
    )

    table_train, table_test = train_test_split(
        table_real, test_size=0.5, random_state=42
    )

    scaler = StandardScaler()
    scaler.fit(table_train[numerical_columns])

    table_train[numerical_columns] = scaler.transform(table_train[numerical_columns])
    table_test[numerical_columns] = scaler.transform(table_test[numerical_columns])
    table_syn[numerical_columns] = scaler.transform(table_syn[numerical_columns])

    return table_train, table_test, table_syn


def evaluate_dcr(table_train, table_syn, table_test, seed=42):
    if seed is not None:
        np.random.seed(seed)
    real_data = table_train.values
    syn_data = table_syn.sample(n=table_test.shape[0], replace=True).values
    test_data = table_test.values

    dcrs_syn = []
    dcrs_test = []
    batch_size = 100

    for i in range((syn_data.shape[0] // batch_size) + 1):
        syn_data_batch = syn_data[i * batch_size : (i + 1) * batch_size]
        test_data_batch = test_data[i * batch_size : (i + 1) * batch_size]
        if syn_data_batch.shape[0] == 0:
            break

        dcr_syn = pairwise_distances(syn_data_batch, real_data, metric="euclidean").min(
            axis=1
        )
        dcr_test = pairwise_distances(
            test_data_batch, real_data, metric="euclidean"
        ).min(axis=1)
        dcrs_syn.append(dcr_syn)
        dcrs_test.append(dcr_test)

    dcrs_syn = np.concatenate(dcrs_syn)
    dcrs_test = np.concatenate(dcrs_test)

    score = (dcrs_test < dcrs_syn).mean()
    return dcrs_syn, dcrs_test, score


def estimate_dcr_score(tables, tables_synthetic, table_name, metadata, m=10, seed=42):
    table_real = tables[table_name]
    table_syn = tables_synthetic[table_name]
    table_syn = table_syn[table_real.columns]
    table_train, table_test, table_syn = prepare_data(
        table_real.copy(), table_syn.copy(), metadata, table_name
    )
    scores = []
    dcr_syn = []
    dcr_test = []
    for i in tqdm(range(m)):
        dcrs_syn, dcrs_test, score = evaluate_dcr(
            table_train, table_syn, table_test, seed=seed + i
        )
        scores.append(score)
        dcr_syn.append(dcrs_syn)
        dcr_test.append(dcrs_test)

    print(
        f"Mean score: {np.mean(scores):.4f} +- {np.std(scores) / np.sqrt(len(scores)) :.4f}"
    )
    dcr_syn = np.concatenate(dcr_syn)
    dcr_test = np.concatenate(dcr_test)
    return scores, dcr_syn, dcr_test


if __name__ == "__main__":
    dataset_name = "airbnb-simplified_subsampled"
    tables, tables_synthetic, metadata = load_data(dataset_name)
    table_name = "users"

    scores, dcr_syn, dcr_test = estimate_dcr_score(
        tables, tables_synthetic, table_name, metadata, m=100, seed=1
    )

    nbins = 100
    max_val = np.quantile(dcr_syn, 0.975)

    plt.figure(figsize=(10, 10))
    bins = plt.hist(dcr_test, bins=nbins, range=(0, max_val), alpha=0.5, label="Real")
    plt.hist(dcr_syn, bins=bins[1], range=(0, max_val), alpha=0.5, label="Synthetic")
    plt.legend(fontsize=15)
    plt.savefig(f"results/dcr_{dataset_name}_{table_name}.png")

    dataset_name = "rossmann_subsampled"
    tables, tables_synthetic, metadata = load_data(dataset_name)
    table_name = "store"

    scores, dcr_syn, dcr_test = estimate_dcr_score(
        tables, tables_synthetic, table_name, metadata, m=100, seed=1
    )

    nbins = 100
    max_val = np.quantile(dcr_syn, 0.99)

    plt.figure(figsize=(10, 10))
    bins = plt.hist(dcr_test, bins=nbins, range=(0, max_val), alpha=0.5, label="Real")
    plt.hist(dcr_syn, bins=bins[1], range=(0, max_val), alpha=0.5, label="Synthetic")
    plt.legend(fontsize=15)
    plt.savefig(f"results/dcr_{dataset_name}_{table_name}.png")
