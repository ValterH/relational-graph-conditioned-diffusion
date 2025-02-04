import argparse

import torch
import numpy as np
import pandas as pd
from syntherela.metadata import Metadata
from syntherela.data import save_tables, load_tables, remove_sdv_columns

from relgdiff.generation.diffusion import sample_diff

DATA_PATH = "data"


############################################################################################


def sample(
    dataset_name,
    factor_missing=True,
    model_type="mlp",
    seed=None,
    denoising_steps=50,
    normalization="quantile",
    sample_idx=None,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # read data
    metadata = Metadata().load_from_json(
        f"{DATA_PATH}/original/{dataset_name}/metadata.json"
    )
    tables_orig = load_tables(f"{DATA_PATH}/original/{dataset_name}/", metadata)
    tables_orig, metadata = remove_sdv_columns(tables_orig, metadata)
    tables = dict()

    # for each table in dataset
    for table in metadata.get_tables():
        # skip foreign key only tables
        if metadata.get_column_names(table) == metadata.get_column_names(
            table, sdtype="id"
        ):
            continue
        table_save_path = f"{dataset_name}/{table}{'_factor' if factor_missing else ''}"

        # sample diffusion
        df = sample_diff(
            table_save_path,
            run="baseline",
            num_samples=len(tables_orig[table]),
            is_cond=False,
            model_type=model_type,
            device=device,
            denoising_steps=denoising_steps,
            normalization=normalization,
        )

        # postprocess the data
        table_metadata = metadata.get_table_meta(table, to_dict=False)

        # handle missing values
        cat_columns = df.select_dtypes(include=["object"]).columns.to_list()
        for col in cat_columns:
            if col not in table_metadata.columns and factor_missing:
                imputed_column = col.split("_missing")[0]
                missing_mask = df[col].astype(int).astype(bool)
                df[imputed_column] = df[imputed_column].astype("float64")
                df.loc[missing_mask, imputed_column] = np.nan
                df = df.drop(columns=[col])
                continue
            elif "?" in df[col].unique():
                df[col] = df[col].replace("?", np.nan)

        # convert dates to datetime
        datetime_columns = table_metadata.get_column_names(sdtype="datetime")
        for col in datetime_columns:
            date_columns = [f"{col}_Year", f"{col}_Month", f"{col}_Day"]
            date_df = pd.DataFrame(
                df[date_columns].values, columns=["year", "month", "day"]
            ).round(0)
            fmt = "%Y%m%d"
            if f"{col}_Hour" in df.columns:
                date_df["hour"] = df[f"{col}_Hour"].values.round(0)
                date_df["minute"] = df[f"{col}_Minute"].values.round(0)
                date_df["second"] = df[f"{col}_Second"].values.round(0)
                date_columns.extend([f"{col}_Hour", f"{col}_Minute", f"{col}_Second"])
                fmt += "%H%M%S"
            df[col] = pd.to_datetime(dict(date_df), format=fmt, errors="coerce")
            df = df.drop(columns=date_columns)

        # add primary key
        pk = metadata.get_primary_key(table)
        if pk is not None:
            df[pk] = np.arange(len(df))

        print(f"Successfully sampled data for table {table}")
        print(df.head())
        tables[table] = df

    save_path = f"{DATA_PATH}/synthetic/{dataset_name}/baseline"
    if sample_idx is not None:
        save_path = f"{save_path}/sample{sample_idx}"
    save_tables(tables, save_path)


############################################################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="rossmann_subsampled")
    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("--num-structures", default=None, type=int)
    parser.add_argument("--gnn-hidden", type=int, default=128)
    parser.add_argument("--denoising-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--factor-missing", action="store_true")
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument(
        "--model-type", type=str, default="mlp", choices=["mlp", "unet"]
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="quantile",
        choices=["quantile", "standard", "cdf"],
    )
    parser.add_argument(
        "--embedding-task",
        type=str,
        default="reconstruction",
        choices=["reconstruction", "node_classification"],
    )
    parser.add_argument("--use-original-structure", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset_name
    num_samples = args.num_samples
    denoising_steps = args.denoising_steps
    factor_missing = args.factor_missing
    seed = args.seed
    model_type = args.model_type
    normalization = args.normalization

    for i in range(1, num_samples + 1):
        sample(
            dataset_name=dataset_name,
            model_type=model_type,
            normalization=normalization,
            factor_missing=factor_missing,
            denoising_steps=denoising_steps,
            seed=seed,
            sample_idx=i,
        )


if __name__ == "__main__":
    main()
