import os
import argparse

import torch
import numpy as np
import pandas as pd
from syntherela.metadata import Metadata
from syntherela.data import save_tables

from relgdiff.data.update_graph import update_graph_features
from relgdiff.data.sample_structures import sample_structures
from relgdiff.generation.diffusion import sample_diff
from relgdiff.embedding_generation.embeddings import compute_hetero_gnn_embeddings
from relgdiff.data.utils import get_table_order, get_positional_encoding

DATA_PATH = "data"


############################################################################################


def sample(
    dataset_name,
    num_samples,
    run,
    factor_missing=True,
    model_type="mlp",
    seed=None,
    denoising_steps=50,
    gnn_hidden=128,
    mlp_layers=3,
    positional_enc=True,
    normalization="quantile",
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # read data
    metadata = Metadata().load_from_json(
        f"{DATA_PATH}/original/{dataset_name}/metadata.json"
    )
    tables = dict()

    gnn_layers = len(metadata.get_tables())
    if positional_enc:
        pos_enc, positional_enc = get_positional_encoding(dataset_name)
    else:
        pos_enc = None

    # create graph
    masked_tables = metadata.get_tables()
    hetero_data = sample_structures(
        data_path=f"{DATA_PATH}/original/{dataset_name}",
        metadata=metadata,
        num_structures=num_samples,
        pos_enc=pos_enc,
    )

    # Read latentent embeddings dimensionts to obtain GNN dimensions
    embedding_dims = {}
    for table in metadata.get_tables():
        # skip foreign key only tables
        if metadata.get_column_names(table) == metadata.get_column_names(
            table, sdtype="id"
        ):
            continue
        table_save_path = f'{dataset_name}/{table}{"_factor" if factor_missing else ""}'
        table_latents = np.load(f"ckpt/{table_save_path}/vae/latents.npy")
        _, T, C = table_latents.shape
        embedding_dims[table] = (T - 1) * C

    # for each table in dataset
    for table in get_table_order(dataset_name):
        # skip foreign key only tables
        if metadata.get_column_names(table) == metadata.get_column_names(
            table, sdtype="id"
        ):
            # add primary key
            pk = metadata.get_primary_key(table)
            df = pd.DataFrame(columns=[pk])
            df[pk] = np.arange(hetero_data[table].x.shape[0])
            # add foreign keys
            for parent in metadata.get_parents(table):
                for foreign_key in metadata.get_foreign_keys(parent, table):
                    fks = (
                        hetero_data[(parent, foreign_key, table)]
                        .edge_index[0]
                        .cpu()
                        .numpy()
                    )
                    df[foreign_key] = fks
            print(f"Successfully sampled data for table {table}")
            print(df.head())
            tables[table] = df
            continue
        table_save_path = f'{dataset_name}/{table}{"_factor" if factor_missing else ""}'
        # compute GNN embeddings
        gnn_save_dir = f'ckpt/{dataset_name}/hetero_gnn/{"factor" if factor_missing else ""}{"pe" if positional_enc else ""}'
        conditional_embeddings = compute_hetero_gnn_embeddings(
            hetero_data,
            embedding_table=table,
            model_save_dir=gnn_save_dir,
            embedding_dim=embedding_dims,
            hidden_channels=gnn_hidden,
            num_layers=gnn_layers,
            mlp_layers=mlp_layers,
            pos_enc=pos_enc,
        )
        masked_tables.remove(table)

        os.makedirs(f"ckpt/{table_save_path}/{run}/gen", exist_ok=True)
        np.save(f"ckpt/{table_save_path}/{run}/gen/cond_z.npy", conditional_embeddings)

        # sample diffusion
        df = sample_diff(
            table_save_path,
            run,
            is_cond=True,
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
        # add foreign keys
        for parent in metadata.get_parents(table):
            for foreign_key in metadata.get_foreign_keys(parent, table):
                fks = (
                    hetero_data[(parent, foreign_key, table)]
                    .edge_index[0]
                    .cpu()
                    .numpy()
                )
                df[foreign_key] = fks

        print(f"Successfully sampled data for table {table}")
        print(df.head())
        tables[table] = df

        # update the features of hetero_data
        if len(tables) < len(metadata.get_tables()):
            hetero_data = update_graph_features(hetero_data, df, table, metadata)

    save_tables(tables, f"{DATA_PATH}/synthetic/{dataset_name}/ours/{run}/1/sample1")


############################################################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="rossmann_subsampled")
    parser.add_argument("--num-samples", default=None, type=int)
    parser.add_argument("--gnn-hidden", type=int, default=128)
    parser.add_argument("--denoising-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--factor-missing", action="store_true")
    parser.add_argument("--positional-enc", action="store_true")
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
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset_name
    num_samples = args.num_samples
    denoising_steps = args.denoising_steps
    factor_missing = args.factor_missing
    positional_enc = args.positional_enc
    seed = args.seed
    model_type = args.model_type
    gnn_hidden = args.gnn_hidden
    normalization = args.normalization
    if args.run is not None:
        run = args.run
    else:
        run = f'{model_type}{"_factor" if factor_missing else ""}{"_pe" if positional_enc else ""}'

    sample(
        dataset_name=dataset_name,
        num_samples=num_samples,
        run=run,
        model_type=model_type,
        normalization=normalization,
        factor_missing=factor_missing,
        positional_enc=positional_enc,
        denoising_steps=denoising_steps,
        gnn_hidden=gnn_hidden,
        seed=seed,
    )


if __name__ == "__main__":
    main()
