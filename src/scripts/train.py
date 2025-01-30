import os
import argparse

import torch
import numpy as np
from syntherela.metadata import Metadata
from syntherela.data import load_tables, remove_sdv_columns

from relgdiff.generation.diffusion import train_diff
from relgdiff.generation.autoencoder import train_vae
from relgdiff.generation.utils_train import preprocess
from relgdiff.generation.tabsyn.latent_utils import get_input_train
from relgdiff.embedding_generation.embeddings import (
    train_hetero_gnn,
    compute_hetero_gnn_embeddings,
)
from relgdiff.data.utils import get_table_order, get_positional_encoding

DATA_PATH = "data"


############################################################################################


def train_pipline(
    dataset_name,
    run,
    retrain_vae=False,
    factor_missing=True,
    model_type="mlp",
    normalization="quantile",
    gnn_hidden=128,
    mlp_layers=3,
    positional_enc=True,
    epochs_gnn=250,
    epochs_vae=4000,
    epochs_diff=4000,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # read data
    metadata = Metadata().load_from_json(
        f"{DATA_PATH}/original/{dataset_name}/metadata.json"
    )
    tables = load_tables(f"{DATA_PATH}/original/{dataset_name}/", metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)
    gnn_layers = len(metadata.get_tables())
    if positional_enc:
        pos_enc, positional_enc = get_positional_encoding(dataset_name)
    else:
        pos_enc = None

    masked_tables = metadata.get_tables()

    # train variational autoencoders
    latents = {}
    embedding_dims = {}
    for table in metadata.get_tables():
        # skip foreign key only tables
        if metadata.get_column_names(table) == metadata.get_column_names(
            table, sdtype="id"
        ):
            continue
        table_save_path = f'{dataset_name}/{table}{"_factor" if factor_missing else ""}'
        if retrain_vae or not os.path.exists(
            f"ckpt/{table_save_path}/vae/{run}/decoder.pt"
        ):
            print(f"Training VAE for table {table}")
            X_num, X_cat, idx, categories, d_numerical = preprocess(
                dataset_path=f"{DATA_PATH}/processed/{table_save_path}",
                normalization=normalization,
            )
            train_vae(
                X_num,
                X_cat,
                idx,
                categories,
                d_numerical,
                ckpt_dir=f"ckpt/{table_save_path}/vae/{run}",
                epochs=epochs_vae,
                device=device,
                seed=seed,
            )
        else:
            print(f"Reusing VAE for table {table}")
        table_latents = np.load(f"ckpt/{table_save_path}/vae/{run}/latents.npy")
        latents[table] = table_latents
        _, T, C = table_latents.shape
        embedding_dims[table] = (T - 1) * C

    # train generative model for each table (latent conditional diffusion)
    for table in get_table_order(metadata):
        # skip foreign key only tables
        if metadata.get_column_names(table) == metadata.get_column_names(
            table, sdtype="id"
        ):
            continue
        table_save_path = f'{dataset_name}/{table}{"_factor" if factor_missing else ""}'

        # train GNN
        gnn_save_dir = f'ckpt/{dataset_name}/hetero_gnn/{"factor" if factor_missing else ""}{"pe" if positional_enc else ""}'
        _, hetero_data = train_hetero_gnn(
            tables,
            metadata,
            embedding_table=table,
            masked_tables=masked_tables,
            latents=latents,
            pos_enc=pos_enc,
            model_save_dir=gnn_save_dir,
            hidden_channels=gnn_hidden,
            num_layers=gnn_layers,
            mlp_layers=mlp_layers,
            embedding_dim=embedding_dims,
            epochs=epochs_gnn,
            seed=seed,
        )
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

        os.makedirs(f"ckpt/{table_save_path}/", exist_ok=True)
        np.save(f"ckpt/{table_save_path}/cond_train_z.npy", conditional_embeddings)

        # train conditional diffusion
        train_z, train_z_cond, _, ckpt_path, _ = get_input_train(
            table_save_path, is_cond=True, run=run
        )
        print(f"Training conditional diffusion for table {table}")
        train_diff(
            train_z,
            train_z_cond,
            ckpt_path,
            epochs=epochs_diff,
            is_cond=True,
            model_type=model_type,
            device=device,
            seed=seed,
        )


############################################################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="rossmann_subsampled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain_vae", action="store_true")
    parser.add_argument("--epochs-vae", type=int, default=4000)
    parser.add_argument("--epochs-diff", type=int, default=10000)
    parser.add_argument("--epochs-gnn", type=int, default=1000)
    parser.add_argument("--gnn-hidden", type=int, default=128)
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
    retrain_vae = args.retrain_vae
    epochs_vae = args.epochs_vae
    epochs_diff = args.epochs_diff
    epochs_gnn = args.epochs_gnn
    gnn_hidden = args.gnn_hidden
    factor_missing = args.factor_missing
    positional_enc = args.positional_enc
    model_type = args.model_type
    normalization = args.normalization
    if args.run is not None:
        run = args.run
    else:
        run = f'{model_type}{"_factor" if factor_missing else ""}{"_pe" if positional_enc else ""}'

    train_pipline(
        dataset_name=dataset_name,
        run=run,
        retrain_vae=retrain_vae,
        factor_missing=factor_missing,
        positional_enc=positional_enc,
        epochs_vae=epochs_vae,
        epochs_diff=epochs_diff,
        epochs_gnn=epochs_gnn,
        model_type=model_type,
        gnn_hidden=gnn_hidden,
        normalization=normalization,
    )


if __name__ == "__main__":
    main()
