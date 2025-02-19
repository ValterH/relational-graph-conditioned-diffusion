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
from relgdiff.data.utils import get_table_order

DATA_PATH = "data"


############################################################################################


def train_pipline(
    dataset_name,
    retrain_vae=False,
    factor_missing=True,
    model_type="mlp",
    normalization="quantile",
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

    # train variational autoencoders
    for table in metadata.get_tables():
        # skip foreign key only tables
        if metadata.get_column_names(table) == metadata.get_column_names(
            table, sdtype="id"
        ):
            continue
        table_save_path = f"{dataset_name}/{table}{'_factor' if factor_missing else ''}"
        if retrain_vae or not os.path.exists(
            f"ckpt/{table_save_path}/vae/baseline/decoder.pt"
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
                ckpt_dir=f"ckpt/{table_save_path}/vae/baseline",
                epochs=epochs_vae,
                device=device,
                seed=seed,
            )
        else:
            print(f"Reusing VAE for table {table}")

    # train generative model for each table (latent conditional diffusion)
    for table in get_table_order(metadata):
        # skip foreign key only tables
        if metadata.get_column_names(table) == metadata.get_column_names(
            table, sdtype="id"
        ):
            continue
        table_save_path = f"{dataset_name}/{table}{'_factor' if factor_missing else ''}"

        os.makedirs(f"ckpt/{table_save_path}/", exist_ok=True)

        # train conditional diffusion
        train_z, _, _, ckpt_path, _ = get_input_train(
            table_save_path, is_cond=False, run="baseline"
        )
        print(f"Training unconditional diffusion for table {table}")
        train_diff(
            train_z,
            None,
            ckpt_path,
            epochs=epochs_diff,
            is_cond=False,
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
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset_name
    retrain_vae = args.retrain_vae
    epochs_vae = args.epochs_vae
    epochs_diff = args.epochs_diff
    factor_missing = args.factor_missing
    model_type = args.model_type
    normalization = args.normalization
    train_pipline(
        dataset_name=dataset_name,
        retrain_vae=retrain_vae,
        factor_missing=factor_missing,
        epochs_vae=epochs_vae,
        epochs_diff=epochs_diff,
        model_type=model_type,
        normalization=normalization,
    )


if __name__ == "__main__":
    main()
