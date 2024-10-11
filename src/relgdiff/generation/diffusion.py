import os
import warnings
import time

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from relgdiff.generation.tabsyn.model import MLPDiffusion, Model
from relgdiff.generation.tabsyn.tabular_unet import tabularUnet
from relgdiff.generation.tabsyn.latent_utils import (
    get_input_train,
    get_input_generate,
    recover_data,
    split_num_cat,
)
from relgdiff.generation.tabsyn.diffusion_utils import sample

warnings.filterwarnings("ignore")


def get_denoise_function(
    in_dim,
    in_dim_cond,
    is_cond,  # MLPDiffusion only
    encoder_dim=[64, 128, 256],  # TabularUnet only
    embed_dim=16,  # TabularUnet only
    device="cuda:0",
    model_type="mlp",
):
    if model_type == "mlp":
        denoise_fn = MLPDiffusion(
            in_dim, 1024, is_cond=is_cond, d_in_cond=in_dim_cond
        ).to(device)
    elif model_type == "unet":
        denoise_fn = tabularUnet(
            in_dim, in_dim_cond, embed_dim=embed_dim, encoder_dim=encoder_dim
        )
    else:
        raise ValueError(f"Model type {model_type} not recognized")

    return denoise_fn


def train_diff(
    train_z,
    train_z_cond,
    ckpt_path,
    epochs=4000,
    early_stopping_patience=1000,
    is_cond=False,
    model_type="mlp",
    device="cuda:0",
    seed=None,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1]
    if is_cond:
        in_dim_cond = train_z_cond.shape[1]
    else:
        in_dim_cond = None

    mean, _ = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / 2
    train_data = train_z

    if is_cond:
        train_data = torch.cat([train_z, train_z_cond], dim=1)
    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    denoise_fn = get_denoise_function(
        in_dim, in_dim_cond, is_cond=is_cond, device=device, model_type=model_type
    )

    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1], is_cond=is_cond).to(
        device
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=20, verbose=True
    )

    model.train()

    best_loss = float("inf")
    patience = 0
    start_time = time.time()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)

            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                {
                    "Loss": loss.item(),
                    "Best Loss": best_loss,
                    "LR": optimizer.param_groups[0]["lr"],
                }
            )

        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = loss.item()
            patience = 0
            torch.save(model.state_dict(), f"{ckpt_path}/model.pt")
        else:
            patience += 1
            if patience == early_stopping_patience:
                print("Early stopping")
                break

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f"{ckpt_path}/model_{epoch}.pt")

    end_time = time.time()
    print("Time: ", end_time - start_time)


def sample_diff(
    dataname,
    run,
    is_cond=True,
    model_type="mlp",
    device="cuda:0",
    num_samples=None,
    ids=None,
    denoising_steps=50,
    normalization="quantile",
):
    if is_cond:
        cond_embedding_save_path = f"ckpt/{dataname}/{run}/gen/cond_z.npy"
        train_z_cond = torch.tensor(np.load(cond_embedding_save_path)).float()
        B, in_dim_cond = train_z_cond.size()
        train_z_cond = train_z_cond.view(B, in_dim_cond).to(device)
        num_samples = B
        if ids is None:
            ids = np.arange(num_samples)
    else:
        train_z_cond = None
        in_dim_cond = None
        if ids is None:
            ids = np.arange(num_samples)

    train_z, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(
        dataname, run=run, normalization=normalization
    )
    in_dim = train_z.shape[1]

    mean = train_z.mean(0)

    denoise_fn = get_denoise_function(
        in_dim, in_dim_cond, is_cond, device="cuda:0", model_type=model_type
    )

    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1], is_cond=is_cond).to(
        device
    )

    model.load_state_dict(torch.load(f"{ckpt_path}/model.pt"))

    """
        Generating samples    
    """
    start_time = time.time()

    sample_dim = in_dim

    x_next = sample(
        model.denoise_fn_D,
        num_samples,
        sample_dim,
        device=device,
        z_cond=train_z_cond,
        num_steps=denoising_steps,
    )
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat = split_num_cat(syn_data, info, num_inverse, cat_inverse)

    syn_df = recover_data(syn_num, syn_cat, info)

    idx_name_mapping = info["idx_name_mapping"]
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    # convert data type
    for col in syn_df.columns:
        datatype = info["column_info"][str(col)]["subtype"]
        if datatype == "int":
            if syn_df[col].dtype == "object":
                # parse float from string (model outputs floats)
                syn_df[col] = syn_df[col].apply(lambda x: float(x))
            elif syn_df[col].dtype == "float32":
                # round float to integer to avoid poor conversion to int
                syn_df[col] = syn_df[col].round()
        syn_df[col] = syn_df[col].astype(datatype)
        # convert missing values ('nan') to NaN to match original data
        if datatype == "str":
            syn_df[col] = syn_df[col].replace("nan", np.nan)
    syn_df.rename(columns=idx_name_mapping, inplace=True)

    end_time = time.time()
    print("Time:", end_time - start_time)

    return syn_df


if __name__ == "__main__":
    train_z, train_z_cond, _, ckpt_path, _ = get_input_train("store", is_cond=False)
    train_diff(
        train_z, train_z_cond, ckpt_path, epochs=10, is_cond=False, device="cuda:0"
    )
