import json
import numpy as np
import pandas as pd
import torch

from relgdiff.generation.utils_train import preprocess
from relgdiff.generation.vae.model import Decoder_model


def get_input_train(dataname, is_cond, run):
    dataset_dir = f"data/processed/{dataname}"

    with open(f"{dataset_dir}/info.json", "r") as f:
        info = json.load(f)

    ckpt_dir = f"ckpt/{dataname}/{run}"
    embedding_save_path = f"ckpt/{dataname}/vae/{run}/latents.npy"
    if is_cond:
        cond_embedding_save_path = f"ckpt/{dataname}/cond_train_z.npy"
        train_z_cond = torch.tensor(np.load(cond_embedding_save_path)).float()
    else:
        train_z_cond = None
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim

    train_z = train_z.view(B, in_dim)

    return train_z, train_z_cond, dataset_dir, ckpt_dir, info


def get_input_generate(dataname, run, normalization="quantile"):
    dataset_dir = f"data/processed/{dataname}"
    ckpt_dir = f"ckpt/{dataname}/{run}"

    with open(f"{dataset_dir}/info.json", "r") as f:
        info = json.load(f)

    _, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(
        dataset_dir, inverse=True, normalization=normalization
    )

    embedding_save_path = f"ckpt/{dataname}/vae/{run}/latents.npy"
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]

    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim

    train_z = train_z.view(B, in_dim)
    pre_decoder = Decoder_model(2, d_numerical, categories, 4, n_head=1, factor=32)

    decoder_save_path = f"ckpt/{dataname}/vae/{run}/decoder.pt"
    pre_decoder.load_state_dict(torch.load(decoder_save_path))

    info["pre_decoder"] = pre_decoder
    info["token_dim"] = token_dim

    return train_z, dataset_dir, ckpt_dir, info, num_inverse, cat_inverse


@torch.no_grad()
def split_num_cat(syn_data, info, num_inverse, cat_inverse):
    if torch.cuda.is_available():
        pre_decoder = info["pre_decoder"].cuda()
    else:
        pre_decoder = info["pre_decoder"].cpu()

    token_dim = info["token_dim"]

    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)
    if torch.cuda.is_available():
        norm_input = pre_decoder(torch.tensor(syn_data).cuda())
    else:
        norm_input = pre_decoder(torch.tensor(syn_data).cpu())
    x_hat_num, x_hat_cat = norm_input

    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim=-1))

    syn_num = x_hat_num.cpu().numpy()
    syn_cat = torch.stack(syn_cat).t().cpu().numpy()

    syn_num = num_inverse(syn_num)
    syn_cat = cat_inverse(syn_cat)
    return syn_num, syn_cat


def recover_data(syn_num, syn_cat, info):
    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]

    idx_mapping = info["idx_mapping"]
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    for i in range(len(num_col_idx) + len(cat_col_idx)):
        if i in set(num_col_idx):
            syn_df[i] = syn_num[:, idx_mapping[i]]
        elif i in set(cat_col_idx):
            syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]

    return syn_df


def process_invalid_id(syn_cat, min_cat, max_cat):
    syn_cat = np.clip(syn_cat, min_cat, max_cat)

    return syn_cat
