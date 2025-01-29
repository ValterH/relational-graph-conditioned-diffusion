import os
import copy
import pickle

import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler

from relgdiff.data.tables_to_heterodata import tables_to_heterodata
from relgdiff.embedding_generation.hetero_gnns import GraphConditioning


mse_loss = torch.nn.MSELoss()


def train(
    model,
    optimizer,
    data,
    embedding_table,
    train_index,
    pos_enc=None,
    alpha=0.75,
):
    model.train()
    optimizer.zero_grad()
    if pos_enc is not None:
        pos_enc = data.pe_dict
    _, output = model(data.x_dict, data.edge_index_dict, pos_enc=pos_enc)

    loss_other = torch.tensor(0.0).to(data.x_dict[embedding_table].device)
    for i, table in enumerate(output.keys()):
        idx = train_index[table]
        if table == embedding_table:
            loss_target = mse_loss(output[table][idx], data[table].y[idx])
        else:
            loss_other += mse_loss(output[table][idx], data[table].y[idx])

    if i > 1:
        loss_other /= i

    # TODO: should the loss be weighted?
    loss = alpha * loss_target + (1 - alpha) * loss_other

    loss.backward()
    optimizer.step()
    return loss.item()


def test(
    model,
    data,
    embedding_table,
    test_index,
    best_model=None,
    best_val=0,
    pos_enc=None,
):
    model.eval()

    if pos_enc is not None:
        pos_enc = data.pe_dict
    _, output = model(data.x_dict, data.edge_index_dict, pos_enc=pos_enc)

    idx = test_index[embedding_table]
    loss = mse_loss(output[embedding_table][idx], data[embedding_table].y[idx])

    if loss < best_val:
        best_val = loss.item()
        best_model = copy.deepcopy(model)

    return loss.item(), best_model, best_val


def train_hetero_gnn(
    tables,
    metadata,
    embedding_table,
    masked_tables,
    latents={},
    pos_enc={},
    model_save_dir="ckpt/hetero_gnn",
    gnn_model="GIN",
    hidden_channels=128,
    num_layers=2,
    mlp_layers=3,
    embedding_dim=64,
    epochs=200,
    lr=0.008,
    weight_decay=1e-05,
    device="cuda",
    transform="standard",
    seed=None,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    hetero_data = tables_to_heterodata(
        copy.deepcopy(tables),
        metadata,
        masked_tables,
        embedding_table=embedding_table,
        latents=latents,
        pos_enc=pos_enc,
    )

    # select 90% of random nodes as train nodes
    train_index = dict()
    test_index = dict()
    for node_type in hetero_data.x_dict.keys():
        # if node_type == embedding_table:
        n = hetero_data.x_dict[node_type].shape[0]
        idx = torch.randperm(n)
        train_index[node_type] = idx[: int(n * 0.9)]
        test_index[node_type] = idx[int(n * 0.9) :]

    if type(embedding_dim) is int:
        out_channels = {embedding_table: embedding_dim}
    else:
        out_channels = embedding_dim

    model = GraphConditioning(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        types=list(hetero_data.x_dict.keys()),
        data=hetero_data,
        num_layers=num_layers,
        mlp_layers=mlp_layers,
        model_type=gnn_model,
        pos_enc=pos_enc,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=1
    )
    model.to(device)
    hetero_data.to(device)
    pbar = tqdm(range(epochs))
    best_model = None
    best_val = float("inf")
    for epoch in pbar:
        loss = train(
            model,
            optimizer,
            hetero_data,
            embedding_table,
            train_index,
            pos_enc=pos_enc,
        )
        val_loss, best_model, best_val = test(
            model,
            hetero_data,
            embedding_table,
            test_index,
            best_model,
            best_val,
            pos_enc=pos_enc,
        )
        pbar.set_description(
            f"Epoch {epoch + 1}: loss {round(loss, 5)}, val loss {round(val_loss, 5)} best val {round(best_val, 5)}"
        )
        scheduler.step()

    model = best_model

    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(
        model.state_dict(), os.path.join(model_save_dir, f"model_{embedding_table}.pt")
    )

    if transform is not None:
        if pos_enc is not None:
            pos_enc = hetero_data.pe_dict
        output, _ = model(
            hetero_data.x_dict, hetero_data.edge_index_dict, pos_enc=pos_enc
        )
        embeddings = output[embedding_table].cpu().detach().numpy()
        if transform == "minmax":
            scaler = MinMaxScaler(feature_range=(-1, 1))
        elif transform == "standard":
            scaler = StandardScaler()
        elif transform == "quantile":
            scaler = QuantileTransformer(output_distribution="normal")
        scaler.fit(embeddings)

        with open(
            os.path.join(model_save_dir, f"scaler_{embedding_table}.pkl"), "wb"
        ) as f:
            pickle.dump(scaler, f)

    return best_model, hetero_data


def compute_hetero_gnn_embeddings(
    hetero_data,
    embedding_table,
    model_save_dir="ckpt/hetero_gnn",
    gnn_model="GIN",
    hidden_channels=128,
    num_layers=2,
    mlp_layers=3,
    embedding_dim=60,
    pos_enc=None,
    transform="standard",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_save_path = os.path.join(model_save_dir, f"model_{embedding_table}.pt")

    if type(embedding_dim) is int:
        out_channels = {embedding_table: embedding_dim}
    else:
        out_channels = embedding_dim

    model = GraphConditioning(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        types=list(hetero_data.x_dict.keys()),
        data=hetero_data,
        num_layers=num_layers,
        mlp_layers=mlp_layers,
        model_type=gnn_model,
        pos_enc=pos_enc,
    ).to("cpu")

    # pass the data through the model to initialize the lazy dimensions
    hetero_data.to("cpu")
    if pos_enc is not None:
        pos_enc = hetero_data.pe_dict
    model(hetero_data.x_dict, hetero_data.edge_index_dict, pos_enc=pos_enc)
    model.load_state_dict(torch.load(model_save_path))

    # save last layer embeddings
    model.to(device)
    hetero_data.to(device)
    with torch.no_grad():
        model.eval()
        h, _ = model(hetero_data.x_dict, hetero_data.edge_index_dict, pos_enc=pos_enc)
    h = h[embedding_table].cpu().detach().numpy()

    if transform is not None:
        with open(
            os.path.join(model_save_dir, f"scaler_{embedding_table}.pkl"), "rb"
        ) as f:
            scaler = pickle.load(f)
        h = scaler.transform(h)
    return h


def main():
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from syntherela.metadata import Metadata
    from syntherela.data import load_tables, remove_sdv_columns
    from relgdiff.data.utils import get_table_order, get_positional_encoding

    seed = 42
    epochs = 100
    mlp_layers = 3
    hidden_channels = 128
    run = 1
    gnn_model = "GIN"  # "GAT"  # "GraphSAGE"  # "GATv2"  # "GAT"  #
    dataset_name = "walmart_subsampled"  # "CORA_v1"  #   "rossmann_subsampled"  #
    pos_enc = True
    factor_missing = True  #

    # load data
    metadata = Metadata().load_from_json(f"data/original/{dataset_name}/metadata.json")
    tables = load_tables(f"data/original/{dataset_name}/", metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)

    masked_tables = metadata.get_tables()
    table_order = get_table_order(metadata)
    num_layers = len(table_order)

    # load latents for supervision
    latents = {}
    embedding_dims = {}
    if pos_enc:
        pos_enc, _ = get_positional_encoding(dataset_name)

    for table in metadata.get_tables():
        if metadata.get_column_names(table) == metadata.get_column_names(
            table, sdtype="id"
        ):
            table_order.remove(table)
            continue

        table_save_path = f"{table}{'_factor' if factor_missing else ''}"
        latents[table] = np.load(
            f"ckpt/{dataset_name}/{table_save_path}/vae/{run}/latents.npy"
        )
        _, T, C = latents[table].shape
        embedding_dims[table] = (T - 1) * C
    print(embedding_dims)
    embeddings = []
    for table in table_order:
        _, hetero_data = train_hetero_gnn(
            tables,
            metadata,
            embedding_table=table,
            masked_tables=masked_tables,
            latents=latents,
            pos_enc=pos_enc,
            model_save_dir="ckpt/DEBUG",
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            mlp_layers=mlp_layers,
            embedding_dim=embedding_dims,
            epochs=epochs,
            gnn_model=gnn_model,
            transform="standard",
            seed=seed,
        )
        conditional_embeddings = compute_hetero_gnn_embeddings(
            hetero_data,
            embedding_table=table,
            model_save_dir="ckpt/DEBUG",
            embedding_dim=embedding_dims,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            mlp_layers=mlp_layers,
            pos_enc=pos_enc,
            gnn_model=gnn_model,
            transform="standard",
        )
        masked_tables.remove(table)
        embeddings.append(conditional_embeddings)

    fig, axes = plt.subplots(1, len(table_order), figsize=(len(table_order) * 5, 5))
    cmap = plt.get_cmap("Set1")
    for i, table in enumerate(table_order):
        print(f"{table} embeddings shape: {embeddings[i].shape}")
        pca = PCA(n_components=3)
        pca.fit(embeddings[i])
        hetero_gnn_embeddings = pca.transform(embeddings[i])
        axes[i].scatter(
            hetero_gnn_embeddings[:, 0],
            hetero_gnn_embeddings[:, 1],
            s=10,
            color=cmap(i),
        )
        axes[i].set_title(table)
        axes[i].axis("equal")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
