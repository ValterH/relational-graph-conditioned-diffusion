import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero, HeteroDictLinear
from torch_geometric.nn.models import GAT, EdgeCNN, GraphSAGE, GIN, MLP

from relgdiff.generation.tabsyn.model import PositionalEmbedding


class GraphConditioning(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: dict,
        types: list,
        data: HeteroData,
        num_layers: int = 2,
        mlp_layers: int = 3,
        model_type: str = "GIN",
        aggr: str = "sum",
        model_kwargs: dict = {"jk": "cat"},
        dropout: float = 0.0,
        pos_enc: dict = None,
        embedding_dim: int = 64,
    ):
        super(GraphConditioning, self).__init__()

        model_kwargs["dropout"] = dropout
        self.proj = HeteroDictLinear(
            in_channels=-1, out_channels=hidden_channels, types=types
        )
        self.gnn = build_hetero_gnn(
            model_type=model_type,
            data=data,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            aggr=aggr,
            model_kwargs=model_kwargs,
            out_channels=embedding_dim,
        )

        self.mlps = HeteroMLP(
            in_channels=embedding_dim,
            hidden_channels=2 * hidden_channels,
            out_channels=out_channels,
            num_layers=mlp_layers,
            dropout=dropout,
        )

        if pos_enc is not None:
            self.pos_enc = nn.ModuleDict()
            self.pos_embeds = nn.ModuleDict()
            for table_name, _ in pos_enc.items():
                if table_name not in types:
                    continue
                self.pos_enc[table_name] = PositionalEmbedding(
                    num_channels=hidden_channels,
                    max_positions=int(data.pe_dict[table_name].max() * 2),
                )
                self.pos_embeds[table_name] = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )

    def forward(self, x_dict, edge_index, pos_enc=None):
        x_dict = self.proj(x_dict)
        if pos_enc is not None:
            for table_name, positions in pos_enc.items():
                pos_emb = self.pos_enc[table_name](
                    positions.to(x_dict[table_name].device)
                )
                pos_emb = self.pos_embeds[table_name](pos_emb)
                x_dict[table_name] += pos_emb
        x_dict = self.gnn(x_dict, edge_index)
        out = self.mlps(x_dict)
        return x_dict, out


class HeteroMLP(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super(HeteroMLP, self).__init__()
        self.mlps = torch.nn.ModuleDict()
        for key in out_channels.keys():
            self.mlps[key] = MLP(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels[key],
                num_layers=num_layers,
                dropout=dropout,
                norm="layer_norm",
            )

    def forward(self, x_dict):
        out = dict()
        for key, x in x_dict.items():
            if key not in self.mlps:
                continue
            out[key] = self.mlps[key](x)
        return out


def build_hetero_gnn(
    model_type,
    data: HeteroData,
    hidden_channels: int = 64,
    num_layers: int = 2,
    aggr: str = "sum",
    model_kwargs: dict = {},
    out_channels: int = 64,
) -> nn.Module:
    """
    model_types: GAT, GATv2, EdgeCNN, GCN, GraphSAGE, GIN
    """

    if model_type == "GAT":
        model = GAT(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            add_self_loops=False,
            heads=hidden_channels // 16,
            **model_kwargs,
        )
    elif model_type == "GATv2":
        model = GAT(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            add_self_loops=False,
            heads=hidden_channels // 16,
            v2=True,
            **model_kwargs,
        )
    elif model_type == "EdgeCNN":
        model = EdgeCNN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            **model_kwargs,
        )
    elif model_type == "GraphSAGE":
        model = GraphSAGE(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            **model_kwargs,
        )
    elif model_type == "GIN":
        model = GIN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            **model_kwargs,
        )
    else:
        raise ValueError(f"Model type {model_type} not supported.")

    return to_hetero(model, data.metadata(), aggr=aggr)
