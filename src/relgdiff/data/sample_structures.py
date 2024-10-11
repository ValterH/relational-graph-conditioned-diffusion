import random

import numpy as np
import scipy.sparse as sp
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix
from syntherela.data import load_tables, remove_sdv_columns

from relgdiff.data.tables_to_heterodata import tables_to_heterodata


def get_connected_components(data):
    homo = data.to_homogeneous()
    adj = to_scipy_sparse_matrix(homo.edge_index.cpu())

    num_components, component = sp.csgraph.connected_components(adj, connection="weak")
    components = dict()
    for i, key in enumerate(data.x_dict.keys()):
        components[key] = component[homo.node_type.cpu() == i]

    connected_components = []

    for component in np.arange(num_components):
        nodes = dict()
        for key, ccs in components.items():
            nodes[key] = np.argwhere(ccs == component).flatten()
        connected_components.append(data.subgraph(nodes))

    return connected_components


def sample_structures(data_path, metadata, num_structures=None, pos_enc={}):
    tables = load_tables(data_path, metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)
    data = tables_to_heterodata(
        tables, metadata, masked_tables=metadata.get_tables(), pos_enc=pos_enc
    )
    subgraphs = get_connected_components(data)
    if num_structures is None:
        num_structures = len(subgraphs)

    if num_structures > len(subgraphs):
        samples = random.choices(subgraphs, k=num_structures)
    else:
        samples = random.sample(subgraphs, k=num_structures)
    # use the dataloader to stitch the samples to a single HeteroData object
    dataloader = DataLoader(samples, batch_size=num_structures)
    hetero_data = next(iter(dataloader))
    for key in hetero_data.x_dict.keys():
        hetero_data[key]["mean"] = hetero_data[key]["mean"][0]
        hetero_data[key]["std"] = hetero_data[key]["std"][0]
        for column in hetero_data[key]["categories"]:
            hetero_data[key]["categories"][column] = hetero_data[key]["categories"][
                column
            ][0]
    return hetero_data
