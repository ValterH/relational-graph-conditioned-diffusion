import random
import warnings

import numpy as np
import scipy.sparse as sp
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix
from syntherela.data import load_tables, remove_sdv_columns

from relgdiff.data.tables_to_heterodata import (
    tables_to_heterodata,
    subgraph_has_all_tables,
)
from relgdiff.data.utils import sort_rows_by_child_count


def get_connected_components(data):
    homo = data.to_homogeneous()
    adj = to_scipy_sparse_matrix(homo.edge_index.cpu())

    num_components, component = sp.csgraph.connected_components(adj, connection="weak")
    components = dict()
    for i, key in enumerate(data.x_dict.keys()):
        components[key] = component[homo.node_type.cpu() == i]

    connected_components = []
    largest_cc_size = 0
    for component in np.arange(num_components):
        nodes = dict()
        for key, ccs in components.items():
            nodes[key] = np.argwhere(ccs == component).flatten()
        subgraph = data.subgraph(nodes)
        connected_components.append(subgraph)
        if subgraph.num_nodes > largest_cc_size:
            largest_cc_size = subgraph.num_nodes

    return connected_components, largest_cc_size


def sample_structures(
    data_path, metadata, num_structures=None, pos_enc={}, fix_structure=False
):
    tables = load_tables(data_path, metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)
    if not fix_structure:
        for table_name in metadata.get_tables():
            tables = sort_rows_by_child_count(tables, table_name, metadata)
    data = tables_to_heterodata(
        tables, metadata, masked_tables=metadata.get_tables(), pos_enc=pos_enc
    )
    if fix_structure:
        return data
    subgraphs, num_nodes_largest = get_connected_components(data)

    pct_largest = num_nodes_largest / data.num_nodes
    if pct_largest > 0.5:
        warnings.warn(
            f""""The largest connected component is larger than 50% of the dataset ({pct_largest * 100: .2f}%). 
            Keeping the original structure as the foreign key graph forms a network. This will be addressed in future work."""
        )
        return data

    if num_structures is None:
        num_structures = len(subgraphs)

    if num_structures >= len(subgraphs):
        samples = random.choices(subgraphs, k=num_structures)
    else:
        samples = random.sample(subgraphs, k=num_structures)

    # reporder the samples, s.t. the last sample has all tables (this will create a valid edge_index)
    for i in range(len(samples)):
        if subgraph_has_all_tables(samples[i], metadata):
            # move the current subgraph to the end of the list
            samples.append(samples.pop(i))
            break
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
