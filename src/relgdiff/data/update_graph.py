import torch
from relgdiff.data.tables_to_heterodata import preprocess_data


def update_graph_features(heterodata, df, table_name, metadata, dim_empty=16):
    # set the node features for the selected table
    # tranform the data to all numerical values
    temp_table, _, _, _ = preprocess_data(
        df,
        table_name=table_name,
        metadata=metadata,
        categories=heterodata[table_name]["categories"],
    )
    # standardize the data
    means = heterodata[table_name]["mean"]
    stds = heterodata[table_name]["std"]
    temp_table[means.index] = temp_table[means.index] - means
    temp_table[stds.index] = temp_table[stds.index] / stds

    table_values = temp_table.values.astype("float32")
    if table_values.size == 0:
        heterodata[table_name].x = torch.ones(
            (temp_table.shape[0], dim_empty), dtype=torch.float32
        )
    else:
        heterodata[table_name].x = torch.tensor(table_values, dtype=torch.float32)

    return heterodata
