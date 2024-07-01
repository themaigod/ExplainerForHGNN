import json
import os
import scipy.sparse
import pickle
import numpy as np


def process_node_classification_dataset_to_edge_directions(dataset_path,
                                                           target_node_type=None,
                                                           label_path=None,
                                                           consider_self_loop=False,
                                                           ):
    assert target_node_type is not None or label_path is not None, "Either target_node_type or label_path should be provided."

    with open(os.path.join(dataset_path, "edges.pkl"), "rb") as f:
        edges = pickle.load(f)

    node_types = np.load(os.path.join(dataset_path, "node_types.npy"))

    if not target_node_type:
        with open(label_path, "rb") as f:
            labels = pickle.load(f)
        # assuming that the all nodes of the labels have the same type
        target_node_type = node_types[labels[0][0][0]]
        target_node_type = int(target_node_type)  # convert to int for json serialization

    edge_directions = {
        "target_node_type": target_node_type,
        "edge_types": []
    }

    for edge in edges:
        edge: scipy.sparse.csr_matrix
        edge = edge.tocoo()
        if consider_self_loop:
            edge.set_diag(0)
            edge.eliminate_zeros()

        # use the first edge to determine the source node type
        source = node_types[edge.row[0]]
        target = node_types[edge.col[0]]

        edge_directions["edge_types"].append([int(source), int(target)])

    with open(os.path.join(dataset_path, "edge_directions.json"), "w") as f:
        json.dump(edge_directions, f)


if __name__ == "__main__":
    process_node_classification_dataset_to_edge_directions("../data/acm", label_path="../data/acm/labels.pkl")