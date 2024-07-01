from datasets import ACM
import numpy as np


def pretest_acm():
    dataset = ACM("./data/ACM",
                  {"labels": "./data/ACM/labels.pkl", "check_data_size": True})
    node_features_shape = dataset.node_features.shape
    print("Num of nodes:", node_features_shape[0], "Num of features:",
          node_features_shape[1])
    node_types = dataset.node_types
    print("Node type count:", len(set(node_types)), "Num of nodes:", len(node_types))
    for i in range(len(set(node_types))):
        print(f"Node type {i} count:", len(np.argwhere(node_types == i)))
    print("Subgraph sizes:", [subgraph.nnz for subgraph in dataset.edges])
    print("Train size:", len(dataset.labels[0]), "Valid size:", len(dataset.labels[1]),
          "Test size:", len(dataset.labels[2]))
    print("Target node type:", dataset.edge_directions['target_node_type'],
          "Edge types:", dataset.edge_directions['edge_types'])
    print("Dataset is hetero:", dataset.hetero, "Dataset is single graph:",
          dataset.single_graph)

    assert dataset.node_features.shape[0] == 8994
    assert dataset.node_features.shape[1] == 1902
    assert len(set(dataset.node_types)) == 3
    assert len(np.argwhere(dataset.node_types == 0)) == 3025
    assert len(np.argwhere(dataset.node_types == 1)) == 5912
    assert len(np.argwhere(dataset.node_types == 2)) == 57
    assert [subgraph.nnz for subgraph in dataset.edges] == [
        9936, 9936, 3025, 3025]
    assert len(dataset.labels[0]) == 600
    assert len(dataset.labels[1]) == 300
    assert len(dataset.labels[2]) == 2125
    assert dataset.edge_directions['target_node_type'] == 0
    assert dataset.edge_directions['edge_types'] == [[0, 1], [1, 0], [0, 2], [2, 0]]
    assert dataset.hetero is True
    assert dataset.single_graph is True
