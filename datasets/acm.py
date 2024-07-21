from .dataset import NodeClassificationDataset
import pickle
import numpy as np
import json


class ACM(NodeClassificationDataset):
    def __init__(self, dataset_path, config):
        super(ACM, self).__init__(dataset_path, config)
        self.load_data()
        self.hetero = True
        if config.get('check_data_size', False):
            self.check_data_size()

        self.num_classes = len(set(self.labels[0][:, 1]))
        self.num_features = self.node_features.shape[1]

    def load_data(self):
        with open(self.node_features_path, 'rb') as f:
            self.node_features = pickle.load(f)

        self.node_types = np.load(self.node_types_path)

        with open(self.edges_path, 'rb') as f:
            self.edges = pickle.load(f)

        if self.config.get('labels', None) is None:
            with open(self.labels_path, 'rb') as f:
                self.labels = pickle.load(f)
        else:
            self.labels = self.config['labels']
            with open(self.labels, 'rb') as f:
                self.labels = pickle.load(f)

        with open(self.edge_directions_path, 'rb') as f:
            self.edge_directions = json.load(f)

    def check_data_size(self):
        """
        Check the size of each data file
        :return:
        """
        num_nodes = self.node_features.shape[0]
        assert num_nodes == len(
            self.node_types), f"Node features and node types have different sizes"
        edge_shapes = [subgraph.shape for subgraph in self.edges]
        for i, shape in enumerate(edge_shapes):
            assert shape[0] == shape[1], f"Subgraph {i} is not square"

        for i, shape in enumerate(edge_shapes):
            assert num_nodes == shape[
                0], f"Node features and subgraph {i} have different sizes"


# if __name__ == "__main__":
#     dataset = ACM("../data/acm", {"labels": "../data/acm/labels.pkl", "check_data_size": True})
#     node_features_shape = dataset.node_features.shape
#     print("Num of nodes:", node_features_shape[0], "Num of features:",
#           node_features_shape[1])
#     node_types = dataset.node_types
#     print("Node type count:", len(set(node_types)), "Num of nodes:", len(node_types))
#     for i in range(len(set(node_types))):
#         print(f"Node type {i} count:", len(np.argwhere(node_types == i)))
#     print("Subgraph sizes:", [subgraph.nnz for subgraph in dataset.edges])
#     print("Train size:", len(dataset.labels[0]), "Valid size:", len(dataset.labels[1]),
#           "Test size:", len(dataset.labels[2]))
#     print("Target node type:", dataset.edge_directions['target_node_type'],
#           "Edge types:", dataset.edge_directions['edge_types'])
#     print("Dataset is hetero:", dataset.hetero, "Dataset is single graph:",
#           dataset.single_graph)
