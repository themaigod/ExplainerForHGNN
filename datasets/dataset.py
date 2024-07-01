import torch.utils.data as data
import os


class Dataset(data.Dataset):
    def __init__(self):
        pass


class NodeClassificationDataset(Dataset):
    def __init__(self, dataset_path, config):
        """

        :param dataset_path: where the dataset is stored
            Note: the structure of the dataset should be like:
            node_features.pkl
            node_types.npy
            edges.pkl
            labels.pkl
            edge_directions.json

            labels.pkl will be covered if config['labels'] is not None
            labels.pkl:
            [
                [
                [node_id, label], # train set
                    ...
                ],
                [
                    [node_id, label], # valid set
                    ...
                ],
                [
                    [node_id, label], # test set
                    ...
                ]
            ]

            edge_directions.json:
            {
                target_node_type: num
                edge_types: [
                    [source_node_type, target_node_type],
                    ...
                ]
            }
        :param config:
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.config = config
        self.dataset_name = dataset_path.split('/')[-1]

        # for model to recognize if the dataset is for single graph or multiple graphs
        self.single_graph = True
        self.hetero = False

        self.get_data_files_path()
        if self.config.get('labels', None) is not None:
            self.labels = self.config['labels']
            assert os.path.exists(self.labels), f"{self.labels} not found"

    def get_data_files_path(self):
        node_features_path = os.path.join(self.dataset_path, 'node_features.pkl')
        node_types_path = os.path.join(self.dataset_path, 'node_types.npy')
        edges_path = os.path.join(self.dataset_path, 'edges.pkl')
        labels_path = os.path.join(self.dataset_path, 'labels.pkl')
        edge_directions_path = os.path.join(self.dataset_path, 'edge_directions.json')

        assert os.path.exists(
            node_features_path), f"node_features.pkl not found in {self.dataset_path}"
        assert os.path.exists(
            node_types_path), f"node_types.npy not found in {self.dataset_path}"
        assert os.path.exists(edges_path), f"edges.pkl not found in {self.dataset_path}"
        assert os.path.exists(
            edge_directions_path), f"edge_directions.json not found in {self.dataset_path}"

        if self.config.get('labels', None) is None:
            assert os.path.exists(
                labels_path), f"labels.pkl not found in {self.dataset_path}"

        self.node_features_path = node_features_path
        self.node_types_path = node_types_path
        self.edges_path = edges_path
        self.labels_path = labels_path
        self.edge_directions_path = edge_directions_path

    def check_data_size(self):
        raise NotImplementedError
