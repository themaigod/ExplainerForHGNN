import json

__all__ = [
    "load_dataset",
    "ACM",
    "NodeClassificationDataset",
]

from .acm import ACM
from .dataset import NodeClassificationDataset


def load_dataset(dataset_path):
    dataset_name = dataset_path.split("/")[-1]

    with open(f"./dataset_configs/{dataset_name}.json", "r") as f:
        config = json.load(f)

    try:
        return globals()[dataset_name](dataset_path, config)
    except:
        print(
            f"Warning: Dataset {dataset_name} not found specific loader." + \
            "This may cause errors. \n Using the default (NodeClassificationDataset).")
        return NodeClassificationDataset(dataset_path, config)
