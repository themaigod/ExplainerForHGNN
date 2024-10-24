import json
import pickle
import os

__all__ = [
    "load_dataset",
    "ACM",
    "DBLP",
    "IMDB",
    "NodeClassificationDataset",
]

from .acm import ACM
from .dblp import DBLP
from .imdb import IMDB
from .dataset import NodeClassificationDataset


def load_dataset(dataset_path, dataset_config=None, is_load_dataset=False,
                           only_load_test_label_shuffle=False):
    dataset_name = dataset_path.split("/")[-1]

    if dataset_config is not None:
        with open(dataset_config, "r") as f:
            config = json.load(f)
    else:
        with open(f"./dataset_configs/{dataset_name}.json", "r") as f:
            config = json.load(f)

    # ensure the shuffle of test labels is consistent
    if is_load_dataset:
        if only_load_test_label_shuffle:
            if config.get("save_path", None):
                dataset_path_processed = config["save_path"]
            else:
                dataset_path_processed = os.path.join(dataset_path, "processed.pkl")
            dataset = globals()[dataset_name](dataset_path, config)
            if not os.path.exists(dataset_path_processed):
                with open(dataset_path_processed, "wb") as f:
                    pickle.dump(dataset.labels[2], f)
                return dataset
            with open(dataset_path_processed, "rb") as f:
                test_labels = pickle.load(f)
            dataset.labels[2] = test_labels
            return dataset
        else:
            if config.get("save_path", None):
                dataset_path_processed = config["save_path"]
            else:
                dataset_path_processed = os.path.join(dataset_path, "processed.pkl")
            if not os.path.exists(dataset_path_processed):
                dataset = globals()[dataset_name](dataset_path, config)
                with open(dataset_path_processed, "wb") as f:
                    pickle.dump(dataset.to_dict(), f)
                return dataset
            with open(dataset_path_processed, "rb") as f:
                dataset_dict = pickle.load(f)
            return globals()[dataset_name].from_dict(dataset_dict)

    try:
        globals()[dataset_name]
    except KeyError:
        print(
            f"Warning: Dataset {dataset_name} not found specific loader." + \
            "This may cause errors. \n Using the default (NodeClassificationDataset).")
        return NodeClassificationDataset(dataset_path, config)
    return globals()[dataset_name](dataset_path, config)
