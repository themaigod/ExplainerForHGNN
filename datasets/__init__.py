import json

__all__ = [
    "load_dataset",
    "ACM",
    "DBLP",
    "IMDB",
    "BaseDataset",
]

from .acm import ACM


def load_dataset(dataset_path):
    dataset_name = dataset_path.split("/")[-1]

    with open(f"./dataset_configs/{dataset_name}.json", "r") as f:
        config = json.load(f)

    try:
        return globals()[dataset_name](dataset_path, config)
    except:
        print(
            f"Warning: Dataset {dataset_name} not found specific loader. This may cause errors.")
        return BaseDataset(dataset_path, config)
