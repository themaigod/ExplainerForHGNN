import json

all = [
    "load_model",
    "HAN",
    "MAGNN",
    "HGT",
    "RGCN",
]

from .han import HAN
# from .gcn import GCN
# from .gat import GAT
# from .magnn import MAGNN
# from .hgt import HGT
# from .rgcn import RGCN
from .model import BaseModel


def load_model(model_name, dataset, model_config=None):
    # load config first
    if model_config is not None:
        with open(model_config, "r") as f:
            config = json.load(f)
    else:
        with open(f"./model_configs/{model_name}_{dataset.dataset_name}.json",
                  "r") as f:
            config = json.load(f)

    if model_name == "HAN":
        return HAN(config, dataset)
    # elif model_name == "MAGNN":
    #     return MAGNN(config, dataset)
    # elif model_name == "HGT":
    #     return HGT(config, dataset)
    # elif model_name == "RGCN":
    #     return RGCN(config, dataset)
    else:
        return BaseModel(config, dataset)
