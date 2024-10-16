import json

__all__ = [
    "load_explainer",
    "GradExplainerMeta",
    "GradExplainer",
    "GradExplainerOriginal",
    # "SAExplainer",
    # "GOAtExplainer",
    # "SubgraphXExplainer",
    "GNNExplainerMeta",
    "GNNExplainerOriginal",
    # "PGExplainer",
    # "CGEExplainer",
    "Explainer",
]

from .grad import GradExplainerMeta, GradExplainer, GradExplainerOriginal
# from .saliency import SAExplainer
# from .goat import GOAtExplainer
# from .subgraphx import SubgraphXExplainer
from .gnnexplainer import GNNExplainerMeta, GNNExplainerOriginal
# from .pgexplainer import PGExplainer
# from .cge import CGEExplainer
from .explainer import Explainer
import os


def load_default_config(explainer_name, model_name, dataset_name):
    if os.path.exists(
        f"./explainer_configs/{explainer_name}_{model_name}_{dataset_name}.json"):
        with open(
            f"./explainer_configs/{explainer_name}_{model_name}_{dataset_name}.json",
            "r") as f:
            config = json.load(f)
    else:
        with open(f"./explainer_configs/{explainer_name}.json", "r") as f:
            config = json.load(f)
        for key in config.keys():
            if key.endswith("_path"):
                config[key] = config[key].format(model_name=model_name,
                                                 dataset_name=dataset_name,
                                                 explainer_name=explainer_name)
    return config


def load_explainer(explainer_name, model_name, dataset_name, explainer_config=None):
    # load config first
    if explainer_config is not None:
        with open(explainer_config, "r") as f:
            config = json.load(f)
    else:
        config = load_default_config(explainer_name, model_name, dataset_name)

    if explainer_name == "GNNExplainerMeta":
        return GNNExplainerMeta(config)
    elif explainer_name == "GNNExplainerOriginal":
        return GNNExplainerOriginal(config)
    elif explainer_name == "GradExplainerMeta":
        return GradExplainerMeta(config)
    elif explainer_name == "GradExplainer":
        return GradExplainer(config)
    elif explainer_name == "GradExplainerOriginal":
        return GradExplainerOriginal(config)
    # elif explainer_name == "SAExplainer":
    #     return SAExplainer(config)
    # elif explainer_name == "GOAtExplainer":
    #     return GOAtExplainer(config)
    # elif explainer_name == "SubgraphXExplainer":
    #     return SubgraphXExplainer(config)
    # elif explainer_name == "GNNExplainer":
    #     return GNNExplainer(config)
    # elif explainer_name == "PGExplainer":
    #     return PGExplainer(config)
    # elif explainer_name == "CGEExplainer":
    #     return CGEExplainer(config)
    else:
        return Explainer(config)
