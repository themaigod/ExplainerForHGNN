import json

__all__ = [
    "load_explainer",
    "GradExplainer",
    "SAExplainer",
    "GOAtExplainer",
    "SubgraphXExplainer",
    "GNNExplainer",
    "PGExplainer",
    "CGEExplainer",
    "Explainer",
]

from .grad import GradExplainer
from .saliency import SAExplainer
from .goat import GOAtExplainer
from .subgraphx import SubgraphXExplainer
from .gnnexplainer import GNNExplainer
from .pgexplainer import PGExplainer
from .cge import CGEExplainer
from .explainer import Explainer


def load_explainer(explainer_name):
    # load config first
    with open(f"./explainer_configs/{explainer_name}.json", "r") as f:
        config = json.load(f)

    if explainer_name == "GradExplainer":
        return GradExplainer(config)
    elif explainer_name == "SAExplainer":
        return SAExplainer(config)
    elif explainer_name == "GOAtExplainer":
        return GOAtExplainer(config)
    elif explainer_name == "SubgraphXExplainer":
        return SubgraphXExplainer(config)
    elif explainer_name == "GNNExplainer":
        return GNNExplainer(config)
    elif explainer_name == "PGExplainer":
        return PGExplainer(config)
    elif explainer_name == "CGEExplainer":
        return CGEExplainer(config)
    else:
        return Explainer(config)
