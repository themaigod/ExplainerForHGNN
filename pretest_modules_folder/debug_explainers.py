from explainers import GNNExplainerMeta
from datasets import ACM
from models import HAN


def debug_GNNExplainerMeta():
    dataset = ACM("./data/ACM",
                  {"labels": "./data/ACM/labels.pkl", "check_data_size": True})

    import json
    with open('model_configs/HAN_ACM.json') as f:
        model_config = json.load(f)

    model = HAN(model_config, dataset)

    model.fit()

    summary = model.get_summary()

    with open("./explainer_configs/GNNExplainerMeta_HAN_ACM.json", "r") as f:
        config = json.load(f)

    explainer = GNNExplainerMeta(config)

    result = explainer.explain(model)

    return result
