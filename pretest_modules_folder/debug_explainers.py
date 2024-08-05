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

    selected_nodes = dataset.labels[2][: 10]
    selected_nodes = [i[0] for i in selected_nodes]

    result = explainer.explain_selected_nodes(model, selected_nodes)

    return result
