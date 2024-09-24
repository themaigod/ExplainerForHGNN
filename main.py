import argparse
import torch
import numpy as np
import random

from models import load_model
from explainers import load_explainer
from datasets import load_dataset


def getargs():
    parser = argparse.ArgumentParser(description='Explainable AI')
    parser.add_argument('--dataset', type=str, default='./data/ACM',
                        help='Path to dataset, the folder name should be the same as the dataset name')
    parser.add_argument('--model', type=str, default='HAN',
                        help='Model to use. \n./model_configs/{Model}_{Dataset}.json will be used for configuration')
    parser.add_argument('--explainer', type=str, default='GNNExplainerMeta',
                        help='Explainer to use\n./explainer_configs/{Explainer}_{Model}_{Dataset}.json will be used for configuration')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed. You can try multiple seeds to get overall performance')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device')
    return parser.parse_args()


def set_seed(seed, ensure_reproducibility=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    if ensure_reproducibility:
        torch.backends.cudnn.benchmark = False


def train_model(model_name, dataset_path, device):
    dataset = load_dataset(dataset_path)
    model = load_model(model_name, dataset)
    model.to(device)

    # Train model
    print("Training model...")
    model.train()
    model.fit()
    summary = model.get_summary()

    # print summary
    print("Model Summary:")
    print("----------------")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return model


def explain(model, explainer_name, device):
    explainer = load_explainer(explainer_name)
    explainer.to(device)
    result = explainer.explain(model)
    print("Explanation Summary:")
    print("----------------")
    for key, value in result.items():
        print(f"{key}: {value}")
    explainer.save_explanation()
    return explainer


# Do not implement now
# def visualize(explainer):
#     if explainer.metrics is not None:
#         explainer.get_metrics()
#         explainer.save_metrics()
#
#     if explainer.model.summary is not None:
#         explainer.model.get_summary()
#         explainer.model.save_summary()
#
#     explainer.visualize()


def main():
    args = getargs()
    set_seed(args.random_seed)
    model = train_model(args.model, args.dataset, args.device)
    explainer = explain(model, args.explainer, args.device)
    # visualize(explainer)
