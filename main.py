import argparse
import torch
import numpy as np
import random


def getargs():
    parser = argparse.ArgumentParser(description='Explainable AI')
    parser.add_argument('--dataset', type=str, default='./data/ACM',
                        help='Path to dataset, the folder name should be the same as the dataset name')
    parser.add_argument('--model', type=str, default='HAN',
                        help='Model to use. \n./model_configs/{Model}.json will be used for configuration')
    parser.add_argument('--explainer', type=str, default='GNNExplainer',
                        help='Explainer to use\n./explainer_configs/{Explainer}.json will be used for configuration')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed. You can try multiple seeds to get overall performance')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device')
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def train_model(model_name, dataset_path, device):
    dataset = load_dataset(dataset_path)
    model = load_model(model_name, dataset)
    model.to(device)

    # Train model
    model.train()
    model.fit()
    return model


def explain(model, explainer_name, device):
    explainer = load_explainer(explainer_name)
    explainer.to(device)
    explainer.explain(model)
    explainer.eval()
    return explainer


def visualize(explainer):
    if explainer.metrics is not None:
        explainer.get_metrics()
        explainer.save_metrics()

    if explainer.model.summary is not None:
        explainer.model.get_summary()
        explainer.model.save_summary()

    explainer.visualize()


def main():
    args = getargs()
    set_seed(args.random_seed)
    model = train_model(args.model, args.dataset, args.device)
    explainer = explain(model, args.explainer, args.device)
    visualize(explainer)
