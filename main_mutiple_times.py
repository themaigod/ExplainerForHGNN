import argparse
import torch
import numpy as np
import random
import os

from models import load_model
from explainers import load_explainer
from datasets import load_dataset


def getargs():
    parser = argparse.ArgumentParser(description='Explainable AI')
    parser.add_argument('--dataset', type=str, default='./data/ACM',
                        help='Path to dataset, the folder name should be the same as the dataset name')
    parser.add_argument('--model', type=str, default='HAN',
                        help='Model to use.')
    parser.add_argument('--explainer', type=str, default='GNNExplainerMeta',
                        help='Explainer to use')
    parser.add_argument('--random_seed', type=int, nargs='+', default=[0],
                        help='Random seed. You can try multiple seeds to get overall performance')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device')
    parser = getargs_optional(parser)
    return parser.parse_args()


def getargs_optional(parser):
    parser.add_argument('--dataset_config', type=str, default=None,
                        help='Dir to dataset config file')
    parser.add_argument('--model_config', type=str, default=None,
                        help='Dir to model config file')
    parser.add_argument('--explainer_config', type=str, default=None,
                        help='Dir to explainer config file')
    parser.add_argument('--ensure_reproducibility', action='store_true',
                        default=False,
                        help='Ensure reproducibility')
    parser.add_argument('--save_attention', action='store_true',
                        default=False,
                        help='Save attention weights')
    parser.add_argument('--save_model', action='store_true',
                        default=False,
                        help='Save model')
    parser.add_argument('--minimize_explanation', action='store_true',
                        default=False,
                        help='Only save edge mask')
    parser.add_argument('--save_gat_attention', action='store_true',
                        default=False,
                        help='Save GAT attention weights')
    parser.add_argument('--explanation_keep_keys', type=str, nargs='+', default=[],
                        help='Keys to keep in explanation')
    return parser


def set_seed(seed, ensure_reproducibility=False, times=0):
    if len(seed) > 1:
        seed = seed[times]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    import os
    # because of the suggestion from
    # https://pytorch.org/docs/stable/notes/randomness.html
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if ensure_reproducibility:
        torch.backends.cudnn.benchmark = False


def train_model(model_name, dataset_path, device, dataset_configs=None,
                model_configs=None
                ):
    dataset_config = dataset_configs.__next__()
    model_config = model_configs.__next__()
    dataset = load_dataset(dataset_path, dataset_config)
    model = load_model(model_name, dataset, model_config)
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


def explain(model, explainer_name, device, explainer_configs=None, minimize=False,
            filter_keys=None
            ):
    explainer_config = explainer_configs.__next__()
    explainer = load_explainer(explainer_name, model.__class__.__name__,
                               model.dataset.__class__.__name__,
                               explainer_config)
    explainer.to(device)
    result = explainer.explain(model)
    print("Explanation Summary:")
    print("----------------")
    for key, value in result.items():
        print(f"{key}: {value}")
    if minimize:
        explainer.save_explanation(filter_keys=['edge_mask'])
    elif filter_keys is not None and len(filter_keys) > 0:
        explainer.save_explanation(filter_keys=filter_keys)
    else:
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


def get_times(random_seed, dataset_config, model_config, explainer_config):
    seed_times = 0
    if len(random_seed) > 1:
        seed_times = len(random_seed)

    dataset_times = 0
    if dataset_config is not None:
        if len(os.listdir(dataset_config)) > 1:
            dataset_times = len(os.listdir(dataset_config))

    model_times = 0
    if model_config is not None:
        if len(os.listdir(model_config)) > 1:
            model_times = len(os.listdir(model_config))

    explainer_times = 0
    if explainer_config is not None:
        if len(os.listdir(explainer_config)) > 1:
            explainer_times = len(os.listdir(explainer_config))

    times = 1
    if seed_times != 0:
        times = seed_times

    if dataset_times != 0:
        if dataset_times != times and times != 1:
            raise ValueError("Number of dataset times does not match")
        times = dataset_times

    if model_times != 0:
        if model_times != times and times != 1:
            raise ValueError("Number of model times does not match")
        times = model_times

    if explainer_times != 0:
        if explainer_times != times and times != 1:
            raise ValueError("Number of explainer times does not match")
        times = explainer_times
    return times


def multiple_times(name):
    if name is None:
        return [None], False
    if os.listdir(name) == 1:
        return [name], False
    sorted_list = sorted(os.listdir(name))
    return [os.path.join(name, i) for i in sorted_list], True


def main():
    args = getargs()
    times = get_times(args.random_seed, args.dataset_config, args.model_config,
                      args.explainer_config)
    dataset_configs, explainer_configs, model_configs = prepare_configs(
        args.dataset_config,
        args.explainer_config,
        args.model_config,
        times)

    if times == 1:
        raise ValueError(
            "This script is for multiple times running, please use main.py for single time running")
    for i in range(times):
        set_seed(args.random_seed, args.ensure_reproducibility, i)
        print("Running {}-th time / {} times".format(i + 1, times))
        model = train_model(args.model, args.dataset, args.device, dataset_configs,
                            model_configs)
        if args.save_model:
            model.save()
        if args.save_attention:
            if args.save_gat_attention and model.model_name in ['HAN']:
                model.save_attention(gat_attention=args.save_gat_attention)
            else:
                model.save_attention()
        explainer = explain(model, args.explainer, args.device, explainer_configs,
                            minimize=args.minimize_explanation,
                            filter_keys=args.explanation_keep_keys)
    # visualize(explainer)


def prepare_configs(dataset_config, explainer_config, model_config, times):
    dataset_configs, dataset_is_multiple = multiple_times(dataset_config)
    model_configs, model_is_multiple = multiple_times(model_config)
    explainer_configs, explainer_is_multiple = multiple_times(explainer_config)
    if not dataset_is_multiple:
        dataset_configs = prepare_dataset_configs(dataset_configs, times)
    else:
        dataset_configs = iter(dataset_configs)
    if not model_is_multiple:
        model_configs = prepare_model_configs(model_configs, times)
    else:
        model_configs = iter(model_configs)
    if not explainer_is_multiple:
        explainer_configs = prepare_explainer_configs(explainer_configs, times)
    else:
        explainer_configs = iter(explainer_configs)
    return dataset_configs, explainer_configs, model_configs


def prepare_dataset_configs(dataset_configs, times=None):
    yield dataset_configs[0]


def prepare_model_configs(model_configs, times):
    import json
    path = model_configs[0]
    with open(path, "r") as f:
        config = json.load(f)

    for t in range(1, times + 1):
        new_config_path = "tmp/model_configs/model_config_{}.json".format(t)
        os.makedirs(os.path.dirname(new_config_path), exist_ok=True)
        new_config = {}
        for key, value in config.items():
            if key.endswith("_path"):
                new_config[key] = os.path.join(os.path.dirname(value),
                                               "{}_{}".format(t,
                                                              os.path.basename(value)))
            else:
                new_config[key] = value
        with open(new_config_path, "w") as f:
            json.dump(new_config, f)
        yield new_config_path


def prepare_explainer_configs(explainer_configs, times):
    import json
    path = explainer_configs[0]
    with open(path, "r") as f:
        config = json.load(f)

    for t in range(1, times + 1):
        new_config_path = "tmp/explainer_configs/explainer_config_{}.json".format(t)
        os.makedirs(os.path.dirname(new_config_path), exist_ok=True)
        new_config = {}
        for key, value in config.items():
            if key.endswith("_path"):
                new_config[key] = os.path.join(os.path.dirname(value),
                                               "{}_{}".format(t,
                                                              os.path.basename(value)))
            else:
                new_config[key] = value
        with open(new_config_path, "w") as f:
            json.dump(new_config, f)
        yield new_config_path


if __name__ == '__main__':
    main()