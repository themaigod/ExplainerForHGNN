# Do We Really Need to Access the meta-path in Heterogeneous Information Networks: An Explainable Graph Neural Network Approach

## Introduction

This repository implements several explainable graph neural network models, and applies them to heterogeneous GNNs.
We do fair comparison between different models and evaluate their performance on several benchmark datasets.

## Requirements

- Python 3.10
- Pytorch (developed with 2.3.0)
- Cuda 11 or higher (if you want to use GPU. Not sure about the lower version of Cuda)
- Numpy

Note: Pytorch seems to do a lot of changes in the sparse tensor (Compared to the previous version), and give
a better support for the sparse tensor, which we rely on. So the elder version of Pytorch may not work with this code.

Note: Currently, we not rely on other common libraries like `torch_geometric`, `dgl`, etc. We hope to make the code more
compatible with different environments. But if it is tough to implement the model without these libraries in the future,
we will use them.

## Run the code

To run the code, you can use the following command:

```bash
python main.py --dataset {dataset_name} --model {model_name} --explainer {explainer_name} --random_seed {seed}
```

You can do some customization by changing `./dataset_configs/{dataset_name}.json`, `./model_configs/{model_name}_{dataset_name}.json`
and `./explainer_configs/{explainer_name}_{model_name}_{dataset_name}.json`.

Note: Although we recommend using `./explainer_configs/{explainer_name}_{model_name}_{dataset_name}.json` to set
the details for different models and datasets, you can just use `./explainer_configs/{explainer_name}.json` to set the
new model and dataset. You can just provide the `./explainer_configs/{explainer_name}.json` if you design a new explainer.

More details can be found in the `python main.py -h`.

### Multi-run

You can run multiple experiments by using the following command:

```bash
python main_multiple_times.py --dataset {dataset_name} --model {model_name} --explainer {explainer_name} --random_seed {seeds}
```

Seeds should be a list of integers, e.g., `--random_seed 1 2 3 4 5`.

We also offer some customization settings in `python main_multiple_times.py -h`. For example, you can find setting that
allows you to run the experiments in different model settings, and then you do not need to give the random seeds.

## Explainer

Now Dataset available: ACM, DBLP, IMDB

Now model available: HAN (with GAT), HAN (with GCN)

Now explainer available: GNNExplainer, Grad

We provide two versions of explainer: one is the original version (directly for heterogeneous), and the other is the
version that further modify for the heterogeneous.

Officially, you can use `GNNExplainerMeta`, `GNNExplainerOriginal`, `GradExplainerMeta`, `GradExplainerOriginal` to
run the experiments.

### Plan:

PGExplainer

```bibtex
@article{luo2020parameterized,
  title={Parameterized Explainer for Graph Neural Network},
  author={Luo, Dongsheng and Cheng, Wei and Xu, Dongkuan and Yu, Wenchao and Zong, Bo and Chen, Haifeng and Zhang, Xiang},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2024}
}
```

GNNExplainer

```bibtex
@article{Ying2019GNNExplainerGE,
  title={GNNExplainer: Generating Explanations for Graph Neural Networks},
  author={Rex Ying and Dylan Bourgeois and Jiaxuan You and Marinka Zitnik and Jure Leskovec},
  journal={Advances in neural information processing systems},
  year={2019},
  volume={32},
  pages={
          9240-9251
        },
  url={https://api.semanticscholar.org/CorpusID:202572927}
}
```

SubgraphX

```bibtex
@inproceedings{Yuan2021OnEO,
  title={On Explainability of Graph Neural Networks via Subgraph Explorations},
  author={Hao Yuan and Haiyang Yu and Jie Wang and Kang Li and Shuiwang Ji},
  booktitle={International Conference on Machine Learning},
  year={2021},
  url={https://api.semanticscholar.org/CorpusID:231861768}
}
```

SA

```bibtex
@inproceedings{baldassarre2019explainability,
  title={Explainability techniques for graph convolutional networks},
  author={Baldassarre, Federico and Azizpour, Hossein},
  booktitle={Proceedings of the ICML 2019 Workshop on Learning and Reasoning with Graph-Structured Representations},
  year={2019},
  note={arXiv preprint arXiv:1905.13686}
}
```

GOAt

```bibtex
@inproceedings{Lu2024GOAtEG,
  title={GOAt: Explaining Graph Neural Networks via Graph Output Attribution},
  author={Shengyao Lu and Keith G. Mills and Jiao He and Bang Liu and Di Niu},
  booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:267301280},
  note={arXiv preprint arXiv:2401.14578}
}

```

Grad

```bibtex
@article{Selvaraju2016GradCAMVE,
  title={Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization},
  author={Ramprasaath R. Selvaraju and Abhishek Das and Ramakrishna Vedantam and Michael Cogswell and Devi Parikh and Dhruv Batra},
  journal={International Journal of Computer Vision},
  year={2016},
  volume={128},
  pages={336 - 359},
  url={https://api.semanticscholar.org/CorpusID:15019293}
}
```

The applicable version for GNN is:

```bibtex
@INPROCEEDINGS{8954227,
  author={Pope, Phillip E. and Kolouri, Soheil and Rostami, Mohammad and Martin, Charles E. and Hoffmann, Heiko},
  booktitle={2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Explainability Methods for Graph Convolutional Neural Networks}, 
  year={2019},
  volume={},
  number={},
  pages={10764-10773},
  keywords={Deep Learning;Deep Learning},
  doi={10.1109/CVPR.2019.01103}}
```
