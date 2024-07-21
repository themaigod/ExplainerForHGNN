# Do We Really Need to Access the meta-path in Heterogeneous Information Networks: An Explainable Graph Neural Network Approach

## Introduction

This repository implements several explainable graph neural network models, and applies them to heterogeneous GNNs.
We do fair comparison between different models and evaluate their performance on several benchmark datasets.

## Requirements

- Python 3.10

## Run the code

To run the code, you can use the following command:

```bash
python main.py --dataset {dataset_name} --model {model_name} --explainer {explainer_name} --random_seed {seed}
```

You can do some customization by changing `./dataset_configs/{dataset_name}.json`, `./model_configs/{model_name}_{dataset_name}.json`
and `./explainer_configs/{explainer_name}.json`.

## Explainer

[//]: # ()[test.py](..%2F..%2F..%2F..%2Fwechat%20file%2FWeChat%20Files%2Fwxid_87nrr49t3h7222%2FFileStorage%2FFile%2F2024-06%2Ftest.py)
[//]: # (```bibtex)

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

Traditionally, author code use threshould to get the hard mask. In practically, there is no work since it can all below the threshold. we use control sparsity instead.





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

[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
[//]: #
