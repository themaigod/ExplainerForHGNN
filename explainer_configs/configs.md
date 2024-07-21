# GNN Explainer:

Example:

```json
{
  "record_metrics": [
    "mask_density"
  ],
  "extract_neighbors": true,
  "n_hop": 3,
  "init_strategy_for_edge": "normal",
  "use_mask_bias": false,
  "init_strategy_for_feature": "constant",
  "edge_mask_threshold": 0.5,
  "feature_mask_threshold": 0.5,
  "opt_scheduler": null,
  "opt": "adam",
  "edge_mask_activation": "sigmoid",
  "epochs": 100,
  "record": true,
  "record_step": 1,
  "feature_mask_use_sigmoid": true,
  "coff_normal": 1.0,
  "coff_edge_size": 0.005,
  "coff_feature_size": 1.0,
  "coff_edge_ent": 1.0,
  "coff_feature_ent": 0.1,
  "coff_laplacian": 1.0,
  "mask_features": true,
  "feature_mask_marginalize": false,
  "opt_lr": 0.1,
  "opt_wd": 0.0,
  "opt_decay_rate": null,
  "opt_decay_step": null,
  "opt_restart": null,
  "eval_metrics": [
    "fidelity_neg",
    "fidelity_pos",
    "characterization_score",
    "fidelity_curve_auc",
    "unfaithfulness",
    "sparsity",
    "graph_exp_faith_feature",
    "graph_exp_faith_edge",
    "graph_exp_stability_feature",
    "graph_exp_stability_edge",
    "Macro-F1",
    "Micro-F1",
    "roc_auc_score"
  ],
  "summary_path": null
}
```

(key: available options):
- `record_metrics`: (list[str]) List of metrics to be recorded during training.
	- mask_density: Density of the mask.
- `extract_neighbors`: (true, false) If set to `true`, the neighbors will be extracted. Otherwise, use the whole graph.
- `n_hop`: (int) Number of hops to extract neighbors. If `extract_neighbors` is set to `false`, this value will be ignored.
You can count the number of graph convolutions stacked in the model to determine the number of hops.
- `init_strategy_for_edge`: (normal, constant) Initialization strategy for edge mask.
- `use_mask_bias`: (true, false) If set to `true`, the mask bias will be used.
- `init_strategy_for_feature`: (normal, constant) Initialization strategy for feature mask.
- `edge_mask_threshold`: (float) Threshold for edge mask.
- `feature_mask_threshold`: (float) Threshold for feature mask.
- `opt_scheduler`: (null, step, cos) Scheduler for optimizer. If set to `null`, the optimizer will not use scheduler.
- `opt`: (adam, sgd, rmsprop, adagrad) Optimizer.
- `edge_mask_activation`: (sigmoid, relu) Activation function for edge mask.
- `epochs`: (int) Number of epochs for training.
- `record`: (true, false) If set to `true`, the training process will be recorded.
- `record_step`: (int) Step for recording. If `record` is set to `false`, this value will be ignored.
- `feature_mask_use_sigmoid`: (true, false) If set to `true`, the feature mask will use sigmoid activation.
- `coff_normal`: (float) Coefficient for normal loss.
- `coff_edge_size`: (float) Coefficient for edge size loss.
- `coff_feature_size`: (float) Coefficient for feature size loss.
- `coff_edge_ent`: (float) Coefficient for edge entropy loss.
- `coff_feature_ent`: (float) Coefficient for feature entropy loss.
- `coff_laplacian`: (float) Coefficient for laplacian loss.
- `mask_features`: (true, false) If set to `true`, the features will be masked.
- `feature_mask_marginalize`: (true, false) If set to `true`, the feature mask will be marginalized.
- `opt_lr`: (float) Learning rate for optimizer.
- `opt_wd`: (float) Weight decay for optimizer.
- `opt_decay_rate`: (float, null) Decay rate for optimizer. Only used when `opt_scheduler` is set to `step`.
- `opt_decay_step`: (int, null) Decay step for optimizer. Only used when `opt_scheduler` is set to `step`.
- `opt_restart`: (int, null) Restart for optimizer. Only used when `opt_scheduler` is set to `cos`.
- `eval_metrics`: (list[str]) List of metrics to be evaluated.
	- fidelity_neg: Negative fidelity.
	- fidelity_pos: Positive fidelity.
	- characterization_score: Characterization score.
	- fidelity_curve_auc: Fidelity curve AUC.
	- unfaithfulness: Unfaithfulness.
	- sparsity: Sparsity.
	- graph_exp_faith_feature: Graph explanation faithfulness for feature.
	- graph_exp_faith_edge: Graph explanation faithfulness for edge.
	- graph_exp_stability_feature: Graph explanation stability for feature.
	- graph_exp_stability_edge: Graph explanation stability for edge.
	- Macro-F1: Macro-F1 score.
	- Micro-F1: Micro-F1 score.
	- roc_auc_score: ROC AUC score.
- `summary_path`: (string, null) Path to save the summary. If set to `null`, the summary will not be saved.
