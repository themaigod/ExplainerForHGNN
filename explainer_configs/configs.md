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
  "summary_path": null,
  "edge_mask_hard_method": "original",
  "feature_mask_hard_method": "original",
  "threshold_auc": [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
  ],
  "auc_use_feature_mask": true,
  "auc_use_edge_mask": true,
  "top_k_for_faith_feature": 0.25,
  "top_k_for_faith_edge": 0.25,
  "perturb_ratio_in_gs_for_stability": 0.001,
  "perturb_std_in_features_for_stability": 0.01,
  "stability_times": 25,
  "pos_weight_characterization_score": 0.5,
  "neg_weight_characterization_score": 0.5,
  "top_k_for_stability_feature": 0.25,
  "top_k_for_stability_edge": 0.25
}
```

(key: available options):

- `record_metrics`: (list[str]) List of metrics to be recorded during training.
  - mask_density: Density of the mask.
- `extract_neighbors`: (true, false) If set to `true`, the neighbors will be extracted. Otherwise, use the whole graph.
- `n_hop`: (int) Number of hops to extract neighbors. If `extract_neighbors` is set to `false`, this value will be
  ignored.
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
- `apply_mata_mask`: (true, false) If set to `true`, the mask will be applied to the meta-path. Noting that this
  parameter is only supported in the those heterogeneous models that provide the meta-path importance tensor.
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
  (key: available options, eval_metrics):
- `edge_mask_hard_method`: (threshold, auto_threshold, top_k, original)
  [sparsity, fidelity_neg, fidelity_pos, unfaithfulness, graph_exp_faith_edge, MacroF1, MicroF1, fidelity_curve_auc]
  Hard method for edge mask.
- `edge_mask_threshold`: (float)
  [sparsity, fidelity_neg, fidelity_pos, unfaithfulness, graph_exp_faith_edge, MacroF1, MicroF1]
  Threshold for edge mask. Only used when `edge_mask_hard_method` is set to `threshold`.
- `threshold_percentage_edge`: (float)
  [sparsity, fidelity_neg, fidelity_pos, unfaithfulness, graph_exp_faith_edge, MacroF1, MicroF1]
  Threshold percentage for edge mask. Only used when `edge_mask_hard_method` is set to `auto_threshold`.
- `top_k_for_edge_mask`: (float)
  [sparsity, fidelity_neg, fidelity_pos, unfaithfulness, graph_exp_faith_edge, MacroF1, MicroF1]
  Top k for edge mask. Only used when `edge_mask_hard_method` is set to `top_k`.
- `feature_mask_hard_method`: (threshold, auto_threshold, top_k, original)
  [sparsity, fidelity_neg, fidelity_pos, unfaithfulness, graph_exp_faith_feature, MacroF1, MicroF1, fidelity_curve_auc]
  Hard method for feature mask.
- `feature_mask_threshold`: (float)
  [sparsity, fidelity_neg, fidelity_pos, unfaithfulness, graph_exp_faith_feature, MacroF1, MicroF1]
  Threshold for feature mask. Only used when `feature_mask_hard_method` is set to `threshold`.
- `threshold_percentage_feature`: (float)
  [sparsity, fidelity_neg, fidelity_pos, unfaithfulness, graph_exp_faith_feature, MacroF1, MicroF1]
  Threshold percentage for feature mask. Only used when `feature_mask_hard_method` is set to `auto_threshold`.
- `top_k_for_feature_mask`: (float)
  [sparsity, fidelity_neg, fidelity_pos, unfaithfulness, graph_exp_faith_feature, MacroF1, MicroF1]
  Top k for feature mask. Only used when `feature_mask_hard_method` is set to `top_k`.
- `threshold_auc`: (list[float])
  [fidelity_curve_auc]
  Thresholds for AUC calculation. Only used when `edge_mask_threshold` or `feature_mask_threshold` is set to
  `original` or `threshold`.
- `top_k_for_auc`: (list[float])
  [fidelity_curve_auc]
  Top k for AUC calculation. Only used when `edge_mask_threshold` or `feature_mask_threshold` is set to
  `top_k`.
- `threshold_percentage_auc`: (list[float])
  [fidelity_curve_auc]
  Threshold percentage for AUC calculation. Only used when `edge_mask_threshold` or `feature_mask_threshold` is set to
  `auto_threshold`.
- `auc_use_feature_mask`: (true, false)
  [fidelity_curve_auc]
  If set to `true`, the feature mask will be used for AUC calculation. `auc_use_feature_mask` and `auc_use_edge_mask`
  cannot be set to `false` at the same time.
- `auc_use_edge_mask`: (true, false)
  [fidelity_curve_auc]
  If set to `true`, the edge mask will be used for AUC calculation. `auc_use_feature_mask` and `auc_use_edge_mask`
  cannot be set to `false` at the same time.
- `top_k_for_faith_feature`: (float)
  [graph_exp_faith_feature]
  Top k for graph explanation faithfulness for feature.
- `top_k_for_faith_edge`: (float)
  [graph_exp_faith_edge]
  Top k for graph explanation faithfulness for edge.
- `perturb_ratio_in_gs_for_stability`: (float)
  [graph_exp_stability_feature, graph_exp_stability_edge]
  Perturbation ratio in graph structure for stability. We use rewiring (swapping) method to perturb the graph structure.
  u--v u v
  becomes | |
  x--y x y
- `perturb_std_in_features_for_stability`: (float)
  [graph_exp_stability_feature, graph_exp_stability_edge]
  Standard deviation in features for stability. We use Gaussian noise to perturb the features.
- `stability_times`: (int)
  [graph_exp_stability_feature, graph_exp_stability_edge]
  Number of times to calculate stability to get the maximum value.
- `pos_weight_characterization_score`: (float)
  [characterization_score]
  Weight for fidelity_pos in characterization
  score. `pos_weight_characterization_score` + `neg_weight_characterization_score`
  should be equal to 1.
- `neg_weight_characterization_score`: (float)
  [characterization_score]
  Weight for fidelity_neg in characterization
  score. `pos_weight_characterization_score` + `neg_weight_characterization_score`
  should be equal to 1.
- `top_k_for_stability_feature`: (float)
  [graph_exp_stability_feature]
  Top k for graph explanation stability for feature.
- `top_k_for_stability_edge`: (float)
  [graph_exp_stability_edge]
  Top k for graph explanation stability for edge.
