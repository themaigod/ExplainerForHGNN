Example:

```json
{
  "metrics": [
    "Micro-F1",
    "Macro-F1"
  ],
  "meta_paths": [
    [
      0,
      1,
      0
    ],
    [
      0,
      2,
      0
    ]
  ],
  "lr": 0.01,
  "num_heads": [
    8
  ],
  "hidden_units": 8,
  "dropout": 0.6,
  "weight_decay": 0.001,
  "num_epochs": 200,
  "patience": 100,
  "summary_path": "model_summary/HAN_ACM_summary.json",
  "device": 0
}
```

(key: available options):
- `metrics`: (list[str]) List of metrics to be used for evaluation.
- `meta_paths`: (list[list[int]]) List of meta-paths to be used for HAN model.
- `lr`: (float) Learning rate.
- `num_heads`: (list[int]) List of number of heads for each layer.
- `hidden_units`: (int) Number of hidden units.
- `dropout`: (float) Dropout rate.
- `weight_decay`: (float) Weight decay.
- `num_epochs`: (int) Number of epochs.
- `patience`: (int) Patience for early stopping.
- `summary_path`: (str) Path to save the model summary.
- `device`: (int) Device to run the model on.
