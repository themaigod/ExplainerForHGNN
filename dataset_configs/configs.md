Example:

```json
{
  "check_data_size": true,
  "labels": "../data/ACM/labels_5_fold_cross_validation_1.pkl",
  "test_label_shuffle": false
}
```

(key: available options):
- `check_data_size`: (true, false) If set to `true`, the data size will be checked.
- `labels`: (string, null) Path to the labels file. If set to `null`, the labels will be found in default path.
- `test_label_shuffle`: (true, false) If set to `true`, the test labels will be shuffled.

