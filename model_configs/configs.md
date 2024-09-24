### Example Configuration:
```json
{
  "metrics": [
    "Micro-F1",
    "Macro-F1"
  ],
  "meta_paths": [
    [0, 1, 0],
    [0, 2, 0]
  ],
  "lr": 0.01,
  "num_heads": [8],
  "hidden_units": 8,
  "dropout": 0.6,
  "weight_decay": 0.001,
  "num_epochs": 200,
  "patience": 100,
  "summary_path": "model_summary/HAN_ACM_summary.json",
  "device": 0
}
```

### Configuration Keys (with descriptions):

1. **Model Parameters:**
   - **`meta_paths`**: (list of lists of integers)  
     Defines the meta-paths to be used in the HAN model. Each meta-path is represented by a sequence of node types.
     - Example: `[[0, 1, 0], [0, 2, 0]]` indicates two meta-paths.

   - **`num_heads`**: (list of integers)  
     Specifies the number of attention heads for each layer of the model.  
     - Example: `[8]` means 8 attention heads are used.

   - **`hidden_units`**: (integer)  
     Number of hidden units in each GAT (Graph Attention Network) layer.  
     - Example: `8` hidden units.

2. **Training Parameters:**
   - **`lr` (Learning Rate)**: (float)  
     Learning rate for the optimizer.  
     - Example: `0.01`.

   - **`dropout`**: (float)  
     Dropout rate to prevent overfitting.  
     - Example: `0.6`.

   - **`weight_decay`**: (float)  
     Weight decay (L2 regularization) applied during optimization.  
     - Example: `0.001`.

   - **`num_epochs`**: (integer)  
     Number of epochs for training the model.  
     - Example: `200`.

   - **`patience`**: (integer)  
     Number of epochs without improvement before early stopping occurs.  
     - Example: `100`.

3. **Evaluation and Metrics:**
   - **`metrics`**: (list of strings)  
     Metrics to evaluate model performance.  
     - Example: `["Micro-F1", "Macro-F1"]`.

4. **Device Configuration:**
   - **`device`**: (integer)  
     Specifies the device to run the model on.  
     - Example: `0` refers to the first GPU or CPU.

5. **File Paths:**
   - **`summary_path`**: (string)  
     Path to save the model's summary after training.  
     - Example: `"model_summary/HAN_ACM_summary.json"`.
