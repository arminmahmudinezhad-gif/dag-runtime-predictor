# Dataset

The full synthetic dataset is not tracked in Git. It contains more than ten thousand generated files and should be reproduced locally with:

```bash
jupyter lab notebooks/01_generate_execution_time_dataset_25k.ipynb
```

The notebook creates `data/dag_runtime_dataset_25k/`, including split records, graph descriptions, node and edge features, simulator parameters, and metadata used by the training and evaluation pipeline.

Expected high-level layout:

```text
data/dag_runtime_dataset_25k/
├── features/
├── graphs/
├── metadata/
├── records/
└── simulator/
```

Keep the generated files local. They are ignored by `.gitignore`.
