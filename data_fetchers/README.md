# data_fetchers

Small utilities to download and load hallucination-related datasets into memory.

Quick start

1. Install requirements (recommended in a venv):

```bash
pip install datasets
```

2. Download TruthfulQA (example):

```python
from data_fetchers import download_dataset

download_dataset("truthfulqa", save_dir="data/truthfulqa", config="mc1", split="validation")
```

3. Load into memory:

```python
from data_fetchers import load_dataset_into_memory

items = load_dataset_into_memory("truthfulqa", path="data/truthfulqa/validation.jsonl")
print(len(items))
print(items[0].keys())
```

Design notes
- Connectors are implemented as modules (e.g., `truthfulqa.py`) and expose
  `download(save_dir, ...)` and `load_into_memory(path)`.
- The `loader` dispatches based on dataset name and provides sensible
  defaults for `data/<name>/` locations.
