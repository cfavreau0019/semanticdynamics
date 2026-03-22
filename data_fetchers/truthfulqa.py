"""TruthfulQA connector.

This connector uses the Hugging Face `datasets` library to fetch TruthfulQA
and persist it as JSONL for reproducible local loading. The connector keeps
the saved format generic (one JSON object per line) so other loaders can
consume it easily.

Notes
- Requires `datasets` (Hugging Face) to be installed: `pip install datasets`.
"""
import os
import json
from typing import List, Dict, Optional
import argparse

def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ('yes','true','t','y','1'): return True
    if v in ('no','false','f','n','0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected')


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def download(save_dir: str = "data/truthfulqa", config: Optional[str] = None, split: Optional[str] = None) -> str:
    """Download TruthfulQA via `datasets` and save as JSONL.

    Args:
        save_dir: directory to write dataset files into.
        config: optional dataset config (e.g., 'mc1' for multiple-choice config).
        split: optional split name ('train'|'validation'|'test'). If omitted,
            all available splits are saved into `save_dir` with names like
            `validation.jsonl`.

    Returns:
        Path to the primary saved file (if split provided) or the directory
        containing saved split files.
    """
    try:
        # Import here to avoid heavy dependency at module import time.
        from datasets import load_dataset as hf_load_dataset
    except Exception as e:
        raise RuntimeError("Missing dependency 'datasets'. Install with `pip install datasets`.") from e

    _ensure_dir(save_dir)

    dataset_name = "truthful_qa"
    config = config or 'generation'  # default to generation config

    if split is None:
        print("Config: ", config)
        ds = hf_load_dataset(dataset_name, name=config)
        # ds is likely a DatasetDict with splits; write each split to file
        for sname, subset in ds.items():
            out_path = os.path.join(save_dir, f"{sname}.jsonl")
            subset.to_json(out_path)
        return save_dir
    else:
        ds = hf_load_dataset(dataset_name, name=config, split=split)
        out_path = os.path.join(save_dir, f"{split}.jsonl")
        ds.to_json(out_path)
        return out_path


def load_into_memory(path: str) -> List[Dict]:
    """Load a JSONL file produced by `download()` and return list of dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TruthfulQA dataset and save as JSONL.")
    parser.add_argument("--save_dir", default="data/truthfulqa", help="Directory to write dataset files into.")

    args = parser.parse_args()
    # Example usage: download all splits with default config
    download(save_dir="data/truthfulqa")