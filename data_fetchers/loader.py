"""Generic loader that dispatches to dataset connectors.

This small API lets you call `download_dataset(name, ...)` and
`load_dataset_into_memory(name, path=...)` so additional connectors can be
added under this package and automatically used.
"""
from typing import List, Dict, Optional
import os


def _module_for(name: str):
    nm = name.lower()
    if nm in ("truthfulqa", "truthful_qa", "truthful-qa"):
        from . import truthfulqa as mod
        return mod
    raise ValueError(f"Unknown dataset connector: {name}")


def download_dataset(name: str, save_dir: Optional[str] = None, **kwargs) -> str:
    """Download dataset `name` using its connector.

    Returns path to saved file or directory.
    """
    mod = _module_for(name)
    if save_dir is None:
        save_dir = os.path.join("data", name)
    return mod.download(save_dir=save_dir, **kwargs)


def load_dataset_into_memory(name: str, path: Optional[str] = None) -> List[Dict]:
    """Load dataset into memory. If `path` is omitted, uses a conventional
    location under `data/<name>/` and picks a single file when obvious.
    """
    mod = _module_for(name)
    if path:
        return mod.load_into_memory(path)

    base_dir = os.path.join("data", name)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"No dataset directory found at {base_dir}; specify `path=` or run download_dataset()`")

    # prefer common split files
    for candidate in ("validation.jsonl", "test.jsonl", "train.jsonl"):
        p = os.path.join(base_dir, candidate)
        if os.path.exists(p):
            return mod.load_into_memory(p)

    # otherwise pick the first jsonl in the dir
    for fn in os.listdir(base_dir):
        if fn.endswith(".jsonl"):
            return mod.load_into_memory(os.path.join(base_dir, fn))

    raise FileNotFoundError(f"No .jsonl files found in {base_dir}")
