"""Simple dataset fetcher utilities.

Expose a small loader API for dataset connectors.
"""
from .loader import download_dataset, load_dataset_into_memory

__all__ = ["download_dataset", "load_dataset_into_memory"]
