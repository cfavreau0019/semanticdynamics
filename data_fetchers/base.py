"""Base dataset connector abstractions.

This file defines a minimal interface for dataset connectors so additional
connectors (e.g., FEVER, XSum annotations) can be added with the same API.
"""
from typing import List, Dict


class DatasetConnector:
    """Minimal connector interface.

    Implementations should provide `download()` which persists raw data to
    disk and `load_into_memory()` which returns a list of example dicts.
    """

    def download(self, save_dir: str, **kwargs) -> str:
        """Download and persist dataset to `save_dir`.

        Returns the path where the dataset was saved.
        """
        raise NotImplementedError()

    def load_into_memory(self, path: str = None) -> List[Dict]:
        """Load dataset content into memory and return as list of dicts."""
        raise NotImplementedError()
