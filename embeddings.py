from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import math


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs = embs / norms
        return embs


def chord_distance(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(u - v))


def arc_distance(u: np.ndarray, v: np.ndarray) -> float:
    dot = float(np.dot(u, v))
    dot = max(min(dot, 1.0), -1.0)
    return float(math.acos(dot))


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    return 1.0 - float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def semantic_volume(embs: np.ndarray) -> float:
    # Compute simplex volume for N points: volume = parallelepiped_volume / factorial(N-1)
    # For numerical stability use SVD on the difference matrix
    n, d = embs.shape
    if n < 2:
        return 0.0
    M = embs[1:] - embs[0]
    # Parallelepiped volume = product of singular values
    s = np.linalg.svd(M, compute_uv=False)
    parallelepiped = float(np.prod(s))
    denom = math.factorial(n - 1)
    return parallelepiped / denom


def pairwise_distances(embs: np.ndarray, metric: str = "chord") -> np.ndarray:
    n = embs.shape[0]
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if metric == "chord":
                d = chord_distance(embs[i], embs[j])
            elif metric == "arc":
                d = arc_distance(embs[i], embs[j])
            elif metric == "cosine":
                d = cosine_distance(embs[i], embs[j])
            else:
                d = float(np.linalg.norm(embs[i] - embs[j]))
            out[i, j] = out[j, i] = d
    return out
