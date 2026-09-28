"""
Sentence / chunk embeddings via any provider with an `embed` method: API providers
(OpenAI, Featherless) or a local SentenceTransformerEmbedder.

    from llm import embed, embed_documents

    vecs = embed(["a sentence", "another"], provider="openai")          # (2, dim) np.ndarray
    docs = embed_documents({i: s["reading"] for i, s in samples.items()}, provider="featherless")
    docs[0]["chunks"], docs[0]["embeddings"]                            # list[str], (n_chunks, dim)
"""
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Hashable, Mapping, Optional, Sequence

import numpy as np

from llm.providers import LLMProvider, get_provider
from llm.types import EmbeddingResponse


def split_into_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation or newlines; drops empty pieces."""
    chunks = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [c.strip() for c in chunks if c.strip()]


class SentenceTransformerEmbedder:
    """
    Local embedding backend with the same `embed` signature as API providers, so it can
    be passed as `provider=` to the functions below. Requires `sentence-transformers`.
    """
    name = "sentence-transformers"
    supports_embeddings = True

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None, **kwargs):
        from sentence_transformers import SentenceTransformer
        self.default_embedding_model = model_name
        self.model = SentenceTransformer(model_name, device=device, **kwargs)

    def embed(self, texts: Sequence[str], model: Optional[str] = None, batch_size: int = 64, **kwargs) -> EmbeddingResponse:
        if model is not None and model != self.default_embedding_model:
            raise ValueError(f"This embedder is loaded with {self.default_embedding_model!r}, not {model!r}.")
        vecs = self.model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                                 show_progress_bar=False, **kwargs)
        return EmbeddingResponse(embeddings=vecs.astype(np.float32), model=self.default_embedding_model)


def _embedder(provider: Any):
    if provider is None or isinstance(provider, (str, LLMProvider)):
        return get_provider(provider)
    if not hasattr(provider, "embed"):
        raise TypeError(f"{provider!r} has no embed() method.")
    return provider


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def embed(
    texts: str | Sequence[str],
    provider: Any = None,
    model: Optional[str] = None,
    normalize: bool = False,
    batch_size: int = 256,
    max_workers: int = 1,
    **kwargs,
) -> np.ndarray:
    """
    Embed one text (-> shape (dim,)) or many (-> shape (n, dim), input order preserved).

    batch_size  — texts per API call
    max_workers — >1 sends several batch_size-sized calls concurrently
    kwargs      — forwarded to the backend (e.g. dimensions=512 for OpenAI text-embedding-3-*)
    """
    single = isinstance(texts, str)
    texts = [texts] if single else list(texts)
    backend = _embedder(provider)
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    call = lambda b: backend.embed(b, model=model, batch_size=batch_size, **kwargs).embeddings
    if max_workers > 1 and len(batches) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            parts = list(ex.map(call, batches))
    else:
        parts = [call(b) for b in batches]

    out = np.concatenate(parts, axis=0)
    if normalize:
        out = _normalize(out)
    return out[0] if single else out


def embed_documents(
    documents: Mapping[Hashable, str] | Sequence[str],
    splitter: Optional[Callable[[str], list[str]]] = split_into_sentences,
    provider: Any = None,
    model: Optional[str] = None,
    normalize: bool = False,
    batch_size: int = 256,
    max_workers: int = 2,
    **kwargs,
) -> dict[Hashable, dict[str, Any]]:
    """
    Split each document into chunks and embed every chunk. All chunks from all documents
    are pooled into shared API calls (far fewer requests than one call per chunk).

    splitter — any str -> list[str] callable (sentence splitter, a DynamicalEmbedding
               subtext grouping, ...). None embeds each document whole.
    Returns {key: {"chunks": list[str], "embeddings": np.ndarray (n_chunks, dim)}};
    list input is keyed by position.
    """
    items = documents.items() if isinstance(documents, Mapping) else enumerate(documents)
    keys, all_chunks, spans = [], [], []
    for key, text in items:
        chunks = splitter(text) if splitter is not None else [text]
        keys.append(key)
        spans.append((len(all_chunks), len(all_chunks) + len(chunks)))
        all_chunks.extend(chunks)

    vecs = embed(all_chunks, provider=provider, model=model, normalize=normalize,
                 batch_size=batch_size, max_workers=max_workers, **kwargs)
    return {
        key: {"chunks": all_chunks[a:b], "embeddings": vecs[a:b]}
        for key, (a, b) in zip(keys, spans)
    }


def embed_samples(samples: dict, text_key: str = "reading", **kwargs) -> dict:
    """
    In-place helper for the notebook's samples dict: adds 'chunks' and 'embeddings' to each
    samples[i] based on samples[i][text_key]. kwargs go to embed_documents.
    """
    results = embed_documents({i: s[text_key] for i, s in samples.items()}, **kwargs)
    for i, r in results.items():
        samples[i]["chunks"] = r["chunks"]
        samples[i]["embeddings"] = r["embeddings"]
    return samples
