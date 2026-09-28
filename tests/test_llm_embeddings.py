import numpy as np
import pytest

from llm import embed, embed_documents, embed_samples, split_into_sentences
from llm.types import EmbeddingResponse


class CountingEmbedder:
    """Duck-typed backend (like SentenceTransformerEmbedder): vector = [len(text), index-in-call]."""

    def __init__(self):
        self.calls = []

    def embed(self, texts, model=None, batch_size=None, **kwargs):
        self.calls.append(list(texts))
        return EmbeddingResponse(np.array([[len(t), i] for i, t in enumerate(texts)], dtype=np.float32))


def test_split_into_sentences():
    text = "One. Two three!  Four?\n\nFive\nsix."
    assert split_into_sentences(text) == ["One.", "Two three!", "Four?", "Five", "six."]
    assert split_into_sentences("   ") == []


def test_embed_single_string_returns_1d(provider):
    v = embed("abc", provider=provider)
    assert v.shape == (2,) and v[0] == 3


def test_embed_list_order_across_batches_and_workers(provider):
    texts = ["a" * n for n in range(1, 11)]
    v = embed(texts, provider=provider, batch_size=3, max_workers=3)
    assert v.shape == (10, 2) and v[:, 0].tolist() == list(range(1, 11))


def test_embed_normalize(provider):
    v = embed(["abc", "a"], provider=provider, normalize=True)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, rtol=1e-6)


def test_embed_empty(provider):
    assert embed([], provider=provider).size == 0


def test_embed_accepts_duck_typed_backend():
    backend = CountingEmbedder()
    v = embed(["ab", "abcd"], provider=backend)
    assert v[:, 0].tolist() == [2, 4]


def test_embed_rejects_object_without_embed():
    with pytest.raises(TypeError):
        embed(["a"], provider=object())


def test_embed_documents_pools_chunks_into_shared_calls():
    backend = CountingEmbedder()
    docs = {"d1": "One. Two three!", "d2": "Solo", "d3": "A. B. C."}
    out = embed_documents(docs, provider=backend, batch_size=100)
    assert len(backend.calls) == 1  # all 6 chunks in one request
    assert out["d1"]["chunks"] == ["One.", "Two three!"]
    assert out["d3"]["embeddings"].shape == (3, 2)
    assert out["d2"]["embeddings"][:, 0].tolist() == [4]


def test_embed_documents_list_input_and_no_splitter():
    out = embed_documents(["x. y.", "zz"], splitter=None, provider=CountingEmbedder())
    assert list(out) == [0, 1]
    assert out[0]["chunks"] == ["x. y."]


def test_embed_documents_custom_splitter():
    out = embed_documents({"k": "a b c"}, splitter=str.split, provider=CountingEmbedder())
    assert out["k"]["chunks"] == ["a", "b", "c"]


def test_embed_samples_in_place():
    samples = {0: {"reading": "Hi. There."}, 1: {"reading": "Solo", "prompt": "p"}}
    result = embed_samples(samples, provider=CountingEmbedder())
    assert result is samples
    assert samples[0]["chunks"] == ["Hi.", "There."]
    assert samples[1]["embeddings"].shape == (1, 2) and samples[1]["prompt"] == "p"
