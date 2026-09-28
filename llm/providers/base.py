from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from llm.types import BatchJob, ChatRequest, ChatResponse, EmbeddingResponse


class LLMProvider(ABC):
    """
    Common interface for black-box LLM APIs.

    Only `chat` is mandatory. Embeddings and batch are optional capabilities: a
    provider that lacks one leaves the default implementation, which raises
    NotImplementedError, and advertises it via `supports_embeddings` /
    `supports_batch`.

    To add a provider (e.g. Anthropic):
      1. subclass LLMProvider, implement `chat` (translate ChatRequest -> native call
         -> ChatResponse) and, if available, the batch methods;
      2. register it in llm/providers/__init__.py (PRESETS or register_provider).
    Everything built on top (generate_many, run_batch, embed, ...) then works unchanged.
    """

    name: str = "base"
    supports_embeddings: bool = False
    supports_batch: bool = False

    def __init__(
        self,
        default_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        default_params: Optional[dict[str, Any]] = None,
    ):
        self.default_model = default_model
        self.default_embedding_model = default_embedding_model
        self.default_params = dict(default_params or {})

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, default_model={self.default_model!r})"

    def resolve_model(self, model: Optional[str]) -> str:
        model = model or self.default_model
        if not model:
            raise ValueError(
                f"No model given and provider '{self.name}' has no default_model "
                f"(set it in .env or pass model=...)."
            )
        return model

    def resolve_embedding_model(self, model: Optional[str]) -> str:
        model = model or self.default_embedding_model
        if not model:
            raise ValueError(f"No embedding model given and provider '{self.name}' has no default.")
        return model

    # ---- chat -------------------------------------------------------------------
    @abstractmethod
    def chat(self, request: ChatRequest) -> ChatResponse:
        """Run one request synchronously. May raise; callers decide how to handle."""
        ...

    # ---- embeddings -------------------------------------------------------------
    def embed(self, texts: Sequence[str], model: Optional[str] = None, **kwargs) -> EmbeddingResponse:
        raise NotImplementedError(f"Provider '{self.name}' does not support embeddings.")

    # ---- batch ------------------------------------------------------------------
    def submit_batch(self, requests: Sequence[ChatRequest], **kwargs) -> BatchJob:
        raise NotImplementedError(f"Provider '{self.name}' does not support batch jobs.")

    def get_batch(self, batch_id: str) -> BatchJob:
        raise NotImplementedError(f"Provider '{self.name}' does not support batch jobs.")

    def cancel_batch(self, batch_id: str) -> BatchJob:
        raise NotImplementedError(f"Provider '{self.name}' does not support batch jobs.")

    def batch_results(self, job: BatchJob | str, requests: Optional[Sequence[ChatRequest]] = None) -> list[ChatResponse]:
        """
        Fetch results of a finished chat batch. If the original `requests` are given,
        results come back in the same order with their metadata re-attached, and any
        request with no result gets a ChatResponse carrying an error.
        """
        raise NotImplementedError(f"Provider '{self.name}' does not support batch jobs.")
