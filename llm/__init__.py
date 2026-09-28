"""
LLM access for the project:

  inference   generate / generate_many     live chat calls (single + concurrent)
  batch       run_batch / wait_for_batch   asynchronous provider batch jobs (OpenAI)
  embeddings  embed / embed_documents      sentence & chunk embeddings
  whitebox    WhiteBoxModel                NNsight hidden states (imported lazily: pulls in torch)

Providers are chosen per call with provider="openai" | "featherless" | <LLMProvider>.
"""
from llm.types import (
    BatchJob, ChatRequest, ChatResponse, EmbeddingResponse, Message, Usage,
)
from llm.providers import (
    LLMProvider, OpenAICompatibleProvider, available_providers, get_provider, register_provider,
)
from llm.inference import chat, generate, generate_many
from llm.batch import (
    load_requests, load_responses, read_jsonl, run_batch, run_batch_results,
    save_requests, save_responses, wait_for_batch, write_jsonl,
)
from llm.embeddings import (
    SentenceTransformerEmbedder, embed, embed_documents, embed_samples, split_into_sentences,
)

_LAZY = {"WhiteBoxModel", "HiddenStates", "GenerationTrace"}


def __getattr__(name):
    if name in _LAZY:
        from llm import whitebox
        return getattr(whitebox, name)
    raise AttributeError(f"module 'llm' has no attribute {name!r}")


__all__ = [
    "BatchJob", "ChatRequest", "ChatResponse", "EmbeddingResponse", "Message", "Usage",
    "LLMProvider", "OpenAICompatibleProvider", "available_providers", "get_provider", "register_provider",
    "chat", "generate", "generate_many",
    "load_requests", "load_responses", "read_jsonl", "run_batch", "run_batch_results",
    "save_requests", "save_responses", "wait_for_batch", "write_jsonl",
    "SentenceTransformerEmbedder", "embed", "embed_documents", "embed_samples", "split_into_sentences",
    *sorted(_LAZY),
]
