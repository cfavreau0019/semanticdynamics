"""
Provider registry. `get_provider("openai")` / `get_provider("featherless")` build a
provider from .env settings; instances are cached so the HTTP client is reused.

Env vars read (first one found wins):
    openai:       OPENAI_API_KEY | OPENAI_API_TOKEN, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL
    featherless:  FEATHERLESS_API_TOKEN | FEATHERLESS_API_KEY, FEATHERLESS_MODEL, EMBEDDING_MODEL
    LLM_PROVIDER  default provider name when none is passed (falls back to "openai")
"""
import os
from typing import Any, Callable, Optional

from dotenv import load_dotenv

from llm.providers.base import LLMProvider
from llm.providers.openai_compatible import OpenAICompatibleProvider

load_dotenv()

PRESETS: dict[str, dict[str, Any]] = {
    "openai": dict(
        base_url=None,
        api_key_env=("OPENAI_API_KEY", "OPENAI_API_TOKEN"),
        model_env=("OPENAI_MODEL",),
        embedding_model_env=("OPENAI_EMBEDDING_MODEL",),
        default_embedding_model="text-embedding-3-small",
        supports_batch=True,
        # current OpenAI models reject max_tokens; keep calling code provider-neutral
        param_renames={"max_tokens": "max_completion_tokens"},
    ),
    "featherless": dict(
        base_url="https://api.featherless.ai/v1",
        api_key_env=("FEATHERLESS_API_TOKEN", "FEATHERLESS_API_KEY"),
        model_env=("FEATHERLESS_MODEL",),
        embedding_model_env=("EMBEDDING_MODEL",),
        default_embedding_model="Qwen/Qwen3-Embedding-8B",
        supports_batch=False,
    ),
}

# name -> zero/kw-arg callable returning an LLMProvider. Presets are registered below;
# add e.g. register_provider("anthropic", lambda **kw: AnthropicProvider(**kw)).
_FACTORIES: dict[str, Callable[..., LLMProvider]] = {}
_CACHE: dict[str, LLMProvider] = {}


def _first_env(names: tuple[str, ...]) -> Optional[str]:
    for n in names:
        if os.getenv(n):
            return os.getenv(n)
    return None


def _preset_factory(name: str) -> Callable[..., LLMProvider]:
    def build(**overrides) -> LLMProvider:
        p = PRESETS[name]
        kwargs = dict(
            name=name,
            base_url=p["base_url"],
            api_key=_first_env(p["api_key_env"]),
            default_model=_first_env(p["model_env"]),
            default_embedding_model=_first_env(p["embedding_model_env"]) or p["default_embedding_model"],
            supports_batch=p["supports_batch"],
            param_renames=p.get("param_renames"),
        )
        kwargs.update(overrides)
        if not kwargs["api_key"] and kwargs.get("client") is None:
            raise ValueError(f"No API key for '{name}'. Set one of {p['api_key_env']} in .env or pass api_key=.")
        return OpenAICompatibleProvider(**kwargs)
    return build


def register_provider(name: str, factory: Callable[..., LLMProvider]) -> None:
    _FACTORIES[name] = factory
    _CACHE.pop(name, None)


for _name in PRESETS:
    register_provider(_name, _preset_factory(_name))


def available_providers() -> list[str]:
    return sorted(_FACTORIES)


def get_provider(provider: str | LLMProvider | None = None, **overrides) -> LLMProvider:
    """
    Resolve a provider. Accepts an instance (returned as-is), a registered name, or
    None (-> $LLM_PROVIDER or "openai"). Passing overrides (default_model=..., etc.)
    builds a fresh, uncached instance.
    """
    if isinstance(provider, LLMProvider):
        return provider
    name = provider or os.getenv("LLM_PROVIDER", "openai")
    if name not in _FACTORIES:
        raise KeyError(f"Unknown provider '{name}'. Registered: {available_providers()}")
    if overrides:
        return _FACTORIES[name](**overrides)
    if name not in _CACHE:
        _CACHE[name] = _FACTORIES[name]()
    return _CACHE[name]


__all__ = [
    "LLMProvider",
    "OpenAICompatibleProvider",
    "PRESETS",
    "available_providers",
    "get_provider",
    "register_provider",
]
