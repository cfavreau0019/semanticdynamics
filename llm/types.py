"""Provider-agnostic request/response containers shared by every part of `llm`."""
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

Message = dict[str, Any]  # {"role": "system" | "user" | "assistant", "content": str}


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class ChatRequest:
    """
    One chat-completion request.

    messages — OpenAI-style message list. Providers that store the system prompt
               separately (e.g. Anthropic) are responsible for pulling it out.
    model    — None means "use the provider's default model".
    params   — sampling/generation params (max_tokens, temperature, ...). Merged over
               the provider's default_params at call time.
    id       — correlation id. Becomes `custom_id` in batch jobs and `request_id` on
               the matching ChatResponse, so results can be joined back to inputs.
    metadata — arbitrary user data (e.g. the tarot `instant`). Never sent to the
               provider; copied onto the ChatResponse.
    extra_body — provider-specific fields outside the standard OpenAI schema, merged
               into the JSON body as-is, e.g. {"top_k": 40, "min_p": 0.02,
               "repetition_penalty": 1.1} for vLLM-based servers such as Featherless.
               Merged over the provider's default_extra_body. OpenAI itself rejects
               unknown fields.
    """
    messages: list[Message]
    model: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
    metadata: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        id: Optional[str] = None,
        metadata: Optional[dict] = None,
        extra_body: Optional[dict] = None,
        **params,
    ) -> "ChatRequest":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return cls(
            messages=messages,
            model=model,
            params=params,
            id=id if id is not None else _new_id(),
            metadata=metadata or {},
            extra_body=extra_body or {},
        )

    @property
    def prompt(self) -> str:
        """Content of the last user message (handy for logging / DataFrames)."""
        for m in reversed(self.messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChatRequest":
        return cls(**d)


@dataclass
class Usage:
    """
    cached_tokens — how many of prompt_tokens the provider served from its prompt (KV) cache,
                    because the prompt began like an earlier one. 0 when the provider reports
                    none, or does not report them.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
            self.cached_tokens + other.cached_tokens,
        )


@dataclass
class ChatResponse:
    """
    Result of a ChatRequest. On failure `text` is None and `error` holds the message,
    so a large concurrent/batch run doesn't die on one bad request.

    `validation` holds the output of a content validator (see the `validate=` hook on
    generate_many / run_batch / apply_validation) as a plain JSON dict with at least
    {"passed": bool | None}. It is kept apart from `metadata` (the request's inputs) and
    from `error` (the API call itself failing): ok=True, valid=False means the call
    succeeded but the content is wrong.

    When a response is regenerated because it failed validation, `attempts` counts every
    generation made for the request and `failed_attempts` keeps a record (text, validation,
    finish_reason, usage, error) of each one that was not returned, oldest first. `usage`
    covers only the returned attempt; add failed_attempts' usage for the total cost.
    """
    request_id: str
    text: Optional[str]
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Usage] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    validation: Optional[dict[str, Any]] = None
    batch_id: Optional[str] = None                                       # batch that produced it (None = live)
    attempts: int = 1                                                    # generations made for this request
    failed_attempts: list[dict[str, Any]] = field(default_factory=list)  # records of attempts not returned
    raw: Any = field(default=None, repr=False)  # provider-native response object/dict

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def valid(self) -> Optional[bool]:
        """True/False once validated; None if not validated (or the validator itself crashed)."""
        return None if self.validation is None else self.validation.get("passed")

    def to_dict(self) -> dict:
        """JSON-serialisable view (drops `raw`)."""
        d = asdict(self)
        d.pop("raw", None)
        return d


@dataclass
class EmbeddingResponse:
    embeddings: np.ndarray            # shape (n_texts, dim), row order == input order
    model: Optional[str] = None
    usage: Optional[Usage] = None


# Normalised batch lifecycle, shared by all providers.
BATCH_PENDING = "pending"        # accepted, not started (validating / queued)
BATCH_RUNNING = "running"        # in progress / finalizing / cancelling
BATCH_COMPLETED = "completed"
BATCH_FAILED = "failed"
BATCH_EXPIRED = "expired"        # window elapsed; partial results may exist
BATCH_CANCELLED = "cancelled"
BATCH_TERMINAL = {BATCH_COMPLETED, BATCH_FAILED, BATCH_EXPIRED, BATCH_CANCELLED}


@dataclass
class BatchJob:
    """
    Handle for an asynchronous provider batch. Persist it with to_dict() (or just keep
    `id`) so results can be collected from a later session: provider.get_batch(id).
    """
    id: str
    provider: str
    status: str                       # one of the BATCH_* constants above
    raw_status: Optional[str] = None  # provider's own status string
    endpoint: Optional[str] = None
    kind: str = "chat"                # "chat" | "embedding"
    request_counts: dict[str, int] = field(default_factory=dict)
    input_file: Optional[str] = None  # local path of the submitted input, if any
    created_at: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = field(default=None, repr=False)

    @property
    def done(self) -> bool:
        return self.status in BATCH_TERMINAL

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d
