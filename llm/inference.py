"""
Black-box text generation: one-off calls and concurrent fan-out over many requests.

    from llm import generate, generate_many, ChatRequest

    text = generate("Hello", provider="featherless", max_tokens=100)
    text = generate("Hello", provider="featherless", extra_body={"top_k": 40, "min_p": 0.02})

    reqs = [ChatRequest.from_prompt(celtic_cross_prompt(inst), max_tokens=6400, metadata={"instant": inst})
            for inst in instants]
    responses = generate_many(reqs, provider="featherless", max_workers=4)
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Sequence

from tqdm.auto import tqdm

from llm.hooks import DEFAULT_VALIDATION_RETRIES, ValidateFn, attach_validation, merge_retry, needs_retry
from llm.providers import LLMProvider, get_provider
from llm.types import ChatRequest, ChatResponse

DEFAULT_SYSTEM = "You are a helpful assistant."


def chat(request: ChatRequest, provider: str | LLMProvider | None = None) -> ChatResponse:
    """Run a single ChatRequest. Raises on API errors."""
    return get_provider(provider).chat(request)


def generate(
    prompt: str,
    system: Optional[str] = DEFAULT_SYSTEM,
    provider: str | LLMProvider | None = None,
    model: Optional[str] = None,
    **params,
) -> str:
    """
    Prompt in, text out. Extra kwargs (max_tokens, temperature, ...) are sent as params;
    extra_body={...} carries provider-specific fields (top_k, min_p, repetition_penalty, ...).
    """
    request = ChatRequest.from_prompt(prompt, system=system, model=model, **params)
    return chat(request, provider).text


def generate_many(
    requests: Sequence[ChatRequest | str],
    provider: str | LLMProvider | None = None,
    max_workers: int = 4,
    system: Optional[str] = DEFAULT_SYSTEM,
    model: Optional[str] = None,
    raise_on_error: bool = False,
    on_result: Optional[Callable[[ChatResponse], None]] = None,
    progress: bool = True,
    validate: Optional[ValidateFn] = None,
    validation_retries: int = DEFAULT_VALIDATION_RETRIES,
    **params,
) -> list[ChatResponse]:
    """
    Run many requests concurrently against a live endpoint (threads; the SDK client is
    thread-safe and retries 429/5xx itself).

    requests       — ChatRequests, or plain prompt strings (wrapped using system/model/params,
                     including extra_body={...} if given)
    max_workers    — concurrent in-flight requests; keep within your provider's rate/concurrency
                     limits (Featherless plans cap concurrent connections)
    raise_on_error — False: a failed request yields ChatResponse(error=...) and the rest continue
    on_result      — called (in the main thread) as each response lands, e.g. to append to a
                     JSONL checkpoint so a crash doesn't lose finished work
    validate       — (response, request) -> dict, run on each successful response in its worker
                     thread *before* on_result, so checkpoints already carry response.validation
                     (see validation.response_validator)
    validation_retries — when `validate` is given, regenerate a response that fails validation
                     up to this many extra times (default 2; 0 disables). Retries happen inside
                     the worker, so on_result/checkpoints only see the final response, whose
                     `attempts` and `failed_attempts` record what came before. API errors and
                     validator crashes are not retried here (the SDK retries transient errors).
    Returns responses in the same order as `requests`.
    """
    llm = get_provider(provider)
    reqs = [
        r if isinstance(r, ChatRequest) else ChatRequest.from_prompt(r, system=system, model=model, **params)
        for r in requests
    ]

    def attempt(req: ChatRequest, raising: bool) -> ChatResponse:
        try:
            return attach_validation(llm.chat(req), validate, req)
        except Exception as e:
            if raising:
                raise
            return ChatResponse(request_id=req.id, text=None, error=f"{type(e).__name__}: {e}",
                                metadata=req.metadata)

    def run(req: ChatRequest) -> ChatResponse:
        resp = attempt(req, raise_on_error)
        retries_left = validation_retries if validate is not None else 0
        while retries_left > 0 and needs_retry(resp):
            retries_left -= 1
            retry = attempt(req, raising=False)
            resp = merge_retry(resp, retry)
            if not retry.ok:  # keep the last usable (if invalid) response rather than looping on errors
                break
        return resp

    results: list[Optional[ChatResponse]] = [None] * len(reqs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run, r): i for i, r in enumerate(reqs)}
        bar = tqdm(total=len(reqs), desc=f"{llm.name} chat", disable=not progress)
        try:
            for future in as_completed(futures):
                resp = future.result()
                results[futures[future]] = resp
                if on_result is not None:
                    on_result(resp)
                bar.update(1)
        finally:
            bar.close()

    n_failed = sum(not r.ok for r in results)
    if n_failed and progress:
        print(f"{n_failed}/{len(results)} requests failed; inspect `.error` on those responses.")
    if validate is not None and progress:
        n_retried = sum(r.attempts > 1 for r in results)
        n_invalid = sum(r.valid is False for r in results)
        if n_retried or n_invalid:
            print(f"{n_retried} responses regenerated after failing validation; "
                  f"{n_invalid} still invalid (see .validation / .failed_attempts).")
    return results
