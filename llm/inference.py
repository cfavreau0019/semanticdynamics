"""
Black-box text generation: one-off calls and concurrent fan-out over many requests.

    from llm import generate, generate_many, ChatRequest

    text = generate("Hello", provider="featherless", max_tokens=100)

    reqs = [ChatRequest.from_prompt(celtic_cross_prompt(inst), max_tokens=6400, metadata={"instant": inst})
            for inst in instants]
    responses = generate_many(reqs, provider="featherless", max_workers=4)
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Sequence

from tqdm.auto import tqdm

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
    """Prompt in, text out. Extra kwargs (max_tokens, temperature, ...) are sent as params."""
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
    **params,
) -> list[ChatResponse]:
    """
    Run many requests concurrently against a live endpoint (threads; the SDK client is
    thread-safe and retries 429/5xx itself).

    requests       — ChatRequests, or plain prompt strings (wrapped using system/model/params)
    max_workers    — concurrent in-flight requests; keep within your provider's rate/concurrency
                     limits (Featherless plans cap concurrent connections)
    raise_on_error — False: a failed request yields ChatResponse(error=...) and the rest continue
    on_result      — called (in the main thread) as each response lands, e.g. to append to a
                     JSONL checkpoint so a crash doesn't lose finished work
    Returns responses in the same order as `requests`.
    """
    llm = get_provider(provider)
    reqs = [
        r if isinstance(r, ChatRequest) else ChatRequest.from_prompt(r, system=system, model=model, **params)
        for r in requests
    ]

    def run(req: ChatRequest) -> ChatResponse:
        try:
            return llm.chat(req)
        except Exception as e:
            if raise_on_error:
                raise
            return ChatResponse(request_id=req.id, text=None, error=f"{type(e).__name__}: {e}",
                                metadata=req.metadata)

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
    return results
