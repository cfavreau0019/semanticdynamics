"""
Provider-agnostic helpers around asynchronous batch jobs, plus JSONL persistence.

Typical flow (OpenAI; results can take up to 24h, at ~50% of the live price):

    from llm import ChatRequest, get_provider, run_batch, save_requests, load_requests

    reqs = [ChatRequest.from_prompt(p, id=f"sample-{i}", max_tokens=2000) for i, p in enumerate(prompts)]
    save_requests(reqs, "data/llm_batches/tarot_requests.jsonl")   # keep metadata for later
    job = run_batch(reqs, provider="openai", wait=False)            # returns a BatchJob
    print(job.id)                                                   # note this down

    # ... later / in another session ...
    reqs = load_requests("data/llm_batches/tarot_requests.jsonl")
    responses = run_batch_results(job.id, provider="openai", requests=reqs)
"""
import json
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

from llm.hooks import DEFAULT_VALIDATION_RETRIES, ValidateFn, apply_validation, merge_retry, needs_retry
from llm.inference import generate_many
from llm.providers import LLMProvider, get_provider
from llm.types import BatchJob, ChatRequest, ChatResponse, Usage


# ---- JSONL ----------------------------------------------------------------------
def write_jsonl(records: Iterable[dict], path: str | Path, append: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_requests(requests: Sequence[ChatRequest], path: str | Path) -> Path:
    return write_jsonl((r.to_dict() for r in requests), path)


def load_requests(path: str | Path) -> list[ChatRequest]:
    return [ChatRequest.from_dict(d) for d in read_jsonl(path)]


def save_responses(responses: Sequence[ChatResponse], path: str | Path, append: bool = False) -> Path:
    return write_jsonl((r.to_dict() for r in responses), path, append=append)


def load_responses(path: str | Path) -> list[ChatResponse]:
    out = []
    for d in read_jsonl(path):
        if d.get("usage") is not None:
            d["usage"] = Usage(**d["usage"])
        out.append(ChatResponse(**d))
    return out


# ---- batch lifecycle --------------------------------------------------------------
def wait_for_batch(
    job: BatchJob | str,
    provider: str | LLMProvider | None = None,
    poll_interval: float = 60.0,
    timeout: Optional[float] = None,
    verbose: bool = True,
) -> BatchJob:
    """Block until the batch reaches a terminal state (completed/failed/expired/cancelled)."""
    llm = get_provider(provider)
    input_file = job.input_file if isinstance(job, BatchJob) else None
    batch_id = job.id if isinstance(job, BatchJob) else job
    start = time.monotonic()
    while True:
        current = llm.get_batch(batch_id)
        current.input_file = input_file
        if verbose:
            c = current.request_counts
            print(f"[{time.strftime('%H:%M:%S')}] batch {batch_id}: {current.raw_status} "
                  f"({c.get('completed', 0)}/{c.get('total', '?')} done, {c.get('failed', 0)} failed)")
        if current.done:
            return current
        if timeout is not None and time.monotonic() - start > timeout:
            raise TimeoutError(f"Batch {batch_id} not finished after {timeout}s (status {current.raw_status}).")
        time.sleep(poll_interval)


def run_batch(
    requests: Sequence[ChatRequest],
    provider: str | LLMProvider | None = None,
    wait: bool = True,
    poll_interval: float = 60.0,
    timeout: Optional[float] = None,
    validate: Optional[ValidateFn] = None,
    validation_retries: int = DEFAULT_VALIDATION_RETRIES,
    **submit_kwargs,
) -> list[ChatResponse] | BatchJob:
    """
    Submit a chat batch. wait=True blocks and returns ordered ChatResponses; wait=False
    returns the BatchJob immediately (collect later with run_batch_results, then
    retry_invalid if you want regenerations).

    With `validate` and wait=True, responses that fail validation are resubmitted as a
    follow-up batch of just those requests, up to `validation_retries` rounds (default 2;
    0 disables). Each round is another batch job, so it can add up to the completion
    window (24h) of waiting per round.
    """
    llm = get_provider(provider)
    job = llm.submit_batch(requests, **submit_kwargs)
    print(f"Submitted batch {job.id} ({len(requests)} requests) to {llm.name}; input: {job.input_file}")
    if not wait:
        return job
    job = wait_for_batch(job, llm, poll_interval=poll_interval, timeout=timeout)
    responses = run_batch_results(job, llm, requests, validate=validate)
    if validate is not None and validation_retries > 0:
        responses = retry_invalid(responses, requests, validate, llm, retries=validation_retries, mode="batch",
                                  poll_interval=poll_interval, timeout=timeout, **submit_kwargs)
    return responses


def run_batch_results(
    job: BatchJob | str,
    provider: str | LLMProvider | None = None,
    requests: Optional[Sequence[ChatRequest]] = None,
    validate: Optional[ValidateFn] = None,
) -> list[ChatResponse]:
    """
    Fetch results for a finished batch (e.g. submitted in an earlier session). With
    `validate`, each successful response gets response.validation before being returned,
    i.e. before you save it.

    This never submits new work. To regenerate responses that failed validation, pass the
    result to retry_invalid (live or as a follow-up batch).
    """
    responses = get_provider(provider).batch_results(job, requests)
    return apply_validation(responses, validate, requests) if validate else responses


def retry_invalid(
    responses: Sequence[ChatResponse],
    requests: Sequence[ChatRequest],
    validate: ValidateFn,
    provider: str | LLMProvider | None = None,
    retries: int = DEFAULT_VALIDATION_RETRIES,
    mode: str = "live",
    max_workers: int = 4,
    poll_interval: float = 60.0,
    timeout: Optional[float] = None,
    progress: bool = True,
    **submit_kwargs,
) -> list[ChatResponse]:
    """
    Regenerate responses that failed validation (valid is False), up to `retries` rounds.

    mode="live"  — regenerate with generate_many (seconds, full price)
    mode="batch" — submit the failures as a follow-up batch and wait (up to 24h, ~half price)

    `requests` supplies the original request for each response (matched by id). Returns a
    new list in the same order; each regenerated response carries `attempts` and
    `failed_attempts`. Responses that are valid, unvalidated, or API failures are untouched.
    """
    if mode not in ("live", "batch"):
        raise ValueError(f"mode must be 'live' or 'batch', got {mode!r}")
    llm = get_provider(provider)
    by_id = {r.id: r for r in requests}
    out = list(responses)
    for round_ in range(1, retries + 1):
        bad = [i for i, r in enumerate(out) if needs_retry(r) and r.request_id in by_id]
        if not bad:
            break
        retry_reqs = [by_id[out[i].request_id] for i in bad]
        if progress:
            print(f"Validation retry round {round_}/{retries}: regenerating {len(bad)} response(s) ({mode})")
        if mode == "live":
            new = generate_many(retry_reqs, provider=llm, max_workers=max_workers, validate=validate,
                                validation_retries=0, progress=progress)
        else:
            job = llm.submit_batch(retry_reqs, **submit_kwargs)
            job = wait_for_batch(job, llm, poll_interval=poll_interval, timeout=timeout, verbose=progress)
            new = run_batch_results(job, llm, retry_reqs, validate=validate)
        for i, n in zip(bad, new):
            out[i] = merge_retry(out[i], n)
    return out
