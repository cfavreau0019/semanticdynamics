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
    **submit_kwargs,
) -> list[ChatResponse] | BatchJob:
    """
    Submit a chat batch. wait=True blocks and returns ordered ChatResponses;
    wait=False returns the BatchJob immediately (collect later with run_batch_results).
    """
    llm = get_provider(provider)
    job = llm.submit_batch(requests, **submit_kwargs)
    print(f"Submitted batch {job.id} ({len(requests)} requests) to {llm.name}; input: {job.input_file}")
    if not wait:
        return job
    job = wait_for_batch(job, llm, poll_interval=poll_interval, timeout=timeout)
    return llm.batch_results(job, requests)


def run_batch_results(
    job: BatchJob | str,
    provider: str | LLMProvider | None = None,
    requests: Optional[Sequence[ChatRequest]] = None,
) -> list[ChatResponse]:
    """Fetch results for a finished batch (e.g. submitted in an earlier session)."""
    return get_provider(provider).batch_results(job, requests)
