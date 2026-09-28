import pytest

from llm import (
    BatchJob, ChatRequest, ChatResponse, Usage, load_requests, load_responses, read_jsonl, run_batch,
    run_batch_results, save_requests, save_responses, wait_for_batch, write_jsonl,
)
from llm.types import BATCH_COMPLETED
from conftest import FakeOpenAIClient


def test_jsonl_write_read_append(tmp_path):
    path = tmp_path / "sub" / "x.jsonl"
    write_jsonl([{"a": 1}, {"b": "é"}], path)
    write_jsonl([{"c": 3}], path, append=True)
    assert read_jsonl(path) == [{"a": 1}, {"b": "é"}, {"c": 3}]


def test_requests_round_trip(tmp_path):
    reqs = [ChatRequest.from_prompt(f"p{i}", id=f"s{i}", metadata={"inst": {"Present": "The Fool"}},
                                    max_tokens=3) for i in range(3)]
    save_requests(reqs, tmp_path / "r.jsonl")
    assert load_requests(tmp_path / "r.jsonl") == reqs


def test_responses_round_trip(tmp_path):
    resps = [ChatResponse("a", "t", model="m", usage=Usage(1, 2, 3), metadata={"k": 1}, raw=object()),
             ChatResponse("b", None, error="bad")]
    save_responses(resps, tmp_path / "o.jsonl")
    loaded = load_responses(tmp_path / "o.jsonl")
    assert loaded[0].usage == Usage(1, 2, 3) and loaded[0].raw is None
    assert (loaded[1].text, loaded[1].error) == (None, "bad")


def test_wait_for_batch_polls_until_terminal(make_provider):
    client = FakeOpenAIClient(statuses=("validating", "in_progress", "finalizing", "completed"))
    p = make_provider(client=client)
    job = p.submit_batch([ChatRequest.from_prompt("q")])
    final = wait_for_batch(job, p, poll_interval=0, verbose=False)
    assert final.status == BATCH_COMPLETED
    assert final.input_file == job.input_file
    assert sum(c[0] == "batch_retrieve" for c in client.calls) == 3


def test_wait_for_batch_timeout(make_provider):
    p = make_provider(client=FakeOpenAIClient(statuses=("validating", "in_progress")))
    with pytest.raises(TimeoutError):
        wait_for_batch("batch_1", p, poll_interval=0.01, timeout=0.03, verbose=False)


def test_run_batch_blocking_returns_ordered_responses(provider):
    reqs = [ChatRequest.from_prompt(f"q{i}", id=f"s{i}") for i in range(4)]
    out = run_batch(reqs, provider=provider, poll_interval=0)
    assert [r.text for r in out] == [f"B:q{i}" for i in range(4)]


def test_run_batch_non_blocking_then_collect(provider, tmp_path):
    reqs = [ChatRequest.from_prompt(f"q{i}", id=f"s{i}", metadata={"i": i}) for i in range(2)]
    job = run_batch(reqs, provider=provider, wait=False)
    assert isinstance(job, BatchJob)
    save_requests(reqs, tmp_path / "r.jsonl")
    wait_for_batch(job.id, provider, poll_interval=0, verbose=False)
    out = run_batch_results(job.id, provider=provider, requests=load_requests(tmp_path / "r.jsonl"))
    assert [(r.text, r.metadata["i"]) for r in out] == [("B:q0", 0), ("B:q1", 1)]
