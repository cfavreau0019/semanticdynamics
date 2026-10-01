"""Regenerating responses that fail validation: generate_many, run_batch, retry_invalid."""
import threading

import pytest

from llm import (
    ChatRequest, ChatResponse, Usage, generate_many, load_responses, retry_invalid, run_batch, run_batch_results,
    save_responses,
)
from llm.hooks import DEFAULT_VALIDATION_RETRIES, merge_retry, needs_retry
from llm.providers import LLMProvider


class ScriptedProvider(LLMProvider):
    """Returns scripted outputs per prompt, in order; an Exception entry is raised. Last entry repeats."""
    name = "scripted"

    def __init__(self, script):
        super().__init__(default_model="m")
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = {}
        self.lock = threading.Lock()

    def chat(self, request):
        with self.lock:
            n = self.calls.get(request.prompt, 0)
            self.calls[request.prompt] = n + 1
        outputs = self.script[request.prompt]
        out = outputs[min(n, len(outputs) - 1)]
        if isinstance(out, Exception):
            raise out
        return ChatResponse(request.id, out, finish_reason="stop", usage=Usage(1, len(out), 1 + len(out)),
                            metadata=request.metadata)


def good_if_ok(response, request):
    return {"passed": response.text.startswith("ok"), "n_errors": 0 if response.text.startswith("ok") else 1}


def reqs(*prompts):
    return [ChatRequest.from_prompt(p, id=p) for p in prompts]


# ---- helpers --------------------------------------------------------------------------
def test_default_is_two_retries():
    assert DEFAULT_VALIDATION_RETRIES == 2


def test_needs_retry():
    assert needs_retry(ChatResponse("a", "x", validation={"passed": False}))
    assert not needs_retry(ChatResponse("a", "x", validation={"passed": True}))
    assert not needs_retry(ChatResponse("a", "x"))                                   # not validated
    assert not needs_retry(ChatResponse("a", "x", validation={"passed": None}))      # validator crashed
    assert not needs_retry(ChatResponse("a", None, error="boom", validation={"passed": False}))


def test_merge_retry_success_and_error():
    first = ChatResponse("a", "bad", validation={"passed": False}, usage=Usage(1, 2, 3))
    second = ChatResponse("a", "ok", validation={"passed": True})
    merged = merge_retry(first, second)
    assert merged is second and merged.attempts == 2
    assert merged.failed_attempts == [{"text": "bad", "finish_reason": None, "validation": {"passed": False},
                                       "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                                       "error": None, "model": None, "batch_id": None}]
    errored = merge_retry(merged, ChatResponse("a", None, error="boom"))
    assert errored is second and errored.attempts == 3
    assert [f["error"] for f in errored.failed_attempts] == [None, "boom"]


# ---- generate_many --------------------------------------------------------------------
def test_retries_until_valid():
    p = ScriptedProvider({"a": ["bad 1", "bad 2", "ok!"], "b": ["ok"]})
    a, b = generate_many(reqs("a", "b"), provider=p, validate=good_if_ok, progress=False)
    assert (a.text, a.valid, a.attempts) == ("ok!", True, 3)
    assert [f["text"] for f in a.failed_attempts] == ["bad 1", "bad 2"]
    assert all(f["validation"]["passed"] is False for f in a.failed_attempts)
    assert (b.attempts, b.failed_attempts) == (1, [])
    assert p.calls == {"a": 3, "b": 1}


def test_gives_up_after_retries_and_returns_last():
    p = ScriptedProvider({"a": ["bad 1", "bad 2", "bad 3", "ok too late"]})
    [a] = generate_many(reqs("a"), provider=p, validate=good_if_ok, progress=False)
    assert (a.text, a.valid, a.attempts, len(a.failed_attempts)) == ("bad 3", False, 3, 2)
    assert p.calls["a"] == 1 + DEFAULT_VALIDATION_RETRIES


@pytest.mark.parametrize("retries, calls", [(0, 1), (1, 2), (5, 4)])
def test_retry_count_configurable(retries, calls):
    p = ScriptedProvider({"a": ["bad", "bad", "bad", "ok"]})
    generate_many(reqs("a"), provider=p, validate=good_if_ok, validation_retries=retries, progress=False)
    assert p.calls["a"] == calls


def test_no_validator_no_retries():
    p = ScriptedProvider({"a": ["bad"]})
    [a] = generate_many(reqs("a"), provider=p, progress=False)
    assert a.attempts == 1 and p.calls["a"] == 1


def test_validator_crash_not_retried():
    def broken(response, request):
        raise RuntimeError("bug")

    p = ScriptedProvider({"a": ["whatever"]})
    [a] = generate_many(reqs("a"), provider=p, validate=broken, progress=False)
    assert a.valid is None and p.calls["a"] == 1


def test_api_error_on_first_attempt_not_retried():
    p = ScriptedProvider({"a": [RuntimeError("down"), "ok"]})
    [a] = generate_many(reqs("a"), provider=p, validate=good_if_ok, progress=False)
    assert not a.ok and a.attempts == 1 and p.calls["a"] == 1


def test_api_error_on_retry_keeps_last_invalid_response():
    p = ScriptedProvider({"a": ["bad", RuntimeError("down"), "ok"]})
    [a] = generate_many(reqs("a"), provider=p, validate=good_if_ok, progress=False)
    assert (a.text, a.ok, a.valid, a.attempts) == ("bad", True, False, 2)
    assert "down" in a.failed_attempts[-1]["error"]
    assert p.calls["a"] == 2  # stops rather than hammering a failing endpoint


def test_on_result_sees_only_final_response():
    p = ScriptedProvider({"a": ["bad", "ok"], "b": ["ok"]})
    seen = []
    generate_many(reqs("a", "b"), provider=p, validate=good_if_ok, progress=False,
                  on_result=lambda r: seen.append((r.request_id, r.text, r.attempts)))
    assert sorted(seen) == [("a", "ok", 2), ("b", "ok", 1)]


def test_history_survives_save_and_load(tmp_path):
    p = ScriptedProvider({"a": ["bad", "ok"]})
    out = generate_many(reqs("a"), provider=p, validate=good_if_ok, progress=False)
    save_responses(out, tmp_path / "r.jsonl")
    [loaded] = load_responses(tmp_path / "r.jsonl")
    assert loaded.attempts == 2 and loaded.failed_attempts == out[0].failed_attempts


def test_summary_printed(capsys):
    p = ScriptedProvider({"a": ["bad", "ok"], "b": ["bad"]})
    generate_many(reqs("a", "b"), provider=p, validate=good_if_ok, max_workers=1)
    assert "2 responses regenerated after failing validation; 1 still invalid" in capsys.readouterr().out


# ---- batch --------------------------------------------------------------------------
def fails_first_time(ids):
    seen = {}

    def validate(response, request):
        seen[response.request_id] = seen.get(response.request_id, 0) + 1
        return {"passed": not (response.request_id in ids and seen[response.request_id] == 1)}
    return validate


def batch_creates(provider):
    return [c for c in provider.client.calls if c[0] == "batch_create"]


def test_run_batch_resubmits_only_invalid_as_follow_up_batch(provider):
    requests = [ChatRequest.from_prompt(f"q{i}", id=f"s{i}", metadata={"i": i}) for i in range(3)]
    out = run_batch(requests, provider=provider, poll_interval=0, validate=fails_first_time({"s1"}))
    assert len(batch_creates(provider)) == 2
    assert [l["custom_id"] for l in provider.client.uploaded] == ["s1"]    # follow-up batch contents
    assert [r.attempts for r in out] == [1, 2, 1] and all(r.valid for r in out)
    assert out[1].metadata == {"i": 1} and out[1].failed_attempts[0]["validation"] == {"passed": False}


def test_run_batch_reports_batches_and_stamps_batch_id(provider):
    events = []
    requests = [ChatRequest.from_prompt(f"q{i}", id=f"s{i}") for i in range(2)]
    out = run_batch(requests, provider=provider, poll_interval=0, validate=fails_first_time({"s1"}),
                    on_batch=lambda job, rnd: events.append((rnd, job.status)))
    assert events == [(0, "pending"), (0, "completed"), (1, "pending"), (1, "completed")]
    assert [r.batch_id for r in out] == ["batch_1", "batch_2"]       # s1 was regenerated by the retry batch
    assert out[1].failed_attempts[0]["batch_id"] == "batch_1"


def test_run_batch_retries_disabled(provider):
    requests = [ChatRequest.from_prompt("q", id="s0")]
    out = run_batch(requests, provider=provider, poll_interval=0, validate=fails_first_time({"s0"}),
                    validation_retries=0)
    assert len(batch_creates(provider)) == 1 and out[0].valid is False


def test_run_batch_results_never_submits(provider):
    requests = [ChatRequest.from_prompt("q", id="s0")]
    provider.submit_batch(requests)
    out = run_batch_results("batch_1", provider=provider, requests=requests, validate=fails_first_time({"s0"}))
    assert out[0].valid is False and len(batch_creates(provider)) == 1


def test_retry_invalid_live_after_collecting_batch(provider):
    requests = [ChatRequest.from_prompt(f"q{i}", id=f"s{i}") for i in range(2)]
    provider.submit_batch(requests)
    validate = fails_first_time({"s0"})
    collected = run_batch_results("batch_1", provider=provider, requests=requests, validate=validate)
    fixed = retry_invalid(collected, requests, validate, provider=provider, mode="live", progress=False)
    assert [r.valid for r in fixed] == [True, True] and fixed[0].attempts == 2
    assert fixed[0].text == "echo:q0"                        # regenerated live
    assert collected[0].valid is False                       # input list not modified
    assert len(batch_creates(provider)) == 1


def test_retry_invalid_batch_mode(provider):
    requests = [ChatRequest.from_prompt(f"q{i}", id=f"s{i}") for i in range(2)]
    provider.submit_batch(requests)
    validate = fails_first_time({"s1"})
    collected = run_batch_results("batch_1", provider=provider, requests=requests, validate=validate)
    fixed = retry_invalid(collected, requests, validate, provider=provider, mode="batch", poll_interval=0,
                          progress=False)
    assert fixed[1].valid and fixed[1].attempts == 2 and len(batch_creates(provider)) == 2


def test_retry_invalid_leaves_others_alone_and_validates_mode():
    p = ScriptedProvider({"a": ["ok"]})
    responses = [ChatResponse("v", "fine", validation={"passed": True}),
                 ChatResponse("n", "unvalidated"),
                 ChatResponse("e", None, error="down")]
    out = retry_invalid(responses, reqs("v", "n", "e"), good_if_ok, provider=p, progress=False)
    assert out == responses and p.calls == {}
    with pytest.raises(ValueError, match="mode"):
        retry_invalid(responses, [], good_if_ok, provider=p, mode="later")


def test_batch_file_names_unique_within_a_second(provider):
    lines = provider.build_batch_lines([ChatRequest.from_prompt("q", id="a")])
    assert provider.write_batch_file(lines) != provider.write_batch_file(lines)
