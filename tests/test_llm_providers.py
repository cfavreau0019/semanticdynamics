import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

import llm.providers as providers
from llm.providers import LLMProvider, OpenAICompatibleProvider, get_provider, register_provider
from llm.providers.openai_compatible import MAX_BATCH_REQUESTS
from llm.types import (
    BATCH_COMPLETED, BATCH_EXPIRED, BATCH_PENDING, BATCH_RUNNING, BatchJob, ChatRequest, ChatResponse,
)
from conftest import FakeOpenAIClient, batch_body


# ---- chat -----------------------------------------------------------------------
def test_chat_sends_body_and_parses_response(provider):
    resp = provider.chat(ChatRequest.from_prompt("hello", id="r1", metadata={"m": 1}, max_tokens=9))
    name, kwargs = provider.client.calls[-1]
    assert name == "chat"
    assert kwargs["model"] == "fake-model"
    assert kwargs["max_completion_tokens"] == 9 and "max_tokens" not in kwargs
    assert (resp.request_id, resp.text, resp.finish_reason, resp.metadata) == ("r1", "echo:hello", "stop", {"m": 1})
    assert resp.usage.total_tokens == 3


def test_request_model_overrides_default(provider):
    provider.chat(ChatRequest.from_prompt("x", model="other"))
    assert provider.client.calls[-1][1]["model"] == "other"


def test_default_params_merged_under_request_params(make_provider):
    p = make_provider(default_params={"temperature": 0.1, "top_p": 0.5})
    body = p.build_chat_body(ChatRequest.from_prompt("x", temperature=0.9))
    assert body["temperature"] == 0.9 and body["top_p"] == 0.5


def test_param_rename_does_not_clobber_explicit_new_name(provider):
    body = provider.build_chat_body(ChatRequest.from_prompt("x", max_tokens=1, max_completion_tokens=2))
    assert body["max_completion_tokens"] == 2


def test_no_renames_keeps_max_tokens(make_provider):
    body = make_provider(param_renames=None).build_chat_body(ChatRequest.from_prompt("x", max_tokens=4))
    assert body["max_tokens"] == 4


def test_missing_model_raises(make_provider):
    with pytest.raises(ValueError, match="No model"):
        make_provider(default_model=None).chat(ChatRequest.from_prompt("x"))


# ---- extra_body -----------------------------------------------------------------
SAMPLING = {"repetition_penalty": 1.1, "top_k": 40, "min_p": 0.02}


def test_live_chat_sends_extra_body_via_sdk_kwarg(provider):
    provider.chat(ChatRequest.from_prompt("x", max_tokens=5, extra_body=SAMPLING))
    kwargs = provider.client.calls[-1][1]
    assert kwargs["extra_body"] == SAMPLING
    assert "top_k" not in kwargs  # not passed as a (non-existent) SDK argument
    assert kwargs["max_completion_tokens"] == 5


def test_no_extra_body_sends_none(provider):
    provider.chat(ChatRequest.from_prompt("x"))
    assert provider.client.calls[-1][1]["extra_body"] is None


def test_extra_body_precedence(make_provider):
    p = make_provider(default_extra_body={"top_k": 10, "min_p": 0.1, "a": 1},
                      default_params={"extra_body": {"a": 2, "b": 2}})
    req = ChatRequest.from_prompt("x", extra_body={"top_k": 40})
    req.params["extra_body"] = {"b": 3, "min_p": 0.05}
    # default_extra_body < default_params["extra_body"] < params["extra_body"] < request.extra_body
    assert p.resolve_extra_body(req) == {"top_k": 40, "min_p": 0.05, "a": 2, "b": 3}


def test_extra_body_in_params_is_routed_not_sent_as_kwarg(provider):
    req = ChatRequest(messages=[{"role": "user", "content": "x"}], params={"extra_body": {"top_k": 5}})
    provider.chat(req)
    kwargs = provider.client.calls[-1][1]
    assert kwargs["extra_body"] == {"top_k": 5}


def test_build_chat_body_and_batch_lines_flatten_extra_body(make_provider):
    p = make_provider(default_extra_body={"min_p": 0.02})
    req = ChatRequest.from_prompt("q", id="s0", max_tokens=7, extra_body={"top_k": 40})
    body = p.build_batch_lines([req])[0]["body"]
    assert body["top_k"] == 40 and body["min_p"] == 0.02 and body["max_completion_tokens"] == 7
    assert "extra_body" not in body
    assert p.build_chat_body(req) == body


def test_extra_body_keys_are_not_renamed(provider):
    body = provider.build_chat_body(ChatRequest.from_prompt("x", extra_body={"max_tokens": 3}))
    assert body["max_tokens"] == 3  # sent verbatim; renames only apply to standard params


def test_get_provider_default_extra_body_override(monkeypatch):
    monkeypatch.setenv("FEATHERLESS_API_TOKEN", "test-key")
    p = get_provider("featherless", default_extra_body=SAMPLING)
    assert p.default_extra_body == SAMPLING
    assert get_provider("featherless") is not p


# ---- embeddings -----------------------------------------------------------------
def test_embed_batches_calls_and_restores_order(provider):
    out = provider.embed(["a", "abc", "ab", "abcd", "abcde"], batch_size=2)
    assert out.embeddings.dtype == np.float32
    assert out.embeddings[:, 0].tolist() == [1, 3, 2, 4, 5]
    assert sum(c[0] == "embed" for c in provider.client.calls) == 3
    assert out.usage.prompt_tokens == 5 and out.model == "fake-emb"


def test_embed_forwards_extra_kwargs(provider):
    provider.embed(["a"], dimensions=64)
    assert provider.client.calls[-1][1]["dimensions"] == 64


def test_embed_unsupported_raises(make_provider):
    with pytest.raises(NotImplementedError):
        make_provider(supports_embeddings=False).embed(["a"])


# ---- batch lines / files --------------------------------------------------------
def test_build_batch_lines_format(provider):
    lines = provider.build_batch_lines([ChatRequest.from_prompt("q", id="s0", max_tokens=7)])
    assert lines == [{
        "custom_id": "s0", "method": "POST", "url": "/v1/chat/completions",
        "body": {"model": "fake-model", "messages": [{"role": "user", "content": "q"}],
                 "max_completion_tokens": 7},
    }]


@pytest.mark.parametrize("requests_, match", [
    ([], "empty"),
    ([ChatRequest.from_prompt("a", id="x"), ChatRequest.from_prompt("b", id="x")], "unique"),
    ([ChatRequest.from_prompt("a"), ChatRequest.from_prompt("b", model="other")], "one model"),
])
def test_batch_validation_errors(provider, requests_, match):
    with pytest.raises(ValueError, match=match):
        provider.build_batch_lines(requests_)


def test_batch_request_limit(provider):
    lines = [{"custom_id": str(i), "body": {"model": "m"}} for i in range(MAX_BATCH_REQUESTS + 1)]
    with pytest.raises(ValueError, match="limit"):
        provider._validate_batch_lines(lines)


def test_embedding_batch_lines_from_mapping_and_list(provider):
    from_map = provider.build_embedding_batch_lines({"k1": "t1"}, dimensions=8)
    assert from_map == [{"custom_id": "k1", "method": "POST", "url": "/v1/embeddings",
                         "body": {"model": "fake-emb", "input": "t1", "dimensions": 8}}]
    assert [l["custom_id"] for l in provider.build_embedding_batch_lines(["a", "b"])] == ["0", "1"]


def test_write_batch_file_default_location_is_valid_jsonl(provider):
    lines = provider.build_batch_lines([ChatRequest.from_prompt("héllo", id="a")])
    path = provider.write_batch_file(lines)
    assert path.parent == provider.batch_dir and path.name.endswith("_input.jsonl")
    assert [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] == lines


# ---- batch lifecycle / results --------------------------------------------------
def test_submit_batch_uploads_and_creates(provider):
    job = provider.submit_batch([ChatRequest.from_prompt("q", id="s0")], metadata={"run": "t"})
    names = [c[0] for c in provider.client.calls]
    assert names == ["file_create", "batch_create"]
    create_kwargs = provider.client.calls[-1][1]
    assert create_kwargs["endpoint"] == "/v1/chat/completions"
    assert create_kwargs["completion_window"] == "24h"
    assert create_kwargs["metadata"] == {"kind": "chat", "run": "t"}
    assert provider.client.uploaded[0]["custom_id"] == "s0"
    assert job.status == BATCH_PENDING and job.input_file.endswith("_input.jsonl")


@pytest.mark.parametrize("raw, norm", [("validating", BATCH_PENDING), ("finalizing", BATCH_RUNNING),
                                       ("completed", BATCH_COMPLETED), ("expired", BATCH_EXPIRED)])
def test_status_normalisation(make_provider, raw, norm):
    p = make_provider(client=FakeOpenAIClient(statuses=(raw, raw)))
    assert p.get_batch("batch_1").status == norm


def test_batch_results_ordered_with_metadata_and_saved_files(provider):
    reqs = [ChatRequest.from_prompt(f"q{i}", id=f"s{i}", metadata={"i": i}) for i in range(3)]
    job = provider.submit_batch(reqs)
    results = provider.batch_results(job, reqs)
    assert [r.request_id for r in results] == ["s0", "s1", "s2"]
    assert [r.text for r in results] == ["B:q0", "B:q1", "B:q2"]
    assert [r.metadata for r in results] == [{"i": 0}, {"i": 1}, {"i": 2}]
    base = job.input_file[:-len("_input.jsonl")]
    assert open(base + "_output.jsonl", encoding="utf-8").read().count("\n") == 2


def test_batch_results_errors_and_missing(make_provider):
    def outputs(uploaded):
        ok = [{"custom_id": "s0", "response": {"status_code": 200, "body": batch_body("fine")}, "error": None}]
        err = [
            {"custom_id": "s1", "response": {"status_code": 400, "body": {"error": {"message": "bad input"}}},
             "error": None},
            {"custom_id": "s2", "response": None, "error": {"code": "batch_expired", "message": "too slow"}},
        ]
        return ok, err

    p = make_provider(client=FakeOpenAIClient(statuses=("validating", "expired"), batch_outputs=outputs))
    reqs = [ChatRequest.from_prompt(f"q{i}", id=f"s{i}") for i in range(4)]
    job = p.submit_batch(reqs)
    r0, r1, r2, r3 = p.batch_results(job, reqs)
    assert r0.ok and r0.text == "fine"
    assert r1.error == "HTTP 400: bad input"
    assert r2.error == "batch_expired: too slow"
    assert "no result" in r3.error


def test_batch_results_without_requests_uses_custom_ids(provider):
    provider.submit_batch([ChatRequest.from_prompt("q", id="only")])
    results = provider.batch_results("batch_1")
    assert [(r.request_id, r.text) for r in results] == [("only", "B:q")]


def test_batch_results_before_done_raises(make_provider):
    p = make_provider(client=FakeOpenAIClient(statuses=("validating", "in_progress")))
    p.submit_batch([ChatRequest.from_prompt("q")])
    with pytest.raises(RuntimeError, match="still"):
        p.batch_results("batch_1")


def test_failed_batch_raises_with_validation_errors(make_provider):
    client = FakeOpenAIClient(statuses=("validating", "failed"))
    client.batch_errors = NS(data=[NS(line=1, message="bad json")])
    p = make_provider(client=client)
    p.submit_batch([ChatRequest.from_prompt("q")])
    with pytest.raises(RuntimeError, match="line 1: bad json"):
        p.batch_results("batch_1")


def test_embedding_batch_round_trip(make_provider):
    def outputs(uploaded):
        ok = [{"custom_id": l["custom_id"], "error": None,
               "response": {"status_code": 200, "body": {"data": [{"index": 0, "embedding": [0.5, 0.25]}]}}}
              for l in uploaded[:1]]
        err = [{"custom_id": uploaded[1]["custom_id"], "response": None, "error": {"code": "x", "message": "y"}}]
        return ok, err

    client = FakeOpenAIClient(batch_outputs=outputs, endpoint="/v1/embeddings")
    p = make_provider(client=client)
    job = p.submit_embedding_batch({"a": "text a", "b": "text b"})
    assert job.kind == "embedding" and client.calls[-1][1]["endpoint"] == "/v1/embeddings"
    vectors, errors = p.embedding_batch_results(job)
    np.testing.assert_allclose(vectors["a"], [0.5, 0.25])
    assert errors == {"b": "x: y"}


def test_batch_methods_disabled_when_unsupported(make_provider):
    p = make_provider(supports_batch=False)
    with pytest.raises(NotImplementedError):
        p.submit_batch([ChatRequest.from_prompt("q")])
    with pytest.raises(NotImplementedError):
        p.get_batch("b")


# ---- registry -------------------------------------------------------------------
@pytest.fixture
def clean_registry(monkeypatch):
    monkeypatch.setattr(providers, "_CACHE", {})
    monkeypatch.setattr(providers, "_FACTORIES", dict(providers._FACTORIES))
    for var in ("OPENAI_API_KEY", "OPENAI_API_TOKEN", "OPENAI_MODEL", "OPENAI_EMBEDDING_MODEL",
                "FEATHERLESS_API_TOKEN", "FEATHERLESS_API_KEY", "FEATHERLESS_MODEL", "EMBEDDING_MODEL",
                "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_presets_read_env(clean_registry):
    clean_registry.setenv("FEATHERLESS_API_TOKEN", "test-key")
    clean_registry.setenv("FEATHERLESS_MODEL", "some/model")
    p = get_provider("featherless")
    assert isinstance(p, OpenAICompatibleProvider)
    assert p.default_model == "some/model"
    assert p.default_embedding_model == "Qwen/Qwen3-Embedding-8B"
    assert str(p.client.base_url).startswith("https://api.featherless.ai/v1")
    assert not p.supports_batch


def test_openai_preset_second_key_name_and_renames(clean_registry):
    clean_registry.setenv("OPENAI_API_TOKEN", "test-key")
    p = get_provider("openai")
    assert p.supports_batch and p.param_renames == {"max_tokens": "max_completion_tokens"}


def test_missing_key_raises(clean_registry):
    with pytest.raises(ValueError, match="No API key"):
        get_provider("openai")


def test_instances_cached_unless_overridden(clean_registry):
    clean_registry.setenv("OPENAI_API_KEY", "test-key")
    assert get_provider("openai") is get_provider("openai")
    custom = get_provider("openai", default_model="x")
    assert custom is not get_provider("openai") and custom.default_model == "x"


def test_default_provider_from_env(clean_registry):
    clean_registry.setenv("FEATHERLESS_API_TOKEN", "test-key")
    clean_registry.setenv("LLM_PROVIDER", "featherless")
    assert get_provider().name == "featherless"


def test_instance_passthrough_and_unknown_name(clean_registry, provider):
    assert get_provider(provider) is provider
    with pytest.raises(KeyError, match="Unknown provider"):
        get_provider("nope")


def test_register_custom_provider(clean_registry):
    class Dummy(LLMProvider):
        name = "dummy"

        def chat(self, request):
            return ChatResponse(request.id, "dummy")

    register_provider("dummy", lambda **kw: Dummy(**kw))
    p = get_provider("dummy")
    assert p.chat(ChatRequest.from_prompt("x")).text == "dummy"
    with pytest.raises(NotImplementedError):
        p.embed(["x"])
    with pytest.raises(NotImplementedError):
        p.batch_results(BatchJob("b", "dummy", "completed"))
