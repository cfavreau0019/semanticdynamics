import threading
import time

import pytest

from llm import ChatRequest, ChatResponse, chat, generate, generate_many
from llm.providers import LLMProvider


def test_generate_returns_text_with_default_system(provider):
    assert generate("hi", provider=provider, max_tokens=5) == "echo:hi"
    kwargs = provider.client.calls[-1][1]
    assert kwargs["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert kwargs["max_completion_tokens"] == 5


def test_generate_passes_extra_body(provider):
    generate("hi", provider=provider, max_tokens=5, extra_body={"top_k": 40, "min_p": 0.02})
    kwargs = provider.client.calls[-1][1]
    assert kwargs["extra_body"] == {"top_k": 40, "min_p": 0.02}


def test_generate_many_strings_share_extra_body(provider):
    generate_many(["a", "b"], provider=provider, progress=False, extra_body={"repetition_penalty": 1.1})
    assert [c[1]["extra_body"] for c in provider.client.calls] == [{"repetition_penalty": 1.1}] * 2


def test_generate_without_system(provider):
    generate("hi", system=None, provider=provider)
    assert [m["role"] for m in provider.client.calls[-1][1]["messages"]] == ["user"]


def test_chat_passes_request_through(provider):
    assert chat(ChatRequest.from_prompt("x", id="z"), provider).request_id == "z"


def test_generate_many_preserves_order_and_metadata(provider):
    reqs = [ChatRequest.from_prompt(f"p{i}", metadata={"i": i}) for i in range(20)]
    out = generate_many(reqs, provider=provider, max_workers=5, progress=False)
    assert [r.text for r in out] == [f"echo:p{i}" for i in range(20)]
    assert [r.request_id for r in out] == [r.id for r in reqs]
    assert [r.metadata["i"] for r in out] == list(range(20))


def test_generate_many_accepts_strings_with_shared_params(provider):
    out = generate_many(["a", "b"], provider=provider, system="S", model="m2", progress=False, temperature=0.2)
    assert [r.text for r in out] == ["echo:a", "echo:b"]
    for _, kwargs in provider.client.calls:
        assert kwargs["model"] == "m2" and kwargs["temperature"] == 0.2
        assert kwargs["messages"][0]["content"] == "S"


def test_generate_many_captures_errors(provider):
    out = generate_many(["ok", "FAIL", "ok2"], provider=provider, progress=False)
    assert [r.ok for r in out] == [True, False, True]
    assert out[1].text is None and out[1].error == "RuntimeError: boom"


def test_generate_many_raise_on_error(provider):
    with pytest.raises(RuntimeError, match="boom"):
        generate_many(["FAIL"], provider=provider, raise_on_error=True, progress=False)


def test_generate_many_on_result_called_once_per_request(provider):
    seen = []
    generate_many([f"p{i}" for i in range(6)], provider=provider, on_result=seen.append, progress=False)
    assert sorted(r.text for r in seen) == sorted(f"echo:p{i}" for i in range(6))


def test_generate_many_respects_max_workers():
    class Slow(LLMProvider):
        name = "slow"

        def __init__(self):
            super().__init__(default_model="m")
            self.active = self.peak = 0
            self.lock = threading.Lock()

        def chat(self, request):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.02)
            with self.lock:
                self.active -= 1
            return ChatResponse(request.id, "ok")

    p = Slow()
    generate_many([str(i) for i in range(12)], provider=p, max_workers=3, progress=False)
    assert 1 < p.peak <= 3


def test_generate_many_empty(provider):
    assert generate_many([], provider=provider, progress=False) == []
