"""
Shared fixtures. API tests run against FakeOpenAIClient, an in-memory stand-in for
openai.OpenAI that implements only the calls the llm package makes, so no network
access or API credits are needed.
"""
import json
import os
from types import SimpleNamespace as NS

import pytest

# torch + numpy from different wheels can both ship libiomp5md.dll on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from llm.providers.openai_compatible import OpenAICompatibleProvider


def completion(content, model="fake-model", finish_reason="stop"):
    return NS(
        model=model,
        choices=[NS(message=NS(content=content), finish_reason=finish_reason)],
        usage=NS(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )


def batch_body(content, model="fake-model"):
    return {
        "model": model,
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class FakeOpenAIClient:
    """
    chat:        echoes the last user message as "echo:<content>"; raises if it contains "FAIL"
    embeddings:  vector [len(text), 1.0]; data returned in reverse order to test re-sorting
    batches:     status sequence set by `statuses`; output/error file contents are built by
                 `batch_outputs(uploaded_lines) -> (output_lines, error_lines)`
    Every call is recorded in `self.calls` as (name, kwargs).
    """

    def __init__(self, statuses=("validating", "completed"), batch_outputs=None, endpoint="/v1/chat/completions"):
        self.calls = []
        self.chat = NS(completions=NS(create=self._chat))
        self.embeddings = NS(create=self._embed)
        self.files = NS(create=self._file_create, content=self._file_content)
        self.batches = NS(create=self._batch_create, retrieve=self._batch_retrieve,
                          cancel=self._batch_cancel, list=self._batch_list)
        self.statuses = list(statuses)
        self.status_idx = 0
        self.endpoint = endpoint
        self.uploaded = []
        self.batch_outputs = batch_outputs or self._default_outputs
        self.batch_errors = None
        self.n_batches = 0  # each create gets a new id: batch_1, batch_2, ...

    # chat / embeddings
    def _chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        content = kwargs["messages"][-1]["content"]
        if "FAIL" in content:
            raise RuntimeError("boom")
        return completion("echo:" + content, model=kwargs["model"])

    def _embed(self, **kwargs):
        self.calls.append(("embed", kwargs))
        data = [NS(index=i, embedding=[float(len(t)), 1.0]) for i, t in enumerate(kwargs["input"])]
        n = len(kwargs["input"])
        return NS(data=list(reversed(data)), usage=NS(prompt_tokens=n, completion_tokens=0, total_tokens=n))

    # files / batches
    def _file_create(self, file, purpose):
        self.calls.append(("file_create", {"purpose": purpose}))
        self.uploaded = [json.loads(l) for l in file.read().decode("utf-8").splitlines()]
        return NS(id="file-in")

    def _batch(self, status, batch_id=None):
        done = status in ("completed", "expired", "cancelled", "failed")
        return NS(
            id=batch_id or f"batch_{self.n_batches}", status=status, endpoint=self.endpoint, created_at=123,
            metadata={"kind": "embedding" if "embeddings" in self.endpoint else "chat"},
            request_counts=NS(total=len(self.uploaded), completed=len(self.uploaded), failed=0),
            output_file_id="file-out" if done and status != "failed" else None,
            error_file_id="file-err" if done and status != "failed" else None,
            errors=self.batch_errors,
        )

    def _batch_create(self, **kwargs):
        self.calls.append(("batch_create", kwargs))
        self.n_batches += 1
        self.status_idx = 0  # a new batch starts from the first status again
        return self._batch(self.statuses[0])

    def _batch_retrieve(self, batch_id):
        self.calls.append(("batch_retrieve", {"id": batch_id}))
        self.status_idx = min(self.status_idx + 1, len(self.statuses) - 1)
        return self._batch(self.statuses[self.status_idx], batch_id)

    def _batch_cancel(self, batch_id):
        return self._batch("cancelling", batch_id)

    def _batch_list(self, limit):
        return [self._batch(self.statuses[-1])]

    @staticmethod
    def _default_outputs(uploaded):
        out = [{"id": "r", "custom_id": l["custom_id"],
                "response": {"status_code": 200, "body": batch_body("B:" + l["body"]["messages"][-1]["content"])},
                "error": None} for l in uploaded]
        return list(reversed(out)), []

    def _file_content(self, file_id):
        out, err = self.batch_outputs(self.uploaded)
        lines = out if file_id == "file-out" else err
        return NS(text="\n".join(json.dumps(l) for l in lines))


@pytest.fixture
def fake_client():
    return FakeOpenAIClient()


@pytest.fixture
def make_provider(tmp_path):
    def make(client=None, **kwargs):
        defaults = dict(
            name="fake", client=client or FakeOpenAIClient(), default_model="fake-model",
            default_embedding_model="fake-emb", supports_batch=True,
            param_renames={"max_tokens": "max_completion_tokens"}, batch_dir=tmp_path / "batches",
        )
        defaults.update(kwargs)
        return OpenAICompatibleProvider(**defaults)
    return make


@pytest.fixture
def provider(make_provider):
    return make_provider()
