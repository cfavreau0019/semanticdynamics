"""
Provider for any OpenAI-compatible endpoint: OpenAI itself, Featherless, vLLM,
Together, OpenRouter, a local server, ... Chat and embeddings go through the
`openai` SDK; the Batch API is only enabled where the server implements it (OpenAI).
"""
import json
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from openai import OpenAI

from llm.providers.base import LLMProvider
from llm.types import (
    BATCH_CANCELLED, BATCH_COMPLETED, BATCH_EXPIRED, BATCH_FAILED, BATCH_PENDING, BATCH_RUNNING,
    BatchJob, ChatRequest, ChatResponse, EmbeddingResponse, Usage,
)

CHAT_ENDPOINT = "/v1/chat/completions"
EMBEDDINGS_ENDPOINT = "/v1/embeddings"

# OpenAI Batch API limits (per input file)
MAX_BATCH_REQUESTS = 50_000
MAX_BATCH_FILE_BYTES = 200 * 1024 * 1024

DEFAULT_BATCH_DIR = Path(__file__).resolve().parents[2] / "data" / "llm_batches"

_STATUS_MAP = {
    "validating": BATCH_PENDING,
    "in_progress": BATCH_RUNNING,
    "finalizing": BATCH_RUNNING,
    "cancelling": BATCH_RUNNING,
    "completed": BATCH_COMPLETED,
    "failed": BATCH_FAILED,
    "expired": BATCH_EXPIRED,
    "cancelled": BATCH_CANCELLED,
}


def _usage(u: Any) -> Optional[Usage]:
    if u is None:
        return None
    get = u.get if isinstance(u, dict) else lambda k, d=0: getattr(u, k, d)
    details = get("prompt_tokens_details", None)      # absent on providers that don't report caching
    cached = (details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", 0))
    return Usage(
        prompt_tokens=get("prompt_tokens", 0) or 0,
        completion_tokens=get("completion_tokens", 0) or 0,
        total_tokens=get("total_tokens", 0) or 0,
        cached_tokens=cached or 0,
    )


class OpenAICompatibleProvider(LLMProvider):
    """
    Args:
        name:                    label used in logs / BatchJob.provider
        api_key, base_url:       passed to openai.OpenAI (base_url=None -> api.openai.com)
        default_model:           chat model used when a request doesn't name one
        default_embedding_model: embedding model used when embed() doesn't name one
        default_params:          merged under every request's params (e.g. {"temperature": 0.7})
        default_extra_body:      non-standard body fields sent with every chat request, e.g.
                                 {"top_k": 40, "min_p": 0.02, "repetition_penalty": 1.1} for
                                 vLLM-based servers (Featherless). OpenAI rejects unknown fields.
        param_renames:           rename param keys before sending, e.g. OpenAI's newer models
                                 reject `max_tokens` and want `max_completion_tokens`
        supports_batch:          whether this endpoint implements the Files + Batches API
        supports_embeddings:     whether this endpoint implements /embeddings
        max_retries, timeout:    forwarded to the SDK (which retries 429/5xx with backoff)
        batch_dir:               where batch input/output JSONL files are written
        client:                  pass a pre-built OpenAI client instead of api_key/base_url
    """

    def __init__(
        self,
        name: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        default_params: Optional[dict[str, Any]] = None,
        default_extra_body: Optional[dict[str, Any]] = None,
        param_renames: Optional[dict[str, str]] = None,
        supports_batch: bool = False,
        supports_embeddings: bool = True,
        max_retries: int = 3,
        timeout: Optional[float] = 600.0,
        batch_dir: str | Path = DEFAULT_BATCH_DIR,
        client: Optional[OpenAI] = None,
        **client_kwargs,
    ):
        super().__init__(default_model, default_embedding_model, default_params, default_extra_body)
        self.name = name
        self.param_renames = dict(param_renames or {})
        self.supports_batch = supports_batch
        self.supports_embeddings = supports_embeddings
        self.batch_dir = Path(batch_dir)
        self.client = client or OpenAI(
            api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout, **client_kwargs
        )

    # ---- request building -------------------------------------------------------
    def _params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        merged = {**self.default_params, **params}
        merged.pop("extra_body", None)  # handled by resolve_extra_body
        for old, new in self.param_renames.items():
            if old in merged and new not in merged:
                merged[new] = merged.pop(old)
        return merged

    def _request_parts(self, request: ChatRequest) -> tuple[dict[str, Any], dict[str, Any]]:
        """(standard SDK kwargs, provider-specific extra_body)."""
        kwargs = {
            "model": self.resolve_model(request.model),
            "messages": request.messages,
            **self._params(request.params),
        }
        return kwargs, self.resolve_extra_body(request)

    def build_chat_body(self, request: ChatRequest) -> dict[str, Any]:
        """
        The exact JSON body the server receives (also used for batch lines): standard
        params plus extra_body fields merged in at the top level, as the SDK does.
        """
        kwargs, extra = self._request_parts(request)
        return {**kwargs, **extra}

    def _to_response(self, request: ChatRequest, completion: Any) -> ChatResponse:
        """Works on both SDK objects (live calls) and plain dicts (batch output)."""
        if isinstance(completion, dict):
            choice = completion["choices"][0]
            text = choice["message"].get("content")
            finish = choice.get("finish_reason")
            model = completion.get("model")
            usage = completion.get("usage")
        else:
            choice = completion.choices[0]
            text = choice.message.content
            finish = choice.finish_reason
            model = completion.model
            usage = completion.usage
        return ChatResponse(
            request_id=request.id,
            text=text,
            model=model,
            finish_reason=finish,
            usage=_usage(usage),
            metadata=request.metadata,
            raw=completion,
        )

    # ---- chat -------------------------------------------------------------------
    def chat(self, request: ChatRequest) -> ChatResponse:
        kwargs, extra = self._request_parts(request)
        completion = self.client.chat.completions.create(**kwargs, extra_body=extra or None)
        return self._to_response(request, completion)

    # ---- embeddings -------------------------------------------------------------
    def embed(
        self,
        texts: Sequence[str],
        model: Optional[str] = None,
        batch_size: int = 256,
        **kwargs,
    ) -> EmbeddingResponse:
        """
        Embed texts, sending up to `batch_size` inputs per HTTP call (OpenAI allows 2048).
        Extra kwargs (e.g. dimensions=256) go straight to embeddings.create.
        """
        if not self.supports_embeddings:
            return super().embed(texts, model, **kwargs)
        model = self.resolve_embedding_model(model)
        texts = list(texts)
        vectors: list[list[float]] = []
        usage = Usage()
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            resp = self.client.embeddings.create(model=model, input=chunk, **kwargs)
            vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
            if resp.usage is not None:
                usage = usage + _usage(resp.usage)
        return EmbeddingResponse(embeddings=np.asarray(vectors, dtype=np.float32), model=model, usage=usage)

    # ---- batch: input files -----------------------------------------------------
    def build_batch_lines(self, requests: Sequence[ChatRequest]) -> list[dict]:
        """
        Convert requests to OpenAI Batch API lines:
            {"custom_id": ..., "method": "POST", "url": "/v1/chat/completions", "body": {...}}
        Enforces the API's rules: unique custom_ids, a single model per file, <=50k lines.
        """
        lines = [
            {"custom_id": r.id, "method": "POST", "url": CHAT_ENDPOINT, "body": self.build_chat_body(r)}
            for r in requests
        ]
        self._validate_batch_lines(lines)
        return lines

    def build_embedding_batch_lines(
        self,
        texts: Sequence[str] | Mapping[str, str],
        model: Optional[str] = None,
        **params,
    ) -> list[dict]:
        """One line per text. `texts` may be a {custom_id: text} mapping; a list gets ids "0", "1", ..."""
        items = texts.items() if isinstance(texts, Mapping) else ((str(i), t) for i, t in enumerate(texts))
        model = self.resolve_embedding_model(model)
        lines = [
            {"custom_id": str(cid), "method": "POST", "url": EMBEDDINGS_ENDPOINT,
             "body": {"model": model, "input": text, **params}}
            for cid, text in items
        ]
        self._validate_batch_lines(lines)
        return lines

    @staticmethod
    def _validate_batch_lines(lines: list[dict]) -> None:
        if not lines:
            raise ValueError("Batch is empty.")
        if len(lines) > MAX_BATCH_REQUESTS:
            raise ValueError(f"{len(lines)} requests exceeds the {MAX_BATCH_REQUESTS} per-batch limit; split it.")
        ids = [l["custom_id"] for l in lines]
        if len(set(ids)) != len(ids):
            raise ValueError("custom_id (ChatRequest.id) values must be unique within a batch.")
        models = {l["body"]["model"] for l in lines}
        if len(models) > 1:
            raise ValueError(f"A batch file may only target one model; got {sorted(models)}.")

    def write_batch_file(self, lines: list[dict], path: Optional[str | Path] = None) -> Path:
        """Write batch lines to JSONL (default: data/llm_batches/<name>_<timestamp>_<id>_input.jsonl)."""
        if path is None:
            self.batch_dir.mkdir(parents=True, exist_ok=True)
            stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"  # unique even within a second
            path = self.batch_dir / f"{self.name}_{stamp}_input.jsonl"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        size = path.stat().st_size
        if size > MAX_BATCH_FILE_BYTES:
            raise ValueError(f"Batch file is {size / 1e6:.0f} MB; the limit is 200 MB. Split the requests.")
        return path

    # ---- batch: lifecycle -------------------------------------------------------
    def _require_batch(self) -> None:
        if not self.supports_batch:
            raise NotImplementedError(f"Provider '{self.name}' does not support the Batch API.")

    def _submit(self, path: Path, endpoint: str, kind: str, completion_window: str,
                metadata: Optional[dict[str, str]]) -> BatchJob:
        with open(path, "rb") as f:
            uploaded = self.client.files.create(file=f, purpose="batch")
        batch = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata={"kind": kind, **(metadata or {})},
        )
        job = self._to_job(batch)
        job.input_file = str(path)
        return job

    def submit_batch(
        self,
        requests: Sequence[ChatRequest],
        completion_window: str = "24h",
        metadata: Optional[dict[str, str]] = None,
        input_path: Optional[str | Path] = None,
    ) -> BatchJob:
        """Write the JSONL input, upload it, and create a /v1/chat/completions batch."""
        self._require_batch()
        path = self.write_batch_file(self.build_batch_lines(requests), input_path)
        return self._submit(path, CHAT_ENDPOINT, "chat", completion_window, metadata)

    def submit_embedding_batch(
        self,
        texts: Sequence[str] | Mapping[str, str],
        model: Optional[str] = None,
        completion_window: str = "24h",
        metadata: Optional[dict[str, str]] = None,
        input_path: Optional[str | Path] = None,
        **params,
    ) -> BatchJob:
        """Same as submit_batch but against /v1/embeddings (50% cheaper than live calls)."""
        self._require_batch()
        path = self.write_batch_file(self.build_embedding_batch_lines(texts, model, **params), input_path)
        return self._submit(path, EMBEDDINGS_ENDPOINT, "embedding", completion_window, metadata)

    def _to_job(self, batch: Any) -> BatchJob:
        counts = batch.request_counts
        meta = dict(batch.metadata or {})
        errors = getattr(batch, "errors", None)
        if errors is not None and getattr(errors, "data", None):
            meta["errors"] = "; ".join(f"line {e.line}: {e.message}" for e in errors.data)
        return BatchJob(
            id=batch.id,
            provider=self.name,
            status=_STATUS_MAP.get(batch.status, batch.status),
            raw_status=batch.status,
            endpoint=batch.endpoint,
            kind=meta.get("kind", "embedding" if batch.endpoint == EMBEDDINGS_ENDPOINT else "chat"),
            request_counts=(
                {"total": counts.total, "completed": counts.completed, "failed": counts.failed}
                if counts is not None else {}
            ),
            created_at=batch.created_at,
            metadata=meta,
            raw=batch,
        )

    def get_batch(self, batch_id: str) -> BatchJob:
        self._require_batch()
        return self._to_job(self.client.batches.retrieve(batch_id))

    def cancel_batch(self, batch_id: str) -> BatchJob:
        self._require_batch()
        return self._to_job(self.client.batches.cancel(batch_id))

    def list_batches(self, limit: int = 20) -> list[BatchJob]:
        self._require_batch()
        return [self._to_job(b) for b in self.client.batches.list(limit=limit)]

    # ---- batch: results ---------------------------------------------------------
    def _download_lines(self, file_id: Optional[str], save_to: Optional[Path]) -> list[dict]:
        if not file_id:
            return []
        text = self.client.files.content(file_id).text
        if save_to is not None:
            save_to.write_text(text, encoding="utf-8")
        return [json.loads(l) for l in text.splitlines() if l.strip()]

    def _raw_results(self, job: BatchJob | str) -> tuple[BatchJob, dict[str, dict]]:
        """
        Download output + error files, keyed by custom_id. Each value is either
        {"body": <response body>} or {"error": <message>}. Raw files are saved next
        to the input file when there is one.
        """
        input_file = job.input_file if isinstance(job, BatchJob) else None
        job = self.get_batch(job.id if isinstance(job, BatchJob) else job)  # refresh file ids
        job.input_file = input_file
        if not job.done:
            raise RuntimeError(f"Batch {job.id} is still '{job.raw_status}'; use wait_for_batch first.")
        if job.status == BATCH_FAILED and not job.raw.output_file_id:
            raise RuntimeError(f"Batch {job.id} failed: {job.metadata.get('errors', 'no details')}")

        out_path = err_path = None
        if job.input_file:
            stem = Path(job.input_file).with_suffix("")
            base = str(stem)[:-len("_input")] if str(stem).endswith("_input") else str(stem)
            out_path, err_path = Path(base + "_output.jsonl"), Path(base + "_errors.jsonl")

        results: dict[str, dict] = {}
        for line in self._download_lines(job.raw.output_file_id, out_path) + \
                self._download_lines(job.raw.error_file_id, err_path):
            cid = line["custom_id"]
            resp, err = line.get("response"), line.get("error")
            if err:
                results[cid] = {"error": f"{err.get('code')}: {err.get('message')}"}
            elif resp and resp.get("status_code") == 200:
                results[cid] = {"body": resp["body"]}
            else:
                body = (resp or {}).get("body") or {}
                msg = (body.get("error") or {}).get("message", json.dumps(body))
                results[cid] = {"error": f"HTTP {(resp or {}).get('status_code')}: {msg}"}
        return job, results

    def batch_results(self, job: BatchJob | str, requests: Optional[Sequence[ChatRequest]] = None) -> list[ChatResponse]:
        job, results = self._raw_results(job)
        if requests is None:
            requests = [ChatRequest(messages=[], id=cid) for cid in results]
        responses = []
        for req in requests:
            r = results.get(req.id)
            if r is None:
                responses.append(ChatResponse(req.id, None, error="no result returned for this request",
                                              metadata=req.metadata))
            elif "error" in r:
                responses.append(ChatResponse(req.id, None, error=r["error"], metadata=req.metadata))
            else:
                responses.append(self._to_response(req, r["body"]))
        return responses

    def embedding_batch_results(self, job: BatchJob | str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        """Returns ({custom_id: vector}, {custom_id: error message})."""
        job, results = self._raw_results(job)
        vectors, errors = {}, {}
        for cid, r in results.items():
            if "error" in r:
                errors[cid] = r["error"]
            else:
                vectors[cid] = np.asarray(r["body"]["data"][0]["embedding"], dtype=np.float32)
        return vectors, errors
